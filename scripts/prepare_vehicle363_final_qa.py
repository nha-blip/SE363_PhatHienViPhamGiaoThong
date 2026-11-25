import hashlib
import html
import json
import math
import random
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import apply_vehicle363_duplicate_policy as integrity
from scripts import audit_duplicate_resolution as audit
from scripts import dryrun_final_duplicate_resolution as duplicates
from scripts import simulate_duplicate_conflict_drop as statistics


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/Vehicle363-v1"
OUTPUT = ROOT / "data/vehicle_audit/final_qa"
SEED = 3634808
NAMES = audit.NAMES
CONDITIONS = ("small_objects", "medium_objects", "large_objects", "crowded_scenes", "sparse_scenes", "overlapping_vehicles", "near_boundaries", "many_boxes", "very_few_boxes")
COLORS = {"car": "#ffd700", "motorcycle": "#ff66dd", "bus": "#00ddff", "truck": "#66ff66"}
EXPECTED = {"images": 153174, "boxes": 673687, "class_counts": {"car": 327150, "motorcycle": 236236, "bus": 21435, "truck": 88866}, "per_split": {"train": {"images": 110478, "boxes": 528720}, "val": {"images": 21460, "boxes": 78772}, "test": {"images": 21236, "boxes": 66195}}}


def write_json(name, value):
    destination = integrity.safe_path(OUTPUT, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def signature(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def read_labels(record):
    content = (DATASET / integrity.label_relative(record)).read_bytes()
    return content, [tuple([int(fields[0]), *map(float, fields[1:])]) for fields in (line.split() for line in content.decode("utf-8").splitlines())]


def conditions_for(labels, areas):
    result = set()
    for area in areas:
        result.add("small_objects" if area < 32 ** 2 else "medium_objects" if area < 96 ** 2 else "large_objects")
    count = len(labels)
    if count >= 10:
        result.add("crowded_scenes")
    if count <= 2:
        result.add("sparse_scenes")
    if count >= 15:
        result.add("many_boxes")
    if count == 1:
        result.add("very_few_boxes")
    boxes = audit.pixel_boxes(labels, 1, 1)
    if any(min(box[1], box[2], 1 - box[3], 1 - box[4]) <= 0.02 + 1e-12 for box in boxes):
        result.add("near_boundaries")
    if any(audit.iou(first, second) >= 0.15 for first_index, first in enumerate(boxes) for second in boxes[first_index + 1:]):
        result.add("overlapping_vehicles")
    return sorted(result)


def inspect(record):
    image_path = DATASET / record["final_image"]
    label_path = DATASET / integrity.label_relative(record)
    before = (signature(image_path), signature(label_path))
    result = integrity.inspect_record(record, decode_image=True)
    content, labels = read_labels(record)
    if hashlib.sha256(content).hexdigest() != result["label_sha256"] or before != (signature(image_path), signature(label_path)):
        raise RuntimeError("Active image/label changed during audit")
    return {"conditions": conditions_for(labels, [entry[3] for entry in result["sizes"]]), "label_sha256": result["label_sha256"], "image_signature": before[0], "label_signature": before[1]}


def select_samples(records, features, policy, bounded):
    pools = defaultdict(list)
    auto = {entry["group_id"]: entry for entry in policy["applied_decisions"]}
    real = set(policy["real_conflict_group_ids"])
    for index, record in enumerate(records):
        split, source = record["final_split"], record["source_dataset"]
        pools[f"split_source/{split}/{source}"].append(index)
        for name, count in record["target_class_counts"].items():
            if count:
                pools[f"class/{name}/{split}/{source}"].append(index)
        for condition in features[index]["conditions"]:
            pools[f"condition/{condition}/{split}/{source}"].append(index)
        if source == "HCMC":
            pools[f'domain/{record["daytime_nighttime"]}/{split}'].append(index)
        if record["duplicate_group"] in auto:
            pools[f'canonicalized/{auto[record["duplicate_group"]]["category"]}'].append(index)
        if record["duplicate_group"] in real:
            pools[f"retained_real_conflict/{split}"].append(index)
        if record["duplicate_group"] in bounded:
            pools["bounded_jitter"].append(index)
    for split in statistics.SPLITS:
        for source in statistics.SOURCES:
            pools[f"split_source/{split}/{source}"]
            for name in NAMES:
                pools[f"class/{name}/{split}/{source}"]
            for condition in CONDITIONS:
                pools[f"condition/{condition}/{split}/{source}"]
    chosen = defaultdict(set)
    strata = {}
    for category in sorted(pools):
        requested = 12 if category.startswith(("split_source/", "canonicalized/", "retained_real_conflict/")) else 10 if category.startswith("class/") else 18 if category == "bounded_jitter" else 5
        candidates = sorted(pools[category], key=lambda index: records[index]["final_image"])
        derived_seed = int(hashlib.sha256(f"{SEED}|{category}".encode()).hexdigest()[:16], 16)
        random.Random(derived_seed).shuffle(candidates)
        used_groups = set()
        selected = []
        for index in candidates:
            record = records[index]
            group = record["source_group"] if record["source_dataset"] == "HCMC" else record["duplicate_group"]
            if group not in used_groups:
                selected.append(index)
                used_groups.add(group)
                if len(selected) == requested:
                    break
        if not category.startswith("retained_real_conflict/"):
            for index in candidates:
                if len(selected) >= min(requested, len(candidates)):
                    break
                if index not in selected:
                    selected.append(index)
        for index in selected:
            chosen[index].add(category)
        strata[category] = {"available_images": len(candidates), "requested_images_or_groups": requested, "selected_images": len(selected), "selected_distinct_diversity_groups": len(used_groups), "unavailable": not candidates, "selected_final_images": [records[index]["final_image"] for index in selected]}
    by_group = defaultdict(list)
    for index, record in enumerate(records):
        if record["duplicate_group"] in real:
            by_group[record["duplicate_group"]].append(index)
    selected_real_groups = {records[index]["duplicate_group"] for index in chosen if records[index]["duplicate_group"] in real}
    for group in sorted(selected_real_groups):
        members = sorted(by_group[group], key=lambda index: (records[index]["target_bbox_count"], records[index]["final_image"]))
        for index in (members[0], members[-1]):
            chosen[index].add(f'retained_real_conflict/{records[index]["final_split"]}')
            chosen[index].add("real_conflict_pair")
    return chosen, strata, auto, real


def render_preview(index, record, categories, feature, auto, real, bounded):
    _, labels = read_labels(record)
    with Image.open(DATASET / record["final_image"]) as source:
        image = source.convert("RGB")
    width, height = image.size
    banner_height = 160
    canvas = Image.new("RGB", (max(width, 760), height + banner_height), "#162131")
    canvas.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f'#{index:04d} | split={record["final_split"]} | source={record["source_dataset"]} | boxes={len(labels)} | native={width}x{height}', font=audit.font(16), fill="white")
    for row_index, line in enumerate(textwrap.wrap(record["final_image"], width=max(65, canvas.width // 8))[:2]):
        draw.text((10, 34 + row_index * 17), line, font=audit.font(12), fill="white")
    draw.text((10, 72), "source ID: " + record["source_image"], font=audit.font(12), fill="white")
    draw.text((10, 94), "group: " + record["duplicate_group"], font=audit.font(12), fill="white")
    for class_index, name in enumerate(NAMES):
        draw.text((10 + class_index * 150, 118), f'{name}: {record["target_class_counts"][name]}', font=audit.font(14), fill=COLORS[name])
    boxes = audit.pixel_boxes(labels, width, height)
    occupied = []
    for box_index, box in enumerate(boxes):
        class_id, left, top, right, bottom = box
        name = NAMES[class_id]
        position = (left, top + banner_height, right, bottom + banner_height)
        draw.rectangle(position, outline="black", width=4)
        draw.rectangle(position, outline=COLORS[name], width=2)
        caption = f"{name} #{box_index}"
        caption_width = int(draw.textlength(caption, font=audit.font(12))) + 6
        positions = [(left, top - 17), (left, bottom + 2), (right - caption_width, top - 34), (right + 3, top)]
        possible = []
        for proposed_x, proposed_y in positions:
            caption_x = max(0, min(width - caption_width, proposed_x))
            caption_y = max(0, min(height - 17, proposed_y)) + banner_height
            rectangle = (caption_x, caption_y, caption_x + caption_width, caption_y + 16)
            intersections = sum(not (rectangle[2] <= other[0] or other[2] <= rectangle[0] or rectangle[3] <= other[1] or other[3] <= rectangle[1]) for other in occupied)
            possible.append((intersections, rectangle))
        rectangle = min(possible, key=lambda entry: entry[0])[1]
        occupied.append(rectangle)
        draw.rectangle(rectangle, fill="black")
        draw.text((rectangle[0] + 3, rectangle[1]), caption, font=audit.font(12), fill=COLORS[name])
    filename = f'{index:04d}__{Path(record["final_image"]).stem}.jpg'
    canvas.save(OUTPUT / "previews" / filename, quality=95)
    crop_path = None
    small_indices = sorted([box_index for box_index, box in enumerate(boxes) if (box[3] - box[1]) * (box[4] - box[2]) < 32 ** 2], key=lambda box_index: (boxes[box_index][3] - boxes[box_index][1]) * (boxes[box_index][4] - boxes[box_index][2]))[:3]
    if small_indices:
        sheet = Image.new("RGB", (320 * len(small_indices), 280), "#162131")
        crop_draw = ImageDraw.Draw(sheet)
        for cell_index, box_index in enumerate(small_indices):
            class_id, left, top, right, bottom = boxes[box_index]
            margin_x = max(8, (right - left) * 0.20)
            margin_y = max(8, (bottom - top) * 0.20)
            crop_left, crop_top = max(0, math.floor(left - margin_x)), max(0, math.floor(top - margin_y))
            crop_right, crop_bottom = min(width, math.ceil(right + margin_x)), min(height, math.ceil(bottom + margin_y))
            crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            scale = min(300 / crop.width, 215 / crop.height)
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.NEAREST)
            cropped_draw = ImageDraw.Draw(crop)
            cropped_draw.rectangle(((left - crop_left) * scale, (top - crop_top) * scale, (right - crop_left) * scale, (bottom - crop_top) * scale), outline=COLORS[NAMES[class_id]], width=2)
            sheet.paste(crop, (cell_index * 320 + 10, 54))
            crop_draw.text((cell_index * 320 + 10, 8), f'{NAMES[class_id]} #{box_index} | magnification {scale:.1f}x', font=audit.font(14), fill=COLORS[NAMES[class_id]])
            crop_draw.text((cell_index * 320 + 10, 29), f'Native bbox {(right-left):.1f}x{(bottom-top):.1f}px', font=audit.font(12), fill="white")
        crop_path = "previews/" + filename.replace(".jpg", "__small_crops.jpg")
        sheet.save(OUTPUT / crop_path, quality=95)
    group_id = record["duplicate_group"]
    return {"sample_id": f"sample_{index:04d}", "preview": "previews/" + filename, "small_object_crops": crop_path, "final_image": record["final_image"], "label": integrity.label_relative(record), "source_image": record["source_image"], "split": record["final_split"], "source": record["source_dataset"], "domain": record["daytime_nighttime"] or record["original_train_test_status"], "class_counts": record["target_class_counts"], "bbox_count": record["target_bbox_count"], "duplicate_group": group_id, "atomic_group": record["atomic_group"], "selection_strata": sorted(categories), "conditions": feature["conditions"], "canonicalized_category": auto[group_id]["category"] if group_id in auto else None, "retained_real_conflict": group_id in real, "bounded_jitter_case": group_id in bounded, "image_sha256": record["sha256"], "label_sha256": feature["label_sha256"], "human_review_status": "REVIEW_REQUIRED"}


def write_html(manifest):
    identity = {"seed": manifest["seed"], "samples": [{key: sample[key] for key in ("sample_id", "final_image", "image_sha256", "label_sha256")} for sample in manifest["samples"]]}
    manifest["qa_package_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    for sample in manifest["samples"]:
        browse = [f"condition/{condition}" for condition in sample["conditions"]]
        if sample["retained_real_conflict"]:
            browse.append("annotation/REAL_CONFLICT_KEEP_AS_IS")
        if sample["canonicalized_category"]:
            browse.append("annotation/canonicalized/" + sample["canonicalized_category"])
        if sample["bounded_jitter_case"]:
            browse.append("annotation/bounded_jitter")
        sample["browse_categories"] = sorted(browse)
    samples = json.dumps(manifest["samples"], ensure_ascii=False).replace("<", "\\u003c")
    categories = sorted({category for sample in manifest["samples"] for category in sample["selection_strata"] + sample["browse_categories"]})
    options = '<option value="">All categories</option>' + "".join(f'<option value="{html.escape(category)}">{html.escape(category)}</option>' for category in categories)
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Vehicle363 — Work 4.8 Human QA</title>
<style>body{font:15px system-ui;margin:0;background:#101824;color:#edf2fa}header{padding:20px;background:#1c2a3f;position:sticky;top:0;z-index:2}h1{font-size:23px;margin:0 0 8px}select,button,input,textarea{padding:8px;margin:3px;border-radius:5px;border:1px solid #66758b}main{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:16px}.card{background:#1c2a3f;padding:12px;border-radius:8px;overflow-wrap:anywhere}.card img{width:100%;background:#111}.muted{color:#b9c7dc;font-size:13px}a{color:#89c9ff}textarea{width:94%;min-height:45px}summary{cursor:pointer}.tags{font-size:12px;color:#bce4c8}#count{font-weight:bold}@media(max-width:600px){main{grid-template-columns:1fr}header{position:static}}</style>
<header><h1>Work 4.8 — CURRENT ACTIVE Vehicle363 Human QA</h1><div>Automated integrity: PASS · HUMAN_QA_STATUS: REVIEW_REQUIRED · Do not freeze/version or train.</div>
<div class="muted">Seed __SEED__. Stratified review sample, not a random prevalence estimate. Click previews for native resolution. REAL_CONFLICT is insufficient automated evidence, not proof of incorrect labels.</div>
<select id="category">__OPTIONS__</select><select id="split"><option value="">All splits</option><option>train</option><option>val</option><option>test</option></select><select id="source"><option value="">All sources</option><option>HCMC</option><option>MIO</option></select><select id="class"><option value="">All classes</option><option>car</option><option>motorcycle</option><option>bus</option><option>truck</option></select><input id="search" placeholder="Image ID / group ID"><button id="export">Export draft review JSON</button><span id="count"></span>
<div class="muted">Review drafts are stored only in this browser; export them to retain/share evidence. They never change the authoritative HUMAN_QA_STATUS or annotations. Contact sheets/group filters are representative, not exhaustive.</div></header><main id="gallery"></main>
<script>const samples=__SAMPLES__;const key='vehicle363-qa-__SNAPSHOT__';let reviews={};try{reviews=JSON.parse(localStorage.getItem(key)||'{}')}catch(error){}const escape=value=>String(value).replace(/[&<>"']/g,character=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));const cards=new Map();const gallery=document.getElementById('gallery');
for(const sample of samples){const card=document.createElement('article');card.className='card';card.dataset.id=sample.sample_id;let related=samples.filter(other=>other.duplicate_group===sample.duplicate_group&&other.sample_id!==sample.sample_id);card.innerHTML=`<b>${escape(sample.sample_id)} · ${escape(sample.split)} · ${escape(sample.source)} · ${sample.bbox_count} boxes</b><div class="muted">${escape(sample.source_image)}<br>${escape(sample.duplicate_group)}</div><div class="tags">${escape(sample.conditions.join(' · '))}${sample.retained_real_conflict?' · REAL_CONFLICT KEEP AS-IS':''}${sample.canonicalized_category?' · '+escape(sample.canonicalized_category):''}${sample.bounded_jitter_case?' · BOUNDED JITTER':''}</div><a href="${escape(sample.preview)}" target="_blank"><img loading="lazy" src="${escape(sample.preview)}" alt="${escape(sample.source_image)} bounding boxes"></a>${sample.small_object_crops?`<details><summary>Small-object magnified crops</summary><a href="${escape(sample.small_object_crops)}" target="_blank"><img loading="lazy" src="${escape(sample.small_object_crops)}" alt="Small-object crops"></a></details>`:''}<div>${related.length?'Same active duplicate group: '+related.map(other=>`<a href="#${other.sample_id}" data-related="${other.sample_id}">${other.sample_id}</a>`).join(' · '):''}</div><details><summary>Selection and class counts</summary><div class="muted">${escape(sample.selection_strata.join(' | '))}<br>${escape(JSON.stringify(sample.class_counts))}</div></details><select class="decision"><option value="REVIEW_REQUIRED">Not reviewed</option><option value="PASS_SAMPLE">Sample acceptable</option><option value="ISSUE_FOUND">Issue found</option><option value="UNCLEAR">Unclear</option></select><textarea placeholder="Which bbox index? Missing object, wrong class, duplicate bbox, geometry, or normal occlusion?"></textarea>`;card.id=sample.sample_id;const select=card.querySelector('.decision'),notes=card.querySelector('textarea');select.value=reviews[sample.sample_id]?.decision||'REVIEW_REQUIRED';notes.value=reviews[sample.sample_id]?.notes||'';const save=()=>{reviews[sample.sample_id]={decision:select.value,notes:notes.value};try{localStorage.setItem(key,JSON.stringify(reviews))}catch(error){}};select.onchange=save;notes.oninput=save;gallery.appendChild(card);cards.set(sample.sample_id,card)}
function filter(){let count=0;for(const sample of samples){const category=document.getElementById('category').value,split=document.getElementById('split').value,source=document.getElementById('source').value,name=document.getElementById('class').value,query=document.getElementById('search').value.toLowerCase();const visible=(!category||sample.selection_strata.includes(category))&&(!split||sample.split===split)&&(!source||sample.source===source)&&(!name||sample.class_counts[name]>0)&&(!query||(sample.final_image+' '+sample.source_image+' '+sample.duplicate_group).toLowerCase().includes(query));cards.get(sample.sample_id).hidden=!visible;if(visible)count++}document.getElementById('count').textContent=` ${count}/${samples.length} images`}
for(const id of ['category','split','source','class','search'])document.getElementById(id).addEventListener('input',filter);document.querySelectorAll('[data-related]').forEach(link=>link.onclick=()=>{for(const id of ['category','split','source','class','search'])document.getElementById(id).value='';filter();cards.get(link.dataset.related).scrollIntoView({block:'center'})});document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({work:'4.8',human_qa_status:'REVIEW_REQUIRED',scope:'Human sample-review drafts only; not a full-dataset approval',dataset_snapshot:'__SNAPSHOT__',reviews},null,2)],{type:'application/json'});const anchor=document.createElement('a');anchor.href=URL.createObjectURL(blob);anchor.download='vehicle363_human_qa_draft.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(anchor.href),1000)};filter();</script></html>'''
    template = template.replace("sample.selection_strata.includes(category)", "sample.selection_strata.concat(sample.browse_categories).includes(category)")
    template = template.replace("Contact sheets/group filters are representative", "Group comparisons are representative")
    template = template.replace("const key='vehicle363-qa-__SNAPSHOT__'", "const key='vehicle363-qa-__PACKAGE__'")
    template = template.replace("dataset_snapshot:'__SNAPSHOT__',reviews", "dataset_snapshot:'__SNAPSHOT__',qa_package_sha256:'__PACKAGE__',sample_identities:samples.map(sample=>({sample_id:sample.sample_id,final_image:sample.final_image,image_sha256:sample.image_sha256,label_sha256:sample.label_sha256})),reviews")
    template = template.replace("__SEED__", str(SEED)).replace("__OPTIONS__", options).replace("__SNAPSHOT__", manifest["dataset_snapshot_sha256"]).replace("__PACKAGE__", manifest["qa_package_sha256"]).replace("__SAMPLES__", samples)
    (OUTPUT / "index.html").write_text(template, encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "previews").mkdir(exist_ok=True)
    input_paths = sorted([path for path in (DATASET / "metadata").glob("*.json")] + [DATASET / "data.yaml", DATASET / "final_qa_report.json", ROOT / "data/vehicle_audit/duplicate_resolution/work_4_7c_dryrun.json", ROOT / "data/vehicle_audit/duplicate_resolution/work_4_7c_apply.json"])
    before = {str(path.relative_to(ROOT)): audit.digest_file(path) for path in input_paths}
    records = audit.load_json(DATASET / "metadata/source_manifest.json")
    if audit.load_json(DATASET / "metadata/class_mapping.json")["schema"] != dict(zip(map(str, range(4)), NAMES)):
        raise RuntimeError("Active target class schema differs from locked Vehicle363 mapping")
    policy = audit.load_json(DATASET / "metadata/applied_duplicate_policy.json")
    applied = audit.load_json(ROOT / "data/vehicle_audit/duplicate_resolution/work_4_7c_apply.json")
    dry = audit.load_json(ROOT / "data/vehicle_audit/duplicate_resolution/work_4_7c_dryrun.json")
    if policy["status"] != "PASS" or applied["status"] != "PASS":
        raise RuntimeError("Work 4.7C application has not passed")
    if applied["dryrun_sha256"] != before[str((ROOT / "data/vehicle_audit/duplicate_resolution/work_4_7c_dryrun.json").relative_to(ROOT))]:
        raise RuntimeError("Applied/dry-run evidence differs")
    pairing = integrity.inventory_check(records)
    actual = statistics.aggregate(records)
    statistics.verify_statistics(actual, audit.load_json(DATASET / "metadata/dataset_statistics.json"))
    for field in ("images", "boxes", "class_counts"):
        if actual[field] != EXPECTED[field]:
            raise RuntimeError(f"Actual {field} differs from expected: {actual[field]}")
    for split, expected in EXPECTED["per_split"].items():
        for field, value in expected.items():
            if actual["per_split"][split][field] != value:
                raise RuntimeError(f"Actual {split}/{field} differs from expected")
    features = []
    print("FINAL READ-ONLY AUDIT: parsing labels, decoding and freshly hashing every active image...", flush=True)
    with ThreadPoolExecutor(max_workers=12) as executor:
        for offset in range(0, len(records), 4096):
            features.extend(executor.map(inspect, records[offset:offset + 4096]))
            print(f"Audited {min(offset + 4096, len(records))}/{len(records)}", flush=True)
    leaks = duplicates.leakage(records)
    if any(leaks.values()):
        raise RuntimeError(f"Leakage found: {leaks}")
    by_group = defaultdict(list)
    for record in records:
        by_group[record["duplicate_group"]].append(record)
    for decision in policy["applied_decisions"]:
        members = by_group[decision["group_id"]]
        if len(members) != 1 or f'{members[0]["source_dataset"]}:{members[0]["source_image"]}' != decision["canonical_member"]:
            raise RuntimeError("Active canonical selection does not match applied policy")
    for group in policy["real_conflict_group_ids"]:
        if not by_group[group]:
            raise RuntimeError("Retained REAL_CONFLICT group missing")
    bounded = {entry["group_id"] for entry in dry["results"] if entry["auto_resolvable"] and any(pair["evidence"] == "BOUNDED_2PX_JITTER" for attempt in entry["candidate_attempts"] if not attempt["reasons"] for comparison in attempt["comparisons"] for pair in comparison["pairs"])}
    chosen, strata, auto, real = select_samples(records, features, policy, bounded)
    selected = sorted(chosen, key=lambda index: records[index]["final_image"])
    print(f"Generating {len(selected)} unique current-active image previews...", flush=True)
    samples = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for offset in range(0, len(selected), 64):
            batch = [(number + 1, index) for number, index in enumerate(selected) if offset <= number < offset + 64]
            samples.extend(executor.map(lambda item: render_preview(item[0], records[item[1]], chosen[item[1]], features[item[1]], auto, real, bounded), batch))
    coverage = {"splits": dict(Counter(sample["split"] for sample in samples)), "sources": dict(Counter(sample["source"] for sample in samples)), "split_sources": dict(Counter(f'{sample["split"]}/{sample["source"]}' for sample in samples)), "classes": {name: sum(sample["class_counts"][name] > 0 for sample in samples) for name in NAMES}, "conditions": {name: sum(name in sample["conditions"] for sample in samples) for name in CONDITIONS}, "canonicalized_categories": dict(Counter(sample["canonicalized_category"] for sample in samples if sample["canonicalized_category"])), "retained_real_conflict_images": sum(sample["retained_real_conflict"] for sample in samples), "retained_real_conflict_groups": len({sample["duplicate_group"] for sample in samples if sample["retained_real_conflict"]}), "bounded_jitter_images": sum(sample["bounded_jitter_case"] for sample in samples), "hcmc_domains": dict(Counter(sample["domain"] for sample in samples if sample["source"] == "HCMC"))}
    if any(not count for section in ("splits", "sources", "classes", "conditions") for count in coverage[section].values()):
        raise RuntimeError("Required QA coverage missing")
    for path in (sample["preview"] for sample in samples):
        with Image.open(OUTPUT / path) as image:
            image.verify()
    def unchanged(item):
        record, feature = item
        if signature(DATASET / record["final_image"]) != tuple(feature["image_signature"]) or signature(DATASET / integrity.label_relative(record)) != tuple(feature["label_signature"]):
            raise RuntimeError("Active file stat changed during QA generation")
        if audit.digest_file(DATASET / integrity.label_relative(record)) != feature["label_sha256"]:
            raise RuntimeError("Active label bytes changed during QA generation")
    print("Verifying read-only fingerprints and active files after preview generation...", flush=True)
    with ThreadPoolExecutor(max_workers=12) as executor:
        for offset in range(0, len(records), 4096):
            list(executor.map(unchanged, zip(records[offset:offset + 4096], features[offset:offset + 4096])))
    after = {str(path.relative_to(ROOT)): audit.digest_file(path) for path in input_paths}
    if before != after:
        raise RuntimeError("Active metadata or applied-policy evidence changed")
    snapshot = before[str((DATASET / "metadata/source_manifest.json").relative_to(ROOT))]
    checks = {**pairing, "invalid_yolo_rows": 0, "invalid_class_ids": 0, "invalid_boxes": 0, "non_finite_coordinates": 0, "empty_labels": 0, "orphan_labels": 0, "unreadable_images": 0, "image_hash_mismatches": 0, "annotation_count_mismatches": 0, "metadata_modified": False, "active_images_or_labels_modified": False, "all_active_images_freshly_sha256_verified": True, "all_active_images_decoded": True, "all_active_labels_parsed_and_rehashed": True, "generated_boundary_floating_point_tolerance": 1e-12}
    automated = {"work": "4.8", "automated_integrity": "PASS", "human_qa_status": "REVIEW_REQUIRED", "created_at_utc": datetime.now(timezone.utc).isoformat(), "actual": actual, "expected": EXPECTED, "expected_totals_match": True, "leakage": leaks, "checks": checks, "input_sha256_before": before, "input_sha256_after": after, "readonly_verification": "All image content freshly hashed against manifest, image/label stats checked again after QA, all label bytes freshly rehashed, all input metadata SHA-256 unchanged.", "trained_yolo": False, "frozen_or_versioned": False}
    manifest = {"work": "4.8", "status": "REVIEW_REQUIRED", "automated_integrity": "PASS", "HUMAN_QA_STATUS": "REVIEW_REQUIRED", "seed": SEED, "dataset": "data/Vehicle363-v1", "dataset_snapshot_sha256": snapshot, "sample_count": len(samples), "crop_sheet_count": sum(bool(sample["small_object_crops"]) for sample in samples), "sampling": {"method": "Independent per-stratum deterministic seeded shuffle of sorted current-active paths; camera/group diversity first, then fill; deduplicated union, plus diagnostic min/max-box active REAL_CONFLICT pairs", "quotas": "Split/source 12; class/split/source 10; condition/split/source 5; HCMC domain/split 5; canonical category 12; REAL_CONFLICT distinct groups/split 12; bounded jitter 18", "thresholds": {"small_area_pixels": "<1024", "medium_area_pixels": "1024..9216 exclusive upper", "large_area_pixels": ">=9216", "crowded_target_boxes": ">=10", "sparse_target_boxes": "<=2", "many_target_boxes": ">=15", "very_few_target_boxes": "1", "overlapping_bbox_iou": ">=0.15", "near_boundary_normalized_margin": "<=0.02"}, "limitations": ["Conditions are target-bbox geometric proxies, not independently verified scene semantics.", "Object-size thresholds use native image pixels; definitions are engineering QA strata.", "Overlapping boxes may be normal occlusion or duplicate annotation; human judgment required.", "Crowding counts target boxes only; non-target excluded objects may still be visible.", "Sample is deliberately enriched for rare classes/hard conditions, not a prevalence estimate.", "Unavailable per-source/split condition strata are explicitly recorded; no fabricated samples.", "Canonicalized/bounded-jitter previews show retained current labels only, not quarantined/source alternatives.", "REAL_CONFLICT pairs show active copies; not all 6,010 groups are reviewed."]}, "coverage": coverage, "strata": strata, "samples": samples, "next": "Wait for human QA. DO NOT freeze/version Vehicle363 yet."}
    write_html(manifest)
    write_json("qa_manifest.json", manifest)
    write_json("automated_integrity_report.json", automated)
    lines = ["# Work 4.8 — Final Vehicle363 human QA", "", "**AUTOMATED INTEGRITY: PASS**", "**HUMAN_QA_STATUS: REVIEW_REQUIRED**", "", "Open [index.html](index.html) in a browser. Filter by split, source, class or selection category; click images for native-resolution previews, expand small-object crops, and compare links within retained REAL_CONFLICT groups. Every displayed annotation comes from a CURRENT ACTIVE image/label, never a quarantined or original-source copy.", "", f'Unique image samples: {len(samples)}; small-object crop sheets: {manifest["crop_sheet_count"]}; fixed seed: {SEED}.', "", "## Coverage", *[f'- {section}: {coverage[section]}' for section in coverage], "", "Selection quotas/availability and actual image IDs are in qa_manifest.json. Each image may belong to multiple categories, so category counts do not sum to unique sample count. Camera/group diversity is prioritized; only explicit real-conflict comparison pairs intentionally repeat exact-image groups. Unavailable strata are reported instead of fabricated.", "", "Small/medium/large areas: <32² / 32²..<96² / >=96² native pixels. Crowded >=10 target boxes, sparse <=2, many >=15, very few=1; overlap IoU>=0.15; near boundary within 2% of any image edge. These are QA engineering proxies, not proof of physical object identity or scene quality.", "", "## Review checklist", "- Correct target class: car / motorcycle / bus / truck?", "- Bbox fits the intended vehicle, without abnormal truncation or zero-size geometry?", "- Missing target objects or duplicate annotations of the same physical object?", "- Overlap is normal occlusion/crowding, or a suspicious duplicate/class contradiction?", "- Small objects remain identifiable; inspect magnified crops and native frames.", "- Canonicalized groups retain valid complete labels; bounded-jitter cases need attention.", "- REAL_CONFLICT copies remain unchanged; this label is insufficient automatic evidence, not necessarily wrong annotations.", "", "Use per-card draft decisions/notes and Export draft review JSON. Browser storage is local and may be unavailable for file:// pages; export drafts regularly. An individual sample PASS is not a full dataset PASS. Exports keep HUMAN_QA_STATUS=REVIEW_REQUIRED and do not modify labels or automatically approve the dataset.", "", "## Automated integrity", "", f'Images: {actual["images"]:,}; boxes: {actual["boxes"]:,}. Class/split/source counts match current metadata and expected totals. Every image decoded and freshly SHA-256 verified; every label parsed for syntax/classes/finite normalized values/bounds. No empty/orphan/unexpected files or leakage. Existing 1e-12 generated-boundary roundoff tolerance is retained; no coordinate is rewritten or clamped.', "", f'Leakage: {leaks}. Metadata fingerprints, active-file stats and label hashes stayed unchanged. See automated_integrity_report.json.', "", "## Next", "", "Wait for human QA. DO NOT freeze/version Vehicle363 yet. No YOLO training, source modifications, active image/label changes, deletions, remapping, or split changes were performed."]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f'WORK 4.8 STATUS: REVIEW_REQUIRED\nAUTOMATED INTEGRITY: PASS\nQA SAMPLES GENERATED: {len(samples)}')
    for split in statistics.SPLITS:
        print(f'{split.upper()} SAMPLES: {coverage["splits"].get(split, 0)}')
    print(f'SOURCE COVERAGE: {coverage["sources"]}\nCLASS COVERAGE: {coverage["classes"]}\nCONDITION COVERAGE: {coverage["conditions"]}\nFINAL IMAGES: {actual["images"]}\nFINAL BOXES: {actual["boxes"]}\nLEAKAGE: {leaks}\nHUMAN_QA_STATUS: REVIEW_REQUIRED\nQA LOCATION: {OUTPUT}\nNEXT: Wait for human QA. DO NOT freeze/version Vehicle363 yet.')


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, AssertionError, KeyError, ValueError, OSError) as error:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_json("automated_integrity_report.json", {"work": "4.8", "automated_integrity": "FAIL", "human_qa_status": "REVIEW_REQUIRED", "error": str(error), "dataset_modified_by_script": False})
        write_json("qa_manifest.json", {"work": "4.8", "status": "FAIL", "automated_integrity": "FAIL", "HUMAN_QA_STATUS": "REVIEW_REQUIRED", "error": str(error), "samples": []})
        (OUTPUT / "README.md").write_text(f"# Work 4.8\n\nAUTOMATED INTEGRITY: FAIL\nHUMAN_QA_STATUS: REVIEW_REQUIRED\n\n{error}\n\nDo not freeze/version or train. Investigate before regenerating QA.\n", encoding="utf-8")
        (OUTPUT / "index.html").write_text("<!doctype html><meta charset='utf-8'><h1>Automated integrity: FAIL</h1><p>HUMAN_QA_STATUS: REVIEW_REQUIRED</p><pre>" + html.escape(str(error)) + "</pre><p>Do not freeze/version or train.</p>", encoding="utf-8")
        raise
