"""Generate read-only visual and statistical QA for HCMC class IDs."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
HCMC = ROOT / "data" / "archive (1)"
OUT = ROOT / "data" / "vehicle_audit" / "hcmc_class_semantics"
IMAGE_SIZE = (1280, 720)
CLASS_IDS = [str(index) for index in range(8)]


def percentile(values, ratio):
    values = sorted(values)
    index = (len(values) - 1) * ratio
    low, high = math.floor(index), math.ceil(index)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)


def camera_key(relative: str) -> str:
    match = re.search(r"(?:cam|src)[_-]?(\d+)", Path(relative).name, re.I)
    return ("day" if relative.startswith("daytime-dataset/") else "night") + ":" + (match.group(1) if match else "unknown")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def parse():
    images = sorted(HCMC.rglob("*.jpg"))
    image_records, objects, area_by_class = {}, defaultdict(list), defaultdict(list)
    counts, image_counts, camera_counts = Counter(), defaultdict(set), defaultdict(Counter)
    cooccur = [[0] * 8 for _ in range(8)]
    for index, image in enumerate(images, 1):
        relative = str(image.relative_to(HCMC)).replace("\\", "/")
        label = image.with_suffix(".txt")
        labels_here = set()
        rows = []
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            class_id, cx, cy, width, height = fields[0], *map(float, fields[1:])
            x1, y1 = (cx - width / 2) * 1280, (cy - height / 2) * 720
            x2, y2 = (cx + width / 2) * 1280, (cy + height / 2) * 720
            box = (class_id, x1, y1, x2, y2)
            rows.append(box)
            labels_here.add(class_id)
            counts[class_id] += 1
            image_counts[class_id].add(relative)
            camera_counts[class_id][camera_key(relative)] += 1
            area_by_class[class_id].append(max(0, x2 - x1) * max(0, y2 - y1))
            objects[class_id].append((relative, x1, y1, x2, y2, camera_key(relative)))
        image_records[relative] = rows
        indices = sorted(int(value) for value in labels_here if value in CLASS_IDS)
        for left in indices:
            for right in indices:
                cooccur[left][right] += 1
        if index % 10000 == 0:
            print(f"parsed {index}/{len(images)}", flush=True)
    return images, image_records, objects, area_by_class, counts, image_counts, camera_counts, cooccur


def select(objects, areas):
    threshold = percentile(areas, .5)
    candidates = [item for item in objects if (item[3] - item[1]) * (item[4] - item[2]) >= threshold]
    buckets = defaultdict(list)
    for item in candidates:
        buckets[item[5]].append(item)
    selected, used = [], set()
    # Round-robin camera/time groups and stride filenames to prevent near-consecutive selections.
    ordered = sorted(buckets)
    cursors = {key: 0 for key in ordered}
    while len(selected) < 50 and ordered:
        next_order = []
        for key in ordered:
            bucket = buckets[key]
            cursor = cursors[key]
            while cursor < len(bucket) and bucket[cursor][0] in used:
                cursor += 1
            if cursor < len(bucket):
                chosen = bucket[cursor]
                selected.append(chosen)
                used.add(chosen[0])
                cursors[key] = cursor + max(1, len(bucket) // 50)
                next_order.append(key)
                if len(selected) == 50:
                    break
        ordered = next_order
        if not next_order:
            break
    if len(selected) < 50:
        for item in candidates:
            if item[0] not in used:
                selected.append(item)
                used.add(item[0])
                if len(selected) == 50:
                    break
    return selected


def crop_sheet(class_id, selected):
    columns, cell_w, cell_h, title_h = 5, 260, 190, 30
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (columns * cell_w, title_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"HCMC class ID {class_id}: 50 padded object crops", fill="black")
    for index, (relative, x1, y1, x2, y2, _) in enumerate(selected):
        source = HCMC / relative
        width, height = x2 - x1, y2 - y1
        pad_x, pad_y = width * .15, height * .15
        box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)), min(1280, int(x2 + pad_x)), min(720, int(y2 + pad_y)))
        with Image.open(source) as image:
            crop = ImageOps.exif_transpose(image).convert("RGB").crop(box)
            crop.thumbnail((cell_w - 8, cell_h - 35))
        row, col = divmod(index, columns)
        left, top = col * cell_w, title_h + row * cell_h
        canvas.paste(crop, (left + 4, top + 3))
        draw.text((left + 4, top + cell_h - 28), f"ID {class_id} {Path(relative).name[:22]}", fill="black")
        draw.text((left + 4, top + cell_h - 14), f"{width:.0f}x{height:.0f}", fill="black")
    canvas.save(OUT / f"class_{class_id}_crops.jpg", quality=92)


def full_frame_sheet(class_id, selected, records):
    # Twelve frames sampled from the crop selection; every same-ID box in each frame is shown.
    frames, seen = [], set()
    for item in selected:
        if item[0] not in seen:
            frames.append(item[0]); seen.add(item[0])
        if len(frames) == 12:
            break
    columns, cell_w, cell_h, title_h = 3, 420, 270, 30
    canvas = Image.new("RGB", (columns * cell_w, title_h + math.ceil(len(frames) / columns) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"HCMC class ID {class_id}: full-frame bbox QA", fill="black")
    for index, relative in enumerate(frames):
        with Image.open(HCMC / relative) as image:
            frame = ImageOps.exif_transpose(image).convert("RGB")
        box_draw = ImageDraw.Draw(frame)
        for label, x1, y1, x2, y2 in records[relative]:
            if label == class_id:
                box_draw.rectangle((x1, y1, x2, y2), outline="red", width=5)
                box_draw.text((x1 + 3, max(0, y1 - 16)), f"ID {class_id}", fill="red", stroke_width=1, stroke_fill="white")
        frame.thumbnail((cell_w - 8, cell_h - 25))
        row, col = divmod(index, columns)
        left, top = col * cell_w, title_h + row * cell_h
        canvas.paste(frame, (left + 4, top + 3))
        draw.text((left + 4, top + cell_h - 18), Path(relative).name[:45], fill="black")
    canvas.save(OUT / f"class_{class_id}_fullframes.jpg", quality=92)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images, records, objects, areas, counts, image_counts, cameras, cooccur = parse()
    selections = {}
    for class_id in CLASS_IDS:
        selections[class_id] = select(objects[class_id], areas[class_id])
        crop_sheet(class_id, selections[class_id])
        full_frame_sheet(class_id, selections[class_id], records)
    hash_groups = defaultdict(list)
    for image in images:
        relative = str(image.relative_to(HCMC)).replace("\\", "/")
        hash_groups[digest(image)].append(relative)
    duplicate_groups = [group for group in hash_groups.values() if len(group) > 1]
    pair_results = {}
    for left, right in (("0", "4"), ("1", "5"), ("2", "6"), ("3", "7")):
        related = 0
        for group in duplicate_groups:
            labels = [set(box[0] for box in records[relative]) for relative in group]
            if any(left in label_set for label_set in labels) and any(right in label_set for label_set in labels):
                related += 1
        pair_results[f"{left}<->{right}"] = related
    size = {}
    for class_id in CLASS_IDS:
        widths = [item[3] - item[1] for item in objects[class_id]]
        heights = [item[4] - item[2] for item in objects[class_id]]
        size[class_id] = {"bbox_count": counts[class_id], "image_count": len(image_counts[class_id]),
                          "median_width": round(statistics.median(widths), 3), "median_height": round(statistics.median(heights), 3),
                          "median_area": round(statistics.median(areas[class_id]), 3), "p25_area": round(percentile(areas[class_id], .25), 3),
                          "p75_area": round(percentile(areas[class_id], .75), 3), "camera_distribution": cameras[class_id].most_common()}
    result = {"authoritative_class_map_found": False, "local_metadata_evidence": [], "object_size_analysis": size,
              "cooccurrence_matrix": {class_id: {other: cooccur[int(class_id)][int(other)] for other in CLASS_IDS} for class_id in CLASS_IDS},
              "duplicate_image_investigation": {"exact_duplicate_groups": len(duplicate_groups), "tested_pairs_groups_with_both_ids": pair_results,
                                                   "different_label_sets_in_duplicate_groups": sum(1 for group in duplicate_groups if len({tuple(sorted(box[0] for box in records[relative])) for relative in group}) > 1)},
              "selection": {class_id: [{"source": item[0], "bbox_xyxy": [round(value, 2) for value in item[1:5]], "camera_group": item[5]} for item in chosen] for class_id, chosen in selections.items()}}
    (OUT / "hcmc_class_semantics_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("HCMC CLASS QA COMPLETE", flush=True)


if __name__ == "__main__":
    main()
