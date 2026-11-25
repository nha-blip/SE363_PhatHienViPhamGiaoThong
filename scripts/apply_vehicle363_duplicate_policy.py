import argparse
import copy
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_duplicate_resolution as audit
from scripts import dryrun_final_duplicate_resolution as dryrun
from scripts import simulate_duplicate_conflict_drop as simulation


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/Vehicle363-v1"
OUTPUT = ROOT / "data/vehicle_audit/duplicate_resolution"
NAMES = audit.NAMES
SPLITS = simulation.SPLITS
ALLOWED = {"SAME_OBJECTS_JITTER", "SUPERSET_ANNOTATION", "MORE_COMPLETE_VALID", "JITTER_EQUIVALENT"}


def safe_path(base, relative):
    path = (base / relative).resolve()
    if base.resolve() not in path.parents:
        raise RuntimeError(f"Path outside intended directory: {path}")
    return path


def save_json(path, value):
    path = safe_path(ROOT, path.relative_to(ROOT))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def label_relative(record):
    return f'labels/{record["final_split"]}/{Path(record["final_image"]).with_suffix(".txt").name}'


def inspect_record(record, decode_image=False):
    path = safe_path(DATASET, label_relative(record))
    content = path.read_bytes()
    labels = []
    for line in content.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"0", "1", "2", "3"}:
            raise RuntimeError(f"Invalid YOLO syntax/class: {path}")
        class_id = int(fields[0])
        center_x, center_y, width, height = map(float, fields[1:])
        values = (center_x, center_y, width, height)
        if not all(math.isfinite(value) for value in values) or not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise RuntimeError(f"Invalid normalized bbox values: {path}")
        if not (-1e-12 <= center_x - width / 2 < center_x + width / 2 <= 1 + 1e-12 and -1e-12 <= center_y - height / 2 < center_y + height / 2 <= 1 + 1e-12):
            raise RuntimeError(f"BBox outside normalized image bounds: {path}")
        labels.append((class_id, *values))
    if not labels:
        raise RuntimeError(f"Empty label: {path}")
    actual_counts = dict(zip(NAMES, audit.bbox_counts(labels)))
    if len(labels) != record["target_bbox_count"] or actual_counts != record["target_class_counts"]:
        raise RuntimeError(f"Annotation/class count mismatch: {path}")
    result = {"record": record, "label_sha256": hashlib.sha256(content).hexdigest()}
    if decode_image:
        image_path = safe_path(DATASET, record["final_image"])
        actual_hash = audit.digest_file(image_path)
        if actual_hash != record["sha256"]:
            raise RuntimeError(f"Image SHA-256 mismatch: {image_path}")
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Invalid image dimensions: {image_path}")
        result["sha256"] = actual_hash
        result["sizes"] = [(NAMES[label[0]], label[3] * width, label[4] * height, label[3] * width * label[4] * height) for label in labels]
    return result


def inventory_check(records):
    expected_images = {record["final_image"] for record in records}
    expected_labels = {label_relative(record) for record in records}
    if len(expected_images) != len(records) or len(expected_labels) != len(records):
        raise RuntimeError("Repeated image or label path in manifest")
    for record in records:
        if Path(record["final_image"]).parts[:2] != ("images", record["final_split"]):
            raise RuntimeError("Manifest path/split disagreement")
    actual_images = {path.relative_to(DATASET).as_posix() for split in SPLITS for path in (DATASET / "images" / split).rglob("*") if path.is_file()}
    actual_labels = {path.relative_to(DATASET).as_posix() for split in SPLITS for path in (DATASET / "labels" / split).rglob("*") if path.is_file()}
    differences = {"missing_images": len(expected_images - actual_images), "missing_labels": len(expected_labels - actual_labels), "unexpected_images": len(actual_images - expected_images), "unexpected_labels": len(actual_labels - expected_labels)}
    if any(differences.values()):
        raise RuntimeError(f"Image/label pairing or orphan files: {differences}")
    return differences


def full_scan(records, decode_image=False, baseline_label_hashes=None):
    checks = inventory_check(records)
    hashes = {}
    sizes = defaultdict(list)
    actual_records = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for offset in range(0, len(records), 4096):
            batch = records[offset:offset + 4096]
            for result in executor.map(lambda record: inspect_record(record, decode_image), batch):
                record = result["record"]
                label_path = label_relative(record)
                hashes[label_path] = result["label_sha256"]
                if baseline_label_hashes is not None and result["label_sha256"] != baseline_label_hashes[label_path]:
                    raise RuntimeError(f"Retained label bytes changed: {label_path}")
                actual_records.append({**record, "sha256": result.get("sha256", record["sha256"])})
                for name, width, height, area in result.get("sizes", []):
                    value = (width, height, area)
                    for key in ("global", f"class:{name}", f'source:{record["source_dataset"]}'):
                        sizes[key].append(value)
            print(f'{"FULL post-apply image/label validation" if decode_image else "Preflight label validation"}: {min(offset + 4096, len(records))}/{len(records)}', flush=True)
    checks.update({"invalid_yolo_rows": 0, "invalid_boxes": 0, "empty_labels": 0, "orphan_labels": 0, "invalid_class_ids": 0, "unreadable_images": 0, "image_hash_mismatches": 0, "annotation_mismatches": 0, "retained_label_byte_changes": 0, "all_final_image_bytes_sha256_revalidated": decode_image, "all_final_images_decoded": decode_image, "generated_boundary_floating_point_tolerance": 1e-12})
    leaks = dryrun.leakage(actual_records)
    if any(leaks.values()):
        raise RuntimeError(f"Cross-split leakage: {leaks}")
    checks.update({"exact_hash_cross_split_overlap": leaks["exact_hash_overlap"], "atomic_group_overlap": leaks["atomic_group_overlap"], "hcmc_sequence_overlap": leaks["hcmc_sequence_overlap"], "mio_duplicate_overlap": leaks["mio_duplicate_overlap"]})
    bbox_statistics = {}
    for key, values in sizes.items():
        data = np.asarray(values)
        bbox_statistics[key] = {"count": len(values), "units": "pixels / square pixels", **{name: {"min": float(data[:, column].min()), "median": float(np.median(data[:, column])), "p25": float(np.percentile(data[:, column], 25)), "p75": float(np.percentile(data[:, column], 75)), "max": float(data[:, column].max())} for column, name in enumerate(("width", "height", "area"))}}
    return simulation.aggregate(actual_records), checks, leaks, hashes, bbox_statistics


def rollback(transaction):
    if (DATASET / "RELEASE_MANIFEST.json").exists():
        raise RuntimeError("Frozen release exists. In-place rollback is prohibited; create a new version.")
    transaction = safe_path(OUTPUT, transaction.relative_to(OUTPUT))
    plan = audit.load_json(transaction / "plan.json")
    for item in reversed(plan["moves"]):
        original = safe_path(DATASET, item["relative"])
        parked = safe_path(transaction / "quarantine", item["relative"])
        if parked.exists():
            if original.exists():
                raise RuntimeError(f"Cannot rollback over an existing file: {original}")
            original.parent.mkdir(parents=True, exist_ok=True)
            os.replace(parked, original)
    for relative in plan["metadata_files"]:
        original = safe_path(DATASET, relative)
        backup = safe_path(transaction / "backup", relative)
        if backup.exists():
            shutil.copy2(backup, original)
        elif relative in plan["new_metadata_files"] and original.exists():
            original.unlink()
    save_json(transaction / "state.json", {"phase": "ROLLED_BACK", "dataset_restored": True})


def write_dataset_metadata(records, actual, checks, transaction, report):
    statistics = audit.load_json(transaction / "backup/metadata/dataset_statistics.json")
    group_records = defaultdict(list)
    for record in records:
        group_records[record["atomic_group"]].append(record)
    old_groups = {group["group_id"]: group for group in audit.load_json(transaction / "backup/metadata/group_manifest.json")}
    groups = []
    for group_id, members in sorted(group_records.items()):
        previous = old_groups[group_id]
        count = simulation.counts(members)
        group_type = "HCMC_CAMERA_SEQUENCE" if any(member["source_dataset"] == "HCMC" for member in members) else "MIO_EXACT_DUPLICATE_GROUP" if any(len(member["source_aliases"]) > 1 for member in members) else "MIO_SINGLETON"
        groups.append({**previous, "group_type": group_type, "image_count": count["images"], "bbox_count": count["boxes"], "class_counts": count["class_counts"], "duplicate_group_ids": sorted({member["duplicate_group"] for member in members}), "source_groups": sorted({member["source_group"] for member in members})})
    def format_counts(summary):
        return {key: summary[key] for key in ("images", "boxes", "class_counts")}
    statistics.update(format_counts(actual))
    statistics["source_counts"] = {source: {key: entry[key] for key in ("images", "boxes")} for source, entry in actual["per_source"].items()}
    statistics["per_source"] = {source: format_counts(entry) for source, entry in actual["per_source"].items()}
    statistics["per_split"] = {split: {**format_counts(entry), "source_counts": {source: {key: source_entry[key] for key in ("images", "boxes")} for source, source_entry in entry["per_source"].items()}} for split, entry in actual["per_split"].items()}
    statistics["hcmc_day_night"] = {domain: format_counts(actual["per_domain"][f"HCMC:{domain}"]) for domain in ("daytime", "nighttime")}
    statistics["bbox_size_statistics"] = report.pop("bbox_size_statistics")
    statistics["atomic_group_statistics"] = {"total": len(groups), "by_type": dict(Counter(group["group_type"] for group in groups)), "per_split": dict(Counter(group["final_split"] for group in groups)), "largest_groups": sorted(groups, key=lambda group: group["image_count"], reverse=True)[:10]}
    statistics["duplicate_annotation_conflicts"] = report["real_conflict_retained"]
    statistics["duplicate_resolution_work_4_7c"] = {"auto_groups_applied": report["applied_auto_resolution_groups"], "removed_images": report["removed"]["images"], "removed_boxes": report["removed"]["boxes"], "real_conflict_groups_kept_as_is": report["real_conflict_retained"]}
    statistics["reconciliation"].update({"work_4_7c_noncanonical_bbox_copies_removed": report["removed"]["boxes"], "final_boxes": actual["boxes"]})
    duplicates = audit.load_json(transaction / "backup/metadata/duplicate_report.json")
    applied = {entry["group_id"]: entry for entry in report["applied_decisions"]}
    real_ids = set(report["real_conflict_group_ids"])
    duplicates["annotation_conflicts"] = [group for group in duplicates["annotation_conflicts"] if group["group_id"] in real_ids]
    duplicates["compatible_annotation_groups_requiring_review"] = [group for group in duplicates.get("compatible_annotation_groups_requiring_review", []) if group["group_id"] not in applied]
    current_by_duplicate = defaultdict(list)
    for record in records:
        current_by_duplicate[record["duplicate_group"]].append(record)
    for group in duplicates["groups"]:
        group_id = group["group_id"]
        group["current_retained_members"] = [f'{record["source_dataset"]}:{record["source_image"]}' for record in current_by_duplicate[group_id]]
        if group_id in applied:
            group["pre_work_4_7c_decision"] = group["decision"]
            group["decision"] = applied[group_id]["category"]
            group["canonical"] = applied[group_id]["canonical_member"]
        elif group_id in real_ids:
            group["decision"] = "REAL_CONFLICT_KEEP_AS_IS"
            group["canonical"] = None
    duplicates["work_4_7c_application"] = statistics["duplicate_resolution_work_4_7c"]
    duplicates["source_inventory_preserved"] = True
    save_json(DATASET / "metadata/source_manifest.json", records)
    save_json(DATASET / "metadata/group_manifest.json", groups)
    save_json(DATASET / "metadata/dataset_statistics.json", statistics)
    save_json(DATASET / "metadata/duplicate_report.json", duplicates)
    save_json(DATASET / "metadata/output_validation.json", checks)
    plan = audit.load_json(transaction / "backup/metadata/split_plan.json")
    plan.update({"per_split": statistics["per_split"], "atomic_groups": len(groups), "work_4_7c_split_reassignments": 0})
    save_json(DATASET / "metadata/split_plan.json", plan)
    policy = {"work": "4.7C APPLY", "status": "PASS", "applied_auto_resolution_groups": report["applied_auto_resolution_groups"], "real_conflict_retained": report["real_conflict_retained"], "real_conflict_policy": "KEEP_AS_IS; no canonical choice; no label modification", "dryrun_sha256": report["dryrun_sha256"], "transaction": str(transaction.relative_to(ROOT)), "applied_decisions": report["applied_decisions"], "real_conflict_group_ids": report["real_conflict_group_ids"], "source_aliases_preserved": True, "split_reassignments": 0, "no_union_or_coordinate_averaging": True, "historical_build_checkpoints": "Pre-application state; do not rerun the original resume/build against this applied dataset."}
    save_json(DATASET / "metadata/applied_duplicate_policy.json", policy)
    near = audit.load_json(transaction / "backup/metadata/near_duplicate_report.json")
    names = {Path(record["final_image"]).name for record in records}
    near["baseline_candidate_count"] = near["candidate_count"]
    near["candidates"] = [item for item in near["candidates"] if Path(item["first_image"]).name in names and Path(item["second_image"]).name in names]
    near["candidate_count"] = len(near["candidates"])
    near["work_4_7c_scope"] = "Filtered prior sampled candidates to retained images only; not a new near-duplicate scan. Original sampling counts describe the pre-apply sample."
    save_json(DATASET / "metadata/near_duplicate_report.json", near)
    qa = audit.load_json(transaction / "backup/qa/manifest.json")
    for item in qa["previews"]:
        item["retained_in_current_dataset"] = Path(item["final_image"]).name in names
    qa["work_4_7c_scope"] = "Historical build QA previews; retained_in_current_dataset identifies current members. Refresh QA in Work 4.8."
    save_json(DATASET / "qa/manifest.json", qa)
    old_final = audit.load_json(transaction / "backup/final_qa_report.json")
    final = {"status": "REVIEW_REQUIRED", "work_4_7c_apply_status": "PASS", "data_integrity": "PASS", "processing_complete": True, "input_validation": old_final["input_validation"], "mapping_validation": old_final["mapping_validation"], "label_validation": checks, "source_validation": old_final["source_validation"], "class_distribution": actual["class_counts"], "split_validation": statistics["per_split"], "group_validation": {"atomic_groups": len(groups), "leakage": report["leakage"]}, "duplicate_validation": {"source_historical_inventory": "metadata/duplicate_report.json", "applied_auto_resolution_groups": report["applied_auto_resolution_groups"], "real_conflict_retained": report["real_conflict_retained"], "real_conflict_is_not_proof_of_incorrect_annotations": True}, "perceptual_near_duplicate_validation": {"candidate_count": near["candidate_count"], "exhaustive": False, "scope": near["work_4_7c_scope"]}, "visual_qa_manifest": {"manifest": "qa/manifest.json", "final_dataset_qa": "PENDING WORK 4.8", "representative_duplicate_qa": "Human PASS for representative samples only; not all 6,010 groups reviewed."}, "known_limitations": old_final["known_limitations"], "license_status": old_final["license_status"], "next": "WORK 4.8 FINAL VEHICLE363 HUMAN QA", "trained_yolo": False}
    save_json(DATASET / "final_qa_report.json", final)
    readme = (transaction / "backup/README.md").read_text(encoding="utf-8")
    prefix, suffix = readme.split("## Statistics", 1)
    _, suffix = suffix.split("## Grouping and duplicates", 1)
    prefix = prefix.replace("Work 4.7 status: **FAIL**. Final human QA is pending.", "Work 4.7C APPLY: **PASS**. Full data integrity validation passes. Work 4.8 final human QA remains pending.")
    start = prefix.index("Integrity blocker:")
    end = prefix.index("## Schema")
    prefix = prefix[:start] + "Applied only validated automatic canonicalizations. All 6,010 REAL_CONFLICT groups remain unchanged; geometric evidence was insufficient, not proof of incorrect annotations. Human QA PASS covered representative samples only.\n\nDo not rerun the original Work 4.7 build/resume checkpoints: they describe the pre-application dataset. Audit backups and quarantine are outside this dataset.\n\n" + prefix[end:]
    table = ["## Statistics", "", "| Split | Images | Boxes | Car | Motorcycle | Bus | Truck | HCMC images | MIO images |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, entry in actual["per_split"].items():
        values = [entry["images"], entry["boxes"], *[entry["class_counts"][name] for name in NAMES], *[entry["per_source"][source]["images"] for source in simulation.SOURCES]]
        table.append(f"| {split} | " + " | ".join(f"{value:,}" for value in values) + " |")
    table += ["", f'Total: {actual["images"]:,} images, {actual["boxes"]:,} boxes.', "", "## Grouping and duplicates", ""]
    suffix = suffix.replace("Canonicalization removes a duplicate pair only when target annotation multisets agree. Conflicting duplicate annotations are retained and fully reported; they prevent PASS.", "Work 4.7C canonicalizes validated jitter/superset/more-complete groups using existing annotations only. REAL_CONFLICT groups are retained without canonical selection; they do not invalidate the approved KEEP_AS-IS policy's technical PASS.")
    (DATASET / "README.md").write_text(prefix + "\n".join(table) + suffix, encoding="utf-8")
    (DATASET / "WORK_4_7_SUMMARY.txt").write_text("Historical Work 4.7 build completed; Work 4.7C APPLY STATUS: PASS\nDATA INTEGRITY: PASS\n" + f'FINAL IMAGES: {actual["images"]}\nFINAL BOXES: {actual["boxes"]}\n' + "\n".join(f'{name}: {actual["class_counts"][name]}' for name in NAMES) + "\nREAL_CONFLICT RETAINED: 6010 (KEEP AS-IS)\nNEXT: WORK 4.8 FINAL VEHICLE363 HUMAN QA\n", encoding="utf-8")
    (DATASET / "qa/FINAL_HUMAN_REVIEW.md").write_text("# Vehicle363 final human review — Work 4.8\n\nWork 4.7C APPLY and full technical validation: PASS. Final dataset human QA is pending.\n\nAll 6,010 REAL_CONFLICT groups are retained unchanged with no canonical selection. This category means insufficient automatic evidence, not necessarily incorrect labels. Human duplicate QA PASS covered representative samples only.\n\nUse qa/manifest.json retained_in_current_dataset flags for historical previews; create fresh final-dataset samples in Work 4.8. Prior guides are preserved in the Work 4.7C transaction backup. Review broad truck merges, all four classes, HCMC day/night, small boxes and source/split balance. Near-duplicate review remains sampled, and HCMC license review remains outstanding. Do not train as part of this application.\n", encoding="utf-8")


def main():
    if (DATASET / "RELEASE_MANIFEST.json").exists():
        raise RuntimeError("Frozen release exists. Create a new version instead of modifying Vehicle363-v1.")
    marker = DATASET / "metadata/applied_duplicate_policy.json"
    if marker.exists():
        raise RuntimeError("Work 4.7C policy already applied; refusing double application")
    dry = audit.load_json(OUTPUT / "work_4_7c_dryrun.json")
    if dry["status"] not in {"PASS", "REVIEW_REQUIRED"} or not dry["dry_run_only"]:
        raise RuntimeError("No valid dry-run is available")
    print("Preflight: checking every dry-run input fingerprint...", flush=True)
    expected = dry["validation"]["input_sha256"]
    paths = [safe_path(ROOT, relative) for relative in expected]
    with ThreadPoolExecutor(max_workers=12) as executor:
        hashes = list(executor.map(audit.digest_file, paths))
    if any(actual != wanted for actual, wanted in zip(hashes, expected.values())):
        raise RuntimeError("Dry-run input changed; investigate rather than forcing expected counts")
    records = audit.load_json(DATASET / "metadata/source_manifest.json")
    current = simulation.aggregate(records)
    simulation.verify_statistics(current, audit.load_json(DATASET / "metadata/dataset_statistics.json"))
    selected = {}
    real_ids = set()
    for entry in dry["results"] + dry["prior_auto_resolutions"]:
        if entry["category"] == "REAL_CONFLICT":
            if entry["auto_resolvable"] or entry["canonical_member"] is not None:
                raise RuntimeError("REAL_CONFLICT must not have an automatic canonical")
            real_ids.add(entry["group_id"])
        else:
            if entry["category"] not in ALLOWED or not entry["auto_resolvable"] or entry["canonical_member"] is None:
                raise RuntimeError("Unvalidated resolution decision")
            if entry["group_id"] in selected:
                raise RuntimeError("Repeated auto decision")
            selected[entry["group_id"]] = entry
    if selected.keys() & real_ids:
        raise RuntimeError("Auto and REAL_CONFLICT groups overlap")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["duplicate_group"]].append(record)
    retained = []
    discarded = []
    for group_id, members in grouped.items():
        if group_id not in selected:
            retained.extend(members)
            continue
        decision = selected[group_id]
        matches = [record for record in members if f'{record["source_dataset"]}:{record["source_image"]}' == decision["canonical_member"]]
        if len(matches) != 1 or len({record["final_split"] for record in members}) != 1:
            raise RuntimeError("Canonical missing, ambiguous, or cross-split")
        canonical = copy.deepcopy(matches[0])
        aliases = {(alias["source_dataset"], alias["source_image"]): alias for record in members for alias in record["source_aliases"]}
        canonical["source_aliases"] = [aliases[key] for key in sorted(aliases)]
        retained.append(canonical)
        discarded.extend(record for record in members if record is not matches[0])
    if len(selected) != dry["auto_resolvable_primary_groups"] + dry["supplemental_auto_groups"] or len(real_ids) != dry["human_review_remaining"]:
        raise RuntimeError("Decision inventory inconsistent")
    projected = simulation.aggregate(retained)
    if projected != dry["simulation"]["remaining"]:
        raise RuntimeError("Derived plan differs from validated projection")
    if any(record["duplicate_group"] in real_ids for record in discarded):
        raise RuntimeError("Removal would affect REAL_CONFLICT")
    for split in ("val", "test"):
        if {record["final_image"]: record for record in records if record["final_split"] == split} != {record["final_image"]: record for record in retained if record["final_split"] == split}:
            raise RuntimeError("Validation/test records changed unexpectedly")
    preflight, _, _, baseline_hashes, _ = full_scan(records)
    if preflight != current:
        raise RuntimeError("Actual preflight labels disagree with current metadata")
    print(f'Validated plan: {len(selected)} auto groups; {len(discarded)} noncanonical pairs to quarantine; {len(real_ids)} REAL_CONFLICT groups unchanged.', flush=True)
    transaction = OUTPUT / "apply_transactions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    transaction.mkdir(parents=True)
    metadata_files = [path.relative_to(DATASET).as_posix() for path in (DATASET / "metadata").glob("*.json")]
    metadata_files += ["final_qa_report.json", "README.md", "WORK_4_7_SUMMARY.txt", "qa/manifest.json", "qa/FINAL_HUMAN_REVIEW.md", "metadata/applied_duplicate_policy.json"]
    new_files = [relative for relative in metadata_files if not safe_path(DATASET, relative).exists()]
    moves = [{"relative": relative} for record in discarded for relative in (record["final_image"], label_relative(record))]
    plan = {"moves": moves, "metadata_files": metadata_files, "new_metadata_files": new_files, "dryrun_sha256": audit.digest_file(OUTPUT / "work_4_7c_dryrun.json")}
    save_json(transaction / "plan.json", plan)
    for relative in metadata_files:
        source = safe_path(DATASET, relative)
        if source.exists():
            destination = safe_path(transaction / "backup", relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    immutable_paths = [DATASET / "data.yaml", *list((DATASET / "evaluation_subsets").glob("*.txt"))]
    immutable_paths += [ROOT / relative for relative in expected if not str(ROOT / relative).startswith(str(DATASET))]
    immutable = {str(path.relative_to(ROOT)): audit.digest_file(path) for path in immutable_paths}
    report = {"work": "4.7C APPLY", "status": "FAIL", "data_integrity": "FAIL", "applied_auto_resolution_groups": len(selected), "applied_primary_groups": dry["auto_resolvable_primary_groups"], "applied_supplemental_groups": dry["supplemental_auto_groups"], "category_counts": dict(Counter(entry["category"] for entry in selected.values())), "real_conflict_retained": len(real_ids), "real_conflict_group_ids": sorted(real_ids), "applied_decisions": [{key: entry[key] for key in ("group_id", "category", "canonical_member")} for entry in selected.values()], "removed": simulation.counts(discarded), "current_before": current, "projection": dry["simulation"]["remaining"], "dryrun_sha256": plan["dryrun_sha256"], "transaction": str(transaction.relative_to(ROOT)), "human_qa": {"status": "PASS", "scope": "Representative final-conflict QA samples only; not all 6,010 REAL_CONFLICT groups reviewed."}, "real_conflict_interpretation": "Insufficient automated geometric evidence, not necessarily incorrect annotations", "labels_modified": False, "source_images_or_annotations_modified": False, "union_or_coordinate_averaging": False, "split_reassignments": 0}
    try:
        save_json(transaction / "state.json", {"phase": "APPLYING", "pairs": len(discarded)})
        for index, item in enumerate(moves):
            source = safe_path(DATASET, item["relative"])
            destination = safe_path(transaction / "quarantine", item["relative"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError("Quarantine destination already exists")
            os.replace(source, destination)
            if index % 500 == 0:
                save_json(transaction / "state.json", {"phase": "APPLYING", "moved_files": index + 1})
        save_json(transaction / "state.json", {"phase": "VALIDATING", "moved_files": len(moves)})
        actual, checks, leaks, after_hashes, size_statistics = full_scan(retained, decode_image=True, baseline_label_hashes=baseline_hashes)
        if actual != projected:
            raise RuntimeError("Actual fully validated counts differ from projection")
        for path, digest in immutable.items():
            if audit.digest_file(ROOT / path) != digest:
                raise RuntimeError(f"Immutable source/config/evaluation file changed: {path}")
        real_records = [record for record in records if record["duplicate_group"] in real_ids]
        if [record for record in retained if record["duplicate_group"] in real_ids] != real_records:
            if {record["final_image"]: record for record in retained if record["duplicate_group"] in real_ids} != {record["final_image"]: record for record in real_records}:
                raise RuntimeError("REAL_CONFLICT metadata altered")
        if any(after_hashes[label_relative(record)] != baseline_hashes[label_relative(record)] for record in real_records):
            raise RuntimeError("REAL_CONFLICT label bytes altered")
        report.update({"status": "PASS", "data_integrity": "PASS", "final": actual, "validation": checks, "leakage": leaks, "projection_matches_actual": True, "real_conflict_current_images_retained": len(real_records), "real_conflict_labels_unchanged": True, "all_retained_labels_byte_identical": True, "immutable_file_sha256": immutable, "bbox_size_statistics": size_statistics, "next": "WORK 4.8 FINAL VEHICLE363 HUMAN QA"})
        write_dataset_metadata(retained, actual, checks, transaction, report)
        written = audit.load_json(DATASET / "metadata/source_manifest.json")
        simulation.verify_statistics(simulation.aggregate(written), audit.load_json(DATASET / "metadata/dataset_statistics.json"))
        if {record["final_image"]: record for record in written} != {record["final_image"]: record for record in retained}:
            raise RuntimeError("Written manifest differs from validated plan")
        inventory_check(written)
        save_json(OUTPUT / "work_4_7c_apply.json", report)
        lines = ["# Work 4.7C APPLY", "", "**STATUS: PASS — DATA INTEGRITY: PASS**", "", f'Applied {len(selected):,} validated auto-resolution groups ({dry["auto_resolvable_primary_groups"]:,} primary + {dry["supplemental_auto_groups"]} supplemental). Removed {len(discarded):,} noncanonical image/label pairs from the active dataset into reversible quarantine, discarding {report["removed"]["boxes"]:,} duplicate annotation copies from active counts. No source file or retained label byte was modified; no union, averaging or split reassignment.', "", f'All {len(real_ids):,} REAL_CONFLICT groups / {len(real_records):,} current images are kept AS-IS without canonical selection. This category means insufficient automated evidence, not necessarily wrong annotations. Human QA PASS applies to representative samples only; it does not cover all 6,010 groups.', "", "## Final counts", "", f'Images: {actual["images"]:,}; boxes: {actual["boxes"]:,}.', *[f'- {name}: {actual["class_counts"][name]:,}' for name in NAMES], "", "| Split | Images | Boxes |", "|---|---:|---:|", *[f'| {split} | {entry["images"]:,} | {entry["boxes"]:,} |' for split, entry in actual["per_split"].items()], "", "## Full post-apply validation", "", "Every active image was decoded and freshly SHA-256 hashed; every label was parsed, checked for syntax/classes/finite normalized bbox bounds, paired with its image, and compared byte-for-byte to pre-apply hashes. No missing/empty/orphan/unexpected labels/images or count mismatches. Bounds allow only the existing 1e-12 generated floating-point tolerance; no coordinate is clamped.", "", f'Leakage: {leaks}. Actual totals, split/source/class counts match the dry-run projection exactly. Both sources and all target classes remain represented. REAL_CONFLICT labels and metadata are unchanged. Source GT/cache, class mapping and evaluation lists remain untouched.', "", "## Metadata and provenance", "", "Updated current source/group manifests, counts/bbox statistics, split-plan counts, output validation and duplicate-policy records. Historical duplicate source membership is preserved; canonical aliases preserve provenance. Prior near-duplicate candidates were only filtered to active images, not rescanned; historical QA previews are marked current/historical. Final dataset human QA and licensing review are still pending, so final_qa_report.json remains REVIEW_REQUIRED even though this application stage passes.", "", f'Transaction backup/quarantine: `{transaction.relative_to(ROOT).as_posix()}`. Old builder checkpoints describe the pre-policy dataset; do not rerun them against applied outputs.', "", "NEXT: WORK 4.8 FINAL VEHICLE363 HUMAN QA. No YOLO training performed."]
        (OUTPUT / "work_4_7c_apply.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        save_json(transaction / "state.json", {"phase": "COMMITTED", "status": "PASS", "final_images": actual["images"], "final_boxes": actual["boxes"]})
        print(f'WORK 4.7C APPLY STATUS: PASS\nAPPLIED AUTO-RESOLUTION GROUPS: {len(selected)}\nREAL_CONFLICT RETAINED: {len(real_ids)}\nFINAL IMAGES: {actual["images"]}\nFINAL BOXES: {actual["boxes"]}', flush=True)
        for name in NAMES:
            print(f'{name.upper()}: {actual["class_counts"][name]}')
        print(f'LEAKAGE: {leaks}\nDATA INTEGRITY: PASS\nNEXT: WORK 4.8 FINAL VEHICLE363 HUMAN QA')
    except BaseException as error:
        rollback(transaction)
        report.update({"status": "FAIL", "data_integrity": "FAIL", "error": str(error), "rolled_back": True, "next": "Investigate mismatch/integrity issue before retrying"})
        report.pop("bbox_size_statistics", None)
        save_json(OUTPUT / "work_4_7c_apply.json", report)
        (OUTPUT / "work_4_7c_apply.md").write_text(f"# Work 4.7C APPLY\n\nSTATUS: FAIL\n\nApplication rolled back to baseline. Error: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollback", type=Path)
    arguments = parser.parse_args()
    if arguments.rollback:
        rollback(arguments.rollback.resolve())
    else:
        main()
