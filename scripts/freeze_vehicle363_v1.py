import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import apply_vehicle363_duplicate_policy as integrity
from scripts import audit_duplicate_resolution as audit
from scripts import dryrun_final_duplicate_resolution as duplicate_audit
from scripts import prepare_vehicle363_final_qa as qa
from scripts import simulate_duplicate_conflict_drop as statistics


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/Vehicle363-v1"
OUTPUT = ROOT / "data/vehicle_audit/final"
DATE = "2026-10-04"
REQUEST = Path("C:/Users/NHA/.codex/attachments/c611fdfc-74cc-4a3e-b189-d7fac02d6d65/Pasted text.txt")
YOLO_REFERENCE = "https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py"
CONTROLS = {"FILE_HASHES.sha256", "DATASET_FINGERPRINT.sha256"}


def authoritative_paths():
    return sorted(list((DATASET / "metadata").glob("*.json")) + [DATASET / relative for relative in ("dataset.yaml", "data.yaml", "README.md", "RELEASE_MANIFEST.json", "final_qa_report.json", "WORK_4_7_SUMMARY.txt", "qa/manifest.json", "qa/duplicate_review_manifest.json", "qa/FINAL_HUMAN_REVIEW.md")])


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def parse_hash_manifest(content):
    rows = []
    for line in content.decode("utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError("Invalid SHA-256 manifest")
        integrity.safe_path(DATASET, relative)
        rows.append((relative, digest))
    if [relative for relative, _ in rows] != sorted({relative for relative, _ in rows}):
        raise RuntimeError("Hash manifest paths are not unique and sorted")
    return rows


def verify_release():
    content = (DATASET / "FILE_HASHES.sha256").read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    if (DATASET / "DATASET_FINGERPRINT.sha256").read_text().strip() != fingerprint + "  FILE_HASHES.sha256":
        raise RuntimeError("Dataset fingerprint mismatch")
    rows = parse_hash_manifest(content)
    with ThreadPoolExecutor(max_workers=12) as executor:
        for offset in range(0, len(rows), 8192):
            batch = rows[offset:offset + 8192]
            for (relative, expected), actual in zip(batch, executor.map(lambda entry: audit.digest_file(integrity.safe_path(DATASET, entry[0])), batch)):
                if actual != expected:
                    raise RuntimeError(f"Release file changed: {relative}")
            print(f"Release hash verification: {min(offset + 8192, len(rows))}/{len(rows)}", flush=True)
    records = audit.load_json(DATASET / "metadata/source_manifest.json")
    integrity.inventory_check(records)
    indexed = {relative for relative, _ in rows}
    required = {record["final_image"] for record in records} | {integrity.label_relative(record) for record in records}
    authoritative = {path.relative_to(DATASET).as_posix() for path in authoritative_paths()}
    if indexed != required | authoritative or len(required) != len(records) * 2:
        raise RuntimeError("Release hash manifest does not cover all images/labels")
    if any(relative in CONTROLS for relative in indexed):
        raise RuntimeError("Self-referential hash manifest")
    return fingerprint, len(rows)


def portable_yaml():
    return yaml.safe_dump({"train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(audit.NAMES))}, sort_keys=False, allow_unicode=True).encode("utf-8")


def verify_yaml(content, root):
    config = yaml.safe_load(content)
    if config.get("path") is not None or config["names"] != dict(enumerate(audit.NAMES)):
        raise RuntimeError("Non-portable YAML or incorrect target schema")
    resolved = {split: (root / config[split]).resolve() for split in statistics.SPLITS}
    if any(not path.is_dir() or root.resolve() not in path.parents for path in resolved.values()):
        raise RuntimeError("YOLO image directories do not resolve within dataset root")
    return {split: str(path.relative_to(root)) for split, path in resolved.items()}


def readme(actual):
    rows = []
    for split, entry in actual["per_split"].items():
        values = [entry["images"], entry["boxes"], *[entry["class_counts"][name] for name in audit.NAMES]]
        rows.append(f"| {split} | " + " | ".join(f"{value:,}" for value in values) + " |")
    return "\n".join(["# Vehicle363-v1 — FROZEN", "", f"Official Project 363 release, frozen {DATE}. Purpose: four-class vehicle detection. Images, labels, split assignments and class mapping are immutable. Changes require Vehicle363-v1.1 or Vehicle363-v2; do not modify v1 in place.", "", "## Sources and schema", "", "Sources: local HCMC daytime/nighttime vehicle dataset and MIO-TCD Localization. Full source membership, exclusions, grouping and mapping evidence remain in metadata/ and ../../vehicle_audit/. No source image/annotation was changed.", "", "| ID | Class | Boxes |", "|---|---|---:|", *[f'| {class_id} | {name} | {actual["class_counts"][name]:,} |' for class_id, name in enumerate(audit.NAMES)], "", "HCMC source IDs 0/4 → motorcycle, 1/5 → car, 2/6 → bus, 3/7 → truck. MIO car/motorcycle/bus map directly; articulated_truck, single_unit_truck, pickup_truck and work_van map to truck. MIO motorized_vehicle, bicycle, non-motorized_vehicle and pedestrian are explicitly excluded. See metadata/class_mapping.json.", "", f'Totals: {actual["images"]:,} images / {actual["images"]:,} labels / {actual["boxes"]:,} bboxes.', "", "| Split | Images | Boxes | Car | Motorcycle | Bus | Truck |", "|---|---:|---:|---:|---:|---:|---:|", *rows, "", *[f'- {source}: {entry["images"]:,} images / {entry["boxes"]:,} boxes.' for source, entry in actual["per_source"].items()], "", "## Duplicate handling and leakage", "", "Work 4.7C applied 3,572 validated automatic decisions (3,571 primary plus one supplemental compatible subset): deterministic existing canonical sets for jitter, validated complete existing sets for supersets/more-complete cases. No coordinate averaging or annotation union. Noncanonical copies are quarantined outside the active dataset; never restore them into v1.", "", "All 6,010 REAL_CONFLICT groups are KEPT AS-IS, not resolved. Their current labels/images are unchanged; no canonical member was selected. REAL_CONFLICT means insufficient automated geometric evidence, not necessarily incorrect annotations. See metadata/applied_duplicate_policy.json and the preserved Work 4.7 audit trail.", "", "Exact hashes and atomic groups remain in one split. HCMC camera/video sequences and MIO exact duplicate groups remain atomic. Pre-freeze exact-hash/atomic/HCMC-sequence/MIO-duplicate cross-split leakage: zero.", "", "## Automated and human QA", "", "AUTOMATED QA: PASS. Pre-freeze validation parsed every YOLO row, checked class IDs/finite normalized bounds, pairing/counts/empty/orphan files, decoded every image and freshly SHA-256 hashed every image and label. Only the existing 1e-12 generated boundary roundoff tolerance applies; no clamping or annotation rewrite.", "", f"HUMAN_QA_STATUS: PASS; human QA date {DATE}. The reviewer confirmed tight/acceptable boxes, correct target classes, no abnormal duplicate boxes, and no obvious missing target vehicles in reviewed representative stratified samples. This is NOT a claim that all 153,174 images or all 6,010 retained conflict groups were manually inspected.", "", "697 representative Work 4.8 images cover train/val/test, both sources, four classes, size/crowding/overlap/boundary conditions, canonicalized/retained-conflict/bounded-jitter cases. QA evidence remains at ../vehicle_audit/final_qa/. Individual sample verdicts were not fabricated; the overall representative-review declaration is recorded separately.", "", "## YOLO configuration", "", "Use dataset.yaml; data.yaml is an identical compatibility configuration. Both omit the optional root path so the YAML directory supplies the dataset root; train/val/test paths are relative. No machine-specific absolute dataset root is required. Pass the YAML by a valid path when consuming it.", "", f'This path-resolution convention was checked against the [official Ultralytics loader]({YOLO_REFERENCE}). YAML/schema/directory/pairing checks passed; Ultralytics is not installed in this environment, so no runtime training or training-loader execution was performed. YOLO_TRAINING_READY=YES refers to dataset readiness, not an installed training environment.', "", "Do not run original build/resume/duplicate-apply/rollback commands against this frozen release. Training must not repair images or write label caches inside v1; arrange caches/output paths separately and verify release hashes after experiments. Images may be hardlinks to source bytes: never edit them or apply shared-inode permission changes.", "", "## Fingerprint and verification", "", "FILE_HASHES.sha256 is UTF-8 with LF newlines, lowercase SHA-256, two spaces, and dataset-relative POSIX paths sorted lexicographically. It includes every active image/label and authoritative metadata/config/document files, including RELEASE_MANIFEST.json. DATASET_FINGERPRINT.sha256 is SHA-256 of the exact FILE_HASHES.sha256 bytes. The two hash-control files exclude themselves to avoid circular hashing; RELEASE_MANIFEST.json records locations/algorithm, not an embedded self-referential digest.", "", "From the project workspace run `python freeze_vehicle363_v1.py --verify` for read-only full verification. Audit scripts and their hashes are recorded in RELEASE_MANIFEST.json; they are provenance, not files inside the release payload. The manifest can also be verified by any SHA-256 checker after moving the dataset directory.", "", "## Known limitations and licensing", "", "- HCMC semantics depend on representative human/statistical QA; no authoritative source class map was recovered.", "- HCMC sequence IDs are conservative filename-derived groupings.", "- work_van → truck is a deliberate broad-superclass merge with MEDIUM confidence.", "- Non-target/excluded objects may remain visible without target boxes; source/class/domain imbalance remains possible.", "- 6,010 retained REAL_CONFLICT groups remain unresolved under the approved KEEP-AS-IS policy.", "- Perceptual near-duplicate screening was sampled, not exhaustive; zero exact leakage does not prove no near-duplicate leakage.", "- Original MIO test lacks local ground truth and is not included as labeled active images.", "- Source invalid boxes were excluded without rewriting original coordinates; details remain in historical reports.", "- HCMC source license/usage terms remain UNKNOWN — MANUAL REVIEW REQUIRED. MIO local documentation states CC BY-NC-SA 4.0. Frozen technical release does not grant redistribution permissions; combined redistribution is not claimed.", "", "## Future versions", "", "v1 contains only car, motorcycle, bus and truck. Direction classes are reserved for a future version if required. Dataset branch is complete; next: Work 5 — Temporal / Tracking Dataset Audit. DO NOT TRAIN YET.", ""])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata_paths = sorted((DATASET / "metadata").glob("*.json"))
    work8 = ROOT / "data/vehicle_audit/final_qa"
    baseline = {str(path.relative_to(ROOT)): audit.digest_file(path) for path in metadata_paths + [DATASET / "data.yaml", DATASET / "final_qa_report.json", work8 / "qa_manifest.json", work8 / "automated_integrity_report.json"]}
    previous = audit.load_json(work8 / "automated_integrity_report.json")
    samples = audit.load_json(work8 / "qa_manifest.json")
    if previous["automated_integrity"] != "PASS":
        raise RuntimeError("Work 4.8 automated integrity did not pass")
    for path, expected in previous["input_sha256_after"].items():
        if audit.digest_file(ROOT / path) != expected:
            raise RuntimeError(f"Work 4.8 input changed: {path}")
    records = audit.load_json(DATASET / "metadata/source_manifest.json")
    pairing = integrity.inventory_check(records)
    mapping = audit.load_json(DATASET / "metadata/class_mapping.json")
    if mapping["schema"] != dict(zip(map(str, range(4)), audit.NAMES)):
        raise RuntimeError("Class mapping changed")
    actual = statistics.aggregate(records)
    if actual != previous["actual"]:
        raise RuntimeError("Counts/source/split properties differ from Work 4.8")
    for key in ("images", "boxes", "class_counts"):
        if actual[key] != qa.EXPECTED[key]:
            raise RuntimeError(f"Expected count mismatch: {key}")
    statistics.verify_statistics(actual, audit.load_json(DATASET / "metadata/dataset_statistics.json"))
    hashes = {}
    file_stats = {}
    print("Final read-only pre-freeze validation: all images and labels...", flush=True)
    def inspect(record):
        image = DATASET / record["final_image"]
        label = DATASET / integrity.label_relative(record)
        before = (qa.signature(image), qa.signature(label))
        result = integrity.inspect_record(record, decode_image=True)
        if before != (qa.signature(image), qa.signature(label)):
            raise RuntimeError("Active file changed during pre-freeze validation")
        return record, result, before
    with ThreadPoolExecutor(max_workers=12) as executor:
        for offset in range(0, len(records), 4096):
            for record, result, stamps in executor.map(inspect, records[offset:offset + 4096]):
                image_path, label_path = record["final_image"], integrity.label_relative(record)
                hashes[image_path], hashes[label_path] = result["sha256"], result["label_sha256"]
                file_stats[image_path], file_stats[label_path] = stamps
            print(f"Validated {min(offset + 4096, len(records))}/{len(records)}", flush=True)
    for sample in samples["samples"]:
        if hashes[sample["final_image"]] != sample["image_sha256"] or hashes[sample["label"]] != sample["label_sha256"]:
            raise RuntimeError("Reviewed representative sample changed since Work 4.8")
    leaks = duplicate_audit.leakage(records)
    if any(leaks.values()) or leaks != previous["leakage"]:
        raise RuntimeError("Leakage differs from Work 4.8")
    if any(audit.digest_file(ROOT / path) != expected for path, expected in baseline.items()):
        raise RuntimeError("Pre-freeze metadata changed while validating")
    config = portable_yaml()
    resolved = verify_yaml(config, DATASET)
    human = {"status": "PASS", "date": DATE, "reviewer": "Project human reviewer (user)", "scope": "Reviewed representative stratified Work 4.8 QA samples; not all 153174 images and not all 6010 retained REAL_CONFLICT groups", "observations": ["bbox localization visually acceptable/tight", "target classes visually correct", "no abnormal duplicate annotations observed", "no obvious missing target vehicles observed"], "evidence_attachment": str(REQUEST), "evidence_sha256": audit.digest_file(REQUEST), "qa_package_sha256": samples["qa_package_sha256"], "qa_sample_package_count": samples["sample_count"], "individual_sample_verdicts": "Not supplied; no per-image review result is fabricated"}
    policy = audit.load_json(DATASET / "metadata/applied_duplicate_policy.json")
    scripts = sorted(ROOT.glob("*.py"))
    release = {"dataset_name": "Vehicle363", "version": "Vehicle363-v1", "freeze_date": DATE, "status": "FROZEN", "purpose": "4-class vehicle detection for Project 363", "image_count": actual["images"], "label_count": actual["images"], "bbox_count": actual["boxes"], "class_mapping": mapping["schema"], "source_class_mapping": mapping, "class_counts": actual["class_counts"], "split_counts": actual["per_split"], "source_counts": actual["per_source"], "human_qa_status": "PASS", "automated_qa_status": "PASS", "human_qa": human, "duplicate_resolution_policy": {"applied_auto_groups": policy["applied_auto_resolution_groups"], "retained_real_conflict_groups": policy["real_conflict_retained"], "retained_real_conflict_status": "UNRESOLVED; KEEP AS-IS; no canonical selection or annotation rewrite", "policy_file": "metadata/applied_duplicate_policy.json", "annotation_union_or_averaging": False}, "leakage_results": leaks, "source_provenance": {"HCMC": {"workspace_source_root": "data/archive (1)", "domains": ["daytime", "nighttime"], "license_status": "UNKNOWN — MANUAL REVIEW REQUIRED"}, "MIO": {"workspace_source_root": "data/MIO-TCD-Localization/MIO-TCD-Localization", "active_source_partition": "train", "original_test": "No local ground truth; not included as labeled active images", "license_status": "CC BY-NC-SA 4.0 according to local documentation"}, "combined_redistribution_claimed": False}, "audit_trail": {"workspace_relative_location": "data/vehicle_audit", "preserved": True, "quarantine_transaction": policy["transaction"], "work_4_8_qa": "data/vehicle_audit/final_qa", "work_4_9_report": "data/vehicle_audit/final/work_4_9_freeze_report.json"}, "generation_and_audit_scripts": [{"workspace_relative_path": str(path.relative_to(ROOT)), "sha256": audit.digest_file(path)} for path in scripts], "fingerprint": {"algorithm": "SHA-256 of exact UTF-8 LF FILE_HASHES.sha256 bytes", "file_manifest": "FILE_HASHES.sha256", "fingerprint_file": "DATASET_FINGERPRINT.sha256", "entry_format": "lowercase sha256, two spaces, POSIX relative path, LF; lexicographic path order", "scope": "Every active image/label plus authoritative metadata/config/document files, including this release manifest", "self_reference_exclusions": sorted(CONTROLS)}, "immutable_rule": "Never modify images, labels, split assignments or class mapping in place; use Vehicle363-v1.1 or Vehicle363-v2", "immutability_enforcement": "Release policy, SHA-256 verification, local build/apply/rollback guards. No OS ACL/chmod applied to shared hardlink inodes.", "future_direction_classes": "Reserved for future dataset/version; not included in v1", "yolo_training_ready": True, "yolo_config": {"official": "dataset.yaml", "compatibility": "data.yaml", "root_resolution": "Omit optional path; YAML file parent is dataset root", "resolved_relative_directories": resolved, "official_loader_reference": YOLO_REFERENCE, "ultralytics_runtime_executed": False, "readiness_scope": "Configuration/schema/directories/pairing/classes validated; no training runtime installation required or training started"}, "trained_yolo": False}
    final = audit.load_json(DATASET / "final_qa_report.json")
    final.update({"status": "PASS", "human_qa_status": "PASS", "human_qa": human, "automated_qa_status": "PASS", "release_status": "FROZEN", "release_manifest": "RELEASE_MANIFEST.json", "next": "Work 5 — Temporal / Tracking Dataset Audit. DO NOT TRAIN YET."})
    final["visual_qa_manifest"]["final_dataset_qa"] = "PASS — representative Work 4.8 QA samples only"
    staged = {DATASET / "dataset.yaml": config, DATASET / "data.yaml": config, DATASET / "RELEASE_MANIFEST.json": json_bytes(release), DATASET / "README.md": readme(actual).encode(), DATASET / "final_qa_report.json": json_bytes(final), DATASET / "WORK_4_7_SUMMARY.txt": (f"Vehicle363-v1 STATUS: FROZEN\nWORK 4.9: PASS\nIMAGES: {actual['images']}\nBOXES: {actual['boxes']}\nHUMAN QA: PASS (representative samples only)\nREAL_CONFLICT: 6010 retained unresolved KEEP AS-IS\nAuthoritative release: RELEASE_MANIFEST.json\nNEXT: Work 5 — Temporal / Tracking Dataset Audit\nDO NOT TRAIN YET.\n").encode()}
    staged[DATASET / "README.md"] = staged[DATASET / "README.md"].replace(b"../../vehicle_audit/", b"../vehicle_audit/")
    backup = OUTPUT / "pre_freeze_backup"
    backup.mkdir(parents=True, exist_ok=True)
    originals = {path: path.read_bytes() if path.exists() else None for path in staged}
    for path, content in originals.items():
        if content is not None:
            (backup / path.name).write_bytes(content)
    controls = [DATASET / relative for relative in CONTROLS]
    try:
        for path, content in staged.items():
            path.write_bytes(content)
        for path in authoritative_paths():
            hashes[path.relative_to(DATASET).as_posix()] = audit.digest_file(path)
        for relative, expected in file_stats.items():
            if qa.signature(DATASET / relative) != expected:
                raise RuntimeError("Active image/label changed before freeze commit")
        content = "".join(f"{hashes[relative]}  {relative}\n" for relative in sorted(hashes)).encode("utf-8")
        (DATASET / "FILE_HASHES.sha256").write_bytes(content)
        fingerprint = hashlib.sha256(content).hexdigest()
        (DATASET / "DATASET_FINGERPRINT.sha256").write_bytes(f"{fingerprint}  FILE_HASHES.sha256\n".encode())
        checked, count = verify_release()
        if checked != fingerprint:
            raise RuntimeError("Post-release fingerprint verification failed")
        human_path = work8 / "human_qa_confirmation.json"
        save_json(human_path, human)
        report = {"work": "4.9", "status": "PASS", "dataset": "Vehicle363-v1", "dataset_status": "FROZEN", "freeze_date": DATE, "human_qa_status": "PASS", "human_qa": human, "automated_qa_status": "PASS", "actual": actual, "expected_matches_work_4_8": True, "validation": {**pairing, "label_count": len(records), "invalid_yolo_rows": 0, "invalid_class_ids": 0, "non_finite_coordinates": 0, "invalid_normalized_bounds": 0, "empty_labels": 0, "orphan_files": 0, "unreadable_images": 0, "image_hash_mismatches": 0, "all_active_images_decoded": True, "all_image_label_hashes_verified_after_release": True, "generated_boundary_floating_point_tolerance": 1e-12}, "leakage": leaks, "dataset_fingerprint": fingerprint, "fingerprint_manifest_entries": count, "authoritative_metadata_sha256": {relative: digest for relative, digest in hashes.items() if not relative.startswith(("images/", "labels/"))}, "yolo_training_ready": True, "yolo_runtime_executed": False, "audit_trail": "PRESERVED", "real_conflict_groups": 6010, "real_conflict_status": "RETAINED UNRESOLVED KEEP AS-IS", "active_images_or_labels_modified": False, "split_or_class_mapping_modified": False, "source_datasets_modified": False, "quarantine_restored": False, "pre_freeze_input_sha256": baseline, "next": "Vehicle detection dataset branch COMPLETE. Proceed to Work 5 — Temporal / Tracking Dataset Audit. DO NOT TRAIN YET."}
        save_json(OUTPUT / "work_4_9_freeze_report.json", report)
        (OUTPUT / "work_4_9_freeze_report.md").write_text("\n".join(["# Work 4.9 — Freeze Vehicle363-v1", "", "WORK 4.9 STATUS: PASS", "DATASET STATUS: FROZEN", "HUMAN QA: PASS — representative samples only, 2026-10-04", "AUTOMATED QA: PASS", "", f'Images/labels: {actual["images"]:,}; boxes: {actual["boxes"]:,}.', *[f'- {name}: {actual["class_counts"][name]:,}' for name in audit.NAMES], "", *[f'- {split}: {entry["images"]:,} images / {entry["boxes"]:,} boxes' for split, entry in actual["per_split"].items()], "", f'Leakage: {leaks}.', "", f'Fingerprint: `{fingerprint}`; sorted file manifest entries: {count:,}. Includes every active image/label and authoritative metadata/config/documents; excludes only the two self-referential hash-control files.', "", "All active images/labels were freshly validated, and full release hashes rechecked after metadata/config staging. Expected Work 4.8 properties matched. No active image/label, split, source dataset or class map was modified. All 6,010 REAL_CONFLICT groups remain unresolved and unchanged; no quarantine copy was restored.", "", "Human QA confirmation is recorded at ../final_qa/human_qa_confirmation.json. This attests to the representative stratified review, not every image. Work 4.8's original automated audit/manifest remain historical evidence; per-sample verdicts were not fabricated.", "", f'YOLO_TRAINING_READY: YES — portable config/directory/schema/pairing checks, using [official path-resolution semantics]({YOLO_REFERENCE}). Ultralytics runtime is not installed/executed. DO NOT TRAIN YET.', "", "AUDIT TRAIL: PRESERVED. License limitations and sampled near-duplicate limitations remain documented. Immutability is enforced by release policy/fingerprint and local mutation guards, not shared-hardlink chmod/ACL changes.", "", report["next"], ""]), encoding="utf-8")
        print(f"WORK 4.9 STATUS: PASS\nDATASET: Vehicle363-v1\nSTATUS: FROZEN\nHUMAN QA: PASS\nAUTOMATED QA: PASS\nIMAGES: {actual['images']}\nBOXES: {actual['boxes']}\nCLASS COUNTS: {actual['class_counts']}\nSPLITS: " + str({split: {key: entry[key] for key in ('images','boxes')} for split, entry in actual['per_split'].items()}) + f"\nLEAKAGE: {leaks}\nDATASET FINGERPRINT: {fingerprint}\nYOLO_TRAINING_READY: YES\nAUDIT TRAIL: PRESERVED\nNEXT: {report['next']}")
    except BaseException:
        for path, original in originals.items():
            if original is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(original)
        for path in controls:
            if path.exists():
                path.unlink()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify:
        print("RELEASE VERIFICATION PASS:", verify_release())
    elif (DATASET / "RELEASE_MANIFEST.json").exists():
        raise RuntimeError("Release already exists; use --verify. Never regenerate a frozen release in place.")
    else:
        try:
            main()
        except (RuntimeError, AssertionError, KeyError, ValueError, OSError) as error:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            save_json(OUTPUT / "work_4_9_freeze_report.json", {"work": "4.9", "status": "REVIEW_REQUIRED", "dataset_status": "NOT_FROZEN", "automated_qa_status": "FAIL", "human_qa_status": "PASS", "discrepancy": str(error), "active_images_or_labels_modified": False})
            (OUTPUT / "work_4_9_freeze_report.md").write_text(f"# Work 4.9\n\nSTATUS: REVIEW_REQUIRED\nDATASET: NOT_FROZEN\n\n{error}\n\nInvestigate discrepancy; do not freeze or train.\n", encoding="utf-8")
            raise
