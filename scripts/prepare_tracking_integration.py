import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_tracking_sources import ROOT, inventory
from src.tracking.io import load_frame_map, sha256, snapshot_hash, verify_snapshot


SEQUENCE = ROOT / "data/Tracking363_working/sequences/vntraffic_001"
REPORTS = ROOT / "runs/project363/tracking"


def content(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def contract_schema(tracks=False):
    observation = {"type": "object", "additionalProperties": False,
                   "required": ["bbox_xyxy", "confidence", "class_id", "class_name"],
                   "properties": {"bbox_xyxy": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0}},
                                  "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                  "class_id": {"type": "integer", "minimum": 0},
                                  "class_name": {"enum": ["car", "motorcycle", "bus", "truck"]}}}
    if tracks:
        observation["required"].append("track_id")
        observation["properties"]["track_id"] = {"type": "integer", "minimum": 1}
    field = "tracks" if tracks else "detections"
    properties = {"sequence_id": {"type": "string", "minLength": 1},
                  "source_frame_id": {"type": "integer", "minimum": 0},
                  "mot_frame_id": {"type": "integer", "minimum": 1},
                  "width": {"type": "integer", "minimum": 1}, "height": {"type": "integer", "minimum": 1},
                  "fps": {"type": "number", "exclusiveMinimum": 0},
                  "sequence_length": {"type": "integer", "minimum": 1},
                  field: {"type": "array", "items": observation}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "TrackingResult" if tracks else "DetectionResult",
            "type": "object", "additionalProperties": False, "required": list(properties), "properties": properties,
            "$comment": "Runtime checks additionally enforce mot=source+1, finite coordinates, positive extent, image bounds, original class metadata and per-frame unique track IDs. Empty arrays are valid."}


def main():
    print("Verifying pilot/input provenance and explicit owner human-QA approval...", flush=True)
    before = inventory()
    original_manifest = json.loads((SEQUENCE / "metadata/provenance_manifest.json").read_text(encoding="utf-8"))
    protected_hashes = {**original_manifest["input_sha256"], **original_manifest["derived_artifacts_sha256"]}
    for path, expected in protected_hashes.items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"Original source or immutable pilot changed: {path}")
    metadata = json.loads((SEQUENCE / "metadata/sequence.json").read_text(encoding="utf-8"))
    machine = json.loads((SEQUENCE / "qa/machine_qa_report.json").read_text(encoding="utf-8"))
    approval_path = SEQUENCE / "qa/human_review.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval["human_qa_status"] != "PASS" or approval.get("approval_basis") != "explicit owner confirmation in Work 5.3 request":
        raise RuntimeError("Explicit owner human-QA approval required")
    if machine["status"] != "PASS" or (machine["pilot_frames"], machine["gt_observations"], machine["track_count"]) != (501, 16035, 85):
        raise RuntimeError("Pilot machine QA mismatch")
    if any(machine[name] for name in ("duplicate_frame_track_keys", "invalid_bbox", "out_of_bounds_bbox")) or not machine["work_5_1b_corrections_preserved"]:
        raise RuntimeError("Pilot geometry/correction integrity failed")
    frames = load_frame_map(SEQUENCE / "metadata/frame_map.csv", metadata)
    if len(frames) != 501 or metadata["semantic_class_status"] != "PENDING" or metadata["dataset_role"] != "PILOT_DEV":
        raise RuntimeError("Pilot role/semantic policy changed")
    for filename, tracks in (("detection_result.schema.json", False), ("tracking_result.schema.json", True)):
        path = ROOT / "configs/tracking" / filename
        desired = content(contract_schema(tracks))
        if path.exists() and path.read_text(encoding="utf-8") != desired:
            raise FileExistsError(f"Existing contract differs: {path}")
        if not path.exists():
            path.write_text(desired, encoding="utf-8")
    files = sorted((SEQUENCE / "img1").glob("*.jpg"))
    if [path.name for path in files] != [f"{index:06d}.jpg" for index in range(1, 502)]:
        raise RuntimeError("Pilot image set mismatch")
    files.extend(SEQUENCE / path for path in ("gt/gt.txt", "metadata/frame_map.csv", "metadata/sequence.json", "metadata/annotations.jsonl", "qa/human_review.json", "qa/track_gap_report.json"))
    payload = {"work": "5.3", "sequence_id": "vntraffic_001", "dataset_role": "PILOT_DEV", "pilot_frames": 501,
               "gt_observations": 16035, "tracks": 85, "human_qa_status": "PASS", "approved_by": "project owner",
               "approval_basis": "explicit owner confirmation", "manual_review_frame_ids": None,
               "exhaustive_review_claimed": False, "semantic_gt_classes": "PENDING", "tracks_with_internal_gaps": 55,
               "total_internal_missing_observations": 3878, "track_gap_repair": "NONE", "excluded_source_frames": list(range(501, 510)),
               "files_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in files},
               "snapshot_scope": "Integration input snapshot only, not final Tracking363-v1 or an unbiased benchmark"}
    snapshot = {"hash_algorithm": "SHA-256 of canonical payload JSON (sorted keys, compact separators, UTF-8)",
                "snapshot_hash": snapshot_hash(payload), "payload": payload}
    snapshot_path = REPORTS / "work_5_3_integration_snapshot.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text(encoding="utf-8")) != snapshot:
        raise FileExistsError("Approved integration snapshot differs; create an explicitly versioned snapshot instead of overwriting")
    if not snapshot_path.exists():
        snapshot_path.write_text(content(snapshot), encoding="utf-8")
    verify_snapshot(snapshot, ROOT)
    test_xml = REPORTS / "work_5_3_tests.xml"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    print("Running integration tests with stub backend only; no YOLO/ByteTrack execution...", flush=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/tracking", "-q", "-p", "no:cacheprovider", f"--junitxml={test_xml}"], cwd=ROOT, env=environment, capture_output=True, text=True)
    (REPORTS / "work_5_3_tests.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError("Integration scaffold tests failed; inspect work_5_3_tests.log")
    suites = ElementTree.parse(test_xml).getroot()
    cases = suites.findall(".//testcase")
    if not cases or suites.findall(".//failure") or suites.findall(".//error") or suites.findall(".//skipped"):
        raise RuntimeError("Test evidence is empty, failed or skipped")
    for path, expected in protected_hashes.items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"Protected input changed during tests: {path}")
    verify_snapshot(snapshot, ROOT)
    if inventory() != before:
        raise RuntimeError("Data workspace changed during integration snapshot/testing")
    scaffold_paths = sorted(path for directory in (ROOT / "src/tracking", ROOT / "configs/tracking", ROOT / "tests/tracking") for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    report = {"work": "5.3", "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "human_qa_status": "PASS", "approved_by": "project owner", "approval_basis": "explicit owner confirmation",
              "manual_review_frame_ids": None, "exhaustive_501_frame_review_claimed": False,
              "pilot_machine_qa_status": "PASS", "sequence": "vntraffic_001", "dataset_role": "PILOT_DEV",
              "frames": 501, "gt_observations": 16035, "tracks": 85, "semantic_gt_classes": "PENDING",
              "integration_snapshot": snapshot_path.relative_to(ROOT).as_posix(), "integration_snapshot_hash": snapshot["snapshot_hash"],
              "integration_snapshot_file_sha256": sha256(snapshot_path), "snapshot_file_count": len(files),
              "schema_test_status": "PASS", "frame_mapping_test_status": "PASS", "empty_detection_test_status": "PASS",
              "vehicle_filter_test_status": "PASS", "test_count": len(cases), "test_exit_code": result.returncode,
              "integration_contract": "READY", "yolo_adapter": "READY", "bytetrack_adapter": "READY",
              "adapter_readiness_scope": "Interface/scaffold tested with fake Results and stub backend only; actual Ultralytics runtime not exercised",
              "real_backend_runtime_verified": False, "detector_checkpoint_selected": False, "execution_enabled": False,
              "raw_data_modified": False, "gt_modified": False, "yolo_executed": False, "bytetrack_executed": False,
              "models_trained": False, "track_gap_repair": "NONE", "frames_501_to_509_included": False,
              "final_tracking363_v1_created": False, "split_created": False,
              "input_and_immutable_pilot_sha256_reverified": True, "data_inventory_unchanged_during_tests": True,
              "authorized_qa_change": "Derived human_review.json records owner PASS; GT/images/immutable construction QA unchanged",
              "scaffold_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in scaffold_paths},
              "known_limitations": metadata["known_limitations"],
              "next": "Work 5.4 — Run YOLO -> ByteTrack on vntraffic_001 after selecting checkpoint and explicitly enabling execution"}
    (REPORTS / "work_5_3_report.json").write_text(content(report), encoding="utf-8")
    markdown = ["# Work 5.3 — Human QA closeout and integration preparation", "", "WORK 5.3 STATUS: PASS", "WORK 5.2 MACHINE QA: PASS", "HUMAN QA: PASS — explicit project-owner approval; supersedes previous pilot human-QA gate REVIEW_REQUIRED.", "", "Exact inspected frame IDs and exhaustive 501-frame inspection are NOT claimed. Existing construction manifests are historical immutable evidence; the derived human_review.json and this report carry the new approval.", "", "## Pilot and snapshot", "", "Sequence: vntraffic_001; role PILOT_DEV; 501 frames, 16035 observations, 85 tracks. Semantic GT classes remain PENDING.", f"Integration snapshot hash: `{snapshot['snapshot_hash']}`", f"Snapshot: `{snapshot_path.relative_to(ROOT).as_posix()}`", f"Hashed files: {len(files)} (501 extracted frames plus GT, mapping, sequence metadata, internal annotations, human approval and structural gap audit).", "", "## Integration contracts", "", "- DetectionResult -> pixel bbox_xyxy, confidence, original class_id/class_name and source/MOT frame IDs.", "- TrackingResult -> predicted track ID, pixel bbox_xyxy, score and original detector class metadata. Predicted IDs are not GT IDs.", "- Filter only car/motorcycle/bus/truck by checkpoint names; traffic-light predictions excluded without changing unified IDs.", "- Empty detections remain valid and still update the injected tracker once per frame.", "- No silent clipping, semantic GT assignment, gap interpolation or cross-sequence tracker state.", "- YOLO/ByteTrack adapters READY means interface preparation; real backend/checkpoint execution has NOT been tested or run.", "- Model path/device remain null. execution_enabled=false. Work 5.4 must verify snapshot, select/hash checkpoint, check/pin actual Ultralytics API and explicitly authorize execution.", "", "## Tests", "", f"{len(cases)} tests passed. Schema round trips, bbox conversions, native-frame mapping, empty detections, temporal-order guards, original unified-ID filtering and immutable source/pilot hashes checked. Only fake detector outputs and a stub backend used.", "Test log: `runs/project363/tracking/work_5_3_tests.log`", "", "## Known limitations retained", "", *[f"- {item}" for item in metadata["known_limitations"]], "", "## Protection and scope", "", "RAW DATA MODIFIED: NO. GT MODIFIED: NO. YOLO EXECUTED: NO. BYTETRACK EXECUTED: NO. No training, tuning, interpolation, split or final Tracking363-v1.", "Only the derived human-QA record changed for owner approval; snapshot/code/config/tests and reports were generated. Source and immutable pilot hashes verified before/after tests. This is not a fresh full frozen dataset payload rehash.", "", "Primary interface references: [Ultralytics Results](https://docs.ultralytics.com/reference/engine/results/), [ByteTrack API](https://docs.ultralytics.com/reference/trackers/byte_tracker/).", "", report["next"], ""]
    (REPORTS / "work_5_3_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(content({"WORK 5.3 STATUS": "PASS", "WORK 5.2 MACHINE QA": "PASS", "HUMAN QA": "PASS", "SEQUENCE": "vntraffic_001",
                   "FRAMES": 501, "GT OBSERVATIONS": 16035, "TRACKS": 85, "SEMANTIC GT CLASSES": "PENDING",
                   "INTEGRATION CONTRACT": "READY", "YOLO ADAPTER": "READY", "BYTETRACK ADAPTER": "READY (scaffold only)",
                   "INTEGRATION SNAPSHOT HASH": snapshot["snapshot_hash"], "TESTS PASSED": len(cases),
                   "RAW SOURCE MODIFIED": "NO", "GT MODIFIED": "NO", "YOLO EXECUTED": "NO", "BYTETRACK EXECUTED": "NO", "NEXT": report["next"]}), flush=True)


if __name__ == "__main__":
    main()
