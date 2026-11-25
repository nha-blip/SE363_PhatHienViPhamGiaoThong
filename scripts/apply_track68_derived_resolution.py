import json
import math
from datetime import datetime, timezone
from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256, verify_snapshot


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/Tracking363_working/sequences/vntraffic_001"
OUTPUT = ROOT / "data/Tracking363_working/gt_revisions/vntraffic_001_remove_track68"
KEY = (383, 68)


def row_key(line):
    columns = line.decode("utf-8").strip().split(",")
    return int(columns[0]), int(columns[1])


def validate(lines):
    identities = set()
    tracks = set()
    for line in lines:
        values = [float(value) for value in line.decode("utf-8").strip().split(",")]
        if len(values) != 10 or not all(math.isfinite(value) for value in values):
            raise ValueError("Invalid MOT row")
        frame, identity, left, top, width, height = values[:6]
        if frame != int(frame) or identity != int(identity) or not 1 <= frame <= 501 or identity <= 0:
            raise ValueError("Invalid frame/track ID")
        if left < 0 or top < 0 or width <= 0 or height <= 0 or left + width > 1920 or top + height > 1080:
            raise ValueError("Invalid bbox bounds")
        if (frame, identity) in identities:
            raise ValueError("Duplicate frame/track key")
        identities.add((frame, identity))
        tracks.add(identity)
    return len(tracks)


def main():
    snapshot = json.loads((ROOT / "runs/project363/tracking/work_5_3_integration_snapshot.json").read_text(encoding="utf-8"))
    verify_snapshot(snapshot, ROOT)
    gt = BASE / "gt/gt.txt"
    annotations = BASE / "metadata/annotations.jsonl"
    before_gt, before_annotations = sha256(gt), sha256(annotations)
    original = gt.read_bytes().splitlines(keepends=True)
    internal = annotations.read_bytes().splitlines(keepends=True)
    removed_gt = [line for line in original if row_key(line) == KEY]
    removed_internal = [line for line in internal if (json.loads(line)["mot_frame_id"], json.loads(line)["track_id"]) == KEY]
    if len(original) != 16035 or len(internal) != 16035 or len(removed_gt) != 1 or len(removed_internal) != 1:
        raise ValueError("Unexpected baseline or deletion scope")
    removed_record = json.loads(removed_internal[0])
    if removed_record["bbox_xywh"] != [0.0, 684.0, 133.0, 62.0] or removed_record["source_frame_id"] != 382:
        raise ValueError("Approved annotation differs from baseline")
    retained = [line for line in original if row_key(line) != KEY]
    retained_internal = [line for line in internal if (json.loads(line)["mot_frame_id"], json.loads(line)["track_id"]) != KEY]
    if validate(retained) != 84 or len(retained) != 16034:
        raise ValueError("Unexpected derived totals")
    if set(map(row_key, retained)) != {(json.loads(line)["mot_frame_id"], json.loads(line)["track_id"]) for line in retained_internal}:
        raise ValueError("Internal/MOT key mismatch")
    if any(json.loads(line)["class"] is not None for line in retained_internal):
        raise ValueError("Semantic classes must remain PENDING")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "gt").mkdir()
    (OUTPUT / "metadata").mkdir()
    (OUTPUT / "gt/gt.txt").write_bytes(b"".join(retained))
    (OUTPUT / "metadata/annotations.jsonl").write_bytes(b"".join(retained_internal))
    if (OUTPUT / "gt/gt.txt").read_bytes().splitlines(keepends=True) != retained or (OUTPUT / "metadata/annotations.jsonl").read_bytes().splitlines(keepends=True) != retained_internal:
        raise RuntimeError("Derived output mismatch")
    verify_snapshot(snapshot, ROOT)
    if sha256(gt) != before_gt or sha256(annotations) != before_annotations:
        raise RuntimeError("Baseline GT modified")
    report = {"status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "approval_basis": "Owner explicitly approved removal after confirming white car ID 74 and black car ID 72",
              "supersedes": "Earlier target-ID68 bbox proposal is rejected; no replacement bbox is applied",
              "base_sequence": str(BASE), "derived_revision": str(OUTPUT),
              "image_directory": str(BASE / "img1"), "frame_map": str(BASE / "metadata/frame_map.csv"),
              "sequence_id": "vntraffic_001", "source_frames": [0, 500], "mot_frames": [1, 501], "image_frames": 501,
              "original_gt_sha256": before_gt, "original_annotations_sha256": before_annotations,
              "derived_gt_sha256": sha256(OUTPUT / "gt/gt.txt"), "derived_annotations_sha256": sha256(OUTPUT / "metadata/annotations.jsonl"),
              "source_rows": 16035, "derived_rows": 16034, "source_tracks": 85, "derived_tracks": 84,
              "removed_rows": 1, "removed_mot_row": removed_gt[0].decode().strip(), "removed_internal_record": removed_record,
              "preserved_frame_track_keys": [[383, 74], [383, 72]], "all_other_rows_byte_identical": True,
              "duplicate_frame_track_keys": 0, "invalid_bbox_count": 0, "baseline_snapshot_unchanged": True,
              "semantic_class_status": "PENDING", "classes_applied": False,
              "human_class_review_note": "85-track review is historical; ID 68's approval is excluded from the 84-track derived revision",
              "track30_decision": "KEEP: owner confirmed motorcycle occlusion within car bbox",
              "raw_source_modified": False, "baseline_gt_modified": False, "new_ids_created": False,
              "images_copied_or_modified": False, "bytetrack_executed": False, "final_dataset_created": False,
              "dataset_role": "PILOT_DEV", "full_dataset_quality_claim": False,
              "known_limitations": "Existing GT gaps and unresolved camera overlap remain; correction does not certify exhaustive bbox completeness"}
    (OUTPUT / "metadata/resolution_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text("# VNTraffic derived GT revision\n\nRemoved only source frame 382 / MOT frame 383 / track ID 68, a redundant oversized bbox, with explicit owner approval. White car remains ID 74; black car remains ID 72. All other GT rows are byte-identical to the original pilot. Track 30 remains unchanged after human clarification of valid occlusion.\n\n16034 annotations, 84 tracks, 501 existing frames. Semantic classes remain null/PENDING; approved CSV classes were not applied. This is a GT-only revision: images and frame_map are referenced from the original vntraffic_001 sequence, not copied. The revision is not a standalone sequence directory; consumers must pair this gt/gt.txt with the base img1 and frame_map paths recorded in metadata/resolution_manifest.json. Original raw and approved pilot GT remain intact; historical QA reports are not overwritten. No split, final dataset, new ID, gap repair or ByteTrack run.\n", encoding="utf-8")
    print(json.dumps({name: report[name] for name in ("status", "derived_revision", "removed_rows", "derived_rows", "derived_tracks", "baseline_snapshot_unchanged", "semantic_class_status")}, indent=2))


if __name__ == "__main__":
    main()
