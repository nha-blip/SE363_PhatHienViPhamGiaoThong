import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_tracking_sources import ROOT, digest, inventory


SOURCE = ROOT / "data/Vehicle_Tracking/VNTraffic"
WORKING = ROOT / "data/Tracking363_working/VNTraffic"
AUDIT = ROOT / "runs/project363/tracking_audit"
SEQUENCE = "vntraffic_001"


def relative(path):
    return path.relative_to(ROOT).as_posix()


def parse_lines(lines):
    rows = []
    for number, line in enumerate(lines, 1):
        values = list(map(float, next(csv.reader([line.decode("utf-8").strip()]))))
        if len(values) != 10 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid source row {number}")
        if values[0] != int(values[0]) or values[1] != int(values[1]):
            raise ValueError(f"Noninteger frame/track at row {number}")
        rows.append({"source_row_1based": number, "values": values})
    return rows


def decode_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Original video cannot be opened")
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    resolution = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    decoded = 0
    timestamps = []
    unexpected_resolutions = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            unexpected_resolutions += [frame.shape[1], frame.shape[0]] != resolution
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
            decoded += 1
    finally:
        capture.release()
    return {"reported_frames": reported, "decoded_frames": decoded, "fps": fps,
            "resolution": resolution, "unexpected_frame_resolutions": unexpected_resolutions,
            "metadata_frame_count_minus_decoded": reported - decoded,
            "timestamp_nonfinite_count": sum(not math.isfinite(value) for value in timestamps),
            "timestamp_non_increasing_pairs": sum(after <= before for before, after in zip(timestamps, timestamps[1:])),
            "scope": "Full sequential in-memory decode; no video/frame copies or extraction"}


def validate(rows, video):
    counts = Counter((int(row["values"][0]), int(row["values"][1])) for row in rows)
    tracks = defaultdict(set)
    invalid_boxes = []
    invalid_frames = []
    invalid_tracks = []
    width, height = video["resolution"]
    for row in rows:
        frame, identity, left, top, box_width, box_height = row["values"][:6]
        frame, identity = int(frame), int(identity)
        tracks[identity].add(frame)
        if not 0 <= frame < video["decoded_frames"]:
            invalid_frames.append(row["source_row_1based"])
        if identity <= 0:
            invalid_tracks.append(row["source_row_1based"])
        if not (left >= 0 and top >= 0 and box_width > 0 and box_height > 0 and left + box_width <= width and top + box_height <= height):
            invalid_boxes.append(row["source_row_1based"])
    summaries = {}
    for identity, frames in sorted(tracks.items()):
        ordered = sorted(frames)
        gaps = [{"from_frame": before, "to_frame": after, "unannotated_internal_frames": after - before - 1}
                for before, after in zip(ordered, ordered[1:]) if after - before > 1]
        summaries[str(identity)] = {"first_frame": ordered[0], "last_frame": ordered[-1],
                                   "unique_annotated_frames": len(ordered), "gaps": gaps,
                                   "unannotated_internal_frame_count": sum(gap["unannotated_internal_frames"] for gap in gaps)}
    all_frames = sorted({frame for frames in tracks.values() for frame in frames})
    return {"row_count": len(rows), "duplicate_frame_track_keys": [
                {"frame_id": frame, "track_id": identity, "count": count}
                for (frame, identity), count in sorted(counts.items()) if count > 1],
            "duplicate_excess_rows": sum(count - 1 for count in counts.values() if count > 1),
            "invalid_bbox_count": len(invalid_boxes), "invalid_bbox_source_rows": invalid_boxes,
            "invalid_frame_count": len(invalid_frames), "invalid_frame_source_rows": invalid_frames,
            "invalid_track_id_count": len(invalid_tracks), "invalid_track_source_rows": invalid_tracks,
            "track_count": len(tracks), "frame_range": [all_frames[0], all_frames[-1]],
            "frames_with_gt": len(all_frames),
            "video_frames_without_gt_rows": sorted(set(range(video["decoded_frames"])) - set(all_frames)),
            "tracks": summaries,
            "continuity_interpretation": "Internal annotation gaps are reported, not filled; they can reflect occlusion or incomplete GT. Geometric/temporal checks do not certify every physical identity."}


def put(path, content):
    if path.resolve() != WORKING.resolve() and WORKING.resolve() not in path.resolve().parents:
        raise RuntimeError("Derived output escaped authorized working area")
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Existing derived file differs; do not overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main():
    before = inventory()
    source_path = SOURCE / "VNTraffic_GroundTruth.txt"
    video_path = SOURCE / "VNTraffic_Original-video.mp4"
    review_path = AUDIT / "work_5_1/human_review.json"
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)
    decisions = {item["check"]: item for item in review["observations"]}
    first_decision = decisions["VNTraffic frame 2 ID 31"]
    second_decision = decisions["VNTraffic frame 208 ID 49"]
    if first_decision.get("approved_bbox_to_keep") != "A" or (second_decision.get("vehicle_A_existing_id"), second_decision.get("vehicle_B_existing_id")) != (44, 49):
        raise RuntimeError("Recorded human decisions do not match authorized Work 5.1B request")
    prior = json.loads((AUDIT / "work_5_1_report.json").read_text(encoding="utf-8"))
    protected = prior["protected_and_track_annotation_sha256_after"]
    protected_before = {path: digest(ROOT / path) for path in protected}
    if protected_before != protected:
        raise RuntimeError("Prior protected/source evidence changed; stop before creating derived GT")
    raw_files = sorted(path for path in SOURCE.iterdir() if path.is_file())
    raw_hashes = {relative(path): digest(path) for path in raw_files}
    preview_manifest = json.loads((AUDIT / "work_5_1/conflict_qa/qa_manifest.json").read_text(encoding="utf-8"))
    for path, expected in raw_hashes.items():
        if preview_manifest["source_sha256_after"].get(path) != expected:
            raise RuntimeError("Raw VNTraffic file differs from reviewed QA evidence")
    lines = source_path.read_bytes().splitlines(keepends=True)
    source_rows = parse_lines(lines)
    correction_rows = {}
    expected_pairs = {(2, 31): [[1697, 776, 55, 83], [1684, 800, 44, 68]],
                      (208, 49): [[682, 683, 54, 46], [712, 710, 60, 57]]}
    for (frame, identity), expected in expected_pairs.items():
        matched = [row for row in source_rows if row["values"][:2] == [frame, identity]]
        if [row["values"][2:6] for row in matched] != expected:
            raise RuntimeError("Source A/B row order or geometry differs from human-reviewed boxes")
        sample = next(item for item in preview_manifest["samples"] if item["category"] == "VNTraffic conflict" and item["source_frame"] == frame)
        if [item["box"] for item in sample["txt_objects"]] != expected:
            raise RuntimeError("Preview A/B provenance mismatch")
        correction_rows[frame] = matched
    kept = correction_rows[2][0]["source_row_1based"]
    removed = correction_rows[2][1]["source_row_1based"]
    reassigned = correction_rows[208][0]["source_row_1based"]
    retained = correction_rows[208][1]["source_row_1based"]
    working_lines = []
    working_rows = []
    for row, line in zip(source_rows, lines):
        number = row["source_row_1based"]
        if number == removed:
            continue
        if number == reassigned:
            fields = line.decode("utf-8").split(",")
            fields[1] = "44"
            line = ",".join(fields).encode("utf-8")
        values = list(row["values"])
        if number == reassigned:
            values[1] = 44
        working_lines.append(line)
        working_rows.append({"source_row_1based": number, "values": values})
    video = decode_video(video_path)
    if video["decoded_frames"] != 510 or video["metadata_frame_count_minus_decoded"] != 0 or video["unexpected_frame_resolutions"] or video["timestamp_nonfinite_count"] or video["timestamp_non_increasing_pairs"]:
        raise RuntimeError(f"Original video frame integrity mismatch: {video}")
    source_qa = validate(source_rows, video)
    working_qa = validate(working_rows, video)
    if working_qa["duplicate_frame_track_keys"] or any(working_qa[key] for key in ("invalid_bbox_count", "invalid_frame_count", "invalid_track_id_count")):
        raise RuntimeError(f"Derived validation failed: {working_qa}")
    source_tracks = {int(row["values"][1]) for row in source_rows}
    working_tracks = {int(row["values"][1]) for row in working_rows}
    if source_tracks != working_tracks or len(working_rows) != len(source_rows) - 1:
        raise RuntimeError("Unexpected track namespace or row-count change")
    if source_qa["video_frames_without_gt_rows"] != working_qa["video_frames_without_gt_rows"]:
        raise RuntimeError("Unexpected frame-coverage change")
    source_counts = Counter(tuple(row["values"]) for row in source_rows)
    working_counts = Counter(tuple(row["values"]) for row in working_rows)
    old_A = tuple(correction_rows[208][0]["values"])
    new_A = (old_A[0], 44, *old_A[2:])
    expected_removed = Counter([tuple(correction_rows[2][1]["values"]), old_A])
    if source_counts - working_counts != expected_removed or working_counts - source_counts != Counter([new_A]):
        raise RuntimeError("Diff includes changes outside the two approved decisions")
    gt_path = WORKING / "gt/gt.txt"
    internal_path = WORKING / "gt/annotations.jsonl"
    internal = []
    for row in working_rows:
        values = row["values"]
        number = row["source_row_1based"]
        internal.append({"sequence_id": SEQUENCE, "frame_id": int(values[0]), "track_id": int(values[1]),
                         "bbox_xywh": values[2:6], "class": None, "class_status": "PENDING",
                         "source": "VNTraffic", "source_annotation": relative(source_path),
                         "source_annotation_sha256": raw_hashes[relative(source_path)],
                         "source_row_1based": number, "human_corrected": number in {kept, reassigned},
                         "human_correction_kind": "duplicate_partner_removed" if number == kept else "existing_id_reassignment" if number == reassigned else None,
                         "confidence": values[6], "world_coordinates_placeholders": values[7:10]})
    put(gt_path, b"".join(working_lines))
    put(internal_path, ("\n".join(json.dumps(item, ensure_ascii=False) for item in internal) + "\n").encode("utf-8"))
    reloaded = parse_lines(gt_path.read_bytes().splitlines(keepends=True))
    if [row["values"] for row in reloaded] != [row["values"] for row in working_rows]:
        raise RuntimeError("Persisted working TXT differs from validated rows")
    changes = [
        {"operation": "REMOVE_DUPLICATE_BBOX_B", "frame_id": 2, "track_id": 31,
         "removed_source_row_1based": removed, "before": correction_rows[2][1]["values"], "after": None,
         "kept_bbox_A_source_row_1based": kept, "kept_bbox_A": correction_rows[2][0]["values"],
         "reason": "Human confirmed one physical vehicle; keep original A (red) without modifying its bbox"},
        {"operation": "REASSIGN_A_TO_EXISTING_TXT_ID", "frame_id": 208,
         "source_row_1based": reassigned, "before": list(old_A), "after": list(new_A),
         "unchanged_bbox_B_source_row_1based": retained, "unchanged_bbox_B": correction_rows[208][1]["values"],
         "reason": "Human confirmed A -> existing ID 44 and B -> existing ID 49; no new track ID"}
    ]
    manifest = {"work": "5.1B", "sequence_id": SEQUENCE, "source": "VNTraffic", "baseline": "TXT namespace",
                "authorization": {"request": "User explicit Work 5.1B derived working GT authorization",
                                  "scope": "Only two recorded human decisions in separate working area; raw edits and class inference prohibited",
                                  "historical_review_flags": "Earlier correction_authorized=false referred to read-only Work 5.1; this explicit new request authorizes only derived-copy changes"},
                "human_review_path": relative(review_path), "human_review_sha256": digest(review_path),
                "preview_manifest_path": relative(AUDIT / "work_5_1/conflict_qa/qa_manifest.json"),
                "preview_manifest_sha256": digest(AUDIT / "work_5_1/conflict_qa/qa_manifest.json"),
                "raw_source_sha256": raw_hashes, "source_gt": relative(source_path),
                "source_gt_sha256": raw_hashes[relative(source_path)], "working_gt": relative(gt_path),
                "working_gt_sha256": digest(gt_path), "internal_annotations": relative(internal_path),
                "internal_annotations_sha256": digest(internal_path), "changes": changes,
                "removed_rows": 1, "id_changed_rows": 1, "added_rows": 0, "bbox_geometry_changed_rows": 0,
                "new_track_ids": [], "source_row_count": len(source_rows), "working_row_count": len(working_rows),
                "frame_index_base": 0, "bbox_units": "native video pixels, left/top/width/height",
                "semantic_class_policy": "null/PENDING for all objects; confidence/world placeholders are not classes",
                "namespace_policy": "Original TXT IDs preserved except approved frame 208 A reassignment to existing TXT ID 44; JSON IDs untouched",
                "raw_source_modified": False}
    qa = {"status": "PASS", "scope": "Derived TXT geometry/ID-key integrity and frame availability, not complete semantic benchmark certification",
          "video": video, "source": source_qa, "working": working_qa,
          "target_track_continuity_before_after": {str(identity): {"before": source_qa["tracks"][str(identity)], "after": working_qa["tracks"][str(identity)]} for identity in (31, 44, 49)},
          "all_classes_null": all(item["class"] is None for item in internal),
          "human_corrected_internal_records": sum(item["human_corrected"] for item in internal),
          "video_frames_missing_for_gt": working_qa["invalid_frame_count"],
          "frames_without_gt_interpretation": "Frames 501..509 have no GT rows; not confirmed empty/background. Do not score them as fully annotated without additional review.",
          "no_gap_interpolation": True, "no_identity_completeness_claim": True}
    put(WORKING / "metadata/provenance_diff_manifest.json", json_bytes(manifest))
    put(WORKING / "metadata/qa_report.json", json_bytes(qa))
    put(WORKING / "metadata/human_review_snapshot.json", review_bytes)
    readme = """# VNTraffic derived working GT — Work 5.1B

Baseline: original TXT track-ID namespace. This is NOT final Tracking363-v1.

`../gt/gt.txt`: original ten-field rows with only approved removal/reassignment.
Frame IDs remain ZERO-BASED; do not feed this directly to a 1-based evaluator without an explicit later conversion. Bboxes remain native pixel xywh, NOT YOLO normalized coordinates. Final three -1 fields are world-coordinate placeholders, not class IDs.

`../gt/annotations.jsonl`: canonical internal records; every class is null/PENDING. Each record points to exact source file hash and original 1-based row number. `human_corrected` marks the retained frame-2 bbox whose duplicate partner was removed and frame-208 A whose ID changed. Frame-208 B was confirmed but not changed.

`provenance_diff_manifest.json`: exact before/after rows, retained partners and authorization scope. `qa_report.json`: frame availability, box validity, track gaps and duplicate-key checks. `human_review_snapshot.json`: the historical read-only decisions; Work 5.1B explicitly authorizes derived-only application.

Frames 0..500 have GT; original video has 510 decoded frames (0..509). Frames 501..509 are UNREVIEWED coverage, not known empty. Internal annotation gaps are not interpolated or claimed incorrect automatically. Class semantics, complete trajectories and detector camera-exposure risk remain pending.

No raw/source edit, video duplication/extraction, new IDs, YOLO/ByteTrack run, AICC22 merge, final dataset, or train/val/test split.
"""
    put(WORKING / "metadata/README.md", readme.encode("utf-8"))
    raw_after = {relative(path): digest(path) for path in raw_files}
    protected_after = {path: digest(ROOT / path) for path in protected}
    if raw_after != raw_hashes or protected_after != protected_before or review_path.read_bytes() != review_bytes:
        raise RuntimeError("Protected/raw/review evidence changed during build")
    after = inventory()
    prefix = relative(WORKING) + "/"
    unauthorized = [path for path in before.keys() | after.keys() if not path.startswith(prefix) and before.get(path) != after.get(path)]
    if unauthorized:
        raise RuntimeError(f"Data files changed outside working area: {unauthorized[:10]}")
    report = {"work": "5.1B", "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "raw_source_modified": False, "source_files_modified": False, "base_gt": "VNTraffic TXT",
              "working_gt_ready": True, "readiness_scope": "Class-pending initial pilot working GT only; not final benchmark",
              "source_gt_hash": manifest["source_gt_sha256"], "working_gt_hash": manifest["working_gt_sha256"],
              "source_row_count": len(source_rows), "working_row_count": len(working_rows),
              "human_corrections_applied": changes, "remaining_duplicate_frame_track_keys": len(working_qa["duplicate_frame_track_keys"]),
              "invalid_bbox_count": working_qa["invalid_bbox_count"], "track_count": working_qa["track_count"],
              "frame_range": working_qa["frame_range"], "semantic_class_status": "PENDING",
              "working_gt_path": relative(gt_path), "internal_annotations_path": relative(internal_path),
              "provenance_diff_manifest": relative(WORKING / "metadata/provenance_diff_manifest.json"),
              "qa_report": relative(WORKING / "metadata/qa_report.json"), "video": video,
              "frames_with_gt": working_qa["frames_with_gt"], "video_frames_without_gt": working_qa["video_frames_without_gt_rows"],
              "track_continuity": {"tracks_with_internal_gaps": sum(bool(track["gaps"]) for track in working_qa["tracks"].values()),
                                   "total_internal_unannotated_frames": sum(track["unannotated_internal_frame_count"] for track in working_qa["tracks"].values()),
                                   "interpretation": working_qa["continuity_interpretation"]},
              "source_sha256_before": raw_hashes, "source_sha256_after": raw_after,
              "protected_release_and_annotation_sha256_unchanged": protected_before == protected_after,
              "data_inventory_files_checked_before": len(before), "data_inventory_files_checked_after": len(after),
              "data_files_modified_outside_working_area": [], "fresh_full_frozen_payload_hash_audit": False,
              "next": "Work 5.2 — Build Tracking363 temporal pilot sequences; retain pending class/coverage/camera-overlap caveats"}
    (AUDIT / "work_5_1b_report.json").write_bytes(json_bytes(report))
    markdown = ["# Work 5.1B — VNTraffic derived working GT", "", "WORK 5.1B STATUS: PASS", "RAW SOURCE MODIFIED: NO", "BASE GT: VNTraffic TXT", "SEMANTIC CLASS STATUS: PENDING (all null)", "WORKING GT READY: YES — class-pending pilot only, not final benchmark", "", "## Integrity summary", "", f"- SOURCE GT HASH: `{report['source_gt_hash']}`", f"- WORKING GT HASH: `{report['working_gt_hash']}`", f"- SOURCE ROW COUNT: {len(source_rows)}", f"- WORKING ROW COUNT: {len(working_rows)}", f"- REMAINING DUPLICATE FRAME/TRACK KEYS: {report['remaining_duplicate_frame_track_keys']}", f"- INVALID BBOX COUNT: {report['invalid_bbox_count']}", f"- TRACK COUNT: {report['track_count']}", f"- FRAME RANGE: {working_qa['frame_range']} (0-based)", f"- ORIGINAL VIDEO: {video['decoded_frames']} fully decoded frames, {video['fps']} fps, {video['resolution']}", "", "## Exact approved changes", "", f"1. Frame 2 / ID 31: keep A red, source row {kept}; remove only B, source row {removed}.", f"2. Frame 208: A, source row {reassigned}, ID 49 -> existing TXT ID 44; B, source row {retained}, keeps ID 49.", "", "No other source row edited. No bbox coordinates changed. No added rows or new IDs. All other TXT bytes and source row order preserved.", "", "```json", json.dumps(changes, indent=2), "```", "", "## Coverage and continuity caveats", "", "GT covers frames 0..500. Video decodes 0..509. Frames 501..509 have no GT rows and are NOT approved empty/background frames.", f"Tracks with internal annotation gaps: {report['track_continuity']['tracks_with_internal_gaps']}. Total missing internal observations: {report['track_continuity']['total_internal_unannotated_frames']}. Full gap lists are in qa_report.json; no interpolation or arbitrary identity reassignment.", "", "PASS is structural readiness for a derived pilot working GT. All four vehicle semantics, complete trajectory/box coverage, and HCMC detector camera-exposure risk remain unresolved; Work 5.1/global benchmark status is not automatically promoted to PASS.", "", "## Outputs", "", f"- Working TXT: `{relative(gt_path)}`", f"- Internal JSONL: `{relative(internal_path)}`", f"- Provenance/diff: `{report['provenance_diff_manifest']}`", f"- QA: `{report['qa_report']}`", "", "## Source protection", "", "All raw VNTraffic file SHA-256 hashes unchanged. Existing frozen release controls/source annotation hashes unchanged; data size/mtime inventory shows no changes outside the authorized derived area. This is not a fresh full frozen payload rehash.", "", "No YOLO/ByteTrack execution, AICC22 merge, new arbitrary IDs, semantic class assignment, final Tracking363-v1 or split created.", "", report["next"], ""]
    (AUDIT / "work_5_1b_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"WORK 5.1B STATUS": "PASS", "RAW SOURCE MODIFIED": "NO", "BASE GT": "VNTraffic TXT",
                      "HUMAN CORRECTIONS APPLIED": ["frame 2 ID 31: keep bbox A", "frame 208: A -> ID 44, B -> ID 49"],
                      "SEMANTIC CLASS STATUS": "PENDING", "SOURCE GT HASH": report["source_gt_hash"],
                      "WORKING GT HASH": report["working_gt_hash"], "SOURCE ROW COUNT": len(source_rows),
                      "WORKING ROW COUNT": len(working_rows), "DUPLICATE FRAME/TRACK KEYS": report["remaining_duplicate_frame_track_keys"],
                      "INVALID BBOX COUNT": report["invalid_bbox_count"], "TRACK COUNT": report["track_count"],
                      "FRAME RANGE": report["frame_range"], "WORKING GT READY": "YES — class-pending pilot only",
                      "NEXT": report["next"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
