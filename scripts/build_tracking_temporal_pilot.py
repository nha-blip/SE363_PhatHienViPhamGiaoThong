import csv
import hashlib
import html
import io
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_tracking_sources import ROOT, digest, inventory
from scripts.build_vntraffic_working_gt import parse_lines, validate


SEQUENCE_ID = "vntraffic_001"
DESTINATION = ROOT / "data/Tracking363_working/sequences" / SEQUENCE_ID
BASELINE = ROOT / "data/Tracking363_working/VNTraffic"
REPORTS = ROOT / "runs/project363/tracking"
VIDEO = ROOT / "data/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4"


def relative(path):
    return path.relative_to(ROOT).as_posix()


def json_content(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def save(path, content):
    if DESTINATION.resolve() not in path.resolve().parents:
        raise RuntimeError("Sequence output escaped authorized directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content.encode("utf-8") if isinstance(content, str) else content)


def runs_inventory():
    result = {}
    for directory, children, files in os.walk(ROOT / "runs", followlinks=False):
        for name in children + files:
            path = Path(directory) / name
            information = path.stat(follow_symlinks=False)
            if path.is_symlink() or getattr(information, "st_file_attributes", 0) & 0x400:
                raise RuntimeError(f"Unexpected linked run artifact: {path}")
            if path.is_file():
                result[relative(path)] = (information.st_size, information.st_mtime_ns)
    return result


def gap_audit(rows):
    tracks = defaultdict(set)
    for row in rows:
        tracks[int(row[1])].add(int(row[0]))
    gaps = []
    for identity, frames in sorted(tracks.items()):
        ordered = sorted(frames)
        ranges = [{"start_frame": before + 1, "end_frame": after - 1,
                   "source_start_frame": before, "source_end_frame": after - 2,
                   "missing_count": after - before - 1, "classification": "UNCLASSIFIED_STRUCTURAL_GAP"}
                  for before, after in zip(ordered, ordered[1:]) if after - before > 1]
        if ranges:
            gaps.append({"track_id": identity, "first_frame": ordered[0], "last_frame": ordered[-1],
                         "source_first_frame": ordered[0] - 1, "source_last_frame": ordered[-1] - 1,
                         "observed_frames": len(ordered), "observed_mot_frame_ids": ordered,
                         "missing_internal_frames": sum(item["missing_count"] for item in ranges),
                         "gap_ranges": ranges})
    return {"frame_indexing": "first_frame/last_frame/gap start/end are MOT 1-based; source_* fields are 0-based",
            "classification_policy": "Structure only; causes unknown. No claim of occlusion, annotation error or tracking error.",
            "gap_repair": "NONE", "track_count": len(tracks), "tracks_with_internal_gaps": len(gaps),
            "total_internal_missing_observations": sum(item["missing_internal_frames"] for item in gaps), "tracks": gaps}


def draw_preview(frame, rows, source_frame, focus=None):
    image = frame.copy()
    for row in rows:
        identity = int(row[1])
        left, top, width, height = map(int, row[2:6])
        color = (40, 255, 40)
        if identity == focus:
            color = (255, 60, 255)
        if source_frame == 2 and identity == 31 or source_frame == 208 and identity == 44:
            color = (40, 40, 255)
        elif source_frame == 208 and identity == 49:
            color = (255, 160, 0)
        cv2.rectangle(image, (left, top), (left + width, top + height), color, 2)
        cv2.putText(image, f"ID {identity}", (left, max(20, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    label = f"PILOT_DEV | source {source_frame} | MOT {source_frame + 1} | class PENDING"
    if focus is not None:
        present = any(int(row[1]) == focus for row in rows)
        label += f" | gap QA ID {focus}: {'GT present' if present else 'no GT row'}"
    cv2.putText(image, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return image


def verify_preview_video(path):
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not capture.isOpened():
            raise RuntimeError("QA video cannot be reopened")
        while True:
            success, frame = capture.read()
            if not success:
                break
            if (frame.shape[1], frame.shape[0]) != (1280, 720):
                raise RuntimeError("Unexpected QA video resolution")
            count += 1
    finally:
        capture.release()
    if count != 501:
        raise RuntimeError(f"QA video decoded {count} frames, expected 501")
    return count


def main():
    if DESTINATION.exists():
        raise FileExistsError("Pilot directory already exists; refuse to overwrite. Inspect existing outputs before any rerun.")
    print("Checking source and Work 5.1B evidence before extraction...", flush=True)
    data_before = inventory()
    runs_before = runs_inventory()
    baseline_report_path = ROOT / "runs/project363/tracking_audit/work_5_1b_report.json"
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if baseline_report["status"] != "PASS":
        raise RuntimeError("Work 5.1B is not PASS")
    provenance_path = BASELINE / "metadata/provenance_diff_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "runs/project363/tracking_audit/work_5_1_report.json").read_text(encoding="utf-8"))
    input_paths = {ROOT / path for path in prior["protected_and_track_annotation_sha256_after"]}
    input_paths.update(ROOT / path for path in provenance["raw_source_sha256"])
    input_paths.update(path for path in BASELINE.rglob("*") if path.is_file())
    input_paths.add(baseline_report_path)
    input_hashes = {relative(path): digest(path) for path in sorted(input_paths)}
    for path, expected in provenance["raw_source_sha256"].items():
        if input_hashes[path] != expected:
            raise RuntimeError("Raw source hash differs from Work 5.1B")
    gt_path = BASELINE / "gt/gt.txt"
    internal_path = BASELINE / "gt/annotations.jsonl"
    if input_hashes[relative(gt_path)] != provenance["working_gt_sha256"] or input_hashes[relative(gt_path)] != baseline_report["working_gt_hash"]:
        raise RuntimeError("Working GT hash mismatch")
    if input_hashes[relative(internal_path)] != provenance["internal_annotations_sha256"]:
        raise RuntimeError("Working internal annotations hash mismatch")
    source_lines = gt_path.read_bytes().splitlines(keepends=True)
    working_rows = parse_lines(source_lines)
    internal = [json.loads(line) for line in internal_path.read_text(encoding="utf-8").splitlines()]
    if len(working_rows) != 16035 or len(internal) != len(working_rows):
        raise RuntimeError("Baseline observation count mismatch")
    expected_corrected_rows = {provenance["changes"][0]["kept_bbox_A_source_row_1based"], provenance["changes"][1]["source_row_1based"]}
    derived_records = []
    pilot_lines = []
    mot_rows = []
    per_frame = defaultdict(list)
    for parsed, line, record in zip(working_rows, source_lines, internal):
        values = parsed["values"]
        source_frame = int(values[0])
        if not 0 <= source_frame <= 500:
            raise RuntimeError("Annotation outside approved pilot range")
        if [record["frame_id"], record["track_id"], *record["bbox_xywh"], record["confidence"]] != values[:7] or record["class"] is not None:
            raise RuntimeError("Internal record differs from corrected working GT")
        if record["human_corrected"] != (record["source_row_1based"] in expected_corrected_rows):
            raise RuntimeError("Human-corrected flag lacks Work 5.1B provenance")
        converted = dict(record)
        converted.pop("frame_id")
        converted.update(source_frame_id=source_frame, mot_frame_id=source_frame + 1,
                         working_gt=relative(gt_path), working_gt_sha256=provenance["working_gt_sha256"])
        derived_records.append(converted)
        pilot_lines.append(str(source_frame + 1).encode("ascii") + b"," + line.partition(b",")[2])
        mot_rows.append([source_frame + 1, *values[1:]])
        per_frame[source_frame].append(values)
    input_validation = validate(working_rows, {"resolution": [1920, 1080], "decoded_frames": 510})
    if input_validation["duplicate_frame_track_keys"] or any(input_validation[key] for key in ("invalid_bbox_count", "invalid_frame_count", "invalid_track_id_count")):
        raise RuntimeError("Baseline structural QA failed")
    gaps = gap_audit(mot_rows)
    if (gaps["track_count"], gaps["tracks_with_internal_gaps"], gaps["total_internal_missing_observations"]) != (85, 55, 3878):
        raise RuntimeError("Baseline track/gap counts differ; no gap repair is allowed")
    sampled = {0: ["beginning"], 1: ["frame2_neighbor"], 2: ["frame2_approved_correction"], 3: ["frame2_neighbor"],
               207: ["frame208_neighbor"], 208: ["frame208_approved_correction"], 209: ["frame208_neighbor"],
               250: ["middle"], 499: ["end_neighbor"], 500: ["end"]}
    densest = max(per_frame, key=lambda frame: (len(per_frame[frame]), -frame))
    sampled.setdefault(densest, []).append("dense_traffic")
    gap_examples = sorted([(gap["missing_count"], track["track_id"], gap) for track in gaps["tracks"] for gap in track["gap_ranges"]], key=lambda item: (-item[0], item[1], item[2]["start_frame"]))[:3]
    focus = {}
    for _, identity, gap in gap_examples:
        for source_frame in (gap["source_start_frame"] - 1, (gap["source_start_frame"] + gap["source_end_frame"]) // 2, gap["source_end_frame"] + 1):
            sampled.setdefault(source_frame, []).append(f"structural_gap_track_{identity}")
            focus.setdefault(source_frame, identity)
    for directory in ("img1", "gt", "metadata", "qa/previews"):
        (DESTINATION / directory).mkdir(parents=True, exist_ok=False)
    save(DESTINATION / "gt/gt.txt", b"".join(pilot_lines))
    save(DESTINATION / "metadata/annotations.jsonl", "\n".join(json.dumps(record, ensure_ascii=False) for record in derived_records) + "\n")
    save(DESTINATION / "qa/track_gap_report.json", json_content(gaps))
    gap_markdown = ["# Structural track gap audit", "", "Frame ranges below are MOT 1-based; source frame = MOT frame - 1.", "Classification: UNCLASSIFIED_STRUCTURAL_GAP. Causes unknown; no repair/interpolation.", "", "| Track | First | Last | Observed frames | Missing internal observations | Missing MOT ranges |", "|---|---:|---:|---:|---:|---|"]
    for track in gaps["tracks"]:
        ranges = "; ".join(f"{gap['start_frame']}..{gap['end_frame']}" for gap in track["gap_ranges"])
        gap_markdown.append(f"| {track['track_id']} | {track['first_frame']} | {track['last_frame']} | {track['observed_frames']} | {track['missing_internal_frames']} | {ranges} |")
    save(DESTINATION / "qa/track_gap_report.md", "\n".join(gap_markdown) + "\n")
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError("Original video cannot be opened")
    video_info = {"reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                  "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    if video_info != {"reported_frames": 510, "fps": 30.0, "width": 1920, "height": 1080}:
        capture.release()
        raise RuntimeError(f"Unexpected original video metadata: {video_info}")
    preview_path = DESTINATION / "qa/pilot_preview.mp4"
    writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("QA video encoder unavailable; no PASS report generated")
    frame_map = []
    image_manifest = []
    tiles = []
    qa_samples = []
    decoded = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            source_frame = decoded
            decoded += 1
            if frame.shape[:2] != (1080, 1920):
                raise RuntimeError("Unexpected decoded frame size")
            if source_frame > 500:
                continue
            mot_frame = source_frame + 1
            filename = f"{mot_frame:06d}.jpg"
            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise RuntimeError(f"JPEG encoding failed at source frame {source_frame}")
            content = encoded.tobytes()
            save(DESTINATION / "img1" / filename, content)
            frame_map.append({"source_frame_id": source_frame, "mot_frame_id": mot_frame, "image_filename": filename,
                              "source_video": relative(VIDEO), "timestamp_seconds": f"{source_frame / 30:.9f}"})
            image_manifest.append({"source_frame_id": source_frame, "mot_frame_id": mot_frame, "image_filename": filename,
                                   "sha256": hashlib.sha256(content).hexdigest(),
                                   "decoded_source_bgr_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            annotated = draw_preview(frame, per_frame[source_frame], source_frame, focus.get(source_frame))
            writer.write(cv2.resize(annotated, (1280, 720)))
            if source_frame in sampled:
                preview_filename = f"source_{source_frame:06d}_mot_{mot_frame:06d}.jpg"
                success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 93])
                if not success:
                    raise RuntimeError("QA preview JPEG encoding failed")
                save(DESTINATION / "qa/previews" / preview_filename, encoded.tobytes())
                thumbnail = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                thumbnail.thumbnail((630, 355))
                tiles.append((thumbnail, source_frame, mot_frame, sampled[source_frame]))
                qa_samples.append({"source_frame_id": source_frame, "mot_frame_id": mot_frame, "categories": sampled[source_frame],
                                   "annotation_count": len(per_frame[source_frame]), "focus_track_id": focus.get(source_frame), "preview": "previews/" + preview_filename})
            if mot_frame % 100 == 0:
                print(f"Extracted {mot_frame}/501 approved pilot frames", flush=True)
    finally:
        capture.release()
        writer.release()
    if decoded != 510 or len(frame_map) != 501:
        raise RuntimeError(f"Unexpected decoded/extracted frame counts: {decoded}/{len(frame_map)}")
    preview_decoded = verify_preview_video(preview_path)
    csv_stream = io.StringIO(newline="")
    csv_writer = csv.DictWriter(csv_stream, fieldnames=["source_frame_id", "mot_frame_id", "image_filename", "source_video", "timestamp_seconds"], lineterminator="\n")
    csv_writer.writeheader()
    csv_writer.writerows(frame_map)
    save(DESTINATION / "metadata/frame_map.csv", csv_stream.getvalue())
    save(DESTINATION / "metadata/extraction_manifest.json", json_content({"image_encoding": "OpenCV JPEG quality 95; lossy derived pixels, no resize", "opencv_version": cv2.__version__, "timestamp_origin": "source_frame_id / 30, derived index/FPS, NOT original capture wall-clock", "frames": image_manifest}))
    columns = 3
    sheet = Image.new("RGB", (1920, 55 + math.ceil(len(tiles) / columns) * 415), "#eeeeee")
    drawing = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    drawing.text((12, 12), "vntraffic_001 | PILOT_DEV | source 0-based -> MOT 1-based | classes PENDING", fill="black", font=label_font)
    for index, (thumbnail, source_frame, mot_frame, categories) in enumerate(tiles):
        left, top = (index % columns) * 640 + 5, 55 + (index // columns) * 415
        sheet.paste(thumbnail, (left, top))
        drawing.text((left, top + 357), f"source {source_frame} -> MOT {mot_frame}", fill="black", font=label_font)
        drawing.text((left, top + 380), ", ".join(categories)[:63], fill="black", font=label_font)
    sheet.save(DESTINATION / "qa/contact_sheet.jpg", quality=93)
    save(DESTINATION / "qa/preview_manifest.json", json_content({"human_qa_status": "REVIEW_REQUIRED", "sampling": "Deterministic beginning/middle/end, approved correction neighbors, earliest densest frame and three largest structural gaps", "samples": qa_samples}))
    cards = ''.join(f'<article><h3>Source {item["source_frame_id"]} → MOT {item["mot_frame_id"]}: {html.escape(", ".join(item["categories"]))}</h3><a href="{item["preview"]}"><img loading="lazy" src="{item["preview"]}" alt="annotated QA frame"></a></article>' for item in qa_samples)
    save(DESTINATION / "qa/index.html", '<!doctype html><html lang="en"><meta charset="utf-8"><title>Tracking pilot QA</title><style>body{font-family:Arial;margin:24px;background:#eee}article{background:white;padding:12px;margin:16px 0}img{width:100%}</style><h1>vntraffic_001 — PILOT_DEV — HUMAN QA REQUIRED</h1><p>IDs are existing TXT track IDs, NOT vehicle classes. Gap causes are unknown; no missing boxes are drawn. Frame 2 ID31 red; frame208 ID44 red and ID49 blue. Open images at full resolution. MP4 preview uses mp4v; if your browser cannot play it, download it to a compatible player.</p><a href="contact_sheet.jpg">Contact sheet</a> | <a href="pilot_preview.mp4">QA video</a>' + cards + '</html>')
    filenames = sorted(path.name for path in (DESTINATION / "img1").iterdir())
    expected_filenames = [f"{frame:06d}.jpg" for frame in range(1, 502)]
    if filenames != expected_filenames:
        raise RuntimeError("Extracted filenames are not exactly 000001.jpg..000501.jpg")
    for item in image_manifest:
        path = DESTINATION / "img1" / item["image_filename"]
        if digest(path) != item["sha256"]:
            raise RuntimeError("Persisted image SHA mismatch")
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (1080, 1920):
            raise RuntimeError("Extracted JPEG unreadable or resized")
    persisted_rows = [item["values"] for item in parse_lines((DESTINATION / "gt/gt.txt").read_bytes().splitlines(keepends=True))]
    if persisted_rows != mot_rows or any(converted[1:] != original["values"][1:] or converted[0] != original["values"][0] + 1 for converted, original in zip(persisted_rows, working_rows)):
        raise RuntimeError("GT conversion altered fields beyond the frame offset")
    duplicate_keys = [key for key, count in Counter((row[0], row[1]) for row in mot_rows).items() if count > 1]
    invalid_boxes = sum(not all(math.isfinite(value) for value in row[2:6]) or row[4] <= 0 or row[5] <= 0 for row in mot_rows)
    out_of_bounds = sum(row[2] < 0 or row[3] < 0 or row[2] + row[4] > 1920 or row[3] + row[5] > 1080 for row in mot_rows)
    corrections_preserved = ([row[2:6] for row in mot_rows if row[:2] == [3, 31]] == [[1697, 776, 55, 83]]
                             and [row[2:6] for row in mot_rows if row[:2] == [209, 44]] == [[682, 683, 54, 46]]
                             and [row[2:6] for row in mot_rows if row[:2] == [209, 49]] == [[712, 710, 60, 57]])
    with (DESTINATION / "metadata/frame_map.csv").open(newline="", encoding="utf-8") as handle:
        persisted_map = list(csv.DictReader(handle))
    map_valid = len(persisted_map) == 501 and all(int(item["source_frame_id"]) == index and int(item["mot_frame_id"]) == index + 1 and item["image_filename"] == expected_filenames[index] and abs(float(item["timestamp_seconds"]) - index / 30) < 1e-8 for index, item in enumerate(persisted_map))
    observation_frames_exist = all(1 <= row[0] <= 501 and (DESTINATION / "img1" / f"{int(row[0]):06d}.jpg").is_file() for row in mot_rows)
    if duplicate_keys or invalid_boxes or out_of_bounds or not corrections_preserved or not map_valid or not observation_frames_exist:
        raise RuntimeError("Pilot machine QA failed")
    machine_qa = {"status": "PASS", "pilot_frames": len(filenames), "filename_continuity_valid": True,
                  "frame_map_rows": len(persisted_map), "frame_map_valid": map_valid,
                  "duplicate_mot_frame_ids": len(persisted_map) - len({item["mot_frame_id"] for item in persisted_map}),
                  "duplicate_frame_track_keys": len(duplicate_keys), "invalid_bbox": invalid_boxes, "out_of_bounds_bbox": out_of_bounds,
                  "track_count": len({row[1] for row in mot_rows}), "gt_observations": len(mot_rows),
                  "all_observations_map_to_existing_frames": observation_frames_exist, "index_conversion_valid": True,
                  "non_frame_gt_fields_preserved": True, "work_5_1b_corrections_preserved": corrections_preserved,
                  "all_semantic_classes_null": all(record["class"] is None for record in derived_records),
                  "human_corrected_flags_exactly_inherited": True, "human_corrected_observations": sum(record["human_corrected"] for record in derived_records),
                  "tracks_with_internal_gaps": gaps["tracks_with_internal_gaps"], "total_internal_missing_observations": gaps["total_internal_missing_observations"],
                  "track_gap_repair": "NONE", "source_video_decoded_frames": decoded, "qa_video_decoded_frames": preview_decoded,
                  "human_qa_status": "REVIEW_REQUIRED"}
    limitations = ["Semantic classes remain null/PENDING", "55 tracks have 3878 internal missing observations; causes unknown, no interpolation",
                   "Source frames 501..509 excluded, not labeled empty/background", "Complete physical identity and bbox coverage are not certified by structural QA",
                   "Potential HCMC detector-training camera overlap remains unresolved; not an unbiased final benchmark",
                   "PILOT_DEV only; one atomic camera/video/sequence unit, no train/val/test split", "JPEG frames and QA video are lossy derived encodings"]
    sequence = {"sequence_id": SEQUENCE_ID, "dataset_role": "PILOT_DEV", "source_name": "VNTraffic",
                "source_video": relative(VIDEO), "source_video_sha256": input_hashes[relative(VIDEO)],
                "source_gt_sha256": provenance["source_gt_sha256"], "working_gt_sha256": provenance["working_gt_sha256"],
                "source_annotation": provenance["source_gt"], "annotation_baseline": relative(gt_path), "annotation_source_policy": "Corrected Work 5.1B TXT only",
                "fps": 30, "width": 1920, "height": 1080, "source_video_total_frames": decoded,
                "pilot_source_start_frame": 0, "pilot_source_end_frame": 500, "pilot_frame_count": 501,
                "frame_indexing_source": "0-based", "frame_indexing_mot": "1-based; mot_frame_id = source_frame_id + 1",
                "timestamp_policy": "source_frame_id / 30.0; derived index/FPS timestamp, not original capture wall-clock",
                "track_count": 85, "annotation_count": 16035, "semantic_class_status": "PENDING",
                "bbox_units": "Native source pixel xywh; coordinates unchanged, only frame IDs offset",
                "mot_column_meaning": ["mot_frame_id", "existing_txt_track_id", "left", "top", "width", "height", "confidence", "world_x_placeholder", "world_y_placeholder", "world_z_placeholder"],
                "placeholder_policy": "Final three -1 fields are unprovided world coordinates, NOT semantic vehicle class IDs; adapt explicitly for a specific evaluator",
                "atomic_group_id": "VNTraffic/original_video/vntraffic_001", "known_limitations": limitations,
                "models_executed": [], "final_tracking363_v1_created": False, "split_created": False}
    save(DESTINATION / "metadata/sequence.json", json_content(sequence))
    save(DESTINATION / "qa/machine_qa_report.json", json_content(machine_qa))
    save(DESTINATION / "seqinfo.ini", "[Sequence]\nname=vntraffic_001\nimDir=img1\nframeRate=30\nseqLength=501\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n")
    save(DESTINATION / "metadata/README.md", "# vntraffic_001 — temporal pilot\n\nDataset role: PILOT_DEV. Not final Tracking363-v1 or an unbiased train/val/test benchmark.\n\nSource frames 0..500 -> MOT 1..501. Frame 501..509 from the original video are excluded; they are not empty/background annotations. Timestamps derive from source frame index / 30, not original capture wall-clock. Bbox coordinates, existing TXT IDs and confidence are unchanged from Work 5.1B. Final three -1 columns are world-coordinate placeholders, NOT semantic classes. Every internal class remains null/PENDING.\n\nNo gap repair, track merge/split, new IDs, or model execution. Keep this entire camera/video/sequence as one group. See sequence.json, extraction_manifest.json, frame_map.csv, annotations.jsonl and ../qa/track_gap_report.json. Human QA is required: open ../qa/index.html and inspect corrections, sequence endpoints, dense traffic and gap examples. If the MP4 browser codec is unsupported, use a desktop video player.\n")
    output_hashes = {relative(path): digest(path) for path in sorted(DESTINATION.rglob("*")) if path.is_file()}
    save(DESTINATION / "metadata/provenance_manifest.json", json_content({"work": "5.2", "input_sha256": input_hashes,
         "derived_artifacts_sha256": output_hashes, "index_change": "Only frame IDs +1; bbox/track/confidence unchanged",
         "working_provenance": relative(provenance_path), "source_to_pilot_mapping": "metadata/frame_map.csv",
         "human_corrected_flag_policy": "Exact inheritance verified against Work 5.1B provenance source-row identifiers",
         "encoding": "Sequential full source decode; extract 0..500 as native JPEG quality95; excluded 501..509; no source modification"}))
    inputs_after = {relative(path): digest(path) for path in sorted(input_paths)}
    if inputs_after != input_hashes:
        raise RuntimeError("Protected/source/Work 5.1B input bytes changed")
    print("Checking all existing data and run files remain unchanged...", flush=True)
    data_after = inventory()
    prefix = relative(DESTINATION) + "/"
    unauthorized = [path for path in data_before.keys() | data_after.keys() if not path.startswith(prefix) and data_before.get(path) != data_after.get(path)]
    if unauthorized or runs_inventory() != runs_before:
        raise RuntimeError(f"Unexpected existing data/run changes: {unauthorized[:10]}")
    report = {"work": "5.2", "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "sequence": SEQUENCE_ID, "dataset_role": "PILOT_DEV", "source_frames": decoded, "pilot_frames": 501,
              "source_gt_rows": baseline_report["source_row_count"], "working_gt_rows": len(working_rows), "pilot_gt_rows": len(mot_rows),
              "track_count": 85, "semantic_classes": "PENDING", "frame_index": "SOURCE 0-BASED -> MOT 1-BASED",
              "duplicate_frame_track_keys": len(duplicate_keys), "invalid_bbox": invalid_boxes, "out_of_bounds_bbox": out_of_bounds,
              "tracks_with_internal_gaps": gaps["tracks_with_internal_gaps"], "total_internal_missing_observations": gaps["total_internal_missing_observations"],
              "frame_map_valid": map_valid, "index_conversion_valid": True, "work_5_1b_corrections_preserved": corrections_preserved,
              "track_gap_repair": "NONE", "raw_source_modified": False, "working_gt_modified": False,
              "bytetrack_executed": False, "yolo_executed": False, "vlm_executed": False, "training_executed": False,
              "final_tracking363_v1_created": False, "split_created": False, "human_qa_status": "REVIEW_REQUIRED",
              "sequence_path": relative(DESTINATION), "qa_location": relative(DESTINATION / "qa/index.html"),
              "qa_sample_count": len(qa_samples), "input_sha256_before": input_hashes, "input_sha256_after": inputs_after,
              "data_files_checked_before": len(data_before), "data_files_checked_after": len(data_after),
              "existing_run_files_checked_unchanged": len(runs_before), "fresh_full_frozen_payload_hash_audit": False,
              "pilot_gt_sha256": digest(DESTINATION / "gt/gt.txt"), "sequence_metadata_sha256": digest(DESTINATION / "metadata/sequence.json"),
              "known_limitations": limitations, "next": "Work 5.3 — Human QA of temporal pilot + prepare detector/ByteTrack integration"}
    REPORTS.mkdir(parents=True, exist_ok=True)
    for path in (REPORTS / "work_5_2_report.json", REPORTS / "work_5_2_report.md"):
        if path.exists():
            raise FileExistsError("Refuse to overwrite existing Work 5.2 report")
    (REPORTS / "work_5_2_report.json").write_text(json_content(report), encoding="utf-8")
    lines = ["# Work 5.2 — Tracking363 temporal pilot", "", "WORK 5.2 STATUS: PASS", "SEQUENCE: vntraffic_001", "DATASET ROLE: PILOT_DEV", "HUMAN QA STATUS: REVIEW_REQUIRED (Work 5.3)", "", "## Machine validation", "", *[f"- {key}: {report[key]}" for key in ("source_frames", "pilot_frames", "source_gt_rows", "working_gt_rows", "pilot_gt_rows", "track_count", "semantic_classes", "frame_index", "duplicate_frame_track_keys", "invalid_bbox", "out_of_bounds_bbox", "tracks_with_internal_gaps", "total_internal_missing_observations", "frame_map_valid", "index_conversion_valid", "work_5_1b_corrections_preserved", "track_gap_repair", "raw_source_modified", "working_gt_modified")], "", "Only frame IDs change: source 0 -> MOT 1; source 2 -> MOT 3; source 208 -> MOT 209; source 500 -> MOT 501. Timestamp = source frame / 30, not original capture wall-clock. JPEGs are native-resolution lossy derived outputs.", "", "Approved corrections survive: MOT frame3 ID31 has only bbox A [1697,776,55,83]; MOT frame209 A is ID44 [682,683,54,46] and B is ID49 [712,710,60,57]. Existing IDs, bbox fields and confidence unchanged from working GT. All classes null; last three -1 fields mean world-coordinate placeholders, not vehicle semantics.", "", "## QA and artifacts", "", f"- Sequence: `{relative(DESTINATION)}`", f"- Manual QA: `{report['qa_location']}`", f"- Contact sheet: `{relative(DESTINATION / 'qa/contact_sheet.jpg')}`", f"- QA video: `{relative(preview_path)}` (501 decoded QA frames, 1280x720, 30fps, mp4v)", f"- QA samples: {len(qa_samples)}", f"- Full gap audit: `{relative(DESTINATION / 'qa/track_gap_report.json')}`", f"- Provenance/hashes: `{relative(DESTINATION / 'metadata/provenance_manifest.json')}`", "", "## Known limitations", "", *[f"- {item}" for item in limitations], "", "## Source protection", "", "Hashes before/after match for raw source files, Work 5.1B GT/internal metadata and existing protected release controls. Data size/mtime inventory confirms no changes outside the new sequence. All existing run artifacts, including any local YOLO training outputs, remain unchanged. Not a new full frozen payload rehash.", "", "No model, tracking tuning, split or final dataset created. PASS is sequence-construction/machine-QA status, not automatic human QA or benchmark approval.", "", report["next"], ""]
    (REPORTS / "work_5_2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json_content({"WORK 5.2 STATUS": "PASS", "SEQUENCE": SEQUENCE_ID, "DATASET ROLE": "PILOT_DEV",
                        "PILOT FRAMES": 501, "GT OBSERVATIONS": 16035, "TRACKS": 85, "SEMANTIC CLASSES": "PENDING",
                        "FRAME INDEX": report["frame_index"], "DUPLICATE FRAME/TRACK KEYS": len(duplicate_keys),
                        "INVALID BBOX": invalid_boxes, "TRACK GAP REPAIR": "NONE", "RAW SOURCE MODIFIED": "NO",
                        "BYTETrack EXECUTED": "NO", "FINAL TRACKING363-v1 CREATED": "NO", "NEXT": report["next"]}), flush=True)


if __name__ == "__main__":
    main()
