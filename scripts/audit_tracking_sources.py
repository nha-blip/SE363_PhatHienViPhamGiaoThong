import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs/project363/tracking_audit"
PREVIEWS = AUDIT / "work_5_1"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROTECTED = {"Vehicle363-v1", "TrafficLight363-v1", "vehicle_audit"}
TARGETS = ["car", "motorcycle", "bus", "truck"]


def relative(path):
    return path.relative_to(ROOT).as_posix()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def inventory():
    files = {}
    stack = [ROOT / "data"]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                information = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(information, "st_file_attributes", 0) & 0x400:
                    raise RuntimeError(f"Unexpected junction/symlink; cannot safely audit: {entry.path}")
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files[relative(path)] = (information.st_size, information.st_mtime_ns)
    return files


def source_name(path):
    parts = path.relative_to(ROOT / "data").parts
    if parts[0] == "Vehicle_Tracking":
        return parts[1]
    return {"archive (1)": "HCMC", "archive": "Kaggle_Traffic_Light", "dataset_train_rgb": "Bosch_RGB"}.get(parts[0], parts[0])


def summary(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else None


def font():
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def contact_sheet(name, panels, note=""):
    filename = re.sub(r"[^A-Za-z0-9_-]", "_", name) + ".jpg"
    destination = PREVIEWS / filename
    if destination.resolve().parent != PREVIEWS.resolve():
        raise RuntimeError("Preview path escaped permitted directory")
    columns = 3
    rows = max(1, math.ceil(len(panels) / columns))
    canvas = Image.new("RGB", (columns * 520, 65 + rows * 335), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), name[:160], fill="black", font=font())
    draw.text((12, 32), note[:170], fill="black", font=font())
    for index, (image, label) in enumerate(panels):
        image = image.copy()
        image.thumbnail((510, 285))
        position = ((index % columns) * 520 + 5, 65 + (index // columns) * 335)
        canvas.paste(image, position)
        draw.text((position[0], position[1] + 287), label[:72], fill="black", font=font())
    canvas.save(destination, quality=88)
    return relative(destination)


def probe_video(path):
    annotated = "GroundTruth-video" in path.name or path.name in {"sequence_tracks_video.mp4", "sequence_overview_video.mp4", "counting_example_cam_5_1min.mp4"}
    identity = {"path": relative(path), "size_bytes": path.stat().st_size, "role": "annotation_visualization_or_overview" if annotated else "primary_video"}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {**identity, "error": "Cannot open video; frame count/FPS/duration unknown"}, []
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if count <= 0 or fps <= 0 or not math.isfinite(fps) or width <= 0 or height <= 0:
        capture.release()
        return {**identity, "error": "Invalid/unknown video metadata; manual probe required"}, []
    starts = sorted({0, max(0, count // 2 - 1), max(0, count - 3)})
    samples = []
    windows = []
    failures = []
    jpeg_sizes = []
    for start in starts:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        hashes = []
        stamps = []
        differences = []
        previous = None
        for offset in range(3):
            if start + offset >= count:
                break
            success, frame = capture.read()
            if not success:
                failures.append(start + offset)
                break
            reported_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(float)
            if previous is not None:
                differences.append(float(np.abs(gray - previous).mean() / 255))
            previous = gray
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            stamps.append(timestamp)
            if offset == 0:
                success_jpeg, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if success_jpeg:
                    jpeg_sizes.append(len(encoded))
            samples.append((start + offset, timestamp, reported_position, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        windows.append({"start_zero_based": start, "timestamps_ms": stamps, "timestamp_non_increasing": sum(after <= before for before, after in zip(stamps, stamps[1:])), "consecutive_exact_duplicate_pairs": sum(before == after for before, after in zip(hashes, hashes[1:])), "normalized_gray_mad": differences, "large_difference_pairs_over_0_25": sum(value > 0.25 for value in differences)})
    capture.release()
    annotated = "GroundTruth-video" in path.name or path.name in {"sequence_tracks_video.mp4", "sequence_overview_video.mp4", "counting_example_cam_5_1min.mp4"}
    record = {"path": relative(path), "size_bytes": path.stat().st_size, "reported_frames": count, "frame_count_method": "OpenCV container metadata, not full decode", "fps": fps, "resolution": [width, height], "duration_seconds_estimated": count / fps if fps else None, "role": "annotation_visualization_or_overview" if annotated else "primary_video", "sampled_windows": windows, "sampled_frames_decoded": len(samples), "sample_decode_failures": failures, "jpeg90_bytes_per_native_frame_sample": summary(jpeg_sizes), "estimated_jpeg90_extracted_bytes": int(count * statistics.median(jpeg_sizes)) if jpeg_sizes else None}
    return record, samples


def decode_short_video(path):
    capture = cv2.VideoCapture(str(path))
    hashes = []
    stamps = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
        stamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
    capture.release()
    return {"decoded_frames": len(hashes), "exact_duplicate_adjacent_pairs": sum(before == after for before, after in zip(hashes, hashes[1:])), "timestamp_non_increasing": sum(after <= before for before, after in zip(stamps, stamps[1:])), "scope": "Full in-memory decode of this SHORT original video only; no extracted frame files", "first_timestamp_ms": stamps[0] if stamps else None, "last_timestamp_ms": stamps[-1] if stamps else None}


def track_annotations(directory, video):
    text_path = next(directory.glob("*GroundTruth.txt"))
    json_path = next(directory.glob("*GroundTruth.json"))
    text_rows = []
    malformed = []
    with text_path.open(newline="") as handle:
        for index, row in enumerate(csv.reader(handle), 1):
            try:
                values = tuple(map(float, row))
                if len(values) != 10 or not all(math.isfinite(value) for value in values):
                    raise ValueError("Expected ten finite fields")
                text_rows.append(values)
            except ValueError:
                malformed.append(index)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    json_rows = []
    mismatches = []
    for identity, track in document.items():
        if len(track["frames"]) != len(track["boxes"]):
            mismatches.append(identity)
        json_rows.extend((int(Path(frame).stem), int(identity), *box) for frame, box in zip(track["frames"], track["boxes"]))
    frame_count = video["reported_frames"]
    width, height = video["resolution"]
    def checks(rows):
        frames = sorted({int(row[0]) for row in rows})
        identities = {int(row[1]) for row in rows}
        zero_valid = all(0 <= frame < frame_count for frame in frames)
        one_valid = all(1 <= frame <= frame_count for frame in frames)
        base = 0 if zero_valid and not one_valid else 1 if one_valid and not zero_valid else None
        pairs = Counter((row[0], row[1]) for row in rows)
        duplicates = [{"frame": int(frame), "existing_track_id": int(identity), "rows": count, "boxes_xywh": [list(row[2:6]) for row in rows if row[0] == frame and row[1] == identity]} for (frame, identity), count in pairs.items() if count > 1]
        gaps = {}
        for identity in identities:
            track_frames = sorted({int(row[0]) for row in rows if int(row[1]) == identity})
            missing = sum(after - before - 1 for before, after in zip(track_frames, track_frames[1:]))
            if missing:
                gaps[str(identity)] = missing
        valid_ids = all(row[0] == int(row[0]) and row[1] == int(row[1]) and row[1] > 0 for row in rows)
        invalid_boxes = sum(not (row[4] > 0 and row[5] > 0 and row[2] >= 0 and row[3] >= 0 and row[2] + row[4] <= width and row[3] + row[5] <= height) for row in rows)
        return {"rows": len(rows), "nonempty_tracks": len(identities), "frames_with_annotations": len(frames), "frame_range": [min(frames), max(frames)] if frames else None, "frame_index_base_inferred_from_endpoints": base, "frame_base_requires_visual_alignment_check": True, "invalid_boxes": invalid_boxes, "invalid_frame_or_track_integers": not valid_ids, "duplicate_frame_id_keys": sum(count - 1 for count in pairs.values()), "duplicate_frame_id_examples": duplicates[:10], "tracks_with_internal_annotation_gaps": gaps, "absence_is_not_proof_of_empty_frames": True}
    text_geometry = Counter((row[0], *row[2:6]) for row in text_rows)
    json_geometry = Counter((row[0], *row[2:6]) for row in json_rows)
    text_geometry_ids = defaultdict(set)
    for row in text_rows:
        text_geometry_ids[(row[0], *row[2:6])].add(int(row[1]))
    matches = defaultdict(set)
    for row in json_rows:
        matches[int(row[1])].update(text_geometry_ids[(row[0], *row[2:6])])
    mapped_values = [next(iter(values)) for values in matches.values() if len(values) == 1]
    result = {"files": [relative(text_path), relative(json_path)], "txt_format": "frame,id,left,top,width,height,confidence,x,y,z (ten-column MOT-style; last three fields are -1, NOT semantic class IDs)", "json_format": "dictionary keyed by existing track ID, each {frames: [image path strings], boxes: [pixel xywh]}", "semantic_classes": "UNKNOWN: neither annotation file assigns car/motorcycle/bus/truck", "txt": checks(text_rows), "json": checks(json_rows), "malformed_txt_rows": malformed, "json_frame_box_length_mismatch_ids": mismatches, "json_empty_track_entries": sum(not track["frames"] for track in document.values()), "json_only_geometry_rows": sum((json_geometry - text_geometry).values()), "txt_only_geometry_rows": sum((text_geometry - json_geometry).values()), "same_geometry_multiset": text_geometry == json_geometry, "json_is_geometry_superset_of_txt": not (text_geometry - json_geometry) and bool(json_geometry - text_geometry), "shared_geometry_id_correspondence_bijective": all(len(values) == 1 for values in matches.values() if values) and len(mapped_values) == len(set(mapped_values)), "existing_id_correspondence_examples_json_to_txt": {str(identity): sorted(values) for identity, values in list(matches.items())[:10]}, "shared_frame_path_prefix": "./train/S01/c002/img1/ appears in both sources; template path is NOT authoritative evidence of shared camera", "decision": "No conversion/remapping/canonical selection performed; human alignment, completeness and per-class QA required"}
    return result, text_rows


def imptc_annotations(root):
    sequences = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        sync_path = directory / "context/master_timestamp_sync.json"
        sync = json.loads(sync_path.read_text())
        timestamps = [int(value) for value in sync]
        deltas = [after - before for before, after in zip(timestamps, timestamps[1:])]
        classes = Counter()
        keys = set()
        tracks = 0
        observations = 0
        unmatched = 0
        for path in sorted(directory.glob("*/*/track.json")):
            data = json.loads(path.read_text())
            classes[data["overview"].get("class_name", "UNKNOWN")] += 1
            tracks += 1
            for row in data["track_data"].values():
                keys.update(row)
                observations += 1
                unmatched += str(row["ts"]) not in sync
        sequences.append({"sequence": directory.name, "master_frames": len(sync), "timestamp_units_inferred": "microseconds from epoch-scale magnitude and median 40000 increment", "timestamp_steps": summary(deltas), "non_increasing_timestamps": sum(value <= 0 for value in deltas), "duplicate_master_frame_ids": len(sync) - len({row['id'] for row in sync.values()}), "fps_inferred": 1e6 / statistics.median(deltas) if deltas else None, "tracks": tracks, "track_observations": observations, "track_class_counts": dict(classes), "track_data_fields": sorted(keys), "track_timestamps_not_in_master": unmatched, "raw_2d_image_bboxes": "NOT FOUND in inspected local track schema; coordinates/cuboids require documented calibration/projection", "videos": [relative(path) for path in sorted(directory.glob("*.mp4"))]})
    return {"format": "Existing IDs in vehicles/<id>/track.json and vrus/<id>/track.json; overview.class_id/class_name, track_data[*].ts, 3D cuboid/coordinates/velocity; context/master_timestamp_sync.json", "sequences": sequences, "class_counts": dict(sum((Counter(item['track_class_counts']) for item in sequences), Counter())), "direct_mot_readiness": "NOT READY: rendered overview/tracks videos are not guaranteed raw camera RGB; per-camera 2D box alignment/calibration must be located first"}


def frame_sequences(name, paths):
    groups = defaultdict(list)
    for path in paths:
        if name == "HCMC":
            match = re.search(r"cam_(\d+)_(\d+)", path.stem)
            key = relative(path.parent) + "/cam_" + match.group(1) if match else relative(path.parent)
            number = int(match.group(2)) if match else None
        else:
            key = relative(path.parent)
            digits = re.search(r"(\d+)", path.stem)
            number = int(digits.group(1)) if digits else None
        groups[key].append((number, path))
    records = []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda item: (item[0] if item[0] is not None else -1, str(item[1])))
        numbers = [number for number, _ in items if number is not None]
        unique = sorted(set(numbers))
        steps = [after - before for before, after in zip(unique, unique[1:])]
        records.append({"sequence_identifier_candidate": key, "images": len(items), "number_range": [min(numbers), max(numbers)] if numbers else None, "duplicate_numeric_indices": len(numbers) - len(unique), "number_steps": summary(steps), "missing_numeric_indices": sum(max(0, value - 1) for value in steps), "temporal_continuity": "UNVERIFIED: filename order alone is not evidence", "fps": None, "timestamps": "No authoritative frame timestamp metadata located"})
    panels = []
    sampled_steps = []
    resolutions = Counter()
    chosen_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:2]
    for key, items in chosen_groups:
        indices = sorted({0, max(0, len(items) // 2 - 1), max(0, len(items) - 3)})
        for start in indices:
            previous = None
            for number, path in items[start:start + 3]:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    resolutions[f"{image.width}x{image.height}"] += 1
                    gray = np.asarray(image.convert("L").resize((64, 36))).astype(float)
                    if previous is not None:
                        sampled_steps.append(float(np.abs(gray - previous).mean() / 255))
                    previous = gray
                    panels.append((image, f"{Path(key).name} #{number} {path.name[:35]}"))
    preview = contact_sheet(name + "_numbered_neighbors", panels, "Numbered neighbors only; not proof of true consecutive camera frames")
    return records, preview, {"sample_resolutions": dict(resolutions), "neighbor_gray_mad": summary(sampled_steps), "high_difference_neighbor_pairs_over_0_25": sum(value > 0.25 for value in sampled_steps), "files_shuffled_or_temporal_gaps": "Cannot establish without authoritative acquisition timeline"}


def main():
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    print("Inventorying raw sources and protected file signatures...", flush=True)
    before = inventory()
    evidence_paths = [ROOT / key for key in before if key.startswith("data/Vehicle_Tracking/") and Path(key).suffix in {".txt", ".json"}]
    evidence_paths += [ROOT / "data/Vehicle363-v1/RELEASE_MANIFEST.json", ROOT / "data/Vehicle363-v1/FILE_HASHES.sha256", ROOT / "data/Vehicle363-v1/DATASET_FINGERPRINT.sha256", ROOT / "data/TrafficLight363-v1/data.yaml"]
    evidence_before = {relative(path): digest(path) for path in evidence_paths}
    raw_images = defaultdict(list)
    videos = defaultdict(list)
    archives = []
    for key in before:
        path = ROOT / key
        top = Path(key).parts[1]
        if top in PROTECTED or top.endswith(".zip") and top in {"Vehicle363-v1.zip", "TrafficLight363-v1.zip"}:
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos[source_name(path)].append(path)
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            raw_images[source_name(path)].append(path)
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [{"name": entry.filename, "uncompressed_bytes": entry.file_size, "compressed_bytes": entry.compress_size, "crc32": entry.CRC} for entry in archive.infolist() if Path(entry.filename).suffix.lower() in VIDEO_EXTENSIONS]
                if members:
                    archives.append({"path": key, "zip_size_bytes": path.stat().st_size, "video_entries": members})
    sources = []
    decoded_for_preview = 0
    video_records_all = []
    for name, paths in sorted(videos.items()):
        paths = sorted(paths)
        print(f"Auditing videos: {name} ({len(paths)})", flush=True)
        records = []
        sheets = []
        panels = []
        selected = {paths[0], paths[len(paths) // 2]}
        if name in {"VNTraffic", "AICC22-Custom"}:
            selected = {path for path in paths if "Original-video" in path.name}
        for path in sorted(paths):
            record, samples = probe_video(path)
            records.append(record)
            decoded_for_preview += len(samples)
            if path in selected:
                panels.extend((image, f"{path.stem[:36]} f={frame} t={stamp / 1000:.3f}s") for frame, stamp, _, image in samples)
            print(f"  {path.name}: {record.get('reported_frames')} frames", flush=True)
        sheets.append(contact_sheet(name + "_video_continuity", panels, "Three consecutive-frame windows; starts are widely separated"))
        primary = [record for record in records if record.get("role") == "primary_video"]
        usable_records = primary
        annotations = {"track_ids": False, "format": "No per-frame track ID annotations located"}
        suitability = "GOOD_FOR_TRACKING"
        continuity = "Consecutive decode windows sampled; whole long videos NOT exhaustively verified"
        classes = {target: "VISUAL_REVIEW_REQUIRED; not established by source annotations" for target in TARGETS}
        notes = ["Requires new per-frame identities/classes before quantitative tracking evaluation", "Possible scene cuts or inter-clip camera overlap require human QA; similarity is only a screening signal"]
        if name in {"AICC22-Custom", "VNTraffic"}:
            original = next(record for record in primary if "Original-video" in record["path"])
            original_path = ROOT / original["path"]
            original["full_short_decode"] = decode_short_video(original_path)
            annotations, rows = track_annotations(original_path.parent, original)
            annotations["track_ids"] = True
            base = annotations["txt"]["frame_index_base_inferred_from_endpoints"]
            overlay = []
            capture = cv2.VideoCapture(str(original_path))
            frames = sorted({int(row[0]) for row in rows})
            for frame in [frames[0], frames[len(frames) // 2], frames[-1]]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame - (base or 0))
                success, image = capture.read()
                if not success:
                    continue
                for row in rows:
                    if row[0] == frame:
                        left, top, width, height = map(int, row[2:6])
                        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)
                        cv2.putText(image, str(int(row[1])), (left, max(15, top)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                overlay.append((Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), f"Existing TXT IDs: frame={frame}, inferred base={base}"))
            capture.release()
            sheets.append(contact_sheet(name + "_existing_GT_alignment", overlay, "Existing annotations ONLY; verify frame offset/classes/completeness manually"))
            notes = ["No semantic vehicle class in either track annotation format", "JSON/TXT export correspondence audited; no format selected or changed", "GT frame coverage and identity correctness still need human review", "Generic S01/c002 JSON paths reused across sources; do NOT group cameras by this template"]
            if name == "VNTraffic":
                notes += ["TXT has two conflicting frame/ID keys: frame 2 ID 31 and frame 208 ID 49 each have different boxes; JSON also has two repeated frame/ID keys. No rows corrected or discarded", "GT stops at frame 500 but original decodes frames 0..509; last nine frames have no GT rows, not confirmed empty", "Human preview inspection suggests the same camera scene as HCMC cam_15; exact frame/video overlap NOT established. Resolve detector-training camera exposure before claiming an unbiased unseen-camera benchmark"]
            else:
                notes += ["JSON includes 1254 extra geometry observations over TXT; annotation editions require review", "Human preview inspection: large foreground vehicle appears unboxed in middle TXT preview; completeness is not approved"]
        elif name == "imptc_samples":
            annotations = imptc_annotations(paths[0].parent.parent)
            annotations["track_ids"] = True
            suitability = "POSSIBLE_FOR_TRACKING"
            continuity = "Master timestamp timeline checked; image-space camera/box alignment unresolved"
            classes = {target: annotations["class_counts"].get(target, 0) for target in TARGETS}
            usable_records = [record for record in records if Path(record["path"]).name == "sequence_overview_video.mp4"]
            notes = ["All local videos are rendered overviews/track visualizations, not proven original camera streams", "Existing 3D trajectories are not directly a 2D ByteTrack benchmark", "Vehicle and VRU identity namespaces must remain separate"]
        elif name == "AICity21_Track1_Vehicle_Counting":
            annotations = {"track_ids": False, "format": "Counting CSV video_clip_id,frame_id,movement_id,vehicle_class_id; ROI polygons and movement labels. Movement ID is NOT persistent object ID", "class_schema": "Local README: four-wheel vehicles and freight trucks; two-wheelers not counted", "metadata": "Dataset_A/datasetA_vid_stats.txt lists fps and frame counts"}
            classes = {"car": "Broad four-wheel counting category, not exact 4-class labels", "truck": "Freight truck counting category", "bus": "Not separated", "motorcycle": "Excluded from counting ground truth; visual presence requires QA"}
        elif name == "Dataset_Traffic_Evaluation":
            suitability = "POSSIBLE_FOR_TRACKING"
            notes.append("rlr_occasions/video1..video6/teste2_6 may be edited clips; segment at cuts before sequence selection")
        failures = [record["path"] for record in records if record.get("error") or record.get("sample_decode_failures")]
        if failures:
            suitability = "POSSIBLE_FOR_TRACKING"
            notes.append(f"Decode problems: {failures}")
        video_records_all.extend(records)
        root_path = paths[0].parent if name in {"AICC22-Custom", "VNTraffic"} else ROOT / "data" / name
        camera_ids = sorted({match.group(1) for path in paths if (match := re.search(r"(cam_\d+|Bellevue_[A-Za-z0-9_]+?)__", path.stem + "__"))})
        item = {"source_name": name, "path": relative(root_path), "type": "video", "number_of_videos_or_sequences": len(usable_records), "all_video_files": len(records), "number_of_frames": sum(record.get("reported_frames", 0) for record in usable_records), "fps": sorted({record["fps"] for record in usable_records if "fps" in record}), "resolution": sorted({f"{record['resolution'][0]}x{record['resolution'][1]}" for record in usable_records if "resolution" in record}), "duration_seconds": sum(record.get("duration_seconds_estimated") or 0 for record in usable_records), "camera_source_identifiers": camera_ids or [name + " (filename/root evidence; physical camera identity requires confirmation)"], "existing_annotations": annotations, "existing_track_ids": annotations["track_ids"], "target_classes": classes, "temporal_continuity": continuity, "tracking_suitability": suitability, "grouping_unit": "Physical camera -> original video -> continuous no-cut segment; paired originals/GT visualizations stay together", "leakage_risks": notes, "notes": notes, "video_records": records, "previews": sheets, "candidate_frame_scope": "Metadata count upper bound, not approved usable frames", "extracted_jpeg90_bytes_estimate": sum(record.get("estimated_jpeg90_extracted_bytes") or 0 for record in usable_records)}
        sources.append(item)
    for name in ("HCMC", "MIO-TCD-Localization", "Bosch_RGB", "Kaggle_Traffic_Light"):
        paths = sorted(raw_images.get(name, []))
        if not paths:
            continue
        print(f"Auditing image source: {name} ({len(paths)})", flush=True)
        groups, sheet, visual = frame_sequences(name, paths)
        possible = name in {"HCMC", "Bosch_RGB"}
        notes = ["No authoritative fps/timestamps or persistent per-object IDs found; numerical neighbors alone cannot validate temporal order", "No frame/label conversion performed"]
        if name == "HCMC":
            notes.append("Camera names/domain suggest grouping, but frame-order/FPS evidence must be recovered; cross-version duplicate suffixes exist")
            notes += ["Human contact-sheet review: unparsed filename bucket mixes camera views; burned-in clock changes from 21:31:28 to 21:31:18 in lexical neighbors. This bucket is NOT a validated temporal sequence", "cam_15 view visually resembles VNTraffic camera. Same-scene/camera overlap is a risk, NOT established exact-frame duplication; inspect detector release split provenance before tracking evaluation"]
            mapping = {target: "Present in locked source class mapping; not track annotations" for target in TARGETS}
        elif name == "MIO-TCD-Localization":
            notes.append("Local README describes localization on independent images; detection CSV is not trajectory/track ID GT")
            mapping = {target: "Present in localization class list; no temporal identity" for target in TARGETS}
        else:
            mapping = {target: "No vehicle-class ground truth; requires visual review" for target in TARGETS}
        historical = ROOT / "data/vehicle_audit" / ("hcmc_audit.json" if name == "HCMC" else "mio_tcd_audit.json")
        duplicate_history = json.loads(historical.read_text())["duplicate_analysis"] if name in {"HCMC", "MIO-TCD-Localization"} and historical.exists() else None
        sources.append({"source_name": name, "path": relative(ROOT / "data" / {"HCMC": "archive (1)", "Bosch_RGB": "dataset_train_rgb", "Kaggle_Traffic_Light": "archive"}.get(name, name)), "type": "frame_sequence", "number_of_videos_or_sequences": len(groups), "number_of_frames": len(paths), "fps": None, "resolution": visual["sample_resolutions"], "duration_seconds": None, "camera_source_identifiers": [row["sequence_identifier_candidate"] for row in groups], "existing_annotations": "YOLO detection labels" if name == "HCMC" else "Detection CSV with classes and boxes" if name == "MIO-TCD-Localization" else "Traffic-light detection annotations, no track IDs", "existing_track_ids": False, "target_classes": mapping, "temporal_continuity": "UNVERIFIED (numbered image groups are NOT confirmed consecutive sequences)", "tracking_suitability": "POSSIBLE_FOR_TRACKING" if possible else "NOT_SUITABLE", "grouping_unit": "Recover physical camera/acquisition sequence before selecting frames; never random frame split", "leakage_risks": ["HCMC/Bosch images used in detection releases; avoid detector-training overlap in strict unseen tracking evaluation"], "notes": notes, "numbered_groups": groups, "visual_continuity_screen": visual, "historical_duplicate_audit_not_recomputed": duplicate_history, "previews": [sheet], "candidate_frame_scope": "Potential frame images only; continuity not established", "extracted_jpeg90_bytes_estimate": 0})
    if archives:
        entries = [entry for archive in archives for entry in archive["video_entries"]]
        blank = Image.new("RGB", (510, 285), "#dddddd")
        ImageDraw.Draw(blank).text((15, 30), "ARCHIVE ONLY: video not extracted\nNo real frame preview available", fill="black", font=font())
        sheet = contact_sheet("Bellevue_NE8th_archive_inventory", [(blank, f"{len(entries)} video entries in {len(archives)} ZIPs")], "Metadata-only placeholder, NOT a decoded visual QA sample")
        sources.append({"source_name": "Bellevue_NE8th_ARCHIVE_ONLY", "path": "data/Bellevue_Bellevue_NE8th-*.zip", "type": "video", "number_of_videos_or_sequences": len({entry['name'] for entry in entries}), "number_of_frames": None, "fps": None, "resolution": None, "duration_seconds": None, "camera_source_identifiers": ["Bellevue_Bellevue_NE8th (archive filenames)"], "existing_annotations": "No track annotations in video-only archive inventory", "existing_track_ids": False, "target_classes": {target: "UNKNOWN; archive not decoded" for target in TARGETS}, "temporal_continuity": "UNVERIFIED: no extraction/decode", "tracking_suitability": "POSSIBLE_FOR_TRACKING", "grouping_unit": "Camera NE8th -> clip timestamp; verify overlap with loose Bellevue clips", "notes": ["All ZIP central directories read; no archive member extracted", "Frames/durations unknown and excluded from numeric frame totals"], "archives": archives, "previews": [sheet], "extracted_jpeg90_bytes_estimate": None})
    print("Checking source/protected file signatures after audit...", flush=True)
    after = inventory()
    changes = [key for key in before.keys() | after.keys() if before.get(key) != after.get(key)]
    if changes:
        raise RuntimeError(f"Source/protected inventory changed during read-only audit: {changes[:10]}")
    evidence_after = {relative(path): digest(path) for path in evidence_paths}
    if evidence_before != evidence_after:
        raise RuntimeError("Protected/track annotation evidence bytes changed")
    candidates = [source for source in sources if source["tracking_suitability"] != "NOT_SUITABLE"]
    good = [source for source in sources if source["tracking_suitability"] == "GOOD_FOR_TRACKING"]
    totals = {"total_sources": len(sources), "tracking_suitable_sources": len(good), "possible_sources": sum(source['tracking_suitability'] == 'POSSIBLE_FOR_TRACKING' for source in sources), "candidate_sequences_upper_bound": sum(source["number_of_videos_or_sequences"] for source in candidates), "candidate_frames_known_upper_bound": sum(source["number_of_frames"] or 0 for source in candidates), "candidate_sequences_continuity_unverified": sum(source["number_of_videos_or_sequences"] for source in candidates if source["type"] == "frame_sequence" or "ARCHIVE_ONLY" in source["source_name"]), "candidate_frames_unknown_sources": [source['source_name'] for source in candidates if source['number_of_frames'] is None], "all_loose_video_files": len(video_records_all), "all_loose_video_bytes": sum(record.get('size_bytes', 0) for record in video_records_all), "all_loose_video_reported_frames_including_visualization_copies": sum(record.get('reported_frames', 0) for record in video_records_all), "archive_video_compressed_zip_bytes": sum(archive['zip_size_bytes'] for archive in archives), "archive_video_uncompressed_bytes": sum(entry['uncompressed_bytes'] for archive in archives for entry in archive['video_entries']), "estimated_candidate_jpeg90_extraction_bytes": sum(source['extracted_jpeg90_bytes_estimate'] or 0 for source in candidates), "decoder_sample_frames": decoded_for_preview, "short_original_videos_fully_decoded_frames": sum(record.get('full_short_decode', {}).get('decoded_frames', 0) for record in video_records_all)}
    report = {"work": "5.1", "status": "REVIEW_REQUIRED", "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_datasets_modified": False, "vehicle363_modified": False, "trafficlight363_modified": False, "models_trained": False, "tracking_dataset_created": False, "track_ids_generated": False, "source_inventory_files_checked_before_after": len(before), "protected_and_track_annotation_sha256_before": evidence_before, "protected_and_track_annotation_sha256_after": evidence_after, "sampling": "Deterministic lexicographic representative selection; first/middle/end consecutive windows; full in-memory decode only of the two short original tracking clips; no bulk frame extraction", "totals": totals, "sources": sources, "best_source": "VNTraffic", "reason": "Original 1920x1080/30fps short video with existing pixel MOT-style boxes and persistent IDs; JSON/TXT geometry agrees with bijective ID correspondence. Still needs manual identity/completeness/class/frame-alignment QA", "existing_track_annotations": True, "temporal_leakage_risk": "HIGH if splitting individual frames; same-camera clips, visualization/original pairs, archive copies, reused generic JSON paths and detector-training overlap require grouping/review; no split or cross-source exact-video rehash performed", "storage_estimate_scope": "Decimal bytes/GB. Frame counts from container metadata except two short full decodes. JPEG90 native-resolution estimate from bounded actual frame encodes; archive-only extraction size unknown; not a promise of usable frame count", "recommended_strategy": ["Use VNTraffic as a small initial alignment/ID evaluation pilot; retain native existing IDs", "Review AICC22 JSON superset vs TXT before selecting annotation edition; verify inferred 1-based vs VNTraffic 0-based frame origin", "Use AICity/Bellevue/local continuous clips as expansion pool requiring new human track/class GT; segment edited clips at actual cuts", "Do not use MIO/random Kaggle images as tracking sequences", "Do not treat IMPTC 3D rendered overview/trajectory assets as raw 2D MOT without calibration and per-camera images", "Group camera -> video -> continuous segment; build held-out cameras where possible and audit overlap with detector training", "Do not build Tracking363-v1 until human temporal/annotation QA and licensing review"], "limitations": ["All four target classes are not authoritatively covered by any current 2D track GT; classes must be verified/annotated", "Long videos checked only at bounded windows; hidden cuts/duplicates/shuffles can remain", "Archive-only NE8th videos have no real frame QA yet", "Sources with no per-object IDs can supply raw footage, not a ready benchmark", "Full decoded frame equality is a duplicate screen, not proof of real-world capture continuity", "No license authorization for future redistribution inferred from local availability"], "references": [{"topic": "VideoCapture metadata/position properties", "url": "https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html"}, {"topic": "MOT ten-field format reference", "url": "https://github.com/DanceTrack/DanceTrack/blob/main/TrackEval/docs/MOTChallenge-format.txt"}], "next": "Work 5.2 — Build Tracking363 temporal sequences, only after resolving QA prerequisites"}
    totals["candidate_frames_unknown_sources"] = [source["source_name"] for source in candidates if source["number_of_frames"] is None or any("reported_frames" not in record for record in source.get("video_records", []) if record.get("role") == "primary_video")]
    totals["unopenable_or_unknown_metadata_videos"] = [record["path"] for record in video_records_all if record.get("error")]
    totals["videos_with_sample_decode_failures"] = [record["path"] for record in video_records_all if record.get("sample_decode_failures")]
    totals["human_approved_benchmark_frames"] = 0
    report["reason"] = "VNTraffic is best only for an initial alignment/identity QA pilot: 510 fully decoded 1920x1080/30fps frames and existing 2D IDs. NOT benchmark-ready: two repeated frame/ID keys with different boxes, nine trailing frames without GT rows, no four-class semantics, and visually suspected HCMC cam_15 camera overlap. Exact frame/video overlap is unverified. Do not claim unbiased held-out detector evaluation."
    report["recommended_strategy"][0] = "Start with VNTraffic read-only QA; resolve conflicting existing frame/ID rows and HCMC camera-exposure risk before selecting evaluation data. No automatic GT repair."
    report["critical_findings"] = [
        "VNTraffic TXT frame 2 ID 31 and frame 208 ID 49 each contain different boxes under one frame/ID key; JSON also has two repeated keys. Requires human resolution, not automatic deduplication.",
        "AICC22 JSON has 1254 additional box observations over TXT. Middle TXT preview appears to omit a large foreground vehicle; neither edition is approved without human QA.",
        "Five videos fail bounded tail decoding; video3.mp4 and video6.mp4 in Dataset_Traffic_Evaluation cannot open. Their bytes are counted, their unknown frames are not invented; metadata totals are upper bounds.",
        "HCMC lexical neighbor previews include mixed cameras and a burned-in timestamp reversal; catalog groups are not validated continuous sequences.",
        "Human preview comparison suggests VNTraffic and HCMC cam_15 share a camera scene. Exact overlapping frames/video are NOT established; inspect frozen detector split provenance before claiming independent tracking evaluation.",
        "No source currently has approved, complete 2D tracking ground truth for all four target classes. Fully decoded frames are not automatically approved benchmark frames.",
        "IMPTC previews are multi-camera/map mosaics with some black tiles; existing 3D tracks are not direct native per-camera 2D MOT labels."
    ]
    (AUDIT / "work_5_1_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Work 5.1 — Tracking data source audit", "", "STATUS: REVIEW_REQUIRED — audit completed read-only; source/GT suitability still requires human QA, not an execution failure.", "", "Vehicle363 / TrafficLight363 / sources modified: NO. No model/ByteTrack run, ID generation, conversion, split or final Tracking363 dataset created.", "", "## Sources", "", "| SOURCE | TYPE | SEQUENCES | FRAMES | FPS | RESOLUTION | TEMPORAL_CONTINUITY | EXISTING_TRACK_IDS | TARGET_CLASSES | TRACKING_SUITABILITY | NOTES |", "|---|---|---:|---:|---|---|---|---|---|---|---|"]
    for source in sources:
        lines.append(f"| {source['source_name']} | {source['type']} | {source['number_of_videos_or_sequences']} | {source['number_of_frames'] if source['number_of_frames'] is not None else 'UNKNOWN'} | {source['fps']} | {source['resolution']} | {source['temporal_continuity']} | {'YES' if source['existing_track_ids'] else 'NO'} | {source['target_classes']} | {source['tracking_suitability']} | {'; '.join(source['notes'])} |")
    lines += ["", "GOOD_FOR_TRACKING means suitable raw temporal material, NOT an approved per-class benchmark. Sequence/frame totals are upper bounds: numbered HCMC/Bosch image groups and archive-only clips are not validated sequences. Random independent detection images are not counted as usable tracking material.", "", "## Totals / storage", "", *[f"- {key}: {value}" for key, value in totals.items()], "", report["storage_estimate_scope"], "", "Video visualization copies are listed for provenance but excluded from direct raw candidate sequence counts. IMPTC's four rendered overview timelines are counted only as possible research candidates; image-space MOT readiness is NOT established. Archive durations/frame counts remain UNKNOWN; never estimate them just from ZIP bytes.", "", "## Existing track annotations", ""]
    for source in sources:
        lines += [f"### {source['source_name']}", "", f"Path: `{source['path']}`", f"Grouping: {source.get('grouping_unit', 'Unresolved')}", "", *[f"- QA: [{Path(preview).name}]({Path(preview).relative_to(AUDIT.relative_to(ROOT)).as_posix()})" for preview in source['previews']], ""]
        if source["existing_track_ids"]:
            lines += ["```json", json.dumps(source["existing_annotations"], indent=2, ensure_ascii=False), "```", ""]
        elif source["source_name"] == "AICity21_Track1_Vehicle_Counting":
            lines += [str(source["existing_annotations"]), ""]
    lines += ["## Best initial source", "", report["best_source"], report["reason"], "", "AICC22 TXT/JSON disagreement is geometry SUBSET/SUPERSET, not silently canonicalized. VNTraffic JSON contains empty track entries and changes some numeric IDs; matching geometry is not alone proof of identity correctness. Neither export provides authoritative four-class semantic labels. Existing ten-column TXT final -1 fields are world-coordinate placeholders, not classes.", "", "## Split safety", "", report["temporal_leakage_risk"], "", "## Recommended strategy", "", *[f"- {item}" for item in report['recommended_strategy']], "", "## Limitations", "", *[f"- {item}" for item in report['limitations']], "", "## Read-only verification", "", f"Checked size/mtime inventory for {len(before):,} data files before/after; no additions/removals/modifications. Protected release control files and 2D track GT SHA-256 unchanged. This is NOT a fresh full frozen release hash audit; no payload repair was attempted.", "", "Metadata property semantics: [OpenCV](https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html). MOT-style format reference: [TrackEval format documentation](https://github.com/DanceTrack/DanceTrack/blob/main/TrackEval/docs/MOTChallenge-format.txt). Exact local schemas and caveats above take priority over generic format assumptions.", "", report['next'], ""]
    lines[8:8] = ["## Critical findings / human review blockers", "", *[f"- {item}" for item in report["critical_findings"]], ""]
    (AUDIT / "work_5_1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"WORK 5.1 STATUS": report['status'], "VEHICLE363 MODIFIED": "NO", "TOTAL SOURCES FOUND": len(sources), "TRACKING-SUITABLE SOURCES": len(good), "TOTAL CANDIDATE SEQUENCES": totals['candidate_sequences_upper_bound'], "TOTAL CANDIDATE FRAMES": totals['candidate_frames_known_upper_bound'], "BEST SOURCE FOR TRACKING": report['best_source'], "REASON": report['reason'], "EXISTING TRACK ANNOTATIONS": "YES", "TEMPORAL LEAKAGE RISK": report['temporal_leakage_risk'], "STORAGE ESTIMATE": totals, "RECOMMENDED DATASET STRATEGY": report['recommended_strategy'], "NEXT": report['next']}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
