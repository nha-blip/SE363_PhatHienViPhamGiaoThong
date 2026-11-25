import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.bytetrack_adapter import create_ultralytics_adapter
from src.tracking.class_suggestions import overlap
from src.tracking.experiment_filters import cross_class_suppression
from src.tracking.io import sha256
from src.tracking.pilot_runtime import LoggedBoundsBackend, diagnostic_matches
from src.tracking.schemas import DetectionResult, TrackingResult


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/project363/tracking/work_5_4_baseline"
OUTPUT = ROOT / "runs/project363/tracking/work_5_4_experiments"
PILOT = ROOT / "data/Tracking363_working/sequences/vntraffic_001"
GT = ROOT / "data/Tracking363_working/gt_revisions/vntraffic_001_remove_track68/metadata/annotations.jsonl"
NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
CASES = {5: {"gt_id": 13, "center": 36, "start": 20, "end": 50}, 24: {"gt_id": 39, "center": 222, "start": 205, "end": 240}}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def target_preview(result, gt_row):
    import cv2
    import numpy as np

    image = cv2.imread(str(PILOT / "img1" / f"{result.frame.mot_frame_id:06d}.jpg"))
    if image is None:
        raise RuntimeError("QA frame unavailable")
    left, top, width, height = gt_row["bbox_xywh"]
    region = (max(0, int(left - 80)), max(0, int(top - 80)), min(1920, int(left + width + 80)), min(1080, int(top + height + 80)))
    cv2.rectangle(image, (int(left), int(top)), (int(left + width), int(top + height)), (0, 255, 0), 2)
    for track in result.tracks:
        if overlap(gt_row["bbox_xywh"], track.bbox_xyxy) > 0.1:
            track_left, track_top, track_right, track_bottom = track.bbox_xyxy
            color = (int((track.track_id * 71) % 200 + 55), int((track.track_id * 37) % 200 + 55), 255)
            cv2.rectangle(image, (int(track_left), int(track_top)), (int(track_right), int(track_bottom)), color, 2)
            cv2.putText(image, f"P{track.track_id}", (int(track_left), max(15, int(track_top))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    crop = image[region[1]:region[3], region[0]:region[2]]
    scale = min(960 / crop.shape[1], 480 / crop.shape[0])
    resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    panel = np.zeros((540, 960, 3), dtype=np.uint8)
    panel[60:60 + resized.shape[0], :resized.shape[1]] = resized
    cv2.putText(panel, f"GT{gt_row['track_id']} green | source {result.frame.source_frame_id} / MOT {result.frame.mot_frame_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(panel, "Prediction IDs differ between experiments; compare physical continuity", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def execute(name, changes, suppression, detections, gt_frames, baseline):
    import cv2

    settings = yaml.safe_load((ROOT / "configs/tracking/bytetrack_work_5_4.yaml").read_text())["tracker"]
    settings.update(changes)
    tracker = create_ultralytics_adapter(settings, NAMES, 30, allow_execution=True)
    tracker.backend.reset()
    bounds = LoggedBoundsBackend(tracker.backend, 1920, 1080)
    tracker.backend = bounds
    directory = OUTPUT / name
    directory.mkdir()
    write_json(directory / "effective_config.json", {"tracker": settings, "post_cache_crossclass_suppression_iou": suppression, "detector_rerun": False, "not_identical_to_rerunning_yolo_agnostic_nms": True})
    case_data = {identity: {"matched_frames": 0, "unmatched_frames": 0, "multi_prediction_overlap_frames": 0, "matched_prediction_ids": [], "consecutive_match_changes": [], "frame_assignments": []} for identity in CASES}
    previous = {}
    event_count = 0
    match_count = 0
    observations = 0
    ids = set()
    dropped = []
    duplicate_pair_count = 0
    results = []
    for detected in detections:
        frame_id = detected.frame.mot_frame_id
        if suppression is not None:
            detected, removed = cross_class_suppression(detected, suppression)
            dropped.extend({"mot_frame_id": frame_id, **item} for item in removed)
        result = tracker.update(detected)
        if name == "control" and result.to_dict() != baseline[frame_id].to_dict():
            raise RuntimeError(f"Control replay diverges at frame {frame_id}; abort experiments")
        results.append(result)
        observations += len(result.tracks)
        ids.update(track.track_id for track in result.tracks)
        for position, track in enumerate(result.tracks):
            left, top, right, bottom = track.bbox_xyxy
            duplicate_pair_count += sum(overlap((left, top, right-left, bottom-top), other.bbox_xyxy) >= 0.7 for other in result.tracks[position+1:])
        matches = diagnostic_matches(list(gt_frames[frame_id].values()), result.tracks)
        match_count += len(matches)
        assigned = {}
        for gt_id, prediction_id, _ in matches:
            assigned[gt_id] = prediction_id
            earlier = previous.get(gt_id)
            if earlier and earlier[0] == frame_id - 1 and earlier[1] != prediction_id:
                event_count += 1
            previous[gt_id] = (frame_id, prediction_id)
        for identity, case in CASES.items():
            if not case["start"] <= frame_id <= case["end"] or case["gt_id"] not in gt_frames[frame_id]:
                continue
            stats = case_data[identity]
            prediction_id = assigned.get(case["gt_id"])
            if prediction_id is not None:
                stats["matched_frames"] += 1
                stats["matched_prediction_ids"].append(prediction_id)
            else:
                stats["unmatched_frames"] += 1
            previous_assignment = stats["frame_assignments"][-1] if stats["frame_assignments"] else None
            if previous_assignment and previous_assignment["mot_frame_id"] == frame_id - 1 and previous_assignment["prediction_id"] is not None and prediction_id is not None and previous_assignment["prediction_id"] != prediction_id:
                stats["consecutive_match_changes"].append(frame_id)
            stats["frame_assignments"].append({"mot_frame_id": frame_id, "prediction_id": prediction_id})
            stats["multi_prediction_overlap_frames"] += sum(overlap(gt_frames[frame_id][case["gt_id"]]["bbox_xywh"], track.bbox_xyxy) >= 0.5 for track in result.tracks) >= 2
    with (directory / "tracking_results.jsonl").open("x", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), allow_nan=False) + "\n")
    write_json(directory / "suppressed_detections.json", dropped)
    write_json(directory / "bounds_events.json", bounds.events)
    for identity, case in CASES.items():
        video_path = directory / f"case_{identity:02d}.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 540))
        if not writer.isOpened():
            raise RuntimeError("QA writer unavailable")
        count = 0
        try:
            for result in results[case["start"]-1:case["end"]]:
                gt_row = gt_frames[result.frame.mot_frame_id].get(case["gt_id"])
                if gt_row:
                    rendered = target_preview(result, gt_row)
                    writer.write(rendered)
                    count += 1
                    if result.frame.mot_frame_id in (case["center"]-1, case["center"], case["center"]+1):
                        if not cv2.imwrite(str(directory / f"case_{identity:02d}_mot{result.frame.mot_frame_id}.jpg"), rendered):
                            raise RuntimeError("Preview write failed")
        finally:
            writer.release()
        capture = cv2.VideoCapture(str(video_path))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        if decoded != count:
            raise RuntimeError("QA video validation failed")
        case_data[identity]["qa_video_frames"] = count
        case_data[identity]["qa_playback_fps"] = 10
        case_data[identity]["unique_matched_ids"] = len(set(case_data[identity]["matched_prediction_ids"]))
    summary = {"name": name, "frames": 501, "changes": changes, "suppression_iou": suppression,
               "prediction_observations": observations, "unique_prediction_ids": len(ids), "diagnostic_matches": match_count,
               "diagnostic_id_change_candidates": event_count, "overlapping_prediction_pairs_iou_0_7": duplicate_pair_count,
               "suppressed_detections": len(dropped), "case_diagnostics": case_data, "human_qa_status": "REVIEW_REQUIRED",
               "official_metrics": None, "selected_as_final": False, "caveat": "Geometric diagnostics only; reducing matches/recall can trivially reduce candidate counts. Cache suppression may remove real occluded vehicles. No calibrated improvement or test generalization claim."}
    write_json(directory / "report.json", summary)
    print(json.dumps({key: summary[key] for key in ("name", "diagnostic_matches", "diagnostic_id_change_candidates", "overlapping_prediction_pairs_iou_0_7", "suppressed_detections")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Owner authorization and --execute required")
    inputs = [GT, BASE / "detections.jsonl", BASE / "tracking_results.jsonl", BASE / "candidate_analysis/human_targeted_review.json", ROOT / "configs/tracking/bytetrack_work_5_4.yaml"]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    detections = [DetectionResult.from_dict(json.loads(line)) for line in inputs[1].read_text().splitlines()]
    baseline = {row.frame.mot_frame_id: row for row in (TrackingResult.from_dict(json.loads(line)) for line in inputs[2].read_text().splitlines())}
    if len(detections) != 501 or [row.frame.mot_frame_id for row in detections] != list(range(1,502)):
        raise ValueError("Invalid detection cache ordering")
    gt_frames = defaultdict(dict)
    for line in GT.read_text().splitlines():
        row = json.loads(line)
        gt_frames[row["mot_frame_id"]][row["track_id"]] = row
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / "input_hashes.json", hashes)
    experiments = [("control", {}, None), ("crossclass_suppression_0_7", {}, 0.7), ("new_track_0_4", {"new_track_thresh": 0.4}, None), ("match_0_9", {"match_thresh": 0.9}, None)]
    summaries = [execute(name, changes, suppression, detections, gt_frames, baseline) for name, changes, suppression in experiments]
    for path in inputs:
        if sha256(path) != hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError("Protected inputs changed")
    report = {"status": "EXECUTION_PASS_HUMAN_REVIEW_REQUIRED", "control_matches_baseline_exactly": True,
              "experiments": summaries, "source_gt_and_baseline_unchanged": True, "yolo_rerun": False,
              "selected_configuration": None, "tuning_complete": False, "evaluation_policy_still_pending": True,
              "dataset_role": "PILOT_DEV", "full_metrics": None,
              "next": "Inspect case5/case24 slow QA and side effects; do not rank by candidate count alone"}
    write_json(OUTPUT / "comparison.json", report)
    sections = []
    for summary in summaries:
        name = summary["name"]
        sections.append(f'<section><h2>{name}</h2><p>Diagnostic candidates: {summary["diagnostic_id_change_candidates"]}; GT matches: {summary["diagnostic_matches"]}; overlapping pairs: {summary["overlapping_prediction_pairs_iou_0_7"]}. Not official accuracy metrics.</p>')
        for identity in CASES:
            stats = summary["case_diagnostics"][identity]
            sections.append(f'<h3>Case {identity}</h3><p>Matched/unmatched frames: {stats["matched_frames"]}/{stats["unmatched_frames"]}; multiple overlapping prediction frames: {stats["multi_prediction_overlap_frames"]}; consecutive assignment changes: {len(stats["consecutive_match_changes"])}.</p><video controls preload="metadata" src="{name}/case_{identity:02d}.mp4"></video><p>10fps slow playback from 30fps source. GT green; colored P-ID boxes predictions. P-ID numbers between experiments need not match.</p>')
            for frame_id in (CASES[identity]["center"]-1, CASES[identity]["center"], CASES[identity]["center"]+1):
                sections.append(f'<img loading="lazy" src="{name}/case_{identity:02d}_mot{frame_id}.jpg">')
        sections.append('</section>')
    (OUTPUT / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>ByteTrack controlled experiments</title><style>body{background:#181818;color:white;font:18px sans-serif;margin:20px}video{width:80%}img{width:32%}section{border-top:1px solid #777}</style><h1>Controlled experiments — HUMAN QA REQUIRED</h1><p>Four separate 501-frame replays. Control reproduces original predictions exactly; each other run changes only one setting or one post-cache suppression stage. Check identity continuity, duplicate boxes, and missed vehicles. No final configuration or official metrics selected.</p>' + "\n".join(sections), encoding="utf-8")
    (OUTPUT / "README.md").write_text("# Work 5.4 controlled parameter experiments\n\nOpen index.html to compare slow case5/case24 video and still frames. Each run starts a fresh tracker at frame1, consumes all501 cached low-confidence detections, and does not rerun YOLO or alter GT/source/baseline. Control replay is bit-for-bit identical in serialized tracking results.\n\nVariants separately add post-cache cross-class suppression at IoU>0.7, raise new_track_thresh to0.4, or raise match_thresh to0.9. Post-cache suppression is not identical to rerunning detector agnostic NMS and may remove real vehicles under occlusion. Higher new-track confidence can miss weak vehicles; lenient matching can misassociate neighboring vehicles. Do not select parameters solely by lower candidate/duplicate counts. All counts are geometry diagnostics, not IDF1/HOTA/MOTA or final accuracy. Full evaluation completeness/ignore policy remains pending. No final config selected. Videos play at10fps (one-third source speed). Original report unchanged; current evidence is comparison.json.\n\nParameter reference: https://docs.ultralytics.com/modes/track/\n", encoding="utf-8")
    print("CONTROLLED_EXPERIMENTS_COMPLETE: " + str(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
