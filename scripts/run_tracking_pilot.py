import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.bytetrack_adapter import create_ultralytics_adapter
from src.tracking.io import load_frame_map, sha256, verify_snapshot
from src.tracking.pilot_runtime import LoggedBoundsBackend, diagnostic_matches
from src.tracking.yolo_adapter import YoloAdapter


ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
NAMES_6 = {**NAMES, 4: "pickup", 5: "minibus"}  # 6-class taxonomy, spec v1.1 §15.3


def detector_names(model_names):
    """The checkpoint's own class table, accepted only if it is the old 4-class or the spec 6-class taxonomy (FR-06)."""
    if model_names not in (NAMES, NAMES_6):
        raise ValueError(f"Unexpected model namespace: {model_names}")
    return dict(model_names)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_review_table(output, changes, fps=30):
    fields = ["gt_track_id", "previous_mot_frame", "mot_frame_id", "previous_prediction_id", "prediction_id", "decision", "notes"]
    with (output / "human_tracking_review.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in changes:
            writer.writerow({**{key: row[key] for key in fields[:5]}, "decision": "REVIEW_REQUIRED", "notes": ""})
    rows = []
    for row in changes:
        stamp = max(0, (row["mot_frame_id"] - 1) / fps - 1)
        rows.append(f'<tr><td>GT{row["gt_track_id"]}</td><td>source {row["mot_frame_id"]-1} / MOT {row["mot_frame_id"]}</td><td>P{row["previous_prediction_id"]} → P{row["prediction_id"]}</td><td><button onclick="const video=document.querySelector(\'video\');video.currentTime={stamp:.3f};video.play();video.scrollIntoView();">Watch nearby frames</button></td></tr>')
    return '<h2>ID-change candidates — not certified errors</h2><p>Review all cases in human_tracking_review.csv. Set decision to CONFIRMED_ID_CHANGE, MATCHING_AMBIGUITY, or REVIEW_REQUIRED; add notes. A numeric P-ID differing from GT is normal.</p><table><tr><th>GT target</th><th>Frame</th><th>Predicted change</th><th>Video</th></tr>' + "".join(rows) + '</table>'


def render_frame(image, gt_rows, result):
    import cv2
    import numpy as np

    truth, prediction = image.copy(), image.copy()
    for row in gt_rows:
        left, top, width, height = row["bbox_xywh"]
        cv2.rectangle(truth, (int(left), int(top)), (int(left + width), int(top + height)), (0, 255, 0), 2)
        cv2.putText(truth, f"GT{row['track_id']}", (int(left), max(35, int(top))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for track in result.tracks:
        left, top, right, bottom = track.bbox_xyxy
        cv2.rectangle(prediction, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), 2)
        cv2.putText(prediction, f"P{track.track_id}", (int(left), max(35, int(top))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    for panel, title in ((truth, "DERIVED GT (green)"), (prediction, "BYTETRACK PREDICTIONS (cyan) - IDs need not equal GT")):
        cv2.rectangle(panel, (0, 0), (1920, 45), (0, 0, 0), -1)
        cv2.putText(panel, f"{title} | source {result.frame.source_frame_id} / MOT {result.frame.mot_frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.hstack([cv2.resize(truth, (960, 540)), cv2.resize(prediction, (960, 540))])


def main():
    parser = argparse.ArgumentParser(description="Explicitly authorized baseline YOLO -> ByteTrack pilot; GT stays read-only")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs/tracking/bytetrack_work_5_4.yaml"))
    parser.add_argument("--output", default=str(ROOT / "runs/project363/tracking/work_5_4_baseline"))
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Use --execute only with owner authorization")
    configuration = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    if not configuration["execution_enabled"] or configuration["evaluation"]["full_metrics_enabled"]:
        raise ValueError("Baseline requires execution permission and disabled full metrics")
    settings = configuration["detector"]
    if settings["minimum_confidence"] > configuration["tracker"]["track_low_thresh"]:
        raise ValueError("Detector discards ByteTrack low-confidence candidates")
    output = Path(arguments.output).resolve()
    if (ROOT / "runs/project363/tracking").resolve() not in output.parents:
        raise ValueError("Pilot outputs must stay in project tracking runs")
    base = ROOT / configuration["sequence_directory"]
    revision = ROOT / configuration["derived_revision"]
    snapshot = json.loads((ROOT / configuration["integration_snapshot"]).read_text(encoding="utf-8"))
    verify_snapshot(snapshot, ROOT)
    manifest = json.loads((revision / "metadata/resolution_manifest.json").read_text(encoding="utf-8"))
    protected = {revision / "gt/gt.txt": manifest["derived_gt_sha256"], revision / "metadata/annotations.jsonl": manifest["derived_annotations_sha256"], ROOT / settings["model_path"]: settings["expected_sha256"]}
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
    sequence = json.loads((base / "metadata/sequence.json").read_text(encoding="utf-8"))
    frames = load_frame_map(base / "metadata/frame_map.csv", sequence)
    records = [json.loads(line) for line in (revision / "metadata/annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 16034 or len({row["track_id"] for row in records}) != 84:
        raise ValueError("Unexpected derived GT totals")
    by_frame = defaultdict(list)
    for row in records:
        by_frame[row["mot_frame_id"]].append(row)
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(settings["cpu_threads"])
    model = YOLO(str(ROOT / settings["model_path"]))
    if model.names != NAMES:
        raise ValueError("Unexpected model namespace")
    tracker = create_ultralytics_adapter(configuration["tracker"], NAMES, sequence["fps"], allow_execution=True)
    bounded_backend = LoggedBoundsBackend(tracker.backend, sequence["width"], sequence["height"])
    tracker.backend = bounded_backend
    output.mkdir(parents=True, exist_ok=False)
    (output / "previews").mkdir()
    for extension in ("json", "md"):
        previous = ROOT / f"runs/project363/tracking/work_5_4_report.{extension}"
        if previous.exists():
            shutil.copyfile(previous, output / f"previous_work_5_4_report.{extension}")
    write_json(output / "effective_config.json", configuration)
    write_json(output / "environment.json", {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "opencv": cv2.__version__, "device": settings["device"], "checkpoint_sha256": settings["expected_sha256"], "baseline_snapshot": snapshot["snapshot_hash"], "created_at_utc": datetime.now(timezone.utc).isoformat()})
    adapter = YoloAdapter(NAMES, settings["minimum_confidence"])
    video = cv2.VideoWriter(str(output / "tracking_qa.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), sequence["fps"], (1920, 540))
    if not video.isOpened():
        raise RuntimeError("QA video writer unavailable")
    prediction_ids = set()
    observation_count = 0
    detection_count = 0
    low_count = 0
    inference_seconds = 0
    tracker_seconds = 0
    matched_count = 0
    previous_matches = {}
    changes = []
    preview_ids = set(range(1, 502, 25)) | {3, 208, 209, 210, 382, 383, 384, 501}
    preview_manifest = []
    frame_log = []
    try:
        with (output / "detections.jsonl").open("x", encoding="utf-8") as detections, (output / "tracking_results.jsonl").open("x", encoding="utf-8") as tracking, (output / "raw_backend_outputs.jsonl").open("x", encoding="utf-8") as raw, (output / "predictions_mot.txt").open("x", encoding="utf-8") as mot:
            for frame in frames:
                image = cv2.imread(str(base / "img1" / f"{frame.mot_frame_id:06d}.jpg"))
                if image is None or image.shape[:2] != (frame.height, frame.width):
                    raise ValueError("Missing or incorrectly sized frame")
                started = time.perf_counter()
                prediction = model.predict(image, device=settings["device"], imgsz=settings["imgsz"], conf=settings["minimum_confidence"], iou=settings["nms_iou"], max_det=settings["max_det"], save=False, verbose=False)[0]
                inference_seconds += time.perf_counter() - started
                detected = adapter.adapt(prediction, frame)
                started = time.perf_counter()
                result = tracker.update(detected)
                tracker_seconds += time.perf_counter() - started
                detections.write(json.dumps(detected.to_dict(), allow_nan=False) + "\n")
                tracking.write(json.dumps(result.to_dict(), allow_nan=False) + "\n")
                raw.write(json.dumps({"mot_frame_id": frame.mot_frame_id, "raw_rows": bounded_backend.raw_output}, allow_nan=False) + "\n")
                detection_count += len(detected.detections)
                low_count += sum(configuration["tracker"]["track_low_thresh"] < item.confidence < configuration["tracker"]["track_high_thresh"] for item in detected.detections)
                observation_count += len(result.tracks)
                for track in result.tracks:
                    prediction_ids.add(track.track_id)
                    left, top, right, bottom = track.bbox_xyxy
                    mot.write(f"{frame.mot_frame_id},{track.track_id},{left:.6f},{top:.6f},{right-left:.6f},{bottom-top:.6f},{track.confidence:.6f},-1,-1,-1\n")
                matches = diagnostic_matches(by_frame[frame.mot_frame_id], result.tracks, configuration["evaluation"]["diagnostic_iou"])
                matched_count += len(matches)
                for gt_id, prediction_id, score in matches:
                    previous = previous_matches.get(gt_id)
                    if previous and previous[0] == frame.mot_frame_id - 1 and previous[1] != prediction_id:
                        changes.append({"gt_track_id": gt_id, "previous_mot_frame": previous[0], "mot_frame_id": frame.mot_frame_id, "previous_prediction_id": previous[1], "prediction_id": prediction_id, "iou": score, "status": "VISUAL_REVIEW_REQUIRED_NOT_OFFICIAL_IDSW"})
                        if len(changes) <= 20:
                            preview_ids.add(frame.mot_frame_id)
                    previous_matches[gt_id] = (frame.mot_frame_id, prediction_id)
                rendered = render_frame(image, by_frame[frame.mot_frame_id], result)
                video.write(rendered)
                if frame.mot_frame_id in preview_ids:
                    filename = f"frame_{frame.mot_frame_id:06d}.jpg"
                    if not cv2.imwrite(str(output / "previews" / filename), rendered):
                        raise RuntimeError("Preview write failed")
                    preview_manifest.append({"mot_frame_id": frame.mot_frame_id, "source_frame_id": frame.source_frame_id, "filename": filename})
                frame_log.append({"mot_frame_id": frame.mot_frame_id, "detections": len(detected.detections), "predicted_tracks": len(result.tracks), "gt_observations": len(by_frame[frame.mot_frame_id]), "diagnostic_matches": len(matches)})
                if frame.mot_frame_id % 50 == 0:
                    print(f"BYTETRACK_BASELINE: {frame.mot_frame_id}/{len(frames)}", flush=True)
    finally:
        video.release()
    decoded = cv2.VideoCapture(str(output / "tracking_qa.mp4"))
    video_frames = 0
    while decoded.read()[0]:
        video_frames += 1
    decoded.release()
    if video_frames != 501:
        raise RuntimeError("QA video frame count mismatch")
    verify_snapshot(snapshot, ROOT)
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Protected input changed: {path}")
    write_json(output / "bounds_events.json", bounded_backend.events)
    write_json(output / "qa_manifest.json", preview_manifest)
    write_json(output / "diagnostic_id_changes.json", changes)
    write_json(output / "frame_diagnostics.json", frame_log)
    policy = {"status": "REVIEW_REQUIRED", "class_agnostic": True, "pilot_role": "PILOT_DEV", "full_metrics_enabled": False, "IDF1": None, "HOTA": None, "MOTA": None, "official_ID_switches": None, "gap_policy": "No interpolation; missing observations not automatically interpreted as negatives or ignore regions", "completeness": "UNVERIFIED", "required_before_metrics": ["Human identify fully annotated/evaluable frames and regions", "Record explicit ignore policy for unverifiable areas", "Choose independent holdout before final generalization claims"], "diagnostics": "One-to-one IoU>=0.5 matches without semantic classes; ID-change candidates only for consecutive matched frames, not official metrics", "tuning": "Untuned baseline only; wait for visual QA before selecting parameters", "camera_overlap": "Potential HCMC detector-training exposure unresolved"}
    write_json(output / "evaluation_policy.json", policy)
    review_table = create_review_table(output, changes, sequence["fps"])
    sections = [f'<section><h2>Source {row["source_frame_id"]} / MOT {row["mot_frame_id"]}</h2><img loading="lazy" src="previews/{row["filename"]}"></section>' for row in preview_manifest]
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>ByteTrack pilot QA</title><style>body{background:#181818;color:white;font:18px sans-serif;margin:20px}img,video{max-width:100%}section{border-top:1px solid #777}</style><h1>HUMAN QA: REVIEW_REQUIRED</h1><p>Left: corrected GT (green). Right: tracker predictions P-ID (cyan). Numeric P-IDs need not equal GT IDs: check physical identity continuity, not numeric equality. GT semantics remain PENDING. No official metrics or tuning PASS.</p><video controls src="tracking_qa.mp4"></video>' + review_table + '<p>If MP4 playback is unsupported, open the video externally or inspect JPEG previews. Review nearby frames in the video for loss/reappearance, occlusion, ID changes; diagnostic_id_changes.json is a candidate list, not certified errors.</p>' + "\n".join(sections), encoding="utf-8")
    report = {"work": "5.4", "status": "REVIEW_REQUIRED", "automated_execution_status": "PASS", "human_tracking_qa_status": "REVIEW_REQUIRED", "bytetrack_executed": True, "tracker_tuned": False, "frames": 501, "gt_observations": 16034, "gt_tracks": 84, "detector_observations": detection_count, "low_score_detections": low_count, "predicted_observations": observation_count, "unique_predicted_ids": len(prediction_ids), "diagnostic_matched_observations": matched_count, "consecutive_match_id_change_candidates": len(changes), "bounds_event_count": len(bounded_backend.events), "preview_images": len(preview_manifest), "qa_video_decoded_frames": video_frames, "device": settings["device"], "inference_seconds": inference_seconds, "tracker_update_seconds": tracker_seconds, "cpu_inference_plus_tracker_fps_excluding_io_qa": len(frames) / (inference_seconds + tracker_seconds), "fps_scope": "Offline CPU with concurrent training; includes model warmup, excludes input image read/QA/evaluation; not deployment real-time FPS", "checkpoint_completed_epochs": 19, "checkpoint_sha256": settings["expected_sha256"], "derived_gt_sha256": manifest["derived_gt_sha256"], "raw_and_baseline_unchanged": True, "gt_classes": "PENDING", "class_agnostic_association": True, "tracking_metrics": {"IDF1": None, "HOTA": None, "MOTA": None, "official_ID_switches": None}, "dataset_role": "PILOT_DEV", "final_dataset_created": False, "split_created": False, "qa_directory": str(output), "next": "Human tracking QA and evaluation completeness/ignore policy, then parameter experiments", "references": ["https://docs.ultralytics.com/reference/trackers/byte_tracker/", "https://docs.ultralytics.com/modes/track/"]}
    write_json(output / "report.json", report)
    write_json(ROOT / "runs/project363/tracking/work_5_4_report.json", report)
    summary = "# Work 5.4 — ByteTrack baseline execution\n\nSTATUS: REVIEW_REQUIRED. Automated execution: PASS. Human tracking QA: REVIEW_REQUIRED.\n\n" + "\n".join(f"- {key}: {report[key]}" for key in ("frames", "gt_observations", "gt_tracks", "predicted_observations", "unique_predicted_ids", "low_score_detections", "consecutive_match_id_change_candidates", "preview_images", "bounds_event_count")) + "\n\nExecuted real ByteTrack with fixed epoch-19 checkpoint and class-agnostic association. Predictions are regenerated at confidence 0.1 so the low-score stage receives candidates; the older 0.25 cache is not used. Every frame advances the same tracker, including empty detections. Raw outputs and any explicit bounds adjustments are logged; no GT clipping or changes. Original snapshot and derived GT hashes reverified.\n\nOpen " + str(output / "index.html") + ". GT and prediction IDs have separate namespaces; compare physical identity over time, not numeric equality. Inspect diagnostic_id_changes.json, especially occlusions and returns after gaps. Candidates are not official ID switches or certified annotation/tracking errors.\n\nNo IDF1/HOTA/MOTA claimed, no tuning PASS, no unbiased benchmark claim. Existing GT completeness/gaps and camera-exposure limitations require explicit evaluation policy before full metrics. CPU timing excludes I/O and QA and is not a deployment FPS measurement. Classes stay PENDING. No final dataset/split.\n\nBaseline config: configs/tracking/bytetrack_work_5_4.yaml. Policy and machine evidence are in the QA directory. Historical pre-execution reports are preserved there as previous_work_5_4_report.*.\n\nNEXT: Wait for human tracking QA, define evaluable/ignore regions, then tune with reproducible parameter runs.\n"
    (ROOT / "runs/project363/tracking/work_5_4_report.md").write_text(summary, encoding="utf-8")
    (output / "README.md").write_text(summary, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
