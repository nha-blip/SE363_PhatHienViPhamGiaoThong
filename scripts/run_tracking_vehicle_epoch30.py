import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_tracking_pilot import NAMES, create_review_table, render_frame, write_json
from src.tracking.bytetrack_adapter import create_ultralytics_adapter
from src.tracking.io import load_frame_map, sha256, verify_snapshot
from src.tracking.pilot_runtime import LoggedBoundsBackend, assignment_history, diagnostic_matches, identity_diagnostics
from src.tracking.schemas import TrackingResult
from src.tracking.yolo_adapter import YoloAdapter


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tracking/bytetrack_epoch30_20261031.yaml"
OUTPUT = ROOT / "runs/project363/tracking/vehicle_epoch30_vntraffic_001"
# Cases the owner confirmed as defects on the epoch-19 baseline.
CONFIRMED_CASES = {"case_05_gt13": {"gt_id": 13, "start": 20, "end": 50}, "case_24_gt39": {"gt_id": 39, "start": 205, "end": 240}}


def case_diagnostics(gt_frames, results, minimum_iou):
    history = assignment_history(gt_frames, results, minimum_iou)
    report = {}
    for name, case in CONFIRMED_CASES.items():
        rows = [row for row in history.get(case["gt_id"], []) if case["start"] <= row[0] <= case["end"]]
        present = sum(case["gt_id"] in gt_frames.get(frame, {}) for frame in range(case["start"], case["end"] + 1))
        report[name] = {"gt_frames": present, "matched_frames": len(rows), "prediction_ids": sorted({row[1] for row in rows})}
    return report


def comparison_rows(baseline, current):
    labels = [("matched_observations", "Quan sát GT khớp"), ("coverage", "Độ phủ GT"), ("gt_tracks_never_matched", "Track GT không khớp lần nào"),
              ("gt_tracks_with_several_prediction_ids", "Track GT có từ 2 ID dự đoán"), ("mean_prediction_ids_per_matched_gt_track", "ID dự đoán trung bình mỗi track GT"),
              ("consecutive_id_change_candidates", "Ứng viên đổi ID liên tiếp"), ("unique_prediction_ids", "Số ID dự đoán"), ("prediction_observations", "Quan sát dự đoán"),
              ("predictions_outside_gt", "Dự đoán ngoài GT (chưa kết luận)"), ("overlapping_prediction_pairs_iou_0_7", "Cặp box trùng IoU≥0,7")]
    return "".join(f"<tr><td>{label}</td><td>{baseline[key]:.4g}</td><td>{current[key]:.4g}</td></tr>" for key, label in labels)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description="Rerun the pilot baseline with a newer vehicle checkpoint; only the detector changes")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Use --execute only with owner authorization")
    OUTPUT = arguments.output
    configuration = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    baseline_epochs = configuration.get("baseline_completed_epochs", 19)
    baseline_configuration = yaml.safe_load((ROOT / "configs/tracking/bytetrack_work_5_4.yaml").read_text(encoding="utf-8"))
    if configuration["tracker"] != baseline_configuration["tracker"] or configuration["evaluation"]["full_metrics_enabled"]:
        raise ValueError("Tracker must match the baseline and full metrics stay disabled")
    settings = configuration["detector"]
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
    by_frame = defaultdict(list)
    gt_frames = defaultdict(dict)
    for line in (revision / "metadata/annotations.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_frame[row["mot_frame_id"]].append(row)
        gt_frames[row["mot_frame_id"]][row["track_id"]] = row
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
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "effective_config.json", configuration)
    write_json(OUTPUT / "environment.json", {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "opencv": cv2.__version__, "device": settings["device"], "checkpoint_sha256": settings["expected_sha256"], "baseline_snapshot": snapshot["snapshot_hash"], "created_at_utc": datetime.now(timezone.utc).isoformat()})
    adapter = YoloAdapter(NAMES, settings["minimum_confidence"])
    video = cv2.VideoWriter(str(OUTPUT / "tracking_qa.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), sequence["fps"], (1920, 540))
    if not video.isOpened():
        raise RuntimeError("QA video writer unavailable")
    results = []
    previous_matches = {}
    changes = []
    inference_seconds = tracker_seconds = 0
    try:
        with (OUTPUT / "detections.jsonl").open("x", encoding="utf-8") as detections, (OUTPUT / "tracking_results.jsonl").open("x", encoding="utf-8") as tracking, (OUTPUT / "predictions_mot.txt").open("x", encoding="utf-8") as mot:
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
                results.append(result)
                detections.write(json.dumps(detected.to_dict(), allow_nan=False) + "\n")
                tracking.write(json.dumps(result.to_dict(), allow_nan=False) + "\n")
                for track in result.tracks:
                    left, top, right, bottom = track.bbox_xyxy
                    mot.write(f"{frame.mot_frame_id},{track.track_id},{left:.6f},{top:.6f},{right-left:.6f},{bottom-top:.6f},{track.confidence:.6f},-1,-1,-1\n")
                for gt_id, prediction_id, score in diagnostic_matches(by_frame[frame.mot_frame_id], result.tracks, configuration["evaluation"]["diagnostic_iou"]):
                    earlier = previous_matches.get(gt_id)
                    if earlier and earlier[0] == frame.mot_frame_id - 1 and earlier[1] != prediction_id:
                        changes.append({"gt_track_id": gt_id, "previous_mot_frame": earlier[0], "mot_frame_id": frame.mot_frame_id, "previous_prediction_id": earlier[1], "prediction_id": prediction_id, "iou": score, "status": "VISUAL_REVIEW_REQUIRED_NOT_OFFICIAL_IDSW"})
                    previous_matches[gt_id] = (frame.mot_frame_id, prediction_id)
                video.write(render_frame(image, by_frame[frame.mot_frame_id], result))
                if frame.mot_frame_id % 50 == 0:
                    print(f"BYTETRACK_EPOCH{settings['completed_epochs']}: {frame.mot_frame_id}/{len(frames)}", flush=True)
    finally:
        video.release()
    decoded = cv2.VideoCapture(str(OUTPUT / "tracking_qa.mp4"))
    video_frames = 0
    while decoded.read()[0]:
        video_frames += 1
    decoded.release()
    if video_frames != len(frames):
        raise RuntimeError("QA video frame count mismatch")
    verify_snapshot(snapshot, ROOT)
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Protected input changed: {path}")
    iou = configuration["evaluation"]["diagnostic_iou"]
    baseline_directory = ROOT / configuration["baseline_directory"]
    baseline_results = [TrackingResult.from_dict(json.loads(line)) for line in (baseline_directory / "tracking_results.jsonl").read_text(encoding="utf-8").splitlines()]
    baseline, current = identity_diagnostics(gt_frames, baseline_results, iou), identity_diagnostics(gt_frames, results, iou)
    if current["consecutive_id_change_candidates"] != len(changes):
        raise RuntimeError("Inconsistent ID-change diagnostics")
    report = {"status": "REVIEW_REQUIRED", "automated_execution_status": "PASS", "human_tracking_qa_status": "REVIEW_REQUIRED", "dataset_role": "PILOT_DEV",
              "frames": len(frames), "checkpoint_completed_epochs": settings["completed_epochs"], "checkpoint_sha256": settings["expected_sha256"], "baseline_checkpoint_completed_epochs": baseline_epochs,
              "only_detector_changed": True, "tracker_tuned": False, "baseline": baseline, "current": current,
              "confirmed_case_diagnostics": {"baseline": case_diagnostics(gt_frames, baseline_results, iou), "current": case_diagnostics(gt_frames, results, iou)},
              "bounds_event_count": len(bounded_backend.events), "qa_video_decoded_frames": video_frames, "device": settings["device"], "inference_seconds": inference_seconds, "tracker_update_seconds": tracker_seconds,
              "fps_scope": "Offline CPU; includes model warmup, excludes image read/QA; not deployment real-time FPS",
              "tracking_metrics": {"IDF1": None, "HOTA": None, "MOTA": None, "official_ID_switches": None}, "gt_completeness": "UNVERIFIED", "gt_classes": "PENDING",
              "camera_overlap": "Potential HCMC detector-training exposure unresolved", "gt_modified": False, "test_used": False}
    write_json(OUTPUT / "bounds_events.json", bounded_backend.events)
    write_json(OUTPUT / "diagnostic_id_changes.json", changes)
    write_json(OUTPUT / "report.json", report)
    review_table = create_review_table(OUTPUT, changes, sequence["fps"])
    epochs = settings["completed_epochs"]
    (OUTPUT / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>Tracking epoch {epochs} QA</title><style>body{{background:#181818;color:white;font:18px sans-serif;margin:20px}}video{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #777;padding:4px 10px}}</style><h1>Tracking epoch {epochs} — cần người xem lại</h1><p>Chỉ đổi detector từ epoch {baseline_epochs} sang epoch {epochs}; ByteTrack giữ nguyên. Trái: GT (xanh lá). Phải: dự đoán (xanh dương). So sánh sự liên tục của cùng một xe, không so số ID. Đây là số chẩn đoán, không phải IDF1/HOTA/MOTA.</p><table><tr><th>Chỉ số</th><th>Epoch {baseline_epochs}</th><th>Epoch {epochs}</th></tr>' + comparison_rows(baseline, current) + '</table><video controls src="tracking_qa.mp4"></video>' + review_table, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
