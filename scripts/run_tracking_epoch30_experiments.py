import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_tracking_experiments as experiments
from src.tracking.class_suggestions import overlap
from src.tracking.pilot_runtime import identity_diagnostics
from src.tracking.schemas import DetectionResult, TrackingResult


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/project363/tracking/vehicle_epoch30_vntraffic_001"
OUTPUT = BASE / "experiments"
BASELINE = ROOT / "runs/project363/tracking/work_5_4_baseline"
TRIALS = [("control", {}, None), ("crossclass_suppression_0_7", {}, 0.7), ("new_track_0_4", {"new_track_thresh": 0.4}, None), ("match_0_9", {"match_thresh": 0.9}, None)]


def read_rows(path, contract):
    return [contract.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def detection_profile(detections):
    """Count strongly overlapping detection pairs and whether their predicted classes disagree."""
    same = different = 0
    pairs = Counter()
    for result in detections:
        items = result.detections
        for position, first in enumerate(items):
            left, top, right, bottom = first.bbox_xyxy
            for second in items[position + 1:]:
                if overlap((left, top, right - left, bottom - top), second.bbox_xyxy) >= 0.7:
                    if first.class_id == second.class_id:
                        same += 1
                    else:
                        different += 1
                        pairs["+".join(sorted((first.class_name, second.class_name)))] += 1
    return {"detections": sum(len(result.detections) for result in detections), "same_class_pairs_iou_0_7": same,
            "cross_class_pairs_iou_0_7": different, "cross_class_pair_names": dict(pairs.most_common())}


def duplicate_ids(result):
    found = set()
    for position, track in enumerate(result.tracks):
        left, top, right, bottom = track.bbox_xyxy
        for other in result.tracks[position + 1:]:
            if overlap((left, top, right - left, bottom - top), other.bbox_xyxy) >= 0.7:
                found.update((track.track_id, other.track_id))
    return found


def draw_panel(image, result, title):
    import cv2

    panel = image.copy()
    repeated = duplicate_ids(result)
    for track in result.tracks:
        left, top, right, bottom = map(int, track.bbox_xyxy)
        color = (0, 0, 255) if track.track_id in repeated else (255, 255, 0)
        cv2.rectangle(panel, (left, top), (right, bottom), color, 2)
        cv2.putText(panel, f"P{track.track_id} {track.class_name}", (left, max(60, top)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.rectangle(panel, (0, 0), (1920, 45), (0, 0, 0), -1)
    cv2.putText(panel, f"{title} | MOT {result.frame.mot_frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return cv2.resize(panel, (960, 540))


def write_qa_page():
    """Show the unchanged tracker beside cross-class suppression; both are candidates for human review, neither is selected."""
    import cv2
    import numpy as np

    control = read_rows(OUTPUT / "control/tracking_results.jsonl", TrackingResult)
    variant = read_rows(OUTPUT / "crossclass_suppression_0_7/tracking_results.jsonl", TrackingResult)
    video = cv2.VideoWriter(str(OUTPUT / "control_vs_suppression.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1920, 540))
    if not video.isOpened():
        raise RuntimeError("QA writer unavailable")
    rows = []
    try:
        for left, right in zip(control, variant):
            frame_id = left.frame.mot_frame_id
            image = cv2.imread(str(experiments.PILOT / "img1" / f"{frame_id:06d}.jpg"))
            video.write(np.hstack([draw_panel(image, left, "GIU NGUYEN"), draw_panel(image, right, "LOAI TRUNG KHAC LOP")]))
            rows.append((len(duplicate_ids(left)) - len(duplicate_ids(right)), frame_id))
    finally:
        video.release()
    decoded = cv2.VideoCapture(str(OUTPUT / "control_vs_suppression.mp4"))
    count = 0
    while decoded.read()[0]:
        count += 1
    decoded.release()
    if count != len(control):
        raise RuntimeError("QA video validation failed")
    report = json.loads((OUTPUT / "comparison.json").read_text(encoding="utf-8"))
    labels = [("coverage", "Độ phủ GT"), ("consecutive_id_change_candidates", "Ứng viên đổi ID"), ("overlapping_prediction_pairs_iou_0_7", "Cặp box trùng IoU≥0,7"), ("unique_prediction_ids", "Số ID dự đoán"), ("mean_prediction_ids_per_matched_gt_track", "ID trung bình mỗi track GT")]
    table = "".join(f"<tr><td>{name}</td>" + "".join(f"<td>{report['trials'][trial][key]:.4g}</td>" for trial, _, _ in TRIALS) + "</tr>" for key, name in labels)
    jumps = sorted(rows, reverse=True)[:20]
    buttons = "".join(f'<button onclick="const v=document.querySelector(\'video\');v.currentTime={(frame_id - 1) / 10:.1f};v.pause();">MOT {frame_id}</button> ' for difference, frame_id in sorted(jumps, key=lambda row: row[1]) if difference > 0)
    epochs = json.loads((BASE / "report.json").read_text(encoding="utf-8"))["checkpoint_completed_epochs"]
    (OUTPUT / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>So sánh tracker epoch {epochs}</title>' + '<style>body{background:#181818;color:white;font:18px sans-serif;margin:20px}video{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #777;padding:4px 10px}button{margin:2px;font-size:16px}</style>' + f'<h1>So sánh tracker trên detector epoch {epochs} — cần người xem</h1>' + '<p>Trái: giữ nguyên. Phải: loại box trùng khác lớp IoU&gt;0,7. Box đỏ là dự đoán đang trùng nhau; box xanh là dự đoán khác. Video phát 10 fps (nguồn 30 fps). Khi xem, kiểm tra hai điều: box đỏ bên trái có đúng là một xe bị gán hai lớp không, và bên phải có làm mất xe thật đang bị che không.</p><table><tr><th>Chỉ số chẩn đoán</th><th>Giữ nguyên</th><th>Loại trùng khác lớp</th><th>new_track 0,4</th><th>match 0,9</th></tr>' + table + '</table><p>Đây không phải IDF1/HOTA/MOTA; số ứng viên giảm chưa chứng minh chất lượng tăng. Chưa chọn cấu hình nào.</p><video controls src="control_vs_suppression.mp4"></video><h2>Nhảy tới frame khác biệt nhiều nhất</h2>' + buttons, encoding="utf-8")


def main():
    global BASE, OUTPUT
    parser = argparse.ArgumentParser(description="Replay tracker variants on the cached detections of a detector run")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--qa-only", action="store_true", help="Rebuild the review page from existing replay outputs")
    parser.add_argument("--base", type=Path, default=BASE)
    arguments = parser.parse_args()
    BASE, OUTPUT = arguments.base, arguments.base / "experiments"
    if arguments.qa_only:
        write_qa_page()
        return
    if not arguments.execute:
        raise PermissionError("Owner authorization and --execute required")
    detections = read_rows(BASE / "detections.jsonl", DetectionResult)
    control = {row.frame.mot_frame_id: row for row in read_rows(BASE / "tracking_results.jsonl", TrackingResult)}
    if [row.frame.mot_frame_id for row in detections] != list(range(1, 502)):
        raise ValueError("Invalid detection cache ordering")
    gt_frames = defaultdict(dict)
    for line in experiments.GT.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gt_frames[row["mot_frame_id"]][row["track_id"]] = row
    experiments.OUTPUT = OUTPUT
    OUTPUT.mkdir(exist_ok=False)
    summaries = {}
    for name, changes, suppression in TRIALS:
        experiments.execute(name, changes, suppression, detections, gt_frames, control)
        results = read_rows(OUTPUT / name / "tracking_results.jsonl", TrackingResult)
        summaries[name] = {"changes": changes, "suppression_iou": suppression, **identity_diagnostics(gt_frames, results)}
    epochs = json.loads((BASE / "report.json").read_text(encoding="utf-8"))["checkpoint_completed_epochs"]
    report = {"status": "EXECUTION_PASS_HUMAN_REVIEW_REQUIRED", f"control_matches_epoch{epochs}_run_exactly": True, "yolo_rerun": False, "gt_modified": False,
              "selected_configuration": None, "official_metrics": None, "detection_profile": {"epoch_19": detection_profile(read_rows(BASELINE / "detections.jsonl", DetectionResult)), f"epoch_{epochs}": detection_profile(detections)},
              "trials": summaries}
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_qa_page()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
