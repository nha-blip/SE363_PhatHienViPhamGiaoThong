import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_vehicle_val_errors as base

GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
BASE_CONFIDENCE = 0.25
MIN_HEIGHT = 32


def collect(task, imgsz):
    from ultralytics import YOLO

    configuration = base.TASKS[task]
    if base.sha256(configuration["weights"]) != configuration["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    dataset = configuration["dataset"]
    model = YOLO(str(configuration["weights"]))
    stream = model.predict(source=str(dataset / "images/val"), stream=True, batch=16, imgsz=imgsz, conf=min(GRID), iou=0.7, device=0, half=True, verbose=False)
    images = []
    for result in stream:
        height, width = result.orig_shape
        gt_classes, gt_boxes = base.read_labels(dataset / "labels/val" / (Path(result.path).stem + ".txt"), width, height)
        boxes = result.boxes
        images.append((gt_classes, gt_boxes, boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()))
    return images


def score(images, thresholds):
    """thresholds: per-class confidence. Returns per-class recall/precision over all GT and over GT with height >= MIN_HEIGHT."""
    tp = np.zeros(4); gt_total = np.zeros(4); predicted = np.zeros(4)
    tp_large = np.zeros(4); gt_large = np.zeros(4)
    for gt_classes, gt_boxes, boxes, classes, confidences in images:
        keep = confidences >= np.asarray(thresholds)[classes] if len(classes) else np.zeros(0, bool)
        boxes, classes = boxes[keep], classes[keep]
        matched = {gt: pred for gt, pred in base.match(base.iou_matrix(gt_boxes, boxes))}
        large = (gt_boxes[:, 3] - gt_boxes[:, 1]) >= MIN_HEIGHT if len(gt_boxes) else np.zeros(0, bool)
        for index, label in enumerate(gt_classes):
            gt_total[label] += 1
            gt_large[label] += large[index]
            if index in matched.values() or index in matched:
                pass
            if index in matched and classes[matched[index]] == label:
                tp[label] += 1
                tp_large[label] += large[index]
        for label in classes:
            predicted[label] += 1
    return {"recall": tp / gt_total, "recall_large": tp_large / gt_large, "precision": tp / np.maximum(predicted, 1), "gt": gt_total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--imgsz", type=int, default=960)
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Use --execute; this runs the detector on the local GPU")
    output = base.ROOT / f"runs/project363/light_threshold_sweep_imgsz{arguments.imgsz}_20261104"
    output.mkdir(parents=True, exist_ok=False)
    names = ["red", "yellow", "green", "off"]
    report = {}
    for task in ("light", "light_ft960"):
        images = collect(task, arguments.imgsz)
        rows = []
        for threshold in GRID:
            result = score(images, [threshold] * 4)
            rows.append({"threshold": threshold, **{key: dict(zip(names, map(float, result[key]))) for key in ("recall", "recall_large", "precision")}})
        report[task] = rows
        print(task, flush=True)
        for row in rows:
            print(row["threshold"], " ".join(f"{n}:{row['recall'][n]*100:.1f}/{row['recall_large'][n]*100:.1f}/P{row['precision'][n]*100:.0f}" for n in names), flush=True)
    (output / "sweep.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
