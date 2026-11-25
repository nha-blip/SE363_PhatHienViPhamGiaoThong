import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256

os.environ["YOLO_AUTOINSTALL"] = "false"


ROOT = Path(__file__).resolve().parents[1]
# Validation split extracted from vehicle_trainval.zip (sha256 d4fe27fb...), the exact 21460 images / 78772 boxes the Kaggle run validated on.
DATASET = ROOT / "data/Vehicle363_val_from_kaggle_archive_20261102"
WEIGHTS = ROOT / "weights/yolo26_vehicle_restart03_20261031/yolo26_fullscene_resume_20261028/vehicle/checkpoints/best.pt"
EXPECTED_SHA256 = "2cefe325048ba9338da75b76d893b7f8e0ba1e5aba6f2db2ac392aac8af50e67"
OUTPUT = ROOT / "runs/project363/yolo_error_analysis_vehicle_epoch41_kaggle_val_20261102"
NAMES = ["car", "motorcycle", "bus", "truck"]
TASKS = {
    "vehicle": {"dataset": DATASET, "weights": WEIGHTS, "sha256": EXPECTED_SHA256, "names": NAMES},
    "light": {"dataset": ROOT / "data/Light363_val_copy_20261102", "weights": ROOT / "weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt",
              "sha256": "4275b891940217966ab03ff7629d588063477eb60c03a57014a2941cce73c056", "names": ["red", "yellow", "green", "off"]},
    # YOLOv8s baselines (Ultralytics 8.4.178 checkpoints from the October Kaggle runs), scored on the same validation splits as YOLO26s.
    "vehicle_v8s": {"dataset": DATASET, "weights": ROOT / "weights/kaggle_small_vehicle_20261019/best.pt",
                    "sha256": "e1254d4c171593ded3f57939163d1cb2b2aa4e93a0a29dfa332824555a7cc35f", "names": NAMES},
    "light_v8s": {"dataset": ROOT / "data/Light363_val_copy_20261102", "weights": ROOT / "weights/trained_yolo_20261018/small/traffic_light/best.pt",
                  "sha256": "cdbe8a06f9324bca1058f215467b24b6cc2a94e43aaeeefd034304ff5b6dabe4", "names": ["red", "yellow", "green", "off"]},
    "light_ft960": {"dataset": ROOT / "data/Light363_val_copy_20261102", "weights": ROOT / "runs/project363/yolo26_light_finetune960_local_20261103/train/weights/best.pt",
                    "sha256": "14464b9dd2c6dcb06f1c02640c729dd4e2a0680a58eeb2e0fb4bc67507b6678a", "names": ["red", "yellow", "green", "off"]},
    "vehicle_local49": {"dataset": DATASET, "weights": ROOT / "runs/project363/yolo26_vehicle_local_resume_20261104/epoch49_last_copy.pt",
                        "sha256": "9650eb68b47febb5799b1194a99d4f4a5806ef1b4b401b22f242f91d60fd02f8", "names": NAMES},
    "vehicle_local_best": {"dataset": DATASET, "weights": ROOT / "runs/project363/yolo26_vehicle_local_resume_20261104/final_best_copy.pt",
                           "sha256": "2ae7f03ca5d53da6447e687d0d66ff89d304aa661adef4aba8136d43875a26d9", "names": NAMES},
    "test_vehicle_e63": {"dataset": ROOT / "data/_test_view/vehicle", "weights": ROOT / "runs/project363/yolo26_vehicle_local_resume_20261104/final_best_copy.pt", "sha256": "2ae7f03ca5d53da6447e687d0d66ff89d304aa661adef4aba8136d43875a26d9", "names": NAMES},
    "test_vehicle_v8s": {"dataset": ROOT / "data/_test_view/vehicle", "weights": ROOT / "weights/kaggle_small_vehicle_20261019/best.pt", "sha256": "e1254d4c171593ded3f57939163d1cb2b2aa4e93a0a29dfa332824555a7cc35f", "names": NAMES},
    "test_light_A": {"dataset": ROOT / "data/_test_view/light", "weights": ROOT / "weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt", "sha256": "4275b891940217966ab03ff7629d588063477eb60c03a57014a2941cce73c056", "names": ["red", "yellow", "green", "off"]},
}
CONFIDENCE = 0.25
MATCH_IOU = 0.5
BUCKETS = [(0, 16, "<16"), (16, 32, "16-32"), (32, 64, "32-64"), (64, 10 ** 6, ">=64")]


def read_labels(path, width, height):
    classes, boxes = [], []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            identity, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            classes.append(identity)
            boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(classes, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(first, second):
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)))
    size = np.maximum(0, np.minimum(first[:, None, 2:], second[None, :, 2:]) - np.maximum(first[:, None, :2], second[None, :, :2]))
    intersection = size.prod(axis=2)
    area_first = np.maximum(0, first[:, 2:] - first[:, :2]).prod(axis=1)
    area_second = np.maximum(0, second[:, 2:] - second[:, :2]).prod(axis=1)
    return intersection / (area_first[:, None] + area_second[None, :] - intersection + 1e-9)


def match(overlap):
    """Greedy class-agnostic one-to-one matching at MATCH_IOU, highest overlap first."""
    pairs = np.argwhere(overlap >= MATCH_IOU)
    pairs = pairs[np.argsort(-overlap[pairs[:, 0], pairs[:, 1]])] if len(pairs) else pairs
    used_gt, used_pred, matches = set(), set(), []
    for gt_index, pred_index in pairs:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((int(gt_index), int(pred_index)))
    return matches


def bucket(height):
    return next(label for low, high, label in BUCKETS if low <= height < high)


def camera_key(name):
    parts = name.split("__")
    if len(parts) < 2:
        return "unknown"
    camera = re.sub(r"_\d+_jpg$|_\d+$", "", parts[1])
    return f"{parts[0]}/{camera}"


def contact_sheet(path, items, title):
    import cv2

    tiles = []
    for item in items:
        image = cv2.imread(item["image"])
        if image is None:
            continue
        left, top, right, bottom = map(int, item["box"])
        margin = int(max(right - left, bottom - top) * 0.6) + 20
        x0, y0, x1, y1 = max(0, left - margin), max(0, top - margin), min(image.shape[1], right + margin), min(image.shape[0], bottom + margin)
        crop = image[y0:y1, x0:x1].copy()
        cv2.rectangle(crop, (left - x0, top - y0), (right - x0, bottom - y0), (0, 255, 255), 2)
        scale = 256 / max(crop.shape[:2])
        tile = np.zeros((256, 256, 3), dtype=np.uint8)
        resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
        tile[:resized.shape[0], :resized.shape[1]] = resized
        cv2.putText(tile, item["label"], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros((256, 256, 3), dtype=np.uint8))
    rows = [np.hstack(tiles[start:start + 6]) for start in range(0, len(tiles), 6)]
    cv2.imwrite(str(path), np.vstack(rows))
    return title


def main():
    parser = argparse.ArgumentParser(description="Error analysis of the epoch-41 vehicle checkpoint on the local validation split")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-official-val", action="store_true")
    parser.add_argument("--task", choices=sorted(TASKS), default="vehicle")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", choices=("val", "test"), default="val", help="test points the task at the junction view of the held-out test split")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Use --execute; this runs the detector on the local GPU")
    global DATASET, WEIGHTS, EXPECTED_SHA256, NAMES, OUTPUT
    task = TASKS[arguments.task]
    DATASET, WEIGHTS, EXPECTED_SHA256, NAMES = task["dataset"], task["weights"], task["sha256"], task["names"]
    if (arguments.task, arguments.imgsz) != ("vehicle", 640):
        OUTPUT = ROOT / f"runs/project363/yolo_error_analysis_{arguments.task}_imgsz{arguments.imgsz}_20261102"
    if arguments.split == "test":
        DATASET = ROOT / ("data/_test_view/vehicle" if NAMES[0] == "car" else "data/_test_view/light")
        OUTPUT = ROOT / f"runs/project363/yolo_error_analysis_TEST_{arguments.task}_imgsz{arguments.imgsz}_20261112"
    if sha256(WEIGHTS) != EXPECTED_SHA256:
        raise ValueError("Checkpoint hash mismatch")
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(WEIGHTS))
    if [model.names[index] for index in sorted(model.names)] != NAMES:
        raise ValueError("Unexpected class names")
    report = {"checkpoint_sha256": EXPECTED_SHA256, "test_used": arguments.split == "test" or arguments.task.startswith("test_"), "dataset": str(DATASET),
              "note": "Same 21460 images / 78772 boxes as the Kaggle validation; the local YOLO363_balanced_720p variant is a different split and scores about 4 points lower"}
    if not arguments.skip_official_val:
        metrics = model.val(data=str(DATASET / "data.yaml"), split="val", imgsz=arguments.imgsz, batch=16, conf=0.001, iou=0.7, device=0, workers=2,
                            plots=False, project=str(OUTPUT), name="official_val", exist_ok=False, verbose=False)
        report["official_val"] = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "map50": float(metrics.box.map50),
                                  "map50_95": float(metrics.box.map), "map50_95_per_class": dict(zip(NAMES, map(float, metrics.box.maps)))}
        print(json.dumps(report["official_val"]), flush=True)
    images = sorted(path for path in (DATASET / "images/val").iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    counts = {name: Counter() for name in NAMES}
    confusion = defaultdict(Counter)
    recall_by_size = {name: {label: [0, 0] for _, _, label in BUCKETS} for name in NAMES}
    cameras = defaultdict(Counter)
    false_positives, missed = [], []
    started = time.perf_counter()
    # A directory source is read lazily; a list of paths makes Ultralytics open every file at once and hits the Windows open-file limit.
    stream = model.predict(source=str(DATASET / "images/val"), stream=True, batch=16, imgsz=arguments.imgsz, conf=CONFIDENCE, iou=0.7, device=0, half=True, verbose=False)
    for position, result in enumerate(stream):
        path = Path(result.path)
        height, width = result.orig_shape
        gt_classes, gt_boxes = read_labels(DATASET / "labels/val" / (path.stem + ".txt"), width, height)
        predicted = result.boxes
        pred_boxes = predicted.xyxy.cpu().numpy()
        pred_classes = predicted.cls.cpu().numpy().astype(int)
        confidences = predicted.conf.cpu().numpy()
        overlap = iou_matrix(gt_boxes, pred_boxes)
        matches = match(overlap)
        matched_gt = {gt for gt, _ in matches}
        matched_pred = {pred for _, pred in matches}
        camera = camera_key(path.name)
        for gt_index, pred_index in matches:
            gt_name, pred_name = NAMES[gt_classes[gt_index]], NAMES[pred_classes[pred_index]]
            size = bucket(gt_boxes[gt_index, 3] - gt_boxes[gt_index, 1])
            recall_by_size[gt_name][size][1] += 1
            if gt_classes[gt_index] == pred_classes[pred_index]:
                counts[gt_name]["tp"] += 1
                recall_by_size[gt_name][size][0] += 1
                cameras[camera]["tp"] += 1
            else:
                counts[gt_name]["wrong_class_gt"] += 1
                counts[pred_name]["wrong_class_pred"] += 1
                confusion[gt_name][pred_name] += 1
                cameras[camera]["wrong"] += 1
        for gt_index in range(len(gt_classes)):
            if gt_index in matched_gt:
                continue
            name = NAMES[gt_classes[gt_index]]
            size = bucket(gt_boxes[gt_index, 3] - gt_boxes[gt_index, 1])
            recall_by_size[name][size][1] += 1
            counts[name]["missed"] += 1
            cameras[camera]["missed"] += 1
            area = float((gt_boxes[gt_index, 2] - gt_boxes[gt_index, 0]) * (gt_boxes[gt_index, 3] - gt_boxes[gt_index, 1]))
            missed.append({"image": str(path), "box": gt_boxes[gt_index].tolist(), "class": name, "area": area})
        for pred_index in range(len(pred_classes)):
            if pred_index in matched_pred:
                continue
            name = NAMES[pred_classes[pred_index]]
            nearest = float(overlap[:, pred_index].max()) if len(gt_classes) else 0.0
            kind = "fp_near_gt" if nearest >= 0.1 else "fp_isolated"
            counts[name][kind] += 1
            cameras[camera][kind] += 1
            if kind == "fp_isolated":
                false_positives.append({"image": str(path), "box": pred_boxes[pred_index].tolist(), "class": name, "confidence": float(confidences[pred_index])})
        cameras[camera]["gt"] += len(gt_classes)
        if (position + 1) % 2000 == 0:
            print(f"ANALYSIS {position + 1}/{len(images)} {time.perf_counter() - started:.0f}s", flush=True)
    summary = {}
    for name in NAMES:
        entry = counts[name]
        gt_total = entry["tp"] + entry["wrong_class_gt"] + entry["missed"]
        predicted_total = entry["tp"] + entry["wrong_class_pred"] + entry["fp_near_gt"] + entry["fp_isolated"]
        localized = entry["tp"] + entry["wrong_class_gt"]
        predicted_localized = entry["tp"] + entry["wrong_class_pred"]
        summary[name] = {**entry, "gt": gt_total, "recall_at_conf_0_25": entry["tp"] / gt_total if gt_total else None,
                         "class_accuracy_among_localized": entry["tp"] / localized if localized else None,
                         "class_precision_among_localized": entry["tp"] / predicted_localized if predicted_localized else None,
                         "precision_at_conf_0_25": entry["tp"] / predicted_total if predicted_total else None,
                         "recall_by_gt_height_px": {label: (hit / total if total else None, total) for label, (hit, total) in recall_by_size[name].items()}}
    total_tp = sum(entry["tp"] for entry in counts.values())
    total_localized = total_tp + sum(entry["wrong_class_gt"] for entry in counts.values())
    report.update(task=arguments.task, imgsz=arguments.imgsz, class_accuracy_among_localized=total_tp / total_localized if total_localized else None,
                  images=len(images), confidence=CONFIDENCE, match_iou=MATCH_IOU, per_class=summary,
                  confusion_gt_to_predicted={gt: dict(row) for gt, row in confusion.items()})
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (OUTPUT / "by_camera.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["camera", "gt", "tp", "wrong_class", "missed", "fp_near_gt", "fp_isolated", "recall"])
        for camera, entry in sorted(cameras.items(), key=lambda item: -item[1]["gt"]):
            writer.writerow([camera, entry["gt"], entry["tp"], entry["wrong"], entry["missed"], entry["fp_near_gt"], entry["fp_isolated"],
                             round(entry["tp"] / entry["gt"], 4) if entry["gt"] else ""])
    top_false = sorted(false_positives, key=lambda item: -item["confidence"])[:24]
    large_missed = sorted(missed, key=lambda item: -item["area"])[:24]
    contact_sheet(OUTPUT / "fp_isolated_top_confidence.jpg", [{**item, "label": f"FP {item['class']} {item['confidence']:.2f}"} for item in top_false], "fp")
    contact_sheet(OUTPUT / "missed_largest.jpg", [{**item, "label": f"MISSED {item['class']}"} for item in large_missed], "missed")
    (OUTPUT / "examples.json").write_text(json.dumps({"fp_isolated_top": top_false, "missed_largest": large_missed}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
