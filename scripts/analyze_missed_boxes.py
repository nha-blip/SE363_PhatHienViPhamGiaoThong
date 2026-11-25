import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_vehicle_val_errors import CONFIDENCE, ROOT, TASKS, bucket, contact_sheet, iou_matrix, match, read_labels
from src.tracking.io import sha256

os.environ["YOLO_AUTOINSTALL"] = "false"

LOW_CONFIDENCE = 0.05
FIELDS = ["image", "source", "cls", "status", "reason", "x1", "y1", "x2", "y2", "width_px", "height_px", "size_bucket", "area_fraction", "aspect_w_over_h",
          "center_y_fraction", "touches_edge", "max_iou_other_gt", "overlapping_gt", "gt_in_image", "brightness", "best_iou_any_pred", "best_pred_confidence"]


def analyse_image(gt_classes, gt_boxes, pred_boxes, pred_classes, confidences, width, height):
    """Classify every ground-truth box as correct, wrong class or missed at CONFIDENCE, and say why a miss happened."""
    overlap = iou_matrix(gt_boxes, pred_boxes)
    strong = np.flatnonzero(confidences >= CONFIDENCE)
    matched = {int(gt): int(strong[pred]) for gt, pred in match(overlap[:, strong])} if len(strong) else {}
    taken = set(matched.values())
    among_gt = iou_matrix(gt_boxes, gt_boxes)
    np.fill_diagonal(among_gt, 0)
    rows = []
    for index in range(len(gt_classes)):
        left, top, right, bottom = gt_boxes[index]
        row_overlap = overlap[index] if overlap.shape[1] else np.zeros(0)
        best = float(row_overlap.max()) if len(row_overlap) else 0.0
        best_confidence = float(confidences[int(row_overlap.argmax())]) if len(row_overlap) and best > 0 else 0.0
        if index in matched:
            status = "TP" if gt_classes[index] == pred_classes[matched[index]] else "WRONG_CLASS"
            reason = ""
        else:
            status = "MISSED"
            close = np.flatnonzero(row_overlap >= 0.5)
            if any(int(pred) in taken for pred in close):
                reason = "box_used_by_another_gt"
            elif len(close):
                reason = "low_confidence_correct_class" if pred_classes[close[0]] == gt_classes[index] else "low_confidence_wrong_class"
            elif best >= 0.1:
                reason = "poor_localization"
            else:
                reason = "no_detection"
        rows.append({"cls": int(gt_classes[index]), "status": status, "reason": reason, "x1": left, "y1": top, "x2": right, "y2": bottom,
                     "width_px": right - left, "height_px": bottom - top, "size_bucket": bucket(bottom - top),
                     "area_fraction": (right - left) * (bottom - top) / (width * height), "aspect_w_over_h": (right - left) / max(bottom - top, 1e-6),
                     "center_y_fraction": (top + bottom) / 2 / height, "touches_edge": int(left <= 2 or top <= 2 or right >= width - 2 or bottom >= height - 2),
                     "max_iou_other_gt": float(among_gt[index].max()) if len(gt_classes) > 1 else 0.0,
                     "overlapping_gt": int((among_gt[index] > 0.3).sum()), "gt_in_image": len(gt_classes), "best_iou_any_pred": best, "best_pred_confidence": best_confidence})
    return rows


LEVELS = {
    "size_bucket": lambda row: row["size_bucket"],
    "crowding_max_iou_with_other_gt": lambda row: "<0.1" if row["max_iou_other_gt"] < 0.1 else "0.1-0.3" if row["max_iou_other_gt"] < 0.3 else "0.3-0.5" if row["max_iou_other_gt"] < 0.5 else ">=0.5",
    "touches_image_edge": lambda row: str(bool(row["touches_edge"])),
    "image_brightness": lambda row: "dark<50" if row["brightness"] < 50 else "mid50-100" if row["brightness"] < 100 else "bright>=100",
    "vertical_position": lambda row: "top_third" if row["center_y_fraction"] < 1 / 3 else "middle_third" if row["center_y_fraction"] < 2 / 3 else "bottom_third",
    "objects_in_image": lambda row: "1-5" if row["gt_in_image"] <= 5 else "6-15" if row["gt_in_image"] <= 15 else "16+",
    "aspect_w_over_h": lambda row: "<0.5" if row["aspect_w_over_h"] < 0.5 else "0.5-1" if row["aspect_w_over_h"] < 1 else "1-2" if row["aspect_w_over_h"] < 2 else ">=2",
    "source": lambda row: row["source"],
}


def tables(rows, names):
    total_missed = sum(row["status"] == "MISSED" for row in rows)
    result = {"all_classes": {}, "motorcycle_only": {}}
    for scope, selected in (("all_classes", rows), ("motorcycle_only", [row for row in rows if names[row["cls"]] == "motorcycle"])):
        for factor, level in LEVELS.items():
            groups = {}
            for row in selected:
                entry = groups.setdefault(level(row), [0, 0])
                entry[0] += 1
                entry[1] += row["status"] == "MISSED"
            result[scope][factor] = {key: {"gt": gt, "missed": missed, "miss_rate": round(missed / gt, 4), "share_of_all_misses": round(missed / total_missed, 4)}
                                     for key, (gt, missed) in sorted(groups.items())}
    return result


def main():
    parser = argparse.ArgumentParser(description="Characterise ground-truth boxes the detector misses")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task", choices=sorted(TASKS), default="vehicle")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="Analyse a random sample of this many validation images instead of all")
    parser.add_argument("--batch", type=int, default=16, help="Images per forward pass; lower it for large --imgsz on a 6 GB GPU")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Use --execute; this runs the detector on the local GPU")
    task = TASKS[arguments.task]
    names, dataset, weights = task["names"], task["dataset"], task["weights"]
    if sha256(weights) != task["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    suffix = f"_sample{arguments.limit}" if arguments.limit else ""
    output = ROOT / f"runs/project363/yolo_missed_boxes_{arguments.task}_imgsz{arguments.imgsz}{suffix}_20261102"
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(weights))
    rows = []
    started = time.perf_counter()
    settings = dict(stream=True, batch=arguments.batch, imgsz=arguments.imgsz, conf=LOW_CONFIDENCE, iou=0.7, device=0, half=True, verbose=False)

    def predictions():
        if not arguments.limit:
            # A directory source is read lazily; a long list of paths opens every file at once and hits the Windows open-file limit.
            yield from model.predict(source=str(dataset / "images/val"), **settings)
            return
        images = sorted(path for path in (dataset / "images/val").iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        chosen = random.Random(363).sample(images, min(arguments.limit, len(images)))
        # A list source is run as one single batch, so the list length must be the batch size.
        for start in range(0, len(chosen), arguments.batch):
            yield from model.predict(source=[str(path) for path in chosen[start:start + arguments.batch]], **settings)

    for position, result in enumerate(predictions()):
        path = Path(result.path)
        height, width = result.orig_shape
        gt_classes, gt_boxes = read_labels(dataset / "labels/val" / (path.stem + ".txt"), width, height)
        if not len(gt_classes):
            continue
        brightness = float(result.orig_img[::4, ::4].mean())
        found = analyse_image(gt_classes, gt_boxes, result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int),
                              result.boxes.conf.cpu().numpy(), width, height)
        for row in found:
            row.update(image=str(path), source=path.name.split("__")[0], brightness=brightness)
        rows.extend(found)
        if (position + 1) % 2000 == 0:
            print(f"MISSED_BOX_ANALYSIS {position + 1} {time.perf_counter() - started:.0f}s", flush=True)
    with (output / "gt_boxes.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "cls": names[row["cls"]], **{key: round(row[key], 4) for key in ("x1", "y1", "x2", "y2", "width_px", "height_px", "area_fraction",
                             "aspect_w_over_h", "center_y_fraction", "max_iou_other_gt", "brightness", "best_iou_any_pred", "best_pred_confidence")}})
    missed = [row for row in rows if row["status"] == "MISSED"]
    reasons = {"all_classes": dict(Counter(row["reason"] for row in missed)),
               **{f"{name}_{label}": dict(Counter(row["reason"] for row in missed if names[row["cls"]] == name and row["size_bucket"] == label))
                  for name in names for label in ("32-64", ">=64", "16-32")}}
    report = {"task": arguments.task, "imgsz": arguments.imgsz, "confidence": CONFIDENCE, "low_confidence_floor": LOW_CONFIDENCE, "test_used": False,
              "gt_boxes": len(rows), "status": dict(Counter(row["status"] for row in rows)), "miss_reasons": reasons, "miss_rate_tables": tables(rows, names)}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    generator = random.Random(363)
    groups = {"missed_random": missed, "missed_motorcycle_32_64": [row for row in missed if names[row["cls"]] == "motorcycle" and row["size_bucket"] == "32-64"],
              "missed_box_used_by_another_gt": [row for row in missed if row["reason"] == "box_used_by_another_gt"]}
    for name, pool in groups.items():
        picked = generator.sample(pool, min(24, len(pool)))
        contact_sheet(output / f"{name}.jpg", [{"image": row["image"], "box": [row["x1"], row["y1"], row["x2"], row["y2"]],
                                                "label": f"{names[row['cls']]} {row['reason'][:22]} h={row['height_px']:.0f}"} for row in picked], name)
    print(json.dumps({"status": report["status"], "miss_reasons": reasons["all_classes"]}), flush=True)


if __name__ == "__main__":
    main()
