import argparse
import csv
import html
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_vehicle_val_errors import CONFIDENCE, ROOT, TASKS, iou_matrix, match, read_labels
from src.tracking.io import sha256

os.environ["YOLO_AUTOINSTALL"] = "false"

OUTPUT = ROOT / "runs/project363/vehicle_label_audit_20261103"
STYLE = ('<!doctype html><meta charset="utf-8"><style>body{background:#181818;color:#eee;font:15px sans-serif;margin:20px}'
         '.grid{display:flex;flex-wrap:wrap;gap:14px}.card{width:740px;background:#242424;padding:8px}img{max-width:100%}p{margin:6px 0}small{color:#999}</style>')


def crop_card(image_path, boxes, focus, destination, name):
    """Save a zoomed crop around the focus box; the focus box is thick red, the other boxes are thin yellow."""
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    left, top, right, bottom = focus
    span = max(360, 4 * max(right - left, bottom - top))
    x0 = int(max(0, min(width - span, (left + right) / 2 - span / 2)))
    y0 = int(max(0, min(height - span * 0.75, (top + bottom) / 2 - span * 0.375)))
    x1, y1 = int(min(width, x0 + span)), int(min(height, y0 + span * 0.75))
    crop = image[y0:y1, x0:x1].copy()
    for box in boxes:
        cv2.rectangle(crop, (int(box[0] - x0), int(box[1] - y0)), (int(box[2] - x0), int(box[3] - y0)), (0, 220, 255), 1)
    cv2.rectangle(crop, (int(left - x0), int(top - y0)), (int(right - x0), int(bottom - y0)), (0, 0, 255), 2)
    crop = cv2.resize(crop, None, fx=min(3.0, 720 / crop.shape[1]), fy=min(3.0, 720 / crop.shape[1]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(destination / f"{name}.jpg"), crop)
    return True


def write_page(path, title, note, cards):
    path.write_text(STYLE + f"<h1>{html.escape(title)}</h1><p>{html.escape(note)}</p><div class=\"grid\">{''.join(cards)}</div>", encoding="utf-8")


def truck_called_car(dataset, weights, imgsz, batch, count, seed):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images = sorted(path for path in (dataset / "images/val").iterdir() if path.name.startswith("mio__"))
    candidates = [path for path in images
                  if any(line.split()[0] == "3" for line in (dataset / "labels/val" / (path.stem + ".txt")).read_text().splitlines() if line.strip())]
    found = []
    for start in range(0, len(candidates), batch):
        chunk = candidates[start:start + batch]
        for result in model.predict(source=[str(path) for path in chunk], imgsz=imgsz, conf=0.05, iou=0.7, device=0, half=True, verbose=False, batch=batch):
            path = Path(result.path)
            height, width = result.orig_shape
            classes, boxes = read_labels(dataset / "labels/val" / (path.stem + ".txt"), width, height)
            predicted = result.boxes.cls.cpu().numpy().astype(int)
            confidence = result.boxes.conf.cpu().numpy()
            strong = np.flatnonzero(confidence >= CONFIDENCE)
            overlap = iou_matrix(boxes, result.boxes.xyxy.cpu().numpy())
            matched = {int(gt): int(strong[pred]) for gt, pred in match(overlap[:, strong])} if len(strong) else {}
            for gt, pred in matched.items():
                if classes[gt] == 3 and predicted[pred] == 0:
                    found.append({"image": path, "box": boxes[gt].tolist(), "others": [box.tolist() for index, box in enumerate(boxes) if index != gt],
                                  "confidence": float(confidence[pred]), "height": float(boxes[gt][3] - boxes[gt][1])})
        print(f"TRUCK_AS_CAR {start + len(chunk)}/{len(candidates)} found={len(found)}", flush=True)
    return len(candidates), found, random.Random(seed).sample(found, min(count, len(found)))


def overlapping_motorcycles(dataset, count, seed):
    pairs = []
    for label in sorted((dataset / "labels/val").glob("*.txt")):
        if sum(line.split()[0] == "1" for line in label.read_text().splitlines() if line.strip()) < 2:
            continue
        path = next((dataset / "images/val").glob(label.stem + ".*"), None)
        if path is None:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        classes, boxes = read_labels(label, width, height)
        motorcycles = [boxes[index] for index in range(len(classes)) if classes[index] == 1]
        if len(motorcycles) < 2:
            continue
        overlap = iou_matrix(np.array(motorcycles), np.array(motorcycles))
        np.fill_diagonal(overlap, 0)
        first, second = np.unravel_index(overlap.argmax(), overlap.shape)
        if overlap[first, second] >= 0.3:
            pairs.append({"image": path, "box": motorcycles[first].tolist(), "other": motorcycles[second].tolist(), "iou": float(overlap[first, second]),
                          "others": [box.tolist() for box in motorcycles]})
    return pairs, random.Random(seed).sample(pairs, min(count, len(pairs)))


def main():
    parser = argparse.ArgumentParser(description="Export label-audit galleries: GT truck predicted as car, and overlapping motorcycle labels")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--count", type=int, default=60)
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Use --execute; this runs the detector on the local GPU")
    task = TASKS["vehicle"]
    if sha256(task["weights"]) != task["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    dataset = task["dataset"]

    folder = OUTPUT / "truck_called_car"
    folder.mkdir()
    total, found, picked = truck_called_car(dataset, task["weights"], arguments.imgsz, arguments.batch, arguments.count, 363)
    cards = []
    for number, item in enumerate(picked, 1):
        if crop_card(item["image"], item["others"], item["box"], folder, f"{number:03d}"):
            cards.append(f'<div class="card"><img src="{number:03d}.jpg"><p>cao {item["height"]:.0f}px · model gọi car, độ tin cậy {item["confidence"]:.2f}<br><small>{html.escape(item["image"].name)}</small></p></div>')
    write_page(folder / "index.html", "Nhãn truck (MIO) mà model gọi là car",
               f"{len(found)} box trong {total} ảnh MIO có nhãn truck ở val; hiển thị {len(cards)} box ngẫu nhiên. Box đỏ dày là nhãn truck; box vàng là các nhãn khác. "
               "Hãy ghi lại: đây có đúng là truck, hay là pickup/van/xe chở khách mà nhãn đang gộp vào truck?", cards)

    folder = OUTPUT / "overlapping_motorcycle_labels"
    folder.mkdir()
    pairs, picked = overlapping_motorcycles(dataset, arguments.count, 363)
    cards = []
    for number, item in enumerate(picked, 1):
        if crop_card(item["image"], item["others"], item["box"], folder, f"{number:03d}"):
            cards.append(f'<div class="card"><img src="{number:03d}.jpg"><p>IoU hai nhãn xe máy {item["iou"]:.2f}<br><small>{html.escape(item["image"].name)}</small></p></div>')
    write_page(folder / "index.html", "Nhãn xe máy chồng nhau",
               f"{len(pairs)} ảnh val có ít nhất hai nhãn xe máy chồng nhau (IoU ≥ 0,3); hiển thị {len(cards)} ảnh ngẫu nhiên. Box đỏ dày là một nhãn, box vàng là các nhãn xe máy khác. "
               "Hãy ghi lại: hai nhãn là hai xe thật chồng nhau, hay một xe và người ngồi bị gắn hai nhãn?", cards)
    print(f"DONE truck_called_car={len(found)} of {total} images, overlapping_motorcycle_images={len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
