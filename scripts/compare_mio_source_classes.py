import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_vehicle_val_errors import iou_matrix, read_labels

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data/MIO_gt_download_20261103/gt_train.csv"
DATASET = ROOT / "data/Vehicle363_val_from_kaggle_archive_20261102"
AUDIT = ROOT / "runs/project363/vehicle_label_audit_20261103"
OUTPUT = AUDIT / "mio_source_class_comparison.json"
MAPPING = {"articulated_truck": "truck", "single_unit_truck": "truck", "pickup_truck": "truck", "work_van": "truck", "car": "car", "bus": "bus", "motorcycle": "motorcycle"}
# Người dùng chấm 60 box truck bị gọi là car (thứ tự = 001..060 trong qa_app.html).
SIXTEEN_SEATER = {7, 12, 15, 19, 23, 26, 27, 30, 32, 36, 41, 49, 52, 58}
PICKUP = {16, 31}
TRUCK = {33, 35}


def user_label(number):
    return "16_cho" if number in SIXTEEN_SEATER else "ban_tai" if number in PICKUP else "truck" if number in TRUCK else "car"


def load_source():
    rows = defaultdict(list)
    with GT.open(encoding="utf-8") as handle:
        for image, label, x1, y1, x2, y2 in csv.reader(handle):
            rows[image].append((label, float(x1), float(y1), float(x2), float(y2)))
    return rows


def source_classes(path, source, wanted_class):
    """For each val label of the wanted target class, find the best-matching source row and return its original class and IoU."""
    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    classes, boxes = read_labels(DATASET / "labels/val" / (path.stem + ".txt"), width, height)
    image_id = re.search(r"train_(\d+)_jpg", path.name).group(1)
    candidates = source.get(image_id, [])
    out = []
    if not candidates:
        return out
    overlap = iou_matrix(np.array([c[1:] for c in candidates]), boxes)
    for index in range(len(classes)):
        if classes[index] != wanted_class:
            continue
        best = int(overlap[:, index].argmax())
        out.append((candidates[best][0], float(overlap[best, index])))
    return out


def main():
    source = load_source()
    page = (AUDIT / "qa_app.html").read_text(encoding="utf-8")
    sets = json.loads(re.search(r"const SETS = (.*?);\nconst KEY", page, re.S).group(1))
    truck_items = next(entry for entry in sets if entry["id"] == "truck")["items"]
    qa = []
    for number, item in enumerate(truck_items, 1):
        matches = source_classes(DATASET / "images/val" / item["name"], source, 3)
        qa.append({"card": number, "image": item["name"], "user_label": user_label(number), "truck_labels_in_image": len(matches),
                   "source_classes": [m[0] for m in matches], "min_match_iou": round(min((m[1] for m in matches), default=0), 3)})
    unique = [entry for entry in qa if entry["truck_labels_in_image"] == 1]
    by_user = defaultdict(Counter)
    for entry in unique:
        by_user[entry["user_label"]][entry["source_classes"][0]] += 1

    overall = Counter()
    low_iou = 0
    images = sorted(path for path in (DATASET / "images/val").iterdir() if path.name.startswith("mio__"))
    for path in images:
        label_file = DATASET / "labels/val" / (path.stem + ".txt")
        if not any(line.split()[0] == "3" for line in label_file.read_text().splitlines() if line.strip()):
            continue
        for label, iou in source_classes(path, source, 3):
            overall[label] += 1
            low_iou += iou < 0.9
    report = {"gt_csv_rows": sum(len(v) for v in source.values()), "gt_csv_sha256": "89ff69dbfb7a35b57221ef773e0a594b9e4a27b4ace003236f93621e07acac1e",
              "qa_cards_with_single_truck_label": len(unique), "qa_cards_total": len(qa),
              "source_class_by_user_label": {key: dict(value) for key, value in by_user.items()},
              "all_val_mio_truck_labels_by_source_class": dict(overall), "all_val_mio_truck_labels_match_iou_below_0.9": low_iou, "cards": qa}
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cards"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
