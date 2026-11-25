import html
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_vehicle_val_errors import iou_matrix, read_labels
from scripts.audit_vehicle_label_candidates import crop_card, write_page
from scripts.compare_mio_source_classes import AUDIT, DATASET, load_source

SAMPLE = 60
SEED = 363
SOURCE_CLASSES = {"pickup_truck": "pickup_truck_sample", "work_van": "work_van_sample"}


def main():
    source = load_source()
    pools = {name: [] for name in SOURCE_CLASSES}
    for path in sorted(path for path in (DATASET / "images/val").iterdir() if path.name.startswith("mio__")):
        label_file = DATASET / "labels/val" / (path.stem + ".txt")
        if not any(line.split()[0] == "3" for line in label_file.read_text().splitlines() if line.strip()):
            continue
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        classes, boxes = read_labels(label_file, width, height)
        candidates = source.get(re.search(r"train_(\d+)_jpg", path.name).group(1), [])
        if not candidates:
            continue
        overlap = iou_matrix(np.array([c[1:] for c in candidates]), boxes)
        for index in range(len(classes)):
            if classes[index] != 3:
                continue
            best = int(overlap[:, index].argmax())
            if overlap[best, index] >= 0.9 and candidates[best][0] in pools:
                pools[candidates[best][0]].append({"image": path, "box": boxes[index].tolist(), "others": [box.tolist() for k, box in enumerate(boxes) if k != index],
                                                   "height": float(boxes[index][3] - boxes[index][1])})
    for source_class, folder_name in SOURCE_CLASSES.items():
        folder = AUDIT / folder_name
        folder.mkdir(exist_ok=False)
        picked = random.Random(SEED).sample(pools[source_class], SAMPLE)
        cards = []
        for number, item in enumerate(picked, 1):
            if crop_card(item["image"], item["others"], item["box"], folder, f"{number:03d}"):
                cards.append(f'<div class="card"><img src="{number:03d}.jpg"><p>cao {item["height"]:.0f}px · lớp gốc MIO {source_class}<br><small>{html.escape(item["image"].name)}</small></p></div>')
        write_page(folder / "index.html", f"Mẫu ngẫu nhiên nhãn {source_class} (MIO, val)",
                   f"{SAMPLE} box ngẫu nhiên trong {len(pools[source_class])} box {source_class} của val MIO, không lọc theo dự đoán của model. Box đỏ dày là box cần xem.", cards)
        print(source_class, len(pools[source_class]), len(cards), flush=True)


if __name__ == "__main__":
    main()
