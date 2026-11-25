"""Read-only MIO-TCD class policy audit for Project 363 Work 4.6."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MIO = ROOT / "data" / "MIO-TCD-Localization" / "MIO-TCD-Localization"
OUT = ROOT / "data" / "vehicle_audit" / "mio_mapping"
AMBIGUOUS = ("pickup_truck", "work_van", "motorized_vehicle")


def percentile(values, ratio):
    values = sorted(values)
    index = (len(values) - 1) * ratio
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] if lower == upper else values[lower] + (values[upper] - values[lower]) * (index - lower)


def image_group(image_id: str) -> str:
    return image_id[:3]


def select(candidates, areas):
    median = percentile(areas, .5)
    candidates = [candidate for candidate in candidates if (candidate[4] - candidate[2]) * (candidate[5] - candidate[3]) >= median]
    groups = defaultdict(list)
    for candidate in candidates:
        groups[image_group(candidate[0])].append(candidate)
    selected, used, cursors = [], set(), {group: 0 for group in groups}
    active = sorted(groups)
    while active and len(selected) < 50:
        next_active = []
        for group in active:
            bucket, cursor = groups[group], cursors[group]
            while cursor < len(bucket) and bucket[cursor][0] in used:
                cursor += 1
            if cursor < len(bucket):
                candidate = bucket[cursor]
                selected.append(candidate)
                used.add(candidate[0])
                cursors[group] = cursor + max(1, len(bucket) // 50)
                next_active.append(group)
                if len(selected) == 50:
                    break
        active = next_active
    for candidate in candidates:
        if len(selected) == 50:
            break
        if candidate[0] not in used:
            selected.append(candidate)
            used.add(candidate[0])
    return selected


def crop_sheet(label, selected):
    columns, cell_w, cell_h, title_h = 5, 260, 190, 30
    canvas = Image.new("RGB", (columns * cell_w, title_h + math.ceil(len(selected) / columns) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"MIO-TCD {label}: 50 padded object crops", fill="black")
    for index, (image_id, _, x1, y1, x2, y2) in enumerate(selected):
        source = MIO / "train" / f"{image_id}.jpg"
        width, height = x2 - x1, y2 - y1
        with Image.open(source) as image:
            pad_x, pad_y = width * .15, height * .15
            crop = ImageOps.exif_transpose(image).convert("RGB").crop((max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)), min(image.width, int(x2 + pad_x)), min(image.height, int(y2 + pad_y))))
            crop.thumbnail((cell_w - 8, cell_h - 35))
        row, col = divmod(index, columns)
        left, top = col * cell_w, title_h + row * cell_h
        canvas.paste(crop, (left + 4, top + 3))
        draw.text((left + 4, top + cell_h - 28), f"{label} {image_id}", fill="black")
        draw.text((left + 4, top + cell_h - 14), f"{width:.0f}x{height:.0f}", fill="black")
    canvas.save(OUT / f"{label}_crops.jpg", quality=92)


def full_frame_sheet(label, selected, image_boxes):
    ids = []
    for image_id, *_ in selected:
        if image_id not in ids:
            ids.append(image_id)
        if len(ids) == 12:
            break
    columns, cell_w, cell_h, title_h = 3, 420, 270, 30
    canvas = Image.new("RGB", (columns * cell_w, title_h + math.ceil(len(ids) / columns) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"MIO-TCD {label}: full-frame bbox QA", fill="black")
    for index, image_id in enumerate(ids):
        with Image.open(MIO / "train" / f"{image_id}.jpg") as image:
            frame = ImageOps.exif_transpose(image).convert("RGB")
        frame_draw = ImageDraw.Draw(frame)
        for source_label, x1, y1, x2, y2 in image_boxes[image_id]:
            if source_label == label:
                frame_draw.rectangle((x1, y1, x2, y2), outline="red", width=5)
                frame_draw.text((x1 + 3, max(0, y1 - 16)), label, fill="red", stroke_width=1, stroke_fill="white")
        frame.thumbnail((cell_w - 8, cell_h - 25))
        row, col = divmod(index, columns)
        left, top = col * cell_w, title_h + row * cell_h
        canvas.paste(frame, (left + 4, top + 3))
        draw.text((left + 4, top + cell_h - 18), f"{image_id}.jpg", fill="black")
    canvas.save(OUT / f"{label}_fullframes.jpg", quality=92)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_counts, source_images, metrics = Counter(), defaultdict(set), defaultdict(lambda: defaultdict(list))
    candidates, image_boxes = defaultdict(list), defaultdict(list)
    with (MIO / "gt_train.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            image_id, label, x1_s, y1_s, x2_s, y2_s = row
            x1, y1, x2, y2 = map(float, (x1_s, y1_s, x2_s, y2_s))
            width, height = x2 - x1, y2 - y1
            source_counts[label] += 1
            source_images[label].add(image_id)
            metrics[label]["width"].append(width)
            metrics[label]["height"].append(height)
            metrics[label]["area"].append(width * height)
            image_boxes[image_id].append((label, x1, y1, x2, y2))
            if label in AMBIGUOUS:
                candidates[label].append((image_id, label, x1, y1, x2, y2))
    selections = {}
    for label in AMBIGUOUS:
        selections[label] = select(candidates[label], metrics[label]["area"])
        crop_sheet(label, selections[label])
        full_frame_sheet(label, selections[label], image_boxes)
    class_data = {}
    for label in sorted(source_counts):
        areas = metrics[label]["area"]
        class_data[label] = {
            "bbox_count": source_counts[label], "image_count": len(source_images[label]),
            "train_bbox_count": source_counts[label], "test_bbox_count": 0,
            "median_width": round(statistics.median(metrics[label]["width"]), 4),
            "median_height": round(statistics.median(metrics[label]["height"]), 4),
            "median_area": round(statistics.median(areas), 4),
        }
    result = {"source_classes": class_data,
              "test_annotation_note": "No local test ground-truth annotation file is present; test bbox counts are zero, not an assertion that test images contain no objects.",
              "selections": {label: [{"image_id": item[0], "bbox_xyxy": [item[2], item[3], item[4], item[5]]} for item in selected] for label, selected in selections.items()}}
    (OUT / "mio_mapping_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("MIO MAPPING QA COMPLETE")


if __name__ == "__main__":
    main()
