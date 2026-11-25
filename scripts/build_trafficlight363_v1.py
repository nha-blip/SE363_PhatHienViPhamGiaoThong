#!/usr/bin/env python3
"""Build the final TrafficLight363-v1 dataset from validated processed sources."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
BOSCH = ROOT / "data" / "bosch_yolo_v1"
KAGGLE = ROOT / "data" / "kaggle_tl_yolo_v1"
OUTPUT = ROOT / "data" / "TrafficLight363-v1"
CLASSES = {0: "red", 1: "yellow", 2: "green", 3: "off"}
EXPECTED = {
    "bosch": {"images": 5093, "boxes": 10755, "classes": {0: 4164, 1: 444, 2: 5421, 3: 726}},
    "kaggle": {"images": 2271, "boxes": 5919, "classes": {0: 3203, 1: 123, 2: 2593, 3: 0}},
}
SEED = 363
POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
KNOWN_KAGGLE_LIMITATION = (
    "Kaggle bounding boxes generally fit traffic lights well. "
    "A small number of visible traffic lights appear to lack annotations. "
    "The omission rate was judged visually to be low and was accepted for TrafficLight363-v1."
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def image_feature(path: Path) -> dict:
    try:
        with Image.open(path) as opened:
            opened.load()
            image = opened.convert("RGB")
            width, height = image.size
            thumb = np.asarray(image.resize((32, 18), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            gray9 = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.float32)
            gray32 = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}
    x = np.arange(32)
    k = np.arange(32)[:, None]
    cosine = np.cos(np.pi * (2 * x + 1) * k / 64)
    cosine[0] *= math.sqrt(1 / 32)
    cosine[1:] *= math.sqrt(2 / 32)
    low = (cosine @ gray32 @ cosine.T)[:8, :8]
    return {
        "width": width,
        "height": height,
        "thumbnail": thumb,
        "dhash": bits_to_int(gray9[:, 1:] > gray9[:, :-1]),
        "phash": bits_to_int(low > np.median(low[1:])),
        "sha256": digest(path),
    }


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[dict]]:
    rows = []
    errors = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return rows, [{"label": str(path), "reason": f"read_error: {exc}"}]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append({"label": str(path), "line": line_number, "reason": "field_count", "row": line})
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append({"label": str(path), "line": line_number, "reason": "parse_error", "row": line})
            continue
        finite = all(math.isfinite(value) for value in (x_center, y_center, width, height))
        values_valid = finite and class_id in CLASSES and 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1
        boundary_valid = values_valid and x_center - width / 2 >= -1e-7 and x_center + width / 2 <= 1 + 1e-7 and y_center - height / 2 >= -1e-7 and y_center + height / 2 <= 1 + 1e-7
        if not values_valid:
            errors.append({"label": str(path), "line": line_number, "reason": "value_or_class_range", "row": line})
        elif not boundary_valid:
            errors.append({"label": str(path), "line": line_number, "reason": "implied_boundary", "row": line})
        rows.append((class_id, x_center, y_center, width, height))
    return rows, errors


def load_bosch() -> list[dict]:
    items = []
    seen = set()
    for split in ("train", "val", "test"):
        manifest = BOSCH / "splits" / f"{split}.txt"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image = Path(line.strip())
            try:
                image_rel = image.relative_to(BOSCH / "images")
            except ValueError:
                raise RuntimeError(f"Bosch manifest path is outside processed root: {image}")
            label = BOSCH / "labels" / image_rel.with_suffix(".txt")
            sequence = image.parent.name
            final_name = f"bosch__{sequence}__{image.name}"
            if final_name in seen:
                raise RuntimeError(f"Bosch destination collision: {final_name}")
            seen.add(final_name)
            items.append({
                "source_dataset": "bosch", "source_image_path": image, "source_label_path": label,
                "source_image": rel(image), "source_label": rel(label), "source_group_id": sequence,
                "source_split": split, "final_split": split, "final_name": final_name,
            })
    return items


def load_kaggle() -> list[dict]:
    manifest = json.loads((KAGGLE / "metadata" / "source_manifest.json").read_text(encoding="utf-8"))
    items = []
    seen = set()
    for entry in manifest["images"]:
        image = KAGGLE / entry["output_image"]
        label = KAGGLE / entry["output_label"]
        final_name = f"kaggle__{image.name}"
        if final_name in seen:
            raise RuntimeError(f"Kaggle destination collision: {final_name}")
        seen.add(final_name)
        items.append({
            "source_dataset": "kaggle", "source_image_path": image, "source_label_path": label,
            "source_image": rel(image), "source_label": rel(label), "source_group_id": entry["inferred_group_id"],
            "source_split": entry["split"], "final_split": entry["split"], "final_name": final_name,
            "raw_source_image": entry["source_filename"], "raw_annotation_indices": entry["safe_source_annotation_indices"],
        })
    return items


def validate_input(items: list[dict], source: str) -> dict:
    label_errors = []
    missing_images = []
    missing_labels = []
    class_counts = Counter()
    boxes = 0
    empty_labels = 0
    groups_by_split = defaultdict(set)
    parsed = {}
    for item in items:
        image = item["source_image_path"]
        label = item["source_label_path"]
        if not image.is_file():
            missing_images.append(str(image))
        if not label.is_file():
            missing_labels.append(str(label))
            continue
        rows, errors = parse_label(label)
        parsed[str(label)] = rows
        label_errors.extend(errors)
        boxes += len(rows)
        class_counts.update(row[0] for row in rows)
        if not rows:
            empty_labels += 1
        groups_by_split[item["source_group_id"]].add(item["source_split"])
    group_overlap = {group: sorted(splits) for group, splits in groups_by_split.items() if len(splits) > 1}
    actual = {"images": len(items), "boxes": boxes, "classes": {class_id: class_counts[class_id] for class_id in CLASSES}}
    expected = EXPECTED[source]
    if actual != expected or missing_images or missing_labels or label_errors or group_overlap:
        raise RuntimeError(json.dumps({
            "source": source, "actual": actual, "expected": expected, "missing_images": missing_images[:20],
            "missing_labels": missing_labels[:20], "label_errors": label_errors[:20], "group_overlap": group_overlap,
        }, indent=2))
    return {"actual": actual, "empty_labels": empty_labels, "group_count": len(groups_by_split), "parsed": parsed}


def strong_near_pairs(left: list[int], right: list[int], features: dict[int, dict], same_collection: bool = False) -> list[dict]:
    results = []
    right_phash = np.asarray([features[index]["phash"] for index in right], dtype=np.uint64)
    chunk = 256
    for start in range(0, len(left), chunk):
        left_chunk = left[start:start + chunk]
        left_phash = np.asarray([features[index]["phash"] for index in left_chunk], dtype=np.uint64)
        xor = np.bitwise_xor(left_phash[:, None], right_phash[None, :])
        distances = POPCOUNT[xor.view(np.uint8).reshape(len(left_chunk), len(right), 8)].sum(axis=2)
        for local_left, right_position in np.argwhere(distances <= 4):
            left_index = left_chunk[int(local_left)]
            right_index = right[int(right_position)]
            if same_collection and left_index >= right_index:
                continue
            first, second = features[left_index], features[right_index]
            dhash_distance = (first["dhash"] ^ second["dhash"]).bit_count()
            if dhash_distance > 5:
                continue
            mae = float(np.mean(np.abs(first["thumbnail"] - second["thumbnail"])))
            if mae <= 0.07:
                results.append({
                    "left_index": left_index, "right_index": right_index,
                    "phash_hamming": int(distances[local_left, right_position]),
                    "dhash_hamming": dhash_distance, "thumbnail_mae": mae,
                })
    return results


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "min": float(array.min()), "p5": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)), "median": float(np.median(array)), "mean": float(array.mean()),
        "p75": float(np.percentile(array, 75)), "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def deterministic_rank(indices: list[int], category: str, items: list[dict]) -> list[int]:
    return sorted(set(indices), key=lambda index: hashlib.sha256(f"{SEED}:{category}:{items[index]['final_name']}".encode()).hexdigest())


def render_preview(item: dict, rows: list[tuple], destination: Path) -> None:
    with Image.open(item["final_image_path"]) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    palette = {0: "#ff453a", 1: "#ffd60a", 2: "#30d158", 3: "#d0d4da"}
    display_name = Path(item["source_image"]).name
    for class_id, x_center, y_center, width, height in rows:
        xmin = (x_center - width / 2) * image.width
        ymin = (y_center - height / 2) * image.height
        xmax = (x_center + width / 2) * image.width
        ymax = (y_center + height / 2) * image.height
        color = palette[class_id]
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=5)
        text = f"{CLASSES[class_id]} | {item['source_dataset']} | {display_name}"
        position = (xmin, max(0, ymin - 28))
        text_box = draw.textbbox(position, text, font=font)
        draw.rectangle(text_box, fill="#101419")
        draw.text(position, text, fill=color, font=font)
    if not rows:
        text = f"empty label | {item['source_dataset']} | {display_name}"
        draw.rectangle((0, 0, min(image.width, 800), 40), fill="#101419")
        draw.text((8, 8), text, fill="white", font=font)
    if image.width > 1600:
        image.thumbnail((1600, 1000), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=90)


def write_readme(path: Path, statistics: dict) -> None:
    splits = statistics["split_distribution"]
    text = f"""# TrafficLight363-v1

## Purpose

Traffic-light state detection for Project 363. This dataset is prepared for future YOLO training; no model was trained during TL-5.

## Classes

- 0 red
- 1 yellow
- 2 green
- 3 off

No direction classes were introduced. Kaggle empty lamp annotations were not interpreted as off.

## Sources

- Bosch Small Traffic Lights processed dataset: {statistics['source_distribution']['bosch']['images']:,} images / {statistics['source_distribution']['bosch']['boxes']:,} boxes.
- Kaggle Traffic Light Safe Subset: {statistics['source_distribution']['kaggle']['images']:,} images / {statistics['source_distribution']['kaggle']['boxes']:,} boxes.

Every final filename is source-prefixed. `metadata/source_manifest.json` preserves processed-source paths, source group, source split, and final split.

## Bosch Processing

The completed Bosch V1 conversion retained 5,093 images and 10,755 validated outer traffic-light boxes. Directional source labels were collapsed to red/green state classes according to its documented V1 mapping; yellow and off were retained. Invalid source geometry was not silently repaired: the completed conversion documented clipping of intersecting edge boxes and one dropped invalid box. Machine QA and human bounding-box QA passed. Recording sequence is preserved as the atomic group.

## Kaggle Processing

The completed TL-4 safe subset retained only valid, non-ignored outer traffic-light boxes with a non-empty lamp list whose explicit red/yellow/green colors all agreed. Empty lamp lists, ignored records, mixed colors, inferred directions, and the unlabeled source test set were excluded. It contains 2,271 images and 5,919 boxes, with no off contribution. `INFERRED_SCENE_GROUP` identifiers are preserved as atomic groups.

{KNOWN_KAGGLE_LIMITATION}

## Split Policy

Bosch recording sequences are atomic. Kaggle inferred scene groups are atomic. Individual frames were never randomly split. The already validated group-safe source splits were retained as the final split because their combination gives appropriate ratios, all four classes in every split, and both sources in validation and test. Source split and final split are both recorded even where they are identical.

Kaggle inferred groups are heuristic and are not authoritative source-video sequence labels.

## Dataset Statistics

| Split | Images | Boxes | Red | Yellow | Green | Off | Bosch images | Kaggle images |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | {splits['train']['images']:,} | {splits['train']['boxes']:,} | {splits['train']['classes']['red']:,} | {splits['train']['classes']['yellow']:,} | {splits['train']['classes']['green']:,} | {splits['train']['classes']['off']:,} | {splits['train']['sources']['bosch']['images']:,} | {splits['train']['sources']['kaggle']['images']:,} |
| Val | {splits['val']['images']:,} | {splits['val']['boxes']:,} | {splits['val']['classes']['red']:,} | {splits['val']['classes']['yellow']:,} | {splits['val']['classes']['green']:,} | {splits['val']['classes']['off']:,} | {splits['val']['sources']['bosch']['images']:,} | {splits['val']['sources']['kaggle']['images']:,} |
| Test | {splits['test']['images']:,} | {splits['test']['boxes']:,} | {splits['test']['classes']['red']:,} | {splits['test']['classes']['yellow']:,} | {splits['test']['classes']['green']:,} | {splits['test']['classes']['off']:,} | {splits['test']['sources']['bosch']['images']:,} | {splits['test']['sources']['kaggle']['images']:,} |
| Total | {statistics['total_images']:,} | {statistics['total_boxes']:,} | {statistics['class_distribution']['red']:,} | {statistics['class_distribution']['yellow']:,} | {statistics['class_distribution']['green']:,} | {statistics['class_distribution']['off']:,} | {statistics['source_distribution']['bosch']['images']:,} | {statistics['source_distribution']['kaggle']['images']:,} |

Detailed size, source, class, split, minority-class, empty-label, and group statistics are in `metadata/dataset_statistics.json`.

## Known Limitations

- The classes are imbalanced; yellow and off are minority states.
- Kaggle contributes no off boxes.
- {KNOWN_KAGGLE_LIMITATION}
- Kaggle inferred groups are heuristic, not authoritative sequences.
- Bosch and Kaggle differ in geography, camera, signal design, resolution, and annotation policy.
- Empty Bosch frames are retained as validated negative examples.
- Strong perceptual checks cannot prove the absence of every subtle near-duplicate or related scene.

## License Status

**Bosch:** no license/redistribution terms were established in the local processed or source documentation inspected for TL-5. Manual verification is required.

**Kaggle:** **UNKNOWN — MANUAL REVIEW REQUIRED**.

Redistribution of the combined dataset must not be assumed permitted until the usage and redistribution terms for every source are verified. No license was invented or transferred from one source to the other.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    bosch_items = load_bosch()
    kaggle_items = load_kaggle()
    bosch_validation = validate_input(bosch_items, "bosch")
    kaggle_validation = validate_input(kaggle_items, "kaggle")
    items = bosch_items + kaggle_items
    if len({item["final_name"] for item in items}) != len(items):
        raise RuntimeError("Final filename collision detected")

    # Decode and hash every source image before creating the final dataset.
    features = {}
    unreadable = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(image_feature, item["source_image_path"]): index for index, item in enumerate(items)}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            value = future.result()
            if "error" in value:
                unreadable.append(value)
            else:
                features[index] = value
    if unreadable or len(features) != len(items):
        raise RuntimeError(json.dumps({"unreadable_images": unreadable[:50]}, indent=2))

    hash_groups = defaultdict(list)
    for index, value in features.items():
        hash_groups[value["sha256"]].append(index)
    cross_source_exact = []
    excluded_indices = set()
    for image_hash, indices in hash_groups.items():
        sources = {items[index]["source_dataset"] for index in indices}
        if len(sources) > 1:
            cross_source_exact.append({"sha256": image_hash, "items": [{"index": index, "source": items[index]["source_dataset"], "image": items[index]["source_image"]} for index in indices]})
            # Deterministic rule: retain the first Bosch item and exclude Kaggle copies.
            keep = min((index for index in indices if items[index]["source_dataset"] == "bosch"), default=min(indices))
            excluded_indices.update(index for index in indices if index != keep and items[index]["source_dataset"] == "kaggle")

    bosch_indices = [index for index, item in enumerate(items) if item["source_dataset"] == "bosch"]
    kaggle_indices = [index for index, item in enumerate(items) if item["source_dataset"] == "kaggle"]
    cross_source_near = strong_near_pairs(bosch_indices, kaggle_indices, features)
    for pair in cross_source_near:
        pair["left_image"] = items[pair["left_index"]]["source_image"]
        pair["right_image"] = items[pair["right_index"]]["source_image"]

    kept_indices = [index for index in range(len(items)) if index not in excluded_indices]
    kept_items = [items[index] for index in kept_indices]
    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    kept_features = {old_to_new[old]: features[old] for old in kept_indices}
    items = kept_items
    features = kept_features

    for split in ("train", "val", "test"):
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metadata").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "qa").mkdir(parents=True, exist_ok=True)

    def copy_item(index: int) -> None:
        item = items[index]
        split = item["final_split"]
        image_destination = OUTPUT / "images" / split / item["final_name"]
        label_destination = OUTPUT / "labels" / split / f"{Path(item['final_name']).stem}.txt"
        shutil.copy2(item["source_image_path"], image_destination)
        shutil.copy2(item["source_label_path"], label_destination)
        item["final_image_path"] = image_destination
        item["final_label_path"] = label_destination
        item["final_image"] = rel(image_destination)
        item["final_label"] = rel(label_destination)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(copy_item, range(len(items))))

    # Parse final labels independently and calculate statistics.
    final_errors = []
    final_rows = {}
    class_counts = Counter()
    source_counts = defaultdict(Counter)
    split_counts = defaultdict(Counter)
    split_sources = defaultdict(lambda: defaultdict(Counter))
    minority_images = defaultdict(lambda: Counter())
    empty_labels = 0
    widths, heights, areas, normalized_areas = [], [], [], []
    small_indices = []
    for index, item in enumerate(items):
        rows, errors = parse_label(item["final_label_path"])
        final_rows[index] = rows
        final_errors.extend(errors)
        if not rows:
            empty_labels += 1
        source = item["source_dataset"]
        split = item["final_split"]
        source_counts[source]["images"] += 1
        source_counts[source]["boxes"] += len(rows)
        split_counts[split]["images"] += 1
        split_counts[split]["boxes"] += len(rows)
        split_sources[split][source]["images"] += 1
        split_sources[split][source]["boxes"] += len(rows)
        image_classes = set()
        feature = features[index]
        for class_id, _, _, width, height in rows:
            class_counts[class_id] += 1
            split_counts[split][f"class_{class_id}"] += 1
            source_counts[source][f"class_{class_id}"] += 1
            image_classes.add(class_id)
            pixel_width, pixel_height = width * feature["width"], height * feature["height"]
            widths.append(pixel_width); heights.append(pixel_height); areas.append(pixel_width * pixel_height); normalized_areas.append(width * height)
            if pixel_width < 8 or pixel_height < 8:
                small_indices.append(index)
        for class_id in image_classes:
            minority_images[split][class_id] += 1

    # Leakage and strong cross-split near-duplicate analysis.
    split_index = {split: [index for index, item in enumerate(items) if item["final_split"] == split] for split in ("train", "val", "test")}
    image_overlap = sum(len({items[index]["final_name"] for index in split_index[a]} & {items[index]["final_name"] for index in split_index[b]}) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    group_splits = defaultdict(set)
    for item in items:
        group_splits[(item["source_dataset"], item["source_group_id"])].add(item["final_split"])
    bosch_group_overlap = sum(len(splits) > 1 for (source, _), splits in group_splits.items() if source == "bosch")
    kaggle_group_overlap = sum(len(splits) > 1 for (source, _), splits in group_splits.items() if source == "kaggle")
    exact_cross_split = []
    final_hash_groups = defaultdict(list)
    for index, feature in features.items():
        final_hash_groups[feature["sha256"]].append(index)
    for image_hash, indices in final_hash_groups.items():
        splits = {items[index]["final_split"] for index in indices}
        if len(splits) > 1:
            exact_cross_split.append({"sha256": image_hash, "images": [items[index]["final_image"] for index in indices], "splits": sorted(splits)})
    cross_split_near = []
    for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
        pairs = strong_near_pairs(split_index[left_split], split_index[right_split], features)
        for pair in pairs:
            pair.update({
                "left_split": left_split, "right_split": right_split,
                "left_image": items[pair["left_index"]]["final_image"],
                "right_image": items[pair["right_index"]]["final_image"],
                "left_group": items[pair["left_index"]]["source_group_id"],
                "right_group": items[pair["right_index"]]["source_group_id"],
            })
        cross_split_near.extend(pairs)

    source_manifest = []
    for item in items:
        entry = {key: item[key] for key in ("final_image", "final_label", "source_dataset", "source_image", "source_label", "source_group_id", "source_split", "final_split")}
        if item["source_dataset"] == "kaggle":
            entry["raw_source_image"] = item["raw_source_image"]
            entry["raw_annotation_indices"] = item["raw_annotation_indices"]
        source_manifest.append(entry)
    (OUTPUT / "metadata" / "source_manifest.json").write_text(json.dumps({
        "naming_policy": "bosch__<recording_sequence>__<filename> or kaggle__<filename>",
        "copy_method": "byte-for-byte copies with shutil.copy2; processed sources unchanged",
        "exact_cross_source_duplicate_rule": "If present, retain Bosch and exclude Kaggle exact copies; no exact matches were expected from prior audit.",
        "images": source_manifest,
    }, indent=2), encoding="utf-8")

    groups = []
    for (source, group_id), splits in sorted(group_splits.items()):
        group_items = [index for index, item in enumerate(items) if item["source_dataset"] == source and item["source_group_id"] == group_id]
        group_classes = Counter(row[0] for index in group_items for row in final_rows[index])
        groups.append({
            "source_dataset": source, "source_group_id": group_id, "final_split": next(iter(splits)),
            "image_count": len(group_items), "box_count": sum(len(final_rows[index]) for index in group_items),
            "class_distribution": {CLASSES[class_id]: group_classes[class_id] for class_id in CLASSES},
            "final_images": [items[index]["final_image"] for index in group_items],
        })
    (OUTPUT / "metadata" / "group_manifest.json").write_text(json.dumps({
        "atomic_units": {"bosch": "recording_sequence", "kaggle": "INFERRED_SCENE_GROUP"},
        "groups": groups,
    }, indent=2), encoding="utf-8")

    split_distribution = {}
    for split in ("train", "val", "test"):
        split_distribution[split] = {
            "images": split_counts[split]["images"], "boxes": split_counts[split]["boxes"],
            "classes": {CLASSES[class_id]: split_counts[split][f"class_{class_id}"] for class_id in CLASSES},
            "sources": {source: {"images": split_sources[split][source]["images"], "boxes": split_sources[split][source]["boxes"]} for source in ("bosch", "kaggle")},
            "minority_classes": {
                "yellow_image_count": minority_images[split][1], "yellow_bbox_count": split_counts[split]["class_1"],
                "off_image_count": minority_images[split][3], "off_bbox_count": split_counts[split]["class_3"],
            },
        }
    statistics = {
        "total_images": len(items), "total_boxes": sum(class_counts.values()),
        "source_distribution": {source: {"images": source_counts[source]["images"], "boxes": source_counts[source]["boxes"], "classes": {CLASSES[class_id]: source_counts[source][f"class_{class_id}"] for class_id in CLASSES}} for source in ("bosch", "kaggle")},
        "class_distribution": {CLASSES[class_id]: class_counts[class_id] for class_id in CLASSES},
        "split_distribution": split_distribution,
        "bbox_size_statistics": {"pixel_width": summarize(widths), "pixel_height": summarize(heights), "pixel_area": summarize(areas), "normalized_area": summarize(normalized_areas)},
        "empty_label_image_count": empty_labels,
        "atomic_group_counts": {"bosch_recording_sequences": len({item["source_group_id"] for item in items if item["source_dataset"] == "bosch"}), "kaggle_inferred_scene_groups": len({item["source_group_id"] for item in items if item["source_dataset"] == "kaggle"}), "total": len(groups)},
    }
    (OUTPUT / "metadata" / "dataset_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    data_yaml = f"""path: {OUTPUT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: red
  1: yellow
  2: green
  3: off
"""
    (OUTPUT / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # Deterministic visual QA.
    class_images = defaultdict(list)
    for index, rows in final_rows.items():
        for class_id in {row[0] for row in rows}:
            class_images[class_id].append(index)
    selections = {
        "random": deterministic_rank(list(range(len(items))), "random", items)[:50],
        "red": deterministic_rank(class_images[0], "red", items)[:30],
        "yellow": deterministic_rank(class_images[1], "yellow", items)[:75],
        "green": deterministic_rank(class_images[2], "green", items)[:30],
        "off": deterministic_rank(class_images[3], "off", items)[:75],
        "bosch": deterministic_rank([index for index, item in enumerate(items) if item["source_dataset"] == "bosch"], "bosch", items)[:30],
        "kaggle": deterministic_rank([index for index, item in enumerate(items) if item["source_dataset"] == "kaggle"], "kaggle", items)[:30],
        "small_boxes": deterministic_rank(small_indices, "small_boxes", items)[:30],
    }
    visual_manifest = {"seed": SEED, "categories": {}}
    for category, indices in selections.items():
        visual_manifest["categories"][category] = []
        for index in indices:
            preview = OUTPUT / "qa" / category / f"{Path(items[index]['final_name']).stem}.jpg"
            render_preview(items[index], final_rows[index], preview)
            visual_manifest["categories"][category].append({"final_image": items[index]["final_image"], "preview": rel(preview)})
    (OUTPUT / "qa" / "visual_qa_manifest.json").write_text(json.dumps(visual_manifest, indent=2), encoding="utf-8")

    missing_images = sum(not item["final_image_path"].is_file() for item in items)
    missing_labels = sum(not item["final_label_path"].is_file() for item in items)
    output_images = list((OUTPUT / "images").rglob("*.*"))
    output_labels = list((OUTPUT / "labels").rglob("*.txt"))
    structure_validation = {
        "expected_images": len(items), "actual_images": len(output_images), "expected_labels": len(items), "actual_labels": len(output_labels),
        "missing_images": missing_images, "missing_labels": missing_labels, "unreadable_images": len(unreadable),
    }
    expected_after_dedup = {
        class_id: EXPECTED["bosch"]["classes"][class_id] + EXPECTED["kaggle"]["classes"][class_id]
        for class_id in CLASSES
    }
    class_validation = {
        "schema": CLASSES, "actual": {class_id: class_counts[class_id] for class_id in CLASSES},
        "expected_before_cross_source_dedup": expected_after_dedup,
        "matches_expected": not excluded_indices and all(class_counts[class_id] == expected_after_dedup[class_id] for class_id in CLASSES),
        "all_four_classes_present": all(class_counts[class_id] > 0 for class_id in CLASSES),
    }
    source_validation = {
        "bosch": {key: value for key, value in bosch_validation.items() if key != "parsed"},
        "kaggle": {key: value for key, value in kaggle_validation.items() if key != "parsed"},
        "cross_source_exact_duplicates_removed": len(excluded_indices),
    }
    leakage = {
        "image_overlap": image_overlap, "bosch_sequence_overlap": bosch_group_overlap,
        "kaggle_group_overlap": kaggle_group_overlap, "exact_cross_split_duplicates": len(exact_cross_split),
        "exact_cross_split_details": exact_cross_split,
        "strong_near_duplicate_method": "pHash Hamming <=4, dHash Hamming <=5, 32x18 RGB thumbnail MAE <=0.07",
        "strong_near_duplicate_cross_split_candidates": len(cross_split_near),
        "strong_near_duplicate_cross_split_details": cross_split_near[:500],
    }
    duplicate_analysis = {
        "exact_method": "SHA-256", "cross_source_exact_duplicates": len(cross_source_exact),
        "cross_source_exact_details": cross_source_exact,
        "cross_source_exact_handling": "Retain Bosch and exclude Kaggle exact copies; no removal when count is zero.",
        "cross_source_strong_near_duplicate_candidates": len(cross_source_near),
        "cross_source_strong_near_duplicate_details": cross_source_near[:500],
    }
    technical_failures = (
        final_errors or missing_images or missing_labels or unreadable or image_overlap or bosch_group_overlap or kaggle_group_overlap
        or exact_cross_split or not class_validation["all_four_classes_present"]
        or len(output_images) != len(items) or len(output_labels) != len(items)
    )
    technical_validation = "FAIL" if technical_failures else "PASS"
    report = {
        "structure_validation": structure_validation,
        "label_validation": {"invalid_labels": len(final_errors), "invalid_boxes": sum(error.get("reason") == "implied_boundary" for error in final_errors), "details": final_errors[:500]},
        "class_validation": class_validation,
        "source_validation": source_validation,
        "duplicate_analysis": duplicate_analysis,
        "leakage_analysis": leakage,
        "split_analysis": {"policy": "retain validated group-safe source splits; source_split equals final_split", "atomic_units": {"bosch": "recording_sequence", "kaggle": "INFERRED_SCENE_GROUP"}, "statistics": split_distribution},
        "minority_class_analysis": {split: split_distribution[split]["minority_classes"] for split in ("train", "val", "test")},
        "visual_qa_manifest": "qa/visual_qa_manifest.json",
        "visual_qa_counts": {category: len(indices) for category, indices in selections.items()},
        "known_limitations": [
            "Yellow and off are minority classes.", "Kaggle contributes no off boxes.", KNOWN_KAGGLE_LIMITATION,
            "Kaggle inferred scene groups are heuristic and not authoritative video sequences.",
            "Source-domain and annotation-policy differences remain.",
            "Source redistribution permissions are not established locally.",
        ],
        "technical_validation": technical_validation,
        "status": "FAIL" if technical_failures else "REVIEW_REQUIRED",
        "next_step": "FINAL HUMAN QA",
    }
    (OUTPUT / "final_qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_readme(OUTPUT / "README.md", statistics)
    print(json.dumps({
        "status": report["status"], "technical_validation": technical_validation,
        "images": statistics["total_images"], "boxes": statistics["total_boxes"],
        "sources": statistics["source_distribution"], "classes": statistics["class_distribution"],
        "splits": statistics["split_distribution"], "duplicates": duplicate_analysis,
        "leakage": {key: value for key, value in leakage.items() if not key.endswith("details")},
        "validation": {**structure_validation, "invalid_labels": len(final_errors)},
    }, indent=2))


if __name__ == "__main__":
    main()
