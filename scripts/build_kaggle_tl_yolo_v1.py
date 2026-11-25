#!/usr/bin/env python3
"""Build the audited Kaggle traffic-light safe subset without touching sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "archive"
TRAIN = SOURCE / "train_dataset"
IMAGES = TRAIN / "train_images"
ANNOTATIONS = TRAIN / "train.json"
OUTPUT = ROOT / "data" / "kaggle_tl_yolo_v1"
EXPECTED = {"red": 3203, "yellow": 123, "green": 2593}
CLASS_ID = {"red": 0, "yellow": 1, "green": 2}
SEED = 363

# Adjacent source IDs are joined when enough independent similarity tests agree.
GROUPING_THRESHOLDS = {
    "maximum_numeric_gap": 3,
    "thumbnail_mae": 0.16,
    "rgb_histogram_l1": 0.20,
    "dhash_hamming": 18,
    "phash_hamming": 16,
    "minimum_votes_gap_1": 2,
    "minimum_votes_gap_2_or_3": 3,
    "global_near_duplicate_phash_hamming": 4,
    "global_near_duplicate_dhash_hamming": 5,
    "global_near_duplicate_thumbnail_mae": 0.07,
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p5": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def feature_record(path: Path) -> dict:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        thumb = np.asarray(image.resize((32, 18), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        gray9 = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.float32)
        dhash = bits_to_int(gray9[:, 1:] > gray9[:, :-1])
        gray32 = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    n = 32
    x = np.arange(n)
    k = np.arange(n)[:, None]
    cosine = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    cosine[0] *= math.sqrt(1 / n)
    cosine[1:] *= math.sqrt(2 / n)
    dct = cosine @ gray32 @ cosine.T
    low = dct[:8, :8]
    phash = bits_to_int(low > np.median(low[1:]))
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(thumb[:, :, channel], bins=16, range=(0, 1))
        hist_parts.append(hist.astype(np.float32) / hist.sum())
    return {
        "width": width,
        "height": height,
        "thumbnail": thumb,
        "histogram": np.concatenate(hist_parts),
        "dhash": dhash,
        "phash": phash,
        "sha256": sha256(path),
    }


def similarity(first: dict, second: dict) -> dict:
    return {
        "thumbnail_mae": float(np.mean(np.abs(first["thumbnail"] - second["thumbnail"]))),
        "rgb_histogram_l1": float(np.mean(np.abs(first["histogram"] - second["histogram"]))),
        "dhash_hamming": (first["dhash"] ^ second["dhash"]).bit_count(),
        "phash_hamming": (first["phash"] ^ second["phash"]).bit_count(),
    }


class UnionFind:
    def __init__(self, items: list[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def valid_outer_box(record: dict, width: int, height: int) -> bool:
    box = record.get("bndbox")
    if not isinstance(box, dict):
        return False
    try:
        xmin, ymin, xmax, ymax = [float(box[key]) for key in ("xmin", "ymin", "xmax", "ymax")]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)) and 0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height


def select_safe(records: list[dict], image_sizes: dict[str, tuple[int, int]]) -> tuple[dict, Counter]:
    selected: dict[str, list[dict]] = defaultdict(list)
    excluded = Counter()
    for index, record in enumerate(records):
        filename = str(record.get("filename", "")).replace("\\", "/")
        size = image_sizes.get(filename)
        if size is None or not valid_outer_box(record, *size):
            excluded["invalid_or_missing_outer_bbox"] += 1
            continue
        if record.get("ignore") != 0:
            excluded["ignore_equals_1"] += 1
            continue
        inbox = record.get("inbox")
        if not isinstance(inbox, list) or not inbox:
            excluded["empty_inbox"] += 1
            continue
        colors = [lamp.get("color") if isinstance(lamp, dict) else None for lamp in inbox]
        if any(color not in CLASS_ID for color in colors):
            excluded["missing_or_unknown_color"] += 1
            continue
        if len(set(colors)) != 1:
            excluded["mixed_colors"] += 1
            continue
        selected[filename].append({"annotation_index": index, "color": colors[0], "record": record})
    return selected, excluded


def make_groups(ids: list[int], features: dict[int, dict]) -> tuple[list[list[int]], list[dict], list[dict], int]:
    union = UnionFind(ids)
    boundaries = []
    adjacency = []
    thresholds = GROUPING_THRESHOLDS
    for left, right in zip(ids, ids[1:]):
        gap = right - left
        if gap > thresholds["maximum_numeric_gap"]:
            boundaries.append({"left_id": left, "right_id": right, "numeric_gap": gap, "decision": "boundary_numeric_gap"})
            continue
        metrics = similarity(features[left], features[right])
        votes = sum((
            metrics["thumbnail_mae"] <= thresholds["thumbnail_mae"],
            metrics["rgb_histogram_l1"] <= thresholds["rgb_histogram_l1"],
            metrics["dhash_hamming"] <= thresholds["dhash_hamming"],
            metrics["phash_hamming"] <= thresholds["phash_hamming"],
        ))
        required = thresholds["minimum_votes_gap_1"] if gap == 1 else thresholds["minimum_votes_gap_2_or_3"]
        joined = votes >= required
        if joined:
            union.union(left, right)
        item = {"left_id": left, "right_id": right, "numeric_gap": gap, **metrics, "votes": votes, "required_votes": required, "decision": "join" if joined else "boundary_visual_discontinuity"}
        adjacency.append(item)
        # Boundaries close to the decision threshold are prioritized for QA.
        if not joined or votes == required:
            boundaries.append(item)

    global_merges = 0
    # Strong near-duplicates can be non-adjacent. Hash gates keep this O(n^2)
    # scan cheap while the thumbnail test rejects visually unrelated collisions.
    for position, left in enumerate(ids):
        first = features[left]
        for right in ids[position + 1:]:
            if union.find(left) == union.find(right):
                continue
            second = features[right]
            if (first["phash"] ^ second["phash"]).bit_count() > thresholds["global_near_duplicate_phash_hamming"]:
                continue
            if (first["dhash"] ^ second["dhash"]).bit_count() > thresholds["global_near_duplicate_dhash_hamming"]:
                continue
            mae = float(np.mean(np.abs(first["thumbnail"] - second["thumbnail"])))
            if mae <= thresholds["global_near_duplicate_thumbnail_mae"]:
                union.union(left, right)
                global_merges += 1

    grouped: dict[int, list[int]] = defaultdict(list)
    for image_id in ids:
        grouped[union.find(image_id)].append(image_id)
    groups = sorted((sorted(values) for values in grouped.values()), key=lambda values: values[0])
    return groups, adjacency, boundaries, global_merges


def group_metrics(groups: list[list[int]], safe_by_id: dict[int, list[dict]]) -> list[dict]:
    result = []
    for index, ids in enumerate(groups, 1):
        classes = Counter(item["color"] for image_id in ids for item in safe_by_id.get(image_id, []))
        result.append({
            "group_id": f"ISG_{index:04d}",
            "image_ids": ids,
            "first_id": min(ids),
            "last_id": max(ids),
            "image_count": len(ids),
            "safe_image_count": sum(image_id in safe_by_id for image_id in ids),
            "safe_bbox_count": sum(classes.values()),
            "red_count": classes["red"],
            "yellow_count": classes["yellow"],
            "green_count": classes["green"],
        })
    return result


def split_score(counts: dict[str, Counter], targets: dict[str, Counter]) -> float:
    score = 0.0
    for split in ("train", "val", "test"):
        for key, weight in (("images", 2.0), ("boxes", 2.0), ("red", 1.0), ("yellow", 4.0), ("green", 1.0)):
            target = max(1.0, targets[split][key])
            error = (counts[split][key] - target) / target
            score += weight * error * error
            if counts[split][key] > target * 1.20:
                score += weight * (counts[split][key] / target - 1.20) * 5
    for split in ("train", "val", "test"):
        for color in ("red", "yellow", "green"):
            if counts[split][color] == 0:
                score += 10000
    return score


def choose_split(groups: list[dict]) -> tuple[dict[str, list[str]], dict[str, Counter], float]:
    active = [group for group in groups if group["safe_image_count"]]
    totals = Counter()
    for group in active:
        totals.update({
            "images": group["safe_image_count"], "boxes": group["safe_bbox_count"],
            "red": group["red_count"], "yellow": group["yellow_count"], "green": group["green_count"],
        })
    ratios = {"train": 0.75, "val": 0.125, "test": 0.125}
    targets = {split: Counter({key: totals[key] * ratio for key in totals}) for split, ratio in ratios.items()}
    best = None
    for trial in range(2500):
        rng = random.Random(SEED + trial)
        order = list(active)
        rng.shuffle(order)
        order.sort(key=lambda group: (group["yellow_count"] > 0, group["safe_bbox_count"] + rng.random()), reverse=True)
        assignment = {split: [] for split in ratios}
        counts = {split: Counter() for split in ratios}
        for group in order:
            additions = Counter({
                "images": group["safe_image_count"], "boxes": group["safe_bbox_count"],
                "red": group["red_count"], "yellow": group["yellow_count"], "green": group["green_count"],
            })
            options = []
            for split in ratios:
                proposal = {name: current.copy() for name, current in counts.items()}
                proposal[split].update(additions)
                options.append((split_score(proposal, targets), rng.random(), split))
            split = min(options)[2]
            assignment[split].append(group["group_id"])
            counts[split].update(additions)
        score = split_score(counts, targets)
        signature = tuple(tuple(sorted(assignment[split])) for split in ("train", "val", "test"))
        candidate = (score, signature, assignment, counts)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2], best[3], best[0]


def annotate_preview(source: Path, destination: Path, entries: list[dict]) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    colors = {"red": "#ff453a", "yellow": "#ffd60a", "green": "#30d158"}
    for item in entries:
        record = item["record"]
        color = item["color"]
        box = [float(record["bndbox"][key]) for key in ("xmin", "ymin", "xmax", "ymax")]
        draw.rectangle(box, outline=colors[color], width=5)
        label = f"{color} {source.name}"
        text_box = draw.textbbox((box[0], max(0, box[1] - 28)), label, font=font)
        draw.rectangle(text_box, fill="#101419")
        draw.text((box[0], max(0, box[1] - 28)), label, fill=colors[color], font=font)
    if image.width > 1600:
        image.thumbnail((1600, 1000), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=90)


def group_sheet(destination: Path, title: str, ids: list[int], source_by_id: dict[int, Path], safe_by_id: dict[int, list[dict]]) -> None:
    # At most 24 deterministic positions, always retaining both ends.
    if len(ids) > 24:
        positions = np.linspace(0, len(ids) - 1, 24).round().astype(int)
        shown = [ids[position] for position in sorted(set(positions))]
    else:
        shown = ids
    cols = 4
    cell_w, cell_h, heading_h = 360, 235, 45
    rows = math.ceil(len(shown) / cols)
    canvas = Image.new("RGB", (cols * cell_w, heading_h + rows * cell_h), "#161a20")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill="white", font=ImageFont.load_default(size=20))
    for index, image_id in enumerate(shown):
        with Image.open(source_by_id[image_id]) as opened:
            thumb = opened.convert("RGB")
        thumb.thumbnail((cell_w - 8, cell_h - 34), Image.Resampling.LANCZOS)
        x, y = (index % cols) * cell_w + 4, heading_h + (index // cols) * cell_h + 30
        canvas.paste(thumb, (x, y))
        count = len(safe_by_id.get(image_id, []))
        draw.text((x, y - 24), f"{image_id:05d}.jpg | safe boxes={count}", fill="#f4c95d", font=ImageFont.load_default(size=15))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=88)


def write_readme(path: Path, summary: dict) -> None:
    split = summary["split_statistics"]
    text = f"""# Kaggle Traffic Light YOLO V1 Safe Subset

This is a local, mechanically processed subset of `data/archive/`, identified by the Project 363 audit as the Kaggle `wjybuqi/traffic-light-detection-dataset` download. Source files were not modified. This dataset has not been merged with Bosch and was not used to train a model.

## Source annotation schema

`train_dataset/train.json` contains outer traffic-light objects with pixel `bndbox`, `ignore`, `occluded`, `truncated`, and nested `inbox` lamp records. Lamp records contain their own box, `color`, numeric-string `shape`, `value`, and quality flags. Color and shape are lamp attributes; no direction labels were inferred.

## Safe subset policy

An outer box is included only when it is valid and in bounds, `ignore == 0`, `inbox` is non-empty, every lamp has an explicit red/yellow/green color, and all nested colors agree. The **outer traffic-light box**, not a lamp box, is converted to YOLO.

- `red` → class 0
- `yellow` → class 1
- `green` → class 2
- class 3 (`off`) is reserved for Bosch compatibility and has no boxes here.

Empty `inbox` was not mapped to off because absence of a nested annotation is not an explicit off label. Mixed-color signals were excluded because V1 has no compound state and lane/arrow semantics are unresolved. `ignore == 1` records were excluded because the source explicitly marks them unusable. The 400 source test images were not used because they have no ground-truth annotations.

The resulting dataset contains {summary['safe_images']:,} images and {summary['safe_boxes']:,} boxes: {summary['class_distribution']['red']:,} red, {summary['class_distribution']['yellow']:,} yellow, {summary['class_distribution']['green']:,} green, and 0 off.

Images were copied byte-for-byte with `shutil.copy2` and retained their numeric filenames. `metadata/source_manifest.json` links every output image and YOLO row to its source path and original JSON annotation index.

## Inferred scene grouping and split

All 2,600 annotated source images were analyzed in numeric-ID order using thumbnail pixel difference, RGB histogram distance, dHash, and pHash. Nearby IDs were joined when a deterministic voting rule indicated similar global appearance. A stricter all-pairs perceptual check merged strong non-adjacent near-duplicates. Groups are named `INFERRED_SCENE_GROUPS` (`ISG_####`). Large visual discontinuities and numeric gaps form candidate boundaries.

The group is the atomic split unit. A deterministic class-aware search targets 75/12.5/12.5 while giving extra weight to rare yellow coverage. It does not break groups to force exact percentages.

| Split | Images | Boxes | Red | Yellow | Green |
|---|---:|---:|---:|---:|---:|
| Train | {split['train']['images']:,} | {split['train']['boxes']:,} | {split['train']['red']:,} | {split['train']['yellow']:,} | {split['train']['green']:,} |
| Val | {split['val']['images']:,} | {split['val']['boxes']:,} | {split['val']['red']:,} | {split['val']['yellow']:,} | {split['val']['green']:,} |
| Test | {split['test']['images']:,} | {split['test']['boxes']:,} | {split['test']['red']:,} | {split['test']['yellow']:,} | {split['test']['green']:,} |

**The inferred scene groups are heuristic and are not authoritative source-video sequence labels.** Human review of `qa/sequence_groups/` is required. The supplied source train/test partition was not reused as evidence of leakage safety. Even after exact and strong perceptual checks, subtle near-duplicates or scene relationships may remain.

## Known limitations

- Only red, yellow, and green are contributed; class 3 remains empty.
- Yellow is rare and the class distribution is imbalanced.
- Direction/type shape codes are preserved only in the source; their meanings are undocumented and were not inferred.
- Groups are visual heuristics without authoritative timestamps or camera-session IDs.
- Geographic, signal-design, and camera-viewpoint domain differ from Bosch and fixed traffic cameras.
- The local source includes no license or redistribution terms.

## License warning

**LICENSE STATUS:  
UNKNOWN — MANUAL REVIEW REQUIRED**

This processed dataset must not be presented as redistributable until the source usage terms are verified. License ambiguity does not prevent this local technical preprocessing, but it remains an unresolved release blocker.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    document = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    records = document["annotations"]
    source_paths = sorted(IMAGES.glob("*.jpg"))
    source_by_id = {int(path.stem): path for path in source_paths}
    filename_by_id = {image_id: f"train_images/{image_id:05d}.jpg" for image_id in source_by_id}

    features = {}
    image_sizes = {}
    for image_id, path in source_by_id.items():
        features[image_id] = feature_record(path)
        image_sizes[filename_by_id[image_id]] = (features[image_id]["width"], features[image_id]["height"])
    safe_by_filename, excluded = select_safe(records, image_sizes)
    safe_by_id = {int(Path(filename).stem): items for filename, items in safe_by_filename.items()}
    actual = Counter(item["color"] for items in safe_by_id.values() for item in items)
    if dict(actual) != EXPECTED:
        raise SystemExit(f"PRECONDITION FAIL: independently calculated {dict(actual)}, expected {EXPECTED}; no conversion performed")

    groups_raw, adjacency, boundaries, global_merges = make_groups(sorted(source_by_id), features)
    groups = group_metrics(groups_raw, safe_by_id)
    assignment, split_counts, selection_score = choose_split(groups)
    split_by_group = {group_id: split for split, group_ids in assignment.items() for group_id in group_ids}
    group_by_id = {image_id: group["group_id"] for group in groups for image_id in group["image_ids"]}
    split_by_id = {image_id: split_by_group[group_by_id[image_id]] for image_id in safe_by_id}

    # Strong all-pairs cross-split check, plus exact SHA-256 validation.
    exact_groups = defaultdict(list)
    for image_id in safe_by_id:
        exact_groups[features[image_id]["sha256"]].append(image_id)
    exact_cross = []
    for digest, ids in exact_groups.items():
        if len({split_by_id[image_id] for image_id in ids}) > 1:
            exact_cross.append({"sha256": digest, "image_ids": sorted(ids), "splits": sorted({split_by_id[image_id] for image_id in ids})})
    near_cross = []
    safe_ids = sorted(safe_by_id)
    for position, left in enumerate(safe_ids):
        for right in safe_ids[position + 1:]:
            if split_by_id[left] == split_by_id[right]:
                continue
            first, second = features[left], features[right]
            ph = (first["phash"] ^ second["phash"]).bit_count()
            if ph > GROUPING_THRESHOLDS["global_near_duplicate_phash_hamming"]:
                continue
            dh = (first["dhash"] ^ second["dhash"]).bit_count()
            if dh > GROUPING_THRESHOLDS["global_near_duplicate_dhash_hamming"]:
                continue
            mae = float(np.mean(np.abs(first["thumbnail"] - second["thumbnail"])))
            if mae <= GROUPING_THRESHOLDS["global_near_duplicate_thumbnail_mae"]:
                near_cross.append({"left_id": left, "right_id": right, "left_split": split_by_id[left], "right_split": split_by_id[right], "phash_hamming": ph, "dhash_hamming": dh, "thumbnail_mae": mae})

    for directory in ("images", "labels", "metadata", "qa", "splits"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    manifest_images = []
    widths, heights, areas, normalized_areas = [], [], [], []
    yolo_errors = []
    for image_id in safe_ids:
        source_path = source_by_id[image_id]
        destination = output / "images" / source_path.name
        shutil.copy2(source_path, destination)
        width, height = image_sizes[filename_by_id[image_id]]
        rows = []
        mappings = []
        all_indices = [index for index, record in enumerate(records) if str(record.get("filename", "")).replace("\\", "/") == filename_by_id[image_id]]
        for row_index, item in enumerate(safe_by_id[image_id]):
            box = item["record"]["bndbox"]
            xmin, ymin, xmax, ymax = [float(box[key]) for key in ("xmin", "ymin", "xmax", "ymax")]
            pixel_width, pixel_height = xmax - xmin, ymax - ymin
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            norm_width, norm_height = pixel_width / width, pixel_height / height
            values = (x_center, y_center, norm_width, norm_height)
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < norm_width <= 1 and 0 < norm_height <= 1):
                yolo_errors.append({"image_id": image_id, "annotation_index": item["annotation_index"], "values": values})
            rows.append(f"{CLASS_ID[item['color']]} {x_center:.10f} {y_center:.10f} {norm_width:.10f} {norm_height:.10f}")
            mappings.append({"yolo_row": row_index + 1, "source_annotation_index": item["annotation_index"], "color": item["color"], "class_id": CLASS_ID[item["color"]]})
            widths.append(pixel_width); heights.append(pixel_height); areas.append(pixel_width * pixel_height); normalized_areas.append(norm_width * norm_height)
        label_path = output / "labels" / f"{image_id:05d}.txt"
        label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        manifest_images.append({
            "output_image": f"images/{image_id:05d}.jpg", "output_label": f"labels/{image_id:05d}.txt",
            "source_filename": relative(source_path), "source_annotation_filename": filename_by_id[image_id],
            "source_numeric_id": image_id, "inferred_group_id": group_by_id[image_id], "split": split_by_id[image_id],
            "all_source_annotation_indices": all_indices,
            "safe_source_annotation_indices": [item["annotation_index"] for item in safe_by_id[image_id]],
            "yolo_rows": mappings,
        })
    (output / "metadata" / "source_manifest.json").write_text(json.dumps({"source": relative(SOURCE), "copy_method": "byte-for-byte file copy with shutil.copy2; original numeric filenames retained", "images": manifest_images}, indent=2), encoding="utf-8")

    for split in ("train", "val", "test"):
        lines = [f"../images/{image_id:05d}.jpg" for image_id in safe_ids if split_by_id[image_id] == split]
        (output / "splits" / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    data_yaml = f"""path: {output.as_posix()}
train: splits/train.txt
val: splits/val.txt
test: splits/test.txt

names:
  0: red
  1: yellow
  2: green
  3: off
"""
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")

    group_sizes = [group["image_count"] for group in groups]
    active_groups = [group for group in groups if group["safe_image_count"]]
    uncertain = sorted(boundaries, key=lambda item: (abs(item.get("votes", -99) - item.get("required_votes", 0)), item["left_id"]))[:10]
    sequence_report = {
        "group_type": "INFERRED_SCENE_GROUPS",
        "authoritative_video_sequences": False,
        "method": "Deterministic union of adjacent numeric IDs using four similarity votes, plus strict non-adjacent perceptual near-duplicate merging",
        "thresholds": GROUPING_THRESHOLDS,
        "number_of_source_images": len(source_paths),
        "number_of_safe_images": len(safe_ids),
        "number_of_inferred_groups": len(groups),
        "number_of_active_groups": len(active_groups),
        "singleton_groups": sum(size == 1 for size in group_sizes),
        "largest_group": max(group_sizes),
        "median_group_size": statistics.median(group_sizes),
        "global_strong_near_duplicate_merges": global_merges,
        "adjacent_comparisons": len(adjacency),
        "uncertain_boundary_candidates": uncertain,
        "groups": [{**group, "split": split_by_group.get(group["group_id"])} for group in groups],
    }
    (output / "sequence_analysis.json").write_text(json.dumps(sequence_report, indent=2), encoding="utf-8")

    # Grouping QA.
    qa_groups = output / "qa" / "sequence_groups"
    largest = sorted(groups, key=lambda group: (-group["image_count"], group["group_id"]))[:10]
    rng = random.Random(SEED)
    random_groups = rng.sample(groups, min(10, len(groups)))
    yellow_groups = [group for group in groups if group["yellow_count"]]
    for category, selected_groups in (("largest", largest), ("random", random_groups), ("yellow", yellow_groups)):
        for number, group in enumerate(selected_groups, 1):
            group_sheet(qa_groups / category / f"{number:03d}_{group['group_id']}.jpg", f"{category} | {group['group_id']} | IDs {group['first_id']:05d}-{group['last_id']:05d}", group["image_ids"], source_by_id, safe_by_id)
    for number, boundary in enumerate(uncertain, 1):
        ids = [image_id for image_id in range(max(min(source_by_id), boundary["left_id"] - 3), min(max(source_by_id), boundary["right_id"] + 3) + 1) if image_id in source_by_id]
        group_sheet(qa_groups / "uncertain_boundaries" / f"{number:02d}_{boundary['left_id']:05d}_{boundary['right_id']:05d}.jpg", f"uncertain boundary {boundary['left_id']:05d}|{boundary['right_id']:05d}", ids, source_by_id, safe_by_id)

    # Annotation QA; deterministic selection by SHA-256 ranking.
    def ranked(ids: list[int], label: str) -> list[int]:
        return sorted(set(ids), key=lambda image_id: hashlib.sha256(f"{SEED}:{label}:{image_id}".encode()).hexdigest())
    red_ids = ranked([image_id for image_id, items in safe_by_id.items() if any(item["color"] == "red" for item in items)], "red")[:30]
    yellow_ids = ranked([image_id for image_id, items in safe_by_id.items() if any(item["color"] == "yellow" for item in items)], "yellow")
    green_ids = ranked([image_id for image_id, items in safe_by_id.items() if any(item["color"] == "green" for item in items)], "green")[:30]
    random_ids = ranked(safe_ids, "random")[:50]
    small_ids = ranked([image_id for image_id, items in safe_by_id.items() if any((float(item["record"]["bndbox"]["xmax"]) - float(item["record"]["bndbox"]["xmin"]) < 8 or float(item["record"]["bndbox"]["ymax"]) - float(item["record"]["bndbox"]["ymin"]) < 8) for item in items)], "small")[:30]
    for category, ids in (("red", red_ids), ("yellow", yellow_ids), ("green", green_ids), ("random", random_ids), ("small_boxes", small_ids)):
        for image_id in ids:
            annotate_preview(source_by_id[image_id], output / "qa" / category / f"{image_id:05d}.jpg", safe_by_id[image_id])

    split_sets = {split: {image_id for image_id in safe_ids if split_by_id[image_id] == split} for split in ("train", "val", "test")}
    group_sets = {split: set(assignment[split]) for split in ("train", "val", "test")}
    split_statistics = {split: {key: int(value) for key, value in split_counts[split].items()} | {"groups": len(assignment[split])} for split in ("train", "val", "test")}
    validation = {
        "expected_safe_boxes": sum(EXPECTED.values()), "actual_safe_boxes": sum(actual.values()), "class_counts_match_expected": dict(actual) == EXPECTED,
        "invalid_yolo_rows": len(yolo_errors), "invalid_yolo_details": yolo_errors,
        "missing_images": sum(not (output / item["output_image"]).is_file() for item in manifest_images),
        "missing_labels": sum(not (output / item["output_label"]).is_file() for item in manifest_images),
        "duplicate_output_image_ids": len(safe_ids) - len(set(safe_ids)),
        "image_overlap": sum(len(split_sets[a] & split_sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))),
        "inferred_group_overlap": sum(len(group_sets[a] & group_sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))),
        "safe_image_coverage": len(set().union(*split_sets.values())),
        "status": "PASS",
    }
    if any((validation["invalid_yolo_rows"], validation["missing_images"], validation["missing_labels"], validation["duplicate_output_image_ids"], validation["image_overlap"], validation["inferred_group_overlap"])) or not validation["class_counts_match_expected"] or exact_cross or near_cross:
        validation["status"] = "FAIL"
    report = {
        "source": relative(SOURCE),
        "selection_policy": "valid in-bounds outer bbox; ignore=0; non-empty inbox; all nested lamps explicitly red/yellow/green; all nested colors agree",
        "excluded_annotation_reasons": {
            "invalid_or_missing_outer_bbox": excluded["invalid_or_missing_outer_bbox"],
            "ignore_equals_1": excluded["ignore_equals_1"],
            "empty_inbox": excluded["empty_inbox"],
            "missing_or_unknown_color": excluded["missing_or_unknown_color"],
            "mixed_colors": excluded["mixed_colors"],
        },
        "source_annotated_images": len(source_paths),
        "source_images_referenced_by_annotations": len({str(record.get('filename', '')).replace('\\', '/') for record in records}),
        "safe_images": len(safe_ids), "excluded_images": len(source_paths) - len(safe_ids), "safe_boxes": sum(actual.values()),
        "class_distribution": {**{color: actual[color] for color in ("red", "yellow", "green")}, "off": 0},
        "bbox_statistics": {"pixel_width": stats(widths), "pixel_height": stats(heights), "pixel_area": stats(areas), "normalized_area": stats(normalized_areas)},
        "inferred_group_statistics": {key: sequence_report[key] for key in ("number_of_inferred_groups", "number_of_active_groups", "singleton_groups", "largest_group", "median_group_size")},
        "split_statistics": split_statistics,
        "split_selection": {"target_ratios": {"train": 0.75, "val": 0.125, "test": 0.125}, "objective_score": selection_score, "trials": 2500, "atomic_unit": "INFERRED_SCENE_GROUP"},
        "cross_split_duplicate_analysis": {"exact_method": "SHA-256", "exact_cross_split_duplicates": len(exact_cross), "exact_details": exact_cross, "near_duplicate_method": "all safe cross-split pairs gated by pHash<=4, dHash<=5, thumbnail MAE<=0.07", "near_duplicate_cross_split_candidates": len(near_cross), "near_duplicate_details": near_cross},
        "qa": {"red": len(red_ids), "yellow": len(yellow_ids), "green": len(green_ids), "random": len(random_ids), "small_boxes": len(small_ids), "largest_groups": len(largest), "random_groups": len(random_groups), "yellow_groups": len(yellow_groups), "uncertain_boundaries": len(uncertain)},
        "license": "UNKNOWN — MANUAL REVIEW REQUIRED",
        "validation": validation,
        "status": "REVIEW_REQUIRED" if validation["status"] == "PASS" else "FAIL",
        "next_step": "HUMAN GROUP REVIEW",
    }
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_readme(output / "README.md", report)
    print(json.dumps({"status": report["status"], "safe_images": len(safe_ids), "safe_boxes": sum(actual.values()), "classes": dict(actual), "groups": report["inferred_group_statistics"], "splits": split_statistics, "duplicates": report["cross_split_duplicate_analysis"], "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
