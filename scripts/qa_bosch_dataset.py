#!/usr/bin/env python3
"""Run structural, statistical, duplicate, and sampling QA for Bosch YOLO V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


SPLITS = ("train", "val", "test")
CLASS_NAMES = ("red", "yellow", "green", "off")
ORIGINAL_TO_V1 = {
    "Red": "red",
    "RedLeft": "red",
    "RedRight": "red",
    "RedStraight": "red",
    "RedStraightLeft": "red",
    "Yellow": "yellow",
    "Green": "green",
    "GreenLeft": "green",
    "GreenRight": "green",
    "GreenStraight": "green",
    "GreenStraightRight": "green",
    "GreenStraightLeft": "green",
    "off": "off",
}
SEED = 363
EXPECTED = {
    "images": 5093,
    "boxes": 10755,
    "classes": {"red": 4164, "yellow": 444, "green": 5421, "off": 726},
}


class QaError(RuntimeError):
    """Raised when required QA inputs cannot be read."""


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repository_root / "data" / "bosch_yolo_v1",
        help="Processed Bosch YOLO V1 root.",
    )
    return parser.parse_args()


def read_png_dimensions(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as image_file:
        header = image_file.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise QaError("not_a_readable_png")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise QaError("invalid_png_dimensions")
    return width, height


def quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def metric_statistics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("count", "min", "max", "mean", "median", "p1", "p5", "p25", "p75", "p95", "p99")}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "p1": quantile(values, 0.01),
        "p5": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
    }


def bbox_stats(boxes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_width": metric_statistics([box["pixel_width"] for box in boxes]),
        "pixel_height": metric_statistics([box["pixel_height"] for box in boxes]),
        "pixel_area": metric_statistics([box["pixel_area"] for box in boxes]),
        "normalized_area": metric_statistics([box["normalized_area"] for box in boxes]),
    }


def image_key(dataset_root: Path, image_path: Path) -> str:
    return image_path.relative_to(dataset_root).as_posix()


def source_image_key(image_path: Path) -> str:
    parts = image_path.parts
    try:
        images_index = parts.index("images")
    except ValueError as error:
        raise QaError(f"image path does not include images directory: {image_path}") from error
    return Path(*parts[images_index + 1 :]).as_posix()


def parse_label(
    label_path: Path, width: int, height: int, tolerance: float = 1e-6
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    boxes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return boxes, [{"reason": "unreadable_label", "detail": str(error)}]
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            errors.append({"reason": "malformed_row", "line": line_number, "value": line})
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, box_width, box_height = (float(value) for value in fields[1:])
        except ValueError:
            errors.append({"reason": "non_numeric_row", "line": line_number, "value": line})
            continue
        if class_id not in range(len(CLASS_NAMES)):
            errors.append({"reason": "unexpected_class_id", "line": line_number, "value": line})
            continue
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
            errors.append({"reason": "invalid_normalized_values", "line": line_number, "value": line})
            continue
        left = x_center - box_width / 2
        right = x_center + box_width / 2
        top = y_center - box_height / 2
        bottom = y_center + box_height / 2
        if left < -tolerance or right > 1 + tolerance or top < -tolerance or bottom > 1 + tolerance:
            errors.append({"reason": "box_exceeds_normalized_boundary", "line": line_number, "value": line})
            continue
        boxes.append(
            {
                "class_id": class_id,
                "class": CLASS_NAMES[class_id],
                "x_center": x_center,
                "y_center": y_center,
                "width": box_width,
                "height": box_height,
                "left": max(0.0, left),
                "right": min(1.0, right),
                "top": max(0.0, top),
                "bottom": min(1.0, bottom),
                "pixel_width": box_width * width,
                "pixel_height": box_height * height,
                "pixel_area": box_width * width * box_height * height,
                "normalized_area": box_width * box_height,
                "image_width": width,
                "image_height": height,
                "line": line_number,
            }
        )
    return boxes, errors


def load_manifests(dataset_root: Path) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    manifests: dict[str, list[Path]] = {}
    errors: list[dict[str, Any]] = []
    for split in SPLITS:
        manifest_path = dataset_root / "splits" / f"{split}.txt"
        if not manifest_path.is_file():
            errors.append({"split": split, "reason": "missing_manifest", "path": str(manifest_path)})
            manifests[split] = []
            continue
        raw_paths = manifest_path.read_text(encoding="utf-8").splitlines()
        manifests[split] = [Path(path) for path in raw_paths if path]
    return manifests, errors


def collect_records(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests, manifest_errors = load_manifests(dataset_root)
    records: list[dict[str, Any]] = []
    structural = {
        "manifest_errors": manifest_errors,
        "missing_images": [],
        "unreadable_images": [],
        "missing_labels": [],
        "label_errors": [],
    }
    for split, image_paths in manifests.items():
        for image_path in image_paths:
            if not image_path.is_file():
                structural["missing_images"].append({"split": split, "image": str(image_path)})
                continue
            try:
                width, height = read_png_dimensions(image_path)
            except (OSError, QaError) as error:
                structural["unreadable_images"].append(
                    {"split": split, "image": str(image_path), "reason": str(error)}
                )
                continue
            relative_image = image_path.relative_to(dataset_root / "images")
            label_path = dataset_root / "labels" / relative_image.with_suffix(".txt")
            if not label_path.is_file():
                structural["missing_labels"].append({"split": split, "image": str(image_path)})
                continue
            boxes, label_errors = parse_label(label_path, width, height)
            for error in label_errors:
                structural["label_errors"].append(
                    {"split": split, "image": image_key(dataset_root, image_path), **error}
                )
            records.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "image": image_key(dataset_root, image_path),
                    "source_image": source_image_key(image_path),
                    "sequence": relative_image.parent.name,
                    "label_path": label_path,
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                }
            )
    return records, structural


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_data = {}
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        boxes = [box for record in split_records for box in record["boxes"]]
        classes = Counter(box["class"] for box in boxes)
        split_data[split] = {
            "images": len(split_records),
            "annotated_images": sum(bool(record["boxes"]) for record in split_records),
            "empty_images": sum(not record["boxes"] for record in split_records),
            "boxes": len(boxes),
            "class_distribution": {class_name: classes[class_name] for class_name in CLASS_NAMES},
            "bbox_statistics": bbox_stats(boxes),
        }
    all_boxes = [box for record in records for box in record["boxes"]]
    classes = Counter(box["class"] for box in all_boxes)
    per_class = {
        class_name: bbox_stats([box for box in all_boxes if box["class"] == class_name])
        for class_name in CLASS_NAMES
    }
    return {
        "images": len(records),
        "annotated_images": sum(bool(record["boxes"]) for record in records),
        "empty_images": sum(not record["boxes"] for record in records),
        "boxes": len(all_boxes),
        "class_distribution": {class_name: classes[class_name] for class_name in CLASS_NAMES},
        "bbox_statistics": {"global": bbox_stats(all_boxes), "per_class": per_class, "per_split": split_data},
        "splits": split_data,
    }


def review_category(boxes: list[dict[str, Any]], predicate: Any) -> dict[str, Any]:
    selected = [box for box in boxes if predicate(box)]
    classes = Counter(box["class"] for box in selected)
    return {
        "count": len(selected),
        "percentage_of_boxes": (len(selected) / len(boxes) * 100) if boxes else 0.0,
        "class_distribution": {class_name: classes[class_name] for class_name in CLASS_NAMES},
    }


def edge_analysis(boxes: list[dict[str, Any]]) -> dict[str, Any]:
    edge_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0, "any_edge": 0}
    edge_classes = {edge: Counter() for edge in edge_counts}
    for box in boxes:
        tolerance_x = 1.0 / box["image_width"]
        tolerance_y = 1.0 / box["image_height"]
        edges = {
            "left": box["left"] <= tolerance_x,
            "right": 1.0 - box["right"] <= tolerance_x,
            "top": box["top"] <= tolerance_y,
            "bottom": 1.0 - box["bottom"] <= tolerance_y,
        }
        for edge, present in edges.items():
            if present:
                edge_counts[edge] += 1
                edge_classes[edge][box["class"]] += 1
        if any(edges.values()):
            edge_counts["any_edge"] += 1
            edge_classes["any_edge"][box["class"]] += 1
    return {
        "definition": "box edge within one source pixel of an image boundary",
        "counts": edge_counts,
        "class_distribution": {
            edge: {class_name: edge_classes[edge][class_name] for class_name in CLASS_NAMES}
            for edge in edge_counts
        },
    }


def clipped_box_analysis(dataset_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    report_path = dataset_root / "conversion_report.json"
    if not report_path.is_file():
        return {"available": False, "reason": "conversion_report.json is missing"}
    conversion = json.loads(report_path.read_text(encoding="utf-8"))
    clipped = conversion.get("clipped_boxes", [])
    record_index = {record["source_image"]: record for record in records}
    errors: list[dict[str, Any]] = []
    matched = 0
    for item in clipped:
        record = record_index.get(item["image"])
        if record is None:
            errors.append({"image": item["image"], "reason": "output_image_not_found"})
            continue
        bbox = item["clipped_bbox"]
        target = (
            (bbox["x_min"] + bbox["x_max"]) / 2 / record["width"],
            (bbox["y_min"] + bbox["y_max"]) / 2 / record["height"],
            (bbox["x_max"] - bbox["x_min"]) / record["width"],
            (bbox["y_max"] - bbox["y_min"]) / record["height"],
        )
        matching_boxes = [
            box
            for box in record["boxes"]
            if box["class"] == ORIGINAL_TO_V1.get(item["class"])
            and max(
                abs(box["x_center"] - target[0]),
                abs(box["y_center"] - target[1]),
                abs(box["width"] - target[2]),
                abs(box["height"] - target[3]),
            ) <= 1e-6
            and box["normalized_area"] > 0
        ]
        if matching_boxes:
            matched += 1
        else:
            errors.append({"image": item["image"], "class": item["class"], "reason": "clipped_box_not_matched"})
    return {
        "available": True,
        "source_clipped_boxes": len(clipped),
        "matched_valid_output_boxes": matched,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }


def empty_image_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"requires_visual_review": True, "per_split": {}, "per_sequence": {}}
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        empty = sum(not record["boxes"] for record in split_records)
        result["per_split"][split] = {
            "empty_images": empty,
            "annotated_images": len(split_records) - empty,
            "empty_image_ratio": empty / len(split_records) if split_records else 0.0,
        }
    for sequence in sorted({record["sequence"] for record in records}):
        sequence_records = [record for record in records if record["sequence"] == sequence]
        empty = sum(not record["boxes"] for record in sequence_records)
        result["per_sequence"][sequence] = {
            "split": sequence_records[0]["split"],
            "empty_images": empty,
            "annotated_images": len(sequence_records) - empty,
            "empty_image_ratio": empty / len(sequence_records),
        }
    return result


def duplicate_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        digest = hashlib.sha256()
        with record["image_path"].open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[digest.hexdigest()].append(record)
    cross_split_groups = []
    pair_count = 0
    for digest, group in hashes.items():
        splits = sorted({record["split"] for record in group})
        if len(splits) > 1:
            cross_split_groups.append(
                {"sha256": digest, "splits": splits, "images": [record["image"] for record in group]}
            )
            pair_count += sum(
                1
                for left, right in itertools.combinations(group, 2)
                if left["split"] != right["split"]
            )
    return {
        "method": "sha256_exact_file_hash",
        "exact_cross_split_duplicate_groups": len(cross_split_groups),
        "exact_cross_split_duplicate_pairs": pair_count,
        "groups": cross_split_groups,
        "near_duplicate_analysis": {
            "performed": False,
            "candidates": 0,
            "reason": "not required for machine pass; temporal near-duplicates within sequences are expected",
        },
    }


def sample_records(records: list[dict[str, Any]], amount: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: record["image"])
    if len(ordered) <= amount:
        return ordered
    return sorted(random.Random(seed).sample(ordered, amount), key=lambda record: record["image"])


def make_preview_manifest(dataset_root: Path, records: list[dict[str, Any]], clipped: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = {}

    def add_group(group: str, selected: list[dict[str, Any]]) -> None:
        items = []
        for record in selected:
            safe_name = record["image"].replace("/", "__")
            items.append(
                {
                    "image": str(record["image_path"]),
                    "label": str(record["label_path"]),
                    "output": f"{group}/{safe_name}",
                }
            )
        groups[group] = items

    for offset, split in enumerate(SPLITS):
        amount = {"train": 30, "val": 20, "test": 20}[split]
        add_group(f"random/{split}", sample_records([r for r in records if r["split"] == split], amount, SEED + offset))
    for class_id, class_name in enumerate(CLASS_NAMES):
        selected = [record for record in records if any(box["class_id"] == class_id for box in record["boxes"])]
        add_group(f"by_class/{class_name}", sample_records(selected, 30, SEED + 100 + class_id))
    small_images = sorted(
        [record for record in records if record["boxes"]],
        key=lambda record: min(box["normalized_area"] for box in record["boxes"]),
    )
    add_group("small_boxes", small_images[:30])
    clipped_sources = {item["image"] for item in clipped}
    add_group("clipped_boxes", [record for record in records if record["source_image"] in clipped_sources])
    for offset, split in enumerate(SPLITS):
        amount = {"train": 20, "val": 10, "test": 10}[split]
        add_group(
            f"empty_images/{split}",
            sample_records([record for record in records if record["split"] == split and not record["boxes"]], amount, SEED + 200 + offset),
        )
    return {"seed": SEED, "groups": groups}


def anomaly_list(records: list[dict[str, Any]], structural: dict[str, Any], clipped: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    anomalies: list[dict[str, Any]] = []
    for error_type in ("missing_images", "unreadable_images", "missing_labels", "label_errors"):
        for error in structural[error_type]:
            anomalies.append({"severity": "CRITICAL", "reason": error_type, **error})
    clipped_sources = {item["image"] for item in clipped}
    for record in records:
        if record["source_image"] in clipped_sources:
            anomalies.append({"image": record["image"], "split": record["split"], "sequence": record["sequence"], "reason": "contains_clipped_box", "severity": "REVIEW"})
        if not record["boxes"]:
            anomalies.append({"image": record["image"], "split": record["split"], "sequence": record["sequence"], "reason": "empty_image_requires_visual_review", "severity": "INFO"})
        if len(record["boxes"]) >= 10:
            anomalies.append({"image": record["image"], "split": record["split"], "sequence": record["sequence"], "reason": "many_boxes", "box_count": len(record["boxes"]), "severity": "REVIEW"})
        for box in record["boxes"]:
            if box["pixel_width"] < 4 or box["pixel_height"] < 4:
                anomalies.append({"image": record["image"], "split": record["split"], "sequence": record["sequence"], "reason": "very_small_box", "class": box["class"], "bbox": {key: box[key] for key in ("x_center", "y_center", "width", "height")}, "severity": "REVIEW"})
            elif box["normalized_area"] > 0.05:
                anomalies.append({"image": record["image"], "split": record["split"], "sequence": record["sequence"], "reason": "very_large_box", "class": box["class"], "bbox": {key: box[key] for key in ("x_center", "y_center", "width", "height")}, "severity": "REVIEW"})
    severity_rank = {"CRITICAL": 0, "REVIEW": 1, "INFO": 2}
    ordered = sorted(anomalies, key=lambda item: (severity_rank[item["severity"]], item.get("reason", ""), item.get("image", "")))
    return ordered[:1000], sum(item["severity"] == "CRITICAL" for item in anomalies), sum(item["severity"] == "REVIEW" for item in anomalies)


def write_human_review_checklist(qa_root: Path, preview_manifest: dict[str, Any]) -> None:
    group_counts = {group: len(items) for group, items in preview_manifest["groups"].items()}
    lines = ["# Bosch YOLO V1 Human Review", "", "Preview groups were selected deterministically (seed 363). Do not alter labels automatically from this review.", ""]
    for class_name in CLASS_NAMES:
        lines.extend([f"## {class_name.title()} ({group_counts[f'by_class/{class_name}']} previews)", "", "- [ ] bbox surrounds the correct traffic light", f"- [ ] {class_name} state is visually correct", ""])
    lines.extend(["## Random samples", "", "- [ ] preview annotations look plausible across train, validation, and test", "", f"## Small boxes ({group_counts['small_boxes']} previews)", "", "- [ ] objects are still recognizable traffic lights", "- [ ] boxes are correctly positioned", "", f"## Clipped boxes ({group_counts['clipped_boxes']} previews)", "", "- [ ] clipped boxes still correspond to partially visible traffic lights", "- [ ] edge clipping did not remove the traffic-light object", "", "## Empty images", "", "- [ ] no obvious usable traffic light annotation is missing", "- [ ] negative examples are appropriate for the intended detector", ""])
    for split in SPLITS:
        lines.append(f"- [ ] `{split}` empty-image sample ({group_counts[f'empty_images/{split}']} previews) reviewed")
    (qa_root / "HUMAN_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    dataset_root = parse_args().dataset_root.resolve()
    qa_root = dataset_root / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    records, structural = collect_records(dataset_root)
    summary = aggregate(records)
    all_boxes = [box for record in records for box in record["boxes"]]
    small = {
        "thresholds": {"very_small": "pixel_width < 4 OR pixel_height < 4", "small": "pixel_width < 8 OR pixel_height < 8", "very_large": "normalized_area > 0.05"},
        "very_small": review_category(all_boxes, lambda box: box["pixel_width"] < 4 or box["pixel_height"] < 4),
        "small": review_category(all_boxes, lambda box: box["pixel_width"] < 8 or box["pixel_height"] < 8),
        "very_large": review_category(all_boxes, lambda box: box["normalized_area"] > 0.05),
    }
    clipped = json.loads((dataset_root / "conversion_report.json").read_text(encoding="utf-8")).get("clipped_boxes", [])
    clipped_analysis = clipped_box_analysis(dataset_root, records)
    duplicates = duplicate_analysis(records)
    preview_manifest = make_preview_manifest(dataset_root, records, clipped)
    (qa_root / "preview_manifest.json").write_text(json.dumps(preview_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_human_review_checklist(qa_root, preview_manifest)
    anomalies, critical_count, review_count = anomaly_list(records, structural, clipped)
    split_report = json.loads((dataset_root / "split_report.json").read_text(encoding="utf-8"))
    split_mismatches = []
    for split in SPLITS:
        actual = summary["splits"][split]
        expected = split_report[split]
        for key in ("images", "annotated_images", "empty_images", "boxes"):
            if actual[key] != expected[key]:
                split_mismatches.append(f"{split}.{key}: expected {expected[key]}, got {actual[key]}")
        if actual["class_distribution"] != expected["class_distribution"]:
            split_mismatches.append(f"{split}.class_distribution mismatch")
    image_sets = {split: {str(record["image_path"]) for record in records if record["split"] == split} for split in SPLITS}
    image_overlap = sum(len(image_sets[left] & image_sets[right]) for left, right in (("train", "val"), ("train", "test"), ("val", "test")))
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sequence_splits[record["sequence"]].add(record["split"])
    sequence_overlap = sum(len(splits) > 1 for splits in sequence_splits.values())
    failures = []
    if summary["images"] != EXPECTED["images"]:
        failures.append(f"images={summary['images']}")
    if summary["boxes"] != EXPECTED["boxes"]:
        failures.append(f"boxes={summary['boxes']}")
    if summary["class_distribution"] != EXPECTED["classes"]:
        failures.append("class_distribution mismatch")
    if any(structural[key] for key in structural):
        failures.append("structural validation errors")
    if split_mismatches:
        failures.append("split statistics mismatch")
    if image_overlap or sequence_overlap:
        failures.append("split leakage")
    if duplicates["exact_cross_split_duplicate_pairs"]:
        failures.append("exact cross-split duplicates")
    if clipped_analysis.get("status") == "FAIL":
        failures.append("clipped-box validation failed")
    status = "FAIL" if failures else "REVIEW_REQUIRED"
    report = {
        "dataset": {key: summary[key] for key in ("images", "annotated_images", "empty_images", "boxes")},
        "structural_validation": {key: {"count": len(value), "items": value} for key, value in structural.items()},
        "split_validation": {"work_2_mismatches": split_mismatches, "image_overlap": image_overlap, "sequence_overlap": sequence_overlap, "splits": summary["splits"]},
        "class_distribution": summary["class_distribution"],
        "bbox_statistics": summary["bbox_statistics"],
        "small_box_analysis": small,
        "edge_box_analysis": edge_analysis(all_boxes),
        "clipped_box_analysis": clipped_analysis,
        "empty_image_analysis": empty_image_analysis(records),
        "duplicate_analysis": duplicates,
        "visual_samples": {
            "manifest": "qa/preview_manifest.json",
            "seed": SEED,
            "groups": {group: len(items) for group, items in preview_manifest["groups"].items()},
            "rendered_preview_count": len(list(qa_root.rglob("*.png"))),
        },
        "anomalies": anomalies,
        "critical_issue_count": critical_count,
        "review_issue_count": review_count,
        "status": status,
        "failures": failures,
    }
    (dataset_root / "qa_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "images": summary["images"], "boxes": summary["boxes"], "failures": failures}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
