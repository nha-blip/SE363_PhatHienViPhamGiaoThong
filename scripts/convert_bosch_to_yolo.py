#!/usr/bin/env python3
"""Convert the Bosch Small Traffic Lights annotations into YOLO detection files.

The source dataset is read-only. The converter writes a separate processed
dataset and refuses to replace it unless --overwrite is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_MAPPING = {
    "Red": (0, "red"),
    "RedLeft": (0, "red"),
    "RedRight": (0, "red"),
    "RedStraight": (0, "red"),
    "RedStraightLeft": (0, "red"),
    "Yellow": (1, "yellow"),
    "Green": (2, "green"),
    "GreenLeft": (2, "green"),
    "GreenRight": (2, "green"),
    "GreenStraight": (2, "green"),
    "GreenStraightRight": (2, "green"),
    "GreenStraightLeft": (2, "green"),
    "off": (3, "off"),
}

EXPECTED_SOURCE_SUMMARY = {
    "total_images": 5093,
    "total_boxes_source": 10756,
}


@dataclass(frozen=True)
class Box:
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class ImageRecord:
    relative_path: Path
    boxes: tuple[Box, ...]


class ConversionError(RuntimeError):
    """Raised when source data cannot be converted without an explicit choice."""


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repository_root / "data" / "dataset_train_rgb",
        help="Bosch source root containing train.yaml and rgb/train/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root / "data" / "bosch_yolo_v1",
        help="Destination for the processed YOLO dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly delete and recreate an existing output directory.",
    )
    return parser.parse_args()


def parse_bosch_yaml(annotation_path: Path) -> list[ImageRecord]:
    """Parse the fixed Bosch YAML schema without requiring a third-party YAML package."""
    lines = annotation_path.read_text(encoding="utf-8").splitlines()
    records: list[ImageRecord] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.startswith("- boxes:"):
            raise ConversionError(f"Unexpected YAML syntax at line {index + 1}: {line!r}")

        box_lines = [line.partition(":")[2]]
        index += 1
        while index < len(lines) and not lines[index].startswith("  path:"):
            box_lines.append(lines[index])
            index += 1

        if index == len(lines):
            raise ConversionError("Annotation record has no path field.")

        path_match = re.fullmatch(r"\s*path:\s*(.+?)\s*", lines[index])
        if not path_match:
            raise ConversionError(f"Unexpected path syntax at line {index + 1}: {lines[index]!r}")
        raw_path = path_match.group(1)
        index += 1

        boxes: list[Box] = []
        box_text = "\n".join(box_lines)
        for match in re.finditer(r"\{([^{}]+)\}", box_text, flags=re.DOTALL):
            fields = {
                key: value.strip().strip("'\"")
                for key, value in re.findall(r"([A-Za-z_]+):\s*([^,}\n]+)", match.group(1))
            }
            required = {"label", "x_min", "y_min", "x_max", "y_max"}
            missing = required.difference(fields)
            if missing:
                raise ConversionError(f"Box for {raw_path} is missing fields: {sorted(missing)}")
            boxes.append(
                Box(
                    label=fields["label"],
                    x_min=float(fields["x_min"]),
                    y_min=float(fields["y_min"]),
                    x_max=float(fields["x_max"]),
                    y_max=float(fields["y_max"]),
                )
            )

        normalized = raw_path.removeprefix("./")
        records.append(ImageRecord(relative_path=Path(normalized), boxes=tuple(boxes)))

    return records


def png_dimensions(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as image_file:
        header = image_file.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ConversionError(f"Unsupported or corrupt PNG image: {image_path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ConversionError(f"Invalid PNG dimensions for {image_path}: {width}x{height}")
    return width, height


def relative_source_path(source_root: Path, relative_path: Path) -> Path:
    candidate = (source_root / relative_path).resolve()
    try:
        return candidate.relative_to(source_root.resolve())
    except ValueError as error:
        raise ConversionError(f"Image path escapes source root: {relative_path}") from error


def box_coordinates(box: Box) -> dict[str, float]:
    return {
        "x_min": box.x_min,
        "y_min": box.y_min,
        "x_max": box.x_max,
        "y_max": box.y_max,
    }


def dropped_box(image_path: Path, box: Box, reason: str) -> dict[str, Any]:
    return {
        "image": image_path.as_posix(),
        "class": box.label,
        "original_bbox": box_coordinates(box),
        "reason": reason,
    }


def validate_records(
    records: list[ImageRecord], source_root: Path
) -> tuple[dict[str, Any], list[tuple[ImageRecord, int, int, tuple[Box, ...]]]]:
    original_classes: Counter[str] = Counter(
        box.label for record in records for box in record.boxes
    )
    mapped_classes: Counter[str] = Counter()
    missing_images: list[str] = []
    dropped_boxes: list[dict[str, Any]] = []
    source_out_of_bounds_boxes: list[dict[str, Any]] = []
    unknown_labels: list[dict[str, str]] = []
    clipped_boxes: list[dict[str, Any]] = []
    prepared: list[tuple[ImageRecord, int, int, tuple[Box, ...]]] = []

    for record in records:
        relative_path = relative_source_path(source_root, record.relative_path)
        image_path = source_root / relative_path
        if not image_path.is_file():
            missing_images.append(relative_path.as_posix())
            dropped_boxes.extend(
                dropped_box(relative_path, box, "missing_source_image") for box in record.boxes
            )
            continue

        width, height = png_dimensions(image_path)
        output_boxes: list[Box] = []
        for box in record.boxes:
            if box.label not in CLASS_MAPPING:
                unknown_labels.append({"image": relative_path.as_posix(), "label": box.label})
                dropped_boxes.append(dropped_box(relative_path, box, "unknown_label"))
                continue
            if box.x_min >= box.x_max or box.y_min >= box.y_max:
                dropped_boxes.append(dropped_box(relative_path, box, "invalid_source_geometry"))
                continue

            outside = box.x_min < 0 or box.y_min < 0 or box.x_max > width or box.y_max > height
            converted = box
            if outside:
                source_out_of_bounds_boxes.append(dropped_box(relative_path, box, "outside_image_boundary"))
                converted = Box(
                    label=box.label,
                    x_min=max(0.0, box.x_min),
                    y_min=max(0.0, box.y_min),
                    x_max=min(float(width), box.x_max),
                    y_max=min(float(height), box.y_max),
                )
                if converted.x_min >= converted.x_max or converted.y_min >= converted.y_max:
                    dropped_boxes.append(dropped_box(relative_path, box, "empty_after_clipping"))
                    continue
                clipped_boxes.append(
                    {
                        "image": relative_path.as_posix(),
                        "class": box.label,
                        "original_bbox": box_coordinates(box),
                        "clipped_bbox": box_coordinates(converted),
                    }
                )

            mapped_classes[CLASS_MAPPING[converted.label][1]] += 1
            output_boxes.append(converted)
        prepared.append((ImageRecord(relative_path, tuple(output_boxes)), width, height, tuple(output_boxes)))

    report: dict[str, Any] = {
        "source_root": str(source_root.resolve()),
        "total_images": len(records),
        "images_with_annotations": sum(bool(record.boxes) for record in records),
        "images_without_annotations": sum(not record.boxes for record in records),
        "total_boxes_source": sum(len(record.boxes) for record in records),
        "valid_boxes": sum(len(boxes) for _, _, _, boxes in prepared),
        "normal_converted_count": sum(len(boxes) for _, _, _, boxes in prepared) - len(clipped_boxes),
        "clipped_boxes_count": len(clipped_boxes),
        "clipped_boxes": clipped_boxes,
        "dropped_boxes_count": len(dropped_boxes),
        "dropped_boxes": dropped_boxes,
        "total_boxes_output": sum(len(boxes) for _, _, _, boxes in prepared),
        "original_class_distribution": dict(sorted(original_classes.items())),
        "mapped_class_distribution_output": dict(sorted(mapped_classes.items())),
        "missing_images": missing_images,
        "out_of_bounds_boxes_count": len(source_out_of_bounds_boxes),
        "out_of_bounds_boxes": source_out_of_bounds_boxes,
        "unknown_labels": unknown_labels,
    }
    return report, prepared


def validate_conversion(report: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in EXPECTED_SOURCE_SUMMARY.items():
        if report[key] != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {report[key]!r}")
    for key in ("missing_images", "unknown_labels"):
        if report[key]:
            mismatches.append(f"{key}: expected no entries, got {len(report[key])}")
    accounted_boxes = (
        report["normal_converted_count"]
        + report["clipped_boxes_count"]
        + report["dropped_boxes_count"]
    )
    if report["total_boxes_source"] != accounted_boxes:
        mismatches.append(
            "source accounting invariant failed: "
            f"{report['total_boxes_source']} != {accounted_boxes}"
        )
    if report["valid_boxes"] != report["total_boxes_output"]:
        mismatches.append("valid_boxes does not match total_boxes_output")
    return mismatches


def write_metadata(output_root: Path, report: dict[str, Any]) -> None:
    (output_root / "data.yaml").write_text(
        "# No train/val/test split is created by this conversion (Project 363 Work 2).\n"
        "path: .\n"
        "train: images\n"
        "names:\n"
        "  0: red\n"
        "  1: yellow\n"
        "  2: green\n"
        "  3: off\n",
        encoding="utf-8",
    )
    (output_root / "class_mapping.yaml").write_text(
        "original_to_v1:\n"
        "  Red: red\n"
        "  RedLeft: red\n"
        "  RedRight: red\n"
        "  RedStraight: red\n"
        "  RedStraightLeft: red\n"
        "  Yellow: yellow\n"
        "  Green: green\n"
        "  GreenLeft: green\n"
        "  GreenRight: green\n"
        "  GreenStraight: green\n"
        "  GreenStraightRight: green\n"
        "  GreenStraightLeft: green\n"
        "  off: off\n"
        "v1_class_ids:\n"
        "  0: red\n"
        "  1: yellow\n"
        "  2: green\n"
        "  3: off\n",
        encoding="utf-8",
    )
    (output_root / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# Bosch YOLO V1\n\n"
        "## Source\n\n"
        "Read-only conversion from `data/dataset_train_rgb/train.yaml` and "
        "`data/dataset_train_rgb/rgb/train/`. Source files are not changed.\n\n"
        "## Conversion\n\n"
        "Images are copied to `images/` with their source-relative paths. YOLO label files "
        "are stored under `labels/` with matching paths. Images with no Bosch boxes receive "
        "an empty `.txt` label file, an Ultralytics-compatible convention. No split is made.\n\n"
        "## Class mapping\n\n"
        "`0=red`, `1=yellow`, `2=green`, `3=off`. Directional Bosch labels are collapsed "
        "only in this processed dataset; the source annotation remains unchanged. Full mapping "
        "is in `class_mapping.yaml`.\n\n"
        "## Statistics\n\n"
        f"- Images: {report['total_images']} ({report['images_with_annotations']} annotated, "
        f"{report['images_without_annotations']} empty)\n"
        f"- Source boxes: {report['total_boxes_source']}\n"
        f"- Normal converted: {report['normal_converted_count']}\n"
        f"- Clipped and converted: {report['clipped_boxes_count']}\n"
        f"- Dropped: {report['dropped_boxes_count']}\n"
        f"- Output boxes: {report['total_boxes_output']}\n"
        f"- Mapped output boxes: {report['mapped_class_distribution_output']}\n\n"
        "## Validation result\n\n"
        f"- Missing images: {len(report['missing_images'])}\n"
        f"- Source out-of-bounds boxes: {report['out_of_bounds_boxes_count']}\n"
        f"- Clipped boxes: {report['clipped_boxes_count']}\n"
        f"- Dropped boxes: {report['dropped_boxes_count']}\n"
        f"- Unknown labels: {len(report['unknown_labels'])}\n\n"
        "## Cleaning policy\n\n"
        "A source box that intersects the image but exceeds an image boundary is clipped to "
        "the valid boundary and retained only if it still has positive width and height. The "
        "report stores its original and clipped coordinates. Invalid source geometry is never "
        "repaired by swapping coordinates; it is dropped with an explicit reason.\n\n"
        "## Known issues\n\n"
        "`data.yaml` deliberately has no validation split. Create a leakage-safe split by "
        "recording sequence in Project 363 Work 2 before training.\n",
        encoding="utf-8",
    )


def write_dataset(
    output_root: Path, source_root: Path, prepared: list[tuple[ImageRecord, int, int, tuple[Box, ...]]]
) -> None:
    for record, width, height, boxes in prepared:
        relative_path = record.relative_path
        source_image = source_root / relative_path
        destination_image = output_root / "images" / relative_path
        destination_label = output_root / "labels" / relative_path.with_suffix(".txt")
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination_image)

        lines = []
        for box in boxes:
            class_id = CLASS_MAPPING[box.label][0]
            x_center = ((box.x_min + box.x_max) / 2.0) / width
            y_center = ((box.y_min + box.y_max) / 2.0) / height
            box_width = (box.x_max - box.x_min) / width
            box_height = (box.y_max - box.y_min) / height
            coordinates = (x_center, y_center, box_width, box_height)
            if not all(0.0 <= coordinate <= 1.0 for coordinate in coordinates):
                raise ConversionError(f"YOLO coordinate out of range for {relative_path}: {coordinates}")
            lines.append(f"{class_id} " + " ".join(f"{coordinate:.8f}" for coordinate in coordinates))
        destination_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_output_dataset(output_root: Path, prepared: list[tuple[ImageRecord, int, int, tuple[Box, ...]]]) -> list[str]:
    """Validate every emitted YOLO label line and the copied image/label pairing."""
    errors: list[str] = []
    output_boxes = 0
    for record, _, _, expected_boxes in prepared:
        image_path = output_root / "images" / record.relative_path
        label_path = output_root / "labels" / record.relative_path.with_suffix(".txt")
        if not image_path.is_file():
            errors.append(f"missing output image: {record.relative_path.as_posix()}")
        if not label_path.is_file():
            errors.append(f"missing output label: {record.relative_path.as_posix()}")
            continue
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(expected_boxes):
            errors.append(f"label count mismatch: {record.relative_path.as_posix()}")
            continue
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                errors.append(f"invalid YOLO field count: {record.relative_path.as_posix()}:{line_number}")
                continue
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                errors.append(f"non-numeric YOLO label: {record.relative_path.as_posix()}:{line_number}")
                continue
            if class_id not in {0, 1, 2, 3}:
                errors.append(f"invalid class ID: {record.relative_path.as_posix()}:{line_number}")
            x_center, y_center, box_width, box_height = coordinates
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                errors.append(f"invalid YOLO center: {record.relative_path.as_posix()}:{line_number}")
            if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
                errors.append(f"invalid YOLO dimensions: {record.relative_path.as_posix()}:{line_number}")
            output_boxes += 1
    if output_boxes != sum(len(boxes) for _, _, _, boxes in prepared):
        errors.append(f"output box count mismatch: {output_boxes}")
    return errors


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    annotation_path = source_root / "train.yaml"
    if not annotation_path.is_file():
        raise ConversionError(f"Missing source annotation: {annotation_path}")

    records = parse_bosch_yaml(annotation_path)
    report, prepared = validate_records(records, source_root)
    mismatches = validate_conversion(report)

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise ConversionError(f"Output exists and is not empty: {output_root}. Use --overwrite to recreate it.")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if mismatches:
        report["validation"] = {"status": "FAIL", "mismatches": mismatches}
        write_metadata(output_root, report)
        raise ConversionError("Validation failed; conversion output was not generated: " + "; ".join(mismatches))

    write_dataset(output_root, source_root, prepared)
    output_errors = validate_output_dataset(output_root, prepared)
    report["validation"] = {
        "status": "PASS" if not output_errors else "FAIL",
        "mismatches": output_errors,
    }
    write_metadata(output_root, report)
    if output_errors:
        raise ConversionError("Output validation failed: " + "; ".join(output_errors))
    print(json.dumps(report["validation"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConversionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
