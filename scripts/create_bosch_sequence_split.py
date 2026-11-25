#!/usr/bin/env python3
"""Create a deterministic leakage-safe Bosch YOLO split by recording sequence."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
CLASS_NAMES = ("red", "yellow", "green", "off")
TARGET_RATIOS = {"train": 0.75, "val": 0.125, "test": 0.125}
EXPECTED_TOTALS = {
    "images": 5093,
    "boxes": 10755,
    "classes": {"red": 4164, "yellow": 444, "green": 5421, "off": 726},
}


class SplitError(RuntimeError):
    """Raised when input data cannot satisfy the split requirements."""


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


def blank_statistics() -> dict[str, Any]:
    return {
        "images": 0,
        "annotated_images": 0,
        "empty_images": 0,
        "boxes": 0,
        "classes": {class_name: 0 for class_name in CLASS_NAMES},
        "image_paths": [],
    }


def read_label(label_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise SplitError(f"Invalid YOLO label field count: {label_path}:{line_number}")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise SplitError(f"Non-numeric YOLO label: {label_path}:{line_number}") from error
        if class_id not in range(len(CLASS_NAMES)):
            raise SplitError(f"Invalid class ID: {label_path}:{line_number}")
        x_center, y_center, box_width, box_height = coordinates
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise SplitError(f"Invalid YOLO center: {label_path}:{line_number}")
        if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
            raise SplitError(f"Invalid YOLO dimensions: {label_path}:{line_number}")
        counts[CLASS_NAMES[class_id]] += 1
    return counts


def discover_sequence_statistics(dataset_root: Path) -> dict[str, dict[str, Any]]:
    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise SplitError("Processed images/ or labels/ directory is missing.")

    sequences: dict[str, dict[str, Any]] = {}
    for image_path in sorted(image_root.rglob("*.png")):
        relative_path = image_path.relative_to(image_root)
        if len(relative_path.parts) < 2:
            raise SplitError(f"Cannot infer recording sequence from: {relative_path.as_posix()}")
        sequence_name = relative_path.parent.name
        label_path = label_root / relative_path.with_suffix(".txt")
        if not label_path.is_file():
            raise SplitError(f"Missing label file: {label_path}")

        statistics = sequences.setdefault(sequence_name, blank_statistics())
        label_counts = read_label(label_path)
        statistics["images"] += 1
        statistics["image_paths"].append((image_root / relative_path).resolve().as_posix())
        if label_counts:
            statistics["annotated_images"] += 1
        else:
            statistics["empty_images"] += 1
        statistics["boxes"] += sum(label_counts.values())
        for class_name in CLASS_NAMES:
            statistics["classes"][class_name] += label_counts[class_name]
    if not sequences:
        raise SplitError("No PNG images found in processed dataset.")
    return dict(sorted(sequences.items()))


def totals(statistics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = blank_statistics()
    for sequence in statistics.values():
        for key in ("images", "annotated_images", "empty_images", "boxes"):
            result[key] += sequence[key]
        for class_name in CLASS_NAMES:
            result["classes"][class_name] += sequence["classes"][class_name]
    return result


def assert_work_one_totals(sequence_statistics: dict[str, dict[str, Any]]) -> None:
    overall = totals(sequence_statistics)
    mismatches = []
    for key in ("images", "boxes"):
        if overall[key] != EXPECTED_TOTALS[key]:
            mismatches.append(f"{key}: expected {EXPECTED_TOTALS[key]}, got {overall[key]}")
    if overall["classes"] != EXPECTED_TOTALS["classes"]:
        mismatches.append(
            f"classes: expected {EXPECTED_TOTALS['classes']}, got {overall['classes']}"
        )
    if mismatches:
        raise SplitError("Work 1 totals do not match: " + "; ".join(mismatches))


def aggregate_assignment(
    sequence_statistics: dict[str, dict[str, Any]], assignment: dict[str, str]
) -> dict[str, dict[str, Any]]:
    split_statistics = {split: blank_statistics() for split in SPLITS}
    for sequence_name, split in assignment.items():
        target = split_statistics[split]
        source = sequence_statistics[sequence_name]
        target["images"] += source["images"]
        target["annotated_images"] += source["annotated_images"]
        target["empty_images"] += source["empty_images"]
        target["boxes"] += source["boxes"]
        target["image_paths"].extend(source["image_paths"])
        for class_name in CLASS_NAMES:
            target["classes"][class_name] += source["classes"][class_name]
    return split_statistics


def candidate_score(split_statistics: dict[str, dict[str, Any]], overall: dict[str, Any]) -> tuple[float, ...]:
    image_error = sum(
        abs(split_statistics[split]["images"] / overall["images"] - TARGET_RATIOS[split])
        for split in SPLITS
    )
    box_error = sum(
        abs(split_statistics[split]["boxes"] / overall["boxes"] - TARGET_RATIOS[split])
        for split in SPLITS
    )
    class_error = sum(
        abs(
            split_statistics[split]["classes"][class_name] / overall["classes"][class_name]
            - TARGET_RATIOS[split]
        )
        for split in SPLITS
        for class_name in CLASS_NAMES
    )
    missing_class_count = sum(
        split_statistics[split]["classes"][class_name] == 0
        for split in SPLITS
        for class_name in CLASS_NAMES
    )
    return (float(missing_class_count), image_error, box_error, class_error)


def select_assignment(sequence_statistics: dict[str, dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    sequence_names = tuple(sequence_statistics)
    overall = totals(sequence_statistics)
    candidates_evaluated = 0
    best: tuple[tuple[float, ...], tuple[str, ...], dict[str, str], dict[str, dict[str, Any]]] | None = None

    for values in itertools.product(SPLITS, repeat=len(sequence_names)):
        if set(values) != set(SPLITS):
            continue
        candidates_evaluated += 1
        assignment = dict(zip(sequence_names, values, strict=True))
        split_statistics = aggregate_assignment(sequence_statistics, assignment)
        score = candidate_score(split_statistics, overall)
        candidate = (score, values, assignment, split_statistics)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        raise SplitError("No assignment has non-empty train, validation, and test splits.")
    score, _, assignment, split_statistics = best
    selection = {
        "method": "exhaustive_sequence_assignment",
        "candidates_evaluated": candidates_evaluated,
        "target_ratios": TARGET_RATIOS,
        "score": {
            "missing_class_count": int(score[0]),
            "image_ratio_l1_error": score[1],
            "box_ratio_l1_error": score[2],
            "class_ratio_l1_error": score[3],
        },
        "criteria": [
            "minimize missing class coverage across splits",
            "then minimize image-count ratio error against 75/12.5/12.5",
            "then minimize bbox-count ratio error",
            "then minimize per-class ratio error",
            "break exact ties lexicographically by sorted sequence assignment",
        ],
    }
    return assignment, {"selection": selection, "statistics": split_statistics}


def output_sequence_statistics(sequence_statistics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        sequence_name: {
            key: value
            for key, value in statistics.items()
            if key != "image_paths"
        }
        for sequence_name, statistics in sequence_statistics.items()
    }


def write_manifests(dataset_root: Path, split_statistics: dict[str, dict[str, Any]]) -> None:
    split_root = dataset_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        paths = sorted(split_statistics[split]["image_paths"])
        (split_root / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")


def write_data_yaml(dataset_root: Path) -> None:
    (dataset_root / "data.yaml").write_text(
        "# Leakage-safe split by recording sequence; generated by create_bosch_sequence_split.py.\n"
        "path: .\n"
        "train: splits/train.txt\n"
        "val: splits/val.txt\n"
        "test: splits/test.txt\n"
        "names:\n"
        "  0: red\n"
        "  1: yellow\n"
        "  2: green\n"
        "  3: off\n",
        encoding="utf-8",
    )


def validate_split(
    dataset_root: Path,
    sequence_statistics: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    split_statistics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_sets: dict[str, set[str]] = {}
    duplicate_images = 0
    for split in SPLITS:
        paths = (dataset_root / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        manifest_sets[split] = set(paths)
        duplicate_images += len(paths) - len(manifest_sets[split])
    image_overlap = sum(
        len(manifest_sets[left].intersection(manifest_sets[right]))
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    all_manifest_images = set().union(*manifest_sets.values())
    expected_images = {
        image_path
        for statistics in sequence_statistics.values()
        for image_path in statistics["image_paths"]
    }
    missing_images = len(expected_images.difference(all_manifest_images))
    invalid_manifest_paths = sum(
        not Path(image_path).is_file()
        for image_paths in manifest_sets.values()
        for image_path in image_paths
    )
    sequence_split_sets = {
        split: {sequence for sequence, assigned_split in assignment.items() if assigned_split == split}
        for split in SPLITS
    }
    sequence_overlap = sum(
        len(sequence_split_sets[left].intersection(sequence_split_sets[right]))
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    output_boxes = sum(split_statistics[split]["boxes"] for split in SPLITS)
    return {
        "image_overlap": image_overlap,
        "sequence_overlap": sequence_overlap,
        "missing_images": missing_images,
        "invalid_manifest_paths": invalid_manifest_paths,
        "duplicate_images": duplicate_images,
        "total_images": len(all_manifest_images),
        "total_boxes": output_boxes,
        "class_coverage": {
            split: {
                class_name: split_statistics[split]["classes"][class_name] > 0
                for class_name in CLASS_NAMES
            }
            for split in SPLITS
        },
    }


def split_report(
    assignment: dict[str, str],
    selection: dict[str, Any],
    split_statistics: dict[str, dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    total_images = sum(statistics["images"] for statistics in split_statistics.values())
    total_boxes = sum(statistics["boxes"] for statistics in split_statistics.values())
    report: dict[str, Any] = {
        "strategy": "recording_sequence",
        "target_ratios": TARGET_RATIOS,
        "actual_ratios": {
            split: {
                "images": split_statistics[split]["images"] / total_images,
                "boxes": split_statistics[split]["boxes"] / total_boxes,
            }
            for split in SPLITS
        },
        "selection": selection,
        "validation": validation,
    }
    for split in SPLITS:
        statistics = split_statistics[split]
        report[split] = {
            "sequences": sorted(sequence for sequence, assigned in assignment.items() if assigned == split),
            "images": statistics["images"],
            "annotated_images": statistics["annotated_images"],
            "empty_images": statistics["empty_images"],
            "boxes": statistics["boxes"],
            "class_distribution": statistics["classes"],
        }
    return report


def update_readme(dataset_root: Path, report: dict[str, Any]) -> None:
    readme_path = dataset_root / "README.md"
    existing = readme_path.read_text(encoding="utf-8")
    section_header = "## Train / Validation / Test Split\n"
    base = existing.split(section_header, maxsplit=1)[0].rstrip()
    lines = [base, "", "", section_header.rstrip(), ""]
    lines.append(
        "Frames from one recording are temporally related, so this dataset is split by "
        "recording sequence rather than randomly by frame. Each sequence belongs to exactly one split."
    )
    lines.extend(["", "The split was selected deterministically by exhaustive evaluation of all non-empty "
                  "sequence-level train/val/test assignments. It first minimizes missing class coverage, then "
                  "image, bbox, and per-class ratio error against a 75/12.5/12.5 target.", ""])
    for split in SPLITS:
        details = report[split]
        ratios = report["actual_ratios"][split]
        lines.append(f"### {split.title()}")
        lines.append("")
        lines.append(f"- Sequences: {', '.join(details['sequences'])}")
        lines.append(
            f"- Images: {details['images']} ({ratios['images']:.2%}); boxes: "
            f"{details['boxes']} ({ratios['boxes']:.2%})"
        )
        lines.append(
            f"- Annotated/empty images: {details['annotated_images']}/{details['empty_images']}"
        )
        lines.append(f"- Classes: {details['class_distribution']}")
        lines.append("")
    validation = report["validation"]
    lines.extend(
        [
            "### Leakage validation",
            "",
            f"- Image overlap: {validation['image_overlap']}; sequence overlap: {validation['sequence_overlap']}",
            f"- Missing images: {validation['missing_images']}; invalid manifest paths: "
            f"{validation['invalid_manifest_paths']}; duplicate images: {validation['duplicate_images']}",
            f"- Coverage: {validation['total_images']} images and {validation['total_boxes']} boxes.",
            "",
        ]
    )
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    dataset_root = parse_args().dataset_root.resolve()
    sequence_statistics = discover_sequence_statistics(dataset_root)
    assert_work_one_totals(sequence_statistics)
    (dataset_root / "sequence_statistics.json").write_text(
        json.dumps(output_sequence_statistics(sequence_statistics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assignment, selection_result = select_assignment(sequence_statistics)
    split_statistics = selection_result["statistics"]
    write_manifests(dataset_root, split_statistics)
    write_data_yaml(dataset_root)
    validation = validate_split(dataset_root, sequence_statistics, assignment, split_statistics)
    report = split_report(assignment, selection_result["selection"], split_statistics, validation)

    failures = [
        f"{key}={validation[key]}"
        for key in (
            "image_overlap",
            "sequence_overlap",
            "missing_images",
            "invalid_manifest_paths",
            "duplicate_images",
        )
        if validation[key] != 0
    ]
    if validation["total_images"] != EXPECTED_TOTALS["images"]:
        failures.append(f"total_images={validation['total_images']}")
    if validation["total_boxes"] != EXPECTED_TOTALS["boxes"]:
        failures.append(f"total_boxes={validation['total_boxes']}")
    for split in SPLITS:
        for class_name, present in validation["class_coverage"][split].items():
            if not present:
                failures.append(f"missing_class={split}:{class_name}")
    report["validation"]["status"] = "PASS" if not failures else "FAIL"
    report["validation"]["failures"] = failures
    (dataset_root / "split_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    update_readme(dataset_root, report)
    if failures:
        raise SplitError("Split validation failed: " + "; ".join(failures))
    print(json.dumps({"status": "PASS", "selection": report["selection"]["score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SplitError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
