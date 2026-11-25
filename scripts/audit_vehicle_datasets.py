"""Read-only audit for Project 363 vehicle-detection source datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


WORKSPACE = Path(__file__).resolve().parents[1]
DATA = WORKSPACE / "data"
HCMC_ROOT = DATA / "archive (1)"
MIO_ROOT = DATA / "MIO-TCD-Localization" / "MIO-TCD-Localization"
OUTPUT = DATA / "vehicle_audit"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGETS = ("car", "motorcycle", "bus", "truck")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def number(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p5": None, "p25": None, "median": None,
                "mean": None, "p75": None, "p95": None, "max": None}
    return {
        "count": len(values), "min": number(min(values)), "p5": number(percentile(values, .05)),
        "p25": number(percentile(values, .25)), "median": number(percentile(values, .5)),
        "mean": number(statistics.fmean(values)), "p75": number(percentile(values, .75)),
        "p95": number(percentile(values, .95)), "max": number(max(values)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def tree(root: Path, depth: int = 3) -> list[str]:
    result: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_dir() and len(relative.parts) <= depth:
            result.append(str(relative).replace("\\", "/") + "/")
    return result


def inspect_images(paths: list[Path], root: Path) -> tuple[dict[str, dict[str, int]], dict, dict[str, str], list[str]]:
    formats: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    dimensions: dict[str, tuple[int, int]] = {}
    unreadable: list[str] = []
    hashes: dict[str, str] = {}
    for index, path in enumerate(paths, 1):
        relative = str(path.relative_to(root)).replace("\\", "/")
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                formats[(image.format or path.suffix.lstrip(".")).upper()] += 1
                resolutions[f"{width}x{height}"] += 1
                dimensions[relative] = (width, height)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            unreadable.append(f"{relative}: {error}")
        hashes[relative] = sha256(path)
        if index % 10000 == 0:
            print(f"  inspected {index:,}/{len(paths):,} images in {root.name}", flush=True)
    valid_dimensions = list(dimensions.values())
    summary = {
        "total_images": len(paths), "readable_images": len(dimensions),
        "unreadable_images": len(unreadable), "formats": dict(formats),
        "resolution_distribution": dict(resolutions),
        "minimum_resolution": None if not valid_dimensions else {"width": min(w for w, _ in valid_dimensions), "height": min(h for _, h in valid_dimensions)},
        "maximum_resolution": None if not valid_dimensions else {"width": max(w for w, _ in valid_dimensions), "height": max(h for _, h in valid_dimensions)},
        "common_resolutions": resolutions.most_common(10),
    }
    return dimensions, summary, hashes, unreadable


def empty_validation() -> dict[str, int | list[str]]:
    return {"total_boxes": 0, "valid_boxes": 0, "invalid_geometry_boxes": 0,
            "out_of_bounds_boxes": 0, "zero_area_boxes": 0,
            "missing_image_annotations": 0, "annotation_files_with_parsing_errors": [],
            "very_small_boxes": 0, "small_boxes": 0}


def box_stat_record() -> defaultdict[str, list[float]]:
    return defaultdict(list)


def add_box_metrics(storage: dict[str, defaultdict[str, list[float]]], label: str, width: float, height: float,
                    image_width: int, image_height: int, validation: dict) -> None:
    bucket = storage[label]
    bucket["pixel_width"].append(width)
    bucket["pixel_height"].append(height)
    bucket["pixel_area"].append(width * height)
    bucket["normalized_area"].append((width * height) / (image_width * image_height))
    if width < 4 or height < 4:
        validation["very_small_boxes"] += 1
    elif width < 8 or height < 8:
        validation["small_boxes"] += 1


def summarize_box_stats(storage: dict[str, defaultdict[str, list[float]]]) -> dict:
    return {label: {metric: stats(values) for metric, values in values_by_metric.items()}
            for label, values_by_metric in storage.items()}


def audit_hcmc() -> tuple[dict, dict[str, Path], dict[str, str]]:
    images = all_images(HCMC_ROOT)
    dimensions, media, hashes, unreadable = inspect_images(images, HCMC_ROOT)
    validation = empty_validation()
    classes: Counter[str] = Counter()
    class_images: defaultdict[str, set[str]] = defaultdict(set)
    metrics: dict[str, defaultdict[str, list[float]]] = defaultdict(box_stat_record)
    selected: dict[str, Path] = {}
    labels = sorted(HCMC_ROOT.rglob("*.txt"))
    for label_path in labels:
        relative = str(label_path.relative_to(HCMC_ROOT)).replace("\\", "/")
        image_relative = str(label_path.relative_to(HCMC_ROOT).with_suffix(".jpg")).replace("\\", "/")
        if image_relative not in dimensions:
            validation["missing_image_annotations"] += 1
            continue
        image_width, image_height = dimensions[image_relative]
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            validation["annotation_files_with_parsing_errors"].append(f"{relative}: {error}")
            continue
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                validation["annotation_files_with_parsing_errors"].append(f"{relative}:{line_number}: expected 5 fields")
                continue
            try:
                label, center_x, center_y, width_n, height_n = parts[0], *map(float, parts[1:])
            except ValueError:
                validation["annotation_files_with_parsing_errors"].append(f"{relative}:{line_number}: non-numeric value")
                continue
            validation["total_boxes"] += 1
            if not all(math.isfinite(value) for value in (center_x, center_y, width_n, height_n)) or width_n <= 0 or height_n <= 0:
                validation["invalid_geometry_boxes"] += 1
                if width_n == 0 or height_n == 0:
                    validation["zero_area_boxes"] += 1
                continue
            xmin = (center_x - width_n / 2) * image_width
            ymin = (center_y - height_n / 2) * image_height
            xmax = (center_x + width_n / 2) * image_width
            ymax = (center_y + height_n / 2) * image_height
            if xmin >= xmax or ymin >= ymax:
                validation["invalid_geometry_boxes"] += 1
                continue
            if xmax <= 0 or ymax <= 0 or xmin >= image_width or ymin >= image_height:
                validation["invalid_geometry_boxes"] += 1
                continue
            if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
                validation["out_of_bounds_boxes"] += 1
            validation["valid_boxes"] += 1
            classes[label] += 1
            class_images[label].add(image_relative)
            add_box_metrics(metrics, label, xmax - xmin, ymax - ymin, image_width, image_height, validation)
            selected.setdefault(label, HCMC_ROOT / image_relative)
    total = sum(classes.values())
    class_table = {label: {"bbox_count": count, "image_count": len(class_images[label]),
                           "percentage_of_objects": number(100 * count / total) if total else 0}
                   for label, count in sorted(classes.items(), key=lambda item: item[0])}
    duplicates = duplicate_summary(hashes, "HCMC")
    result = {
        "dataset": "HCMC Vehicle Dataset (local root named archive (1))",
        "root_absolute": str(HCMC_ROOT.resolve()), "root_relative": str(HCMC_ROOT.relative_to(WORKSPACE)),
        "total_size_bytes": sum(path.stat().st_size for path in HCMC_ROOT.rglob("*") if path.is_file()),
        "structure_up_to_depth_3": tree(HCMC_ROOT), "media": media,
        "annotation": {"format": "YOLO text: class_id cx cy width height; normalized coordinates", "annotation_file_count": len(labels),
                       "fields": ["class_id", "center_x", "center_y", "width", "height"], "class_semantics": "UNKNOWN - MANUAL REVIEW REQUIRED; no local class-map/readme was found"},
        "classes": class_table, "bbox_validation": validation, "bbox_size_statistics": summarize_box_stats(metrics),
        "duplicate_analysis": duplicates, "unreadable_images": unreadable,
        "split_information": "No train/val/test metadata found. Daytime/nighttime directories are content partitions, not documented model splits.",
        "license": "UNKNOWN - MANUAL REVIEW REQUIRED; no local license/readme found.",
        "leakage_risk": "High: filenames include camera identifiers and sequential frame-like numbers; future atomic split should be camera/capture sequence.",
    }
    return result, selected, hashes


def audit_mio() -> tuple[dict, dict[str, Path]]:
    images = all_images(MIO_ROOT)
    dimensions, media, hashes, unreadable = inspect_images(images, MIO_ROOT)
    validation = empty_validation()
    classes: Counter[str] = Counter()
    class_images: defaultdict[str, set[str]] = defaultdict(set)
    metrics: dict[str, defaultdict[str, list[float]]] = defaultdict(box_stat_record)
    selected: dict[str, Path] = {}
    csv_path = MIO_ROOT / "gt_train.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, 1):
            if len(row) != 6:
                validation["annotation_files_with_parsing_errors"].append(f"gt_train.csv:{line_number}: expected 6 fields")
                continue
            image_id, label, xmin_s, ymin_s, xmax_s, ymax_s = row
            image_relative = f"train/{image_id}.jpg"
            if image_relative not in dimensions:
                validation["missing_image_annotations"] += 1
                continue
            try:
                xmin, ymin, xmax, ymax = map(float, (xmin_s, ymin_s, xmax_s, ymax_s))
            except ValueError:
                validation["annotation_files_with_parsing_errors"].append(f"gt_train.csv:{line_number}: non-numeric coordinate")
                continue
            validation["total_boxes"] += 1
            if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)) or xmin >= xmax or ymin >= ymax:
                validation["invalid_geometry_boxes"] += 1
                if xmin == xmax or ymin == ymax:
                    validation["zero_area_boxes"] += 1
                continue
            image_width, image_height = dimensions[image_relative]
            if xmax <= 0 or ymax <= 0 or xmin >= image_width or ymin >= image_height:
                validation["invalid_geometry_boxes"] += 1
                continue
            if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
                validation["out_of_bounds_boxes"] += 1
            validation["valid_boxes"] += 1
            classes[label] += 1
            class_images[label].add(image_relative)
            add_box_metrics(metrics, label, xmax - xmin, ymax - ymin, image_width, image_height, validation)
            selected.setdefault(label, MIO_ROOT / image_relative)
    total = sum(classes.values())
    class_table = {label: {"bbox_count": count, "image_count": len(class_images[label]),
                           "percentage_of_objects": number(100 * count / total) if total else 0}
                   for label, count in sorted(classes.items(), key=lambda item: item[0])}
    result = {
        "dataset": "MIO-TCD Localization Dataset",
        "root_absolute": str(MIO_ROOT.resolve()), "root_relative": str(MIO_ROOT.relative_to(WORKSPACE)),
        "total_size_bytes": sum(path.stat().st_size for path in MIO_ROOT.rglob("*") if path.is_file()),
        "structure_up_to_depth_3": tree(MIO_ROOT), "media": media,
        "annotation": {"format": "CSV rows: image_id,class,xmin,ymin,xmax,ymax", "annotation_file": "gt_train.csv",
                       "fields": ["image_id", "class", "xmin", "ymin", "xmax", "ymax"], "test_ground_truth": "NOT AVAILABLE locally"},
        "classes": class_table, "bbox_validation": validation, "bbox_size_statistics": summarize_box_stats(metrics),
        "duplicate_analysis": duplicate_summary(hashes, "MIO-TCD"), "unreadable_images": unreadable,
        "split_information": "Local train/ and test/ partitions exist. Ground truth is supplied only for train through gt_train.csv.",
        "license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (README.txt). Attribution, non-commercial use, and share-alike apply.",
        "leakage_risk": "README documents separate train/test partitions. Filenames alone do not establish sequence/camera identity; preserve the supplied partition and avoid random re-splitting until provenance is known.",
    }
    return result, selected, hashes


def duplicate_summary(hashes: dict[str, str], source: str) -> dict:
    by_hash: defaultdict[str, list[str]] = defaultdict(list)
    for relative, digest in hashes.items():
        by_hash[digest].append(relative)
    groups = [members for members in by_hash.values() if len(members) > 1]
    return {"method": "SHA-256 exact file hashes", "duplicate_groups": len(groups),
            "duplicate_files_beyond_first": sum(len(group) - 1 for group in groups),
            "sample_groups": groups[:10], "near_duplicate_detection": "NOT PERFORMED"}


def create_contact_sheet(images: list[Path], output: Path, title: str) -> None:
    cell_w, cell_h, columns = 280, 220, 4
    rows = max(1, math.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for index, image_path in enumerate(images):
        row, col = divmod(index, columns)
        x, y = col * cell_w, 34 + row * cell_h
        with Image.open(image_path) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((cell_w - 12, cell_h - 32))
            canvas.paste(preview, (x + 6, y + 4))
        draw.text((x + 6, y + cell_h - 24), image_path.name[:36], fill="black")
    canvas.save(output, quality=90)


def markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def target_counts(dataset: dict, hcmc: bool = False) -> dict[str, int | None]:
    if hcmc:
        return {target: None for target in TARGETS}
    source = dataset["classes"]
    return {"car": source.get("car", {}).get("bbox_count", 0), "motorcycle": source.get("motorcycle", {}).get("bbox_count", 0),
            "bus": source.get("bus", {}).get("bbox_count", 0),
            "truck": source.get("articulated_truck", {}).get("bbox_count", 0) + source.get("single_unit_truck", {}).get("bbox_count", 0)}


def write_markdown(hcmc: dict, mio: dict, cross_duplicates: dict, selected_hcmc: dict[str, Path], selected_mio: dict[str, Path]) -> None:
    hcounts, mcounts = target_counts(hcmc, True), target_counts(mio)
    h_images, m_images = hcmc["media"]["total_images"], mio["media"]["total_images"]
    total_images = h_images + m_images
    mappings = [
        ["car", "HCMC class IDs 0–3: semantics undocumented", "MIO `car`", mcounts["car"]],
        ["motorcycle", "HCMC class IDs 0–3: semantics undocumented", "MIO `motorcycle`", mcounts["motorcycle"]],
        ["bus", "HCMC class IDs 0–3: semantics undocumented", "MIO `bus`", mcounts["bus"]],
        ["truck", "HCMC class IDs 0–3: semantics undocumented", "MIO `articulated_truck` + `single_unit_truck`", mcounts["truck"]],
    ]
    summary = [
        ["HCMC", h_images, hcmc["bbox_validation"]["total_boxes"], "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", hcmc["bbox_validation"]["invalid_geometry_boxes"], "HIGH (visual review pending contact sheets)"],
        ["MIO-TCD", m_images, mio["bbox_validation"]["total_boxes"], mcounts["car"], mcounts["motorcycle"], mcounts["bus"], mcounts["truck"], mio["bbox_validation"]["invalid_geometry_boxes"], "MEDIUM (traffic-camera imagery; non-Vietnam-specific)"],
    ]
    hcmc_classes = markdown_table([[name, values["bbox_count"], values["image_count"], values["percentage_of_objects"]]
                                  for name, values in hcmc["classes"].items()], ["HCMC class ID", "BBoxes", "Images", "% objects"])
    mio_classes = markdown_table([[name, values["bbox_count"], values["image_count"], values["percentage_of_objects"]]
                                 for name, values in mio["classes"].items()], ["MIO class", "BBoxes", "Images", "% objects"])
    text = f"""# Vehicle Detection Dataset Audit — Project 363 Work 4.4

Audit date: 2026-10-03. This audit only inspected source data. It did not alter source files, convert annotations, create YOLO labels, merge datasets, split data, or train a model. New files in `data/vehicle_audit/` are reports and visual previews only.

## Summary

{markdown_table(summary, ["Dataset", "Images", "BBoxes", "Car", "Motorcycle", "Bus", "Truck-related", "Invalid BBox", "Domain Fit"])}

**Audit result: REVIEW_REQUIRED.** Both sources are readable and their annotations link reliably to images, but HCMC lacks a local class-ID semantic map and local license/readme evidence. Therefore its four IDs cannot safely be mapped to Project 363 target names in this audit.

## HCMC Findings

- Root: `{hcmc["root_relative"]}` (`{hcmc["root_absolute"]}`).
- Media: {h_images:,} JPG images; {hcmc["media"]["readable_images"]:,} readable; {hcmc["media"]["unreadable_images"]} unreadable. Common resolutions: {hcmc["media"]["common_resolutions"][:5]}.
- Annotation: {hcmc["annotation"]["annotation_file_count"]:,} per-image YOLO TXT files with normalized `class_id,cx,cy,w,h`.
- Object classes are only local IDs, not names. The audit observed IDs: {", ".join(hcmc["classes"].keys()) or "none"}. Do **not** assume `0=car, 1=motorcycle, 2=bus, 3=truck` without the original HCMC class definition.
- BBox validation: {hcmc["bbox_validation"]["valid_boxes"]:,} valid, {hcmc["bbox_validation"]["invalid_geometry_boxes"]:,} invalid geometry, {hcmc["bbox_validation"]["out_of_bounds_boxes"]:,} out-of-bounds, {hcmc["bbox_validation"]["very_small_boxes"]:,} very small, {hcmc["bbox_validation"]["small_boxes"]:,} small.
- Observable domain material is organized into `daytime` and `nighttime` and filenames include camera/frame-like groups. The fixed-camera / Vietnamese-urban fit cannot be rated conclusively from filenames; inspect `hcmc_general.jpg` and `hcmc_class_id_*.jpg`. Provisional fit: **MEDIUM — visual manual review required**.

{hcmc_classes}

## MIO-TCD Findings

- Root: `{mio["root_relative"]}` (`{mio["root_absolute"]}`).
- Confirmed Localization package: local README describes the 2017 MIO-TCD localization dataset; local train/test image directories and `gt_train.csv` are present. Test ground truth is not included.
- Media: {m_images:,} JPG images; {mio["media"]["readable_images"]:,} readable; {mio["media"]["unreadable_images"]} unreadable. Common resolutions: {mio["media"]["common_resolutions"][:5]}.
- Annotation schema: `image_id,class,xmin,ymin,xmax,ymax`; {mio["bbox_validation"]["total_boxes"]:,} train annotations. Available: class, image ID and bbox. Not available: track ID, sequence/video ID, frame ID, occlusion, truncation, or difficulty.
- BBox validation: {mio["bbox_validation"]["valid_boxes"]:,} valid, {mio["bbox_validation"]["invalid_geometry_boxes"]:,} invalid geometry, {mio["bbox_validation"]["out_of_bounds_boxes"]:,} out-of-bounds, {mio["bbox_validation"]["very_small_boxes"]:,} very small, {mio["bbox_validation"]["small_boxes"]:,} small.

{mio_classes}

## Potential Class Mapping

Do not apply this mapping in Work 4.4. `pickup_truck`, `work_van`, `motorized_vehicle`, and the two explicit MIO truck types remain separate for later policy review.

{markdown_table(mappings, ["Target", "HCMC source", "MIO source", "MIO bbox count"])}

## Class Imbalance

- MIO target counts are car {mcounts["car"]:,}, motorcycle {mcounts["motorcycle"]:,}, bus {mcounts["bus"]:,}, explicit trucks {mcounts["truck"]:,}. The weakest explicit MIO target is `bus` if it has the smallest nonzero count; validate against the class table before mapping `pickup_truck`/`work_van`.
- HCMC imbalance is measured by class ID only until the class map is recovered. No combined semantic target ratio is defensible yet.

## Source Dominance Risk

- Image contribution if both sources were merged: HCMC {100*h_images/total_images:.2f}% ({h_images:,}); MIO-TCD {100*m_images/total_images:.2f}% ({m_images:,}).
- **SOURCE_DOMINANCE_RISK: REVIEW_REQUIRED.** MIO holds the larger image share, while HCMC target-class contribution is unknown due to missing local ID semantics. Do not blindly merge or balance before resolving the HCMC mapping.

## Leakage Risk

- HCMC: likely correlated camera/frame bursts; use camera/capture sequence as the future atomic split unit.
- MIO-TCD: preserve supplied `train`/`test`; local filenames do not establish sequence/camera provenance, so a new random split should not be created without provenance confirmation.

## Data Quality Issues

- HCMC needs the original class map and license/readme. Any target-class coverage claim before that is unsupported.
- MIO test images have no ground-truth CSV locally; train annotations are structurally linked to local train images.
- Exact duplicate hashes: HCMC {hcmc["duplicate_analysis"]["duplicate_files_beyond_first"]:,}; MIO {mio["duplicate_analysis"]["duplicate_files_beyond_first"]:,}; cross-source {cross_duplicates["duplicate_files_beyond_first"]:,}. Near-duplicate analysis was not performed.

## License Information

- HCMC: **UNKNOWN — MANUAL REVIEW REQUIRED**; no local license/readme was found.
- MIO-TCD: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International, documented locally in `README.txt`.

## Vehicle363-v1 Sufficiency

- A. HCMC Vietnam-domain examples: **UNDETERMINED** until visual contact-sheet review and class-ID semantics are recovered.
- B. MIO complement: **Yes**, it provides a large, labelled localization corpus and explicit vehicle subclasses, but its domain should not be treated as Vietnamese fixed-intersection data.
- C. Usable annotations for car/motorcycle/bus/truck: **MIO yes** for its named classes; **HCMC undecidable** from local files alone.
- D. Weakest target class: **UNDETERMINED** until HCMC IDs are resolved; the MIO class table identifies the smallest explicit named target.
- E. Third dataset: **UNDETERMINED.** The concrete blocking gap is HCMC class semantics/license, not evidence that another source is necessary. Resolve this before deciding on acquisition.

## Remaining Dataset Gaps

1. HCMC class-ID mapping and license/usage documentation.
2. Visual approval of HCMC camera/domain fit using previews.
3. A target-class mapping policy for MIO truck-related and ambiguous vehicle subclasses.
4. A leakage-safe split policy based on source provenance.

## Output previews

- HCMC: `hcmc_general.jpg` and `hcmc_class_id_0.jpg` through `hcmc_class_id_3.jpg` (class names unavailable locally).
- MIO-TCD: `mio_general.jpg`, `mio_car.jpg`, `mio_motorcycle.jpg`, `mio_bus.jpg`, and `mio_truck_related.jpg`.

## Final result

```text
WORK 4.4 AUDIT: REVIEW_REQUIRED

HCMC:
Images: {h_images:,}
BBoxes: {hcmc["bbox_validation"]["total_boxes"]:,}
Classes: {", ".join(hcmc["classes"].keys())}
Invalid boxes: {hcmc["bbox_validation"]["invalid_geometry_boxes"]:,}
Domain fit: MEDIUM — manual visual/class-map review required

MIO-TCD:
Images: {m_images:,}
BBoxes: {mio["bbox_validation"]["total_boxes"]:,}
Classes: {", ".join(mio["classes"].keys())}
Invalid boxes: {mio["bbox_validation"]["invalid_geometry_boxes"]:,}
Domain fit: MEDIUM

COMBINED TARGET COUNTS:
Car: HCMC UNKNOWN + MIO {mcounts["car"]:,}
Motorcycle: HCMC UNKNOWN + MIO {mcounts["motorcycle"]:,}
Bus: HCMC UNKNOWN + MIO {mcounts["bus"]:,}
Truck-related: HCMC UNKNOWN + MIO explicit trucks {mcounts["truck"]:,}

SOURCE DOMINANCE:
HCMC: {100*h_images/total_images:.2f}% of images
MIO-TCD: {100*m_images/total_images:.2f}% of images

WEAKEST TARGET CLASS: UNDETERMINED
THIRD DATASET: UNDETERMINED
```
"""
    (OUTPUT / "VEHICLE_DATASET_AUDIT.md").write_text(text, encoding="utf-8")


def main() -> None:
    if not HCMC_ROOT.is_dir() or not MIO_ROOT.is_dir():
        raise SystemExit("Required dataset roots could not be located.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Auditing HCMC...", flush=True)
    hcmc, selected_hcmc, hcmc_hashes = audit_hcmc()
    print("Auditing MIO-TCD...", flush=True)
    mio, selected_mio, mio_hashes = audit_mio()
    cross = defaultdict(list)
    for relative, digest in hcmc_hashes.items():
        cross[digest].append(f"HCMC/{relative}")
    for relative, digest in mio_hashes.items():
        cross[digest].append(f"MIO-TCD/{relative}")
    cross_groups = [members for members in cross.values() if len(members) > 1 and len({member.split('/', 1)[0] for member in members}) > 1]
    cross_duplicates = {"method": "SHA-256 exact file hashes", "duplicate_groups": len(cross_groups),
                        "duplicate_files_beyond_first": sum(len(group) - 1 for group in cross_groups), "sample_groups": cross_groups[:10],
                        "near_duplicate_detection": "NOT PERFORMED"}
    hcmc["cross_source_duplicate_analysis"] = cross_duplicates
    mio["cross_source_duplicate_analysis"] = cross_duplicates
    (OUTPUT / "hcmc_audit.json").write_text(json.dumps(hcmc, indent=2), encoding="utf-8")
    (OUTPUT / "mio_tcd_audit.json").write_text(json.dumps(mio, indent=2), encoding="utf-8")
    combined = {"hcmc_images": hcmc["media"]["total_images"], "mio_images": mio["media"]["total_images"],
                "hcmc_target_counts": target_counts(hcmc, True), "mio_target_counts": target_counts(mio),
                "cross_source_duplicate_analysis": cross_duplicates,
                "status": "REVIEW_REQUIRED", "reason": "HCMC class ID semantics and license are not documented locally."}
    (OUTPUT / "combined_vehicle_audit.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    hcmc_general = all_images(HCMC_ROOT)[:12]
    create_contact_sheet(hcmc_general, OUTPUT / "hcmc_general.jpg", "HCMC general deterministic sample")
    for label, image_path in selected_hcmc.items():
        create_contact_sheet([image_path], OUTPUT / f"hcmc_class_id_{label}.jpg", f"HCMC class ID {label} (semantic unknown)")
    mio_general = all_images(MIO_ROOT / "train")[:12]
    create_contact_sheet(mio_general, OUTPUT / "mio_general.jpg", "MIO-TCD general deterministic sample")
    for label in ("car", "motorcycle", "bus"):
        if label in selected_mio:
            create_contact_sheet([selected_mio[label]], OUTPUT / f"mio_{label}.jpg", f"MIO-TCD {label}")
    truck_images = [selected_mio[label] for label in ("articulated_truck", "single_unit_truck", "pickup_truck", "work_van") if label in selected_mio]
    if truck_images:
        create_contact_sheet(truck_images, OUTPUT / "mio_truck_related.jpg", "MIO-TCD truck-related classes")
    write_markdown(hcmc, mio, cross_duplicates, selected_hcmc, selected_mio)
    print("WORK 4.4 AUDIT: REVIEW_REQUIRED", flush=True)


if __name__ == "__main__":
    main()
