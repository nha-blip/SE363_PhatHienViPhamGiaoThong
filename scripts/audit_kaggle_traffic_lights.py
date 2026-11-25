#!/usr/bin/env python3
"""Read-only audit of data/archive (Kaggle wjybuqi traffic-light dataset)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "data" / "archive"
TRAIN_ROOT = SOURCE / "train_dataset"
TEST_ROOT = SOURCE / "test_dataset"
ANNOTATION_FILE = TRAIN_ROOT / "train.json"
BOSCH_ROOT = WORKSPACE / "data" / "bosch_yolo_v1"
OUTPUT = WORKSPACE / "data" / "kaggle_traffic_light_audit"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SAMPLE_COUNT = 12
SAMPLE_SEED = 363


def rel(path: Path) -> str:
    return path.relative_to(WORKSPACE).as_posix()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_score(text: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}:{text}".encode("utf-8")).hexdigest()


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("count", "min", "p5", "p25", "median", "mean", "p75", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p5": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def validate_box(box: object, width: int | None, height: int | None) -> tuple[dict, list[float] | None]:
    result = {
        "valid": False,
        "invalid_geometry": False,
        "non_finite": False,
        "zero_area": False,
        "out_of_bounds": False,
        "does_not_intersect_image": False,
    }
    if not isinstance(box, dict) or any(key not in box for key in ("xmin", "ymin", "xmax", "ymax")):
        result["invalid_geometry"] = True
        return result, None
    try:
        coords = [float(box[key]) for key in ("xmin", "ymin", "xmax", "ymax")]
    except (TypeError, ValueError):
        result["invalid_geometry"] = True
        return result, None
    if not all(math.isfinite(value) for value in coords):
        result["invalid_geometry"] = True
        result["non_finite"] = True
        return result, coords
    xmin, ymin, xmax, ymax = coords
    if xmin == xmax or ymin == ymax:
        result["zero_area"] = True
    if xmin >= xmax or ymin >= ymax:
        result["invalid_geometry"] = True
    if width is not None and height is not None:
        result["out_of_bounds"] = xmin < 0 or ymin < 0 or xmax > width or ymax > height
        result["does_not_intersect_image"] = xmax <= 0 or ymax <= 0 or xmin >= width or ymin >= height
    result["valid"] = not any(
        result[key]
        for key in ("invalid_geometry", "out_of_bounds", "does_not_intersect_image")
    ) and width is not None and height is not None
    return result, coords


def counter_table(counter: Counter, image_sets: dict) -> list[dict]:
    return [
        {"label": str(label), "bbox_count": int(count), "image_count": len(image_sets[label])}
        for label, count in sorted(counter.items(), key=lambda item: str(item[0]))
    ]


def dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def bbox_label(annotation: dict) -> str:
    colors = [str(item.get("color")) for item in annotation.get("inbox", [])]
    shapes = [str(item.get("shape")) for item in annotation.get("inbox", [])]
    color_text = "+".join(colors) if colors else "NO_COLOR"
    shape_text = "+".join(shapes) if shapes else "NO_SHAPE"
    return f"ignore={annotation.get('ignore')} color={color_text} shape={shape_text}"


def create_placeholder(path: Path, heading: str, message: str) -> None:
    canvas = Image.new("RGB", (1440, 900), "#20242a")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 90), heading, fill="white", font=ImageFont.load_default(size=42))
    draw.multiline_text((70, 190), message, fill="#f4c95d", spacing=16, font=ImageFont.load_default(size=28))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def create_contact_sheet(path: Path, heading: str, selected: list[tuple[dict, Path]], all_by_file: dict[str, list[dict]]) -> None:
    if not selected:
        create_placeholder(path, heading, "No source annotation satisfies this category.\nNo source image was relabeled or inferred.")
        return
    cols, rows = 3, 4
    cell_w, cell_h, header_h = 480, 330, 50
    canvas = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), "#15181d")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), heading, fill="white", font=ImageFont.load_default(size=24))
    for index, (target, image_path) in enumerate(selected[: cols * rows]):
        col, row = index % cols, index // cols
        x0, y0 = col * cell_w, header_h + row * cell_h
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        outer = target["bndbox"]
        ox1, oy1, ox2, oy2 = [float(outer[k]) for k in ("xmin", "ymin", "xmax", "ymax")]

        full_w, full_h = 235, 250
        thumb = image.copy()
        thumb.thumbnail((full_w, full_h), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (cell_w, cell_h), "#22272e")
        px, py = 5, 48
        panel.paste(thumb, (px, py))
        td = ImageDraw.Draw(panel)
        sx, sy = thumb.width / width, thumb.height / height
        for annotation in all_by_file[target["filename"]]:
            box = annotation.get("bndbox", {})
            try:
                coords = [float(box[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
            except (KeyError, TypeError, ValueError):
                continue
            color = "#ff3cac" if annotation is target else "#b7bec7"
            td.rectangle((px + coords[0] * sx, py + coords[1] * sy, px + coords[2] * sx, py + coords[3] * sy), outline=color, width=2)

        pad_x = max(18.0, (ox2 - ox1) * 1.7)
        pad_y = max(18.0, (oy2 - oy1) * 0.8)
        crop_box = (
            max(0, int(ox1 - pad_x)), max(0, int(oy1 - pad_y)),
            min(width, int(ox2 + pad_x)), min(height, int(oy2 + pad_y)),
        )
        crop = image.crop(crop_box)
        crop.thumbnail((230, 250), Image.Resampling.LANCZOS)
        cx, cy = 245, 48
        panel.paste(crop, (cx, cy))
        csx, csy = crop.width / max(1, crop_box[2] - crop_box[0]), crop.height / max(1, crop_box[3] - crop_box[1])
        td.rectangle(
            (
                cx + (ox1 - crop_box[0]) * csx,
                cy + (oy1 - crop_box[1]) * csy,
                cx + (ox2 - crop_box[0]) * csx,
                cy + (oy2 - crop_box[1]) * csy,
            ),
            outline="#ff3cac", width=3,
        )
        for lamp in target.get("inbox", []):
            box = lamp.get("bndbox", {})
            try:
                coords = [float(box[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
            except (KeyError, TypeError, ValueError):
                continue
            color = {"red": "#ff3b30", "yellow": "#ffd60a", "green": "#32d74b"}.get(lamp.get("color"), "white")
            td.rectangle(
                (
                    cx + (coords[0] - crop_box[0]) * csx,
                    cy + (coords[1] - crop_box[1]) * csy,
                    cx + (coords[2] - crop_box[0]) * csx,
                    cy + (coords[3] - crop_box[1]) * csy,
                ),
                outline=color, width=2,
            )
        td.text((6, 5), Path(target["filename"]).name, fill="white", font=ImageFont.load_default(size=16))
        td.text((6, 25), bbox_label(target), fill="#f4c95d", font=ImageFont.load_default(size=13))
        canvas.paste(panel, (x0, y0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_files = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    image_files = sorted(path for path in source_files if path.suffix.lower() in IMAGE_EXTENSIONS)
    train_images = sorted((TRAIN_ROOT / "train_images").glob("*"))
    train_images = [path for path in train_images if path.suffix.lower() in IMAGE_EXTENSIONS]
    test_images = sorted((TEST_ROOT / "test_images").glob("*"))
    test_images = [path for path in test_images if path.suffix.lower() in IMAGE_EXTENSIONS]

    media_errors: list[dict] = []
    media_info: dict[str, dict] = {}
    resolutions = Counter()
    formats = Counter()
    modes = Counter()
    exif_images = 0
    kaggle_hash_groups: dict[str, list[Path]] = defaultdict(list)
    dhashes: dict[int, int] = {}
    image_by_id: dict[int, Path] = {}
    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                image_format = image.format or image_path.suffix.lstrip(".").upper()
                exif = image.getexif()
                if exif:
                    exif_images += 1
                try:
                    numeric_id = int(image_path.stem)
                    image_by_id[numeric_id] = image_path
                    dhashes[numeric_id] = dhash(image)
                except ValueError:
                    pass
                media_info[image_path.name] = {"width": width, "height": height, "path": rel(image_path)}
                resolutions[f"{width}x{height}"] += 1
                formats[image_format] += 1
                modes[image.mode] += 1
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            media_errors.append({"path": rel(image_path), "error": f"{type(exc).__name__}: {exc}"})
        kaggle_hash_groups[hash_file(image_path)].append(image_path)

    try:
        annotations_document = json.loads(ANNOTATION_FILE.read_text(encoding="utf-8"))
        annotations = annotations_document["annotations"]
        parse_errors: list[str] = []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        annotations = []
        parse_errors = [f"{type(exc).__name__}: {exc}"]

    all_by_file: dict[str, list[dict]] = defaultdict(list)
    outer_validation = Counter()
    inner_validation = Counter()
    missing_refs = set()
    outer_widths: list[float] = []
    outer_heights: list[float] = []
    outer_areas: list[float] = []
    outer_normalized_areas: list[float] = []
    safe_widths: list[float] = []
    safe_heights: list[float] = []
    safe_areas: list[float] = []
    safe_normalized_areas: list[float] = []
    inner_colors = Counter()
    inner_color_images: dict[str, set[str]] = defaultdict(set)
    inner_shapes = Counter()
    inner_shape_images: dict[str, set[str]] = defaultdict(set)
    inner_values = Counter()
    inner_value_images: dict[str, set[str]] = defaultdict(set)
    signal_states = Counter()
    signal_state_images: dict[str, set[str]] = defaultdict(set)
    safe_states = Counter()
    safe_state_images: dict[str, set[str]] = defaultdict(set)
    ignored = Counter()
    candidates: dict[str, list[tuple[dict, Path]]] = defaultdict(list)

    for annotation in annotations:
        filename = str(annotation.get("filename", ""))
        normalized = filename.replace("\\", "/")
        image_path = TRAIN_ROOT / Path(normalized)
        info = media_info.get(image_path.name)
        width = info["width"] if info else None
        height = info["height"] if info else None
        if not image_path.is_file():
            missing_refs.add(filename)
        all_by_file[filename].append(annotation)
        ignored[str(annotation.get("ignore"))] += 1

        result, coords = validate_box(annotation.get("bndbox"), width, height)
        for key, value in result.items():
            if value:
                outer_validation[key] += 1
        if image_path.is_file() and result["valid"] and coords:
            xmin, ymin, xmax, ymax = coords
            box_width, box_height = xmax - xmin, ymax - ymin
            area = box_width * box_height
            normalized_area = area / (width * height)
            outer_widths.append(box_width)
            outer_heights.append(box_height)
            outer_areas.append(area)
            outer_normalized_areas.append(normalized_area)
            candidates["general"].append((annotation, image_path))
            if box_width < 8 or box_height < 8:
                candidates["small_boxes"].append((annotation, image_path))

        colors = [str(item.get("color")) for item in annotation.get("inbox", [])]
        color_set = sorted(set(colors))
        state = "none" if not color_set else "+".join(color_set)
        signal_states[state] += 1
        signal_state_images[state].add(filename)
        if annotation.get("ignore") == 0 and not color_set:
            candidates["ambiguous"].append((annotation, image_path))
        if annotation.get("ignore") == 0 and len(color_set) > 1:
            candidates["ambiguous"].append((annotation, image_path))

        safe_state = color_set[0] if annotation.get("ignore") == 0 and len(color_set) == 1 and color_set[0] in {"red", "yellow", "green"} and result["valid"] else None
        if safe_state:
            safe_states[safe_state] += 1
            safe_state_images[safe_state].add(filename)
            candidates[safe_state].append((annotation, image_path))
            if coords:
                xmin, ymin, xmax, ymax = coords
                box_width, box_height = xmax - xmin, ymax - ymin
                area = box_width * box_height
                safe_widths.append(box_width)
                safe_heights.append(box_height)
                safe_areas.append(area)
                safe_normalized_areas.append(area / (width * height))

        for lamp in annotation.get("inbox", []):
            color = str(lamp.get("color"))
            shape = str(lamp.get("shape"))
            value = str(lamp.get("value"))
            inner_colors[color] += 1
            inner_color_images[color].add(filename)
            inner_shapes[shape] += 1
            inner_shape_images[shape].add(filename)
            inner_values[value] += 1
            inner_value_images[value].add(filename)
            if shape != "-1":
                candidates["directional"].append((annotation, image_path))
            lamp_result, _ = validate_box(lamp.get("bndbox"), width, height)
            for key, flag in lamp_result.items():
                if flag:
                    inner_validation[key] += 1

    very_small = sum(width < 4 or height < 4 for width, height in zip(outer_widths, outer_heights))
    small = sum(width < 8 or height < 8 for width, height in zip(outer_widths, outer_heights))
    safe_very_small = sum(width < 4 or height < 4 for width, height in zip(safe_widths, safe_heights))
    safe_small = sum(width < 8 or height < 8 for width, height in zip(safe_widths, safe_heights))

    within_duplicate_groups = [paths for paths in kaggle_hash_groups.values() if len(paths) > 1]
    bosch_hash_groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((BOSCH_ROOT / "images").rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            bosch_hash_groups[hash_file(path)].append(path)
    cross_hashes = sorted(set(kaggle_hash_groups).intersection(bosch_hash_groups))
    cross_kaggle_images = sum(len(kaggle_hash_groups[value]) for value in cross_hashes)
    cross_bosch_images = sum(len(bosch_hash_groups[value]) for value in cross_hashes)
    cross_pairs = sum(len(kaggle_hash_groups[value]) * len(bosch_hash_groups[value]) for value in cross_hashes)

    adjacent_distances = []
    adjacent_cross_split = []
    for numeric_id in sorted(image_by_id):
        if numeric_id + 1 not in image_by_id:
            continue
        distance = (dhashes[numeric_id] ^ dhashes[numeric_id + 1]).bit_count()
        first_split = "train" if "train_dataset" in image_by_id[numeric_id].parts else "test"
        second_split = "train" if "train_dataset" in image_by_id[numeric_id + 1].parts else "test"
        adjacent_distances.append(distance)
        if first_split != second_split:
            adjacent_cross_split.append(distance)

    bosch_qa = json.loads((BOSCH_ROOT / "qa_report.json").read_text(encoding="utf-8"))
    bosch_stats = bosch_qa["bbox_statistics"]["global"]

    annotation_keys = Counter(tuple(sorted(item.keys())) for item in annotations)
    inner_records = [lamp for item in annotations for lamp in item.get("inbox", [])]
    inner_keys = Counter(tuple(sorted(item.keys())) for item in inner_records)

    report = {
        "audit": {
            "title": "Project 363 Kaggle Traffic Light Dataset Extension Audit",
            "status": "REVIEW_REQUIRED",
            "recommendation": "MANUAL_REVIEW_REQUIRED",
            "subset_assessment": "USE_SUBSET",
            "deterministic_sample_seed": SAMPLE_SEED,
            "source_mutated": False,
        },
        "dataset_structure": {
            "dataset_root": rel(SOURCE),
            "total_size_bytes": sum(path.stat().st_size for path in source_files),
            "total_files": len(source_files),
            "tree": {
                "train_dataset": {"train_images": len(train_images), "train.json": ANNOTATION_FILE.stat().st_size},
                "test_dataset": {"test_images": len(test_images)},
                "submit_example.json": (SOURCE / "submit_example.json").stat().st_size,
            },
            "image_directories": [rel(TRAIN_ROOT / "train_images"), rel(TEST_ROOT / "test_images")],
            "annotation_files": [rel(ANNOTATION_FILE), rel(SOURCE / "submit_example.json")],
            "readme_or_license_files": [],
            "existing_split": {"train": len(train_images), "validation": 0, "test": len(test_images), "test_is_unlabeled": True},
        },
        "media": {
            "total_images": len(image_files),
            "train_images": len(train_images),
            "validation_images": 0,
            "test_images": len(test_images),
            "formats": dict(formats),
            "modes": dict(modes),
            "corrupted_images": len(media_errors),
            "corruption_details": media_errors,
            "resolution_distribution": dict(sorted(resolutions.items(), key=lambda item: (-item[1], item[0]))),
            "common_resolutions": [{"resolution": value, "images": count} for value, count in resolutions.most_common(10)],
            "images_with_exif": exif_images,
        },
        "annotation_schema": {
            "format": "JSON object-detection annotations with outer signal bndbox and nested inbox lamp records",
            "top_level": {"annotations": "array"},
            "outer_record_keys": [{"keys": list(keys), "records": count} for keys, count in annotation_keys.items()],
            "inner_record_keys": [{"keys": list(keys), "records": count} for keys, count in inner_keys.items()],
            "image_path": "annotations[].filename, relative to train_dataset",
            "traffic_light_bbox": "annotations[].bndbox (xmin, ymin, xmax, ymax; pixel coordinates)",
            "lamp_bbox": "annotations[].inbox[].bndbox (xmin, ymin, xmax, ymax; pixel coordinates)",
            "color_state": "annotations[].inbox[].color",
            "type_or_direction": "annotations[].inbox[].shape (numeric-string code; no local legend)",
            "other_attributes": {
                "outer": ["truncated", "occluded", "ignore"],
                "inner": ["truncated", "occluded", "difficult", "value"],
            },
            "important_semantics": "Color belongs to nested lamp records, not directly to the outer traffic-light box. Shape also belongs to a lamp record and has no local legend.",
        },
        "original_labels": {
            "explicit_outer_category_field": False,
            "colors": counter_table(inner_colors, inner_color_images),
            "shape_codes": counter_table(inner_shapes, inner_shape_images),
            "value_labels": counter_table(inner_values, inner_value_images),
            "outer_signal_color_combinations_all": counter_table(signal_states, signal_state_images),
            "ignore_flag": dict(ignored),
        },
        "potential_mapping": {
            "red": {"bosch_label": "red", "rating": "SAFE", "condition": "ignore=0, valid outer bbox, and all explicit nested colors are red"},
            "yellow": {"bosch_label": "yellow", "rating": "SAFE", "condition": "ignore=0, valid outer bbox, and all explicit nested colors are yellow"},
            "green": {"bosch_label": "green", "rating": "SAFE", "condition": "ignore=0, valid outer bbox, and all explicit nested colors are green"},
            "none": {"bosch_label": "off", "rating": "AMBIGUOUS", "condition": "empty inbox is not an explicit off label and may mean unrecognized/ignored/unannotated"},
            "green+red": {"bosch_label": None, "rating": "AMBIGUOUS", "condition": "multiple simultaneous color records; may represent different arrows/lanes or transition"},
            "red+yellow": {"bosch_label": None, "rating": "AMBIGUOUS", "condition": "multiple simultaneous color records; Bosch V1 has no compound state"},
            "ignored_records": {"bosch_label": None, "rating": "UNMAPPABLE", "condition": "source explicitly marks ignore=1"},
            "safe_subset": "Only non-ignored, geometrically valid outer boxes whose non-empty nested colors all agree on red, yellow, or green.",
            "apply_mapping": False,
        },
        "color_distribution": {
            "nested_lamp_annotations": dict(inner_colors),
            "potential_usable_outer_boxes": {"red": safe_states["red"], "yellow": safe_states["yellow"], "green": safe_states["green"], "off": 0},
            "potential_usable_images": {"red": len(safe_state_images["red"]), "yellow": len(safe_state_images["yellow"]), "green": len(safe_state_images["green"]), "off": 0},
            "class_imbalance_ratio_max_to_min_nonzero": float(max(safe_states.values()) / min(safe_states.values())),
            "bosch_v1": {"red": 4164, "yellow": 444, "green": 5421, "off": 726},
            "minority_class_assessment": "Adds a limited number of safe yellow boxes and no explicitly labeled off boxes.",
        },
        "direction_information": {
            "support": "PARTIAL",
            "shape_attribute_present": True,
            "shape_code_legend_present_locally": False,
            "codes": counter_table(inner_shapes, inner_shape_images),
            "left": "UNKNOWN — MANUAL REVIEW REQUIRED",
            "right": "UNKNOWN — MANUAL REVIEW REQUIRED",
            "straight": "UNKNOWN — MANUAL REVIEW REQUIRED",
            "u_turn": "UNKNOWN — MANUAL REVIEW REQUIRED",
            "lane_specific": "UNKNOWN — MANUAL REVIEW REQUIRED",
            "other_directional_states": "Numeric shape codes and auxiliary values exist, but their meanings are undocumented locally.",
            "v2_assessment": "Potentially useful after obtaining an authoritative shape-code legend and manually validating code semantics; color and shape are separate nested attributes.",
        },
        "bbox_validation": {
            "outer_traffic_light_boxes": {**{key: int(outer_validation[key]) for key in ("valid", "invalid_geometry", "out_of_bounds", "zero_area", "non_finite", "does_not_intersect_image")}, "total": len(annotations)},
            "inner_lamp_boxes": {**{key: int(inner_validation[key]) for key in ("valid", "invalid_geometry", "out_of_bounds", "zero_area", "non_finite", "does_not_intersect_image")}, "total": len(inner_records)},
            "missing_image_references": len(missing_refs),
            "missing_image_reference_details": sorted(missing_refs),
            "parse_errors": parse_errors,
            "note": "Counts are reported separately because outer boxes are traffic-light objects while nested boxes are illuminated lamp regions.",
        },
        "object_size_analysis": {
            "population": "all geometrically valid, in-bounds outer traffic-light boxes, including ignore=1",
            "pixel_width": describe(outer_widths),
            "pixel_height": describe(outer_heights),
            "pixel_area": describe(outer_areas),
            "normalized_area": describe(outer_normalized_areas),
            "very_small": {"definition": "width < 4 px OR height < 4 px", "count": very_small, "percentage": 100 * very_small / len(outer_widths)},
            "small": {"definition": "width < 8 px OR height < 8 px", "count": small, "percentage": 100 * small / len(outer_widths)},
            "safe_subset": {
                "boxes": len(safe_widths),
                "pixel_width": describe(safe_widths),
                "pixel_height": describe(safe_heights),
                "pixel_area": describe(safe_areas),
                "normalized_area": describe(safe_normalized_areas),
                "very_small": safe_very_small,
                "small": safe_small,
            },
            "bosch_comparison": {
                "bosch_pixel_width": bosch_stats["pixel_width"],
                "bosch_pixel_height": bosch_stats["pixel_height"],
                "bosch_pixel_area": bosch_stats["pixel_area"],
                "bosch_normalized_area": bosch_stats["normalized_area"],
                "bosch_very_small": bosch_qa["small_box_analysis"]["very_small"],
                "bosch_small": bosch_qa["small_box_analysis"]["small"],
                "assessment": "Kaggle outer traffic-light boxes are substantially larger in pixels and normalized area than Bosch boxes.",
            },
        },
        "duplicate_analysis": {
            "method": "SHA-256 exact file hash",
            "within_kaggle": {
                "duplicate_groups": len(within_duplicate_groups),
                "images_in_duplicate_groups": sum(len(group) for group in within_duplicate_groups),
                "redundant_copies": sum(len(group) - 1 for group in within_duplicate_groups),
                "groups": [[rel(path) for path in group] for group in within_duplicate_groups],
            },
            "kaggle_to_bosch": {
                "matching_hashes": len(cross_hashes),
                "kaggle_images": cross_kaggle_images,
                "bosch_images": cross_bosch_images,
                "image_pairs": cross_pairs,
            },
        },
        "leakage_risk": {
            "filenames": "Numeric names 00001.jpg through 03000.jpg are partitioned between train and test rather than reset per split.",
            "metadata": f"{exif_images} of {len(image_files)} images contain EXIF metadata.",
            "adjacent_numeric_pairs": len(adjacent_distances),
            "adjacent_dhash_distance": describe(adjacent_distances),
            "adjacent_pairs_crossing_existing_split": len(adjacent_cross_split),
            "cross_split_adjacent_pairs_dhash_le_5": sum(value <= 5 for value in adjacent_cross_split),
            "assessment": "Sequential naming and visual similarity among neighboring IDs indicate likely bursts/sequences, while no authoritative sequence/session IDs are provided.",
            "safest_future_split_unit": "UNKNOWN — MANUAL REVIEW REQUIRED",
            "split_guidance": "Recover source video/camera-session boundaries; if unavailable, keep conservative contiguous filename runs together and validate with near-duplicate analysis. Do not reuse the existing random-looking train/test partition as evidence of leakage safety.",
        },
        "license": {
            "local_license_file": None,
            "local_readme_file": None,
            "license": "UNKNOWN — MANUAL REVIEW REQUIRED",
            "research_restrictions": "UNKNOWN — MANUAL REVIEW REQUIRED",
            "redistribution_restrictions": "UNKNOWN — MANUAL REVIEW REQUIRED",
            "attribution_requirements": "UNKNOWN — MANUAL REVIEW REQUIRED",
        },
        "domain_fit": {
            "rating": "MEDIUM",
            "basis": "Deterministic contact sheets show vehicle-mounted forward road views in Chinese urban/intersection settings, mostly day with clear, overcast, rainy, and some night scenes. Traffic-light scale and context are relevant, but viewpoint differs from fixed traffic cameras.",
        },
        "bosch_comparison": {
            "bosch": {
                "images": 5093, "usable_boxes": 10755, "red": 4164, "yellow": 444, "green": 5421, "off": 726,
                "very_small_percentage": bosch_qa["small_box_analysis"]["very_small"]["percentage_of_boxes"],
                "small_percentage": bosch_qa["small_box_analysis"]["small"]["percentage_of_boxes"],
                "median_bbox_size": [bosch_stats["pixel_width"]["median"], bosch_stats["pixel_height"]["median"]],
                "directional_information": "Collapsed in V1; available in Bosch source labels",
                "domain_fit": "HIGH for vehicle-mounted road scenes",
                "invalid_bbox_count": 0,
                "license_clarity": "Not stated in local processed README/QA",
            },
            "kaggle": {
                "images": len(image_files), "usable_boxes": sum(safe_states.values()), "red": safe_states["red"], "yellow": safe_states["yellow"], "green": safe_states["green"], "off": 0,
                "very_small_percentage": 100 * safe_very_small / len(safe_widths),
                "small_percentage": 100 * safe_small / len(safe_widths),
                "median_bbox_size": [describe(safe_widths)["median"], describe(safe_heights)["median"]],
                "directional_information": "PARTIAL: shape codes present, legend absent",
                "domain_fit": "MEDIUM",
                "invalid_bbox_count": int(outer_validation["invalid_geometry"] + outer_validation["out_of_bounds"]),
                "license_clarity": "UNKNOWN — MANUAL REVIEW REQUIRED",
            },
            "does_kaggle_add_information": "Yes: substantially larger signals, non-US designs/environments, and separate shape codes. It does not add safe off labels, adds relatively few yellow boxes, and introduces unresolved semantics, leakage, and license risks.",
        },
        "final_recommendation": {
            "recommendation": "MANUAL_REVIEW_REQUIRED",
            "use_subset": True,
            "safe_subset": "Non-ignored, valid outer boxes with one or more nested lamps whose explicit color labels all agree on red, yellow, or green.",
            "do_not_use_without_review": ["ignore=1 records", "empty inbox records as off", "mixed-color outer signals", "shape codes as named directions", "unlabeled test images"],
            "rationale": "The safe color subset is valuable and much larger-object-biased, but local files do not establish license terms, off semantics, shape-code meanings, or leakage-safe sequence units.",
        },
    }

    for category in ("general", "red", "yellow", "green", "small_boxes", "directional", "ambiguous"):
        unique = {}
        for annotation, image_path in candidates[category]:
            if not image_path.is_file():
                continue
            key = (annotation.get("filename"), json.dumps(annotation.get("bndbox", {}), sort_keys=True))
            unique[key] = (annotation, image_path)
        selected = sorted(unique.values(), key=lambda item: deterministic_score(f"{category}:{item[0].get('filename')}:{item[0].get('bndbox')}"))[:SAMPLE_COUNT]
        create_contact_sheet(OUTPUT / category / "contact_sheet.jpg", f"Kaggle traffic-light audit — {category}", selected, all_by_file)
    create_placeholder(
        OUTPUT / "off" / "contact_sheet.jpg",
        "Kaggle traffic-light audit — off",
        "No explicit 'off' color/state occurs in train.json.\nEmpty inbox records were NOT inferred as off.",
    )

    (OUTPUT / "kaggle_tl_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "images": len(image_files),
        "outer_boxes": len(annotations),
        "safe_states": dict(safe_states),
        "outer_validation": dict(outer_validation),
        "inner_validation": dict(inner_validation),
        "very_small": very_small,
        "small": small,
        "duplicates": report["duplicate_analysis"],
        "resolution_top": resolutions.most_common(10),
        "output": rel(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
