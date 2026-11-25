import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "Vehicle363-v1"
OUT = ROOT / "data" / "vehicle_audit" / "duplicate_resolution"
NAMES = ["car", "motorcycle", "bus", "truck"]
SEED = 36347


def save_json(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def member_key(member):
    return member["source"] + ":" + member["image"]


def pixel_boxes(labels, width, height):
    result = []
    for class_id, x_center, y_center, box_width, box_height, *_ in labels:
        result.append([class_id, (x_center - box_width / 2) * width,
                       (y_center - box_height / 2) * height,
                       (x_center + box_width / 2) * width,
                       (y_center + box_height / 2) * height])
    return result


def bbox_counts(labels):
    return [sum(label[0] == class_id for label in labels) for class_id in range(4)]


def valid_box(box, width, height):
    class_id, left, top, right, bottom = box
    return class_id in range(4) and all(math.isfinite(value) for value in box) and -1e-9 <= left < right <= width + 1e-9 and -1e-9 <= top < bottom <= height + 1e-9


def iou(first, second):
    intersection = max(0, min(first[3], second[3]) - max(first[1], second[1])) * max(0, min(first[4], second[4]) - max(first[2], second[2]))
    first_area = (first[3] - first[1]) * (first[4] - first[2])
    second_area = (second[3] - second[1]) * (second[4] - second[2])
    denominator = first_area + second_area - intersection
    return intersection / denominator if denominator > 0 else 0.0


def iou_matrix(first, second, same_class=True):
    return [[iou(left, right) if not same_class or left[0] == right[0] else 0.0 for right in second] for left in first]


def maximum_weight_assignment(weights):
    row_count = len(weights)
    col_count = len(weights[0]) if weights else 0
    dimension = max(row_count, col_count)
    if not dimension:
        return []
    costs = np.ones((dimension, dimension))
    if row_count and col_count:
        costs[:row_count, :col_count] -= np.asarray(weights)
    row_potential = [0.0] * (dimension + 1)
    col_potential = [0.0] * (dimension + 1)
    row_at_column = [0] * (dimension + 1)
    predecessor = [0] * (dimension + 1)
    for row in range(1, dimension + 1):
        row_at_column[0] = row
        current_column = 0
        minimum = [float("inf")] * (dimension + 1)
        visited = [False] * (dimension + 1)
        while True:
            visited[current_column] = True
            current_row = row_at_column[current_column]
            delta, next_column = float("inf"), 0
            for column in range(1, dimension + 1):
                if not visited[column]:
                    reduced = costs[current_row - 1, column - 1] - row_potential[current_row] - col_potential[column]
                    if reduced < minimum[column]:
                        minimum[column] = reduced
                        predecessor[column] = current_column
                    if minimum[column] < delta:
                        delta, next_column = minimum[column], column
            for column in range(dimension + 1):
                if visited[column]:
                    row_potential[row_at_column[column]] += delta
                    col_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if row_at_column[current_column] == 0:
                break
        while current_column:
            previous_column = predecessor[current_column]
            row_at_column[current_column] = row_at_column[previous_column]
            current_column = previous_column
    return sorted((row_at_column[column] - 1, column - 1) for column in range(1, dimension + 1)
                  if 0 < row_at_column[column] <= row_count and column <= col_count)


def best_pairs(first, second):
    matrix = iou_matrix(first, second)
    pairs = []
    for class_id in range(4):
        left_indices = [index for index, box in enumerate(first) if box[0] == class_id]
        right_indices = [index for index, box in enumerate(second) if box[0] == class_id]
        if not left_indices or not right_indices:
            continue
        weights = [[matrix[left][right] for right in right_indices] for left in left_indices]
        for left, right in maximum_weight_assignment(weights):
            first_index, second_index = left_indices[left], right_indices[right]
            pairs.append((first_index, second_index, matrix[first_index][second_index]))
    return sorted(pairs)


def high_confidence_match(first, second, threshold):
    matrix = iou_matrix(first, second)
    graph = [[column for column, value in enumerate(row) if value >= threshold and first[index][0] == second[column][0]]
             for index, row in enumerate(matrix)]

    def find_assignment(forbidden=None):
        assigned_columns = {}

        def augment(row, visited):
            for column in graph[row]:
                if (row, column) == forbidden or column in visited:
                    continue
                visited.add(column)
                if column not in assigned_columns or augment(assigned_columns[column], visited):
                    assigned_columns[column] = row
                    return True
            return False

        for row in range(len(first)):
            if not augment(row, set()):
                return None
        return sorted((row, column) for column, row in assigned_columns.items())

    assignment = find_assignment()
    if assignment is None:
        return {"complete": False, "unique": False, "pairs": [], "unmatched": list(range(len(second))), "reason": "no_complete_high_iou_same_class_matching"}
    if any(len(row) > 1 for row in graph):
        for edge in assignment:
            if find_assignment(edge) is not None:
                return {"complete": True, "unique": False, "pairs": [(left, right, matrix[left][right]) for left, right in assignment],
                        "unmatched": [], "reason": "multiple_possible_one_to_one_matchings"}
    matched = {right for _, right in assignment}
    return {"complete": True, "unique": True, "pairs": [(left, right, matrix[left][right]) for left, right in assignment],
            "unmatched": [index for index in range(len(second)) if index not in matched], "reason": "unique_high_iou_same_class_matching"}


def suspicious_pairs(boxes, threshold=0.5):
    result = []
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            overlap = iou(boxes[first], boxes[second])
            if overlap >= threshold:
                result.append({"first_box": first, "second_box": second, "iou": overlap,
                               "reason": "same_class_duplicate_overlap" if boxes[first][0] == boxes[second][0] else "cross_class_contradiction_overlap"})
    return result


def verify_current_labels(member):
    record = member["final_record"]
    if record is None:
        return None
    image_relative = Path(record["final_image"])
    label_path = DATASET / "labels" / record["final_split"] / image_relative.with_suffix(".txt").name
    content = label_path.read_bytes()
    rows = []
    for line in content.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"0", "1", "2", "3"}:
            raise RuntimeError(f"Invalid current label: {label_path}")
        rows.append(tuple([int(fields[0]), *[round(float(value), 12) for value in fields[1:]]]))
    expected = sorted(tuple([label[0], *[round(value, 12) for value in label[1:5]]]) for label in member["target_annotations"])
    if sorted(rows) != expected:
        raise RuntimeError(f"Current label disagrees with duplicate report: {label_path}")
    return str(label_path.relative_to(ROOT)), hashlib.sha256(content).hexdigest()


def load_inputs():
    paths = [DATASET / "metadata" / name for name in ["duplicate_report.json", "source_manifest.json", "class_mapping.json", "dataset_statistics.json", "output_validation.json"]]
    paths += [DATASET / "qa" / "duplicate_review_manifest.json"]
    snapshot = {str(path.relative_to(ROOT)): digest_file(path) for path in paths}
    duplicates = load_json(paths[0])
    conflicts = duplicates["annotation_conflicts"]
    supplemental = duplicates.get("compatible_annotation_groups_requiring_review", [])
    if len({group["group_id"] for group in conflicts}) != len(conflicts):
        raise RuntimeError("Duplicate conflict IDs are not unique")
    source_manifest = load_json(paths[1])
    mapping = load_json(paths[2])
    statistics_data = load_json(paths[3])
    qa_metadata = load_json(paths[5])
    if mapping["schema"] != dict(zip(map(str, range(4)), NAMES)):
        raise RuntimeError("Target schema does not match locked Vehicle363 mapping")
    relevant_keys = {(member["source"], member["image"]) for group in conflicts + supplemental for member in group["members"]}
    relevant = {}
    group_splits = defaultdict(set)
    atomic_splits = defaultdict(set)
    hash_splits = defaultdict(set)
    image_names = set()
    for record in source_manifest:
        if record["final_image"] in image_names:
            raise RuntimeError("Repeated final image in existing source manifest")
        image_names.add(record["final_image"])
        group_splits[record["duplicate_group"]].add(record["final_split"])
        atomic_splits[record["atomic_group"]].add(record["final_split"])
        hash_splits[record["sha256"]].add(record["final_split"])
        key = (record["source_dataset"], record["source_image"])
        if key in relevant_keys:
            relevant[key] = record
    if len(source_manifest) != statistics_data["images"]:
        raise RuntimeError("Source manifest and dataset image totals disagree")
    del source_manifest
    existing_leakage = {"exact_duplicate_group_cross_split_overlap": sum(len(values) > 1 for values in group_splits.values()),
                        "atomic_group_cross_split_overlap": sum(len(values) > 1 for values in atomic_splits.values()),
                        "exact_hash_cross_split_overlap": sum(len(values) > 1 for values in hash_splits.values())}
    if any(existing_leakage.values()):
        raise RuntimeError(f"Existing dataset leakage: {existing_leakage}")
    with (ROOT / "data" / "vehicle_audit" / "vehicle363_build" / "image_scan.pkl").open("rb") as handle:
        image_cache = pickle.load(handle)
    dimensions = {key: (image_cache[key]["width"], image_cache[key]["height"]) for key in relevant_keys}
    for group in conflicts + supplemental:
        hashes = {image_cache[(member["source"], member["image"])]["sha256"] for member in group["members"]}
        if len(hashes) != 1 or group["group_id"] != "exact__" + next(iter(hashes)):
            raise RuntimeError(f"Hash metadata cannot be reconciled: {group['group_id']}")
        if len({dimensions[(member["source"], member["image"])] for member in group["members"]}) != 1:
            raise RuntimeError(f"Dimensions disagree within exact group: {group['group_id']}")
        for member in group["members"]:
            key = (member["source"], member["image"])
            final_record = relevant.get(key)
            if member["target_annotations"] and final_record is None:
                raise RuntimeError(f"Positive annotation member absent from source manifest: {key}")
            if final_record and (final_record["duplicate_group"] != group["group_id"] or final_record["sha256"] not in hashes
                                 or final_record["target_bbox_count"] != len(member["target_annotations"])
                                 or final_record["target_class_counts"] != dict(zip(NAMES, bbox_counts(member["target_annotations"])))):
                raise RuntimeError(f"Annotation/source manifest disagreement: {key}")
            member["width"], member["height"] = dimensions[key]
            member["pixel_boxes"] = pixel_boxes(member["target_annotations"], *dimensions[key])
            member["member_id"] = member_key(member)
            member["final_record"] = final_record
    del image_cache
    source_path = ROOT / "data" / "MIO-TCD-Localization" / "MIO-TCD-Localization" / "gt_train.csv"
    snapshot[str(source_path.relative_to(ROOT))] = digest_file(source_path)
    locked_map = {policy["source_class"]: policy["target_class_id"] for policy in mapping["mio_source_policies"] if policy["target_class_id"] is not None}
    source_labels = {key: [] for key in relevant_keys if key[0] == "MIO"}
    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        for image_id, source_class, *coordinates in csv.reader(handle):
            key = ("MIO", f"train/{image_id}.jpg")
            if key not in source_labels or source_class not in locked_map:
                continue
            width, height = dimensions[key]
            left, top, right, bottom = map(float, coordinates)
            if 0 <= left < right <= width and 0 <= top < bottom <= height:
                source_labels[key].append((locked_map[source_class], (left + right) / (2 * width), (top + bottom) / (2 * height),
                                           (right - left) / width, (bottom - top) / height))
    for group in conflicts + supplemental:
        for member in group["members"]:
            key = (member["source"], member["image"])
            if key[0] != "MIO":
                raise RuntimeError("Unexpected non-MIO conflict needs source validation implementation")
            expected = sorted(tuple([label[0], *[round(value, 12) for value in label[1:5]]]) for label in source_labels[key])
            recorded = sorted(tuple([label[0], *[round(value, 12) for value in label[1:5]]]) for label in member["target_annotations"])
            if expected != recorded:
                raise RuntimeError(f"Existing duplicate annotations differ from local gt_train.csv: {key}")
    relevant_members = [member for group in conflicts + supplemental for member in group["members"]]
    with ThreadPoolExecutor(max_workers=12) as executor:
        label_checksums = [result for result in executor.map(verify_current_labels, relevant_members) if result is not None]
    snapshot.update(label_checksums)
    save_json("input_validation.json", {"recalculated_conflict_groups": len(conflicts), "supplemental_compatible_groups": len(supplemental),
              "current_positive_member_labels_verified": len(label_checksums), "source_annotations_reconciled": True,
              "dimensions_source": "Existing Work 4.7 decoded-image cache", "source_image_rescan_performed": False,
              "baseline_images": statistics_data["images"], "baseline_boxes": statistics_data["boxes"],
              "input_sha256": snapshot, "existing_split_safety": existing_leakage})
    return {"conflicts": conflicts, "supplemental": supplemental, "statistics": statistics_data,
            "snapshot": snapshot, "split_safety": existing_leakage,
            "qa_metadata": {"existing_sample_groups": len(qa_metadata["examples"]), "recalculated_conflicts": len(conflicts)},
            "source_annotations_reconciled": True, "source_image_rescan_performed": False}


def distribution(values):
    if not values:
        return {"count": 0}
    histogram, edges = np.histogram(values, bins=[0, .1, .3, .5, .7, .8, .85, .9, .95, 1.0000001])
    return {"count": len(values), "min": min(values), "max": max(values),
            "quantiles": {str(percentile): float(np.percentile(values, percentile)) for percentile in [5, 10, 25, 50, 75, 90, 95]},
            "histogram": [{"lower_inclusive": float(edges[index]), "upper_exclusive": float(edges[index + 1]), "count": int(count)} for index, count in enumerate(histogram)]}


def calibrate(inputs):
    strata = defaultdict(list)
    for group in inputs["conflicts"]:
        sizes = {len(member["pixel_boxes"]) for member in group["members"]}
        stratum = "equal_box_count" if len(sizes) == 1 else "different_box_count"
        if len(group["members"]) > 2:
            stratum = "multiple_copies"
        strata[stratum].append(group)
    sampled = []
    for name, groups in sorted(strata.items()):
        random.Random(SEED).shuffle(groups)
        sampled.extend(groups[:200])
    rows = []
    buckets = defaultdict(list)
    for group in sampled:
        ordered = sorted(group["members"], key=lambda member: (len(member["pixel_boxes"]), member["member_id"]))
        first, second = ordered[0], ordered[-1]
        pairs = best_pairs(first["pixel_boxes"], second["pixel_boxes"])
        matched_left, matched_right = set(), set()
        for left, right, overlap in pairs:
            first_box, second_box = first["pixel_boxes"][left], second["pixel_boxes"][right]
            corner_shift = max(abs(first_box[index] - second_box[index]) for index in range(1, 5))
            size = min((first_box[3] - first_box[1]) * (first_box[4] - first_box[2]),
                       (second_box[3] - second_box[1]) * (second_box[4] - second_box[2]))
            if overlap > 0 and corner_shift <= 2.0:
                bucket = "same_object_small_jitter_geometric_proxy"
            elif overlap == 0:
                bucket = "clearly_different_geometric_proxy"
            else:
                bucket = "uncertain_geometric_proxy"
            buckets[bucket].append(overlap)
            if bucket == "same_object_small_jitter_geometric_proxy" and size >= 32 ** 2:
                buckets["small_jitter_non_tiny_boxes"].append(overlap)
            rows.append({"group_id": group["group_id"], "first_member": first["member_id"], "second_member": second["member_id"],
                         "first_box": left, "second_box": right, "class": NAMES[first_box[0]], "iou": overlap,
                         "maximum_corner_shift_pixels": corner_shift, "minimum_bbox_area": size, "proxy_category": bucket})
            matched_left.add(left)
            matched_right.add(right)
        for member, matched in [(first, matched_left), (second, matched_right)]:
            for index, box in enumerate(member["pixel_boxes"]):
                if index not in matched:
                    rows.append({"group_id": group["group_id"], "unmatched_member": member["member_id"], "box": index, "class": NAMES[box[0]], "iou": None, "proxy_category": "unmatched_object_not_same_class_pair"})
    report = {"seed": SEED, "sample_groups": len(sampled), "sampling": "Up to 200 groups per equal-count, unequal-count and multiple-copy strata; independent deterministic shuffle",
              "matching": "Maximum-total-IoU one-to-one Hungarian assignment separately per target class",
              "dimensions": "Previously decoded Work 4.7 image-dimension cache; no source image rescan",
              "proxy_warning": "Geometric strata are not human ground truth: <=2px maximum corner shift identifies the small-jitter proxy; zero IoU identifies clearly different geometry; other positive overlaps remain uncertain.",
              "distributions": {name: distribution(values) for name, values in buckets.items()},
              "unmatched_boxes": sum(row["iou"] is None for row in rows), "observations": rows,
              "threshold_candidates": {str(threshold): {name: sum(value >= threshold for value in values) for name, values in buckets.items()}
                                       for threshold in [.75, .8, .85, .9, .95]},
              "HIGH_CONFIDENCE_IOU": None, "threshold_selection_status": "PENDING_DISTRIBUTION_AND_VISUAL_REVIEW"}
    save_json("iou_calibration.json", report)
    text = ["# IoU calibration — Work 4.7A", "", f"Seed: {SEED}. Sample: {len(sampled)} conflict groups.", "", report["matching"], "", report["proxy_warning"], "",
            "| Geometric proxy | Matches | P05 | P25 | Median | P95 |", "|---|---:|---:|---:|---:|---:|"]
    for name, data in report["distributions"].items():
        quantiles = data.get("quantiles", {})
        text.append(f"| {name} | {data['count']} | " + " | ".join(f"{quantiles.get(str(percentile), 0):.4f}" for percentile in [5, 25, 50, 95]) + " |")
    text += ["", "HIGH_CONFIDENCE_IOU: pending review of these observed distributions and representative matched boxes. Raw coordinate equality is not the matching rule.", ""]
    (OUT / "iou_calibration.md").write_text("\n".join(text), encoding="utf-8")
    save_json("calibration_sample_groups.json", {"group_ids": [group["group_id"] for group in sampled]})
    print(json.dumps({"sample_groups": len(sampled), "distributions": report["distributions"], "threshold_candidates": report["threshold_candidates"]}, indent=2), flush=True)
    return report


def font(size=14):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def source_image(member):
    record = member["final_record"]
    if record is None:
        return ROOT / "data" / "MIO-TCD-Localization" / "MIO-TCD-Localization" / member["image"]
    return DATASET / record["final_image"]


def calibration_qa(inputs, report):
    groups = {group["group_id"]: group for group in inputs["conflicts"]}
    rows = [row for row in report["observations"] if row["iou"] is not None]
    selections = []
    for label, predicate in [("high IoU >=0.90", lambda row: row["iou"] >= .9),
                             ("near threshold 0.80..0.90", lambda row: .8 <= row["iou"] < .9),
                             ("low IoU positive", lambda row: 0 < row["iou"] < .75),
                             ("zero overlap", lambda row: row["iou"] == 0)]:
        candidates = [row for row in rows if predicate(row)]
        random.Random(SEED).shuffle(candidates)
        selections.extend((label, row) for row in candidates[:3])
    canvas = Image.new("RGB", (1440, 300 * math.ceil(len(selections) / 3)), "white")
    examples = []
    for index, (description, row) in enumerate(selections):
        cell = Image.new("RGB", (480, 300), "white")
        draw = ImageDraw.Draw(cell)
        draw.text((4, 2), f"{description} | {row['class']} IoU={row['iou']:.4f}", font=font(), fill="black")
        draw.text((4, 20), f"max corner shift: {row['maximum_corner_shift_pixels']:.1f}px", font=font(), fill="black")
        group = groups[row["group_id"]]
        for panel_index, (member_id, box_index) in enumerate([(row["first_member"], row["first_box"]), (row["second_member"], row["second_box"])]):
            member = next(member for member in group["members"] if member["member_id"] == member_id)
            box = member["pixel_boxes"][box_index]
            with Image.open(source_image(member)) as source:
                image = source.convert("RGB")
            _, left, top, right, bottom = box
            context = max(right - left, bottom - top) * .2 + 2
            crop_left, crop_top = max(0, math.floor(left - context)), max(0, math.floor(top - context))
            crop_right, crop_bottom = min(image.width, math.ceil(right + context)), min(image.height, math.ceil(bottom + context))
            crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            ImageDraw.Draw(crop).rectangle((left - crop_left, top - crop_top, right - crop_left, bottom - crop_top), outline="lime", width=2)
            scale = min(232 / crop.width, 205 / crop.height)
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
            cell.paste(crop, (panel_index * 240 + 4, 78))
            draw.text((panel_index * 240 + 4, 44), member["image"], font=font(), fill="black")
            draw.text((panel_index * 240 + 4, 60), f"bbox index {box_index}, set size {len(member['pixel_boxes'])}", font=font(), fill="black")
        canvas.paste(cell, ((index % 3) * 480, (index // 3) * 300))
        examples.append({"description": description, **row})
    directory = OUT / "qa" / "calibration"
    directory.mkdir(parents=True, exist_ok=True)
    canvas.save(directory / "matched_box_crops.jpg", quality=94)
    save_json("qa/calibration/manifest.json", {"examples": examples, "preview": "matched_box_crops.jpg"})
    return canvas


def select_threshold(report):
    threshold = .90
    report["HIGH_CONFIDENCE_IOU"] = threshold
    report["threshold_selection_status"] = "SELECTED_FROM_OBSERVED_DISTRIBUTIONS_AND_REPRESENTATIVE_CROP_REVIEW"
    report["threshold_reason"] = (
        "The 600-group calibration has small-jitter (<=2px) median IoU 0.9132 and non-tiny small-jitter median 0.9412. "
        "At 0.90, 150/177 non-tiny small-jitter pairs remain eligible; 0.95 leaves only 77/177. "
        "Use 0.90 conservatively rather than lowering to 0.80/0.85 to rescue tiny-object jitter. "
        "Tiny/low-IoU pairs remain human-review cases. Reviewed crop examples around and above the threshold "
        "show the same objects with changed edges, while zero-overlap assignments show distinct objects. "
        "No classification accuracy or semantic ground truth is inferred from geometric proxy strata. "
        "IoU is combined with unique one-to-one same-class matching, valid geometry, and overlap/extra-box checks.")
    report["visual_calibration_qa"] = "qa/calibration/matched_box_crops.jpg"
    report["suspicious_internal_overlap_iou"] = .50
    report["cross_class_overlap_review_iou"] = .50
    report["pixel_boundary_roundoff_tolerance"] = 1e-9
    save_json("iou_calibration.json", report)
    path = OUT / "iou_calibration.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("HIGH_CONFIDENCE_IOU: pending review of these observed distributions and representative matched boxes. Raw coordinate equality is not the matching rule.",
                        f"**HIGH_CONFIDENCE_IOU = {threshold:.2f}.**\n\n{report['threshold_reason']}\n\nSee `qa/calibration/matched_box_crops.jpg`. The categories above are geometric proxies, not verified semantic labels.\n\nSuspicious internal overlap threshold: IoU >=0.50, same-class duplicate or cross-class contradiction. This deliberately increases review load when valid occlusion and duplicate labeling cannot be distinguished automatically. Pixel conversion tolerates 1e-9 pixels of floating-point roundoff only; it never repairs coordinates.")
    path.write_text(text, encoding="utf-8")
    return threshold


def comparison_record(first, second, match, accepted):
    pairs = match["pairs"]
    diagnostic_only = not match["complete"]
    if diagnostic_only:
        pairs = best_pairs(first["pixel_boxes"], second["pixel_boxes"])
    return {"from_member": first["member_id"], "to_member": second["member_id"], "complete": match["complete"],
            "unique": match["unique"], "accepted": accepted, "reason": match["reason"],
            "pairs_are_diagnostics_only": diagnostic_only,
            "pairs": [{"from_box": left, "to_box": right, "class_id": first["pixel_boxes"][left][0],
                       "class_name": NAMES[first["pixel_boxes"][left][0]], "iou": overlap} for left, right, overlap in pairs],
            "additional_box_indices": match["unmatched"]}


def unresolved(group, reasons, diagnostics=None):
    members = group["members"]
    if diagnostics is None:
        first, second = members[0], members[1]
        pairs = best_pairs(first["pixel_boxes"], second["pixel_boxes"])
        diagnostics = [comparison_record(first, second, {"complete": False, "unique": False, "reason": "best_pair_diagnostics_only_not_accepted",
                                                       "pairs": pairs, "unmatched": []}, False)]
    return {"group_id": group["group_id"], "members": [member["member_id"] for member in members],
            "category": "TRUE_CONFLICT", "canonical_member": None,
            "annotation_counts": {member["member_id"]: len(member["pixel_boxes"]) for member in members},
            "matched_pairs": diagnostics, "match_ious": [pair["iou"] for comparison in diagnostics for pair in comparison["pairs"]],
            "additional_boxes": [], "reason": "; ".join(sorted(set(reasons))), "confidence": "LOW", "auto_resolvable": False}


def classify_group(group, threshold):
    members = sorted(group["members"], key=lambda member: member["member_id"])
    for member in members:
        if not all(valid_box(box, member["width"], member["height"]) for box in member["pixel_boxes"]):
            return unresolved(group, ["invalid_class_or_pixel_geometry"])
    if any(not member["pixel_boxes"] for member in members):
        return unresolved(group, ["empty_target_set_has_no_positive_high_iou_subset_evidence"])
    annotation_counts = [len(member["pixel_boxes"]) for member in members]
    if len(set(annotation_counts)) == 1:
        if any(bbox_counts(member["pixel_boxes"]) != bbox_counts(members[0]["pixel_boxes"]) for member in members[1:]):
            first = members[0]
            second = next(member for member in members[1:] if bbox_counts(member["pixel_boxes"]) != bbox_counts(first["pixel_boxes"]))
            diagnostic = comparison_record(first, second, {"complete": False, "unique": False, "pairs": [], "unmatched": [],
                                            "reason": "target_class_multiset_disagreement"}, False)
            return unresolved(group, ["same_box_count_but_different_target_class_multisets"], [diagnostic])
        comparisons = []
        for first_index in range(len(members)):
            for second_index in range(first_index + 1, len(members)):
                first, second = members[first_index], members[second_index]
                match = high_confidence_match(first["pixel_boxes"], second["pixel_boxes"], threshold)
                accepted = match["complete"] and match["unique"]
                comparisons.append(comparison_record(first, second, match, accepted))
                if not accepted:
                    return unresolved(group, [match["reason"]], comparisons)
        canonical_candidates = [member for member in members if not suspicious_pairs(member["pixel_boxes"])]
        if not canonical_candidates:
            return unresolved(group, ["all_jitter_candidate_sets_have_suspicious_duplicate_or_cross_class_overlaps"], comparisons)
        canonical = canonical_candidates[0]
        return {"group_id": group["group_id"], "members": [member["member_id"] for member in members],
                "category": "SAME_OBJECTS_JITTER", "canonical_member": canonical["member_id"],
                "annotation_counts": {member["member_id"]: len(member["pixel_boxes"]) for member in members},
                "matched_pairs": comparisons, "match_ious": [pair["iou"] for comparison in comparisons for pair in comparison["pairs"]],
                "additional_boxes": [], "reason": "All member pairs have equal target class multisets and unique high-IoU matches; select clean valid annotation set by source filename ordering, without averaging or union.",
                "confidence": "HIGH", "auto_resolvable": True}
    maximum_count = max(annotation_counts)
    candidates = [member for member in members if len(member["pixel_boxes"]) == maximum_count]
    failures = []
    best_diagnostics = None
    for canonical in candidates:
        suspicious = suspicious_pairs(canonical["pixel_boxes"])
        if suspicious:
            failures.append("superset_candidate_has_suspicious_internal_duplicate_or_cross_class_overlap")
            continue
        comparisons = []
        valid_candidate = True
        for smaller in members:
            if smaller is canonical:
                continue
            match = high_confidence_match(smaller["pixel_boxes"], canonical["pixel_boxes"], threshold)
            accepted = match["complete"] and match["unique"]
            comparisons.append(comparison_record(smaller, canonical, match, accepted))
            if not accepted:
                failures.append(match["reason"])
                valid_candidate = False
                break
            for additional in match["unmatched"]:
                added_box = canonical["pixel_boxes"][additional]
                if not valid_box(added_box, canonical["width"], canonical["height"]):
                    failures.append("invalid_additional_box")
                    valid_candidate = False
                for existing in smaller["pixel_boxes"]:
                    if iou(added_box, existing) >= .50:
                        failures.append("additional_box_overlaps_an_already_annotated_object")
                        valid_candidate = False
        if not valid_candidate:
            best_diagnostics = comparisons
            continue
        for other_candidate in candidates:
            if other_candidate is canonical:
                continue
            match = high_confidence_match(canonical["pixel_boxes"], other_candidate["pixel_boxes"], threshold)
            if not match["complete"] or not match["unique"]:
                valid_candidate = False
                failures.append("multiple_largest_sets_do_not_share_unique_same_semantics")
                break
        if valid_candidate:
            baseline = min(members, key=lambda member: (len(member["pixel_boxes"]), member["member_id"]))
            baseline_match = high_confidence_match(baseline["pixel_boxes"], canonical["pixel_boxes"], threshold)
            additions = [{"canonical_box_index": index, "class_id": canonical["pixel_boxes"][index][0],
                          "class_name": NAMES[canonical["pixel_boxes"][index][0]], "bbox_xyxy_pixels": canonical["pixel_boxes"][index][1:],
                          "relative_to_member": baseline["member_id"]} for index in baseline_match["unmatched"]]
            return {"group_id": group["group_id"], "members": [member["member_id"] for member in members],
                    "category": "SUPERSET_ANNOTATION", "canonical_member": canonical["member_id"],
                    "annotation_counts": {member["member_id"]: len(member["pixel_boxes"]) for member in members},
                    "matched_pairs": comparisons, "match_ious": [pair["iou"] for comparison in comparisons for pair in comparison["pairs"]],
                    "additional_boxes": additions, "reason": "One clean largest existing annotation set uniquely matches every other member at high same-class IoU; valid additional boxes are non-overlapping with matched objects. Keep that set only; never create a union.",
                    "confidence": "HIGH", "auto_resolvable": True,
                    "baseline_smaller_member": baseline["member_id"]}
    return unresolved(group, failures or ["no_single_existing_annotation_set_dominates"], best_diagnostics)


def render_member(member, extra_indices=None, highlighted=False):
    with Image.open(source_image(member)) as source:
        image = source.convert("RGB")
    image.thumbnail((450, 290))
    panel = Image.new("RGB", (470, 365), "white")
    panel.paste(image, (4, 64))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 2), member["member_id"], font=font(), fill="black")
    histogram = dict(zip(NAMES, bbox_counts(member["pixel_boxes"])))
    draw.text((4, 22), f"bbox={len(member['pixel_boxes'])} " + " ".join(f"{name}:{count}" for name, count in histogram.items() if count), font=font(12), fill="black")
    draw.text((4, 40), "COMMON: green | ADDITIONAL: orange" if extra_indices is not None else "Final classes labeled; original annotation geometry", font=font(12), fill="black")
    scale_x, scale_y = image.width / member["width"], image.height / member["height"]
    for index, box in enumerate(member["pixel_boxes"]):
        class_id, left, top, right, bottom = box
        is_extra = extra_indices is not None and index in extra_indices
        color = "orange" if is_extra else "lime"
        position = (left * scale_x + 4, top * scale_y + 64, right * scale_x + 4, bottom * scale_y + 64)
        draw.rectangle(position, outline=color, width=2)
        caption = NAMES[class_id] + (" ADD" if is_extra else "")
        caption_left = min(position[0], max(4, image.width - draw.textlength(caption, font=font(12))))
        draw.text((caption_left, max(64, position[1] - 14)), caption, font=font(12), fill=color, stroke_width=1, stroke_fill="black")
    return panel


def group_preview(group, result):
    members = {member["member_id"]: member for member in group["members"]}
    if result["canonical_member"]:
        canonical = members[result["canonical_member"]]
        if result["category"] == "SUPERSET_ANNOTATION":
            competing = members[result["baseline_smaller_member"]]
            extra_indices = {item["canonical_box_index"] for item in result["additional_boxes"]}
        else:
            competing = next(member for member in group["members"] if member["member_id"] != canonical["member_id"])
            extra_indices = None
    else:
        failed_comparison = next((item for item in result["matched_pairs"] if not item["accepted"]), None)
        if failed_comparison is not None:
            competing = members[failed_comparison["from_member"]]
            canonical = members[failed_comparison["to_member"]]
        else:
            ordered = sorted(group["members"], key=lambda member: (len(member["pixel_boxes"]), member["member_id"]))
            competing, canonical = ordered[0], ordered[-1]
        extra_indices = None
    cell = Image.new("RGB", (960, 470), "white")
    draw = ImageDraw.Draw(cell)
    label = result["category"]
    draw.text((6, 2), label + " | " + group["group_id"][:28], font=font(), fill="black")
    overlaps = result["match_ious"]
    information = f"matching IoU: min={min(overlaps):.3f}, median={float(np.median(overlaps)):.3f}" if overlaps else "No accepted high-confidence matching"
    draw.text((6, 22), information + f" | copies={len(group['members'])}", font=font(), fill="black")
    decision = f"Proposed canonical: {result['canonical_member']}" if result["canonical_member"] else "Decision: HUMAN REVIEW; canonical=null"
    draw.text((6, 42), decision, font=font(), fill="black")
    cell.paste(render_member(competing), (6, 65))
    cell.paste(render_member(canonical, extra_indices=extra_indices), (482, 65))
    draw.text((6, 432), "\n".join(textwrap.wrap(result["reason"], width=135)[:2]), font=font(12), fill="black")
    return cell


def category_qa(groups, results):
    indexed_groups = {group["group_id"]: group for group in groups}
    manifest = []
    limits = {"SAME_OBJECTS_JITTER": 50, "SUPERSET_ANNOTATION": 50, "TRUE_CONFLICT": 100}
    for category, limit in limits.items():
        selected = [result for result in results if result["category"] == category]
        random.Random(SEED).shuffle(selected)
        selected = selected[:limit]
        directory = OUT / "qa" / category.lower()
        directory.mkdir(parents=True, exist_ok=True)
        for page_start in range(0, len(selected), 10):
            page_results = selected[page_start:page_start + 10]
            sheet = Image.new("RGB", (1920, math.ceil(len(page_results) / 2) * 470), "white")
            for index, result in enumerate(page_results):
                image = group_preview(indexed_groups[result["group_id"]], result)
                name = f"{page_start + index + 1:03d}__{result['group_id'][7:23]}.jpg"
                individual = directory / name
                image.save(individual, quality=93)
                sheet.paste(image, ((index % 2) * 960, (index // 2) * 470))
                manifest.append({"group_id": result["group_id"], "category": category,
                                 "preview": individual.relative_to(OUT).as_posix(), "canonical_member": result["canonical_member"]})
            sheet.save(directory / f"contact_sheet_{page_start // 10 + 1:02d}.jpg", quality=93)
    save_json("qa/manifest.json", {"seed": SEED, "sample_counts": dict(Counter(item["category"] for item in manifest)),
                                  "groups": manifest, "full_annotation_members": "../duplicate_resolution_manifest.json"})
    index = ["# Work 4.7A — Human QA", "", "Review proposed resolutions without modifying Vehicle363-v1. TRUE_CONFLICT requires human judgment. Green = common boxes; orange = additional boxes in the proposed superset.", ""]
    for category in limits:
        index += [f"## {category}", ""]
        for item in manifest:
            if item["category"] == category:
                index.append(f"- [{item['group_id'][:30]}]({item['preview'].removeprefix('qa/')}) — canonical: {item['canonical_member'] or 'HUMAN REVIEW'}")
        index.append("")
    (OUT / "qa" / "README.md").write_text("\n".join(index), encoding="utf-8")
    return dict(Counter(item["category"] for item in manifest))


def group_statistics(groups, results):
    indexed = {group["group_id"]: group for group in groups}
    statistics_data = {}
    for category in ["SAME_OBJECTS_JITTER", "SUPERSET_ANNOTATION", "TRUE_CONFLICT"]:
        subset = [result for result in results if result["category"] == category]
        involved = [member for result in subset for member in indexed[result["group_id"]]["members"]]
        data = {"groups": len(subset), "percent_of_conflict_groups": 100 * len(subset) / len(results),
                "images": len(involved), "positive_target_images": sum(bool(member["pixel_boxes"]) for member in involved),
                "boxes_involved": sum(len(member["pixel_boxes"]) for member in involved)}
        if category == "SUPERSET_ANNOTATION":
            data["boxes_in_smaller_sets"] = sum(len(member["pixel_boxes"]) for result in subset for member in indexed[result["group_id"]]["members"]
                                                  if len(member["pixel_boxes"]) < max(result["annotation_counts"].values()))
            data["boxes_in_selected_canonical_sets"] = sum(result["annotation_counts"][result["canonical_member"]] for result in subset)
            data["additional_boxes_retained"] = sum(len(result["additional_boxes"]) for result in subset)
            data["additional_boxes_definition"] = "Unique additional canonical boxes relative to each group's smallest matched member, not an annotation union"
        statistics_data[category] = data
    return statistics_data


def simulate(inputs, groups, results):
    projected_images = inputs["statistics"]["images"]
    projected_counts = [inputs["statistics"]["class_counts"][name] for name in NAMES]
    removed_images = 0
    removed_boxes = [0, 0, 0, 0]
    canonical_boxes = 0
    split_removed = defaultdict(lambda: {"images": 0, "boxes": 0})
    indexed = {group["group_id"]: group for group in groups}
    for result in results:
        if not result["auto_resolvable"]:
            continue
        members = indexed[result["group_id"]]["members"]
        canonical = next(member for member in members if member["member_id"] == result["canonical_member"])
        if canonical["final_record"] is None:
            raise RuntimeError("Canonical candidate has no current positive target image")
        canonical_split = canonical["final_record"]["final_split"]
        canonical_boxes += len(canonical["pixel_boxes"])
        for member in members:
            if member is canonical or member["final_record"] is None:
                continue
            if member["final_record"]["final_split"] != canonical_split:
                raise RuntimeError("Canonical selection would span splits")
            removed_images += 1
            split_removed[canonical_split]["images"] += 1
            split_removed[canonical_split]["boxes"] += len(member["pixel_boxes"])
            for class_id, count in enumerate(bbox_counts(member["pixel_boxes"])):
                removed_boxes[class_id] += count
                projected_counts[class_id] -= count
    projected_images -= removed_images
    return {"current": {"images": inputs["statistics"]["images"], "boxes": inputs["statistics"]["boxes"], "class_counts": inputs["statistics"]["class_counts"]},
            "projected": {"images": projected_images, "boxes": sum(projected_counts), "class_counts": dict(zip(NAMES, projected_counts))},
            "images_removed_as_redundant_copies": removed_images, "boxes_added_relative_to_current_dataset": 0,
            "boxes_removed_with_noncanonical_copies": sum(removed_boxes), "removed_class_counts": dict(zip(NAMES, removed_boxes)),
            "boxes_in_retained_canonical_sets": canonical_boxes, "coordinates_averaged_or_unioned": False,
            "explanation": "All proposed canonical sets already exist in current Vehicle363-v1. Superset extras are retained, not newly added; only noncanonical duplicate image/label pairs would be removed.",
            "removed_per_split": dict(split_removed), "remaining_unresolved_groups": sum(not result["auto_resolvable"] for result in results)}


def output_report(inputs, results, supplement_results, threshold, qa_counts):
    stats = group_statistics(inputs["conflicts"], results)
    all_groups = inputs["conflicts"] + inputs["supplemental"]
    all_results = results + supplement_results
    main_projection = simulate(inputs, inputs["conflicts"], results)
    full_projection = simulate(inputs, all_groups, all_results)
    unchanged = all(digest_file(ROOT / path) == digest for path, digest in inputs["snapshot"].items())
    if not unchanged:
        raise RuntimeError("Input metadata or source annotation changed during this read-only audit")
    auto_groups = sum(result["auto_resolvable"] for result in results)
    true_groups = len(results) - auto_groups
    status = "REVIEW_REQUIRED" if any(not result["auto_resolvable"] for result in all_results) else "PASS"
    safety = {**inputs["split_safety"], "canonical_members_preserve_original_split": True,
              "proposed_cross_split_duplicate_leakage": 0, "proposed_group_overlap": 0, "proposed_source_leakage": 0,
              "split_changes_required": False, "scope": "Selection only removes within-group duplicate copies, retaining the existing group/split/source identities",
              "dataset_metadata_unchanged_sha256": unchanged, "vehicle363_modified": False,
              "source_datasets_modified": False, "source_image_rescan_performed": False}
    report = {"work": "4.7A", "status": status, "total_conflict_groups": len(results),
              "HIGH_CONFIDENCE_IOU": threshold, "category_statistics": stats,
              "auto_resolvable_conflict_groups": auto_groups, "human_review_required_conflict_groups": true_groups,
              "primary_conflicts_only_simulation": main_projection,
              "supplemental_existing_compatible_subset_groups": {"count": len(supplement_results), "results": supplement_results,
                                                               "why_included": "One previously reported compatible-subset group was kept separate from the 9,581 conflict groups; apply the same conservative tests and report separately."},
              "projected_after_all_high_confidence_recommendations": full_projection,
              "simulation_scope": "Main conflict groups plus explicitly separated compatible-subset supplement; no resolution applied",
              "split_leakage_impact": safety, "qa_samples": qa_counts,
              "calibration": "iou_calibration.json", "resolution_manifest": "duplicate_resolution_manifest.json",
              "reason_counts_for_true_conflicts": dict(Counter(result["reason"] for result in results if result["category"] == "TRUE_CONFLICT")),
              "confidence_interpretation": "HIGH is automatic resolution confidence under the documented conservative rules, not an empirically measured semantic accuracy. LOW unresolved entries have no canonical member.",
              "important_limitations": ["Calibration strata are geometric proxies rather than human-labeled truth.",
                                        "Tiny-object jitter can fall below 0.90 despite identical objects and is intentionally retained for human review.",
                                        "The 0.50 suspicious-overlap guard can flag legitimate occlusion; such groups are not auto-resolved.",
                                        "Completeness means geometric same-class dominance of existing target boxes, not verification of every object's true semantic identity.",
                                        "No annotation unions, coordinate averaging, or split reassignment are proposed.",
                                        "Existing unrelated near-duplicate candidates from Work 4.7 remain outside this exact-duplicate resolution audit."],
              "next": "HUMAN REVIEW OF TRUE_CONFLICT SAMPLE", "resolution_applied": False, "trained_yolo": False}
    save_json("work_4_7a_report.json", report)
    projection = full_projection["projected"]
    text = ["# Work 4.7A — Duplicate annotation conflict resolution", "", f"**Status: {status}. No resolution has been applied.**", "",
            f"HIGH_CONFIDENCE_IOU: **{threshold:.2f}**. Recalculated conflict groups: **{len(results):,}**.", "",
            "## Classification", "", "| Category | Groups | Percent | Member images | Boxes involved |", "|---|---:|---:|---:|---:|"]
    for category, data in stats.items():
        text.append(f"| {category} | {data['groups']:,} | {data['percent_of_conflict_groups']:.2f}% | {data['images']:,} | {data['boxes_involved']:,} |")
    text += ["", f"Auto-resolvable primary groups: **{auto_groups:,}**. Human-review primary groups: **{true_groups:,}**.", "",
             "## Conservative resolution rules", "", "Boxes are converted to pixels using previously decoded dimensions. Matching uses only the final target class. A complete maximum-cardinality same-class IoU>=0.90 graph match must be unique; alternate complete matchings are rejected. Jitter groups require successful unique matching between every member pair. Canonical jitter selection prefers valid, non-suspicious geometry, then source filename/ID ordering.", "",
             "A superset candidate must be one existing largest annotation set that matches every other set, including intermediate members in a chain. Extras must be inside the image with positive area, and neither duplicates nor cross-class suspicious overlaps. IoU>=0.50 internal overlaps or overlaps between extras and already annotated objects block automatic resolution. Box count alone never determines the winner. No boxes are averaged or unioned.", "",
             "## Superset completeness statistics", ""]
    superset = stats["SUPERSET_ANNOTATION"]
    text += [f"- Boxes in smaller sets: {superset['boxes_in_smaller_sets']:,} (all strictly smaller members).",
             f"- Boxes in selected canonical sets: {superset['boxes_in_selected_canonical_sets']:,}.",
             f"- Additional boxes retained: {superset['additional_boxes_retained']:,}, relative to each group's smallest matched member.", "",
             "## Supplemental compatible group", "", f"The existing report also contains {len(supplement_results)} compatible-subset group outside the {len(results):,} conflict groups. It is classified separately in the resolution manifest; main category totals above exclude it. The combined projection below includes every HIGH-confidence primary and supplemental recommendation. Primary-only projections are also recorded in JSON.", "",
             "## Count-only simulation", "", "| Dataset state | Images | Boxes | Car | Motorcycle | Bus | Truck |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, data in [("Current Vehicle363-v1", full_projection["current"]), ("Projected after HIGH recommendations", projection)]:
        text.append(f"| {label} | {data['images']:,} | {data['boxes']:,} | " + " | ".join(f"{data['class_counts'][name]:,}" for name in NAMES) + " |")
    text += ["", f"Boxes added relative to current dataset: **0**. Boxes removed with noncanonical copies: **{full_projection['boxes_removed_with_noncanonical_copies']:,}**. Retained canonical coordinates are unchanged.", "", full_projection["explanation"], "",
             "## Split/leakage safety", "", "Split changes required: **NO**. Exact duplicate hashes and atomic groups are already contained in one split. Canonical selection retains each source record's existing split and atomic group and removes only other copies in that same group. Existing and projected cross-split hash/group overlaps: **0**. Source identities remain intact.", "",
             "## Human QA", "", "See [QA index](qa/README.md), [calibration crops](qa/calibration/matched_box_crops.jpg), and `qa/manifest.json`. Superset previews distinguish COMMON (green) and ADDITIONAL (orange). Competing TRUE_CONFLICT annotations are shown side by side. Only the displayed pair is shown for multi-copy QA; complete member/pair comparisons are preserved in the manifest.", "",
             "## Limitations", ""] + [f"- {item}" for item in report["important_limitations"]]
    text += ["", "## Next", "", "**HUMAN REVIEW OF TRUE_CONFLICT SAMPLE. DO NOT APPLY THE RESOLUTION YET.**", "",
             "Vehicle363-v1 and both source datasets remain unchanged. No Work 4.7 rebuild or whole-source image rescan was performed. Input metadata and local source annotation SHA-256 values were unchanged after this audit.", ""]
    (OUT / "work_4_7a_report.md").write_text("\n".join(text), encoding="utf-8")
    lines = ["WORK 4.7A STATUS:", status, "", "TOTAL CONFLICT GROUPS:", str(len(results)), ""]
    lines += [f"{category}: {data['groups']} ({data['percent_of_conflict_groups']:.2f}%)" for category, data in stats.items()]
    lines += ["", f"AUTO-RESOLVABLE: {auto_groups}", f"HUMAN REVIEW REQUIRED: {true_groups}",
              f"HIGH_CONFIDENCE_IOU: {threshold:.2f}", "", "CURRENT Vehicle363-v1:",
              f"images: {full_projection['current']['images']}", f"boxes: {full_projection['current']['boxes']}", "",
              "PROJECTED AFTER RESOLUTION:", f"images: {projection['images']}", f"boxes: {projection['boxes']}"]
    lines += [f"{name}: {projection['class_counts'][name]}" for name in NAMES]
    lines += ["", f"Supplemental compatible groups: {len(supplement_results)} (separate from primary conflict counts)",
              "SPLIT/LEAKAGE IMPACT:", "Split changes required: NO; projected cross-split exact hash/group/source leakage: 0.", "",
              "NEXT:", "HUMAN REVIEW OF TRUE_CONFLICT SAMPLE", "", "DO NOT APPLY THE RESOLUTION YET.", ""]
    console = "\n".join(lines)
    (OUT / "WORK_4_7A_SUMMARY.txt").write_text(console, encoding="utf-8")
    print(console, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    calibration = calibrate(inputs)
    calibration_qa(inputs, calibration)
    if args.calibrate_only:
        return
    threshold = select_threshold(calibration)
    results = []
    for index, group in enumerate(inputs["conflicts"], 1):
        result = classify_group(group, threshold)
        result["scope"] = "PRIMARY_ANNOTATION_CONFLICT"
        results.append(result)
        if index % 2000 == 0:
            print(json.dumps({"classified": index, "total": len(inputs["conflicts"]), "categories": dict(Counter(item["category"] for item in results))}), flush=True)
    supplemental = [classify_group(group, threshold) for group in inputs["supplemental"]]
    for result in supplemental:
        result["scope"] = "SUPPLEMENTAL_PREVIOUS_COMPATIBLE_SUBSET"
    save_json("duplicate_resolution_manifest.json", {"work": "4.7A", "HIGH_CONFIDENCE_IOU": threshold,
              "bbox_indexing": "Zero-based index in the target_annotations arrays of Work 4.7 duplicate_report.json",
              "source_annotations_reconciled_with_gt_train_csv": inputs["source_annotations_reconciled"],
              "source_image_rescan_performed": False, "resolution_applied": False,
              "groups": results, "supplemental_groups": supplemental})
    qa_counts = category_qa(inputs["conflicts"], results)
    output_report(inputs, results, supplemental, threshold, qa_counts)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_json("work_4_7a_report.json", {"work": "4.7A", "status": "FAIL", "processing_complete": False,
                  "error": f"{type(error).__name__}: {error}", "resolution_applied": False, "vehicle363_modified": False})
        print(f"WORK 4.7A STATUS: FAIL\n{type(error).__name__}: {error}", flush=True)
        raise
