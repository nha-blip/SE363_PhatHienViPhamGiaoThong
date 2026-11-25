import csv
import hashlib
import io
import json
import math
import os
import pickle
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "Vehicle363-v1"
CACHE = ROOT / "data" / "vehicle_audit" / "vehicle363_build"
SOURCE_ROOTS = {
    "HCMC": ROOT / "data" / "archive (1)",
    "MIO": ROOT / "data" / "MIO-TCD-Localization" / "MIO-TCD-Localization",
}
NAMES = ["car", "motorcycle", "bus", "truck"]
HCMC_MAP = {0: 1, 1: 0, 2: 2, 3: 3, 4: 1, 5: 0, 6: 2, 7: 3}
MIO_MAP = {"car": 0, "motorcycle": 1, "bus": 2, "articulated_truck": 3,
           "single_unit_truck": 3, "pickup_truck": 3, "work_van": 3}
RATIOS = {"train": 0.75, "val": 0.125, "test": 0.125}
EXPECTED = {"hcmc_images": 49270, "hcmc_boxes": 366921,
            "hcmc_classes": [95806, 235493, 11187, 24435],
            "mio_images": 137743, "mio_source_annotations": 351549,
            "mio_mapped_classes": [233497, 1837, 10598, 68034], "mio_excluded": 37583}
COSINE = np.cos(np.pi * np.arange(8)[:, None] * (np.arange(32)[None, :] + 0.5) / 32)
POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
START = time.monotonic()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def progress(stage, **details):
    message = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": round(time.monotonic() - START, 1), **details}
    print(json.dumps(message, ensure_ascii=False), flush=True)
    json_write(CACHE / "progress.json", message)


def checkpoint(name, value=None):
    path = CACHE / (name + ".pkl")
    if value is None:
        if path.exists():
            with path.open("rb") as handle:
                return pickle.load(handle)
        return None
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def counts(labels):
    result = [0, 0, 0, 0]
    for label in labels:
        result[label[0]] += 1
    return result


def summed_counts(records):
    result = [0, 0, 0, 0]
    for record in records:
        for class_id, count in enumerate(counts(record["labels"])):
            result[class_id] += count
    return result


def named(values):
    return dict(zip(NAMES, values))


def safe_name(source, relative):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", relative).strip("_")[:150]
    return f"{source.lower()}__{slug}__{hashlib.sha1(relative.encode()).hexdigest()[:12]}.jpg"


def inspect_image(item):
    source, relative, size, modified = item
    path = SOURCE_ROOTS[source] / relative
    result = {"source": source, "relative": relative, "file_size": size, "modified_ns": modified}
    try:
        content = path.read_bytes()
        result["sha256"] = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            result["width"], result["height"] = image.size
            gray = image.convert("L")
            values = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
            frequencies = COSINE @ values @ COSINE.T
            threshold = np.median(frequencies.flatten()[1:])
            bits = (frequencies > threshold).flatten()
            result["phash"] = int.from_bytes(np.packbits(bits).tobytes(), "big")
            differences = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
            result["dhash"] = int.from_bytes(np.packbits(differences[:, 1:] > differences[:, :-1]).tobytes(), "big")
        result["error"] = None
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def scan_sources():
    previous = checkpoint("image_scan") or {}
    work = []
    inventory = {}
    for source, directory in SOURCE_ROOTS.items():
        if not directory.is_dir():
            raise RuntimeError(f"Source root not found: {directory}")
        for path in sorted(directory.rglob("*.jpg")):
            stat = path.stat()
            relative = path.relative_to(directory).as_posix()
            key = (source, relative)
            cached = previous.get(key)
            if cached and cached["file_size"] == stat.st_size and cached["modified_ns"] == stat.st_mtime_ns:
                inventory[key] = cached
            else:
                work.append((source, relative, stat.st_size, stat.st_mtime_ns))
    progress("scan_images", cached=len(inventory), pending=len(work), free_bytes=shutil.disk_usage(OUT.parent).free)
    with ThreadPoolExecutor(max_workers=12) as executor:
        for index, result in enumerate(executor.map(inspect_image, work), 1):
            inventory[(result["source"], result["relative"])] = result
            if index % 5000 == 0:
                progress("scan_images", completed=index, pending=len(work) - index)
            if index % 20000 == 0:
                checkpoint("image_scan", inventory)
    checkpoint("image_scan", inventory)
    unreadable = [result for result in inventory.values() if result["error"]]
    if unreadable:
        json_write(OUT / "metadata" / "unreadable_source_images.json", unreadable)
        raise RuntimeError(f"Unreadable source images: {len(unreadable)}; review before building")
    return inventory


def hcmc_stream(relative):
    name = Path(relative).stem
    name = re.sub(r"_jpg\.rf\..*$", "", name, flags=re.I).rstrip("_-")
    for pattern in [r"^(cam[_-]?\d+)[_-](\d+)$", r"^(src[_-]?\d+)[_-]frame[_-](\d+)$", r"^(.+?[_-]mp4)[_-](\d+)$"]:
        match = re.fullmatch(pattern, name, flags=re.I)
        if match:
            return match.group(1).lower(), int(match.group(2))
    raise RuntimeError(f"Cannot reliably group HCMC filename: {relative}")


def parse_hcmc_item(item):
    relative, info = item
    label_path = (SOURCE_ROOTS["HCMC"] / relative).with_suffix(".txt")
    if not label_path.exists():
        raise RuntimeError(f"Missing HCMC annotation: {label_path}")
    labels, excluded = [], []
    raw_counts = [0, 0, 0, 0]
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"Malformed HCMC row: {relative}:{line_number}")
        source_id = int(fields[0])
        target = HCMC_MAP[source_id]
        raw_counts[target] += 1
        values = [Decimal(value) for value in fields[1:]]
        x_center, y_center, width, height = values
        valid = all(value.is_finite() for value in values)
        valid = valid and 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1
        valid = valid and x_center - width / 2 >= 0 and x_center + width / 2 <= 1 and y_center - height / 2 >= 0 and y_center + height / 2 <= 1
        if not valid:
            excluded.append({"source": "HCMC", "image": relative, "source_class": source_id,
                             "bbox": fields[1:], "reason": "invalid_source_yolo_geometry", "source_line": line_number})
        else:
            labels.append((target, *[float(value) for value in values], str(source_id)))
    stream, frame = hcmc_stream(relative)
    record = {"source": "HCMC", "relative": relative, "partition": "daytime" if relative.startswith("daytime-") else "nighttime",
              "stream": stream, "frame": frame, "labels": labels, "info": info,
              "source_group": f"hcmc_stream__{stream}", "final_image": safe_name("HCMC", relative)}
    return record, raw_counts, excluded


def parse_annotations(inventory):
    progress("parse_annotations")
    excluded = []
    records = []
    raw_hcmc = [0, 0, 0, 0]
    hcmc_items = [(relative, info) for (source, relative), info in inventory.items() if source == "HCMC"]
    with ThreadPoolExecutor(max_workers=12) as executor:
        for record, source_counts, bad_rows in executor.map(parse_hcmc_item, hcmc_items):
            records.append(record)
            excluded.extend(bad_rows)
            for class_id, count in enumerate(source_counts):
                raw_hcmc[class_id] += count
    mio_by_image = {}
    raw_mio = Counter()
    with (SOURCE_ROOTS["MIO"] / "gt_train.csv").open(newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            image_id, source_class, *coordinates = row
            relative = f"train/{image_id}.jpg"
            info = inventory.get(("MIO", relative))
            if info is None:
                raise RuntimeError(f"Missing annotated MIO image: {relative}")
            raw_mio[source_class] += 1
            bbox = [float(value) for value in coordinates]
            if len(bbox) != 4:
                raise RuntimeError(f"Malformed MIO row {line_number}")
            record = mio_by_image.setdefault(relative, {"source": "MIO", "relative": relative, "partition": "train", "original_image_id": image_id,
                                                        "labels": [], "info": info, "final_image": safe_name("MIO", relative)})
            if source_class not in MIO_MAP:
                excluded.append({"source": "MIO", "image": relative, "source_class": source_class, "bbox": bbox,
                                 "reason": "non_target_policy_exclusion", "source_line": line_number})
                continue
            left, top, right, bottom = bbox
            if not all(math.isfinite(value) for value in bbox) or not (0 <= left < right <= info["width"] and 0 <= top < bottom <= info["height"]):
                excluded.append({"source": "MIO", "image": relative, "source_class": source_class, "bbox": bbox,
                                 "reason": "invalid_source_pixel_geometry", "source_line": line_number})
                continue
            record["labels"].append((MIO_MAP[source_class], (left + right) / (2 * info["width"]),
                                     (top + bottom) / (2 * info["height"]), (right - left) / info["width"],
                                     (bottom - top) / info["height"], source_class))
    records.extend(mio_by_image.values())
    raw_mapped = [sum(count for source, count in raw_mio.items() if MIO_MAP.get(source) == class_id) for class_id in range(4)]
    actual = {"hcmc_images": len(hcmc_items), "hcmc_boxes": sum(raw_hcmc), "hcmc_classes": raw_hcmc,
              "mio_images": sum(source == "MIO" for source, _ in inventory), "mio_source_annotations": sum(raw_mio.values()),
              "mio_mapped_classes": raw_mapped, "mio_excluded": sum(count for source, count in raw_mio.items() if source not in MIO_MAP)}
    validation = {"expected": EXPECTED, "actual": actual, "matches_expectations": actual == EXPECTED,
                  "mio_class_counts_raw": dict(raw_mio), "sources": {key: str(value) for key, value in SOURCE_ROOTS.items()}}
    json_write(OUT / "metadata" / "input_validation.json", validation)
    if actual != EXPECTED:
        raise RuntimeError(f"Material source count differences: {actual}")
    valid_pre_dedupe = {source: named(summed_counts([record for record in records if record["source"] == source])) for source in SOURCE_ROOTS}
    validation["valid_target_boxes_before_deduplication"] = valid_pre_dedupe
    json_write(OUT / "metadata" / "input_validation.json", validation)
    checkpoint("parsed_annotations", (records, excluded, validation))
    return records, excluded, validation


class UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            next_key = self.parent[key]
            self.parent[key] = root
            key = next_key
        return root

    def union(self, first, second):
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def signature(record):
    return tuple(sorted((label[0], *[round(value, 12) for value in label[1:5]]) for label in record["labels"]))


def deduplicate(records, inventory):
    progress("duplicate_annotation_comparison")
    hash_members = defaultdict(list)
    record_by_key = {(record["source"], record["relative"]): record for record in records}
    for key, info in inventory.items():
        hash_members[info["sha256"]].append(key)
    report = {"method": "Recalculated SHA-256 over all HCMC and all MIO train/test images", "groups": [],
              "hcmc_duplicate_groups": 0, "mio_duplicate_groups": 0, "cross_source_exact_duplicates": 0,
              "removed_canonical_duplicates": 0, "removed_equivalent_bbox_copies": [0, 0, 0, 0],
              "annotation_conflicts": [], "compatible_annotation_groups_requiring_review": [], "annotation_union_performed": False,
              "equivalence_rule": "Sorted multisets of target class and normalized coordinates rounded to 12 decimal places; multiplicity preserved"}
    kept = []
    for digest, members in sorted(hash_members.items()):
        group_id = "exact__" + digest
        source_counts = Counter(source for source, _ in members)
        report["hcmc_duplicate_groups"] += source_counts["HCMC"] > 1
        report["mio_duplicate_groups"] += source_counts["MIO"] > 1
        report["cross_source_exact_duplicates"] += bool(source_counts["HCMC"] and source_counts["MIO"])
        annotated = [record_by_key[key] for key in members if key in record_by_key]
        target_members = [record for record in annotated if record["labels"]]
        signatures = {signature(record) for record in annotated}
        disagreement = len(signatures) > 1
        ordered_signatures = sorted(signatures, key=len)
        compatible_subset = disagreement and bool(ordered_signatures[0]) and all(
            not (Counter(earlier) - Counter(later)) for earlier, later in zip(ordered_signatures, ordered_signatures[1:]))
        decision = "NO_TARGET_OBJECTS"
        canonical = None
        if target_members:
            if disagreement:
                decision = "COMPATIBLE_SUBSET_REVIEW_BEFORE_UNION" if compatible_subset else "DUPLICATE_ANNOTATION_CONFLICT"
                review_key = "compatible_annotation_groups_requiring_review" if compatible_subset else "annotation_conflicts"
                report[review_key].append({"group_id": group_id,
                    "members": [{"source": record["source"], "image": record["relative"], "target_annotations": record["labels"]} for record in annotated]})
                selected = target_members
            else:
                decision = "KEEP_CANONICAL_EQUIVALENT_ANNOTATIONS"
                canonical = min(target_members, key=lambda record: (record["source"], record["relative"]))
                selected = [canonical]
                report["removed_canonical_duplicates"] += len(target_members) - 1
                for class_id, count in enumerate(counts(canonical["labels"])):
                    report["removed_equivalent_bbox_copies"][class_id] += count * (len(target_members) - 1)
            for record in selected:
                record["duplicate_group"] = group_id
                record["source_group"] = record.get("source_group", group_id)
                record["source_aliases"] = [{"source_dataset": source, "source_image": relative,
                                              "annotation_available": (source, relative) in record_by_key} for source, relative in members]
                kept.append(record)
        if len(members) > 1:
            report["groups"].append({"group_id": group_id, "sha256": digest,
                                     "members": [{"source": source, "image": relative, "annotation_available": (source, relative) in record_by_key} for source, relative in members],
                                     "decision": decision, "canonical": f"{canonical['source']}:{canonical['relative']}" if canonical else None})
    json_write(OUT / "metadata" / "duplicate_report.json", report)
    return kept, report


def create_groups(records):
    initial = {record["source_group"] for record in records}
    union = UnionFind(initial)
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["info"]["sha256"]].append(record)
        for alias in record["source_aliases"]:
            if alias["source_dataset"] == "HCMC":
                stream, _ = hcmc_stream(alias["source_image"])
                alias_group = f"hcmc_stream__{stream}"
                union.parent.setdefault(alias_group, alias_group)
                union.union(record["source_group"], alias_group)
    for members in by_hash.values():
        for member in members[1:]:
            union.union(members[0]["source_group"], member["source_group"])
    groups = defaultdict(list)
    for record in records:
        group_id = union.find(record["source_group"])
        record["group_id"] = group_id
        groups[group_id].append(record)
    return groups


def split_groups(groups):
    sources = defaultdict(list)
    for group_id, members in groups.items():
        sources[tuple(sorted({record["source"] for record in members}))].append((group_id, members))
    for source_groups in sources.values():
        total = np.array([sum(len(members) for _, members in source_groups),
                          *summed_counts([record for _, members in source_groups for record in members])], dtype=float)
        target = {split: total * ratio for split, ratio in RATIOS.items()}
        running = {split: np.zeros(5) for split in RATIOS}
        random.Random(363).shuffle(source_groups)
        source_groups.sort(key=lambda item: max(np.array([len(item[1]), *summed_counts(item[1])]) / np.maximum(total, 1)), reverse=True)
        weights = np.array([2, 1, 1, 3, 1])
        for group_id, members in source_groups:
            addition = np.array([len(members), *summed_counts(members)])
            scores = {}
            for split in RATIOS:
                before = np.sum(weights * ((running[split] - target[split]) / np.maximum(total, 1)) ** 2)
                after = np.sum(weights * ((running[split] + addition - target[split]) / np.maximum(total, 1)) ** 2)
                scores[split] = after - before
            chosen = min(scores, key=lambda split: (scores[split], list(RATIOS).index(split)))
            running[chosen] += addition
            for record in members:
                record["split"] = chosen


def near_duplicate_check(records):
    progress("perceptual_cross_split_check")
    samples = []
    for source in SOURCE_ROOTS:
        for split in RATIOS:
            candidates = sorted((record for record in records if record["source"] == source and record["split"] == split), key=lambda record: record["final_image"])
            random.Random(363).shuffle(candidates)
            samples.extend(candidates[:1500])
    hashes = np.array([record["info"]["phash"] for record in samples], dtype=np.uint64)
    dhashes = np.array([record["info"]["dhash"] for record in samples], dtype=np.uint64)
    splits = np.array([list(RATIOS).index(record["split"]) for record in samples])
    suspects = []
    for start in range(0, len(samples), 128):
        stop = min(start + 128, len(samples))
        xor = hashes[start:stop, None] ^ hashes[None, :]
        distance = POPCOUNT[xor.view(np.uint8).reshape(stop - start, len(samples), 8)].sum(axis=2)
        mask = (distance <= 4) & (splits[start:stop, None] != splits[None, :])
        first_indices, second_indices = np.where(mask)
        for offset, second_index in zip(first_indices.tolist(), second_indices.tolist()):
            first_index = start + offset
            if first_index >= second_index or (int(dhashes[first_index]) ^ int(dhashes[second_index])).bit_count() > 4:
                continue
            first, second = samples[first_index], samples[second_index]
            suspects.append({"first_image": first["final_image"], "second_image": second["final_image"],
                             "first_split": first["split"], "second_split": second["split"],
                             "first_group": first["group_id"], "second_group": second["group_id"],
                             "phash_distance": int(distance[offset, second_index]),
                             "dhash_distance": (int(dhashes[first_index]) ^ int(dhashes[second_index])).bit_count(),
                             "first_source": first["source"], "second_source": second["source"]})
    report = {"method": "Deterministic whole-frame 64-bit DCT pHash and dHash; both Hamming distances <=4",
              "scope": "Up to 1,500 images per source per split, seed 363; all cross-split pairs within sampled pool including cross-source",
              "sample_images": len(samples), "sample_per_source_split": dict(Counter(f"{record['source']}:{record['split']}" for record in samples)),
              "exhaustive": False, "candidate_count": len(suspects), "candidates": suspects,
              "limitations": "Perceptual similarity is a review signal, not proof of identical frames. No claim of exhaustive near-duplicate absence."}
    json_write(OUT / "metadata" / "near_duplicate_report.json", report)
    return report


def label_text(record):
    return "".join(f"{label[0]} " + " ".join(format(value, ".17g") for value in label[1:5]) + "\n" for label in record["labels"])


def migrate_outputs(records, all_records):
    progress("write_normalized_outputs", images=len(records))
    original_names = {record["final_image"] for record in all_records}
    old_images = defaultdict(list)
    old_labels = defaultdict(list)
    for split in RATIOS:
        for kind in ["images", "labels"]:
            directory = OUT / kind / split
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.iterdir():
                if not path.is_file():
                    raise RuntimeError(f"Unexpected output directory: {path}")
                name = path.name if kind == "images" else path.stem + ".jpg"
                if name not in original_names:
                    raise RuntimeError(f"Refusing to modify unrecognized existing file: {path}")
                (old_images if kind == "images" else old_labels)[name].append(path)
    desired = {record["final_image"] for record in records}
    migration = {"images_moved": 0, "labels_written": 0, "redundant_generated_files_removed": 0,
                 "source_files_modified": False, "physical_policy": "One generated image pathname per canonical pair; same-volume hardlinks when available, otherwise byte copies"}
    for index, record in enumerate(records, 1):
        image_path = OUT / "images" / record["split"] / record["final_image"]
        existing = old_images.get(record["final_image"], [])
        if not image_path.exists():
            if existing:
                existing[0].replace(image_path)
                migration["images_moved"] += 1
            else:
                source_path = SOURCE_ROOTS[record["source"]] / record["relative"]
                try:
                    os.link(source_path, image_path)
                except OSError:
                    shutil.copyfile(source_path, image_path)
        for old_path in existing:
            if old_path != image_path and old_path.exists():
                old_path.unlink()
                migration["redundant_generated_files_removed"] += 1
        label_path = OUT / "labels" / record["split"] / (Path(record["final_image"]).stem + ".txt")
        label_path.write_text(label_text(record), encoding="utf-8")
        migration["labels_written"] += 1
        for old_path in old_labels.get(record["final_image"], []):
            if old_path != label_path and old_path.exists():
                old_path.unlink()
        if index % 10000 == 0:
            progress("write_normalized_outputs", completed=index, total=len(records))
    for name in set(old_images) | set(old_labels):
        if name not in desired:
            for path in old_images.get(name, []) + old_labels.get(name, []):
                if path.exists():
                    if OUT.resolve() not in path.resolve().parents:
                        raise RuntimeError(f"Unsafe generated-file target: {path}")
                    path.unlink()
                    migration["redundant_generated_files_removed"] += 1
    json_write(OUT / "metadata" / "resume_migration.json", migration)


def validate_record(record):
    result = Counter()
    image_path = OUT / "images" / record["split"] / record["final_image"]
    label_path = OUT / "labels" / record["split"] / (Path(record["final_image"]).stem + ".txt")
    if not image_path.exists():
        result["missing_images"] += 1
    else:
        try:
            content = image_path.read_bytes()
            if hashlib.sha256(content).hexdigest() != record["info"]["sha256"]:
                result["image_hash_mismatches"] += 1
            with Image.open(io.BytesIO(content)) as image:
                image.load()
        except Exception:
            result["unreadable_images"] += 1
    if not label_path.exists():
        result["missing_labels"] += 1
    else:
        text = label_path.read_text(encoding="utf-8")
        if text != label_text(record):
            result["annotation_mismatches"] += 1
        for line in text.splitlines():
            try:
                fields = line.split()
                if len(fields) != 5 or fields[0] not in {"0", "1", "2", "3"}:
                    raise ValueError("Invalid row")
                x_center, y_center, width, height = [float(value) for value in fields[1:]]
                valid = all(math.isfinite(value) for value in [x_center, y_center, width, height])
                valid = valid and 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1
                valid = valid and x_center - width / 2 >= -1e-12 and y_center - height / 2 >= -1e-12 and x_center + width / 2 <= 1 + 1e-12 and y_center + height / 2 <= 1 + 1e-12
                if not valid:
                    result["invalid_boxes"] += 1
            except ValueError:
                result["invalid_yolo_rows"] += 1
    return result


def validate_outputs(records):
    progress("validate_every_output", total=len(records))
    totals = Counter({key: 0 for key in ["invalid_yolo_rows", "invalid_boxes", "missing_images", "missing_labels", "unreadable_images",
                                      "image_hash_mismatches", "annotation_mismatches", "unexpected_images", "unexpected_labels"]})
    with ThreadPoolExecutor(max_workers=12) as executor:
        for index, result in enumerate(executor.map(validate_record, records), 1):
            totals.update(result)
            if index % 10000 == 0:
                progress("validate_every_output", completed=index, total=len(records))
    desired = {record["final_image"] for record in records}
    for split in RATIOS:
        totals["unexpected_images"] += sum(path.name not in desired for path in (OUT / "images" / split).iterdir())
        totals["unexpected_labels"] += sum(path.stem + ".jpg" not in desired for path in (OUT / "labels" / split).iterdir())
    kinds = {"image_overlap": lambda record: record["final_image"], "atomic_group_overlap": lambda record: record["group_id"],
             "exact_hash_cross_split_overlap": lambda record: record["info"]["sha256"],
             "hcmc_sequence_overlap": lambda record: record.get("stream") if record["source"] == "HCMC" else None,
             "mio_duplicate_overlap": lambda record: record["duplicate_group"] if record["source"] == "MIO" else None}
    for name, getter in kinds.items():
        mapping = defaultdict(set)
        for record in records:
            key = getter(record)
            if key is not None:
                mapping[key].add(record["split"])
        totals[name] = sum(len(splits) > 1 for splits in mapping.values())
    result = {**dict(totals), "generated_boundary_floating_point_tolerance": 1e-12,
              "source_boundaries_checked_strictly": True, "all_final_image_bytes_sha256_revalidated": True,
              "train_val_image_intersection": 0, "train_test_image_intersection": 0, "val_test_image_intersection": 0}
    split_sets = {split: {path.name for path in (OUT / "images" / split).iterdir()} for split in RATIOS}
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        result[f"{first}_{second}_image_intersection"] = len(split_sets[first] & split_sets[second])
    json_write(OUT / "metadata" / "output_validation.json", result)
    return result


def subset_statistics(records):
    return {"images": len(records), "boxes": sum(len(record["labels"]) for record in records), "class_counts": named(summed_counts(records)),
            "source_counts": {source: {"images": sum(record["source"] == source for record in records),
                                      "boxes": sum(len(record["labels"]) for record in records if record["source"] == source)} for source in SOURCE_ROOTS}}


def size_statistics(records):
    values = defaultdict(list)
    for record in records:
        for label in record["labels"]:
            width, height = label[3] * record["info"]["width"], label[4] * record["info"]["height"]
            for key in ["global", f"class:{NAMES[label[0]]}", f"source:{record['source']}"]:
                values[key].append((width, height, width * height))
    result = {}
    for key, items in values.items():
        data = np.asarray(items)
        result[key] = {"count": len(items), "units": "pixels / square pixels",
                       **{metric: {"min": float(data[:, column].min()), "median": float(np.median(data[:, column])),
                                   "p25": float(np.percentile(data[:, column], 25)), "p75": float(np.percentile(data[:, column], 75)),
                                   "max": float(data[:, column].max())} for column, metric in enumerate(["width", "height", "area"])}}
    return result


def write_metadata(records, groups, excluded, duplicates, inputs, near):
    progress("write_manifests_and_statistics")
    class_mapping = {"schema": dict(enumerate(NAMES)), "hcmc_source_to_target": HCMC_MAP,
                     "mio_source_policies": [{"source_class": source, "target_class_id": target,
                                              "target_class": NAMES[target], "policy": "MERGE_MAP" if source in {"pickup_truck", "work_van"} else "DIRECT_MAP",
                                              "confidence": "MEDIUM" if source == "work_van" else "HIGH"} for source, target in MIO_MAP.items()] +
                                            [{"source_class": source, "target_class_id": None, "target_class": None, "policy": "EXCLUDE", "confidence": "HIGH"}
                                             for source in ["motorized_vehicle", "bicycle", "non-motorized_vehicle", "pedestrian"]]}
    json_write(OUT / "metadata" / "class_mapping.json", class_mapping)
    manifest = [{"final_image": f"images/{record['split']}/{record['final_image']}", "source_dataset": record["source"],
                 "source_image": record["relative"], "source_partition": record["partition"], "source_group": record["source_group"],
                 "atomic_group": record["group_id"], "duplicate_group": record["duplicate_group"], "final_split": record["split"],
                 "target_bbox_count": len(record["labels"]), "target_class_counts": named(counts(record["labels"])),
                 "daytime_nighttime": record["partition"] if record["source"] == "HCMC" else None,
                 "camera_id": record.get("stream"), "frame_id": record.get("frame"), "original_image_id": record.get("original_image_id"),
                 "original_train_test_status": record["partition"] if record["source"] == "MIO" else None,
                 "sha256": record["info"]["sha256"], "source_aliases": record["source_aliases"]} for record in records]
    json_write(OUT / "metadata" / "source_manifest.json", manifest)
    group_manifest = []
    for group_id, members in sorted(groups.items()):
        sources = sorted({record["source"] for record in members})
        group_type = "HCMC_CAMERA_SEQUENCE" if "HCMC" in sources else "MIO_EXACT_DUPLICATE_GROUP" if any(len(record["source_aliases"]) > 1 for record in members) else "MIO_SINGLETON"
        group_manifest.append({"group_id": group_id, "source_dataset": "+".join(sources), "group_type": group_type,
                               "image_count": len(members), "bbox_count": sum(len(record["labels"]) for record in members),
                               "class_counts": named(summed_counts(members)), "duplicate_group_ids": sorted({record["duplicate_group"] for record in members}),
                               "source_groups": sorted({record["source_group"] for record in members}), "final_split": members[0]["split"]})
    json_write(OUT / "metadata" / "group_manifest.json", group_manifest)
    json_write(OUT / "metadata" / "excluded_annotations.json", {"records": excluded, "counts_by_reason": dict(Counter(item["reason"] for item in excluded))})
    result = {**subset_statistics(records), "per_source": {source: subset_statistics([record for record in records if record["source"] == source]) for source in SOURCE_ROOTS},
              "per_split": {split: subset_statistics([record for record in records if record["split"] == split]) for split in RATIOS},
              "hcmc_day_night": {partition: subset_statistics([record for record in records if record["source"] == "HCMC" and record["partition"] == partition]) for partition in ["daytime", "nighttime"]},
              "bbox_size_statistics": size_statistics(records), "atomic_group_statistics": {"total": len(groups), "by_type": dict(Counter(item["group_type"] for item in group_manifest)),
                                         "per_split": dict(Counter(item["final_split"] for item in group_manifest)),
                                         "largest_groups": sorted(group_manifest, key=lambda item: item["image_count"], reverse=True)[:10]},
              "duplicate_statistics": {key: duplicates[key] for key in ["hcmc_duplicate_groups", "mio_duplicate_groups", "cross_source_exact_duplicates", "removed_canonical_duplicates", "removed_equivalent_bbox_copies"]},
              "duplicate_annotation_conflicts": len(duplicates["annotation_conflicts"]),
              "excluded_annotation_statistics": dict(Counter(item["reason"] for item in excluded)),
              "reconciliation": {"expected_raw_target_boxes": 680887,
                                 "invalid_source_boxes_excluded": sum(item["reason"] != "non_target_policy_exclusion" for item in excluded),
                                 "equivalent_duplicate_boxes_removed": sum(duplicates["removed_equivalent_bbox_copies"]),
                                 "final_boxes": sum(len(record["labels"]) for record in records)}}
    json_write(OUT / "metadata" / "dataset_statistics.json", result)
    for name, predicate in {"test_hcmc": lambda record: record["source"] == "HCMC", "test_mio": lambda record: record["source"] == "MIO",
                            "test_hcmc_day": lambda record: record["source"] == "HCMC" and record["partition"] == "daytime",
                            "test_hcmc_night": lambda record: record["source"] == "HCMC" and record["partition"] == "nighttime",
                            "test_overall": lambda record: True}.items():
        path = OUT / "evaluation_subsets" / (name + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join((OUT / "images" / "test" / record["final_image"]).as_posix() + "\n" for record in records if record["split"] == "test" and predicate(record)), encoding="utf-8")
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: car\n  1: motorcycle\n  2: bus\n  3: truck\n", encoding="utf-8")
    return result


def generate_preview(item):
    category, index, record = item
    image_path = OUT / "images" / record["split"] / record["final_image"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((1280, 960))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    measuring = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    source_lines = []
    current = ""
    for character in record["relative"]:
        if current and measuring.textlength(current + character, font=font) > image.width - 16:
            source_lines.append(current)
            current = ""
        current += character
    source_lines.append(current)
    header_lines = [f"{record['source']} | {record['split']} | {record.get('partition')}"] + source_lines
    header_height = 22 * len(header_lines) + 8
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    for header_index, line in enumerate(header_lines):
        draw.text((8, 4 + 22 * header_index), line, fill="black", font=font)
    for label in record["labels"]:
        class_id, x_center, y_center, width, height, source_class = label
        left, top = (x_center - width / 2) * image.width, (y_center - height / 2) * image.height + header_height
        right, bottom = (x_center + width / 2) * image.width, (y_center + height / 2) * image.height + header_height
        color = ["cyan", "yellow", "lime", "magenta"][class_id]
        draw.rectangle((left, top, right, bottom), outline=color, width=2)
        caption = NAMES[class_id] + (f" <- {source_class}" if record["source"] == "MIO" else "")
        caption_width = draw.textlength(caption, font=font)
        caption_left = min(max(2, left), max(2, image.width - caption_width - 2))
        draw.text((caption_left, max(header_height, top - 17)), caption, font=font, fill=color, stroke_width=1, stroke_fill="black")
    path = OUT / "qa" / category / f"{index:03d}__{record['final_image']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)
    return {"category": category, "preview": path.relative_to(OUT).as_posix(), "final_image": record["final_image"],
            "source_image": record["relative"], "split": record["split"], "boxes": len(record["labels"])}


def visual_qa(records):
    progress("generate_visual_qa")
    specs = {"random": (lambda record: True, 50),
             **{name: (lambda record, class_id=class_id: any(label[0] == class_id for label in record["labels"]), 75 if name == "bus" else 50) for class_id, name in enumerate(NAMES)},
             "hcmc": (lambda record: record["source"] == "HCMC", 30), "mio": (lambda record: record["source"] == "MIO", 30),
             "hcmc_day": (lambda record: record["source"] == "HCMC" and record["partition"] == "daytime", 30),
             "hcmc_night": (lambda record: record["source"] == "HCMC" and record["partition"] == "nighttime", 30),
             "small_boxes": (lambda record: any(label[3] * record["info"]["width"] * label[4] * record["info"]["height"] < 32 ** 2 for label in record["labels"]), 50),
             "mio_pickup_to_truck": (lambda record: any(label[-1] == "pickup_truck" for label in record["labels"]), 50),
             "mio_workvan_to_truck": (lambda record: any(label[-1] == "work_van" for label in record["labels"]), 50)}
    jobs = []
    for name, (predicate, minimum) in specs.items():
        candidates = sorted((record for record in records if predicate(record)), key=lambda record: record["final_image"])
        random.Random(363).shuffle(candidates)
        jobs.extend((name, index, record) for index, record in enumerate(candidates[:minimum], 1))
    with ThreadPoolExecutor(max_workers=8) as executor:
        manifest = list(executor.map(generate_preview, jobs))
    json_write(OUT / "qa" / "manifest.json", {"seed": 363, "category_counts": dict(Counter(item["category"] for item in manifest)), "previews": manifest})
    return {"seed": 363, "category_counts": dict(Counter(item["category"] for item in manifest)), "manifest": "qa/manifest.json", "human_confirmation": "PENDING"}


def final_report(inputs, statistics_data, validation, duplicates, near, qa):
    checks = {"all_classes_in_each_split": all(all(count > 0 for count in data["class_counts"].values()) for data in statistics_data["per_split"].values()),
              "both_sources_in_each_split": all(all(data["source_counts"][source]["images"] > 0 for source in SOURCE_ROOTS) for data in statistics_data["per_split"].values()),
              "qa_minimums_met": all(count >= (75 if category == "bus" else 30 if category in {"hcmc", "mio", "hcmc_day", "hcmc_night"} else 50) for category, count in qa["category_counts"].items())}
    failure_counts = [value for key, value in validation.items() if key not in {"generated_boundary_floating_point_tolerance", "source_boundaries_checked_strictly", "all_final_image_bytes_sha256_revalidated"}]
    status = "FAIL" if any(failure_counts) or duplicates["annotation_conflicts"] or not all(checks.values()) else "REVIEW_REQUIRED"
    limitations = ["HCMC semantics rely on day/night statistics, representative crop QA and human review; no authoritative map recovered.",
                   "HCMC duplicated source IDs 0/4 motorcycle, 1/5 car, 2/6 bus, 3/7 truck.",
                   "MIO work_van -> truck is a deliberate broad-superclass merge, MEDIUM confidence.",
                   "MIO motorized_vehicle excluded because it is semantically indeterminate.",
                   "Excluded MIO objects may remain visible without target annotations.",
                   "Source/domain and class imbalance remains possible.",
                   "HCMC camera/video ID groups are conservative filename inferences; human confirmation required.",
                   "Near-duplicate check is sampled, not exhaustive.",
                   "MIO original test images have no local ground truth and are not included; duplicate aliases are retained in metadata.",
                   "Generated images may be hardlinks to source bytes; treat images as immutable, editing a hardlink would affect its source."]
    boundary_path = OUT / "metadata" / "hcmc_boundary_diagnostics.json"
    if boundary_path.exists():
        boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        limitations.append(f"Strict HCMC boundary validation excludes {boundary['invalid_hcmc_boxes']} rows: {boundary['overshoot_within_previous_1e6_tolerance']} have overshoot <=1e-6, and {boundary['overshoot_beyond_previous_1e6_tolerance']} exceed the earlier audit tolerance. Source coordinates are never clamped.")
    qa["duplicate_conflict_review_manifest"] = "qa/duplicate_review_manifest.json"
    qa["duplicate_conflict_review_guide"] = "qa/FINAL_HUMAN_REVIEW.md"
    licenses = {"HCMC": {"status": "UNKNOWN — MANUAL REVIEW REQUIRED", "evidence": "No README/license/config found under actual local HCMC source root"},
                "MIO": {"status": "CC BY-NC-SA 4.0", "evidence": str(SOURCE_ROOTS["MIO"] / "README.txt")},
                "combined_redistribution_claimed": False}
    json_write(OUT / "metadata" / "license_status.json", licenses)
    report = {"status": status, "processing_complete": True,
              "input_validation": inputs, "mapping_validation": {"locked_hcmc_map": HCMC_MAP, "locked_mio_map": MIO_MAP, "applied_without_policy_change": True},
              "conversion_validation": statistics_data["reconciliation"], "duplicate_validation": {key: value for key, value in duplicates.items() if key != "groups"},
              "group_validation": {"rule": "All frames of each parsed HCMC camera/video ID, across day/night and any frame gaps, remain atomic. Exact hash links connect whole source groups. MIO exact hashes are atomic; numeric IDs imply no temporal sequence.",
                                   "group_statistics": statistics_data["atomic_group_statistics"]},
              "split_validation": {**checks, **validation}, "label_validation": validation,
              "source_validation": {"source_files_modified": False, "source_roots": {key: str(value) for key, value in SOURCE_ROOTS.items()}},
              "class_distribution": statistics_data["class_counts"], "bbox_distribution": statistics_data["bbox_size_statistics"],
              "perceptual_near_duplicate_validation": {key: value for key, value in near.items() if key != "candidates"},
              "visual_qa_manifest": qa, "known_limitations": limitations, "license_status": licenses,
              "next": "FINAL HUMAN QA", "trained_yolo": False}
    json_write(OUT / "final_qa_report.json", report)
    rows = ["# Vehicle363-v1", "", f"Work 4.7 status: **{status}**. Final human QA is pending.", "",
            f"Integrity blocker: {len(duplicates['annotation_conflicts']):,} exact-image groups have differing target annotations. These groups are preserved atomically and are not automatically reconciled. See `qa/FINAL_HUMAN_REVIEW.md` and `metadata/duplicate_report.json` before using this dataset.", "",
            "## Schema", "", "0 car · 1 motorcycle · 2 bus · 3 truck", "",
            "## Build and reproducibility", "", "Run `python scripts/build_vehicle363_v1.py` from the project root. Progress and resumable local checkpoints are in `data/vehicle_audit/vehicle363_build/`. The build verifies source counts, decodes every source and final image, recalculates SHA-256, compares duplicate target annotations, and validates every YOLO row. No source label or image is edited.", "",
            "Image inclusion requires at least one valid mapped target bbox. Original MIO test images have no local ground truth and are included in the full hash audit only. If a training image has both target and excluded objects, it is retained with target labels only.", "",
            "## Statistics", "", "| Split | Images | Boxes | Car | Motorcycle | Bus | Truck | HCMC images | MIO images |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, data in statistics_data["per_split"].items():
        rows.append(f"| {split} | {data['images']:,} | {data['boxes']:,} | " + " | ".join(f"{data['class_counts'][name]:,}" for name in NAMES) + f" | {data['source_counts']['HCMC']['images']:,} | {data['source_counts']['MIO']['images']:,} |")
    rows += ["", f"Total: {statistics_data['images']:,} images, {statistics_data['boxes']:,} boxes.", "", "## Grouping and duplicates", "", report["group_validation"]["rule"], "",
             "Canonicalization removes a duplicate pair only when target annotation multisets agree. Conflicting duplicate annotations are retained and fully reported; they prevent PASS. The split minimizes source-specific image/class deficits against 75/12.5/12.5 while preserving groups. Bus receives extra weight. Group integrity overrides exact ratios.", "",
             "## Evaluation", "", "`evaluation_subsets/` contains absolute-path image lists for overall test, HCMC-only, MIO-only and HCMC day/night. These are evaluation manifests, not extra datasets. All four classes and both sources must occur in every split.", "",
             "## Validation and human QA", "", "See `final_qa_report.json`, `metadata/output_validation.json`, `metadata/near_duplicate_report.json` and `qa/manifest.json`. Preview categories contain individual annotated images with full source filename, target class and MIO source class. Human review must confirm merged-class examples and filename-based grouping. Near-duplicate candidates require review of whole atomic groups; no individual image is moved to correct them.", "",
             "## Known limitations", ""] + [f"- {item}" for item in limitations]
    rows += ["", "## Source license and usage", "", "HCMC: **UNKNOWN — MANUAL REVIEW REQUIRED**. No local source usage terms were recovered.", "", "MIO-TCD: the local README states **CC BY-NC-SA 4.0**. This file reports local documentation and does not invent additional permissions.", "", "Combined dataset redistribution is not claimed. Review source terms separately before redistribution.", "", "## Next", "", "FINAL HUMAN QA. Do not train YOLO during Work 4.7.", ""]
    (OUT / "README.md").write_text("\n".join(rows), encoding="utf-8")
    return report


def review_diagnostics(excluded, duplicates, records):
    bad_hcmc = [item for item in excluded if item["source"] == "HCMC"]
    overshoots = []
    for item in bad_hcmc:
        x_center, y_center, width, height = [Decimal(value) for value in item["bbox"]]
        overshoots.append(max(Decimal(0), -(x_center - width / 2), -(y_center - height / 2),
                              x_center + width / 2 - 1, y_center + height / 2 - 1))
    boundary_report = {"invalid_hcmc_boxes": len(bad_hcmc),
                       "overshoot_within_previous_1e6_tolerance": sum(value <= Decimal("0.000001") for value in overshoots),
                       "overshoot_beyond_previous_1e6_tolerance": sum(value > Decimal("0.000001") for value in overshoots),
                       "maximum_normalized_overshoot": str(max(overshoots, default=Decimal(0))),
                       "reason_for_difference_from_work44": "Previous audit used a 1e-6 implied-boundary tolerance; this conversion checks source decimals strictly.",
                       "policy": "Exclude every genuinely out-of-bounds source row; no coordinate clamping or annotation repair."}
    json_write(OUT / "metadata" / "hcmc_boundary_diagnostics.json", boundary_report)
    record_by_key = {(record["source"], record["relative"]): record for record in records}
    comparison_summary = Counter()
    examples = []
    for conflict in duplicates["annotation_conflicts"]:
        member_counts = [counts(member["target_annotations"]) for member in conflict["members"]]
        if all(member == member_counts[0] for member in member_counts):
            comparison_summary["same_target_class_counts_different_geometry"] += 1
        else:
            comparison_summary["different_target_class_or_object_counts"] += 1
    for group_index, conflict in enumerate(duplicates["annotation_conflicts"][:50], 1):
        images = []
        for member_index, member in enumerate(conflict["members"][:2], 1):
            record = record_by_key.get((member["source"], member["image"]))
            if record is not None:
                preview = generate_preview(("duplicate_annotation_conflicts", group_index * 2 + member_index, record))
                images.append(preview)
        examples.append({"group_id": conflict["group_id"], "previews": images})
    json_write(OUT / "qa" / "duplicate_review_manifest.json", {"unresolved_conflict_groups": len(duplicates["annotation_conflicts"]),
               "comparison_summary": dict(comparison_summary), "compatible_subset_groups": len(duplicates["compatible_annotation_groups_requiring_review"]),
               "examples": examples, "automatic_reconciliation_performed": False})
    text = ["# Final human review", "", f"Unresolved duplicate annotation conflicts: {len(duplicates['annotation_conflicts']):,} groups.", "",
            "Inspect the paired full-frame previews below. They show the same exact image bytes with different source annotations. No annotation union or arbitrary canonical annotation choice was made for these groups.", "",
            "A human policy decision is needed before these source conflicts can be reconciled. All copies stay in one split. See `metadata/duplicate_report.json` for complete annotations and source identities.", "",
            "| Exact group | First annotation | Second annotation |", "|---|---|---|"]
    for example in examples:
        paths = [item["preview"].removeprefix("qa/") for item in example["previews"]]
        text.append(f"| {example['group_id'][:23]} | " + " | ".join(f"[Preview {index + 1}]({path})" for index, path in enumerate(paths)) + " |")
    (OUT / "qa" / "FINAL_HUMAN_REVIEW.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return boundary_report


def write_summary(statistics_data, validation, duplicates, near, status):
    lines = ["WORK 4.7 STATUS:", status, "", "FINAL DATASET:", "Vehicle363-v1", "",
             "TOTAL:", f"Images: {statistics_data['images']}", f"Boxes: {statistics_data['boxes']}", "", "CLASS COUNTS:"]
    lines += [f"{name}: {count}" for name, count in statistics_data["class_counts"].items()]
    lines += ["", "SOURCE:"]
    for source, data in statistics_data["per_source"].items():
        lines += [f"{source} images: {data['images']}", f"{source} boxes: {data['boxes']}"]
    for split, data in statistics_data["per_split"].items():
        lines += ["", split.upper() + ":", f"images: {data['images']}", f"boxes: {data['boxes']}"]
        lines += [f"{name}: {count}" for name, count in data["class_counts"].items()]
        lines += [f"{source} images: {data['source_counts'][source]['images']}" for source in SOURCE_ROOTS]
    lines += ["", "DUPLICATES:", f"HCMC duplicate groups: {duplicates['hcmc_duplicate_groups']}",
              f"MIO duplicate groups: {duplicates['mio_duplicate_groups']}",
              f"cross-source exact duplicates: {duplicates['cross_source_exact_duplicates']}",
              f"removed canonical duplicates: {duplicates['removed_canonical_duplicates']}",
              f"annotation conflicts: {len(duplicates['annotation_conflicts'])}",
              f"compatible subsets requiring review before union: {len(duplicates['compatible_annotation_groups_requiring_review'])}", "", "LEAKAGE:"]
    for display, key in {"image overlap": "image_overlap", "atomic group overlap": "atomic_group_overlap",
                         "HCMC sequence overlap": "hcmc_sequence_overlap", "MIO duplicate overlap": "mio_duplicate_overlap",
                         "exact-hash cross-split overlap": "exact_hash_cross_split_overlap"}.items():
        lines.append(f"{display}: {validation[key]}")
    lines += [f"strong near-duplicate candidates: {near['candidate_count']} (sampled perceptual check; review signals)", "", "EXCLUDED:"]
    exclusion_counts = statistics_data["excluded_annotation_statistics"]
    lines += [f"invalid HCMC boxes: {exclusion_counts.get('invalid_source_yolo_geometry', 0)}",
              f"invalid MIO boxes: {exclusion_counts.get('invalid_source_pixel_geometry', 0)}",
              f"MIO non-target annotations: {exclusion_counts.get('non_target_policy_exclusion', 0)}", "", "VALIDATION:"]
    for display, key in {"invalid YOLO rows": "invalid_yolo_rows", "invalid boxes": "invalid_boxes", "missing images": "missing_images",
                         "missing labels": "missing_labels", "unreadable images": "unreadable_images"}.items():
        lines.append(f"{display}: {validation[key]}")
    lines += ["", "LICENSE:", "HCMC: UNKNOWN — MANUAL REVIEW REQUIRED", "MIO: CC BY-NC-SA 4.0 (local README)", "",
              "NEXT:", "FINAL HUMAN QA — resolve duplicate annotation conflicts before declaring this dataset ready.", "", "DO NOT TRAIN YOLO.", ""]
    (OUT / "WORK_4_7_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


def refresh_review_report():
    records, excluded, inputs, duplicates = checkpoint("build_plan")
    validation = json.loads((OUT / "metadata" / "output_validation.json").read_text(encoding="utf-8"))
    statistics_data = json.loads((OUT / "metadata" / "dataset_statistics.json").read_text(encoding="utf-8"))
    near = json.loads((OUT / "metadata" / "near_duplicate_report.json").read_text(encoding="utf-8"))
    qa_manifest_path = OUT / "qa" / "manifest.json"
    qa_manifest = json.loads(qa_manifest_path.read_text(encoding="utf-8"))
    ordinal = Counter()
    for item in qa_manifest["previews"]:
        ordinal[item["category"]] += 1
        old_path = OUT / item["preview"]
        new_name = f"{ordinal[item['category']]:03d}__{item['final_image']}"
        new_path = old_path.with_name(new_name)
        if old_path != new_path:
            if new_path.exists():
                raise RuntimeError(f"Unexpected duplicate QA preview: {new_path}")
            old_path.replace(new_path)
            item["preview"] = new_path.relative_to(OUT).as_posix()
    json_write(qa_manifest_path, qa_manifest)
    review_diagnostics(excluded, duplicates, records)
    qa = {"seed": 363, "category_counts": qa_manifest["category_counts"], "manifest": "qa/manifest.json", "human_confirmation": "PENDING"}
    report = final_report(inputs, statistics_data, validation, duplicates, near, qa)
    write_summary(statistics_data, validation, duplicates, near, report["status"])
    progress("COMPLETE", status=report["status"], images=statistics_data["images"], boxes=statistics_data["boxes"])


def main():
    if (OUT / "RELEASE_MANIFEST.json").exists():
        raise RuntimeError("Frozen release exists. Create a new dataset version; do not rebuild Vehicle363-v1 in place.")
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        inventory = scan_sources()
        all_records, excluded, inputs = parse_annotations(inventory)
        records, duplicates = deduplicate(all_records, inventory)
        groups = create_groups(records)
        split_groups(groups)
        json_write(OUT / "metadata" / "split_plan.json", {
            "target_ratios": RATIOS,
            "per_split": {split: subset_statistics([record for record in records if record["split"] == split]) for split in RATIOS},
            "atomic_groups": len(groups), "seed": 363})
        progress("split_plan_ready", **{split: subset_statistics([record for record in records if record["split"] == split]) for split in RATIOS})
        checkpoint("build_plan", (records, excluded, inputs, duplicates))
        near = near_duplicate_check(records)
        migrate_outputs(records, all_records)
        validation = validate_outputs(records)
        statistics_data = write_metadata(records, groups, excluded, duplicates, inputs, near)
        qa = visual_qa(records)
        review_diagnostics(excluded, duplicates, records)
        report = final_report(inputs, statistics_data, validation, duplicates, near, qa)
        write_summary(statistics_data, validation, duplicates, near, report["status"])
        progress("COMPLETE", status=report["status"], images=statistics_data["images"], boxes=statistics_data["boxes"])
    except Exception as error:
        progress("FAILED", error=f"{type(error).__name__}: {error}")
        json_write(OUT / "final_qa_report.json", {"status": "FAIL", "processing_complete": False,
                                                 "error": f"{type(error).__name__}: {error}", "source_files_modified": False})
        raise


if __name__ == "__main__":
    main()
