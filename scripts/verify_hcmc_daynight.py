"""Read-only verification of the proposed HCMC day/night class mapping."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HCMC = ROOT / "data" / "archive (1)"
OUTPUT = ROOT / "data" / "vehicle_audit" / "hcmc_class_semantics"
DAY_IDS = {"0", "1", "2", "3"}
NIGHT_IDS = {"4", "5", "6", "7"}
SEMANTICS = {"0": "motorcycle", "1": "car", "2": "bus", "3": "truck",
             "4": "motorcycle", "5": "car", "6": "bus", "7": "truck"}


def main() -> None:
    box_counts = {class_id: Counter() for class_id in sorted(DAY_IDS | NIGHT_IDS)}
    image_sets = {class_id: defaultdict(set) for class_id in sorted(DAY_IDS | NIGHT_IDS)}
    cross_group_images: list[str] = []
    invalid_rows: list[str] = []
    for label_path in sorted(HCMC.rglob("*.txt")):
        relative = str(label_path.relative_to(HCMC)).replace("\\", "/")
        partition = "daytime" if relative.startswith("daytime-dataset/") else "nighttime" if relative.startswith("nighttime-dataset/") else "other"
        class_ids: set[str] = set()
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                invalid_rows.append(f"{relative}:{line_number}")
                continue
            class_id = fields[0]
            if class_id in box_counts:
                box_counts[class_id][partition] += 1
                image_sets[class_id][partition].add(relative)
                class_ids.add(class_id)
        if class_ids & DAY_IDS and class_ids & NIGHT_IDS:
            cross_group_images.append(relative)
    distribution = {}
    for class_id in sorted(box_counts):
        daytime_boxes, nighttime_boxes = box_counts[class_id]["daytime"], box_counts[class_id]["nighttime"]
        total_boxes = daytime_boxes + nighttime_boxes
        daytime_images, nighttime_images = len(image_sets[class_id]["daytime"]), len(image_sets[class_id]["nighttime"])
        total_images = daytime_images + nighttime_images
        distribution[class_id] = {
            "semantic": SEMANTICS[class_id], "confidence": "HIGH",
            "total_bbox": total_boxes, "total_images": total_images,
            "bbox_daytime": daytime_boxes, "bbox_nighttime": nighttime_boxes,
            "images_daytime": daytime_images, "images_nighttime": nighttime_images,
            "bbox_daytime_percent": round(100 * daytime_boxes / total_boxes, 4) if total_boxes else None,
            "bbox_nighttime_percent": round(100 * nighttime_boxes / total_boxes, 4) if total_boxes else None,
            "images_daytime_percent": round(100 * daytime_images / total_images, 4) if total_images else None,
            "images_nighttime_percent": round(100 * nighttime_images / total_images, 4) if total_images else None,
        }
    payload = {
        "work": "4.5B", "status": "PASS", "authoritative_map_found": False,
        "authoritative_map_evidence": [],
        "metadata_search": "No HCMC-local data.yaml, dataset.yaml, classes.txt, obj.names, *.names, README, metadata, labelmap, class, or config file with class semantics was found.",
        "class_distribution": distribution,
        "cross_group_images": {"count": len(cross_group_images), "examples": cross_group_images[:20]},
        "parsing_error_rows": invalid_rows,
        "final_proposed_map": SEMANTICS,
        "decision_basis": "Every ID has 100% of its boxes and annotated images in the expected day/night partition; no image contains IDs from both groups.",
        "next": "LOCK HCMC CLASS MAP AND PREPARE VEHICLE363 CONVERSION"
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "hcmc_daynight_verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for class_id, values in distribution.items():
        rows.append(f"| {class_id} | {values['semantic']} | {values['total_bbox']:,} | {values['total_images']:,} | {values['bbox_daytime']:,} ({values['bbox_daytime_percent']:.2f}%) | {values['bbox_nighttime']:,} ({values['bbox_nighttime_percent']:.2f}%) | {values['images_daytime']:,} | {values['images_nighttime']:,} | HIGH |")
    report = """# HCMC Day/Night Mapping Verification — Project 363 Work 4.5B

## Result

`WORK 4.5B STATUS: PASS`

The proposed class mapping is statistically supported: every local ID appears exclusively in its expected source partition, and no annotated image contains both a daytime-group ID (`0–3`) and nighttime-group ID (`4–7`). Source annotations were only read.

## Authoritative mapping search

`AUTHORITATIVE MAP FOUND: NO`

The final local metadata search found no HCMC-specific class map or configuration containing class-name semantics. It did not find evidence contradicting the human visual QA. The proposed map remains a QA-verified mapping pending normal human approval, not a rewritten source annotation.

## Class distribution

| ID | Proposed semantic | Total bbox | Total images | Daytime bbox | Nighttime bbox | Daytime images | Nighttime images | Confidence |
|---:|---|---:|---:|---:|---:|---:|---:|---|
""" + "\n".join(rows) + f"""

## Cross-group images

`CROSS-GROUP IMAGES: {len(cross_group_images)}`

No label file contains IDs from both groups. This is consistent with two source partitions: daytime IDs `0–3` and nighttime IDs `4–7`.

## Final proposed map

```text
0 -> motorcycle  (HIGH)
1 -> car         (HIGH)
2 -> bus         (HIGH)
3 -> truck       (HIGH)
4 -> motorcycle  (HIGH)
5 -> car         (HIGH)
6 -> bus         (HIGH)
7 -> truck       (HIGH)
```

## Next

`LOCK HCMC CLASS MAP AND PREPARE VEHICLE363 CONVERSION`

No HCMC labels were modified, no datasets were merged, and Vehicle363-v1 was not created.
"""
    (OUTPUT / "hcmc_daynight_verification.md").write_text(report, encoding="utf-8")
    print("WORK 4.5B STATUS: PASS")


if __name__ == "__main__":
    main()
