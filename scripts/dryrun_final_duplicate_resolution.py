import json
import math
import pickle
import random
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_duplicate_resolution as audit
from scripts import simulate_duplicate_conflict_drop as simulation


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/Vehicle363-v1"
OUTPUT = ROOT / "data/vehicle_audit/duplicate_resolution"
NAMES = audit.NAMES
RULES = {
    "same_class_high_iou": 0.90,
    "small_jitter_maximum_corner_shift_pixels": 2.0,
    "small_jitter_minimum_iou": 0.70,
    "small_jitter_minimum_width_height_ratio": 0.85,
    "near_duplicate_internal_iou": 0.90,
    "near_duplicate_tight_footprint_iou": 0.85,
    "tight_footprint_minimum_width_height_ratio": 0.90,
    "tight_footprint_maximum_relative_center_shift": 0.05,
    "cross_class_same_footprint_iou": 0.80,
    "cross_class_minimum_width_height_ratio": 0.85,
    "cross_class_maximum_relative_center_shift": 0.10,
}


def shape_metrics(first, second):
    first_width, first_height = first[3] - first[1], first[4] - first[2]
    second_width, second_height = second[3] - second[1], second[4] - second[2]
    ratios = (min(first_width, second_width) / max(first_width, second_width),
              min(first_height, second_height) / max(first_height, second_height))
    center_shift = (abs(first[1] + first[3] - second[1] - second[3]) / (2 * min(first_width, second_width)),
                    abs(first[2] + first[4] - second[2] - second[4]) / (2 * min(first_height, second_height)))
    return min(ratios), max(center_shift)


def matching_evidence(first, second):
    overlap = audit.iou(first, second)
    if first[0] != second[0]:
        return None
    if overlap >= RULES["same_class_high_iou"]:
        return "HIGH_IOU"
    ratio, _ = shape_metrics(first, second)
    corner_shift = max(abs(first[index] - second[index]) for index in range(1, 5))
    centers_inside = all(left[1] <= (right[1] + right[3]) / 2 <= left[3] and left[2] <= (right[2] + right[4]) / 2 <= left[4] for left, right in ((first, second), (second, first)))
    if overlap >= RULES["small_jitter_minimum_iou"] and corner_shift <= RULES["small_jitter_maximum_corner_shift_pixels"] + 1e-9 and ratio >= RULES["small_jitter_minimum_width_height_ratio"] and centers_inside:
        return "BOUNDED_2PX_JITTER"
    return None


def same_footprint(first, second):
    ratio, shift = shape_metrics(first, second)
    return audit.iou(first, second) >= RULES["cross_class_same_footprint_iou"] and ratio >= RULES["cross_class_minimum_width_height_ratio"] and shift <= RULES["cross_class_maximum_relative_center_shift"]


def internally_suspicious(boxes):
    findings = []
    advisory = []
    for first_index, first in enumerate(boxes):
        for second_index in range(first_index + 1, len(boxes)):
            second = boxes[second_index]
            overlap = audit.iou(first, second)
            ratio, shift = shape_metrics(first, second)
            near_duplicate = overlap >= RULES["near_duplicate_internal_iou"] or (overlap >= RULES["near_duplicate_tight_footprint_iou"] and ratio >= RULES["tight_footprint_minimum_width_height_ratio"] and shift <= RULES["tight_footprint_maximum_relative_center_shift"])
            if near_duplicate or (first[0] != second[0] and same_footprint(first, second)):
                findings.append({"first_box": first_index, "second_box": second_index, "iou": overlap, "reason": "likely_same_object_duplicate" if first[0] == second[0] else "same_footprint_different_classes"})
            elif overlap >= 0.50:
                advisory.append({"first_box": first_index, "second_box": second_index, "iou": overlap, "reason": "overlap_alone_not_rejected_possible_occlusion"})
    return findings, advisory


def compare_sets(first, second):
    graph = [[index for index, target in enumerate(second) if matching_evidence(box, target)] for box in first]

    def assign(forbidden=None):
        columns = {}

        def augment(row, visited):
            for column in graph[row]:
                if (row, column) == forbidden or column in visited:
                    continue
                visited.add(column)
                if column not in columns or augment(columns[column], visited):
                    columns[column] = row
                    return True
            return False

        if not all(augment(row, set()) for row in range(len(first))):
            return None
        return sorted((row, column) for column, row in columns.items())

    pairs = assign()
    unique = pairs is not None and not any(assign(pair) is not None for pair in pairs if len(graph[pair[0]]) > 1)
    return {
        "complete": pairs is not None, "unique": unique,
        "empty_source_set": not first,
        "pairs": [{"from_box": left, "to_box": right, "class": NAMES[first[left][0]], "iou": audit.iou(first[left], second[right]), "evidence": matching_evidence(first[left], second[right])} for left, right in (pairs or [])],
        "unmatched_source_boxes": [index for index, neighbors in enumerate(graph) if not neighbors],
        "additional_candidate_boxes": [index for index in range(len(second)) if pairs is not None and index not in {right for _, right in pairs}],
    }


def geometry_valid(member):
    for row in member["target_annotations"]:
        class_id, center_x, center_y, width, height = row[:5]
        if not isinstance(class_id, int) or class_id not in range(4):
            return False
        if not all(math.isfinite(value) for value in row[:5]):
            return False
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            return False
        if not (-1e-12 <= center_x - width / 2 < center_x + width / 2 <= 1 + 1e-12 and -1e-12 <= center_y - height / 2 < center_y + height / 2 <= 1 + 1e-12):
            return False
    return all(audit.valid_box(box, member["width"], member["height"]) for box in member["pixel_boxes"])


def classify(group):
    members = sorted(group["members"], key=lambda member: (-len(member["pixel_boxes"]), member["member_id"]))
    largest_count = len(members[0]["pixel_boxes"])
    candidates = [member for member in members if len(member["pixel_boxes"]) == largest_count]
    attempts = []
    for candidate in candidates:
        blockers, advisory = internally_suspicious(candidate["pixel_boxes"])
        attempt = {"candidate_member": candidate["member_id"], "valid_geometry_and_classes": geometry_valid(candidate), "internal_suspicious_pairs": blockers, "normal_overlap_advisories": advisory, "comparisons": [], "cross_class_contradictions": [], "reasons": []}
        if not attempt["valid_geometry_and_classes"]:
            attempt["reasons"].append("invalid_candidate_geometry_or_class")
        if blockers:
            attempt["reasons"].append("candidate_has_likely_duplicate_or_contradictory_boxes")
        for member in members:
            if member is candidate:
                continue
            if not geometry_valid(member):
                attempt["reasons"].append("invalid_other_member_geometry")
                continue
            comparison = compare_sets(member["pixel_boxes"], candidate["pixel_boxes"])
            comparison["member"] = member["member_id"]
            attempt["comparisons"].append(comparison)
            if not comparison["complete"]:
                attempt["reasons"].append("objects_not_provably_preserved")
            elif not comparison["unique"]:
                attempt["reasons"].append("ambiguous_one_to_one_correspondence")
            for first_index, first in enumerate(member["pixel_boxes"]):
                for second_index, second in enumerate(candidate["pixel_boxes"]):
                    if first[0] != second[0] and same_footprint(first, second):
                        attempt["cross_class_contradictions"].append({"member": member["member_id"], "member_box": first_index, "candidate_box": second_index, "member_class": NAMES[first[0]], "candidate_class": NAMES[second[0]], "iou": audit.iou(first, second)})
        if attempt["cross_class_contradictions"]:
            attempt["reasons"].append("same_object_footprint_assigned_different_classes")
        attempt["reasons"] = sorted(set(attempt["reasons"]))
        attempts.append(attempt)
        if not attempt["reasons"]:
            category = "JITTER_EQUIVALENT" if all(len(member["pixel_boxes"]) == largest_count for member in members) else "MORE_COMPLETE_VALID"
            return {"group_id": group["group_id"], "category": category, "auto_resolvable": True, "canonical_member": candidate["member_id"], "confidence": "HIGH" if all(pair["evidence"] == "HIGH_IOU" for comparison in attempt["comparisons"] for pair in comparison["pairs"]) else "MEDIUM", "candidate_attempts": attempts, "reason": "One existing largest internally consistent annotation set uniquely preserves all other valid target objects under the documented matching rules. Empty target sets provide no negative evidence; no boxes are added or unioned."}
    return {"group_id": group["group_id"], "category": "REAL_CONFLICT", "auto_resolvable": False, "canonical_member": None, "confidence": "LOW", "candidate_attempts": attempts, "reason": "; ".join(sorted({reason for attempt in attempts for reason in attempt["reasons"]})), "interpretation": "Unresolved contradiction or insufficient geometric evidence, not proof all annotations are wrong."}


def leakage(records):
    dimensions = {name: defaultdict(set) for name in ("exact_hash_overlap", "atomic_group_overlap", "hcmc_sequence_overlap", "mio_duplicate_overlap")}
    for record in records:
        split = record["final_split"]
        dimensions["exact_hash_overlap"][record["sha256"]].add(split)
        dimensions["atomic_group_overlap"][record["atomic_group"]].add(split)
        if record["source_dataset"] == "HCMC":
            if record["camera_id"] is None:
                raise RuntimeError("HCMC camera/sequence missing")
            dimensions["hcmc_sequence_overlap"][record["camera_id"]].add(split)
        else:
            dimensions["mio_duplicate_overlap"][record["duplicate_group"]].add(split)
    return {name: sum(len(splits) > 1 for splits in groups.values()) for name, groups in dimensions.items()}


def generate_qa(groups, results):
    unresolved = [result for result in results if result["category"] == "REAL_CONFLICT"]
    directory = OUTPUT / "qa/final_real_conflict"
    directory.mkdir(parents=True, exist_ok=True)
    generator = random.Random(36348)
    shuffled = sorted(unresolved, key=lambda result: result["group_id"])
    generator.shuffle(shuffled)
    chosen = {}
    for reason in sorted({reason for result in unresolved for attempt in result["candidate_attempts"] for reason in attempt["reasons"]}):
        for result in [result for result in shuffled if any(reason in attempt["reasons"] for attempt in result["candidate_attempts"])][:10]:
            chosen[result["group_id"]] = result
    for class_id in (3, 1, 2, 0):
        for result in [result for result in shuffled if any(box[0] == class_id for member in groups[result["group_id"]]["members"] for box in member["pixel_boxes"])][:15]:
            chosen[result["group_id"]] = result
    for result in shuffled:
        if len(chosen) >= min(100, len(unresolved)):
            break
        chosen[result["group_id"]] = result
    samples = list(chosen.values())[:100]
    previews = []
    entries = []
    for index, result in enumerate(samples, 1):
        group = groups[result["group_id"]]
        members = {member["member_id"]: member for member in group["members"]}
        attempt = result["candidate_attempts"][0]
        candidate = members[attempt["candidate_member"]]
        comparison = next((entry for entry in attempt["comparisons"] if not entry["complete"] or not entry["unique"]), None)
        contradiction = next(iter(attempt["cross_class_contradictions"]), None)
        partner_id = contradiction["member"] if contradiction else comparison["member"] if comparison else next(member_id for member_id in members if member_id != candidate["member_id"])
        preview = Image.new("RGB", (960, 475), "white")
        draw = ImageDraw.Draw(preview)
        draw.text((6, 4), f'REAL_CONFLICT {result["group_id"]}', font=audit.font(12), fill="black")
        draw.text((6, 24), "MAX candidate is NOT approved. Compare against competing annotations.", font=audit.font(), fill="red")
        preview.paste(audit.render_member(members[partner_id]), (6, 58))
        preview.paste(audit.render_member(candidate), (482, 58))
        draw.text((6, 427), "\n".join(textwrap.wrap(result["reason"], 130)[:3]), font=audit.font(12), fill="black")
        filename = f'{index:03d}__{result["group_id"].removeprefix("exact__")[:16]}.jpg'
        preview.save(directory / filename, quality=92)
        previews.append(preview)
        entries.append({"group_id": result["group_id"], "preview": filename, "max_candidate": candidate["member_id"], "competing_member": partner_id, "source_member_count": len(members), "only_two_members_displayed": len(members) > 2, "reason": result["reason"]})
    sheets = []
    for offset in range(0, len(previews), 10):
        page = Image.new("RGB", (1920, 2375), "white")
        for cell_index, preview in enumerate(previews[offset:offset + 10]):
            page.paste(preview, ((cell_index % 2) * 960, (cell_index // 2) * 475))
        filename = f"contact_sheet_{offset // 10 + 1:02d}.jpg"
        page.save(directory / filename, quality=92)
        sheets.append(filename)
    qa = {"seed": 36348, "total_real_conflicts": len(unresolved), "sample_count": len(entries), "selection": "Stratified by failure reason and class, then deterministic random fill; representative, not exhaustive or a prevalence estimate.", "contact_sheets": sheets, "examples": entries}
    (directory / "manifest.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    lines = ["# Work 4.7C — unresolved real-conflict QA", "", "No candidate shown here is approved. Green boxes show each member's existing annotations; green is not a correctness verdict. Compare object identity, class, missing objects, near-duplicate boxes, and whether apparent overlap is normal occlusion. Multi-member groups show only a diagnostic pair; full comparisons are in work_4_7c_dryrun.json.", "", f"Remaining groups: {len(unresolved):,}; displayed representative groups: {len(entries)}. Do not modify annotations or approve all groups based on this sample.", "", "Review record: group_id | candidate/other member | same objects? | class contradiction? | duplicate bbox? | decision/reason.", "", "## Contact sheets", *[f"- [{name}]({name})" for name in sheets], "", "## Individual groups", *[f'- [{entry["group_id"]}]({entry["preview"]}) — {entry["reason"]}' for entry in entries]]
    (directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return qa


def report_markdown(report):
    result = report["simulation"]
    lines = ["# Work 4.7C — final duplicate resolution DRY RUN", "", f'**STATUS: {report["status"]}**', "", "DO NOT APPLY CHANGES YET. No dataset/source image, label, or split was modified. The Work 4.7B DROP-ALL policy is not used.", "", "## Human QA evidence", "", "The human reviewer reports inspecting representative duplicate examples: bbox geometry was visually acceptable; additional boxes represented real objects with correct target classes; no abnormal duplicate overlap was observed. This is user-supplied representative evidence, not a claim that all 9,581 groups were manually reviewed or that every proposed extra was visually verified.", "", "## Reclassification of previous TRUE_CONFLICT", "", *[f'- {name}: {count:,}' for name, count in report["reclassification_counts"].items()], "", f'Primary automatically resolvable groups including Work 4.7A: {report["auto_resolvable_primary_groups"]:,}. Supplemental compatible-subset auto group: {report["supplemental_auto_groups"]}. Remaining human-review groups: {report["human_review_remaining"]:,}.', "", "REAL_CONFLICT means genuine contradiction OR insufficient/ambiguous evidence. No member is automatically chosen for these groups; every current copy stays in the projected dataset.", "", "## Matching and overlap rules", "", "Largest box count only proposes a candidate. It must pass strict geometry/classes, internal near-duplicate checks, and unique one-to-one same-class containment against every other member. Candidates are ordered by descending valid target count, then member ID; tied maximum candidates are tested in that order. No smaller candidate substitutes for a failed maximum set.", "", "Same-class correspondence requires IoU >=0.90, OR <=2px corner jitter with IoU >=0.70, width/height ratios >=0.85, and each center inside the other box. The 2px rule follows the earlier geometric jitter proxy; it is a conservative engineering heuristic, not an independently measured semantic accuracy guarantee. No broad IoU-threshold lowering is used. Empty target sets add no negative-object evidence and may be contained in a validated positive set.", "", "Internal IoU >=0.90 is suspicious; IoU >=0.85 with >=0.90 size ratios and <=5% relative center shift is also suspicious. Mere IoU >=0.50 is recorded as an advisory, not rejected as duplicate: nearby vehicles can overlap normally. Across members, different classes with IoU >=0.80, >=0.85 size ratios and <=10% center shift indicate an unresolved same-footprint contradiction. All thresholds and per-group pair evidence are in JSON.", "", "Visual object identity cannot be proven from boxes alone; ambiguous matches remain for human review. No union, coordinate averaging, new boxes, or split reassignment occurs.", "", "## Previous TRUE_CONFLICT impact", "", "| Reclassified category | Groups | Current affected images | Current boxes | Retained images | Retained boxes | Discarded copies' boxes |", "|---|---:|---:|---:|---:|---:|---:|"]
    reason_lines = ["## Why groups remain unresolved", "", *[f'- {reason}: {count:,} groups' for reason, count in report["real_conflict_primary_reason_counts"].items()], "", "Primary reasons are mutually exclusive: cross-class contradiction, internal suspicious boxes, ambiguous matching, then unproven preservation. Detailed blocker counts in JSON may overlap. Geometric same-footprint class disagreement is a review signal, not visual adjudication. Failure to meet a preservation criterion is not proof an object is missing or mislabeled.", ""]
    reason_lines = ["The same-footprint cross-class contradiction check also applies inside a candidate annotation set, not only across member copies.", "", *reason_lines]
    insertion = lines.index("## Previous TRUE_CONFLICT impact")
    lines[insertion:insertion] = reason_lines
    for name, entry in report["previous_true_conflict_impact_by_category"].items():
        lines.append(f'| {name} | {entry["groups"]:,} | {entry["current"]["images"]:,} | {entry["current"]["boxes"]:,} | {entry["retained"]["images"]:,} | {entry["retained"]["boxes"]:,} | {entry["discarded"]["boxes"]:,} |')
    lines += ["", "| Class | Current affected boxes | Retained | Discarded | Gain vs actual current dataset | Gain vs deterministic single-copy reference |", "|---|---:|---:|---:|---:|---:|"]
    scope = report["previous_true_conflict_impact"]
    for name in NAMES:
        lines.append(f'| {name} | {scope["current"]["class_counts"][name]:,} | {scope["retained"]["class_counts"][name]:,} | {scope["discarded"]["class_counts"][name]:,} | 0 | {scope["gain_vs_deterministic_reference"]["class_counts"][name]:,} |')
    lines += ["", "**Current canonical-selection comparison is undefined:** existing conflict metadata has canonical=null and retains multiple copies. There are zero boxes gained relative to the actual current dataset because all selected sets already exist. For an explicitly hypothetical single-copy comparison only, reference = lexicographically first currently retained member per resolved group; gain against that reference is reported separately, never presented as actual new boxes. Unresolved groups are excluded from that hypothetical gain.", "", "## Projected final dataset", "", "| Metric | Current | Discarded with noncanonical copies | Projected | Retained % |", "|---|---:|---:|---:|---:|"]
    for name in ("images", "boxes", *NAMES):
        values = [entry[name] if name in ("images", "boxes") else entry["class_counts"][name] for entry in (result["current"], result["removed"], result["remaining"])]
        retention = result["retention_percent"][name] if name in ("images", "boxes") else result["retention_percent"]["class_counts"][name]
        lines.append(f'| {name} | {values[0]:,} | {values[1]:,} | {values[2]:,} | {retention:.3f}% |')
    lines += ["", "| Split | Images | Boxes | car | motorcycle | bus | truck |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in simulation.SPLITS:
        entry = result["remaining"]["per_split"][split]
        lines.append(f'| {split} | {entry["images"]:,} | {entry["boxes"]:,} | ' + " | ".join(f'{entry["class_counts"][name]:,}' for name in NAMES) + " |")
    lines += ["", f'Truck retention: {report["truck_retention_percent"]:.3f}%. MIO motorcycle retention: {report["mio_motorcycle_retention_percent"]:.3f}%. These are raw annotation-copy retention rates, not unique-object recall estimates; removing duplicate annotated copies lowers counts without necessarily losing object coverage.', "", "## Leakage and read-only safety", "", f'Current leakage: {report["leakage_current"]}. Projected leakage: {report["leakage_projected"]}. Split reassignment: zero.', "", "All splits retain all four target classes; validation/test retain both sources. Baseline statistics and candidate counts are reconciled to current metadata; all 20,004 affected positive label files and source GT are checked against Work 4.7A fingerprints. Source/cache and metadata fingerprints remain unchanged. No fresh exact/perceptual duplicate detection was performed.", "", "## Remaining review", "", f'{report["human_review_remaining"]:,} groups remain. {report["human_review_workload"]}', "", "See [final conflict QA](qa/final_real_conflict/README.md). This is representative QA, not exhaustive. Per-group candidates, failures, matching pairs, gains/reference definitions, split/source, retained/discarded counts and input checks are in work_4_7c_dryrun.json.", "", "## Final recommendation", "", report["final_recommendation"], "", "DO NOT APPLY CHANGES YET."]
    return "\n".join(lines) + "\n"


def main():
    print("Checking existing Work 4.7A source/label fingerprints (read only)...", flush=True)
    validation = audit.load_json(OUTPUT / "input_validation.json")
    expected = validation["input_sha256"]
    paths = sorted(set([ROOT / path for path in expected] + list((DATASET / "metadata").glob("*.json")) + [OUTPUT / "duplicate_resolution_manifest.json", OUTPUT / "work_4_7a_report.json", OUTPUT / "input_validation.json", ROOT / "data/vehicle_audit/vehicle363_build/image_scan.pkl"]))

    def snapshot():
        with ThreadPoolExecutor(max_workers=12) as executor:
            hashes = list(executor.map(audit.digest_file, paths))
        return {str(path.relative_to(ROOT)): digest for path, digest in zip(paths, hashes)}

    before = snapshot()
    if any(before[path] != digest for path, digest in expected.items()):
        raise RuntimeError("Source/current annotations changed since validated Work 4.7A")
    manifest = audit.load_json(OUTPUT / "duplicate_resolution_manifest.json")
    previous = audit.load_json(OUTPUT / "work_4_7a_report.json")
    duplicates = audit.load_json(DATASET / "metadata/duplicate_report.json")
    records = audit.load_json(DATASET / "metadata/source_manifest.json")
    statistics = audit.load_json(DATASET / "metadata/dataset_statistics.json")
    current = simulation.aggregate(records)
    simulation.verify_statistics(current, statistics)
    if len({record["final_image"] for record in records}) != len(records):
        raise RuntimeError("Repeated current image")
    by_member = {(record["source_dataset"], record["source_image"]): record for record in records}
    if len(by_member) != len(records):
        raise RuntimeError("Repeated current source member")
    groups = {group["group_id"]: group for group in duplicates["annotation_conflicts"] + duplicates.get("compatible_annotation_groups_requiring_review", [])}
    if len(groups) != len(manifest["groups"]) + len(manifest["supplemental_groups"]):
        raise RuntimeError("Existing conflict inventory disagrees")
    with (ROOT / "data/vehicle_audit/vehicle363_build/image_scan.pkl").open("rb") as handle:
        cache = pickle.load(handle)
    for group in groups.values():
        for member in group["members"]:
            key = (member["source"], member["image"])
            image_metadata = cache[key]
            if group["group_id"] != "exact__" + image_metadata["sha256"]:
                raise RuntimeError("Cached group identity disagrees")
            member["width"], member["height"] = image_metadata["width"], image_metadata["height"]
            member["member_id"] = audit.member_key(member)
            member["pixel_boxes"] = audit.pixel_boxes(member["target_annotations"], member["width"], member["height"])
            member["final_record"] = by_member.get(key)
            if not geometry_valid(member):
                raise RuntimeError(f'Invalid source target annotation: {member["member_id"]}')
            record = member["final_record"]
            if member["target_annotations"] and record is None:
                raise RuntimeError("Positive target member absent from current manifest")
            if record and (record["duplicate_group"] != group["group_id"] or record["target_bbox_count"] != len(member["pixel_boxes"]) or record["target_class_counts"] != dict(zip(NAMES, audit.bbox_counts(member["pixel_boxes"])))):
                raise RuntimeError("Current member annotation statistics disagree")
        if len({(member["width"], member["height"]) for member in group["members"]}) != 1:
            raise RuntimeError("Exact duplicates have inconsistent dimensions")
        if len({member["final_record"]["final_split"] for member in group["members"] if member["final_record"]}) != 1:
            raise RuntimeError("Conflict group spans splits")
    del cache
    true_previous = [group for group in manifest["groups"] if group["category"] == "TRUE_CONFLICT"]
    if len(true_previous) != 6650 or len(manifest["groups"]) != previous["total_conflict_groups"]:
        raise RuntimeError("Unexpected primary conflict inventory")
    print("Re-evaluating 6,650 previous TRUE_CONFLICT groups...", flush=True)
    results = []
    removed_paths = set()
    auto_prior = []
    for old in manifest["groups"] + manifest["supplemental_groups"]:
        group = groups[old["group_id"]]
        actual_ids = {member["member_id"] for member in group["members"]}
        if actual_ids != set(old["members"]):
            raise RuntimeError("Work 4.7A group members disagree")
        if old["category"] == "TRUE_CONFLICT":
            result = classify(group)
            results.append(result)
        else:
            if not old["auto_resolvable"] or old["confidence"] != "HIGH":
                raise RuntimeError("Prior auto-resolution lacks HIGH confidence")
            result = {"group_id": old["group_id"], "category": old["category"], "canonical_member": old["canonical_member"], "auto_resolvable": True, "scope": old["scope"]}
            auto_prior.append(result)
        current_members = [member for member in group["members"] if member["final_record"]]
        retained = current_members
        if result["auto_resolvable"]:
            canonical = next(member for member in group["members"] if member["member_id"] == result["canonical_member"])
            if canonical["final_record"] is None:
                raise RuntimeError("Selected canonical absent from current dataset")
            retained = [canonical]
            removed_paths.update(member["final_record"]["final_image"] for member in current_members if member is not canonical)
        if old["category"] == "TRUE_CONFLICT":
            discarded = [member for member in current_members if member not in retained]
            reference = min(current_members, key=lambda member: member["member_id"]) if result["auto_resolvable"] else None
            delta = {name: retained[0]["final_record"]["target_class_counts"][name] - reference["final_record"]["target_class_counts"][name] for name in NAMES} if reference else dict.fromkeys(NAMES, 0)
            if any(value < 0 for value in delta.values()):
                raise RuntimeError("Accepted candidate loses a reference class object")
            result.update({"split": current_members[0]["final_record"]["final_split"], "source_dataset": current_members[0]["final_record"]["source_dataset"], "source_member_count": len(group["members"]), "current": simulation.counts(member["final_record"] for member in current_members), "retained": simulation.counts(member["final_record"] for member in retained), "discarded": simulation.counts(member["final_record"] for member in discarded), "reference_member": reference["member_id"] if reference else None, "gain_vs_deterministic_reference": {"boxes": sum(delta.values()), "class_counts": delta}, "boxes_gained_vs_actual_current": 0, "boxes_gained_vs_current_canonical_selection": None})
    projected = simulation.scenario(records, removed_paths, current)
    remaining = [record for record in records if record["final_image"] not in removed_paths]
    baseline_leakage, projected_leakage = leakage(records), leakage(remaining)
    if any(baseline_leakage.values()) or any(projected_leakage.values()):
        raise RuntimeError("Structural leakage exists")
    categories = {category: sum(result["category"] == category for result in results) for category in ("MORE_COMPLETE_VALID", "JITTER_EQUIVALENT", "REAL_CONFLICT")}
    if sum(categories.values()) != 6650:
        raise RuntimeError("Reclassification total incorrect")
    blocker_counts = Counter()
    primary_reason_counts = Counter()
    reason_order = ("same_object_footprint_assigned_different_classes", "candidate_has_likely_duplicate_or_contradictory_boxes", "ambiguous_one_to_one_correspondence", "objects_not_provably_preserved")
    for result in results:
        if result["category"] == "REAL_CONFLICT":
            reasons = {reason for attempt in result["candidate_attempts"] for reason in attempt["reasons"]}
            blocker_counts.update(reasons)
            primary_reason_counts[next((reason for reason in reason_order if reason in reasons), "other_geometry_or_consistency_failure")] += 1

    def combined_impact(entries):
        summary = {"groups": len(entries)}
        for stage in ("current", "retained", "discarded"):
            summary[stage] = {"images": sum(entry[stage]["images"] for entry in entries), "boxes": sum(entry[stage]["boxes"] for entry in entries), "class_counts": {name: sum(entry[stage]["class_counts"][name] for entry in entries) for name in NAMES}}
        summary["gain_vs_deterministic_reference"] = {"boxes": sum(entry["gain_vs_deterministic_reference"]["boxes"] for entry in entries), "class_counts": {name: sum(entry["gain_vs_deterministic_reference"]["class_counts"][name] for entry in entries) for name in NAMES}}
        return summary

    print("Generating representative QA for unresolved groups...", flush=True)
    qa = generate_qa(groups, results)
    after = snapshot()
    if before != after:
        raise RuntimeError("Input bytes changed during dry-run")
    total_impact = combined_impact(results)
    if total_impact["current"]["images"] != 13943 or total_impact["current"]["boxes"] != 44650:
        raise RuntimeError("Previous TRUE_CONFLICT current counts disagree")
    for stage in ("images", "boxes"):
        if total_impact["current"][stage] != total_impact["retained"][stage] + total_impact["discarded"][stage]:
            raise RuntimeError("Count conservation failed")
    safety = projected["evaluation_safety"]
    if not all(safety["all_classes_present_by_split"].values()) or not all(safety["both_sources_present_in_val_test"].values()):
        raise RuntimeError("Projected evaluation representation lost")
    status = "REVIEW_REQUIRED" if categories["REAL_CONFLICT"] else "PASS"
    report = {"work": "4.7C", "status": status, "dry_run_only": True, "dataset_modified": False, "created_at_utc": datetime.now(timezone.utc).isoformat(), "previous_true_conflict": 6650, "reclassification_counts": categories, "auto_resolvable_previous_true_groups": 6650 - categories["REAL_CONFLICT"], "auto_resolvable_primary_groups": len([entry for entry in manifest["groups"] if entry["auto_resolvable"]]) + 6650 - categories["REAL_CONFLICT"], "supplemental_auto_groups": len(manifest["supplemental_groups"]), "human_review_remaining": categories["REAL_CONFLICT"], "human_review_workload": "Small enough for targeted individual review." if categories["REAL_CONFLICT"] <= 100 else "More than 100 groups: representative QA is only an initial triage; full review requires batches or explicit additional evidence, not blanket approval.", "matching_rules": RULES, "human_qa_evidence": {"provenance": "User-supplied statement in Work 4.7C request", "scope": "Representative samples only, not all 9,581 groups", "observations": ["bbox geometry visually acceptable", "additional boxes represented real objects", "additional target class labels correct", "no abnormal overlapping duplicate boxes observed"]}, "results": results, "prior_auto_resolutions": auto_prior, "previous_true_conflict_impact": total_impact, "previous_true_conflict_impact_by_category": {name: combined_impact([result for result in results if result["category"] == name]) for name in categories}, "current_canonical_baseline": {"available": False, "reason": "Current conflict groups have canonical=null and multiple retained copies", "boxes_gained_vs_actual_current_dataset": 0, "hypothetical_reference": "Lexicographically first currently retained member per resolved group only"}, "simulation": projected, "truck_retention_percent": projected["retention_percent"]["class_counts"]["truck"], "mio_motorcycle_retention_percent": projected["per_source"]["MIO"]["retention_percent"]["class_counts"]["motorcycle"], "leakage_current": baseline_leakage, "leakage_projected": projected_leakage, "qa": qa, "validation": {"work_4_7a_source_and_label_fingerprints_match": True, "current_positive_label_files_verified": validation["current_positive_member_labels_verified"], "source_gt_validated_by_unchanged_work_4_7a_evidence": True, "strict_candidate_geometry_checked": True, "current_statistics_reconciled": True, "input_fingerprints_unchanged": True, "input_sha256": before, "duplicate_detection_rerun": False, "split_reassignments": 0}, "final_recommendation": "Review unresolved groups and the bounded-jitter engineering rule before approving any application. Keep all REAL_CONFLICT members unchanged; do not drop entire groups. DO NOT APPLY CHANGES YET." if categories["REAL_CONFLICT"] else "All groups have proposed resolutions under the documented checks. Review the dry-run before authorizing application. DO NOT APPLY CHANGES YET."}
    report["real_conflict_primary_reason_counts"] = dict(primary_reason_counts)
    report["real_conflict_blocker_counts_nonexclusive"] = dict(blocker_counts)
    (OUTPUT / "work_4_7c_dryrun.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "work_4_7c_dryrun.md").write_text(report_markdown(report), encoding="utf-8")
    print(f"\nWORK 4.7C DRY-RUN STATUS: {status}\nPREVIOUS TRUE_CONFLICT: 6650\nRECLASSIFIED:")
    for category, count in categories.items():
        print(f"{category}: {count}")
    print(f'AUTO-RESOLVABLE: {report["auto_resolvable_primary_groups"]} primary + {report["supplemental_auto_groups"]} supplemental\nHUMAN REVIEW REMAINING: {categories["REAL_CONFLICT"]}\nPROJECTED FINAL:')
    for name in ("images", "boxes", *NAMES):
        print(f'{name}: {projected["remaining"][name] if name in ("images", "boxes") else projected["remaining"]["class_counts"][name]}')
    for split in simulation.SPLITS:
        print(f'{split.upper()}:\nimages: {projected["remaining"]["per_split"][split]["images"]}\nboxes: {projected["remaining"]["per_split"][split]["boxes"]}')
    print(f'TRUCK RETENTION: {report["truck_retention_percent"]:.3f}%\nMIO MOTORCYCLE RETENTION: {report["mio_motorcycle_retention_percent"]:.3f}%\nLEAKAGE: {projected_leakage}\nFINAL RECOMMENDATION: {report["final_recommendation"]}')


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, AssertionError, KeyError, ValueError, OSError) as error:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        failure = {"work": "4.7C", "status": "FAIL", "dry_run_only": True, "dataset_modified": False, "structural_or_integrity_error": str(error), "final_recommendation": "DO NOT APPLY CHANGES YET. Resolve the integrity problem before proceeding."}
        (OUTPUT / "work_4_7c_dryrun.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        (OUTPUT / "work_4_7c_dryrun.md").write_text(f'# Work 4.7C DRY RUN\n\nSTATUS: FAIL\n\n{error}\n\nDO NOT APPLY CHANGES YET.\n', encoding="utf-8")
        raise
