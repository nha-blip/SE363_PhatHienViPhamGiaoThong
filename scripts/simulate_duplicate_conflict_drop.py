import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/Vehicle363-v1"
OUTPUT = ROOT / "data/vehicle_audit/duplicate_resolution"
CLASSES = ("car", "motorcycle", "bus", "truck")
SPLITS = ("train", "val", "test")
SOURCES = ("HCMC", "MIO")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(records):
    result = {"images": 0, "boxes": 0, "class_counts": dict.fromkeys(CLASSES, 0)}
    for record in records:
        result["images"] += 1
        result["boxes"] += record["target_bbox_count"]
        for name in CLASSES:
            result["class_counts"][name] += record["target_class_counts"][name]
    assert result["boxes"] == sum(result["class_counts"].values())
    return result


def aggregate(records):
    split_records = defaultdict(list)
    source_records = defaultdict(list)
    domain_records = defaultdict(list)
    for record in records:
        split_records[record["final_split"]].append(record)
        source_records[record["source_dataset"]].append(record)
        domain = record["daytime_nighttime"] if record["source_dataset"] == "HCMC" else record["original_train_test_status"]
        assert domain is not None
        domain_records[f'{record["source_dataset"]}:{domain}'].append(record)
    result = counts(records)
    result["per_split"] = {}
    for split in SPLITS:
        entry = counts(split_records[split])
        entry["per_source"] = {
            source: counts(record for record in split_records[split] if record["source_dataset"] == source)
            for source in SOURCES
        }
        result["per_split"][split] = entry
    result["per_source"] = {source: counts(source_records[source]) for source in SOURCES}
    result["per_domain"] = {domain: counts(domain_records[domain]) for domain in ("HCMC:daytime", "HCMC:nighttime", "MIO:train", "MIO:test")}
    return result


def percentage(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def impact(current, removed, remaining):
    result = {"current": current, "removed": removed, "remaining": remaining}
    result["removal_percent"] = {
        "images": percentage(removed["images"], current["images"]),
        "boxes": percentage(removed["boxes"], current["boxes"]),
        "class_counts": {name: percentage(removed["class_counts"][name], current["class_counts"][name]) for name in CLASSES},
    }
    result["retention_percent"] = {
        "images": percentage(remaining["images"], current["images"]),
        "boxes": percentage(remaining["boxes"], current["boxes"]),
        "class_counts": {name: percentage(remaining["class_counts"][name], current["class_counts"][name]) for name in CLASSES},
    }
    result["class_flags"] = {
        name: "N/A" if result["removal_percent"]["class_counts"][name] is None else "CRITICAL" if result["removal_percent"]["class_counts"][name] > 20 else "WARNING" if result["removal_percent"]["class_counts"][name] > 10 else "OK"
        for name in CLASSES
    }
    for key in ("images", "boxes"):
        assert current[key] == removed[key] + remaining[key]
    for name in CLASSES:
        assert current["class_counts"][name] == removed["class_counts"][name] + remaining["class_counts"][name]
    return result


def scenario(records, removed_paths, current):
    removed = aggregate([record for record in records if record["final_image"] in removed_paths])
    remaining = aggregate([record for record in records if record["final_image"] not in removed_paths])
    result = impact(current, removed, remaining)
    for dimension in ("per_split", "per_source", "per_domain"):
        result[dimension] = {
            key: impact(current[dimension][key], removed[dimension][key], remaining[dimension][key])
            for key in current[dimension]
        }
    result["evaluation_safety"] = {
        "all_classes_present_by_split": {split: all(remaining["per_split"][split]["class_counts"][name] > 0 for name in CLASSES) for split in SPLITS},
        "both_sources_present_in_val_test": {split: all(remaining["per_split"][split]["per_source"][source]["images"] > 0 for source in SOURCES) for split in ("val", "test")},
        "minimum_remaining_boxes_per_class_across_splits": {name: min(remaining["per_split"][split]["class_counts"][name] for split in SPLITS) for name in CLASSES},
        "split_reassignments": 0,
    }
    result["source_share_changes"] = {}
    for split in ("overall", *SPLITS):
        before = current if split == "overall" else current["per_split"][split]
        after = remaining if split == "overall" else remaining["per_split"][split]
        shares = {}
        for source in SOURCES:
            shares[source] = {}
            for unit in ("images", "boxes"):
                initial = percentage(before["per_source"][source][unit], before[unit])
                final = percentage(after["per_source"][source][unit], after[unit])
                shares[source][unit] = {"current_percent": initial, "remaining_percent": final, "change_percentage_points": final - initial}
        result["source_share_changes"][split] = shares
    return result


def verify_statistics(current, statistics):
    for section in (None, "per_split", "per_source"):
        pairs = [(current, statistics)] if section is None else [(current[section][key], statistics[section][key]) for key in current[section]]
        for actual, expected in pairs:
            for key in ("images", "boxes", "class_counts"):
                assert actual[key] == expected[key], (section, key, actual[key], expected[key])
    for split in SPLITS:
        for source in SOURCES:
            for key in ("images", "boxes"):
                assert current["per_split"][split]["per_source"][source][key] == statistics["per_split"][split]["source_counts"][source][key]
    for domain in ("daytime", "nighttime"):
        for key in ("images", "boxes", "class_counts"):
            assert current["per_domain"][f"HCMC:{domain}"][key] == statistics["hcmc_day_night"][domain][key]


def format_percent(value):
    return "N/A" if value is None else f"{value:.3f}%"


def render_report(report):
    drop = report["drop_only"]
    combined = report["combined_final_policy"]
    lines = ["# Work 4.7B — TRUE_CONFLICT drop simulation", "", f'**WORK 4.7B STATUS: {report["status"]}**', "", "SIMULATION ONLY. No image, label, split assignment, source annotation, or Vehicle363-v1 file was changed. No duplicate detection or build was rerun.", "", "## Scope and reconciliation", "", f'Primary groups: {report["primary_category_counts"]}. One supplemental HIGH-confidence compatible-subset group is canonicalized only in the combined scenario; it is not part of the 9,581 primary groups.', "", f'TRUE_CONFLICT groups: {len(report["true_conflict_groups"]):,}; source members: {report["true_conflict_source_members"]:,}; currently retained images: {drop["removed"]["images"]:,}; already-absent source members: {report["true_conflict_absent_source_members"]:,}. Absent members and source aliases are never counted as additional removals.', "", "Counts come from actual current source_manifest records, independently reconciled against dataset_statistics and Work 4.7A. Per-group split/source/member/image/box/class details are in the companion JSON. Image/label validation and matching evidence are inherited from Work 4.7A; this simulation does not reopen image or annotation content.", "", "## Drop-only overall and class impact", "", "| Metric | Current | Removed | Remaining | Removed % | Retained % | Flag |", "|---|---:|---:|---:|---:|---:|---|"]
    for metric in ("images", "boxes", *CLASSES):
        values = [entry[metric] if metric in ("images", "boxes") else entry["class_counts"][metric] for entry in (drop["current"], drop["removed"], drop["remaining"])]
        loss = drop["removal_percent"][metric] if metric in ("images", "boxes") else drop["removal_percent"]["class_counts"][metric]
        retention = drop["retention_percent"][metric] if metric in ("images", "boxes") else drop["retention_percent"]["class_counts"][metric]
        lines.append(f'| {metric} | {values[0]:,} | {values[1]:,} | {values[2]:,} | {format_percent(loss)} | {format_percent(retention)} | {drop["class_flags"].get(metric, "—")} |')
    lines += ["", "## Split impact — drop only", "", "| Split | Current images | Removed images | Remaining images | Current boxes | Removed boxes | Remaining boxes | car | motorcycle | bus | truck |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        entry = drop["per_split"][split]
        values = [entry[stage][unit] for unit in ("images", "boxes") for stage in ("current", "removed", "remaining")]
        values += [entry["remaining"]["class_counts"][name] for name in CLASSES]
        lines.append(f'| {split} | ' + " | ".join(f"{value:,}" for value in values) + " |")
    lines += ["", "## Source and domain impact — drop only", "", "| Source/domain | Current images | Removed images | Remaining images | Images removed % | Current boxes | Removed boxes | Remaining boxes | Boxes removed % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dimension in ("per_source", "per_domain"):
        for name, entry in drop[dimension].items():
            lines.append(f'| {name} | {entry["current"]["images"]:,} | {entry["removed"]["images"]:,} | {entry["remaining"]["images"]:,} | {format_percent(entry["removal_percent"]["images"])} | {entry["current"]["boxes"]:,} | {entry["removed"]["boxes"]:,} | {entry["remaining"]["boxes"]:,} | {format_percent(entry["removal_percent"]["boxes"])} |')
    lines += ["", "All TRUE_CONFLICT removals are MIO. MIO:train/test refer to original source partitions, not final split names. Original MIO test has no local ground truth and contributes zero current images; its loss rate is N/A, not 0%. HCMC day/night remain unchanged. Camera-level or visual-domain balance for MIO cannot be inferred from these metadata.", "", "## Combined final policy", "", f'Canonicalization additionally removes {report["auto_resolution"]["removed"]["images"]:,} noncanonical images / {report["auto_resolution"]["removed"]["boxes"]:,} boxes. No boxes are added, averaged, or unioned. The HIGH-confidence canonical sets already exist in the current dataset.', "", "| Metric | Projected final | Combined removed | Removed % | Retained % | Flag |", "|---|---:|---:|---:|---:|---|"]
    for metric in ("images", "boxes", *CLASSES):
        final = combined["remaining"][metric] if metric in ("images", "boxes") else combined["remaining"]["class_counts"][metric]
        removed = combined["removed"][metric] if metric in ("images", "boxes") else combined["removed"]["class_counts"][metric]
        loss = combined["removal_percent"][metric] if metric in ("images", "boxes") else combined["removal_percent"]["class_counts"][metric]
        retention = combined["retention_percent"][metric] if metric in ("images", "boxes") else combined["retention_percent"]["class_counts"][metric]
        lines.append(f'| {metric} | {final:,} | {removed:,} | {format_percent(loss)} | {format_percent(retention)} | {combined["class_flags"].get(metric, "—")} |')
    lines += ["", "| Split | Images | Boxes | car | motorcycle | bus | truck | HCMC images | MIO images |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        entry = combined["remaining"]["per_split"][split]
        values = [entry["images"], entry["boxes"], *[entry["class_counts"][name] for name in CLASSES], *[entry["per_source"][source]["images"] for source in SOURCES]]
        lines.append(f'| {split} | ' + " | ".join(f"{value:,}" for value in values) + " |")
    lines += ["", "## Evaluation and domain safety", "", "Both scenarios retain all four classes in all splits and both HCMC/MIO in validation/test. No split reassignment is simulated.", "", "| Scenario / split | HCMC image share before | After | Change (pp) | HCMC box share before | After | Change (pp) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, result in (("Drop", drop), ("Combined", combined)):
        for split, entry in result["source_share_changes"].items():
            shares = entry["HCMC"]
            lines.append(f'| {label} / {split} | {shares["images"]["current_percent"]:.3f}% | {shares["images"]["remaining_percent"]:.3f}% | {shares["images"]["change_percentage_points"]:.3f} | {shares["boxes"]["current_percent"]:.3f}% | {shares["boxes"]["remaining_percent"]:.3f}% | {shares["boxes"]["change_percentage_points"]:.3f} |')
    lines += ["", "## Source-specific class imbalance", "", "| Source / class | Current boxes | Drop-only removed % | Drop-only remaining | Combined removed % | Combined remaining |", "|---|---:|---:|---:|---:|---:|"]
    for source in SOURCES:
        for name in CLASSES:
            initial = drop["per_source"][source]["current"]["class_counts"][name]
            drop_entry = drop["per_source"][source]
            final_entry = combined["per_source"][source]
            lines.append(f'| {source} / {name} | {initial:,} | {format_percent(drop_entry["removal_percent"]["class_counts"][name])} | {drop_entry["remaining"]["class_counts"][name]:,} | {format_percent(final_entry["removal_percent"]["class_counts"][name])} | {final_entry["remaining"]["class_counts"][name]:,} |')
    lines += ["", "These percentages use each source's own class denominator, not the whole dataset. Large HCMC motorcycle counts can mask substantial loss of MIO motorcycles in the global balance. Per-source class counts for each split are also preserved in JSON.", "", "## Decision and bus impact", "", *[f"- {reason}" for reason in report["decision_reasons"]], "", "User thresholds: WARNING for class loss >10%; CRITICAL for >20%. SAFE_TO_DROP requires no such class warning in either scenario. Loss of a class in any split or either source in validation/test is treated as UNSAFE_TO_DROP. Other concerning metrics require REVIEW_REQUIRED; a CRITICAL rate alone is not automatically proof of unusability.", "", "No numeric definition of 'well represented', bus sufficiency, or serious domain imbalance was supplied. Remaining counts, minimum per-split class counts, source loss rates, and source-share changes are provided for human judgment; presence alone does not prove statistical sufficiency. MIO camera/day-night data are unavailable. The decision does not claim qualitative TRUE_CONFLICT annotations are all incorrect; Work 4.7A's conservative match rejection may include small-object jitter.", "", f'NEXT: {report["next"]}', "", "DO NOT MODIFY DATASET.", "", "## Read-only verification", "", "SHA-256 of all current Vehicle363-v1 metadata files plus final_qa_report.json, and both Work 4.7A input files, remained unchanged before/after computation. The script writes only the two Work 4.7B outputs in this audit directory; no dataset file operations are performed. Full input fingerprints and arithmetic/evaluation checks are in JSON."]
    return "\n".join(lines) + "\n"


def main():
    paths = list((DATASET / "metadata").rglob("*")) + [DATASET / "final_qa_report.json", OUTPUT / "duplicate_resolution_manifest.json", OUTPUT / "work_4_7a_report.json"]
    paths = sorted(path for path in paths if path.is_file())
    before = {str(path.relative_to(ROOT)): fingerprint(path) for path in paths}
    records = json.loads((DATASET / "metadata/source_manifest.json").read_text(encoding="utf-8"))
    statistics = json.loads((DATASET / "metadata/dataset_statistics.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUTPUT / "duplicate_resolution_manifest.json").read_text(encoding="utf-8"))
    previous = json.loads((OUTPUT / "work_4_7a_report.json").read_text(encoding="utf-8"))
    current = aggregate(records)
    verify_statistics(current, statistics)
    grouped = defaultdict(list)
    assert len({record["final_image"] for record in records}) == len(records)
    for record in records:
        assert record["final_split"] in SPLITS and record["source_dataset"] in SOURCES
        assert record["target_bbox_count"] == sum(record["target_class_counts"].values())
        grouped[record["duplicate_group"]].append(record)
    categories = dict(Counter(group["category"] for group in manifest["groups"]))
    assert len(manifest["groups"]) == previous["total_conflict_groups"]
    assert categories == {key: entry["groups"] for key, entry in previous["category_statistics"].items()}
    all_groups = manifest["groups"] + manifest["supplemental_groups"]
    assert len({group["group_id"] for group in all_groups}) == len(all_groups)
    true_paths = set()
    auto_paths = set()
    true_groups = []
    auto_groups = []
    for group in all_groups:
        members = grouped[group["group_id"]]
        assert members, group["group_id"]
        assert len({record["final_split"] for record in members}) == 1
        assert len({record["source_dataset"] for record in members}) == 1
        present = {f'{record["source_dataset"]}:{record["source_image"]}' for record in members}
        assert present <= set(group["members"])
        assert len(present) == len(members)
        for record in members:
            member_id = f'{record["source_dataset"]}:{record["source_image"]}'
            assert record["target_bbox_count"] == group["annotation_counts"][member_id]
        if group["category"] == "TRUE_CONFLICT":
            assert group["canonical_member"] is None and not group["auto_resolvable"]
            true_paths.update(record["final_image"] for record in members)
            true_groups.append({"group_id": group["group_id"], "scope": group["scope"], "split": members[0]["final_split"], "source_dataset": members[0]["source_dataset"], "source_member_images": len(group["members"]), "currently_retained_images": len(members), "already_absent_source_members": sorted(set(group["members"]) - present), "current_bbox_count": counts(members)["boxes"], "class_counts": counts(members)["class_counts"], "current_final_images": [record["final_image"] for record in members]})
        else:
            assert group["category"] in ("SAME_OBJECTS_JITTER", "SUPERSET_ANNOTATION")
            assert group["confidence"] == "HIGH" and group["auto_resolvable"]
            assert group["canonical_member"] in present
            noncanonical = [record for record in members if f'{record["source_dataset"]}:{record["source_image"]}' != group["canonical_member"]]
            auto_paths.update(record["final_image"] for record in noncanonical)
            auto_groups.append({"group_id": group["group_id"], "scope": group["scope"], "category": group["category"], "canonical_member": group["canonical_member"], "split": members[0]["final_split"], "removed": counts(noncanonical)})
    assert not true_paths & auto_paths
    drop = scenario(records, true_paths, current)
    auto = scenario(records, auto_paths, current)
    combined = scenario(records, true_paths | auto_paths, current)
    assert len(true_groups) == categories["TRUE_CONFLICT"]
    for key, previous_key in (("images", "positive_target_images"), ("boxes", "boxes_involved")):
        assert drop["removed"][key] == previous["category_statistics"]["TRUE_CONFLICT"][previous_key]
    for key in ("images", "boxes", "class_counts"):
        assert auto["remaining"][key] == previous["projected_after_all_high_confidence_recommendations"]["projected"][key]
    for key in ("images", "boxes"):
        assert combined["removed"][key] == drop["removed"][key] + auto["removed"][key]
    assert drop["removed"]["per_source"]["HCMC"]["images"] == 0
    reasons = []
    unsafe = False
    for label, result in (("Drop only", drop), ("Combined final policy", combined)):
        safety = result["evaluation_safety"]
        if not all(safety["all_classes_present_by_split"].values()) or not all(safety["both_sources_present_in_val_test"].values()):
            unsafe = True
            reasons.append(f"{label}: required class/source representation is lost.")
        for name, flag in result["class_flags"].items():
            if flag != "OK":
                reasons.append(f'{label}: {name} loses {format_percent(result["removal_percent"]["class_counts"][name])} ({flag}); {result["remaining"]["class_counts"][name]:,} boxes remain.')
    for label, result in (("Drop only", drop), ("Combined", combined)):
        minimum = result["evaluation_safety"]["minimum_remaining_boxes_per_class_across_splits"]["bus"]
        reasons.append(f'{label}: bus remaining {result["remaining"]["class_counts"]["bus"]:,}; minimum across splits {minimum:,}.')
    concerning = any(flag != "OK" for result in (drop, combined) for flag in result["class_flags"].values())
    decision = "UNSAFE_TO_DROP" if unsafe else "REVIEW_REQUIRED"
    if not concerning and not unsafe:
        reasons.append("No class exceeds the 10% loss threshold, but quantitative sufficiency/domain acceptance needs human confirmation because no minimum-count or domain-imbalance criterion was supplied.")
    reasons.append("MIO alone bears the removals; source partition counts do not establish camera-level or visual-domain safety. Validation/test source and class presence are checked independently.")
    for label, result in (("Drop only", drop), ("Combined", combined)):
        losses = result["per_source"]["MIO"]["removal_percent"]["class_counts"]
        reasons.append(f'{label}: within MIO, motorcycle loss is {format_percent(losses["motorcycle"])} and truck loss is {format_percent(losses["truck"])}; source-specific class/domain imbalance requires review even when global motorcycle loss is small.')
    after = {str(path.relative_to(ROOT)): fingerprint(path) for path in paths}
    assert before == after, "An input changed during simulation"
    report = {
        "work": "4.7B", "status": decision, "final_decision": decision,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_only": True, "dataset_modified": False, "split_reassignments": 0,
        "duplicate_detection_rerun": False, "primary_category_counts": categories,
        "true_conflict_source_members": sum(group["source_member_images"] for group in true_groups),
        "true_conflict_absent_source_members": sum(len(group["already_absent_source_members"]) for group in true_groups),
        "true_conflict_groups": true_groups, "auto_resolution_groups": auto_groups,
        "supplemental_auto_resolution_groups": len(manifest["supplemental_groups"]),
        "current": current, "drop_only": drop, "auto_resolution": auto,
        "combined_final_policy": combined, "decision_reasons": reasons,
        "decision_criteria": {"warning_class_loss_percent_strictly_greater_than": 10, "critical_class_loss_percent_strictly_greater_than": 20, "unsafe_if_required_class_or_evaluation_source_disappears": True, "qualitative_sufficiency_and_domain_thresholds": "Not supplied; human review required, no invented cutoff."},
        "validation": {"current_statistics_reconciled": True, "work_4_7a_counts_reconciled": True, "canonical_members_currently_present": True, "group_splits_atomic": True, "removal_sets_disjoint": True, "count_conservation": True, "input_fingerprints_unchanged": True, "input_sha256": before},
        "limitations": ["No fresh image/label content validation; uses previously validated actual current metadata.", "No authoritative MIO camera or day/night domain labels are available.", "Source aliases and absent members do not count as removals.", "TRUE_CONFLICT is a conservative Work 4.7A matching category, not proof every member annotation is semantically wrong."],
        "next": "Review class loss and domain impact before authorizing Work 4.7C. DO NOT MODIFY DATASET.",
    }
    for filename, content in (("work_4_7b_drop_simulation.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"), ("work_4_7b_drop_simulation.md", render_report(report))):
        destination = OUTPUT / filename
        assert DATASET not in destination.resolve().parents
        destination.write_text(content, encoding="utf-8")
    print(f"WORK 4.7B STATUS: {decision}\nTRUE_CONFLICT GROUPS: {len(true_groups)}")
    for label, entry in (("CURRENT", current), ("REMOVED (TRUE_CONFLICT ONLY)", drop["removed"]), ("REMOVAL RATE (TRUE_CONFLICT ONLY)", drop["removal_percent"]), ("PROJECTED FINAL (COMBINED)", combined["remaining"])):
        print(f"\n{label}:")
        for metric in ("images", "boxes", *CLASSES):
            value = entry[metric] if metric in ("images", "boxes") else entry["class_counts"][metric]
            print(f"{metric}: {format_percent(value) if label.startswith('REMOVAL RATE') else value}")
    for split in SPLITS:
        print(f"\n{split.upper()}:")
        entry = combined["remaining"]["per_split"][split]
        for metric in ("images", "boxes", *CLASSES):
            print(f'{metric}: {entry[metric] if metric in ("images", "boxes") else entry["class_counts"][metric]}')
    print("\nSOURCE IMPACT (TRUE_CONFLICT ONLY):")
    for source, entry in drop["per_source"].items():
        print(f'{source}: images removed {entry["removed"]["images"]} ({format_percent(entry["removal_percent"]["images"])}); boxes removed {entry["removed"]["boxes"]} ({format_percent(entry["removal_percent"]["boxes"])})')
    print("\nBUS IMPACT:\n" + "\n".join(reason for reason in reasons if "bus" in reason.lower()))
    print(f'\nDATA RETENTION (COMBINED):\nimages: {format_percent(combined["retention_percent"]["images"])}\nboxes: {format_percent(combined["retention_percent"]["boxes"])}')
    print(f'\nFINAL DECISION: {decision}\nNEXT: {report["next"]}')


if __name__ == "__main__":
    main()
