import json
from datetime import datetime, timezone

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_tracking_sources import AUDIT, ROOT, digest, inventory


def main():
    before = inventory()
    report_path = AUDIT / "work_5_1_report.json"
    markdown_path = AUDIT / "work_5_1_report.md"
    review_path = AUDIT / "work_5_1/human_review.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    reference = report["protected_and_track_annotation_sha256_after"]
    verified = {path: digest(ROOT / path) for path in reference}
    if verified != reference:
        raise RuntimeError("Protected release evidence or source GT changed since audit; stop for review")
    required = [AUDIT / "work_5_1/conflict_qa/index.html", AUDIT / "work_5_1/neighbor_qa/index.html"]
    for source in report["sources"]:
        required.extend(ROOT / path for path in source["previews"])
    if any(not path.is_file() for path in required):
        raise RuntimeError("QA output missing; cannot finalize audit")
    for directory in ("conflict_qa", "neighbor_qa"):
        manifest = json.loads((AUDIT / "work_5_1" / directory / "qa_manifest.json").read_text(encoding="utf-8"))
        if any(digest(ROOT / path) != expected for path, expected in manifest["source_sha256_after"].items()):
            raise RuntimeError("QA source hashes no longer match current files")
    after = inventory()
    if before != after:
        raise RuntimeError("Data workspace changed during read-only finalization")
    decisions = {item["check"]: item for item in review["observations"]}
    frame_two = decisions["VNTraffic frame 2 ID 31"]
    selected_box = frame_two.get("approved_bbox_to_keep")
    frame_two_summary = (f"VNTraffic frame 2 ID 31: same vehicle; owner selected bbox {selected_box}. Decision recorded, not applied; source remains unchanged."
                         if selected_box else "VNTraffic frame 2 ID 31: same vehicle; bbox A/B selection still unspecified.")
    frame_208 = decisions["VNTraffic frame 208 ID 49"]
    identity_A = frame_208.get("vehicle_A_existing_id")
    identity_B = frame_208.get("vehicle_B_existing_id")
    ids_selected = identity_A is not None and identity_B is not None and identity_A != identity_B
    frame_208_summary = (f"VNTraffic frame 208: owner confirmed A -> existing TXT ID {identity_A}, B -> existing TXT ID {identity_B}. Decision recorded, not applied."
                         if ids_selected else "VNTraffic frame 208 ID 49: two different vehicles remain unresolved under the same source ID.")
    pending = [
        "Confirmed VNTraffic repair decisions have not been applied; any future repair must use a separate authorized working GT, never overwrite the raw sources",
        "AICC22: stable/nonduplicate IDs approved only on representative samples; complete additional bbox geometry and final annotation edition remain unapproved",
        "Authoritative four-class semantic GT, frame alignment/coverage, and detector-training camera overlap must be resolved before a quantitative benchmark",
        "Long/archived sources and decode-damaged videos remain unverified expansion candidates, not approved continuous sequences"
    ]
    if not selected_box:
        pending.insert(0, frame_two_summary)
    if not ids_selected:
        pending.insert(0, frame_208_summary)
    report.update(status="REVIEW_REQUIRED", audit_execution_status="COMPLETE_READ_ONLY",
                  human_qa_status="REVIEW_REQUIRED", human_review=review,
                  pending_decisions=pending,
                  finalized_at_utc=datetime.now(timezone.utc).isoformat(),
                  read_only_finalization={"data_inventory_files_checked_before_after": len(before),
                                          "data_inventory_unchanged": True,
                                          "protected_and_annotation_sha256_current": verified,
                                          "source_video_sha256_rechecked_against_qa_manifests": True,
                                          "fresh_full_frozen_payload_hash_audit": False},
                  next="Work 5.2 preparation only after explicit QA/source policy decisions; no Tracking363 dataset creation or GT repair authorized")
    report["totals"]["human_approved_benchmark_frames"] = 0
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    heading = "## Final audit closeout and human feedback"
    markdown = markdown_path.read_text(encoding="utf-8").split(heading)[0].rstrip()
    block = [heading, "", "AUDIT_EXECUTION_STATUS: COMPLETE_READ_ONLY", "WORK_5_1_STATUS: REVIEW_REQUIRED", "HUMAN_QA_STATUS: REVIEW_REQUIRED", "", "The source discovery audit is complete. Human feedback is recorded; neither source annotations nor benchmark readiness are automatically approved.", "", "### Human confirmations (reviewed samples only)", "", "- Consecutive frames and bbox alignment: confirmed by owner.", "- Existing track IDs and neighbor QA: reported OK by owner, without explicit repair decisions.", "- VNTraffic frame 2 ID 31: same vehicle; bbox A/B selection still unspecified.", "- VNTraffic frame 208 ID 49: two vehicles; correct existing trajectory IDs still unspecified.", "- AICC22 additional object IDs: stable and nonduplicate on reviewed samples; all 1254 added boxes are NOT claimed manually approved.", "", "Human feedback record: [human_review.json](work_5_1/human_review.json)", "Neighbor decision QA: [index.html](work_5_1/neighbor_qa/index.html)", "AICC22 side-by-side QA: [index.html](work_5_1/conflict_qa/index.html)", "", "### Pending decisions", "", *[f"- {item}" for item in pending], "", "### Source strategy", "", "- VNTraffic: initial tracking mechanics/alignment pilot only; not yet an unbiased, complete four-class benchmark.", "- AICC22 JSON: more-complete candidate edition, not selected or remapped automatically.", "- AICity21: expansion footage candidate, counting labels are not persistent object track GT.", "- MIO/random detection images: do not construct artificial tracking sequences.", "- Preserve camera -> video -> continuous segment grouping and native source annotations.", "", "### Read-only finalization", "", f"Data size/mtime inventory unchanged for {len(before):,} files during finalization. Existing release controls/source annotation hashes and QA source-video hashes still match. This is not a new full frozen dataset payload hash verification.", "", "No source writes, annotation repair, ID generation, split, Tracking363 dataset creation, ByteTrack execution or model training performed by this audit closeout.", "", report["next"], ""]
    replacements = {
        "- VNTraffic frame 2 ID 31: same vehicle; bbox A/B selection still unspecified.": "- " + frame_two_summary,
        "- VNTraffic frame 208 ID 49: two vehicles; correct existing trajectory IDs still unspecified.": "- " + frame_208_summary,
        "- Existing track IDs and neighbor QA: reported OK by owner, without explicit repair decisions.": "- Existing track IDs and neighbor QA: reported OK on inspected samples; individual repair decisions recorded below."
    }
    block = [replacements.get(item, item) for item in block]
    markdown_path.write_text(markdown + "\n\n" + "\n".join(block), encoding="utf-8")
    for directory in ("conflict_qa", "neighbor_qa"):
        path = AUDIT / "work_5_1" / directory / "qa_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["human_feedback_record"] = "../human_review.json"
        manifest["human_qa_status"] = "REVIEW_REQUIRED"
        manifest["human_feedback_scope"] = review["scope"]
        if directory == "neighbor_qa":
            for case in manifest["cases"]:
                decision = decisions[f"VNTraffic frame {case['conflict_frame']} ID {case['existing_track_id']}"]
                case["human_decision"] = decision
                if case["conflict_frame"] == 2:
                    case["human_observation"] = frame_two_summary
                    case["decision_status"] = "DECISION_RECORDED_NOT_APPLIED" if selected_box else "REVIEW_REQUIRED"
                else:
                    case["human_observation"] = frame_208_summary
                    case["decision_status"] = "DECISION_RECORDED_NOT_APPLIED" if ids_selected else "REVIEW_REQUIRED_ID_COLLISION"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_execution": report["audit_execution_status"], "work_5_1_status": report["status"],
                      "source_modified": False, "human_feedback_recorded": True, "pending_decisions": pending}, indent=2))


if __name__ == "__main__":
    main()
