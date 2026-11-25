import csv
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "runs/project363/tracking/class_preannotation_vntraffic_001"
FIELDS = ["track_id", "candidate_class", "suggested_class", "matched_observations", "majority_ratio", "mixed_predictions", "approved_class", "decision", "notes"]
CLASSES = {"car", "motorcycle", "bus", "truck"}


def normalize_row(values):
    if len(values) >= 10 and not values[6].strip() and values[7].strip() == "REVIEW_REQUIRED":
        approved, decision = values[8:10]
        notes = ",".join(values[10:]).strip()
    elif len(values) == len(FIELDS):
        approved, decision, notes = values[6:9]
    else:
        raise ValueError(f"Ambiguous CSV layout for track {values[0] if values else 'missing'}")
    approved = approved.strip().lower()
    if approved == "motor":
        approved = "motorcycle"
    decision = decision.strip().upper()
    if decision == "APPROVE":
        decision = "APPROVED"
    if decision not in {"APPROVED", "REVIEW_REQUIRED"} or approved and approved not in CLASSES:
        raise ValueError(f"Invalid decision/class for track {values[0]}")
    if decision == "APPROVED" and not approved:
        raise ValueError(f"Approved class missing for track {values[0]}")
    return dict(zip(FIELDS, values[:6] + [approved, decision, notes.strip()]))


def main():
    source = DIRECTORY / "human_review.csv"
    initial_hash = sha256(source)
    rows = list(csv.reader(io.StringIO(source.read_text(encoding="utf-8-sig"))))
    if rows[0] != FIELDS:
        raise ValueError("Unexpected review header")
    normalized = [normalize_row(values) for values in rows[1:] if values]
    baseline = {str(row["track_id"]): row for row in json.loads((DIRECTORY / "track_class_suggestions.json").read_text(encoding="utf-8"))}
    identities = [row["track_id"] for row in normalized]
    if len(identities) != len(set(identities)) or set(identities) != set(baseline):
        raise ValueError("Missing, repeated, or unknown track IDs")
    for row in normalized:
        saved = baseline[row["track_id"]]
        if row["candidate_class"] != (saved["candidate_class"] or "") or row["suggested_class"] != (saved["suggested_class"] or "") or int(row["matched_observations"]) != saved["matched_observations"] or float(row["majority_ratio"]) != saved["majority_ratio"] or row["mixed_predictions"] != str(saved["mixed_predictions"]):
            raise ValueError(f"Prediction columns changed for track {row['track_id']}")
    output = DIRECTORY / "human_review.normalized.csv"
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    issues = [{"track_id": int(row["track_id"]), "approved_class": row["approved_class"], "note": row["notes"], "bbox_review_status": "REVIEW_REQUIRED"} for row in normalized if "bbox" in row["notes"].lower()]
    report = {"status": "NORMALIZED_NOT_APPLIED", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(source), "source_sha256": initial_hash, "normalized_csv": str(output), "normalized_sha256": sha256(output),
              "tracks": len(normalized), "approved_tracks": sum(row["decision"] == "APPROVED" for row in normalized),
              "approved_class_counts": dict(Counter(row["approved_class"] for row in normalized if row["decision"] == "APPROVED")),
              "aliases": {"motor": "motorcycle", "approve": "APPROVED"},
              "layout_repair": "Moved explicitly appended owner labels/decisions into their proper columns; original CSV preserved",
              "bbox_review_issues": issues, "gt_modified": False, "semantic_classes_applied_to_gt": False,
              "overall_dataset_qa_status": "REVIEW_REQUIRED" if issues else "NOT_EVALUATED"}
    if sha256(source) != initial_hash:
        raise RuntimeError("Owner CSV changed during normalization")
    with (DIRECTORY / "human_review_validation.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
