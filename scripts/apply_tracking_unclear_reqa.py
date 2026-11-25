"""Apply the re-QA of the "unclear" items and close the vntraffic_001 MOT 201-300 label revision.

* Track answered "mixed" whose switch frames cannot be resolved on the blurry video: its boxes in the
  window become ignore regions (no GT, not scored) instead of guessing the split.
* Candidates answered "multi_vehicle_box": their detector boxes become ignore regions.
* Tracks answered "one": unchanged.
Crowd (multi-vehicle) regions stay ignore regions by policy; individual vehicles there are not labelled.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tracking_unclear_reqa import BASE, DETECTORS, TRACKING  # noqa: E402

SOURCE = BASE / "vntraffic_001_mot201_300_after_missing_qa"
QA = BASE / "vntraffic_001_mot201_300_unclear_reqa"
OUTPUT = BASE / "vntraffic_001_mot201_300_reviewed_provisional"
MIXED_WINDOW = (231, 300)


def main():
    draft = json.loads((SOURCE / "annotations_draft.json").read_text(encoding="utf-8"))
    answers = json.loads((QA / "tracking_track_qa_result.json").read_text(encoding="utf-8"))
    page = json.loads(__import__("re").search(r"const DATA = (\{.*?\});\nconst KEY", (QA / "track_qa.html").read_text(encoding="utf-8"), __import__("re").S).group(1))
    windows = {m["key"]: m["window"] for m in page["missing"]}
    ignore, moved = list(draft.get("ignore_regions") or []), {}
    mixed = {int(k) for k, v in answers["tracks"].items() if v["label"] == "mixed"}
    for frame in draft["frames"]:
        if MIXED_WINDOW[0] <= frame["mot_frame_id"] <= MIXED_WINDOW[1]:
            keep = []
            for o in frame["objects"]:
                if o["track_id"] in mixed:
                    ignore.append({"frame": frame["mot_frame_id"], "bbox_xywh": o["bbox_xywh"], "source": f"gt_track_{o['track_id']}_mixed_unresolved"})
                    moved[o["track_id"]] = moved.get(o["track_id"], 0) + 1
                else:
                    keep.append(o)
            frame["objects"] = keep
    added = 0
    for key, value in answers["missing"].items():
        if value["label"] != "multi_vehicle_box":
            continue
        detector, identity = key.split(":")
        first, last = windows[key]
        for line in (TRACKING / DETECTORS[detector] / "predictions_mot.txt").read_text().splitlines():
            p = line.split(",")
            if int(p[1]) == int(identity) and first <= int(p[0]) <= last:
                ignore.append({"frame": int(p[0]), "bbox_xywh": [float(v) for v in p[2:6]], "source": key})
                added += 1
    draft["ignore_regions"] = ignore
    draft["ignore_region_policy"] = ("crowd boxes enclosing several vehicles and tracks judged mixed but unresolvable on this blurry "
                                     "video are ignore regions: a prediction matching only an ignore region is neither TP nor FP")
    draft["draft_revision"] = "reviewed_provisional_201_300"
    draft["status"] = "REVIEWED_PROVISIONAL"
    draft["is_ground_truth"] = False
    draft["review_summary"] = {
        "window": [201, 300], "track_qa": ["201-230", "231-300"], "frame_split_qa": ["201-230", "231-300"],
        "missing_vehicle_search": "candidates from 4 detectors (e30, e41_960, e63_960, v8s_960)",
        "unclear_items_reasked": True, "limits": ["vehicles missed by all four detectors may still be absent",
                                                  "semantic classes not labelled", "blurry 1080p source"]}
    draft["source_draft_sha256"] = hashlib.sha256((SOURCE / "annotations_draft.json").read_bytes()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "annotations_draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {"mixed_tracks_moved_to_ignore": moved, "multi_vehicle_ignore_boxes_added": added, "ignore_region_boxes_total": len(ignore),
               "gt_boxes": sum(len(f["objects"]) for f in draft["frames"] if 201 <= f["mot_frame_id"] <= 300)}
    (OUTPUT / "qa_applied.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
