"""Apply the MOT 231-300 track QA to a new copy of the (201-230 QA'd) vntraffic_001 draft.

* extra_frames from the frame-level split QA are removed (track 1602: whole window, per the user's note);
* pairs answered "same" are merged (lower ID kept);
* YOLOv8s tracks answered "missing_vehicle" are added, only in frames the draft did not cover,
  with new IDs 2000 + v8s ID. Their boxes are detector boxes, not hand-drawn.
"Unclear" answers are left untouched and listed in qa_applied.json.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tracking_track_qa as builder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/Tracking363_working/annotation_drafts"
DRAFT = BASE / "vntraffic_001_mot201_300_after_track_qa"
QA = BASE / "vntraffic_001_mot231_300_track_qa"
OUTPUT = BASE / "vntraffic_001_mot201_300_after_track_qa_231_300"
FIRST_FRAME, LAST_FRAME = 231, 300
WHOLE_WINDOW_REMOVAL = {1602: "ghi chú người dùng: 'tất cả đều thừa'"}
NEW_ID_OFFSET = 2000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true", help="Write the new draft; without it only the plan is printed")
    arguments = parser.parse_args()
    source = DRAFT / "annotations_draft.json"
    draft = json.loads(source.read_text(encoding="utf-8"))
    track_qa = json.loads((QA / "tracking_track_qa_result.json").read_text(encoding="utf-8"))
    split_qa = json.loads((QA / "track_split_qa_result.json").read_text(encoding="utf-8"))
    page = json.loads(__import__("re").search(r"const DATA = (\{.*?\});\nconst KEY", (QA / "track_qa.html").read_text(encoding="utf-8"), __import__("re").S).group(1))

    mixed = {k for k, v in track_qa["tracks"].items() if v["label"] == "mixed"}
    if mixed - set(split_qa) or any(not split_qa[k]["status"] for k in mixed):
        raise RuntimeError("Frame-level QA incomplete")
    if any(entry["switch_frames"] for entry in split_qa.values()):
        raise RuntimeError("ID-switch frames were marked; splitting is not implemented")

    removals = {(int(t), f) for t, e in split_qa.items() for f in e["extra_frames"]}
    for track in WHOLE_WINDOW_REMOVAL:
        removals |= {(track, fr["mot_frame_id"]) for fr in draft["frames"] if FIRST_FRAME <= fr["mot_frame_id"] <= LAST_FRAME
                     for o in fr["objects"] if o["track_id"] == track}

    same = [tuple(sorted(map(int, k.split("-")))) for k, v in track_qa["pairs"].items() if v["label"] == "same"]
    mapping = {b: a for a, b in same}

    builder.FIRST_FRAME, builder.LAST_FRAME = FIRST_FRAME, LAST_FRAME
    candidates = builder.missing_candidates(draft, Path(page["missing_source"]))
    confirmed = sorted(int(k) for k, v in track_qa["missing"].items() if v["label"] == "missing_vehicle")
    additions = {}
    for identity in confirmed:
        for frame, box, _, _ in candidates[identity]:
            additions.setdefault(frame, []).append({"track_id": NEW_ID_OFFSET + identity, "bbox_xywh": [round(v, 2) for v in box], "class": None,
                                                    "annotation_origin": "V8S_TRACK_HUMAN_CONFIRMED_MISSING", "human_corrected": False})

    result = copy.deepcopy(draft)
    removed, remapped, added, duplicates_dropped = [], Counter(), Counter(), []
    for frame in result["frames"]:
        kept = []
        for item in frame["objects"]:
            if (item["track_id"], frame["mot_frame_id"]) in removals:
                removed.append((item["track_id"], frame["mot_frame_id"]))
                continue
            if item["track_id"] in mapping:
                remapped[f"{item['track_id']}->{mapping[item['track_id']]}"] += 1
                item["_merged_from"] = item["track_id"]
                item["track_id"] = mapping[item["track_id"]]
            kept.append(item)
        # A merged pair that was a duplicate box on one vehicle: keep the box of the kept (lower) ID.
        present = {item["track_id"] for item in kept if not item.get("_merged_from")}
        deduped = []
        for item in kept:
            if item.get("_merged_from") and item["track_id"] in present:
                duplicates_dropped.append((item.pop("_merged_from"), frame["mot_frame_id"]))
                continue
            item.pop("_merged_from", None)
            deduped.append(item)
        kept = deduped
        for item in additions.get(frame["mot_frame_id"], []):
            kept.append(item)
            added[item["track_id"]] += 1
        ids = Counter(item["track_id"] for item in kept)
        if any(c > 1 for c in ids.values()):
            raise RuntimeError(f"Duplicate IDs in frame {frame['mot_frame_id']}: {[i for i, c in ids.items() if c > 1]}")
        frame["objects"] = kept
    if removals - set(removed):
        raise RuntimeError(f"Marked boxes not found: {sorted(removals - set(removed))}")

    plan = {"removed_boxes": len(removed), "removed_by_track": dict(Counter(str(t) for t, _ in removed)),
            "whole_window_removals": {str(k): v for k, v in WHOLE_WINDOW_REMOVAL.items()},
            "merged_pairs": same, "boxes_reassigned": dict(remapped),
            "duplicate_boxes_dropped_on_merge": {str(k): [f for t, f in duplicates_dropped if t == k] for k in {t for t, _ in duplicates_dropped}},
            "added_tracks": {str(k): n for k, n in added.items()}, "added_boxes": sum(added.values()),
            "left_unclear": {"tracks": [k for k, v in track_qa["tracks"].items() if v["label"] == "unclear"],
                             "missing": [k for k, v in track_qa["missing"].items() if v["label"] == "unclear"],
                             "multi_vehicle_box_candidates_not_added": [k for k, v in track_qa["missing"].items() if v["label"] == "multi_vehicle_box"]}}
    print(json.dumps(plan, indent=1))
    if not arguments.execute:
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result["draft_revision"] = "after_track_qa_201_230_and_231_300"
    result["source_draft_sha256"] = sha256(source)
    result["status"] = "UNREVIEWED"
    result["is_ground_truth"] = False
    (OUTPUT / "annotations_draft.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUTPUT / "qa_applied.json").write_text(json.dumps({**plan, "scope": "frames 231-300; 201-230 changes come from the source draft",
                                                        "limitation": "added boxes are YOLOv8s boxes; vehicles missed by both the draft and YOLOv8s are still absent"},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", OUTPUT)


if __name__ == "__main__":
    main()
