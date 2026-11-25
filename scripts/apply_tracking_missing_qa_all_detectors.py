"""Add the vehicles confirmed missing in the all-detector QA, and record crowd ("box ôm nhiều xe") regions.

* "missing_vehicle": the lead track's uncovered boxes are added with IDs 3000+n (origin recorded).
* "multi_vehicle_box" answers from both missing-vehicle QA rounds become per-frame ignore regions:
  a detector box there covers a cluster whose individual vehicles may not be labeled.
Writes a new draft revision; the source draft is not modified.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tracking_track_qa as builder  # noqa: E402
from build_tracking_missing_qa_all_detectors import DETECTORS, TRACKING  # noqa: E402

BASE = Path(__file__).resolve().parents[1] / "data/Tracking363_working/annotation_drafts"
DRAFT = BASE / "vntraffic_001_mot201_300_after_track_qa_231_300"
QA_ALL = BASE / "vntraffic_001_mot201_300_missing_qa_all_detectors"
QA_V8S = BASE / "vntraffic_001_mot231_300_track_qa"
OUTPUT = BASE / "vntraffic_001_mot201_300_after_missing_qa"


def page_data(folder):
    return json.loads(re.search(r"const DATA = (\{.*?\});\nconst KEY", (folder / "track_qa.html").read_text(encoding="utf-8"), re.S).group(1))


def track_rows(detector, identity, first, last):
    rows = []
    for line in (TRACKING / DETECTORS[detector] / "predictions_mot.txt").read_text().splitlines():
        p = line.split(",")
        if int(p[1]) == identity and first <= int(p[0]) <= last:
            rows.append((int(p[0]), [float(v) for v in p[2:6]]))
    return rows


def main():
    source = DRAFT / "annotations_draft.json"
    draft = json.loads(source.read_text(encoding="utf-8"))
    frames = {f["mot_frame_id"]: f for f in draft["frames"]}
    builder.DRAFT, builder.FIRST_FRAME, builder.LAST_FRAME = DRAFT, 201, 300
    answers = json.loads((QA_ALL / "tracking_track_qa_result.json").read_text(encoding="utf-8"))["missing"]
    added, ignore = {}, []
    next_id = 3000
    for item in page_data(QA_ALL)["missing"]:
        label = answers[item["key"]]["label"]
        detector, identity = item["key"].split(":")
        if label == "missing_vehicle":
            cands = builder.missing_candidates(draft, TRACKING / DETECTORS[detector] / "predictions_mot.txt")
            next_id += 1
            for frame, box, _, _ in cands[int(identity)]:
                frames[frame]["objects"].append({"track_id": next_id, "bbox_xywh": [round(v, 2) for v in box], "class": None,
                                                 "annotation_origin": f"{detector.upper()}_TRACK_HUMAN_CONFIRMED_MISSING", "human_corrected": False})
                added[next_id] = added.get(next_id, 0) + 1
        elif label == "multi_vehicle_box":
            for member in item["members"]:
                d, i = member.split(":")
                ignore += [{"frame": f, "bbox_xywh": b, "source": member} for f, b in track_rows(d, int(i), 201, 300)]
    v8s_answers = json.loads((QA_V8S / "tracking_track_qa_result.json").read_text(encoding="utf-8"))["missing"]
    for key, value in v8s_answers.items():
        if value["label"] == "multi_vehicle_box":
            ignore += [{"frame": f, "bbox_xywh": b, "source": f"v8s_960:{key}"} for f, b in track_rows("v8s_960", int(key), 231, 300)]
    for f in draft["frames"]:
        ids = [o["track_id"] for o in f["objects"]]
        assert len(ids) == len(set(ids)), f["mot_frame_id"]
    draft["ignore_regions"] = ignore
    draft["ignore_region_policy"] = "detector boxes judged by the user to enclose several vehicles; per-frame boxes, frames 201-300"
    draft["draft_revision"] = "after_track_qa_and_missing_qa_all_detectors"
    draft["source_draft_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    draft["status"], draft["is_ground_truth"] = "UNREVIEWED", False
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "annotations_draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {"added_tracks": added, "added_boxes": sum(added.values()), "ignore_region_boxes": len(ignore),
               "ignore_region_sources": sorted({r["source"] for r in ignore}),
               "unclear_left_out": [k for k, v in answers.items() if v["label"] == "unclear"]}
    (OUTPUT / "qa_applied.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
