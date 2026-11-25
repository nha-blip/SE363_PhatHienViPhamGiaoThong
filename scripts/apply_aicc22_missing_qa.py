"""AICC22: add the vehicle confirmed missing and record crowd-box ignore regions (new draft revision)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tracking_track_qa as builder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/Tracking363_working/annotation_drafts"
DRAFT = BASE / "aicc22_custom_001_from_json_gt"
QA = BASE / "aicc22_custom_001_missing_qa_all_detectors"
OUTPUT = BASE / "aicc22_custom_001_after_missing_qa"
TRACKING = ROOT / "runs/project363/tracking"
DETECTORS = {"e30": "vehicle_epoch30_aicc22_custom_001", "e41_960": "vehicle_epoch41_imgsz960_aicc22_custom_001",
             "e63_960": "vehicle_epoch63_imgsz960_aicc22_custom_001", "v8s_960": "vehicle_v8s_imgsz960_aicc22_custom_001"}
FIRST, LAST = 3, 200


def main():
    draft = json.loads((DRAFT / "annotations_draft.json").read_text(encoding="utf-8"))
    frames = {f["mot_frame_id"]: f for f in draft["frames"]}
    page = json.loads(re.search(r"const DATA = (\{.*?\});\nconst KEY", (QA / "track_qa.html").read_text(encoding="utf-8"), re.S).group(1))
    answers = json.loads((QA / "tracking_track_qa_result.json").read_text(encoding="utf-8"))["missing"]
    builder.DRAFT, builder.FIRST_FRAME, builder.LAST_FRAME = DRAFT, FIRST, LAST
    added, ignore, next_id = {}, [], 3000
    for item in page["missing"]:
        label = answers[item["key"]]["label"]
        if label == "missing_vehicle":
            detector, identity = item["key"].split(":")
            next_id += 1
            for frame, box, _, _ in builder.missing_candidates(draft, TRACKING / DETECTORS[detector] / "predictions_mot.txt")[int(identity)]:
                frames[frame]["objects"].append({"track_id": next_id, "bbox_xywh": [round(v, 2) for v in box], "class": None,
                                                 "annotation_origin": f"{detector.upper()}_TRACK_HUMAN_CONFIRMED_MISSING", "human_corrected": False})
                added[next_id] = added.get(next_id, 0) + 1
        elif label == "multi_vehicle_box":
            for member in item["members"]:
                d, i = member.split(":")
                for line in (TRACKING / DETECTORS[d] / "predictions_mot.txt").read_text().splitlines():
                    p = line.split(",")
                    if int(p[1]) == int(i) and FIRST <= int(p[0]) <= LAST:
                        ignore.append({"frame": int(p[0]), "bbox_xywh": [float(v) for v in p[2:6]], "source": member})
    for f in draft["frames"]:
        ids = [o["track_id"] for o in f["objects"]]
        assert len(ids) == len(set(ids)), f["mot_frame_id"]
    draft.update({"ignore_regions": ignore, "draft_revision": "after_missing_qa_all_detectors"})
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "annotations_draft.json").write_text(json.dumps(draft, indent=1), encoding="utf-8")
    summary = {"added_tracks": added, "ignore_region_boxes": len(ignore), "answers": {k: v["label"] for k, v in answers.items()}}
    (OUTPUT / "qa_applied.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
