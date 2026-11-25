"""AICC22-Custom: draft from the owner-reviewed JSON GT candidate + missing-vehicle QA from all 4 detectors.

MOT frames 1-2 are unlabeled in the source (not background), so the window is 3-200.
The source GT file is read only; the draft is a new file.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tracking_missing_qa_all_detectors as missing  # noqa: E402
import build_tracking_track_qa as builder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "data/Tracking363_working/sequences/aicc22_custom_001"
BASE = ROOT / "data/Tracking363_working/annotation_drafts"
DRAFT = BASE / "aicc22_custom_001_from_json_gt"
QA = BASE / "aicc22_custom_001_missing_qa_all_detectors"
FIRST, LAST = 3, 200


def build_draft():
    frames = {f: {"mot_frame_id": f, "image": f"../../sequences/aicc22_custom_001/img1/{f:06d}.jpg", "objects": []} for f in range(1, 201)}
    for line in (SEQ / "gt/gt_candidate_json.txt").read_text().splitlines():
        p = line.split(",")
        frames[int(p[0])]["objects"].append({"track_id": int(p[1]), "bbox_xywh": [float(v) for v in p[2:6]], "class": None,
                                              "annotation_origin": "EXISTING_GT", "human_corrected": False})
    draft = {"sequence_id": "aicc22_custom_001", "draft_revision": "from_json_gt_candidate", "width": 1280, "height": 720,
             "status": "UNREVIEWED", "is_ground_truth": False, "ignore_regions": [], "unlabeled_frames": [1, 2],
             "source_gt": str(SEQ / "gt/gt_candidate_json.txt"), "frames": [frames[f] for f in sorted(frames)]}
    DRAFT.mkdir(parents=True, exist_ok=False)
    (DRAFT / "annotations_draft.json").write_text(json.dumps(draft, indent=1), encoding="utf-8")
    print("draft boxes", sum(len(f["objects"]) for f in draft["frames"]))


def main():
    if not DRAFT.exists():
        build_draft()
    missing.DRAFT, missing.OUTPUT, missing.FIRST, missing.LAST = DRAFT, QA, FIRST, LAST
    missing.DETECTORS = {k: v.replace("vntraffic_001", "aicc22_custom_001") for k, v in missing.DETECTORS.items()}
    missing.PREVIOUS = None
    original = missing.json.loads

    def loads(text, *args, **kwargs):  # no earlier answers exist for this sequence
        return original(text, *args, **kwargs)
    missing.PREVIOUS = type("NoAnswers", (), {"read_text": lambda self, encoding=None: '{"missing": {}}'})()
    missing.main()


if __name__ == "__main__":
    main()
