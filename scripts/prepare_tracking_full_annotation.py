import json
import os
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.class_suggestions import overlap
from src.tracking.io import sha256


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = ROOT / "data/Tracking363_working/sequences/vntraffic_001"
GT = ROOT / "data/Tracking363_working/gt_revisions/vntraffic_001_remove_track68/metadata/annotations.jsonl"
PREDICTIONS = ROOT / "runs/project363/tracking/vehicle_epoch30_vntraffic_001/experiments/crossclass_suppression_0_7/tracking_results.jsonl"
OUTPUT = ROOT / "data/Tracking363_working/annotation_drafts/vntraffic_001_mot201_300"
FIRST_FRAME, LAST_FRAME = 201, 300
PROPOSAL_ID_OFFSET = 1000
# Lower than the matching IoU so a vehicle partly hidden behind a labelled one can still be proposed.
PROPOSAL_MAX_GT_IOU = 0.3


def draft_frame(frame_id, gt_rows, tracks, image):
    """Keep existing boxes and IDs; add model boxes that no existing box explains, in a separate ID range."""
    objects = [{"track_id": row["track_id"], "bbox_xywh": row["bbox_xywh"], "class": None, "annotation_origin": "EXISTING_GT", "human_corrected": False} for row in gt_rows]
    for track in tracks:
        if any(overlap(row["bbox_xywh"], track["bbox_xyxy"]) >= PROPOSAL_MAX_GT_IOU for row in gt_rows):
            continue
        left, top, right, bottom = track["bbox_xyxy"]
        objects.append({"track_id": PROPOSAL_ID_OFFSET + track["track_id"], "bbox_xywh": [round(left, 2), round(top, 2), round(right - left, 2), round(bottom - top, 2)],
                        "class": None, "annotation_origin": "MODEL_PROPOSAL", "human_corrected": False})
    return {"mot_frame_id": frame_id, "image": image, "review_status": "UNREVIEWED", "objects": objects}


def main():
    gt_rows = defaultdict(list)
    for line in GT.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gt_rows[row["mot_frame_id"]].append(row)
    tracks = {}
    for line in PREDICTIONS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        tracks[row["mot_frame_id"]] = row["tracks"]
    OUTPUT.mkdir(exist_ok=False)
    images = Path(os.path.relpath(SEQUENCE / "img1", OUTPUT)).as_posix()
    frames = [draft_frame(frame_id, gt_rows[frame_id], tracks[frame_id], f"{images}/{frame_id:06d}.jpg") for frame_id in range(FIRST_FRAME, LAST_FRAME + 1)]
    for frame in frames:
        if not (SEQUENCE / "img1" / f"{frame['mot_frame_id']:06d}.jpg").is_file():
            raise ValueError("Missing frame image")
    payload = {"sequence_id": f"vntraffic_001_mot{FIRST_FRAME}_{LAST_FRAME}", "draft_revision": "full_annotation_001", "source_prediction_sha256": sha256(PREDICTIONS),
               "source_gt_sha256": sha256(GT), "width": 1920, "height": 1080, "status": "UNREVIEWED", "is_ground_truth": False, "ignore_regions": [], "frames": frames}
    (OUTPUT / "annotations_draft.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    template = (ROOT / "src/tracking/annotation_editor.html").read_text(encoding="utf-8")
    (OUTPUT / "editor.html").write_text(template.replace("__ANNOTATION_PAYLOAD__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c")), encoding="utf-8")
    summary = {"frames": len(frames), "existing_gt_boxes": sum(obj["annotation_origin"] == "EXISTING_GT" for frame in frames for obj in frame["objects"]),
               "model_proposals": sum(obj["annotation_origin"] == "MODEL_PROPOSAL" for frame in frames for obj in frame["objects"]),
               "proposal_tracks": len({obj["track_id"] for frame in frames for obj in frame["objects"] if obj["annotation_origin"] == "MODEL_PROPOSAL"}), "is_ground_truth": False}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text("# Bản nháp gán nhãn đầy đủ vntraffic_001, MOT 201–300\n\nMở editor.html. Bản nháp này chưa phải GT. Box xanh lá lấy từ GT sẵn có, box cam là đề xuất của detector epoch 30 sau khi loại trùng khác lớp, chưa được kiểm chứng. File gốc trong gt_revisions không bị sửa.\n\nKhi xong, bấm Xuất file kết quả và gửi file vntraffic_001_mot201_300_human_review.json để kiểm tra trước khi tạo revision GT.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
