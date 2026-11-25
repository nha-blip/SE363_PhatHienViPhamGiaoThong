from scripts.prepare_tracking_full_annotation import PROPOSAL_ID_OFFSET, draft_frame


def test_existing_boxes_stay_and_only_unexplained_predictions_become_proposals():
    gt_rows = [{"track_id": 7, "bbox_xywh": [0, 0, 100, 100]}]
    tracks = [{"track_id": 1, "bbox_xyxy": [0, 0, 100, 100]}, {"track_id": 2, "bbox_xyxy": [500, 500, 550.123, 560]}]
    frame = draft_frame(201, gt_rows, tracks, "img/000201.jpg")
    assert [(item["track_id"], item["annotation_origin"]) for item in frame["objects"]] == [(7, "EXISTING_GT"), (PROPOSAL_ID_OFFSET + 2, "MODEL_PROPOSAL")]
    assert frame["objects"][1]["bbox_xywh"] == [500, 500, 50.12, 60]
    assert frame["review_status"] == "UNREVIEWED"
    assert all(item["class"] is None for item in frame["objects"])


def test_vehicle_partly_behind_a_labelled_box_is_still_proposed():
    gt_rows = [{"track_id": 7, "bbox_xywh": [0, 0, 100, 100]}]
    frame = draft_frame(201, gt_rows, [{"track_id": 3, "bbox_xyxy": [80, 0, 180, 100]}], "x.jpg")
    assert len(frame["objects"]) == 2
