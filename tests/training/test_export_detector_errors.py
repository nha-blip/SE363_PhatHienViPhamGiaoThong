import numpy as np

from scripts.export_detector_errors import choose_errors, match_errors


def test_wrong_class_and_unmatched_boxes_are_distinct():
    identities = np.array([0, 1])
    boxes = np.array([[0, 0, 10, 10], [30, 30, 40, 40]])
    predictions = np.array([[0, 0, 10, 10, .8, 2], [50, 50, 60, 60, .9, 1]])
    gt_status, pred_status, matches = match_errors(identities, boxes, predictions)
    assert gt_status == ['WRONG', 'MISSED']
    assert pred_status == ['WRONG', 'EXTRA']
    assert len(matches) == 1


def test_empty_gt_and_duplicate_prediction():
    prediction = np.array([[0, 0, 10, 10, .8, 0], [0, 0, 10, 10, .7, 0]])
    gt_status, pred_status, _ = match_errors(np.array([0]), np.array([[0, 0, 10, 10]]), prediction)
    assert gt_status == ['OK']
    assert sorted(pred_status) == ['EXTRA', 'OK']
    gt_status, pred_status, _ = match_errors(np.array([]), np.empty((0, 4)), prediction)
    assert gt_status == []
    assert pred_status == ['EXTRA', 'EXTRA']
    gt_status, pred_status, _ = match_errors(np.array([0]), np.array([[0, 0, 10, 10]]), np.empty((0, 6)))
    assert gt_status == ['MISSED']
    assert pred_status == []


def test_selection_includes_wrong_class_and_has_no_duplicates():
    records = [{'error_types': ['MISSED']}, {'error_types': ['WRONG', 'EXTRA']}, {'error_types': ['EXTRA']}]
    selected = choose_errors(records, 2)
    assert records[1] in selected
    assert len(selected) == 2
    assert selected[0] is not selected[1]
