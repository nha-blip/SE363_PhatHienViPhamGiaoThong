from types import SimpleNamespace

from src.tracking.class_suggestions import summarize_votes, unique_matches


def test_unique_match_preserves_gt():
    records = [{"bbox_xywh": [10, 10, 20, 20], "track_id": 49}]
    prediction = SimpleNamespace(bbox_xyxy=(10, 10, 30, 30))
    assert unique_matches(records, [prediction]) == {0: (0, 1.0)}
    assert records[0]["track_id"] == 49


def test_shared_prediction_is_not_assigned_twice():
    records = [{"bbox_xywh": [10, 10, 20, 20]}] * 2
    assert unique_matches(records, [SimpleNamespace(bbox_xyxy=(10, 10, 30, 30))]) == {}


def test_multiple_predictions_remain_ambiguous():
    records = [{"bbox_xywh": [10, 10, 20, 20]}]
    prediction = SimpleNamespace(bbox_xyxy=(10, 10, 30, 30))
    assert unique_matches(records, [prediction, prediction]) == {}


def test_classes_remain_pending_even_with_consistent_votes():
    result = summarize_votes([{"candidate_class": "car"}] * 3)
    assert result["suggested_class"] == "car"
    assert result["class"] is None
    assert result["human_qa_status"] == "REVIEW_REQUIRED"


def test_no_evidence_and_mixed_votes_do_not_force_class():
    assert summarize_votes([{"candidate_class": None}])["suggested_class"] is None
    assert summarize_votes([{"candidate_class": "car"}, {"candidate_class": "truck"}, {"candidate_class": "car"}])["suggested_class"] is None
