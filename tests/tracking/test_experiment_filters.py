import pytest

from src.tracking.experiment_filters import cross_class_suppression
from src.tracking.schemas import Detection, DetectionResult, FrameInfo


def result(detections):
    return DetectionResult(FrameInfo("pilot", 0, 1, 100, 100, 30, 1), tuple(detections))


def test_crossclass_overlap_suppresses_lower_score_only():
    source = result([Detection((0, 0, 20, 20), 0.8, 0, "car"), Detection((0, 0, 20, 20), 0.5, 3, "truck")])
    filtered, log = cross_class_suppression(source)
    assert len(filtered.detections) == 1
    assert log[0]["class_name"] == "truck"
    assert len(source.detections) == 2


def test_sameclass_or_separate_objects_are_preserved():
    source = result([Detection((0, 0, 20, 20), 0.8, 0, "car"), Detection((0, 0, 20, 20), 0.5, 0, "car"), Detection((50, 50, 70, 70), 0.5, 3, "truck")])
    assert len(cross_class_suppression(source)[0].detections) == 3


def test_empty_frame_and_invalid_threshold():
    assert cross_class_suppression(result([])) == (result([]), [])
    with pytest.raises(ValueError):
        cross_class_suppression(result([]), 0)
