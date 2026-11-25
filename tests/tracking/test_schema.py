import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.tracking.io import read_results, write_results
from src.tracking.schemas import Detection, DetectionResult, FrameInfo, Track, TrackingResult, xywh_to_xyxy, xyxy_to_xywh
from src.tracking.yolo_adapter import YoloAdapter


def frame():
    return FrameInfo("test_sequence", 0, 1, 1920, 1080, 30, 2)


def test_detection_and_tracking_round_trip(tmp_path):
    detection = Detection((10, 20, 30, 60), 0.95, 7, "car")
    detection_result = DetectionResult(frame(), (detection,))
    assert DetectionResult.from_dict(json.loads(json.dumps(detection_result.to_dict()))) == detection_result
    track = Track(detection.bbox_xyxy, 0.91, 7, "car", 17)
    tracking_result = TrackingResult(frame(), (track,))
    assert TrackingResult.from_dict(json.loads(json.dumps(tracking_result.to_dict()))) == tracking_result
    path = tmp_path / "results.jsonl"
    write_results(path, (detection_result,))
    assert read_results(path, DetectionResult) == (detection_result,)
    with pytest.raises(FileExistsError):
        write_results(path, (detection_result,))


def test_bbox_conversion_and_bounds():
    assert xywh_to_xyxy((682, 683, 54, 46)) == (682, 683, 736, 729)
    assert xyxy_to_xywh((682, 683, 736, 729)) == (682, 683, 54, 46)
    with pytest.raises(ValueError):
        DetectionResult(frame(), (Detection((0, 0, 1921, 10), 0.5, 0, "car"),))


@pytest.mark.parametrize("box,score", [((0, 0, 0, 1), 0.5), ((-1, 0, 2, 2), 0.5), ((0, 0, float('nan'), 2), 0.5), ((0, 0, 2, 2), 1.1)])
def test_invalid_detection_rejected(box, score):
    with pytest.raises(ValueError):
        Detection(box, score, 0, "car")


def test_vehicle_filter_preserves_unified_ids():
    names = {0: "red", 1: "yellow", 2: "green", 3: "off", 7: "car", 11: "motorcycle", 15: "bus", 20: "truck"}
    identities = list(names)
    result = SimpleNamespace(names=names, orig_shape=(1080, 1920), boxes=SimpleNamespace(
        xyxy=np.array([[10, 20, 30, 40]] * len(names)), conf=np.array([0.8] * len(names)), cls=np.array(identities)))
    converted = YoloAdapter(names).adapt(result, frame())
    assert [(item.class_id, item.class_name) for item in converted.detections] == [(7, "car"), (11, "motorcycle"), (15, "bus"), (20, "truck")]
    result.names = {**names, 7: "truck"}
    with pytest.raises(ValueError):
        YoloAdapter(names).adapt(result, frame())


def test_confidence_filter_retains_low_stage_candidates():
    names = {0: "car"}
    result = SimpleNamespace(names=names, orig_shape=(1080, 1920), boxes=SimpleNamespace(
        xyxy=[[10, 20, 30, 40], [10, 20, 30, 40]], conf=[0.15, 0.05], cls=[0, 0]))
    converted = YoloAdapter(names, minimum_confidence=0.1).adapt(result, frame())
    assert [item.confidence for item in converted.detections] == [0.15]


def test_duplicate_predicted_track_rejected():
    track = Track((10, 20, 30, 40), 0.8, 0, "car", 1)
    with pytest.raises(ValueError):
        TrackingResult(frame(), (track, track))
