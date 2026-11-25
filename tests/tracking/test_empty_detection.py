from types import SimpleNamespace

import numpy as np
import pytest

from src.tracking.bytetrack_adapter import ByteTrackAdapter, DetectionBatch, create_ultralytics_adapter
from src.tracking.schemas import Detection, DetectionResult, FrameInfo
from src.tracking.yolo_adapter import YoloAdapter


class FakeBackend:
    def __init__(self, output=()):
        self.calls = []
        self.output = output

    def update(self, batch, img=None):
        self.calls.append(batch)
        return np.asarray(self.output)


def frame(source=0, sequence="test_sequence"):
    return FrameInfo(sequence, source, source + 1, 1920, 1080, 30, 3)


def test_empty_detections_advance_backend_each_frame():
    backend = FakeBackend()
    tracker = ByteTrackAdapter(backend, {7: "car"})
    for source in range(3):
        result = tracker.update(DetectionResult(frame(source)))
        assert result.tracks == ()
        assert backend.calls[-1].xywh.shape == (0, 4)
        assert backend.calls[-1].conf.shape == (0,)
        assert len(backend.calls[-1][np.zeros(0, dtype=bool)]) == 0
    assert len(backend.calls) == 3
    assert tracker.last_frame == 3


def test_empty_yolo_none_and_array_outputs():
    adapter = YoloAdapter({0: "car"})
    result = SimpleNamespace(names={0: "car"}, orig_shape=(1080, 1920), boxes=None)
    assert adapter.adapt(result, frame()).detections == ()
    result.boxes = SimpleNamespace(xyxy=np.empty((0, 4)), conf=np.empty(0), cls=np.empty(0))
    assert adapter.adapt(result, frame()).detections == ()


def test_backend_xywh_uses_center_not_top_left():
    batch = DetectionBatch([[10, 20, 30, 60]], [0.8], [7])
    np.testing.assert_array_equal(batch.xywh, [[20, 40, 20, 40]])
    np.testing.assert_array_equal(batch[np.array([True])].cls, [7])


def test_stub_tracker_output_contract():
    backend = FakeBackend([[10, 20, 30, 40, 17, 0.91, 7, 0]])
    adapter = ByteTrackAdapter(backend, {7: "car"})
    result = adapter.update(DetectionResult(frame(), (Detection((10, 20, 30, 40), 0.95, 7, "car"),)))
    assert result.tracks[0].track_id == 17
    assert result.tracks[0].class_id == 7
    assert result.tracks[0].class_name == "car"


def test_out_of_order_or_cross_sequence_rejected():
    adapter = ByteTrackAdapter(FakeBackend(), {0: "car"})
    with pytest.raises(ValueError):
        adapter.update(DetectionResult(frame(1)))
    adapter.update(DetectionResult(frame()))
    with pytest.raises(ValueError):
        adapter.update(DetectionResult(frame()))
    with pytest.raises(ValueError):
        adapter.update(DetectionResult(frame(1, "another_sequence")))


def test_model_execution_factory_requires_explicit_authorization():
    with pytest.raises(PermissionError):
        create_ultralytics_adapter({}, {0: "car"}, 30)
