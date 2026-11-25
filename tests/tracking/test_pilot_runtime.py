from types import SimpleNamespace

import numpy as np
import pytest

from src.tracking.pilot_runtime import LoggedBoundsBackend, diagnostic_matches, identity_diagnostics, size_bucket


def test_bounds_policy_logs_and_preserves_raw_prediction():
    backend = SimpleNamespace(update=lambda batch, img=None: np.array([[-1, 2, 105, 12, 1, 0.9, 0, 0]]))
    wrapper = LoggedBoundsBackend(backend, 100, 50)
    result = wrapper.update(None)
    assert result[0, :4].tolist() == [0, 2, 100, 12]
    assert wrapper.raw_output[0][0] == -1
    assert wrapper.events[0]["action"] == "CLIP_PREDICTION_ONLY"


def test_degenerate_output_is_explicitly_dropped():
    backend = SimpleNamespace(update=lambda batch, img=None: np.array([[101, 2, 105, 12, 1, 0.9, 0, 0]]))
    wrapper = LoggedBoundsBackend(backend, 100, 50)
    assert wrapper.update(None).shape == (0, 8)
    assert wrapper.events[0]["action"] == "DROP_DEGENERATE_PREDICTION"


def test_nonfinite_backend_output_is_rejected():
    backend = SimpleNamespace(update=lambda batch, img=None: np.array([[0, 2, np.nan, 12, 1, 0.9, 0, 0]]))
    with pytest.raises(ValueError):
        LoggedBoundsBackend(backend, 100, 50).update(None)


def test_diagnostic_match_is_one_to_one_and_class_agnostic():
    gt_rows = [{"track_id": 74, "bbox_xywh": [0, 0, 10, 10]}, {"track_id": 72, "bbox_xywh": [20, 20, 10, 10]}]
    tracks = [SimpleNamespace(track_id=1, bbox_xyxy=(0, 0, 10, 10)), SimpleNamespace(track_id=2, bbox_xyxy=(20, 20, 30, 30))]
    assert diagnostic_matches(gt_rows, tracks) == [(74, 1, 1.0), (72, 2, 1.0)]
    assert diagnostic_matches(gt_rows, []) == []


def frame_result(frame_id, *tracks):
    return SimpleNamespace(frame=SimpleNamespace(mot_frame_id=frame_id),
                           tracks=[SimpleNamespace(track_id=identity, bbox_xyxy=box) for identity, box in tracks])


def test_identity_diagnostics_counts_fragments_and_keeps_outside_predictions_separate():
    gt_frames = {frame: {7: {"track_id": 7, "bbox_xywh": [0, 0, 20, 40]}, 8: {"track_id": 8, "bbox_xywh": [100, 100, 80, 80]}} for frame in (1, 2, 3)}
    results = [frame_result(1, (1, (0, 0, 20, 40))), frame_result(2, (2, (0, 0, 20, 40)), (9, (300, 300, 340, 340))),
               frame_result(3, (2, (0, 0, 20, 40)), (3, (0, 0, 20, 39)))]
    report = identity_diagnostics(gt_frames, results)
    assert (report["gt_observations"], report["gt_tracks"], report["matched_observations"]) == (6, 2, 3)
    assert report["gt_tracks_never_matched"] == 1
    assert report["gt_tracks_with_several_prediction_ids"] == 1
    assert report["consecutive_id_change_candidates"] == 1
    assert report["predictions_outside_gt"] == 1
    assert report["overlapping_prediction_pairs_iou_0_7"] == 1
    assert report["coverage_by_short_side_px"]["<32"] == {"gt": 3, "matched": 3}
    assert report["coverage_by_short_side_px"][">=64"] == {"gt": 3, "matched": 0}


def test_size_bucket_boundaries():
    assert [size_bucket(value) for value in (15.9, 16, 32, 64)] == ["<16", "<32", "<64", ">=64"]
