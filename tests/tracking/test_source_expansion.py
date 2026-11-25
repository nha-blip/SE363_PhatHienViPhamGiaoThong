import pytest

from scripts.expand_tracking_pilot_sources import validate_rows


def test_valid_unlabeled_prefix_is_allowed():
    validate_rows([{"mot_frame_id": 3, "track_id": 1, "bbox_xywh": [0, 0, 10, 10]}])


def test_duplicate_keys_are_rejected():
    row = {"mot_frame_id": 3, "track_id": 1, "bbox_xywh": [0, 0, 10, 10]}
    with pytest.raises(ValueError):
        validate_rows([row, row])


def test_invalid_source_bounds_are_rejected():
    with pytest.raises(ValueError):
        validate_rows([{"mot_frame_id": 3, "track_id": 1, "bbox_xywh": [1270, 0, 20, 10]}])
