import pytest

from scripts.prepare_vlm_local_vehicle_review import validate_seed


def test_valid_manual_reference():
    seed = {'clip_seconds':[0,4], 'reference_frame':30, 'bbox':[1,2,50,80], 'class':None}
    validate_seed(seed, {'nominal_fps':30,'decoded_clip_frames':900,'source_start_frame':0})


def test_reference_outside_context_rejected():
    seed = {'clip_seconds':[0,4], 'reference_frame':300, 'bbox':[1,2,50,80], 'class':'car'}
    with pytest.raises(ValueError):
        validate_seed(seed, {'nominal_fps':30,'decoded_clip_frames':900,'source_start_frame':0})


def test_invalid_bbox_rejected():
    seed = {'clip_seconds':[0,4], 'reference_frame':30, 'bbox':[50,2,1,80], 'class':'car'}
    with pytest.raises(ValueError):
        validate_seed(seed, {'nominal_fps':30,'decoded_clip_frames':900,'source_start_frame':0})
