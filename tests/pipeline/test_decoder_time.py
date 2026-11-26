import pytest

pytest.importorskip("cv2")
from scripts.run_red_light_pipeline import decoder_time


def test_valid_pts_is_used_instead_of_nominal_time():
    assert decoder_time(10.404, 10.30, 10.216) == (10.404, True)


def test_missing_negative_or_non_increasing_pts_falls_back_to_nominal():
    assert decoder_time(None, None, 1.0) == (1.0, False)
    assert decoder_time(-0.1, None, 1.0) == (1.0, False)
    assert decoder_time(2.0, 2.0, 2.1) == (2.1, False)
