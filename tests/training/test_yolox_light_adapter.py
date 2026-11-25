import numpy as np
import pytest

from scripts.yolox_light_adapter import decode_yolox


def test_raw_grid_stride_decoding_does_not_mutate_input():
    raw = np.zeros((1, 8400, 19), dtype=np.float32)
    decoded = decode_yolox(raw)
    np.testing.assert_array_equal(decoded[0, :4], [0, 0, 8, 8])
    np.testing.assert_array_equal(decoded[1, :4], [8, 0, 8, 8])
    np.testing.assert_array_equal(decoded[6400, :4], [0, 0, 16, 16])
    assert not raw.any()


def test_unexpected_shape_is_rejected():
    with pytest.raises(ValueError):
        decode_yolox(np.zeros((1, 8400, 85), dtype=np.float32))
