import math

from scripts.propose_ne12th_initial_boxes import BLACK_BOXES, WHITE_BOXES


def test_proposal_scope_is_explicit_per_frame():
    assert set(BLACK_BOXES) == set(range(1, 27))
    assert set(WHITE_BOXES) == set(range(1, 11))


def test_proposed_boxes_are_valid_native_coordinates():
    for table in (BLACK_BOXES, WHITE_BOXES):
        for left, top, width, height in table.values():
            assert all(math.isfinite(value) for value in (left, top, width, height))
            assert left >= 0 and top >= 0 and width > 0 and height > 0
            assert left + width <= 1280 and top + height <= 720
