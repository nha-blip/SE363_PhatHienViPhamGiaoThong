import numpy as np
import cv2

from src.signal.lamp_position import read_by_position

ORANGE, RED, GREEN = (0, 120, 255), (60, 40, 240), (120, 230, 40)  # BGR


def head(lit_zone, colour, layout="vertical3", size=(30, 90)):
    w, h = size
    img = np.full((h, w, 3), 30, np.uint8)  # dark housing
    centres = [(w // 2, h // 6), (w // 2, h // 2), (w // 2, 5 * h // 6)]
    for c in centres:
        cv2.circle(img, c, 8, (50, 50, 50), -1)  # unlit lamps
    if lit_zone is not None:
        cv2.circle(img, centres[lit_zone], 8, colour, -1)
    return img.transpose(1, 0, 2).copy() if layout == "horizontal3" else img


def test_orange_middle_lamp_is_yellow():
    assert read_by_position(head(1, ORANGE), "vertical3")[0] == "yellow"


def test_orange_top_lamp_is_red():  # VN red often looks orange: position wins over hue
    assert read_by_position(head(0, ORANGE), "vertical3")[0] == "red"


def test_green_bottom_and_horizontal_layout():
    assert read_by_position(head(2, GREEN), "vertical3")[0] == "green"
    assert read_by_position(head(0, RED, "horizontal3"), "horizontal3")[0] == "red"


def test_nothing_lit_or_unknown_layout_gives_none():
    assert read_by_position(head(None, RED), "vertical3") is None
    assert read_by_position(head(1, ORANGE), "arrow") is None


def test_green_blob_in_red_zone_is_rejected():
    assert read_by_position(head(0, GREEN), "vertical3") is None
