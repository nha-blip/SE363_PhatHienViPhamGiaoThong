import pytest

from scripts.run_tracking_pilot import NAMES, NAMES_6, detector_names
from src.tracking.yolo_adapter import YoloAdapter


def test_four_and_six_class_checkpoints_are_accepted():
    assert detector_names(NAMES) == NAMES and detector_names(NAMES_6) == NAMES_6
    YoloAdapter(detector_names(NAMES_6))


def test_unknown_class_table_is_rejected():
    with pytest.raises(ValueError):
        detector_names({0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "van"})
    with pytest.raises(ValueError):
        detector_names({0: "car", 1: "truck", 2: "bus", 3: "motorcycle"})
