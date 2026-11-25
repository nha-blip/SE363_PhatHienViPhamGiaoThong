import pytest

from scripts.evaluate_yolo_candidates import candidate_mapping


def test_motor_alias_does_not_guess_van_or_other_classes():
    mapping, excluded = candidate_mapping({0: 'motor', 1: 'van', 2: 'others', 3: 'truck'}, 'vehicle')
    assert mapping == {0: 1, 3: 3}
    assert excluded == {}


def test_color_direction_and_lisa_mapping_preserve_missing_off():
    mapping, excluded = candidate_mapping({0: 'left-green', 1: 'traffic light red', 2: 'warningLeft',
                                           3: 'goForward', 4: 'UNKNOWN', 5: 'red-amber-circle'}, 'state')
    assert mapping == {0: 2, 1: 0, 2: 1, 3: 2}
    assert set(excluded) == {4, 5}
    assert 3 not in mapping.values()


def test_generic_light_is_not_treated_as_a_color():
    with pytest.raises(ValueError):
        candidate_mapping({0: 'traffic light'}, 'state')
    assert candidate_mapping({0: 'traffic light'}, 'localization')[0] == {0: 0}


def test_yolox_special_states_are_not_forced_into_off_or_red():
    from scripts.yolox_light_adapter import NAMES
    mapping, excluded = candidate_mapping(NAMES, 'state')
    assert mapping[3] == 1
    assert mapping[13] == 2
    assert set(excluded) == {0, 2}
    assert 3 not in mapping.values()
