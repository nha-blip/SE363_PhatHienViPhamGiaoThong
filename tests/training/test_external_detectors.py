import numpy as np
import pytest
import torch

from scripts.evaluate_external_detectors import class_mapping, mapped_predictions, read_labels


def test_no_guessed_vehicle_subtype_mapping():
    mapping, excluded = class_mapping({0: 'Car', 1: 'Van', 2: 'Truck', 3: 'Ambulance',
                                       4: 'Motorcycle', 5: 'Bus', 6: 'Taxi'}, 'vehicle')
    assert mapping == {0: 0, 2: 3, 4: 1, 5: 2}
    assert excluded == {}


def test_light_color_pooling_preserves_unsupported_red_yellow():
    mapping, excluded = class_mapping({0: 'circle_red', 1: 'arrow_left_green',
                                       2: 'arrow_right_yellow', 3: 'off',
                                       4: 'circle_red_yellow'}, 'state')
    assert mapping == {0: 0, 1: 2, 2: 1, 3: 3}
    assert excluded == {4: 'circle_red_yellow'}
    with pytest.raises(ValueError):
        class_mapping({0: 'Traffic light'}, 'state')


def test_trained_light_colors_map_without_direction_suffix():
    mapping, excluded = class_mapping({0: 'red', 1: 'yellow', 2: 'green', 3: 'off'}, 'state')
    assert mapping == {0: 0, 1: 1, 2: 2, 3: 3}
    assert excluded == {}


def test_label_collapse_and_native_coordinates(tmp_path):
    path = tmp_path / 'labels.txt'
    path.write_text('2 0.5 0.5 0.2 0.4\n')
    identities, boxes = read_labels(path, 100, 50, collapse=True)
    np.testing.assert_array_equal(identities, [0])
    np.testing.assert_allclose(boxes, [[40, 15, 60, 35]])
    path.write_text('0 0.5 0.5 -0.1 0.4')
    with pytest.raises(ValueError):
        read_labels(path, 100, 50)


def test_mapped_predictions_sorted_by_confidence():
    from types import SimpleNamespace
    result = SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor([
        [0, 0, 10, 10, 0.2, 5], [0, 0, 10, 10, 0.8, 9], [0, 0, 10, 10, 0.9, 4],
    ])))
    predictions = mapped_predictions(result, {5: 0, 9: 2})
    np.testing.assert_allclose(predictions[:, 4], [0.8, 0.2])
    np.testing.assert_array_equal(predictions[:, 5], [2, 0])
