import numpy as np

from scripts.prepare_detector_improvement import capped_weights, focus_preview, spread_groups, train_only


def test_candidate_weights_are_capped_and_handle_missing_classes():
    weights = capped_weights({'green': 6000, 'red': 5500, 'yellow': 400, 'off': 0})
    assert weights['green'] == 1
    assert weights['yellow'] == 3
    assert weights['off'] is None


def test_training_selection_excludes_validation_and_test():
    records = [{'final_split': split} for split in ('train', 'val', 'test')]
    assert train_only(records) == [records[0]]


def test_sampling_spreads_groups_and_is_reproducible():
    records = [{'group': group, 'final_image': f'{group}/{index}'} for group in ('a', 'b') for index in range(3)]
    chosen = spread_groups(records, 2, 'group')
    assert {record['group'] for record in chosen} == {'a', 'b'}
    assert spread_groups(records, 2, 'group') == chosen


def test_zoom_does_not_paint_over_tiny_lamp_interior():
    image = np.full((100, 100, 3), (0, 0, 200), dtype=np.uint8)
    board = focus_preview(image, [49, 49, 51, 51], 'sample', 'red')
    assert board.shape == (682, 1440, 3)
    np.testing.assert_array_equal(board[362, 1200], [0, 0, 200])
