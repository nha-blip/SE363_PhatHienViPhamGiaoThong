import pytest

from scripts.summarize_detector_errors import summarize_matrix


def test_error_categories_do_not_double_count_wrong_class_boxes():
    matrix = [[7, 2, 3], [1, 4, 5], [6, 8, 0]]
    result = summarize_matrix(matrix, ['car', 'truck'])
    assert result['totals'] == {'gt': 28, 'correct': 11, 'unmatched_gt': 14,
                                'wrong_class_gt': 3, 'unmatched_predictions': 8}
    assert result['per_class'][0]['gt'] == 14
    assert result['confusions'][0] == {'gt': 'truck', 'predicted': 'car', 'count': 2}


def test_empty_prediction_class_and_invalid_matrix():
    result = summarize_matrix([[0, 0], [3, 0]], ['red'])
    assert result['per_class'][0]['precision_at_025'] is None
    assert result['per_class'][0]['recall_at_025'] == 0
    with pytest.raises(ValueError):
        summarize_matrix([[1]], ['red'])
    with pytest.raises(ValueError):
        summarize_matrix([[0.5, 0], [0, 0]], ['red'])
