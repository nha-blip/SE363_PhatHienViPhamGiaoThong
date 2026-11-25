import numpy as np

from scripts.render_external_detector_qa import compose, select_pairs


def test_selection_covers_classes_and_background(tmp_path):
    pairs = []
    for index in range(25):
        label = tmp_path / f'{index}.txt'
        label.write_text('' if index % 5 == 4 else f'{index % 5} 0.5 0.5 0.2 0.2\n')
        pairs.append((tmp_path / f'{index}.jpg', label))
    selected = select_pairs(pairs, 10)
    assert len(selected) == 10
    assert len({image for image, _ in selected}) == 10
    assert select_pairs(pairs, 10) == selected
    assert any(not label.read_text() for _, label in selected)
    assert {int(label.read_text().split()[0]) for _, label in selected if label.read_text()} == {0, 1, 2, 3}


def test_comparison_board_preserves_input_and_contains_zoom():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    annotations = [{'class_id': 0, 'label': 'red', 'bbox_xyxy': [20, 20, 40, 60], 'scored': True}]
    output = compose(image, annotations, annotations, 'QA fixture', 'state')
    assert output.shape == (978, 1920, 3)
    assert not image.any()
    assert output.any()
    assert compose(image, [], [], 'Empty background', 'localization').shape == (746, 1920, 3)
