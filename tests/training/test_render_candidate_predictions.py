import numpy as np

from scripts.render_candidate_predictions import build_record, focus_pairs, focus_title, render


def test_focus_contains_only_matched_object_label():
    record = {'ground_truth': [{'label': 'car', 'bbox_xyxy': [0, 0, 10, 10], 'status': 'WRONG'},
                               {'label': 'motorcycle', 'bbox_xyxy': [12, 0, 20, 10], 'status': 'OK'}],
              'predictions': [{'label': 'truck', 'bbox_xyxy': [0, 0, 10, 10], 'status': 'WRONG', 'confidence': .8}],
              'matches': [{'gt_index': 0, 'prediction_index': 0}]}
    focus = focus_pairs(record, 1)[0]
    assert focus_title(focus['gt'], 'GT') == 'GT: car'
    assert focus_title(focus['prediction'], 'PRED') == 'PRED: truck 0.80'


def test_generic_light_matching_does_not_claim_color_correctness(tmp_path):
    label = tmp_path / 'label.txt'
    label.write_text('2 0.5 0.5 0.2 0.2\n')
    cached = {'label': str(label), 'image': 'example.png', 'predictions': [[40, 40, 60, 60, .8, 0]],
              'native_predictions': [[40, 40, 60, 60, .8, 9]]}
    report = {'job': 'bdd_yolo11s_localization', 'mapping': {'mapped_source_ids': {'9': 0},
               'source_names': {'9': 'traffic light'}}}
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    record = build_record(cached, report, image)
    assert record['ground_truth'][0]['label'] == 'green'
    assert record['ground_truth'][0]['status'] == 'OK'
    assert record['predictions'][0]['label'] == 'traffic light'
    assert record['localization_only']
    assert render(record, image).shape == (978, 1920, 3)


def test_no_predictions_are_still_renderable(tmp_path):
    label = tmp_path / 'label.txt'
    label.write_text('0 0.5 0.5 0.2 0.2\n')
    report = {'job': 'signal_state', 'mapping': {'mapped_source_ids': {'0': 0}, 'source_names': {'0': 'red'}}}
    cached = {'label': str(label), 'image': 'example.png', 'predictions': [], 'native_predictions': []}
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    record = build_record(cached, report, image)
    assert record['ground_truth'][0]['status'] == 'MISSED'
    assert render(record, image).shape == (978, 1920, 3)
