import json
from pathlib import Path

import pytest

from scripts.build_vlm_annotation_batch import validate_proposal


ROOT = Path(__file__).resolve().parents[2]


def test_manual_proposals_match_source_bounds_and_scope():
    document = json.loads((ROOT / 'data/VLM363_working/annotation_batches/batch_001/manual_proposals.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'runs/project363/vlm/source_survey/source_manifest.json').read_text(encoding='utf-8'))
    sources = {row['survey_id']: row['decoder'] for row in manifest}
    assert not document['is_ground_truth']
    assert not document['is_training_ready']
    for sample in document['samples']:
        validate_proposal(sample, sources[sample['survey_id']])


def test_invalid_bbox_is_rejected():
    sample = {'scope_status': 'in_scope', 'package_assessment_proposed': 'insufficient_evidence',
              'clip': [0, 1], 'frames': [[0, [-1, 0, 10, 10]]]}
    with pytest.raises(ValueError, match='bounding box'):
        validate_proposal(sample, {'width': 100, 'height': 100, 'nominal_fps': 30, 'reported_frame_count': 30})


def test_out_of_scope_cannot_have_violation_label():
    sample = {'scope_status': 'out_of_scope', 'package_assessment_proposed': 'red_light_crossing',
              'clip': [0, 1], 'frames': [[0, [0, 0, 10, 10]]]}
    with pytest.raises(ValueError, match='Out-of-scope'):
        validate_proposal(sample, {'width': 100, 'height': 100, 'nominal_fps': 30, 'reported_frame_count': 30})
