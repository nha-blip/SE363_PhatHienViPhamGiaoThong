import pytest

from scripts.import_vlm_human_review import normalize


def fixture_review():
    package = {'sample_id': 'vlm_007', 'source_sha256': 'source', 'vehicle_type': 'truck',
               'scope_status': 'out_of_scope', 'package_assessment_proposed': None, 'evidence': [],
               'assessment': None, 'full_video_assessment': None}
    row = {'sample_id': 'vlm_007', 'source_sha256': 'source', 'decision': 'CONFIRM_PROPOSAL',
           'final_scope_status': 'in_scope', 'final_package_assessment': 'no_red_light_crossing',
           'target_bbox_confirmed': True, 'signal_mapping_confirmed': True, 'stop_line_confirmed': True}
    return {'batch_id': 'batch_001', 'draft_sha256': 'draft', 'reviews': [row]}, [package]


def test_truck_scope_and_changed_decision():
    review, packages = fixture_review()
    rows, results = normalize(review, packages, 'draft')
    assert rows[0]['decision'] == 'CHANGE_LABEL'
    assert results[0]['scope_status'] == 'in_scope'
    assert results[0]['assessment'] is None
    assert results[0]['is_ground_truth'] is False
    assert packages[0]['scope_status'] == 'out_of_scope'


def test_reject_mismatched_hash():
    review, packages = fixture_review()
    with pytest.raises(ValueError):
        normalize(review, packages, 'wrong')


def test_reject_duplicate_rows():
    review, packages = fixture_review()
    review['reviews'] *= 2
    with pytest.raises(ValueError):
        normalize(review, packages, 'draft')


def test_insufficient_evidence_allows_unknown_signal():
    review, packages = fixture_review()
    review['reviews'][0].update(final_package_assessment='insufficient_evidence', signal_mapping_confirmed=False)
    assert normalize(review, packages, 'draft')[1][0]['package_assessment'] == 'insufficient_evidence'


def test_definitive_label_needs_confirmations():
    review, packages = fixture_review()
    review['reviews'][0]['signal_mapping_confirmed'] = False
    with pytest.raises(ValueError):
        normalize(review, packages, 'draft')
