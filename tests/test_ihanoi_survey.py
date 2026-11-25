import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.survey_ihanoi_cameras import Survey, center, next_section


class IhanoiSurveyTests(unittest.TestCase):
    def test_listing_retries_read_only_during_ui_transition(self):
        survey = Survey.__new__(Survey)
        survey.dump = Mock(side_effect=[b'<hierarchy/>',
            '<hierarchy><node content-desc="Camera giao thông"/></hierarchy>'.encode()])
        with patch('scripts.survey_ihanoi_cameras.time.sleep'):
            raw, rows = survey.listing('emulator-5554')
        self.assertEqual(rows, {})
        self.assertEqual(survey.dump.call_count, 2)

    def test_resume_preserves_completed_and_waiting_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows = [{'camera_id': 'C001', 'status': 'RECORDED_AWAITING_USER_REVIEW'},
                    {'camera_id': 'C002', 'status': 'WAITING_SURVEY'}]
            (output/'survey_index.json').write_text(json.dumps({
                'status': 'SURVEY_ENDED_REQUIRES_USER_SELECTION', 'cameras': rows}))
            with patch('scripts.survey_ihanoi_cameras.OUTPUT', output):
                survey = Survey(resume=True, serials=('emulator-5554', 'emulator-5584'))
            self.assertEqual(list(survey.records.values()), rows)
            self.assertEqual(survey.serials, ('emulator-5554', 'emulator-5584'))

    def test_active_survey_cannot_be_resumed_by_second_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output/'survey_index.json').write_text(json.dumps({'status': 'SURVEY_RUNNING'}))
            with patch('scripts.survey_ihanoi_cameras.OUTPUT', output):
                with self.assertRaisesRegex(ValueError, 'duplicate writer'):
                    Survey(resume=True)

    def test_unfinished_claim_blocks_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output/'survey_index.json').write_text(json.dumps({
                'status': 'SURVEY_ENDED_REQUIRES_USER_SELECTION',
                'cameras': [{'camera_id': 'C001', 'status': 'CLAIMED_ONCE'}]}))
            with patch('scripts.survey_ihanoi_cameras.OUTPUT', output):
                with self.assertRaisesRegex(ValueError, 'Unfinished claim'):
                    Survey(resume=True)

    def test_row_center_and_clipped_row(self):
        self.assertEqual(center('[0,560][900,725]'), (450, 642))
        self.assertIsNone(center('[0,1554][900,1600]'))
        self.assertIsNone(center('[0,100][900,200]'))

    def test_last_section_can_expand(self):
        raw = '<hierarchy><node clickable="true" content-desc="Xã Quang Minh" bounds="[0,1481][900,1600]"/></hierarchy>'
        self.assertEqual(next_section(raw, set()), ('Xã Quang Minh', (450, 1540)))
        self.assertIsNone(next_section(raw, {'Xã Quang Minh'}))
