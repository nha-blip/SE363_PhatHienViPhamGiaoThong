import json
import tempfile
import unittest
from pathlib import Path

from scripts.review_light_dataset import ReviewStore, correction_objects


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'review.json'
        self.original = {'is_ground_truth': False, 'status': 'PENDING_HUMAN_QA', 'reviews': [
            {'id': 'sample', 'decision': 'PENDING', 'bbox_scope': 'UNREVIEWED',
             'class_mapping_confirmed': False, 'notes': '',
             'objects': [{'source_class': 'red', 'proposed_class': 'red', 'bbox_xyxy': [1, 2, 3, 4]}]}]}
        self.path.write_text(json.dumps(self.original), encoding='utf-8')

    def test_backup_and_preserve_source_labels(self):
        store = ReviewStore(self.path)
        store.save(0, 'KEEP', 'SIGNAL_HEAD', True, 'Đúng')
        saved = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(saved['reviews'][0]['objects'], self.original['reviews'][0]['objects'])
        self.assertEqual(json.loads(store.backup.read_text()), self.original)
        self.assertFalse(saved['is_ground_truth'])
        self.assertEqual(saved['status'], 'HUMAN_QA_COMPLETED_REQUIRES_VALIDATION')
        store.save(0, 'NEEDS_FIX', 'SIGNAL_HEAD', False, 'Sửa')
        self.assertEqual(len(list(self.path.parent.glob('review_backups/*.json'))), 1)

    def test_noop_does_not_write(self):
        store = ReviewStore(self.path)
        self.assertFalse(store.save(0, 'PENDING', 'UNREVIEWED', False, ''))
        self.assertIsNone(store.backup)

    def test_external_change_is_not_overwritten(self):
        store = ReviewStore(self.path)
        self.path.write_text('external edit', encoding='utf-8')
        with self.assertRaises(RuntimeError):
            store.save(0, 'KEEP', 'SIGNAL_HEAD', False, '')
        self.assertEqual(self.path.read_text(), 'external edit')
        self.assertEqual(store.rows[0]['decision'], 'PENDING')

    def test_unknown_mapping_cannot_be_confirmed(self):
        data = self.original
        data['reviews'][0]['objects'][0]['proposed_class'] = None
        self.path.write_text(json.dumps(data), encoding='utf-8')
        store = ReviewStore(self.path)
        with self.assertRaises(ValueError):
            store.save(0, 'KEEP', 'SIGNAL_HEAD', True, '')

    def test_resume_first_pending(self):
        store = ReviewStore(self.path)
        self.assertEqual(store.first_pending(), 0)
        store.save(0, 'UNCERTAIN', 'UNCERTAIN', False, 'Chưa rõ')
        reopened = ReviewStore(self.path)
        self.assertEqual(reopened.rows[0]['decision'], 'UNCERTAIN')

    def test_reconstruct_removed_box_without_mutating_review(self):
        row = {'objects': [{'source_class': 'red'}], 'correction': {
            'action': {'object_number': 1}, 'before': {'source_class': 'wait_on'}, 'after': None}}
        before, after = correction_objects(row)
        self.assertEqual([obj['source_class'] for obj in before], ['wait_on', 'red'])
        self.assertEqual([obj['source_class'] for obj in after], ['red'])
        self.assertEqual(len(row['objects']), 1)

    def test_reconstruct_previous_class(self):
        row = {'objects': [{'source_class': 'red'}], 'correction': {
            'action': {'object_number': 1}, 'before': {'source_class': 'off'}, 'after': {'source_class': 'red'}}}
        before, after = correction_objects(row)
        self.assertEqual(before[0]['source_class'], 'off')
        self.assertEqual(after[0]['source_class'], 'red')


if __name__ == '__main__':
    unittest.main()
