import json
import tempfile
import unittest
from pathlib import Path

from scripts.review_light_dataset import ReviewStore


class FocusReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'review.json'
        obj = {'source_class': 'truck', 'proposed_class': 'truck', 'bbox_xyxy': [1, 1, 5, 5]}
        self.original = {'allow_class_edit': True, 'allow_focus_removal': True, 'is_ground_truth': False,
                         'reviews': [{'id': 'focus', 'decision': 'PENDING', 'bbox_scope': 'UNREVIEWED',
                                      'class_mapping_confirmed': False, 'notes': '', 'allowed_classes': ['car', 'truck'],
                                      'objects': [obj], 'correction': {'action': {'object_number': 1}, 'before': obj, 'after': obj}}]}
        self.path.write_text(json.dumps(self.original), encoding='utf-8')

    def test_class_edit_preserves_original_snapshot(self):
        store = ReviewStore(self.path)
        store.save_class(0, 'car')
        row = store.rows[0]
        self.assertEqual(row['objects'][0]['source_class'], 'car')
        self.assertEqual(row['correction']['before']['source_class'], 'truck')
        self.assertEqual(row['class_edits'][0]['before'], 'truck')
        self.assertFalse(row['class_mapping_confirmed'])
        store.save(0, 'KEEP', 'WHOLE_OBJECT', True, '')
        self.assertEqual(json.loads(store.backup.read_text()), self.original)

    def test_remove_and_restore_last_proposal(self):
        store = ReviewStore(self.path)
        store.save_class(0, 'car')
        store.set_removed(0, True)
        self.assertEqual(store.rows[0]['objects'], [])
        self.assertEqual(store.rows[0]['decision'], 'REMOVE_BOX')
        self.assertIsNone(store.rows[0]['correction']['after'])
        with self.assertRaises(ValueError):
            store.save(0, 'KEEP', 'WHOLE_OBJECT', True, '')
        store.set_removed(0, False)
        self.assertEqual(store.rows[0]['objects'][0]['source_class'], 'car')
        self.assertEqual(store.rows[0]['decision'], 'PENDING')
        self.assertFalse(store.data['is_ground_truth'])

    def test_unsupported_class_and_unauthorized_edit_rejected(self):
        store = ReviewStore(self.path)
        with self.assertRaises(ValueError):
            store.save_class(0, 'bus')
        store.data['allow_class_edit'] = False
        with self.assertRaises(ValueError):
            store.save_class(0, 'car')
        store.data['allow_focus_removal'] = False
        with self.assertRaises(ValueError):
            store.set_removed(0, True)

    def test_remove_decision_needs_explicit_removal(self):
        store = ReviewStore(self.path)
        with self.assertRaises(ValueError):
            store.save(0, 'REMOVE_BOX', 'UNREVIEWED', False, '')


if __name__ == '__main__':
    unittest.main()
