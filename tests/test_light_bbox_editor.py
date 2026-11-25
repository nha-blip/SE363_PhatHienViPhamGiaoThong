import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.light_bbox_editor import box_from_drag, source_point, validate_box
from scripts.review_light_dataset import ReviewStore


class GeometryTests(unittest.TestCase):
    def test_zoomed_scrolled_point_to_source(self):
        self.assertEqual(source_point(100, 50, 2, (300, 200, 500, 400)), (350, 225))
        self.assertEqual(source_point(-20, 1000, 2, (300, 200, 500, 400)), (300, 400))

    def test_reverse_drag(self):
        self.assertEqual(box_from_drag((30, 40), (10, 20)), [10, 20, 30, 40])

    def test_reject_invalid_geometry(self):
        for box in ([0, 0, 0, 10], [-1, 0, 10, 10], [0, 0, 101, 10], [0, 0, float('nan'), 10]):
            with self.assertRaises(ValueError):
                validate_box(box, 100, 50)

    def test_save_geometry_is_draft_only_and_resets_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'source.png'
            Image.new('RGB', (100, 50)).save(image)
            original_pixels = image.read_bytes()
            path = root / 'review.json'
            original = {'allow_bbox_edit': True, 'is_ground_truth': False, 'reviews': [{
                'id': 'sample', 'source_image': str(image), 'objects': [{'source_class': 'red', 'proposed_class': 'red', 'bbox_xyxy': [10, 10, 30, 30]}],
                'edit_object_numbers': [1], 'geometry_required': True, 'decision': 'PENDING', 'bbox_scope': 'SIGNAL_HEAD',
                'class_mapping_confirmed': True, 'notes': 'Keep notes'}]}
            path.write_text(json.dumps(original), encoding='utf-8')
            store = ReviewStore(path)
            with self.assertRaises(ValueError):
                store.save(0, 'KEEP', 'SIGNAL_HEAD', True, '')
            self.assertTrue(store.save_geometry(0, 0, [9, 8, 32, 33]))
            row = store.rows[0]
            self.assertEqual(row['objects'][0]['bbox_xyxy'], [9, 8, 32, 33])
            self.assertEqual(row['geometry_edits'][0]['before'], [10, 10, 30, 30])
            self.assertEqual(row['decision'], 'PENDING')
            self.assertEqual(row['bbox_scope'], 'UNREVIEWED')
            self.assertEqual(row['notes'], 'Keep notes')
            self.assertFalse(row['class_mapping_confirmed'])
            self.assertFalse(store.data['is_ground_truth'])
            self.assertEqual(image.read_bytes(), original_pixels)
            self.assertEqual(json.loads(store.backup.read_text()), original)
            store.save(0, 'KEEP', 'SIGNAL_HEAD', True, 'Keep notes')
            self.assertEqual(ReviewStore(path).rows[0]['decision'], 'KEEP')
            store.data['allow_bbox_edit'] = False
            with self.assertRaises(ValueError):
                store.save_geometry(0, 0, [8, 8, 32, 33])


if __name__ == '__main__':
    unittest.main()
