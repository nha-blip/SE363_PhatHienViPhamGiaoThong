import copy
import unittest

from scripts.finalize_light_corrections import passed, validate_edit


class FinalizeCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {'objects': [{'source_class': 'red', 'proposed_class': 'red', 'bbox_xyxy': [10, 10, 20, 20]}]}
        self.row = {'id': 'sample', 'decision': 'KEEP', 'bbox_scope': 'SIGNAL_HEAD',
                    'class_mapping_confirmed': True, 'notes': '', 'objects': copy.deepcopy(self.baseline['objects']),
                    'geometry_edits': [{'object_number': 1, 'before': [10, 10, 20, 20],
                                        'after': [9, 9, 21, 21], 'coordinate_system': 'native_source_pixels'}]}
        self.row['objects'][0]['bbox_xyxy'] = [9, 9, 21, 21]

    def test_accepts_consistent_confirmed_geometry(self):
        validate_edit(self.baseline, self.row, 1, 'geometry')

    def test_rejects_broken_history(self):
        self.row['geometry_edits'][0]['before'] = [1, 1, 2, 2]
        with self.assertRaises(ValueError):
            validate_edit(self.baseline, self.row, 1, 'geometry')

    def test_keep_with_error_note_is_not_accepted(self):
        self.row['notes'] = 'BBox vẫn quá lớn'
        self.assertFalse(passed(self.row))
        with self.assertRaises(ValueError):
            validate_edit(self.baseline, self.row, 1, 'geometry')

    def test_does_not_allow_unrelated_object_changes(self):
        other = {'source_class': 'green', 'proposed_class': 'green', 'bbox_xyxy': [30, 30, 40, 40]}
        self.baseline['objects'].append(other)
        self.row['objects'].append({**other, 'proposed_class': 'red'})
        with self.assertRaises(ValueError):
            validate_edit(self.baseline, self.row, 1, 'geometry')


if __name__ == '__main__':
    unittest.main()
