import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.build_light_expansion_pool import sequence_group, verify_fingerprint


class ExpansionPoolTests(unittest.TestCase):
    def test_720_sequences_are_atomic(self):
        first = {'source': 'S2TLD_720x1280', 'image_member': 'archive/normal_1/JPEGImages/000001.jpg'}
        second = {**first, 'image_member': 'archive/normal_1/JPEGImages/000200.jpg'}
        self.assertEqual(sequence_group(first), sequence_group(second))
        self.assertNotEqual(sequence_group(first), sequence_group({**first, 'image_member': 'archive/normal_2/JPEGImages/000001.jpg'}))

    def test_1080_same_capture_day_stays_together(self):
        row = {'source': 'S2TLD_1080x1920', 'image_member': 'archive/JPEGImages/2020-03-30 11_30_04.803536736.jpg'}
        self.assertEqual(sequence_group(row), 'S2TLD_1080x1920/2020-03-30')

    def test_atlas_synchronized_cameras_share_group(self):
        first = {'source': 'ATLAS_sample', 'image_member': 'ATLAS/train/front_medium/images/front_medium_1722622519-50002175.jpg'}
        second = {**first, 'image_member': 'ATLAS/train/front_tele/images/front_tele_1722622519-50002175.jpg'}
        self.assertEqual(sequence_group(first), sequence_group(second))

    def test_changed_image_blocks_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.jpg'
            path.write_bytes(b'original')
            row = {'path': str(path), 'sha256': hashlib.sha256(b'original').hexdigest(), 'dhash': '0000000000000000'}
            self.assertEqual(verify_fingerprint(row), str(path))
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_fingerprint(row)

    def test_unknown_sequence_is_not_randomly_assigned(self):
        with self.assertRaises(ValueError):
            sequence_group({'source': 'S2TLD_720x1280', 'image_member': 'unknown/0001.jpg'})


if __name__ == '__main__':
    unittest.main()
