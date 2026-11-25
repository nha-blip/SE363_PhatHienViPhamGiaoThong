import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from config.kaggle.yolo26_vehicle_local_20261027 import train


class VehicleLocalTests(unittest.TestCase):
    def test_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'file'
            path.write_bytes(b'abc')
            self.assertEqual(train.digest_file(path),
                             'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('vehicle/images/train/one.jpg', b'image')
                archive.writestr('vehicle/labels/train/one.txt', b'0 0.5 0.5 0.2 0.2')
            with zipfile.ZipFile(path) as archive, patch.object(train, 'COUNTS', {'train': 1}):
                self.assertEqual(len(train.checked_members(archive)), 2)

    def test_reject_unpaired_and_traversal(self):
        for name in ('vehicle/images/train/one.jpg', 'vehicle/../escape'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'data.zip'
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr(name, b'data')
                with zipfile.ZipFile(path) as archive, patch.object(train, 'COUNTS', {'train': 1}):
                    with self.assertRaises(ValueError):
                        train.checked_members(archive)
