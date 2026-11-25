import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.review_bbox_readability import render, size_bucket


class ReadabilityReviewTests(unittest.TestCase):
    def test_bucket_edges(self):
        self.assertEqual(size_bucket(3.99), 'under4')
        self.assertEqual(size_bucket(4), '4to8')
        self.assertEqual(size_bucket(8), '8to12')
        self.assertEqual(size_bucket(32), '32plus')

    def test_preview_does_not_change_source(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'source.png'
            destination = Path(folder) / 'preview.png'
            Image.new('RGB', (1280, 720), 'gray').save(image)
            original = image.read_bytes()
            render({'id': 'test', 'kind': 'light', 'class_name': 'red',
                    'image': str(image), 'box': [0, 0, 3, 8],
                    'width': 3, 'height': 8}, destination)
            with Image.open(destination) as opened:
                self.assertEqual(opened.size, (1200, 850))
            self.assertEqual(image.read_bytes(), original)
