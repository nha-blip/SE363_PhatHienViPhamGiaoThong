import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config.kaggle.yolo26_parallel_20261027 import train
from scripts.package_readable_kaggle import label_for, validate_label


class KaggleTrainingTests(unittest.TestCase):
    def test_label_mapping(self):
        self.assertEqual(label_for(Path('data/task/images/train/car.jpg')), Path('data/task/labels/train/car.txt'))

    def test_label_validation_from_bytes(self):
        self.assertEqual(validate_label(Path('unused'), b'0 0.5 0.5 0.2 0.3\n'), 1)
        with self.assertRaises(ValueError):
            validate_label(Path('unused'), b'4 0.5 0.5 0.2 0.3\n')

    def test_light_worker_checkpoint_and_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            last = root / 'source_last.pt'
            best = root / 'source_best.pt'
            last.write_bytes(b'last with optimizer')
            best.write_bytes(b'best with optimizer')
            callbacks = {}
            settings = {}

            class Model:
                def add_callback(self, name, function):
                    callbacks[name] = function

                def train(self, **kwargs):
                    settings.update(kwargs)
                    callbacks['on_model_save'](SimpleNamespace(last=last, best=best, epoch=2, best_fitness=0.6))

            with patch.object(train, 'WORK', root), patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda path: Model())}):
                train.worker('light')
            self.assertEqual(settings['hsv_h'], 0)
            self.assertEqual(settings['mosaic'], 0)
            self.assertEqual(settings['device'], 0)
            self.assertEqual(settings['save_period'], -1)
            self.assertEqual(settings['epochs'], 100)
            self.assertNotIn('time', settings)
            self.assertEqual((root / 'light/checkpoints/last.pt').read_bytes(), last.read_bytes())
            self.assertEqual((root / 'light/checkpoints/best.pt').read_bytes(), best.read_bytes())
            self.assertEqual(json.loads((root / 'light/epoch_status.json').read_text())['epoch_completed'], 3)

    def test_metadata_private_and_no_secret_sources(self):
        metadata = json.loads((Path(train.__file__).parent / 'kernel-metadata.json').read_text())
        self.assertTrue(metadata['is_private'])
        self.assertEqual(metadata['dataset_sources'], ['v5nhdf/yolo363-readable-crops-20261027'])
        self.assertTrue(metadata['enable_gpu'])
