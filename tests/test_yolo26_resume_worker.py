import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config.kaggle.yolo26_vehicle_local_20261027 import train


class ResumeWorkerTests(unittest.TestCase):
    def run_worker(self, start_epoch, expected_epoch=11):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'light').mkdir()
            dataset = root / 'dataset'
            dataset.mkdir()
            (dataset / 'data.yaml').write_text('names: [red, yellow, green, off]\ntrain: images/train\nval: images/val\n')
            resume = root / 'last.pt'
            best = root / 'best.pt'
            results = root / 'results.csv'
            resume.write_bytes(b'optimizer checkpoint')
            best.write_bytes(b'previous best')
            results.write_text('epoch,metric\n11,0.6\n')
            trainer_best = root / 'restored_best.pt'
            trainer_csv = root / 'restored_results.csv'
            callbacks = {}
            captured = {}

            class Model:
                def add_callback(self, name, function):
                    callbacks[name] = function

                def train(self, **kwargs):
                    captured.update(kwargs)
                    callbacks['on_train_start'](SimpleNamespace(start_epoch=start_epoch,
                        best=trainer_best, csv=trainer_csv, batch_size=16, device='cuda:0', train_loader=[1]))

            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
            with patch.object(train, 'WORK', root), patch.object(train, 'stage_light', return_value=dataset), \
                    patch.dict('sys.modules', {'torch': fake_torch, 'ultralytics': SimpleNamespace(YOLO=lambda path: Model())}):
                train.worker('light', resume, best, results, resume_start_epoch=expected_epoch)
            self.assertTrue(captured['resume'])
            self.assertEqual(captured['data'], str(root / 'light/data.yaml'))
            self.assertEqual(captured['save_dir'], str(root / 'light/train'))
            self.assertEqual(trainer_best.read_bytes(), best.read_bytes())
            self.assertEqual(trainer_csv.read_bytes(), results.read_bytes())

    def test_resume_keeps_history_best_and_new_data_path(self):
        self.run_worker(11)

    def test_wrong_resume_epoch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'epoch 12'):
            self.run_worker(0)

    def test_later_light_checkpoint_resumes_exact_epoch(self):
        self.run_worker(42, expected_epoch=42)
