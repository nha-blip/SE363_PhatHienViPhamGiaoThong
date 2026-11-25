import json
import tempfile
import unittest
from pathlib import Path

import psutil

from scripts.run_light_ab_experiment import blocker_alive, digest, metric_record, select_winner, write_json


class LightExperimentTests(unittest.TestCase):
    def test_winner_uses_validation_not_test(self):
        metrics = {'baseline': {'combined': {'map5095': 0.2}},
                   'A_unique': {'combined': {'map5095': 0.3}},
                   'B_rare3': {'combined': {'map5095': 0.25}}}
        self.assertEqual(select_winner(metrics), 'A_unique')

    def test_baseline_kept_when_tied(self):
        metrics = {name: {'combined': {'map5095': 0.2}} for name in ('baseline', 'A_unique', 'B_rare3')}
        self.assertEqual(select_winner(metrics), 'baseline')

    def test_process_identity_checks_creation_time(self):
        process = psutil.Process()
        self.assertTrue(blocker_alive(process.pid, process.create_time()))
        self.assertFalse(blocker_alive(process.pid, process.create_time() + 1))

    def test_atomic_json_and_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'status.json'
            write_json(path, {'status': 'WAITING_GPU'})
            initial = digest(path)
            write_json(path, {'status': 'COMPLETE'})
            self.assertEqual(json.loads(path.read_text())['status'], 'COMPLETE')
            self.assertNotEqual(initial, digest(path))
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_class_metrics_respect_class_index(self):
        class Box:
            map50 = 0.5
            map = 0.3
            ap_class_index = [1, 3]
            ap50 = [0.4, 0.6]
            ap = [0.2, 0.4]
            p = [0.7, 0.8]
            r = [0.3, 0.5]

        class Result:
            box = Box()
            speed = {'inference': 10.0}

        result = metric_record(Result())
        self.assertEqual(result['per_class']['yellow']['ap50'], 0.4)
        self.assertEqual(result['per_class']['off']['recall'], 0.5)


if __name__ == '__main__':
    unittest.main()
