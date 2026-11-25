import unittest

from scripts.resume_vehicle_epoch30_20261031 import require_resumable


def checkpoint(**changes):
    metadata = {'completed_epochs': 30, 'target_epochs': 100, 'optimizer_present': True, 'best_epoch': 30, 'patience': 15,
                'names': {0: 'car', 1: 'motorcycle', 2: 'bus', 3: 'truck'}}
    metadata.update(changes)
    return metadata


class RequireResumableTests(unittest.TestCase):
    def test_accepts_epoch_30_with_optimizer(self):
        require_resumable(checkpoint())

    def test_rejects_other_epoch_missing_optimizer_or_classes(self):
        for changes in ({'completed_epochs': 27}, {'optimizer_present': False}, {'names': {0: 'red', 1: 'yellow'}}):
            with self.assertRaises((RuntimeError, ValueError)):
                require_resumable(checkpoint(**changes))

    def test_rejects_early_stopped_checkpoint(self):
        with self.assertRaises(RuntimeError):
            require_resumable(checkpoint(best_epoch=10))


if __name__ == '__main__':
    unittest.main()
