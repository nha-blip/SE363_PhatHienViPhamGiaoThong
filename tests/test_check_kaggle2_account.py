import unittest
from datetime import timedelta
from types import SimpleNamespace

from scripts.check_kaggle2_account import quota_hours


class KaggleQuotaTests(unittest.TestCase):
    def test_remaining_hours(self):
        result = quota_hours(SimpleNamespace(time_used=timedelta(hours=2), total_time_allowed=timedelta(hours=30)))
        self.assertEqual(result, {'used_hours': 2, 'total_hours': 30, 'remaining_hours': 28})

    def test_absent_quota(self):
        self.assertIsNone(quota_hours(None))

    def test_overrun_is_clamped(self):
        result = quota_hours(SimpleNamespace(time_used=timedelta(hours=32), total_time_allowed=timedelta(hours=30)))
        self.assertEqual(result['remaining_hours'], 0)
