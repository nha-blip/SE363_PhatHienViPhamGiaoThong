import json
import sys
from unittest.mock import Mock

import torch

from scripts import pause_training_after_checkpoint


def test_stop_only_after_verified_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'source.pt'
    checkpoint.write_bytes(b'owner checkpoint fixture')
    results = tmp_path / 'results.csv'
    results.write_text('epoch\n60\n')
    output = tmp_path / 'output'
    process = Mock()
    process.create_time.return_value = 123
    process.is_running.return_value = True
    monkeypatch.setattr(pause_training_after_checkpoint.psutil, 'Process', lambda pid: process)
    monkeypatch.setattr(pause_training_after_checkpoint.time, 'sleep', lambda duration: None)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'epoch': 59, 'optimizer': {}})
    monkeypatch.setattr(sys, 'argv', ['pause', '--pid', '123', '--completed-epoch', '60',
                                   '--checkpoint', str(checkpoint), '--results', str(results),
                                   '--output', str(output)])
    pause_training_after_checkpoint.main()
    process.terminate.assert_called_once()
    assert (output / 'last.pt').read_bytes() == checkpoint.read_bytes()
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'STOPPED_AFTER_CONFIRMED_CHECKPOINT'
    assert status['completed_epoch'] == 60
    assert status['optimizer_preserved'] is True


def test_no_stop_before_target_epoch(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'source.pt'
    checkpoint.write_bytes(b'previous checkpoint')
    results = tmp_path / 'results.csv'
    results.write_text('epoch\n59\n')
    output = tmp_path / 'output'
    process = Mock()
    process.create_time.return_value = 123
    process.is_running.side_effect = [True, False]
    monkeypatch.setattr(pause_training_after_checkpoint.psutil, 'Process', lambda pid: process)
    monkeypatch.setattr(pause_training_after_checkpoint.time, 'sleep', lambda duration: None)
    monkeypatch.setattr(sys, 'argv', ['pause', '--pid', '123', '--completed-epoch', '60',
                                   '--checkpoint', str(checkpoint), '--results', str(results),
                                   '--output', str(output)])
    pause_training_after_checkpoint.main()
    process.terminate.assert_not_called()
    assert not (output / 'last.pt').exists()
