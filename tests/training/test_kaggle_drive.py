import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from config.kaggle import train_drive


def test_backup_receipt_uploaded_last(tmp_path):
    last = tmp_path / 'last.pt'
    best = tmp_path / 'best.pt'
    last.write_bytes(b'current epoch')
    best.write_bytes(b'best so far')
    backup = train_drive.DriveBackup('rclone', tmp_path / 'private.conf', 'gdrive2:run', tmp_path)
    backup.copy = Mock()
    backup.after_save(SimpleNamespace(epoch=4, last=last, best=best))
    destinations = [call.args[1] for call in backup.copy.call_args_list]
    assert destinations == [
        'gdrive2:run/epoch_0005/last.pt',
        'gdrive2:run/epoch_0005/best.pt',
        'gdrive2:run/epoch_0005/UPLOAD_COMPLETE.json',
    ]
    receipt = json.loads((tmp_path / 'receipts/epoch_0005.json').read_text())
    assert receipt['files']['last.pt']['sha256'] == train_drive.digest(last)
    assert backup.completed_epochs == [5]
    backup.after_train(SimpleNamespace(epoch=4, last=last, best=best))
    assert backup.copy.call_args.args[1] == 'gdrive2:run/final/UPLOAD_COMPLETE.json'
    assert backup.completed_epochs == [5]


def test_backup_failure_never_uploads_receipt(tmp_path):
    last = tmp_path / 'last.pt'
    best = tmp_path / 'best.pt'
    last.write_bytes(b'current')
    best.write_bytes(b'best')
    backup = train_drive.DriveBackup('rclone', tmp_path / 'private.conf', 'gdrive2:run', tmp_path)
    backup.copy = Mock(side_effect=[None, RuntimeError('upload failed')])
    with pytest.raises(RuntimeError):
        backup.after_save(SimpleNamespace(epoch=0, last=last, best=best))
    assert backup.copy.call_count == 2
    assert backup.completed_epochs == []
    assert json.loads((tmp_path / 'backup_status.json').read_text())['status'] == 'FAILED'
    assert last.is_file()


@pytest.mark.parametrize('task,device', [('vehicle', 0), ('traffic_light', 1)])
@pytest.mark.parametrize('variant', ['nano', 'small'])
def test_worker_device_and_save_policy(tmp_path, monkeypatch, task, device, variant):
    configuration = tmp_path / 'train.yaml'
    configuration.write_text(yaml.safe_dump({'save': False, 'save_period': 5, 'device': 0}))
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=factory))
    monkeypatch.setattr(train_drive.shutil, 'which', lambda name: 'rclone')
    arguments = SimpleNamespace(config=configuration, output=tmp_path / 'output', task=task,
                                weights=train_drive.MODELS[variant],
                                rclone_config=tmp_path / 'secret.conf', destination='gdrive2:run')
    train_drive.worker(arguments)
    factory.assert_called_once_with(train_drive.MODELS[variant])
    settings = model.train.call_args.kwargs
    assert settings['device'] == device
    assert settings['save'] is True
    assert settings['save_period'] == -1
    assert settings['resume'] is False
    assert [call.args[0] for call in model.add_callback.call_args_list] == ['on_model_save', 'on_train_end']


@pytest.mark.parametrize('variant', ['nano', 'small'])
def test_notebook_cells_compile_and_wait(variant):
    path = Path(__file__).resolve().parents[2] / f'config/kaggle/notebook_{variant}.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for source in sources:
        compile(source, str(path), 'exec')
    assert f'"{variant}"' in sources[-1]
    assert '"--wait"' in sources[-1]
    assert 'base64.b64decode' in sources[-2]
