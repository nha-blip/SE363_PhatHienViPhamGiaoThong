import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import review_drive_training


@pytest.mark.parametrize('corrupt', [False, True])
def test_only_completed_models_download_and_verify(tmp_path, monkeypatch, corrupt):
    data = b'checkpoint test fixture'
    sha256 = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    completed = 'nano/run/vehicle'
    pending = 'small/run/vehicle'
    inventory = []
    for path in (
        f'{completed}/epoch_0054/last.pt',
        f'{completed}/epoch_0054/UPLOAD_COMPLETE.json',
        f'{completed}/TRAIN_COMPLETE.json', f'{completed}/best.pt',
        f'{pending}/epoch_0058/last.pt', f'{pending}/epoch_0058/UPLOAD_COMPLETE.json',
    ):
        inventory.append({'Path': path, 'Name': path.rsplit('/', 1)[-1],
                          'ModTime': '2026-10-18T00:00:00Z'})
    inventory_path = tmp_path / 'inventory.json'
    inventory_path.write_text(json.dumps(inventory))
    receipt = {'epoch': 54, 'best_sha256_at_backup': sha256}
    payloads = {
        f'{completed}/TRAIN_COMPLETE.json': {'completed_epoch': 54},
        f'{completed}/epoch_0054/UPLOAD_COMPLETE.json': receipt,
        f'{pending}/epoch_0058/UPLOAD_COMPLETE.json': {'epoch': 58},
    }

    def fake_run(command, **kwargs):
        operation = command[1]
        path = command[2].removeprefix('gdrive2:')
        if operation == 'cat':
            output = json.dumps(payloads[path])
        elif operation == 'lsjson':
            output = json.dumps({'Size': len(data), 'Hashes': {'md5': md5}})
        elif operation == 'copyto':
            assert path == f'{completed}/best.pt'
            Path(command[3]).write_bytes(b'corrupted' if corrupt else data)
            output = ''
        else:
            raise AssertionError(operation)
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(review_drive_training.subprocess, 'run', fake_run)
    arguments = SimpleNamespace(rclone='rclone', folder_id='folder', inventory=inventory_path,
                                output=tmp_path / 'models', report=tmp_path / 'report.json')
    if corrupt:
        with pytest.raises(RuntimeError, match='verification failed'):
            review_drive_training.review(arguments)
        assert not (arguments.output / 'nano/vehicle/best.pt').exists()
    else:
        review_drive_training.review(arguments)
        report = json.loads(arguments.report.read_text())
        assert report['models'][0]['status'] == 'DOWNLOADED_VERIFIED'
        assert report['models'][0]['receipt_sha256_verified'] is True
        assert report['models'][1]['status'] == 'NO_CONFIRMED_COMPLETION_NOT_DOWNLOADED'
        assert not (arguments.output / 'small/vehicle/best.pt').exists()
