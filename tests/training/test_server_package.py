import pytest
import yaml
import hashlib
import json
import zipfile

from config.server import configure_server
from scripts.package_yolo_server import validate_label
from scripts import package_yolo_server


def test_label_validation_keeps_background_and_rejects_bad_values(tmp_path):
    label = tmp_path / 'label.txt'
    label.write_text('', encoding='utf-8')
    assert validate_label(label) == 0
    label.write_text('3 0.5 0.5 0.2 0.3\n', encoding='utf-8')
    assert validate_label(label) == 1
    for invalid in ('4 0.5 0.5 0.2 0.3', '0 nan 0.5 0.2 0.3', '0 0.5 0.5 0 0.3'):
        label.write_text(invalid, encoding='utf-8')
        with pytest.raises(ValueError):
            validate_label(label)


def test_server_configs_resolve_portable_root(tmp_path, monkeypatch):
    config = tmp_path / 'config/server'
    config.mkdir(parents=True)
    for dataset, classes in configure_server.DATASETS.values():
        for kind in ('images', 'labels'):
            for split in ('train', 'val', 'test'):
                (tmp_path / 'data' / dataset / kind / split).mkdir(parents=True)
    monkeypatch.setattr(configure_server, 'ROOT', tmp_path)
    monkeypatch.setattr(configure_server, 'CONFIG', config)
    configure_server.configure()
    assert len(list(config.glob('*.yaml'))) == 6
    for task in configure_server.DATASETS:
        for variant, model in (('nano', 'yolov8n.pt'), ('small', 'yolov8s.pt')):
            settings = yaml.safe_load((config / f'{task}_{variant}_train.yaml').read_text())
            assert settings['model'] == str(tmp_path / 'weights' / model)
            assert settings['data'] == str(config / f'{task}_dataset.yaml')
            assert settings['resume'] is False
            assert settings['device'] == 0
            assert settings['workers'] == 2
    light = yaml.safe_load((config / 'traffic_light_dataset.yaml').read_text())
    assert light['names'][3] == 'off'


def test_missing_dataset_does_not_generate_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(configure_server, 'ROOT', tmp_path)
    monkeypatch.setattr(configure_server, 'CONFIG', tmp_path)
    with pytest.raises(FileNotFoundError):
        configure_server.configure()
    assert not list(tmp_path.glob('*.yaml'))


def test_config_refresh_includes_lf_slurm_and_keeps_dataset_archive(tmp_path, monkeypatch):
    output = tmp_path / 'export'
    output.mkdir()
    config = tmp_path / 'config/server'
    config.mkdir(parents=True)
    for name in ('configure_server.py', 'README.md', 'train.slurm'):
        (config / name).write_bytes(b'first\r\nsecond\r\n')
    dataset = output / 'Vehicle363-v1.zip'
    dataset.write_bytes(b'unchanged dataset archive')
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps({
        'status': 'READY_FOR_UPLOAD', 'archives': [
            {'file': dataset.name, 'sha256': dataset_hash},
            {'file': 'server_config.zip', 'sha256': 'old'},
        ],
    }))
    monkeypatch.setattr(package_yolo_server, 'ROOT', tmp_path)
    monkeypatch.setattr(package_yolo_server, 'OUTPUT', output)
    package_yolo_server.refresh_config_bundle()
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == dataset_hash
    assert b'\r' not in (output / 'SHA256SUMS.txt').read_bytes()
    with zipfile.ZipFile(output / 'server_config.zip') as archive:
        assert len(archive.namelist()) == 3
        assert archive.read('config/server/train.slurm') == b'first\nsecond\n'
