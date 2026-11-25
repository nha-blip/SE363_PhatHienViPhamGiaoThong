from pathlib import Path

import pytest
import yaml

from config.kaggle.prepare_configs import SCHEMAS, build_configs, check_pairs
from config.kaggle import prepare_configs


def roots_for_test(tmp_path):
    roots = {}
    for task in SCHEMAS:
        root = tmp_path / task
        for kind in ('images', 'labels'):
            for split in ('train', 'val', 'test'):
                (root / kind / split).mkdir(parents=True)
        roots[task] = root
    return roots


def test_four_profiles_and_unknown_mount_names(tmp_path):
    roots = roots_for_test(tmp_path)
    configs = build_configs(roots, tmp_path / 'output')
    assert len(configs) == 6
    for task in SCHEMAS:
        for variant, model, batch in (('nano', 'yolov8n.pt', 8), ('small', 'yolov8s.pt', 4)):
            settings = configs[tmp_path / 'output' / f'{task}_{variant}_train.yaml']
            assert settings['model'] == model
            assert settings['batch'] == batch
            assert settings['device'] == 0
            assert settings['resume'] is False
            if task == 'traffic_light':
                assert settings['hsv_h'] == settings['hsv_s'] == 0.0
    light = configs[tmp_path / 'output' / 'traffic_light_dataset.yaml']
    assert yaml.safe_load(yaml.safe_dump(light))['names'][3] == 'off'
    assert light['path'] == roots['traffic_light'].as_posix()


def test_refuses_config_inside_source(tmp_path):
    roots = roots_for_test(tmp_path)
    with pytest.raises(ValueError):
        build_configs(roots, roots['vehicle'] / 'config')


def test_background_pair_kept_and_orphan_rejected(tmp_path, monkeypatch):
    roots = roots_for_test(tmp_path)
    root = roots['traffic_light']
    monkeypatch.setitem(prepare_configs.EXPECTED, 'traffic_light', dict.fromkeys(('train', 'val', 'test'), 1))
    for split in ('train', 'val', 'test'):
        (root / 'images' / split / 'sample.jpg').write_bytes(b'test fixture')
        (root / 'labels' / split / 'sample.txt').write_text('')
    check_pairs(root, 'traffic_light')
    (root / 'labels/train/orphan.txt').write_text('')
    with pytest.raises(ValueError):
        check_pairs(root, 'traffic_light')


def test_templates_consistent_with_profiles(tmp_path):
    root = Path(__file__).resolve().parents[2] / 'config/kaggle'
    configs = build_configs(roots_for_test(tmp_path), tmp_path / 'output')
    for path, generated in configs.items():
        template = yaml.safe_load((root / path.name).read_text())
        if path.name.endswith('_dataset.yaml'):
            template.pop('path')
            generated.pop('path')
        else:
            template.pop('data')
            generated.pop('data')
        assert template == generated
