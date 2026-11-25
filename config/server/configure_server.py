import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent
DATASETS = {'vehicle': ('Vehicle363-v1', ['car', 'motorcycle', 'bus', 'truck']),
            'traffic_light': ('TrafficLight363-v1', ['red', 'yellow', 'green', 'off'])}


def configure():
    for task, (dataset, classes) in DATASETS.items():
        directory = ROOT / 'data' / dataset
        for kind in ('images', 'labels'):
            for split in ('train', 'val', 'test'):
                if not (directory / kind / split).is_dir():
                    raise FileNotFoundError(directory / kind / split)
    for task, (dataset, classes) in DATASETS.items():
        dataset_config = CONFIG / f'{task}_dataset.yaml'
        dataset_config.write_text(yaml.safe_dump({
            'path': str(ROOT / 'data' / dataset), 'train': 'images/train',
            'val': 'images/val', 'test': 'images/test',
            'names': dict(enumerate(classes)),
        }, sort_keys=False), encoding='utf-8')
        for variant in ('nano', 'small'):
            configuration = {
                'task': 'detect', 'mode': 'train',
                'model': str(ROOT / 'weights' / ('yolov8n.pt' if variant == 'nano' else 'yolov8s.pt')),
                'data': str(dataset_config), 'device': 0, 'epochs': 100,
                'imgsz': 640, 'batch': 8 if variant == 'nano' else 4,
                'workers': 2, 'patience': 20, 'seed': 363,
                'deterministic': True, 'pretrained': True, 'amp': True,
                'cache': False, 'optimizer': 'AdamW', 'lr0': 0.001,
                'save': True, 'save_period': 5, 'val': True, 'plots': True,
                'project': str(ROOT / 'runs' / 'server_training'),
                'name': f'{task}_{variant}_100', 'exist_ok': False,
                'resume': False, 'single_cls': False,
            }
            if task == 'traffic_light':
                configuration.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.2)
            (CONFIG / f'{task}_{variant}_train.yaml').write_text(
                yaml.safe_dump(configuration, sort_keys=False), encoding='utf-8')
    (ROOT / 'weights').mkdir(exist_ok=True)
    print(json.dumps({'project_root': str(ROOT), 'configs_created': 6,
                      'training_started': False}, indent=2))


if __name__ == '__main__':
    configure()
