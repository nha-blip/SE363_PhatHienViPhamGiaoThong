import argparse
from pathlib import Path

import yaml


SCHEMAS = {'vehicle': {0: 'car', 1: 'motorcycle', 2: 'bus', 3: 'truck'},
           'traffic_light': {0: 'red', 1: 'yellow', 2: 'green', 3: 'off'}}
EXPECTED = {'vehicle': {'train': 110478, 'val': 21460, 'test': 21236},
            'traffic_light': {'train': 5541, 'val': 933, 'test': 890}}


def build_configs(roots, output, device=0):
    output = Path(output).resolve()
    resolved = {task: Path(root).resolve(strict=True) for task, root in roots.items()}
    for root in resolved.values():
        if output == root or root in output.parents:
            raise ValueError('Output configs must be outside input datasets')
        for split in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                directory = root / kind / split
                if not directory.is_dir() or root not in directory.resolve().parents:
                    raise ValueError(f'Missing or escaped dataset directory: {directory}')
    configs = {}
    for task, root in resolved.items():
        dataset = output / f'{task}_dataset.yaml'
        configs[dataset] = {'path': root.as_posix(), 'train': 'images/train',
                            'val': 'images/val', 'test': 'images/test', 'names': SCHEMAS[task]}
        for variant, model, batch in (('nano', 'yolov8n.pt', 8), ('small', 'yolov8s.pt', 4)):
            train = {'task': 'detect', 'mode': 'train', 'model': model,
                     'data': dataset.as_posix(), 'device': device, 'epochs': 100,
                     'imgsz': 640, 'batch': batch, 'workers': 2, 'patience': 20,
                     'seed': 363, 'deterministic': True, 'pretrained': True,
                     'amp': True, 'cache': False, 'optimizer': 'AdamW', 'lr0': 0.001,
                     'save': True, 'save_period': 5, 'val': True, 'plots': True,
                     'project': '/kaggle/working/runs', 'name': f'{task}_{variant}_100',
                     'exist_ok': False, 'resume': False, 'single_cls': False}
            if task == 'traffic_light':
                train.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.2)
            configs[output / f'{task}_{variant}_train.yaml'] = train
    return configs


def check_pairs(root, task):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    for split, expected in EXPECTED[task].items():
        images = [path for path in (root / 'images' / split).rglob('*') if path.is_file() and path.suffix.lower() in extensions]
        labels = list((root / 'labels' / split).rglob('*.txt'))
        image_keys = {path.relative_to(root / 'images' / split).with_suffix('') for path in images}
        label_keys = {path.relative_to(root / 'labels' / split).with_suffix('') for path in labels}
        if len(images) != expected or len(labels) != expected or len(image_keys) != expected or image_keys != label_keys:
            raise ValueError(f'{task}/{split}: image/label pairing or expected count mismatch')


def main():
    parser = argparse.ArgumentParser(description='Create six Kaggle configs from explicit mounted dataset roots; no training')
    parser.add_argument('--vehicle-root', required=True)
    parser.add_argument('--traffic-light-root', required=True)
    parser.add_argument('--output-dir', default='/kaggle/working/configs')
    parser.add_argument('--device', type=int, choices=(0, 1), default=0)
    parser.add_argument('--check-pairs', action='store_true')
    args = parser.parse_args()
    roots = {'vehicle': Path(args.vehicle_root), 'traffic_light': Path(args.traffic_light_root)}
    configs = build_configs(roots, args.output_dir, args.device)
    output = Path(args.output_dir).resolve()
    if output == Path('/kaggle/input') or Path('/kaggle/input') in output.parents:
        raise ValueError('Kaggle input is read-only; use /kaggle/working/configs')
    if args.check_pairs:
        for task, root in roots.items():
            check_pairs(root.resolve(strict=True), task)
    payloads = {path: yaml.safe_dump(value, sort_keys=False) for path, value in configs.items()}
    for path, content in payloads.items():
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise FileExistsError(f'Config differs: {path}; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    for path, content in payloads.items():
        if not path.exists():
            path.write_text(content, encoding='utf-8', newline='\n')
    print(f'Created/reused {len(configs)} configs in {output}; training not started')


if __name__ == '__main__':
    main()
