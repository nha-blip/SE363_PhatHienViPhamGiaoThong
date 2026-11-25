import argparse
import hashlib
import json
import math
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'exports/yolo26_kaggle2_20261027'
BASE = ROOT / 'data/YOLO363_readable_crops_20261027'
LIGHT = ROOT / 'data/YOLO363_light_color_supplement_20261027'
EXPECTED = {'vehicle': {'train': 165233, 'val': 21460}, 'light': {'train': 10810, 'val': 1898}}


def label_for(image):
    parts = list(image.parts)
    parts[len(parts)-1-list(reversed(parts)).index('images')] = 'labels'
    return Path(*parts).with_suffix('.txt')


def validate_label(path, content=None):
    count = 0
    for line in (path.read_text(encoding='utf-8-sig') if content is None else content.decode('utf-8-sig')).splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError('Invalid label: ' + str(path))
        identity, center_x, center_y, width, height = values
        if identity not in range(4) or not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError('Invalid normalized label: ' + str(path))
        count += 1
    return count


def images_for(settings, split):
    value = Path(settings[split])
    if not value.is_absolute():
        value = Path(settings['path']) / value
    if value.suffix == '.txt':
        return [Path(line) for line in value.read_text(encoding='utf-8').splitlines() if line.strip()]
    return sorted(path for path in value.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})


def package(kind):
    config = (BASE if kind == 'vehicle' else LIGHT) / kind / 'data.yaml'
    settings = yaml.safe_load(config.read_text(encoding='utf-8'))
    path = OUTPUT / (kind + '_trainval.zip')
    if path.exists():
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError('Existing archive CRC failed')
            counts = json.loads(archive.read(kind + '/PACKAGING.json'))['counts']
            if any(counts[split]['images'] != expected for split, expected in EXPECTED[kind].items()):
                raise ValueError('Existing archive inventory differs')
            members = len(archive.infolist())
        checksum = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                checksum.update(chunk)
        return {'kind': kind, 'archive': str(path), 'bytes': path.stat().st_size, 'sha256': checksum.hexdigest(),
                'members': members, 'counts': counts, 'crc_verified': True}
    temporary = path.with_suffix('.zip.partial')
    counts = {}
    inventory = []
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for split, expected in EXPECTED[kind].items():
            images = images_for(settings, split)
            if len(images) != expected or len({image.stem.casefold() for image in images}) != len(images):
                raise ValueError('Unexpected count or filename collision')
            boxes = 0
            def load(image):
                label = label_for(image)
                label_bytes = label.read_bytes()
                return image, label, image.read_bytes(), label_bytes, validate_label(label, label_bytes)

            with ThreadPoolExecutor(max_workers=8) as readers:
                for offset in range(0, len(images), 64):
                    for image, label, image_bytes, label_bytes, count in readers.map(load, images[offset:offset+64]):
                        boxes += count
                        archive.writestr(f'{kind}/images/{split}/{image.name}', image_bytes)
                        archive.writestr(f'{kind}/labels/{split}/{image.stem}.txt', label_bytes)
                        inventory.append({'split': split, 'image': str(image), 'label': str(label),
                                          'image_bytes': len(image_bytes), 'label_bytes': len(label_bytes)})
                    if offset % 6400 == 0:
                        print(f'{kind}/{split}: {min(offset+64,len(images))}/{len(images)}', flush=True)
            counts[split] = {'images': len(images), 'bbox': boxes}
        portable = {'path': '.', 'train': 'images/train', 'val': 'images/val', 'names': settings['names']}
        archive.writestr(kind + '/data.yaml', yaml.safe_dump(portable, sort_keys=False))
        archive.writestr(kind + '/PACKAGING.json', json.dumps({'counts': counts, 'source_config': str(config), 'test_not_included': True}))
    os.replace(temporary, path)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
        members = len(archive.infolist())
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            checksum.update(chunk)
    (OUTPUT / (kind + '_source_inventory.json')).write_text(json.dumps(inventory), encoding='utf-8')
    return {'kind': kind, 'archive': str(path), 'bytes': path.stat().st_size, 'sha256': checksum.hexdigest(),
            'members': members, 'counts': counts, 'crc_verified': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    approval = json.loads((LIGHT / 'sample_qa_approval.json').read_text())
    if approval['status'] != 'SAMPLE_QA_APPROVED_FOR_CONTROLLED_TRAINING':
        raise ValueError('Latest sample QA not approved')
    OUTPUT.mkdir(parents=True, exist_ok=args.resume)
    if (OUTPUT / 'upload').exists():
        raise ValueError('Upload folder already exists; do not rebuild a submitted package')
    manifest = {'status': 'BUILDING', 'dataset_slug': 'v5nhdf/yolo363-readable-crops-20261027',
                'private_upload_required': True, 'test_included': False, 'source_modified': False, 'archives': []}
    (OUTPUT / 'package_status.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if shutil.disk_usage(ROOT).free < 20*1024**3:
        raise OSError('Need at least 20 GiB free for packaging')
    with ThreadPoolExecutor(max_workers=2) as pool:
        manifest['archives'] = list(pool.map(package, ('vehicle', 'light')))
    upload = OUTPUT / 'upload'
    upload.mkdir()
    for archive in manifest['archives']:
        shutil.move(archive['archive'], upload / Path(archive['archive']).name)
        archive['archive'] = str(upload / Path(archive['archive']).name)
    metadata = {'id': manifest['dataset_slug'], 'title': 'YOLO363 Readable Crops Train Val 20261027',
                'licenses': [{'name': 'other'}],
                'description': 'Private research training package. Vehicle and signal-state crops reviewed by sampling; full-frame validation inherited unchanged. Source datasets retain their original licenses. No credentials, model checkpoints or test images included.'}
    (upload / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    manifest['status'] = 'COMPLETE_CRC_SHA256_VERIFIED'
    (OUTPUT / 'package_status.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
