import argparse
import hashlib
import json
import math
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config.server.configure_server import DATASETS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/YOLO363_training_working'
OUTPUT = ROOT / 'exports/yolo363_server_20261013'
EXPECTED = {'Vehicle363-v1': {'train': 110478, 'val': 21460, 'test': 21236},
            'TrafficLight363-v1': {'train': 5541, 'val': 933, 'test': 890}}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def validate_label(path):
    count = 0
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'{path}:{line_number}: expected five fields')
        class_id, center_x, center_y, width, height = map(float, parts)
        if not all(math.isfinite(value) for value in (class_id, center_x, center_y, width, height)):
            raise ValueError(f'{path}:{line_number}: nonfinite values')
        if class_id not in (0, 1, 2, 3) or not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f'{path}:{line_number}: invalid class or normalized bbox')
        count += 1
    return count


def inventory(dataset):
    records = []
    counts = {}
    for split, expected in EXPECTED[dataset].items():
        with os.scandir(SOURCE / dataset / 'images' / split) as entries:
            images = sorted(Path(entry.path) for entry in entries if entry.is_file() and Path(entry.name).suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        with os.scandir(SOURCE / dataset / 'labels' / split) as entries:
            labels = sorted(Path(entry.path) for entry in entries if entry.is_file() and entry.name.endswith('.txt'))
        image_stems = {path.stem for path in images}
        if len(images) != expected or len(labels) != expected or len(image_stems) != expected:
            raise ValueError(f'{dataset}/{split}: unexpected count or duplicate image stems')
        if image_stems != {path.stem for path in labels}:
            raise ValueError(f'{dataset}/{split}: image/label mismatch')
        boxes = 0
        empty = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            for offset in range(0, len(labels), 512):
                for label_boxes in executor.map(validate_label, labels[offset:offset + 512]):
                    boxes += label_boxes
                    empty += int(label_boxes == 0)
                if offset % 25600 == 0:
                    print(f'{dataset}/{split}: checked {min(offset + 512, len(labels))}/{len(labels)} labels', flush=True)
        counts[split] = {'images': len(images), 'labels': len(labels), 'boxes': boxes, 'empty_labels': empty}
        records.extend(images + labels)
    return records, counts


def finalize_archive(path):
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f'CRC mismatch: {bad_member}')
        members = len(archive.infolist())
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path),
            'archive_members': members, 'crc_verified': True}


def build():
    if OUTPUT.exists():
        raise FileExistsError(f'Refusing to overwrite: {OUTPUT}')
    validated = {}
    for task, (dataset, classes) in DATASETS.items():
        print(f'Validating {dataset}', flush=True)
        validated[dataset] = inventory(dataset)
    required_bytes = sum(path.stat().st_size for paths, counts in validated.values() for path in paths)
    if shutil.disk_usage(ROOT).free < required_bytes + 1024 ** 3:
        raise OSError('Insufficient disk space for ZIP_STORED archives')
    OUTPUT.mkdir(parents=True)
    manifest = {'status': 'BUILDING', 'source_modified': False, 'label_validation': 'class IDs, finite normalized center/size, pairing and expected split counts; not visual QA',
                'archives': [], 'datasets': {}, 'training_started': False,
                'compression': 'ZIP_STORED; JPEG already compressed; ZIP64 enabled',
                'excluded': ['loader caches', 'tracking', 'VLM', 'trained checkpoints'],
                'builder_sha256': digest(Path(__file__))}
    for dataset, (paths, counts) in validated.items():
        print(f'Packing {dataset}: {len(paths)} files', flush=True)
        archive_path = OUTPUT / f'{dataset}.zip'
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in paths:
                archive.write(path, (Path('data') / path.relative_to(SOURCE)).as_posix())
        manifest['datasets'][dataset] = counts
        manifest['archives'].append(finalize_archive(archive_path))
        print(f'{dataset}: archive CRC and SHA256 verified', flush=True)
    config_archive = OUTPUT / 'server_config.zip'
    with zipfile.ZipFile(config_archive, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('configure_server.py', 'README.md', 'train.slurm'):
            path = ROOT / 'config/server' / name
            archive.write(path, f'config/server/{name}')
    manifest['archives'].append(finalize_archive(config_archive))
    manifest['status'] = 'READY_FOR_UPLOAD'
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (OUTPUT / 'SHA256SUMS.txt').write_bytes(''.join(f"{record['sha256']}  {record['file']}\n" for record in manifest['archives']).encode('utf-8'))
    shutil.copyfile(ROOT / 'config/server/README.md', OUTPUT / 'README_UPLOAD.md')
    print(f'READY_FOR_UPLOAD: {OUTPUT}', flush=True)


def refresh_config_bundle():
    manifest_path = OUTPUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['status'] != 'READY_FOR_UPLOAD':
        raise ValueError('Dataset packaging must finish before configuration refresh')
    temporary = OUTPUT / 'server_config.refresh.zip'
    if temporary.exists():
        raise FileExistsError(temporary)
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('configure_server.py', 'README.md', 'train.slurm'):
            path = ROOT / 'config/server' / name
            archive.writestr(f'config/server/{name}', path.read_text(encoding='utf-8').replace('\r\n', '\n').encode('utf-8'))
    record = finalize_archive(temporary)
    destination = OUTPUT / 'server_config.zip'
    temporary.replace(destination)
    record['file'] = destination.name
    manifest['archives'] = [record if item['file'] == destination.name else item for item in manifest['archives']]
    manifest['configuration_refresh_builder_sha256'] = digest(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (OUTPUT / 'SHA256SUMS.txt').write_bytes(''.join(f"{item['sha256']}  {item['file']}\n" for item in manifest['archives']).encode('utf-8'))
    shutil.copyfile(ROOT / 'config/server/README.md', OUTPUT / 'README_UPLOAD.md')
    print('Slurm/config bundle refreshed; dataset archives unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-config', action='store_true')
    arguments = parser.parse_args()
    if arguments.refresh_config:
        refresh_config_bundle()
    else:
        build()
