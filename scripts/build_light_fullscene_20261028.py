import argparse
import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/TrafficLight363_expanded_candidate_20261026'
OUTPUT = ROOT / 'data/Light363_fullscene_20261028'
EXPORT = ROOT / 'exports/yolo26_fullscene_resume_20261028/upload'
NAMES = ['red', 'yellow', 'green', 'off']


def label_for(image):
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index('images')
    parts[index] = 'labels'
    return Path(*parts).with_suffix('.txt')


def rows_for(label):
    rows = []
    for line in label.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError('Malformed YOLO row')
        identity, center_x, center_y, width, height = values
        if (identity != int(identity) or not 0 <= identity < 4 or width <= 0 or height <= 0
                or center_x-width/2 < -1e-6 or center_y-height/2 < -1e-6
                or center_x+width/2 > 1+1e-6 or center_y+height/2 > 1+1e-6):
            raise ValueError('Invalid normalized annotation')
        rows.append((int(identity), center_x, center_y, width, height))
    return rows


def geometry(width, height):
    ratio = min(1.0, 1280/width, 720/height)
    target_width, target_height = round(width*ratio), round(height*ratio)
    return target_width, target_height, (1280-target_width)//2, (720-target_height)//2


def transform(rows, native_width, native_height):
    width, height, left, top = geometry(native_width, native_height)
    return [(identity, (center_x*width+left)/1280, (center_y*height+top)/720,
             box_width*width/1280, box_height*height/720)
            for identity, center_x, center_y, box_width, box_height in rows]


def source_paths(filename):
    paths = [Path(line.strip()) for line in (SOURCE / filename).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(paths) != len(set(paths)):
        raise ValueError('Unexpected repeated paths in unique source split')
    if not all(path.is_file() and label_for(path).is_file() for path in paths):
        raise FileNotFoundError('Source image/label payload missing')
    return paths


def build():
    train = source_paths('train_unique.txt')
    val = source_paths('val_combined.txt')
    test = source_paths('test_original.txt')
    if set(train) & (set(val) | set(test)):
        raise ValueError('Source split path leakage')
    OUTPUT.mkdir(exist_ok=False)
    for split in ('train', 'val'):
        for category in ('images', 'labels'):
            (OUTPUT / category / split).mkdir(parents=True)
    print('HASHING_HELDOUT', flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        heldout = set(executor.map(lambda path: hashlib.sha256(path.read_bytes()).hexdigest(), val+test))
    seen = {}
    counts = Counter()
    effective = Counter()
    manifest = []
    train_list = []
    rejected = []

    def build(image):
        source_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        rows = rows_for(label_for(image))
        if source_hash in heldout:
            return {'excluded': 'EXACT_HELDOUT_IMAGE', 'image': str(image)}
        identity = hashlib.sha256(image.as_posix().encode()).hexdigest()[:24]
        output_image = OUTPUT / 'images/train' / (identity + '.jpg')
        output_label = OUTPUT / 'labels/train' / (identity + '.txt')
        with Image.open(image) as original:
            original.load()
            native_width, native_height = original.size
            width, height, left, top = geometry(native_width, native_height)
            resized = original.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
            frame = Image.new('RGB', (1280, 720), (114, 114, 114))
            frame.paste(resized, (left, top))
            frame.save(output_image, quality=95, subsampling=0)
        new_rows = transform(rows, native_width, native_height)
        output_label.write_text(''.join(str(row[0])+' '+ ' '.join(f'{value:.10f}' for value in row[1:])+'\n'
                                       for row in new_rows))
        verified = rows_for(output_label)
        if len(verified) != len(rows):
            raise ValueError('Annotation count changed')
        classes = Counter(NAMES[row[0]] for row in rows)
        return {'id': identity, 'source_image': str(image), 'source_label': str(label_for(image)),
                'source_sha256': source_hash, 'source_label_sha256': hashlib.sha256(label_for(image).read_bytes()).hexdigest(),
                'native_size': [native_width, native_height], 'output_size': [1280, 720],
                'class_counts': dict(classes), 'repeat_count': 3 if classes['yellow'] or classes['off'] else 1}

    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, record in enumerate(executor.map(build, train), 1):
            if 'excluded' in record:
                rejected.append(record)
                continue
            prior = seen.get(record['source_sha256'])
            if prior and prior['source_label_sha256'] != record['source_label_sha256']:
                raise ValueError('Same source image has conflicting annotations; QA required')
            seen[record['source_sha256']] = record
            manifest.append(record)
            counts.update(record['class_counts'])
            effective.update({key: value*record['repeat_count'] for key, value in record['class_counts'].items()})
            train_list.extend(['./images/train/'+record['id']+'.jpg']*record['repeat_count'])
            if index % 500 == 0:
                print(json.dumps({'built_train': index, 'total_train': len(train)}), flush=True)
    validation_counts = Counter()
    for image in val:
        identity = hashlib.sha256(image.as_posix().encode()).hexdigest()[:24]
        shutil.copyfile(image, OUTPUT / 'images/val' / (identity+image.suffix.lower()))
        shutil.copyfile(label_for(image), OUTPUT / 'labels/val' / (identity+'.txt'))
        validation_counts.update(NAMES[row[0]] for row in rows_for(label_for(image)))
    (OUTPUT / 'train_balanced.txt').write_text('\n'.join(train_list)+'\n')
    (OUTPUT / 'data.yaml').write_text(yaml.safe_dump({'path': '.', 'train': 'train_balanced.txt',
                                                    'val': 'images/val', 'names': NAMES}, sort_keys=False))
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = {'status': 'FULLSCENE_TRAIN_BUILT', 'train_images': len(manifest), 'train_effective': len(train_list),
               'train_bbox': dict(counts), 'train_effective_bbox': dict(effective),
               'validation_images': len(val), 'validation_bbox': dict(validation_counts),
               'validation_policy': 'Byte-identical source images/labels; renamed only; no filtering or repetition',
               'train_policy': 'Full scene, downscale/pad 1280x720 without upscaling; retain every source annotation; no crops',
               'heldout_content_hash_check': True, 'excluded_train': rejected,
               'test_uploaded': False, 'source_modified': False}
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    package(summary)


def package(summary):
    EXPORT.mkdir(parents=True, exist_ok=True)
    archive_path = EXPORT / 'light_fullscene.zip.bin'
    if archive_path.exists() and zipfile.is_zipfile(archive_path):
        raise FileExistsError('Refusing overwrite of completed archive')
    print('PACKAGING_FULLSCENE', flush=True)
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(OUTPUT.rglob('*')):
            if path.is_file():
                archive.write(path, str(Path('light') / path.relative_to(OUTPUT)).replace('\\', '/'))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC failure')
    digest = hashlib.sha256()
    with archive_path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    package = {'file': archive_path.name, 'bytes': archive_path.stat().st_size,
               'sha256': digest.hexdigest(), 'crc_verified': True, **summary}
    (EXPORT.parent / 'light_package.json').write_text(json.dumps(package, indent=2))
    print(json.dumps(package), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-existing', action='store_true')
    args = parser.parse_args()
    if args.package_existing:
        summary = json.loads((OUTPUT / 'summary.json').read_text())
        manifest = json.loads((OUTPUT / 'manifest.json').read_text())
        if len(manifest) != summary['train_images'] or summary['status'] != 'FULLSCENE_TRAIN_BUILT':
            raise ValueError('Incomplete existing build')
        for split, expected in [('train', summary['train_images']), ('val', summary['validation_images'])]:
            images = list((OUTPUT / 'images' / split).iterdir())
            labels = list((OUTPUT / 'labels' / split).iterdir())
            if len(images) != expected or {path.stem for path in images} != {path.stem for path in labels}:
                raise ValueError('Incomplete existing pairs')
            for label in labels:
                rows_for(label)
        package(summary)
    else:
        build()
