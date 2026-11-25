import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import runtime_base as base


base.WORK = Path('/kaggle/working/yolo26_fullscene_resume_20261028')
original_worker = base.worker


def input_file(name):
    matches = list(Path('/kaggle/input').glob('**/' + name))
    if len(matches) != 1:
        raise RuntimeError('Expected one input file: ' + name)
    return matches[0]


def package():
    return json.loads(input_file('training_package.json').read_text())


def verify(path, expected):
    if base.digest_file(path) != expected:
        raise RuntimeError('Input SHA256 mismatch: ' + path.name)


def stage_light(destination):
    settings = package()['light']
    source = input_file('light_fullscene.zip.bin')
    verify(source, settings['sha256'])
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if shutil.disk_usage(destination).free < sum(member.file_size for member in members) + 2*1024**3:
            raise RuntimeError('Insufficient local disk for light')
        seen = set()
        for member in members:
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.filename
                    or not path.parts or path.parts[0] != 'light'
                    or member.filename in seen or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError('Unsafe light archive')
            seen.add(member.filename)
        for index, member in enumerate(members, 1):
            archive.extract(member, destination)
            if index % 2000 == 0:
                base.emit('LIGHT_EXTRACT_FULLSCENE', completed=index, total=len(members))
    root = destination / 'light'
    for split, count in [('train', settings['train_images']), ('val', settings['validation_images'])]:
        images = list((root / 'images' / split).iterdir())
        labels = list((root / 'labels' / split).iterdir())
        if len(images) != count or {path.stem for path in images} != {path.stem for path in labels}:
            raise RuntimeError('Invalid fullscene light pairs')
    base.emit('LIGHT_FULLSCENE_READY', train_images=settings['train_images'],
              train_effective=settings['train_effective'], validation_images=settings['validation_images'])
    return root


def worker(kind):
    if kind == 'vehicle':
        settings = package()['vehicle_resume']
        files = {}
        for name in ('vehicle_last.pt', 'vehicle_best.pt', 'vehicle_results.csv'):
            source = input_file(name)
            verify(source, settings['sha256'][name])
            files[name] = source
        original_worker(kind, resume_checkpoint=files['vehicle_last.pt'],
                        resume_best=files['vehicle_best.pt'], resume_results=files['vehicle_results.csv'])
    else:
        original_worker(kind)


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    base.stage_light = stage_light
    base.worker = worker
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('vehicle', 'light'))
    args = parser.parse_args()
    if args.task:
        worker(args.task)
    else:
        base.main()
        for kind in ('vehicle', 'light'):
            staging = (base.WORK / kind / 'dataset').resolve()
            expected_parent = (base.WORK / kind).resolve()
            if staging.parent != expected_parent or not staging.is_relative_to(base.WORK.resolve()):
                raise RuntimeError('Unsafe staging cleanup path')
            if staging.is_dir():
                shutil.rmtree(staging)
        base.emit('STAGING_REMOVED_OUTPUTS_KEPT')
