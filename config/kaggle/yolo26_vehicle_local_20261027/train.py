import argparse
import faulthandler
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor


WORK = Path('/kaggle/working/yolo26s_fresh_local_20261027')
ARCHIVE_SHA256 = 'd4fe27fbbf8d0b670c71a2b7500c8a277f00688d7eded24878373440ee1420fe'
COUNTS = {'train': 165233, 'val': 21460}
TASKS = ('vehicle', 'light')
TRAIN_BUDGET_SECONDS = 8 * 3600
SESSION_BUDGET_SECONDS = 10 * 3600


def emit(stage, **details):
    print(json.dumps({'stage': stage, **details}), flush=True)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_members(archive):
    members = archive.infolist()
    seen = set()
    for member in members:
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or '..' in path.parts or '\\' in member.filename
                or not path.parts or path.parts[0] != 'vehicle'
                or stat.S_ISLNK(member.external_attr >> 16)
                or member.filename in seen):
            raise ValueError('Unsafe or duplicate archive member')
        seen.add(member.filename)
    for split, expected in COUNTS.items():
        images = {PurePosixPath(member.filename).stem for member in members
                  if member.filename.startswith(f'vehicle/images/{split}/') and not member.is_dir()}
        labels = {PurePosixPath(member.filename).stem for member in members
                  if member.filename.startswith(f'vehicle/labels/{split}/') and not member.is_dir()}
        if len(images) != expected or images != labels:
            raise ValueError(f'Invalid image/label pairs for {split}')
    return members


def stage_data(source, destination):
    emit('VERIFY_ARCHIVE', source=str(source))
    if digest_file(source) != ARCHIVE_SHA256:
        raise ValueError('Archive SHA256 mismatch')
    with zipfile.ZipFile(source) as archive:
        members = checked_members(archive)
        required = sum(member.file_size for member in members) + 3 * 1024**3
        if shutil.disk_usage(destination.parent).free < required:
            raise RuntimeError('Insufficient local disk space')
        destination.mkdir(exist_ok=False)
        started = time.monotonic()
        for index, member in enumerate(members, 1):
            archive.extract(member, destination)
            if index % 10000 == 0:
                emit('EXTRACT_LOCAL', completed=index, total=len(members),
                     elapsed_seconds=round(time.monotonic() - started, 1))
    emit('LOCAL_DATA_READY', counts=COUNTS, root=str(destination / 'vehicle'))
    return destination / 'vehicle'


def stage_light(destination):
    source = Path('/kaggle/input/datasets/v5nhdf/yolo363-readable-crops-20261027/light_trainval/light')
    destination.mkdir(exist_ok=False)
    shutil.copyfile(source / 'data.yaml', destination / 'data.yaml')
    for split, expected in {'train': 10810, 'val': 1898}.items():
        images = sorted((source / 'images' / split).iterdir())
        labels = sorted((source / 'labels' / split).iterdir())
        if len(images) != expected or {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError('Unexpected light image/label pairs')
        files = images + labels
        for category in ('images', 'labels'):
            (destination / category / split).mkdir(parents=True)

        def copy(source_file):
            shutil.copyfile(source_file, destination / source_file.relative_to(source))

        with ThreadPoolExecutor(max_workers=8) as executor:
            for index, _ in enumerate(executor.map(copy, files), 1):
                if index % 2000 == 0:
                    emit('LIGHT_COPY_LOCAL', split=split, completed=index, total=len(files))
    emit('LIGHT_LOCAL_READY', root=str(destination))
    return destination


def worker(kind, resume_checkpoint=None, resume_best=None, resume_results=None, resume_start_epoch=11):
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    folder = WORK / kind
    if kind == 'vehicle':
        sources = list(Path('/kaggle/input').glob('**/vehicle_trainval.zip.bin'))
        if len(sources) != 1:
            raise RuntimeError('Expected one opaque vehicle archive, refusing mounted-file fallback')
        root = stage_data(sources[0], folder / 'dataset')
    else:
        root = stage_light(folder / 'dataset')
    import torch
    import yaml
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    settings = yaml.safe_load((root / 'data.yaml').read_text())
    settings['path'] = str(root)
    configuration = folder / 'data.yaml'
    configuration.write_text(yaml.safe_dump(settings, sort_keys=False))
    checkpoints = folder / 'checkpoints'
    checkpoints.mkdir()
    model = YOLO(str(resume_checkpoint or WORK / 'yolo26s.pt'))
    started = time.monotonic()

    def save_epoch(trainer):
        hashes = {}
        for name, source in [('last.pt', trainer.last), ('best.pt', trainer.best)]:
            source = Path(source)
            if source.is_file():
                temporary = checkpoints / (name + '.partial')
                shutil.copyfile(source, temporary)
                temporary.replace(checkpoints / name)
                hashes[name] = digest_file(checkpoints / name)
        status = {'task': kind, 'epoch_completed': trainer.epoch + 1, 'sha256': hashes,
                  'best_fitness': float(trainer.best_fitness), 'external_backup': False}
        temporary = folder / 'epoch_status.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(folder / 'epoch_status.json')
        emit('EPOCH_SAVED', **status)

    def training_started(trainer):
        if resume_checkpoint:
            if trainer.start_epoch != resume_start_epoch:
                raise RuntimeError(f'Resume must start at displayed epoch {resume_start_epoch + 1}')
            if resume_best and not Path(trainer.best).exists():
                shutil.copyfile(resume_best, trainer.best)
            if resume_results and not Path(trainer.csv).exists():
                shutil.copyfile(resume_results, trainer.csv)
        (folder / 'training_started.json').write_text(json.dumps({'task': kind, 'batch': trainer.batch_size}))
        emit('TRAINING_STARTED', task=kind, device=str(trainer.device), batch=trainer.batch_size,
             start_epoch=trainer.start_epoch, steps_per_epoch=len(trainer.train_loader))

    def stop_at_budget(trainer):
        if time.monotonic() - started >= TRAIN_BUDGET_SECONDS:
            trainer.stop = True

    model.add_callback('on_model_save', save_epoch)
    model.add_callback('on_train_start', training_started)
    model.add_callback('on_fit_epoch_end', stop_at_budget)
    emit('BUILD_DATALOADER', local_dataset=str(root))
    model.train(data=str(configuration), project=str(folder), name='train',
                resume=bool(resume_checkpoint), save_dir=str(folder / 'train'),
                epochs=100, patience=15, imgsz=640, batch=16, device=0,
                workers=2, amp=True, cache=False, save=True, save_period=-1,
                optimizer='auto', seed=363, deterministic=True, plots=False,
                hsv_h=0.0 if kind == 'light' else 0.015,
                hsv_s=0.2 if kind == 'light' else 0.7,
                hsv_v=0.3 if kind == 'light' else 0.4,
                mosaic=0.0 if kind == 'light' else 0.5, close_mosaic=10,
                exist_ok=False)
    faulthandler.cancel_dump_traceback_later()
    emit('COMPLETE')


def main():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.178'], check=True)
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Two GPUs required, refusing fallback')
    WORK.mkdir(exist_ok=False)
    model = YOLO('yolo26s.pt')
    shutil.copyfile(model.ckpt_path, WORK / 'yolo26s.pt')
    jobs = []
    started = time.monotonic()
    for kind in TASKS:
        gpu = {'vehicle': 0, 'light': 1}[kind]
        folder = WORK / kind
        folder.mkdir()
        log = (folder / 'train.log').open('w')
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        process = subprocess.Popen([sys.executable, '-u', __file__, '--task', kind],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((kind, process, log))
    while any(process.poll() is None for kind, process, log in jobs):
        progress = {}
        for kind, process, log in jobs:
            folder = WORK / kind
            if process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > SESSION_BUDGET_SECONDS or (elapsed > 45 * 60 and not (folder / 'training_started.json').exists()):
                    emit('WORKER_WATCHDOG_STOP', task=kind, elapsed_seconds=elapsed)
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            status_file = folder / 'epoch_status.json'
            status = json.loads(status_file.read_text()) if status_file.exists() else {'epoch_completed': 0}
            progress[kind] = {'return_code': process.poll(), **status}
            with (folder / 'train.log').open('rb') as handle:
                handle.seek(max(0, (folder / 'train.log').stat().st_size - 2400))
                print(f'[{kind} log tail]\n' + handle.read().decode(errors='replace'), flush=True)
        (WORK / 'progress.json').write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress), flush=True)
        time.sleep(30)
    failures = [kind for kind, process, log in jobs if process.returncode != 0]
    for kind, process, log in jobs:
        log.close()
        emit('WORKER_EXIT', task=kind, return_code=process.returncode)
        if process.returncode != 0:
            print((WORK / kind / 'train.log').read_text(errors='replace')[-12000:], flush=True)
    (WORK / 'completion.json').write_text(json.dumps({'failed_tasks': failures, 'success': not failures,
        'worker_exit_codes': {kind: process.returncode for kind, process, log in jobs}}))
    emit('SESSION_FINISHED', failed_tasks=failures,
         note='Normal script exit preserves outputs even when a worker failed; inspect completion.json')


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('vehicle', 'light'))
    args = parser.parse_args()
    worker(args.task) if args.task else main()
