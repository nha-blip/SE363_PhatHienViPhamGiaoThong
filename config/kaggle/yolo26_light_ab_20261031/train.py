import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath


WORK = Path('/kaggle/working/light_ab')
STAGE = Path('/tmp/light_ab')
NAMES = ['red', 'yellow', 'green', 'off']
EXPERIMENTS = {'A_fullscene': 0, 'B_fullscene_lisa': 1}
FULLSCENE_SHA256 = '30ba2e5fd64c6c0f2ab59bed1b860ddb9332af740d108bb76733764f91137952'
FULLSCENE_TRAIN_ENTRIES = 10211
VALIDATION_IMAGES = 1898
LISA_TRAIN_FRAMES = 643
LISA_HOLDOUT_FRAMES = 255
TRAIN_BUDGET_SECONDS = 8 * 3600
SESSION_BUDGET_SECONDS = 10 * 3600
START_TIMEOUT_SECONDS = 45 * 60
EPOCHS = 100
BATCH = 16
EVAL_BATCH = 32
DEVICE = 0


def emit(event, **values):
    print('LIGHT_AB ' + json.dumps({'event': event, **values}), flush=True)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def input_file(name):
    matches = list(Path('/kaggle/input').glob('**/' + name))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one input {name}, got {len(matches)}')
    return matches[0]


def checked_members(archive, root):
    members = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or '..' in path.parts or '\\' in member.filename or not path.parts or path.parts[0] != root
                or (member.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError('Unsafe archive member')
        if not member.is_dir():
            members.append(member)
    return members


def extract(source, root, destination, expected_sha256):
    if digest_file(source) != expected_sha256:
        raise ValueError('Archive SHA256 mismatch: ' + source.name)
    with zipfile.ZipFile(source) as archive:
        for member in checked_members(archive, root):
            archive.extract(member, destination)
    emit('STAGED', archive=source.name)
    return destination / root


def absolute_entries(lines, root):
    """Resolve the sampler lines of the full-scene dataset, keeping intentional repeats."""
    return [str(root / line.strip()[2:]) for line in lines if line.strip().startswith('./')]


def lisa_entries(images, labels):
    """Use the full-scene sampling policy: frames with yellow appear three times; repeats add no new samples."""
    entries = []
    for image in sorted(images):
        label = labels / (image.stem + '.txt')
        classes = {line.split()[0] for line in label.read_text().splitlines() if line.strip()}
        entries.extend([str(image)] * (3 if '1' in classes else 1))
    return entries


def prepare_data():
    import yaml

    package = json.loads(input_file('light_ab_package.json').read_text())
    if package['fullscene_sha256'] != FULLSCENE_SHA256:
        raise ValueError('Unexpected package')
    STAGE.mkdir(parents=True, exist_ok=False)
    light = extract(input_file('light_fullscene.zip.bin'), 'light', STAGE, FULLSCENE_SHA256)
    # The a_ prefix keeps a separate label cache: Ultralytics names it from the first sorted training file.
    lisa = extract(input_file('lisa_candidate.zip.bin'), 'a_lisa', STAGE, package['lisa_sha256'])
    for folder, expected in [(light / 'images/val', VALIDATION_IMAGES), (lisa / 'images/train', LISA_TRAIN_FRAMES),
                             (lisa / 'images/holdout', LISA_HOLDOUT_FRAMES)]:
        if len(list(folder.iterdir())) != expected:
            raise ValueError(f'Unexpected image count in {folder}')
    base = absolute_entries((light / 'train_balanced.txt').read_text().splitlines(), light)
    if len(base) != FULLSCENE_TRAIN_ENTRIES:
        raise ValueError('Unexpected full-scene sampler size')
    extra = lisa_entries(list((lisa / 'images/train').iterdir()), lisa / 'labels/train')
    entries = {'A_fullscene': base, 'B_fullscene_lisa': base + extra}
    WORK.mkdir(parents=True, exist_ok=True)
    for kind, rows in entries.items():
        listing = WORK / f'{kind}_train.txt'
        listing.write_text('\n'.join(rows) + '\n')
        (WORK / f'{kind}.yaml').write_text(yaml.safe_dump(
            {'path': str(STAGE), 'train': str(listing), 'val': str(light / 'images/val'), 'names': NAMES}, sort_keys=False))
    (WORK / 'lisa_holdout.yaml').write_text(yaml.safe_dump(
        {'path': str(STAGE), 'train': str(lisa / 'images/holdout'), 'val': str(lisa / 'images/holdout'), 'names': NAMES}, sort_keys=False))
    write_json(WORK / 'data_summary.json', {'entries': {kind: len(rows) for kind, rows in entries.items()},
                                            'lisa_entries': len(extra), 'test_used': False})
    emit('DATA_READY', entries={kind: len(rows) for kind, rows in entries.items()})


def per_class_report(metrics):
    box = metrics.box
    classes = {}
    for position, index in enumerate(box.ap_class_index):
        classes[NAMES[int(index)]] = {'precision': float(box.p[position]), 'recall': float(box.r[position]),
                                      'ap50': float(box.ap50[position]), 'ap50_95': float(box.ap[position]),
                                      'bbox': int(metrics.nt_per_class[int(index)])}
    return {'overall': {'precision': float(box.mp), 'recall': float(box.mr), 'map50': float(box.map50),
                        'map50_95': float(box.map)}, 'per_class': classes}


def evaluate(checkpoint, configuration, folder, name):
    from ultralytics import YOLO

    metrics = YOLO(str(checkpoint)).val(data=str(configuration), split='val', imgsz=640, batch=EVAL_BATCH, device=DEVICE, workers=2,
                                        plots=False, project=str(folder), name=name, exist_ok=True, verbose=False)
    return per_class_report(metrics)


def worker(kind):
    from ultralytics import YOLO

    folder = WORK / kind
    checkpoints = folder / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WORK / 'yolo26s.pt'))
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
        write_json(folder / 'epoch_status.json', {'task': kind, 'epoch_completed': trainer.epoch + 1, 'sha256': hashes,
                                                  'best_fitness': float(trainer.best_fitness)})

    def training_started(trainer):
        (folder / 'training_started.json').write_text(json.dumps({'task': kind, 'batch': trainer.batch_size}))
        emit('TRAINING_STARTED', task=kind, device=str(trainer.device), steps_per_epoch=len(trainer.train_loader))

    def stop_at_budget(trainer):
        if time.monotonic() - started >= TRAIN_BUDGET_SECONDS:
            trainer.stop = True

    model.add_callback('on_model_save', save_epoch)
    model.add_callback('on_train_start', training_started)
    model.add_callback('on_fit_epoch_end', stop_at_budget)
    model.train(data=str(WORK / f'{kind}.yaml'), project=str(folder), name='train', save_dir=str(folder / 'train'),
                epochs=EPOCHS, patience=15, imgsz=640, batch=BATCH, device=DEVICE, workers=2, amp=True, cache=False,
                save=True, save_period=-1, optimizer='auto', seed=363, deterministic=True, plots=False,
                hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, mosaic=0.0, close_mosaic=10, exist_ok=False)
    best = checkpoints / 'best.pt'
    report = {'task': kind, 'test_used': False, 'checkpoint_sha256': digest_file(best),
              'not_a_batch1_end_to_end_realtime_benchmark': True}
    report['validation'] = evaluate(best, WORK / f'{kind}.yaml', folder, 'eval_validation')
    report['lisa_holdout'] = evaluate(best, WORK / 'lisa_holdout.yaml', folder, 'eval_lisa_holdout')
    write_json(folder / 'report.json', report)
    emit('WORKER_DONE', task=kind, validation=report['validation']['overall'], lisa_holdout=report['lisa_holdout']['overall'])


def start_worker(kind, gpu):
    folder = WORK / kind
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / 'train.log').open('w')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    process = subprocess.Popen([sys.executable, '-u', __file__, '--task', kind], env=environment, stdout=log, stderr=subprocess.STDOUT)
    return process, log


def stop_process(process):
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.178'], check=True)
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Two GPUs required, refusing fallback')
    prepare_data()
    shutil.copyfile(YOLO('yolo26s.pt').ckpt_path, WORK / 'yolo26s.pt')
    write_json(WORK / 'pretrained.json', {'sha256': digest_file(WORK / 'yolo26s.pt')})
    started = time.monotonic()
    # Start in sequence so the two workers do not write the shared validation label cache at once.
    jobs = {}
    for kind, gpu in EXPERIMENTS.items():
        jobs[kind] = start_worker(kind, gpu)
        waited = time.monotonic()
        while (not (WORK / kind / 'training_started.json').exists() and jobs[kind][0].poll() is None
               and time.monotonic() - waited < START_TIMEOUT_SECONDS):
            time.sleep(15)
    while any(process.poll() is None for process, _ in jobs.values()):
        if time.monotonic() - started > SESSION_BUDGET_SECONDS:
            for kind, (process, _) in jobs.items():
                if process.poll() is None:
                    emit('WATCHDOG_STOP', task=kind)
                    stop_process(process)
        progress = {}
        for kind, (process, _) in jobs.items():
            status = WORK / kind / 'epoch_status.json'
            progress[kind] = {'return_code': process.poll(), **(json.loads(status.read_text()) if status.exists() else {'epoch_completed': 0})}
        write_json(WORK / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
        time.sleep(60)
    failures = [kind for kind, (process, _) in jobs.items() if process.returncode != 0]
    for kind, (process, log) in jobs.items():
        log.close()
        emit('WORKER_EXIT', task=kind, return_code=process.returncode)
        if process.returncode != 0:
            print((WORK / kind / 'train.log').read_text(errors='replace')[-8000:], flush=True)
    write_json(WORK / 'completion.json', {'failed_tasks': failures, 'success': not failures,
                                          'worker_exit_codes': {kind: process.returncode for kind, (process, _) in jobs.items()}})
    shutil.rmtree(STAGE)
    emit('SESSION_FINISHED', failed_tasks=failures)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=tuple(EXPERIMENTS))
    args = parser.parse_args()
    worker(args.task) if args.task else main()
