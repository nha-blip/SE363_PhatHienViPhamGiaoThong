import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORK = Path('/kaggle/working/yolo26s_20261027')
DATASET = 'yolo363-readable-crops-20261027'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def configure():
    import yaml

    roots = list(Path('/kaggle/input').rglob('PACKAGING.json'))
    for kind in ('vehicle', 'light'):
        matches = [path.parent for path in roots if path.parent.name == kind and DATASET in str(path)]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one extracted dataset folder for {kind}, got {len(matches)}')
        root = matches[0]
        settings = yaml.safe_load((root / 'data.yaml').read_text())
        settings['path'] = str(root)
        for split in ('train', 'val'):
            if not (root / settings[split]).is_dir():
                raise FileNotFoundError(root / settings[split])
        (WORK / f'{kind}_data.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')


def worker(kind):
    from ultralytics import YOLO

    checkpoints = WORK / kind / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def save_epoch(trainer):
        files = {}
        for name, source in (('last.pt', trainer.last), ('best.pt', trainer.best)):
            source = Path(source)
            if source.is_file():
                temporary = checkpoints / (name + '.partial')
                shutil.copyfile(source, temporary)
                temporary.replace(checkpoints / name)
                files[name] = hashlib.sha256((checkpoints / name).read_bytes()).hexdigest()
        write_json(WORK / kind / 'epoch_status.json', {'epoch_completed': trainer.epoch+1,
                   'best_fitness': float(trainer.best_fitness), 'sha256': files,
                   'checkpoint_directory': str(checkpoints), 'remote_drive_backup': False})

    model = YOLO(str(WORK / 'yolo26s.pt'))
    model.add_callback('on_model_save', save_epoch)

    def training_started(trainer):
        print('YOLO26_WORKER_TRAINING_STARTED ' + json.dumps({'task': kind, 'device': str(trainer.device),
              'batch': trainer.batch_size, 'steps_per_epoch': len(trainer.train_loader)}), flush=True)

    model.add_callback('on_train_start', training_started)

    def training_started(trainer):
        print('YOLO26_WORKER_TRAINING_STARTED ' + json.dumps({'task': kind, 'device': str(trainer.device),
              'batch': trainer.batch_size, 'steps_per_epoch': len(trainer.train_loader)}), flush=True)

    model.add_callback('on_train_start', training_started)

    def stop_at_budget(trainer):
        if time.monotonic() - started >= 10*3600:
            trainer.stop = True

    model.add_callback('on_fit_epoch_end', stop_at_budget)
    model.train(data=str(WORK / f'{kind}_data.yaml'), project=str(WORK / kind), name='train',
                epochs=100, patience=15, imgsz=640, batch=16, device=0,
                workers=2, amp=True, cache=False, save=True, save_period=-1,
                optimizer='auto', seed=363, deterministic=True, plots=False,
                hsv_h=0.0 if kind == 'light' else 0.015,
                hsv_s=0.2 if kind == 'light' else 0.7, hsv_v=0.3 if kind == 'light' else 0.4,
                mosaic=0.0 if kind == 'light' else 0.5, close_mosaic=10,
                exist_ok=False)


def launch():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.178'], check=True)
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Requires two GPUs; refuse CPU fallback or sharing one GPU')
    WORK.mkdir(parents=True, exist_ok=False)
    print('GPUs:', [torch.cuda.get_device_name(index) for index in range(2)], flush=True)
    configure()
    model = YOLO('yolo26s.pt')
    shutil.copyfile(model.ckpt_path, WORK / 'yolo26s.pt')
    jobs = []
    for gpu, kind in enumerate(('vehicle', 'light')):
        folder = WORK / kind
        folder.mkdir()
        log = (folder / 'train.log').open('w', encoding='utf-8')
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        process = subprocess.Popen([sys.executable, '-u', __file__, '--task', kind],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((kind, process, log))
    while any(process.poll() is None for kind, process, log in jobs):
        state = {}
        for kind, process, log in jobs:
            path = WORK / kind / 'epoch_status.json'
            progress = json.loads(path.read_text()) if path.exists() else {'epoch_completed': 0}
            state[kind] = {'return_code': process.poll(), **progress}
            log_path = WORK / kind / 'train.log'
            if log_path.is_file():
                with log_path.open('rb') as handle:
                    handle.seek(max(0, log_path.stat().st_size-2200))
                    tail = handle.read().decode('utf-8', errors='replace')
                print(f'[{kind} log tail]\n{tail}', flush=True)
            log_path = WORK / kind / 'train.log'
            if log_path.is_file():
                with log_path.open('rb') as handle:
                    handle.seek(max(0, log_path.stat().st_size-2200))
                    tail = handle.read().decode('utf-8', errors='replace')
                print(f'[{kind} log tail]\n{tail}', flush=True)
        write_json(WORK / 'progress.json', state)
        print(json.dumps(state), flush=True)
        time.sleep(30)
    failures = []
    for kind, process, log in jobs:
        log.close()
        print(kind, 'exit:', process.returncode, flush=True)
        if process.returncode:
            failures.append(kind)
            print((WORK / kind / 'train.log').read_text(errors='replace')[-6000:], flush=True)
    write_json(WORK / 'completion.json', {'failed_tasks': failures, 'success': not failures})
    if failures:
        raise RuntimeError('Training failed: ' + ','.join(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('vehicle', 'light'))
    arguments = parser.parse_args()
    worker(arguments.task) if arguments.task else launch()
