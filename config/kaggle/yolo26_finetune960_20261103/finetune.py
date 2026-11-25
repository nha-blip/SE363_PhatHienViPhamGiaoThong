"""Continuation entry for Kaggle2: resume the vehicle run and fine-tune the light model at imgsz 960.

Runs on top of the in-memory runtime_base (yolo26_vehicle_local_20261027/train.py) and runtime_fullscene
(yolo26_fullscene_resume_20261028/train.py) modules, like autoresume.py. The runtime modules are imported lazily so
train_light can be smoke-tested locally with only Ultralytics installed.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

LIGHT_NAMES = ['red', 'yellow', 'green', 'off']


def train_light(data_yaml, weights, folder, epochs, budget_seconds, imgsz=960, batch=8, fraction=1.0, device=0, on_epoch=None):
    """Fine-tune the light checkpoint at a larger size with a fresh optimizer; weights only, no resume."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if [model.names[index] for index in sorted(model.names)] != LIGHT_NAMES:
        raise RuntimeError('Unexpected light class names')
    started = time.monotonic()

    def stop_at_budget(trainer):
        if time.monotonic() - started >= budget_seconds:
            trainer.stop = True

    model.add_callback('on_fit_epoch_end', stop_at_budget)
    if on_epoch:
        model.add_callback('on_model_save', on_epoch)
    return model.train(data=str(data_yaml), project=str(folder), name='train', save_dir=str(folder / 'train'), epochs=epochs, patience=10,
                       imgsz=imgsz, batch=batch, device=device, workers=2, amp=True, cache=False, save=True, save_period=-1,
                       optimizer='AdamW', lr0=0.0005, lrf=0.1, warmup_epochs=1, cos_lr=False, seed=363, deterministic=True, plots=False,
                       hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, mosaic=0.0, close_mosaic=0, fraction=fraction, exist_ok=False)


def settings(fullscene):
    return json.loads(fullscene.input_file('resume_state.json').read_text())


def configure():
    import runtime_base as base
    import runtime_fullscene as fullscene

    configuration = settings(fullscene)
    base.TASKS = tuple(configuration['tasks'])
    if not base.TASKS or any(kind not in {'vehicle', 'light'} for kind in base.TASKS):
        raise RuntimeError('Invalid tasks')
    base.TRAIN_BUDGET_SECONDS = configuration['train_budget_seconds']
    base.SESSION_BUDGET_SECONDS = configuration['session_budget_seconds']
    base.stage_light = fullscene.stage_light
    base.worker = worker
    return base, fullscene, configuration


def worker(kind):
    import faulthandler

    base, fullscene, configuration = configure()
    task = configuration['tasks'][kind]
    if kind == 'vehicle':
        files = {}
        for suffix in ('last.pt', 'best.pt', 'results.csv'):
            name = f'resume_vehicle_{suffix}'
            source = fullscene.input_file(name)
            fullscene.verify(source, task['sha256'][name])
            files[suffix] = source
        ORIGINAL_WORKER('vehicle', resume_checkpoint=files['last.pt'], resume_best=files['best.pt'],
                        resume_results=files['results.csv'], resume_start_epoch=task['completed_epochs'])
        return
    faulthandler.enable()
    folder = base.WORK / 'light'
    weights = fullscene.input_file('finetune_light_best.pt')
    fullscene.verify(weights, task['sha256']['finetune_light_best.pt'])
    root = fullscene.stage_light(folder / 'dataset')
    import yaml

    data = yaml.safe_load((root / 'data.yaml').read_text())
    data['path'] = str(root)
    configuration_file = folder / 'data.yaml'
    configuration_file.write_text(yaml.safe_dump(data, sort_keys=False))
    checkpoints = folder / 'checkpoints'
    checkpoints.mkdir()

    def save_epoch(trainer):
        hashes = {}
        for name, source in [('last.pt', trainer.last), ('best.pt', trainer.best)]:
            if Path(source).is_file():
                temporary = checkpoints / (name + '.partial')
                shutil.copyfile(source, temporary)
                temporary.replace(checkpoints / name)
                hashes[name] = base.digest_file(checkpoints / name)
        status = {'task': 'light', 'epoch_completed': trainer.epoch + 1, 'sha256': hashes, 'best_fitness': float(trainer.best_fitness)}
        (folder / 'epoch_status.tmp').write_text(json.dumps(status, indent=2))
        (folder / 'epoch_status.tmp').replace(folder / 'epoch_status.json')
        base.emit('EPOCH_SAVED', **status)

    (folder / 'training_started.json').write_text(json.dumps({'task': 'light', 'imgsz': task['imgsz'], 'batch': task['batch']}))
    base.emit('TRAINING_STARTED', task='light', imgsz=task['imgsz'], batch=task['batch'])
    train_light(configuration_file, weights, folder, task['epochs'], base.TRAIN_BUDGET_SECONDS, task['imgsz'], task['batch'], on_epoch=save_epoch)
    base.emit('COMPLETE')


ORIGINAL_WORKER = None

if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import runtime_base as _base

    ORIGINAL_WORKER = _base.worker
    base, fullscene, _ = configure()
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('vehicle', 'light'))
    args = parser.parse_args()
    if args.task:
        worker(args.task)
    else:
        base.main()
        for kind in base.TASKS:
            staging = (base.WORK / kind / 'dataset').resolve()
            if staging.parent != (base.WORK / kind).resolve() or not staging.is_relative_to(base.WORK.resolve()):
                raise RuntimeError('Unsafe staging cleanup')
            if staging.is_dir():
                shutil.rmtree(staging)
        base.emit('STAGING_REMOVED_OUTPUTS_KEPT')
