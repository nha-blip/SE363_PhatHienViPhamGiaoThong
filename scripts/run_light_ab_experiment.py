import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/project363/light_ab_20261026'
CONFIG = ROOT / 'config/light363_expansion_20261026'
CHECKPOINT = ROOT / 'weights/trained_yolo_20261018/small/traffic_light/best.pt'
NAMES = ['red', 'yellow', 'green', 'off']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def blocker_alive(pid, creation):
    try:
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - creation) < 0.01
    except psutil.NoSuchProcess:
        return False


def gpu_state():
    command = ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']
    response = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    pids = [int(line.strip()) for line in response.stdout.splitlines() if line.strip().isdigit()]
    response = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True, timeout=30)
    free = int(response.stdout.strip().splitlines()[0])
    return pids, free


def select_winner(metrics):
    return max(sorted(metrics), key=lambda name: (metrics[name]['combined']['map5095'], name == 'baseline'))


def prepare(blocker_pid):
    from ultralytics import YOLO, __version__

    model = YOLO(str(CHECKPOINT))
    names = [name for identity, name in sorted(model.names.items())]
    if names != NAMES or not 10_000_000 <= sum(value.numel() for value in model.model.parameters()) <= 13_000_000:
        raise ValueError('Checkpoint is not the expected small four-state detector')
    creation = psutil.Process(blocker_pid).create_time()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for folder in ('configs', 'logs', 'requests', 'metrics', 'models'):
        (OUTPUT / folder).mkdir()
    original_args = model.ckpt.get('train_args', {})
    common = {'task': 'detect', 'mode': 'train', 'model': str(CHECKPOINT), 'device': 0,
              'epochs': 30, 'patience': 10, 'imgsz': 640, 'batch': 4, 'workers': 2,
              'seed': 363, 'deterministic': True, 'optimizer': 'AdamW', 'lr0': 0.001,
              'lrf': 0.01, 'weight_decay': 0.0005, 'warmup_epochs': 3.0,
              'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.2, 'mosaic': 1.0,
              'close_mosaic': 10, 'pretrained': True, 'amp': True, 'fraction': 1.0,
              'cache': False, 'save': True, 'save_period': -1, 'val': True,
              'plots': True, 'verbose': False, 'resume': False, 'exist_ok': False,
              'project': str(OUTPUT / 'models')}
    inputs = [CHECKPOINT]
    for path in CONFIG.glob('*.yaml'):
        inputs.append(path)
    data_root = ROOT / 'data/TrafficLight363_expanded_candidate_20261026'
    inputs.extend(data_root.glob('*.txt'))
    for name, data in (('A_unique', 'data_unique.yaml'), ('B_rare3', 'data.yaml')):
        settings = {**common, 'name': name, 'data': str(CONFIG / data)}
        path = OUTPUT / 'configs' / (name + '.yaml')
        path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
        inputs.append(path)
    write_json(OUTPUT / 'plan.json', {
        'status': 'PREPARED', 'created_at_utc': timestamp(), 'blocker_pid': blocker_pid,
        'blocker_creation': creation, 'checkpoint': str(CHECKPOINT),
        'input_sha256': {str(path): digest(path) for path in inputs},
        'ultralytics_version': __version__, 'original_checkpoint_train_args': original_args,
        'experiments': ['A_unique', 'B_rare3'], 'epochs_per_arm': 30,
        'selection': 'Highest combined-validation mAP50-95 among baseline/A/B; test ONLY selected model',
        'gpu_policy': 'Do not signal or stop other jobs; wait for blocker exit, no compute PIDs, >=2500 MiB free, twice',
        'resume_policy': 'Fine-tune same checkpoint with fresh optimizer, not resume old Kaggle run'})
    write_json(OUTPUT / 'status.json', {'status': 'QUEUED', 'checked_at_utc': timestamp(), 'blocker_pid': blocker_pid})
    print('Prepared A/B fine-tuning queue; existing GPU job untouched', flush=True)


def wait_gpu(plan, step):
    idle_checks = 0
    while idle_checks < 2:
        try:
            active = blocker_alive(plan['blocker_pid'], plan['blocker_creation'])
            pids, free = gpu_state()
            ready = not active and not pids and free >= 2500
            idle_checks = idle_checks + 1 if ready else 0
            state = {'status': 'WAITING_GPU', 'next_step': step, 'blocker_alive': active,
                     'active_compute_pids': pids, 'free_mib': free, 'idle_checks': idle_checks}
        except Exception as error:
            idle_checks = 0
            state = {'status': 'WAITING_GPU_CHECK_FAILED', 'next_step': step, 'error': str(error)}
        write_json(OUTPUT / 'status.json', {**state, 'checked_at_utc': timestamp(), 'controller_pid': os.getpid()})
        if idle_checks < 2:
            time.sleep(30)


def execute(plan, name, request):
    wait_gpu(plan, name)
    for path, checksum in plan['input_sha256'].items():
        if digest(path) != checksum:
            raise ValueError('Input changed while queued: ' + path)
    request_path = OUTPUT / 'requests' / (name + '.json')
    write_json(request_path, request)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8', 'YOLO_VERBOSE': 'false',
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    with (OUTPUT / 'logs' / (name + '.log')).open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--request', str(request_path)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            write_json(OUTPUT / 'status.json', {'status': 'RUNNING', 'step': name, 'worker_pid': process.pid,
                       'controller_pid': os.getpid(), 'checked_at_utc': timestamp(), 'log': str(log.name)})
            try:
                return_code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                continue
    if return_code:
        raise RuntimeError(f'{name} failed with exit code {return_code}; see {log.name}')


def metric_record(result):
    box = result.box
    classes = {}
    for index, identity in enumerate(box.ap_class_index):
        classes[NAMES[int(identity)]] = {'ap50': float(box.ap50[index]), 'ap5095': float(box.ap[index]),
                                       'precision': float(box.p[index]), 'recall': float(box.r[index])}
    return {'map50': float(box.map50), 'map5095': float(box.map), 'per_class': classes,
            'recall_definition': 'Ultralytics validation operating point, not fixed conf=0.25', 'speed_ms': result.speed}


def worker(request):
    from ultralytics import YOLO

    if request['kind'] == 'train':
        settings = yaml.safe_load(Path(request['config']).read_text(encoding='utf-8'))
        directory = Path(settings['project']) / settings['name']
        if directory.exists():
            raise FileExistsError('Refusing to reuse experiment output: ' + str(directory))
        model = YOLO(settings['model'])

        def progress(trainer):
            write_json(OUTPUT / ('progress_' + settings['name'] + '.json'), {
                'epoch_completed': trainer.epoch + 1, 'max_epochs': trainer.epochs,
                'metrics': {str(key): float(value) for key, value in trainer.metrics.items()},
                'checked_at_utc': timestamp(), 'last_checkpoint': str(trainer.last), 'best_checkpoint': str(trainer.best)})

        model.add_callback('on_fit_epoch_end', progress)
        model.train(**settings)
        if not (directory / 'weights/best.pt').is_file():
            raise FileNotFoundError('Best checkpoint missing')
    elif request['kind'] == 'evaluate':
        model = YOLO(request['model'])
        metrics = {}
        for split, filename in (('combined', 'data.yaml'), ('original', 'data_original_val.yaml'), ('new_source', 'data_new_holdout.yaml')):
            result = model.val(data=str(CONFIG / filename), split='val', imgsz=640, batch=4, device=0,
                               half=True, workers=2, plots=False, verbose=False,
                               project=str(OUTPUT / 'validation'), name=request['name'] + '_' + split, exist_ok=False)
            metrics[split] = metric_record(result)
        write_json(request['output'], metrics)
    elif request['kind'] == 'test':
        result = YOLO(request['model']).val(data=str(CONFIG / 'data.yaml'), split='test', imgsz=640, batch=4,
                                           device=0, half=True, workers=2, plots=True, verbose=False,
                                           project=str(OUTPUT / 'validation'), name='selected_original_test', exist_ok=False)
        write_json(request['output'], metric_record(result))
    elif request['kind'] == 'benchmark':
        import cv2
        import numpy as np
        import torch

        images = (ROOT / 'data/TrafficLight363_expanded_candidate_20261026/val_original.txt').read_text().splitlines()[:50]
        frames = [cv2.imread(path) for path in images]
        if any(frame is None for frame in frames):
            raise ValueError('Benchmark frame unreadable')
        measurements = {}
        for name, checkpoint in request['models'].items():
            model = YOLO(checkpoint)
            arguments = {'imgsz': 640, 'device': 0, 'half': True, 'conf': 0.25, 'iou': 0.7,
                         'max_det': 300, 'verbose': False, 'save': False}
            for index in range(10):
                model.predict(frames[index % len(frames)], **arguments)
            durations = []
            for index in range(100):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model.predict(frames[index % len(frames)], **arguments)
                torch.cuda.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
            measurements[name] = {'fps_predict_call': 1000 / float(np.mean(durations)),
                                  'median_ms': float(np.median(durations)), 'p95_ms': float(np.percentile(durations, 95)),
                                  'scope': 'Batch1 FP16 640, CPU-preloaded frames, predict call incl preprocessing/NMS; not tracking/video pipeline'}
            del model
            torch.cuda.empty_cache()
        write_json(request['output'], measurements)
    else:
        raise ValueError('Unknown worker task')


def run():
    plan = read_json(OUTPUT / 'plan.json')
    metrics = {}
    checkpoints = {'baseline': plan['checkpoint']}
    for name in ('baseline', *plan['experiments']):
        if name != 'baseline':
            execute(plan, 'train_' + name, {'kind': 'train', 'config': str(OUTPUT / 'configs' / (name + '.yaml'))})
            checkpoints[name] = str(OUTPUT / 'models' / name / 'weights/best.pt')
        path = OUTPUT / 'metrics' / (name + '.json')
        execute(plan, 'validate_' + name, {'kind': 'evaluate', 'name': name, 'model': checkpoints[name], 'output': str(path)})
        metrics[name] = read_json(path)
    winner = select_winner(metrics)
    write_json(OUTPUT / 'validation_comparison.json', {'models': metrics, 'selected': winner,
               'criterion': 'Highest combined validation mAP50-95; baseline included; test not used for selection'})
    execute(plan, 'selected_test', {'kind': 'test', 'model': checkpoints[winner], 'output': str(OUTPUT / 'metrics/selected_test.json')})
    execute(plan, 'benchmark', {'kind': 'benchmark', 'models': {'baseline': checkpoints['baseline'], 'selected': checkpoints[winner]},
                              'output': str(OUTPUT / 'metrics/benchmark.json')})
    lines = ['# Light small A/B results', '', '| Model | Combined AP50 | Combined AP50-95 | Original AP50-95 | New AP50-95 |', '|---|---:|---:|---:|---:|']
    for name, values in metrics.items():
        lines.append(f'| {name} | {values["combined"]["map50"]:.4f} | {values["combined"]["map5095"]:.4f} | {values["original"]["map5095"]:.4f} | {values["new_source"]["map5095"]:.4f} |')
    lines.extend(['', 'Selected on validation: ' + winner, 'Only selected checkpoint evaluated on original test. See metrics/ for per-class scores and benchmark scope.'])
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(OUTPUT / 'status.json', {'status': 'COMPLETE', 'selected': winner, 'selected_checkpoint': checkpoints[winner], 'completed_at_utc': timestamp()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--blocker-pid', type=int)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.blocker_pid)
    elif args.mode == 'worker':
        worker(read_json(args.request))
    else:
        if read_json(OUTPUT / 'status.json').get('status') == 'COMPLETE':
            raise RuntimeError('Experiment already complete; refusing to rerun')
        lock = OUTPUT / 'controller.lock'
        with lock.open('x', encoding='utf-8') as handle:
            json.dump({'pid': os.getpid(), 'created_at_utc': timestamp()}, handle)
        try:
            run()
        except Exception as error:
            write_json(OUTPUT / 'status.json', {'status': 'FAILED', 'error': str(error), 'checked_at_utc': timestamp()})
            raise
        finally:
            lock.unlink()


if __name__ == '__main__':
    main()
