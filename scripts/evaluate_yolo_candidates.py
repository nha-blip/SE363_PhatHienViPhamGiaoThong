import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ['YOLO_AUTOINSTALL'] = 'false'

from scripts.evaluate_external_detectors import ROOT, evaluate, sha256, write_json


JOBS = (
    ('signal_yolo11s_state', 'signal_yolo11s', 'traffic_light', 'state'),
    ('joint_yolo11s_state', 'joint_yolo11s', 'traffic_light', 'state'),
    ('lisa_yolo26s_state', 'lisa_yolo26s', 'traffic_light', 'state'),
    ('lisa_yolo26m_state', 'lisa_yolo26m', 'traffic_light', 'state'),
    ('bdd_yolo11s_localization', 'bdd_yolo11s', 'traffic_light', 'localization'),
    ('yolox_light_state', 'yolox_light', 'traffic_light', 'state'),
    ('joint_yolo11s_vehicle', 'joint_yolo11s', 'vehicle', 'vehicle'),
    ('visdrone_yolo11s_vehicle', 'visdrone_yolo11s', 'vehicle', 'vehicle'),
    ('bdd_yolo11s_vehicle', 'bdd_yolo11s', 'vehicle', 'vehicle'),
    ('bdd_yolo26n_vehicle', 'bdd_yolo26n', 'vehicle', 'vehicle'),
)


def candidate_mapping(names, mode):
    mapping = {}
    excluded = {}
    vehicle = {'car': 0, 'motorcycle': 1, 'motor': 1, 'bus': 2, 'truck': 3}
    colors = {'red': 0, 'left-red': 0, 'traffic light red': 0, 'stop': 0, 'stopleft': 0,
              'yellow': 1, 'left-yellow': 1, 'traffic light yellow': 1, 'warning': 1, 'warningleft': 1,
              'green': 2, 'left-green': 2, 'traffic light green': 2, 'go': 2, 'goforward': 2, 'goleft': 2,
              'off': 3}
    colors.update({name: target for name, target in (
        ('red-circle', 0), ('red-arrow-left', 0), ('red-arrow-right', 0), ('red-arrow-straight', 0),
        ('amber-circle', 1), ('amber-arrow-left', 1), ('amber-arrow-right', 1), ('amber-arrow-straight', 1),
        ('green-circle', 2), ('green-arrow-left', 2), ('green-arrow-right', 2), ('green-arrow-straight', 2))})
    for identity, name in names.items():
        normalized = name.lower().strip()
        target = vehicle.get(normalized) if mode == 'vehicle' else colors.get(normalized) if mode == 'state' else (
            0 if normalized == 'traffic light' else None)
        if target is not None:
            mapping[int(identity)] = target
        elif normalized in ('unknown', 'tl_none') or 'red_yellow' in normalized or 'red-amber' in normalized:
            excluded[int(identity)] = name
    if not mapping:
        raise ValueError(f'Unsupported namespace: {mode}: {names}')
    return mapping, excluded


def write_summary(output, reports, failures, state):
    write_json(output / 'report.json', {'status': state, 'jobs': reports, 'failed_jobs': failures})
    lines = ['# Candidate evaluation', '',
             '| Job | Backend | Images | mAP50 all target classes | mAP50-95 | P95 ms | FPS model call | Supported target IDs |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for report in reports:
        metric, latency = report['metrics'], report['latency']
        lines.append(f"| {report['job']} | {report['runtime_backend']} | {report['images']} | {metric['metrics/mAP50(B)']:.4f} | "
                     f"{metric['metrics/mAP50-95(B)']:.4f} | {latency['p95_ms']:.2f} | "
                     f"{latency['fps_model_call']:.2f} | {report['supported_target_ids']} |")
    lines += ['', 'Status: ' + state,
              'Full four-color scores retain off GT even when the model cannot predict off. Supported-class AP is reported separately in JSON.',
              'Generic light localization is not color-state accuracy. No direction GT exists, so arrow direction accuracy is not evaluated.',
              'motor -> motorcycle is an explicit approved semantic alias; van/tricycle/others are not guessed into the four vehicle classes.',
              'Weights/source provenance and taxonomy are retained. Unknown/red-yellow are never forced into off/red/yellow.',
              'Batch 1, 640, CUDA FP16, conf=.001 for AP; conf=.25 for speed/confusion. Timing excludes tracking/VLM/render/encode.',
              'This is one benchmark pass, not proof of deployment realtime. Repeated timing is needed before ranking speed.',
              'YOLOX is measured separately on OpenCV DNN CPU FP32, four OpenCV threads. Its latency is NOT CUDA GPU latency; native 640 BGR 0-255, cubic resize/top-left 114 padding, raw stride decode and global NMS follow the author implementation.',
              'Existing test labels are used as-is; known label QA issues and possible upstream source overlap limit interpretation.',
              'predictions_025.jsonl caches native and mapped boxes above .25 for later visual QA without another inference pass.']
    if failures:
        lines += ['', '## Failed jobs'] + [f"- {entry['job']}: {entry['error']}" for entry in failures]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights/candidates_20261021')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--benchmark-images', type=int, default=200)
    parser.add_argument('--device', default='0')
    parser.add_argument('--jobs', nargs='*')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import ultralytics
    torch.set_num_threads(1)
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    write_json(args.output / 'environment.json', {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
               'gpu': torch.cuda.get_device_name(0) if args.device != 'cpu' else None,
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'limit': args.limit,
               'checkpoint_autoinstall': False})
    jobs = [job for job in JOBS if not args.jobs or job[0] in args.jobs]
    if not jobs or args.jobs and set(args.jobs) - {job[0] for job in JOBS}:
        raise ValueError('Unknown or empty job selection')
    reports, failures = [], []
    for identity, weight_id, dataset, mode in jobs:
        write_json(args.output / 'status.json', {'status': 'RUNNING', 'job': identity,
                   'completed_jobs': len(reports), 'failed_jobs': len(failures), 'total_jobs': len(jobs)})
        is_yolox = weight_id == 'yolox_light'
        weights = args.weights / (weight_id + ('.onnx' if is_yolox else '.pt'))
        try:
            provenance = json.loads((args.weights / (weight_id + '.provenance.json')).read_text())
            if provenance['status'] != 'DOWNLOADED' or sha256(weights) != provenance['sha256']:
                raise ValueError('Missing or changed checkpoint')
            cache_path = args.output / (identity + '.predictions_025.jsonl')
            with cache_path.open('w', encoding='utf-8') as handle:
                def cache(image_path, label_path, predictions, result):
                    native = result.boxes.data.detach().cpu().numpy()
                    value = {'image': str(image_path), 'label': str(label_path),
                             'predictions': predictions[predictions[:, 4] > .25].tolist(),
                             'native_predictions': native[native[:, 4] > .25].tolist()}
                    handle.write(json.dumps(value, separators=(',', ':')) + '\n')

                options = SimpleNamespace(output=args.output, limit=args.limit, device=args.device,
                                          benchmark_images=args.benchmark_images, contention=False,
                                          mapping_function=candidate_mapping, prediction_writer=cache,
                                          policy_overrides={'vehicle_mapping': 'Exact car/bus/truck; motor and motorcycle both mean motorcycle. No guessed van/tricycle/others.',
                                                            'light_mapping': 'Explicit color/LISA semantic map; off unsupported unless explicitly present.',
                                                            'oiv7_color_accuracy': 'NOT_APPLICABLE_TO_THIS_CANDIDATE'})
                if is_yolox:
                    from scripts.yolox_light_adapter import YoloxLight
                    options.model_factory = YoloxLight
                    options.device = 'cpu'
                report = evaluate((identity, str(weights.resolve()), dataset, mode), options)
            mapping = report['mapping']['mapped_source_ids']
            supported = sorted(set(mapping.values()))
            supported_rows = [row for row in report['per_class'] if row['class_id'] in supported and row['ap50'] is not None]
            report.update(supported_target_ids=supported,
                          runtime_backend='OpenCV_DNN_CPU_FP32' if is_yolox else 'PyTorch_CUDA_FP16' if args.device != 'cpu' else 'PyTorch_CPU_FP32',
                          unsupported_target_ids=sorted(set(row['class_id'] for row in report['per_class']) - set(supported)),
                          supported_class_map50=float(np.mean([row['ap50'] for row in supported_rows])) if supported_rows else None,
                          predictions_cache=str(cache_path), source_provenance=provenance)
            report['unsupported_predictions_at_conf_0001'] = report.pop('unsupported_red_yellow_predictions_at_conf_0001')
            write_json(args.output / identity / 'report.json', report)
            reports.append(report)
        except Exception as error:
            failures.append({'job': identity, 'error': f'{type(error).__name__}: {error}'})
            print(f'{identity}: FAILED: {error}', flush=True)
            write_json(args.output / identity / 'status.json', {'status': 'FAILED', 'error': str(error)})
            if args.device != 'cpu':
                torch.cuda.empty_cache()
        write_summary(args.output, reports, failures, 'RUNNING')
    state = 'COMPLETE' if not failures else 'COMPLETE_WITH_FAILURES'
    write_summary(args.output, reports, failures, state)
    write_json(args.output / 'status.json', {'status': state, 'completed_jobs': len(reports),
               'failed_jobs': len(failures), 'scope': 'SMOKE_SUBSET_NOT_FINAL' if args.limit else 'FULL_TEST_SPLITS'})


if __name__ == '__main__':
    main()
