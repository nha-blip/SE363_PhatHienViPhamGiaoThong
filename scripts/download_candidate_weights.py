import argparse
import concurrent.futures
import json
import urllib.request
from pathlib import Path

from scripts.evaluate_external_detectors import ROOT, sha256, write_json


CHECKPOINTS = (
    ('signal_yolo11s', 'https://raw.githubusercontent.com/UsamaMasood12/YOLO11s-Traffic-Signal-Detection/main/runs/detect/yolo11s_custom_dataset/weights/best.pt'),
    ('joint_yolo11s', 'https://raw.githubusercontent.com/mamounyosef/road-object-detection/main/training_results/best_model.pt'),
    ('visdrone_yolo11s', 'https://huggingface.co/dronefreak/visdrone-yolo11s/resolve/c1c9c8f559a7ebdc5271f8d5695b026dd22420cc/best.pt'),
    ('bdd_yolo11s', 'https://huggingface.co/dronefreak/bdd100k-yolo11s/resolve/1efdf83e6711df20fdd34e40976a1ac7a0d1f256/best.pt'),
    ('bdd_yolo26n', 'https://raw.githubusercontent.com/lochuan/yolo26n-bdd100k/main/weights/yolo26n-bdd100k-fp32.pt'),
    ('lisa_yolo26s', 'https://huggingface.co/dronefreak/lisa-yolo26s/resolve/b11bbd2eae7f30af61b807e74b390cf7b4ed7932/best.pt'),
    ('lisa_yolo26m', 'https://huggingface.co/dronefreak/lisa-yolo26m/resolve/main/best.pt'),
    ('yolox_light', 'https://raw.githubusercontent.com/jkk-research/autoware_tensorrt_yolox_traffic_light_classifier/master/model/tl_class.onnx'),
    ('kastel_yolov9', 'https://bwsyncandshare.kit.edu/s/Pc9jKBp6ncNbpf7/download/traffic_lights_yolov9.pt'),
    ('kastel_yolov7', 'https://bwsyncandshare.kit.edu/s/yBrYefxSMm2wkdo/download/traffic_lights_yolov7.pt'),
    ('joint_yolov5s', 'https://raw.githubusercontent.com/williamhyin/yolov5s_bdd100k/master/runs/exp0_yolov5s_bdd_prew/weights/best_yolov5s_bdd_prew.pt'),
)


def download(item, output):
    identity, url = item
    suffix = '.onnx' if identity == 'yolox_light' else '.pt'
    path = output / (identity + suffix)
    record = {'id': identity, 'url': url, 'path': str(path), 'executed': False}
    try:
        if path.exists():
            raise FileExistsError('Refusing to overwrite an existing checkpoint')
        request = urllib.request.Request(url, headers={'User-Agent': 'Project363-evaluation/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response, path.with_suffix(suffix + '.partial').open('wb') as handle:
            record['resolved_url'] = response.geturl()
            record['etag'] = response.headers.get('ETag')
            expected = response.headers.get('Content-Length')
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
        temporary = path.with_suffix(suffix + '.partial')
        if temporary.stat().st_size < 1024 * 1024:
            raise ValueError('Not a full checkpoint: possible HTML, pointer or error response')
        if expected is not None and temporary.stat().st_size != int(expected):
            raise ValueError('Content-Length mismatch')
        temporary.replace(path)
        record.update(status='DOWNLOADED', bytes=path.stat().st_size, sha256=sha256(path))
    except Exception as error:
        record.update(status='FAILED', error=str(error))
    print(f"{identity}: {record['status']}", flush=True)
    write_json(output / (identity + '.provenance.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'weights/candidates_20261021')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda item: download(item, args.output), CHECKPOINTS))
    write_json(args.output / 'manifest.json', {'checkpoints': records, 'no_models_executed': True})
    print(json.dumps({record['id']: record['status'] for record in records}, indent=2))


if __name__ == '__main__':
    main()
