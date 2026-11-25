import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_tracking_mot as metrics

MODELS = {"e41": ROOT / "weights/yolo26_vehicle_restart03_20261031/yolo26_fullscene_resume_20261028/vehicle/checkpoints/best.pt",
          "e63": ROOT / "runs/project363/yolo26_vehicle_local_resume_20261104/final_best_copy.pt",
          "v8s": ROOT / "weights/kaggle_small_vehicle_20261019/best.pt"}


def main():
    parser = argparse.ArgumentParser(description="Detector-level recall of several vehicle checkpoints on one MOT sequence with a GT file")
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True, help="MOT gt txt: frame,id,x,y,w,h,...")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--stride", type=int, default=1)
    arguments = parser.parse_args()
    from ultralytics import YOLO

    gt = collections.defaultdict(list)
    for line in arguments.gt.read_text().splitlines():
        parts = line.split(",")
        gt[int(parts[0])].append(list(map(float, parts[2:6])))
    frames = sorted(gt)[::arguments.stride]
    for name, path in MODELS.items():
        model = YOLO(str(path))
        hits = {0.1: 0, 0.25: 0, 0.4: 0}
        total = 0
        boxes_per = {0.25: 0}
        for frame in frames:
            result = model.predict(str(arguments.sequence / "img1" / f"{frame:06d}.jpg"), imgsz=arguments.imgsz, conf=0.1, iou=0.7, device=0, verbose=False)[0]
            xyxy, conf = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
            predicted = np.column_stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]]) if len(xyxy) else np.zeros((0, 4))
            sim = metrics.iou_matrix(gt[frame], predicted)
            total += len(gt[frame])
            for threshold in hits:
                keep = conf >= threshold
                if keep.any():
                    hits[threshold] += int((sim[:, keep].max(1) >= 0.5).sum())
            boxes_per[0.25] += int((conf >= 0.25).sum())
        print(name, f"frames={len(frames)} gt={total}", {t: f"{h / total * 100:.1f}%" for t, h in hits.items()}, "dets@0.25", boxes_per[0.25], flush=True)


if __name__ == "__main__":
    main()
