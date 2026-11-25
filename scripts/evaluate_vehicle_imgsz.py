import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256

os.environ["YOLO_AUTOINSTALL"] = "false"


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/Vehicle363_val_from_kaggle_archive_20261102"
WEIGHTS = ROOT / "weights/yolo26_vehicle_restart03_20261031/yolo26_fullscene_resume_20261028/vehicle/checkpoints/best.pt"
EXPECTED_SHA256 = "2cefe325048ba9338da75b76d893b7f8e0ba1e5aba6f2db2ac392aac8af50e67"
OUTPUT = ROOT / "runs/project363/yolo_vehicle_imgsz_sweep_20261102"
NAMES = ["car", "motorcycle", "bus", "truck"]


def main():
    parser = argparse.ArgumentParser(description="Validate the epoch-41 vehicle checkpoint at several inference sizes; no retraining")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[960, 1280])
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Use --execute; this runs the detector on the local GPU")
    if sha256(WEIGHTS) != EXPECTED_SHA256:
        raise ValueError("Checkpoint hash mismatch")
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(WEIGHTS))
    results = {"checkpoint_sha256": EXPECTED_SHA256, "test_used": False, "trained_imgsz": 640,
               "note": "Same 21460-image validation split as the Kaggle run; weights trained at 640, inference size changed only", "sizes": {}}
    for size in arguments.sizes:
        metrics = model.val(data=str(DATASET / "data.yaml"), split="val", imgsz=size, batch=8, conf=0.001, iou=0.7, device=0, workers=2, half=True,
                            plots=False, project=str(OUTPUT), name=f"imgsz_{size}", exist_ok=False, verbose=False)
        results["sizes"][str(size)] = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "map50": float(metrics.box.map50),
                                       "map50_95": float(metrics.box.map), "map50_95_per_class": dict(zip(NAMES, map(float, metrics.box.maps)))}
        (OUTPUT / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({size: results["sizes"][str(size)]}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
