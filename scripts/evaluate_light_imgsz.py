import argparse
import json
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256

os.environ["YOLO_AUTOINSTALL"] = "false"


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/Light363_fullscene_20261028"
DATASET = ROOT / "data/Light363_val_copy_20261102"
WEIGHTS = ROOT / "weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt"
EXPECTED_SHA256 = "4275b891940217966ab03ff7629d588063477eb60c03a57014a2941cce73c056"
OUTPUT = ROOT / "runs/project363/yolo_light_imgsz_sweep_20261102"
NAMES = ["red", "yellow", "green", "off"]
EXPECTED_VAL = {"images": 1898, "boxes": 4501}


def copy_validation():
    """Copy the validation split so Ultralytics writes its label cache outside the source dataset."""
    for kind in ("images", "labels"):
        shutil.copytree(SOURCE / kind / "val", DATASET / kind / "val")
    labels = list((DATASET / "labels/val").glob("*.txt"))
    boxes = sum(len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]) for path in labels)
    images = len(list((DATASET / "images/val").iterdir()))
    if {"images": images, "boxes": boxes} != EXPECTED_VAL:
        raise ValueError(f"Validation copy differs from the recorded split: {images} images, {boxes} boxes")
    (DATASET / "data.yaml").write_text(f"path: {DATASET.as_posix()}\ntrain: images/val\nval: images/val\nnames:\n- red\n- yellow\n- green\n- 'off'\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Validate the light A checkpoint at several inference sizes; no retraining")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1280])
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
    if not DATASET.exists():
        copy_validation()
    model = YOLO(str(WEIGHTS))
    results = {"checkpoint_sha256": EXPECTED_SHA256, "test_used": False, "trained_imgsz": 640, "validation": EXPECTED_VAL, "sizes": {}}
    for size in arguments.sizes:
        metrics = model.val(data=str(DATASET / "data.yaml"), split="val", imgsz=size, batch=8, conf=0.001, iou=0.7, device=0, workers=2, half=True,
                            plots=False, project=str(OUTPUT), name=f"imgsz_{size}", exist_ok=False, verbose=False)
        results["sizes"][str(size)] = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "map50": float(metrics.box.map50),
                                       "map50_95": float(metrics.box.map), "map50_95_per_class": dict(zip(NAMES, map(float, metrics.box.maps))),
                                       "recall_per_class": dict(zip(NAMES, map(float, metrics.box.r))), "precision_per_class": dict(zip(NAMES, map(float, metrics.box.p)))}
        (OUTPUT / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({size: results["sizes"][str(size)]}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
