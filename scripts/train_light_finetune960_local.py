import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["YOLO_AUTOINSTALL"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "config/kaggle/yolo26_finetune960_20261103"))
DATASET = ROOT / "data/Light363_train_work_20261103"
WEIGHTS = ROOT / "weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt"
EXPECTED_SHA256 = "4275b891940217966ab03ff7629d588063477eb60c03a57014a2941cce73c056"
OUTPUT = ROOT / "runs/project363/yolo26_light_finetune960_local_20261103"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the light YOLO26s A checkpoint at imgsz 960 on the local GPU")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--budget-hours", type=float, default=6.0)
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Use --execute; this trains on the local GPU")
    from src.tracking.io import sha256

    if sha256(WEIGHTS) != EXPECTED_SHA256:
        raise ValueError("Checkpoint hash mismatch")
    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    counts = {split: len(list((DATASET / "images" / split).iterdir())) for split in ("train", "val")}
    if counts != {"train": 8653, "val": 1898} or len((DATASET / "train_balanced.txt").read_text().splitlines()) != 10211:
        raise ValueError(f"Working copy differs from the recorded dataset: {counts}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    configuration = OUTPUT / "data.yaml"
    configuration.write_text(yaml.safe_dump({"path": str(DATASET), "train": "train_balanced.txt", "val": "images/val",
                                             "names": ["red", "yellow", "green", "off"]}, sort_keys=False))
    (OUTPUT / "run.json").write_text(json.dumps({"checkpoint_sha256": EXPECTED_SHA256, "imgsz": 960, "batch": arguments.batch, "epochs": arguments.epochs,
                                                 "test_used": False, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    import finetune

    finetune.train_light(configuration, WEIGHTS, OUTPUT, arguments.epochs, arguments.budget_hours * 3600, imgsz=960, batch=arguments.batch)


if __name__ == "__main__":
    main()
