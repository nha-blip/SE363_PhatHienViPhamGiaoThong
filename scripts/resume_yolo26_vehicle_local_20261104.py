import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "exports/yolo26_finetune960_20261103/upload/resume_vehicle_last.pt"
EXPECTED_SHA256 = "53a4a8a6"
DATASET = ROOT / "data/YOLO363_training_working/Vehicle363-v1"
OUTPUT = ROOT / "runs/project363/yolo26_vehicle_local_resume_20261104"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Resume the YOLO26s vehicle run from epoch 42 on the local GPU")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if not arguments.execute:
        raise PermissionError("Use --execute; this trains on the local GPU")
    if not sha256(SOURCE).startswith(EXPECTED_SHA256):
        raise ValueError("Checkpoint hash mismatch")
    import torch
    import yaml
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    for split, expected in {"train": 110478, "val": 21460}.items():
        if len(os.listdir(DATASET / "images" / split)) != expected:
            raise ValueError(f"Unexpected {split} size")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configuration = OUTPUT / "data.yaml"
    configuration.write_text(yaml.safe_dump({"path": str(DATASET), "train": "images/train", "val": "images/val",
                                             "names": ["car", "motorcycle", "bus", "truck"]}, sort_keys=False))
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 41 or checkpoint.get("optimizer") is None:
        raise ValueError("Not the expected resumable epoch-42 checkpoint")
    # Redirect the Kaggle paths in a copy; the original checkpoint stays untouched.
    arguments_saved = checkpoint["train_args"]
    arguments_saved.update(data=str(configuration), save_dir=str(OUTPUT / "train"), project=str(OUTPUT), name="train",
                           batch=arguments.batch, workers=arguments.workers, device=0)
    patched = OUTPUT / "resume_last_local.pt"
    torch.save(checkpoint, patched)
    del checkpoint
    model = YOLO(str(patched))
    model.train(resume=True, batch=arguments.batch, workers=arguments.workers, device=0)


if __name__ == "__main__":
    main()
