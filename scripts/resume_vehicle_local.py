import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
EXPECTED_IMAGES = {"train": 110478, "val": 21460, "test": 21236}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stems(directory, extensions):
    result = set()
    for folder, _, filenames in os.walk(directory):
        for filename in filenames:
            path = Path(folder) / filename
            if path.suffix.lower() not in extensions:
                continue
            key = path.relative_to(directory).with_suffix("").as_posix()
            if key in result:
                raise ValueError(f"Duplicate stem: {path}")
            result.add(key)
    return result


def normalize_optimizer_state(optimizer):
    import torch

    changes = []
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(name)
                if value is None:
                    continue
                if value.shape != parameter.shape:
                    raise ValueError("Saved optimizer buffer shape does not match current parameter")
                if parameter.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("Resume requires finite optimizer moments with FP32 model parameters")
                if value.dtype != parameter.dtype or value.device != parameter.device or value.stride() != parameter.stride():
                    state[name] = torch.empty_strided(parameter.shape, parameter.stride(), device=parameter.device, dtype=parameter.dtype).copy_(value)
                    changes.append({"group": group_index, "parameter": parameter_index, "buffer": name,
                                    "old_dtype": str(value.dtype), "new_dtype": str(parameter.dtype),
                                    "old_stride": list(value.stride()), "new_stride": list(parameter.stride())})
            step = state.get("step")
            if torch.is_tensor(step) and (group.get("fused") or group.get("capturable")):
                state["step"] = step.to(device=parameter.device, dtype=torch.float32)
    return changes


def use_nonfused_adamw(optimizer):
    import torch

    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("This compatibility path is only for the saved AdamW optimizer")
    for group in optimizer.param_groups:
        group["fused"] = False
        group["foreach"] = False
    optimizer.defaults["fused"] = False
    optimizer.defaults["foreach"] = False
    optimizer._step_supports_amp_scaling = False


def compatible_trainer_class():
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    class CompatibleResumeTrainer(DetectionTrainer):
        def build_optimizer(self, *args, **kwargs):
            self.model.float()
            optimizer = super().build_optimizer(*args, **kwargs)
            use_nonfused_adamw(optimizer)
            return optimizer

        def _load_checkpoint_state(self, checkpoint):
            self.model.float()
            super()._load_checkpoint_state(checkpoint)
            use_nonfused_adamw(self.optimizer)
            changes = normalize_optimizer_state(self.optimizer)
            self.stopper.patience = self.args.patience or float("inf")
            record = {"status": "APPLIED", "optimizer_reset": False, "optimizer_buffer_conversions": changes,
                      "optimizer_kernel": "nonfused_single_tensor_AdamW",
                      "model_parameter_dtype": "float32", "epoch_restored": checkpoint.get("epoch"),
                      "scaler_restored": checkpoint.get("scaler") is not None,
                      "early_stopping_disabled": self.args.patience == 0,
                      "note": "Cross-PyTorch compatibility only; preserved optimizer moments/steps and original input checkpoint"}
            (self.save_dir / "resume_compatibility.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"RESUME_OPTIMIZER_COMPATIBILITY_OK: {len(changes)} buffers converted; optimizer not reset", flush=True)

    return CompatibleResumeTrainer


def main():
    parser = argparse.ArgumentParser(description="Resume an approved Vehicle training checkpoint; never fine-tune silently")
    parser.add_argument("--config", default=str(ROOT / "config/vehicle_resume.yaml"))
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    configuration = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    checkpoint_path = Path(configuration["model"]).resolve(strict=True)
    output = Path(configuration["save_dir"]).resolve()
    allowed_outputs = (ROOT / "runs/local_training").resolve()
    if allowed_outputs not in output.parents:
        raise ValueError("Resume outputs must stay inside project runs/local_training")
    if output.exists():
        raise FileExistsError("Resume output already exists; do not restart epoch-5 training over current progress. Use that run's latest last.pt with a new explicit resume configuration.")
    dataset_path = Path(configuration["data"]).resolve(strict=True)
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    dataset_root = Path(dataset["path"]).resolve(strict=True)
    if (ROOT / "data/YOLO363_training_working").resolve() not in dataset_root.parents:
        raise ValueError("Use separate local working data, never a frozen/raw dataset")
    if dataset["names"] != EXPECTED_NAMES:
        raise ValueError("Vehicle class namespace mismatch")
    print("Checking local image/label pairs...", flush=True)
    counts = {}
    for split, expected in EXPECTED_IMAGES.items():
        images = stems(dataset_root / "images" / split, IMAGE_EXTENSIONS)
        labels = stems(dataset_root / "labels" / split, {".txt"})
        if images != labels or len(images) != expected:
            raise ValueError(f"Incomplete {split}: images={len(images)}, labels={len(labels)}, expected={expected}")
        counts[split] = len(images)
    import torch
    import ultralytics
    from ultralytics import YOLO

    initial_hash = sha256(checkpoint_path)
    print("Loading owner-provided trusted training checkpoint on CPU...", flush=True)
    model = YOLO(str(checkpoint_path))
    checkpoint = model.ckpt
    saved_args = checkpoint.get("train_args", {})
    saved_epoch = checkpoint.get("epoch")
    if not isinstance(saved_epoch, int) or not 0 <= saved_epoch < 99 or checkpoint.get("optimizer") is None:
        raise ValueError("Checkpoint must contain an unfinished training epoch and optimizer state; no restart/fine-tuning fallback allowed")
    completed_epochs = saved_epoch + 1
    if saved_args.get("epochs") != 100 or configuration.get("epochs") != 100 or not configuration.get("resume"):
        raise ValueError("True resume requires total 100 epochs; do not set remaining budget to epochs=95")
    if model.names != EXPECTED_NAMES:
        raise ValueError("Checkpoint is not the approved four-class Vehicle detector")
    if checkpoint.get("version") != ultralytics.__version__:
        raise ValueError("Use the checkpoint's Ultralytics version before resuming")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; refusing silent CPU training")
    if checkpoint.get("scaler") is None:
        raise ValueError("Expected AMP scaler state missing")
    proof = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": initial_hash,
             "completed_epochs": completed_epochs, "next_epoch": completed_epochs + 1, "total_epochs": 100, "remaining_epochs": 100 - completed_epochs,
             "optimizer_present": True, "scaler_present": True, "checkpoint_version": checkpoint["version"],
             "original_training_arguments": saved_args, "local_configuration": configuration,
             "dataset_pair_counts": counts, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
             "gpu": torch.cuda.get_device_name(0), "original_checkpoint_modified": False}
    print(json.dumps({name: proof[name] for name in ("completed_epochs", "next_epoch", "total_epochs", "remaining_epochs", "gpu", "dataset_pair_counts")}, indent=2), flush=True)
    if arguments.check_only:
        if sha256(checkpoint_path) != initial_hash:
            raise RuntimeError("Original checkpoint changed")
        print("RESUME_PREFLIGHT_PASS; training not started", flush=True)
        return
    receipt_dir = ROOT / "runs/local_training/resume_launch_records"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = receipt_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    receipt.write_text(json.dumps(proof, indent=2, default=str) + "\n", encoding="utf-8")
    options = {name: value for name, value in configuration.items() if name not in {"model", "task", "mode"}}
    model.train(trainer=compatible_trainer_class(), **options)
    if sha256(checkpoint_path) != initial_hash:
        raise RuntimeError("Original input checkpoint changed unexpectedly")


if __name__ == "__main__":
    main()
