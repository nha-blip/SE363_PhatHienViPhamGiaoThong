"""Prepare Kaggle training configurations without modifying datasets."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


SCHEMAS = {
    "vehicle": {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"},
    "traffic_light": {0: "red", 1: "yellow", 2: "green", 3: "off"},
}
EXPECTED_COUNTS = {
    "vehicle": {"train": 110478, "val": 21460, "test": 21236},
    "traffic_light": {"train": 5541, "val": 933, "test": 890},
}
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def normalized_names(value):
    if isinstance(value, list):
        names = dict(enumerate(value))
    elif isinstance(value, dict):
        names = {int(key): name for key, name in value.items()}
    else:
        return {}
    if names == {0: "red", 1: "yellow", 2: "green", 3: False}:
        names[3] = "off"
    return names


def validate_root(root, task):
    root = root.resolve(strict=True)
    for split in SPLITS:
        for kind in ("images", "labels"):
            directory = root / kind / split
            if not directory.is_dir() or root not in directory.resolve().parents:
                raise ValueError(f"Missing/invalid directory: {directory}")
    configs = [root / name for name in ("dataset.yaml", "data.yaml") if (root / name).is_file()]
    if not configs:
        raise ValueError(f"No source dataset YAML found in {root}")
    for config in configs:
        names = normalized_names(yaml.safe_load(config.read_text(encoding="utf-8"))["names"])
        if names != SCHEMAS[task]:
            raise ValueError(f"Incorrect class mapping for {task}: {config}")
    return root


def discover(input_root):
    candidates = {task: set() for task in SCHEMAS}
    for directory, children, filenames in os.walk(input_root, followlinks=False):
        root = Path(directory)
        depth = len(root.relative_to(input_root).parts)
        children[:] = [name for name in children if name not in {"images", "labels", "metadata", "qa", "previews"} and not (root / name).is_symlink() and depth < 6]
        for filename in ("dataset.yaml", "data.yaml"):
            if filename not in filenames:
                continue
            try:
                source = yaml.safe_load((root / filename).read_text(encoding="utf-8"))
                if not isinstance(source, dict):
                    continue
                names = normalized_names(source.get("names"))
                for task, schema in SCHEMAS.items():
                    if names == schema:
                        candidates[task].add(validate_root(root, task))
            except (ValueError, KeyError, TypeError, yaml.YAMLError) as error:
                print(f"Ignored candidate {root / filename}: {error}")
    return candidates


def choose_root(task, explicit, candidates):
    if explicit:
        return validate_root(Path(explicit), task)
    choices = candidates[task]
    if len(choices) != 1:
        raise ValueError(f"Expected one {task} dataset, found {len(choices)}: {sorted(map(str, choices))}. Supply --{task.replace('_', '-')}-root explicitly.")
    return next(iter(choices))


def check_pairs(root, task):
    counts = {}
    for split in SPLITS:
        images = set()
        labels = set()
        for path in (root / "images" / split).rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unexpected image file: {path}")
            key = path.relative_to(root / "images" / split).with_suffix("").as_posix()
            if key in images:
                raise ValueError(f"Repeated image stem: {path}")
            images.add(key)
        for path in (root / "labels" / split).rglob("*"):
            if not path.is_file():
                continue
            if path.suffix != ".txt":
                raise ValueError(f"Unexpected label file: {path}")
            labels.add(path.relative_to(root / "labels" / split).with_suffix("").as_posix())
        if images != labels or len(images) != EXPECTED_COUNTS[task][split]:
            raise ValueError(f"{task}/{split}: images={len(images)}, labels={len(labels)}, missing labels={len(images - labels)}, orphan labels={len(labels - images)}; expected {EXPECTED_COUNTS[task][split]}")
        counts[split] = len(images)
    return counts


def build_configs(roots, output):
    configs = {}
    for task, root in roots.items():
        dataset_path = output / f"{task}_dataset.yaml"
        configs[dataset_path] = {"path": root.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": SCHEMAS[task]}
        train = {
            "task": "detect", "mode": "train", "model": "yolov8n.pt",
            "data": dataset_path.as_posix(), "device": 0 if task == "vehicle" else 1,
            "epochs": 100, "imgsz": 640, "batch": 8, "workers": 2,
            "patience": 20, "seed": 363, "deterministic": True,
            "pretrained": True, "amp": True, "cache": False,
            "optimizer": "AdamW", "lr0": 0.001,
            "save": True, "save_period": 5, "val": True, "plots": True,
            "project": "/kaggle/working/runs", "name": f"{task}_yolov8n_640",
            "exist_ok": False, "resume": False, "single_cls": False,
        }
        if task == "traffic_light":
            train.update(hsv_h=0.0, hsv_s=0.0, hsv_v=0.2)
        configs[output / f"{task}_train.yaml"] = train
    return configs


def command_arguments(argv):
    if argv is not None:
        return list(argv)
    arguments = list(sys.argv[1:])
    if "ipykernel" not in sys.modules:
        return arguments
    for index in range(len(arguments) - 1):
        connection = Path(arguments[index + 1]).name
        if arguments[index] == "-f" and connection.startswith("kernel-") and connection.endswith(".json"):
            return arguments[:index] + arguments[index + 2:]
    return arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate two independent Kaggle YOLOv8 configs; does NOT start training or alter datasets.")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-dir", default="/kaggle/working/configs")
    parser.add_argument("--vehicle-root")
    parser.add_argument("--traffic-light-root")
    parser.add_argument("--check-pairs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(command_arguments(argv))
    input_root = Path(arguments.input_root).resolve(strict=True)
    candidates = discover(input_root)
    roots = {
        "vehicle": choose_root("vehicle", arguments.vehicle_root, candidates),
        "traffic_light": choose_root("traffic_light", arguments.traffic_light_root, candidates),
    }
    output = Path(arguments.output_dir).resolve()
    if output == input_root or input_root in output.parents or any(output == root or root in output.parents for root in roots.values()):
        raise ValueError("Generated configs must be OUTSIDE input datasets; use /kaggle/working/configs")
    pair_counts = {task: check_pairs(root, task) for task, root in roots.items()} if arguments.check_pairs else "Not checked; run --check-pairs before training"
    configs = build_configs(roots, output)
    payloads = {path: yaml.safe_dump(config, sort_keys=False, allow_unicode=True) for path, config in configs.items()}
    for path, content in payloads.items():
        print(f"\n--- {path.name} ---\n{content}")
        if not arguments.dry_run and path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Existing config differs: {path}. Use a new output directory to avoid overwriting settings.")
    if not arguments.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        for path, content in payloads.items():
            if not path.exists():
                path.write_text(content, encoding="utf-8")
    print(json.dumps({"dataset_roots": {task: str(root) for task, root in roots.items()}, "pair_counts": pair_counts, "configs_written": 0 if arguments.dry_run else len(configs), "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
