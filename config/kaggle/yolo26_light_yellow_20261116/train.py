"""Kaggle2 notebook: fine-tune light model A with extra yellow lights from LISA (2026-11-16).

Inputs (private datasets):
  v5nhdf/yolo363-fullscene-resume-20261028  -> light_fullscene.zip.bin + training_package.json (model A's data)
  v5nhdf/yolo363-light-yellow-20261116      -> light_A_best.pt, lisa_yellow.zip.bin, package.json (SHA-256s)
Train = A's full-scene train list (+ LISA yellow-focused train frames). Validation = A's original val (comparable
with A). LISA holdout sequences are evaluated separately as a yellow check; they are never trained on.
Outputs in /kaggle/working/out: best.pt, last.pt, results.csv, metrics.json, train.log.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

OUT = Path("/kaggle/working/out")
DATA = Path("/tmp/light_data")  # not /kaggle/working: keeps the notebook output small
NAMES = ["red", "yellow", "green", "off"]
EPOCHS, PATIENCE, IMGSZ, BATCH = 40, 12, 640, 16
TIME_BUDGET_HOURS = 3.3


def log(event, **fields):
    line = json.dumps({"t": round(time.time()), "event": event, **fields})
    print(line, flush=True)
    with (OUT / "train.log").open("a") as handle:
        handle.write(line + "\n")


def find(name):
    matches = list(Path("/kaggle/input").glob("**/" + name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one input {name}, found {len(matches)}")
    return matches[0]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive_path, destination, top):
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != top:
                raise RuntimeError(f"unsafe member {member.filename}")
        archive.extractall(destination)
    return destination / top


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.178"], check=True)
    from ultralytics import YOLO

    package = json.loads(find("package.json").read_text())
    previous = json.loads(find("training_package.json").read_text())["light"]
    fullscene_zip, lisa_zip, weights = find("light_fullscene.zip.bin"), find("lisa_yellow.zip.bin"), find("light_A_best.pt")
    for path, expected in ((fullscene_zip, previous["sha256"]), (lisa_zip, package["sha256"]["lisa_yellow.zip.bin"]),
                           (weights, package["sha256"]["light_A_best.pt"])):
        if sha256(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch {path.name}")
    log("INPUTS_VERIFIED")
    light = safe_extract(fullscene_zip, DATA, "light")
    lisa = safe_extract(lisa_zip, DATA, "lisa")
    for split in ("train", "val"):
        images = {p.stem for p in (light / "images" / split).iterdir()}
        labels = {p.stem for p in (light / "labels" / split).iterdir()}
        if images != labels:
            raise RuntimeError(f"fullscene {split} image/label mismatch")
    balanced = light / "train_balanced.txt"
    base_list = ([str((light / line.strip().lstrip("./")).resolve()) for line in balanced.read_text().splitlines() if line.strip()]
                 if balanced.exists() else [str(p) for p in sorted((light / "images/train").iterdir())])
    lisa_train = [str(p) for p in sorted((lisa / "images/train").iterdir())]
    train_list = DATA / "train_list.txt"
    train_list.write_text("\n".join(base_list + lisa_train) + "\n")
    data_yaml = DATA / "data.yaml"
    data_yaml.write_text(f"train: {train_list}\nval: {light / 'images/val'}\nnames:\n" + "".join(f"  {k}: '{n}'\n" for k, n in enumerate(NAMES)))  # quoted: YAML reads bare off as False
    holdout_yaml = DATA / "lisa_holdout.yaml"
    holdout_yaml.write_text(f"train: {lisa / 'images/holdout'}\nval: {lisa / 'images/holdout'}\nnames:\n" + "".join(f"  {k}: '{n}'\n" for k, n in enumerate(NAMES)))
    log("DATA_READY", base_train_entries=len(base_list), lisa_train=len(lisa_train), balanced_list=balanced.exists())

    def evaluate(model_path, tag):
        result = {}
        for name, yaml_path in (("light_val", data_yaml), ("lisa_holdout", holdout_yaml)):
            metrics = YOLO(str(model_path)).val(data=str(yaml_path), split="val", imgsz=IMGSZ, batch=BATCH, device=0, plots=False, verbose=False)
            per_class = {}
            for index, class_index in enumerate(metrics.box.ap_class_index):
                p, r, ap50, ap = metrics.box.class_result(index)
                per_class[NAMES[int(class_index)]] = {"precision": float(p), "recall": float(r), "ap50": float(ap50), "ap50_95": float(ap)}
            result[name] = {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_class": per_class}
        log("EVALUATED", model=tag, **{k: {"map50_95": v["map50_95"], "yellow": v["per_class"].get("yellow")} for k, v in result.items()})
        return result

    before = evaluate(weights, "A_before")
    model = YOLO(str(weights))
    model.train(data=str(data_yaml), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ, batch=BATCH, device=0, workers=4, seed=363,
                optimizer="SGD", lr0=0.002, lrf=0.1, warmup_epochs=1.0, cos_lr=True,
                hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=0.0, close_mosaic=0,
                time=TIME_BUDGET_HOURS, project=str(OUT), name="train", exist_ok=True, plots=False)
    run = OUT / "train"
    for name in ("best.pt", "last.pt"):
        shutil.copy2(run / "weights" / name, OUT / name)
    shutil.copy2(run / "results.csv", OUT / "results.csv")
    after = evaluate(OUT / "best.pt", "finetuned_best")
    (OUT / "metrics.json").write_text(json.dumps({"before_A": before, "after_best": after, "best_sha256": sha256(OUT / "best.pt"),
                                                  "settings": {"epochs": EPOCHS, "patience": PATIENCE, "imgsz": IMGSZ, "batch": BATCH,
                                                               "lr0": 0.002, "optimizer": "SGD"}}, indent=2))
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        log("FAILED", error=f"{type(error).__name__}: {error}")
        raise
