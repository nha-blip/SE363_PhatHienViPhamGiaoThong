"""Kaggle2 notebook: fine-tune the chosen vehicle model (YOLOv8s, runs at 960) with owner-QA'd VN frames (2026-11-18).

Inputs (private datasets):
  v5nhdf/yolo363-vehicle-archive-20261027   -> vehicle_trainval.zip.bin (Vehicle363 train/val, vehicle/<images|labels>/<split>)
  v5nhdf/yolo363-vehicle-vnpool-20261118    -> vn_vehicle.zip.bin + vehicle_v8s_best.pt + package.json (SHA-256)
Trains at imgsz 960 = the runtime size (the 640 light fine-tune collapsed at 960). Train = a fixed random subset of Vehicle363
train (the full 110k images do not fit the budget) + VN train x VN_REPEAT. Selection val = Vehicle363 val subset + VN val.
Both vals are reported separately before and after. Outputs in /kaggle/working/out.
"""
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

OUT = Path("/kaggle/working/out")
DATA = Path("/tmp/vehicle_data")
NAMES = ["car", "motorcycle", "bus", "truck"]
ARCHIVE_SHA256 = "d4fe27fbbf8d0b670c71a2b7500c8a277f00688d7eded24878373440ee1420fe"
BASE_TRAIN, BASE_VAL = 9000, 2000
VN_REPEAT = 15  # 80 VN images -> ~12% of each epoch
EPOCHS, PATIENCE, IMGSZ, BATCH = 15, 6, 960, 16
TIME_BUDGET_HOURS = 3.0


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


def safe(member, top):
    p = PurePosixPath(member.filename)
    if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != top:
        raise RuntimeError(f"unsafe member {member.filename}")
    return p


def extract_base(archive_path):
    """Extract a seeded random subset of Vehicle363 train and val (image + label pairs only)."""
    rng = random.Random(363)
    with zipfile.ZipFile(archive_path) as archive:
        pairs = {}
        for m in archive.infolist():
            if m.is_dir():
                continue
            p = safe(m, "vehicle")
            if len(p.parts) == 4 and p.parts[1] in ("images", "labels") and p.parts[2] in ("train", "val"):
                pairs.setdefault((p.parts[2], p.stem), {})[p.parts[1]] = m
        chosen = []
        for split, n in (("train", BASE_TRAIN), ("val", BASE_VAL)):
            stems = sorted(s for (sp, s), v in pairs.items() if sp == split and len(v) == 2)
            for s in rng.sample(stems, n):
                chosen += pairs[(split, s)].values()
        for m in chosen:
            archive.extract(m, DATA)
    return DATA / "vehicle"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.178"], check=True)
    from ultralytics import YOLO

    package = json.loads(find("package.json").read_text())
    base_zip, vn_zip, weights = find("vehicle_trainval.zip.bin"), find("vn_vehicle.zip.bin"), find("vehicle_v8s_best.pt")
    for path, expected in ((base_zip, ARCHIVE_SHA256), (vn_zip, package["sha256"]["vn_vehicle.zip.bin"]),
                           (weights, package["sha256"]["vehicle_v8s_best.pt"])):
        if sha256(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch {path.name}")
    log("INPUTS_VERIFIED")
    base = extract_base(base_zip)
    with zipfile.ZipFile(vn_zip) as z:
        for m in z.infolist():
            safe(m, "vnveh")
        z.extractall(DATA)
    vn = DATA / "vnveh"
    for root in (base, vn):
        for split in ("train", "val"):
            if {p.stem for p in (root / "images" / split).iterdir()} != {p.stem for p in (root / "labels" / split).iterdir()}:
                raise RuntimeError(f"{root.name} {split} image/label mismatch")
    vn_train = [str(p) for p in sorted((vn / "images/train").iterdir())]
    train_list = DATA / "train_list.txt"
    train_list.write_text("\n".join([str(p) for p in sorted((base / "images/train").iterdir())] + vn_train * VN_REPEAT) + "\n")
    names = "names:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(NAMES))
    data_yaml = DATA / "data.yaml"
    data_yaml.write_text(f"train: {train_list}\nval:\n  - {base / 'images/val'}\n  - {vn / 'images/val'}\n" + names)
    vals = {}
    for key, folder in (("base_val", base / "images/val"), ("vn_val", vn / "images/val")):
        vals[key] = DATA / f"{key}.yaml"
        vals[key].write_text(f"train: {folder}\nval: {folder}\n" + names)
    log("DATA_READY", base_train=BASE_TRAIN, base_val=BASE_VAL, vn_train=len(vn_train), vn_repeat=VN_REPEAT)

    def evaluate(model_path, tag):
        result = {}
        for key, yaml_path in vals.items():
            m = YOLO(str(model_path)).val(data=str(yaml_path), split="val", imgsz=IMGSZ, batch=BATCH, device=0, plots=False, verbose=False)
            per_class = {}
            for i, c in enumerate(m.box.ap_class_index):
                p, r, ap50, ap = m.box.class_result(i)
                per_class[NAMES[int(c)]] = {"precision": float(p), "recall": float(r), "ap50": float(ap50), "ap50_95": float(ap)}
            result[key] = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": per_class}
        log("EVALUATED", model=tag, **{k: {"map50_95": v["map50_95"], "motorcycle": v["per_class"].get("motorcycle")} for k, v in result.items()})
        return result

    before = evaluate(weights, "v8s_before")
    YOLO(str(weights)).train(data=str(data_yaml), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ, batch=BATCH, device=0, workers=4, seed=363,
                             optimizer="SGD", lr0=0.002, lrf=0.1, warmup_epochs=1.0, cos_lr=True, close_mosaic=3,
                             time=TIME_BUDGET_HOURS, project=str(OUT), name="train", exist_ok=True, plots=False)
    run = OUT / "train"
    for name in ("best.pt", "last.pt"):
        shutil.copy2(run / "weights" / name, OUT / name)
    shutil.copy2(run / "results.csv", OUT / "results.csv")
    after = evaluate(OUT / "best.pt", "finetuned_best")
    (OUT / "metrics.json").write_text(json.dumps({"before_v8s": before, "after_best": after, "best_sha256": sha256(OUT / "best.pt"),
                                                  "settings": {"epochs": EPOCHS, "patience": PATIENCE, "imgsz": IMGSZ, "batch": BATCH, "lr0": 0.002,
                                                               "optimizer": "SGD", "vn_repeat": VN_REPEAT, "base_train": BASE_TRAIN, "base_val": BASE_VAL}}, indent=2))
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        log("FAILED", error=f"{type(error).__name__}: {error}")
        raise
