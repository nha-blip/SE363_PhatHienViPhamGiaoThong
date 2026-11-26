"""Kaggle2 notebook: YOLOv8s 6-class vehicle detector (spec v1.1 §15.3), 2026-11-26.

Input (private dataset v5nhdf/yolo363-vehicle-6class-20261126): vehicle6.zip.bin (veh6/images|labels/train|val|val_vn),
vehicle_v8s_best.pt (current 4-class pipeline model = start weights; the detect head's class layers are re-initialised
for 6 classes), package.json (SHA-256). Recipe = the original v8s run (AdamW lr0 0.001, lrf 0.01, 640, batch 8, mosaic 1.0,
hsv_h 0.015, close_mosaic 10, patience 20, seed 363) with a 10 h time budget (ultralytics fits the epoch count and LR
schedule to it). Selection val = Vehicle363 val (dev) + VN val. The Vehicle363 test split is not on Kaggle.
After training, best.pt is evaluated on dev and VN val at 640 and 960 (runtime size). Outputs in /kaggle/working/out.
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
DATA = Path("/tmp/veh6data")
NAMES = ["car", "motorcycle", "bus", "truck", "pickup", "minibus"]
TIME_BUDGET_HOURS = 10.0


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.178"], check=True)
    from ultralytics import YOLO

    package = json.loads(find("package.json").read_text())
    data_zip, weights = find("vehicle6.zip.bin"), find("vehicle_v8s_best.pt")
    for path in (data_zip, weights):
        if sha256(path) != package["sha256"][path.name]:
            raise RuntimeError(f"SHA-256 mismatch {path.name}")
    log("INPUTS_VERIFIED")
    with zipfile.ZipFile(data_zip) as z:
        for m in z.infolist():
            p = PurePosixPath(m.filename)
            if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "veh6":
                raise RuntimeError(f"unsafe member {m.filename}")
        z.extractall(DATA)
    root = DATA / "veh6"
    for split in ("train", "val", "val_vn"):
        if {p.stem for p in (root / "images" / split).iterdir()} != {p.stem for p in (root / "labels" / split).iterdir()}:
            raise RuntimeError(f"{split} image/label mismatch")
    names = "names:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(NAMES))
    data_yaml = DATA / "data.yaml"
    data_yaml.write_text(f"path: {root}\ntrain: images/train\nval:\n  - images/val\n  - images/val_vn\n" + names)
    vals = {}
    for key in ("val", "val_vn"):
        vals[key] = DATA / f"{key}.yaml"
        vals[key].write_text(f"path: {root}\ntrain: images/{key}\nval: images/{key}\n" + names)
    log("DATA_READY", **{s: len(list((root / "images" / s).iterdir())) for s in ("train", "val", "val_vn")})

    YOLO(str(weights)).train(data=str(data_yaml), epochs=100, time=TIME_BUDGET_HOURS, patience=20, imgsz=640, batch=8, device=0,
                             workers=2, seed=363, deterministic=True, optimizer="AdamW", lr0=0.001, lrf=0.01, warmup_epochs=3.0,
                             mosaic=1.0, hsv_h=0.015, close_mosaic=10, project=str(OUT), name="train", exist_ok=True, plots=False)
    run = OUT / "train"
    for name in ("best.pt", "last.pt"):
        shutil.copy2(run / "weights" / name, OUT / name)
    shutil.copy2(run / "results.csv", OUT / "results.csv")
    log("TRAINED")

    result = {}
    for imgsz in (640, 960):
        for key, yaml_path in vals.items():
            m = YOLO(str(OUT / "best.pt")).val(data=str(yaml_path), split="val", imgsz=imgsz, batch=16, device=0, plots=False, verbose=False)
            per_class = {}
            for i, c in enumerate(m.box.ap_class_index):
                p, r, ap50, ap = m.box.class_result(i)
                per_class[NAMES[int(c)]] = {"precision": float(p), "recall": float(r), "ap50": float(ap50), "ap50_95": float(ap)}
            result[f"{key}@{imgsz}"] = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": per_class}
            log("EVALUATED", split=key, imgsz=imgsz, map50_95=float(m.box.map))
    (OUT / "metrics.json").write_text(json.dumps({"best": result, "best_sha256": sha256(OUT / "best.pt"), "names": NAMES,
                                                  "settings": {"time_hours": TIME_BUDGET_HOURS, "imgsz": 640, "batch": 8, "optimizer": "AdamW",
                                                               "lr0": 0.001, "lrf": 0.01, "start": "vehicle_v8s_best.pt"}}, indent=2))
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        log("FAILED", error=f"{type(error).__name__}: {error}")
        raise
