"""Kaggle2 notebook: train the light model FROM SCRATCH (COCO yolo26s.pt, like model A) at imgsz 960 on the merged data (2026-11-18).

Same merged train list as yolo26_light_dualcam960_20261118 (A data + VN x3 + DualCam x2), but starting from yolo26s.pt with
model A's own recipe (MuSGD lr0 0.01, lrf 0.01, warmup 3, patience 15, 100 epochs max) so all sources are learned together
from the start instead of nudging A. A 12 h Kaggle session cannot hold 100 epochs at 960 (~9 min each): ultralytics `time`
caps training at TIME_BUDGET_HOURS; a checkpoint is kept every 5 epochs so the owner can pick by the VN/iHanoi evaluations.

Inputs (private datasets):
  v5nhdf/yolo363-fullscene-resume-20261028  -> light_fullscene.zip.bin + training_package.json (model A's data)
  v5nhdf/yolo363-light-yellow-20261116      -> light_A_best.pt (model A) + package.json with its SHA-256
  v5nhdf/yolo363-light-vnpool-20261117      -> vn_pool.zip.bin + package.json (VN round lamps, orange = yellow)
  v5nhdf/yolo363-light-dualcam-20261118     -> dualcam.zip.bin + package.json (DualCam, arrows/countdowns blurred)
Train = A's full-scene train list + VN train x VN_REPEAT + DualCam train x DC_REPEAT, at imgsz 960 = the runtime size (the
640 fine-tunes of 16/11 and 17/11 lost large near-camera heads at 960). A's own recipe is kept (MuSGD, no mosaic, no hue
shift) with a low lr and no warmup (warmup disturbed the 18/11 vehicle fine-tune). Validation for best.pt = A val + VN val +
DualCam val; each is reported separately, before and after. Outputs in /kaggle/working/out.
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
EPOCHS, PATIENCE, IMGSZ, BATCH = 100, 15, 960, 8
VN_REPEAT, DC_REPEAT = 3, 2  # ~10.2k base entries + 1083 VN + 2064 DualCam per epoch
TIME_BUDGET_HOURS = 10.5


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

    packages = [json.loads(p.read_text()) for p in Path("/kaggle/input").glob("**/package.json")]
    weights_sha = next(p["sha256"]["light_A_best.pt"] for p in packages if "light_A_best.pt" in p.get("sha256", {}))
    vn_sha = next(p["sha256"]["vn_pool.zip.bin"] for p in packages if "vn_pool.zip.bin" in p.get("sha256", {}))
    dc_sha = next(p["sha256"]["dualcam.zip.bin"] for p in packages if "dualcam.zip.bin" in p.get("sha256", {}))
    previous = json.loads(find("training_package.json").read_text())["light"]
    fullscene_zip, vn_zip, weights = find("light_fullscene.zip.bin"), find("vn_pool.zip.bin"), find("light_A_best.pt")
    dc_zip = find("dualcam.zip.bin")
    for path, expected in ((fullscene_zip, previous["sha256"]), (vn_zip, vn_sha), (weights, weights_sha), (dc_zip, dc_sha)):
        if sha256(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch {path.name}")
    log("INPUTS_VERIFIED")
    light = safe_extract(fullscene_zip, DATA, "light")
    vn = safe_extract(vn_zip, DATA, "vnpool")
    dc = safe_extract(dc_zip, DATA, "dualcam")
    for root, splits in ((light, ("train", "val")), (vn, ("train", "val")), (dc, ("train", "val"))):
        for split in splits:
            images = {p.stem for p in (root / "images" / split).iterdir()}
            labels = {p.stem for p in (root / "labels" / split).iterdir()}
            if images != labels:
                raise RuntimeError(f"{root.name} {split} image/label mismatch")
    balanced = light / "train_balanced.txt"
    base_list = ([str((light / line.strip().lstrip("./")).resolve()) for line in balanced.read_text().splitlines() if line.strip()]
                 if balanced.exists() else [str(p) for p in sorted((light / "images/train").iterdir())])
    vn_train = [str(p) for p in sorted((vn / "images/train").iterdir())]
    dc_train = [str(p) for p in sorted((dc / "images/train").iterdir())]
    train_list = DATA / "train_list.txt"
    train_list.write_text("\n".join(base_list + vn_train * VN_REPEAT + dc_train * DC_REPEAT) + "\n")
    names = "names:\n" + "".join(f"  {k}: '{n}'\n" for k, n in enumerate(NAMES))  # quoted: YAML reads bare off as False
    data_yaml = DATA / "data.yaml"  # selection val = A's val + VN val + DualCam val
    data_yaml.write_text(f"train: {train_list}\nval:\n  - {light / 'images/val'}\n  - {vn / 'images/val'}\n  - {dc / 'images/val'}\n" + names)
    a_val_yaml, vn_val_yaml, dc_val_yaml = DATA / "a_val.yaml", DATA / "vn_val.yaml", DATA / "dc_val.yaml"
    a_val_yaml.write_text(f"train: {light / 'images/val'}\nval: {light / 'images/val'}\n" + names)
    vn_val_yaml.write_text(f"train: {vn / 'images/val'}\nval: {vn / 'images/val'}\n" + names)
    dc_val_yaml.write_text(f"train: {dc / 'images/val'}\nval: {dc / 'images/val'}\n" + names)
    log("DATA_READY", base_train_entries=len(base_list), vn_train=len(vn_train), vn_repeat=VN_REPEAT, dc_train=len(dc_train),
        dc_repeat=DC_REPEAT, balanced_list=balanced.exists())

    def evaluate(model_path, tag):
        result = {}
        for name, yaml_path in (("a_val", a_val_yaml), ("vn_val", vn_val_yaml), ("dualcam_val", dc_val_yaml)):
            metrics = YOLO(str(model_path)).val(data=str(yaml_path), split="val", imgsz=IMGSZ, batch=BATCH, device=0, plots=False, verbose=False)
            per_class = {}
            for index, class_index in enumerate(metrics.box.ap_class_index):
                p, r, ap50, ap = metrics.box.class_result(index)
                per_class[NAMES[int(class_index)]] = {"precision": float(p), "recall": float(r), "ap50": float(ap50), "ap50_95": float(ap)}
            result[name] = {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_class": per_class}
        log("EVALUATED", model=tag, **{k: {"map50_95": v["map50_95"], "yellow": v["per_class"].get("yellow")} for k, v in result.items()})
        return result

    before = evaluate(weights, "A_before")
    model = YOLO("yolo26s.pt")  # COCO-pretrained, as model A
    model.train(data=str(data_yaml), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ, batch=BATCH, device=0, workers=4, seed=363,
                optimizer="MuSGD", lr0=0.01, lrf=0.01, warmup_epochs=3.0, cos_lr=False, save_period=5,
                hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=0.0, close_mosaic=10,
                time=TIME_BUDGET_HOURS, project=str(OUT), name="train", exist_ok=True, plots=False)
    run = OUT / "train"
    for path in sorted((run / "weights").glob("*.pt")):  # best, last and every 5th epoch
        shutil.copy2(path, OUT / path.name)
    shutil.copy2(run / "results.csv", OUT / "results.csv")
    after = evaluate(OUT / "best.pt", "finetuned_best")
    (OUT / "metrics.json").write_text(json.dumps({"before_A": before, "after_best": after, "best_sha256": sha256(OUT / "best.pt"),
                                                  "settings": {"epochs": EPOCHS, "patience": PATIENCE, "imgsz": IMGSZ, "batch": BATCH,
                                                               "start": "yolo26s.pt", "lr0": 0.01, "lrf": 0.01, "optimizer": "MuSGD", "warmup_epochs": 3.0, "time_h": TIME_BUDGET_HOURS,
                                                               "vn_repeat": VN_REPEAT, "dc_repeat": DC_REPEAT}}, indent=2))
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        log("FAILED", error=f"{type(error).__name__}: {error}")
        raise
