"""Kaggle2 notebook: per-ROI signal colour classifier (red, yellow, green, off, none), 2026-11-17.

Input (private): v5nhdf/yolo363-light-colour-crops-20261117 -> light_crops.zip.bin + package.json (SHA-256).
Crops imitate a hand-drawn signal ROI around one head (scripts/build_light_crops_20261117.py). Colour augmentation is
limited on purpose (no hue shift, no RandAugment): the label IS the colour. Val = model A's val crops + held-out VN videos;
accuracy is also reported per class and for the VN val crops alone. Outputs in /kaggle/working/out.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

OUT = Path("/kaggle/working/out")
DATA = Path("/tmp/crops_data")
EPOCHS, PATIENCE, IMGSZ, BATCH = 60, 15, 64, 256


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
    archive = find("light_crops.zip.bin")
    if sha256(archive) != package["sha256"]["light_crops.zip.bin"]:
        raise RuntimeError("SHA-256 mismatch light_crops.zip.bin")
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            p = PurePosixPath(member.filename)
            if p.is_absolute() or ".." in p.parts or p.parts[0] != "crops":
                raise RuntimeError(f"unsafe member {member.filename}")
        z.extractall(DATA)
    root = DATA / "crops"
    log("DATA_READY", **{f"{s}/{c.name}": len(list(c.iterdir())) for s in ("train", "val") for c in sorted((root / s).iterdir()) if c.is_dir()})

    model = YOLO("yolo11s-cls.pt")
    model.train(data=str(root), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ, batch=BATCH, device=0, workers=4, seed=363,
                hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, auto_augment=None, erasing=0.2, fliplr=0.5, translate=0.1, scale=0.3,
                project=str(OUT), name="train", exist_ok=True, plots=False)
    run = OUT / "train"
    for name in ("best.pt", "last.pt"):
        shutil.copy2(run / "weights" / name, OUT / name)
    shutil.copy2(run / "results.csv", OUT / "results.csv")

    best = YOLO(str(OUT / "best.pt"))
    report = {}
    for subset, keep in (("val_all", lambda p: True), ("val_vn", lambda p: p.name.startswith("vnpool__"))):
        confusion = Counter()
        for cls_dir in sorted((root / "val").iterdir()):
            files = [p for p in cls_dir.iterdir() if keep(p)]
            for k in range(0, len(files), 512):
                for r in best.predict([str(p) for p in files[k:k + 512]], imgsz=IMGSZ, verbose=False):
                    confusion[(cls_dir.name, best.names[r.probs.top1])] += 1
        classes = sorted({t for t, _ in confusion})
        report[subset] = {"accuracy": sum(v for (t, p), v in confusion.items() if t == p) / max(1, sum(confusion.values())),
                          "per_class_recall": {c: confusion[(c, c)] / max(1, sum(v for (t, _), v in confusion.items() if t == c)) for c in classes},
                          "confusion": {f"{t}->{p}": v for (t, p), v in sorted(confusion.items())}}
        log("EVALUATED", subset=subset, accuracy=report[subset]["accuracy"], recall=report[subset]["per_class_recall"])
    (OUT / "metrics.json").write_text(json.dumps({"report": report, "best_sha256": sha256(OUT / "best.pt"), "names": best.names,
                                                  "settings": {"model": "yolo11s-cls", "epochs": EPOCHS, "patience": PATIENCE, "imgsz": IMGSZ}}, indent=2))
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        log("FAILED", error=f"{type(error).__name__}: {error}")
        raise
