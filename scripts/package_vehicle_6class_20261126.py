"""Package the 6-class vehicle training data for Kaggle2: exports/yolov8s_vehicle_6class_20261126/upload/.

vehicle6.zip.bin (ZIP_STORED, top folder veh6/):
  images|labels/train   Vehicle363 full-frame train (data/YOLO363_training_working/Vehicle363-v1) with the 6-class labels
                        (data/Vehicle363_6class_labels_fullframe_20261125) + minibus add (355) + VN pool train (6-class)
  images|labels/val     Vehicle363 val with 6-class labels (= dev, model selection)
  images|labels/val_vn  VN pool val (6-class), reported separately
The Vehicle363 test split is NOT uploaded (kept sealed, spec §15.4). vehicle_v8s_best.pt = the current pipeline model
(start weights; its 4-class head is replaced by a 6-class head). package.json holds SHA-256 of both files.
"""
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports/yolov8s_vehicle_6class_20261126/upload"
V363 = ROOT / "data/YOLO363_training_working/Vehicle363-v1/images"
LAB6 = ROOT / "data/Vehicle363_6class_labels_fullframe_20261125"
MINI = ROOT / "data/Vehicle363_minibus_add_20261122"
VN = ROOT / "data/Vehicle363_vn_pool_6class_20261125"
WEIGHTS = ROOT / "weights/kaggle_small_vehicle_20261019/best.pt"
WEIGHTS_SHA = "e1254d4c171593ded3f57939163d1cb2b2aa4e93a0a29dfa332824555a7cc35f"
NAMES = ["car", "motorcycle", "bus", "truck", "pickup", "minibus"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def pairs():
    """(split, image path, label path) for every packaged image; each image has exactly one label file."""
    for split in ("train", "val"):
        for img in sorted((V363 / split).iterdir()):
            yield split, img, LAB6 / split / f"{img.stem}.txt"
    for img in sorted((MINI / "images/train").iterdir()):
        yield "train", img, MINI / "labels/train" / f"{img.stem}.txt"
    for src, dst in (("train", "train"), ("val", "val_vn")):
        for img in sorted((VN / "images" / src).iterdir()):
            yield dst, img, VN / "labels" / src / f"{img.stem}.txt"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha256(WEIGHTS) != WEIGHTS_SHA:
        raise RuntimeError("start weights hash mismatch")
    OUT.mkdir(parents=True)
    stats, seen = Counter(), set()
    with zipfile.ZipFile(OUT / "vehicle6.zip.bin", "w", zipfile.ZIP_STORED) as z:
        for split, img, label in pairs():
            if (split, img.stem) in seen:
                raise RuntimeError(f"duplicate name {split}/{img.stem}")
            seen.add((split, img.stem))
            lines = [l for l in label.read_text().splitlines() if l.strip()]  # raises if a label is missing
            for l in lines:
                c = int(l.split()[0])
                assert 0 <= c < len(NAMES), (label, l)
                stats[f"{split}_{NAMES[c]}"] += 1
            z.write(img, f"veh6/images/{split}/{img.name}")
            z.writestr(f"veh6/labels/{split}/{img.stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
            stats[f"{split}_images"] += 1
    shutil.copy2(WEIGHTS, OUT / "vehicle_v8s_best.pt")
    package = {"created": "2026-11-26", "purpose": "train YOLOv8s 6-class vehicle detector (spec v1.1 §15.3) from the v8s weights",
               "classes": NAMES, "sha256": {n: sha256(OUT / n) for n in ("vehicle6.zip.bin", "vehicle_v8s_best.pt")},
               "stats": dict(sorted(stats.items())), "test_uploaded": False,
               "license": "MIO-TCD CC BY-NC-SA 4.0; HCMC terms unknown; HeteroTraffic CC BY 4.0; UA-DETRAC terms not verified; "
                          "VN frames from public YouTube videos. Course research only, private, not redistributed"}
    (OUT / "package.json").write_text(json.dumps(package, indent=1), encoding="utf-8")
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "yolo363 vehicle 6class 20261126", "id": "v5nhdf/yolo363-vehicle-6class-20261126",
                                                          "licenses": [{"name": "unknown"}]}, indent=1), encoding="utf-8")
    print(json.dumps(package, indent=1))


if __name__ == "__main__":
    main()
