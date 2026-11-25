"""Export the owner-QA'd VN vehicle frames (data/vn_pool_qa_vehicle_20261117) as a YOLO dataset:
data/Vehicle363_vn_pool_20261118 (classes car, motorcycle, bus, truck, same order as Vehicle363).

Kept boxes = every vehicle box not marked wrong (untouched boxes count as correct, as in the QA app).
Split by source video: VAL_VIDEOS (a fixed-camera intersection, close to the test news shots) is val, the rest train.
16-seat minibuses/vans are truck (owner rule 17/11). Never contains test sources (iHanoi, news report).
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/vn_pool_qa_vehicle_20261117"
OUT = ROOT / "data/Vehicle363_vn_pool_20261118"
NAMES = ["car", "motorcycle", "bus", "truck"]
VAL_VIDEOS = {"Kv2Z_NHz9KE"}


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists; never overwrite a dataset version")
    qa = json.loads((QA / "qa_result.json").read_text(encoding="utf-8"))
    stats, manifest = Counter(), []
    for im in qa["images"]:
        if not im.get("reviewed"):
            raise RuntimeError(f"not reviewed: {im['path']}")
        src = ROOT / im["path"].replace("\\", "/")
        video = src.stem.rsplit("_", 1)[0]
        split = "val" if video in VAL_VIDEOS else "train"
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        lines = []
        for b in im["boxes"]:
            if b["group"] != "vehicle" or b["status"] == "wrong":
                continue
            x1, y1, x2, y2 = max(0, b["xyxy"][0]), max(0, b["xyxy"][1]), min(w, b["xyxy"][2]), min(h, b["xyxy"][3])
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{b['cls']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            stats[f"{split}_{NAMES[b['cls']]}"] += 1
        stem = f"vnpool__{src.stem}"
        for sub in ("images", "labels"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, OUT / "images" / split / f"{stem}{src.suffix}")
        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[f"{split}_images"] += 1
        manifest.append({"image": stem, "video": video, "split": split, "boxes": len(lines)})
    assert stats["val_images"] and stats["train_images"], stats
    (OUT / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(NAMES)), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"source": str(QA.relative_to(ROOT)), "val_videos": sorted(VAL_VIDEOS), "stats": dict(sorted(stats.items())),
                                                  "rules": "boxes not marked wrong; 16-seat/van = truck; never test sources"}, indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
