"""Crops for the per-ROI signal-colour classifier (classes red, yellow, green, off, none), from Light363_merged_20261117.

At run time the pipeline crops each configured signal ROI (grown 20%) and classifies it, so training crops imitate a
hand-drawn ROI: the lamp box expanded 1.3-2.2x with a random shift. 'none' = no round lamp in the crop: boxes the owner
marked wrong in the VN QA (signs, logos, tail lights), arrow/countdown lamps (decision (a): round lamps only), and
random background patches of lamp size. Yellow is rare, so it gets more jittered copies. Val crops are deterministic.
Splits follow the merged dataset (VN val = held-out source videos). Output: data/LightCrops363_20261117/<split>/<class>/.
"""
import json
import random
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/Light363_merged_20261117"
QA = ["vn_pool_qa_light_driving_20261117", "vn_pool_qa_light_20261117", "vn_pool_qa_light_orange_20261117"]
OUT = ROOT / "data/LightCrops363_20261117"
NAMES = ["red", "yellow", "green", "off"]
COPIES = {"red": 1, "green": 1, "off": 2, "yellow": 6}  # per lamp in train; VN lamps x3 on top
rng = random.Random(363)


def crop(img, box, train):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    e = rng.uniform(1.3, 2.2) if train else 1.6
    dx, dy = (rng.uniform(-0.15, 0.15) * bw, rng.uniform(-0.15, 0.15) * bh) if train else (0.0, 0.0)
    cx, cy, hw, hh = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, e * bw / 2 + 2, e * bh / 2 + 2
    a, b, c, d = int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(w, cx + hw)), int(min(h, cy + hh))
    return img[b:d, a:c] if c - a >= 3 and d - b >= 3 else None


def save(patch, split, cls, stem, k, stats):
    folder = OUT / split / cls
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{stem}_{k}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
    stats[f"{split}/{cls}"] += 1


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists; never overwrite a dataset version")
    stats = Counter()
    vn_val = {Path(l).stem for l in (SRC / "val_vn.txt").read_text().split()}
    for split in ("train", "val"):
        for label in sorted((SRC / "labels" / split).iterdir()):
            found = list((SRC / "images" / split).glob(f"{label.stem}.*"))  # A's val keeps some .png
            img = cv2.imread(str(found[0])) if len(found) == 1 else None
            if img is None:
                raise FileNotFoundError(label.stem)
            h, w = img.shape[:2]
            vn = label.stem.startswith("vnpool__")
            boxes = []
            for k, line in enumerate(l for l in label.read_text().splitlines() if l.strip()):
                c, x, y, bw, bh = map(float, line.split())
                box = ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
                boxes.append(box)
                cls = NAMES[int(c)]
                n = (COPIES[cls] * (3 if vn else 1)) if split == "train" else 1
                for j in range(n):
                    p = crop(img, box, split == "train")
                    if p is not None:
                        save(p, split, cls, f"{label.stem}_{k}", j, stats)
            # background 'none' patch of lamp size, away from every lamp (one per A image, three per VN image)
            if boxes:
                for j in range(3 if vn else 1):
                    bw = max(6.0, rng.choice([b[2] - b[0] for b in boxes])); bh = bw * rng.uniform(1.0, 3.0)
                    for _ in range(20):
                        x1, y1 = rng.uniform(0, max(1, w - bw)), rng.uniform(0, max(1, h - bh))
                        cand = (x1, y1, x1 + bw, y1 + bh)
                        if all(cand[2] < b[0] - bw or cand[0] > b[2] + bw or cand[3] < b[1] - bh or cand[1] > b[3] + bh for b in boxes):
                            p = crop(img, cand, split == "train")
                            if p is not None:
                                save(p, split, "none", f"{label.stem}_bg", j, stats)
                            break
    # VN QA negatives: boxes marked wrong, arrows, countdowns, motorcycle lamps (split = the image's split)
    for s in QA:
        for im in json.loads((ROOT / "data" / s / "qa_result.json").read_text(encoding="utf-8"))["images"]:
            if not im.get("reviewed"):
                continue
            stem = f"vnpool__{Path(im['path']).stem}"
            split = "val" if stem in vn_val else "train"
            img = None
            for k, b in enumerate(im["boxes"]):
                if b["group"] != "light" or not (b["status"] == "wrong" or b.get("shape") in ("arrow", "countdown", "other")):
                    continue
                img = cv2.imread(str(ROOT / im["path"])) if img is None else img
                for j in range(2 if split == "train" else 1):
                    p = crop(img, b["xyxy"], split == "train")
                    if p is not None:
                        save(p, split, "none", f"{stem}_neg{k}", j, stats)
    (OUT / "summary.json").write_text(json.dumps({"source": str(SRC.relative_to(ROOT)), "classes": NAMES + ["none"], "copies": COPIES,
                                                  "stats": dict(sorted(stats.items()))}, indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
