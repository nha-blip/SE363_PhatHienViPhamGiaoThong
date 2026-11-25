"""Pre-sort external 'van'/'MPV' boxes into minibus (box-shaped) vs car (car-like) with the current vehicle model (v8s @960).

Why v8s works as a sorter: Vehicle363 maps MIO work_van -> truck and cars -> car, so v8s calls box-shaped vans truck/bus and
car-like MPVs car. Owner rules 21/11/2026: box-shaped (Hiace-like, micro-vans, cargo vans) -> minibus; MPV/minivan/7-seat -> car.

Sources (local, from data/external_vehicle_sources_20261121):
  UA-DETRAC  frames listed in uadetrac/van_frames.txt (every 50th frame with a van), class id 2 = van
  HeteroTraffic roadvision_blur images, class id 8 = MPV; portrait photos and boxes taller than 40% of the image skipped
Each candidate: matched v8s prediction (IoU >= 0.5), suggestion minibus | car | unsure, and auto = confident (conf >= 0.5,
IoU >= 0.6). Writes <out>/candidates.json and a 3x-context crop per candidate in <out>/crops for owner QA.
Usage: minibus_candidates_20261121.py <out_dir>
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/external_vehicle_sources_20261121"
WEIGHTS = ROOT / "weights/kaggle_small_vehicle_20261019/best.pt"
NAMES = ["car", "motorcycle", "bus", "truck"]


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def suggest(pred_cls, conf, match_iou):
    if pred_cls is None:
        return "unsure", False
    label = "car" if pred_cls == "car" else "minibus" if pred_cls in ("truck", "bus") else "unsure"
    return label, label != "unsure" and conf >= 0.5 and match_iou >= 0.6


def items():
    for rel in (SRC / "uadetrac/van_frames.txt").read_text().split():
        split, stem = rel.split("/")
        yield "uadetrac", SRC / f"uadetrac/UA-DETRAC/images/{split}/{stem}.jpg", SRC / f"uadetrac/UA-DETRAC/labels/{split}/{stem}.txt", 2
    hroot = SRC / "heterotraffic/roadvision_blur"
    for lab in sorted((hroot / "labels").glob("*.txt")):
        if any(l.split() and l.split()[0] == "8" for l in lab.read_text().splitlines()):
            found = list((hroot / "images").glob(lab.stem + ".*"))
            if found:
                yield "heterotraffic", found[0], lab, 8


def main():
    from ultralytics import YOLO
    out = Path(sys.argv[1]); (out / "crops").mkdir(parents=True, exist_ok=True)
    model, rows, skipped = YOLO(str(WEIGHTS)), [], {"portrait": 0, "too_close": 0, "missing_image": 0}
    for source, img_path, lab_path, cls in items():
        img = cv2.imread(str(img_path))
        if img is None:
            skipped["missing_image"] += 1
            continue
        H, W = img.shape[:2]
        if source == "heterotraffic" and H > W:
            skipped["portrait"] += 1
            continue
        boxes = []
        for line in lab_path.read_text().splitlines():
            p = line.split()
            if p and int(p[0]) == cls:
                x, y, w, h = map(float, p[1:5])
                if source == "heterotraffic" and h > 0.4:
                    skipped["too_close"] += 1
                    continue
                boxes.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
        if not boxes:
            continue
        r = model.predict(img, imgsz=960, conf=0.1, verbose=False)[0]
        preds = [(b, NAMES[int(c)], float(s)) for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        for k, box in enumerate(boxes):
            best = max(((iou(box, b), c, s) for b, c, s in preds), default=(0.0, None, 0.0))
            pred_cls, conf = (best[1], best[2]) if best[0] >= 0.5 else (None, 0.0)
            label, auto = suggest(pred_cls, conf, best[0])
            name = f"{len(rows):05d}_{source}_{img_path.stem.replace(' ', '_')}_{k}.jpg"
            x1, y1, x2, y2 = box
            cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.9
            crop = img[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
            cv2.imwrite(str(out / "crops" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append({"crop": name, "source": source, "image": str(img_path.relative_to(SRC)), "box": [round(v, 1) for v in box],
                         "v8s": pred_cls, "conf": round(conf, 3), "iou": round(best[0], 3), "suggest": label, "auto": auto})
    (out / "candidates.json").write_text(json.dumps({"skipped": skipped, "rows": rows}, indent=1), encoding="utf-8")
    from collections import Counter
    print(json.dumps({"skipped": skipped, "by": {f"{s}|{g}|{'auto' if a else 'qa'}": n for (s, g, a), n in
                                                  sorted(Counter((r["source"], r["suggest"], r["auto"]) for r in rows).items())}}, indent=1))


if __name__ == "__main__":
    assert suggest("truck", 0.8, 0.7) == ("minibus", True) and suggest("car", 0.3, 0.9) == ("car", False) and suggest(None, 0, 0) == ("unsure", False)
    main()
