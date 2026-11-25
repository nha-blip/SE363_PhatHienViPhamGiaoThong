"""Minibus addition for the 6-class vehicle detector: data/Vehicle363_minibus_add_20261122 (train only).

Classes (6-class taxonomy, owner 21/11/2026; ids 0-3 unchanged from Vehicle363): 0 car, 1 motorcycle, 2 bus, 3 truck,
4 pickup, 5 minibus. Images: the 355 frames whose every van/MPV box was pre-sorted 'minibus' by v8s and then judged by the
owner on the Minibus QA page (data/minibus_candidates_20261121/qa_items_A_labeled.json): minibus -> 5, car -> 0, drop ->
blurred. Every other object is mapped so no target is left unlabelled:
  UA-DETRAC (Beijing/Tianjin CCTV): car->car, bus->bus, others->truck. Motorcycles/e-bikes are NOT labelled in UA-DETRAC,
    so v8s motorcycle detections (conf >= 0.3) are added as labels and weaker ones (0.1-0.3) are blurred.
  HeteroTraffic (Bangladesh, CC BY 4.0): Car->car, Motorbike->motorcycle, Bus->bus, Truck->truck; MPV boxes not on the QA
    page (too close) are blurred; motorised look-alikes outside the 6 classes (CNG, Easybike, Leguna, Bhotbhoti,
    PowerTiller, ShoppingVan) and the ambiguous 'Pickup' (mostly small trucks) are blurred; bicycles, pedestrians,
    rickshaws, cycle-vans and wheelbarrows stay unlabelled background (as in Vehicle363).
Both sources also miss vehicles (UA-DETRAC ignore regions, HeteroTraffic truncated/close vehicles): any v8s detection
(conf >= 0.25) that matches no kept or blurred box is blurred.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/external_vehicle_sources_20261121"
QA = ROOT / "data/minibus_candidates_20261121/qa_items_A_labeled.json"
OUT = ROOT / "data/Vehicle363_minibus_add_20261122"
NAMES = ["car", "motorcycle", "bus", "truck", "pickup", "minibus"]
VERDICT = {"minibus": 5, "car": 0}
DETRAC_MAP = {0: 0, 1: 2, 3: 3}  # van (2) comes from the QA verdicts
HETERO = ['Bicycle', 'Bus', 'Bhotbhoti', 'Car', 'CNG', 'Easybike', 'Leguna', 'Motorbike', 'MPV', 'Pedestrian', 'Pickup',
          'PowerTiller', 'Rickshaw', 'ShoppingVan', 'Truck', 'Van', 'Wheelbarrow']
HETERO_MAP = {"Car": 0, "Motorbike": 1, "Bus": 2, "Truck": 3}
HETERO_BLUR = {"CNG", "Easybike", "Leguna", "Bhotbhoti", "PowerTiller", "ShoppingVan", "Pickup"}


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def blur(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(max(0, box[0] - 2)), int(max(0, box[1] - 2)), int(min(w, box[2] + 2)), int(min(h, box[3] + 2))
    if x2 - x1 > 1 and y2 - y1 > 1:
        img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (0, 0), max(3.0, (y2 - y1) / 4))


def yolo_boxes(label, W, H):
    for line in label.read_text().splitlines():
        p = line.split()
        if p:
            c, x, y, w, h = int(p[0]), *map(float, p[1:5])
            yield c, [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]


def main():
    from ultralytics import YOLO
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists; never overwrite a dataset version")
    (OUT / "images/train").mkdir(parents=True); (OUT / "labels/train").mkdir(parents=True)
    qa = json.loads(QA.read_text())
    by_image = defaultdict(list)
    for it in qa:
        by_image[(it["src"], it["image"])].append(it)
    v8s = YOLO(str(ROOT / "weights/kaggle_small_vehicle_20261019/best.pt"))
    stats, manifest = Counter(), []
    for (src, image), its in sorted(by_image.items()):
        img = cv2.imread(str(SRC / image)); H, W = img.shape[:2]
        rel = Path(image)
        label_path = SRC / (str(rel.parent).replace("images", "labels")) / (rel.stem + ".txt")
        out, blurred = [], []
        judged = [(it["box"], it["label"]) for it in its]
        target_cls = 2 if src == "uadetrac" else HETERO.index("MPV")
        for c, box in yolo_boxes(label_path, W, H):
            if c == target_cls:
                match = max(judged, key=lambda j: iou(j[0], box))
                verdict = match[1] if iou(match[0], box) > 0.9 else None
                if verdict in VERDICT:
                    out.append((VERDICT[verdict], box))
                else:
                    blur(img, box); blurred.append(box); stats[f"{src}_blur_van_{verdict or 'unjudged'}"] += 1
            elif src == "uadetrac":
                out.append((DETRAC_MAP[c], box))
            else:
                name = HETERO[c]
                if name in HETERO_MAP:
                    out.append((HETERO_MAP[name], box))
                elif name in HETERO_BLUR:
                    blur(img, box); blurred.append(box); stats[f"heterotraffic_blur_{name}"] += 1
        # Neither source labels everything: UA-DETRAC skips two-wheelers and its ignore regions, HeteroTraffic skips
        # truncated / very close vehicles. A v8s detection matching no label is labelled (UA-DETRAC motorcycles) or blurred.
        r = v8s.predict(cv2.imread(str(SRC / image)), imgsz=960, conf=0.1, verbose=False)[0]
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if any(iou(b, o[1]) >= 0.3 for o in out) or any(iou(b, z) >= 0.3 for z in blurred):
                continue
            if src == "uadetrac" and int(c) == 1 and s >= 0.3:
                out.append((1, b)); stats["uadetrac_motorcycle_pseudo"] += 1
            elif s >= 0.25 or (src == "uadetrac" and int(c) == 1):
                blur(img, b); blurred.append(b); stats[f"{src}_blur_unlabelled_{['car', 'motorcycle', 'bus', 'truck'][int(c)]}"] += 1
        stem = f"{src}__{rel.stem.replace(' ', '_').replace('(', '').replace(')', '')}"
        cv2.imwrite(str(OUT / "images/train" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for c, (x1, y1, x2, y2) in out:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                stats[f"{src}_{NAMES[c]}"] += 1
        (OUT / "labels/train" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[f"{src}_images"] += 1
        manifest.append({"image": stem, "source": src, "source_image": image})
    (OUT / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/train\nnames:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(NAMES)), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"classes": NAMES, "stats": dict(sorted(stats.items())),
                                                  "licences": {"heterotraffic": "CC BY 4.0", "uadetrac": "Zenodo re-upload says CC BY 4.0; original UA-DETRAC terms not verified"}}, indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1) < 1e-9 and iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0
    main()
