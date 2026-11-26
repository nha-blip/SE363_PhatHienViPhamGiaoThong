"""Per-class recall as defined in spec v1.1 §19.2, for the 6-class vehicle taxonomy.

recall(c) = GT boxes of class c matched (IoU >= 0.5, greedy one-to-one, class-agnostic) by a prediction OF CLASS c, over
all GT boxes of c; missed and wrongly-classed boxes both fail. Also reports localisation recall (any class), precision per
class (predictions of c matched to a GT of c / all predictions of c) and the confusion GT class -> predicted class|missed.
Predictions are compared by class NAME, so a 4-class checkpoint can be scored on 6-class labels (it can never be right on
pickup/minibus); --collapse4 maps pickup/minibus -> truck on both sides for a like-for-like 4-class comparison.
Usage: eval_vehicle_6class_20261126.py --weights W --images DIR --labels DIR [--imgsz 960] [--conf 0.25] --out result.json
  Vehicle363 test: --images data/YOLO363_training_working/Vehicle363-v1/images/test
                   --labels data/Vehicle363_6class_labels_fullframe_20261125/test
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_vehicle_val_errors import iou_matrix, match, read_labels  # noqa: E402  (IoU 0.5 greedy matching)

NAMES = ["car", "motorcycle", "bus", "truck", "pickup", "minibus"]
COLLAPSE4 = {"pickup": "truck", "minibus": "truck"}


def score(pairs, collapse=False):
    """pairs: iterable of (gt_names, gt_boxes, pred_names, pred_boxes) per image -> metrics dict."""
    fold = (lambda n: COLLAPSE4.get(n, n)) if collapse else (lambda n: n)
    gt_n, hit, loc, pred_n, pred_ok, confusion = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    for g_names, g_boxes, p_names, p_boxes in pairs:
        g_names, p_names = [fold(n) for n in g_names], [fold(n) for n in p_names]
        matched = dict(match(iou_matrix(g_boxes, p_boxes)))
        gt_n.update(g_names); pred_n.update(p_names)
        for gi, name in enumerate(g_names):
            if gi in matched:
                pn = p_names[matched[gi]]
                loc[name] += 1; confusion[(name, pn)] += 1
                if pn == name:
                    hit[name] += 1; pred_ok[name] += 1
            else:
                confusion[(name, "missed")] += 1
    classes = sorted(set(gt_n) | set(pred_n), key=lambda n: (NAMES.index(n) if n in NAMES else 99, n))
    per_class = {c: {"gt": gt_n[c], "recall": hit[c] / gt_n[c] if gt_n[c] else None, "localised": loc[c] / gt_n[c] if gt_n[c] else None,
                     "predictions": pred_n[c], "precision": pred_ok[c] / pred_n[c] if pred_n[c] else None} for c in classes}
    return {"per_class": per_class, "confusion": {f"{a}->{b}": v for (a, b), v in sorted(confusion.items())},
            "meets_95": {c: v["recall"] is not None and v["recall"] >= 0.95 for c, v in per_class.items() if v["gt"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True); parser.add_argument("--images", required=True); parser.add_argument("--labels", required=True)
    parser.add_argument("--imgsz", type=int, default=960); parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", required=True); parser.add_argument("--device", default="0")
    a = parser.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    from ultralytics import YOLO
    model = YOLO(a.weights)
    rows = []
    # a directory source is read lazily; a list of paths makes ultralytics open every image up front (MemoryError on 21k)
    for r in model.predict(str(Path(a.images)), imgsz=a.imgsz, conf=a.conf, device=a.device, stream=True, verbose=False, batch=8):
        h, w = r.orig_shape
        g_cls, g_boxes = read_labels(Path(a.labels) / f"{Path(r.path).stem}.txt", w, h)
        rows.append(([NAMES[c] for c in g_cls], g_boxes, [model.names[int(c)] for c in r.boxes.cls.tolist()], r.boxes.xyxy.cpu().numpy().reshape(-1, 4)))
    result = {"weights": a.weights, "images": a.images, "labels": a.labels, "n_images": len(rows), "imgsz": a.imgsz, "conf": a.conf,
              "six_class": score(rows), "collapsed_4class": score(rows, collapse=True)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for key in ("six_class", "collapsed_4class"):
        print(key, {c: (v["gt"], None if v["recall"] is None else round(v["recall"], 3)) for c, v in result[key]["per_class"].items()})


if __name__ == "__main__":
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], float)
    s = score([(["car", "pickup", "truck"], b, ["car", "truck", "truck"], b[:3])])
    assert s["per_class"]["car"]["recall"] == 1 and s["per_class"]["pickup"]["recall"] == 0 and s["confusion"]["pickup->truck"] == 1
    assert s["per_class"]["truck"]["precision"] == 0.5
    s4 = score([(["car", "pickup", "truck"], b, ["car", "truck", "truck"], b[:3])], collapse=True)
    assert s4["per_class"]["truck"]["recall"] == 1 and "pickup" not in s4["per_class"]
    s0 = score([(["car"], b[:1], [], np.zeros((0, 4)))])
    assert s0["per_class"]["car"]["recall"] == 0 and s0["confusion"]["car->missed"] == 1
    main()
