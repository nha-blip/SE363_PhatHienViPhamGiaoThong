"""Signal colour per configured ROI for the light-phase evaluation, written as detections.json so that
evaluate_light_phase_rois.py scores every variant with the same truth and voting.

Usage: build_roi_classifier_detections.py <classifier.pt> <variant> <output root> [min_conf]
  cls   classifier on the ROI crop (ROI grown 20%, as RoiHeadAssociator)
  k1    model A's most confident box inside the ROI is re-read by the classifier (crop 1.6x the box); no A box -> ROI crop
  k1b   like k1, but when the A-box read is 'none' (e.g. A boxed an arrow lamp) the whole ROI crop is read instead
  k2    A and the classifier on the ROI crop must agree, otherwise no observation for that frame
  pos   lamp-position reading only (src/signal/lamp_position.py), on ROIs with a known 3-lamp layout; nothing elsewhere
  k1bpos  lamp-position reading where the layout is known, k1b where it is not or the position read gives nothing
'none' or a classifier confidence below min_conf (default 0.5) is no observation. A's detections come from the saved
data/light_phase_eval_20261114/<source>/detections.json (A @960).
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/light_phase_eval_20261114"
LAYOUTS = WORK / "roi_lamp_layouts_20261118.json"
COLOURS = {"red", "yellow", "green", "off"}


def grow(box, g):
    x1, y1, x2, y2 = box
    return [x1 - g * (x2 - x1), y1 - g * (y2 - y1), x2 + g * (x2 - x1), y2 + g * (y2 - y1)]


def cut(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2] if x2 - x1 >= 3 and y2 - y1 >= 3 else None


def classify(model, patch, min_conf):
    if patch is None:
        return None
    p = model.predict(patch, imgsz=64, verbose=False)[0].probs
    name, conf = model.names[p.top1], float(p.top1conf)
    return (name, conf) if name in COLOURS and conf >= min_conf else None


def a_in_roi(dets, roi):
    x1, y1, x2, y2 = grow(roi, 0.2)
    inside = [d for d in dets if d["conf"] >= 0.1 and x1 <= (d["xyxy"][0] + d["xyxy"][2]) / 2 <= x2 and y1 <= (d["xyxy"][1] + d["xyxy"][3]) / 2 <= y2]
    return max(inside, key=lambda d: d["conf"]) if inside else None


def main():
    from ultralytics import YOLO
    weights, variant, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    min_conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    if variant not in ("cls", "k1", "k1b", "k2", "pos", "k1bpos"):
        raise SystemExit("variant must be cls, k1, k1b, k2, pos or k1bpos")
    sys.path.insert(0, str(ROOT))
    from src.signal.lamp_position import read_by_position
    layouts = json.loads(LAYOUTS.read_text(encoding="utf-8"))["layouts"]
    model = YOLO(str(weights))
    rois = json.loads((WORK / "signal_head_rois.json").read_text(encoding="utf-8"))
    for source, source_rois in sorted(rois.items()):
        if not (WORK / source / "detections.json").exists():
            continue  # sources with no ROI still get a file (no observations): the scorer reads every source
        old = json.loads((WORK / source / "detections.json").read_text(encoding="utf-8"))
        rows = []
        for row in old["frames"]:
            new = dict(row, detections=[])
            if row["scene"]:
                img = cv2.imread(str(WORK / source / "frames" / f"{row['index']:05d}.jpg"))
                for r in source_rois:
                    roi, a = r["xyxy"], a_in_roi(row["detections"], r["xyxy"])
                    layout = layouts.get(f"{source}#{r['id']}")
                    by_pos = read_by_position(cut(img, roi), layout) if layout and variant in ("pos", "k1bpos") else None
                    if variant == "pos":
                        got = by_pos
                    elif variant == "k1bpos":
                        got = by_pos or (classify(model, cut(img, grow(a["xyxy"], 0.3)), min_conf) if a else None) or classify(model, cut(img, grow(roi, 0.2)), min_conf)
                    elif variant == "cls":
                        got = classify(model, cut(img, grow(roi, 0.2)), min_conf)
                    elif variant == "k1":
                        got = classify(model, cut(img, grow(a["xyxy"], 0.3)) if a else cut(img, grow(roi, 0.2)), min_conf)
                    elif variant == "k1b":
                        got = (classify(model, cut(img, grow(a["xyxy"], 0.3)), min_conf) if a else None) or classify(model, cut(img, grow(roi, 0.2)), min_conf)
                    else:
                        c = classify(model, cut(img, grow(roi, 0.2)), min_conf)
                        got = c if c and a and a["cls"] == c[0] else None
                    if got:  # one synthetic detection at the ROI, picked up by RoiHeadAssociator
                        new["detections"].append({"xyxy": [round(v, 1) for v in roi], "cls": got[0], "conf": round(got[1], 4)})
            rows.append(new)
        target = out / source / "detections.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dict(old, weights=str(weights), variant=variant, min_conf=min_conf, frames=rows)), encoding="utf-8")
        print(source, variant, sum(len(r["detections"]) for r in rows), "ROI observations", flush=True)


if __name__ == "__main__":
    main()
