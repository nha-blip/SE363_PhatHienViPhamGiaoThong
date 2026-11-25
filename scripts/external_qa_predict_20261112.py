"""Run the YOLO models on an unseen source (Bellevue intersections) and build QA pre-labels.

Pre-labels are the union of all models (conf >= 0.25, cross-model NMS at IoU 0.5), so the QA'd
result is not biased toward one model. Raw predictions (conf >= 0.05) are kept for scoring later.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "data/Tracking363_working/sequences"
OUT = ROOT / "data/external_qa_bellevue_20261112"
STRIDE = 15
MODELS = {
    "vehicle": {
        "e63": ROOT / "runs/project363/yolo26_vehicle_local_resume_20261104/final_best_copy.pt",
        "v8s": ROOT / "weights/kaggle_small_vehicle_20261019/best.pt",
    },
    "light": {
        "A": ROOT / "weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt",
        "ft960": ROOT / "runs/project363/yolo26_light_finetune960_local_20261103/train/weights/best.pt",
        "v8s": ROOT / "weights/trained_yolo_20261018/small/traffic_light/best.pt",
    },
}
NAMES = {"vehicle": ["car", "motorcycle", "bus", "truck"], "light": ["red", "yellow", "green", "off"]}


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def main():
    global OUT
    images = []
    if len(sys.argv) > 1:  # usage: script.py <work_dir>  -> uses <work_dir>/images/*.jpg
        OUT = Path(sys.argv[1]).resolve()
        images = [str(p) for p in sorted((OUT / "images").glob("*.jpg"))]
    else:
        for seq in ["bellevue_ne12th_001", "bellevue_newport_001"]:
            frames = sorted((SEQ / seq / "img1").glob("*.jpg"))
            images += [str(p) for p in frames[STRIDE // 2::STRIDE]]
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {img: {"vehicle": {}, "light": {}} for img in images}
    for group, models in MODELS.items():
        for name, weights in models.items():
            model = YOLO(str(weights))
            for img in images:
                r = model.predict(img, imgsz=960, conf=0.05, iou=0.7, verbose=False, half=True)[0]
                boxes = r.boxes
                raw[img][group][name] = [
                    {"xyxy": [round(float(v), 1) for v in b], "cls": int(c), "conf": round(float(s), 4)}
                    for b, c, s in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())]
            print(group, name, "done", flush=True)
            del model
            torch.cuda.empty_cache()
    prelabels = {}
    for img in images:
        items = []
        for group in MODELS:
            cands = sorted([dict(p, model=m) for m, ps in raw[img][group].items() for p in ps if p["conf"] >= 0.25],
                           key=lambda p: -p["conf"])
            kept = []
            for p in cands:
                if all(iou(p["xyxy"], k["xyxy"]) < 0.5 for k in kept):
                    kept.append(p)
            items += [{"group": group, "xyxy": k["xyxy"], "cls": k["cls"], "status": "pending", "from": k["model"]} for k in kept]
        prelabels[img] = items
    meta = {"source": "see extract_log.json" if len(sys.argv) > 1 else
            "Bellevue intersection cameras (Tracking363 sequences bellevue_ne12th_001, bellevue_newport_001)",
            "never_in_training": True, "stride": None if len(sys.argv) > 1 else STRIDE, "imgsz": 960, "names": NAMES,
            "models": {g: {m: str(w) for m, w in ms.items()} for g, ms in MODELS.items()},
            "prelabel_policy": "union of all models conf>=0.25, cross-model NMS IoU 0.5"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "predictions_raw.json").write_text(json.dumps(raw), encoding="utf-8")
    (OUT / "prelabels.json").write_text(json.dumps(prelabels, indent=1), encoding="utf-8")
    print("images", len(images), "prelabel boxes", sum(len(v) for v in prelabels.values()))


if __name__ == "__main__":
    main()
