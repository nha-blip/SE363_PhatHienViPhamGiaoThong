"""Re-run another light checkpoint on the saved light-phase evaluation frames (same frames, same settings as model A).

Usage: redetect_light_phase_eval.py <weights.pt> <output root> [imgsz, default 960]
Writes <output root>/<source>/detections.json for every source in data/light_phase_eval_20261114 that has frames,
then score with: evaluate_light_phase_rois.py <output root>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/light_phase_eval_20261114"
NAMES = ["red", "yellow", "green", "off"]


def main():
    from ultralytics import YOLO
    weights, out = Path(sys.argv[1]), Path(sys.argv[2])
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 960
    model = YOLO(str(weights))
    for src in sorted(p for p in WORK.iterdir() if (p / "detections.json").exists()):
        target = out / src.name / "detections.json"
        if target.exists():
            continue
        old = json.loads((src / "detections.json").read_text(encoding="utf-8"))
        rows = []
        for row in old["frames"]:
            new = dict(row, detections=[])
            if row["scene"]:
                r = model.predict(str(src / "frames" / f"{row['index']:05d}.jpg"), imgsz=imgsz, conf=0.1, verbose=False)[0]
                new["detections"] = [{"xyxy": [round(v, 1) for v in b], "cls": NAMES[int(c)], "conf": round(float(s), 4)}
                                     for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            rows.append(new)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dict(old, weights=str(weights), imgsz=imgsz, frames=rows)), encoding="utf-8")
        print(src.name, len(rows), "samples", sum(len(r["detections"]) for r in rows), "detections (A:",
              sum(len(r["detections"]) for r in old["frames"]), ")", flush=True)


if __name__ == "__main__":
    main()
