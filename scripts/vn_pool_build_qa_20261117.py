"""Pick training-candidate frames from the VN YouTube pool and pre-label them for external_qa_app_20261112.py.

light mode: model A (conf 0.15) every 0.5 s. Frames with a light (conf >= 0.3) are kept one per 5 s; model A misses
the orange VN yellow, so frames 1, 2 and 3 s after a green head disappears are kept too (likely yellow phase).
vehicle mode: one frame every --every seconds, at most --max per video, pre-labelled by the vehicle model.
Output: <out>/images/<video>_<ms>.jpg, prelabels.json, meta.json (sources, never the test news report).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/vn_youtube_train_pool_20261116/raw"
LIGHT = ROOT / "weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt"
VEHICLE = ROOT / "weights/kaggle_small_vehicle_20261019/best.pt"


def boxes(result, conf):
    return [(int(c), float(s), [round(v, 1) for v in b]) for b, c, s in
            zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()) if s >= conf]


def frames(video, step_s):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step, k = max(1, round(fps * step_s)), 0
    while True:
        ok = cap.grab()  # sequential grab: much faster than seeking
        if not ok:
            break
        if k % step == 0:
            ok, img = cap.retrieve()
            if ok:
                yield k / fps, img
        k += 1
    cap.release()


def near(a, b):
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a, b
    size = max(ax2 - ax1, ay2 - ay1)
    return abs((ax1 + ax2) - (bx1 + bx2)) / 2 <= 2 * size and abs((ay1 + ay2) - (by1 + by2)) / 2 <= 2 * size


def orange(img, box):
    """Lit lamp colour inside a detected head: orange/yellow hue of the bright pixels (VN yellow is often read as red)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    hsv = cv2.cvtColor(img[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2HSV) if x2 > x1 and y2 > y1 else None
    if hsv is None or hsv.size == 0:
        return False
    lit = hsv[(hsv[..., 2] > 180) & (hsv[..., 1] > 90)]
    return len(lit) >= 3 and 8 <= float(np.median(lit[:, 0])) <= 32


def pick_light(model, video, cap, moving=False):
    seen, orange_t = {}, set()
    for t, img in frames(video, 0.5):
        seen[t] = boxes(model.predict(img, imgsz=960, conf=0.15, verbose=False)[0], 0.15)
        if any(orange(img, b) for c, s, b in seen[t] if c in (0, 1)):
            orange_t.add(t)
    times = sorted(seen)
    why, last = {}, -99.0
    for t in sorted(orange_t):
        why[t] = "orange_lamp"
    for t in times:
        if t not in why and any(s >= 0.3 for _, s, _ in seen[t]) and t - last >= 5:
            why[t], last = "light", t
    greens = {} if moving else {t: [b for c, s, b in seen[t] if c == 2 and s >= 0.3] for t in times}  # a moving camera loses heads by itself
    for i, t in enumerate(times if greens else []):
        for g in greens[t]:
            later = times[i + 1:i + 4]  # next 1.5 s
            if later and not any(near(g, h) for u in later for h in greens[u]):
                for dt in (1.0, 2.0, 3.0):
                    u = min(times, key=lambda x: abs(x - (t + dt)))
                    if abs(u - (t + dt)) < 0.3:
                        why.setdefault(u, "after_green")
    for kind in ("light", "after_green", "orange_lamp"):  # fixed cameras repeat the same view: spread at most `cap` picks evenly
        ts = [t for t in times if why.get(t) == kind]
        for j, t in enumerate(ts):
            if len(ts) > cap and j not in {round(i * (len(ts) - 1) / (cap - 1)) for i in range(cap)}:
                del why[t]
    # second pass: decode only the chosen frames (keeping every frame in memory would need ~20 GB for an hour)
    return {t: (why[t], img, seen[t]) for t, img in frames(video, 0.5) if t in why}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["light", "vehicle"], required=True)
    p.add_argument("--videos", nargs="+", required=True, help="video ids in the pool raw folder")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every", type=float, default=10.0)
    p.add_argument("--max", type=int, default=40)
    p.add_argument("--moving", action="store_true", help="moving camera (driving videos): no after-green picks")
    a = p.parse_args()
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    pre_path = a.out / "prelabels.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8")) if pre_path.exists() else {}
    model = YOLO(str(LIGHT if a.mode == "light" else VEHICLE))
    group = a.mode
    stats = {}
    for vid in a.videos:
        video = RAW / f"{vid}.mp4"
        if not video.exists():
            raise FileNotFoundError(video)
        if a.mode == "light":
            picked = pick_light(model, video, a.max, a.moving)
        else:
            picked = {}
            for t, img in frames(video, a.every):
                if len(picked) >= a.max:
                    break
                picked[t] = ("vehicle", img, boxes(model.predict(img, imgsz=960, conf=0.25, verbose=False)[0], 0.25))
        for t, (why, img, dets) in sorted(picked.items()):
            path = a.out / "images" / f"{vid}_{int(t * 1000):08d}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            pre[str(path)] = [{"group": group, "xyxy": b, "cls": c, "status": "pending", "from": f"{a.mode}_model:{why}"}
                              for c, s, b in dets if s >= 0.25]
        stats[vid] = {k: sum(1 for w, _, _ in picked.values() if w == k) for k in {w for w, _, _ in picked.values()}}
        print(vid, stats[vid], flush=True)
    pre_path.write_text(json.dumps(pre, indent=1), encoding="utf-8")
    meta_path = a.out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"source": "data/vn_youtube_train_pool_20261116 (see LABELING_RULES.md)", "never_in_test": True, "videos": {}}
    meta["videos"].update(stats)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
