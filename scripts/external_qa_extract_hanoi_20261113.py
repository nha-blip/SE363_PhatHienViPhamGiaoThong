"""Build the Vietnam external test set (no new recording; only existing local footage).

Sources:
  * iHanoi phone screen recordings: only frames where the camera video is actually shown
    (scan_video_frames.json, found by external scan: scene band 450-760 px tall, <15% white, not flat).
    Up to PER_CAM frames per camera, spread evenly over all its good frames across sessions.
  * vntraffic_001 (Tracking363 sequence): frames outside 201-300 (already QA'd for tracking).
  * Local VN traffic TV news video (AQO...mp4): a few frames, overlays included.
"""
import json
import re
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external_qa_hanoi_20261113"
SRC = ROOT / "data/iHanoi_capture/city_survey_20261029"
PER_CAM = 6


def scene_band(frame):
    h = frame.shape[0]
    rows = frame.mean(axis=(1, 2))
    top, bottom = int(h * 0.11), int(h * 0.88)  # skip app header / LIVE bar
    best, start = (0, 0), None
    for y in range(top, bottom + 1):
        bright = y < bottom and rows[y] > 12
        if bright and start is None:
            start = y
        elif not bright and start is not None:
            if y - start > best[1] - best[0]:
                best = (start, y)
            start = None
    return best


def camera_of(key):
    m = re.search(r"C(\d{3})[_.](\d{2})", key)
    return f"C{m.group(1)}_{m.group(2)}" if m else key


def main():
    img_dir = OUT / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    scan = json.loads((OUT / "scan_video_frames.json").read_text())
    by_cam = {}
    for key, v in scan.items():
        clip = key if key.endswith(".mp4") else str(SRC / key / "screen.mp4")
        for fi, y0, y1 in v["good"]:
            by_cam.setdefault(camera_of(key), []).append((clip, fi, y0, y1))
    log = []
    for cam, items in sorted(by_cam.items()):
        k = min(PER_CAM, len(items))
        picks = [items[round(i * (len(items) - 1) / max(k - 1, 1))] for i in range(k)]
        for clip, fi, y0, y1 in picks:
            cap = cv2.VideoCapture(clip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            name = f"hanoi_{cam}_{Path(clip).parent.name[-15:]}_f{fi:05d}.jpg"
            cv2.imwrite(str(img_dir / name), frame[y0:y1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            log.append({"image": name, "source": "iHanoi", "camera": cam, "clip": clip, "frame": fi, "band": [y0, y1]})
    seq = ROOT / "data/Tracking363_working/sequences/vntraffic_001/img1"
    for fid in [25, 75, 125, 175, 325, 375, 425, 475]:
        p = seq / f"{fid:06d}.jpg"
        if p.exists():
            shutil.copy2(p, img_dir / f"vntraffic001_{fid:06d}.jpg")
            log.append({"image": f"vntraffic001_{fid:06d}.jpg", "source": "vntraffic_001", "frame": fid})
    news = next((ROOT / "data").glob("AQO*.mp4"))
    cap = cv2.VideoCapture(str(news))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(10):
        fi = int(n * (i + 0.5) / 10)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(img_dir / f"vnnews_f{fi:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            log.append({"image": f"vnnews_f{fi:05d}.jpg", "source": "vn_tv_news", "frame": fi})
    (OUT / "extract_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    from collections import Counter
    print(len(log), Counter(e.get("camera", e["source"]) for e in log))


if __name__ == "__main__":
    main()
