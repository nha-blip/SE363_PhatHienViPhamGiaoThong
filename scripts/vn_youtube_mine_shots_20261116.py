"""Mine traffic-camera shots from the VN YouTube pool (training candidates; never the news test source).

1 frame/s; a frame counts as camera footage when the vehicle model finds >= 3 vehicles (conf >= 0.3).
Consecutive footage frames with similar colour histograms form a shot. Output per video:
frames/<video>/<sec>.jpg, shots.json (video, shot, seconds, light detections of model A), and contact sheets.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data/vn_youtube_train_pool_20261116"
VEH = YOLO(str(ROOT / "weights/kaggle_small_vehicle_20261019/best.pt"))
LIGHT = YOLO(str(ROOT / "weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt"))


def hist(img):
    h = cv2.calcHist([cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def main():
    shots = []
    for video in sorted((POOL / "raw").glob("*.mp4")):
        cap = cv2.VideoCapture(str(video))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = POOL / "frames" / video.stem
        out.mkdir(parents=True, exist_ok=True)
        current, prev_h = None, None
        for sec in range(int(n / fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ok, img = cap.read()
            if not ok:
                break
            v = VEH.predict(img, imgsz=960, conf=0.3, verbose=False)[0]
            if len(v.boxes) < 3:
                current, prev_h = None, None
                continue
            h = hist(img)
            if current is None or prev_h is None or cv2.compareHist(prev_h, h, cv2.HISTCMP_CORREL) < 0.85:
                current = {"video": video.stem, "shot": len([s for s in shots if s["video"] == video.stem]), "seconds": [], "lights": {}}
                shots.append(current)
            prev_h = h
            current["seconds"].append(sec)
            cv2.imwrite(str(out / f"{sec:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            p = LIGHT.predict(img, imgsz=960, conf=0.1, verbose=False)[0]
            current["lights"][sec] = [(LIGHT.names[int(c)], round(float(s), 2), [round(x) for x in b]) for b, c, s in
                                      zip(p.boxes.xyxy.tolist(), p.boxes.cls.tolist(), p.boxes.conf.tolist())]
        cap.release()
        print(video.stem, "shots", len([s for s in shots if s["video"] == video.stem]), flush=True)
    (POOL / "shots.json").write_text(json.dumps(shots, indent=1), encoding="utf-8")
    sheets = POOL / "contact_sheets"
    sheets.mkdir(exist_ok=True)
    keep = [s for s in shots if len(s["seconds"]) >= 2]
    for k in range(0, len(keep), 20):
        tiles = []
        for s in keep[k:k + 20]:
            mid = s["seconds"][len(s["seconds"]) // 2]
            img = cv2.resize(cv2.imread(str(POOL / "frames" / s["video"] / f"{mid:05d}.jpg")), (480, 270))
            cv2.putText(img, f"{keep.index(s)} {s['video']} {s['seconds'][0]}-{s['seconds'][-1]}s", (4, 18), 0, 0.5, (0, 255, 255), 2)
            tiles.append(img)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(sheets / f"sheet_{k // 20:02d}.jpg"), np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]))
    print("shots >=2 s:", len(keep), "sheets:", (len(keep) + 19) // 20)


if __name__ == "__main__":
    main()
