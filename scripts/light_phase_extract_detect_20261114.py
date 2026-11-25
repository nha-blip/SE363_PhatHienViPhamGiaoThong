"""Light-phase voting, step 1: sample existing clips at 5 fps, crop the camera scene, run light model A @960.

Test/evaluation use only (existing recordings; nothing goes into training). iHanoi clips are phone screen
recordings: the scene band is found per frame and frames without camera video are recorded as gaps.
The screen-recording timebase is not verified, so times are approximate.
"""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_qa_extract_hanoi_20261113 import scene_band  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/light_phase_eval_20261114"
WEIGHTS = ROOT / "weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt"
NAMES = ["red", "yellow", "green", "off"]
SAMPLE_FPS = 5
CAP = ROOT / "data/iHanoi_capture"
CLIPS = {  # unique recordings only (selected_signal_stopline files duplicate these)
    "C069_trial01": CAP / "ihanoi_C069_10_20261029_trial01.mp4",
    "C069_retry": CAP / "retry_pair_20261029_131120/C069_10_PTZ.mp4",
    "C057_retry": CAP / "retry_pair_20261029_131120/C057_08_PTZ.mp4",
    "C148_guarded": CAP / "guarded_20261101_075936/C148_14-PTZ.mp4",
    "C140_guarded": CAP / "guarded_20261101_075936/C140_15.mp4",
    "C183_survey": CAP / "city_survey_20261029/C183_31-PTZ/screen.mp4",
    "C047_survey": CAP / "city_survey_20261029/C047_10-PTZ/screen.mp4",
    "C040_survey": CAP / "city_survey_20261029/C040_08-PTZ/screen.mp4",
    "C042_survey": CAP / "city_survey_20261029/C042_08-PTZ/screen.mp4",
}
SEQUENCES = {"bellevue_ne12th": ROOT / "data/Tracking363_working/sequences/bellevue_ne12th_001",
             "aicc22": ROOT / "data/Tracking363_working/sequences/aicc22_custom_001"}
# Vietnamese TV news segments (camera footage with news overlays), cut on 2026-11-16; evaluation only
SEQUENCES.update({p.name: p for p in sorted((ROOT / "data/vn_news_violation_clips_20261116").glob("news_seg*")) if (p / "seqinfo.ini").exists()})


def frames_from_clip(path):
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t = 0.0
    while t * fps < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        y0, y1 = scene_band(frame)
        crop = frame[y0:y1]
        good = 450 <= y1 - y0 <= 760 and (crop.min(axis=2) > 225).mean() < 0.15 and crop.std() > 25
        yield round(t, 2), (crop if good else None)
        t += 1 / SAMPLE_FPS
    cap.release()


def frames_from_sequence(folder):
    fps = float([l for l in (folder / "seqinfo.ini").read_text().splitlines() if l.startswith("frameRate")][0].split("=")[1])
    files = sorted((folder / "img1").glob("*.jpg"))
    step = max(1, round(fps / SAMPLE_FPS))
    for k in range(0, len(files), step):
        yield round(k / fps, 2), cv2.imread(str(files[k]))


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    sources = {name: frames_from_clip(p) for name, p in CLIPS.items() if not (OUT / name / "detections.json").exists()}
    sources.update({name: frames_from_sequence(p) for name, p in SEQUENCES.items()})
    for name, frames in sources.items():
        folder = OUT / name
        if (folder / "detections.json").exists():
            continue  # already extracted
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        rows = []
        for k, (t, image) in enumerate(frames):
            row = {"index": k, "t": t, "scene": image is not None, "detections": []}
            if image is not None:
                cv2.imwrite(str(folder / "frames" / f"{k:05d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                r = model.predict(image, imgsz=960, conf=0.1, verbose=False)[0]
                row["detections"] = [{"xyxy": [round(v, 1) for v in b], "cls": NAMES[int(c)], "conf": round(float(s), 4)}
                                     for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            rows.append(row)
        (folder / "detections.json").write_text(json.dumps({"source": str(CLIPS.get(name) or SEQUENCES[name]), "sample_fps": SAMPLE_FPS,
                                                            "weights": str(WEIGHTS), "frames": rows}), encoding="utf-8")
        print(name, len(rows), "samples,", sum(r["scene"] for r in rows), "with scene,",
              sum(len(r["detections"]) for r in rows), "detections", flush=True)


if __name__ == "__main__":
    main()
