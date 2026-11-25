"""Re-ask the tracking QA items answered "unclear", with larger context crops and more frames.

vntraffic_001 MOT 201-300: tracks 33 and 39 (231-300 track QA), missing-vehicle candidates answered
unclear in the v8s round (231-300) and in the all-detector round (201-300).
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tracking_track_qa as builder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/Tracking363_working/annotation_drafts"
DRAFT = BASE / "vntraffic_001_mot201_300_after_missing_qa"
OUT = BASE / "vntraffic_001_mot201_300_unclear_reqa"
TRACKING = ROOT / "runs/project363/tracking"
DETECTORS = {"v8s_960": "vehicle_v8s_imgsz960_vntraffic_001", "e63_960": "vehicle_epoch63_imgsz960_vntraffic_001",
             "e41_960": "vehicle_epoch41_imgsz960_vntraffic_001", "e30": "vehicle_epoch30_vntraffic_001"}
FRAMES, HEIGHT = 8, 260


def page_data(folder):
    return json.loads(re.search(r"const DATA = (\{.*?\});\nconst KEY", (folder / "track_qa.html").read_text(encoding="utf-8"), re.S).group(1))


def big_strip(rows, path):
    chosen = rows if len(rows) <= FRAMES else [rows[round(i * (len(rows) - 1) / (FRAMES - 1))] for i in range(FRAMES)]
    patches = []
    for frame, box, image_path in chosen:
        image = cv2.imread(str(DRAFT / image_path))
        left, top, width, height = box
        margin = max(width, height)  # one box size of context on each side
        x0, y0 = int(max(0, left - margin)), int(max(0, top - margin))
        x1, y1 = int(min(image.shape[1], left + width + margin)), int(min(image.shape[0], top + height + margin))
        patch = image[y0:y1, x0:x1].copy()
        cv2.rectangle(patch, (int(left - x0), int(top - y0)), (int(left + width - x0), int(top + height - y0)), (0, 255, 0), 2)
        scale = HEIGHT / patch.shape[0]
        patch = cv2.resize(patch, (max(40, int(patch.shape[1] * scale)), HEIGHT), interpolation=cv2.INTER_CUBIC)
        cv2.putText(patch, f"f{frame}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        patches += [patch, np.full((HEIGHT, 6, 3), 40, np.uint8)]
    strip = np.hstack(patches[:-1])
    if strip.shape[1] > 1800:  # wrap to two rows so the app can show it large
        half = len(chosen) // 2 * 2
        top_row, bottom_row = np.hstack(patches[:half - 1]), np.hstack(patches[half:-1])
        width = max(top_row.shape[1], bottom_row.shape[1])
        pad = lambda a: np.hstack([a, np.zeros((HEIGHT, width - a.shape[1], 3), np.uint8)])
        strip = np.vstack([pad(top_row), np.full((8, width, 3), 40, np.uint8), pad(bottom_row)])
    cv2.imwrite(str(path), strip)


def detector_rows(detector, identity, images, first, last):
    rows = []
    for line in (TRACKING / DETECTORS[detector] / "predictions_mot.txt").read_text().splitlines():
        p = line.split(",")
        if int(p[1]) == identity and first <= int(p[0]) <= last:
            rows.append((int(p[0]), [float(v) for v in p[2:6]], images[int(p[0])]))
    return sorted(rows)


def main():
    draft = json.loads((DRAFT / "annotations_draft.json").read_text(encoding="utf-8"))
    images = {f["mot_frame_id"]: f["image"] for f in draft["frames"]}
    (OUT / "strips").mkdir(parents=True, exist_ok=True)
    tracks, missing = [], []
    q231 = json.loads((BASE / "vntraffic_001_mot231_300_track_qa/tracking_track_qa_result.json").read_text(encoding="utf-8"))
    for key, answer in q231["tracks"].items():
        if answer["label"] == "unclear":
            rows = sorted((f["mot_frame_id"], o["bbox_xywh"], f["image"]) for f in draft["frames"] if 231 <= f["mot_frame_id"] <= 300
                          for o in f["objects"] if o["track_id"] == int(key))
            big_strip(rows, OUT / "strips" / f"track_{key}.jpg")
            tracks.append({"key": key, "images": [f"strips/track_{key}.jpg"],
                           "caption": f"ID {key} trong nhãn · f{rows[0][0]}–f{rows[-1][0]} ({len(rows)} frame) · lần trước: không rõ"})
    for key, answer in q231["missing"].items():
        if answer["label"] == "unclear":
            rows = [r for r in detector_rows("v8s_960", int(key), images, 231, 300)]
            big_strip(rows, OUT / "strips" / f"missing_v8s_960_{key}.jpg")
            missing.append({"key": f"v8s_960:{key}", "window": [231, 300], "images": [f"strips/missing_v8s_960_{key}.jpg"],
                            "caption": f"Track {key} của v8s_960 · f{rows[0][0]}–f{rows[-1][0]} · lần trước: không rõ"})
    qall_dir = BASE / "vntraffic_001_mot201_300_missing_qa_all_detectors"
    qall = json.loads((qall_dir / "tracking_track_qa_result.json").read_text(encoding="utf-8"))["missing"]
    for item in page_data(qall_dir)["missing"]:
        if qall[item["key"]]["label"] == "unclear":
            detector, identity = item["key"].split(":")
            rows = detector_rows(detector, int(identity), images, 201, 300)
            name = f"missing_{detector}_{identity}.jpg"
            big_strip(rows, OUT / "strips" / name)
            missing.append({"key": item["key"], "window": [201, 300], "images": [f"strips/{name}"],
                            "caption": f"Track {identity} của {detector} · f{rows[0][0]}–f{rows[-1][0]} · lần trước: không rõ"})
    data = {"sequence_id": draft["sequence_id"], "frames": [201, 300], "tracks": tracks, "missing": missing, "draft": str(DRAFT)}
    page = builder.PAGE.replace("__STYLE__", builder.STYLE).replace("__DATA__", json.dumps(data, ensure_ascii=False))
    (OUT / "track_qa.html").write_text(page, encoding="utf-8")
    print(len(tracks), "tracks,", len(missing), "missing candidates ->", OUT)


if __name__ == "__main__":
    main()
