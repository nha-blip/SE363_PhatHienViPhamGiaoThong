"""Grid search of ByteTrack settings for the chosen detector (YOLOv8s @960) on the two QA'd drafts.

Detections are computed once per sequence (same detector settings as the tracking config), then only
the tracker is re-run per setting. Scores: provisional HOTA/IDF1/MOTA with crowd-box ignore regions,
for all objects and for objects >= 32 px. Drafts are not final GT; two sequences only.
"""
import copy
import itertools
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_tracking_mot as ev  # noqa: E402
import evaluate_tracking_with_ignore as evi  # noqa: E402
from scripts.run_tracking_pilot import NAMES  # noqa: E402
from src.tracking.bytetrack_adapter import create_ultralytics_adapter  # noqa: E402
from src.tracking.io import load_frame_map  # noqa: E402
from src.tracking.pilot_runtime import LoggedBoundsBackend  # noqa: E402
from src.tracking.yolo_adapter import YoloAdapter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tracking/bytetrack_v8s_imgsz960_20261105.yaml"
DRAFTS = ROOT / "data/Tracking363_working/annotation_drafts"
SEQUENCES = {
    "vntraffic": {"dir": ROOT / "data/Tracking363_working/sequences/vntraffic_001", "window": (201, 300),
                  "gt": DRAFTS / "vntraffic_001_mot201_300_after_missing_qa/annotations_draft.json"},
    "aicc22": {"dir": ROOT / "data/Tracking363_working/sequences/aicc22_custom_001", "window": (3, 200),
               "gt": DRAFTS / "aicc22_custom_001_after_missing_qa/annotations_draft.json"},
}
GRID = {"track_high_thresh": [0.25, 0.4, 0.5], "new_track_thresh": [0.25, 0.4, 0.6], "track_buffer": [30, 60],
        "match_thresh": [0.7, 0.8, 0.9], "fuse_score": [True, False]}
OUT = ROOT / "runs/project363/tracking/tune_bytetrack_v8s_20261114"


def detections(configuration, seq):
    import cv2
    from ultralytics import YOLO
    s = configuration["detector"]
    meta = json.loads((seq["dir"] / "metadata/sequence.json").read_text(encoding="utf-8"))
    frames = load_frame_map(seq["dir"] / "metadata/frame_map.csv", meta)
    model, adapter = YOLO(str(ROOT / s["model_path"])), YoloAdapter(NAMES, s["minimum_confidence"])
    out = []
    for frame in frames:
        image = cv2.imread(str(seq["dir"] / "img1" / f"{frame.mot_frame_id:06d}.jpg"))
        p = model.predict(image, device=s["device"], imgsz=s["imgsz"], conf=s["minimum_confidence"], iou=s["nms_iou"],
                          max_det=s["max_det"], save=False, verbose=False)[0]
        out.append(adapter.adapt(p, frame))
    return meta, out


def track(settings, meta, dets):
    tracker = create_ultralytics_adapter(settings, NAMES, meta["fps"], allow_execution=True)
    tracker.backend = LoggedBoundsBackend(tracker.backend, meta["width"], meta["height"])
    pred = {}
    for d in dets:
        result = tracker.update(copy.deepcopy(d))
        pred[d.frame.mot_frame_id] = {t.track_id: [t.bbox_xyxy[0], t.bbox_xyxy[1], t.bbox_xyxy[2] - t.bbox_xyxy[0], t.bbox_xyxy[3] - t.bbox_xyxy[1]]
                                     for t in result.tracks}
    return pred


def scorer(seq, min_height):
    first, last = seq["window"]
    gt = ev.load_gt(seq["gt"], first, last)
    ignore = {}
    for frame in list(gt):
        for identity, box in list(gt[frame].items()):
            if box[3] < min_height:
                ignore.setdefault(frame, []).append(box)
                del gt[frame][identity]
    for r in json.loads(seq["gt"].read_text(encoding="utf-8")).get("ignore_regions") or []:
        ignore.setdefault(r["frame"], []).append(r["bbox_xywh"])

    def score(pred):
        evi.MIN_HEIGHT = min_height
        window = {f: b for f, b in pred.items() if first <= f <= last}
        kept, _ = evi.drop_ignored(gt, window, ignore)
        r = ev.evaluate(gt, kept, first, last)
        return {k: round(r[k], 4) for k in ("HOTA", "IDF1", "MOTA", "AssA", "DetA", "recall", "precision")} | {"IDsw": r["ID_switches"]}
    return score


def main():
    configuration = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base = configuration["tracker"]
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, seq in SEQUENCES.items():
        meta, dets = detections(configuration, seq)
        data[name] = (meta, dets, scorer(seq, 0), scorer(seq, 32))
        print("detections", name, len(dets), flush=True)
    keys = list(GRID)
    rows = []
    for values in itertools.product(*GRID.values()):
        settings = dict(base, **dict(zip(keys, values)))
        if settings["new_track_thresh"] < settings["track_high_thresh"]:
            continue
        row = {"settings": dict(zip(keys, values)), "is_baseline": all(base[k] == v for k, v in zip(keys, values))}
        for name, (meta, dets, all_sizes, large) in data.items():
            pred = track(settings, meta, dets)
            row[name] = {"all": all_sizes(pred), "h32": large(pred)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "grid_results.json").write_text(json.dumps({"base_tracker": base, "grid": GRID, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
