"""Run one detector + ByteTrack (same components and settings as the vntraffic pilot) on any MOT sequence.

Detector and tracker settings come from an existing configs/tracking/*.yaml; only the sequence changes.
Writes predictions_mot.txt and tracking_results.jsonl. No GT is read; evaluation is separate.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_tracking_pilot import NAMES, write_json
from src.tracking.bytetrack_adapter import create_ultralytics_adapter
from src.tracking.io import load_frame_map, sha256
from src.tracking.pilot_runtime import LoggedBoundsBackend
from src.tracking.yolo_adapter import YoloAdapter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="existing tracking yaml (detector + tracker sections are used)")
    parser.add_argument("--sequence", type=Path, required=True, help="sequence folder with img1/ and metadata/")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    configuration = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    settings = configuration["detector"]
    if sha256(ROOT / settings["model_path"]) != settings["expected_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    sequence = json.loads((a.sequence / "metadata/sequence.json").read_text(encoding="utf-8"))
    frames = load_frame_map(a.sequence / "metadata/frame_map.csv", sequence)
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(settings["cpu_threads"])
    model = YOLO(str(ROOT / settings["model_path"]))
    if model.names != NAMES:
        raise ValueError("Unexpected model namespace")
    tracker = create_ultralytics_adapter(configuration["tracker"], NAMES, sequence["fps"], allow_execution=True)
    bounded = LoggedBoundsBackend(tracker.backend, sequence["width"], sequence["height"])
    tracker.backend = bounded
    adapter = YoloAdapter(NAMES, settings["minimum_confidence"])
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "effective_config.json", {"source_config": str(a.config), "sequence": str(a.sequence),
                                                    "detector": settings, "tracker": configuration["tracker"]})
    write_json(a.output / "environment.json", {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                                               "created_at_utc": datetime.now(timezone.utc).isoformat()})
    seconds = 0.0
    with (a.output / "tracking_results.jsonl").open("x", encoding="utf-8") as tracking, \
            (a.output / "predictions_mot.txt").open("x", encoding="utf-8") as mot:
        for frame in frames:
            image = cv2.imread(str(a.sequence / "img1" / f"{frame.mot_frame_id:06d}.jpg"))
            if image is None or image.shape[:2] != (frame.height, frame.width):
                raise ValueError(f"Missing or incorrectly sized frame {frame.mot_frame_id}")
            started = time.perf_counter()
            prediction = model.predict(image, device=settings["device"], imgsz=settings["imgsz"], conf=settings["minimum_confidence"],
                                       iou=settings["nms_iou"], max_det=settings["max_det"], save=False, verbose=False)[0]
            result = tracker.update(adapter.adapt(prediction, frame))
            seconds += time.perf_counter() - started
            tracking.write(json.dumps(result.to_dict(), allow_nan=False) + "\n")
            for track in result.tracks:
                left, top, right, bottom = track.bbox_xyxy
                mot.write(f"{frame.mot_frame_id},{track.track_id},{left:.6f},{top:.6f},{right - left:.6f},{bottom - top:.6f},{track.confidence:.6f},-1,-1,-1\n")
    write_json(a.output / "bounds_events.json", bounded.events)
    write_json(a.output / "report.json", {"frames": len(frames), "seconds_detect_and_track": seconds, "status": "PREDICTIONS_ONLY"})
    print("done", a.output, len(frames), "frames", f"{seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
