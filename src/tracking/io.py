import csv
import hashlib
import json
import math
from pathlib import Path

from .schemas import DetectionResult, FrameInfo, TrackingResult


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def snapshot_hash(payload):
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def verify_snapshot(snapshot, workspace):
    if snapshot_hash(snapshot["payload"]) != snapshot["snapshot_hash"]:
        raise ValueError("Integration snapshot hash mismatch")
    root = Path(workspace).resolve()
    for relative, expected in snapshot["payload"]["files_sha256"].items():
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or sha256(path) != expected:
            raise ValueError(f"Snapshot artifact mismatch: {relative}")
    return True


def load_frame_map(path, sequence):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != sequence["pilot_frame_count"]:
        raise ValueError("Frame-map count mismatch")
    frames = []
    for index, row in enumerate(rows):
        frame = FrameInfo(sequence["sequence_id"], int(row["source_frame_id"]), int(row["mot_frame_id"]),
                          sequence["width"], sequence["height"], sequence["fps"], sequence["pilot_frame_count"])
        if frame.source_frame_id != index or row["image_filename"] != f"{frame.mot_frame_id:06d}.jpg":
            raise ValueError("Frame-map order/name mismatch")
        stamp = float(row["timestamp_seconds"])
        if not math.isfinite(stamp) or abs(stamp - frame.source_frame_id / frame.fps) > 1e-8:
            raise ValueError("Frame-map timestamp mismatch")
        if row["source_video"] != sequence["source_video"]:
            raise ValueError("Frame-map source mismatch")
        frames.append(frame)
    return tuple(frames)


def write_results(path, results):
    destination = Path(path)
    previous = 0
    sequence_id = None
    with destination.open("x", encoding="utf-8") as handle:
        for result in results:
            if not isinstance(result, (DetectionResult, TrackingResult)):
                raise ValueError("Validated contract result required")
            frame = result.frame
            if frame.mot_frame_id != previous + 1 or sequence_id is not None and sequence_id != frame.sequence_id:
                raise ValueError("Results must contain every frame once in temporal order")
            handle.write(json.dumps(result.to_dict(), allow_nan=False) + "\n")
            previous, sequence_id = frame.mot_frame_id, frame.sequence_id


def read_results(path, result_type):
    if result_type not in (DetectionResult, TrackingResult):
        raise ValueError("Unsupported result type")
    with Path(path).open(encoding="utf-8") as handle:
        return tuple(result_type.from_dict(json.loads(line)) for line in handle)
