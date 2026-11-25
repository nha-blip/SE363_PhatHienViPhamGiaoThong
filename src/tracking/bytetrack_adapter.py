import inspect
from types import SimpleNamespace

import numpy as np

from .schemas import DetectionResult, Track, TrackingResult, VEHICLE_CLASSES
from .yolo_adapter import class_names_mapping


class DetectionBatch:
    def __init__(self, xyxy, confidence, classes):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(classes, dtype=np.float32).reshape(-1)
        if len(self.xyxy) != len(self.conf) or len(self.conf) != len(self.cls):
            raise ValueError("DetectionBatch array lengths differ")

    @property
    def xywh(self):
        boxes = self.xyxy.copy()
        boxes[:, :2] = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        boxes[:, 2:] = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return boxes

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        return DetectionBatch(self.xyxy[index], self.conf[index], self.cls[index])


class ByteTrackAdapter:
    def __init__(self, backend, class_names):
        self.backend = backend
        self.class_names = class_names_mapping(class_names)
        self.sequence_id = None
        self.last_frame = 0

    def update(self, result, image=None):
        if not isinstance(result, DetectionResult):
            raise ValueError("DetectionResult required")
        frame = result.frame
        if frame.mot_frame_id != self.last_frame + 1:
            raise ValueError("Frames must start at MOT 1 and advance once per frame, even with zero detections")
        if self.sequence_id is not None and frame.sequence_id != self.sequence_id:
            raise ValueError("New sequence requires a new tracker; no cross-sequence ID state")
        for item in result.detections:
            if self.class_names.get(item.class_id) != item.class_name:
                raise ValueError("Detection metadata differs from original detector namespace")
        batch = DetectionBatch([item.bbox_xyxy for item in result.detections],
                               [item.confidence for item in result.detections],
                               [item.class_id for item in result.detections])
        output = np.asarray(self.backend.update(batch, img=image))
        self.last_frame = frame.mot_frame_id
        self.sequence_id = frame.sequence_id
        if output.size == 0:
            return TrackingResult(frame)
        if output.ndim != 2 or output.shape[1] != 8 or not np.isfinite(output).all():
            raise ValueError("Expected finite Ultralytics axis-aligned ByteTrack Nx8 output")
        observations = []
        for row in output:
            identity, class_id = row[4], row[6]
            if identity != int(identity) or class_id != int(class_id):
                raise ValueError("Tracker/class IDs must be integers")
            class_id = int(class_id)
            name = self.class_names.get(class_id)
            if name not in VEHICLE_CLASSES:
                raise ValueError("Nonvehicle or unknown tracker output class")
            observations.append(Track(tuple(map(float, row[:4])), float(row[5]), class_id, name, int(identity)))
        return TrackingResult(frame, tuple(sorted(observations, key=lambda item: item.track_id)))


def create_ultralytics_adapter(settings, class_names, fps, allow_execution=False):
    if not allow_execution:
        raise PermissionError("Actual ByteTrack creation requires explicit Work 5.4 execution authorization")
    from ultralytics.trackers.byte_tracker import BYTETracker

    arguments = SimpleNamespace(**settings)
    keywords = {"frame_rate": fps} if "frame_rate" in inspect.signature(BYTETracker).parameters else {}
    return ByteTrackAdapter(BYTETracker(arguments, **keywords), class_names)
