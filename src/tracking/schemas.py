import math
from dataclasses import dataclass


VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck"})
FRAME_FIELDS = ("sequence_id", "source_frame_id", "mot_frame_id", "width", "height", "fps", "sequence_length")


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def valid_box(value):
    if len(value) != 4:
        raise ValueError("bbox_xyxy must contain four values")
    coordinates = tuple(finite(item, "bbox coordinate") for item in value)
    left, top, right, bottom = coordinates
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError("Invalid pixel xyxy box; no silent clipping")
    return coordinates


def xywh_to_xyxy(box):
    if len(box) != 4:
        raise ValueError("Expected pixel left/top/width/height")
    left, top, width, height = (finite(item, "bbox coordinate") for item in box)
    return valid_box((left, top, left + width, top + height))


def xyxy_to_xywh(box):
    left, top, right, bottom = valid_box(box)
    return left, top, right - left, bottom - top


@dataclass(frozen=True)
class FrameInfo:
    sequence_id: str
    source_frame_id: int
    mot_frame_id: int
    width: int
    height: int
    fps: float
    sequence_length: int

    def __post_init__(self):
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id is required")
        integer(self.source_frame_id, "source_frame_id")
        integer(self.mot_frame_id, "mot_frame_id", 1)
        integer(self.width, "width", 1)
        integer(self.height, "height", 1)
        integer(self.sequence_length, "sequence_length", 1)
        if finite(self.fps, "fps") <= 0:
            raise ValueError("fps must be positive")
        if self.mot_frame_id != self.source_frame_id + 1 or self.mot_frame_id > self.sequence_length:
            raise ValueError("Expected source 0-based -> MOT 1-based within sequence")

    def to_dict(self):
        return {name: getattr(self, name) for name in FRAME_FIELDS}

    @classmethod
    def from_dict(cls, value):
        return cls(**{name: value[name] for name in FRAME_FIELDS})


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple
    confidence: float
    class_id: int
    class_name: str

    def __post_init__(self):
        object.__setattr__(self, "bbox_xyxy", valid_box(self.bbox_xyxy))
        score = finite(self.confidence, "confidence")
        if not 0 <= score <= 1:
            raise ValueError("confidence outside [0,1]")
        integer(self.class_id, "class_id")
        if self.class_name not in VEHICLE_CLASSES:
            raise ValueError("Only vehicle predictions enter this contract")

    def to_dict(self):
        return {"bbox_xyxy": list(self.bbox_xyxy), "confidence": self.confidence,
                "class_id": self.class_id, "class_name": self.class_name}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class Track(Detection):
    track_id: int

    def __post_init__(self):
        Detection.__post_init__(self)
        integer(self.track_id, "track_id", 1)

    def to_dict(self):
        return {**Detection.to_dict(self), "track_id": self.track_id}


def validate_observations(frame, observations, expected_type):
    if not isinstance(frame, FrameInfo):
        raise ValueError("FrameInfo required")
    for observation in observations:
        if not isinstance(observation, expected_type):
            raise ValueError("Unexpected observation type")
        if observation.bbox_xyxy[2] > frame.width or observation.bbox_xyxy[3] > frame.height:
            raise ValueError("Box outside original image bounds; no silent clipping")


@dataclass(frozen=True)
class DetectionResult:
    frame: FrameInfo
    detections: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "detections", tuple(self.detections))
        validate_observations(self.frame, self.detections, Detection)

    def to_dict(self):
        return {**self.frame.to_dict(), "detections": [item.to_dict() for item in self.detections]}

    @classmethod
    def from_dict(cls, value):
        if set(value) != set(FRAME_FIELDS) | {"detections"}:
            raise ValueError("Unexpected DetectionResult fields")
        return cls(FrameInfo.from_dict(value), tuple(Detection.from_dict(item) for item in value["detections"]))


@dataclass(frozen=True)
class TrackingResult:
    frame: FrameInfo
    tracks: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "tracks", tuple(self.tracks))
        validate_observations(self.frame, self.tracks, Track)
        identities = [item.track_id for item in self.tracks]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate predicted track ID in one frame")

    def to_dict(self):
        return {**self.frame.to_dict(), "tracks": [item.to_dict() for item in self.tracks]}

    @classmethod
    def from_dict(cls, value):
        if set(value) != set(FRAME_FIELDS) | {"tracks"}:
            raise ValueError("Unexpected TrackingResult fields")
        return cls(FrameInfo.from_dict(value), tuple(Track.from_dict(item) for item in value["tracks"]))
