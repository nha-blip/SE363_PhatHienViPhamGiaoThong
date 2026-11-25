from .schemas import Detection, DetectionResult, VEHICLE_CLASSES, finite, integer


def class_names_mapping(value):
    if isinstance(value, list):
        value = dict(enumerate(value))
    if not isinstance(value, dict):
        raise ValueError("Detector class names metadata required")
    names = {}
    for identity, name in value.items():
        if isinstance(identity, str) and identity.isdigit():
            identity = int(identity)
        integer(identity, "detector class_id")
        if not isinstance(name, str) or identity in names:
            raise ValueError("Invalid detector names mapping")
        names[identity] = name
    return names


def plain_list(value):
    for operation in ("detach", "cpu", "tolist"):
        if hasattr(value, operation):
            value = getattr(value, operation)()
    return list(value)


class YoloAdapter:
    def __init__(self, class_names, minimum_confidence=0.1):
        self.class_names = class_names_mapping(class_names)
        self.minimum_confidence = finite(minimum_confidence, "minimum_confidence")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence outside [0,1]")

    def adapt(self, result, frame):
        if class_names_mapping(result.names) != self.class_names:
            raise ValueError("Detector namespace changed; do not remap IDs")
        if tuple(result.orig_shape) != (frame.height, frame.width):
            raise ValueError("Detector output must use native image coordinates")
        if result.boxes is None:
            return DetectionResult(frame)
        boxes = plain_list(result.boxes.xyxy)
        scores = plain_list(result.boxes.conf)
        identities = plain_list(result.boxes.cls)
        if len(boxes) != len(scores) or len(boxes) != len(identities):
            raise ValueError("Detector array length mismatch")
        observations = []
        for box, score, identity in zip(boxes, scores, identities):
            numeric_id = finite(identity, "detector class_id")
            if numeric_id != int(numeric_id) or numeric_id < 0:
                raise ValueError("Invalid detector class_id")
            identity = int(numeric_id)
            if identity not in self.class_names:
                raise ValueError("Unknown class ID, no guessed mapping")
            score = finite(score, "confidence")
            if not 0 <= score <= 1:
                raise ValueError("Invalid confidence")
            name = self.class_names[identity]
            if name not in VEHICLE_CLASSES or score < self.minimum_confidence:
                continue
            observations.append(Detection(tuple(box), score, identity, name))
        return DetectionResult(frame, tuple(observations))
