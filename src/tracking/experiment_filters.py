from .class_suggestions import overlap
from .schemas import DetectionResult


def cross_class_suppression(result, threshold=0.7):
    if not 0 < threshold <= 1:
        raise ValueError("Suppression threshold outside (0,1]")
    kept = []
    suppressed = []
    for index, detection in sorted(enumerate(result.detections), key=lambda item: (-item[1].confidence, item[0])):
        left, top, right, bottom = detection.bbox_xyxy
        competing = next((other for other in kept if other.class_id != detection.class_id and overlap((left, top, right - left, bottom - top), other.bbox_xyxy) > threshold), None)
        if competing is None:
            kept.append(detection)
        else:
            suppressed.append({"detection_index": index, "bbox_xyxy": list(detection.bbox_xyxy), "class_name": detection.class_name, "confidence": detection.confidence, "kept_bbox_xyxy": list(competing.bbox_xyxy), "kept_class_name": competing.class_name})
    return DetectionResult(result.frame, tuple(kept)), suppressed
