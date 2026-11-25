from collections import Counter


def overlap(box, prediction):
    left, top, width, height = box
    right, bottom = left + width, top + height
    intersection = max(0, min(right, prediction[2]) - max(left, prediction[0])) * max(0, min(bottom, prediction[3]) - max(top, prediction[1]))
    union = width * height + (prediction[2] - prediction[0]) * (prediction[3] - prediction[1]) - intersection
    return intersection / union if union > 0 else 0.0


def unique_matches(records, detections, minimum_iou=0.5):
    candidates = []
    for row_index, record in enumerate(records):
        for detection_index, detection in enumerate(detections):
            score = overlap(record["bbox_xywh"], detection.bbox_xyxy)
            if score >= minimum_iou:
                candidates.append((row_index, detection_index, score))
    row_counts = Counter(row for row, _, _ in candidates)
    detection_counts = Counter(detection for _, detection, _ in candidates)
    return {row: (detection, score) for row, detection, score in candidates if row_counts[row] == 1 and detection_counts[detection] == 1}


def summarize_votes(observations):
    votes = Counter(row["candidate_class"] for row in observations if row["candidate_class"] is not None)
    matched = sum(votes.values())
    candidate, count = votes.most_common(1)[0] if votes else (None, 0)
    ratio = count / matched if matched else 0.0
    adequate = matched >= 3 and ratio >= 0.8
    return {"class": None, "class_status": "PENDING", "candidate_class": candidate,
            "suggested_class": candidate if adequate else None, "matched_observations": matched,
            "total_observations": len(observations), "votes": dict(votes), "majority_ratio": ratio,
            "mixed_predictions": len(votes) > 1, "human_qa_status": "REVIEW_REQUIRED",
            "reason": "Candidate only; human confirmation required" if adequate else "Insufficient or ambiguous detector evidence"}
