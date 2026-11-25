import numpy as np
from scipy.optimize import linear_sum_assignment

from .class_suggestions import overlap


class LoggedBoundsBackend:
    def __init__(self, backend, width, height):
        self.backend = backend
        self.width = width
        self.height = height
        self.frame = 0
        self.events = []
        self.raw_output = []

    def update(self, batch, img=None):
        self.frame += 1
        output = np.asarray(self.backend.update(batch, img=img), dtype=np.float32)
        if output.size == 0:
            self.raw_output = []
            return np.empty((0, 8), dtype=np.float32)
        if output.ndim != 2 or output.shape[1] != 8 or not np.isfinite(output).all():
            raise ValueError("Invalid raw ByteTrack output")
        self.raw_output = output.tolist()
        kept = []
        for row in output:
            bounded = row.copy()
            bounded[[0, 2]] = np.clip(bounded[[0, 2]], 0, self.width)
            bounded[[1, 3]] = np.clip(bounded[[1, 3]], 0, self.height)
            invalid = bounded[2] <= bounded[0] or bounded[3] <= bounded[1]
            if invalid or not np.array_equal(row[:4], bounded[:4]):
                self.events.append({"mot_frame_id": self.frame, "predicted_track_id": int(row[4]),
                                    "original_bbox_xyxy": row[:4].tolist(), "bounded_bbox_xyxy": bounded[:4].tolist(),
                                    "action": "DROP_DEGENERATE_PREDICTION" if invalid else "CLIP_PREDICTION_ONLY"})
            if not invalid:
                kept.append(bounded)
        return np.asarray(kept, dtype=np.float32).reshape(-1, 8)


def diagnostic_matches(gt_rows, tracks, minimum_iou=0.5):
    if not gt_rows or not tracks:
        return []
    gt_count, track_count = len(gt_rows), len(tracks)
    cost = np.full((gt_count + track_count, gt_count + track_count), 1e6)
    cost[:gt_count, track_count:] = 0.51
    cost[gt_count:, :track_count] = 0.51
    cost[gt_count:, track_count:] = 0
    scores = {}
    for gt_index, row in enumerate(gt_rows):
        for track_index, track in enumerate(tracks):
            score = overlap(row["bbox_xywh"], track.bbox_xyxy)
            if score >= minimum_iou:
                cost[gt_index, track_index] = 1 - score
                scores[gt_index, track_index] = score
    rows, columns = linear_sum_assignment(cost)
    return [(gt_rows[row]["track_id"], tracks[column].track_id, scores[row, column]) for row, column in zip(rows, columns) if (row, column) in scores]


def assignment_history(gt_frames, results, minimum_iou=0.5):
    history = {}
    for result in results:
        frame_id = result.frame.mot_frame_id
        for gt_id, prediction_id, _ in diagnostic_matches(list(gt_frames.get(frame_id, {}).values()), result.tracks, minimum_iou):
            history.setdefault(gt_id, []).append((frame_id, prediction_id))
    return history


def size_bucket(short_side):
    for limit in (16, 32, 64):
        if short_side < limit:
            return f"<{limit}"
    return ">=64"


def identity_diagnostics(gt_frames, results, minimum_iou=0.5):
    """Counts that stay meaningful while GT completeness is unverified: a prediction outside GT is not called a false positive."""
    history = assignment_history(gt_frames, results, minimum_iou)
    matched = {(frame_id, gt_id) for gt_id, rows in history.items() for frame_id, _ in rows}
    gt_ids = {gt_id for rows in gt_frames.values() for gt_id in rows}
    sizes = {}
    for frame_id, rows in gt_frames.items():
        for gt_id, row in rows.items():
            counts = sizes.setdefault(size_bucket(min(row["bbox_xywh"][2:])), {"gt": 0, "matched": 0})
            counts["gt"] += 1
            counts["matched"] += (frame_id, gt_id) in matched
    predictions = outside = duplicates = 0
    for result in results:
        rows = list(gt_frames.get(result.frame.mot_frame_id, {}).values())
        predictions += len(result.tracks)
        for position, track in enumerate(result.tracks):
            outside += not any(overlap(row["bbox_xywh"], track.bbox_xyxy) >= 0.1 for row in rows)
            left, top, right, bottom = track.bbox_xyxy
            duplicates += sum(overlap((left, top, right - left, bottom - top), other.bbox_xyxy) >= 0.7 for other in result.tracks[position + 1:])
    fragments = {gt_id: len({prediction_id for _, prediction_id in rows}) for gt_id, rows in history.items()}
    changes = sum(after[1] != before[1] and after[0] == before[0] + 1 for rows in history.values() for before, after in zip(rows, rows[1:]))
    return {"gt_observations": sum(len(rows) for rows in gt_frames.values()), "gt_tracks": len(gt_ids), "matched_observations": len(matched),
            "coverage": len(matched) / max(1, sum(len(rows) for rows in gt_frames.values())), "coverage_by_short_side_px": dict(sorted(sizes.items())),
            "gt_tracks_never_matched": len(gt_ids - set(history)), "gt_tracks_with_several_prediction_ids": sum(count > 1 for count in fragments.values()),
            "mean_prediction_ids_per_matched_gt_track": sum(fragments.values()) / max(1, len(fragments)), "consecutive_id_change_candidates": changes,
            "prediction_observations": predictions, "unique_prediction_ids": len({track.track_id for result in results for track in result.tracks}),
            "predictions_outside_gt": outside, "overlapping_prediction_pairs_iou_0_7": duplicates}
