"""Join a vehicle track that is lost at the stop line with the track that continues it (ID switch).

Only inside a continuous video (no quality gap between the two pieces) and only near a stop line, where an
ID switch breaks the crossing evidence. A piece B continues piece A when:
* B starts after A ends, within max_gap_s, with no quality gap in between;
* same class family (car/truck/bus are one family: the detector flips between them; motorcycle is its own);
* A's last reference point is within near_line_px of a stop line;
* B's first reference point is within max(min_px, ratio * A's height) of where A's recent motion predicts it;
* the box heights are within height_ratio of each other.
Matches are made greedily by distance; each piece joins at most one predecessor and one successor.
The merged track keeps A's id; every join is reported so it can be reviewed.
"""
from dataclasses import dataclass, replace

FAMILY = {"car": "4w", "truck": "4w", "bus": "4w", "motorcycle": "2w"}


@dataclass
class StitchSettings:
    max_gap_s: float = 0.6
    near_line_px: float = 80.0
    min_px: float = 25.0
    ratio: float = 0.5
    height_ratio: float = 1.6
    velocity_window_s: float = 0.4


def _family(points):
    votes = {}
    for p in points:
        votes[FAMILY.get(p.cls, p.cls)] = votes.get(FAMILY.get(p.cls, p.cls), 0.0) + p.conf
    return max(votes, key=votes.get)


def _bottom(p):
    x1, y1, x2, y2 = p.bbox
    return (x1 + x2) / 2, y2


def _segment_distance(point, line):
    (ax, ay), (bx, by) = line
    px, py = point
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy or 1e-9
    u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return ((px - ax - u * dx) ** 2 + (py - ay - u * dy) ** 2) ** 0.5


def stitch_tracks(points, stop_lines, gap_times=(), settings=StitchSettings()):
    """points: TrackPoints of all tracks. Returns (new points, joins). gap_times: times of quality gaps."""
    tracks = {}
    for p in points:
        tracks.setdefault(p.track_id, []).append(p)
    for pts in tracks.values():
        pts.sort(key=lambda p: p.t)
    candidates = []
    for a_id, a in tracks.items():
        end = a[-1]
        end_xy = _bottom(end)
        if not stop_lines or min(_segment_distance(end_xy, line) for line in stop_lines) > settings.near_line_px:
            continue
        recent = [p for p in a if p.t >= end.t - settings.velocity_window_s]
        dt = end.t - recent[0].t
        start_xy = _bottom(recent[0])
        velocity = ((end_xy[0] - start_xy[0]) / dt, (end_xy[1] - start_xy[1]) / dt) if dt > 0 else (0.0, 0.0)
        height_a = end.bbox[3] - end.bbox[1]
        for b_id, b in tracks.items():
            first = b[0]
            gap = first.t - end.t
            if b_id == a_id or not 0 < gap <= settings.max_gap_s or _family(a) != _family(b):
                continue
            if any(end.t < g < first.t for g in gap_times):
                continue
            height_b = first.bbox[3] - first.bbox[1]
            if max(height_a, height_b) > settings.height_ratio * max(min(height_a, height_b), 1e-6):
                continue
            predicted = (end_xy[0] + velocity[0] * gap, end_xy[1] + velocity[1] * gap)
            bx, by = _bottom(first)
            distance = ((bx - predicted[0]) ** 2 + (by - predicted[1]) ** 2) ** 0.5
            if distance <= max(settings.min_px, settings.ratio * height_a):
                candidates.append((distance, a_id, b_id, gap))
    successor, predecessor, joins = {}, {}, []
    for distance, a_id, b_id, gap in sorted(candidates):
        if a_id in successor or b_id in predecessor:
            continue
        successor[a_id], predecessor[b_id] = b_id, a_id
        joins.append({"kept_id": None, "from_id": a_id, "to_id": b_id, "gap_s": round(gap, 3), "distance_px": round(distance, 1)})
    root = {}
    for track_id in tracks:
        r = track_id
        while r in predecessor:
            r = predecessor[r]
        root[track_id] = r
    for j in joins:
        j["kept_id"] = root[j["from_id"]]
    out = [replace(p, track_id=root[p.track_id]) if root[p.track_id] != p.track_id else p for p in points]
    return out, joins
