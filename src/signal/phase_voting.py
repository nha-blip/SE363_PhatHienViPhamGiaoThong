"""Traffic-light phase per light head with causal temporal voting.

1. LightHeadAssociator groups per-frame light detections into static "heads" (one physical signal
   head) by IoU / centre distance to the head's last box.
2. PhaseVoter keeps one state per head and changes it only when a new colour clearly dominates the
   last `window_s` seconds (confidence-weighted votes, ratio margin and minimum count). With no
   observation for `unknown_after_s` seconds the state becomes "unknown". Only past frames are used,
   so it can run online.
"""
from dataclasses import dataclass, field

STATES = ("red", "yellow", "green", "off")


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


@dataclass
class Head:
    head_id: int
    box: list
    last_t: float
    observations: list = field(default_factory=list)  # (t, cls, conf)


class LightHeadAssociator:
    def __init__(self, create_conf=0.25, attach_conf=0.1, min_iou=0.2, max_center_ratio=0.75, expire_s=10.0):
        self.create_conf, self.attach_conf = create_conf, attach_conf
        self.min_iou, self.max_center_ratio, self.expire_s = min_iou, max_center_ratio, expire_s
        self.heads, self.next_id = [], 1

    def _cost(self, head, box):
        if _iou(head.box, box) >= self.min_iou:
            return 0.0
        hx, hy = (head.box[0] + head.box[2]) / 2, (head.box[1] + head.box[3]) / 2
        bx, by = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        size = max(head.box[3] - head.box[1], box[3] - box[1], 1.0)
        ratio = (box[3] - box[1]) / max(head.box[3] - head.box[1], 1e-6)
        distance = ((hx - bx) ** 2 + (hy - by) ** 2) ** 0.5 / size
        return distance if distance <= self.max_center_ratio and 0.5 <= ratio <= 2.0 else None

    def update(self, t, detections):
        """detections: [{"xyxy", "cls", "conf"}]. Returns {head_id: (cls, conf, box)} for this frame."""
        alive = [h for h in self.heads if t - h.last_t <= self.expire_s]
        pairs = []
        for k, d in enumerate(detections):
            if d["conf"] < self.attach_conf:
                continue
            for h in alive:
                c = self._cost(h, d["xyxy"])
                if c is not None:
                    pairs.append((c, -d["conf"], k, h.head_id))
        used_d, used_h, out = set(), set(), {}
        by_id = {h.head_id: h for h in self.heads}
        for c, _, k, hid in sorted(pairs):
            if k in used_d or hid in used_h:
                continue
            used_d.add(k)
            used_h.add(hid)
            d, h = detections[k], by_id[hid]
            h.box, h.last_t = d["xyxy"], t
            h.observations.append((t, d["cls"], d["conf"]))
            out[hid] = (d["cls"], d["conf"], d["xyxy"])
        for k, d in enumerate(detections):
            if k in used_d or d["conf"] < self.create_conf:
                continue
            # suppress a second head on top of an already matched one (duplicate boxes)
            if any(_iou(d["xyxy"], by_id[h].box) > 0.3 for h in used_h):
                continue
            h = Head(self.next_id, d["xyxy"], t, [(t, d["cls"], d["conf"])])
            self.heads.append(h)
            by_id[h.head_id] = h
            used_h.add(h.head_id)
            out[h.head_id] = (d["cls"], d["conf"], d["xyxy"])
            self.next_id += 1
        return out


class PhaseVoter:
    def __init__(self, window_s=2.0, ratio=1.5, min_count=2, unknown_after_s=6.0):
        self.window_s, self.ratio, self.min_count, self.unknown_after_s = window_s, ratio, min_count, unknown_after_s
        self.state, self.history = {}, {}

    def update(self, t, observed):
        """observed: {head_id: (cls, conf, box)} for this frame. Returns {head_id: state} for every known head."""
        for hid, (cls, conf, _) in observed.items():
            self.history.setdefault(hid, []).append((t, cls, conf))
            self.state.setdefault(hid, "unknown")
        result = {}
        for hid, rows in self.history.items():
            recent = [r for r in rows if t - self.window_s < r[0] <= t]
            if not rows or t - rows[-1][0] > self.unknown_after_s:
                self.state[hid] = "unknown"
            elif recent:
                score, count = {}, {}
                for _, cls, conf in recent:
                    score[cls] = score.get(cls, 0.0) + conf
                    count[cls] = count.get(cls, 0) + 1
                best = max(score, key=score.get)
                current = self.state[hid]
                needed = 1 if current == "unknown" else self.min_count  # first state: no previous colour to protect
                if best != current and count[best] >= needed and score[best] >= self.ratio * score.get(current, 0.0):
                    self.state[hid] = best
            result[hid] = self.state[hid]
        return result


def signal_intervals(frames_out, head_id, lookback_s=2.0, no_observation_s=6.0, missing_yellow_guard_s=3.5):
    """Turn a per-frame voted timeline of one head into signal intervals (spec §9.4, FR-07).

    * "observed": [first, last] raw observation of a colour inside one voted colour segment. Only this span
      is asserted; `decision_latency_s` = time the voter switched minus the first observation.
    * "transition": between the last observation of one colour and the first observation of the next.
      No colour is asserted there (no back-filling across an unclear change).
    * "no_observation": a gap longer than `no_observation_s` without any observation of the head.
    * "uncertain_colour": the signal cycle is green -> yellow -> red. When green is followed directly by red,
      the first `missing_yellow_guard_s` of that red are not asserted: the light model often reads a
      (Vietnamese, orange-looking) yellow as red. Set to 0 to disable.
    Returns intervals sorted by time.
    """
    rows = [(f["t"], f["raw"].get(head_id), f["voted"].get(head_id, "unknown")) for f in frames_out]
    segments = []  # voted colour segments: [state, decided_at, last_frame_t]
    for t, _, state in rows:
        if segments and segments[-1][0] == state:
            segments[-1][2] = t
        else:
            segments.append([state, t, t])
    observed = []
    previous_last = None
    for k, (state, decided_at, seg_end) in enumerate(segments):
        if state == "unknown":
            continue
        next_decision = next((s[1] for s in segments[k + 1:] if s[0] != "unknown"), float("inf"))
        start_bound = max(decided_at - lookback_s, previous_last if previous_last is not None else float("-inf"))
        obs = [(t, raw) for t, raw, _ in rows if raw == state and start_bound <= t < next_decision]
        obs = [o for o in obs if o[0] >= start_bound]
        if not obs:
            continue
        first, last = obs[0][0], max(t for t, _ in obs if t <= max(seg_end, decided_at))
        conflicting = sum(1 for t, raw, _ in rows if first <= t <= last and raw not in (None, state))
        observed.append({"state": state, "t_start": first, "t_end": last, "kind": "observed",
                         "decision_latency_s": round(decided_at - first, 3), "observations": len([o for o in obs if o[0] <= last]),
                         "conflicting_observations": conflicting})
        previous_last = last
    merged = []
    for item in observed:  # same colour separated only by a short gap (e.g. LED flicker, brief occlusion) is one interval
        if merged and merged[-1]["state"] == item["state"] and item["t_start"] - merged[-1]["t_end"] <= no_observation_s:
            merged[-1]["t_end"] = item["t_end"]
            merged[-1]["observations"] += item["observations"]
            merged[-1]["conflicting_observations"] += item["conflicting_observations"]
        else:
            merged.append(item)
    observed = merged
    intervals = []
    for a, b in zip(observed, observed[1:] + [None]):
        intervals.append(a)
        if b is None:
            continue
        gap_obs = [t for t, raw, _ in rows if a["t_end"] < t < b["t_start"] and raw is not None]
        times = [a["t_end"]] + gap_obs + [b["t_start"]]
        longest = max(y - x for x, y in zip(times, times[1:]))
        if a["state"] != b["state"]:
            intervals.append({"state": "unknown", "t_start": a["t_end"], "t_end": b["t_start"], "kind": "transition",
                              "from": a["state"], "to": b["state"], "longest_gap_s": round(longest, 3)})
        elif longest > no_observation_s:
            intervals.append({"state": "unknown", "t_start": a["t_end"], "t_end": b["t_start"], "kind": "no_observation"})
    if missing_yellow_guard_s > 0:
        guarded = []
        for k, iv in enumerate(intervals):
            previous_colour = next((x["state"] for x in reversed(intervals[:k]) if x["kind"] == "observed"), None)
            if iv["kind"] == "observed" and iv["state"] == "red" and previous_colour == "green":
                cut = min(iv["t_start"] + missing_yellow_guard_s, iv["t_end"])
                guarded.append({"state": "unknown", "t_start": iv["t_start"], "t_end": cut, "kind": "uncertain_colour",
                                "possible": ["yellow", "red"], "reason": "green followed directly by red: expected yellow phase not observed"})
                if cut < iv["t_end"]:
                    guarded.append(dict(iv, t_start=cut, note=f"first {missing_yellow_guard_s}s withheld (possible misread yellow)"))
            else:
                guarded.append(iv)
        intervals = guarded
    return intervals


def run(frames, associator=None, voter=None):
    """frames: [{"t", "scene", "detections"}]. Returns per-frame {"t", "raw": {hid: cls}, "voted": {hid: state}, "boxes": {hid: box}}."""
    associator = associator or LightHeadAssociator()
    voter = voter or PhaseVoter()
    out = []
    for f in frames:
        observed = associator.update(f["t"], f["detections"]) if f["scene"] else {}
        voted = voter.update(f["t"], observed)
        out.append({"t": f["t"], "scene": f["scene"], "raw": {h: v[0] for h, v in observed.items()},
                    "voted": voted, "boxes": {h: v[2] for h, v in observed.items()}})
    return out, associator


class RoiHeadAssociator:
    """Per-camera configured signal heads: each ROI is one head. In every frame the most confident
    detection whose centre lies inside the (slightly enlarged) ROI is that head's observation;
    detections outside every ROI are ignored."""

    def __init__(self, rois, attach_conf=0.1, grow=0.2):
        self.rois, self.attach_conf = [], attach_conf
        for roi in rois:
            x1, y1, x2, y2 = roi["xyxy"]
            gx, gy = grow * (x2 - x1), grow * (y2 - y1)
            self.rois.append((roi["id"], [x1 - gx, y1 - gy, x2 + gx, y2 + gy]))
        self.heads = [Head(rid, box, 0.0) for rid, box in self.rois]

    def update(self, t, detections):
        out = {}
        for rid, (x1, y1, x2, y2) in self.rois:
            inside = [d for d in detections if d["conf"] >= self.attach_conf
                      and x1 <= (d["xyxy"][0] + d["xyxy"][2]) / 2 <= x2 and y1 <= (d["xyxy"][1] + d["xyxy"][3]) / 2 <= y2]
            if inside:
                d = max(inside, key=lambda d: d["conf"])
                out[rid] = (d["cls"], d["conf"], d["xyxy"])
                head = next(h for h in self.heads if h.head_id == rid)
                head.observations.append((t, d["cls"], d["conf"]))
                head.last_t = t
        return out
