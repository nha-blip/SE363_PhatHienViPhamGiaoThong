"""Red-light crossing rules for the pilot scope (spec §3.1, §9.3–9.5, FR-09–FR-11).

Inputs are predictions only: vehicle tracks (per camera session), signal intervals from
src.signal.phase_voting.signal_intervals, and an activated SceneConfig approach.

* Approach (FR-09): one per (track, approach); stopping and moving again stays one approach.
* Crossing (FR-10): the reference point changes from the "before" to the "after" side of the stop line
  while moving along the configured direction. The crossing time is an interval bounded by the last
  observation before and the first after; frame gaps widen it, nothing is interpolated.
* Signal check (FR-11): positive only if the whole crossing interval lies in an observed red interval of
  the controlling signal(s); wholly inside an observed green/yellow interval is negative; anything else
  (transition, no observation, off, signals disagreeing) is insufficient evidence.
"""
from dataclasses import dataclass, field

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "pickup", "minibus"}  # spec v1.1 §3.1
NOT_RED = {"green", "yellow"}
NO_OBSERVATION = "no signal observation at crossing time"


@dataclass
class TrackPoint:
    t: float
    frame: int
    track_id: int
    bbox: tuple  # xyxy, source pixels
    cls: str
    conf: float = 1.0


@dataclass
class RuleSettings:
    reference_point: str = "bottom_center"  # must be calibrated per camera (spec §9.3); see reference()
    confirm_px: float = 8.0                 # a crossing is confirmed once the reference point is this far past the line ...
    confirm_height_ratio: float = 0.5       # ... and at least this fraction of the bbox height
    confirm_max_px: float = 40.0            # ... capped, so large near-camera vehicles are not required to move > 40 px
    max_gap_s: float = 1.0                  # larger observation gaps around a crossing are reported
    straight_cos: float = 0.7               # movement must be within ~45° of the configured direction
    min_travel_px: float = 20.0             # below this the vehicle is considered stopped
    truncate_margin_px: float = 5.0
    near_line_px: float = 40.0              # track ending this close to the line while moving -> lost, not compliant
    moving_window_s: float = 1.0
    moving_px_per_s: float = 15.0
    min_observed_s: float = 0.5             # shorter approaches without a crossing are not evidence of compliance
    direction_past_px: float = 60.0         # the direction check uses the track only up to this far past the line
                                            # (a vehicle that crosses straight and then curves away is still in scope)


@dataclass
class ApproachResult:
    approach_id: str
    track_id: int
    cls: str
    t_first: float
    t_last: float
    scope_status: str = "in_scope"
    assessment: str = None
    crossings: list = field(default_factory=list)
    uncertainty_reasons: list = field(default_factory=list)
    review_required: bool = False  # set by track_stitch.flag_stitched_crossings (spec §17)


def reference(bbox, mode):
    x1, y1, x2, y2 = bbox
    return {"bottom_center": ((x1 + x2) / 2, y2), "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "top_center": ((x1 + x2) / 2, y1)}[mode]


def side(point, line):
    (ax, ay), (bx, by) = line
    return (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def signal_state_over(intervals, t0, t1):
    """'red' / 'not_red' if one observed interval covers [t0, t1] entirely, else 'uncertain' with a reason."""
    for iv in intervals:
        if iv["kind"] == "observed" and iv["t_start"] <= t0 and t1 <= iv["t_end"]:
            if iv["state"] == "red":
                return "red", None
            if iv["state"] in NOT_RED:
                return "not_red", None
            return "uncertain", f"signal {iv['state']} does not prove right of way"
    touching = [iv for iv in intervals if iv["t_start"] <= t1 and t0 <= iv["t_end"]]
    if any(iv["kind"] == "transition" for iv in touching):
        return "uncertain", "crossing overlaps a colour transition"
    if any(iv["kind"] == "uncertain_colour" for iv in touching):
        return "uncertain", "crossing in the first seconds of red after green without an observed yellow (possible misread yellow)"
    if not touching:
        return "uncertain", NO_OBSERVATION
    return "uncertain", "crossing not inside a single observed signal interval"


ARROW_REASON = "signal head has a turn arrow: green may be the arrow only, not the through movement"


def arrow_guard(state, has_turn_arrow):
    """A head with a green turn arrow next to the round lamp: the light model has no arrow class, so a 'green'
    reading may be the arrow while the through movement is red. Such a head cannot prove right of way."""
    if has_turn_arrow and state[0] == "not_red":
        return "uncertain", ARROW_REASON
    return state


def combine_signals(states):
    if not states:
        return "uncertain", "no controlling signal mapped to this approach"
    # Heads controlling the same approach show the same phase. A head with no observation at all around the
    # crossing adds no evidence; the remaining heads decide (they must agree). Any other uncertainty still blocks.
    observed = [(s, r) for s, r in states if r != NO_OBSERVATION]
    if observed:
        states = observed
    values = {s for s, _ in states}
    if len(values) == 1 and values != {"uncertain"}:
        return values.pop(), None
    reasons = [r for _, r in states if r]
    if "uncertain" in values:
        return "uncertain", reasons[0] if reasons else "signal uncertain"
    return "uncertain", "controlling signals disagree"


def evaluate_approach(points, approach, signal_intervals_by_id, settings=RuleSettings()):
    """points: TrackPoints of ONE track, any order. approach: SceneConfig approach dict with
    approach_id, approach_roi, observation_roi, stop_line, direction, signal_ids (controlling signals)."""
    pts = sorted(points, key=lambda p: p.t)
    line, direction = approach["stop_line"], approach["direction"]
    norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    d = (direction[0] / norm, direction[1] / norm)
    mid = ((line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2)
    before_sign = 1 if side((mid[0] - 50 * d[0], mid[1] - 50 * d[1]), line) > 0 else -1
    obs = [p for p in pts if point_in_polygon(reference(p.bbox, settings.reference_point), approach["observation_roi"])]
    result = ApproachResult(approach["approach_id"], pts[0].track_id, majority_class(pts), pts[0].t, pts[-1].t)
    if not obs or not any(point_in_polygon(reference(p.bbox, settings.reference_point), approach["approach_roi"]) for p in obs):
        result.scope_status, result.assessment = "not_in_approach", None
        return result
    if result.cls not in VEHICLE_CLASSES:
        result.scope_status, result.assessment = "out_of_scope", None
        result.uncertainty_reasons.append(f"class {result.cls}")
        return result
    line_length = max(((line[1][0] - line[0][0]) ** 2 + (line[1][1] - line[0][1]) ** 2) ** 0.5, 1e-6)
    # signed distance in pixels from the stop line: > 0 before the line, < 0 after it
    sides = [before_sign * side(reference(p.bbox, settings.reference_point), line) / line_length for p in obs]
    # direction is judged on the approach and the crossing itself, not on what the vehicle does far past the line
    upto = next((k for k, s_value in enumerate(sides) if s_value < -settings.direction_past_px), len(obs))
    start = reference(obs[0].bbox, settings.reference_point)
    end = reference(obs[max(upto, 1) - 1].bbox, settings.reference_point)
    travel = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    if travel >= settings.min_travel_px:
        cos = ((end[0] - start[0]) * d[0] + (end[1] - start[1]) * d[1]) / travel
        if cos < settings.straight_cos:
            result.scope_status = "out_of_scope" if cos > -0.2 else "scope_uncertain"
            result.uncertainty_reasons.append("movement not straight along the configured direction" if cos > -0.2 else "moving against the configured direction")
            return result
    if sides[0] < -settings.truncate_margin_px:
        result.assessment = "insufficient_evidence"
        result.uncertainty_reasons.append("track starts after the stop line (truncated approach)")
        return result
    for k in range(1, len(obs)):
        if sides[k - 1] > 0 >= sides[k]:
            # confirm the vehicle really went through: the reference point must get clearly past the line
            # (not a vehicle stopped at the line whose box nudges over it, nor one that only noses over and stops)
            height = obs[k].bbox[3] - obs[k].bbox[1]
            needed = max(settings.confirm_px, min(settings.confirm_height_ratio * height, settings.confirm_max_px))
            run = []
            for s_value in sides[k:]:
                if s_value > settings.confirm_px:  # clearly back before the line: this attempt is over
                    break
                run.append(s_value)
            overshoot = -min(run)
            if overshoot < needed:
                continue  # nudged over the line and stopped/backed off; judged by the no-crossing branch below
            t0, t1 = obs[k - 1].t, obs[k].t
            state, reason = combine_signals([arrow_guard(signal_state_over(signal_intervals_by_id.get(s, []), t0, t1), s in approach.get("arrow_signal_ids", ()))
                                             for s in approach["signal_ids"]])
            status = {"red": "red_light_crossing", "not_red": "no_red_light_crossing"}.get(state, "insufficient_evidence")
            crossing = {"t_min": t0, "t_max": t1, "frame_before": obs[k - 1].frame, "frame_after": obs[k].frame, "signal": state, "status": status}
            if reason:
                crossing["reason"] = reason
            if t1 - t0 > settings.max_gap_s:  # the wider interval was already checked against the signal as a whole
                crossing["note"] = f"observation gap {t1 - t0:.2f}s around the line"
            result.crossings.append(crossing)
    statuses = [c["status"] for c in result.crossings]
    if "red_light_crossing" in statuses:
        result.assessment = "red_light_crossing"
    elif "insufficient_evidence" in statuses:
        result.assessment = "insufficient_evidence"
        result.uncertainty_reasons += [c.get("reason", "crossing uncertain") for c in result.crossings if c["status"] == "insufficient_evidence"]
    elif statuses:
        result.assessment = "no_red_light_crossing"
    else:  # never crossed: negative only if the approach was observed long enough and the vehicle was seen stopped
        last = reference(obs[-1].bbox, settings.reference_point)
        recent = [p for p in obs if p.t >= obs[-1].t - settings.moving_window_s]
        first_recent = reference(recent[0].bbox, settings.reference_point)
        speed = (((last[0] - first_recent[0]) * d[0] + (last[1] - first_recent[1]) * d[1])
                 / max(obs[-1].t - recent[0].t, 1e-6))
        if obs[-1].t - obs[0].t < settings.min_observed_s:
            result.assessment = "insufficient_evidence"
            result.uncertainty_reasons.append("approach observed too briefly")
        elif speed > settings.moving_px_per_s:
            result.assessment = "insufficient_evidence"
            distance = abs(sides[-1])
            result.uncertainty_reasons.append("track lost near the stop line while moving towards it" if distance <= settings.near_line_px
                                              else "track ended while still moving towards the line")
        else:
            result.assessment = "no_red_light_crossing"
            over = -min(sides)
            result.uncertainty_reasons.append("no crossing observed; vehicle seen stopped before the line" if over <= 0
                                              else f"stopped at the line (front at most {over:.1f}px past it, not a confirmed crossing)")
    return result


def majority_class(points):
    votes = {}
    for p in points:
        votes[p.cls] = votes.get(p.cls, 0.0) + p.conf
    return max(votes, key=votes.get)


def evaluate_scene(points, scene_config, signal_intervals_by_id, settings=RuleSettings()):
    """All tracks x all approaches of an activated SceneConfig. Returns ApproachResults that entered an approach."""
    by_track = {}
    for p in points:
        by_track.setdefault(p.track_id, []).append(p)
    mapping = {}
    arrows = {s["signal_id"] for s in scene_config["signals"] if s.get("has_turn_arrow")}
    for s in scene_config["signals"]:
        for aid in s["controls_approach_ids"]:
            mapping.setdefault(aid, []).append(s["signal_id"])
    results = []
    for approach in scene_config["approaches"]:
        approach = dict(approach, signal_ids=mapping.get(approach["approach_id"], []), arrow_signal_ids=arrows)
        for track_points in by_track.values():
            r = evaluate_approach(track_points, approach, signal_intervals_by_id, settings)
            if r.scope_status != "not_in_approach":
                results.append(r)
    return results
