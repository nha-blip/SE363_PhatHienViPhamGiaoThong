"""Versioned SceneConfig (spec §9.2, FR-05).

A config is stored as one JSON file per camera and version. It can be saved as a draft at any time,
but `activation_problems()` must be empty before it is used for violation rules: stop line, approach
ROI, direction, signal ROIs and a confirmed signal-to-approach mapping are all required.
Coordinates are pixels of the reference image; the reference image hash detects a changed view source.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "scene_config.v1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_config(camera_id, version, reference_image, image_size, signals, created_by, transform=None, notes=None):
    return {
        "schema_version": SCHEMA_VERSION, "scene_config_id": f"{camera_id}:v{version}", "camera_id": camera_id, "version": version,
        "status": "DRAFT", "created_by": created_by, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_image": {"path": str(reference_image), "sha256": sha256_file(reference_image), "width": image_size[0], "height": image_size[1]},
        "coordinate_system": "pixels of the reference image", "source_transform": transform,
        "signals": [{"signal_id": s["signal_id"], "roi_xyxy": s["roi_xyxy"], "controls_approach_ids": s.get("controls_approach_ids", []),
                     "mapping_status": s.get("mapping_status", "UNCONFIRMED")} for s in signals],
        "approaches": [],  # each: approach_id, approach_roi (polygon), observation_roi, stop_line [[x,y],[x,y]], direction [dx,dy]
        "activation": None, "notes": notes or [],
    }


def _inside(box, width, height):
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def activation_problems(config):
    """Reasons the config may not be activated for rules (empty list = can activate).

    Every approach needs ROIs, a non-degenerate stop line and direction, and at least one controlling
    signal with a confirmed mapping. Signals that control no drawn approach are allowed (e.g. heads for
    other directions); they are reported by unmapped_signals() for information only.
    """
    problems = []
    w, h = config["reference_image"]["width"], config["reference_image"]["height"]
    if not config["signals"]:
        problems.append("no signal ROI")
    for s in config["signals"]:
        if not _inside(s["roi_xyxy"], w, h):
            problems.append(f"{s['signal_id']}: ROI outside the reference image")
        if s["controls_approach_ids"] and s["mapping_status"] != "CONFIRMED":
            problems.append(f"{s['signal_id']}: signal-to-approach mapping not confirmed")
    if not config["approaches"]:
        problems.append("no approach with stop line, ROI and direction")
    known = {a.get("approach_id") for a in config["approaches"]}
    for a in config["approaches"]:
        aid = a.get("approach_id")
        for key in ("approach_roi", "observation_roi", "stop_line", "direction"):
            if not a.get(key):
                problems.append(f"{aid}: missing {key}")
        if a.get("approach_roi") and len(a["approach_roi"]) < 3:
            problems.append(f"{aid}: approach_roi needs at least 3 points")
        if a.get("stop_line"):
            (x0, y0), (x1, y1) = a["stop_line"]
            if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 < 10:
                problems.append(f"{aid}: stop_line shorter than 10 px (two clicks on the same spot?)")
        if a.get("direction") and (a["direction"][0] ** 2 + a["direction"][1] ** 2) ** 0.5 < 0.5:
            problems.append(f"{aid}: direction is not a unit vector")
        controlling = [s for s in config["signals"] if aid in s["controls_approach_ids"] and s["mapping_status"] == "CONFIRMED"]
        if not controlling:
            problems.append(f"{aid}: no confirmed controlling signal")
    for s in config["signals"]:
        for aid in s["controls_approach_ids"]:
            if aid not in known:
                problems.append(f"{s['signal_id']}: controls unknown approach {aid}")
    return problems


def _segment_distance(point, line):
    (ax, ay), (bx, by) = line
    dx, dy = bx - ax, by - ay
    u = max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / ((dx * dx + dy * dy) or 1e-9)))
    return ((point[0] - ax - u * dx) ** 2 + (point[1] - ay - u * dy) ** 2) ** 0.5


def _inside_polygon(point, polygon):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def approach_warnings(config, max_gap_px=6.0, min_depth_px=60.0):
    """Drawing mistakes that do not block activation but make vehicles invisible to the rules (news pilot 16/11):
    an approach ROI that does not reach its stop line (vehicles first seen in the gap are never in the ROI) and
    an ROI that is only a thin strip along the travel direction (fast or late-detected vehicles skip it)."""
    warnings = []
    for a in config["approaches"]:
        roi, line, d = a.get("approach_roi"), a.get("stop_line"), a.get("direction")
        if not (roi and line and d):
            continue
        aid = a.get("approach_id")
        edges = list(zip(roi, roi[1:] + roi[:1]))
        # distance from the stop line to the ROI: the closest pair of points among vertices and line endpoints
        gap = min(min(_segment_distance(v, line) for v in roi), min(_segment_distance(e, edge) for e in line for edge in edges))
        samples = [(line[0][0] + (line[1][0] - line[0][0]) * k / 20, line[0][1] + (line[1][1] - line[0][1]) * k / 20) for k in range(21)]
        if any(_inside_polygon(p, roi) for p in samples) or any(_segment_distance(p, edge) < 1e-6 for p in samples for edge in edges):
            gap = 0.0  # the line touches or crosses the ROI
        if gap > max_gap_px:
            warnings.append(f"{aid}: vùng làn cách vạch dừng {gap:.0f} px (xe nằm trong khe này không được xét)")
        (ax, ay), (bx, by) = line  # depth = extent of the ROI perpendicular to the stop line
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 or 1.0
        across = [((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / length for x, y in roi]
        image = config.get("reference_image")
        at_edge = bool(image) and any(min(x, y, image["width"] - x, image["height"] - y) <= 5 for x, y in roi)
        if max(across) - min(across) < min_depth_px and not at_edge:  # a line near the image edge only leaves a thin strip
            warnings.append(f"{aid}: vùng làn chỉ sâu {max(across) - min(across):.0f} px tính từ vạch dừng (nên kéo từ mép ảnh tới vạch dừng)")
    return warnings


def unmapped_signals(config):
    return [s["signal_id"] for s in config["signals"] if not s["controls_approach_ids"]]


def load_active(path):
    """Load a config for rule execution; re-validates instead of trusting the stored status."""
    config = load(path)
    problems = activation_problems(config)
    if config.get("status") != "ACTIVE" or problems:
        raise ValueError(f"{config.get('scene_config_id')} cannot be used for rules: status={config.get('status')} {problems}")
    return config


def save(config, folder):
    path = Path(folder) / f"{config['camera_id']}_v{config['version']}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; a change must be saved as a new version")
    path.write_text(json.dumps(config, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema {config.get('schema_version')}")
    return config
