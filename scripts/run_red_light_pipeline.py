"""End-to-end pilot on one camera clip: vehicle tracking + signal timeline + red-light rules + QA video.

Uses the chosen models/configs: YOLOv8s @960 + tuned ByteTrack (configs/tracking/bytetrack_v8s_imgsz960_tuned_20261114.yaml),
light model A @960 with the SceneConfig signal ROIs, PhaseVoter + signal_intervals, src.rules.red_light.
Frames are sampled at SAMPLE_FPS. iHanoi screen recordings: the scene band must match the reference
image height of the SceneConfig; other frames are quality gaps (fed to the tracker as empty frames,
never interpolated). Outputs go to runs/project363/red_light_pilot/<scene_config_id>/.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from external_qa_extract_hanoi_20261113 import scene_band  # noqa: E402
from light_phase_extract_detect_20261114 import CLIPS, SEQUENCES  # noqa: E402
from build_roi_classifier_detections import a_in_roi, classify, cut, grow  # noqa: E402
from scripts.run_tracking_pilot import detector_names  # noqa: E402
from src.rules.red_light import RuleSettings, TrackPoint, evaluate_scene  # noqa: E402
from src.rules.track_stitch import StitchSettings, flag_stitched_crossings, stitch_tracks  # noqa: E402
from src.scene.config import load_active  # noqa: E402
from src.signal.lamp_position import read_by_position  # noqa: E402
from src.signal.phase_voting import PhaseVoter, RoiHeadAssociator, signal_intervals  # noqa: E402
from src.tracking.bytetrack_adapter import create_ultralytics_adapter  # noqa: E402
from src.tracking.pilot_runtime import LoggedBoundsBackend  # noqa: E402
from src.tracking.schemas import FrameInfo  # noqa: E402
from src.tracking.yolo_adapter import YoloAdapter  # noqa: E402

TRACK_CONFIG = ROOT / "configs/tracking/bytetrack_v8s_imgsz960_tuned_20261114.yaml"
LIGHT_WEIGHTS = ROOT / "weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt"
COLOUR_WEIGHTS = ROOT / "weights/light_colour_cls_20261117/best.pt"  # per-ROI colour classifier (light reader k1b)
LIGHT_NAMES = ["red", "yellow", "green", "off"]
OUT = ROOT / "runs/project363/red_light_pilot"
SAMPLE_FPS = 10
COLOURS = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 200, 0), "off": (160, 160, 160), "unknown": (255, 255, 255)}
VERDICT = {"red_light_crossing": (0, 0, 255), "insufficient_evidence": (0, 200, 255), "no_red_light_crossing": (0, 200, 0)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


TIMEBASE = {}  # camera_id -> how frame times were obtained (spec §9.1), written to the run manifest


def decoder_time(pts_s, previous, nominal):
    """Source time of a decoded frame: the decoder PTS when valid (spec §9.1: never the nominal FPS when a PTS exists).
    Returns (t, used_pts). A missing, negative or non-increasing PTS falls back to the nominal time."""
    if pts_s is not None and pts_s >= 0 and (previous is None or pts_s > previous):
        return pts_s, True
    return nominal, False


def frames_for(camera_id, ref_w, ref_h):
    """Yields (t, source_index, image or None, reason); t is the source time in seconds."""
    if camera_id in SEQUENCES:
        folder = SEQUENCES[camera_id]
        fps = float([l for l in (folder / "seqinfo.ini").read_text().splitlines() if l.startswith("frameRate")][0].split("=")[1])
        files = sorted((folder / "img1").glob("*.jpg"))
        step = max(1, round(fps / SAMPLE_FPS))
        TIMEBASE[camera_id] = {"timebase": "nominal_fps_image_sequence", "fps": fps, "timebase_verified": False}
        for k in range(0, len(files), step):
            yield round(k / fps, 3), k, cv2.imread(str(files[k])), None
        return
    cap = cv2.VideoCapture(str(CLIPS[camera_id]))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_y0 = None  # fixed crop window: where the SceneConfig reference image sits inside the phone screen
    t, last_pts, fallbacks, sampled = 0.0, None, 0, 0
    TIMEBASE[camera_id] = info = {"timebase": "decoder_pts", "nominal_fps": fps, "capture_method": "phone_screen_recording",
                                  "timebase_verified": False}
    while t * fps < n:
        index = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        source_t, used = decoder_time(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, last_pts, t)
        last_pts, fallbacks, sampled = (source_t if used else last_pts), fallbacks + (not used), sampled + 1
        info.update(pts_fallback_frames=fallbacks, sampled_frames=sampled)
        y0, y1 = scene_band(frame)
        if frame.shape[1] != ref_w or y1 - y0 < ref_h - 15:
            yield round(source_t, 3), index, None, f"scene band {y1 - y0}px, reference {ref_h}px (app screen, map or layout change)"
        else:
            if ref_y0 is None:
                located, score = locate_reference(frame, y0, y1) if abs((y1 - y0) - ref_h) <= 15 else (None, 0.0)
                if score < 0.5:
                    yield round(source_t, 3), index, None, f"reference view not found in this frame (match {score:.2f})"
                    t += 1 / SAMPLE_FPS
                    continue
                ref_y0 = located
            if not (y0 - 3 <= ref_y0 and ref_y0 + ref_h <= y1 + 3):
                yield round(source_t, 3), index, None, f"scene band {y0}-{y1} does not contain the reference window {ref_y0}-{ref_y0 + ref_h}"
            else:
                crop = frame[ref_y0:ref_y0 + ref_h]
                if crop.std() < 25 or (crop.min(axis=2) > 225).mean() > 0.15:
                    yield round(source_t, 3), index, None, "black/white or flat frame"
                else:
                    yield round(source_t, 3), index, crop, None
        t += 1 / SAMPLE_FPS
    cap.release()
    if ref_y0 is None:
        raise ValueError("reference view never found in the clip; the camera view may have changed (PTZ) - redraw the SceneConfig")


REFERENCE = {}


def assess(points, config, intervals, gap_times):
    """Stitch ID switches at the stop lines (src.rules.track_stitch), then apply the red-light rules."""
    stitched, joins = stitch_tracks(points, [a["stop_line"] for a in config["approaches"]], gap_times, StitchSettings())
    return flag_stitched_crossings(evaluate_scene(stitched, config, intervals, RuleSettings()), joins), joins, stitched


def write_overlay(path, rows, timeline, config, stitched, verdict, ref_w, ref_h):
    by_frame = {}
    for p in stitched:
        by_frame.setdefault(p.frame, []).append(p)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), SAMPLE_FPS, (ref_w, ref_h))
    for k, ((t, image), tl) in enumerate(zip(rows, timeline)):
        if image is None:
            blank = np.zeros((ref_h, ref_w, 3), "uint8")
            cv2.putText(blank, f"t={t:.1f}s QUALITY GAP", (20, ref_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            writer.write(blank)
        else:
            writer.write(draw(image, t, config, by_frame.get(k, []), tl["voted"], verdict))
    writer.release()


def locate_reference(frame, y0, y1):
    """Vertical offset of the reference image inside the phone frame (template matching on the scene band)."""
    ref = REFERENCE["image"]
    lo, hi = max(0, y0 - 20), min(frame.shape[0], y1 + 20)
    match = cv2.matchTemplate(cv2.cvtColor(frame[lo:hi], cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
    _, score, _, (_, dy) = cv2.minMaxLoc(match)
    return lo + dy, score


def draw(image, t, config, tracks, states, verdict_by_track):
    img = image.copy()
    for a in config["approaches"]:
        cv2.polylines(img, [np.array(a["approach_roi"], np.int32)], True, (0, 255, 120), 1)
        (x0, y0), (x1, y1) = a["stop_line"]
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    for s in config["signals"]:
        x1, y1, x2, y2 = map(int, s["roi_xyxy"])
        state = states.get(s["signal_id"], "unknown")
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOURS[state], 3)
        cv2.putText(img, state, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS[state], 2)
    for p in tracks:
        x1, y1, x2, y2 = map(int, p.bbox)
        colour = VERDICT.get(verdict_by_track.get(p.track_id), (255, 200, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, f"{p.track_id} {p.cls}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    cv2.putText(img, f"t={t:6.1f}s  {config['scene_config_id']}", (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--light-reader", choices=["k1bpos", "k1b", "a"], default="k1bpos",
                        help="k1b: model A's box in each signal ROI re-read by the colour classifier ('none' or no box -> whole ROI); "
                             "k1bpos: lit-lamp position first for signals with lamp_layout, else k1b; a: model A only")
    a = parser.parse_args()
    config = load_active(a.scene_config)
    REFERENCE["image"] = cv2.imread(str(ROOT / config["reference_image"]["path"]))
    ref_w, ref_h = config["reference_image"]["width"], config["reference_image"]["height"]
    tracking = yaml.safe_load(TRACK_CONFIG.read_text(encoding="utf-8"))
    det = tracking["detector"]
    if sha256(ROOT / det["model_path"]) != det["expected_sha256"]:
        raise ValueError("vehicle checkpoint hash mismatch")
    from ultralytics import YOLO
    vehicle_model, light_model = YOLO(str(ROOT / det["model_path"])), YOLO(str(LIGHT_WEIGHTS))
    NAMES = detector_names(vehicle_model.names)
    tracker = create_ultralytics_adapter(tracking["tracker"], NAMES, SAMPLE_FPS, allow_execution=True)
    bounded = LoggedBoundsBackend(tracker.backend, ref_w, ref_h)
    tracker.backend = bounded
    adapter = YoloAdapter(NAMES, det["minimum_confidence"])
    associator = RoiHeadAssociator([{"id": s["signal_id"], "xyxy": s["roi_xyxy"]} for s in config["signals"]])
    voter = PhaseVoter()
    colour_model = YOLO(str(COLOUR_WEIGHTS)) if a.light_reader in ("k1b", "k1bpos") else None
    out_dir = OUT / (config["scene_config_id"].replace(":", "_") + ("" if a.light_reader == "a" else f"__{a.light_reader}"))
    out_dir.mkdir(parents=True, exist_ok=False)
    rows, gaps, points, timeline = [], [], [], []
    all_frames = list(frames_for(config["camera_id"], ref_w, ref_h))
    for k, (t, source_index, image, reason) in enumerate(all_frames):
        info = FrameInfo(config["camera_id"], k, k + 1, ref_w, ref_h, float(SAMPLE_FPS), len(all_frames))
        if image is None:
            gaps.append({"t": t, "source_index": source_index, "reason": reason})
            from src.tracking.schemas import DetectionResult
            tracker.update(DetectionResult(info))
            timeline.append({"t": t, "scene": False, "raw": {}, "voted": voter.update(t, {})})
            rows.append((t, None, []))
            continue
        pred = vehicle_model.predict(image, device=det["device"], imgsz=det["imgsz"], conf=det["minimum_confidence"], iou=det["nms_iou"],
                                     max_det=det["max_det"], verbose=False)[0]
        result = tracker.update(adapter.adapt(pred, info))
        frame_points = [TrackPoint(t, k, tr.track_id, tuple(tr.bbox_xyxy), tr.class_name, tr.confidence) for tr in result.tracks]
        points += frame_points
        lights = light_model.predict(image, imgsz=960, conf=0.1, verbose=False)[0]
        detections = [{"xyxy": b, "cls": LIGHT_NAMES[int(c)], "conf": float(s)}
                      for b, c, s in zip(lights.boxes.xyxy.tolist(), lights.boxes.cls.tolist(), lights.boxes.conf.tolist())]
        if colour_model is not None:  # one reading per signal ROI, same logic as scripts/build_roi_classifier_detections.py (k1b / k1bpos)
            readings = []
            for s in config["signals"]:
                roi, box = s["roi_xyxy"], a_in_roi(detections, s["roi_xyxy"])
                by_pos = read_by_position(cut(image, roi), s.get("lamp_layout")) if a.light_reader == "k1bpos" and s.get("lamp_layout") else None
                got = by_pos or (classify(colour_model, cut(image, grow(box["xyxy"], 0.3)), 0.5) if box else None) or                     classify(colour_model, cut(image, grow(roi, 0.2)), 0.5)
                if got:
                    readings.append({"xyxy": roi, "cls": got[0], "conf": got[1]})
            detections = readings
        observed = associator.update(t, detections)
        timeline.append({"t": t, "scene": True, "raw": {h: v[0] for h, v in observed.items()}, "voted": voter.update(t, observed)})
        rows.append((t, image, frame_points))
    intervals = {s["signal_id"]: signal_intervals(timeline, s["signal_id"]) for s in config["signals"]}
    results, joins, stitched = assess(points, config, intervals, [g["t"] for g in gaps])
    verdict = {r.track_id: r.assessment for r in results if r.assessment}
    write_overlay(out_dir / "qa_overlay.mp4", [(t, image) for t, image, _ in rows], timeline, config, stitched, verdict, ref_w, ref_h)
    events = [{"approach_id": r.approach_id, "track_id": r.track_id, "class": r.cls, "t_first": r.t_first, "t_last": r.t_last,
               "scope_status": r.scope_status, "assessment": r.assessment, "crossings": r.crossings, "uncertainty_reasons": r.uncertainty_reasons,
               "review_required": r.review_required}
              for r in results]
    manifest = {"scene_config_id": config["scene_config_id"], "scene_config_sha256": sha256(a.scene_config), "source": str(CLIPS.get(config["camera_id"]) or SEQUENCES[config["camera_id"]]),
                "sample_fps": SAMPLE_FPS, "timebase": TIMEBASE.get(config["camera_id"]), "frames": len(all_frames), "quality_gaps": len(gaps), "tracker_bounds_events": len(bounded.events), "tracking_config": str(TRACK_CONFIG),
                "vehicle_checkpoint_sha256": det["expected_sha256"], "light_checkpoint_sha256": sha256(LIGHT_WEIGHTS), "light_reader": a.light_reader,
                "colour_checkpoint_sha256": sha256(COLOUR_WEIGHTS) if colour_model is not None else None, "rule_settings": vars(RuleSettings()),
                "stitch_settings": vars(StitchSettings()), "track_joins": joins, "timebase_verified": config["camera_id"] in SEQUENCES, "status": "PREDICTIONS_NOT_GT"}
    for name, value in (("events.json", events), ("signal_intervals.json", intervals), ("quality_gaps.json", gaps), ("manifest.json", manifest),
                        ("signal_timeline.json", timeline)):  # tracks.jsonl written below
        (out_dir / name).write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")
    with (out_dir / "tracks.jsonl").open("w", encoding="utf-8") as f:
        for p in points:
            f.write(json.dumps({"t": p.t, "frame": p.frame, "track_id": p.track_id, "bbox_xyxy": [round(v, 1) for v in p.bbox],
                                "class": p.cls, "conf": round(p.conf, 4)}) + "\n")
    counts = {}
    for e in events:
        key = e["assessment"] or e["scope_status"]
        counts[key] = counts.get(key, 0) + 1
    print(config["scene_config_id"], f"frames {len(all_frames)} gaps {len(gaps)} approaches {len(events)}", counts,
          "| signal:", {k: [(i["kind"][0], i["state"][:1], i["t_start"], i["t_end"]) for i in v] for k, v in intervals.items()}, flush=True)


if __name__ == "__main__":
    main()
