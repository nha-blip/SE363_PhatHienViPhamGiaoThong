"""Re-apply the red-light rules to a finished pilot run with a new SceneConfig version (no detector, no GPU).

Reuses the run's raw tracks (tracks.jsonl), per-frame signal votes (signal_timeline.json) and quality gaps.
The signal ROIs of the new config must equal those of the run (the votes depend on them); stop lines,
approaches, mapping and the turn-arrow flags may differ. Signal intervals are recomputed with the current
src.signal.phase_voting.signal_intervals; track stitching and rules are the current ones.
Output: runs/project363/red_light_pilot/rules_rerun/<scene_config_id>/ (events.json, manifest.json, qa_overlay.mp4).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_red_light_pipeline import OUT, REFERENCE, assess, frames_for, sha256, write_overlay  # noqa: E402
from src.rules.red_light import RuleSettings, TrackPoint  # noqa: E402
from src.rules.track_stitch import StitchSettings  # noqa: E402
from src.scene.config import load, load_active  # noqa: E402
from src.signal.phase_voting import signal_intervals  # noqa: E402


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="finished pilot run folder")
    parser.add_argument("--scene-config", type=Path, required=True, help="ACTIVE SceneConfig (any version of the same camera)")
    parser.add_argument("--no-video", action="store_true")
    a = parser.parse_args()
    config = load_active(a.scene_config)
    old_manifest = json.loads((a.run / "manifest.json").read_text(encoding="utf-8"))
    old_config = load(ROOT / "configs/scene_configs" / (old_manifest["scene_config_id"].replace(":v", "_v") + ".json"))
    if old_config["camera_id"] != config["camera_id"]:
        raise ValueError("the run belongs to another camera")
    if [(s["signal_id"], s["roi_xyxy"]) for s in old_config["signals"]] != [(s["signal_id"], s["roi_xyxy"]) for s in config["signals"]]:
        raise ValueError("signal ROIs changed: the stored votes do not apply, rerun the full pipeline")
    points = []
    with (a.run / "tracks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            points.append(TrackPoint(r["t"], r["frame"], r["track_id"], tuple(r["bbox_xyxy"]), r["class"], r["conf"]))
    timeline = json.loads((a.run / "signal_timeline.json").read_text(encoding="utf-8"))
    gaps = json.loads((a.run / "quality_gaps.json").read_text(encoding="utf-8"))
    intervals = {s["signal_id"]: signal_intervals(timeline, s["signal_id"]) for s in config["signals"]}
    results, joins, stitched = assess(points, config, intervals, [g["t"] for g in gaps])
    out_dir = OUT / "rules_rerun" / config["scene_config_id"].replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=False)
    events = [{"approach_id": r.approach_id, "track_id": r.track_id, "class": r.cls, "t_first": r.t_first, "t_last": r.t_last,
               "scope_status": r.scope_status, "assessment": r.assessment, "crossings": r.crossings, "uncertainty_reasons": r.uncertainty_reasons,
               "review_required": r.review_required}
              for r in results]
    manifest = {"scene_config_id": config["scene_config_id"], "scene_config_sha256": sha256(a.scene_config), "rules_rerun_of": str(a.run.resolve().relative_to(ROOT.resolve())),
                "source_run_manifest": old_manifest, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rule_settings": vars(RuleSettings()), "stitch_settings": vars(StitchSettings()), "track_joins": joins, "status": "PREDICTIONS_NOT_GT"}
    for name, value in (("events.json", events), ("signal_intervals.json", intervals), ("manifest.json", manifest)):
        (out_dir / name).write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")
    if not a.no_video:
        REFERENCE["image"] = cv2.imread(str(ROOT / config["reference_image"]["path"]))
        w, h = config["reference_image"]["width"], config["reference_image"]["height"]
        rows = [(t, image) for t, _, image, _ in frames_for(config["camera_id"], w, h)]
        if len(rows) != len(timeline):
            raise ValueError(f"frame count {len(rows)} differs from the run ({len(timeline)})")
        verdict = {r.track_id: r.assessment for r in results if r.assessment}
        write_overlay(out_dir / "qa_overlay.mp4", rows, timeline, config, stitched, verdict, w, h)
    counts = {}
    for e in events:
        key = e["assessment"] or e["scope_status"]
        counts[key] = counts.get(key, 0) + 1
    red = [(e["track_id"], [c["t_min"] for c in e["crossings"] if c["status"] == "red_light_crossing"]) for e in events if e["assessment"] == "red_light_crossing"]
    print(config["scene_config_id"], f"approaches {len(events)}", counts, "joins", [(j["from_id"], j["to_id"]) for j in joins], "RED", red, flush=True)


if __name__ == "__main__":
    main()
