"""Convert the signal-head ROIs drawn on 14/11/2026 into versioned SceneConfig drafts (v1).

Signals only: stop lines, approaches and the signal-to-approach mapping are not drawn yet, so every
config stays DRAFT and activation_problems() lists what is missing.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scene.config import activation_problems, new_config, save  # noqa: E402

WORK = ROOT / "data/light_phase_eval_20261114"
OUT = ROOT / "configs/scene_configs"


def main():
    rois = json.loads((WORK / "signal_head_rois.json").read_text(encoding="utf-8"))
    for source, items in sorted(rois.items()):
        if not items or list(OUT.glob(f"{source}_v*.json")):
            continue  # nothing drawn, or this camera already has SceneConfig versions
        reference = sorted((WORK / source / "frames").glob("*.jpg"))[0]  # the tool's default background frame
        import cv2
        h, w = cv2.imread(str(reference)).shape[:2]
        meta = json.loads((WORK / source / "detections.json").read_text(encoding="utf-8"))
        transform = None
        from datetime import date
        notes = [f"signal ROIs drawn by the project owner in light_roi_tool.py; exported {date.today().isoformat()}"]
        if "vn_news_violation_clips" in meta["source"]:
            transform = {"type": "tv_news_segment", "source": meta["source"],
                         "note": "news overlays (presenter bottom-right, plate boxes) occlude part of the view"}
        if "iHanoi" in meta["source"]:
            transform = {"type": "scene_band_crop_of_phone_screen_recording", "source": meta["source"],
                         "note": "band detected per frame; timebase_verified=false"}
            notes.append("PTZ camera: a changed view invalidates this config")
        config = new_config(source, 1, reference.relative_to(ROOT), (w, h),
                            [{"signal_id": f"{source}:S{r['id']}", "roi_xyxy": [max(0.0, r["xyxy"][0]), max(0.0, r["xyxy"][1]),
                                                                       min(float(w), r["xyxy"][2]), min(float(h), r["xyxy"][3])]} for r in items],
                            created_by="project owner (ROIs); converted by export_signal_rois_to_scene_configs.py", transform=transform, notes=notes)
        config["reference_image"]["path"] = str(reference.relative_to(ROOT)).replace("\\", "/")
        path = save(config, OUT)
        print(path.name, len(items), "signals; cannot activate yet:", "; ".join(sorted(set(p.split(": ")[-1] for p in activation_problems(config)))))


if __name__ == "__main__":
    main()
