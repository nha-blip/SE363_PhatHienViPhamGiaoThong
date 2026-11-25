import csv
import html
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.class_suggestions import summarize_votes, unique_matches
from src.tracking.io import load_frame_map, sha256, verify_snapshot
from src.tracking.yolo_adapter import YoloAdapter


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/Tracking363_working/sequences/vntraffic_001"
OUTPUT = ROOT / "runs/project363/tracking/class_preannotation_vntraffic_001"
SOURCE_MODEL = ROOT / "runs/local_training/vehicle_resume_epoch17_100/weights/best.pt"
NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_qa(tracks, summaries):
    from PIL import Image, ImageDraw

    previews = OUTPUT / "previews"
    previews.mkdir(exist_ok=True)
    sections = []
    manifest = []
    for identity, observations in sorted(tracks.items()):
        ordered = sorted(observations, key=lambda row: row["mot_frame_id"])
        selected = [ordered[position] for position in sorted({0, len(ordered) // 2, len(ordered) - 1})]
        sheet = Image.new("RGB", (960, 300), "#151515")
        drawing = ImageDraw.Draw(sheet)
        for column, row in enumerate(selected):
            image = Image.open(PILOT / "img1" / f"{row['mot_frame_id']:06d}.jpg").convert("RGB")
            left, top, width, height = row["bbox_xywh"]
            padding = max(width, height) * 0.15
            bounds = (max(0, int(left - padding)), max(0, int(top - padding)), min(image.width, int(left + width + padding + 1)), min(image.height, int(top + height + padding + 1)))
            crop = image.crop(bounds)
            ImageDraw.Draw(crop).rectangle((left - bounds[0], top - bounds[1], left + width - bounds[0], top + height - bounds[1]), outline="lime", width=3)
            scale = min(310 / crop.width, 220 / crop.height)
            crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
            sheet.paste(crop, (column * 320, 60))
            drawing.text((column * 320 + 5, 5), f"GT ID {identity} | source {row['source_frame_id']} | MOT {row['mot_frame_id']}", fill="white")
            drawing.text((column * 320 + 5, 25), f"Prediction: {row['candidate_class'] or 'PENDING'}", fill="yellow")
        crop_name = f"track_{identity:03d}_crops.jpg"
        sheet.save(previews / crop_name, quality=90)
        middle = selected[len(selected) // 2]
        image = Image.open(PILOT / "img1" / f"{middle['mot_frame_id']:06d}.jpg").convert("RGB")
        left, top, width, height = middle["bbox_xywh"]
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((left, top, left + width, top + height), outline="lime", width=5)
        drawing.text((max(0, left), max(0, top - 18)), f"GT ID {identity} - candidate {summaries[identity]['candidate_class'] or 'PENDING'}", fill="yellow", stroke_width=1, stroke_fill="black")
        image.thumbnail((1280, 720))
        frame_name = f"track_{identity:03d}_frame.jpg"
        image.save(previews / frame_name, quality=90)
        title = f"Track {identity}: candidate {summaries[identity]['candidate_class'] or 'PENDING'}; majority {summaries[identity]['majority_ratio']:.0%}; mixed={summaries[identity]['mixed_predictions']}"
        sections.append(f'<section><h2>{html.escape(title)}</h2><img loading="lazy" src="previews/{crop_name}"><details><summary>Full frame - GT bbox only</summary><img loading="lazy" src="previews/{frame_name}"></details></section>')
        manifest.append({"track_id": identity, "source_frames": [row["source_frame_id"] for row in selected], "mot_frames": [row["mot_frame_id"] for row in selected], "crops": crop_name, "full_frame": frame_name})
    (OUTPUT / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>VNTraffic class QA</title><style>body{background:#181818;color:white;font:16px sans-serif;margin:24px}img{max-width:100%}section{border-top:1px solid #666;padding:16px}</style><h1>Class suggestions only — HUMAN QA REQUIRED</h1><p>Green boxes and IDs are approved GT, unchanged. Prediction is NOT an approved semantic label. Check all tracks, especially mixed/PENDING. Source frame = MOT frame - 1.</p>' + "\n".join(sections), encoding="utf-8")
    write_json(OUTPUT / "qa_manifest.json", manifest)


def main():
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(2)
    snapshot = json.loads((ROOT / "runs/project363/tracking/work_5_3_integration_snapshot.json").read_text(encoding="utf-8"))
    verify_snapshot(snapshot, ROOT)
    sequence = json.loads((PILOT / "metadata/sequence.json").read_text(encoding="utf-8"))
    frames = load_frame_map(PILOT / "metadata/frame_map.csv", sequence)
    records = [json.loads(line) for line in (PILOT / "metadata/annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 16035 or len(frames) != 501 or any(row["class"] is not None for row in records):
        raise ValueError("Pilot GT differs from approved pending-class baseline")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    before = sha256(SOURCE_MODEL)
    checkpoint = OUTPUT / "vehicle_best_snapshot.pt"
    shutil.copyfile(SOURCE_MODEL, checkpoint)
    if sha256(checkpoint) != before or sha256(SOURCE_MODEL) != before:
        raise RuntimeError("Training checkpoint changed during copy; no inference performed")
    model = YOLO(str(checkpoint))
    if model.names != NAMES:
        raise ValueError("Checkpoint class namespace mismatch")
    model_epoch = model.ckpt.get("epoch")
    write_json(OUTPUT / "checkpoint_provenance.json", {"source": str(SOURCE_MODEL), "sha256": before, "checkpoint_epoch_zero_based": model_epoch, "completed_epochs": model_epoch + 1 if isinstance(model_epoch, int) and model_epoch >= 0 else None, "ultralytics": ultralytics.__version__, "torch": torch.__version__, "device": "cpu", "cpu_threads": 2, "snapshot_hash": snapshot["snapshot_hash"], "created_at_utc": datetime.now(timezone.utc).isoformat()})
    by_frame = defaultdict(list)
    for row in records:
        by_frame[row["mot_frame_id"]].append(row)
    tracks = defaultdict(list)
    adapter = YoloAdapter(NAMES, minimum_confidence=0.25)
    with (OUTPUT / "observation_suggestions.jsonl").open("x", encoding="utf-8") as suggestions, (OUTPUT / "detections.jsonl").open("x", encoding="utf-8") as predictions:
        for frame in frames:
            result = model.predict(str(PILOT / "img1" / f"{frame.mot_frame_id:06d}.jpg"), device="cpu", imgsz=640, conf=0.25, iou=0.7, max_det=300, save=False, verbose=False)[0]
            adapted = adapter.adapt(result, frame)
            predictions.write(json.dumps(adapted.to_dict(), allow_nan=False) + "\n")
            observations = by_frame[frame.mot_frame_id]
            matches = unique_matches(observations, adapted.detections)
            for row_index, original in enumerate(observations):
                row = dict(original)
                row.update(candidate_class=None, detector_confidence=None, match_iou=None, human_qa_status="REVIEW_REQUIRED")
                if row_index in matches:
                    detection_index, score = matches[row_index]
                    detection = adapted.detections[detection_index]
                    row.update(candidate_class=detection.class_name, detector_confidence=detection.confidence, match_iou=score)
                suggestions.write(json.dumps(row, allow_nan=False) + "\n")
                tracks[row["track_id"]].append(row)
            if frame.mot_frame_id % 50 == 0:
                print(f"CLASS_PREANNOTATION: {frame.mot_frame_id}/{len(frames)} frames", flush=True)
    summaries = {identity: summarize_votes(observations) for identity, observations in tracks.items()}
    write_json(OUTPUT / "track_class_suggestions.json", [{"track_id": identity, **value} for identity, value in sorted(summaries.items())])
    with (OUTPUT / "human_review.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "candidate_class", "suggested_class", "matched_observations", "majority_ratio", "mixed_predictions", "approved_class", "decision", "notes"])
        for identity, value in sorted(summaries.items()):
            writer.writerow([identity, value["candidate_class"], value["suggested_class"], value["matched_observations"], value["majority_ratio"], value["mixed_predictions"], "", "REVIEW_REQUIRED", ""])
    create_qa(tracks, summaries)
    verify_snapshot(snapshot, ROOT)
    if sha256(checkpoint) != before:
        raise RuntimeError("Fixed model snapshot modified")
    report = {"automated_execution_status": "PASS", "human_qa_status": "REVIEW_REQUIRED", "frames": len(frames), "gt_observations": len(records), "tracks": len(tracks), "matched_observations": sum(value["matched_observations"] for value in summaries.values()), "tracks_with_suggestion": sum(value["suggested_class"] is not None for value in summaries.values()), "tracks_without_suggestion": sum(value["suggested_class"] is None for value in summaries.values()), "tracks_with_mixed_predictions": sum(value["mixed_predictions"] for value in summaries.values()), "qa_previews": len(tracks) * 2, "gt_snapshot_unchanged": True, "semantic_gt_classes": "PENDING", "bytetrack_executed": False, "dataset_role": "PILOT_DEV", "final_dataset_created": False, "checkpoint_sha256": before}
    write_json(OUTPUT / "report.json", report)
    (OUTPUT / "README.md").write_text("# VNTraffic semantic class QA\n\nOpen index.html. Review every track's crops and full frame; check class, mixed predictions, and PENDING cases. Fill approved_class and decision in human_review.csv, or report decisions to the agent. No decisions are applied automatically. Original GT bbox, track IDs, frame mapping, and null classes remain unchanged; no ByteTrack or final dataset/split was created.\n\nPredictions use a fixed best.pt snapshot on CPU (2 threads); the running GPU training job is not stopped. Full-frame detections are matched conservatively at IoU >= 0.5, rejecting all ambiguous associations. A track suggestion requires >=3 matched observations and >=80% majority among matches, NOT calibrated confidence or coverage. All tracks require human QA, including strong majorities; unmatched observations stay PENDING. No gap repair or ID generation. Potential HCMC/VNTraffic camera overlap remains unresolved; this is not an unbiased benchmark.\n\nMachine evidence: report.json, checkpoint_provenance.json, detections.jsonl, observation_suggestions.jsonl, track_class_suggestions.json. Sources: https://docs.ultralytics.com/modes/predict/\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
