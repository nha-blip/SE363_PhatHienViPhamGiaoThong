import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.class_suggestions import overlap
from src.tracking.io import sha256
from src.tracking.pilot_runtime import diagnostic_matches
from src.tracking.schemas import TrackingResult


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "runs/project363/tracking/work_5_4_baseline"
PILOT = ROOT / "data/Tracking363_working/sequences/vntraffic_001"
GT = ROOT / "data/Tracking363_working/gt_revisions/vntraffic_001_remove_track68/metadata/annotations.jsonl"
OUTPUT = BASELINE / "candidate_analysis"


def pair_iou(first, second):
    left, top, right, bottom = first
    return overlap((left, top, right - left, bottom - top), second)


def candidates_for(gt_row, tracks):
    return sorted([{"prediction_id": track.track_id, "iou": overlap(gt_row["bbox_xywh"], track.bbox_xyxy), "bbox_xyxy": list(track.bbox_xyxy), "class_name": track.class_name, "confidence": track.confidence} for track in tracks], key=lambda item: (-item["iou"], item["prediction_id"]))[:4]


def create_sheet(case, gt_frames, predictions):
    target = case["gt_track_id"]
    center = case["mot_frame_id"]
    identities = (case["previous_prediction_id"], case["prediction_id"])
    frame_ids = [center - 1, center, min(501, center + 1)]
    boxes = []
    for frame_id in frame_ids:
        row = gt_frames[frame_id].get(target)
        if row:
            left, top, width, height = row["bbox_xywh"]
            boxes.append((left, top, left + width, top + height))
        boxes.extend(track.bbox_xyxy for track in predictions[frame_id].tracks if track.track_id in identities)
    region = (max(0, int(min(box[0] for box in boxes) - 40)), max(0, int(min(box[1] for box in boxes) - 40)), min(1920, int(max(box[2] for box in boxes) + 41)), min(1080, int(max(box[3] for box in boxes) + 41)))
    sheet = Image.new("RGB", (1920, 460), "#181818")
    for position, frame_id in enumerate(frame_ids):
        image = Image.open(PILOT / "img1" / f"{frame_id:06d}.jpg").convert("RGB")
        drawing = ImageDraw.Draw(image)
        row = gt_frames[frame_id].get(target)
        if row:
            left, top, width, height = row["bbox_xywh"]
            drawing.rectangle((left, top, left + width, top + height), outline="lime", width=2)
        for track in predictions[frame_id].tracks:
            if track.track_id in identities:
                color = "cyan" if track.track_id == identities[0] else "orange"
                drawing.rectangle(track.bbox_xyxy, outline=color, width=2)
                left, top = track.bbox_xyxy[:2]
                drawing.text((left, max(0, top - 12)), f"P{track.track_id}", fill=color, stroke_width=1, stroke_fill="black")
        crop = image.crop(region)
        scale = min(630 / crop.width, 380 / crop.height)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
        sheet.paste(crop, (position * 640, 70))
        drawing = ImageDraw.Draw(sheet)
        drawing.text((position * 640 + 5, 5), f"GT {target}: source {frame_id-1} / MOT {frame_id}", fill="white")
        drawing.text((position * 640 + 5, 25), f"Green GT | cyan P{identities[0]} | orange P{identities[1]}", fill="white")
        assigned = dict((gt_id, prediction_id) for gt_id, prediction_id, _ in diagnostic_matches(list(gt_frames[frame_id].values()), predictions[frame_id].tracks))
        drawing.text((position * 640 + 5, 45), f"Diagnostic GT match: {assigned.get(target, 'none')}", fill="yellow")
    filename = f"case_{case['case_id']:02d}_gt{target}_mot{center}.jpg"
    sheet.save(OUTPUT / filename, quality=95)
    return filename


def main():
    inputs = [GT, BASELINE / "tracking_results.jsonl", BASELINE / "diagnostic_id_changes.json", BASELINE / "raw_backend_outputs.jsonl", BASELINE / "human_tracking_qa_confirmation.json"]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    records = [json.loads(line) for line in GT.read_text(encoding="utf-8").splitlines()]
    gt_frames = defaultdict(dict)
    for row in records:
        gt_frames[row["mot_frame_id"]][row["track_id"]] = row
    results = [TrackingResult.from_dict(json.loads(line)) for line in (BASELINE / "tracking_results.jsonl").read_text(encoding="utf-8").splitlines()]
    predictions = {result.frame.mot_frame_id: result for result in results}
    raw = {row["mot_frame_id"]: row["raw_rows"] for row in map(json.loads, (BASELINE / "raw_backend_outputs.jsonl").read_text(encoding="utf-8").splitlines())}
    cases = json.loads((BASELINE / "diagnostic_id_changes.json").read_text(encoding="utf-8"))
    if len(predictions) != 501 or len(cases) != 31 or len(records) != 16034:
        raise ValueError("Unexpected baseline input")
    recreated = []
    previous = {}
    for frame_id, result in predictions.items():
        for gt_id, prediction_id, _ in diagnostic_matches(list(gt_frames[frame_id].values()), result.tracks):
            earlier = previous.get(gt_id)
            if earlier and earlier[0] == frame_id - 1 and earlier[1] != prediction_id:
                recreated.append((gt_id, earlier[0], frame_id, earlier[1], prediction_id))
            previous[gt_id] = (frame_id, prediction_id)
    if recreated != [(case["gt_track_id"], case["previous_mot_frame"], case["mot_frame_id"], case["previous_prediction_id"], case["prediction_id"]) for case in cases]:
        raise ValueError("Cannot reproduce original candidate list")
    OUTPUT.mkdir(exist_ok=False)
    patterns = Counter()
    details = []
    for number, original in enumerate(cases, 1):
        case = {"case_id": number, **original}
        gt_id, before, current = case["gt_track_id"], case["previous_mot_frame"], case["mot_frame_id"]
        old_id, new_id = case["previous_prediction_id"], case["prediction_id"]
        tables = {frame_id: {track.track_id: track for track in predictions[frame_id].tracks} for frame_id in (before, current)}
        coexisting = all(identity in tables[frame_id] for frame_id in (before, current) for identity in (old_id, new_id))
        case["both_prediction_ids_exist_in_both_frames"] = coexisting
        case["top_gt_iou_candidates_before"] = candidates_for(gt_frames[before][gt_id], predictions[before].tracks)
        case["top_gt_iou_candidates_after"] = candidates_for(gt_frames[current][gt_id], predictions[current].tracks)
        case["prediction_pair_iou_by_frame"] = {str(frame_id): pair_iou(tables[frame_id][old_id].bbox_xyxy, tables[frame_id][new_id].bbox_xyxy) if old_id in tables[frame_id] and new_id in tables[frame_id] else None for frame_id in (before, current)}
        previous_box = gt_frames[before][gt_id]["bbox_xywh"]
        current_box = gt_frames[current][gt_id]["bbox_xywh"]
        left, top, width, height = current_box
        case["adjacent_gt_bbox_iou"] = overlap(previous_box, (left, top, left + width, top + height))
        choices = {}
        for threshold in (0.3, 0.5, 0.7):
            choices[str(threshold)] = [dict((identity, prediction) for identity, prediction, _ in diagnostic_matches(list(gt_frames[frame_id].values()), predictions[frame_id].tracks, threshold)).get(gt_id) for frame_id in (before, current)]
        case["assigned_ids_at_iou_thresholds"] = choices
        raw_choices = []
        for frame_id in (before, current):
            raw_tracks = [SimpleNamespace(track_id=int(row[4]), bbox_xyxy=tuple(row[:4])) for row in raw[frame_id]]
            raw_choices.append(dict((identity, prediction) for identity, prediction, _ in diagnostic_matches(list(gt_frames[frame_id].values()), raw_tracks)).get(gt_id))
        case["raw_unclipped_assigned_ids"] = raw_choices
        case["clipping_changes_assignment"] = raw_choices != choices["0.5"]
        overlap_values = [value for value in case["prediction_pair_iou_by_frame"].values() if value is not None]
        if coexisting and overlap_values and min(overlap_values) >= 0.5:
            pattern = "COEXISTING_OVERLAPPING_PREDICTIONS_MATCH_OSCILLATION"
        elif coexisting:
            pattern = "COEXISTING_IDS_GT_ASSOCIATION_CHANGE"
        elif old_id not in tables[current] and new_id not in tables[before]:
            pattern = "OLD_DISAPPEARS_NEW_APPEARS"
        else:
            pattern = "PARTIAL_COEXISTENCE_ASSOCIATION_CHANGE"
        case["automated_evidence_pattern"] = pattern
        case["physical_identity_cause"] = "UNCONFIRMED"
        case["interpretation"] = "GT-to-prediction assignment changed while both P-IDs continued; not evidence of numeric P-ID replacement. Overlap may indicate duplicate predictions or nearby vehicles." if coexisting else "Possible fragmentation or association ambiguity; geometry alone cannot certify physical identity."
        case["human_feedback"] = "Owner reviewed listed segment and observed no same-vehicle P-ID change"
        case["official_id_switch_status"] = "NOT_EVALUATED"
        case["preview"] = create_sheet(case, gt_frames, predictions)
        patterns[pattern] += 1
        details.append(case)
    for path in inputs:
        if sha256(path) != hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError("Input changed during analysis")
    report = {"status": "ANALYSIS_COMPLETE_REVIEW_REQUIRED", "candidates": 31, "reproduced_candidates": 31, "evidence_pattern_counts": dict(patterns), "clipping_sensitive_cases": sum(case["clipping_changes_assignment"] for case in details), "official_id_switches": None, "all_inputs_unchanged": True, "input_sha256": hashes, "human_feedback_scope": "All 31 candidate segments; no observed same-vehicle P-ID changes", "conclusion": "Warnings track changes in per-frame GT-to-P assignment, not direct observation that a displayed P-ID transforms. Coexisting IDs and spatial overlap can cause matching oscillation; no blanket false-alarm or tracker-PASS classification.", "next": "Review enlarged case sheets, focusing on old-disappears/new-appears and overlapping duplicate predictions; define evaluable GT before metrics"}
    (OUTPUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "case_evidence.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    sections = [f'<section><h2>Case {case["case_id"]}: GT{case["gt_track_id"]}, P{case["previous_prediction_id"]} → P{case["prediction_id"]}, MOT {case["mot_frame_id"]}</h2><p>{case["automated_evidence_pattern"]}</p><img loading="lazy" src="{case["preview"]}"></section>' for case in details]
    (OUTPUT / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>ID candidate analysis</title><style>body{background:#181818;color:white;font:18px sans-serif;margin:20px}img{max-width:100%}section{border-top:1px solid #666}</style><h1>31 diagnostic candidates — evidence, not official ID switches</h1><p>Green = target GT. Cyan = earlier matched P-ID. Orange = later matched P-ID. Check whether two prediction boxes already coexist on one vehicle or different vehicles. GT/predictions unchanged. Human physical-identity confirmation is separate from automatic geometry patterns.</p>' + "\n".join(sections), encoding="utf-8")
    text = "# Work 5.4 candidate discrepancy analysis\n\n" + json.dumps(report, indent=2) + "\n\nMatching uses independent class-agnostic one-to-one IoU assignments per frame. A different assigned P-ID does not by itself prove a displayed label changed on the same physical vehicle. Both IDs can coexist, with duplicate/overlapping boxes or adjacent vehicles competing. However, stable displayed IDs do not rule out duplicate tracks or incorrect physical association. Old-disappears/new-appears cases require targeted inspection. No GT-gap ignore policy, official metric, or full tracking PASS is inferred.\n\nOpen index.html for before/current/after crop sheets. Full numerical evidence is in case_evidence.json. No YOLO/ByteTrack rerun or parameter change occurred.\n"
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
