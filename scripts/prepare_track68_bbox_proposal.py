import json
from pathlib import Path

from PIL import Image, ImageDraw

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/Tracking363_working/sequences/vntraffic_001"
OUTPUT = ROOT / "runs/project363/tracking/class_preannotation_vntraffic_001/track68_bbox_review"


def main():
    annotations = PILOT / "metadata/annotations.jsonl"
    original_hash = sha256(annotations)
    records = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    frame_rows = {row["track_id"]: row for row in records if row["mot_frame_id"] == 383}
    old_box = frame_rows[68]["bbox_xywh"]
    proposed = [0, 690, 62, 63]
    OUTPUT.mkdir(exist_ok=False)
    source = PILOT / "img1/000383.jpg"
    image = Image.open(source).convert("RGB")
    region = (0, 620, 300, 820)
    panels = []
    for title, boxes in (
        ("Existing GT: 68 red, 74 cyan, 72 yellow", [(68, old_box, "red"), (74, frame_rows[74]["bbox_xywh"], "cyan"), (72, frame_rows[72]["bbox_xywh"], "yellow")]),
        ("Proposal ONLY: 68 green; existing 74 cyan", [(68, proposed, "lime"), (74, frame_rows[74]["bbox_xywh"], "cyan")]),
    ):
        annotated = image.copy()
        drawing = ImageDraw.Draw(annotated)
        for identity, box, color in boxes:
            left, top, width, height = box
            drawing.rectangle((left, top, left + width, top + height), outline=color, width=1)
            drawing.text((max(0, left), top - 12 if identity != 74 else top + height + 2), str(identity), fill=color, stroke_width=1, stroke_fill="black")
        panel = Image.new("RGB", (900, 650), "#181818")
        ImageDraw.Draw(panel).text((10, 10), title, fill="white")
        panel.paste(annotated.crop(region).resize((900, 600), Image.Resampling.LANCZOS), (0, 50))
        panels.append(panel)
    preview = Image.new("RGB", (1800, 650))
    for position, panel in enumerate(panels):
        preview.paste(panel, (position * 900, 0))
    preview.save(OUTPUT / "comparison.png")
    full_frame = image.copy()
    left, top, width, height = proposed
    ImageDraw.Draw(full_frame).rectangle((left, top, left + width, top + height), outline="lime", width=2)
    full_frame.save(OUTPUT / "proposed_full_frame.jpg", quality=95)
    report = {"status": "REVIEW_REQUIRED", "source_frame_id": 382, "mot_frame_id": 383, "track_id": 68,
              "target": "Owner-confirmed white car on the left", "class": "car", "old_bbox_xywh": old_box,
              "proposed_bbox_xywh": proposed, "proposal_basis": "Visual inspection of native 1920x1080 frame; approximate boundaries for human approval, not YOLO or screenshot coordinates",
              "existing_track74_bbox_xywh": frame_rows[74]["bbox_xywh"],
              "existing_track72_bbox_xywh": frame_rows[72]["bbox_xywh"],
              "potential_duplicate_identity": "Proposed ID 68 bbox overlaps the existing ID 74 bbox on the white car; confirm intended identity before any application",
              "track_observation_summary": {str(identity): {"observations": len([row for row in records if row["track_id"] == identity]), "source_frame_range": [min(row["source_frame_id"] for row in records if row["track_id"] == identity), max(row["source_frame_id"] for row in records if row["track_id"] == identity)]} for identity in (68, 74)},
              "source_image_sha256": sha256(source), "annotations_sha256": original_hash,
              "applied": False, "raw_source_modified": False, "gt_modified": False,
              "no_id_merge_or_deletion": True}
    if sha256(annotations) != original_hash:
        raise RuntimeError("GT changed during preview generation")
    (OUTPUT / "proposal.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Track 68 bbox review</title><style>body{background:#181818;color:white;font:18px sans-serif}img{max-width:100%}</style><h1>Source 382 / MOT 383 — preview only</h1><p>Left: existing GT 68 red, 74 cyan, 72 yellow. Right: proposed 68 green and unchanged 74 cyan. White car already has a bbox with ID 74: resolve this possible duplicate identity before editing. No GT changed.</p><img src="comparison.png"><details><summary>Native full frame with proposed bbox</summary><img src="proposed_full_frame.jpg"></details>', encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
