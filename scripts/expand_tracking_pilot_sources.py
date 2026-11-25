import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256


ROOT = Path(__file__).resolve().parents[1]
WORKING = ROOT / "data/Tracking363_working"
SEQUENCES = WORKING / "sequences"
QA = ROOT / "runs/project363/tracking/source_expansion"
AICC = ROOT / "data/Vehicle_Tracking/AICC22-Custom"
SOURCES = [
    ("aicc22_custom_001", AICC / "AICC22-Custom_Original-video.mp4", 200, "PILOT_GT_CANDIDATE"),
    ("bellevue_ne12th_001", ROOT / "data/Bellevue_116th_NE12th/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4", 300, "VISUAL_ONLY_NO_GT"),
    ("bellevue_newport_001", ROOT / "data/Bellevue_150th_Newport/Bellevue_150th_Newport__2017-09-10_18-08-24.mp4", 300, "VISUAL_ONLY_NO_GT"),
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_rows(rows, width=1280, height=720):
    keys = set()
    for row in rows:
        frame, identity = row["mot_frame_id"], row["track_id"]
        left, top, box_width, box_height = row["bbox_xywh"]
        if not 1 <= frame <= 200 or identity <= 0 or (frame, identity) in keys:
            raise ValueError("Invalid or duplicate AICC frame/track key")
        if not all(math.isfinite(value) for value in row["bbox_xywh"]) or left < 0 or top < 0 or box_width <= 0 or box_height <= 0 or left + box_width > width or top + box_height > height:
            raise ValueError("Invalid AICC bbox")
        keys.add((frame, identity))


def load_aicc():
    source_json = AICC / "AICC22-Custom_GroundTruth.json"
    contents = json.loads(source_json.read_text(encoding="utf-8"))
    source_digest = sha256(source_json)
    rows = []
    for identity, track in contents.items():
        if len(track["frames"]) != len(track["boxes"]):
            raise ValueError("AICC JSON array mismatch")
        for frame_name, box in zip(track["frames"], track["boxes"]):
            frame = int(Path(frame_name).stem)
            rows.append({"sequence_id": "aicc22_custom_001", "mot_frame_id": frame, "source_frame_id": frame-1,
                         "track_id": int(identity), "bbox_xywh": list(map(float, box)), "class": None, "class_status": "PENDING",
                         "source": "AICC22-Custom", "source_annotation": str(source_json.relative_to(ROOT)),
                         "source_annotation_sha256": source_digest, "source_annotation_frame_path": frame_name,
                         "human_corrected": False, "annotation_edition_status": "JSON_PROVISIONAL_NOT_HUMAN_SELECTED",
                         "frame_alignment_status": "ONE_BASED_INFERRED_REQUIRES_QA"})
    rows.sort(key=lambda row: (row["mot_frame_id"], row["track_id"]))
    validate_rows(rows)
    txt_rows = []
    for line in (AICC / "AICC22-Custom_GroundTruth.txt").read_text().splitlines():
        values = line.split(",")
        txt_rows.append({"mot_frame_id": int(values[0]), "track_id": int(values[1]), "bbox_xywh": list(map(float, values[2:6]))})
    validate_rows(txt_rows)
    geometry = lambda row: (row["mot_frame_id"], *row["bbox_xywh"])
    json_geometry = Counter(geometry(row) for row in rows)
    txt_geometry = Counter(geometry(row) for row in txt_rows)
    if len(rows) != 2439 or len(txt_rows) != 1185 or txt_geometry - json_geometry:
        raise ValueError("Source editions differ from audited superset relationship")
    return rows, txt_rows, {"json_rows": len(rows), "txt_rows": len(txt_rows), "json_extra_geometry_observations": sum((json_geometry-txt_geometry).values()), "namespace_policy": "Preserve JSON IDs for candidate; TXT has a separate source namespace, never assign TXT IDs to JSON automatically", "selection_status": "PROVISIONAL_JSON_SUPERSET_PENDING_OWNER_QA"}


def annotated(image, rows, title):
    result = image.copy()
    for row in rows:
        left, top, width, height = row["bbox_xywh"]
        cv2.rectangle(result, (int(left), int(top)), (int(left+width), int(top+height)), (0, 255, 0), 2)
        cv2.putText(result, f"ID{row['track_id']}", (int(left), max(20, int(top))), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(result, (0,0), (result.shape[1], 35), (0,0,0), -1)
    cv2.putText(result, title, (8,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)
    return result


def extract(sequence_id, video, count, role, rows, txt_rows):
    import numpy as np

    output = SEQUENCES / sequence_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "img1").mkdir()
    (output / "metadata").mkdir()
    qa = QA / sequence_id
    qa.mkdir()
    before_video = sha256(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid source video metadata")
    json_by_frame, txt_by_frame = defaultdict(list), defaultdict(list)
    for row in rows:
        json_by_frame[row["mot_frame_id"]].append(row)
    for row in txt_rows:
        txt_by_frame[row["mot_frame_id"]].append(row)
    writer = cv2.VideoWriter(str(qa / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360) if rows else (1280,720))
    if not writer.isOpened():
        raise RuntimeError("Preview writer unavailable")
    selected = {1, 2, 3, count//4, count//2, 3*count//4, count-1, count}
    frame_records = []
    preview_names = []
    try:
        for source_id in range(count):
            success, frame = capture.read()
            if not success or frame.shape[:2] != (height, width):
                raise RuntimeError(f"Decode failed at {sequence_id}/{source_id}")
            mot_id = source_id + 1
            destination = output / "img1" / f"{mot_id:06d}.jpg"
            if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError("Frame extraction write failed")
            frame_records.append({"source_frame_id": source_id, "mot_frame_id": mot_id, "image_filename": destination.name,
                                  "source_video": str(video.relative_to(ROOT)),
                                  "timestamp_seconds": source_id/fps, "timestamp_kind": "DERIVED_CONTAINER_FPS_NOT_AUTHORITATIVE_CAPTURE_TIME",
                                  "source_bgr_sha256": hashlib.sha256(frame.tobytes()).hexdigest(), "jpeg_sha256": sha256(destination),
                                  "gt_coverage_status": "ANNOTATIONS_PRESENT_COMPLETENESS_UNVERIFIED" if json_by_frame[mot_id] else "UNLABELED_NOT_BACKGROUND"})
            if rows:
                left = annotated(frame, json_by_frame[mot_id], f"JSON candidate | source {source_id} / MOT {mot_id}")
                right = annotated(frame, txt_by_frame[mot_id], "TXT reference: different ID namespace")
                rendered = np.hstack([cv2.resize(left,(640,360)), cv2.resize(right,(640,360))])
            else:
                rendered = cv2.resize(annotated(frame, [], f"{sequence_id}: VISUAL ONLY - NO GT | frame {mot_id}"), (1280,720))
            writer.write(rendered)
            if mot_id in selected:
                preview_name = f"frame_{mot_id:06d}.jpg"
                if not cv2.imwrite(str(qa / preview_name), rendered):
                    raise RuntimeError("QA image write failed")
                preview_names.append(preview_name)
    finally:
        capture.release()
        writer.release()
    if sha256(video) != before_video:
        raise RuntimeError("Source video changed")
    preview = cv2.VideoCapture(str(qa / "preview.mp4"))
    decoded = 0
    while preview.read()[0]:
        decoded += 1
    preview.release()
    if decoded != count or len(list((output / "img1").glob("*.jpg"))) != count:
        raise RuntimeError("Output frame count mismatch")
    if rows:
        (output / "gt").mkdir()
        with (output / "gt/gt_candidate_json.txt").open("x", encoding="utf-8") as handle:
            for row in rows:
                coordinates = ",".join(f"{value:g}" for value in row["bbox_xywh"])
                handle.write(f"{row['mot_frame_id']},{row['track_id']},{coordinates},1,-1,-1,-1\n")
        with (output / "metadata/annotations_candidate.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    with (output / "metadata/frame_map.csv").open("x", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=["source_frame_id","mot_frame_id","image_filename","source_video","timestamp_seconds","gt_coverage_status"])
        csv_writer.writeheader()
        csv_writer.writerows({key:row[key] for key in csv_writer.fieldnames} for row in frame_records)
    write_json(output / "metadata/extraction_manifest.json", frame_records)
    settings = {"sequence_id": sequence_id, "dataset_role": role, "source_video": str(video.relative_to(ROOT)),
                "source_video_sha256": before_video, "fps": fps, "width": width, "height": height, "pilot_frame_count": count,
                "source_frame_range": [0,count-1], "mot_frame_range": [1,count], "annotation_observations": len(rows),
                "tracks": len({row["track_id"] for row in rows}) if rows else None, "human_qa_status": "REVIEW_REQUIRED",
                "continuity": "Consecutive frames fully decoded; scene-cut/camera continuity requires visual QA",
                "gt_status": "PROVISIONAL_JSON_EDITION_AND_ALIGNMENT_REVIEW_REQUIRED" if rows else "ABSENT",
                "metric_eligible": False, "semantic_classes": "PENDING" if rows else "NOT_ANNOTATED",
                "unlabeled_frames_are_background": False, "source_modified": False, "video_fully_extracted": sequence_id=="aicc22_custom_001"}
    write_json(output / "metadata/sequence.json", settings)
    (output / "seqinfo.ini").write_text(f"[Sequence]\nname={sequence_id}\nimDir=img1\nframeRate={fps}\nseqLength={count}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n", encoding="utf-8")
    artifact_hashes = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file()}
    write_json(output / "metadata/provenance_manifest.json", {"source": settings, "artifact_sha256": artifact_hashes})
    (output / "README.md").write_text(f"# {sequence_id}\n\nRole: {role}. {count} consecutive frames; source read-only. HUMAN QA: REVIEW_REQUIRED.\n\n" + ("gt/gt_candidate_json.txt is a PROVISIONAL JSON namespace conversion, not the chosen final GT edition. Compare JSON vs TXT previews before approval. Annotation frame IDs are inferred one-based, mapping to source index minus1; visual alignment remains unapproved. Confidence1 in MOT rows is a format placeholder, not source certainty. Original JSON IDs unchanged. Frames1/2 are unlabeled, NOT background. Classes null/PENDING; no final metrics.\n" if rows else "No GT exists: no empty GT/labels were fabricated, no IDs/classes inferred. Use only qualitative QA until genuine reviewed bbox/ID annotations exist.\n") + "\nFrame-map timestamps derive from container FPS. JPEG95 is lossy; original BGR/JPEG hashes recorded. No train/val/test split or independent-camera benchmark certification.\n", encoding="utf-8")
    images = "".join(f'<h3>{name}</h3><img loading="lazy" src="{name}">' for name in preview_names)
    (qa / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>{sequence_id} QA</title><style>body{{background:#181818;color:white;font:18px sans-serif}}img,video{{max-width:100%}}</style><h1>{sequence_id} — REVIEW_REQUIRED</h1><p>{role}. ' + ("Left JSON candidate, right TXT reference. IDs differ between editions. Check frame alignment, full bbox extent, added objects and stable physical identities; sample-only ID approval does not certify all JSON annotations." if rows else "Raw visual clip only. Check uninterrupted camera scene, cuts, moving vehicles and occlusion; no GT and no quantitative evaluation.") + '</p><video controls src="preview.mp4"></video>' + images, encoding="utf-8")
    print(f"ADDED: {sequence_id} {count} frames; role={role}", flush=True)
    return settings


def main():
    QA.mkdir(parents=True, exist_ok=False)
    source_files = [AICC / "AICC22-Custom_GroundTruth.json", AICC / "AICC22-Custom_GroundTruth.txt"]
    baseline_files = [ROOT / "data/Tracking363_working/gt_revisions/vntraffic_001_remove_track68/gt/gt.txt", ROOT / "data/Tracking363_working/sequences/vntraffic_001/metadata/annotations.jsonl"]
    protected = {path:sha256(path) for path in source_files + baseline_files}
    rows, txt_rows, edition_report = load_aicc()
    summaries = []
    for sequence_id, video, count, role in SOURCES:
        summaries.append(extract(sequence_id,video,count,role,rows if sequence_id=="aicc22_custom_001" else [],txt_rows if sequence_id=="aicc22_custom_001" else []))
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError("Protected annotations changed")
    write_json(SEQUENCES / "aicc22_custom_001/metadata/annotation_edition_comparison.json", edition_report)
    refresh_provenance()
    report = {"status":"PREPARATION_PASS_HUMAN_REVIEW_REQUIRED", "created_at_utc":datetime.now(timezone.utc).isoformat(),
              "added_sequences": summaries, "added_frames":800,
              "existing_vntraffic_frames":501,"total_prepared_sequences":4,"total_prepared_frames":1301,
              "existing_reviewed_gt_pilot":{"sequence":"vntraffic_001","observations":16034,"tracks":84},
              "new_candidate_gt":{"sequence":"aicc22_custom_001","observations":len(rows),"tracks":len({row['track_id'] for row in rows}),"metric_eligible":False},
              "visual_only_sequences":2,"final_dataset_created":False,"split_created":False,"sources_modified":False,
              "annotation_edition_comparison":edition_report,
              "next":"Human QA AICC22 JSON/TXT edition and frame alignment; review Bellevue clip continuity; no metrics on raw clips without GT"}
    write_json(QA / "source_expansion_report.json",report)
    catalog = [{"sequence_id":"vntraffic_001","frames":501,"role":"PILOT_DEV_REVIEWED_GT_REVISION","gt_path":"data/Tracking363_working/gt_revisions/vntraffic_001_remove_track68/gt/gt.txt","metric_eligible":False,"reason":"GT completeness/ignore policy unresolved"}] + [{"sequence_id":row["sequence_id"],"frames":row["pilot_frame_count"],"role":row["dataset_role"],"metric_eligible":False,"gt_status":row["gt_status"]} for row in summaries]
    write_json(WORKING / "pilot_sequence_catalog.json",catalog)
    links="".join(f'<li><a href="{row["sequence_id"]}/index.html">{row["sequence_id"]}: {row["pilot_frame_count"]} frames ({row["dataset_role"]})</a></li>' for row in summaries)
    (QA / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Tracking source expansion QA</title><h1>3 new sequences — human QA required</h1><p>4 prepared sequences /1301frames including existing VNTraffic. Only one existing reviewed GT pilot; AICC22 GT candidate pending review; two Bellevue clips have no GT. Do not count visual-only clips as labeled benchmark data.</p><ul>'+links+'</ul>',encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


def refresh_provenance():
    for sequence_id, _, _, _ in SOURCES:
        output = SEQUENCES / sequence_id
        manifest = output / "metadata/provenance_manifest.json"
        source = json.loads((output / "metadata/sequence.json").read_text())
        hashes = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file() and path != manifest}
        write_json(manifest, {"source": source, "artifact_sha256": hashes, "scope": "All sequence files except this self-referencing manifest; external QA stored separately"})


if __name__ == "__main__":
    main()
