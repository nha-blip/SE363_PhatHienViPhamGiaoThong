import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.bytetrack_adapter import create_ultralytics_adapter
from src.tracking.io import load_frame_map, sha256
from src.tracking.pilot_runtime import LoggedBoundsBackend, diagnostic_matches
from src.tracking.yolo_adapter import YoloAdapter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/project363/tracking/expanded_tracking_review"
DRAFTS = ROOT / "data/Tracking363_working/annotation_drafts"
SEQUENCES = ROOT / "data/Tracking363_working/sequences"
IDENTITIES = ("aicc22_custom_001", "bellevue_ne12th_001", "bellevue_newport_001")
NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def draw(image, result, caption):
    import cv2

    rendered = image.copy()
    for track in result.tracks:
        left, top, right, bottom = track.bbox_xyxy
        cv2.rectangle(rendered, (int(left),int(top)),(int(right),int(bottom)),(255,255,0),2)
        cv2.putText(rendered,f"P{track.track_id}",(int(left),max(35,int(top))),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
    cv2.rectangle(rendered,(0,0),(image.shape[1],40),(0,0,0),-1)
    cv2.putText(rendered,caption,(8,27),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    return rendered


def process(identity, configuration, model):
    import cv2
    import numpy as np

    source = SEQUENCES / identity
    metadata = json.loads((source / "metadata/sequence.json").read_text())
    provenance = json.loads((source / "metadata/provenance_manifest.json").read_text())
    protected = {source/relative:expected for relative,expected in provenance["artifact_sha256"].items()}
    for path,expected in protected.items():
        if sha256(path)!=expected:
            raise ValueError(f"Sequence changed: {path}")
    frames = load_frame_map(source / "metadata/frame_map.csv",metadata)
    has_gt = identity == "aicc22_custom_001"
    gt_frames = defaultdict(list)
    if has_gt:
        feedback = json.loads((ROOT / "runs/project363/tracking/source_expansion/aicc22_custom_001/human_review.json").read_text())
        if feedback["status"] != "PASS_ON_OWNER_REVIEWED_MATERIAL":
            raise ValueError("AICC22 owner sample QA not recorded")
        for line in (source / "metadata/annotations_candidate.jsonl").read_text().splitlines():
            row = json.loads(line)
            gt_frames[row["mot_frame_id"]].append(row)
    tracker = create_ultralytics_adapter(configuration["tracker"],NAMES,metadata["fps"],allow_execution=True)
    tracker.backend.reset()
    bounds = LoggedBoundsBackend(tracker.backend,metadata["width"],metadata["height"])
    tracker.backend = bounds
    adapter = YoloAdapter(NAMES,0.1)
    output = OUTPUT / identity
    output.mkdir()
    (output / "previews").mkdir()
    resolution = (1920,540) if has_gt else (1280,720)
    video = cv2.VideoWriter(str(output / "tracking_qa.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),metadata["fps"],resolution)
    if not video.isOpened():
        raise RuntimeError("QA video writer unavailable")
    previous = {}
    changes = []
    predicted_ids = set()
    observation_count = 0
    matched_count = 0
    review_frames = []
    draft_frames = []
    try:
        with (output / "detections.jsonl").open("x",encoding="utf-8") as detections,(output / "tracking_results.jsonl").open("x",encoding="utf-8") as results,(output / "raw_backend_outputs.jsonl").open("x",encoding="utf-8") as raw:
            for frame in frames:
                image_path = source / "img1" / f"{frame.mot_frame_id:06d}.jpg"
                image = cv2.imread(str(image_path))
                if image is None or image.shape[:2]!=(frame.height,frame.width):
                    raise ValueError("Frame decode/size mismatch")
                prediction = model.predict(image,device="cpu",imgsz=640,conf=0.1,iou=0.7,max_det=300,save=False,verbose=False)[0]
                detected = adapter.adapt(prediction,frame)
                tracked = tracker.update(detected)
                detections.write(json.dumps(detected.to_dict(),allow_nan=False)+"\n")
                results.write(json.dumps(tracked.to_dict(),allow_nan=False)+"\n")
                raw.write(json.dumps({"mot_frame_id":frame.mot_frame_id,"raw_rows":bounds.raw_output},allow_nan=False)+"\n")
                observation_count += len(tracked.tracks)
                predicted_ids.update(track.track_id for track in tracked.tracks)
                if has_gt:
                    from scripts.run_tracking_pilot import render_frame

                    rendered = render_frame(image,gt_frames[frame.mot_frame_id],tracked)
                    if frame.mot_frame_id<=2:
                        cv2.putText(rendered,"GT UNLABELED: NOT BACKGROUND",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    matched = diagnostic_matches(gt_frames[frame.mot_frame_id],tracked.tracks)
                    matched_count += len(matched)
                    for gt_id,predicted_id,score in matched:
                        earlier = previous.get(gt_id)
                        if earlier and earlier[0]==frame.mot_frame_id-1 and earlier[1]!=predicted_id:
                            changes.append({"gt_track_id":gt_id,"previous_mot_frame":earlier[0],"mot_frame_id":frame.mot_frame_id,"previous_prediction_id":earlier[1],"prediction_id":predicted_id,"iou":score,"status":"REVIEW_REQUIRED_NOT_OFFICIAL_IDSW"})
                        previous[gt_id]=(frame.mot_frame_id,predicted_id)
                else:
                    rendered = cv2.resize(draw(image,tracked,f"{identity}: AUTO DRAFT NOT GT | source {frame.source_frame_id}"),resolution)
                video.write(rendered)
                if frame.mot_frame_id in {1,2,3,len(frames)} or frame.mot_frame_id % 25==0:
                    filename=f"frame_{frame.mot_frame_id:06d}.jpg"
                    if not cv2.imwrite(str(output/"previews"/filename),rendered):
                        raise RuntimeError("Preview write failed")
                    review_frames.append(filename)
                if not has_gt:
                    objects=[]
                    for track in tracked.tracks:
                        left,top,right,bottom=track.bbox_xyxy
                        objects.append({"track_id":track.track_id,"bbox_xywh":[left,top,right-left,bottom-top],"class":None,
                                        "candidate_class":track.class_name,"detector_confidence":track.confidence,
                                        "annotation_origin":"AUTO_YOLO_BYTETRACK_UNVERIFIED","human_corrected":False})
                    draft_frames.append({"mot_frame_id":frame.mot_frame_id,"source_frame_id":frame.source_frame_id,"image":image_path,
                                         "review_status":"UNREVIEWED","objects":objects})
                if frame.mot_frame_id%50==0:
                    print(f"{identity}: {frame.mot_frame_id}/{len(frames)}",flush=True)
    finally:
        video.release()
    capture=cv2.VideoCapture(str(output/"tracking_qa.mp4"))
    decoded=0
    while capture.read()[0]:
        decoded+=1
    capture.release()
    if decoded!=len(frames):
        raise RuntimeError("QA video frame mismatch")
    for path,expected in protected.items():
        if sha256(path)!=expected:
            raise RuntimeError("Source sequence modified")
    write_json(output/"bounds_events.json",bounds.events)
    draft_directory=None
    if not has_gt:
        draft_directory=DRAFTS/identity
        draft_directory.mkdir(parents=True,exist_ok=False)
        for row in draft_frames:
            row["image"]=Path(os.path.relpath(row["image"],draft_directory)).as_posix()
        payload={"sequence_id":identity,"width":metadata["width"],"height":metadata["height"],"fps":metadata["fps"],
                 "source_video_sha256":metadata["source_video_sha256"],"source_prediction_sha256":sha256(output/"tracking_results.jsonl"),
                 "source_prediction_file":str((output/"tracking_results.jsonl").relative_to(ROOT)),
                 "status":"AUTO_DRAFT_UNREVIEWED","is_ground_truth":False,"classes":"PENDING","frames":draft_frames}
        write_json(draft_directory/"annotations_draft.json",payload)
        template=(ROOT/"src/tracking/annotation_editor.html").read_text(encoding="utf-8")
        safe_payload=json.dumps(payload,allow_nan=False).replace("<","\\u003c")
        (draft_directory/"editor.html").write_text(template.replace("__ANNOTATION_PAYLOAD__",safe_payload),encoding="utf-8")
        with (draft_directory/"tracks_draft_mot.txt").open("x",encoding="utf-8") as handle:
            for frame in draft_frames:
                for item in frame["objects"]:
                    box=",".join(f"{coordinate:.6f}" for coordinate in item["bbox_xywh"])
                    handle.write(f"{frame['mot_frame_id']},{item['track_id']},{box},{item['detector_confidence']:.6f},-1,-1,-1\n")
        with (draft_directory/"human_review_frames.csv").open("x",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle)
            writer.writerow(["mot_frame_id","source_frame_id","decision","notes"])
            writer.writerows((row["mot_frame_id"],row["source_frame_id"],"REVIEW_REQUIRED","") for row in draft_frames)
        (draft_directory/"README.md").write_text("# Bellevue annotation DRAFT, NOT GT\n\nOpen editor.html. Bbox/IDs were suggested by YOLO + ByteTrack; do not use these as reference GT to evaluate the same tracker. Review every frame for missing vehicles, duplicates, bbox bounds and identity continuity. Edit boxes/IDs, add missing objects, delete false positives, then explicitly confirm the frame reviewed. Track IDs are sequence-local suggestions, not authoritative source IDs. Semantic classes remain null/PENDING. No empty prediction means empty/background by default. Browser save is not a project file write: export review JSON and provide it for validation before any gt.txt promotion. Inspect the exported review before declaring it approved; all300 frames need review or an explicit approved coverage policy.\n\nEditor only changes browser draft state and downloadable JSON, not original videos, extracted frames, AICC GT or disk annotations. Manual repairs do not interpolate other frames. Geometry alone cannot certify physical identity. The draft MOT file is deliberately tracks_draft_mot.txt, not gt/gt.txt.\n",encoding="utf-8")
    report={"sequence_id":identity,"automated_execution_status":"PASS","human_tracking_qa_status":"REVIEW_REQUIRED",
            "frames":len(frames),"prediction_observations":observation_count,"unique_predicted_ids":len(predicted_ids),
            "source_gt_observations":2439 if has_gt else None,"source_gt_tracks":25 if has_gt else None,
            "diagnostic_matched_observations":matched_count if has_gt else None,"id_change_candidates":len(changes) if has_gt else None,
            "official_metrics":None,"draft_is_gt":False,"draft_directory":str(draft_directory) if draft_directory else None,
            "source_sequence_unchanged":True,"bounds_events":len(bounds.events),"video_frames":decoded,
            "raw_sources_modified":False,"gt_modified":False,"final_dataset_created":False,
            "gt_unlabeled_mot_frames":[1,2] if has_gt else "ALL_UNANNOTATED"}
    write_json(output/"report.json",report)
    write_json(output/"diagnostic_id_changes.json",changes)
    from scripts.run_tracking_pilot import create_review_table

    table=create_review_table(output,changes,metadata["fps"]) if has_gt else ""
    editor_link=Path(os.path.relpath(draft_directory/"editor.html",output)).as_posix() if draft_directory else None
    intro="Left reviewed JSON GT candidate; right P-IDs. Compare physical continuity, not matching ID numbers. First2 frames unlabeled; no official metrics." if has_gt else f'AUTO ANNOTATION DRAFT ONLY. <a href="{editor_link}">Open bbox/ID editor</a>. Predictions need human correction; no GT exists yet.'
    images="".join(f'<img loading="lazy" src="previews/{name}">' for name in review_frames)
    (output/"index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>{identity} tracking QA</title><style>body{{background:#181818;color:white;font:18px sans-serif}}img,video{{max-width:100%}}</style><h1>{identity} — REVIEW_REQUIRED</h1><p>{intro}</p><video controls src="tracking_qa.mp4"></video>'+table+images,encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    if not parser.parse_args().execute:
        raise PermissionError("Explicit execution authorization required")
    configuration=yaml.safe_load((ROOT/"configs/tracking/bytetrack_work_5_4.yaml").read_text())
    checkpoint=ROOT/configuration["detector"]["model_path"]
    checkpoint_hash=sha256(checkpoint)
    if checkpoint_hash!=configuration["detector"]["expected_sha256"]:
        raise ValueError("Fixed checkpoint hash mismatch")
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(2)
    model=YOLO(str(checkpoint))
    if model.names!=NAMES:
        raise ValueError("Unexpected detector namespace")
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT/"provenance.json",{"checkpoint_sha256":checkpoint_hash,"checkpoint_completed_epochs":19,"device":"cpu","torch":torch.__version__,"ultralytics":ultralytics.__version__,"tracker":configuration["tracker"],"detector_confidence":0.1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"selected_final_config":False})
    reports=[process(identity,configuration,model) for identity in IDENTITIES]
    if sha256(checkpoint)!=checkpoint_hash:
        raise RuntimeError("Fixed checkpoint changed")
    write_json(OUTPUT/"summary.json",{"status":"EXECUTION_PASS_GT_DRAFTS_REVIEW_REQUIRED","sequences":reports,"bellevue_gt_created":False,"bellevue_drafts_created":2,"next":"QA AICC tracking and edit/review both Bellevue drafts before GT promotion"})
    links="".join(f'<li><a href="{row["sequence_id"]}/index.html">{row["sequence_id"]} tracking/annotation review</a></li>' for row in reports)
    (OUTPUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Expanded tracking review</title><h1>AICC tracking + Bellevue annotation drafts</h1><p>AICC GT candidate preserved. Bellevue boxes/IDs are unapproved model-generated drafts, NOT ground truth. No metrics or final dataset PASS.</p><ul>'+links+'</ul>',encoding="utf-8")
    print(json.dumps(reports,indent=2),flush=True)


if __name__=="__main__":
    main()
