import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256
from src.tracking.schemas import TrackingResult


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/project363/tracking/expanded_tracking_review/aicc22_custom_001"
PILOT = ROOT / "data/Tracking363_working/sequences/aicc22_custom_001"
OUTPUT = BASE / "event_qa"


def panel(frame_id, gt_id, prediction_ids, region, gt_frames, results):
    image = Image.open(PILOT / "img1" / f"{frame_id:06d}.jpg").convert("RGB")
    drawing = ImageDraw.Draw(image)
    row = gt_frames[frame_id].get(gt_id)
    if row:
        left, top, width, height = row["bbox_xywh"]
        drawing.rectangle((left,top,left+width,top+height),outline="lime",width=2)
    present = []
    for track in results[frame_id].tracks:
        if track.track_id in prediction_ids:
            color = ("cyan", "orange", "magenta")[prediction_ids.index(track.track_id)]
            drawing.rectangle(track.bbox_xyxy,outline=color,width=2)
            drawing.text((track.bbox_xyxy[0],max(0,track.bbox_xyxy[1]-14)),f"P{track.track_id}",fill=color,stroke_width=1,stroke_fill="black")
            present.append(track.track_id)
    crop = image.crop(region)
    scale = min(630/crop.width,350/crop.height)
    crop = crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))),Image.Resampling.LANCZOS)
    result = Image.new("RGB",(640,430),"#181818")
    result.paste(crop,(0,75))
    drawing = ImageDraw.Draw(result)
    drawing.text((5,5),f"GT{gt_id} green | source {frame_id-1} / MOT {frame_id}",fill="white")
    drawing.text((5,25),f"Cyan P{prediction_ids[0]} | orange P{prediction_ids[1]}",fill="white")
    if len(prediction_ids) > 2:
        drawing.text((330,25),f"Magenta P{prediction_ids[2]}",fill="magenta")
    drawing.text((5,45),f"Present prediction IDs: {present}",fill="yellow")
    return result


def main():
    annotations = PILOT / "metadata/annotations_candidate.jsonl"
    inputs = [annotations,BASE/"tracking_results.jsonl",BASE/"diagnostic_id_changes.json"]
    hashes = {str(path.relative_to(ROOT)):sha256(path) for path in inputs}
    gt_frames = defaultdict(dict)
    for line in annotations.read_text().splitlines():
        row = json.loads(line)
        gt_frames[row["mot_frame_id"]][row["track_id"]]=row
    results = {result.frame.mot_frame_id:result for result in (TrackingResult.from_dict(json.loads(line)) for line in inputs[1].read_text().splitlines())}
    cases = json.loads(inputs[2].read_text())
    groups = defaultdict(list)
    for number,case in enumerate(cases,1):
        groups[case["gt_track_id"]].append({"case_id":number,**case})
    OUTPUT.mkdir(exist_ok=False)
    sections = []
    evidence = []
    for gt_id,events in groups.items():
        prediction_ids = (events[0]["previous_prediction_id"],events[0]["prediction_id"])
        start = max(1,min(case["previous_mot_frame"] for case in events)-7)
        end = min(200,max(case["mot_frame_id"] for case in events)+7)
        boxes = []
        for frame_id in range(start,end+1):
            row = gt_frames[frame_id].get(gt_id)
            if row:
                left,top,width,height=row["bbox_xywh"]
                boxes.append((left,top,left+width,top+height))
            boxes.extend(track.bbox_xyxy for track in results[frame_id].tracks if track.track_id in prediction_ids)
        region=(max(0,int(min(box[0] for box in boxes)-40)),max(0,int(min(box[1] for box in boxes)-40)),min(1280,int(max(box[2] for box in boxes)+41)),min(720,int(max(box[3] for box in boxes)+41)))
        video_name=f"gt{gt_id}_slow.mp4"
        writer=cv2.VideoWriter(str(OUTPUT/video_name),cv2.VideoWriter_fourcc(*"mp4v"),3,(640,430))
        if not writer.isOpened():
            raise RuntimeError("Video writer unavailable")
        timeline=[]
        try:
            for frame_id in range(start,end+1):
                image=panel(frame_id,gt_id,prediction_ids,region,gt_frames,results)
                writer.write(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
                timeline.append({"mot_frame_id":frame_id,"source_frame_id":frame_id-1,"present_prediction_ids":[track.track_id for track in results[frame_id].tracks if track.track_id in prediction_ids]})
        finally:
            writer.release()
        capture=cv2.VideoCapture(str(OUTPUT/video_name))
        decoded=0
        while capture.read()[0]:
            decoded+=1
        capture.release()
        if decoded!=end-start+1:
            raise RuntimeError("Video frame count mismatch")
        sections.append(f'<section><h2>GT{gt_id}: P{prediction_ids[0]} / P{prediction_ids[1]}</h2><p>Source {start-1}..{end-1}; all consecutive frames; playback3fps from original10fps.</p><video controls src="{video_name}"></video>')
        for case in events:
            frame_ids=[case["previous_mot_frame"],case["mot_frame_id"],min(200,case["mot_frame_id"]+1)]
            sheet=Image.new("RGB",(1920,430))
            for position,frame_id in enumerate(frame_ids):
                sheet.paste(panel(frame_id,gt_id,prediction_ids,region,gt_frames,results),(position*640,0))
            filename=f"case_{case['case_id']:02d}_mot{case['mot_frame_id']}.jpg"
            sheet.save(OUTPUT/filename,quality=95)
            sections.append(f'<h3>Case {case["case_id"]}: source {case["mot_frame_id"]-1} / MOT {case["mot_frame_id"]}; diagnostic P{case["previous_prediction_id"]} → P{case["prediction_id"]}</h3><img src="{filename}">')
        sections.append('</section>')
        evidence.append({"gt_track_id":gt_id,"prediction_ids":list(prediction_ids),"mot_window":[start,end],"cases":events,"timeline":timeline,"qa_video":video_name})
    for path in inputs:
        if sha256(path)!=hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError("Inputs changed")
    write=lambda path,value:path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    write(OUTPUT/"event_manifest.json",{"status":"QA_GENERATED_REVIEW_REQUIRED","events":evidence,"inputs_unchanged":True,"input_sha256":hashes,"predictions_rerun":False,"gt_modified":False})
    write(OUTPUT/"human_feedback.json",{"reviewer":"project owner","status":"ISSUE_CONFIRMED_REMAINING_REVIEW_REQUIRED","confirmed_physical_vehicle_id_replacement":{"gt_track_id":9,"previous_prediction_id":2,"prediction_id":26,"owner_reported_source_frame":30,"diagnostic_source_frame":34,"diagnostic_mot_frame":35,"exact_human_frame_confirmation":False,"human_statement":"Same vehicle; P2 replaced by P26 in subsequent frames"},"machine_timing_note":"In the inspected source27..37 range, both IDs are present at source32/33; P2 absent at source34; matching warning at source34. Owner's approximate source30 timing is preserved, not overwritten.","remaining_events":"GT5 P5/P37 and GT11 P8/P92 lack owner confirmation; new consecutive-frame QA provided","official_id_switch_count":None,"gt_modified":False,"predictions_modified":False})
    (OUTPUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>AICC focused event QA</title><style>body{background:#181818;color:white;font:18px sans-serif;margin:20px}img{max-width:100%}video{width:640px;max-width:100%}section{border-top:1px solid #777}</style><h1>AICC: 5 warnings grouped into 3 vehicles</h1><p>Green=GT, cyan=first P-ID, orange=second P-ID. Same vehicle P2→P26 confirmed by owner; precise timeline still shown. Remaining GT5/GT11 groups need human QA. All consecutive frames included in slow videos, not only every25th frame. No predictions or GT edited; no official metric.</p>'+"\n".join(sections),encoding="utf-8")
    (OUTPUT/"README.md").write_text("# Focused AICC tracking QA\n\nOwner confirmed same physical vehicle P2 replaced by P26. Reported source30 is approximate; cached matching event is source34/MOT35, with P2/P26 briefly coexisting at source32/33. Both timings are recorded separately.\n\nFive warnings involve three vehicles: GT9 P2/P26, GT5 P5/P37, GT11 P8/P92. Open index.html for5 before/current/after sheets and3 slow videos containing every consecutive frame around events. Review whether the same vehicle changes ID, two predictions overlap one vehicle, or predictions refer to different vehicles. If unclear, keep REVIEW_REQUIRED. No GT/model/tracker rerun or metric certification.\n",encoding="utf-8")
    parent=BASE/"index.html"
    existing=parent.read_text(encoding="utf-8")
    parent.write_text(existing.replace('<video controls', '<p><a href="event_qa/index.html">Focused event QA: all missing neighboring frames and slow video</a></p><video controls',1),encoding="utf-8")
    print("AICC_EVENT_QA_READY: "+str(OUTPUT),flush=True)


if __name__=="__main__":
    main()
