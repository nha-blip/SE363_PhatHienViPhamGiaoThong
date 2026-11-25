import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256


ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/Tracking363_working/annotation_drafts/bellevue_ne12th_001/cleanup_revision_001/annotations_draft.json"
OUTPUT=ROOT/"runs/project363/tracking/bellevue_draft_cleanup/ne12th_initial_qa"
REGION=(735,100,925,265)


def preview(frame,image_path):
    image=Image.open(image_path).convert("RGB")
    drawing=ImageDraw.Draw(image)
    visible=[]
    colors={2:"magenta",3:"cyan",5:"yellow",8:"orange"}
    for item in frame["objects"]:
        identity=item["track_id"]
        if identity not in colors:
            continue
        left,top,width,height=item["bbox_xywh"]
        drawing.rectangle((left,top,left+width,top+height),outline=colors[identity],width=1)
        drawing.text((left,max(0,top-12)),str(identity),fill=colors[identity],stroke_width=1,stroke_fill="black")
        visible.append(identity)
    panel=Image.new("RGB",(400,410),"#181818")
    crop=image.crop(REGION)
    crop=crop.resize((380,330),Image.Resampling.LANCZOS)
    panel.paste(crop,(0,70))
    drawing=ImageDraw.Draw(panel)
    drawing.text((5,5),f"MOT {frame['mot_frame_id']} / source {frame['source_frame_id']}",fill="white")
    drawing.text((5,25),"Draft3 cyan; draft5 yellow; draft2 magenta",fill="white")
    drawing.text((5,45),f"Present IDs: {visible} | NOT GT",fill="white")
    return panel


def main():
    before=sha256(SOURCE)
    payload=json.loads(SOURCE.read_text())
    selected=payload["frames"][:26]
    first_white=next(frame["mot_frame_id"] for frame in payload["frames"] if any(item["track_id"]==5 for item in frame["objects"]))
    OUTPUT.mkdir(exist_ok=False)
    panels=[]
    evidence=[]
    for frame in selected:
        path=(SOURCE.parent/frame["image"]).resolve(strict=True)
        panels.append(preview(frame,path))
        evidence.append({"mot_frame_id":frame["mot_frame_id"],"source_frame_id":frame["source_frame_id"],"objects":[item for item in frame["objects"] if item["track_id"] in (2,3,5,8)],"source_image_sha256":sha256(path)})
    sheets=[]
    for start,end,name in ((1,10,"frames_01_10.jpg"),(11,18,"frames_11_18.jpg"),(19,26,"frames_19_26.jpg")):
        count=end-start+1
        columns=5 if count==10 else 4
        sheet=Image.new("RGB",(columns*400,820),"#181818")
        for position,frame_id in enumerate(range(start,end+1)):
            sheet.paste(panels[frame_id-1],((position%columns)*400,(position//columns)*410))
        sheet.save(OUTPUT/name,quality=95)
        sheets.append(name)
    writer=cv2.VideoWriter(str(OUTPUT/"first26_slow.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),3,(400,410))
    if not writer.isOpened():raise RuntimeError("QA video writer unavailable")
    try:
        for image in panels:
            writer.write(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture=cv2.VideoCapture(str(OUTPUT/"first26_slow.mp4"))
    decoded=0
    while capture.read()[0]:decoded+=1
    capture.release()
    if decoded!=26 or sha256(SOURCE)!=before:
        raise RuntimeError("QA integrity mismatch")
    report={"status":"QA_PREPARED_NOT_APPLIED","sequence_id":"bellevue_ne12th_001","review_mot_frames":[1,26],"review_source_frames":[0,25],
            "first_draft5_mot_frame":first_white,"first_draft5_source_frame":first_white-1,
            "draft5_absent_mot_frames":list(range(1,first_white)),
            "owner_report":"Initial bbox covers two vehicles; by frame26 they have separate boxes",
            "observed_structure":"Draft5 already exists from MOT11. Draft3 remains comparatively tall through early frames, so separate-ID presence alone does not establish correct single-vehicle bounds.",
            "priority_review_windows":{"mot1_to10":"No separate draft5 observation: check missing white-car bbox and oversized draft3", "mot11_to18":"Draft3 and5 both present: inspect whether draft3 still includes white car", "mot19_to26":"Later reference for separate physical identities and bbox extent"},
            "no_interpolation":True,"no_splitting_applied":True,"no_ids_created":True,"gt_modified":False,"draft_modified":False,
            "source_draft_sha256":before,"native_roi_xyxy":list(REGION),"frames":evidence,
            "remaining_duplicate_observations_note":"Draft2 remains in some early frames under conservative previous cleanup; displayed in magenta, not silently removed",
            "next":"Human-guided native-frame correction of missing/merged boxes in a new derived revision; do not assume frames1..25 all wrong"}
    (OUTPUT/"review_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    images="".join(f'<h2>{name}</h2><img src="{name}">' for name in sheets)
    (OUTPUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>NE12th first26 frames</title><style>body{background:#181818;color:white;font:18px sans-serif}img{max-width:100%}video{height:550px;max-width:100%}</style><h1>NE12th: all first26 frames — no GT edits</h1><p>Cyan draft3 (black car), yellow draft5 (white car); magenta draft2 is a remaining overlapping draft observation. Track5 starts at MOT11/source10, not26. Review missing white-car bbox in1..10 and oversized black-car box in11..18. These are predictions, not authoritative labels; no interpolation/splitting done.</p><video controls src="first26_slow.mp4"></video>'+images,encoding="utf-8")
    print(json.dumps({key:report[key] for key in ('status','first_draft5_mot_frame','draft5_absent_mot_frames','draft_modified')},indent=2))


if __name__=="__main__":
    main()
