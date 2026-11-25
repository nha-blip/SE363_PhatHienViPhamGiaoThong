import copy
import json
import os
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
DESTINATION=SOURCE.parent.parent/"initial_boxes_revision_002"
QA=ROOT/"runs/project363/tracking/bellevue_draft_cleanup/ne12th_initial_correction"
BLACK_BOXES={
    1:[824,167,58,42],2:[823,168,59,42],3:[821,168,60,43],4:[819,168,61,44],
    5:[817,169,61,44],6:[816,169,62,44],7:[814,170,63,44],8:[813,170,63,45],
    9:[812,171,64,45],10:[810,173,63,42],11:[808,173,64,43],12:[806,173,65,43],
    13:[805,174,64,44],14:[803,174,65,44],15:[800,174,65,45],16:[798,176,66,44],
    17:[796,176,65,44],18:[792,177,65,43],19:[790,178,65,44],20:[788,179,67,44],
    21:[786,179,68,44],22:[783,179,69,45],23:[780,180,70,45],24:[777,180,71,45],
    25:[774,181,71,45],26:[772,181,71,46],
}
WHITE_BOXES={
    1:[806,143,56,34],2:[805,142,57,35],3:[804,142,57,35],4:[803,142,58,35],
    5:[802,142,58,35],6:[801,142,58,36],7:[800,142,59,36],8:[799,142,59,36],
    9:[798,142,59,36],10:[797,142,59,37],
}


def draw_crop(frame,image_path):
    image=Image.open(image_path).convert("RGB")
    drawing=ImageDraw.Draw(image)
    for item in frame["objects"]:
        identity=item["track_id"]
        if identity not in (2,3,5):continue
        color={2:"magenta",3:"cyan",5:"yellow"}[identity]
        left,top,width,height=item["bbox_xywh"]
        drawing.rectangle((left,top,left+width,top+height),outline=color,width=1)
        drawing.text((left,max(0,top-12)),str(identity),fill=color,stroke_width=1,stroke_fill="black")
    panel=Image.new("RGB",(400,410),"#181818")
    panel.paste(image.crop((735,100,925,265)).resize((380,330),Image.Resampling.LANCZOS),(0,70))
    drawing=ImageDraw.Draw(panel)
    drawing.text((5,5),f"MOT {frame['mot_frame_id']} / source {frame['source_frame_id']}",fill="white")
    drawing.text((5,25),"ID3 black car: cyan | ID5 white car: yellow",fill="white")
    drawing.text((5,45),"VISUAL PROPOSAL - OWNER QA REQUIRED",fill="white")
    return panel


def main():
    original_hash=sha256(SOURCE)
    original=json.loads(SOURCE.read_text())
    if original["is_ground_truth"] or any(frame["review_status"]!="UNREVIEWED" for frame in original["frames"]):
        raise ValueError("Source is no longer an unreviewed draft")
    revised=copy.deepcopy(original)
    changes=[]
    image_hashes={}
    for frame in revised["frames"]:
        number=frame["mot_frame_id"]
        image_path=(SOURCE.parent/frame["image"]).resolve(strict=True)
        if number<=26:
            image_hashes[str(number)]=sha256(image_path)
            black=[item for item in frame["objects"] if item["track_id"]==3]
            if len(black)!=1:raise ValueError("Expected unique existing black-car ID3")
            before=copy.deepcopy(black[0])
            black[0].update(bbox_xywh=BLACK_BOXES[number],annotation_origin="ASSISTANT_PER_FRAME_VISUAL_PROPOSAL",proposal_status="REVIEW_REQUIRED",human_corrected=False,
                            original_detector_confidence=before["detector_confidence"],detector_confidence=None)
            changes.append({"mot_frame_id":number,"source_frame_id":number-1,"action":"REPLACE_BLACK_CAR_BBOX","track_id":3,"before":before,"after":copy.deepcopy(black[0])})
        if number<=10:
            if any(item["track_id"]==5 for item in frame["objects"]):raise ValueError("White-car ID5 already exists")
            white={"track_id":5,"bbox_xywh":WHITE_BOXES[number],"class":None,"candidate_class":None,"detector_confidence":None,
                   "annotation_origin":"ASSISTANT_PER_FRAME_VISUAL_PROPOSAL","human_corrected":False,"proposal_status":"REVIEW_REQUIRED"}
            frame["objects"].append(white)
            changes.append({"mot_frame_id":number,"source_frame_id":number-1,"action":"ADD_MISSING_WHITE_CAR","track_id":5,"before":None,"after":copy.deepcopy(white)})
        if number in (3,4):
            extra=[item for item in frame["objects"] if item["track_id"]==2]
            if len(extra)!=1:raise ValueError("Expected residual merged ID2 bbox")
            frame["objects"]=[item for item in frame["objects"] if item["track_id"]!=2]
            changes.append({"mot_frame_id":number,"source_frame_id":number-1,"action":"REMOVE_RESIDUAL_MERGED_DUPLICATE","track_id":2,"before":extra[0],"after":None})
        frame["image"]=Path(os.path.relpath(image_path,DESTINATION)).as_posix()
        identities=[item["track_id"] for item in frame["objects"]]
        if len(identities)!=len(set(identities)):raise ValueError("Duplicate frame/ID")
        for item in frame["objects"]:
            left,top,width,height=item["bbox_xywh"]
            if left<0 or top<0 or width<=0 or height<=0 or left+width>1280 or top+height>720 or item["class"] is not None:
                raise ValueError("Invalid proposed bbox/class")
    for before,after in zip(original["frames"],revised["frames"]):
        before_table={item["track_id"]:item for item in before["objects"]}
        after_table={item["track_id"]:item for item in after["objects"]}
        for identity,item in before_table.items():
            permitted=before["mot_frame_id"]<=26 and identity==3 or before["mot_frame_id"] in (3,4) and identity==2
            if not permitted and after_table.get(identity)!=item:raise ValueError("Unrelated object changed")
        if before["mot_frame_id"]>26 and before["objects"]!=after["objects"]:raise ValueError("Later frame changed")
    revised.update(draft_revision="initial_boxes_revision_002",status="PER_FRAME_VISUAL_PROPOSAL_REQUIRES_OWNER_QA",is_ground_truth=False,
                   revision_source_sha256=original_hash,correction_authorization="Owner agreed to a separate proposal after confirming two-car merged bbox issue",
                   coordinate_method="Individually specified on native-frame coordinate rulers; no optical-flow/model boxes/interpolation used",exact_bbox_approval="PENDING")
    DESTINATION.mkdir(exist_ok=False)
    QA.mkdir(exist_ok=False)
    (DESTINATION/"annotations_draft.json").write_text(json.dumps(revised,indent=2)+"\n")
    template=(ROOT/"src/tracking/annotation_editor.html").read_text(encoding="utf-8")
    (DESTINATION/"editor.html").write_text(template.replace("__ANNOTATION_PAYLOAD__",json.dumps(revised,allow_nan=False).replace("<","\\u003c")),encoding="utf-8")
    with (DESTINATION/"tracks_draft_mot.txt").open("x",encoding="utf-8") as handle:
        for frame in revised["frames"]:
            for item in frame["objects"]:
                box=",".join(f"{value:.6f}" for value in item["bbox_xywh"])
                confidence=item["detector_confidence"] if item["detector_confidence"] is not None else -1
                handle.write(f"{frame['mot_frame_id']},{item['track_id']},{box},{confidence},-1,-1,-1\n")
    panels=[draw_crop(frame,(DESTINATION/frame["image"]).resolve()) for frame in revised["frames"][:26]]
    sheets=[]
    for start,end in ((1,10),(11,18),(19,26)):
        columns=5 if end-start+1==10 else 4
        sheet=Image.new("RGB",(columns*400,820),"#181818")
        for position,number in enumerate(range(start,end+1)):
            sheet.paste(panels[number-1],((position%columns)*400,(position//columns)*410))
        name=f"proposed_{start:02d}_{end:02d}.jpg"
        sheet.save(QA/name,quality=95)
        sheets.append(name)
    video=cv2.VideoWriter(str(QA/"proposed_first26.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),3,(400,410))
    if not video.isOpened():raise RuntimeError("QA writer unavailable")
    try:
        for panel in panels:video.write(cv2.cvtColor(np.asarray(panel),cv2.COLOR_RGB2BGR))
    finally:video.release()
    capture=cv2.VideoCapture(str(QA/"proposed_first26.mp4"))
    count=0
    while capture.read()[0]:count+=1
    capture.release()
    if count!=26 or sha256(SOURCE)!=original_hash:raise RuntimeError("Source/video integrity mismatch")
    report={"status":"DRAFT_PROPOSAL_CREATED_NOT_GT","frames":300,"black_bbox_replacements":26,"white_bbox_additions":10,"merged_bbox_deletions":2,
            "remaining_bbox_observations":sum(len(frame["objects"]) for frame in revised["frames"]),"existing_ids_reused":[3,5],"new_ids_created":False,
            "changed_mot_frames":[1,26],"changed_source_frames":[0,25],"later_frames_unchanged":True,"other_retained_objects_unchanged":True,
            "source_draft_sha256":original_hash,"revision_sha256":sha256(DESTINATION/"annotations_draft.json"),"input_image_sha256":image_hashes,
            "changes":changes,"source_modified":False,"raw_predictions_modified":False,"human_bbox_approval":"REVIEW_REQUIRED",
            "no_interpolation":True,"semantic_classes":"PENDING","confidence_note":"Corrected/new visual boxes have confidence null; draft MOT encodes unknown as -1. Prior detector scores retained only as historical provenance.",
            "scope_note":"Only first26 native frames inspected; continuity/extent of later detector boxes not certified", "revision_directory":str(DESTINATION)}
    (DESTINATION/"correction_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    (QA/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    editor=Path(os.path.relpath(DESTINATION/"editor.html",QA)).as_posix()
    images="".join(f'<h2>{name}</h2><img src="{name}">' for name in sheets)
    (QA/"index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>NE12th proposed bbox split</title><style>body{{background:#181818;color:white;font:18px sans-serif}}img{{max-width:100%}}video{{height:550px;max-width:100%}}</style><h1>NE12th: per-frame proposal — owner QA required</h1><p>Black car ID3 cyan; white car ID5 yellow. Added10 missing white boxes, replaced26 black boxes, removed2 residual merged boxes. No interpolation and no raw/old draft edits. Approximate pixel boundaries require human confirmation; later frames are unchanged, not certified. <a href="{editor}">Open new revision editor</a></p><video controls src="proposed_first26.mp4"></video>'+images,encoding="utf-8")
    (DESTINATION/"README.md").write_text("# NE12th initial per-frame visual proposal\n\nOwner authorized a separate draft after confirming two vehicles were merged. ID3 is black car; ID5 is white car. Proposed native pixel boxes are independently specified for each inspected frame, not interpolated or inferred from YOLO boxes. AddID5 at MOT1..10; replaceID3 at MOT1..26 because early merged/oversized boxes persisted into later reference frames; remove residual merged ID2 at MOT3/4. No arbitrary IDs, no source/current GT edit, no label-confidence fabricated. All proposed extents still require owner QA. Later frames and other boxes preserved; their completeness/continuity not certified. Classes remain null/PENDING.\n\nOpen editor.html for refinements; separate browser draft revision avoids restoring old uncorrected data. All frame statuses remain UNREVIEWED. Export review JSON for import validation before GT promotion. Draft MOT score-1 means unknown visual-proposal confidence; it is not authoritative GT. Original draft and prior cleanup revision remain intact. See correction_manifest.json for exact before/after records and hashes.\n")
    print(json.dumps({key:report[key] for key in ('status','black_bbox_replacements','white_bbox_additions','merged_bbox_deletions','remaining_bbox_observations','source_modified')},indent=2))


if __name__=="__main__":
    main()
