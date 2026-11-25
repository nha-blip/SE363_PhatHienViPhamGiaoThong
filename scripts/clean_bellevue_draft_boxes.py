import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.class_suggestions import overlap
from src.tracking.io import sha256


ROOT=Path(__file__).resolve().parents[1]
DRAFTS=ROOT/"data/Tracking363_working/annotation_drafts"
QA=ROOT/"runs/project363/tracking/bellevue_draft_cleanup"


def xyxy(item):
    left,top,width,height=item["bbox_xywh"]
    return left,top,left+width,top+height


def draw(image,objects,title,removed_ids=()):
    result=image.copy()
    for item in objects:
        left,top,right,bottom=xyxy(item)
        color=(0,0,255) if item["track_id"] in removed_ids else (255,255,0)
        cv2.rectangle(result,(int(left),int(top)),(int(right),int(bottom)),color,2)
        cv2.putText(result,f"Draft {item['track_id']}",(int(left),max(35,int(top))),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    cv2.rectangle(result,(0,0),(image.shape[1],40),(0,0,0),-1)
    cv2.putText(result,title,(8,27),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
    return result


def process(identity):
    source=DRAFTS/identity
    source_json=source/"annotations_draft.json"
    source_hash=sha256(source_json)
    original=json.loads(source_json.read_text())
    if original["is_ground_truth"] or any(frame["review_status"]!="UNREVIEWED" or any(item.get("human_corrected") for item in frame["objects"]) for frame in original["frames"]):
        raise ValueError("Do not alter human-reviewed draft without separate scope approval")
    cleaned=copy.deepcopy(original)
    removed=[]
    output=source/"cleanup_revision_001"
    output.mkdir(exist_ok=False)
    previews=QA/identity
    previews.mkdir()
    for frame in cleaned["frames"]:
        table={item["track_id"]:item for item in frame["objects"]}
        removals=[]
        for item in frame["objects"]:
            reason=None
            if identity=="bellevue_ne12th_001" and item["track_id"] in (2,8) and 3 in table:
                score=overlap(item["bbox_xywh"],xyxy(table[3]))
                if score>=0.85:
                    reason={"reason":"DUPLICATE_OF_CONTINUOUS_DRAFT_ID3","kept_track_id":3,"iou":score}
            if identity=="bellevue_newport_001" and item["track_id"]==15 and 172<=frame["mot_frame_id"]<=175:
                if not {7,8}.issubset(table) or min(overlap(item["bbox_xywh"],xyxy(table[kept])) for kept in (7,8))<0.3:
                    raise ValueError("Newport confirmed two-car geometry changed")
                reason={"reason":"BBOX_SPANS_TWO_SEPARATE_CARS_VISUALLY_CHECKED_NATIVE_FRAMES172_TO175","kept_track_ids":[7,8]}
            if reason:
                removals.append(item["track_id"])
                removed.append({"mot_frame_id":frame["mot_frame_id"],"source_frame_id":frame["source_frame_id"],"removed_object":copy.deepcopy(item),**reason})
        frame["objects"]=[item for item in frame["objects"] if item["track_id"] not in removals]
        image_path=(source/frame["image"]).resolve(strict=True)
        frame["image"]=Path(os.path.relpath(image_path,output)).as_posix()
    cleaned["draft_revision"]="cleanup_revision_001"
    cleaned["cleanup_source_sha256"]=source_hash
    cleaned["status"]="CLEANED_DRAFT_REQUIRES_HUMAN_REVIEW"
    cleaned["is_ground_truth"]=False
    (output/"annotations_draft.json").write_text(json.dumps(cleaned,indent=2)+"\n")
    template=(ROOT/"src/tracking/annotation_editor.html").read_text(encoding="utf-8")
    (output/"editor.html").write_text(template.replace("__ANNOTATION_PAYLOAD__",json.dumps(cleaned,allow_nan=False).replace("<","\\u003c")),encoding="utf-8")
    with (output/"tracks_draft_mot.txt").open("x",encoding="utf-8") as handle:
        for frame in cleaned["frames"]:
            for item in frame["objects"]:
                box=",".join(f"{value:.6f}" for value in item["bbox_xywh"])
                handle.write(f"{frame['mot_frame_id']},{item['track_id']},{box},{item['detector_confidence']:.6f},-1,-1,-1\n")
    video=cv2.VideoWriter(str(previews/"cleaned_draft.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),cleaned["fps"],(1280,720))
    if not video.isOpened():
        raise RuntimeError("Video writer unavailable")
    changed_frames=sorted({row["mot_frame_id"] for row in removed})
    selected=set(changed_frames[:2]+changed_frames[-2:])
    if changed_frames:selected.add(changed_frames[len(changed_frames)//2])
    names=[]
    try:
        for before,after in zip(original["frames"],cleaned["frames"]):
            image_path=(source/before["image"]).resolve(strict=True)
            image=cv2.imread(str(image_path))
            if image is None:raise RuntimeError("Missing image")
            expected=[item for item in before["objects"] if not any(record["mot_frame_id"]==before["mot_frame_id"] and record["removed_object"]["track_id"]==item["track_id"] for record in removed)]
            if after["objects"]!=expected:
                raise RuntimeError("Unexpected retained bbox/ID change")
            rendered=draw(image,after["objects"],f"CLEANED DRAFT NOT GT | source {after['source_frame_id']} / MOT {after['mot_frame_id']}")
            video.write(cv2.resize(rendered,(1280,720)))
            if after["mot_frame_id"] in selected:
                deleted=[item["track_id"] for item in before["objects"] if item not in after["objects"]]
                comparison=np.hstack([cv2.resize(draw(image,before["objects"],"BEFORE: removed boxes red",deleted),(640,360)),cv2.resize(rendered,(640,360))])
                name=f"frame_{after['mot_frame_id']:06d}.jpg"
                if not cv2.imwrite(str(previews/name),comparison):raise RuntimeError("Preview write failed")
                names.append(name)
    finally:
        video.release()
    capture=cv2.VideoCapture(str(previews/"cleaned_draft.mp4"))
    decoded=0
    while capture.read()[0]:decoded+=1
    capture.release()
    if decoded!=300 or sha256(source_json)!=source_hash:
        raise RuntimeError("Video validation or original draft hash check failed")
    for frame in cleaned["frames"]:
        ids=[item["track_id"] for item in frame["objects"]]
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate IDs after cleanup")
    report={"sequence_id":identity,"status":"APPLIED_TO_NEW_DRAFT_REVISION_NOT_GT","removed_bbox_observations":len(removed),
            "changed_mot_frames":changed_frames,"source_bbox_observations":sum(len(frame["objects"]) for frame in original["frames"]),
            "remaining_bbox_observations":sum(len(frame["objects"]) for frame in cleaned["frames"]),"removal_manifest":removed,
            "original_draft_sha256":source_hash,"cleaned_draft_sha256":sha256(output/"annotations_draft.json"),
            "all_retained_boxes_and_ids_unchanged":True,"raw_predictions_modified":False,"original_draft_modified":False,
            "human_review_status":"REVIEW_REQUIRED","is_ground_truth":False,"revision_directory":str(output),"qa_video_frames":decoded,
            "scope":"Only explicitly identified duplicate IDs/geometry; no arbitrary global suppression, ID merge, or semantic label assignment"}
    (output/"cleanup_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"README.md").write_text("# Cleaned Bellevue draft revision, NOT GT\n\nOpen editor.html for further review. Original draft and raw detector/tracker caches are preserved. Only identified redundant boxes were removed; every retained bbox and ID is unchanged. No track ID merge or rename, no full completeness certification. All300frames still UNREVIEWED. Previously saved browser edits from the original editor are not automatically imported; this revision uses a separate browser-save key. Export JSON after human corrections/review for import validation before GT promotion. See cleanup_manifest.json for every deletion and source hashes.\n")
    editor=Path(os.path.relpath(output/"editor.html",previews)).as_posix()
    images="".join(f'<img src="{name}">' for name in names)
    (previews/"index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>{identity} draft cleanup</title><style>body{{background:#181818;color:white;font:18px sans-serif}}img,video{{max-width:100%}}</style><h1>{identity}: removed {len(removed)} redundant draft observations</h1><p>NOT ground truth. Original draft preserved. <a href="{editor}">Open cleaned draft editor</a></p><video controls src="cleaned_draft.mp4"></video>'+images,encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("sequence_id","removed_bbox_observations","remaining_bbox_observations","changed_mot_frames")}),flush=True)
    return report


def main():
    QA.mkdir(exist_ok=False)
    reports=[process(identity) for identity in ("bellevue_ne12th_001","bellevue_newport_001")]
    (QA/"report.json").write_text(json.dumps(reports,indent=2)+"\n")
    links="".join(f'<li><a href="{row["sequence_id"]}/index.html">{row["sequence_id"]}: cleaned draft and editor</a></li>' for row in reports)
    (QA/"index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Bellevue draft cleanup</h1><p>Selected redundant boxes removed in separate revisions; original drafts remain. No final GT or full QA PASS.</p><ul>'+links+'</ul>')


if __name__=="__main__":
    main()
