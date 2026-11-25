import copy
import json
from pathlib import Path

import cv2
import numpy as np

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256


ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / "data/Tracking363_working/annotation_drafts/bellevue_ne12th_001"
SOURCE = AREA / "initial_boxes_revision_002/annotations_draft.json"
OUTPUT = AREA / "id_cleanup_revision_003"
QA = ROOT / "runs/project363/tracking/bellevue_draft_cleanup/ne12th_id_cleanup"
ALIASES = {9: 3, 23: 13}


def overlay(image, objects, frame, title, removed_ids=()):
    rendered = image.copy()
    for item in objects:
        left, top, width, height = item["bbox_xywh"]
        color = (0,0,255) if item["track_id"] in removed_ids else (255,255,0)
        cv2.rectangle(rendered,(int(left),int(top)),(int(left+width),int(top+height)),color,2)
        cv2.putText(rendered,f"ID{item['track_id']}",(int(left),max(55,int(top)-4)),cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2)
    cv2.rectangle(rendered,(0,0),(1280,40),(0,0,0),-1)
    cv2.putText(rendered,f"{title} | MOT {frame} / source {frame-1}",(8,27),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    return rendered


def main():
    before_hash = sha256(SOURCE)
    before = json.loads(SOURCE.read_text())
    after = copy.deepcopy(before)
    removed = []
    for frame in after["frames"]:
        objects = {item["track_id"]: item for item in frame["objects"]}
        for alias, retained in ALIASES.items():
            if alias in objects:
                if retained not in objects:
                    raise ValueError("Canonical bbox absent; do not silently delete a vehicle")
                removed.append({"mot_frame_id":frame["mot_frame_id"],"source_frame_id":frame["source_frame_id"],
                                "removed_track_id":alias,"retained_track_id":retained,
                                "removed_object":copy.deepcopy(objects[alias]),"retained_object":copy.deepcopy(objects[retained]),
                                "basis":"Owner explicitly confirmed same physical vehicle"})
        frame["objects"] = [item for item in frame["objects"] if item["track_id"] not in ALIASES]
    if len(removed)!=9:
        raise ValueError("Unexpected duplicate scope")
    for original, revised in zip(before["frames"],after["frames"]):
        expected=[item for item in original["objects"] if item["track_id"] not in ALIASES]
        if expected!=revised["objects"]:
            raise ValueError("Unrelated bbox or ID changed")
        ids=[item["track_id"] for item in revised["objects"]]
        if len(ids)!=len(set(ids)):
            raise ValueError("Duplicate IDs remain within a frame")
        if (SOURCE.parent/original["image"]).resolve()!=(OUTPUT/revised["image"]).resolve():
            raise ValueError("Image reference mismatch")
    if before["frames"][:26]!=after["frames"][:26]:
        raise ValueError("Approved first26 frames changed")
    approval = AREA / "initial_boxes_revision_002/human_bbox_approval.json"
    approval_hash = sha256(approval)
    after.update(draft_revision="id_cleanup_revision_003",status="OWNER_CONFIRMED_DUPLICATE_IDS_REMOVED_DRAFT_NOT_FINAL_GT",
                 revision_source_sha256=before_hash,is_ground_truth=False,
                 inherited_scoped_bbox_approval=str(approval.relative_to(ROOT)),
                 inherited_scoped_approval_sha256=approval_hash,
                 duplicate_identity_decisions={str(alias):retained for alias,retained in ALIASES.items()})
    OUTPUT.mkdir(exist_ok=False)
    QA.mkdir(exist_ok=False)
    (OUTPUT/"annotations_draft.json").write_text(json.dumps(after,indent=2)+"\n")
    template=(ROOT/"src/tracking/annotation_editor.html").read_text(encoding="utf-8")
    (OUTPUT/"editor.html").write_text(template.replace("__ANNOTATION_PAYLOAD__",json.dumps(after,allow_nan=False).replace("<","\\u003c")),encoding="utf-8")
    with (OUTPUT/"tracks_draft_mot.txt").open("x",encoding="utf-8") as handle:
        for frame in after["frames"]:
            for item in frame["objects"]:
                confidence=item.get("detector_confidence")
                confidence=confidence if confidence is not None else -1
                box=",".join(f"{value:.6f}" for value in item["bbox_xywh"])
                handle.write(f"{frame['mot_frame_id']},{item['track_id']},{box},{confidence},-1,-1,-1\n")
    changed={record["mot_frame_id"] for record in removed}
    previews=[]
    writer=cv2.VideoWriter(str(QA/"ne12th_frames_027_300_updated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),10,(1280,720))
    if not writer.isOpened():raise RuntimeError("QA writer unavailable")
    try:
        for original,revised in zip(before["frames"],after["frames"]):
            image=cv2.imread(str((SOURCE.parent/original["image"]).resolve(strict=True)))
            if image is None:raise ValueError("Image unavailable")
            frame=revised["mot_frame_id"]
            rendered=overlay(image,revised["objects"],frame,"ID CLEANUP DRAFT NOT GT")
            if frame>=27:writer.write(rendered)
            if frame in changed:
                deleted=[record["removed_track_id"] for record in removed if record["mot_frame_id"]==frame]
                combined=np.hstack([cv2.resize(overlay(image,original["objects"],frame,"BEFORE: duplicate red",deleted),(640,360)),cv2.resize(rendered,(640,360))])
                name=f"frame_{frame:06d}.jpg"
                if not cv2.imwrite(str(QA/name),combined):raise RuntimeError("Preview write failed")
                previews.append(name)
    finally:writer.release()
    capture=cv2.VideoCapture(str(QA/"ne12th_frames_027_300_updated.mp4"))
    decoded=0
    while capture.read()[0]:decoded+=1
    capture.release()
    if decoded!=274 or sha256(SOURCE)!=before_hash or sha256(approval)!=approval_hash:
        raise RuntimeError("Protected input/video integrity mismatch")
    report={"status":"APPLIED_TO_NEW_DRAFT_REVISION","owner_confirmed_identity_pairs":[[3,9],[13,23]],"retained_ids":[3,13],
            "removed_bbox_observations":9,"removed_id9_mot_frames":[36,37],"removed_id23_mot_frames":list(range(150,157)),
            "source_observations":sum(len(frame["objects"]) for frame in before["frames"]),
            "remaining_observations":sum(len(frame["objects"]) for frame in after["frames"]),
            "remaining_track_ids":sorted({item["track_id"] for frame in after["frames"] for item in frame["objects"]}),
            "source_draft_sha256":before_hash,"revision_sha256":sha256(OUTPUT/"annotations_draft.json"),
            "removal_manifest":removed,"all_retained_bbox_coordinates_and_ids_unchanged":True,"approved_first26_frames_unchanged":True,
            "raw_source_modified":False,"original_draft_modified":False,"predictions_modified":False,"is_ground_truth":False,
            "scope":"Remove duplicate observations only where owner-confirmed canonical ID is also present; no track renaming, interpolation or unreviewed bbox selection",
            "human_qa_status":"REVIEW_UPDATED_VIDEO_BEFORE_FINAL_GT","newport_status":"Owner provisionally acceptable; no new Newport edits",
            "revision_directory":str(OUTPUT),"qa_video_frames":decoded}
    (OUTPUT/"identity_resolution_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    (QA/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUTPUT/"README.md").write_text("# NE12th ID-cleanup revision003, still draft\n\nOwner confirmed3/9 and13/23 are the same physical vehicles. Keep continuousIDs3 and13; remove duplicateID9 observations at MOT36/37 andID23 at150..156 because canonical boxes already exist. All retained bbox coordinates andIDs are unchanged, as are approved first26-frame corrections inherited from revision002. No ID remapping, gap repair, source/cache changes or finalGT promotion. See identity_resolution_manifest.json for exact before/after observations and hashes.\n\nOpen editor.html for further manual review. Distinct browser-save key prevents importing stale duplicate boxes from old editors. Frame-review flags remain historical/unreviewed; scoped approval does not certify every frame/object.\n")
    images="".join(f'<h2>{name}</h2><img src="{name}">' for name in previews)
    (QA/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>NE12th ID cleanup</title><style>body{background:#181818;color:white;font:18px sans-serif}img,video{max-width:100%}</style><h1>NE12th: keep IDs3/13, remove duplicates9/23</h1><p>Owner-confirmed same-vehicle identities. Updated MOT27..300 video at10fps; original drafts/caches and approved first26 frames unchanged. Still draft, not finalGT.</p><video controls src="ne12th_frames_027_300_updated.mp4"></video>'+images,encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("status","removed_bbox_observations","remaining_observations","remaining_track_ids","approved_first26_frames_unchanged")},indent=2))


if __name__=="__main__":
    main()
