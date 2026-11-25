import json
from pathlib import Path

import cv2

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.io import sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/project363/tracking/bellevue_remaining_qa_videos"
JOBS = (
    ("ne12th_frames_027_300", "bellevue_ne12th_001", "initial_boxes_revision_002", 27, 300),
    ("newport_frames_001_300", "bellevue_newport_001", "cleanup_revision_001", 1, 300),
)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    reports = []
    for name, sequence, revision, start, end in JOBS:
        draft = ROOT / "data/Tracking363_working/annotation_drafts" / sequence / revision / "annotations_draft.json"
        original_hash = sha256(draft)
        payload = json.loads(draft.read_text(encoding="utf-8"))
        selected = payload["frames"][start-1:end]
        if [row["mot_frame_id"] for row in selected] != list(range(start, end+1)):
            raise ValueError("Missing or unordered draft frames")
        video = OUTPUT / f"{name}.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (payload["width"], payload["height"]))
        if not writer.isOpened():
            raise RuntimeError("Video writer unavailable")
        image_hashes = {}
        try:
            for row in selected:
                image_path = (draft.parent / row["image"]).resolve(strict=True)
                image_hashes[str(row["mot_frame_id"])] = sha256(image_path)
                image = cv2.imread(str(image_path))
                if image is None or image.shape[:2] != (payload["height"], payload["width"]):
                    raise ValueError("Image decode/size mismatch")
                for item in row["objects"]:
                    left, top, width, height = item["bbox_xywh"]
                    identity = item["track_id"]
                    if sequence == "bellevue_ne12th_001" and identity == 5:
                        color = (0, 255, 255)
                    elif sequence == "bellevue_ne12th_001" and identity == 3:
                        color = (255, 255, 0)
                    else:
                        color = (int((identity*73)%200+55), int((identity*41)%200+55), 255)
                    cv2.rectangle(image, (int(left), int(top)), (int(left+width), int(top+height)), color, 2)
                    label = f"ID {identity}"
                    text_left, text_top = int(left), max(60, int(top)-5)
                    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
                    cv2.rectangle(image, (text_left, text_top-23), (min(payload["width"],text_left+text_width+4), text_top+3), (0,0,0), -1)
                    cv2.putText(image, label, (text_left, text_top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(image, (0,0), (payload["width"],40), (0,0,0), -1)
                title = f"{sequence} | MOT {row['mot_frame_id']} / source {row['source_frame_id']} | DRAFT - QA REQUIRED"
                cv2.putText(image, title, (8,27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)
                writer.write(image)
        finally:
            writer.release()
        capture = cv2.VideoCapture(str(video))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        if decoded != end-start+1 or sha256(draft) != original_hash:
            raise RuntimeError("Video count or protected-draft integrity mismatch")
        for row in selected:
            if sha256((draft.parent/row["image"]).resolve(strict=True)) != image_hashes[str(row["mot_frame_id"])]:
                raise RuntimeError("Original frame changed")
        report = {"sequence":sequence, "revision":revision, "video":video.name, "mot_frame_range":[start,end],
                  "source_frame_range":[start-1,end-1], "decoded_frames":decoded, "playback_fps":10,
                  "source_fps":payload["fps"], "playback_duration_seconds":decoded/10,
                  "all_frames_included":True, "no_interpolation_or_frame_skipping":True,
                  "source_draft":str(draft.relative_to(ROOT)), "source_draft_sha256":original_hash,
                  "source_image_sha256":image_hashes, "video_sha256":sha256(video), "images_and_annotations_unchanged":True,
                  "human_qa_status":"REVIEW_REQUIRED", "is_ground_truth":False}
        reports.append(report)
        print(f"VIDEO_READY: {video.name} | {decoded} frames | 10fps slow playback", flush=True)
    (OUTPUT/"manifest.json").write_text(json.dumps(reports,indent=2)+"\n",encoding="utf-8")
    sections = []
    for report in reports:
        sections.append(f'<section><h2>{report["sequence"]}: MOT {report["mot_frame_range"][0]}–{report["mot_frame_range"][1]}</h2><p>All {report["decoded_frames"]} consecutive frames, 10fps slow playback. Actual MOT/source numbers are printed on each frame.</p><video controls preload="metadata" src="{report["video"]}"></video><p><a href="{report["video"]}">Open/download MP4</a></p></section>')
    (OUTPUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Remaining Bellevue QA videos</title><style>body{background:#181818;color:white;font:18px sans-serif;margin:20px}video{width:100%;max-width:1280px}section{border-top:1px solid #777}</style><h1>Remaining-frame bbox/ID QA</h1><p>These are current draft overlays, not final GT. Videos play at10fps, about one-third source speed. No source or annotation changes. If your browser cannot play MP4v, use VLC.</p><label>Playback speed <select onchange="document.querySelectorAll(\'video\').forEach(video=>video.playbackRate=Number(this.value))"><option value="1">1x (10fps)</option><option value="0.5">0.5x</option><option value="0.25">0.25x</option></select></label>'+"\n".join(sections),encoding="utf-8")
    (OUTPUT/"README.md").write_text("# Remaining Bellevue QA videos\n\nNE12th: MOT27..300/source26..299 using initial_boxes_revision_002. Newport: MOT1..300/source0..299 using cleanup_revision_001. All574 selected images appear exactly once, in order, with current draft bbox/ID overlays and frame numbers. Slow playback10fps, not a statement of sourceFPS. Native resolution1280x720; no interpolation/skipped frames.\n\nOpen index.html or theMP4 files inVLC. NE12th ID3 is cyan and ID5 yellow; other IDs use deterministic colors. Check missing vehicles, bbox extent, duplicate/merged boxes and physical identity continuity. Draft-only overlays do not certify GT. Images/draft hashes were reverified unchanged. See manifest.json for exact source revisions and hashes. No model inference, tracker rerun, annotation changes or automated QA approval.\n",encoding="utf-8")


if __name__ == "__main__":
    main()
