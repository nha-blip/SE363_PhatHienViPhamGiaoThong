import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_tracking_conflict_qa import ROOT, SOURCES, annotations, decoded_frames, draw_boxes, font, sha256


OUTPUT = ROOT / "runs/project363/tracking_audit/work_5_1/neighbor_qa"
CASES = ((2, 31, 0, 12), (208, 49, 193, 223))


def overlaps(box, region):
    left, top, width, height = box
    return left < region[2] and left + width > region[0] and top < region[3] and top + height > region[1]


def panel_sheet(filename, panels, heading):
    canvas = Image.new("RGB", (2400, 590), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), heading, fill="black", font=font(22))
    for index, (image, caption) in enumerate(panels):
        scale = min(790 / image.width, 490 / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
        canvas.paste(resized, (index * 800 + 5, 55))
        draw.text((index * 800 + 8, 553), caption, fill="black", font=font(17))
    canvas.save(OUTPUT / filename, quality=92)


def main():
    directory = SOURCES / "VNTraffic"
    evidence = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".txt", ".json", ".mp4"})
    before = {path.relative_to(ROOT).as_posix(): sha256(path) for path in evidence}
    text, structured = annotations(directory, "VNTraffic")
    wanted = {frame for _, _, first, last in CASES for frame in range(first, last + 1)}
    frames = decoded_frames(directory / "VNTraffic_Original-video.mp4", wanted)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    sections = []
    for conflict, identity, first, last in CASES:
        target_boxes = [item["box"] for frame in range(first, last + 1) for item in text[frame] if item["id"] == identity]
        if len([item for item in text[conflict] if item["id"] == identity]) != 2:
            raise ValueError("Expected two original conflicting boxes")
        width, height = frames[first].size
        region = (max(0, int(min(box[0] for box in target_boxes) - 90)), max(0, int(min(box[1] for box in target_boxes) - 90)), min(width, int(max(box[0] + box[2] for box in target_boxes) + 90)), min(height, int(max(box[1] + box[3] for box in target_boxes) + 90)))
        samples = []
        animation = []
        for frame in range(first, last + 1):
            objects = [dict(item) for item in text[frame]]
            colors = []
            occurrence = 0
            for item in objects:
                color = "#32ff55"
                if item["id"] == identity:
                    occurrence += 1
                    color = "#ff55ff"
                    if frame == conflict:
                        marker = "A" if occurrence == 1 else "B"
                        item["label"] = f"{marker}: ID {identity}"
                        color = "#ff4040" if occurrence == 1 else "#00aaff"
                colors.append(color)
            annotated = draw_boxes(frames[frame], objects, colors)
            filename = f"VNTraffic_case_{conflict:06d}_frame_{frame:06d}.jpg"
            panel_sheet(filename, [(annotated, "Full frame: all existing TXT IDs"), (annotated.crop(region), "Fixed ROI: target magenta; conflict A red / B blue"), (frames[frame].crop(region), "Same ROI without boxes: inspect vehicle appearance")], f"VNTraffic | case frame {conflict}, ID {identity} | frame {frame} (0-based) | {frame / 30:.3f}s")
            nearby = [item for item in objects if overlaps(item["box"], region)]
            samples.append({"frame_zero_based": frame, "preview": filename, "txt_objects_in_roi": nearby, "json_objects_in_roi": [item for item in structured[frame] if overlaps(item["box"], region)], "target_id_count": occurrence})
            image = annotated.crop(region)
            image.thumbnail((600, 420))
            animation_frame = Image.new("RGB", (600, 455), "#eeeeee")
            animation_frame.paste(image, (0, 35))
            ImageDraw.Draw(animation_frame).text((8, 5), f"Frame {frame} | target ID {identity} | TXT", fill="black", font=font(20))
            animation.append(animation_frame)
        gif = f"VNTraffic_case_{conflict:06d}_neighbors.gif"
        animation[0].save(OUTPUT / gif, save_all=True, append_images=animation[1:], duration=200, loop=0)
        statement = "Human reports A/B are the same vehicle; selection of a valid box remains pending." if conflict == 2 else "Human reports A/B are two different vehicles; correct trajectory IDs remain pending."
        cases.append({"conflict_frame": conflict, "existing_track_id": identity, "frame_window": [first, last], "fixed_roi_xyxy": region, "human_observation": statement, "decision_status": "REVIEW_REQUIRED", "animation": gif, "samples": samples})
        cards = ''.join(f'<article id="case-{conflict}-frame-{item["frame_zero_based"]}"><h3>Frame {item["frame_zero_based"]}</h3><a href="{item["preview"]}"><img loading="lazy" src="{item["preview"]}" alt="frame {item["frame_zero_based"]}"></a></article>' for item in samples)
        sections.append(f'<section><h2>Case frame {conflict} / existing ID {identity}</h2><p>{html.escape(statement)}</p><p>Frames {first}–{last}; GIF playback 5 fps, slower than original 30 fps. Colors are display markers, not new IDs.</p><img class="gif" src="{gif}" alt="neighbor animation">{cards}</section>')
    after = {path.relative_to(ROOT).as_posix(): sha256(path) for path in evidence}
    if before != after:
        raise RuntimeError("Source bytes changed during QA generation")
    manifest = {"human_qa_status": "REVIEW_REQUIRED", "source_modified": False, "gt_repaired": False, "track_ids_generated": False, "source_sha256_before": before, "source_sha256_after": after, "sampling": "Deterministic consecutive windows 0..12 and 193..223, decoded sequentially from original video; no original frames saved separately", "prior_human_feedback": {"reviewed_track_id_continuity": "PASS on reviewed samples, not whole source", "aicc22_added_ids": "Stable and nonduplicate on reviewed samples; not confirmation of all added box geometry"}, "cases": cases}
    (OUTPUT / "qa_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>VNTraffic neighbor QA</title><style>body{font-family:Arial;background:#eee;margin:24px}article{background:white;padding:12px;margin:16px 0}article img{width:100%}.gif{max-width:600px}p{max-width:1100px;line-height:1.6}</style><h1>VNTraffic neighbor QA — REVIEW_REQUIRED</h1><p>Open full-size images to inspect. All TXT IDs are displayed. Target ID is magenta except at the conflict frame, where original row A is red and row B is blue. A/B are row markers only; colors do not establish vehicle identity. Third panel has no boxes.</p><p>Case 2: choose A or B only if the box fits the same vehicle and neighboring frames. Case 208: trace each vehicle before/after; report existing neighboring ID if identifiable, otherwise UNCLEAR. No automatic repair or new ID.</p>' + ''.join(sections) + '</html>'
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    instructions = """# VNTraffic — QA frame lân cận

Mở `index.html` bằng trình duyệt. Nhấn ảnh để xem đầy đủ.

- Frame 2 / ID 31: xem frame 0–12. A đỏ và B xanh dương là hai dòng bbox gốc; chọn bbox ôm xe đúng hơn, hoặc ghi CHƯA RÕ.
- Frame 208 / ID 49: xem frame 193–223. Lần theo hai xe để xác định xe nào thuộc ID 49, xe còn lại có ID gốc nào ở frame trước/sau. Không đoán hoặc tự tạo ID mới.
- Các frame khác: ID đang xét màu tím; các ID khác màu xanh lá. Khung giữa phóng to cùng vùng cố định, khung phải không vẽ bbox.
- GIF chạy chậm 5 fps để xem; nguồn 30 fps. Nếu nhãn che xe, đối chiếu khung không bbox.

Phản hồi: case / frame / A hoặc B / ID gốc quan sát được / nhận xét hoặc CHƯA RÕ.

HUMAN_QA_STATUS: REVIEW_REQUIRED. Không sửa GT, không remap ID, không tạo dataset. SHA-256 các video và annotation VNTraffic không đổi.
"""
    (OUTPUT / "README.md").write_text(instructions, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "preview_images": sum(len(case["samples"]) for case in cases), "animations": len(cases), "source_sha256_unchanged": before == after, "human_qa_status": "REVIEW_REQUIRED"}))


if __name__ == "__main__":
    main()
