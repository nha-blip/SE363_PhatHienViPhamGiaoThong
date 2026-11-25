import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/project363/tracking_audit/work_5_1/conflict_qa"
SOURCES = ROOT / "data/Vehicle_Tracking"


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def annotations(directory, name):
    text = defaultdict(list)
    with (directory / f"{name}_GroundTruth.txt").open(newline="") as handle:
        for row in csv.reader(handle):
            values = list(map(float, row))
            text[int(values[0])].append({"id": int(values[1]), "box": values[2:6]})
    document = json.loads((directory / f"{name}_GroundTruth.json").read_text(encoding="utf-8"))
    structured = defaultdict(list)
    for identity, track in document.items():
        if len(track["frames"]) != len(track["boxes"]):
            raise ValueError("JSON frame/box length mismatch")
        for frame, box in zip(track["frames"], track["boxes"]):
            structured[int(Path(frame).stem)].append({"id": int(identity), "box": box})
    return text, structured


def decoded_frames(path, wanted):
    capture = cv2.VideoCapture(str(path))
    frames = {}
    if not capture.isOpened():
        raise ValueError(f"Cannot open {path}")
    try:
        for index in range(max(wanted) + 1):
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Cannot sequentially decode frame {index}: {path}")
            if index in wanted:
                frames[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def font(size=20):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def draw_boxes(image, objects, colors=None):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(objects):
        color = colors[index] if colors else "#32ff55"
        left, top, width, height = item["box"]
        draw.rectangle((left, top, left + width, top + height), outline=color, width=3)
        label = item.get("label", f"ID {item['id']}")
        position = (max(0, left), max(0, top - 24))
        bounds = draw.textbbox(position, label, font=font())
        draw.rectangle(bounds, fill="black")
        draw.text(position, label, fill=color, font=font())
    return canvas


def sheet(filename, panels, heading, note):
    width, height = 960, 620
    canvas = Image.new("RGB", (width * len(panels), height + 90), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), heading, fill="black", font=font(24))
    draw.text((12, 42), note, fill="black", font=font(18))
    for index, (image, label) in enumerate(panels):
        image = image.copy()
        scale = min((width - 12) / image.width, (height - 40) / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
        canvas.paste(image, (index * width + 6, 90))
        draw.text((index * width + 10, height + 52), label, fill="black", font=font(18))
    canvas.save(OUTPUT / filename, quality=93)
    return filename


def main():
    evidence = [path for name in ("VNTraffic", "AICC22-Custom") for path in (SOURCES / name).iterdir() if path.suffix.lower() in {".mp4", ".txt", ".json"}]
    before = {path.relative_to(ROOT).as_posix(): sha256(path) for path in evidence}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = []
    text, structured = annotations(SOURCES / "VNTraffic", "VNTraffic")
    frames = decoded_frames(SOURCES / "VNTraffic/VNTraffic_Original-video.mp4", {2, 208})
    for frame, identity in ((2, 31), (208, 49)):
        objects = [dict(item) for item in text[frame] if item["id"] == identity]
        if len(objects) != 2:
            raise ValueError("Expected two conflicting source boxes")
        for label, item in zip(("A", "B"), objects):
            item["label"] = label
        annotated = draw_boxes(frames[frame], objects, ["#ff4040", "#00aaff"])
        left = min(item["box"][0] for item in objects)
        top = min(item["box"][1] for item in objects)
        right = max(item["box"][0] + item["box"][2] for item in objects)
        bottom = max(item["box"][1] + item["box"][3] for item in objects)
        context = max(40, int(max(right - left, bottom - top) * 0.2))
        crop = (max(0, int(left - context)), max(0, int(top - context)), min(annotated.width, int(right + context)), min(annotated.height, int(bottom + context)))
        filename = sheet(f"VNTraffic_frame_{frame:06d}_ID_{identity}.jpg", [(annotated, "Full frame: only the conflicting pair is drawn"), (annotated.crop(crop), "Zoom: A red / B blue; ID labels are not vehicle classes")], f"VNTraffic | source frame {frame} | decoded index {frame} | existing ID {identity}", "Decide: same vehicle / different vehicles / invalid box / unclear. No annotation changed.")
        samples.append({"category": "VNTraffic conflict", "preview": filename, "source_frame": frame, "decoded_zero_based_index": frame, "existing_track_id": identity, "txt_objects": objects, "json_objects": [item for item in structured[frame] if item["id"] == identity], "human_decision": "REVIEW_REQUIRED"})
    text, structured = annotations(SOURCES / "AICC22-Custom", "AICC22-Custom")
    extra_counts = {}
    for frame in sorted(set(text) | set(structured)):
        original = Counter(tuple(item["box"]) for item in text[frame])
        expanded = Counter(tuple(item["box"]) for item in structured[frame])
        if original - expanded:
            raise ValueError(f"TXT geometry not subset at frame {frame}")
        extra_counts[frame] = sum((expanded - original).values())
    eligible = [frame for frame, count in extra_counts.items() if count]
    anchors = {eligible[0], eligible[len(eligible) // 2], eligible[-1]}
    for band in range(6):
        candidates = [frame for frame in eligible if min(5, (frame - 1) * 6 // 200) == band]
        if candidates:
            anchors.add(max(candidates, key=lambda frame: (extra_counts[frame], -frame)))
    selected = sorted({neighbor for frame in anchors for neighbor in (frame - 1, frame, frame + 1) if neighbor in structured})
    frames = decoded_frames(SOURCES / "AICC22-Custom/AICC22-Custom_Original-video.mp4", {frame - 1 for frame in selected})
    for frame in selected:
        remaining = Counter(tuple(item["box"]) for item in text[frame])
        colors = []
        extra_objects = []
        for item in structured[frame]:
            geometry = tuple(item["box"])
            shared = remaining[geometry] > 0
            if shared:
                remaining[geometry] -= 1
            else:
                extra_objects.append(item)
            colors.append("#32ff55" if shared else "#ff9b22")
        filename = sheet(f"AICC22_frame_{frame:06d}_TXT_vs_JSON.jpg", [(draw_boxes(frames[frame - 1], text[frame]), f"TXT: {len(text[frame])} boxes / source IDs"), (draw_boxes(frames[frame - 1], structured[frame], colors), f"JSON: {len(structured[frame])} boxes / +{extra_counts[frame]} orange")], f"AICC22 | source frame {frame} | decoded index {frame - 1} | inferred 1-based alignment", "Green: shared geometry. Orange: JSON-only. IDs differ between exports; do not compare ID numbers directly.")
        samples.append({"category": "AICC22 TXT vs JSON", "preview": filename, "source_frame": frame, "decoded_zero_based_index": frame - 1, "frame_base": "Inferred 1-based; requires human alignment confirmation", "anchor": frame in anchors, "txt_boxes": len(text[frame]), "json_boxes": len(structured[frame]), "json_only_objects": extra_objects, "human_decision": "REVIEW_REQUIRED"})
    after = {path.relative_to(ROOT).as_posix(): sha256(path) for path in evidence}
    if before != after:
        raise RuntimeError("Source evidence changed during preview generation")
    manifest = {"human_qa_status": "REVIEW_REQUIRED", "source_files_modified": False, "source_sha256_before": before, "source_sha256_after": after, "sampling": "Deterministic: VNTraffic two known conflicts; AICC22 first/middle/last extra frames plus maximum-extra frame in each of six temporal bands and immediate annotated neighbors. Not exhaustive manual review.", "aicc22_json_extra_observations_total": sum(extra_counts.values()), "aicc22_anchor_frames": sorted(anchors), "samples": samples}
    (OUTPUT / "qa_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    instructions = "Check VNTraffic red/blue boxes: one vehicle, two vehicles, invalid box, or unclear. Check AICC22 orange boxes: real vehicle, correct extent, no duplicate, and stable existing JSON ID across neighboring frames. Green matches geometry, not necessarily ID. Track IDs are NOT car/motorcycle/bus/truck class labels. Record filename, ID and decision. Do not edit source annotations."
    sections = []
    for category in ("VNTraffic conflict", "AICC22 TXT vs JSON"):
        cards = [f'<article><h3>{html.escape(item["preview"])}</h3><a href="{item["preview"]}"><img loading="lazy" src="{item["preview"]}" alt="{html.escape(item["preview"])}"></a></article>' for item in samples if item["category"] == category]
        sections.append(f"<h2>{category}</h2>" + "\n".join(cards))
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Tracking conflict QA</title><style>body{font-family:Arial;margin:24px;background:#eee}article{background:white;padding:12px;margin:18px 0}img{width:100%;height:auto}p{max-width:1100px;line-height:1.6}</style><h1>Tracking conflict QA — REVIEW_REQUIRED</h1><p>' + html.escape(instructions) + '</p><p>AICC22 alignment is inferred; confirm boxes align with the displayed vehicles. Open images for full resolution. Representative samples do not approve all extra observations.</p>' + "\n".join(sections) + '</html>'
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    (OUTPUT / "README.md").write_text("# Tracking conflict QA\n\nOpen `index.html` in a browser.\n\n" + instructions + "\n\nHUMAN_QA_STATUS: REVIEW_REQUIRED. Source SHA-256 unchanged. No repair, ID remapping, annotation edition selection or dataset creation performed.\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "samples": len(samples), "vnt_conflicts": 2, "aicc22_samples": len(selected), "extra_observations_total": sum(extra_counts.values()), "source_sha256_unchanged": before == after}))


if __name__ == "__main__":
    main()
