"""Tighten hand-drawn boxes in qa_result.json.

For each human box: crop a context window (box size x4, at least 96 px), upscale it to 640, run every
light (or vehicle) model at conf 0.01 and take the predicted box with the highest IoU to the hand box.
Accepted only if IoU >= 0.3; otherwise the hand box is kept. The original is kept as xyxy_manual.
A review sheet (white = hand, cyan = refined) is written for human confirmation.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from external_qa_predict_20261112 import MODELS, iou  # noqa: E402

WORK = Path(__file__).resolve().parents[1] / "data/external_qa_bellevue_20261112"


def main():
    result = WORK / "qa_result.json"
    state = json.loads(result.read_text(encoding="utf-8"))
    (WORK / "qa_result_before_refine.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    models = {g: [YOLO(str(w)) for w in ms.values()] for g, ms in MODELS.items()}
    tiles, report = [], []
    for im in state["images"]:
        pil = Image.open(im["path"]).convert("RGB")
        W, H = pil.size
        for b in im["boxes"]:
            if b.get("from") != "human":
                continue
            hand = b.get("xyxy_manual", b["xyxy"])
            x1, y1, x2, y2 = hand
            side = max(96.0, 4 * max(x2 - x1, y2 - y1))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cx1, cy1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
            cx2, cy2 = min(W, int(cx + side / 2)), min(H, int(cy + side / 2))
            crop = pil.crop((cx1, cy1, cx2, cy2))
            scale = 640 / max(crop.size)
            big = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.BICUBIC)
            best, best_iou = None, 0.0
            for m in models[b["group"]]:
                r = m.predict(big, imgsz=640, conf=0.01, verbose=False)[0]
                for p in r.boxes.xyxy.tolist():
                    q = [p[0] / scale + cx1, p[1] / scale + cy1, p[2] / scale + cx1, p[3] / scale + cy1]
                    v = iou(q, hand)
                    if v > best_iou:
                        best, best_iou = q, v
            b["xyxy_manual"] = hand
            if best is not None and best_iou >= 0.3:
                b["xyxy"] = [round(v, 1) for v in best]
                b["refined"] = round(best_iou, 3)
            else:
                b["xyxy"] = hand
                b["refined"] = None
            report.append(b["refined"])
            # review tile
            t = crop.resize((round(crop.width * scale / 2), round(crop.height * scale / 2)), Image.BICUBIC)
            d = ImageDraw.Draw(t)
            s = scale / 2
            d.rectangle([(hand[0] - cx1) * s, (hand[1] - cy1) * s, (hand[2] - cx1) * s, (hand[3] - cy1) * s], outline="white", width=1)
            if b["refined"]:
                bx = b["xyxy"]
                d.rectangle([(bx[0] - cx1) * s, (bx[1] - cy1) * s, (bx[2] - cx1) * s, (bx[3] - cy1) * s], outline="cyan", width=2)
            d.text((3, 3), f"#{len(tiles) + 1} {Path(im['path']).parent.parent.name[9:15]}/{Path(im['path']).stem} "
                           f"{'IoU %.2f' % b['refined'] if b['refined'] else 'GIU TAY'}", fill="yellow")
            tiles.append(t)
    result.write_text(json.dumps(state, indent=1), encoding="utf-8")
    cols, size = 6, 320
    sheet = Image.new("RGB", (cols * size, ((len(tiles) + cols - 1) // cols) * size), "black")
    for k, t in enumerate(tiles):
        t.thumbnail((size - 4, size - 4))
        sheet.paste(t, ((k % cols) * size, (k // cols) * size))
    sheet.save(WORK / "refine_review_sheet.jpg", quality=92)
    print("human boxes", len(report), "refined", sum(r is not None for r in report), "kept manual", sum(r is None for r in report))


if __name__ == "__main__":
    main()
