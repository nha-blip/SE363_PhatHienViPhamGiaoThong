import argparse
import csv
import html
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [
    ("xe máy cao 32–64 px bị bỏ sót", lambda row: row["cls"] == "motorcycle" and row["size_bucket"] == "32-64"),
    ("xe khác bị bỏ sót, model không phát hiện gì", lambda row: row["cls"] != "motorcycle" and row["reason"] == "no_detection"),
    ("box GT bị bỏ sót vì model chỉ có một box cho hai nhãn chồng nhau", lambda row: row["reason"] == "box_used_by_another_gt"),
    ("model có thấy nhưng độ tin cậy dưới 0,25", lambda row: row["reason"].startswith("low_confidence")),
    ("box lớn từ 64 px vẫn bị bỏ sót", lambda row: row["size_bucket"] == ">=64"),
]
COLORS = {"TP": (80, 200, 80), "WRONG_CLASS": (0, 220, 255), "MISSED": (60, 60, 255)}


def render(row, others, destination):
    image = cv2.imread(row["image"])
    if image is None:
        return False
    height, width = image.shape[:2]
    left, top, right, bottom = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
    span = max(360, 4 * max(right - left, bottom - top))
    x0, x1 = int(max(0, min(width - span, (left + right) / 2 - span / 2))), int(min(width, max(span, (left + right) / 2 + span / 2)))
    y0, y1 = int(max(0, min(height - span * 0.75, (top + bottom) / 2 - span * 0.375))), int(min(height, max(span * 0.75, (top + bottom) / 2 + span * 0.375)))
    crop = image[y0:y1, x0:x1].copy()
    for other in others:
        ol, ot, orr, ob = (float(other[key]) for key in ("x1", "y1", "x2", "y2"))
        if other is row:
            continue
        cv2.rectangle(crop, (int(ol - x0), int(ot - y0)), (int(orr - x0), int(ob - y0)), COLORS[other["status"]], 1)
    cv2.rectangle(crop, (int(left - x0), int(top - y0)), (int(right - x0), int(bottom - y0)), (0, 0, 255), 2)
    scale = min(3.0, 720 / crop.shape[1])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    context = image.copy()
    cv2.rectangle(context, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), max(3, width // 300))
    context = cv2.resize(context, (480, int(480 * height / width)))
    cv2.imwrite(str(destination / (row["card"] + "_zoom.jpg")), crop)
    cv2.imwrite(str(destination / (row["card"] + "_full.jpg")), context)
    return True


def main():
    parser = argparse.ArgumentParser(description="Export a browsable gallery of missed ground-truth boxes")
    parser.add_argument("--run", type=Path, required=True, help="Output folder of analyze_missed_boxes.py")
    parser.add_argument("--per-group", type=int, default=8)
    arguments = parser.parse_args()
    run = arguments.run if arguments.run.is_absolute() else ROOT / arguments.run
    rows = list(csv.DictReader((run / "gt_boxes.csv").open(encoding="utf-8")))
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image"]].append(row)
    missed = [row for row in rows if row["status"] == "MISSED"]
    gallery = run / "gallery"
    gallery.mkdir(exist_ok=False)
    generator = random.Random(363)
    sections, number = [], 0
    for title, test in GROUPS:
        pool = [row for row in missed if test(row)]
        cards = []
        for row in generator.sample(pool, min(arguments.per_group, len(pool))):
            number += 1
            row["card"] = f"{number:03d}"
            if render(row, by_image[row["image"]], gallery):
                cards.append(f'<div class="card"><img src="{row["card"]}_zoom.jpg"><img class="full" src="{row["card"]}_full.jpg">'
                             f'<p><b>{html.escape(row["cls"])}</b> · cao {float(row["height_px"]):.0f}px · {html.escape(row["reason"])}<br>'
                             f'nguồn {html.escape(row["source"])} · độ sáng {float(row["brightness"]):.0f} · {row["overlapping_gt"]} nhãn chồng · IoU tốt nhất với dự đoán {float(row["best_iou_any_pred"]):.2f}<br>'
                             f'<small>{html.escape(Path(row["image"]).name)}</small></p></div>')
        sections.append(f"<h2>{html.escape(title)} <small>({len(pool)} box trong mẫu)</small></h2><div class=\"grid\">{''.join(cards)}</div>")
    (gallery / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Box bị bỏ sót</title><style>body{background:#181818;color:#eee;font:15px sans-serif;margin:20px}'
        '.grid{display:flex;flex-wrap:wrap;gap:14px}.card{width:740px;background:#242424;padding:8px}img{max-width:100%}.full{width:300px}p{margin:6px 0}small{color:#999}</style>'
        '<h1>Box GT bị model bỏ sót</h1><p>Ô vuông đỏ dày là box bị bỏ sót. Các box mảnh: xanh lá là GT model đã gọi đúng, vàng là gọi sai lớp, đỏ mảnh là GT khác cũng bị bỏ sót. '
        'Đây là mẫu ngẫu nhiên, không phải đánh giá nhãn: một box bị bỏ sót có thể do model, do nhãn thừa/trùng, hoặc do vật thể quá nhỏ.</p>' + "".join(sections), encoding="utf-8")
    print(f"{number} cards -> {gallery / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
