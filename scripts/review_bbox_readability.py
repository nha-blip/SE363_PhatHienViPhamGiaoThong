import argparse
import json
import math
import random
import sqlite3
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

if __package__:
    from .filter_detector_datasets_720p import NAMES, ROOT, write_json
else:
    from filter_detector_datasets_720p import NAMES, ROOT, write_json


OUTPUT = ROOT / 'data/bbox_readability_qa_20261027'
AUDIT = ROOT / 'runs/project363/quality_filter_balanced_20261027/audit.sqlite'
LIMITS = (4, 8, 12, 20, 32, float('inf'))
BUCKETS = ('under4', '4to8', '8to12', '12to20', '20to32', '32plus')


def size_bucket(short_side):
    return next(name for name, limit in zip(BUCKETS, LIMITS) if short_side < limit)


def render(sample, destination):
    with Image.open(sample['image']) as opened:
        source = opened.convert('RGB')
    box = sample['box']
    left, top = math.floor(box[0]), math.floor(box[1])
    right, bottom = math.ceil(box[2]), math.ceil(box[3])
    margin = max(12, math.ceil(max(right - left, bottom - top) * 0.6))
    context = source.crop((max(0, left - margin), max(0, top - margin),
                           min(source.width, right + margin), min(source.height, bottom + margin)))
    raw = source.crop((left, top, right, bottom))
    canvas = Image.new('RGB', (1200, 850), '#202020')
    draw = ImageDraw.Draw(canvas)
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    title = f"{sample['id']} | {sample['kind']} | label: {sample['class_name']} (not verified)"
    draw.text((15, 10), title, fill='white', font=font)
    draw.text((15, 40), f"bbox at 720p: {sample['width']:.1f} x {sample['height']:.1f}px; at imgsz640: {sample['width']/2:.1f} x {sample['height']/2:.1f}px", fill='yellow', font=font)
    draw.text((15, 70), 'Zoom is display only: no extra detail. Judge identity/color, not just presence.', fill='white', font=font)
    overview = source.copy()
    marking = ImageDraw.Draw(overview)
    marking.rectangle(box, outline='black', width=5)
    marking.rectangle(box, outline='yellow', width=2)
    overview.thumbnail((640, 360))
    canvas.paste(overview, (15, 120))
    if raw.width <= 450 and raw.height <= 300:
        canvas.paste(raw, (700, 140))
    draw.text((700, 110), 'Native bbox (1:1 if it fits)', fill='white', font=font)
    for crop, position, title in ((context, (15, 535), 'Context, nearest-neighbor zoom'),
                                   (raw, (615, 535), 'Raw bbox, nearest-neighbor zoom')):
        factor = min(560 / crop.width, 260 / crop.height)
        crop = crop.resize((max(1, round(crop.width * factor)), max(1, round(crop.height * factor))), Image.Resampling.NEAREST)
        draw.text((position[0], position[1] - 30), title, fill='white', font=font)
        canvas.paste(crop, position)
    canvas.save(destination)


def build():
    if OUTPUT.exists():
        raise FileExistsError('Review already exists; use --review to preserve decisions')
    generator = random.Random(363)
    seen = Counter()
    selected = {}
    connection = sqlite3.connect(f'file:{AUDIT.as_posix()}?mode=ro', uri=True)
    for record, in connection.execute("SELECT record FROM images WHERE split='train' ORDER BY identity"):
        row = json.loads(record)
        for index, obj in enumerate(row['objects']):
            if 'target_box' not in obj:
                continue
            box = obj['target_box']
            width, height = box[2] - box[0], box[3] - box[1]
            bucket = size_bucket(min(width, height) / 2)
            group = (row['kind'], obj['class'], bucket)
            seen[group] += 1
            reservoir = selected.setdefault(group, [])
            sample = {'kind': row['kind'], 'class_name': NAMES[row['kind']][obj['class']],
                      'bucket_at_imgsz640': bucket, 'width': width, 'height': height,
                      'audit_identity': row['key'], 'object_index': index,
                      'original_image': row['image'], 'source': row['source'],
                      'image': row.get('output_image', row['image']),
                      'box': box if row['decision'] == 'KEEP' else obj['box'],
                      'filter_decision': row['decision'], 'decision': 'PENDING'}
            if len(reservoir) < 3:
                reservoir.append(sample)
            else:
                position = generator.randrange(seen[group])
                if position < 3:
                    reservoir[position] = sample
    connection.close()
    (OUTPUT / 'images').mkdir(parents=True)
    reviews = []
    for group in sorted(selected):
        for sample in selected[group]:
            sample['id'] = f"{sample['kind']}_{len(reviews)+1:03d}"
            sample['preview'] = str(OUTPUT / 'images' / (sample['id'] + '.png'))
            render(sample, sample['preview'])
            reviews.append(sample)
    write_json(OUTPUT / 'human_review.json', {'status': 'PENDING_HUMAN_READABILITY_QA',
               'training_imgsz_reference': 640, 'dataset_modified': False, 'reviews': reviews,
               'population_counts': {'/'.join(map(str, key)): value for key, value in seen.items()}})
    (OUTPUT / 'MO_QA.cmd').write_text('@echo off\r\ncd /d "%~dp0\\..\\.."\r\n"C:\\Users\\NHA\\miniconda3\\envs\\my_env\\python.exe" scripts\\review_bbox_readability.py --review\r\nif errorlevel 1 pause\r\n', encoding='ascii')
    (OUTPUT / 'README.md').write_text('# QA độ rõ bbox\n\nMở MO_QA.cmd. Chọn 1: nhận ra loại xe hoặc màu đèn; 2: không chắc; 3: không nhận ra.\nKết quả tự lưu sau từng lựa chọn; có thể đóng và mở lại. Không cần sửa JSON.\n\nMỗi nhóm kích thước cạnh ngắn tại imgsz=640 lấy ngẫu nhiên tối đa 3 bbox mỗi lớp,\nchỉ từ train, gồm cả ảnh được giữ và ảnh bị loại. Không đại diện chính xác tỷ lệ lỗi toàn bộ dữ liệu.\nẢnh phóng to dùng nearest-neighbor, không tạo thêm chi tiết. Nhãn hiện tại chỉ là tham khảo, không bảo đảm đúng.\n\nBộ mẫu dùng để chốt ngưỡng riêng theo xe/đèn, không tự xóa bbox hay sửa dataset.\nNếu đổi imgsz cần tính lại kích thước; mắt người còn phụ thuộc độ mờ và che khuất.\n', encoding='utf-8')
    print(json.dumps({'samples': len(reviews), 'folder': str(OUTPUT)}), flush=True)


def review():
    path = OUTPUT / 'human_review.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = data['reviews']
    current = next((index for index, row in enumerate(rows) if row['decision'] == 'PENDING'), 0)
    window = tk.Tk()
    window.title('QA độ rõ bbox — không sửa dataset')
    status = ttk.Label(window)
    status.pack()
    picture = ttk.Label(window)
    picture.pack()
    buttons = ttk.Frame(window)
    buttons.pack(pady=8)

    def show():
        row = rows[current]
        completed = sum(item['decision'] != 'PENDING' for item in rows)
        status.config(text=f"{current+1}/{len(rows)} | Đã đánh giá {completed} | {row['decision']} | img640: {row['bucket_at_imgsz640']} px")
        with Image.open(row['preview']) as opened:
            preview = opened.copy()
        preview.thumbnail((1100, max(400, window.winfo_screenheight() - 220)))
        picture.image = ImageTk.PhotoImage(preview)
        picture.config(image=picture.image)

    def choose(value):
        nonlocal current
        rows[current]['decision'] = value
        data['status'] = 'COMPLETE_REQUIRES_THRESHOLD_REVIEW' if all(row['decision'] != 'PENDING' for row in rows) else 'IN_PROGRESS'
        write_json(path, data)
        current = min(current + 1, len(rows) - 1)
        show()

    def previous():
        nonlocal current
        current = max(0, current - 1)
        show()

    for title, value in (('1 — Nhìn rõ loại xe / màu đèn', 'CLEAR'), ('2 — Không chắc', 'UNCERTAIN'), ('3 — Không nhận ra', 'UNREADABLE')):
        ttk.Button(buttons, text=title, command=lambda choice=value: choose(choice)).pack(side='left', padx=5)
    ttk.Button(buttons, text='Quay lại', command=previous).pack(side='left')
    for key, value in (('1', 'CLEAR'), ('2', 'UNCERTAIN'), ('3', 'UNREADABLE')):
        window.bind(key, lambda event, choice=value: choose(choice))
    show()
    window.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', action='store_true')
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if arguments.review:
        OUTPUT = arguments.folder.resolve()
    review() if arguments.review else build()
