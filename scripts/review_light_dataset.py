import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import tkinter as tk
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

if __package__:
    from .light_bbox_editor import BboxEditor, validate_box
else:
    from light_bbox_editor import BboxEditor, validate_box


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = ROOT / 'data/traffic_light_external_audit_20261022/human_review.json'
DECISIONS = {'PENDING', 'KEEP', 'NEEDS_FIX', 'UNCERTAIN', 'REMOVE_BOX'}
SCOPES = {'UNREVIEWED', 'SIGNAL_HEAD', 'LIT_BULB', 'UNCERTAIN', 'WHOLE_OBJECT'}


def correction_objects(row):
    change = row['correction']
    before = copy.deepcopy(row['objects'])
    index = change['action']['object_number'] - 1
    if change['after'] is None:
        before.insert(index, copy.deepcopy(change['before']))
    else:
        before[index] = copy.deepcopy(change['before'])
    return before, copy.deepcopy(row['objects'])


def correction_focus(row):
    with Image.open(row['source_image']) as opened:
        source = opened.convert('RGB')
    target = row['correction']['before']['bbox_xyxy']
    margin = max(35, math.ceil(max(target[2] - target[0], target[3] - target[1]) * 0.8))
    left, top = max(0, math.floor(target[0]) - margin), max(0, math.floor(target[1]) - margin)
    right, bottom = min(source.width, math.ceil(target[2]) + margin), min(source.height, math.ceil(target[3]) + margin)
    crop = source.crop((left, top, right, bottom))
    factor = min(690 / crop.width, 480 / crop.height)
    size = (max(1, round(crop.width * factor)), max(1, round(crop.height * factor)))
    before, after = correction_objects(row)
    canvas = Image.new('RGB', (1440, size[1] + 240), '#202020')
    font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/segoeui.ttf'
    font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for position, objects in enumerate((before, after)):
        offset = position * 720 + 10
        draw.text((offset, 8), 'TRƯỚC SỬA' if position == 0 else 'SAU SỬA — kiểm tra bên này', fill='white', font=font)
        panel = crop.resize(size, Image.Resampling.NEAREST)
        marking = ImageDraw.Draw(panel)
        visible = []
        for number, obj in enumerate(objects, 1):
            box = obj['bbox_xyxy']
            if box[2] < left or box[0] > right or box[3] < top or box[1] > bottom:
                continue
            coordinates = [(value - (left if index % 2 == 0 else top)) * factor for index, value in enumerate(box)]
            marking.rectangle(coordinates, outline='black', width=6)
            marking.rectangle(coordinates, outline='yellow', width=3)
            anchor = (max(0, min(size[0] - 45, coordinates[0])), max(0, coordinates[1] - 30))
            marking.rectangle(marking.textbbox(anchor, f'#{number}', font=font), fill='black')
            marking.text(anchor, f'#{number}', fill='yellow', font=font)
            visible.append(f'#{number}: {obj["source_class"]}')
        canvas.paste(panel, (offset, 50))
        for line, text in enumerate(visible[:4]):
            draw.text((offset, size[1] + 62 + line * 30), text, fill='yellow', font=font)
        if position == 1 and row['correction']['after'] is None:
            draw.text((offset, size[1] + 188), 'Đã bỏ bbox thừa; ảnh gốc vẫn giữ nguyên.', fill='white', font=font)
    return canvas


class ReviewStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.original = self.path.read_bytes()
        self.expected_hash = hashlib.sha256(self.original).hexdigest()
        self.data = json.loads(self.original)
        self.rows = self.data['reviews']
        if not self.rows or len({row['id'] for row in self.rows}) != len(self.rows):
            raise ValueError('Danh sách review trống hoặc trùng ID')
        self.backup = None

    def first_pending(self):
        return next((index for index, row in enumerate(self.rows) if row['decision'] == 'PENDING'), 0)

    def save(self, index, decision, scope, confirmed, notes):
        if decision not in DECISIONS or scope not in SCOPES:
            raise ValueError('Giá trị đánh giá không hợp lệ')
        row = self.rows[index]
        if decision == 'REMOVE_BOX' and not row.get('remove_focus_object'):
            raise ValueError('Dùng nút Bỏ bbox thừa để xác nhận đối tượng cần loại')
        if decision == 'KEEP' and row.get('remove_focus_object'):
            raise ValueError('Khôi phục bbox trước khi chọn Đúng; đề xuất loại đã được lưu riêng')
        if decision == 'KEEP' and row.get('geometry_required') and not row.get('geometry_edits'):
            raise ValueError('Ảnh này cần vẽ lại bbox trước khi chọn Đúng')
        if confirmed and any(obj.get('proposed_class') is None for obj in row.get('objects', [])):
            raise ValueError('Có trạng thái chưa ánh xạ; hãy bỏ xác nhận ánh xạ và ghi chú')
        updates = {'decision': decision, 'bbox_scope': scope,
                   'class_mapping_confirmed': bool(confirmed), 'notes': notes}
        return self.write_updates(index, updates)

    def save_class(self, index, name):
        row = self.rows[index]
        if not self.data.get('allow_class_edit') or name not in row.get('allowed_classes', []) or len(row['objects']) != 1:
            raise ValueError('Không được đổi lớp này')
        objects = copy.deepcopy(row['objects'])
        before = objects[0]['proposed_class']
        if before == name:
            return False
        objects[0].update(source_class=name, proposed_class=name)
        correction = copy.deepcopy(row['correction'])
        correction['after'] = copy.deepcopy(objects[0])
        edit = {'before': before, 'after': name, 'edited_at_utc': datetime.now(timezone.utc).isoformat()}
        return self.write_updates(index, {'objects': objects, 'correction': correction,
                                  'class_edits': row.get('class_edits', []) + [edit],
                                  'decision': 'PENDING', 'class_mapping_confirmed': False})

    def set_removed(self, index, removed):
        row = self.rows[index]
        if not self.data.get('allow_focus_removal'):
            raise ValueError('Không được loại đối tượng trong phiên này')
        correction = copy.deepcopy(row['correction'])
        if removed:
            if len(row['objects']) != 1:
                raise ValueError('Chỉ được đề xuất bỏ đối tượng focus duy nhất')
            snapshot = copy.deepcopy(row['objects'][0])
            correction['after'] = None
            updates = {'removed_object_snapshot': snapshot, 'objects': [], 'decision': 'REMOVE_BOX'}
        else:
            if not row.get('remove_focus_object'):
                return False
            snapshot = copy.deepcopy(row['removed_object_snapshot'])
            correction['after'] = snapshot
            updates = {'objects': [snapshot], 'decision': 'PENDING'}
        updates.update(remove_focus_object=removed, correction=correction,
                       bbox_scope='UNREVIEWED', class_mapping_confirmed=False)
        updates['removal_edits'] = row.get('removal_edits', []) + [{'removed': removed, 'edited_at_utc': datetime.now(timezone.utc).isoformat()}]
        return self.write_updates(index, updates)

    def save_geometry(self, index, object_index, box):
        row = self.rows[index]
        if not self.data.get('allow_bbox_edit') or object_index + 1 not in row.get('edit_object_numbers', []):
            raise ValueError('Chỉ được sửa bbox được chỉ định trong bản nháp riêng')
        if not 0 <= object_index < len(row['objects']):
            raise ValueError('Không có đối tượng này')
        with Image.open(row['source_image']) as source:
            box = validate_box(box, source.width, source.height)
        objects = copy.deepcopy(row['objects'])
        before = list(objects[object_index]['bbox_xyxy'])
        if before == box:
            return False
        objects[object_index].update(bbox_xyxy=box, valid=True)
        edit = {'object_number': object_index + 1, 'before': before, 'after': box,
                'coordinate_system': 'native_source_pixels', 'edited_at_utc': datetime.now(timezone.utc).isoformat()}
        updates = {'objects': objects, 'geometry_edits': row.get('geometry_edits', []) + [edit],
                   'decision': 'PENDING', 'bbox_scope': 'UNREVIEWED', 'class_mapping_confirmed': False}
        if 'correction' in row:
            correction = copy.deepcopy(row['correction'])
            correction['after'] = copy.deepcopy(objects[object_index])
            updates['correction'] = correction
        return self.write_updates(index, updates)

    def write_updates(self, index, updates):
        row = self.rows[index]
        if all(row.get(key) == value for key, value in updates.items()):
            return False
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.expected_hash:
            raise RuntimeError('JSON đã bị thay đổi bởi chương trình khác. Đóng và mở lại công cụ để tránh ghi đè.')
        payload = {**self.data, 'reviews': [dict(item) for item in self.rows], 'is_ground_truth': False}
        payload['reviews'][index].update(updates)
        payload['status'] = ('PENDING_HUMAN_QA' if any(item['decision'] == 'PENDING' for item in payload['reviews'])
                             else 'HUMAN_QA_COMPLETED_REQUIRES_VALIDATION')
        payload['last_reviewed_at_utc'] = datetime.now(timezone.utc).isoformat()
        content = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
        if self.backup is None:
            backup_dir = self.path.parent / 'review_backups'
            backup_dir.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            self.backup = backup_dir / f'{self.path.stem}_{stamp}.json'
            with self.backup.open('xb') as handle:
                handle.write(self.original)
        descriptor, temporary = tempfile.mkstemp(prefix=self.path.name + '.', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.data = payload
        self.rows = payload['reviews']
        self.expected_hash = hashlib.sha256(content).hexdigest()
        return True


class ReviewWindow:
    def __init__(self, root, store):
        self.root = root
        self.store = store
        self.index = store.first_pending()
        self.scale = 1.0
        self.image = None
        self.photo = None
        self.loading = False
        self.scheduled = None
        self.decision = tk.StringVar()
        self.scope = tk.StringVar()
        self.confirmed = tk.BooleanVar()
        self.mode = tk.StringVar(value='Vùng sửa phóng to' if 'correction' in store.rows[self.index] else 'Ảnh QA')
        self.progress = tk.StringVar()
        self.status = tk.StringVar(value='Chọn đánh giá; kết quả được tự lưu. Không sửa nhãn nguồn.')
        root.title('QA nhãn xe — bản sửa riêng' if store.data.get('annotation_kind') == 'vehicle' else 'QA dataset đèn — lưu trên máy')
        root.geometry('1450x900')
        root.minsize(1000, 650)
        root.protocol('WM_DELETE_WINDOW', self.close)
        toolbar = ttk.Frame(root, padding=6)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='◀ Trước', command=lambda: self.navigate(-1)).pack(side='left')
        ttk.Button(toolbar, text='Lưu và tiếp ▶', command=lambda: self.navigate(1)).pack(side='left', padx=5)
        ttk.Button(toolbar, text='Ảnh chưa đánh giá', command=self.next_pending).pack(side='left')
        ttk.Label(toolbar, textvariable=self.progress).pack(side='left', padx=15)
        body = ttk.Panedwindow(root, orient='horizontal')
        body.pack(fill='both', expand=True)
        viewer = ttk.Frame(body)
        controls = ttk.Frame(body, padding=10, width=355)
        body.add(viewer, weight=4)
        body.add(controls, weight=1)
        tools = ttk.Frame(viewer, padding=4)
        tools.pack(fill='x')
        selector = ttk.Combobox(tools, textvariable=self.mode, values=['Vùng sửa phóng to', 'Ảnh QA', 'Ảnh gốc + bbox', 'Ảnh gốc'], state='readonly', width=22)
        selector.pack(side='left')
        selector.bind('<<ComboboxSelected>>', lambda event: self.load_picture())
        self.edit_button = ttk.Button(tools, text='Vẽ lại bbox', command=self.edit_bbox)
        self.edit_button.pack(side='left', padx=3)
        for text, command in [('Vừa cửa sổ', self.fit), ('−', lambda: self.zoom(0.8)), ('+', lambda: self.zoom(1.25)), ('100%', lambda: self.zoom_to(1.0))]:
            ttk.Button(tools, text=text, command=command).pack(side='left', padx=3)
        grid = ttk.Frame(viewer)
        grid.pack(fill='both', expand=True)
        grid.rowconfigure(0, weight=1)
        grid.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(grid, background='#202020', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        vertical = ttk.Scrollbar(grid, orient='vertical', command=self.canvas.yview)
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal = ttk.Scrollbar(grid, orient='horizontal', command=self.canvas.xview)
        horizontal.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<ButtonPress-1>', lambda event: self.canvas.scan_mark(event.x, event.y))
        self.canvas.bind('<B1-Motion>', lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1))
        self.title = ttk.Label(controls, wraplength=330, font=('Segoe UI', 11, 'bold'))
        self.title.pack(anchor='w', pady=(0, 10))
        ttk.Label(controls, text='Đánh giá ảnh', font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        for label, value in [('Đúng', 'KEEP'), ('Cần sửa', 'NEEDS_FIX'), ('Không chắc', 'UNCERTAIN'), ('Chưa đánh giá', 'PENDING')]:
            button = ttk.Radiobutton(controls, text=label, variable=self.decision, value=value, command=self.save)
            button.pack(anchor='w', pady=3)
            if value == 'KEEP':
                self.keep_button = button
        ttk.Separator(controls).pack(fill='x', pady=9)
        ttk.Label(controls, text='BBox bao vùng nào?').pack(anchor='w')
        scopes = ([('Toàn bộ phương tiện', 'WHOLE_OBJECT'), ('Không chắc', 'UNCERTAIN'), ('Chưa kiểm tra', 'UNREVIEWED')]
                  if store.data.get('annotation_kind') == 'vehicle' else
                  [('Cả cụm đèn', 'SIGNAL_HEAD'), ('Chỉ bóng sáng', 'LIT_BULB'), ('Không chắc', 'UNCERTAIN'), ('Chưa kiểm tra', 'UNREVIEWED')])
        for label, value in scopes:
            ttk.Radiobutton(controls, text=label, variable=self.scope, value=value, command=self.save).pack(anchor='w', pady=2)
        self.mapping = ttk.Checkbutton(controls, text='Xác nhận nhãn hiện tại', variable=self.confirmed, command=self.save)
        self.mapping.pack(anchor='w', pady=(12, 5))
        self.details = tk.Text(controls, height=6, width=39, wrap='word', background='#eeeeee')
        self.details.pack(fill='x')
        self.class_choice = tk.StringVar()
        self.class_selector = ttk.Combobox(controls, textvariable=self.class_choice, state='readonly')
        if store.data.get('allow_class_edit'):
            self.class_selector.pack(fill='x', pady=(5, 2))
            ttk.Button(controls, text='Áp dụng lớp đã chọn', command=self.change_class).pack(fill='x')
        self.remove_button = ttk.Button(controls, text='Bỏ bbox thừa', command=self.toggle_removal)
        if store.data.get('allow_focus_removal'):
            self.remove_button.pack(fill='x', pady=4)
        ttk.Label(controls, text='Ghi chú (tự lưu sau khi ngừng gõ)').pack(anchor='w', pady=(10, 4))
        self.notes = tk.Text(controls, height=4, width=39, wrap='word', undo=True)
        self.notes.pack(fill='both', expand=True)
        self.notes.bind('<<Modified>>', self.notes_changed)
        ttk.Button(controls, text='Lưu ngay', command=self.save).pack(fill='x', pady=8)
        ttk.Label(controls, text='Lăn chuột: cuộn ảnh\nCtrl + lăn chuột: phóng to/thu nhỏ\nGiữ chuột trái để kéo ảnh', wraplength=330).pack(anchor='w')
        ttk.Label(root, textvariable=self.status, padding=6, wraplength=1400).pack(fill='x')
        root.bind('<Control-s>', lambda event: self.save())
        self.load_row()

    def notes_changed(self, event=None):
        if not self.notes.edit_modified():
            return
        self.notes.edit_modified(False)
        if self.loading:
            return
        if self.scheduled is not None:
            self.root.after_cancel(self.scheduled)
        self.scheduled = self.root.after(700, self.save)

    def save(self):
        if self.scheduled is not None:
            self.root.after_cancel(self.scheduled)
            self.scheduled = None
        try:
            changed = self.store.save(self.index, self.decision.get(), self.scope.get(), self.confirmed.get(), self.notes.get('1.0', 'end-1c'))
            if changed:
                self.status.set('Đã lưu trên máy. Nhãn nguồn không thay đổi; đánh giá chưa tự trở thành GT.')
            self.update_progress()
            return True
        except Exception as error:
            self.status.set('CHƯA LƯU: ' + str(error))
            messagebox.showerror('Không lưu được', str(error), parent=self.root)
            return False

    def update_progress(self):
        counts = Counter(row['decision'] for row in self.store.rows)
        self.progress.set(f'{self.index + 1}/{len(self.store.rows)} | Đã đánh giá: {len(self.store.rows) - counts["PENDING"]} | Còn: {counts["PENDING"]}')

    def load_row(self):
        self.loading = True
        row = self.store.rows[self.index]
        self.edit_button.configure(state='normal' if self.store.data.get('allow_bbox_edit') and row.get('edit_object_numbers') and row['objects'] else 'disabled')
        self.keep_button.configure(state='disabled' if (row.get('geometry_required') and not row.get('geometry_edits')) or row.get('remove_focus_object') else 'normal')
        self.title.configure(text=row['id'])
        self.decision.set(row['decision'])
        self.scope.set(row['bbox_scope'])
        self.confirmed.set(row['class_mapping_confirmed'])
        self.notes.delete('1.0', 'end')
        self.notes.insert('1.0', row.get('notes', ''))
        self.notes.edit_modified(False)
        objects = row.get('objects', [])
        self.class_selector.configure(values=row.get('allowed_classes', []), state='readonly' if objects else 'disabled')
        self.class_choice.set(objects[0]['proposed_class'] if len(objects) == 1 else '')
        self.remove_button.configure(text='Khôi phục bbox vừa bỏ' if row.get('remove_focus_object') else 'Bỏ bbox thừa')
        unmapped = any(obj.get('proposed_class') is None for obj in objects)
        self.mapping.configure(state='disabled' if unmapped or not objects else 'normal')
        lines = []
        if row.get('task_hint'):
            lines.append(row['task_hint'] + '\n')
        if row.get('remove_focus_object'):
            lines.append('ĐÃ LƯU ĐỀ XUẤT BỎ BBOX FOCUS. Không xóa nhãn nguồn.\n')
        if unmapped:
            lines.append('Có trạng thái chưa ánh xạ; không tự gộp wait_on / red_yellow.\n')
        for identity, obj in enumerate(objects, 1):
            target = obj.get('proposed_class') or 'CHƯA ÁNH XẠ'
            warning = ' [BBOX LỖI]' if not obj.get('valid', True) else ''
            lines.append(f'#{identity}: {obj["source_class"]} → {target}{warning}')
        self.details.configure(state='normal')
        self.details.delete('1.0', 'end')
        self.details.insert('1.0', '\n'.join(lines))
        self.details.configure(state='disabled')
        self.loading = False
        self.update_progress()
        self.load_picture()

    def change_class(self):
        if not self.save():
            return
        try:
            self.store.save_class(self.index, self.class_choice.get())
            self.load_row()
            self.status.set('Đã lưu lớp đề xuất. Kiểm tra lại rồi chọn Đúng và xác nhận nhãn.')
        except Exception as error:
            messagebox.showerror('Chưa đổi lớp', str(error), parent=self.root)

    def toggle_removal(self):
        if not self.save():
            return
        row = self.store.rows[self.index]
        removed = not row.get('remove_focus_object', False)
        if removed and not messagebox.askyesno('Bỏ đối tượng focus?', 'Chỉ đề xuất bỏ bbox đang được hỏi, không xóa các đối tượng khác hoặc nhãn nguồn. Tiếp tục?', parent=self.root):
            return
        try:
            self.store.set_removed(self.index, removed)
            self.load_row()
        except Exception as error:
            messagebox.showerror('Chưa đổi đề xuất loại bbox', str(error), parent=self.root)

    def edit_bbox(self):
        if not self.save():
            return
        row = self.store.rows[self.index]
        numbers = row.get('edit_object_numbers', [])
        if not numbers:
            return
        object_index = numbers[0] - 1

        def apply(box):
            self.store.save_geometry(self.index, object_index, box)
            self.load_row()
            self.status.set('Đã lưu bbox mới vào bản nháp. Kiểm tra lại rồi chọn Đúng và xác nhận phạm vi/màu.')

        self.editor = BboxEditor(self.root, row, object_index, apply)

    def load_picture(self):
        row = self.store.rows[self.index]
        if (self.mode.get() == 'Vùng sửa phóng to' or (self.mode.get() == 'Ảnh QA' and row.get('geometry_edits'))) and 'correction' in row:
            try:
                self.image = correction_focus(row)
                self.root.after_idle(self.fit)
            except Exception as error:
                self.image = None
                self.canvas.delete('all')
                self.canvas.create_text(20, 20, anchor='nw', text=f'Không đọc được vùng sửa: {error}', fill='white')
            return
        if self.mode.get() == 'Vùng sửa phóng to':
            self.mode.set('Ảnh QA')
        path = Path(row['qa_image'] if self.mode.get() == 'Ảnh QA' else row['source_image'])
        try:
            with Image.open(path) as original:
                self.image = original.convert('RGB')
            if self.mode.get() == 'Ảnh gốc + bbox':
                draw = ImageDraw.Draw(self.image)
                font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/segoeui.ttf'
                font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
                for identity, obj in enumerate(row['objects'], 1):
                    left, top, right, bottom = obj['bbox_xyxy']
                    if not all(isinstance(value, (int, float)) for value in (left, top, right, bottom)):
                        continue
                    box = (left, top, max(left, right), max(top, bottom))
                    draw.rectangle(box, outline='black', width=6)
                    draw.rectangle(box, outline='yellow', width=3)
                    label = f'#{identity} {obj["source_class"]}'
                    origin = (max(0, left), max(0, top - 25))
                    draw.rectangle(draw.textbbox(origin, label, font=font), fill='black')
                    draw.text(origin, label, font=font, fill='yellow')
            self.root.after_idle(self.fit)
        except Exception as error:
            self.image = None
            self.canvas.delete('all')
            self.canvas.create_text(20, 20, anchor='nw', text=f'Không đọc được ảnh: {error}', fill='white', width=700)

    def fit(self):
        if self.image is not None:
            self.root.update_idletasks()
            self.zoom_to(min(max(100, self.canvas.winfo_width()) / self.image.width,
                             max(100, self.canvas.winfo_height()) / self.image.height))
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)

    def zoom(self, factor):
        self.zoom_to(self.scale * factor)

    def zoom_to(self, scale):
        if self.image is None:
            return
        self.scale = max(0.1, min(scale, 6.0, 9000 / max(self.image.size)))
        size = tuple(max(1, round(dimension * self.scale)) for dimension in self.image.size)
        self.photo = ImageTk.PhotoImage(self.image.resize(size, Image.Resampling.NEAREST))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, *size))

    def wheel(self, event):
        if event.state & 4:
            self.zoom(1.2 if event.delta > 0 else 1 / 1.2)
        else:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')
        return 'break'

    def navigate(self, delta):
        if self.save():
            self.index = max(0, min(len(self.store.rows) - 1, self.index + delta))
            self.load_row()

    def next_pending(self):
        if not self.save():
            return
        for offset in range(1, len(self.store.rows) + 1):
            index = (self.index + offset) % len(self.store.rows)
            if self.store.rows[index]['decision'] == 'PENDING':
                self.index = index
                self.load_row()
                return
        messagebox.showinfo('Đã đánh giá hết', 'Không còn ảnh PENDING. Kết quả vẫn cần được kiểm tra trước khi ghép dataset.', parent=self.root)

    def close(self):
        if self.save():
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='Local traffic light dataset QA; no web server')
    parser.add_argument('--review', type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    root = tk.Tk()
    try:
        store = ReviewStore(args.review)
        ReviewWindow(root, store)
        root.mainloop()
    except Exception as error:
        messagebox.showerror('Không mở được công cụ QA', str(error), parent=root)
        root.destroy()
        raise


if __name__ == '__main__':
    main()
