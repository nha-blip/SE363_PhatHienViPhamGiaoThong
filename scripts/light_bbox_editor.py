import math
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


def validate_box(box, width, height):
    if len(box) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
        raise ValueError('Tọa độ bbox không hợp lệ')
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError('BBox phải nằm trong ảnh và có chiều rộng, chiều cao lớn hơn 0')
    return list(map(float, box))


def source_point(display_x, display_y, scale, crop):
    if scale <= 0:
        raise ValueError('Invalid display scale')
    return (min(crop[2], max(crop[0], display_x / scale + crop[0])),
            min(crop[3], max(crop[1], display_y / scale + crop[1])))


def box_from_drag(start, end):
    return [min(start[0], end[0]), min(start[1], end[1]),
            max(start[0], end[0]), max(start[1], end[1])]


class BboxEditor:
    def __init__(self, parent, row, object_index, on_apply):
        self.window = tk.Toplevel(parent)
        self.window.title(f'Vẽ lại bbox #{object_index + 1} — {row["id"]}')
        self.window.geometry('1100x760')
        self.window.transient(parent)
        self.window.grab_set()
        self.window.protocol('WM_DELETE_WINDOW', self.window.destroy)
        with Image.open(row['source_image']) as opened:
            self.source = opened.convert('RGB')
        self.objects = row['objects']
        self.index = object_index
        self.original = list(self.objects[object_index]['bbox_xyxy'])
        self.box = list(self.original)
        self.on_apply = on_apply
        self.scale = 1
        self.start = None
        self.has_drawing = False
        self.photo = None
        instruction = ('Bao trọn phương tiện đang được hỏi, không gộp xe bên cạnh.' if row.get('annotation_kind') == 'vehicle'
                       else 'Bao cả cụm đèn, không chỉ bóng sáng.')
        self.instructions = tk.StringVar(value='Kéo chuột trái từ một góc sang góc đối diện của khung MỚI. ' + instruction)
        ttk.Label(self.window, textvariable=self.instructions, padding=8, wraplength=1050).pack(fill='x')
        toolbar = ttk.Frame(self.window, padding=5)
        toolbar.pack(fill='x')
        for text, command in [('Vùng đèn', self.focus), ('Toàn ảnh', self.full),
                              ('−', lambda: self.zoom(0.8)), ('+', lambda: self.zoom(1.25)),
                              ('Hoàn tác', self.reset)]:
            ttk.Button(toolbar, text=text, command=command).pack(side='left', padx=4)
        frame = ttk.Frame(self.window)
        frame.pack(fill='both', expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(frame, background='#202020', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        vertical = ttk.Scrollbar(frame, orient='vertical', command=self.canvas.yview)
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal = ttk.Scrollbar(frame, orient='horizontal', command=self.canvas.xview)
        horizontal.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.coordinates = tk.StringVar()
        ttk.Label(self.window, textvariable=self.coordinates, padding=5).pack(fill='x')
        buttons = ttk.Frame(self.window, padding=8)
        buttons.pack(fill='x')
        ttk.Button(buttons, text='Hủy — không lưu', command=self.window.destroy).pack(side='right', padx=5)
        ttk.Button(buttons, text='Áp dụng bbox mới', command=self.apply).pack(side='right')
        self.window.after_idle(self.focus)

    def focus(self):
        left, top, right, bottom = self.original
        margin = max(50, math.ceil(max(right - left, bottom - top) * 1.2))
        self.crop = (max(0, math.floor(left) - margin), max(0, math.floor(top) - margin),
                     min(self.source.width, math.ceil(right) + margin), min(self.source.height, math.ceil(bottom) + margin))
        self.fit()

    def full(self):
        self.crop = (0, 0, self.source.width, self.source.height)
        self.fit()

    def fit(self):
        self.window.update_idletasks()
        self.scale = min(max(100, self.canvas.winfo_width()) / (self.crop[2] - self.crop[0]),
                         max(100, self.canvas.winfo_height()) / (self.crop[3] - self.crop[1]))
        self.render()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def zoom(self, factor):
        self.scale = max(0.1, min(self.scale * factor, 30, 9000 / max(self.crop[2] - self.crop[0], self.crop[3] - self.crop[1])))
        self.render()

    def render(self):
        crop = self.source.crop(self.crop)
        size = tuple(max(1, round(value * self.scale)) for value in crop.size)
        self.photo = ImageTk.PhotoImage(crop.resize(size, Image.Resampling.NEAREST))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, *size))
        for index, obj in enumerate(self.objects):
            box = self.box if index == self.index else obj['bbox_xyxy']
            color = '#ffff00' if index == self.index else '#00ffff'
            coordinates = [(value - self.crop[position % 2]) * self.scale for position, value in enumerate(box)]
            self.canvas.create_rectangle(*coordinates, outline='black', width=6, tags='bbox')
            self.canvas.create_rectangle(*coordinates, outline=color, width=2, tags='bbox')
            self.canvas.create_text(coordinates[0], max(12, coordinates[1] - 12), text=f'#{index + 1} {obj["source_class"]}', anchor='sw', fill=color, font=('Segoe UI', 13, 'bold'), tags='bbox')
        self.coordinates.set('Tọa độ ảnh gốc: ' + ', '.join(f'{value:.2f}' for value in self.box) + ' | Vàng: khung đang sửa; xanh: đối tượng khác')

    def point(self, event):
        return source_point(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y), self.scale, self.crop)

    def press(self, event):
        self.start = self.point(event)

    def drag(self, event):
        if self.start is not None:
            self.box = box_from_drag(self.start, self.point(event))
            self.render()

    def release(self, event):
        if self.start is None:
            return
        self.box = box_from_drag(self.start, self.point(event))
        self.start = None
        self.has_drawing = self.box != self.original
        self.render()

    def reset(self):
        self.box = list(self.original)
        self.has_drawing = False
        self.render()

    def wheel(self, event):
        if event.state & 4:
            self.zoom(1.2 if event.delta > 0 else 1 / 1.2)
        else:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')
        return 'break'

    def apply(self):
        try:
            if not self.has_drawing:
                raise ValueError('Bạn chưa vẽ khung mới. Kéo chuột để vẽ hoặc bấm Hủy.')
            box = validate_box(self.box, self.source.width, self.source.height)
            self.on_apply(box)
            self.window.destroy()
        except Exception as error:
            messagebox.showerror('Chưa áp dụng bbox', str(error), parent=self.window)
