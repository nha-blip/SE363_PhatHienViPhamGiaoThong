import json
import math
import os
import time
import tkinter as tk
from datetime import datetime, timezone
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from scripts.survey_ihanoi_cameras import OUTPUT, ROOT


CHOICE_LABELS = {
    'CHUA_CHON': 'Chưa chọn',
    'THEO_DOI_DEN_VACH': 'Giữ: đèn + vạch',
    'CHI_TRACKING': 'Giữ: chỉ tracking',
    'BO_QUA': 'Loại',
    'CAN_XEM_LAI': 'Cần xem lại',
}
CHOICES = tuple(CHOICE_LABELS)
STATUS_LABELS = {
    'RECORDED_AWAITING_USER_REVIEW': 'Chờ bạn đánh giá',
    'EXISTING_REVIEW_VIDEO': 'Video có sẵn',
    'EXCLUDED_BLACK_SCREEN': 'Đã tách: nền đen',
    'PREVIOUSLY_REJECTED_BY_USER': 'Bạn đã loại trước đó',
    'WAITING_SURVEY': 'Chưa quay',
    'CLAIMED_ONCE': 'Đang xử lý',
    'FAILED_CAMERA_OR_RECORDING_CHECK': 'Cần kiểm tra lỗi',
}


def save_choice(path, identity, choice, notes):
    if choice not in CHOICES:
        raise ValueError('Lựa chọn không hợp lệ')
    latest = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    latest[identity] = {'choice': choice, 'notes': notes,
                        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                        'is_ground_truth': False}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return latest


def review_hint(row):
    if row.get('excluded_from_review_default'):
        return 'Bộ lọc phát hiện nền đen; có thể bật xem lại.'
    if row.get('black_filter') == 'KEEP_NONBLACK_CONTENT':
        return 'Có nội dung khác nền đen; bạn vẫn cần kiểm tra đèn, vạch và độ rõ.'
    if row.get('status') == 'PREVIOUSLY_REJECTED_BY_USER':
        return 'Đã bị bạn loại ở lượt trước; không tự khôi phục vào dataset.'
    return 'Chưa có kết luận chất lượng tự động.'


class LocalVideo:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError('Không mở được video')
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and 0 < fps <= 240 else 25
        self.duration = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)/self.fps
        self.last_time = None

    def read(self):
        valid, frame = self.capture.read()
        if not valid:
            return None
        seconds = self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000
        fallback = max(0, self.capture.get(cv2.CAP_PROP_POS_FRAMES)-1)/self.fps
        if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and fallback > 0):
            seconds = fallback
        if self.last_time is not None and seconds < self.last_time:
            seconds = max(fallback, self.last_time+1/self.fps)
        self.last_time = seconds
        return frame, seconds

    def seek(self, seconds):
        self.last_time = None
        return self.capture.set(cv2.CAP_PROP_POS_MSEC, max(0, min(seconds, self.duration))*1000)

    def close(self):
        self.capture.release()


class Review:
    def __init__(self, window):
        self.window = window
        self.rows = {}
        self.current_id = None
        self.saved_value = None
        self.reader = None
        self.pending = None
        self.last_frame = None
        self.position = 0
        self.playing = False
        self.after_id = None
        self.refreshing = False
        self.path = OUTPUT / 'human_choices.json'
        self.choices = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        window.title('Chọn camera iHanoi — video và ghi chú')
        width = min(1550, max(800, window.winfo_screenwidth()-100))
        height = min(850, max(550, window.winfo_screenheight()-100))
        window.geometry(f'{width}x{height}+25+25')
        window.minsize(min(1050, width), min(650, height))
        window.protocol('WM_DELETE_WINDOW', self.close)
        window.bind('<Control-s>', lambda event: self.save())
        ttk.Label(window, text='Chọn camera để xem video bên phải. Ghi chú và lựa chọn tự lưu khi đổi camera.').pack(pady=8)
        panes = ttk.Panedwindow(window, orient='horizontal')
        panes.pack(fill='both', expand=True, padx=10)
        left, right = ttk.Frame(panes), ttk.Frame(panes)
        panes.add(left, weight=3)
        panes.add(right, weight=2)
        self.initial_layout = False

        def arrange(event):
            if not self.initial_layout and event.width > 100:
                self.initial_layout = True
                panes.sashpos(0, int(event.width*0.56))

        panes.bind('<Configure>', arrange)
        table_frame = ttk.Frame(left)
        table_frame.pack(fill='both', expand=True)
        self.table = ttk.Treeview(table_frame, columns=('name', 'status', 'choice', 'notes'),
                                  show='tree headings', selectmode='browse')
        self.table.heading('#0', text='Mã camera')
        self.table.column('#0', width=110, stretch=False)
        for column, label, width in [('name', 'Tên / vị trí', 220), ('status', 'Thu thập', 125),
                                     ('choice', 'Giữ / loại', 130), ('notes', 'Ghi chú của bạn', 200)]:
            self.table.heading(column, text=label)
            self.table.column(column, width=width, minwidth=90)
        vertical = ttk.Scrollbar(table_frame, command=self.table.yview)
        horizontal = ttk.Scrollbar(table_frame, orient='horizontal', command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.table.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        self.table.bind('<<TreeviewSelect>>', self.selected)
        controls = ttk.Frame(left)
        controls.pack(fill='x', pady=8)
        ttk.Button(controls, text='Cập nhật', command=self.refresh).pack(side='left', padx=3)
        ttk.Button(controls, text='Mở ngoài', command=lambda: self.open_file('video')).pack(side='left', padx=3)
        ttk.Button(controls, text='Ảnh mẫu', command=lambda: self.open_file('preview')).pack(side='left', padx=3)
        self.choice = tk.StringVar(value=CHOICE_LABELS['CHUA_CHON'])
        ttk.Combobox(controls, values=list(CHOICE_LABELS.values()), textvariable=self.choice,
                     state='readonly', width=21).pack(side='left', padx=4)
        ttk.Button(controls, text='Lưu', command=self.save).pack(side='left', padx=3)
        quick = ttk.Frame(left)
        quick.pack(fill='x', pady=3)
        for code in ('THEO_DOI_DEN_VACH', 'CHI_TRACKING', 'BO_QUA', 'CAN_XEM_LAI'):
            ttk.Button(quick, text=CHOICE_LABELS[code], command=lambda value=code: self.choose(value)).pack(side='left', padx=3)
        self.show_black = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text='Hiện video nền đen đã tách riêng', variable=self.show_black,
                         command=self.refresh).pack(anchor='w', pady=5)
        ttk.Label(left, text='Ghi chú của bạn:').pack(anchor='w')
        self.note = tk.StringVar()
        ttk.Entry(left, textvariable=self.note).pack(fill='x', pady=(3, 8))
        self.hint = tk.StringVar()
        ttk.Label(left, textvariable=self.hint, wraplength=700).pack(anchor='w', pady=5)
        self.video_title = tk.StringVar(value='Chọn một camera để xem video')
        ttk.Label(right, textvariable=self.video_title, wraplength=520).pack(fill='x', padx=8, pady=5)
        self.canvas = tk.Canvas(right, background='#161616', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=8)
        self.image_item = self.canvas.create_image(0, 0, anchor='center')
        self.canvas.bind('<Configure>', lambda event: self.paint(self.last_frame))
        playback = ttk.Frame(right)
        playback.pack(fill='x', pady=6, padx=8)
        self.play_button = ttk.Button(playback, text='Phát', command=self.toggle)
        self.play_button.pack(side='left')
        ttk.Button(playback, text='Về đầu', command=lambda: self.seek(0)).pack(side='left', padx=5)
        self.clock = tk.StringVar(value='0.0 / 0.0 giây')
        ttk.Label(playback, textvariable=self.clock).pack(side='right')
        self.seek_position = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(right, from_=0, to=1, variable=self.seek_position)
        self.slider.pack(fill='x', padx=8)
        self.slider.bind('<ButtonPress-1>', self.begin_seek)
        self.slider.bind('<ButtonRelease-1>', self.finish_seek)
        self.focus_camera = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text='Phóng vùng camera (chỉ khi xem, không sửa file)',
                         variable=self.focus_camera, command=lambda: self.paint(self.last_frame)).pack(anchor='w', padx=8, pady=8)
        self.message = tk.StringVar(value='Lựa chọn không phải nhãn GT và không tự thay đổi dataset.')
        ttk.Label(window, textvariable=self.message).pack(fill='x', padx=12, pady=7)
        self.refresh()

    def choice_code(self):
        return next(code for code, label in CHOICE_LABELS.items() if label == self.choice.get())

    def save(self, force=True):
        if self.current_id is None:
            return True
        value = (self.choice_code(), self.note.get())
        if not force and value == self.saved_value:
            return True
        try:
            self.choices = save_choice(self.path, self.current_id, *value)
            self.saved_value = value
            if self.table.exists(self.current_id):
                self.table.set(self.current_id, 'choice', CHOICE_LABELS[value[0]])
                self.table.set(self.current_id, 'notes', value[1])
            self.message.set('Đã lưu lựa chọn và ghi chú: ' + self.current_id)
            return True
        except (OSError, ValueError) as error:
            messagebox.showerror('Chưa lưu được đánh giá', str(error))
            return False

    def choose(self, code):
        self.choice.set(CHOICE_LABELS[code])
        self.save()

    def refresh(self):
        if not self.save(force=False):
            return
        try:
            data = json.loads((OUTPUT / 'survey_index.json').read_text(encoding='utf-8'))
            self.rows = {row['camera_id']: row for row in data['cameras']}
            self.refreshing = True
            for identity in self.table.get_children():
                self.table.delete(identity)
            for identity, row in self.rows.items():
                if row.get('excluded_from_review_default') and not self.show_black.get():
                    continue
                review = self.choices.get(identity, {})
                self.table.insert('', 'end', iid=identity, text=identity,
                    values=(row.get('name', ''), STATUS_LABELS.get(row['status'], row['status']),
                            CHOICE_LABELS.get(review.get('choice'), CHOICE_LABELS['CHUA_CHON']), review.get('notes', '')))
            if self.current_id and self.table.exists(self.current_id):
                self.table.selection_set(self.current_id)
            else:
                self.current_id = None
                self.saved_value = None
                self.close_video()
                children = self.table.get_children()
                available = [identity for identity in children if self.rows[identity].get('video')]
                if available or children:
                    self.table.selection_set((available or children)[0])
        except (OSError, ValueError, KeyError) as error:
            messagebox.showerror('Không đọc được danh mục', str(error))
        finally:
            self.refreshing = False

    def selected(self, event=None):
        selection = self.table.selection()
        if self.refreshing or not selection or selection[0] == self.current_id:
            return
        if not self.save(force=False):
            self.table.selection_set(self.current_id)
            return
        self.current_id = selection[0]
        review = self.choices.get(self.current_id, {})
        self.choice.set(CHOICE_LABELS.get(review.get('choice'), CHOICE_LABELS['CHUA_CHON']))
        self.note.set(review.get('notes', ''))
        self.saved_value = (self.choice_code(), self.note.get())
        self.hint.set(review_hint(self.rows[self.current_id]))
        self.load_video()

    def file_path(self, kind):
        row = self.rows.get(self.current_id, {})
        relative = row.get('video') if kind == 'video' else next(iter(row.get('previews', [])), None)
        path = (ROOT / relative).resolve() if relative else None
        if path and path.is_relative_to((ROOT / 'data/iHanoi_capture').resolve()) and path.is_file():
            return path
        return None

    def open_file(self, kind):
        path = self.file_path(kind)
        if path:
            os.startfile(path)
        else:
            messagebox.showinfo('Chưa có file', 'Camera này chưa có file tương ứng.')

    def close_video(self):
        self.pause()
        if self.reader:
            self.reader.close()
        self.reader = None
        self.pending = None
        self.last_frame = None
        self.position = 0
        self.canvas.itemconfigure(self.image_item, image='')
        self.video_title.set('Chọn một camera để xem video')
        self.clock.set('0.0 / 0.0 giây')

    def load_video(self):
        self.close_video()
        path = self.file_path('video')
        self.video_title.set(self.current_id + ' — ' + self.rows[self.current_id].get('name', ''))
        if path is None:
            self.message.set('Camera này chưa có video. Có thể xem trạng thái hoặc ghi chú lý do loại.')
            return
        try:
            self.reader = LocalVideo(path)
            self.slider.configure(to=max(self.reader.duration, 0.1))
            self.seek_position.set(0)
            self.play()
        except (ValueError, cv2.error) as error:
            self.message.set('Không phát được video: ' + str(error))

    def paint(self, frame):
        if frame is None:
            return
        self.last_frame = frame
        height, width = frame.shape[:2]
        shown = frame[int(height*0.35):int(height*0.75)] if self.focus_camera.get() and (width, height) == (900, 1600) else frame
        picture = Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        picture.thumbnail((canvas_width-8, canvas_height-8), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(picture)
        self.canvas.coords(self.image_item, canvas_width/2, canvas_height/2)
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def pause(self):
        self.playing = False
        self.play_button.configure(text='Phát')
        if self.after_id is not None:
            self.window.after_cancel(self.after_id)
            self.after_id = None

    def play(self):
        if self.reader is None or self.playing:
            return
        self.playing = True
        self.play_button.configure(text='Tạm dừng')
        self.origin = time.monotonic()-self.position
        self.schedule_frame()

    def toggle(self):
        self.pause() if self.playing else self.play()

    def schedule_frame(self):
        if not self.playing or self.reader is None:
            return
        try:
            if self.pending is None:
                self.pending = self.reader.read()
            if self.pending is None:
                self.pause()
                return
            delay = max(1, int((self.origin+self.pending[1]-time.monotonic())*1000))
            self.after_id = self.window.after(min(delay, 60000), self.present_frame)
        except cv2.error as error:
            self.pause()
            self.message.set('Lỗi giải mã video: ' + str(error))

    def present_frame(self):
        self.after_id = None
        frame, self.position = self.pending
        self.pending = None
        self.paint(frame)
        self.seek_position.set(self.position)
        self.clock.set(f'{self.position:.1f} / {self.reader.duration:.1f} giây')
        self.schedule_frame()

    def seek(self, seconds, resume=None):
        if self.reader is None:
            return
        was_playing = self.playing if resume is None else resume
        self.pause()
        self.pending = None
        self.reader.seek(seconds)
        result = self.reader.read()
        if result:
            frame, self.position = result
            self.paint(frame)
            self.seek_position.set(self.position)
            self.clock.set(f'{self.position:.1f} / {self.reader.duration:.1f} giây')
        if was_playing:
            self.play()

    def begin_seek(self, event):
        self.resume_after_seek = self.playing
        self.pause()

    def finish_seek(self, event):
        self.seek(self.seek_position.get(), resume=getattr(self, 'resume_after_seek', False))

    def close(self):
        if self.save(force=False):
            self.close_video()
            self.window.destroy()


if __name__ == '__main__':
    cv2.setNumThreads(1)
    root = tk.Tk()
    Review(root)
    root.mainloop()
