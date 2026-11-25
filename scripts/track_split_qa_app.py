import json
import re
import sys
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "data/Tracking363_working/annotation_drafts/vntraffic_001_mot201_300"
FOLDER = ROOT / "data/Tracking363_working/annotation_drafts/vntraffic_001_mot201_230_track_qa"
RESULT = FOLDER / "track_split_qa_result.json"
TRACKS = [32, 1189, 1363, 1493, 1494, 1517]
FIRST_FRAME, LAST_FRAME = 201, 230
COLUMNS, CELL_W, CELL_H = 6, 190, 175
if len(sys.argv) > 1:  # usage: track_split_qa_app.py <track_qa folder>; tracks = those answered "mixed"
    FOLDER = Path(sys.argv[1]).resolve()
    RESULT = FOLDER / "track_split_qa_result.json"
    _page = (FOLDER / "track_qa.html").read_text(encoding="utf-8")
    _data = json.loads(re.search(r"const DATA = (\{.*?\});\nconst KEY", _page, re.S).group(1))
    DRAFT = Path(_data["draft"])
    FIRST_FRAME, LAST_FRAME = _data["frames"]
    _answers = json.loads((FOLDER / "tracking_track_qa_result.json").read_text(encoding="utf-8"))["tracks"]
    TRACKS = sorted(int(k) for k, v in _answers.items() if v.get("label") == "mixed")
    COLUMNS, CELL_W, CELL_H = 9, 132, 128


def load_rows():
    draft = json.loads((DRAFT / "annotations_draft.json").read_text(encoding="utf-8"))
    rows = defaultdict(list)
    for frame in draft["frames"]:
        if FIRST_FRAME <= frame["mot_frame_id"] <= LAST_FRAME:
            for item in frame["objects"]:
                if item["track_id"] in TRACKS:
                    rows[item["track_id"]].append((frame["mot_frame_id"], item["bbox_xywh"], frame["image"]))
    for value in rows.values():
        value.sort()
    return rows


def thumbnail(row):
    frame, (left, top, width, height), path = row
    image = cv2.imread(str(DRAFT / path))
    margin = 0.25 * max(width, height)
    x0, y0 = int(max(0, left - margin)), int(max(0, top - margin))
    x1, y1 = int(min(image.shape[1], left + width + margin)), int(min(image.shape[0], top + height + margin))
    patch = image[y0:y1, x0:x1].copy()
    cv2.rectangle(patch, (int(left - x0), int(top - y0)), (int(left + width - x0), int(top + height - y0)), (0, 255, 0), 1)
    scale = min((CELL_W - 8) / patch.shape[1], (CELL_H - 40) / patch.shape[0])
    patch = cv2.resize(patch, (max(8, int(patch.shape[1] * scale)), max(8, int(patch.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))


class App:
    def __init__(self, root):
        self.root = root
        root.title("QA tách xe trong track (khung hình đổi ID)")
        root.geometry("1250x900")
        self.rows = load_rows()
        self.answers = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.is_file() else {}
        self.position = 0
        self.cells = {}
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        self.heading = tk.Label(top, font=("Segoe UI", 13, "bold"), anchor="w")
        self.heading.pack(side="left")
        self.status = tk.Label(top, anchor="e", fg="#555")
        self.status.pack(side="right")
        tk.Label(root, anchor="w", justify="left", wraplength=1200,
                 text="Chọn kiểu đánh dấu rồi bấm vào ảnh; bấm lại để bỏ. “Xe mới” = ảnh ĐẦU TIÊN của một xe khác với ảnh trước. "
                      "“Box thừa” = box ở khung đó không phải xe, hoặc là box trùng/thừa. Track không có gì sai thì bấm “Không có lỗi”.").pack(fill="x", padx=8)
        self.mode = tk.StringVar(value="switch")
        choice = tk.Frame(root)
        choice.pack(fill="x", padx=8)
        tk.Radiobutton(choice, text="Đánh dấu: XE MỚI (đỏ)", variable=self.mode, value="switch", fg="#d33").pack(side="left")
        tk.Radiobutton(choice, text="Đánh dấu: BOX THỪA (cam)", variable=self.mode, value="extra", fg="#d77a00").pack(side="left", padx=12)
        tk.Label(choice, text="Ghi chú:").pack(side="left", padx=(24, 4))
        self.note = tk.Entry(choice, width=60)
        self.note.pack(side="left")
        self.note.bind("<KeyRelease>", lambda event: self.save_note())
        bottom = tk.Frame(root)
        bottom.pack(side="bottom", fill="x", padx=8, pady=6)
        holder = tk.Frame(root)
        holder.pack(fill="both", expand=True, padx=8, pady=6)
        scroll = tk.Scrollbar(holder, orient="vertical")
        scroll.pack(side="right", fill="y")
        self.scroll_canvas = tk.Canvas(holder, yscrollcommand=scroll.set, highlightthickness=0)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.scroll_canvas.yview)
        self.grid_frame = tk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", lambda event: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        root.bind_all("<MouseWheel>", lambda event: self.scroll_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"))
        tk.Button(bottom, text="← Track trước", command=lambda: self.move(-1)).pack(side="left")
        tk.Button(bottom, text="Track sau →", command=lambda: self.move(1)).pack(side="left", padx=6)
        tk.Button(bottom, text="Không có lỗi", command=self.no_switch).pack(side="left", padx=18)
        tk.Button(bottom, text="Xong track này → track sau", command=self.done).pack(side="left")
        root.bind("<Right>", lambda event: self.move(1) if event.widget is not self.note else None)
        root.bind("<Left>", lambda event: self.move(-1) if event.widget is not self.note else None)
        self.render()

    def entry(self):
        return self.answers.setdefault(str(TRACKS[self.position]), {"switch_frames": [], "extra_frames": [], "status": "", "note": ""})

    def save(self):
        RESULT.write_text(json.dumps(self.answers, indent=2), encoding="utf-8")

    def render(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        track = TRACKS[self.position]
        rows = self.rows[track]
        self.images = []
        self.cells = {}
        for index, row in enumerate(rows):
            image = thumbnail(row)
            self.images.append(image)
            cell = tk.Frame(self.grid_frame, width=CELL_W, height=CELL_H, bd=3, relief="flat")
            cell.grid(row=index // COLUMNS, column=index % COLUMNS, padx=2, pady=2)
            cell.grid_propagate(False)
            picture = tk.Label(cell, image=image)
            picture.pack()
            caption = tk.Label(cell, text=f"f{row[0]}")
            caption.pack()
            for widget in (cell, picture, caption):
                widget.bind("<Button-1>", lambda event, frame=row[0]: self.toggle(frame))
            self.cells[row[0]] = (cell, caption)
        self.scroll_canvas.yview_moveto(0)
        self.refresh()

    def refresh(self):
        track = TRACKS[self.position]
        entry = self.entry()
        entry.setdefault("extra_frames", [])
        for frame, (cell, caption) in self.cells.items():
            if frame in entry["switch_frames"]:
                cell.config(bg="#d33")
                caption.config(text=f"f{frame}  ← XE MỚI", fg="#d33")
            elif frame in entry["extra_frames"]:
                cell.config(bg="#d77a00")
                caption.config(text=f"f{frame}  ← BOX THỪA", fg="#d77a00")
            else:
                cell.config(bg=self.root.cget("bg"))
                caption.config(text=f"f{frame}", fg="black")
        self.note.delete(0, "end")
        self.note.insert(0, entry.get("note", ""))
        self.heading.config(text=f"Track {track}  ({self.position + 1}/{len(TRACKS)})  ·  {len(self.rows[track])} khung hình")
        done = sum(1 for key in map(str, TRACKS) if self.answers.get(key, {}).get("status"))
        self.status.config(text=f"Trạng thái: {entry['status'] or 'chưa xong'}  ·  đã xong {done}/{len(TRACKS)}")

    def toggle(self, frame):
        entry = self.entry()
        key = "switch_frames" if self.mode.get() == "switch" else "extra_frames"
        entry.setdefault("extra_frames", [])
        if frame in entry[key]:
            entry[key].remove(frame)
        else:
            entry[key].append(frame)
            entry[key].sort()
        entry["status"] = ""
        self.save()
        self.refresh()

    def save_note(self):
        self.entry()["note"] = self.note.get()
        self.save()

    def no_switch(self):
        entry = self.entry()
        entry["switch_frames"], entry["extra_frames"], entry["status"] = [], [], "no_error"
        self.save()
        self.move(1)

    def done(self):
        entry = self.entry()
        if not entry["switch_frames"] and not entry["extra_frames"] and not entry.get("note"):
            messagebox.showinfo("Chưa đánh dấu", "Chưa đánh dấu khung nào hoặc ghi chú. Nếu track không có lỗi, bấm “Không có lỗi”.")
            return
        entry["status"] = "marked"
        self.save()
        self.move(1)

    def move(self, step):
        self.position = max(0, min(len(TRACKS) - 1, self.position + step))
        self.render()


if __name__ == "__main__":
    window = tk.Tk()
    App(window)
    window.mainloop()
