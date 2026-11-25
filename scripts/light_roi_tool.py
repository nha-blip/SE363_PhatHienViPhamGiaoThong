"""Draw the signal-head ROIs for each camera once (per-camera configuration).

Drag to draw a box around each traffic-light head that should be monitored; right-click a box to delete it.
↑ ↓ change the background frame (pick a clear one); Enter / → next camera; ← previous camera.
Saved to data/light_phase_eval_20261114/signal_head_rois.json after every change.
"""
import json
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/light_phase_eval_20261114"
RESULT = WORK / "signal_head_rois.json"
SKIP = {"C047_survey", "qa"}
SCALE_TO = 1250


class App:
    def __init__(self, root):
        self.root = root
        self.sources = sorted(p.name for p in WORK.iterdir() if p.is_dir() and p.name not in SKIP)
        self.rois = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
        self.position = next((k for k, name in enumerate(self.sources) if name not in self.rois), 0)  # first camera not done yet
        self.frame_pos, self.start = 0, None
        root.title("Khai báo cột đèn theo camera")
        root.geometry("1300x900")
        bar = tk.Frame(root)
        bar.pack(side="bottom", fill="x", padx=8, pady=6)
        tk.Button(bar, text="← Camera trước", command=lambda: self.move(-1)).pack(side="left")
        tk.Button(bar, text="Camera sau → (Enter)", command=lambda: self.move(1)).pack(side="left", padx=6)
        tk.Button(bar, text="Ảnh nền khác ↓", command=lambda: self.change_frame(1)).pack(side="left", padx=6)
        tk.Button(bar, text="Xóa hết khung camera này", command=self.clear).pack(side="left", padx=20)
        self.status = tk.Label(bar, fg="#555")
        self.status.pack(side="left", padx=10)
        tk.Label(root, anchor="w", bg="#ffffe0", text="Kéo chuột: vẽ khung quanh MỘT cột đèn cần theo dõi (chỉ đèn giao thông thật, mỗi cột một khung). "
                 "Chuột phải vào khung: xóa. ↑↓: đổi ảnh nền. Enter/→: camera sau. Không có đèn cần theo dõi thì để trống.").pack(side="bottom", fill="x")
        self.canvas = tk.Canvas(root, bg="black", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<Button-3>", self.delete)
        root.bind("<Return>", lambda e: self.move(1))
        root.bind("<Right>", lambda e: self.move(1))
        root.bind("<Left>", lambda e: self.move(-1))
        root.bind("<Down>", lambda e: self.change_frame(1))
        root.bind("<Up>", lambda e: self.change_frame(-1))
        self.load()

    def save(self):
        RESULT.write_text(json.dumps(self.rois, indent=1), encoding="utf-8")

    @property
    def name(self):
        return self.sources[self.position]

    def load(self):
        frames = sorted((WORK / self.name / "frames").glob("*.jpg"))
        self.frames = frames
        self.frame_pos = min(self.frame_pos, len(frames) - 1) if frames else 0
        self.render()

    def change_frame(self, step):
        if self.frames:
            self.frame_pos = (self.frame_pos + step * max(1, len(self.frames) // 10)) % len(self.frames)
            self.render()

    def render(self):
        image = Image.open(self.frames[self.frame_pos]).convert("RGB")
        self.scale = min(SCALE_TO / image.width, 780 / image.height)
        self.photo = ImageTk.PhotoImage(image.resize((int(image.width * self.scale), int(image.height * self.scale)), Image.BICUBIC))
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, image=self.photo, anchor="nw")
        for roi in self.rois.get(self.name, []):
            x1, y1, x2, y2 = [v * self.scale for v in roi["xyxy"]]
            c.create_rectangle(x1, y1, x2, y2, outline="#00ffff", width=2)
            c.create_text(x1, y1 - 2, text=f"đèn {roi['id']}", anchor="sw", fill="#00ffff", font=("Segoe UI", 10, "bold"))
        self.status.config(text=f"Camera {self.position + 1}/{len(self.sources)}: {self.name} · ảnh {self.frame_pos + 1}/{len(self.frames)} · "
                                f"{len(self.rois.get(self.name, []))} cột đèn · đã khai báo {sum(1 for s in self.sources if s in self.rois)}/{len(self.sources)} camera")

    def press(self, event):
        self.start = (event.x, event.y)

    def drag(self, event):
        self.canvas.delete("drag")
        if self.start:
            self.canvas.create_rectangle(*self.start, event.x, event.y, outline="white", dash=(3, 2), tags="drag")

    def release(self, event):
        if not self.start:
            return
        (x0, y0), (x1, y1) = self.start, (event.x, event.y)
        self.start = None
        self.canvas.delete("drag")
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return
        rois = self.rois.setdefault(self.name, [])
        box = [round(min(x0, x1) / self.scale, 1), round(min(y0, y1) / self.scale, 1), round(max(x0, x1) / self.scale, 1), round(max(y0, y1) / self.scale, 1)]
        rois.append({"id": max([r["id"] for r in rois], default=0) + 1, "xyxy": box})
        self.save()
        self.render()

    def delete(self, event):
        x, y = event.x / self.scale, event.y / self.scale
        rois = self.rois.get(self.name, [])
        hits = [r for r in rois if r["xyxy"][0] <= x <= r["xyxy"][2] and r["xyxy"][1] <= y <= r["xyxy"][3]]
        if hits:
            rois.remove(min(hits, key=lambda r: (r["xyxy"][2] - r["xyxy"][0]) * (r["xyxy"][3] - r["xyxy"][1])))
            self.save()
            self.render()

    def clear(self):
        self.rois[self.name] = []
        self.save()
        self.render()

    def move(self, step):
        self.rois.setdefault(self.name, [])  # visiting a camera counts as declared (possibly with no heads)
        self.save()
        self.position = max(0, min(len(self.sources) - 1, self.position + step))
        self.frame_pos = 0
        self.load()


if __name__ == "__main__":
    window = tk.Tk()
    App(window)
    window.mainloop()
