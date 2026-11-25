"""Light-phase QA: label the true colour of each light head once per second.

First run builds crops (data/light_phase_eval_20261114/qa/). Heads with >= MIN_OBS detections are shown.
The model's predictions are NOT shown, to avoid anchoring the reviewer.
Controls: 1 red, 2 yellow, 3 green, 4 off (unlit), 5 cannot see / not a light; click a cell to paint the
selected colour; "Tô các ô chưa gán" paints every unlabeled cell; ← → previous/next head.
"""
import json
import sys
import tkinter as tk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.signal.phase_voting import run  # noqa: E402

WORK = ROOT / "data/light_phase_eval_20261114"
QA = WORK / "qa"
RESULT = QA / "light_phase_qa_result.json"
SKIP = {"C047_survey"}
MIN_OBS = 8
LABELS = {"1": ("red", "#e33", "Đỏ"), "2": ("yellow", "#e0b000", "Vàng"), "3": ("green", "#2a2", "Xanh"),
          "4": ("off", "#777", "Tắt"), "5": ("not_visible", "#36c", "Không thấy / không phải đèn")}
COLUMNS, CELL = 10, 120


def build():
    items = []
    for d in sorted(p for p in WORK.iterdir() if p.is_dir() and p.name not in SKIP and p.name != "qa"):
        frames = json.loads((d / "detections.json").read_text(encoding="utf-8"))["frames"]
        out, assoc = run(frames)
        for h in assoc.heads:
            if len(h.observations) < MIN_OBS:
                continue
            seen = [(o["t"], o["boxes"][h.head_id]) for o in out if h.head_id in o["boxes"]]
            first, last = seen[0][0], seen[-1][0]
            cells = []
            for k, f in enumerate(frames):
                if not (first <= f["t"] <= last) or abs(f["t"] - round(f["t"])) > 1e-6 or not f["scene"]:
                    continue
                box = min(seen, key=lambda s: abs(s[0] - f["t"]))[1]
                image = cv2.imread(str(d / "frames" / f"{f['index']:05d}.jpg"))
                x1, y1, x2, y2 = box
                m = max(x2 - x1, y2 - y1) * 1.2 + 6
                ox, oy = int(max(0, x1 - m)), int(max(0, y1 - m))
                patch = image[oy:int(min(image.shape[0], y2 + m)), ox:int(min(image.shape[1], x2 + m))].copy()
                cv2.rectangle(patch, (int(x1 - ox) - 2, int(y1 - oy) - 2), (int(x2 - ox) + 2, int(y2 - oy) + 2), (255, 255, 255), 1)
                name = f"{d.name}_H{h.head_id}_t{f['t']:06.1f}.jpg"
                cv2.imwrite(str(QA / "crops" / name), patch)
                cells.append({"t": f["t"], "index": f["index"], "crop": name, "box": box})
            if cells:
                items.append({"key": f"{d.name}:H{h.head_id}", "source": d.name, "head_id": h.head_id, "cells": cells})
    (QA / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    return items


class App:
    def __init__(self, root, items):
        self.root, self.items = root, items
        self.answers = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
        self.position = next((k for k, it in enumerate(items) if not self.complete(it)), 0)
        self.current = "1"
        root.title("QA màu đèn theo thời gian (mỗi ô = 1 giây)")
        root.geometry("1300x900")
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=4)
        self.heading = tk.Label(top, font=("Segoe UI", 13, "bold"), anchor="w")
        self.heading.pack(side="left")
        self.status = tk.Label(top, fg="#555")
        self.status.pack(side="right")
        bar = tk.Frame(root)
        bar.pack(fill="x", padx=8)
        self.buttons = {}
        for key, (_, color, text) in LABELS.items():
            b = tk.Button(bar, text=f"{key} · {text}", fg=color, command=lambda k=key: self.pick(k))
            b.pack(side="left", padx=2)
            self.buttons[key] = b
        tk.Button(bar, text="Tô các ô chưa gán bằng màu đang chọn", command=self.fill).pack(side="left", padx=16)
        tk.Label(root, anchor="w", fg="#555", text="Chọn màu (phím 1–5) rồi bấm vào ô để tô. Mẹo: chọn màu chiếm đa số → “Tô các ô chưa gán” → sửa các ô "
                 "đổi màu. Ô 5 = đèn bị che/ra khỏi khung/không phải đèn. ← → chuyển cột đèn.").pack(fill="x", padx=8)
        nav = tk.Frame(root)
        nav.pack(side="bottom", fill="x", padx=8, pady=6)
        tk.Button(nav, text="← Cột đèn trước", command=lambda: self.move(-1)).pack(side="left")
        tk.Button(nav, text="Cột đèn sau →", command=lambda: self.move(1)).pack(side="left", padx=6)
        holder = tk.Frame(root)
        holder.pack(fill="both", expand=True, padx=8, pady=4)
        sb = tk.Scrollbar(holder)
        sb.pack(side="right", fill="y")
        self.canvas = tk.Canvas(holder, yscrollcommand=sb.set, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.config(command=self.canvas.yview)
        self.grid = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.grid, anchor="nw")
        self.grid.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        root.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        for key in LABELS:
            root.bind(key, lambda e, k=key: self.pick(k))
        root.bind("<Right>", lambda e: self.move(1))
        root.bind("<Left>", lambda e: self.move(-1))
        self.render()

    def complete(self, item):
        a = self.answers.get(item["key"], {})
        return all(str(c["t"]) in a for c in item["cells"])

    def save(self):
        RESULT.write_text(json.dumps(self.answers, indent=1), encoding="utf-8")

    def pick(self, key):
        self.current = key
        for k, b in self.buttons.items():
            b.config(relief="sunken" if k == key else "raised")

    def paint(self, t):
        self.answers.setdefault(self.items[self.position]["key"], {})[str(t)] = LABELS[self.current][0]
        self.save()
        self.refresh()

    def fill(self):
        item = self.items[self.position]
        a = self.answers.setdefault(item["key"], {})
        for c in item["cells"]:
            a.setdefault(str(c["t"]), LABELS[self.current][0])
        self.save()
        self.refresh()

    def render(self):
        for w in self.grid.winfo_children():
            w.destroy()
        item = self.items[self.position]
        self.photos, self.cells = [], {}
        for k, c in enumerate(item["cells"]):
            img = Image.open(QA / "crops" / c["crop"]).convert("RGB")
            s = (CELL - 14) / max(img.size)
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC)
            photo = ImageTk.PhotoImage(img)
            self.photos.append(photo)
            cell = tk.Frame(self.grid, width=CELL, height=CELL + 18, bd=4, relief="flat")
            cell.grid(row=k // COLUMNS, column=k % COLUMNS, padx=2, pady=2)
            cell.grid_propagate(False)
            pic = tk.Label(cell, image=photo)
            pic.pack()
            cap = tk.Label(cell, text=f"{c['t']:.0f}s")
            cap.pack()
            for w in (cell, pic, cap):
                w.bind("<Button-1>", lambda e, t=c["t"]: self.paint(t))
            self.cells[str(c["t"])] = (cell, cap)
        self.canvas.yview_moveto(0)
        self.pick(self.current)
        self.refresh()

    def refresh(self):
        item = self.items[self.position]
        a = self.answers.get(item["key"], {})
        colors = {v[0]: (v[1], v[2]) for v in LABELS.values()}
        for t, (cell, cap) in self.cells.items():
            label = a.get(t)
            cell.config(bg=colors[label][0] if label else self.root.cget("bg"))
            cap.config(text=f"{float(t):.0f}s" + (f" {colors[label][1].split(' ')[0]}" if label else ""))
        done = sum(self.complete(it) for it in self.items)
        self.heading.config(text=f"{item['source']} · cột đèn H{item['head_id']} ({self.position + 1}/{len(self.items)}) · {len(item['cells'])} giây")
        self.status.config(text=f"Đã xong {done}/{len(self.items)} cột đèn")

    def move(self, step):
        self.position = max(0, min(len(self.items) - 1, self.position + step))
        self.render()


def main():
    (QA / "crops").mkdir(parents=True, exist_ok=True)
    items = json.loads((QA / "items.json").read_text(encoding="utf-8")) if (QA / "items.json").exists() else build()
    if "--build-only" in sys.argv:
        print(len(items), "heads,", sum(len(i["cells"]) for i in items), "cells")
        return
    root = tk.Tk()
    App(root, items)
    root.mainloop()


if __name__ == "__main__":
    main()
