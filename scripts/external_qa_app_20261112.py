"""QA app for the Bellevue external test images (vehicle + traffic light boxes).

Pre-labels are the union of all models. The reviewer fixes them so the result becomes ground truth.
Controls (also shown in the window):
  click box        select            drag on image   draw a new box (class = last used)
  1 2 3 4          car / motorcycle / bus / truck
  5 6 7 8          red / yellow / green / off
  x or Delete      mark/unmark box as WRONG (no object there)
  i                mark/unmark box as UNSURE (excluded from scoring)
  Enter / Right    save, mark image reviewed, next image      Left  previous image
  + / -  or Ctrl+wheel   zoom                                 Tab   show both / vehicle only / light only
"""
import json
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "data/external_qa_bellevue_20261112"
RESULT = WORK / "qa_result.json"
CLASSES = [("vehicle", 0, "car"), ("vehicle", 1, "motorcycle"), ("vehicle", 2, "bus"), ("vehicle", 3, "truck"),
           ("light", 0, "red"), ("light", 1, "yellow"), ("light", 2, "green"), ("light", 3, "off")]
COLORS = {("vehicle", 0): "#00c8ff", ("vehicle", 1): "#ff00ff", ("vehicle", 2): "#ffa500", ("vehicle", 3): "#00ff66",
          ("light", 0): "#ff2020", ("light", 1): "#ffe000", ("light", 2): "#20ff20", ("light", 3): "#b0b0b0"}
LAYERS = ["Cả hai", "Chỉ xe", "Chỉ đèn"]


def name_of(item):
    return next(n for g, c, n in CLASSES if g == item["group"] and c == item["cls"])


class App:
    def __init__(self, root):
        self.root = root
        if RESULT.exists():
            self.state = json.loads(RESULT.read_text(encoding="utf-8"))
        else:
            pre = json.loads((WORK / "prelabels.json").read_text(encoding="utf-8"))
            self.state = {"images": [{"path": p, "reviewed": False, "boxes": boxes} for p, boxes in pre.items()]}
        self.index = next((k for k, im in enumerate(self.state["images"]) if not im["reviewed"]), 0)
        self.zoom, self.layer, self.selected, self.last_class = 1.0, 0, None, CLASSES[0]
        self.press = None
        self.drag_rect = None

        root.title(f"QA ảnh ngoài ({WORK.name}) - xe + đèn")
        bar = tk.Frame(root)
        bar.pack(side=tk.BOTTOM, fill=tk.X)  # pack first so it never gets pushed off-screen
        tk.Button(bar, text="◀ Trước", command=self.prev).pack(side=tk.LEFT)
        tk.Button(bar, text="Xong ảnh ▶ (Enter)", command=self.next, bg="#c8f0c8").pack(side=tk.LEFT)
        tk.Button(bar, text="Phóng +", command=lambda: self.set_zoom(self.zoom * 1.5)).pack(side=tk.LEFT)
        tk.Button(bar, text="Thu −", command=lambda: self.set_zoom(self.zoom / 1.5)).pack(side=tk.LEFT)
        tk.Button(bar, text="Lớp hiển thị (Tab)", command=self.cycle_layer).pack(side=tk.LEFT)
        self.status = tk.Label(bar, anchor="w", justify=tk.LEFT, font=("Segoe UI", 10))
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        help_text = ("Bấm box: chọn | Kéo chuột: vẽ box bị sót | 1-4: car/moto/bus/truck | 5-8: đỏ/vàng/xanh/tắt | "
                     "x/Delete: box SAI (box tự vẽ thì xóa luôn) | i: không chắc | Enter/→: xong, ảnh sau | ←: ảnh trước | +/-: phóng | Tab: lớp")
        tk.Label(root, text=help_text, anchor="w", bg="#ffffe0").pack(side=tk.BOTTOM, fill=tk.X)

        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(frame, bg="black", cursor="crosshair")
        xs = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        ys = tk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        xs.pack(side=tk.BOTTOM, fill=tk.X)
        ys.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        for k, (g, c, n) in enumerate(CLASSES):
            root.bind(str(k + 1), lambda e, cl=(g, c, n): self.set_class(cl))
        root.bind("x", lambda e: self.toggle("wrong"))
        root.bind("<Delete>", lambda e: self.toggle("wrong"))
        root.bind("i", lambda e: self.toggle("unsure"))
        root.bind("<Return>", lambda e: self.next())
        root.bind("<Right>", lambda e: self.next())
        root.bind("<Left>", lambda e: self.prev())
        root.bind("g", lambda e: self.goto())
        root.bind("+", lambda e: self.set_zoom(self.zoom * 1.5))
        root.bind("=", lambda e: self.set_zoom(self.zoom * 1.5))
        root.bind("-", lambda e: self.set_zoom(self.zoom / 1.5))
        root.bind("<Tab>", lambda e: (self.cycle_layer(), "break")[1])
        self.load()

    # ---------- data ----------
    @property
    def image(self):
        return self.state["images"][self.index]

    def save(self):
        tmp = RESULT.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(RESULT)

    def visible(self, box):
        return self.layer == 0 or (self.layer == 1) == (box["group"] == "vehicle")

    # ---------- drawing ----------
    def load(self):
        self.pil = Image.open(self.image["path"]).convert("RGB")
        self.selected = None
        self.render()

    def set_zoom(self, z):
        self.zoom = min(max(z, 0.5), 6.0)
        self.render()

    def cycle_layer(self):
        self.layer = (self.layer + 1) % 3
        self.selected = None
        self.render()

    def render(self):
        w, h = self.pil.size
        z = self.zoom
        self.tk_img = ImageTk.PhotoImage(self.pil.resize((int(w * z), int(h * z)), Image.BILINEAR))
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, image=self.tk_img, anchor="nw")
        c.configure(scrollregion=(0, 0, w * z, h * z))
        for k, b in enumerate(self.image["boxes"]):
            if not self.visible(b):
                continue
            x1, y1, x2, y2 = [v * z for v in b["xyxy"]]
            color = COLORS[(b["group"], b["cls"])]
            width = 3 if k == self.selected else 1
            dash = (4, 3) if b["status"] == "unsure" else None
            c.create_rectangle(x1, y1, x2, y2, outline=color, width=width, dash=dash)
            if b["status"] == "wrong":
                c.create_line(x1, y1, x2, y2, fill="red", width=2)
                c.create_line(x1, y2, x2, y1, fill="red", width=2)
            tag = name_of(b) + (" SAI" if b["status"] == "wrong" else " ?" if b["status"] == "unsure" else "") + \
                (" +" if b.get("from") == "human" else "")
            if k == self.selected or z >= 1.5 or b["group"] == "vehicle":
                c.create_text(x1, y1 - 2, text=tag, anchor="sw", fill=color, font=("Segoe UI", 8 if z < 1.5 else 10))
        self.update_status()

    def update_status(self):
        im = self.image
        n_rev = sum(i["reviewed"] for i in self.state["images"])
        boxes = [b for b in im["boxes"] if self.visible(b)]
        sel = ""
        if self.selected is not None:
            b = im["boxes"][self.selected]
            sel = f" | Đang chọn: {name_of(b)} ({b['status']}, từ {b.get('from')})"
        self.status.config(text=f"Ảnh {self.index + 1}/{len(self.state['images'])}  {Path(im['path']).parent.parent.name}/"
                                f"{Path(im['path']).name}  {'ĐÃ QA' if im['reviewed'] else 'chưa QA'} | Đã QA {n_rev} ảnh | "
                                f"{LAYERS[self.layer]} | {len(boxes)} box | zoom {self.zoom:.1f}x | Vẽ mới: {self.last_class[2]}{sel}")

    # ---------- interaction ----------
    def to_image(self, event):
        return self.canvas.canvasx(event.x) / self.zoom, self.canvas.canvasy(event.y) / self.zoom

    def on_press(self, event):
        self.press = self.to_image(event)

    def on_drag(self, event):
        if self.press is None:
            return
        x, y = self.to_image(event)
        z = self.zoom
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
        self.drag_rect = self.canvas.create_rectangle(self.press[0] * z, self.press[1] * z, x * z, y * z,
                                                      outline="white", dash=(3, 2))

    def on_release(self, event):
        if self.press is None:
            return
        x0, y0 = self.press
        x1, y1 = self.to_image(event)
        self.press = None
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None
        if abs(x1 - x0) * self.zoom > 5 and abs(y1 - y0) * self.zoom > 5:
            g, c, _ = self.last_class
            w, h = self.pil.size
            box = [max(0, min(x0, x1)), max(0, min(y0, y1)), min(w, max(x0, x1)), min(h, max(y0, y1))]
            self.image["boxes"].append({"group": g, "xyxy": [round(v, 1) for v in box], "cls": c, "status": "ok", "from": "human"})
            self.selected = len(self.image["boxes"]) - 1
            self.save()
        else:  # click: smallest visible box under the cursor; clicking again cycles to the next larger one (overlapping boxes)
            hits = [k for _, k in sorted((abs((b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1])), k)
                    for k, b in enumerate(self.image["boxes"]) if self.visible(b)
                    and b["xyxy"][0] - 2 <= x1 <= b["xyxy"][2] + 2 and b["xyxy"][1] - 2 <= y1 <= b["xyxy"][3] + 2)]
            self.selected = (hits[(hits.index(self.selected) + 1) % len(hits)] if self.selected in hits else hits[0]) if hits else None
        self.render()

    def on_wheel(self, event):
        if event.state & 0x0004:  # Ctrl
            self.set_zoom(self.zoom * (1.25 if event.delta > 0 else 0.8))
        elif event.state & 0x0001:  # Shift
            self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
        else:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def set_class(self, cl):
        self.last_class = cl
        if self.selected is not None:
            b = self.image["boxes"][self.selected]
            b["group"], b["cls"] = cl[0], cl[1]
            if b["status"] == "pending":
                b["status"] = "ok"
            self.save()
        self.render()

    def toggle(self, status):
        if self.selected is None:
            return
        b = self.image["boxes"][self.selected]
        if status == "wrong" and b.get("from") == "human":  # a box you drew yourself is removed outright
            del self.image["boxes"][self.selected]
            self.selected = None
        else:
            b["status"] = "pending" if b["status"] == status else status
        self.save()
        self.render()

    def next(self):
        self.image["reviewed"] = True
        self.save()
        if self.index < len(self.state["images"]) - 1:
            self.index += 1
            self.load()
        else:
            self.render()
            self.status.config(text=self.status.cget("text") + "   >>> ĐÃ HẾT ẢNH, kết quả đã lưu.")

    def goto(self):
        from tkinter import simpledialog
        n = simpledialog.askinteger("Đi tới ảnh", f"Số ảnh (1..{len(self.state['images'])})", parent=self.root,
                                    minvalue=1, maxvalue=len(self.state["images"]))
        if n:
            self.save()
            self.index = n - 1
            self.load()

    def prev(self):
        self.save()
        if self.index > 0:
            self.index -= 1
            self.load()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1340x880")
    App(root)
    root.mainloop()
    sys.exit(0)
