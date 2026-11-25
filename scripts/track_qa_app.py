import json
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

FOLDER = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else
          Path(__file__).resolve().parents[1] / "data/Tracking363_working/annotation_drafts/vntraffic_001_mot201_230_track_qa")
RESULT = FOLDER / "tracking_track_qa_result.json"
MODES = {
    "tracks": {"title": "Từng track", "question": "Dải ảnh này là MỘT xe duy nhất xuyên suốt?",
               "choices": [("1", "one", "Một xe duy nhất"), ("2", "mixed", "Lẫn nhiều xe"), ("3", "not_vehicle", "Không phải xe (box thừa)"), ("4", "unclear", "Không rõ")]},
    "pairs": {"title": "Cặp cần gộp", "question": "Hai dải ảnh này là CÙNG MỘT xe?",
              "choices": [("1", "same", "Cùng một xe"), ("2", "different", "Khác xe"), ("3", "unclear", "Không rõ")]},
    "missing": {"title": "Xe có thể thiếu nhãn", "question": "Đây có phải xe THẬT mà bản nháp còn thiếu box?",
                "choices": [("1", "missing_vehicle", "Đúng, một xe thật bị thiếu"), ("2", "multi_vehicle_box", "Box ôm nhiều xe / nhảy sang xe khác"),
                            ("3", "not_vehicle_or_duplicate", "Không phải xe / xe đã có box"), ("4", "unclear", "Không rõ")]},
}


def load_data():
    page = (FOLDER / "track_qa.html").read_text(encoding="utf-8")
    return json.loads(re.search(r"const DATA = (\{.*?\});\nconst KEY", page, re.S).group(1))


class App:
    def __init__(self, root):
        self.root = root
        self.data = load_data()
        self.items = {name: self.data.get(name, []) for name in MODES if self.data.get(name)}
        self.answers = {name: {} for name in self.items}
        if RESULT.exists():
            saved = json.loads(RESULT.read_text(encoding="utf-8"))
            self.answers = {name: saved.get(name, {}) for name in self.items}
        self.mode, self.index = next(iter(self.items)), 0
        self.photos = []
        root.title(f"QA track MOT {self.data['frames'][0]}-{self.data['frames'][1]}")
        root.geometry("1200x820")
        top = tk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        self.tab_buttons = {}
        for name in self.items:
            button = tk.Button(top, command=lambda n=name: self.switch(n))
            button.pack(side="left", padx=2)
            self.tab_buttons[name] = button
        if "tracks" in self.items:
            tk.Button(top, text="Đặt tất cả track chưa trả lời = Một xe duy nhất", command=self.all_tracks_ok).pack(side="right", padx=2)
        self.question = tk.Label(root, font=("Segoe UI", 14, "bold"), anchor="w")
        self.question.pack(fill="x", padx=10)
        self.caption = tk.Label(root, anchor="w", fg="#555")
        self.caption.pack(fill="x", padx=10)
        nav = tk.Frame(root)
        nav.pack(side="bottom", fill="x", padx=10, pady=6)
        self.choices = tk.Frame(root)
        self.choices.pack(side="bottom", fill="x", padx=10, pady=4)
        self.canvas = tk.Frame(root)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=6)
        tk.Button(nav, text="← Trước", command=lambda: self.move(-1)).pack(side="left")
        tk.Button(nav, text="Sau →", command=lambda: self.move(1)).pack(side="left", padx=4)
        tk.Button(nav, text="Mục chưa trả lời (Tab)", command=self.next_open).pack(side="left", padx=4)
        tk.Label(nav, text="Phím: 1–4 chọn, ← → chuyển, Tab tới mục chưa trả lời. Tự lưu vào " + RESULT.name, fg="#555").pack(side="right")
        root.bind("<Key>", self.key)
        root.bind("<Tab>", lambda event: (self.next_open(), "break")[1])
        self.render()

    def save(self):
        RESULT.write_text(json.dumps({"sequence_id": self.data["sequence_id"], "frames": self.data["frames"], **self.answers},
                                     ensure_ascii=False, indent=2), encoding="utf-8")

    def done(self, name):
        return sum(1 for answer in self.answers[name].values() if answer.get("label"))

    def switch(self, name):
        self.mode, self.index = name, 0
        self.render()

    def all_tracks_ok(self):
        if not messagebox.askyesno("Xác nhận", "Đặt mọi track chưa trả lời thành 'Một xe duy nhất'? Các track bạn đã trả lời không bị đổi."):
            return
        for item in self.items["tracks"]:
            if not self.answers["tracks"].get(item["key"], {}).get("label"):
                self.answers["tracks"][item["key"]] = {"label": "one", "note": "đặt hàng loạt theo xác nhận của người dùng"}
        self.save()
        self.render()

    def render(self):
        mode = MODES[self.mode]
        item = self.items[self.mode][self.index]
        for name, button in self.tab_buttons.items():
            button.config(text=f"{MODES[name]['title']} ({self.done(name)}/{len(self.items[name])})", relief="sunken" if name == self.mode else "raised")
        self.question.config(text=mode["question"])
        self.caption.config(text=f"Mục {self.index + 1}/{len(self.items[self.mode])} · {item['caption']}")
        for widget in self.canvas.winfo_children():
            widget.destroy()
        self.photos = []
        width = max(400, self.root.winfo_width() - 40)
        for source in item["images"]:
            image = Image.open(FOLDER / source)
            scale = min(1.0, width / image.width, (360 if len(item["images"]) > 1 else 520) / image.height)
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
            photo = ImageTk.PhotoImage(image)
            self.photos.append(photo)
            tk.Label(self.canvas, image=photo).pack(anchor="w", pady=2)
        current = self.answers[self.mode].get(item["key"], {}).get("label")
        for widget in self.choices.winfo_children():
            widget.destroy()
        for key, label, text in mode["choices"]:
            tk.Button(self.choices, text=f"{key} · {text}", font=("Segoe UI", 11), bg="#cfe3ff" if current == label else None,
                      command=lambda value=label: self.choose(value)).pack(side="left", padx=4)

    def choose(self, label):
        item = self.items[self.mode][self.index]
        self.answers[self.mode][item["key"]] = {"label": label, "note": ""}
        self.save()
        if self.index < len(self.items[self.mode]) - 1:
            self.index += 1
        self.render()

    def move(self, step):
        self.index = max(0, min(len(self.items[self.mode]) - 1, self.index + step))
        self.render()

    def next_open(self):
        items = self.items[self.mode]
        for step in range(1, len(items) + 1):
            position = (self.index + step) % len(items)
            if not self.answers[self.mode].get(items[position]["key"], {}).get("label"):
                self.index = position
                break
        self.render()

    def key(self, event):
        for key, label, _ in MODES[self.mode]["choices"]:
            if event.char == key:
                self.choose(label)
                return
        if event.keysym == "Left":
            self.move(-1)
        elif event.keysym == "Right":
            self.move(1)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
