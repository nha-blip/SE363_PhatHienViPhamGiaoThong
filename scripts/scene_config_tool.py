"""Add approaches (approach ROI, stop line, direction, controlling signals) to the signal-only SceneConfigs.

For each camera config in configs/scene_configs/ (latest version), draw one or more approaches:
  1. click 3+ points, then Enter or right-click: approach ROI (the road area where vehicles approach the line)
  2. click 2 points: the stop line (edge on the approaching side)
  3. click 2 points: driving direction (from -> to, straight through the junction)
  4. press the number keys of the signal heads that control this approach, then Enter to confirm
Backspace undoes the last click; "Xóa làn cuối" deletes the last approach; ↑↓ change the background frame;
"Lưu phiên bản mới" writes a NEW version (old versions are never modified); → / ← camera.
The observation ROI defaults to the whole image.
"Đèn có mũi tên" marks heads with a green turn arrow (a green reading there cannot prove right of way).
"Xóa hết làn" removes all approaches of the camera to redraw them. --cameras a,b limits the tool to those cameras.
After each approach, drawing mistakes (ROI not reaching the stop line, thin ROI) are shown as warnings.
"""
import argparse
import copy
import json
import sys
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path

from tkinter import messagebox, simpledialog

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scene.config import activation_problems, approach_warnings, save  # noqa: E402

CONFIGS = ROOT / "configs/scene_configs"
STEPS = [("approach_roi", -1, "Vùng làn: bấm các điểm bao cả mặt đường từ MÉP ẢNH tới SÁT vạch dừng (≥3 điểm), xong Enter hoặc chuột phải"),
         ("stop_line", 2, "Vạch dừng THẬT (vạch trắng ngang ngay trước vạch đi bộ): bấm 2 điểm hoặc kéo dọc vạch"),
         ("direction", 2, "Hướng đi: bấm 2 điểm hoặc kéo chuột từ điểm xuất phát tới điểm đích"), ("signals", 0, "Nhấn số của đèn điều khiển làn này, rồi Enter")]


def latest_configs():
    by_camera = {}
    for path in sorted(CONFIGS.glob("*_v*.json")):
        c = json.loads(path.read_text(encoding="utf-8"))
        if c["version"] >= by_camera.get(c["camera_id"], {}).get("version", 0):
            by_camera[c["camera_id"]] = c
    return [by_camera[k] for k in sorted(by_camera)]


class App:
    def __init__(self, root, cameras=None):
        self.root = root
        self.configs = [c for c in latest_configs() if not cameras or c["camera_id"] in cameras]
        if cameras:
            missing = set(cameras) - {c["camera_id"] for c in self.configs}
            if missing:
                raise SystemExit(f"no SceneConfig for {sorted(missing)}")
        self.position = 0 if cameras else next((k for k, c in enumerate(self.configs) if not c["approaches"]), 0)  # first camera without lanes
        root.title("Khai báo vạch dừng / làn / đèn điều khiển")
        root.geometry("1320x930")
        bar = tk.Frame(root)
        bar.pack(side="bottom", fill="x", padx=8, pady=6)
        tk.Button(bar, text="← Camera trước", command=lambda: self.move(-1)).pack(side="left")
        tk.Button(bar, text="Camera sau →", command=lambda: self.move(1)).pack(side="left", padx=4)
        tk.Button(bar, text="Ảnh nền khác ↓", command=lambda: self.change_frame(1)).pack(side="left", padx=4)
        tk.Button(bar, text="Xóa làn cuối", command=self.delete_last).pack(side="left", padx=12)
        tk.Button(bar, text="Xóa hết làn", command=self.delete_all).pack(side="left", padx=4)
        tk.Button(bar, text="Đèn có mũi tên", command=self.toggle_arrow).pack(side="left", padx=4)
        tk.Button(bar, text="Chép làn từ camera cùng góc", command=self.copy_same_view).pack(side="left", padx=4)
        tk.Button(bar, text="Lưu phiên bản mới", command=self.save_version, bg="#c8f0c8").pack(side="left", padx=12)
        self.status = tk.Label(bar, fg="#555", anchor="w", justify="left")
        self.status.pack(side="left", fill="x", expand=True)
        self.hint = tk.Label(root, anchor="w", bg="#ffffe0", font=("Segoe UI", 11))
        self.hint.pack(side="bottom", fill="x")
        self.canvas = tk.Canvas(root, bg="black", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<Button-3>", lambda e: self.close_polygon())
        root.bind("<BackSpace>", lambda e: self.undo())
        root.bind("<Return>", lambda e: self.enter())
        root.bind("<Right>", lambda e: self.move(1))
        root.bind("<Left>", lambda e: self.move(-1))
        root.bind("<Down>", lambda e: self.change_frame(1))
        root.bind("<Up>", lambda e: self.change_frame(-1))
        for n in range(1, 10):
            root.bind(str(n), lambda e, n=n: self.toggle_signal(n))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.load()

    # ----- state -----
    def load(self):
        c = self.configs[self.position]
        self.config = copy.deepcopy(c)
        self.dirty = False
        folder = ROOT / Path(c["reference_image"]["path"]).parent
        self.frames = sorted(folder.glob("*.jpg")) or [ROOT / c["reference_image"]["path"]]
        self.frame_pos = 0
        self.new_approach()
        self.render()

    def new_approach(self):
        self.step, self.points, self.chosen = 0, [], set()
        self.draft = {}

    def change_frame(self, step):
        self.frame_pos = (self.frame_pos + step * max(1, len(self.frames) // 10)) % len(self.frames)
        self.render()

    # ----- input -----
    def release(self, event):
        """A drag (press, move, release) also gives the 2 points of a stop line or direction arrow."""
        key, needed, _ = STEPS[self.step]
        if needed == 2 and len(self.points) == 1:
            x0, y0 = self.points[0]
            if ((event.x / self.scale - x0) ** 2 + (event.y / self.scale - y0) ** 2) ** 0.5 >= 8:
                self.click(event)

    def click(self, event):
        key, needed, _ = STEPS[self.step]
        if needed == 0:
            return
        point = [round(event.x / self.scale, 1), round(event.y / self.scale, 1)]
        if self.points and ((point[0] - self.points[-1][0]) ** 2 + (point[1] - self.points[-1][1]) ** 2) ** 0.5 < 8:
            self.status.config(text="Điểm này trùng/quá gần điểm trước (<8 px) nên bị bỏ qua. Bấm điểm thứ hai ở xa hơn, hoặc kéo chuột.")
            return
        self.points.append(point)
        if len(self.points) == needed:
            if key == "direction":
                (x0, y0), (x1, y1) = self.points
                norm = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1e-6)
                self.draft[key] = [round((x1 - x0) / norm, 4), round((y1 - y0) / norm, 4)]
                self.draft["direction_points"] = self.points
            else:
                self.draft[key] = self.points
            self.points = []
            self.step += 1
        self.render()

    def close_polygon(self):
        if STEPS[self.step][1] == -1 and len(self.points) >= 3:
            self.draft[STEPS[self.step][0]] = self.points
            self.points = []
            self.step += 1
            self.render()

    def enter(self):
        if STEPS[self.step][1] == -1:
            self.close_polygon()
        else:
            self.confirm_signals()

    def undo(self):
        if self.points:
            self.points.pop()
        elif self.step > 0:
            self.step -= 1
            self.draft.pop(STEPS[self.step][0], None)
        self.render()

    def toggle_signal(self, n):
        if STEPS[self.step][0] != "signals" or n > len(self.config["signals"]):
            return
        self.chosen ^= {n}
        self.render()

    def confirm_signals(self):
        if STEPS[self.step][0] != "signals" or not self.chosen:
            return
        w, h = self.config["reference_image"]["width"], self.config["reference_image"]["height"]
        aid = f"A{len(self.config['approaches']) + 1}"
        self.config["approaches"].append({"approach_id": aid, "approach_roi": self.draft["approach_roi"],
                                          "observation_roi": [[0, 0], [w, 0], [w, h], [0, h]], "stop_line": self.draft["stop_line"],
                                          "direction": self.draft["direction"], "direction_points": self.draft["direction_points"]})
        for k in self.chosen:
            s = self.config["signals"][k - 1]
            if aid not in s["controls_approach_ids"]:
                s["controls_approach_ids"].append(aid)
            s["mapping_status"] = "CONFIRMED"
        self.dirty = True
        self.new_approach()
        self.render()
        warnings = approach_warnings({"approaches": [self.config["approaches"][-1]], "reference_image": self.config["reference_image"]})
        if warnings:
            messagebox.showwarning("Kiểm tra làn vừa vẽ", "\n".join(warnings) + "\n\nNếu đúng là lỗi: bấm 'Xóa làn cuối' và vẽ lại.")

    def delete_all(self):
        if not self.config["approaches"] or not messagebox.askyesno("Xóa hết làn", f"Xóa {len(self.config['approaches'])} làn của camera này để vẽ lại?"):
            return
        while self.config["approaches"]:
            self.delete_last()

    def toggle_arrow(self):
        n = len(self.config["signals"])
        text = simpledialog.askstring("Đèn có mũi tên", f"Số của cột đèn có MŨI TÊN xanh (rẽ) cạnh đèn tròn (1..{n}).\nNhập lại số đó để bỏ đánh dấu.", parent=self.root)
        if not text or not text.strip().isdigit() or not 1 <= int(text) <= n:
            return
        sig = self.config["signals"][int(text) - 1]
        sig["has_turn_arrow"] = not sig.get("has_turn_arrow", False)
        self.dirty = True
        self.render()

    def delete_last(self):
        if not self.config["approaches"]:
            return
        aid = self.config["approaches"].pop()["approach_id"]
        for s in self.config["signals"]:
            if aid in s["controls_approach_ids"]:
                s["controls_approach_ids"].remove(aid)
            if not s["controls_approach_ids"]:
                s["mapping_status"] = "UNCONFIRMED"
        self.dirty = True
        self.render()

    def copy_same_view(self):
        """Copy approaches and signal mapping from another clip of the same physical camera (e.g. C069_trial01 -> C069_retry)."""
        me = self.config
        import re
        match = re.match(r"(C\d{3})_", me["camera_id"])  # only clips of the same physical camera code (e.g. C069_*); news segments are different cameras
        if not match:
            self.status.config(text="Camera này không có mã camera chung với clip khác, không chép được. Hãy vẽ trực tiếp.")
            return
        prefix = match.group(1)
        donors = [c for c in latest_configs() if c["camera_id"] != me["camera_id"] and c["camera_id"].startswith(prefix + "_") and c["approaches"]]
        if not donors:
            self.status.config(text=f"Chưa có camera {prefix} nào khác đã lưu làn để chép.")
            return
        donor = donors[0]
        if len(donor["signals"]) != len(me["signals"]):
            self.status.config(text=f"{donor['camera_id']} có {len(donor['signals'])} cột đèn, camera này có {len(me['signals'])}: không chép tự động được.")
            return
        me["approaches"] = copy.deepcopy(donor["approaches"])
        for mine, theirs in zip(me["signals"], donor["signals"]):
            mine["controls_approach_ids"] = list(theirs["controls_approach_ids"])
            mine["mapping_status"] = theirs["mapping_status"]
        me.setdefault("notes", []).append(f"approaches copied from {donor['scene_config_id']}; checked visually by the owner before saving")
        self.dirty = True
        self.render()
        self.status.config(text=f"Đã chép {len(me['approaches'])} làn từ {donor['scene_config_id']}. KIỂM TRA bằng mắt (↑↓ đổi ảnh nền), lệch thì Xóa làn cuối và vẽ lại.")

    def save_version(self):
        if not self.dirty and not (self.config["status"] != "ACTIVE" and self.config["approaches"] and not activation_problems(self.config)):
            self.status.config(text="Không có thay đổi để lưu.")
            return
        c = self.config
        c["version"] = max(x["version"] for x in latest_configs() if x["camera_id"] == c["camera_id"]) + 1
        c["scene_config_id"] = f"{c['camera_id']}:v{c['version']}"
        c["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        c["created_by"] = "project owner (scene_config_tool.py)"
        problems = activation_problems(c)
        if problems:
            c["status"], c["activation"] = "DRAFT", None
        else:
            c["status"] = "ACTIVE"
            c["activation"] = {"activated_by": "project owner", "activated_at_utc": c["created_at_utc"],
                               "note": "signal-to-approach mapping confirmed in scene_config_tool.py"}
        path = save(c, CONFIGS)
        self.configs[self.position] = copy.deepcopy(c)
        self.dirty = False
        self.render()
        self.status.config(text=f"Đã lưu {path.name} · trạng thái {c['status']}" + (f" · thiếu: {problems[0]}" if problems else ""))

    def leave_ok(self):
        """Before leaving a camera: auto-save finished approaches, ask about an unfinished one."""
        if self.step > 0 or self.points:
            if not messagebox.askyesno("Làn đang vẽ dở", "Làn đang vẽ chưa xong (chưa chọn đèn + Enter). Bỏ làn đang vẽ dở này?"):
                return False
            self.new_approach()
        if self.dirty:
            self.save_version()
        return True

    def move(self, step):
        if not self.leave_ok():
            return
        self.position = max(0, min(len(self.configs) - 1, self.position + step))
        self.load()

    def close(self):
        if self.leave_ok():
            self.root.destroy()

    # ----- drawing -----
    def render(self):
        image = Image.open(self.frames[self.frame_pos]).convert("RGB")
        self.scale = min(1290 / image.width, 800 / image.height)
        self.photo = ImageTk.PhotoImage(image.resize((int(image.width * self.scale), int(image.height * self.scale)), Image.BICUBIC))
        c, s = self.canvas, self.scale
        c.delete("all")
        c.create_image(0, 0, image=self.photo, anchor="nw")
        for k, sig in enumerate(self.config["signals"], 1):
            x1, y1, x2, y2 = [v * s for v in sig["roi_xyxy"]]
            colour = "#ffff00" if k in self.chosen else "#00ffff"
            c.create_rectangle(x1, y1, x2, y2, outline=colour, width=2)
            c.create_text(x1, y1 - 2, anchor="sw", fill=colour, font=("Segoe UI", 12, "bold"),
                          text=f"{k}" + (" (mũi tên)" if sig.get("has_turn_arrow") else "")
                          + (f" → {','.join(sig['controls_approach_ids'])}" if sig["controls_approach_ids"] else ""))
        for a in self.config["approaches"]:
            self.draw_approach(a, "#00ff66", a["approach_id"])
        if self.draft or self.points:
            self.draw_approach(self.draft, "#ff9900", "mới")
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:]):
            c.create_line(x0 * s, y0 * s, x1 * s, y1 * s, fill="#ff9900", width=2)
        if STEPS[self.step][1] == -1 and len(self.points) >= 3:  # preview of the closing edge
            (x0, y0), (x1, y1) = self.points[-1], self.points[0]
            c.create_line(x0 * s, y0 * s, x1 * s, y1 * s, fill="#ff9900", width=1, dash=(4, 3))
        for k, (x, y) in enumerate(self.points, 1):
            c.create_oval(x * s - 5, y * s - 5, x * s + 5, y * s + 5, fill="#ff9900")
            c.create_text(x * s + 8, y * s - 8, text=str(k), fill="#ff9900", font=("Segoe UI", 10, "bold"))
        self.hint.config(text=f"Bước {self.step + 1}/4: {STEPS[self.step][2]}" + (f" — đang chọn: {sorted(self.chosen)}" if self.chosen else "")
                         + "   |   Backspace: hoàn tác   ·   Xong làn thì làm làn tiếp hoặc bấm Lưu phiên bản mới")
        cfg = self.config
        self.status.config(text=f"Camera {self.position + 1}/{len(self.configs)}: {cfg['camera_id']} · phiên bản hiện tại v{self.configs[self.position]['version']} "
                                f"({self.configs[self.position]['status']}) · {len(cfg['approaches'])} làn" + (" · CHƯA LƯU" if self.dirty else ""))

    def draw_approach(self, a, colour, label):
        c, s = self.canvas, self.scale
        if a.get("approach_roi"):
            pts = [v * s for p in a["approach_roi"] for v in p]
            c.create_polygon(*pts, outline=colour, fill="", width=2, dash=(5, 3))
            c.create_text(pts[0], pts[1], anchor="sw", fill=colour, text=label, font=("Segoe UI", 12, "bold"))
        if a.get("stop_line"):
            (x0, y0), (x1, y1) = a["stop_line"]
            c.create_line(x0 * s, y0 * s, x1 * s, y1 * s, fill="#ff3333", width=3)
        if a.get("direction_points"):
            (x0, y0), (x1, y1) = a["direction_points"]
            c.create_line(x0 * s, y0 * s, x1 * s, y1 * s, fill=colour, width=3, arrow="last", arrowshape=(16, 20, 6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cameras", help="comma-separated camera ids to show (default: all)")
    arguments = parser.parse_args()
    window = tk.Tk()
    App(window, [c.strip() for c in arguments.cameras.split(",")] if arguments.cameras else None)
    window.mainloop()
