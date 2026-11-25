import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "data/Tracking363_working/annotation_drafts/vntraffic_001_mot201_300"
OUTPUT = ROOT / "data/Tracking363_working/annotation_drafts/vntraffic_001_mot201_230_track_qa"
FIRST_FRAME, LAST_FRAME = 201, 230
STRIP_FRAMES = 5
MAX_GAP_FRAMES = 12
# Merges the user already identified by hand; asked again as pair questions so the answers are recorded.
KNOWN_PAIRS = [(1363, 54), (1473, 33), (1494, 44), (1189, 32)]

STYLE = """:root{--bg:#f6f6f4;--fg:#1b1b1b;--card:#fff;--line:#d8d8d2;--accent:#1f6feb;--muted:#6b6b66}
@media (prefers-color-scheme:dark){:root{--bg:#181818;--fg:#ececec;--card:#242424;--line:#3a3a3a;--accent:#5b9dff;--muted:#a0a0a0}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;z-index:2}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
main{max-width:1100px;margin:0 auto;padding:16px}img{max-width:100%;border:1px solid var(--line);background:#000;display:block;margin:6px 0}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0}.bar i{display:block;height:100%;background:var(--accent)}
.choices{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.choices button{padding:10px 14px}.muted{color:var(--muted);font-size:13px}
textarea{width:100%;box-sizing:border-box;background:var(--card);color:inherit;border:1px solid var(--line);border-radius:6px;padding:6px;font:inherit}"""

PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QA track</title><style>__STYLE__</style></head><body>
<header><b>QA theo track, MOT 201–230</b><span id="tabs"></span><span class="muted" id="status"></span><span style="margin-left:auto"><button id="json">Xuất JSON</button></span></header>
<main><div class="bar"><i id="bar"></i></div><div id="count" class="muted"></div><h2 id="question" style="font-size:17px"></h2><div id="view"></div>
<div class="choices" id="choices"></div><textarea id="note" rows="2" placeholder="Ghi chú (không bắt buộc)"></textarea>
<p><button id="prev">← Trước</button> <button id="next">Sau →</button> <button id="nextOpen">Mục chưa trả lời ⇥</button></p>
<p class="muted">Phím tắt: 1–4 chọn đáp án, ← → chuyển mục, Tab nhảy đến mục chưa trả lời. Tự lưu trong trình duyệt; bấm “Xuất JSON” khi xong.</p></main>
<script>
const DATA = __DATA__;
const KEY = "tracking-track-qa-201-230";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let mode = "tracks", index = 0;
const $ = id => document.getElementById(id);
const MODES = {
  tracks: {title: "Từng track", items: DATA.tracks, question: "Dải ảnh này là MỘT xe duy nhất xuyên suốt?",
    choices: [["1", "one", "Một xe duy nhất"], ["2", "mixed", "Lẫn nhiều xe"], ["3", "not_vehicle", "Không phải xe (box thừa)"], ["4", "unclear", "Không rõ"]]},
  pairs: {title: "Cặp cần gộp", items: DATA.pairs, question: "Hai dải ảnh này là CÙNG MỘT xe?",
    choices: [["1", "same", "Cùng một xe"], ["2", "different", "Khác xe"], ["3", "unclear", "Không rõ"]]},
};
function key(item) { return item.key; }
function answers(m) { return saved[m] = saved[m] || {}; }
function persist() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) { $("status").textContent = "Không lưu được, hãy xuất sớm"; } }
function render() {
  const m = MODES[mode], item = m.items[index], a = answers(mode)[key(item)] || {};
  $("tabs").innerHTML = "";
  Object.keys(MODES).forEach(name => { const b = document.createElement("button"); const done = Object.values(answers(name)).filter(x => x.label).length;
    b.textContent = MODES[name].title + " (" + done + "/" + MODES[name].items.length + ")"; b.className = name === mode ? "on" : ""; b.onclick = () => { mode = name; index = 0; render(); }; $("tabs").appendChild(b); });
  $("question").textContent = m.question;
  $("count").textContent = "Mục " + (index + 1) + " / " + m.items.length + " · " + item.caption;
  $("view").innerHTML = item.images.map(src => '<img src="' + src + '">').join("");
  $("choices").innerHTML = "";
  m.choices.forEach(c => { const b = document.createElement("button"); b.textContent = c[0] + " · " + c[2]; b.className = a.label === c[1] ? "on" : ""; b.onclick = () => choose(c[1]); $("choices").appendChild(b); });
  $("note").value = a.note || "";
  $("bar").style.width = (100 * Object.values(answers(mode)).filter(x => x.label).length / m.items.length) + "%";
}
function choose(label) { const m = MODES[mode], item = m.items[index]; answers(mode)[key(item)] = {label, note: $("note").value}; persist(); if (index < m.items.length - 1) index++; render(); }
function move(step) { index = Math.max(0, Math.min(MODES[mode].items.length - 1, index + step)); render(); }
function nextOpen() { const m = MODES[mode]; for (let k = 1; k <= m.items.length; k++) { const j = (index + k) % m.items.length; if (!(answers(mode)[key(m.items[j])] || {}).label) { index = j; break; } } render(); }
$("prev").onclick = () => move(-1); $("next").onclick = () => move(1); $("nextOpen").onclick = nextOpen;
$("note").oninput = () => { const item = MODES[mode].items[index]; const a = answers(mode)[key(item)] || {}; answers(mode)[key(item)] = {label: a.label || "", note: $("note").value}; persist(); };
document.addEventListener("keydown", e => { if (e.target.tagName === "TEXTAREA") return; const c = MODES[mode].choices.find(x => x[0] === e.key);
  if (c) choose(c[1]); else if (e.key === "ArrowLeft") move(-1); else if (e.key === "ArrowRight") move(1); else if (e.key === "Tab") { e.preventDefault(); nextOpen(); } });
$("json").onclick = () => { const out = {sequence_id: DATA.sequence_id, frames: DATA.frames, tracks: answers("tracks"), pairs: answers("pairs")};
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], {type: "application/json"})); a.download = "tracking_track_qa_result.json"; a.click(); };
render();
</script></body></html>
"""


def observations(draft):
    tracks = defaultdict(list)
    for frame in draft["frames"]:
        if FIRST_FRAME <= frame["mot_frame_id"] <= LAST_FRAME:
            for item in frame["objects"]:
                tracks[item["track_id"]].append((frame["mot_frame_id"], item["bbox_xywh"], item["annotation_origin"], frame["image"]))
    for rows in tracks.values():
        rows.sort(key=lambda row: row[0])
    return tracks


def crop(image, box, height=170):
    left, top, width, box_height = box
    margin = 0.2 * max(width, box_height)
    x0, y0 = int(max(0, left - margin)), int(max(0, top - margin))
    x1, y1 = int(min(image.shape[1], left + width + margin)), int(min(image.shape[0], top + box_height + margin))
    patch = image[y0:y1, x0:x1].copy()
    cv2.rectangle(patch, (int(left - x0), int(top - y0)), (int(left + width - x0), int(top + box_height - y0)), (0, 255, 0), 2)
    scale = height / patch.shape[0]
    return cv2.resize(patch, (max(30, int(patch.shape[1] * scale)), height), interpolation=cv2.INTER_CUBIC)


def strip(rows, path):
    """Five crops of one track spread over its lifetime, labelled with the frame number."""
    chosen = rows if len(rows) <= STRIP_FRAMES else [rows[round(index * (len(rows) - 1) / (STRIP_FRAMES - 1))] for index in range(STRIP_FRAMES)]
    patches = []
    for frame, box, _, image_path in chosen:
        image = cv2.imread(str(DRAFT / image_path))
        patch = crop(image, box)
        cv2.putText(patch, f"f{frame}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        patches.append(patch)
        patches.append(np.full((patch.shape[0], 6, 3), 40, np.uint8))
    cv2.imwrite(str(path), np.hstack(patches[:-1]))


def center(box):
    return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])


def iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def candidate_pairs(tracks):
    found = {}
    ids = sorted(tracks)
    for first in ids:
        for second in ids:
            if first >= second:
                continue
            a, b = tracks[first], tracks[second]
            shared = {row[0] for row in a} & {row[0] for row in b}
            if shared:
                boxes_a = {row[0]: row[1] for row in a}
                boxes_b = {row[0]: row[1] for row in b}
                overlap = [iou(boxes_a[frame], boxes_b[frame]) for frame in sorted(shared)]
                if sum(value > 0.5 for value in overlap) >= 2:
                    found[(first, second)] = f"hai box chồng nhau ở {sum(value > 0.5 for value in overlap)} frame chung (có thể là box trùng)"
                continue
            early, late = (a, b) if a[-1][0] < b[0][0] else (b, a)
            gap = late[0][0] - early[-1][0]
            if gap > MAX_GAP_FRAMES:
                continue
            end, start = early[-1][1], late[0][1]
            distance = float(np.linalg.norm(center(end) - center(start)))
            ratio = (start[2] * start[3]) / (end[2] * end[3] + 1e-9)
            if distance <= 1.2 * max(end[2], end[3]) + 6 * gap and 0.4 <= ratio <= 2.5:
                found[tuple(sorted((first, second)))] = f"track kết thúc ở f{early[-1][0]}, track kia bắt đầu ở f{late[0][0]} (cách {gap} frame, lệch {distance:.0f}px)"
    for pair in KNOWN_PAIRS:
        if all(identity in tracks for identity in pair):
            found.setdefault(tuple(sorted(pair)), "bạn đã ghi nhận cặp này cần gộp")
    return found


def missing_candidates(draft, prediction_path, min_frames=5, uncovered_share=0.6, cover_iou=0.3):
    """Tracks of another detector that the draft does not cover (possible unlabeled vehicles)."""
    images = {frame["mot_frame_id"]: frame["image"] for frame in draft["frames"]}
    labeled = defaultdict(list)
    for frame in draft["frames"]:
        for item in frame["objects"]:
            labeled[frame["mot_frame_id"]].append(item["bbox_xywh"])
    tracks = defaultdict(list)
    for line in Path(prediction_path).read_text().splitlines():
        parts = line.split(",")
        frame = int(parts[0])
        if FIRST_FRAME <= frame <= LAST_FRAME:
            box = [float(v) for v in parts[2:6]]
            covered = any(iou(box, other) >= cover_iou for other in labeled[frame])
            tracks[int(parts[1])].append((frame, box, "OTHER_DETECTOR", images[frame], covered))
    found = {}
    for identity, rows in tracks.items():
        rows.sort(key=lambda row: row[0])
        uncovered = [row[:4] for row in rows if not row[4]]
        if len(rows) >= min_frames and len(uncovered) >= uncovered_share * len(rows):
            found[identity] = uncovered
    return found


def main():
    global DRAFT, OUTPUT, FIRST_FRAME, LAST_FRAME
    parser = argparse.ArgumentParser(description="Build the track-level QA page for a frame window of the draft")
    parser.add_argument("--first", type=int, default=FIRST_FRAME)
    parser.add_argument("--last", type=int, default=LAST_FRAME)
    parser.add_argument("--draft", type=Path, default=DRAFT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--missing-from", type=Path, help="predictions_mot.txt of another detector, to list possibly unlabeled vehicles")
    arguments = parser.parse_args()
    DRAFT, OUTPUT, FIRST_FRAME, LAST_FRAME = arguments.draft.resolve(), arguments.output.resolve(), arguments.first, arguments.last
    draft = json.loads((DRAFT / "annotations_draft.json").read_text(encoding="utf-8"))
    tracks = observations(draft)
    (OUTPUT / "strips").mkdir(parents=True, exist_ok=True)
    items = []
    for identity in sorted(tracks):
        rows = tracks[identity]
        strip(rows, OUTPUT / "strips" / f"track_{identity}.jpg")
        origin = "đề xuất của model" if rows[0][2] == "MODEL_PROPOSAL" else "nhãn có sẵn"
        items.append({"key": str(identity), "images": [f"strips/track_{identity}.jpg"],
                      "caption": f"ID {identity} · {origin} · f{rows[0][0]}–f{rows[-1][0]} ({len(rows)} frame)"})
    pairs = []
    for (first, second), reason in sorted(candidate_pairs(tracks).items()):
        pairs.append({"key": f"{first}-{second}", "images": [f"strips/track_{first}.jpg", f"strips/track_{second}.jpg"],
                      "caption": f"ID {first} (trên) và ID {second} (dưới): {reason}"})
    data = {"sequence_id": draft["sequence_id"], "frames": [FIRST_FRAME, LAST_FRAME], "tracks": items, "pairs": pairs,
            "draft": str(DRAFT)}
    if arguments.missing_from:
        missing = []
        for identity, rows in sorted(missing_candidates(draft, arguments.missing_from).items()):
            strip(rows, OUTPUT / "strips" / f"missing_{identity}.jpg")
            missing.append({"key": str(identity), "images": [f"strips/missing_{identity}.jpg"],
                            "caption": f"Track {identity} của detector khác · chưa có box trong bản nháp ở {len(rows)} frame (f{rows[0][0]}–f{rows[-1][0]})"})
        data["missing"] = missing
        data["missing_source"] = str(arguments.missing_from)
    (OUTPUT / "track_qa.html").write_text(PAGE.replace("__STYLE__", STYLE).replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print(f"{len(items)} tracks, {len(pairs)} candidate pairs, {len(data.get('missing', []))} missing candidates -> {OUTPUT / 'track_qa.html'}")


if __name__ == "__main__":
    main()
