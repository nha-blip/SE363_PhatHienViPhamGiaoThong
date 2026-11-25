import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs/project363/vehicle_label_audit_20261103"
SETS = [
    {"id": "truck", "folder": "truck_called_car", "title": "Nhãn truck (MIO) mà model gọi là car", "tab": "Truck bị gọi là car",
     "question": "Box đỏ dày thật ra là gì?",
     "choices": [["1", "truck", "Truck thật"], ["2", "pickup_van", "Pickup / van / xe chở khách"], ["3", "car", "Là car (nhãn sai)"], ["4", "unclear", "Không rõ / quá nhỏ"]]},
    {"id": "motorcycle", "folder": "overlapping_motorcycle_labels", "title": "Nhãn xe máy chồng nhau", "tab": "Xe máy chồng nhau",
     "question": "Box đỏ dày và box vàng chồng nhau là gì?",
     "choices": [["1", "two_real", "Hai xe thật"], ["2", "rider_extra", "Một xe + người bị gắn nhãn thừa"], ["3", "duplicate", "Nhãn trùng"], ["4", "unclear", "Không rõ"]]},
]
VEHICLE_CHOICES = [["1", "car", "Car (xe con, SUV)"], ["2", "van_16", "Xe 16 chỗ / minibus"], ["3", "pickup", "Bán tải"], ["4", "cargo_van", "Van chở hàng"],
                   ["5", "truck", "Truck thật"], ["6", "unclear", "Không rõ / quá nhỏ"]]
SETS += [
    {"id": "pickup_sample", "folder": "pickup_truck_sample", "title": "Mẫu ngẫu nhiên nhãn pickup_truck (MIO)", "tab": "pickup_truck ngẫu nhiên",
     "question": "Box đỏ dày thật ra là gì?", "choices": VEHICLE_CHOICES},
    {"id": "work_van_sample", "folder": "work_van_sample", "title": "Mẫu ngẫu nhiên nhãn work_van (MIO)", "tab": "work_van ngẫu nhiên",
     "question": "Box đỏ dày thật ra là gì?", "choices": VEHICLE_CHOICES},
]
CARD = re.compile(r'<img src="(\d+\.jpg)"><p>(.*?)<br><small>(.*?)</small>', re.S)

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QA nhãn xe</title>
<style>
:root{--bg:#f6f6f4;--fg:#1b1b1b;--card:#fff;--line:#d8d8d2;--accent:#1f6feb;--muted:#6b6b66}
@media (prefers-color-scheme:dark){:root{--bg:#181818;--fg:#ececec;--card:#242424;--line:#3a3a3a;--accent:#5b9dff;--muted:#a0a0a0}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;z-index:2}
header h1{font-size:16px;margin:0 8px 0 0}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
main{max-width:1000px;margin:0 auto;padding:16px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0}
.bar i{display:block;height:100%;background:var(--accent)}
img{max-width:100%;border:1px solid var(--line);background:#000}
.choices{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.choices button{padding:10px 14px}
.muted{color:var(--muted);font-size:13px}
table{border-collapse:collapse;margin-top:8px}td,th{border:1px solid var(--line);padding:4px 10px;text-align:left}
textarea{width:100%;box-sizing:border-box;background:var(--card);color:inherit;border:1px solid var(--line);border-radius:6px;padding:6px;font:inherit}
</style></head><body>
<header><h1>QA nhãn xe</h1><span id="tabs"></span><span class="muted" id="status"></span>
<span style="margin-left:auto"><button id="csv">Xuất CSV</button> <button id="json">Xuất JSON</button></span></header>
<main>
<div class="bar"><i id="bar"></i></div>
<div id="count" class="muted"></div>
<h2 id="title" style="font-size:17px"></h2>
<img id="photo" alt="">
<p id="caption" class="muted"></p>
<p id="question"><b></b></p>
<div class="choices" id="choices"></div>
<textarea id="note" rows="2" placeholder="Ghi chú (không bắt buộc)"></textarea>
<p><button id="prev">← Trước</button> <button id="next">Sau →</button> <button id="nextOpen">Ảnh chưa đánh giá ⇥</button></p>
<p class="muted">Phím tắt: 1–4 chọn đáp án, ← → chuyển ảnh, Tab nhảy đến ảnh chưa đánh giá. Kết quả tự lưu trong trình duyệt; hãy bấm “Xuất” khi xong.</p>
<div id="summary"></div>
</main>
<script>
const SETS = __DATA__;
const KEY = "vehicle-label-qa-20261103";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let current = 0, index = 0;
const $ = id => document.getElementById(id);
function persist() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) { $("status").textContent = "Không lưu được trong trình duyệt, hãy xuất sớm"; } }
function answer(set, item) { return (saved[set.id] || {})[item.file] || {}; }
function doneCount(set) { return Object.values(saved[set.id] || {}).filter(x => x.label).length; }
function render() {
  const set = SETS[current], item = set.items[index], a = answer(set, item);
  $("tabs").innerHTML = "";
  SETS.forEach((s, i) => { const b = document.createElement("button"); b.textContent = s.tab + " (" + doneCount(s) + "/" + s.items.length + ")"; b.className = i === current ? "on" : ""; b.onclick = () => { current = i; index = 0; render(); }; $("tabs").appendChild(b); });
  $("title").textContent = set.title;
  const photo = $("photo");
  photo.onerror = () => { $("caption").textContent = "Không tải được ảnh " + set.folder + "/" + item.file + " — hãy mở file này từ thư mục chứa các thư mục ảnh."; };
  photo.removeAttribute("src");
  photo.src = set.folder + "/" + item.file + "?v=" + current + "_" + index;
  $("caption").textContent = "Đang xem " + set.folder + "/" + item.file + " · " + item.caption + " · " + item.name;
  $("question").firstChild.textContent = set.question;
  $("count").textContent = "Ảnh " + (index + 1) + " / " + set.items.length;
  const done = doneCount(set);
  $("bar").style.width = (100 * done / set.items.length) + "%";
  $("choices").innerHTML = "";
  set.choices.forEach(c => { const b = document.createElement("button"); b.textContent = c[0] + " · " + c[2]; b.className = a.label === c[1] ? "on" : ""; b.onclick = () => choose(c[1]); $("choices").appendChild(b); });
  $("note").value = a.note || "";
  const counts = {};
  Object.values(saved[set.id] || {}).forEach(x => { if (x.label) counts[x.label] = (counts[x.label] || 0) + 1; });
  $("summary").innerHTML = "<table><tr><th>Đáp án</th><th>Số ảnh</th><th>Tỉ lệ</th></tr>" + set.choices.map(c => "<tr><td>" + c[2] + "</td><td>" + (counts[c[1]] || 0) + "</td><td>" + (done ? (100 * (counts[c[1]] || 0) / done).toFixed(0) + "%" : "–") + "</td></tr>").join("") + "</table>";
}
function choose(label) {
  const set = SETS[current], item = set.items[index];
  (saved[set.id] = saved[set.id] || {})[item.file] = { label, note: $("note").value };
  persist();
  if (index < set.items.length - 1) index++;
  render();
}
function move(step) { index = Math.max(0, Math.min(SETS[current].items.length - 1, index + step)); render(); }
function nextOpen() { const set = SETS[current]; for (let k = 1; k <= set.items.length; k++) { const j = (index + k) % set.items.length; if (!answer(set, set.items[j]).label) { index = j; break; } } render(); }
$("prev").onclick = () => move(-1); $("next").onclick = () => move(1); $("nextOpen").onclick = nextOpen;
$("note").oninput = () => { const set = SETS[current], item = set.items[index]; const a = answer(set, item); (saved[set.id] = saved[set.id] || {})[item.file] = { label: a.label || "", note: $("note").value }; persist(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  const set = SETS[current], c = set.choices.find(x => x[0] === e.key);
  if (c) choose(c[1]); else if (e.key === "ArrowLeft") move(-1); else if (e.key === "ArrowRight") move(1); else if (e.key === "Tab") { e.preventDefault(); nextOpen(); }
});
function download(name, text, type) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); }
function rows() { const out = []; SETS.forEach(s => s.items.forEach(i => { const a = answer(s, i); out.push({ set: s.id, file: s.folder + "/" + i.file, image: i.name, label: a.label || "", note: a.note || "" }); })); return out; }
$("json").onclick = () => download("label_qa_result.json", JSON.stringify(rows(), null, 2), "application/json");
$("csv").onclick = () => download("label_qa_result.csv", "\\ufeffset,file,image,label,note\\n" + rows().map(r => [r.set, r.file, r.image, r.label, '"' + r.note.replace(/"/g, '""') + '"'].join(",")).join("\\n"), "text/csv");
render();
</script></body></html>
"""


def main():
    data = []
    for spec in SETS:
        page = (AUDIT / spec["folder"] / "index.html").read_text(encoding="utf-8")
        items = [{"file": file, "caption": re.sub(r"<[^>]+>", " ", html.unescape(caption)).strip(), "name": html.unescape(name)} for file, caption, name in CARD.findall(page)]
        data.append({**spec, "items": items})
    (AUDIT / "qa_app.html").write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print({entry["id"]: len(entry["items"]) for entry in data}, AUDIT / "qa_app.html")


if __name__ == "__main__":
    main()
