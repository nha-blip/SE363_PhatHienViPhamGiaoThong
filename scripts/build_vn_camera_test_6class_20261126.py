"""VN camera test set (data/external_qa_hanoi_20261113, never in training) in the 6-class taxonomy: data/vn_camera_test_6class_20261126.

Images flagged skip (owner: close-up zooms) are left out, as in the 13/11 scoring. GT = vehicle boxes not marked wrong; truck
and bus boxes take the owner's 26/11 verdict (data/vn_camera_test_truckbus_qa_20261126/items_labeled.json). Boxes marked
unsure or judged 'drop' are not GT (a prediction on them then counts against precision, never against recall).
"""
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/external_qa_hanoi_20261113"
QA = ROOT / "data/vn_camera_test_truckbus_qa_20261126/items_labeled.json"
OUT = ROOT / "data/vn_camera_test_6class_20261126"
NAMES = ["car", "motorcycle", "bus", "truck", "pickup", "minibus"]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    (OUT / "images").mkdir(parents=True); (OUT / "labels").mkdir()
    verdict = {(it["image"], it["box_index"]): it["label"] for it in json.loads(QA.read_text())}
    stats = Counter()
    for im in json.loads((SRC / "qa_result.json").read_text(encoding="utf-8"))["images"]:
        if im.get("skip"):
            stats["skipped_images"] += 1
            continue
        src = SRC / "images" / Path(im["path"]).name
        w, h = __import__("PIL.Image", fromlist=["Image"]).open(src).size
        lines = []
        for k, b in enumerate(im["boxes"]):
            if b["group"] != "vehicle" or b["status"] in ("wrong", "unsure"):
                stats[f"not_gt_{b['status']}"] += b["group"] == "vehicle"
                continue
            name = verdict.get((im["path"], k), NAMES[b["cls"]]) if b["cls"] in (2, 3) else NAMES[b["cls"]]
            if b["cls"] in (2, 3) and (im["path"], k) not in verdict:
                raise RuntimeError(f"truck/bus box without verdict: {im['path']} #{k}")
            if name == "drop":
                stats["not_gt_drop"] += 1
                continue
            x1, y1, x2, y2 = max(0, b["xyxy"][0]), max(0, b["xyxy"][1]), min(w, b["xyxy"][2]), min(h, b["xyxy"][3])
            lines.append(f"{NAMES.index(name)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            stats[name] += 1
        shutil.copy2(src, OUT / "images" / src.name)
        (OUT / "labels" / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
    (OUT / "summary.json").write_text(json.dumps({"classes": NAMES, "stats": dict(sorted(stats.items())), "rule": __doc__.split("\n\n")[1]}, indent=1), encoding="utf-8")
    print(dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
