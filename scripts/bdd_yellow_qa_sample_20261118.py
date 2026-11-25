"""Sample BDD100K 'yellow' boxes for owner QA, to measure label noise per colour-classifier verdict (18/11/2026).

Each QA image is a zoomed crop around ONE BDD yellow box (context 3x the box, upscaled so the long side is 480 px) with that
box pre-labelled yellow. Owner: Enter = really a round yellow lamp; x = not a yellow round lamp (red, green, arrow,
pedestrian, unreadable); keys 5-8 may relabel. Strata come from the classifier read (scratchpad bdd_yellow_cls.json, train split).
Output: data/bdd_yellow_qa_20261118 (images, prelabels.json, meta.json with the stratum and source box of every crop).
"""
import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/Light363_bdd100k_yellow_20261116"
OUT = ROOT / "data/bdd_yellow_qa_20261118"
PLAN = {"red": 70, "none": 50, "green": 20, "lowconf": 20, "yellow": 40}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    reads = json.loads(Path(sys.argv[1]).read_text())["train"]
    rng = random.Random(363)
    strata = {}
    for r in reads:
        key = r["pred"] if r["conf"] >= 0.5 and r["pred"] != "off" else "lowconf"
        strata.setdefault(key, []).append(r)
    (OUT / "images").mkdir(parents=True)
    pre, meta = {}, []
    for key, n in PLAN.items():
        for r in rng.sample(strata[key], min(n, len(strata[key]))):
            line = (SRC / "labels/train" / f"{r['image']}.txt").read_text().splitlines()[r["box"]]
            _, x, y, w, h = map(float, line.split())
            img = cv2.imread(str(next((SRC / "images/train").glob(r["image"] + ".*"))))
            H, W = img.shape[:2]
            bw, bh = w * W, h * H
            half = max(bw, bh) * 1.5 + 8
            cx, cy = x * W, y * H
            x1, y1, x2, y2 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half))
            s = 480 / max(x2 - x1, y2 - y1)
            crop = cv2.resize(img[y1:y2, x1:x2], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            name = f"{len(meta):03d}_{r['image']}_{r['box']}.jpg"
            cv2.imwrite(str(OUT / "images" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            box = [round((cx - bw / 2 - x1) * s, 1), round((cy - bh / 2 - y1) * s, 1), round((cx + bw / 2 - x1) * s, 1), round((cy + bh / 2 - y1) * s, 1)]
            pre[str(Path("data/bdd_yellow_qa_20261118/images") / name)] = [{"group": "light", "xyxy": box, "cls": 1, "status": "pending", "from": f"bdd:{key}"}]
            meta.append({"crop": name, "stratum": key, "image": r["image"], "box": r["box"], "cls_pred": r["pred"], "cls_conf": r["conf"], "box_h_px": round(bh, 1)})
    # shuffle display order so the owner does not see the strata in blocks
    items = list(pre.items())
    rng.shuffle(items)
    (OUT / "prelabels.json").write_text(json.dumps(dict(items), indent=1), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps({"purpose": "BDD yellow label-noise QA", "plan": PLAN, "crops": meta}, indent=1), encoding="utf-8")
    (OUT / "Mo_app_QA.bat").write_text("@echo off\r\ncd /d D:\\se363-bigdata\r\nC:\\Users\\NHA\\miniconda3\\envs\\my_env\\python.exe scripts\\external_qa_app_20261112.py data\\bdd_yellow_qa_20261118\r\npause\r\n", encoding="utf-8")
    print(len(meta), "crops")


if __name__ == "__main__":
    main()
