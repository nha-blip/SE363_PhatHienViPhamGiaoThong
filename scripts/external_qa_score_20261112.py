"""Score every model against the human-QA'd Bellevue labels.

GT = QA boxes not marked wrong; 'unsure' boxes are ignore regions (a prediction matching one is
neither TP nor FP). Matching: same class, greedy by IoU, conf >= 0.25.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "data/external_qa_bellevue_20261112"


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def score(qa, raw, group, model, thr, conf=0.25, min_h=0):
    stats = defaultdict(lambda: {"gt": 0, "tp": 0, "fp": 0})
    for im in qa["images"]:
        if im.get("skip"):
            continue
        gts = [b for b in im["boxes"] if b["group"] == group and b["status"] != "wrong"]
        preds = [p for p in raw[im["path"]][group][model] if p["conf"] >= conf]
        used_g, used_p = set(), set()
        pairs = sorted(((iou(p["xyxy"], g["xyxy"]), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)
                        if p["cls"] == g["cls"]), reverse=True)
        for v, i, j in pairs:
            if v >= thr and i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
        # predictions overlapping an unsure box, or any GT of another class (wrong class), stay counted as FP
        # except unsure regions, which are ignored
        for j, g in enumerate(gts):
            if g["status"] == "unsure" or g["xyxy"][3] - g["xyxy"][1] < min_h:
                continue
            stats[g["cls"]]["gt"] += 1
            stats[g["cls"]]["tp"] += j in used_g
        for i, p in enumerate(preds):
            if i in used_p:
                g = gts[[j for v, a, j in pairs if a == i and j in used_g and v >= thr][0]]
                if g["status"] == "unsure" or g["xyxy"][3] - g["xyxy"][1] < min_h:
                    continue
            else:
                if any(iou(p["xyxy"], g["xyxy"]) >= thr and (g["status"] == "unsure" or g["xyxy"][3] - g["xyxy"][1] < min_h)
                       for g in gts):
                    continue
                if p["xyxy"][3] - p["xyxy"][1] < min_h:
                    continue
            stats[p["cls"]]["fp"] += i not in used_p
    return stats


def main():
    qa = json.loads((WORK / "qa_result.json").read_text(encoding="utf-8"))
    raw = json.loads((WORK / "predictions_raw.json").read_text(encoding="utf-8"))
    meta = json.loads((WORK / "meta.json").read_text(encoding="utf-8"))
    out = {}
    for group, models in meta["models"].items():
        names = meta["names"][group]
        for thr in ([0.5] if group == "vehicle" else [0.5, 0.3]):
            for min_h in [0, 32]:
                print(f"\n== {group}  IoU {thr}  box cao >= {min_h}px")
                for m in models:
                    st = score(qa, raw, group, m, thr, min_h=min_h)
                    row = []
                    T = {"gt": 0, "tp": 0, "fp": 0}
                    for c, n in enumerate(names):
                        s = st[c]
                        for k in T:
                            T[k] += s[k]
                        if s["gt"] or s["fp"]:
                            row.append(f"{n} R={100 * s['tp'] / max(s['gt'], 1):.0f}% ({s['tp']}/{s['gt']}) P={100 * s['tp'] / max(s['tp'] + s['fp'], 1):.0f}%")
                    row.append(f"TONG R={100 * T['tp'] / max(T['gt'], 1):.1f}% P={100 * T['tp'] / max(T['tp'] + T['fp'], 1):.1f}%")
                    print(f"  {m:6s} " + " | ".join(row))
                    out[f"{group}|{m}|iou{thr}|h{min_h}"] = {names[c]: dict(st[c]) for c in st}
    (WORK / "score_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
