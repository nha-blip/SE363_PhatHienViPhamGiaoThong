"""6-class taxonomy: QA items for the truck/bus boxes of a 4-class owner-QA'd set (default: VN pool training frames).

Under the 17/11 rule, 16-seat minibuses, vans and pickups were all labelled truck. Every kept truck box (status not
'wrong') becomes one QA item; the suggestion is 'minibus' when the original model called it bus (the 49 16-seat/van
boxes the owner moved bus -> truck), else 'truck'. With --with-bus, bus boxes are items too (suggestion 'bus').
Writes <out>/items.json and <out>/crops/<id>.jpg.
Usage: vn_pool_truck_qa_items_20261125.py <out dir> [--qa-dir data/<set>] [--with-bus] [--prefix v]
26/11/2026: also used for the 77 VN camera test images (data/external_qa_hanoi_20261113, --with-bus).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/vn_pool_qa_vehicle_20261117"


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out"); parser.add_argument("--qa-dir", default=str(QA)); parser.add_argument("--with-bus", action="store_true")
    parser.add_argument("--prefix", default="v")
    args = parser.parse_args()
    out, qa_dir, classes = Path(args.out), ROOT / args.qa_dir, ({2, 3} if args.with_bus else {3})
    if out.exists():
        raise FileExistsError(out)
    (out / "crops").mkdir(parents=True)
    pre = json.loads((qa_dir / "prelabels.json").read_text(encoding="utf-8"))
    items, stats = [], Counter()
    for im in json.loads((qa_dir / "qa_result.json").read_text(encoding="utf-8"))["images"]:
        model = [b for b in pre.get(im["path"], []) if b["group"] == "vehicle"] if isinstance(pre, dict) else []
        img = None
        for k, b in enumerate(im["boxes"]):
            if b["group"] != "vehicle" or b["status"] == "wrong" or b["cls"] not in classes:
                continue
            m = max(model, key=lambda p: iou(p["xyxy"], b["xyxy"]), default=None)
            was = m["cls"] if m is not None and iou(m["xyxy"], b["xyxy"]) > 0.9 else None
            stats[f"model_{was}"] += 1
            path = Path(im["path"].replace("\\", "/"))
            img = cv2.imread(str(path if path.is_absolute() else ROOT / path)) if img is None else img
            x1, y1, x2, y2 = b["xyxy"]; pw, ph = (x2 - x1) * 0.35, (y2 - y1) * 0.35
            H, W = img.shape[:2]
            cx1, cy1, cx2, cy2 = int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(W, x2 + pw)), int(min(H, y2 + ph))
            view = img[cy1:cy2, cx1:cx2].copy()
            cv2.rectangle(view, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (0, 200, 255), 2)
            s = 360 / max(view.shape[:2])
            if s < 1:
                view = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cid = f"{args.prefix}{len(items):03d}"
            cv2.imwrite(str(out / "crops" / f"{cid}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 88])
            items.append({"id": cid, "split": "VN", "image": im["path"], "box_index": k, "box": b["xyxy"],
                          "clip": "bus" if b["cls"] == 2 else ("minibus" if was == 2 else "truck"), "conf": "", "model_cls": was})
    (out / "items.json").write_text(json.dumps(items, indent=1))
    print(len(items), dict(stats))


if __name__ == "__main__":
    assert iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    main()
