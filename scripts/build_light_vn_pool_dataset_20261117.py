"""Export the owner-QA'd VN light pool (17/11/2026) as a YOLO dataset of ROUND lamps (classes red, yellow, green, off).

Only reviewed images. A box is a label when status is not wrong/unsure and shape == round. Arrow, countdown and
motorcycle-symbol lamps stay unlabelled on purpose (decision (a): round-lamp model first, arrows are step 2), so the
model learns they are not round lamps. 'unsure' boxes are blurred (no unlabelled-but-uncertain object is left).
Split by source video group (spec: clips of one source stay in one split; one TV channel = one group).
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SETS = ["vn_pool_qa_light_driving_20261117", "vn_pool_qa_light_20261117", "vn_pool_qa_light_orange_20261117"]
OUT = ROOT / "data/Light363_vn_pool_20261117"
GROUP = {"byQ3vbAiMAk": "ANTV", "hLys2ZhVsXg": "ANTV"}  # same channel: same group
VAL_GROUPS = {"6NBXnl_MtfE", "hRf_EQ6Zd3Q", "ANTV"}
# Model A often gave two boxes per head (housing + lit lamp) and both were kept. Same class: keep the larger box.
# Different classes were checked by eye (17/11): keep the right class, or drop an unreadable far-away image.
CONFLICTS = {"l-cQOa5K2X0_02948500": 0,  # red lamp lit; the green box also covered the green arrow below
             "byQ3vbAiMAk_00024960": 1,  # middle orange lamp lit: yellow; the lamp-only box said red
             "Qv_mxBpgqrk_04705000": None}  # ~7 px high horizontal heads, overlapping red/green boxes: drop image


def iou(a, b):
    i = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-12)


def dedupe(stem, boxes):
    """boxes: [(cls, [x1, y1, x2, y2])]. Returns the de-duplicated list, or None to drop the image."""
    area = lambda b: (b[1][2] - b[1][0]) * (b[1][3] - b[1][1])
    keep = []
    for b in sorted(boxes, key=area, reverse=True):
        clash = [k for k in keep if iou(k[1], b[1]) > 0.4]
        if not clash:
            keep.append(b)
        elif any(k[0] != b[0] for k in clash):
            if stem not in CONFLICTS:
                raise ValueError(f"{stem}: overlapping boxes of different classes, check by eye")
            if CONFLICTS[stem] is None:
                return None
            keep = [k for k in keep if k not in clash] + [max(clash + [b], key=lambda k: (k[0] == CONFLICTS[stem], area(k)))]
    return keep


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists; never overwrite a dataset version")
    stats, manifest = Counter(), []
    for s in SETS:
        for im in json.loads((ROOT / "data" / s / "qa_result.json").read_text(encoding="utf-8"))["images"]:
            if not im.get("reviewed"):
                continue
            src = ROOT / im["path"]
            vid = src.name[:-13]  # <video id>_<ms:08d>.jpg
            split = "val" if GROUP.get(vid, vid) in VAL_GROUPS else "train"
            img = cv2.imread(str(src))
            h, w = img.shape[:2]
            rounds, blurred, skipped = [], 0, Counter()
            for b in im["boxes"]:
                if b["group"] != "light" or b["status"] == "wrong":
                    continue
                x1, y1, x2, y2 = b["xyxy"]
                x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    raise ValueError(f"{src}: degenerate box {b['xyxy']}")
                if b["status"] == "unsure":
                    xa, ya, xb, yb = int(x1), int(y1), int(x2) + 1, int(y2) + 1
                    img[ya:yb, xa:xb] = cv2.GaussianBlur(img[ya:yb, xa:xb], (0, 0), sigmaX=max(3, (xb - xa) / 3))
                    blurred += 1
                    continue
                shape = b.get("shape")
                if shape is None:
                    raise ValueError(f"{src}: kept light box without shape tag")
                if shape != "round":
                    skipped[shape] += 1
                    continue
                rounds.append((b["cls"], [x1, y1, x2, y2]))
            kept = dedupe(src.stem, rounds)
            if kept is None:
                stats["dropped_unreadable_images"] += 1
                continue
            stats["duplicate_boxes_removed"] += len(rounds) - len(kept)
            rows = []
            for c, (x1, y1, x2, y2) in kept:
                rows.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                stats[f"{split}_{['red', 'yellow', 'green', 'off'][c]}"] += 1
            name = f"vnpool__{src.stem}"
            (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            stats[f"{split}_images"] += 1
            stats[f"{split}_negative_images"] += not rows
            manifest.append({"image": f"images/{split}/{name}.jpg", "source_frame": im["path"], "video": vid, "group": GROUP.get(vid, vid),
                             "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "blurred_unsure": blurred, "unlabelled_by_rule": dict(skipped)})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: 'red'\n  1: 'yellow'\n  2: 'green'\n  3: 'off'\n", encoding="utf-8")
    summary = {"status": "OWNER_QA_ROUND_LAMPS", "source": "data/vn_youtube_train_pool_20261116 (public YouTube, research use, not redistributed)",
               "label_rule": "round lamps only; arrows/countdowns/motorcycle lamps unlabelled by rule; orange = yellow",
               "val_groups": sorted(VAL_GROUPS), "never_in_test": True, "stats": dict(stats)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["stats"], indent=1))


if __name__ == "__main__":
    main()
