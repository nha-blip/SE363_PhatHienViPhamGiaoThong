"""Step 1 of the 6-class vehicle taxonomy (owner 21/11/2026): re-map MIO truck boxes by their original MIO class.

Vehicle363 merged MIO pickup_truck, work_van, articulated_truck and single_unit_truck into truck (3). The original
classes are in data/MIO_gt_download_20261103/gt_train.csv. Each Vehicle363 truck box of an MIO image is matched to the
original boxes of that image (IoU >= 0.9): pickup_truck -> 4 pickup, work_van -> 5 minibus, the rest stay 3 truck;
an unmatched truck box stays 3 and is counted. car/motorcycle/bus are unchanged. HCMC labels are copied unchanged
(their pickups/minibuses are step 2). Vehicle363 itself is never modified: new label files go to
data/Vehicle363_6class_labels_20261124/<split>/, beside a summary.
Usage: remap_mio_6class_20261124.py <Vehicle363 root containing images/<split> and labels/<split>>
25/11/2026: run on the full-frame copy data/YOLO363_training_working/Vehicle363-v1 (train/val/test) into
data/Vehicle363_6class_labels_fullframe_20261125; the 24/11 output from the crop archive is superseded.

The archive (v5nhdf/yolo363-vehicle-archive-20261027) keeps val as full frames (names = Vehicle363 final images) but its
train is the readable-crop set: `<source>__<audit key[:20]>__<focus index>` crops cut WITHOUT rescaling from frames
recorded in runs/project363/quality_filter_balanced_20261027/audit.sqlite (original pixel boxes per object). A crop's
offset = the focus object's original box minus the crop box of the same size; every crop box is shifted back by it
before matching the MIO originals.
"""
import csv
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/Vehicle363-v1/metadata/source_manifest.json"
GT = ROOT / "data/MIO_gt_download_20261103/gt_train.csv"
OUT = ROOT / "data/Vehicle363_6class_labels_fullframe_20261125"
NEW = {"pickup_truck": 4, "work_van": 5, "articulated_truck": 3, "single_unit_truck": 3}


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


AUDIT = ROOT / "runs/project363/quality_filter_balanced_20261027/audit.sqlite"


def crop_offset(lines, W, H, focus_box):
    """Shift (dx, dy) from crop to original pixels: the crop box whose size matches the focus object's original box."""
    fw, fh = focus_box[2] - focus_box[0], focus_box[3] - focus_box[1]
    best = None
    for line in lines:
        _, x, y, w, h = map(float, line.split()[:5])
        err = abs(w * W - fw) + abs(h * H - fh)
        if best is None or err < best[0]:
            best = (err, focus_box[0] - (x - w / 2) * W, focus_box[1] - (y - h / 2) * H)
    return (best[1], best[2]) if best and best[0] <= 3 else None


def remap_line(line, W, H, originals, offset=(0.0, 0.0)):
    """Return (new label line, outcome) for one YOLO line of an MIO image (offset: crop -> original pixels)."""
    p = line.split()
    c = int(p[0])
    if c != 3:
        return line, "kept"
    x, y, w, h = map(float, p[1:5])
    dx, dy = offset
    box = [(x - w / 2) * W + dx, (y - h / 2) * H + dy, (x + w / 2) * W + dx, (y + h / 2) * H + dy]
    best = max(((iou(box, b), name) for name, b in originals), default=(0.0, None))
    if best[0] < 0.9:
        return line, "truck_unmatched"
    return " ".join([str(NEW[best[1]])] + p[1:]), f"truck->{best[1]}"


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists; never overwrite a label version")
    src = Path(sys.argv[1])
    originals = defaultdict(list)
    with open(GT, newline="") as f:
        for img_id, cls, x1, y1, x2, y2 in csv.reader(f):
            if cls in NEW:
                originals[img_id].append((cls, [float(x1), float(y1), float(x2), float(y2)]))
    mio = {Path(e["final_image"]).stem: e["original_image_id"] for e in json.loads(MANIFEST.read_text(encoding="utf-8"))
           if e["source_dataset"] == "MIO"}
    audit = {}
    for key, record in sqlite3.connect(AUDIT).execute("select identity, record from images where kind='vehicle' and source='mio'"):
        audit[key[:20]] = record
    stats = Counter()
    for split in ("train", "val", "test"):
        (OUT / split).mkdir(parents=True)
        images = {p.stem: p for p in (src / "images" / split).iterdir()}  # one listing; a glob per label is O(n^2)
        for label in sorted((src / "labels" / split).glob("*.txt")):
            lines = [l for l in label.read_text().splitlines() if l.strip()]
            mio_id, offset = mio.get(label.stem), (0.0, 0.0)
            if mio_id is None and label.stem.startswith("mio__") and split == "train":
                _, key20, focus = label.stem.split("__")
                rec = json.loads(audit[key20])
                mio_id = re.search(r"train_(\d{8})_jpg", Path(rec["image"]).name).group(1)
                with Image.open(images[label.stem]) as im:
                    offset = crop_offset(lines, *im.size, rec["objects"][int(focus)]["box"])
                if offset is None:
                    stats[f"{split}_mio_crop_offset_failed"] += 1
                    mio_id = None
            if mio_id is not None:
                with Image.open(images[label.stem]) as im:
                    W, H = im.size
                out = []
                for l in lines:
                    new, outcome = remap_line(l, W, H, originals[mio_id], offset)
                    out.append(new); stats[f"{split}_mio_{outcome}"] += 1
                lines = out
            else:
                stats[f"{split}_hcmc_or_unmapped_files"] += 1
            for l in lines:
                stats[f"{split}_class_{l.split()[0]}"] += 1
            (OUT / split / label.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    summary = {"classes": ["car", "motorcycle", "bus", "truck", "pickup", "minibus"], "mio_map": NEW,
               "rule": "MIO truck boxes matched to gt_train.csv at IoU>=0.9; HCMC unchanged (step 2)", "stats": dict(sorted(stats.items()))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["stats"], indent=1))


if __name__ == "__main__":
    orig = [("pickup_truck", [10, 10, 50, 40]), ("work_van", [100, 100, 150, 140])]
    assert remap_line("3 0.15 0.125 0.2 0.15", 200, 200, orig)[0].startswith("4 ")
    assert remap_line("3 0.625 0.6 0.25 0.2", 200, 200, orig)[0].startswith("5 ")
    assert remap_line("0 0.15 0.125 0.2 0.15", 200, 200, orig)[1] == "kept"
    assert remap_line("3 0.9 0.9 0.05 0.05", 200, 200, orig)[1] == "truck_unmatched"
    # crop: the pickup box [10,10,50,40] seen at [0,0,40,30] inside a 100x100 crop -> offset (10,10) -> pickup
    off = crop_offset(["3 0.2 0.15 0.4 0.3"], 100, 100, [10, 10, 50, 40])
    assert off == (10.0, 10.0) and remap_line("3 0.2 0.15 0.4 0.3", 100, 100, orig, off)[0].startswith("4 ")
    main()
