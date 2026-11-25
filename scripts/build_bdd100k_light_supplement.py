"""BDD100K traffic-light supplement (focus: yellow) in the TrafficLight363 YOLO schema.

Classes: 0 red, 1 yellow, 2 green, 3 off. BDD100K det_20 labels: category "traffic light" with
attributes.trafficLightColor in {R, Y, G, NA}.
* Selected images: every image with at least one yellow light (train split -> train, val split -> a
  separate yellow evaluation set, never used for training).
* NA lights (side/back views, colour not visible) are NOT left unlabelled in the image: their boxes are
  blurred out (box + 2 px), recorded in the manifest (spec §15.2: no unlabelled visible objects).
* Images are copied unchanged otherwise (1280x720); boxes are not clipped silently: invalid ones abort.
License: BDD100K, educational / research / not-for-profit use (see manifest).
"""
import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/external_light_sources_20261116"
OUT = ROOT / "data/Light363_bdd100k_yellow_20261116"
COLOUR = {"R": 0, "Y": 1, "G": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=SRC / "bdd100k_images/archive.zip")
    a = parser.parse_args()
    images = zipfile.ZipFile(a.archive)
    names = {Path(n).name: n for n in images.namelist() if n.endswith(".jpg") and "/100k/" in n}
    manifest, stats = [], Counter()
    for split, label_file, target in (("train", "det_train.json.zip", "train"), ("val", "det_val.json.zip", "yellow_eval")):
        z = zipfile.ZipFile(SRC / "bdd100k_labels" / label_file)
        records = json.load(z.open(z.namelist()[0]))
        (OUT / "images" / target).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / target).mkdir(parents=True, exist_ok=True)
        for rec in records:
            lights = [l for l in rec.get("labels") or [] if l["category"] == "traffic light"]
            if not any((l.get("attributes") or {}).get("trafficLightColor") == "Y" for l in lights):
                continue
            member = names.get(rec["name"])
            if member is None:
                stats[f"{split}_image_missing"] += 1
                continue
            raw = images.read(member)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            rows, masked = [], []
            for l in lights:
                b = l["box2d"]
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    raise ValueError(f"{rec['name']}: box outside the image {b}")
                colour = (l.get("attributes") or {}).get("trafficLightColor")
                if colour in COLOUR:
                    rows.append(f"{COLOUR[colour]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                    stats[f"{target}_{colour}"] += 1
                else:  # colour not visible: blur instead of leaving an unlabelled light
                    xa, ya, xb, yb = int(max(0, x1 - 2)), int(max(0, y1 - 2)), int(min(w, x2 + 3)), int(min(h, y2 + 3))
                    image[ya:yb, xa:xb] = cv2.GaussianBlur(image[ya:yb, xa:xb], (0, 0), sigmaX=max(3, (xb - xa) / 3))
                    masked.append([xa, ya, xb, yb])
                    stats[f"{target}_NA_masked"] += 1
            stem = f"bdd__{Path(rec['name']).stem}"
            if masked:
                ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                data = buf.tobytes()
            else:
                data = raw
            (OUT / "images" / target / f"{stem}.jpg").write_bytes(data)
            (OUT / "labels" / target / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            stats[f"{target}_images"] += 1
            manifest.append({"image": f"images/{target}/{stem}.jpg", "source": member, "source_sha256": hashlib.sha256(raw).hexdigest(),
                             "bdd_split": split, "timeofday": (rec.get("attributes") or {}).get("timeofday"),
                             "weather": (rec.get("attributes") or {}).get("weather"), "na_boxes_blurred": masked})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/yellow_eval\nnames:\n  0: red\n  1: yellow\n  2: green\n  3: off\n", encoding="utf-8")
    summary = {"status": "CANDIDATE_NOT_MERGED_NOT_TRAINED", "source": "BDD100K det_20 (Kaggle mirrors alvaromalfaro/bdd100k images, ronanjanet/bdd100k-labels)",
               "license": "BDD100K: educational, research and not-for-profit use", "selection": "images containing >=1 yellow light",
               "na_policy": "NA lights blurred (box + 2 px), not left unlabelled", "yellow_eval_is_holdout": True, "stats": dict(stats)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["stats"], indent=1))


if __name__ == "__main__":
    main()
