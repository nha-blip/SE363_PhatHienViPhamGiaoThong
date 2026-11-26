"""6-class version of the VN pool vehicle dataset: data/Vehicle363_vn_pool_6class_20261125.

Starts from data/Vehicle363_vn_pool_20261118 (4-class, owner QA 17/11) and applies the owner's 25/11 verdicts on every
truck box (data/vn_pool_truck_qa_20261125/items_labeled.json): truck -> 3, pickup -> 4, minibus -> 5, drop -> the box is
blurred in the image and its label removed (spec §15.2: an unlabelled visible vehicle must not become background).
car / motorcycle / bus are unchanged; same images, same train/val split by video. Never contains test sources.
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/Vehicle363_vn_pool_20261118"
QA = ROOT / "data/vn_pool_truck_qa_20261125/items_labeled.json"
OUT = ROOT / "data/Vehicle363_vn_pool_6class_20261125"
NAMES = ["car", "motorcycle", "bus", "truck", "pickup", "minibus"]
VERDICT = {"truck": 3, "pickup": 4, "minibus": 5}


def to_xyxy(line, W, H):
    _, x, y, w, h = map(float, line.split()[:5])
    return [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]


def same_box(a, b, W, H):
    """The 18/11 export clipped boxes to the image; compare with 1.5 px tolerance after clipping the QA box."""
    b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
    return max(abs(p - q) for p, q in zip(a, b)) <= 1.5


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    verdicts = {}
    for it in json.loads(QA.read_text()):
        verdicts.setdefault("vnpool__" + Path(it["image"].replace("\\", "/")).stem, []).append((it["box"], it["label"]))
    stats = Counter()
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True); (OUT / "labels" / split).mkdir(parents=True)
        for label in sorted((SRC / "labels" / split).glob("*.txt")):
            image = next((SRC / "images" / split).glob(label.stem + ".*"))
            img = cv2.imread(str(image)); H, W = img.shape[:2]
            todo, out, blurred = list(verdicts.get(label.stem, [])), [], False
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                c = int(line.split()[0])
                if c == 3:
                    box = to_xyxy(line, W, H)
                    hit = [v for v in todo if same_box(box, v[0], W, H)]
                    assert len(hit) == 1, (label.name, box, len(hit))
                    todo.remove(hit[0]); verdict = hit[0][1]
                    stats[f"{split}_truck->{verdict}"] += 1
                    if verdict == "drop":
                        x1, y1, x2, y2 = (int(max(0, box[0] - 2)), int(max(0, box[1] - 2)), int(min(W, box[2] + 2)), int(min(H, box[3] + 2)))
                        img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (0, 0), max(3.0, (y2 - y1) / 4)); blurred = True
                        continue
                    line = " ".join([str(VERDICT[verdict])] + line.split()[1:])
                out.append(line)
                stats[f"{split}_{NAMES[int(line.split()[0])]}"] += 1
            assert not todo, (label.name, todo)
            if blurred:
                cv2.imwrite(str(OUT / "images" / split / f"{label.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                shutil.copy2(image, OUT / "images" / split / image.name)
            (OUT / "labels" / split / label.name).write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
            stats[f"{split}_images"] += 1
    (OUT / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(NAMES)), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"source": [str(SRC.relative_to(ROOT)), str(QA.relative_to(ROOT))], "classes": NAMES,
                                                  "stats": dict(sorted(stats.items())), "rules": __doc__.split("\n\n")[1]}, indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    assert same_box([0, 0, 10, 10], [-3, 0.5, 10, 11], 10, 10) and not same_box([0, 0, 10, 10], [5, 0, 10, 10], 10, 10)
    main()
