"""DualCam traffic-light dataset (Sri Lanka, github.com/harinduravin/DualCam, ICMLA 2023) -> YOLO 4-class light dataset
data/Light363_dualcam_20261118 (red, yellow, green, off), same box convention as model A (whole head housing).

Class map: Red->red, Yellow->yellow, Green->green, Empty (all lamps dark)->off.
Not targets (decision (a), round lamps only): Red-Yellow (not used in VN), Green-up/left/right arrows, Count-down,
Empty-count-down -> blurred (box + 2 px), so the model is neither taught nor penalised on them.
Split: DualCam train -> train; test_1 + test_2 -> val (different drives; narrow/wide pairs stay together).
Licence: no licence file in the repo; README asks to cite the paper. Course research only, never redistributed.
Note: horizontal heads in Sri Lanka have red on the RIGHT (left-hand traffic): do not apply the VN horizontal3 position rule.
"""
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/external_light_sources_20261116/dualcam_download/x"
OUT = ROOT / "data/Light363_dualcam_20261118"
MAP = {"Red": 0, "Yellow": 1, "Green": 2, "Empty": 3}
BLUR = {"Red-Yellow", "Green-up", "Green-left", "Green-right", "Count-down", "Empty-count-down"}
SPLITS = {"train": "train", "test_1": "val", "test_2": "val"}
BOX = re.compile(r"<name>([^<]*)</name>.*?<xmin>([\d.]+)</xmin>\s*<ymin>([\d.]+)</ymin>\s*<xmax>([\d.]+)</xmax>\s*<ymax>([\d.]+)</ymax>", re.S)


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists; never overwrite a dataset version")
    stats, manifest = Counter(), []
    for folder, split in SPLITS.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for xml in sorted((SRC / folder / "xml").glob("*.xml")):
            img_path = SRC / folder / "images" / f"{xml.stem}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(img_path)
            h, w = img.shape[:2]
            lines, blurred = [], 0
            for name, *xyxy in BOX.findall(xml.read_text(encoding="utf-8", errors="replace")):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                if name in MAP:
                    lines.append(f"{MAP[name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                    stats[f"{split}_{name}"] += 1
                elif name in BLUR:
                    a, b, c, d = int(max(0, x1 - 2)), int(max(0, y1 - 2)), int(min(w, x2 + 2)), int(min(h, y2 + 2))
                    if c - a > 1 and d - b > 1:
                        img[b:d, a:c] = cv2.GaussianBlur(img[b:d, a:c], (0, 0), max(3.0, (d - b) / 4))
                    blurred += 1
                    stats[f"{split}_blurred_{name}"] += 1
                else:
                    raise ValueError(f"unknown class {name} in {xml}")
            stem = f"dualcam__{folder}__{xml.stem}"
            if blurred:
                cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                shutil.copy2(img_path, OUT / "images" / split / f"{stem}.jpg")
            (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            stats[f"{split}_images"] += 1
            manifest.append({"image": stem, "source": f"{folder}/{xml.stem}", "split": split, "boxes": len(lines), "blurred": blurred})
    (OUT / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: 'red'\n  1: 'yellow'\n  2: 'green'\n  3: 'off'\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"status": "CANDIDATE_NOT_TRAINED", "source": "DualCam (github.com/harinduravin/DualCam)",
                                                  "licence": "no licence file; cite ICMLA 2023 paper; course research only, private",
                                                  "class_map": MAP, "blurred": sorted(BLUR), "splits": SPLITS, "stats": dict(sorted(stats.items()))}, indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
