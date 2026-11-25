"""Merge model A's light dataset (Light363_fullscene_20261028) and the owner-QA'd VN round lamps (Light363_vn_pool_20261117)
into one YOLO dataset: data/Light363_merged_20261117 (classes red, yellow, green, off; same order in both).

Splits are kept as they are (VN split by source video group). Image files are hard links (no extra disk), copied if a link
fails. train_balanced.txt = A's balanced list + every VN train image once (VN oversampling is a training choice).
val_a.txt / val_vn.txt allow reporting each val separately. The sources are never modified.
"""
import json
import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data/Light363_fullscene_20261028"
VN = ROOT / "data/Light363_vn_pool_20261117"
OUT = ROOT / "data/Light363_merged_20261117"
NAMES = ["red", "yellow", "green", "off"]


def place(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists; never overwrite a dataset version")
    stats, lists = Counter(), {"train": [], "val_a": [], "val_vn": []}
    for name, root in (("A", A), ("VN", VN)):
        for split in ("train", "val"):
            (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
            for img in sorted((root / "images" / split).iterdir()):
                label = root / "labels" / split / f"{img.stem}.txt"
                if not label.exists():
                    raise FileNotFoundError(label)
                dst = OUT / "images" / split / img.name
                if dst.exists():
                    raise FileExistsError(f"name collision {img.name}")
                place(img, dst)
                place(label, OUT / "labels" / split / label.name)
                for line in label.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        stats[f"{name}_{split}_{NAMES[int(line.split()[0])]}"] += 1
                stats[f"{name}_{split}_images"] += 1
                if split == "val":
                    lists["val_a" if name == "A" else "val_vn"].append(f"./images/val/{img.name}")
                elif name == "VN":
                    lists["train"].append(f"./images/train/{img.name}")
    balanced = [l for l in (A / "train_balanced.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    (OUT / "train_balanced.txt").write_text("\n".join(balanced + lists["train"]) + "\n", encoding="utf-8")
    for key in ("val_a", "val_vn"):
        (OUT / f"{key}.txt").write_text("\n".join(lists[key]) + "\n", encoding="utf-8")
    (OUT / "data.yaml").write_text("path: .\ntrain: train_balanced.txt\nval: images/val\nnames:\n  0: 'red'\n  1: 'yellow'\n  2: 'green'\n  3: 'off'\n", encoding="utf-8")
    manifest = [dict(e, dataset="A") for e in json.loads((A / "manifest.json").read_text(encoding="utf-8"))] + \
               [dict(e, dataset="VN") for e in json.loads((VN / "manifest.json").read_text(encoding="utf-8"))]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    summary = {"status": "MERGED", "sources": {"A": str(A.relative_to(ROOT)), "VN": str(VN.relative_to(ROOT))},
               "train_balanced_entries": len(balanced) + len(lists["train"]), "stats": dict(stats),
               "notes": ["VN: round lamps only, arrows/countdowns unlabelled by rule, orange = yellow",
                         "VN split by source video group; never contains test sources (iHanoi, news report)"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["stats"], indent=1), summary["train_balanced_entries"])


if __name__ == "__main__":
    main()
