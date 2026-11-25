"""Yellow-only addition for model A's data (owner decision 21/11/2026): data/Light363_yellow_add_20261121 (train only).

Only images that contain a yellow lamp AND no lit head taught as background:
- DualCam train (data/Light363_dualcam_20261118): images with >=1 Yellow and no Green-up/left/right or Red-Yellow head
  (those heads were blurred there). Countdown displays stay blurred (kept by owner decision: not a lamp head).
- VN pool train (data/Light363_vn_pool_20261117): images with >=1 yellow and no arrow/countdown left unlabelled by rule.
Red/green/off boxes in the chosen images keep their labels. Val sets are unchanged (A val for selection; DualCam/VN vals
for reporting) and come from the existing inputs.
"""
import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DC = ROOT / "data/Light363_dualcam_20261118"
DC_XML = ROOT / "data/external_light_sources_20261116/dualcam_download/x/train/xml"
VN = ROOT / "data/Light363_vn_pool_20261117"
OUT = ROOT / "data/Light363_yellow_add_20261121"
HEAD_NEGATIVES = {"Green-up", "Green-left", "Green-right", "Red-Yellow"}


def has_yellow(label):
    return any(l.split()[0] == "1" for l in label.read_text().splitlines() if l.strip())


def main():
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists; never overwrite a dataset version")
    for sub in ("images/train", "labels/train"):
        (OUT / sub).mkdir(parents=True)
    stats, manifest = Counter(), []
    picks = []
    for label in sorted((DC / "labels/train").glob("*.txt")):
        names = set(re.findall(r"<name>([^<]*)</name>", (DC_XML / f"{label.stem.split('__')[-1]}.xml").read_text(encoding="utf-8", errors="replace")))
        if has_yellow(label) and not names & HEAD_NEGATIVES:
            picks.append(("dualcam", DC, label))
    vn_meta = {Path(e["image"]).stem: e for e in json.loads((VN / "manifest.json").read_text(encoding="utf-8"))}
    for label in sorted((VN / "labels/train").glob("*.txt")):
        if has_yellow(label) and not vn_meta[label.stem]["unlabelled_by_rule"]:
            picks.append(("vnpool", VN, label))
        elif has_yellow(label):
            stats["vnpool_skipped_unlabelled_heads"] += 1
    for source, root, label in picks:
        image = next((root / "images/train").glob(f"{label.stem}.*"))
        shutil.copy2(image, OUT / "images/train" / image.name)
        shutil.copy2(label, OUT / "labels/train" / label.name)
        cls = Counter(l.split()[0] for l in label.read_text().splitlines() if l.strip())
        for k, n in cls.items():
            stats[f"{source}_{['red', 'yellow', 'green', 'off'][int(k)]}"] += n
        stats[f"{source}_images"] += 1
        manifest.append({"image": image.name, "source": source})
    assert stats["dualcam_images"] and stats["vnpool_images"], stats
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"purpose": "yellow-only addition to model A data", "stats": dict(sorted(stats.items())),
                                                  "rule": "yellow present, no arrow/red-yellow head blurred or unlabelled"}, indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
