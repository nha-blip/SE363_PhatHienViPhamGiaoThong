"""Option A (owner 25/11/2026): the Vehicle363 test images are lost, so the full-frame val split becomes dev + test.

HCMC val holds whole cameras, so HCMC is split by camera (no camera in both): the subset of cameras whose image count
and daytime count are both closest to half goes to test. MIO has no camera id and every val image is its own
de-duplicated atomic group, so MIO is split 50/50 by its sha256 (fixed, reproducible). dev is for model selection,
test is opened once after the configuration is fixed (spec §15.4).
Writes data/Vehicle363_6class_split_20261125/{dev.txt,test.txt,summary.json} (image stems of the val split).
"""
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/Vehicle363-v1/metadata/source_manifest.json"
LABELS = ROOT / "data/Vehicle363_6class_labels_20261124/val"
OUT = ROOT / "data/Vehicle363_6class_split_20261125"
NAMES = ["car", "motorcycle", "bus", "truck", "pickup", "minibus"]


def pick_test_cameras(cams):
    """cams: {camera: (images, daytime)} -> the camera set closest to half of both totals."""
    total = [sum(v[i] for v in cams.values()) for i in (0, 1)]
    best = None
    for r in range(1, len(cams)):
        for subset in combinations(sorted(cams), r):
            cost = sum(abs(sum(cams[c][i] for c in subset) / max(1, total[i]) - 0.5) for i in (0, 1))
            if best is None or cost < best[0]:
                best = (cost, set(subset))
    return best[1]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    val = [e for e in json.loads(MANIFEST.read_text(encoding="utf-8")) if e["final_split"] == "val"]
    cams = {}
    for e in val:
        if e["source_dataset"] == "HCMC":
            n, d = cams.get(e["camera_id"], (0, 0))
            cams[e["camera_id"]] = (n + 1, d + (e["daytime_nighttime"] == "daytime"))
    test_cams = pick_test_cameras(cams)
    split, stats = {"dev": [], "test": []}, Counter()
    for e in val:
        stem = Path(e["final_image"]).stem
        if e["source_dataset"] == "HCMC":
            s = "test" if e["camera_id"] in test_cams else "dev"
        else:
            s = "test" if int(hashlib.sha256(e["sha256"].encode()).hexdigest(), 16) % 2 else "dev"
        split[s].append(stem)
        stats[f"{s}_{e['source_dataset']}_images"] += 1
        for line in (LABELS / f"{stem}.txt").read_text().splitlines():
            stats[f"{s}_{NAMES[int(line.split()[0])]}"] += 1
    OUT.mkdir(parents=True)
    for s, stems in split.items():
        (OUT / f"{s}.txt").write_text("\n".join(sorted(stems)) + "\n", encoding="utf-8")
    summary = {"rule": __doc__.split("\n\n")[1], "hcmc_cameras": {c: {"images": v[0], "daytime": v[1], "split": "test" if c in test_cams else "dev"}
                                                                   for c, v in sorted(cams.items())}, "stats": dict(sorted(stats.items()))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    assert pick_test_cameras({"a": (10, 0), "b": (10, 10), "c": (20, 10)}) in ({"c"}, {"a", "b"})
    main()
