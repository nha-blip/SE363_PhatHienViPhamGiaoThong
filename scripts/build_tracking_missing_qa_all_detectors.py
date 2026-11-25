"""Missing-vehicle QA candidates from every detector, so no detector is favoured by the GT search.

For each detector, tracks in the window not covered by the current draft (see
build_tracking_track_qa.missing_candidates) are collected. Candidates of different detectors that
follow the same object (IoU >= 0.5 on >= half of their shared frames) are grouped; the longest one
is shown. YOLOv8s tracks already answered in the 231-300 QA are skipped.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tracking_track_qa as builder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/Tracking363_working/annotation_drafts"
DRAFT = BASE / "vntraffic_001_mot201_300_after_track_qa_231_300"
OUTPUT = BASE / "vntraffic_001_mot201_300_missing_qa_all_detectors"
PREVIOUS = BASE / "vntraffic_001_mot231_300_track_qa/tracking_track_qa_result.json"
TRACKING = ROOT / "runs/project363/tracking"
DETECTORS = {"e30": "vehicle_epoch30_vntraffic_001", "e41_960": "vehicle_epoch41_imgsz960_vntraffic_001",
             "e63_960": "vehicle_epoch63_imgsz960_vntraffic_001", "v8s_960": "vehicle_v8s_imgsz960_vntraffic_001"}
FIRST, LAST = 201, 300


def same_object(a, b):
    boxes_a, boxes_b = {r[0]: r[1] for r in a}, {r[0]: r[1] for r in b}
    shared = set(boxes_a) & set(boxes_b)
    if not shared:
        return False
    return sum(builder.iou(boxes_a[f], boxes_b[f]) >= 0.5 for f in shared) >= 0.5 * len(shared)


def main():
    builder.DRAFT, builder.FIRST_FRAME, builder.LAST_FRAME = DRAFT, FIRST, LAST
    draft = json.loads((DRAFT / "annotations_draft.json").read_text(encoding="utf-8"))
    answered_v8s = set(json.loads(PREVIOUS.read_text(encoding="utf-8"))["missing"])
    candidates = []
    for name, folder in DETECTORS.items():
        for identity, rows in builder.missing_candidates(draft, TRACKING / folder / "predictions_mot.txt").items():
            if name == "v8s_960" and str(identity) in answered_v8s:
                continue
            candidates.append({"detector": name, "id": identity, "rows": rows})
    groups = []
    for cand in sorted(candidates, key=lambda c: -len(c["rows"])):
        for group in groups:
            if any(same_object(cand["rows"], other["rows"]) for other in group):
                group.append(cand)
                break
        else:
            groups.append([cand])
    (OUTPUT / "strips").mkdir(parents=True, exist_ok=True)
    items = []
    for group in sorted(groups, key=lambda g: g[0]["rows"][0][0]):
        lead = group[0]
        key = f"{lead['detector']}:{lead['id']}"
        path = f"strips/missing_{lead['detector']}_{lead['id']}.jpg"
        builder.strip(lead["rows"], OUTPUT / path)
        found_by = sorted({c["detector"] for c in group})
        items.append({"key": key, "images": [path], "members": [f"{c['detector']}:{c['id']}" for c in group],
                      "caption": f"Track {lead['id']} của {lead['detector']} · chưa có box ở {len(lead['rows'])} frame "
                                 f"(f{lead['rows'][0][0]}–f{lead['rows'][-1][0]}) · detector thấy: {', '.join(found_by)}"})
    data = {"sequence_id": draft["sequence_id"], "frames": [FIRST, LAST], "missing": items, "draft": str(DRAFT),
            "detectors": DETECTORS}
    page = builder.PAGE.replace("__STYLE__", builder.STYLE).replace("__DATA__", json.dumps(data, ensure_ascii=False))
    (OUTPUT / "track_qa.html").write_text(page, encoding="utf-8")
    by = {}
    for item in items:
        for d in {m.split(":")[0] for m in item["members"]}:
            by[d] = by.get(d, 0) + 1
    print(f"{len(candidates)} candidate tracks -> {len(items)} objects to review; per detector: {by}")


if __name__ == "__main__":
    main()
