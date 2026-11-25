"""Provisional MOT metrics with and without the draft's ignore regions.

--min-height H: GT boxes shorter than H px become ignore regions (removed from GT), and unmatched
predictions shorter than H px are dropped, so only objects >= H px are scored.

With ignore: a prediction is dropped (neither TP nor FP) when, in that frame, it overlaps an ignore
box with IoU >= 0.5 and matches no GT box with IoU >= 0.5. GT is only removed by --min-height.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import evaluate_tracking_mot as ev

TRACKING = Path(__file__).resolve().parents[1] / "runs/project363/tracking"
DETECTORS = {"e30": "vehicle_epoch30_vntraffic_001", "e41_960": "vehicle_epoch41_imgsz960_vntraffic_001",
             "e63_960": "vehicle_epoch63_imgsz960_vntraffic_001", "v8s_960": "vehicle_v8s_imgsz960_vntraffic_001"}


MIN_HEIGHT = 0


def drop_ignored(gt, pred, ignore):
    out, dropped = defaultdict(dict), 0
    for frame, boxes in pred.items():
        ids = list(boxes)
        g = list(gt.get(frame, {}).values())
        ig = ignore.get(frame, [])
        to_gt = ev.iou_matrix([boxes[i] for i in ids], g) if g else None
        to_ig = ev.iou_matrix([boxes[i] for i in ids], ig) if ig else None
        for k, i in enumerate(ids):
            unmatched = to_gt is None or to_gt[k].max() < 0.5
            if unmatched and ((to_ig is not None and to_ig[k].max() >= 0.5) or boxes[i][3] < MIN_HEIGHT):
                dropped += 1
                continue
            out[frame][i] = boxes[i]
    return out, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--first", type=int, default=201)
    parser.add_argument("--last", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sequence", default="vntraffic_001", help="suffix of the runs/project363/tracking/<detector>_<sequence> folders")
    parser.add_argument("--min-height", type=float, default=0)
    a = parser.parse_args()
    global MIN_HEIGHT
    MIN_HEIGHT = a.min_height
    gt = ev.load_gt(a.gt, a.first, a.last)
    ignore = defaultdict(list)
    if a.min_height:
        for frame in list(gt):
            for identity, box in list(gt[frame].items()):
                if box[3] < a.min_height:
                    ignore[frame].append(box)
                    del gt[frame][identity]
    for r in json.loads(a.gt.read_text(encoding="utf-8")).get("ignore_regions") or []:
        ignore[r["frame"]].append(r["bbox_xywh"])
    report = {}
    for name, folder in DETECTORS.items():
        pred = ev.load_predictions(TRACKING / folder.replace("vntraffic_001", a.sequence) / "predictions_mot.txt", a.first, a.last)
        plain = ev.evaluate(gt, pred, a.first, a.last)
        kept, dropped = drop_ignored(gt, pred, ignore)
        withig = ev.evaluate(gt, kept, a.first, a.last)
        report[name] = {"plain": plain, "with_ignore": withig, "pred_boxes_dropped_by_ignore": dropped}
        for tag, r in (("thường", plain), ("bỏ vùng cụm", withig)):
            print(f"{name:8s} {tag:12s} HOTA {r['HOTA']:.3f} IDF1 {r['IDF1']:.3f} MOTA {r['MOTA']:.3f} IDsw {r['ID_switches']:3d} "
                  f"recall {r['recall']:.3f} prec {r['precision']:.3f}" + (f"  (bỏ {dropped} box)" if tag != "thường" else ""))
    if a.output:
        a.output.write_text(json.dumps({"window": [a.first, a.last], "gt": str(a.gt), "status": "PROVISIONAL_DRAFT_NOT_GT", "results": report}, indent=1))


if __name__ == "__main__":
    main()
