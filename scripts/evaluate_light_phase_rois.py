"""Light phase with per-camera head ROIs vs automatic head association, against the human QA.

Truth per ROI and second: the QA'd automatic heads whose boxes lie inside the ROI (majority of their
red/yellow/green labels at that second). QA heads that are real lights but outside every ROI were
not selected for monitoring; QA heads judged "not a light" are reported if they fall in an ROI.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.signal.phase_voting import PhaseVoter, RoiHeadAssociator, run  # noqa: E402

WORK = ROOT / "data/light_phase_eval_20261114"
COLOURS = ("red", "yellow", "green")


def inside(box, roi, grow=0.2):
    x1, y1, x2, y2 = roi
    gx, gy = grow * (x2 - x1), grow * (y2 - y1)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return x1 - gx <= cx <= x2 + gx and y1 - gy <= cy <= y2 + gy


def main():
    det_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else WORK  # another model's detections.json (same frames)
    rois = json.loads((WORK / "signal_head_rois.json").read_text(encoding="utf-8"))
    items = json.loads((WORK / "qa/items.json").read_text(encoding="utf-8"))
    answers = json.loads((WORK / "qa/light_phase_qa_result.json").read_text(encoding="utf-8"))
    totals, per_roi, notes = Counter(), [], Counter()
    for source, source_rois in sorted(rois.items()):
        frames = json.loads((det_root / source / "detections.json").read_text(encoding="utf-8"))["frames"]
        auto_out, _ = run(frames)
        out, _ = run(frames, RoiHeadAssociator(source_rois), PhaseVoter())
        source_items = [i for i in items if i["source"] == source]
        for item in source_items:
            labels = [answers[item["key"]].get(str(c["t"])) for c in item["cells"]]
            real = any(l in COLOURS for l in labels)
            in_roi = any(all(inside(c["box"], r["xyxy"]) for c in item["cells"]) or
                         sum(inside(c["box"], r["xyxy"]) for c in item["cells"]) >= 0.5 * len(item["cells"]) for r in source_rois)
            notes[("real" if real else "false") + ("_in_roi" if in_roi else "_outside_roi")] += 1
            if not real and not in_roi:  # seconds of colour this false head produced without ROIs
                totals["auto_false_head_voted_seconds_removed"] += sum(
                    auto_out[c["index"]]["voted"].get(item["head_id"], "unknown") != "unknown" for c in item["cells"])
        for roi in source_rois:
            members = [i for i in source_items if sum(inside(c["box"], roi["xyxy"]) for c in i["cells"]) >= 0.5 * len(i["cells"])]
            truth = {}
            for m in members:
                for c in m["cells"]:
                    label = answers[m["key"]].get(str(c["t"]))
                    if label in COLOURS:
                        truth.setdefault(c["index"], Counter())[label] += 1
            stats = Counter()
            seq = []
            for index in sorted(truth):
                label = truth[index].most_common(1)[0][0]
                o = out[index]
                raw, vote = o["raw"].get(roi["id"]), o["voted"].get(roi["id"], "unknown")
                stats["seconds"] += 1
                stats["raw_correct"] += raw == label
                stats["raw_output"] += raw is not None
                stats["raw_wrong"] += raw is not None and raw != label
                stats["vote_correct"] += vote == label
                stats["vote_output"] += vote != "unknown"
                stats["vote_wrong"] += vote not in ("unknown", label)
                seq.append(label)
            stats["true_changes"] = sum(a != b for a, b in zip(seq, seq[1:]))
            if truth:
                t0, t1 = frames[min(truth)]["t"], frames[max(truth)]["t"]
                v = [o["voted"].get(roi["id"]) for o in out if t0 <= o["t"] <= t1 and o["voted"].get(roi["id"]) in COLOURS + ("off",)]
                r = [o["raw"].get(roi["id"]) for o in out if t0 <= o["t"] <= t1 and o["raw"].get(roi["id"])]
                stats["vote_changes"] = sum(a != b for a, b in zip(v, v[1:]))
                stats["raw_changes"] = sum(a != b for a, b in zip(r, r[1:]))
            totals.update(stats)
            per_roi.append({"source": source, "roi": roi["id"], "qa_heads": [m["key"] for m in members], **stats})
    s = totals
    summary = {"qa_heads": dict(notes), "roi_count": sum(len(v) for v in rois.values()),
               "seconds_with_truth": s["seconds"],
               "per_frame": {"accuracy": s["raw_correct"] / s["seconds"], "coverage": s["raw_output"] / s["seconds"], "wrong_seconds": s["raw_wrong"], "colour_changes": s["raw_changes"]},
               "voted": {"accuracy": s["vote_correct"] / s["seconds"], "coverage": s["vote_output"] / s["seconds"], "wrong_seconds": s["vote_wrong"], "colour_changes": s["vote_changes"]},
               "true_changes": s["true_changes"],
               "false_head_voted_seconds_removed_by_rois": s["auto_false_head_voted_seconds_removed"],
               "per_roi": per_roi}
    (det_root / "evaluation_rois.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_roi"}, indent=1, ensure_ascii=False))
    for r in per_roi:
        print(f"{r['source']:16s} ROI{r['roi']} heads {','.join(h.split(':')[1] for h in r['qa_heads']) or '-':12s} sec {r.get('seconds', 0):3d} "
              f"raw ok {r.get('raw_correct', 0):3d} wrong {r.get('raw_wrong', 0):2d} | vote ok {r.get('vote_correct', 0):3d} wrong {r.get('vote_wrong', 0):2d} | "
              f"changes raw {r.get('raw_changes', 0)} vote {r.get('vote_changes', 0)} true {r.get('true_changes', 0)}")


if __name__ == "__main__":
    main()
