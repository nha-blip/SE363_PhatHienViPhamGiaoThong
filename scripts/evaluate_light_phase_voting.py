"""Evaluate per-frame light colour vs temporal voting against the human QA (1 label per head per second).

Truth colours: red/yellow/green/off. "not_visible" cells are excluded; heads labelled entirely
not_visible are false heads (not a light).
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.signal.phase_voting import LightHeadAssociator, PhaseVoter, run  # noqa: E402

WORK = ROOT / "data/light_phase_eval_20261114"
COLOURS = {"red", "yellow", "green", "off"}


def evaluate(voter_kwargs=None, verbose=True):
    items = json.loads((WORK / "qa/items.json").read_text(encoding="utf-8"))
    answers = json.loads((WORK / "qa/light_phase_qa_result.json").read_text(encoding="utf-8"))
    runs = {}
    stats = Counter()
    confusion_raw, confusion_vote = Counter(), Counter()
    delays, per_source = [], {}
    for item in items:
        if item["source"] not in runs:
            frames = json.loads((WORK / item["source"] / "detections.json").read_text(encoding="utf-8"))["frames"]
            runs[item["source"]] = run(frames, LightHeadAssociator(), PhaseVoter(**(voter_kwargs or {})))[0]
        out, hid = runs[item["source"]], item["head_id"]
        truth = answers.get(item["key"], {})
        labels = [truth.get(str(c["t"])) for c in item["cells"]]
        if not any(l in COLOURS for l in labels):
            stats["false_heads"] += 1
            span = [o for o in out if item["cells"][0]["t"] <= o["t"] <= item["cells"][-1]["t"]]
            stats["false_head_seconds_with_voted_colour"] += sum(o["voted"].get(hid, "unknown") != "unknown" for o in span) / 5
            stats["false_head_seconds_with_raw_colour"] += sum(hid in o["raw"] for o in span) / 5
            continue
        stats["real_heads"] += 1
        src = per_source.setdefault(item["source"], Counter())
        for c, label in zip(item["cells"], labels):
            if label not in COLOURS:
                continue
            o = out[c["index"]]
            raw, vote = o["raw"].get(hid, "none"), o["voted"].get(hid, "unknown")
            for k, v in (("cells", 1), ("raw_correct", raw == label), ("raw_detected", raw != "none"),
                         ("vote_correct", vote == label), ("vote_known", vote != "unknown"),
                         ("raw_wrong_colour", raw not in ("none", label)), ("vote_wrong_colour", vote not in ("unknown", label))):
                stats[k] += v
                src[k] += v
            confusion_raw[(label, raw)] += 1
            confusion_vote[(label, vote)] += 1
        # colour changes: truth (1 s cells) vs voted (5 fps) within the head's span
        seq = [(c["t"], l) for c, l in zip(item["cells"], labels) if l in COLOURS]
        true_changes = [(t, b) for (_, a), (t, b) in zip(seq, seq[1:]) if a != b]
        span = [o for o in out if seq[0][0] <= o["t"] <= seq[-1][0]]
        voted = [(o["t"], o["voted"].get(hid, "unknown")) for o in span]
        voted = [v for v in voted if v[1] != "unknown"]
        vote_changes = [(t, b) for (_, a), (t, b) in zip(voted, voted[1:]) if a != b]
        raws = [(o["t"], o["raw"][hid]) for o in span if hid in o["raw"]]
        stats["true_changes"] += len(true_changes)
        stats["voted_changes"] += len(vote_changes)
        stats["raw_changes"] += sum(a[1] != b[1] for a, b in zip(raws, raws[1:]))
        for t, colour in true_changes:
            hit = next((tv for tv, cv in vote_changes if cv == colour and t - 1.0 <= tv <= t + 6.0), None)
            if hit is None:
                stats["true_changes_missed_by_vote"] += 1
            else:
                delays.append(hit - t)
        stats["voted_changes_unmatched"] += sum(not any(cv == colour and t - 1.0 <= tv <= t + 6.0 for t, colour in true_changes)
                                                for tv, cv in vote_changes)
    summary = {
        "real_heads": stats["real_heads"], "false_heads": stats["false_heads"], "labelled_seconds": stats["cells"],
        "raw": {"accuracy": stats["raw_correct"] / stats["cells"], "coverage": stats["raw_detected"] / stats["cells"],
                "accuracy_when_output": stats["raw_correct"] / max(stats["raw_detected"], 1), "wrong_colour_seconds": stats["raw_wrong_colour"],
                "colour_changes": stats["raw_changes"]},
        "voted": {"accuracy": stats["vote_correct"] / stats["cells"], "coverage": stats["vote_known"] / stats["cells"],
                  "accuracy_when_output": stats["vote_correct"] / max(stats["vote_known"], 1), "wrong_colour_seconds": stats["vote_wrong_colour"],
                  "colour_changes": stats["voted_changes"], "changes_not_matching_a_true_change": stats["voted_changes_unmatched"]},
        "true_changes": stats["true_changes"], "true_changes_missed_by_vote": stats["true_changes_missed_by_vote"],
        "delay_seconds": sorted(round(d, 1) for d in delays),
        "false_heads_seconds": {"raw_colour": stats["false_head_seconds_with_raw_colour"], "voted_colour": stats["false_head_seconds_with_voted_colour"]},
        "per_source": {s: {"seconds": c["cells"], "raw_acc": round(c["raw_correct"] / c["cells"], 3), "vote_acc": round(c["vote_correct"] / c["cells"], 3),
                           "vote_acc_when_output": round(c["vote_correct"] / max(c["vote_known"], 1), 3)} for s, c in per_source.items()},
        "confusion_vote": {f"{a}->{b}": n for (a, b), n in sorted(confusion_vote.items()) if a != b},
        "confusion_raw": {f"{a}->{b}": n for (a, b), n in sorted(confusion_raw.items()) if a != b},
    }
    if verbose:
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    s = evaluate()
    (WORK / "evaluation_default_voter.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
