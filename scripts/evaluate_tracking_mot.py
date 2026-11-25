import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.96, 0.05)


def load_gt(path, first, last):
    draft = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = defaultdict(dict)
    for frame in draft["frames"]:
        if first <= frame["mot_frame_id"] <= last:
            for item in frame["objects"]:
                frames[frame["mot_frame_id"]][item["track_id"]] = item["bbox_xywh"]
    return frames


def load_predictions(path, first, last):
    frames = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        parts = line.split(",")
        frame = int(parts[0])
        if first <= frame <= last:
            frames[frame][int(parts[1])] = list(map(float, parts[2:6]))
    return frames


def iou_matrix(first, second):
    a, b = np.array(first, float).reshape(-1, 4), np.array(second, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    width = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    height = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.maximum(width, 0) * np.maximum(height, 0)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def evaluate(gt, pred, first, last, threshold=0.5):
    frames = range(first, last + 1)
    gt_ids = sorted({i for f in gt.values() for i in f})
    pred_ids = sorted({i for f in pred.values() for i in f})
    gi, pi = {g: k for k, g in enumerate(gt_ids)}, {p: k for k, p in enumerate(pred_ids)}
    sims = {}
    for frame in frames:
        g, p = sorted(gt.get(frame, {})), sorted(pred.get(frame, {}))
        sims[frame] = (g, p, iou_matrix([gt[frame][i] for i in g], [pred[frame][i] for i in p]))
    n_gt = sum(len(sims[f][0]) for f in frames)
    n_pred = sum(len(sims[f][1]) for f in frames)

    # CLEAR-MOT: per-frame Hungarian matching that prefers the previous assignment of a GT identity.
    tp = fp = fn = idsw = 0
    previous = {}
    for frame in frames:
        g, p, sim = sims[frame]
        matches = []
        if len(g) and len(p):
            score = np.where(sim >= threshold, sim, -1.0)
            for a, gid in enumerate(g):
                if gid in previous and previous[gid] in p:
                    c = p.index(previous[gid])
                    if sim[a, c] >= threshold:
                        score[a, c] += 1.0
            rows, cols = linear_sum_assignment(-score)
            matches = [(r, c) for r, c in zip(rows, cols) if sim[r, c] >= threshold]
        tp += len(matches)
        fn += len(g) - len(matches)
        fp += len(p) - len(matches)
        for r, c in matches:
            if g[r] in previous and previous[g[r]] != p[c]:
                idsw += 1
            previous[g[r]] = p[c]
    mota = 1 - (fn + fp + idsw) / max(n_gt, 1)

    # IDF1: one global GT-identity to prediction-identity assignment maximising matched frames.
    overlap = np.zeros((len(gt_ids), len(pred_ids)))
    for frame in frames:
        g, p, sim = sims[frame]
        for a, gid in enumerate(g):
            for c, pid in enumerate(p):
                if sim[a, c] >= threshold:
                    overlap[gi[gid], pi[pid]] += 1
    idtp = 0.0
    if overlap.size:
        rows, cols = linear_sum_assignment(-overlap)
        idtp = float(overlap[rows, cols].sum())
    idf1 = 2 * idtp / max(n_gt + n_pred, 1)

    # HOTA (TrackEval's two-pass scheme): global alignment first, then per-frame matching at each alpha.
    potential = np.zeros((len(gt_ids), len(pred_ids)))
    gt_total, pred_total = np.zeros(len(gt_ids)), np.zeros(len(pred_ids))
    for frame in frames:
        g, p, sim = sims[frame]
        for gid in g:
            gt_total[gi[gid]] += 1
        for pid in p:
            pred_total[pi[pid]] += 1
        if len(g) and len(p):
            denominator = sim.sum(0, keepdims=True) + sim.sum(1, keepdims=True) - sim
            ratio = np.divide(sim, denominator, out=np.zeros_like(sim), where=denominator > 0)
            potential[np.ix_([gi[i] for i in g], [pi[i] for i in p])] += ratio
    alignment = potential / np.maximum(gt_total[:, None] + pred_total[None, :] - potential, 1e-9)
    hota, deta, assa = [], [], []
    for alpha in ALPHAS:
        htp = hfn = hfp = 0
        counts = np.zeros_like(potential)
        for frame in frames:
            g, p, sim = sims[frame]
            if len(g) and len(p):
                score = alignment[np.ix_([gi[i] for i in g], [pi[i] for i in p])] * sim
                rows, cols = linear_sum_assignment(-score)
                chosen = [(r, c) for r, c in zip(rows, cols) if sim[r, c] >= alpha - 1e-12]
                for r, c in chosen:
                    counts[gi[g[r]], pi[p[c]]] += 1
                htp += len(chosen)
                hfn += len(g) - len(chosen)
                hfp += len(p) - len(chosen)
            else:
                hfn += len(g)
                hfp += len(p)
        association = counts / np.maximum(gt_total[:, None] + pred_total[None, :] - counts, 1e-9)
        assa_value = float((counts * association).sum() / max(htp, 1))
        deta_value = htp / max(htp + hfn + hfp, 1)
        deta.append(deta_value)
        assa.append(assa_value)
        hota.append(float(np.sqrt(deta_value * assa_value)))
    return {"gt_boxes": n_gt, "pred_boxes": n_pred, "gt_ids": len(gt_ids), "pred_ids": len(pred_ids), "HOTA": float(np.mean(hota)), "DetA": float(np.mean(deta)),
            "AssA": float(np.mean(assa)), "IDF1": float(idf1), "MOTA": float(mota), "ID_switches": idsw, "FN": fn, "FP": fp, "TP": tp,
            "recall": tp / max(n_gt, 1), "precision": tp / max(n_pred, 1)}


def main():
    parser = argparse.ArgumentParser(description="Provisional MOT metrics on a draft (not verified GT) for a frame window")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--first", type=int, default=201)
    parser.add_argument("--last", type=int, default=230)
    parser.add_argument("--prediction", action="append", required=True, help="name=path/to/predictions_mot.txt")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    gt = load_gt(arguments.gt, arguments.first, arguments.last)
    report = {}
    for item in arguments.prediction:
        name, path = item.split("=", 1)
        report[name] = evaluate(gt, load_predictions(path, arguments.first, arguments.last), arguments.first, arguments.last)
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report[name].items()}), flush=True)
    if arguments.output:
        arguments.output.write_text(json.dumps({"window": [arguments.first, arguments.last], "gt": str(arguments.gt), "status": "PROVISIONAL_DRAFT_NOT_GT", "results": report}, indent=2))


if __name__ == "__main__":
    main()
