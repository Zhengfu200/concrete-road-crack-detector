"""IoU matching and operating-point metrics, kept independent of model runtime."""
from __future__ import annotations

import numpy as np
from common import NAMES


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1)*max(0, y2-y1)
    area_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return intersection/max(1e-12, area_a+area_b-intersection)


def xywh_to_xyxy(box):
    c, x, y, w, h = box
    return [int(c), x-w/2, y-h/2, x+w/2, y+h/2]


def class_metrics(samples, thresholds, overlap=.5):
    counts = {c: [0, 0, 0] for c in NAMES}  # TP, FP, FN
    scenarios = {"crack": 0, "background": 0}
    for sample in samples:
        gt, predictions = sample["gt"], sample["pred"]
        if any(int(g[0]) not in NAMES for g in gt) or any(int(p[0]) not in NAMES for p in predictions):
            raise ValueError("评估只接受 crack 单类别")
        predictions = [p for p in predictions if p[5] >= thresholds[int(p[0])]]
        key = "crack" if gt else "background"
        scenarios[key] += 1
        for c in NAMES:
            targets = [g for g in gt if g[0] == c]
            pred = sorted((p for p in predictions if p[0] == c), key=lambda p: p[5], reverse=True)
            used = set()
            for p in pred:
                options = [(iou(p[1:5], g[1:5]), j) for j, g in enumerate(targets) if j not in used]
                score, j = max(options, default=(0, -1))
                if score >= overlap:
                    counts[c][0] += 1; used.add(j)
                else:
                    counts[c][1] += 1
            counts[c][2] += len(targets)-len(used)
    metrics = {}
    for c, (tp, fp, fn) in counts.items():
        precision = tp/(tp+fp) if tp+fp else 0.
        recall = tp/(tp+fn) if tp+fn else 0.
        metrics[str(c)] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
                           "recall": recall, "f1": 2*precision*recall/max(1e-12, precision+recall)}
    return {"per_class": metrics, "scenarios": scenarios}


def threshold_curve(samples):
    """Exact confidence breakpoints, using the same greedy matching as class_metrics."""
    events, total = [], 0
    for sample in samples:
        targets = sample["gt"]
        if any(int(g[0]) not in NAMES for g in targets) or any(int(p[0]) not in NAMES for p in sample["pred"]):
            raise ValueError("评估只接受 crack 单类别")
        total += len(targets)
        used = set()
        for p in sorted(sample["pred"], key=lambda p: p[5], reverse=True):
            score, index = max(((iou(p[1:5], g[1:5]), j) for j, g in enumerate(targets)
                                if j not in used), default=(0, -1))
            tp = score >= .5
            if tp:
                used.add(index)
            events.append((float(p[5]), int(tp)))
    events.sort(reverse=True, key=lambda row: row[0])
    points, tp, fp, index = [], 0, 0, 0
    while index < len(events):
        threshold = events[index][0]
        while index < len(events) and events[index][0] == threshold:
            tp += events[index][1]
            fp += 1 - events[index][1]
            index += 1
        precision, recall = tp / (tp + fp), tp / total if total else 0.
        metric = {"tp": tp, "fp": fp, "fn": total-tp, "precision": precision, "recall": recall,
                  "f1": 2*precision*recall/max(1e-12, precision+recall)}
        points.append({"threshold": threshold, "metrics": {"0": metric}})
    if not points:
        points = [{"threshold": .2, "metrics": {"0": {"tp": 0, "fp": 0, "fn": total,
                   "precision": 0., "recall": 0., "f1": 0.}}}]
    return points


def calibration_summary(curve):
    def best(points, key):
        return max(points, key=key) if points else None
    def metric(row):
        return row["metrics"]["0"]
    feasible = [r for r in curve if metric(r)["precision"] >= .8 and metric(r)["recall"] >= .9]
    at_precision = [r for r in curve if metric(r)["precision"] >= .8]
    at_recall = [r for r in curve if metric(r)["recall"] >= .9]
    recall_key = lambda r: (metric(r)["recall"], metric(r)["precision"], r["threshold"])
    fallback = best(curve, lambda r: (metric(r)["f1"], metric(r)["recall"], metric(r)["precision"], r["threshold"]))
    return {"joint_target_feasible": bool(feasible),
            "selection_policy": "joint_precision80_recall90" if feasible else "preview_max_f1_no_feasible_threshold",
            "selected": best(feasible, recall_key) or fallback,
            "recall_at_precision80": best(at_precision, recall_key),
            "precision_at_recall90": best(at_recall, lambda r: (metric(r)["precision"], metric(r)["recall"], r["threshold"]))}


def calibrate(samples):
    curve = threshold_curve(samples)
    selected = calibration_summary(curve)["selected"]
    return {0: selected["threshold"]}, curve


def confusion(samples, thresholds, overlap=.5):
    """Rows=true, columns=predicted; crack=0, background=1."""
    background = len(NAMES)
    matrix = np.zeros((background+1, background+1), dtype=int)
    for sample in samples:
        gt = sample["gt"]
        pred = [p for p in sample["pred"] if p[5] >= thresholds[int(p[0])]]
        pairs = sorted([(iou(g[1:5], p[1:5]), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)], reverse=True)
        used_gt, used_pred = set(), set()
        for score, i, j in pairs:
            if score >= overlap and i not in used_gt and j not in used_pred:
                matrix[int(gt[i][0]), int(pred[j][0])] += 1
                used_gt.add(i); used_pred.add(j)
        for i, g in enumerate(gt):
            if i not in used_gt:
                matrix[int(g[0]), background] += 1
        for j, p in enumerate(pred):
            if j not in used_pred:
                matrix[background, int(p[0])] += 1
    return matrix.tolist()
