"""Evaluation metrics. Numpy/sklearn based -- no torch dependency."""

import numpy as np


def compute_metrics(y_true, y_pred, activity_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = int(max(y_true.max(), y_pred.max())) + 1

    accuracy = float((y_true == y_pred).mean())

    per_class_f1 = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        per_class_f1.append(2 * tp / denom if denom > 0 else 0.0)
    macro_f1 = float(np.mean(per_class_f1))

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    out = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_true)),
    }
    if activity_names:
        out["per_class_f1_named"] = {
            name: per_class_f1[i] for i, name in enumerate(activity_names)
        }
    return out


def expected_calibration_error(y_true, proba, n_bins=10):
    """ECE from softmax probabilities: |acc - conf| weighted by bin mass."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = (pred[mask] == y_true[mask]).mean()
        ece += (mask.mean()) * abs(acc - conf[mask].mean())
    return float(ece)
