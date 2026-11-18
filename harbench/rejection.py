"""Confidence-based window rejection (gap #3).

A windowed classifier forced to label every window fails on ambiguous
(boundary/transition/null) windows. Selective classification lets the model
abstain instead: at coverage c (fraction of windows accepted), report the
accuracy on accepted windows (selective accuracy).

Methods:
  max-softmax : post-hoc; reject if max softmax probability < tau.
  energy      : post-hoc; reject if energy E(x) = -T*logsumexp(logits/T) > tau.
  learned     : abstain head trained with the Deep Gamblers loss
                L = -log(p_y + p_abstain / o); reject if p_abstain > tau.

Key diagnostics: does the rejector catch the windows the classifier gets
wrong (rejection recall/precision on misclassifications), and does it
selectively reject boundary/transition windows?
"""

import numpy as np


def max_softmax_scores(proba):
    """Higher = more confident. Reject if score < tau."""
    return np.asarray(proba).max(axis=1)


def energy_scores(logits, temperature=1.0):
    """Higher = more uncertain. Reject if score > tau."""
    logits = np.asarray(logits) / temperature
    m = logits.max(axis=1, keepdims=True)
    return (-temperature * (m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1))))


def risk_coverage_curve(y_true, y_pred, reject_scores, higher_means_reject,
                        n_thresholds=101):
    """Sweep thresholds -> list of (threshold, coverage, selective_accuracy).

    reject_scores: per-window scores; windows with score on the reject side
    of the threshold are abstained on.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(reject_scores, dtype=np.float64)
    correct = (y_true == y_pred)

    lo, hi = scores.min(), scores.max()
    if lo == hi:
        thresholds = np.array([lo])
    else:
        thresholds = np.linspace(lo, hi, n_thresholds)

    points = []
    for tau in thresholds:
        rejected = scores > tau if higher_means_reject else scores < tau
        coverage = 1.0 - rejected.mean()
        accepted = ~rejected
        if accepted.sum() == 0:
            sel_acc = float("nan")
        else:
            sel_acc = float(correct[accepted].mean())
        points.append({"threshold": float(tau), "coverage": float(coverage),
                       "selective_accuracy": sel_acc,
                       "n_rejected": int(rejected.sum())})
    return points


def point_at_coverage(points, target_coverage):
    """Pick the curve point with coverage closest to (but <=) target."""
    cands = [p for p in points
             if p["coverage"] <= target_coverage + 1e-9
             and not np.isnan(p["selective_accuracy"])]
    if not cands:
        return None
    return min(cands, key=lambda p: abs(p["coverage"] - target_coverage))


def aurc(points):
    """Area under the risk-coverage curve (risk = 1 - selective accuracy),
    integrated over coverage via trapezoidal rule. Lower is better."""
    pts = sorted([p for p in points if not np.isnan(p["selective_accuracy"])],
                 key=lambda p: p["coverage"])
    if len(pts) < 2:
        return float("nan")
    cov = np.array([p["coverage"] for p in pts])
    risk = 1.0 - np.array([p["selective_accuracy"] for p in pts])
    # integrate risk over coverage from min to max coverage
    order = np.argsort(cov)
    c, r = cov[order], risk[order]
    # manual trapezoidal rule (version-proof across numpy 1.x/2.x)
    return float(np.sum((r[:-1] + r[1:]) * 0.5 * np.diff(c)))


def rejection_diagnostics(y_true, y_pred, rejected, boundary_mask=None,
                          transition_mask=None):
    """How good is the rejector at catching misclassifications?

    rejected: bool array, True = abstained.
    Returns dict with coverage, selective accuracy, rejection recall
    (P(rejected | wrong)), rejection precision (P(wrong | rejected)),
    and rejection rates on boundary/transition/pure windows.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rejected = np.asarray(rejected, dtype=bool)
    correct = (y_true == y_pred)
    wrong = ~correct

    out = {
        "coverage": float(1.0 - rejected.mean()),
        "selective_accuracy": float(correct[~rejected].mean()) if (~rejected).sum() else float("nan"),
        "base_accuracy": float(correct.mean()),
        "rejection_recall": float(rejected[wrong].mean()) if wrong.sum() else float("nan"),
        "rejection_precision": float(wrong[rejected].mean()) if rejected.sum() else float("nan"),
        "n_rejected": int(rejected.sum()),
        "n_wrong": int(wrong.sum()),
    }
    if boundary_mask is not None:
        bm = np.asarray(boundary_mask, dtype=bool)
        out["boundary_rejection_rate"] = float(rejected[bm].mean()) if bm.sum() else float("nan")
        out["pure_rejection_rate"] = float(rejected[~bm].mean()) if (~bm).sum() else float("nan")
    if transition_mask is not None:
        tm = np.asarray(transition_mask, dtype=bool)
        out["transition_rejection_rate"] = float(rejected[tm].mean()) if tm.sum() else float("nan")
    return out


def gamblers_loss(logits, targets, reward_o=4.0):
    """Deep Gamblers selective loss.

    logits: (B, K+1) with last column = abstain. targets: (B,) in [0, K).
    L = -log(p_y + p_abstain / o). Larger o -> cheaper abstention.
    """
    import torch
    import torch.nn.functional as F
    targets = torch.as_tensor(targets, dtype=torch.long, device=logits.device)
    proba = F.softmax(logits, dim=1)
    p_y = proba[torch.arange(len(targets), device=logits.device), targets]
    p_a = proba[:, -1]
    return -torch.log(p_y + p_a / reward_o + 1e-12).mean()
