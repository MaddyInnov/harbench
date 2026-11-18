"""Gap #1: proportion-aware soft labeling for mixed windows.

A window straddling an activity transition contains two (or more) labels,
but standard training forces a one-hot majority label on it -- actively
mis-training the model on the minority portion. This module replaces the
one-hot target with the window's label proportions (optionally sharpened),
ported from density-guided label smoothing in video temporal action
localization (arXiv 2403.06616).

Interface: per-sample label sequences per window, shape (N, T) int.
"""

import numpy as np


def proportion_targets(label_seqs, n_classes, temperature=1.0):
    """(N, T) int labels -> (N, C) soft targets (normalized histograms).

    temperature < 1 sharpens toward the majority class; 1.0 = raw proportions.
    """
    label_seqs = np.asarray(label_seqs)
    n, t = label_seqs.shape
    counts = np.zeros((n, n_classes), dtype=np.float64)
    for c in range(n_classes):
        counts[:, c] = (label_seqs == c).sum(axis=1)
    props = counts / t
    if temperature != 1.0:
        # Sharpen in probability space, renormalize.
        logits = np.log(props + 1e-12) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        props = exp / exp.sum(axis=1, keepdims=True)
    return props.astype(np.float32)


def window_purity(label_seqs):
    """Fraction of the majority label in each window. 1.0 = pure."""
    label_seqs = np.asarray(label_seqs)
    n, t = label_seqs.shape
    n_classes = int(label_seqs.max()) + 1
    counts = np.stack([(label_seqs == c).sum(axis=1) for c in range(n_classes)],
                      axis=1)
    return counts.max(axis=1) / t


def boundary_mask(label_seqs):
    """True for windows containing more than one label."""
    return window_purity(label_seqs) < 1.0


def majority_labels(label_seqs):
    """Standard one-hot baseline target: majority vote per window."""
    label_seqs = np.asarray(label_seqs)
    n_classes = int(label_seqs.max()) + 1
    counts = np.stack([(label_seqs == c).sum(axis=1) for c in range(n_classes)],
                      axis=1)
    return counts.argmax(axis=1).astype(np.int64)
