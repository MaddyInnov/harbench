"""PAMAP2 loader with per-sample labels for boundary-window experiments.

Verified column layout (0-based, from data inspection 2026-10-09):
  col 0: timestamp, col 1: activityID, col 2: heart rate (90% NaN, unused)
  per IMU block of 17 cols [hand@3, chest@20, ankle@37]:
    [temperature, acc1 x/y/z, acc2 x/y/z, gyro x/y/z, mag x/y/z, quat x4]
Channel selection: acc1 + gyro from all 3 IMUs = 18 channels @ 100 Hz.
"""

import os
import zipfile

import numpy as np

from .datasets import _download

PAMAP2_URLS = [
    "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
]

IMU_BLOCKS = [3, 20, 37]  # hand, chest, ankle
SELECTED_CHANNELS = [b + o for b in IMU_BLOCKS for o in (1, 2, 3, 7, 8, 9)]

ACTIVITY_NAMES = {
    0: "transient", 1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking", 9: "watching_TV",
    10: "computer_work", 11: "car_driving", 12: "ascending_stairs",
    13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
    18: "folding_laundry", 19: "house_cleaning", 20: "playing_soccer",
    24: "rope_jumping",
}

TRAIN_SUBJECTS = [101, 102, 103, 104, 105, 109]
VAL_SUBJECTS = [106]
TEST_SUBJECTS = [107, 108]

WINDOW = 256   # 2.56 s @ 100 Hz
STRIDE = 128   # 50% overlap


def _interp_nans(a):
    a = a.copy()
    for j in range(a.shape[1]):
        col = a[:, j]
        mask = np.isnan(col)
        if mask.all():
            col[:] = 0.0
        else:
            x = np.arange(len(col))
            col[mask] = np.interp(x[mask], x[~mask], col[~mask])
        a[:, j] = col
    return a


def _extract(data_root):
    outer = _download(PAMAP2_URLS, os.path.join(data_root, "pamap2.zip"))
    inner = os.path.join(data_root, "PAMAP2_Dataset.zip")
    base = os.path.join(data_root, "PAMAP2_Dataset", "Protocol")
    if not os.path.isdir(base):
        print("Extracting outer archive ...", flush=True)
        with zipfile.ZipFile(outer, "r") as zf:
            zf.extractall(data_root)
    if not os.path.isdir(base):
        print("Extracting nested archive ...", flush=True)
        with zipfile.ZipFile(inner, "r") as zf:
            zf.extractall(data_root)
    return os.path.join(data_root, "PAMAP2_Dataset")


def load_pamap2(data_root="data", window=WINDOW, stride=STRIDE,
                protocol_only=True):
    """Returns dict with X (N,T,C), label_seqs (N,T) mapped int,
    subjects, activity_names, sampling_rate."""
    root = _extract(data_root)
    proto_dir = os.path.join(root, "Protocol")

    X_list, seq_list, subj_list = [], [], []
    all_ids = set()
    per_subject = {}
    for subj in TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS:
        path = os.path.join(proto_dir, f"subject{subj}.dat")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping", flush=True)
            continue
        raw = np.loadtxt(path)  # (samples, 54)
        acts = raw[:, 1].astype(np.int64)
        sig = _interp_nans(raw[:, SELECTED_CHANNELS].astype(np.float64))
        per_subject[subj] = (sig, acts)
        all_ids.update(np.unique(acts).tolist())
        print(f"subject{subj}: {len(acts)} samples, "
              f"activities {sorted(set(acts.tolist()))}", flush=True)

    id_map = {aid: i for i, aid in enumerate(sorted(all_ids))}
    activity_names = [ACTIVITY_NAMES.get(aid, f"id_{aid}")
                      for aid in sorted(all_ids)]
    print(f"Activity mapping: {dict(zip(sorted(all_ids), activity_names))}",
          flush=True)

    for subj, (sig, acts) in per_subject.items():
        mapped = np.array([id_map[a] for a in acts], dtype=np.int64)
        n_win = (len(sig) - window) // stride + 1
        for w in range(n_win):
            s = w * stride
            X_list.append(sig[s:s + window])
            seq_list.append(mapped[s:s + window])
            subj_list.append(subj)

    X = np.stack(X_list).astype(np.float32)
    label_seqs = np.stack(seq_list)
    subjects = np.array(subj_list, dtype=np.int64)
    print(f"PAMAP2: X={X.shape}, {len(activity_names)} classes, "
          f"subjects={sorted(set(subjects.tolist()))}", flush=True)
    return {
        "X": X,
        "label_seqs": label_seqs,
        "subjects": subjects,
        "activity_names": activity_names,
        "sampling_rate": 100.0,
        "window_seconds": window / 100.0,
    }


def pamap2_indices(d):
    """Subject-disjoint train/val/test indices."""
    subjects = d["subjects"]
    tr = np.where(np.isin(subjects, TRAIN_SUBJECTS))[0]
    va = np.where(np.isin(subjects, VAL_SUBJECTS))[0]
    te = np.where(np.isin(subjects, TEST_SUBJECTS))[0]
    for a, b in [(tr, va), (tr, te), (va, te)]:
        assert len(set(subjects[a]) & set(subjects[b])) == 0, "subject leakage!"
    return tr, va, te
