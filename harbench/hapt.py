"""HAPT loader with per-sample labels for the gap-1 rerun.

HAPT (Reyes-Ortiz et al.): 30 subjects, waist phone, 50 Hz, acc+gyro.
12 classes: 6 basic activities + 6 explicit postural-transition classes
(stand-to-sit, sit-to-stand, sit-to-lie, lie-to-sit, stand-to-lie,
lie-to-stand) -- the right testbed for mixed-window labeling.

Raw layout (verified 2026-10-09):
  RawData/acc_exp{NN}_user{MM}.txt, gyro_exp{NN}_user{MM}.txt : 3 cols each
  RawData/labels.txt : (exp_id, user_id, activity_id, start, end) segments,
                       1-based sample indices into that experiment's signal.
Official split: 21 train / 9 test subjects (from Train/subject_id_*.txt).
We keep the official TEST subjects and carve val from official train.
"""

import os
import re
import zipfile

import numpy as np

from .datasets import _download

HAPT_URLS = [
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip",
]

HAPT_ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
    "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE",
    "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND",
]
TRANSITION_IDS = [6, 7, 8, 9, 10, 11]  # 0-based ids of the 6 transitions

# Official HAPT split (verified from Train/Test subject_id files).
HAPT_TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]
HAPT_VAL_SUBJECTS = [27, 28, 29, 30]  # carved from official train subjects

WINDOW = 128  # 2.56 s @ 50 Hz
STRIDE = 64   # 50% overlap


def _extract(data_root):
    zip_path = _download(HAPT_URLS, os.path.join(data_root, "hapt.zip"))
    raw_dir = os.path.join(data_root, "RawData")
    if not os.path.isdir(raw_dir) or len(os.listdir(raw_dir)) < 100:
        print("Extracting HAPT ...", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
    return data_root


def _parse_labels(raw_dir):
    """labels.txt -> dict[(exp, user)] -> list of (activity_0based, start0, end0)."""
    segs = {}
    with open(os.path.join(raw_dir, "labels.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            exp, user, act, start, end = map(int, parts)
            segs.setdefault((exp, user), []).append((act - 1, start - 1, end - 1))
    return segs


def load_hapt(data_root="data", window=WINDOW, stride=STRIDE):
    """Returns dict with X (N,T,6), label_seqs (N,T), subjects,
    activity_names, sampling_rate."""
    data_root = _extract(data_root)
    raw_dir = os.path.join(data_root, "RawData")
    segs = _parse_labels(raw_dir)

    # Group experiment files by user.
    user_exps = {}
    for fname in os.listdir(raw_dir):
        m = re.match(r"acc_exp(\d+)_user(\d+)\.txt", fname)
        if m:
            user_exps.setdefault(int(m.group(2)), []).append(int(m.group(1)))

    X_list, seq_list, subj_list = [], [], []
    for user in sorted(user_exps):
        sig_parts, lab_parts = [], []
        for exp in sorted(user_exps[user]):
            acc = np.loadtxt(os.path.join(raw_dir, f"acc_exp{exp:02d}_user{user:02d}.txt"),
                             dtype=np.float32)
            gyr = np.loadtxt(os.path.join(raw_dir, f"gyro_exp{exp:02d}_user{user:02d}.txt"),
                             dtype=np.float32)
            assert len(acc) == len(gyr), f"acc/gyro length mismatch exp{exp} user{user}"
            labels = np.full(len(acc), -1, dtype=np.int64)
            for act, s, e in segs.get((exp, user), []):
                labels[s:e + 1] = act
            sig_parts.append(np.concatenate([acc, gyr], axis=1))
            lab_parts.append(labels)
        sig = np.concatenate(sig_parts, axis=0)
        lab = np.concatenate(lab_parts, axis=0)
        unlabeled = (lab == -1).mean()
        if unlabeled > 0:
            print(f"user{user:02d}: {unlabeled:.1%} samples unlabeled",
                  flush=True)
        n_win = (len(sig) - window) // stride + 1
        kept = 0
        for w in range(n_win):
            s = w * stride
            wlab = lab[s:s + window]
            if (wlab == -1).any():
                continue  # skip windows touching unlabeled regions
            X_list.append(sig[s:s + window])
            seq_list.append(wlab)
            subj_list.append(user)
            kept += 1
        print(f"user{user:02d}: {len(sig)} samples -> {kept} windows",
              flush=True)

    X = np.stack(X_list).astype(np.float32)
    label_seqs = np.stack(seq_list)
    subjects = np.array(subj_list, dtype=np.int64)
    print(f"HAPT: X={X.shape}, classes={len(HAPT_ACTIVITIES)}, "
          f"subjects={sorted(set(subjects.tolist()))}", flush=True)
    return {
        "X": X,
        "label_seqs": label_seqs,
        "subjects": subjects,
        "activity_names": HAPT_ACTIVITIES,
        "sampling_rate": 50.0,
        "window_seconds": window / 50.0,
    }


def hapt_indices(d):
    """Honest indices: official test subjects; val carved from train."""
    subjects = d["subjects"]
    te = np.where(np.isin(subjects, HAPT_TEST_SUBJECTS))[0]
    va = np.where(np.isin(subjects, HAPT_VAL_SUBJECTS))[0]
    tr = np.where(~np.isin(subjects, HAPT_TEST_SUBJECTS + HAPT_VAL_SUBJECTS))[0]
    for a, b in [(tr, va), (tr, te), (va, te)]:
        assert len(set(subjects[a]) & set(subjects[b])) == 0, "subject leakage!"
    return tr, va, te
