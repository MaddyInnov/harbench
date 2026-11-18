"""HARTH loader (UCI id=779; Logacjov et al., Sensors 2021).

22 subjects, two 3-axial Axivity AX3 accelerometers (lower back + thigh),
free-living recordings of ~90-120 min each, professionally annotated with
PER-SAMPLE labels in the released CSVs — the property that makes HARTH
usable for dense (per-timestep) segmentation, unlike pre-cut datasets.

Files: <data_root>/harth/S<sid>.csv with columns
timestamp, back_x/y/z, thigh_x/y/z, label. Acceleration values are in g.

MIXED SAMPLING RATE (verified 2026-11-10 probe): subject 6 is recorded
at 100 Hz (10 ms steps); the other 21 subjects at 50 Hz (20 ms steps).
Downsample subject 6 by 2 before any cross-subject modelling so windows
mean the same duration for everyone.

Label codes (official UCI codebook):
  1 walking, 2 running, 3 shuffling, 4 stairs (ascending),
  5 stairs (descending), 6 standing, 7 sitting, 8 lying,
  13 cycling (sit), 14 cycling (stand),
  130 cycling (sit, inactive), 140 cycling (stand, inactive)
"""

import glob
import os

import numpy as np
import pandas as pd

LABEL_NAMES = {
    1: "walking", 2: "running", 3: "shuffling", 4: "stairs_ascending",
    5: "stairs_descending", 6: "standing", 7: "sitting", 8: "lying",
    13: "cycling_sit", 14: "cycling_stand",
    130: "cycling_sit_inactive", 140: "cycling_stand_inactive",
}
CHANNELS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]


def load_harth(data_root):
    """Load every subject's stream.

    Returns {subject_id (int): {"X": (N, 6) float32 accel in g,
    "y": (N,) int64 original label codes, "t0": first timestamp str,
    "t1": last timestamp str, "median_dt_ms": float}}.
    """
    files = sorted(glob.glob(os.path.join(data_root, "harth", "S*.csv")))
    if not files:
        raise FileNotFoundError(f"no HARTH CSVs under {data_root}/harth")
    subjects = {}
    for path in files:
        sid = int(os.path.basename(path)[1:4])
        df = pd.read_csv(path, usecols=["timestamp"] + CHANNELS + ["label"])
        ts = pd.to_datetime(df["timestamp"])
        dt_ms = ts.diff().dt.total_seconds().median() * 1000.0
        subjects[sid] = {
            "X": df[CHANNELS].to_numpy(dtype=np.float32),
            "y": df["label"].to_numpy(dtype=np.int64),
            "t0": str(ts.iloc[0]), "t1": str(ts.iloc[-1]),
            "median_dt_ms": float(dt_ms),
        }
    return subjects


def dense_windows(subjects, window=256, stride=128):
    """Sliding windows WITH per-timestep labels preserved (dense task).

    Returns X (M, window, 6) float32, Y (M, window) int64 label codes,
    subj (M,) subject ids. Windows never cross subject boundaries;
    transition-containing windows are kept (they are the point).
    """
    Xs, Ys, Ss = [], [], []
    for sid, d in subjects.items():
        X, y = d["X"], d["y"]
        for s in range(0, len(y) - window + 1, stride):
            Xs.append(X[s:s + window])
            Ys.append(y[s:s + window])
            Ss.append(sid)
    return (np.stack(Xs), np.stack(Ys), np.array(Ss, dtype=np.int64))


def loso_subjects(subject_ids):
    """Leave-one-subject-out folds: yields (train_ids, test_id)."""
    ids = list(subject_ids)
    for test_id in ids:
        yield [s for s in ids if s != test_id], test_id
