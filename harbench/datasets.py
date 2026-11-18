"""Dataset loaders for the HAR benchmark suite.

Common interface -- every loader returns plain numpy arrays:
    X              : (N, T, C) float32, windows, channels-last
    y              : (N,) int64, 0-indexed class labels
    subjects       : (N,) int64, subject id per window
    activity_names : list[str]
    sampling_rate  : float, Hz

Splitting is ALWAYS by subject: no window from a validation/test subject
ever appears in training. Model selection uses the validation split only;
the test split is evaluated exactly once per trained model.
"""

import os
import subprocess
import time
import urllib.request
import zipfile

import numpy as np

# ---------------------------------------------------------------- UCI HAR

UCI_URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]

UCI_ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

UCI_SIGNAL_FILES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

# The official UCI-HAR split is by subject (70/30), but NOT by id range --
# the split is an arbitrary subject set baked into the train/test files.
# We respect the FILE split for the test set (never touched in training),
# and carve a subject-disjoint validation set from the train files only.
# Verified from the data: train subjects =
#   {1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30}
# test subjects = {2,4,9,10,12,13,18,20,24}
UCI_VAL_SUBJECTS = [21, 27, 28, 30]  # carved from OFFICIAL train subjects


def _looks_complete(path, min_bytes=1_000_000):
    """True if path exists, is big enough, and passes a zip CRC check."""
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except Exception:
        return False


def _download(urls, dest_path):
    """Download with hard timeouts and retries -- never hangs forever.

    Uses curl (retry, connect/max-time timeouts, resume of partial files)
    with a time-bounded urllib fallback. A corrupt partial file is never
    silently reused; total failure raises loudly instead of stalling the
    notebook cell (which previously let Kaggle kill the session for
    idleness with zero results saved).
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if _looks_complete(dest_path):
        print(f"Reusing cached {dest_path}", flush=True)
        return dest_path
    if os.path.exists(dest_path):
        print(f"Removing incomplete/corrupt {dest_path}", flush=True)
        os.remove(dest_path)
    last_err = None
    for url in urls:
        for attempt in range(4):
            try:
                print(f"Downloading {url} (attempt {attempt + 1}/4) ...",
                      flush=True)
                r = subprocess.run(
                    ["curl", "-fSL", "--retry", "3", "--retry-all-errors",
                     "--connect-timeout", "30", "--max-time", "2400",
                     "-C", "-", "-o", dest_path, url],
                    capture_output=True, text=True, timeout=2500)
                if r.returncode == 0 and _looks_complete(dest_path):
                    print(f"Saved to {dest_path}", flush=True)
                    return dest_path
                last_err = (r.stderr or "")[-500:]
                print(f"curl attempt failed (rc={r.returncode}): {last_err}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"Failed: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
        # urllib fallback with an explicit timeout, one attempt per URL
        try:
            print(f"curl exhausted, trying urllib with timeout for {url}",
                  flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "curl"})
            with urllib.request.urlopen(req, timeout=120) as resp, \
                    open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if _looks_complete(dest_path):
                print(f"Saved to {dest_path} (urllib)", flush=True)
                return dest_path
            last_err = "urllib download failed integrity check"
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"urllib failed: {e}", flush=True)
    raise RuntimeError(f"All download URLs failed. Last error: {last_err}")


def _load_signal_matrix(path):
    # Each row: one 128-sample window of a single channel.
    return np.loadtxt(path, dtype=np.float32)


def load_uci_har(data_root="data"):
    """Load UCI HAR raw inertial signals (9 ch x 128 samples @ 50 Hz)."""
    zip_path = _download(UCI_URLS, os.path.join(data_root, "uci_har.zip"))
    extract_dir = os.path.join(data_root, "UCI HAR Dataset")
    if not os.path.isdir(os.path.join(extract_dir, "train")):
        print("Extracting ...", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        # Newer UCI archive nests the classic zip inside the download.
        nested = os.path.join(data_root, "UCI HAR Dataset.zip")
        if not os.path.isdir(os.path.join(extract_dir, "train")) and os.path.exists(nested):
            print("Extracting nested archive ...", flush=True)
            with zipfile.ZipFile(nested, "r") as zf:
                zf.extractall(data_root)

    def _read_split(split):
        base = os.path.join(extract_dir, split)
        sig_dir = os.path.join(base, "Inertial Signals")
        channels = [_load_signal_matrix(os.path.join(sig_dir, f"{name}_{split}.txt"))
                    for name in UCI_SIGNAL_FILES]
        X = np.stack(channels, axis=-1)  # (N, 128, 9)
        y = np.loadtxt(os.path.join(base, f"y_{split}.txt"), dtype=np.int64) - 1
        subjects = np.loadtxt(os.path.join(base, f"subject_{split}.txt"), dtype=np.int64)
        return X, y, subjects

    Xtr, ytr, str_ = _read_split("train")
    Xte, yte, ste = _read_split("test")
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)
    subjects = np.concatenate([str_, ste], axis=0)
    # source file per window: needed to respect the official split
    source = np.array(["train"] * len(ytr) + ["test"] * len(yte))
    print(f"UCI HAR: X={X.shape}, classes={sorted(np.unique(y).tolist())}, "
          f"subjects={sorted(np.unique(subjects).tolist())}", flush=True)
    return {
        "X": X,
        "y": y,
        "subjects": subjects,
        "source": source,
        "activity_names": UCI_ACTIVITIES,
        "sampling_rate": 50.0,
        "window_seconds": 2.56,
    }


def uci_har_indices(d):
    """Honest indices: test = official test files; val = subject-disjoint
    subset of official train files; train = remaining official train files."""
    subjects, source = d["subjects"], d["source"]
    test_idx = np.where(source == "test")[0]
    train_pool = np.where(source == "train")[0]
    val_mask = np.isin(subjects[train_pool], UCI_VAL_SUBJECTS)
    val_idx = train_pool[val_mask]
    train_idx = train_pool[~val_mask]
    # Safety: assert the val subjects really came from the official train pool
    # and that no subject appears in more than one split.
    for a, b in [(train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)]:
        assert len(set(subjects[a]) & set(subjects[b])) == 0, "subject leakage!"
    return train_idx, val_idx, test_idx


def subject_indices(subjects, wanted):
    wanted = set(wanted)
    return np.array([i for i, s in enumerate(subjects) if s in wanted])


def standardize_fit_apply(X_train, X_val, X_test):
    """Per-channel z-score using TRAIN statistics only."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std
