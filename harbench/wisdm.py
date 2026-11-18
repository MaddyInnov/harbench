"""WISDM Smartphone and Smartwatch Activity and Biometrics Dataset (UCI ID 507).

51 subjects, 18 activities, 20 Hz accelerometer + gyroscope from a smartphone
and a smartwatch. This loader uses the phone accelerometer + gyroscope
(6 channels), merges the two streams on nearest timestamp, windows the stream,
and builds subject-disjoint splits.

Raw line format (per file data_<subject>_<accel|gyro>_<phone|watch>.txt):
    <subject>,<activity_code>,<timestamp>,<x>,<y>,<z>;
e.g. 1600,A,252207666810782,-0.36476135,8.793503,1.0550842;

Activity codes A-S (no N); mapping read from activity_key.txt in the zip.
"""

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

WISDM_URL = ("https://archive.ics.uci.edu/static/public/507/"
             "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip")
FS = 20  # Hz


def download_wisdm(data_root, url=WISDM_URL):
    """Download and extract the WISDM zip. Returns the raw/ directory."""
    dest = os.path.join(data_root, "wisdm")
    raw_dir = os.path.join(dest, "raw")
    if os.path.isdir(raw_dir) and any(os.scandir(raw_dir)):
        return raw_dir
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "wisdm-dataset.zip")
    def _zip_ok(p):
        if not os.path.exists(p):
            return False
        try:
            with zipfile.ZipFile(p) as z:
                return z.testzip() is None
        except Exception:
            return False
    if not _zip_ok(zip_path):
        if os.path.exists(zip_path):
            print("Cached WISDM zip is corrupt/incomplete; re-downloading.", flush=True)
            os.remove(zip_path)
        print(f"Downloading WISDM (~295 MB) from {url} ...", flush=True)
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.", flush=True)
        if not _zip_ok(zip_path):
            os.remove(zip_path)
            raise RuntimeError("Downloaded WISDM zip failed integrity check; deleted cache, retry later")
    with zipfile.ZipFile(zip_path) as z:
        # zip root contains wisdm-dataset/raw/... ; extract preserving tree
        z.extractall(dest)
    # find the raw dir wherever it landed
    for root, dirs, _ in os.walk(dest):
        if "raw" in dirs:
            return os.path.join(root, "raw")
    raise RuntimeError("raw/ directory not found after extracting WISDM zip")


def _activity_key(dest_root):
    """Parse activity_key.txt -> {code: name}. Falls back to known mapping."""
    fallback = {
        "A": "Walking", "B": "Jogging", "C": "Stairs", "D": "Sitting",
        "E": "Standing", "F": "Typing", "G": "BrushingTeeth",
        "H": "EatingSoup", "I": "EatingChips", "J": "EatingPasta",
        "K": "Drinking", "L": "EatingSandwich", "M": "Kicking",
        "O": "PlayingCatch", "P": "Dribbling", "Q": "Writing",
        "R": "Clapping", "S": "FoldingClothes",
    }
    for root, _, files in os.walk(dest_root):
        for fn in files:
            if fn.lower() == "activity_key.txt":
                mapping = {}
                with open(os.path.join(root, fn)) as f:
                    for line in f:
                        line = line.strip()
                        if not line or "=" not in line:
                            continue
                        code, name = line.split("=", 1)
                        code, name = code.strip(), name.strip()
                        if len(code) == 1:
                            mapping[code] = name.replace(" ", "")
                if mapping:
                    return mapping
    return fallback


def _read_sensor_file(path):
    """Parse one raw sensor file -> DataFrame[subject, activity, ts, x, y, z]."""
    df = pd.read_csv(path, header=None,
                     names=["subject", "activity", "ts", "x", "y", "z"],
                     dtype=str)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.rstrip(";").str.strip()
    df = df[(df["subject"] != "") & (df["x"] != "")]
    df = df.astype({"subject": int, "ts": np.int64,
                    "x": float, "y": float, "z": float})
    df["activity"] = df["activity"].astype(str)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def load_wisdm(data_root, win=128, stride=64, device="phone"):
    """Load WISDM, window the phone accel+gyro stream.

    Returns dict with X (N, win, 6), y (int), subjects (N,), activity_names,
    activity_codes. Channels: phone accel xyz + phone gyro xyz.
    """
    raw_dir = download_wisdm(data_root)
    dest_root = os.path.dirname(raw_dir.rstrip("/"))
    code2name = _activity_key(os.path.join(data_root, "wisdm"))

    accel_dir = os.path.join(raw_dir, device, "accel")
    gyro_dir = os.path.join(raw_dir, device, "gyro")
    subjects = sorted({int(fn.split("_")[1]) for fn in os.listdir(accel_dir)
                       if fn.startswith("data_")})

    Xs, ys, ss = [], [], []
    for subj in subjects:
        fa = os.path.join(accel_dir, f"data_{subj}_accel_{device}.txt")
        fg = os.path.join(gyro_dir, f"data_{subj}_gyro_{device}.txt")
        if not (os.path.exists(fa) and os.path.exists(fg)):
            continue
        try:
            da = _read_sensor_file(fa)
            dg = _read_sensor_file(fg)
        except Exception as e:
            print(f"  skip subject {subj}: {e}", flush=True)
            continue
        if len(da) < win or len(dg) < win:
            continue
        # nearest-timestamp merge of gyro onto accel timeline
        m = pd.merge_asof(da, dg, on="ts", direction="nearest",
                          suffixes=("", "_g"))
        stream = m[["x", "y", "z", "x_g", "y_g", "z_g"]].to_numpy(
            dtype=np.float64)
        acts = m["activity"].to_numpy()
        n = len(stream)
        for start in range(0, n - win + 1, stride):
            seg = stream[start:start + win]
            if not np.isfinite(seg).all():
                continue
            acodes = acts[start:start + win]
            vals, counts = np.unique(acodes, return_counts=True)
            maj = vals[int(np.argmax(counts))]
            Xs.append(seg)
            ys.append(maj)
            ss.append(subj)
    X = np.stack(Xs).astype(np.float32)
    codes = np.array(ys)
    subjects_arr = np.array(ss, dtype=np.int64)
    uniq_codes = sorted(set(codes.tolist()))
    code2idx = {c: i for i, c in enumerate(uniq_codes)}
    y = np.array([code2idx[c] for c in codes], dtype=np.int64)
    activity_names = [code2name.get(c, c) for c in uniq_codes]
    print(f"WISDM: X={X.shape} classes={len(uniq_codes)} "
          f"subjects={len(set(ss))}", flush=True)
    return {"X": X, "y": y, "subjects": subjects_arr,
            "activity_names": activity_names, "activity_codes": uniq_codes}


def wisdm_indices(d, n_test=9, n_val=7):
    """Subject-disjoint split: last subjects -> test, previous -> val."""
    subjects = np.array(sorted(set(d["subjects"].tolist())))
    te_subj = set(subjects[-n_test:].tolist())
    va_subj = set(subjects[-(n_test + n_val):-n_test].tolist())
    s = d["subjects"]
    te = np.where(np.isin(s, list(te_subj)))[0]
    va = np.where(np.isin(s, list(va_subj)))[0]
    tr = np.where(~np.isin(s, list(te_subj | va_subj)))[0]
    print(f"WISDM splits: train {len(tr)} / val {len(va)} / test {len(te)} "
          f"windows; subjects {len(subjects) - n_test - n_val}/"
          f"{n_val}/{n_test}", flush=True)
    return tr, va, te
