"""Opportunity Activity Recognition dataset (UCI ID 226).

4 subjects x 6 runs (5 ADL + 1 drill), 30 Hz, 7 IMUs + 12 3D accelerometers +
object/ambient sensors. Files are whitespace-separated .dat with 250 columns
and no header; column names come from column_names.txt in the zip.

Default task: locomotion track (Stand/Walk/Sit/Lie + null). Set
label_track="gesture" for the 18-class mid-level gesture task.

Subject-disjoint splits: train S1+S2, val S3, test S4 (all runs).
"""

import glob
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

OPP_URL = ("https://archive.ics.uci.edu/static/public/226/"
           "opportunity+activity+recognition.zip")
FS = 30  # Hz


def download_opportunity(data_root, url=OPP_URL):
    """Download and extract. Returns the directory containing the .dat files."""
    dest = os.path.join(data_root, "opportunity")
    marker = os.path.join(dest, ".extracted")
    if os.path.exists(marker):
        dats = glob.glob(os.path.join(dest, "**", "S*.dat"), recursive=True)
        if dats:
            return os.path.dirname(dats[0])
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "opportunity.zip")

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
            print("Cached Opportunity zip is corrupt/incomplete; re-downloading.", flush=True)
            os.remove(zip_path)
        print(f"Downloading Opportunity (~700 MB) from {url} ...", flush=True)
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.", flush=True)
        if not _zip_ok(zip_path):
            os.remove(zip_path)
            raise RuntimeError("Downloaded Opportunity zip failed integrity check; deleted cache, retry later")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    open(marker, "w").write("ok")
    dats = glob.glob(os.path.join(dest, "**", "S*.dat"), recursive=True)
    if not dats:
        raise RuntimeError("no S*.dat files found after extracting Opportunity")
    return os.path.dirname(dats[0])


def _column_names(dat_dir):
    """Locate column_names.txt near the dat files -> list of names."""
    cands = (glob.glob(os.path.join(dat_dir, "**", "column_names.txt"),
                       recursive=True)
             + glob.glob(os.path.join(os.path.dirname(dat_dir), "**",
                                      "column_names.txt"), recursive=True))
    for p in cands:
        with open(p) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if len(names) >= 200:
            return names
    return None


def _resolve_columns(names):
    """Find timestamp / locomotion / gesture / sensor columns by name."""
    low = [n.lower() for n in names]
    ts = next((i for i, n in enumerate(low) if "millisec" in n), 0)
    loco = next((i for i, n in enumerate(low)
                 if n in ("ml_both_arms", "locomotion")), None)
    gest = next((i for i, n in enumerate(low) if n == "label"), None)
    label_idx = {i for i in (loco, gest) if i is not None}
    sensor_idx = [i for i in range(len(names))
                  if i != ts and i not in label_idx]
    return ts, loco, gest, sensor_idx


def load_opportunity(data_root, label_track="locomotion", win=128, stride=64):
    """Load Opportunity, interpolate NaNs, window the sensor stream.

    Returns dict with X (N, win, C), y (int), subjects (N,), activity_names.
    """
    dat_dir = download_opportunity(data_root)
    names = _column_names(dat_dir)
    dat_files = sorted(glob.glob(os.path.join(dat_dir, "S*.dat")))
    print(f"Opportunity: {len(dat_files)} dat files", flush=True)

    Xs, ys, ss = [], [], []
    label_values = {}
    # Pass 1: find channels that are all-NaN in EVERY file (drop only those,
    # so the channel dimension stays consistent across files).
    file_meta = []
    drop_all = None
    for fp in dat_files:
        subj = int(os.path.basename(fp)[1])
        df = pd.read_csv(fp, sep=r"\s+", header=None)
        if names is not None and len(names) == df.shape[1]:
            ts_i, loco_i, gest_i, sensor_i = _resolve_columns(names)
        else:  # fallback to canonical layout: 250 cols
            ts_i, loco_i, gest_i = 0, 243, 249
            sensor_i = [i for i in range(df.shape[1])
                        if i not in (ts_i, loco_i, gest_i)]
        lab_i = loco_i if label_track == "locomotion" else gest_i
        if lab_i is None:
            raise RuntimeError(f"could not resolve {label_track} label column")
        sens = df.iloc[:, sensor_i].to_numpy(dtype=np.float64)
        allnan = np.isnan(sens).all(axis=0)
        drop_all = allnan if drop_all is None else (drop_all & allnan)
        file_meta.append((fp, subj, df, ts_i, lab_i, sensor_i))
    n_drop = int(drop_all.sum())
    print(f"Opportunity: dropping {n_drop} channels all-NaN in every file",
          flush=True)
    # Pass 2: apply the global channel mask, interpolate, window.
    for fp, subj, df, ts_i, lab_i, sensor_i in file_meta:
        sens = df.iloc[:, sensor_i].to_numpy(dtype=np.float64)
        sens = sens[:, ~drop_all]
        s = pd.DataFrame(sens).interpolate(limit=FS, limit_direction="both")
        sens = s.to_numpy()
        labels = df.iloc[:, lab_i].to_numpy()
        labels = np.where(np.isnan(labels), 0, labels).astype(np.int64)
        for v in np.unique(labels):
            label_values[int(v)] = label_values.get(int(v), 0) + 1
        n = len(sens)
        for start in range(0, n - win + 1, stride):
            seg = sens[start:start + win]
            if seg.shape != (win, sens.shape[1]) or not np.isfinite(seg).all():
                continue
            lab = labels[start:start + win]
            vals, counts = np.unique(lab, return_counts=True)
            maj = int(vals[int(np.argmax(counts))])
            Xs.append(seg)
            ys.append(maj)
            ss.append(subj)
    X = np.stack(Xs).astype(np.float32)
    y_raw = np.array(ys, dtype=np.int64)
    uniq = sorted(set(y_raw.tolist()))
    v2i = {v: i for i, v in enumerate(uniq)}
    y = np.array([v2i[v] for v in y_raw], dtype=np.int64)
    subjects = np.array(ss, dtype=np.int64)
    if label_track == "locomotion":
        canon = {0: "NULL", 1: "Stand", 2: "Walk", 3: "Sit", 4: "Lie"}
    else:
        canon = {0: "NULL"}
    activity_names = [canon.get(v, f"G{v}") for v in uniq]
    print(f"Opportunity/{label_track}: X={X.shape} classes={uniq} "
          f"subjects={sorted(set(ss))}", flush=True)
    print(f"  label value counts: {label_values}", flush=True)
    return {"X": X, "y": y, "subjects": subjects,
            "activity_names": activity_names,
            "label_values": sorted(uniq)}


def opportunity_indices(d):
    """Subject-disjoint: train S1+S2, val S3, test S4."""
    s = d["subjects"]
    tr = np.where(np.isin(s, [1, 2]))[0]
    va = np.where(s == 3)[0]
    te = np.where(s == 4)[0]
    print(f"Opportunity splits: train {len(tr)} / val {len(va)} / "
          f"test {len(te)} windows", flush=True)
    return tr, va, te
