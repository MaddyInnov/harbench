"""Driver: 4-model comparison on HAR datasets (honest protocol).

Models: tinyhar (reference), moment, gated-deltanet, tinyhar-grkan, rwkv-ts.
Datasets: uci, pamap2, wisdm, opportunity (locomotion).
3 seeds each, strong augmentation for all, subject-disjoint splits,
validation-only selection, test evaluated once.

Usage: python3 driver_compare.py [uci,pamap2|wisdm,opportunity|all] [model1,model2] [seed1,seed2]
Saves incrementally to /kaggle/working/results_compare_<tag>.json
The optional model and seed filters select a CARTESIAN subset to run:
pass EXACTLY the missing (model, seed) pairs, because completed pairs are
NOT skipped automatically -- a broad filter re-burns quota re-running them.
E.g. to resume only gated-deltanet seeds 43,44:
    python3 driver_compare.py pamap2 gated-deltanet 43,44
"""

import json
import platform
import subprocess
import sys
import time

sys.path.insert(0, "/kaggle/working")

import numpy as np
import torch

from harbench import train as T
from harbench import metrics as M
from harbench import soft_labels as SL

DATA_ROOT = "/kaggle/working/data"
SEEDS = [42, 43, 44]
MODELS = ["tinyhar", "moment", "gated-deltanet", "tinyhar-grkan", "rwkv-ts", "inceptionkan"]
MODEL_CFG = {
    "moment": {"lr": 5e-5, "batch_size": 32},
}


def slim(r):
    return {
        "seed": r["seed"],
        "n_params": r["n_params"],
        "best_val_macro_f1": r["best_val_macro_f1"],
        "epochs_trained": r["epochs_trained"],
        "test_accuracy": r["test"]["accuracy"],
        "test_macro_f1": r["test"]["macro_f1"],
        "test_ece": r["test"].get("ece"),
        "per_class_f1": r["test"].get("per_class_f1"),
        "wall_seconds": r.get("wall_seconds"),
    }


def run_dataset(tag, X, y, tr, va, te, activity_names, device, store,
                out_path_global):
    print(f"\n===== dataset {tag}: X={X.shape} "
          f"classes={len(activity_names)} =====", flush=True)
    ds = {"tag": tag, "X_shape": list(X.shape),
          "n_classes": len(activity_names), "activity_names": activity_names,
          "models": {}}
    for model in MODELS:
        recs = []
        for seed in SEEDS:
            print(f"--- {tag} | {model} | seed {seed} ---", flush=True)
            t0 = time.time()
            try:
                r = T.train_one_config(
                    model, X, y, tr, va, te, activity_names,
                    cfg=MODEL_CFG.get(model), seed=seed, verbose=False,
                    augment="strong", device=device)
                r["wall_seconds"] = round(time.time() - t0, 1)
                t = r["test"]
                print(f"    acc={t['accuracy']:.4f} f1={t['macro_f1']:.4f} "
                      f"params={r['n_params']:,} [{r['wall_seconds']}s]",
                      flush=True)
                recs.append(slim(r))
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                recs.append({"seed": seed, "error": f"{type(e).__name__}: {e}"})
        ok = [x for x in recs if "test_accuracy" in x]
        if ok:
            acc = np.array([x["test_accuracy"] for x in ok])
            f1 = np.array([x["test_macro_f1"] for x in ok])
            print(f"  {model}: acc={acc.mean():.4f}+/-{acc.std():.4f} "
                  f"f1={f1.mean():.4f}+/-{f1.std():.4f}", flush=True)
        ds["models"][model] = recs
        store["datasets"][tag] = ds
        with open(out_path_global, "w") as f:
            json.dump({k: v for k, v in store.items() if k != "_path"},
                      f, indent=1)
    return ds


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tag = which.replace(",", "_")
    if len(sys.argv) > 2:
        want_models = [m.strip() for m in sys.argv[2].split(",") if m.strip()]
        unknown = [m for m in want_models if m not in MODELS]
        if unknown:
            raise SystemExit(f"unknown models: {unknown} (choose from {MODELS})")
        MODELS[:] = want_models
        tag += "_" + "_".join(want_models).replace("-", "")
    if len(sys.argv) > 3:
        want_seeds = [int(s.strip()) for s in sys.argv[3].split(",")
                      if s.strip()]
        unknown = [s for s in want_seeds if s not in SEEDS]
        if unknown:
            raise SystemExit(f"unknown seeds: {unknown} (choose from {SEEDS})")
        SEEDS[:] = want_seeds
        tag += "_seeds" + "".join(str(s) for s in want_seeds)
    out_path = f"/kaggle/working/results_compare_{tag}.json"
    run_info = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
        else None,
        "platform": platform.platform(),
        "models": MODELS,
        "seeds": SEEDS,
    }
    print("Environment:", json.dumps(run_info, indent=1), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MOMENT dependency (only needed if 'moment' in MODELS)
    if "moment" in MODELS:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "momentfm"])
        if r.returncode != 0:
            print("full momentfm install failed, retrying --no-deps",
                  flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "--no-deps", "momentfm"], check=True)
        try:
            import momentfm  # noqa: F401
            print("momentfm available", flush=True)
        except ImportError as e:
            print(f"momentfm import failed ({e}); moment runs will be "
                  f"skipped", flush=True)
            MODELS.remove("moment")

    store = {"run_info": run_info, "datasets": {}, "_path": out_path}

    from harbench import datasets as D
    from harbench import pamap2 as P
    from harbench import wisdm as W
    from harbench import opportunity as O

    want = set(which.split(","))
    if "uci" in want or "all" in want:
        du = D.load_uci_har(DATA_ROOT)
        tr, va, te = D.uci_har_indices(du)
        run_dataset("uci", du["X"], du["y"], tr, va, te,
                    du["activity_names"], device, store, out_path)
    if "pamap2" in want or "all" in want:
        dp = P.load_pamap2(DATA_ROOT)
        y = SL.majority_labels(dp["label_seqs"])
        tr, va, te = P.pamap2_indices(dp)
        run_dataset("pamap2", dp["X"], y, tr, va, te,
                    dp["activity_names"], device, store, out_path)
    if "wisdm" in want or "all" in want:
        dw = W.load_wisdm(DATA_ROOT, win=128, stride=64)
        tr, va, te = W.wisdm_indices(dw)
        run_dataset("wisdm", dw["X"], dw["y"], tr, va, te,
                    dw["activity_names"], device, store, out_path)
    if "opportunity" in want or "all" in want:
        do = O.load_opportunity(DATA_ROOT, label_track="locomotion",
                                win=128, stride=64)
        tr, va, te = O.opportunity_indices(do)
        run_dataset("opportunity", do["X"], do["y"], tr, va, te,
                    do["activity_names"], device, store, out_path)

    print("\n================ FINAL SUMMARY ================")
    for dtag, ds in store["datasets"].items():
        for m, recs in ds["models"].items():
            ok = [x for x in recs if "test_accuracy" in x]
            if ok:
                acc = np.array([x["test_accuracy"] for x in ok])
                f1 = np.array([x["test_macro_f1"] for x in ok])
                print(f"{dtag:12s} {m:15s} acc={acc.mean():.4f}+/-"
                      f"{acc.std():.4f} f1={f1.mean():.4f}+/-{f1.std():.4f} "
                      f"({len(ok)}/{len(recs)} ok)")
            else:
                print(f"{dtag:12s} {m:15s} ALL FAILED")
    print("===============================================")
    store.pop("_path", None)
    with open(out_path, "w") as f:
        json.dump(store, f, indent=1)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
