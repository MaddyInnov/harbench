"""Honest training protocol.

Rules enforced here:
  1. Splits are by subject -- callers pass explicit index arrays.
  2. Model selection (early stopping, best checkpoint) uses the VALIDATION
     split only. The test split is evaluated exactly once, after training.
  3. All randomness is seeded; seeds are recorded in the results.
  4. Standardization statistics come from the train split only.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import datasets as D
from . import metrics as M
from . import models as MOD
from .rejection import gamblers_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WindowDataset(Dataset):
    def __init__(self, X, y, soft_targets=None):
        # X: (N, T, C) -> torch wants (N, C, T)
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float().permute(0, 2, 1)
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.soft_targets = (torch.from_numpy(np.ascontiguousarray(soft_targets)).float()
                             if soft_targets is not None else None)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # With soft targets the loader yields (x, soft_vector, hard_label);
        # without them it yields (x, hard_label) as before.
        if self.soft_targets is not None:
            return self.X[i], self.soft_targets[i], self.y[i]
        return self.X[i], self.y[i]


def soft_cross_entropy(logits, targets):
    """Cross-entropy against a full target distribution (N, C)."""
    logp = torch.log_softmax(logits, dim=1)
    return -(targets * logp).sum(dim=1).mean()


DEFAULT_CFG = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "batch_size": 64,
    "epochs": 100,
    "patience": 20,          # early stopping on val macro-F1
    "t0": 10,                # CosineAnnealingWarmRestarts T_0
    "loss": "ce",            # 'ce' or 'focal'
    "focal_gamma": 2.0,
    "class_weight": None,    # None or 'balanced'
}


class TrainDataset(Dataset):
    """Training dataset with optional per-sample loss weights and purity
    targets. Always yields (x, y, w, p); w defaults to ones, p to zeros."""

    def __init__(self, X, y, weights=None, purity=None):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float().permute(0, 2, 1)
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
        n = len(self.y)
        w = np.ones(n, dtype=np.float32) if weights is None else np.asarray(
            weights, dtype=np.float32)
        p = np.zeros(n, dtype=np.float32) if purity is None else np.asarray(
            purity, dtype=np.float32)
        # Normalize weights to mean 1 so the loss scale stays comparable.
        w = w / (w.mean() + 1e-12)
        self.w = torch.from_numpy(np.ascontiguousarray(w)).float()
        self.p = torch.from_numpy(np.ascontiguousarray(p)).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i], self.p[i]


def augment_batch(x):
    """Light IMU augmentation: jitter + per-channel scaling."""
    if torch.rand(1, device=x.device).item() < 0.5:
        x = x + torch.randn_like(x) * 0.03
    if torch.rand(1, device=x.device).item() < 0.5:
        s = torch.randn(x.size(0), x.size(1), 1, device=x.device) * 0.1 + 1.0
        x = x * s
    return x


def augment_batch_strong(x):
    """Stronger IMU augmentation: light + time stretch + magnitude warp +
    small 3D rotations (per channel triplet) + channel dropout."""
    import torch.nn.functional as F
    x = augment_batch(x)
    B, C, T = x.shape
    dev = x.device
    # time stretch 0.8x--1.2x via resampling
    if torch.rand(1, device=dev).item() < 0.5:
        factor = 0.8 + torch.rand(1, device=dev).item() * 0.4
        T2 = max(int(T * factor), 8)
        xw = F.interpolate(x, size=T2, mode="linear", align_corners=False)
        if T2 >= T:
            start = torch.randint(0, T2 - T + 1, (1,)).item()
            x = xw[:, :, start:start + T]
        else:
            x = F.pad(xw, (0, T - T2), mode="reflect")
    # magnitude warp: smooth random scaling curve per channel
    if torch.rand(1, device=dev).item() < 0.5:
        knots = torch.randn(B, C, 4, device=dev) * 0.15 + 1.0
        curve = F.interpolate(knots, size=T, mode="linear",
                              align_corners=False)
        x = x * curve
    # small random rotation per channel triplet (e.g. accel xyz, gyro xyz)
    if C % 3 == 0 and torch.rand(1, device=dev).item() < 0.5:
        ang = (torch.rand(3, device=dev) - 0.5) * 0.6  # +-0.3 rad
        # small-angle rotation R ~= I + skew(ang)
        Rx = torch.tensor([[0, -ang[2], ang[1]],
                           [ang[2], 0, -ang[0]],
                           [-ang[1], ang[0], 0]], device=dev)
        R = torch.eye(3, device=dev) + Rx
        for g in range(C // 3):
            x[:, 3 * g:3 * g + 3, :] = torch.einsum("ij,bjt->bit", R,
                                                   x[:, 3 * g:3 * g + 3, :])
    # channel dropout: zero one random channel
    if torch.rand(1, device=dev).item() < 0.2:
        drop = torch.randint(0, C, (1,)).item()
        x[:, drop, :] = 0
    return x


def apply_augment(x, mode):
    if not mode or mode is False:
        return x
    if mode == "strong":
        return augment_batch_strong(x)
    return augment_batch(x)  # True / "light"


class FocalLoss(nn.Module):
    """Focal loss for hard-example mining (sitting vs standing)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        import torch.nn.functional as F
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


@torch.no_grad()
def predict(model, loader, device, n_eval_classes=None):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        xb, yb = batch[0], batch[-1]
        xb = xb.to(device)
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        k = n_eval_classes if n_eval_classes else logits.shape[1]
        preds.append(logits[:, :k].argmax(dim=1).cpu().numpy())
        trues.append(np.asarray(yb))
    return np.concatenate(trues), np.concatenate(preds)


@torch.no_grad()
def predict_proba_tta(model, loader, device, n_aug=4,
                      augment_mode="strong", n_eval_classes=None):
    """Test-time augmentation: average softmax over the original plus
    n_aug stochastically augmented passes. Returns (y_true, y_pred,
    mean_proba)."""
    import torch.nn.functional as F
    model.eval()
    trues, prob_sums, counts = [], None, 0
    n_passes = 1 + n_aug
    for p in range(n_passes):
        preds, t_, probas = [], [], []
        for batch in loader:
            xb, yb = batch[0], batch[-1]
            xb = xb.to(device)
            if p > 0:
                xb = apply_augment(xb, augment_mode)
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            proba = F.softmax(logits, dim=1)
            k = n_eval_classes if n_eval_classes else proba.shape[1]
            preds.append(proba[:, :k].argmax(dim=1).cpu().numpy())
            probas.append(proba[:, :k].cpu().numpy())
            if p == 0:
                t_.append(np.asarray(yb))
        if p == 0:
            trues = np.concatenate(t_)
            prob_sums = np.concatenate(probas).astype(np.float64)
        else:
            prob_sums += np.concatenate(probas)
        counts += 1
    mean_proba = prob_sums / counts
    y_pred = mean_proba.argmax(axis=1)
    return trues, y_pred, mean_proba


@torch.no_grad()
def predict_proba(model, loader, device, n_eval_classes=None):
    """Returns (y_true, y_pred, proba, logits).

    n_eval_classes: for abstain models (K+1 outputs), restrict the argmax
    to the first K classes; proba/logits keep all K+1 columns.
    """
    import torch.nn.functional as F
    model.eval()
    preds, trues, probas, logitss = [], [], [], []
    for batch in loader:
        xb, yb = batch[0], batch[-1]
        xb = xb.to(device)
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        proba = F.softmax(logits, dim=1)
        k = n_eval_classes if n_eval_classes else proba.shape[1]
        preds.append(proba[:, :k].argmax(dim=1).cpu().numpy())
        trues.append(np.asarray(yb))
        probas.append(proba.cpu().numpy())
        logitss.append(logits.cpu().numpy())
    return (np.concatenate(trues), np.concatenate(preds),
            np.concatenate(probas), np.concatenate(logitss))


def train_one_config(model_name, X, y, train_idx, val_idx, test_idx,
                     activity_names, cfg=None, seed=42, device=None,
                     verbose=True, train_soft_targets=None,
                     train_sample_weights=None, train_purity=None,
                     augment=False, abstain_o=None, tta_n_aug=0):
    """Train once; select on val; evaluate on test exactly once.

    train_soft_targets: optional (N_train, C) float array replacing the
    one-hot training targets (gap #1). Val/test always use hard labels.
    train_sample_weights: optional (N_train,) per-window loss weights
    (gap #2: purity weighting). Normalized to mean 1 internally.
    train_purity: optional (N_train,) window-purity targets for the
    multitask boundary head (model names ending in '-mt').
    augment: False/True/'light'/'strong' -- IMU augmentation level for
    training batches ('strong' adds time stretch, magnitude warp,
    rotations, channel dropout).
    abstain_o: Deep Gamblers reward for the abstain head (model names
    ending in '-abstain', K+1 outputs). None disables.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    multitask = model_name.endswith("-mt")
    abstain = abstain_o is not None
    n_classes = len(activity_names)

    Xtr, Xva, Xte = D.standardize_fit_apply(X[train_idx], X[val_idx], X[test_idx])
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]
    use_soft = train_soft_targets is not None

    if use_soft:
        # Gap #1 path: soft targets via WindowDataset (unchanged behavior).
        train_loader = DataLoader(
            WindowDataset(Xtr, ytr, soft_targets=train_soft_targets),
            batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    elif train_sample_weights is not None or train_purity is not None or augment:
        train_loader = DataLoader(
            TrainDataset(Xtr, ytr,
                         weights=train_sample_weights,
                         purity=train_purity),
            batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    else:
        train_loader = DataLoader(
            WindowDataset(Xtr, ytr),
            batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(WindowDataset(Xva, yva), batch_size=512)
    test_loader = DataLoader(WindowDataset(Xte, yte), batch_size=512)

    n_channels = X.shape[-1]
    model = MOD.build_model(model_name, n_channels, n_classes, seed=seed).to(device)
    n_params = MOD.count_parameters(model)

    # class weights for the boost runs (inverse-frequency, mean-normalized)
    class_weight = cfg.get("class_weight")
    if class_weight == "balanced":
        counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
        w = counts.sum() / (n_classes * np.maximum(counts, 1))
        class_weight = torch.as_tensor(w, dtype=torch.float32, device=device)
    elif class_weight is not None:
        class_weight = torch.as_tensor(np.asarray(class_weight),
                                       dtype=torch.float32, device=device)
    if cfg.get("loss", "ce") == "focal":
        criterion = FocalLoss(gamma=cfg.get("focal_gamma", 2.0),
                              weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg["label_smoothing"], weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["t0"])

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []

    ce_none = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"],
                                  weight=class_weight, reduction="none")
    focal_gamma = cfg.get("focal_gamma", 2.0)
    use_focal = cfg.get("loss", "ce") == "focal"

    def _per_sample_ce_focal(logits_, yb_):
        import torch.nn.functional as F
        ce_i = F.cross_entropy(logits_, yb_, weight=class_weight,
                               reduction="none")
        pt = torch.exp(-ce_i)
        return ((1 - pt) ** focal_gamma) * ce_i

    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            if abstain:
                # batch is (x, y) or (x, y, weights, purity); labels are
                # ALWAYS batch[1] -- batch[-1] is purity in the 4-tuple case.
                xb, yb = batch[0], batch[1]
                xb, yb = xb.to(device), yb.to(device)
                if augment:
                    xb = apply_augment(xb, augment)
                optimizer.zero_grad()
                logits = model(xb)  # (B, K+1)
                loss = gamblers_loss(logits, yb, reward_o=abstain_o)
            elif use_soft:
                xb, soft, _ = batch
                xb, soft = xb.to(device), soft.to(device)
                if augment:
                    xb = apply_augment(xb, augment)
                optimizer.zero_grad()
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                loss = soft_cross_entropy(logits, soft)
            elif len(batch) == 4:
                xb, yb, wb, pb = batch
                xb, yb = xb.to(device), yb.to(device)
                wb, pb = wb.to(device), pb.to(device)
                if augment:
                    xb = apply_augment(xb, augment)
                optimizer.zero_grad()
                out = model(xb)
                if multitask:
                    logits, p_pred = out
                    loss = (wb * ce_none(logits, yb)).mean() \
                        + 0.5 * nn.functional.mse_loss(p_pred, pb)
                else:
                    logits = out[0] if isinstance(out, tuple) else out
                    if use_focal:
                        loss = (wb * _per_sample_ce_focal(logits, yb)).mean()
                    else:
                        loss = (wb * ce_none(logits, yb)).mean()
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                if augment:
                    xb = apply_augment(xb, augment)
                optimizer.zero_grad()
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        _, val_pred = predict(model, val_loader, device,
                              n_eval_classes=n_classes if abstain else None)
        val_f1 = M.compute_metrics(yva, val_pred)["macro_f1"]
        history.append({"epoch": epoch + 1, "val_macro_f1": val_f1})
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}  val_macro_f1={val_f1:.4f} "
                  f"(best {best_val_f1:.4f})", flush=True)
        if epochs_no_improve >= cfg["patience"]:
            if verbose:
                print(f"  early stopping at epoch {epoch+1}", flush=True)
            break

    # Restore best-validation checkpoint, evaluate test ONCE.
    model.load_state_dict(best_state)
    n_eval = n_classes if abstain else None
    y_true, y_pred, proba, logits = predict_proba(model, test_loader, device,
                                                 n_eval_classes=n_eval)
    if abstain:
        # ECE on the K class probabilities renormalized (abstain excluded).
        proba_k = proba[:, :n_classes]
        proba_k = proba_k / proba_k.sum(axis=1, keepdims=True)
        test_metrics = M.compute_metrics(y_true, y_pred, activity_names)
        test_metrics["ece"] = M.expected_calibration_error(y_true, proba_k)
    else:
        test_metrics = M.compute_metrics(y_true, y_pred, activity_names)
        test_metrics["ece"] = M.expected_calibration_error(y_true, proba)

    out = {
        "model": model_name,
        "seed": seed,
        "n_params": n_params,
        "cfg": cfg,
        "use_soft_targets": use_soft,
        "abstain_o": abstain_o,
        "augment": augment,
        "best_val_macro_f1": best_val_f1,
        "epochs_trained": len(history),
        "test": test_metrics,
        "test_y_true": y_true.tolist(),
        "test_y_pred": y_pred.tolist(),
        "test_proba": proba.tolist(),
        "test_logits": logits.tolist(),
        "history": history,
    }
    if tta_n_aug > 0:
        _, y_tta, proba_tta = predict_proba_tta(
            model, test_loader, device, n_aug=tta_n_aug,
            augment_mode="strong" if augment else "light",
            n_eval_classes=n_eval)
        tta_met = M.compute_metrics(y_true, y_tta, activity_names)
        tta_met["ece"] = M.expected_calibration_error(y_true, proba_tta)
        out["test_tta"] = tta_met
    return out
