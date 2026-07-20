"""
Detection/localization evaluation for AnoViT — plan §9.3 (threshold), §9.3bis (model selection), §9.4 (metrics).

Contents:
  - build_ela_cache()        : ELA precompute -> grayscale 384 PNG (once, outside training)
  - SyntheticDevDataset      : synthetic forged dev (ELA + 384 mask, NEAREST)
  - AuthenticELADataset      : authentics (for FPR and Image AUROC), mask = zeros
  - anomaly_map()            : configurable MAE / MSE / SSIM (prepares E4)
  - evaluate()               : inference loop -> dict of metrics
  - pixel_auprc / pixel_auroc / aupro / dice_iou / calibrate_threshold
  - BestDetectionTracker     : best-detection checkpoint + history (AUPRC vs epochs curve)

Dependencies: torch, numpy, pillow, scikit-learn, scipy.

Layout note: generation now writes the images to `images/`, the masks (+ .json) to
`masks/`, and a precomputed RGB ELA to `ela/` (3 qualities stacked as channels, native
resolution). For training: `build_ela_cache(<...>/images, ...)` (multi-quality 384
cache) and `SyntheticDevDataset(<...>/masks, cache, ...)` (1st arg = masks folder).

IMPORTANT — ELA probe qualities: aim for ≈ Q1 (background base = median(Q2)−Q1_GAP,
≈67 with the default config), NOT 90. Probing at Q1 puts authentic text at its fixed
point (min ELA) and makes the forged region explode (measured: forged/auth ~3.2 at Q1
vs ~1.8 at Q90). The 3 E2 qualities bracket Q1: (59, 67, 75). None may equal a Q2 from
the sweep.

Usage (training loop, model selection §9.3bis):

    build_ela_cache("output/images", "cache/dev", qualities=(59, 67, 75))  # ≈ Q1
    dev_ds  = SyntheticDevDataset("output/masks", "cache/dev", qualities=(67,))
    dev_ds  = pilot_subset(dev_ds, n=400, seed=42)          # FIXED subsample per epoch
    dev_ld  = DataLoader(dev_ds, batch_size=48, num_workers=8, pin_memory=True)
    tracker = BestDetectionTracker("experiments/E0/best_model.pt",
                                   history_path="experiments/E0/auprc_curve.json", patience=15)

    for epoch in range(cfg.epochs_max):
        train_one_epoch(model, train_loader)
        res = evaluate(model, dev_ld, error_mode="mae", metrics=("auprc",))
        tracker.update(epoch, res["pixel_auprc"], model)
        if tracker.should_stop:
            break

Final eval (full dev, §9.4):

    res = evaluate(model, dev_loader_full, error_mode="mae", metrics="full",
                   authentic_loader=auth_loader)            # threshold calibrated on dev, never on test
    print(res)   # pixel_auprc, aupro, dice, iou, threshold, fpr_authentic, image_auroc, pixel_auroc
"""

from __future__ import annotations

import csv
import io
import json
import os
from glob import glob
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# 1. ELA cache precompute (once, outside training)
# ---------------------------------------------------------------------------

ELA_SCALE = 15.0  # single GLOBAL scale — identical for all splits (never per image)


def _ela_one(args):
    jpg_path, cache_dir, qualities, scale, img_size = args
    doc_id = os.path.splitext(os.path.basename(jpg_path))[0]
    img = Image.open(jpg_path).convert("RGB")            # NATIVE resolution
    arr = np.asarray(img, np.int16)
    for q in qualities:
        out = os.path.join(cache_dir, f"{doc_id}_q{q}.png")
        if os.path.exists(out):
            continue
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        rec = np.asarray(Image.open(buf), np.int16)
        ela = np.abs(arr - rec).mean(axis=2)             # RGB -> 1 channel
        ela = np.clip(ela * scale, 0, 255).astype(np.uint8)
        Image.fromarray(ela, "L").resize((img_size, img_size), Image.BILINEAR).save(out)
    return doc_id


def build_ela_cache(data_dir, cache_dir, qualities=(59, 67, 75),
                    scale=ELA_SCALE, img_size=384, workers=8):
    """ELA at native resolution -> fixed global scale -> resize -> grayscale PNG.
    NEVER save as JPEG. One pass for the 3 qualities (≈ Q1) serves E0/E1 (67) +
    E2 (59,67,75). Probe ≈ Q1, not 90 (cf. module header)."""
    os.makedirs(cache_dir, exist_ok=True)
    jpgs = sorted(glob(os.path.join(data_dir, "*.jpg")))
    jobs = [(p, cache_dir, qualities, scale, img_size) for p in jpgs]
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_ela_one, jobs, chunksize=16)):
            if (i + 1) % 200 == 0:
                print(f"ELA cache: {i + 1}/{len(jobs)}")
    print(f"ELA cache: {len(jobs)} docs -> {cache_dir}")


# ---------------------------------------------------------------------------
# 2. Evaluation datasets
# ---------------------------------------------------------------------------

def _load_ela_stack(cache_dir, doc_id, qualities):
    """(3, H, W) float32 [0,1] — 1 quality replicated x3 (E0/E1) or 3 qualities stacked (E2)."""
    if len(qualities) not in (1, 3):
        raise ValueError("qualities must contain 1 (replicated x3) or 3 qualities")
    chans = []
    for q in qualities:
        p = os.path.join(cache_dir, f"{doc_id}_q{q}.png")
        chans.append(np.asarray(Image.open(p), np.float32) / 255.0)
    if len(chans) == 1:
        chans = chans * 3
    return torch.from_numpy(np.stack(chans, axis=0))


def _docid_to_mask(doc_id):
    """doc_id `{stem}_{n}` -> mask name `{stem}_mask_{n}.png` (generation schema)."""
    stem, n = doc_id.rsplit("_", 1)
    return f"{stem}_mask_{n}.png"


def _mask_to_docid(fname):
    """`{stem}_mask_{n}.png` -> doc_id `{stem}_{n}` (== image stem / ELA cache key)."""
    stem, n = os.path.basename(fname)[:-4].rsplit("_mask_", 1)
    return f"{stem}_{n}"


class SyntheticDevDataset(Dataset):
    """Annotated synthetic forged dev: ELA from the cache + 384 binary mask.

    data_dir  : folder of MASKS `{stem}_mask_{n}.png` (= generation's `masks/`;
                native resolution, n = number of forgeries). The images live in
                `images/` ({stem}_{n}.jpg), served via the ELA cache, not read here.
    cache_dir : folder of ELA PNGs {doc_id}_q{q}.png (384, grayscale) — cf. build_ela_cache
    qualities : (67,) for E0/E1; (59, 67, 75) for E2 (≈ Q1) <- order = config, never change
    """

    def __init__(self, data_dir, cache_dir, qualities=(67,), img_size=384, doc_ids=None):
        self.cache_dir, self.qualities, self.img_size = cache_dir, tuple(qualities), img_size
        self.data_dir = data_dir
        if doc_ids is None:
            doc_ids = sorted(_mask_to_docid(p)
                             for p in glob(os.path.join(data_dir, "*_mask_*.png")))
        self.doc_ids = list(doc_ids)

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        x = _load_ela_stack(self.cache_dir, doc_id, self.qualities)
        m = Image.open(os.path.join(self.data_dir, _docid_to_mask(doc_id))).convert("L")
        m = m.resize((self.img_size, self.img_size), Image.NEAREST)     # NEAREST mandatory
        mask = (np.asarray(m) > 127).astype(np.float32)
        return x, torch.from_numpy(mask)


class AuthenticELADataset(Dataset):
    """Authentic documents (FPR §9.4, Image AUROC). Mask = zeros by construction."""

    def __init__(self, cache_dir, qualities=(67,), img_size=384, doc_ids=None):
        self.cache_dir, self.qualities, self.img_size = cache_dir, tuple(qualities), img_size
        if doc_ids is None:
            q0 = self.qualities[0]
            doc_ids = sorted(os.path.basename(p)[: -len(f"_q{q0}.png")]
                             for p in glob(os.path.join(cache_dir, f"*_q{q0}.png")))
        self.doc_ids = list(doc_ids)

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        x = _load_ela_stack(self.cache_dir, self.doc_ids[idx], self.qualities)
        return x, torch.zeros(self.img_size, self.img_size)


def _is_negative(v):
    return str(v).strip().lower() in ("true", "1", "yes", "y", "t")


class CSVELADataset(Dataset):
    """Dev/test driven by a flat CSV manifest instead of the folder+cache layout.

    Columns (dataset.csv): `id, type, x_path, negative, mask_path`, where
      x_path    : path to the PRECOMPUTED ELA RGB image (generation's `ela/*_ela.png`,
                  3 qualities ≈ Q1 stacked as channels) — loaded as-is, so NO ELA cache
                  is built here (skip build_ela_cache entirely for this path).
      negative  : truthy -> authentic doc (mask forced to zeros, mask_path ignored).
      mask_path : native-resolution binary mask for a forged doc (NEAREST -> img_size).

    only : None (all rows) | 'forged' (negative falsy) | 'authentic' (negative truthy).
           Split a single manifest into the two loaders `evaluate` expects.

    NOTE: x must match the ELA representation the model was TRAINED on (here RGB 384,
    /255). Adjust the normalization line if training used a different mean/std.
    """

    def __init__(self, csv_path, img_size=384, only=None):
        self.img_size = img_size
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if only == "forged":
            rows = [r for r in rows if not _is_negative(r["negative"])]
        elif only == "authentic":
            rows = [r for r in rows if _is_negative(r["negative"])]
        elif only is not None:
            raise ValueError("only must be None | 'forged' | 'authentic'")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ela = Image.open(r["x_path"]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)          # already 384 -> no-op
        x = torch.from_numpy(np.asarray(ela, np.float32).transpose(2, 0, 1) / 255.0)
        if _is_negative(r["negative"]) or not r.get("mask_path"):
            mask = torch.zeros(self.img_size, self.img_size)
        else:
            m = Image.open(r["mask_path"]).convert("L").resize(
                (self.img_size, self.img_size), Image.NEAREST)       # NEAREST mandatory
            mask = torch.from_numpy((np.asarray(m) > 127).astype(np.float32))
        return x, mask


def pilot_subset(dataset, n, seed=42):
    """FIXED subsample (same docs at each epoch) for the per-epoch model selection."""
    if n >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n, replace=False)
    return Subset(dataset, sorted(idx.tolist()))


# ---------------------------------------------------------------------------
# 3. Anomaly map — MAE / MSE / SSIM (configurable, prepares E4)
# ---------------------------------------------------------------------------

def _gaussian_kernel(win=11, sigma=1.5, channels=3, device="cpu"):
    ax = torch.arange(win, dtype=torch.float32, device=device) - (win - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = (g[:, None] * g[None, :]) / (g.sum() ** 2)
    return k.expand(channels, 1, win, win).contiguous()


def _ssim_map(x, y, win=11, sigma=1.5):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    k = _gaussian_kernel(win, sigma, x.shape[1], x.device)
    pad, g = win // 2, x.shape[1]
    mu_x = F.conv2d(x, k, padding=pad, groups=g)
    mu_y = F.conv2d(y, k, padding=pad, groups=g)
    sx = F.conv2d(x * x, k, padding=pad, groups=g) - mu_x ** 2
    sy = F.conv2d(y * y, k, padding=pad, groups=g) - mu_y ** 2
    sxy = F.conv2d(x * y, k, padding=pad, groups=g) - mu_x * mu_y
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))


def anomaly_map(x, x_hat, mode="mae"):
    """(B, C, H, W) x2 -> (B, H, W). mode: 'mae' | 'mse' | 'ssim' (E4 will decide the default)."""
    if mode == "mae":
        return (x - x_hat).abs().mean(dim=1)
    if mode == "mse":
        return ((x - x_hat) ** 2).mean(dim=1)
    if mode == "ssim":
        return (1.0 - _ssim_map(x, x_hat)).clamp(min=0).mean(dim=1)
    raise ValueError(f"unknown error_mode: {mode}")


# ---------------------------------------------------------------------------
# 4. Metrics §9.4
# ---------------------------------------------------------------------------

_MAX_PIXELS = 20_000_000  # pixel subsampling (seeded) for sklearn on large volumes


def _flat_subsample(scores, masks, max_pixels=_MAX_PIXELS, seed=0):
    s, m = scores.ravel(), masks.ravel()
    if s.size <= max_pixels:
        return s, m
    idx = np.random.default_rng(seed).choice(s.size, size=max_pixels, replace=False)
    return s[idx], m[idx]


def pixel_auprc(scores, masks):
    """MAIN metric. Chance baseline = positive-pixel rate (not 0.5)."""
    s, m = _flat_subsample(scores, masks)
    return float(average_precision_score(m.astype(np.uint8), s))


def pixel_auroc(scores, masks):
    """INDICATIVE only (inflated by the pixel imbalance, §9.4)."""
    s, m = _flat_subsample(scores, masks)
    return float(roc_auc_score(m.astype(np.uint8), s))


def aupro(scores, masks, fpr_limit=0.3, num_bins=512):
    """AUPRO: area under (FPR, mean per-region overlap), FPR <= 0.3, normalized.
    Histogram-based implementation (one pass over the pixels), approximate to num_bins."""
    smin, smax = float(scores.min()), float(scores.max())
    if smax <= smin:
        return 0.0
    edges = np.linspace(smin, smax, num_bins + 1)
    neg_hist = np.zeros(num_bins, np.int64)
    comp_covs = []
    for sc, mk in zip(scores, masks):
        mk = mk.astype(bool)
        neg_hist += np.histogram(sc[~mk], bins=edges)[0]
        lbl, ncomp = ndimage.label(mk)
        for c in range(1, ncomp + 1):
            h = np.histogram(sc[lbl == c], bins=edges)[0]
            comp_covs.append(np.cumsum(h[::-1])[::-1] / max(h.sum(), 1))  # coverage(threshold)
    if not comp_covs:
        return 0.0
    pro = np.mean(comp_covs, axis=0)                                       # (num_bins,)
    fpr = np.cumsum(neg_hist[::-1])[::-1] / max(neg_hist.sum(), 1)
    fpr_a, pro_a = fpr[::-1], pro[::-1]                                    # increasing FPR
    sel = fpr_a <= fpr_limit
    if sel.sum() < 2:
        return 0.0
    return float(np.trapz(pro_a[sel], fpr_a[sel]) / fpr_limit)


def calibrate_threshold(scores, masks):
    """§9.3: threshold = max Dice on the SYNTHETIC DEV (never on the real test). Freeze/save it."""
    s, m = _flat_subsample(scores, masks)
    prec, rec, thr = precision_recall_curve(m.astype(np.uint8), s)
    dice = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(dice[:-1]))])


def dice_iou(scores, masks, threshold):
    pred = scores >= threshold
    gt = masks.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    return {"dice": 2 * tp / max(2 * tp + fp + fn, 1e-12),
            "iou": tp / max(tp + fp + fn, 1e-12)}


def fpr_authentic(auth_scores, threshold):
    """Pixels flagged on authentic documents (the critical industrial point, E7)."""
    return float((auth_scores >= threshold).mean())


def image_scores(scores, q=0.99):
    """Image score = quantile q of the map (more robust than max)."""
    return np.quantile(scores.reshape(scores.shape[0], -1), q, axis=1)


# ---------------------------------------------------------------------------
# 5. Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect(model, loader, device, error_mode):
    """Returns (scores (N,H,W) float32, masks (N,H,W) uint8)."""
    model.eval()
    all_s, all_m = [], []
    for batch in loader:
        x, mask = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        out = model(x)
        x_hat = out[0] if isinstance(out, (tuple, list)) else out
        all_s.append(anomaly_map(x, x_hat, error_mode).float().cpu().numpy())
        all_m.append(mask.numpy())
    return np.concatenate(all_s), (np.concatenate(all_m) > 0.5).astype(np.uint8)


@torch.no_grad()
def evaluate(model, dev_loader, device=None, error_mode="mae",
             metrics=("auprc",), threshold=None, authentic_loader=None):
    """Evaluate on the synthetic dev (and authentics, optionally). Returns a dict.

    metrics : ("auprc",) for the per-epoch model selection (fast) — "full" for §9.4.
    threshold : None -> calibrated on the dev (§9.3) and returned in the dict (freeze it after).
    authentic_loader : required for fpr_authentic and image_auroc.
    """
    if device is None:
        device = next(model.parameters()).device
    if metrics == "full":
        metrics = ("auprc", "aupro", "dice_iou", "fpr", "image_auroc", "pixel_auroc")

    scores, masks = _collect(model, dev_loader, device, error_mode)
    res = {"error_mode": error_mode, "n_dev": int(scores.shape[0])}

    if "auprc" in metrics:
        res["pixel_auprc"] = pixel_auprc(scores, masks)
    if "pixel_auroc" in metrics:
        res["pixel_auroc"] = pixel_auroc(scores, masks)
    if "aupro" in metrics:
        res["aupro"] = aupro(scores, masks)

    if "dice_iou" in metrics or "fpr" in metrics:
        if threshold is None:
            threshold = calibrate_threshold(scores, masks)
        res["threshold"] = threshold
    if "dice_iou" in metrics:
        res.update(dice_iou(scores, masks, threshold))

    if authentic_loader is not None and ("fpr" in metrics or "image_auroc" in metrics):
        auth_scores, _ = _collect(model, authentic_loader, device, error_mode)
        if "fpr" in metrics:
            res["fpr_authentic"] = fpr_authentic(auth_scores, threshold)
        if "image_auroc" in metrics:
            y = np.r_[np.ones(scores.shape[0]), np.zeros(auth_scores.shape[0])]
            s = np.r_[image_scores(scores), image_scores(auth_scores)]
            res["image_auroc"] = float(roc_auc_score(y, s))
    return res


@torch.no_grad()
def evaluate_from_csv(model, csv_path, device=None, error_mode="mae",
                      metrics="full", threshold=None, img_size=384,
                      batch_size=48, num_workers=8):
    """One-call eval from a CSV manifest (cf. CSVELADataset). Splits the manifest
    into forged (negative falsy) and authentic (negative truthy) loaders and calls
    evaluate(). If the manifest has no authentic rows, image_auroc/fpr_authentic are
    skipped (localization metrics still computed).

    VAL/TEST discipline: on the val manifest leave threshold=None (calibrated, returned
    in the dict); on the test manifest pass that frozen value as `threshold=`."""
    def _loader(only):
        ds = CSVELADataset(csv_path, img_size=img_size, only=only)
        return ds, DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    forged_ds, forged_ld = _loader("forged")
    auth_ds, auth_ld = _loader("authentic")
    if len(forged_ds) == 0:
        raise ValueError(f"{csv_path}: no forged rows (negative falsy) to evaluate")
    return evaluate(model, forged_ld, device=device, error_mode=error_mode,
                    metrics=metrics, threshold=threshold,
                    authentic_loader=auth_ld if len(auth_ds) else None)


# ---------------------------------------------------------------------------
# 6. Best-detection model selection (§9.3bis) + AUPRC vs epochs curve
# ---------------------------------------------------------------------------

class BestDetectionTracker:
    """Checkpoint = max dev AUPRC (NOT the last epoch, NOT the reconstruction loss)."""

    def __init__(self, ckpt_path, history_path=None, patience=None):
        self.ckpt_path, self.history_path, self.patience = ckpt_path, history_path, patience
        self.best, self.best_epoch, self.history = -1.0, -1, []
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    def update(self, epoch, auprc, model):
        self.history.append({"epoch": epoch, "auprc": float(auprc)})
        improved = auprc > self.best
        if improved:
            self.best, self.best_epoch = float(auprc), epoch
            torch.save(model.state_dict(), self.ckpt_path)
        if self.history_path:
            with open(self.history_path, "w") as f:
                json.dump({"best_epoch": self.best_epoch, "best_auprc": self.best,
                           "curve": self.history}, f, indent=2)
        return improved

    @property
    def should_stop(self):
        if self.patience is None or not self.history:
            return False
        return (self.history[-1]["epoch"] - self.best_epoch) >= self.patience
