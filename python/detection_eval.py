"""
Évaluation détection/localisation pour AnoViT — plan §9.3 (seuil), §9.3bis (pilotage), §9.4 (métriques).

Contenu :
  - build_ela_cache()        : précalcul ELA -> PNG gris 384 (une fois, hors entraînement)
  - SyntheticDevDataset      : dev synthétique falsifié (ELA + masque 384, NEAREST)
  - AuthenticELADataset      : authentiques (pour FPR et Image AUROC), masque = zéros
  - anomaly_map()            : MAE / MSE / SSIM paramétrable (prépare E4)
  - evaluate()               : boucle d'inférence -> dict de métriques
  - pixel_auprc / pixel_auroc / aupro / dice_iou / calibrate_threshold
  - BestDetectionTracker     : checkpoint best-detection + historique (courbe AUPRC vs époques)

Dépendances : torch, numpy, pillow, scikit-learn, scipy.

Note layout : la génération écrit désormais les images dans `images/`, les masques
(+ .json) dans `masks/`, et une ELA RGB pré-calculée dans `ela/` (3 qualités empilées
en canaux, résolution native). Pour l'entraînement : `build_ela_cache(<...>/images, ...)`
(cache multi-qualité 384) et `SyntheticDevDataset(<...>/masks, cache, ...)` (1er arg =
dossier des masques).

IMPORTANT — qualités de sonde ELA : viser ≈ Q1 (base du fond = médiane(Q2)−Q1_GAP,
≈67 avec la config par défaut), PAS 90. Sonder à Q1 met le texte authentique à son
point fixe (ELA min) et fait exploser la zone falsifiée (mesuré : forgé/auth ~3.2 à
Q1 vs ~1.8 à Q90). Les 3 qualités E2 encadrent Q1 : (59, 67, 75). Aucune ne doit
égaler un Q2 du sweep.

Usage (boucle d'entraînement, pilotage §9.3bis) :

    build_ela_cache("output/images", "cache/dev", qualities=(59, 67, 75))  # ≈ Q1
    dev_ds  = SyntheticDevDataset("output/masks", "cache/dev", qualities=(67,))
    dev_ds  = pilot_subset(dev_ds, n=400, seed=42)          # sous-échantillon FIXE par époque
    dev_ld  = DataLoader(dev_ds, batch_size=48, num_workers=8, pin_memory=True)
    tracker = BestDetectionTracker("experiments/E0/best_model.pt",
                                   history_path="experiments/E0/auprc_curve.json", patience=15)

    for epoch in range(cfg.epochs_max):
        train_one_epoch(model, train_loader)
        res = evaluate(model, dev_ld, error_mode="mae", metrics=("auprc",))
        tracker.update(epoch, res["pixel_auprc"], model)
        if tracker.should_stop:
            break

Éval finale (dev complet, suite §9.4) :

    res = evaluate(model, dev_loader_full, error_mode="mae", metrics="full",
                   authentic_loader=auth_loader)            # seuil calibré sur dev, jamais sur test
    print(res)   # pixel_auprc, aupro, dice, iou, threshold, fpr_authentic, image_auroc, pixel_auroc
"""

from __future__ import annotations

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
from torch.utils.data import Dataset, Subset

# ---------------------------------------------------------------------------
# 1. Précalcul du cache ELA (une fois, hors entraînement)
# ---------------------------------------------------------------------------

ELA_SCALE = 15.0  # échelle GLOBALE unique — identique pour tous les splits (jamais par image)


def _ela_one(args):
    jpg_path, cache_dir, qualities, scale, img_size = args
    doc_id = os.path.splitext(os.path.basename(jpg_path))[0]
    img = Image.open(jpg_path).convert("RGB")            # résolution NATIVE
    arr = np.asarray(img, np.int16)
    for q in qualities:
        out = os.path.join(cache_dir, f"{doc_id}_q{q}.png")
        if os.path.exists(out):
            continue
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        rec = np.asarray(Image.open(buf), np.int16)
        ela = np.abs(arr - rec).mean(axis=2)             # RGB -> 1 canal
        ela = np.clip(ela * scale, 0, 255).astype(np.uint8)
        Image.fromarray(ela, "L").resize((img_size, img_size), Image.BILINEAR).save(out)
    return doc_id


def build_ela_cache(data_dir, cache_dir, qualities=(59, 67, 75),
                    scale=ELA_SCALE, img_size=384, workers=8):
    """ELA à résolution native -> échelle globale fixe -> resize -> PNG gris.
    Ne JAMAIS sauver en JPEG. Une passe pour les 3 qualités (≈ Q1) = E0/E1 (67) +
    E2 (59,67,75) servis. Sonder ≈ Q1, pas 90 (cf. en-tête module)."""
    os.makedirs(cache_dir, exist_ok=True)
    jpgs = sorted(glob(os.path.join(data_dir, "*.jpg")))
    jobs = [(p, cache_dir, qualities, scale, img_size) for p in jpgs]
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_ela_one, jobs, chunksize=16)):
            if (i + 1) % 200 == 0:
                print(f"ELA cache: {i + 1}/{len(jobs)}")
    print(f"ELA cache: {len(jobs)} docs -> {cache_dir}")


# ---------------------------------------------------------------------------
# 2. Datasets d'évaluation
# ---------------------------------------------------------------------------

def _load_ela_stack(cache_dir, doc_id, qualities):
    """(3, H, W) float32 [0,1] — 1 qualité répliquée x3 (E0/E1) ou 3 qualités empilées (E2)."""
    if len(qualities) not in (1, 3):
        raise ValueError("qualities doit contenir 1 (répliqué x3) ou 3 qualités")
    chans = []
    for q in qualities:
        p = os.path.join(cache_dir, f"{doc_id}_q{q}.png")
        chans.append(np.asarray(Image.open(p), np.float32) / 255.0)
    if len(chans) == 1:
        chans = chans * 3
    return torch.from_numpy(np.stack(chans, axis=0))


class SyntheticDevDataset(Dataset):
    """Dev synthétique falsifié annoté : ELA depuis le cache + masque binaire 384.

    data_dir  : dossier des MASQUES `{doc_id}_mask.png` (= `masks/` de la génération ;
                résolution native). Les images vivent dans `images/`, servies via le
                cache ELA (build_ela_cache), pas lues ici.
    cache_dir : dossier des PNG ELA {doc_id}_q{q}.png (384, gris) — cf. build_ela_cache
    qualities : (67,) pour E0/E1 ; (59, 67, 75) pour E2 (≈ Q1) <- ordre = config, ne jamais changer
    """

    def __init__(self, data_dir, cache_dir, qualities=(67,), img_size=384, doc_ids=None):
        self.cache_dir, self.qualities, self.img_size = cache_dir, tuple(qualities), img_size
        self.data_dir = data_dir
        if doc_ids is None:
            doc_ids = sorted(os.path.splitext(os.path.basename(p))[0].replace("_mask", "")
                             for p in glob(os.path.join(data_dir, "*_mask.png")))
        self.doc_ids = list(doc_ids)

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        x = _load_ela_stack(self.cache_dir, doc_id, self.qualities)
        m = Image.open(os.path.join(self.data_dir, f"{doc_id}_mask.png")).convert("L")
        m = m.resize((self.img_size, self.img_size), Image.NEAREST)     # NEAREST obligatoire
        mask = (np.asarray(m) > 127).astype(np.float32)
        return x, torch.from_numpy(mask)


class AuthenticELADataset(Dataset):
    """Documents authentiques (FPR §9.4, Image AUROC). Masque = zéros par construction."""

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


def pilot_subset(dataset, n, seed=42):
    """Sous-échantillon FIXE (mêmes docs à chaque époque) pour le pilotage par époque."""
    if n >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n, replace=False)
    return Subset(dataset, sorted(idx.tolist()))


# ---------------------------------------------------------------------------
# 3. Carte d'anomalie — MAE / MSE / SSIM (paramétrable, prépare E4)
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
    """(B, C, H, W) x2 -> (B, H, W). mode : 'mae' | 'mse' | 'ssim' (E4 tranchera le défaut)."""
    if mode == "mae":
        return (x - x_hat).abs().mean(dim=1)
    if mode == "mse":
        return ((x - x_hat) ** 2).mean(dim=1)
    if mode == "ssim":
        return (1.0 - _ssim_map(x, x_hat)).clamp(min=0).mean(dim=1)
    raise ValueError(f"error_mode inconnu : {mode}")


# ---------------------------------------------------------------------------
# 4. Métriques §9.4
# ---------------------------------------------------------------------------

_MAX_PIXELS = 20_000_000  # sous-échantillonnage pixel (seedé) pour sklearn sur gros volumes


def _flat_subsample(scores, masks, max_pixels=_MAX_PIXELS, seed=0):
    s, m = scores.ravel(), masks.ravel()
    if s.size <= max_pixels:
        return s, m
    idx = np.random.default_rng(seed).choice(s.size, size=max_pixels, replace=False)
    return s[idx], m[idx]


def pixel_auprc(scores, masks):
    """PRINCIPALE. Baseline hasard = taux de pixels positifs (pas 0.5)."""
    s, m = _flat_subsample(scores, masks)
    return float(average_precision_score(m.astype(np.uint8), s))


def pixel_auroc(scores, masks):
    """INDICATIVE seulement (gonflée par le déséquilibre pixel, §9.4)."""
    s, m = _flat_subsample(scores, masks)
    return float(roc_auc_score(m.astype(np.uint8), s))


def aupro(scores, masks, fpr_limit=0.3, num_bins=512):
    """AUPRO : aire sous (FPR, mean per-region overlap), FPR <= 0.3, normalisée.
    Implémentation par histogrammes (une passe sur les pixels), approx. à num_bins près."""
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
            comp_covs.append(np.cumsum(h[::-1])[::-1] / max(h.sum(), 1))  # couverture(seuil)
    if not comp_covs:
        return 0.0
    pro = np.mean(comp_covs, axis=0)                                       # (num_bins,)
    fpr = np.cumsum(neg_hist[::-1])[::-1] / max(neg_hist.sum(), 1)
    fpr_a, pro_a = fpr[::-1], pro[::-1]                                    # FPR croissant
    sel = fpr_a <= fpr_limit
    if sel.sum() < 2:
        return 0.0
    return float(np.trapz(pro_a[sel], fpr_a[sel]) / fpr_limit)


def calibrate_threshold(scores, masks):
    """§9.3 : seuil = max Dice sur le DEV SYNTHÉTIQUE (jamais sur le test réel). À figer/sauver."""
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
    """Pixels flaggés sur documents authentiques (le point industriel critique, E7)."""
    return float((auth_scores >= threshold).mean())


def image_scores(scores, q=0.99):
    """Score image = quantile q de la carte (plus robuste que max)."""
    return np.quantile(scores.reshape(scores.shape[0], -1), q, axis=1)


# ---------------------------------------------------------------------------
# 5. Boucle d'évaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect(model, loader, device, error_mode):
    """Retourne (scores (N,H,W) float32, masks (N,H,W) uint8)."""
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
    """Évalue sur le dev synthétique (et authentiques en option). Retourne un dict.

    metrics : ("auprc",) pour le pilotage par époque (rapide) — "full" pour la suite §9.4.
    threshold : None -> calibré sur le dev (§9.3) et retourné dans le dict (à figer ensuite).
    authentic_loader : requis pour fpr_authentic et image_auroc.
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


# ---------------------------------------------------------------------------
# 6. Pilotage best-detection (§9.3bis) + courbe AUPRC vs époques
# ---------------------------------------------------------------------------

class BestDetectionTracker:
    """Checkpoint = max AUPRC dev (PAS la dernière époque, PAS la loss de reconstruction)."""

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
