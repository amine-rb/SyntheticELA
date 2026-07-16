# `detection_eval.py` — Évaluation détection/localisation AnoViT

Module d'évaluation à plugger dans le codebase d'entraînement. Implémente le protocole du plan :
**§9.3** (calibration du seuil), **§9.3bis** (pilotage best-detection), **§9.4** (métriques finales).

---

## 1. Installation

Copier `detection_eval.py` à la racine du codebase d'entraînement, puis :

```bash
pip install scikit-learn scipy   # torch, numpy, pillow déjà présents
```

---

## 2. Préparer le cache ELA (une seule fois, hors entraînement)

```python
from detection_eval import build_ela_cache

# dev synthétique (falsifiés annotés)
build_ela_cache("output/data", "cache/dev", qualities=(75, 85, 90, 95), workers=8)

# authentiques d'éval (pour FPR / Image AUROC)
build_ela_cache("chemin/vers/authentiques", "cache/auth", qualities=(75, 85, 90, 95), workers=8)
```

Règles appliquées automatiquement (ne pas contourner) :
- ELA calculée à **résolution native**, puis resize 384 (préserve la grille JPEG 8×8)
- Échelle **globale unique** `ELA_SCALE = 15.0` — la même partout (train/dev/test), jamais par image
- Sauvegarde en **PNG gris** (jamais JPEG — ça détruirait le signal ELA)
- Une passe pour les 4 qualités → le même cache sert E0, E1 **et** E2

> ⚠️ Si ton cache d'entraînement existe déjà avec une autre échelle, aligne `ELA_SCALE` dessus.

Layout produit : `cache/dev/{doc_id}_q{75|85|90|95}.png`

---

## 3. Construire les loaders d'évaluation

```python
from torch.utils.data import DataLoader
from detection_eval import SyntheticDevDataset, AuthenticELADataset, pilot_subset

# E0 / E1 : une qualité, répliquée x3 canaux -> (3, 384, 384)
dev_ds = SyntheticDevDataset("output/data", "cache/dev", qualities=(90,))

# E2 : trois qualités empilées (l'ordre = la config, ne jamais le changer)
# dev_ds = SyntheticDevDataset("output/data", "cache/dev", qualities=(75, 85, 95))

# Pilotage par époque : sous-échantillon FIXE (mêmes docs à chaque époque, seedé)
dev_pilot = pilot_subset(dev_ds, n=400, seed=42)
dev_loader = DataLoader(dev_pilot, batch_size=48, num_workers=8, pin_memory=True)

# Authentiques (FPR, Image AUROC) — masque = zéros par construction
auth_ds = AuthenticELADataset("cache/auth", qualities=(90,))
auth_loader = DataLoader(auth_ds, batch_size=48, num_workers=8, pin_memory=True)
```

Chaque item : `(x, mask)` avec `x` en `(3, 384, 384)` float [0,1] et `mask` en `(384, 384)` binaire
(resize **NEAREST**).

---

## 4. Boucle d'entraînement — pilotage §9.3bis

```python
from detection_eval import evaluate, BestDetectionTracker

tracker = BestDetectionTracker(
    ckpt_path="experiments/E0/best_model.pt",
    history_path="experiments/E0/auprc_curve.json",   # la courbe AUPRC vs époques (livrable S1)
    patience=15,                                       # early stopping optionnel
)

for epoch in range(cfg.epochs_max):                    # 100 max (§9.1)
    train_one_epoch(model, train_loader)

    res = evaluate(model, dev_loader, error_mode="mae", metrics=("auprc",))
    tracker.update(epoch, res["pixel_auprc"], model)   # sauvegarde si nouveau max
    print(f"epoch {epoch}: AUPRC={res['pixel_auprc']:.4f} (best={tracker.best:.4f})")

    if tracker.should_stop:
        break
```

- Le checkpoint retenu = **max AUPRC dev** (best-detection), jamais la dernière époque,
  jamais la loss de reconstruction (paradoxe de sur-reconstruction, §4).
- `auprc_curve.json` contient `best_epoch`, `best_auprc` et la courbe complète → à tracer.

---

## 5. Évaluation finale — suite complète §9.4

Sur le **dev complet** (pas le sous-échantillon), avec le best checkpoint rechargé :

```python
model.load_state_dict(torch.load("experiments/E0/best_model.pt"))

dev_full = DataLoader(dev_ds, batch_size=48, num_workers=8, pin_memory=True)
res = evaluate(model, dev_full, error_mode="mae", metrics="full",
               authentic_loader=auth_loader)

# {'pixel_auprc': ..., 'aupro': ..., 'threshold': ..., 'dice': ..., 'iou': ...,
#  'fpr_authentic': ..., 'image_auroc': ..., 'pixel_auroc': ...}

import json
json.dump(res, open("experiments/E0/metrics.json", "w"), indent=2)
```

Le `threshold` retourné est calibré **sur le dev** (max Dice, §9.3). Pour le test réel :
**figer ce seuil** et le repasser explicitement :

```python
res_test = evaluate(model, test_loader, metrics="full",
                    threshold=SEUIL_FIGE,              # jamais recalibré sur le test
                    authentic_loader=test_auth_loader)  # UNE SEULE FOIS (§9.2)
```

---

## 6. Référence API

| Fonction / classe | Entrée → Sortie | Notes |
|---|---|---|
| `build_ela_cache(data_dir, cache_dir, qualities, scale, workers)` | JPEG → PNG gris 384 | une fois |
| `SyntheticDevDataset(data_dir, cache_dir, qualities)` | → `(x, mask)` | `(90,)` ou `(75,85,95)` |
| `AuthenticELADataset(cache_dir, qualities)` | → `(x, zeros)` | pour FPR / image AUROC |
| `pilot_subset(ds, n, seed)` | → Subset fixe | éval par époque |
| `anomaly_map(x, x_hat, mode)` | `(B,C,H,W)` → `(B,H,W)` | `'mae'` \| `'mse'` \| `'ssim'` (E4) |
| `evaluate(model, loader, error_mode, metrics, threshold, authentic_loader)` | → `dict` | `("auprc",)` ou `"full"` |
| `calibrate_threshold(scores, masks)` | → float | max Dice sur dev (§9.3) |
| `BestDetectionTracker(ckpt_path, history_path, patience)` | `.update(epoch, auprc, model)` | `.should_stop` |

Métriques et leur statut (§9.4) :

| Clé du dict | Statut | Lecture |
|---|---|---|
| `pixel_auprc` | **principale** | baseline hasard = % de pixels positifs (~0.02), pas 0.5 |
| `aupro` | **principale** | per-region overlap, FPR ≤ 0.3 (standard MVTec) |
| `dice`, `iou` | opérationnelles | au seuil figé |
| `fpr_authentic` | opérationnelle | le point industriel critique (logos/tampons, E7) |
| `image_auroc` | globale | la plus solide statistiquement sur le test réel 50/50 |
| `pixel_auroc` | indicative | gonflée par le déséquilibre — ne jamais en faire l'argument |

---

## 7. Pièges connus

1. **Ne jamais binariser les scores avant `evaluate`** — AUPRC/AUPRO ont besoin des scores continus.
2. **Sous-échantillon pilote fixe** : toujours le même `seed`, sinon la courbe AUPRC bruite artificiellement.
3. **Seuil** : calibré sur dev → sauvegardé → repassé tel quel au test réel. Jamais l'inverse.
4. **Sortie du modèle** : `evaluate` gère tensor ou tuple (`out[0]`). Si ton AnoViT retourne un dict, adapter 2 lignes dans `_collect()`.
5. **AUPRC/AUROC pixel** sous-échantillonnent à 20M pixels (seedé) — exact à ~1e-3 près, x10 plus rapide.
6. **`qualities` fait partie de la config** de l'expérience (`config.yaml`, §10) — l'ordre des canaux E2 ne doit jamais changer silencieusement.
