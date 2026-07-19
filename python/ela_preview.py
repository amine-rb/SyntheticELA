"""ela_preview — Module 6 du pipeline (QA hors entraînement).

Rôle
----
Contrôle visuel : calculer l'ELA de quelques documents générés, à une qualité
ELA DOCUMENTÉE et DISTINCTE de Q2 (défaut 90), et produire une planche
image | ELA | masque pour vérifier à l'œil que l'incohérence de compression
tombe bien sur la zone falsifiée.

Ce module NE sert PAS à l'entraînement : c'est un outil de sanity-check du
dataset (on veut voir que le signal existe avant de lancer quoi que ce soit).

ELA = |image - recompressée_à_qELA|, étirée pour la lisibilité.

Dépendances : Pillow, NumPy, OpenCV, PyArrow (lecture manifeste).
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np
import cv2
from PIL import Image
import pyarrow.parquet as pq

from orchestrator import load_config, compute_ela_stack, ela_qualities


def _panel(img_bgr, ela_bgr, mask, max_h=700):
    """Assemble [image | ELA | masque] côte à côte, hauteur bornée."""
    h = img_bgr.shape[0]
    scale = min(1.0, max_h / h)
    size = (int(img_bgr.shape[1] * scale), int(h * scale))
    img_r = cv2.resize(img_bgr, size)
    ela_r = cv2.resize(ela_bgr, size)
    mask_r = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), size,
                        interpolation=cv2.INTER_NEAREST)
    sep = np.full((size[1], 6, 3), 255, dtype=np.uint8)
    return np.hstack([img_r, sep, ela_r, sep, mask_r])


def run(cfg: dict, n_samples: int = 12, ela_quality: int = 90,
        prefer_positive: bool = True, scale: float = 15.0, ela_spread: int = 8) -> str:
    out_root = cfg["paths"]["output_dir"]
    # MÊME pile ELA 3 qualités -> RGB que la sortie ela/*.png (le modèle voit ça).
    qs = ela_qualities(int(ela_quality), int(ela_spread))
    manifest_path = os.path.join(out_root, "manifest.parquet")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifeste introuvable : {manifest_path} (lance l'orchestrator d'abord).")

    rows = pq.read_table(manifest_path).to_pylist()
    # Priorité aux positifs (on veut voir le signal), on complète par des négatifs.
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    ordered = (pos + neg) if prefer_positive else rows
    sample = ordered[:n_samples]

    prev_dir = os.path.join(out_root, "ela_preview")
    os.makedirs(prev_dir, exist_ok=True)

    for r in sample:
        img_path = os.path.join(out_root, r["path_img"])
        mask_path = os.path.join(out_root, r["path_mask"])
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        ela_rgb = compute_ela_stack(img_path, qs, scale)     # (H,W,3) RGB = 3 qualités
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ela_bgr = cv2.cvtColor(ela_rgb, cv2.COLOR_RGB2BGR)
        panel = _panel(img_bgr, ela_bgr, mask)

        tag = r["type"]
        out = os.path.join(prev_dir, f"{r['id']}__{tag}__q0-{r['q0']}_q2-{r['q2']}__elaRGB{qs[0]}-{qs[1]}-{qs[2]}.png")
        cv2.imwrite(out, panel)

    print(f"[ela_preview] {len(sample)} planches écrites dans {prev_dir} "
          f"(ELA RGB 3 qualités {qs}, échelle globale ×{scale:g}).")
    return prev_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="ela_preview — QA visuel du dataset généré.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="Écrase paths.output_dir.")
    ap.add_argument("--n", type=int, default=None, help="Nb d'échantillons.")
    ap.add_argument("--ela-quality", type=int, default=None, help="Qualité ELA (distincte de Q2).")
    ap.add_argument("--ela-scale", type=float, default=None,
                    help="Échelle globale fixe de l'ELA (défaut config ela_preview.ela_scale=15).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.out:
        cfg["paths"]["output_dir"] = args.out
    ela_cfg = cfg.get("ela_preview", {})
    n = args.n if args.n is not None else ela_cfg["n_samples"]
    q = args.ela_quality if args.ela_quality is not None else ela_cfg["ela_quality"]
    s = args.ela_scale if args.ela_scale is not None else ela_cfg.get("ela_scale", 15.0)
    spread = int(ela_cfg.get("ela_spread", 8))
    run(cfg, n_samples=n, ela_quality=q, scale=float(s), ela_spread=spread)


if __name__ == "__main__":
    main()
