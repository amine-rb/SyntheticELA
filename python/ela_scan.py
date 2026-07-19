"""ela_scan — génère l'ELA (RGB 3 qualités ≈ Q1) sur un DOSSIER d'images brut.

Rôle
----
Applique EXACTEMENT le même ELA que le pipeline de falsification
(`orchestrator.compute_ela_stack`, mêmes qualités/échelle que `config.sh`) à un
dossier d'images quelconque — falsifié ou non — et écrit une image ELA RGB par
document dans un dossier de sortie, plus un `ela.csv`.

Sert à deux choses :
  1. produire l'entrée ELA d'un modèle sur des images qui ne viennent pas du
     générateur (pas de manifeste, pas de masque requis) ;
  2. inspecter à l'œil ce que "voit" le modèle sur des documents réels.

ATTENTION (Q1 inconnu) : la qualité de sonde (`ELA_QUALITY` ≈ Q1) est celle
choisie pour le corpus SYNTHÉTIQUE. Sur de vraies images dont l'historique de
compression est inconnu, cette sonde n'est pas garantie d'être au point fixe du
fond — le contraste peut être plus faible. Utiliser `--ela-quality` pour balayer
plusieurs sondes si besoin.

ELA = |image - recompressée_à_q|, empilée sur 3 qualités -> canaux R/G/B,
échelle globale fixe (lisibilité + comparabilité inter-images).

Dépendances : Pillow, NumPy, PyYAML (via orchestrator.load_config).
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from PIL import Image

from orchestrator import load_config, compute_ela_stack, ela_qualities

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def _iter_images(src: str, recursive: bool):
    """Chemins d'images sous `src` (récursif ou non), triés, extensions connues."""
    if recursive:
        for root, _, files in os.walk(src):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXT):
                    yield os.path.join(root, f)
    else:
        for f in sorted(os.listdir(src)):
            p = os.path.join(src, f)
            if os.path.isfile(p) and f.lower().endswith(IMG_EXT):
                yield p


def run(src_dir: str, out_dir: str, ela_quality: int, ela_spread: int,
        scale: float, recursive: bool = False, chroma_suppress: float = 0.0,
        grayscale_input: bool = False) -> str:
    if not os.path.isdir(src_dir):
        raise NotADirectoryError(f"Dossier source introuvable : {src_dir}")
    qs = ela_qualities(int(ela_quality), int(ela_spread))   # MÊME pile que la sortie du pipeline
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    n = 0
    for img_path in _iter_images(src_dir, recursive):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ela_name = f"{stem}_ela.png"
        try:
            ela_rgb = compute_ela_stack(img_path, qs, scale,
                                        chroma_suppress=chroma_suppress,
                                        grayscale_input=grayscale_input)   # (H,W,3) RGB
        except Exception as e:                                  # image illisible -> on saute, on signale
            print(f"[ela_scan] ⚠️  ignorée ({e}) : {img_path}")
            continue
        Image.fromarray(ela_rgb, "RGB").save(os.path.join(out_dir, ela_name))
        rows.append({
            "id": stem,
            "ela": ela_name,
            "source_image": os.path.relpath(img_path, src_dir),
            "ela_qualities": ",".join(str(q) for q in qs),
            "ela_scale": scale,
        })
        n += 1

    with open(os.path.join(out_dir, "ela.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "ela", "source_image",
                                          "ela_qualities", "ela_scale"])
        w.writeheader()
        w.writerows(rows)

    cs = f", chroma_suppress={chroma_suppress:g}" if chroma_suppress else ""
    gs = ", grayscale_input" if grayscale_input else ""
    print(f"[ela_scan] {n} ELA RGB écrites dans {out_dir} "
          f"(qualités {qs}, échelle globale ×{scale:g}{gs}{cs}).")
    if n == 0:
        print(f"[ela_scan] ⚠️  aucune image {IMG_EXT} trouvée sous {src_dir}"
              f"{' (essayez --recursive)' if not recursive else ''}.")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ela_scan — ELA RGB (3 qualités ≈ Q1) sur un dossier d'images, "
                    "avec les mêmes paramètres que la génération (config.sh).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--in", dest="src", required=True, help="Dossier d'images source.")
    ap.add_argument("--out", dest="out", required=True, help="Dossier de sortie ELA.")
    ap.add_argument("--recursive", action="store_true", help="Parcourir les sous-dossiers.")
    ap.add_argument("--ela-quality", type=int, default=None,
                    help="Centre de sonde (défaut = config ela.ela_quality ≈ Q1).")
    ap.add_argument("--ela-spread", type=int, default=None,
                    help="Écart des 3 canaux (défaut = config ela.ela_spread).")
    ap.add_argument("--ela-scale", type=float, default=None,
                    help="Échelle globale fixe (défaut = config ela.ela_scale=15).")
    ap.add_argument("--chroma-suppress", type=float, default=None,
                    help="Atténue l'ELA des pixels colorés (logos/tampons) ; 0 = off "
                         "(défaut = config ela.chroma_suppress).")
    ap.add_argument("--grayscale-input", dest="grayscale_input", action="store_true", default=None,
                    help="Passe l'image en gris AVANT l'ELA (défaut = config ela.grayscale_input).")
    ap.add_argument("--no-grayscale-input", dest="grayscale_input", action="store_false",
                    help="Force l'ELA sur l'image couleur (désactive ela.grayscale_input).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ela_cfg = cfg.get("ela", cfg.get("ela_preview", {}))
    q = args.ela_quality if args.ela_quality is not None else int(ela_cfg["ela_quality"])
    spread = args.ela_spread if args.ela_spread is not None else int(ela_cfg.get("ela_spread", 8))
    scale = args.ela_scale if args.ela_scale is not None else float(ela_cfg.get("ela_scale", 15.0))
    chroma = args.chroma_suppress if args.chroma_suppress is not None else float(ela_cfg.get("chroma_suppress", 0.0))
    gray = args.grayscale_input if args.grayscale_input is not None else bool(ela_cfg.get("grayscale_input", False))
    run(args.src, args.out, ela_quality=q, ela_spread=spread,
        scale=scale, recursive=args.recursive, chroma_suppress=chroma, grayscale_input=gray)


if __name__ == "__main__":
    main()
