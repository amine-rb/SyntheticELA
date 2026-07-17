"""jpeg_probe — Module 1 du pipeline de génération synthétique.

Rôle
----
Lire, sur un dossier de JPEG sources (documents Kaggle authentiques), les
informations de compression *déjà inscrites dans chaque fichier* :

    - Q0 : qualité JPEG estimée depuis la table de quantification (JAMAIS choisie,
           toujours LUE — cf. instruction.md, règle impérative sur la compression).
    - table de quantification (luma / chroma).
    - subsampling chroma (4:4:4 / 4:2:2 / 4:2:0).
    - dimensions.

Le module produit un `distribution.json` caractérisant Q0 sur tout le corpus,
afin de voir la distribution des qualités AVANT toute génération. Il ne garde
que les vrais JPEG : une PNG (ou une image sans table de quantification) n'a pas
de Q0, casserait le mismatch de double compression, et est donc EXCLUE avec log.

Estimation de Q0
----------------
Méthode robuste par force brute : pour chaque Q de 1..100 on régénère la table
standard JPEG (Annexe K + scaling libjpeg) et on retient le Q qui minimise
l'écart à la table réelle. Validé : pour un JPEG standard, le match est exact.
Les scanners/téléphones peuvent utiliser des tables custom : on estime quand
même Q0 mais on le marque `nonstandard` si l'écart dépasse un seuil.

Dépendances : Pillow + NumPy uniquement.

Usage
-----
    python jpeg_probe.py --src /chemin/vers/jpeg_kaggle --out output/distribution.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from PIL import Image, JpegImagePlugin

# -----------------------------------------------------------------------------
# Tables de quantification standard JPEG (Annexe K), ordre naturel (row-major).
# PIL renvoie im.quantization dans ce même ordre naturel (vérifié empiriquement).
# -----------------------------------------------------------------------------
STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)

STD_CHROMA = np.array([
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
], dtype=np.float64)

# Précompute des 100 tables standard scalées (Q = 1..100) pour la force brute.
def _scaled_table(base: np.ndarray, quality: int) -> np.ndarray:
    """Table standard scalée à `quality`, façon libjpeg jpeg_quality_scaling."""
    q = max(1, min(100, quality))
    sf = 5000.0 / q if q < 50 else 200.0 - q * 2.0
    t = np.floor((base * sf + 50.0) / 100.0)
    return np.clip(t, 1, 255)

_LUMA_TABLES = {q: _scaled_table(STD_LUMA, q) for q in range(1, 101)}


def estimate_quality(qtable: np.ndarray) -> tuple[int, float]:
    """Estime le facteur qualité d'une table de quantification luma.

    Returns
    -------
    (best_q, absdiff)
        best_q   : Q dans [1, 100] minimisant l'écart à la table standard.
        absdiff  : somme des |diff| au meilleur Q (0 = JPEG standard exact).
                   Sert à décider si la table est "non standard".
    """
    qtable = np.asarray(qtable, dtype=np.float64).reshape(-1)
    best_q, best_err = 100, np.inf
    for q, std in _LUMA_TABLES.items():
        err = float(np.abs(std - qtable).sum())
        if err < best_err:
            best_err, best_q = err, q
    return best_q, best_err


# -----------------------------------------------------------------------------
# Enregistrement par fichier
# -----------------------------------------------------------------------------
@dataclass
class ProbeRecord:
    path: str
    filename: str
    width: int
    height: int
    q0: int                 # qualité estimée (luma)
    absdiff: float          # écart au meilleur Q standard
    nonstandard: bool       # True si table custom (écart > seuil)
    n_qtables: int          # nb de tables (1 = luma seule/grayscale, 2 = luma+chroma)
    subsampling: str        # "4:4:4" / "4:2:2" / "4:2:0" / "unknown" / "none"
    mode: str               # mode PIL (RGB, L, ...)
    is_lossless: bool = False  # True si source sans compression JPEG (PNG...) : Q0 = -1


_SUBSAMPLING_LABELS = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0", -1: "unknown"}


def _read_subsampling(img: Image.Image) -> str:
    try:
        code = JpegImagePlugin.get_sampling(img)
    except Exception:
        code = -1
    return _SUBSAMPLING_LABELS.get(code, "unknown")


def probe_file(
    path: str,
    nonstandard_threshold: float = 40.0,
    allow_lossless: bool = False,
) -> tuple[Optional[ProbeRecord], Optional[str]]:
    """Sonde un fichier unique.

    Returns
    -------
    (record, None) si c'est un vrai JPEG avec table de quantification, OU une
                   source lossless (PNG...) lorsque `allow_lossless=True` (q0=-1).
    (None, reason)  si exclu (format non-JPEG sans allow_lossless, illisible).

    `allow_lossless` : par défaut False (contrat historique : ne garder QUE des
    JPEG, cf. instruction.md). À True, on garde aussi les sources sans historique
    JPEG (PNG). Elles n'ont PAS de Q0 : elles ne sont exploitables qu'en mode Q1
    contrôlé (le Q1 imposé devient l'unique historique de compression du fond).
    """
    try:
        img = Image.open(path)
    except Exception as exc:
        return None, f"unreadable: {exc}"

    quant = getattr(img, "quantization", None)

    if img.format != "JPEG" or not quant:
        if allow_lossless:
            # Source lossless : pas de Q0, marquée is_lossless (Q0 = -1 sentinelle).
            return ProbeRecord(
                path=os.path.abspath(path), filename=os.path.basename(path),
                width=img.width, height=img.height, q0=-1, absdiff=0.0,
                nonstandard=False, n_qtables=0, subsampling="none",
                mode=img.mode, is_lossless=True,
            ), None
        if img.format != "JPEG":
            return None, f"not a JPEG (format={img.format})"
        # JPEG sans table de quantification -> pas de Q0 exploitable.
        return None, "no quantization table"

    # Table luma = index 0 ; c'est celle qui porte le facteur qualité.
    luma = np.array(quant[0], dtype=np.float64)
    q0, absdiff = estimate_quality(luma)

    rec = ProbeRecord(
        path=os.path.abspath(path),
        filename=os.path.basename(path),
        width=img.width,
        height=img.height,
        q0=q0,
        absdiff=round(absdiff, 2),
        nonstandard=absdiff > nonstandard_threshold,
        n_qtables=len(quant),
        subsampling=_read_subsampling(img),
        mode=img.mode,
    )
    return rec, None


# -----------------------------------------------------------------------------
# Parcours de dossier + agrégation
# -----------------------------------------------------------------------------
def _iter_candidates(src: str, recursive: bool, exts: tuple[str, ...]):
    if recursive:
        for root, _, files in os.walk(src):
            for f in files:
                if f.lower().endswith(exts):
                    yield os.path.join(root, f)
    else:
        for f in os.listdir(src):
            p = os.path.join(src, f)
            if os.path.isfile(p) and f.lower().endswith(exts):
                yield p


def probe_dir(
    src: str,
    recursive: bool = True,
    candidate_ext=(".jpg", ".jpeg", ".jpe", ".jfif"),
    nonstandard_threshold: float = 40.0,
    allow_lossless: bool = False,
) -> dict:
    """Sonde tout un dossier, retourne un rapport agrégé (dict prêt pour JSON)."""
    if not os.path.isdir(src):
        raise NotADirectoryError(f"source_dir introuvable : {src}")

    exts = tuple(e.lower() for e in candidate_ext)
    records: list[ProbeRecord] = []
    excluded: list[dict] = []

    for path in _iter_candidates(src, recursive, exts):
        rec, reason = probe_file(path, nonstandard_threshold, allow_lossless)
        if rec is not None:
            records.append(rec)
        else:
            excluded.append({"path": os.path.abspath(path), "reason": reason})

    return _build_report(src, records, excluded)


def _build_report(src: str, records: list[ProbeRecord], excluded: list[dict]) -> dict:
    # Les sources lossless (q0=-1) n'ont pas de Q0 : exclues des stats de Q0.
    n_lossless = int(sum(r.is_lossless for r in records))
    q0_values = [r.q0 for r in records if not r.is_lossless]
    summary = {
        "source_dir": os.path.abspath(src),
        "n_jpeg_kept": len(records),
        "n_lossless_kept": n_lossless,
        "n_excluded": len(excluded),
    }

    if q0_values:
        arr = np.array(q0_values)
        summary["q0_stats"] = {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": round(float(arr.mean()), 2),
            "median": float(np.median(arr)),
            "std": round(float(arr.std()), 2),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
        # Histogramme Q0 (bins de 5), trié.
        hist = Counter((q // 5) * 5 for q in q0_values)
        summary["q0_histogram_bin5"] = {str(k): hist[k] for k in sorted(hist)}
        summary["q0_exact_values"] = {str(k): v for k, v in sorted(Counter(q0_values).items())}
        summary["n_nonstandard_qtables"] = int(sum(r.nonstandard for r in records))

    # Stats indépendantes de Q0 (valables même pour un corpus 100% lossless).
    if records:
        summary["subsampling_distribution"] = dict(Counter(r.subsampling for r in records))
        dims = np.array([(r.width, r.height) for r in records])
        summary["dimensions"] = {
            "width": {"min": int(dims[:, 0].min()), "max": int(dims[:, 0].max()),
                      "median": float(np.median(dims[:, 0]))},
            "height": {"min": int(dims[:, 1].min()), "max": int(dims[:, 1].max()),
                       "median": float(np.median(dims[:, 1]))},
        }

    # Regroupe les raisons d'exclusion pour lecture rapide.
    summary["excluded_reasons"] = dict(Counter(
        e["reason"].split(":")[0].split("(")[0].strip() for e in excluded
    ))

    return {"summary": summary, "records": [asdict(r) for r in records], "excluded": excluded}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _print_summary(report: dict) -> None:
    s = report["summary"]
    print("=" * 64)
    print(f"  jpeg_probe — {s['source_dir']}")
    print("=" * 64)
    print(f"  Sources gardées : {s['n_jpeg_kept']}  (dont lossless/PNG : {s.get('n_lossless_kept', 0)})")
    print(f"  Exclus        : {s['n_excluded']}  {s.get('excluded_reasons', {})}")
    if s.get("n_lossless_kept", 0) and "dimensions" in s:
        print(f"  [lossless]    : pas de Q0 (source PNG) -> mode Q1 contrôlé requis.")
        print(f"  Dimensions    : W {s['dimensions']['width']}  H {s['dimensions']['height']}")
    if "q0_stats" in s:
        st = s["q0_stats"]
        print(f"  Q0 (luma)     : min={st['min']} p05={st['p05']} médiane={st['median']} "
              f"moy={st['mean']} p95={st['p95']} max={st['max']}")
        print(f"  Q0 histogramme (bins de 5) : {s['q0_histogram_bin5']}")
        print(f"  Subsampling   : {s['subsampling_distribution']}")
        print(f"  Tables non standard : {s['n_nonstandard_qtables']} / {s['n_jpeg_kept']}")
        print(f"  Dimensions    : W {s['dimensions']['width']}  H {s['dimensions']['height']}")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="jpeg_probe — distribution de Q0 du corpus source.")
    ap.add_argument("--src", required=True, help="Dossier des JPEG sources (Kaggle authentiques).")
    ap.add_argument("--out", default="output/distribution.json", help="Chemin du rapport JSON.")
    ap.add_argument("--no-recursive", action="store_true", help="Ne pas descendre dans les sous-dossiers.")
    ap.add_argument("--nonstandard-threshold", type=float, default=40.0,
                    help="Seuil d'écart (somme |diff|) au-delà duquel une qtable est 'non standard'.")
    ap.add_argument("--allow-lossless", action="store_true",
                    help="Garder aussi les sources lossless (PNG) : q0=-1, exploitables en Q1 contrôlé.")
    ap.add_argument("--ext", nargs="*", default=None,
                    help="Extensions candidates (défaut JPEG ; ajoute .png avec --allow-lossless).")
    args = ap.parse_args()

    exts = tuple(args.ext) if args.ext else (
        (".jpg", ".jpeg", ".jpe", ".jfif", ".png") if args.allow_lossless
        else (".jpg", ".jpeg", ".jpe", ".jfif"))
    report = probe_dir(
        args.src,
        recursive=not args.no_recursive,
        candidate_ext=exts,
        nonstandard_threshold=args.nonstandard_threshold,
        allow_lossless=args.allow_lossless,
    )
    _print_summary(report)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nRapport écrit : {args.out}  ({len(report['records'])} enregistrements)")


if __name__ == "__main__":
    main()
