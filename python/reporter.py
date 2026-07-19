"""reporter — Rapport lisible auto-généré par lot (REPORT.md).

Rôle
----
Après une génération, produire un `REPORT.md` dans le dossier de sortie qui
explique **les résultats du run** :
    - source, config effective, qualités (Q1 < Q2, l'écart = signal ELA),
    - composition du lot (types / tailles / alignement / négatifs / Q),
    - contrôles d'intégrité (masques positifs/négatifs cohérents),
    - séparabilité ELA échantillonnée (le signal falsifié ressort-il ?).

Ce module lit uniquement les artefacts du dossier de sortie (`manifest.parquet`,
`distribution.json`, `run_config.yaml`) : il peut donc être relancé seul sur un
lot existant.

Usage
-----
    python reporter.py --out output              # rapport seul
    python reporter.py --out output --ela-sample 0   # sans mesure ELA
"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import Counter, defaultdict

import numpy as np
import cv2
from PIL import Image
import pyarrow.parquet as pq


# ---------------------------------------------------------------- helpers
def _load(out_root: str):
    manifest = pq.read_table(os.path.join(out_root, "manifest.parquet")).to_pylist()
    dist, cfg = {}, {}
    p = os.path.join(out_root, "distribution.json")
    if os.path.exists(p):
        dist = json.load(open(p)).get("summary", {})
    p = os.path.join(out_root, "run_config.yaml")
    if os.path.exists(p):
        import yaml
        cfg = yaml.safe_load(open(p))
    return manifest, dist, cfg


def _ela_raw(rgb: np.ndarray, quality: int) -> np.ndarray:
    """Carte ELA brute (moyenne sur canaux, non normalisée) pour la séparabilité."""
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
    return np.abs(rgb.astype(np.int16) - rec).mean(axis=2)


def _ink_mask(rgb: np.ndarray) -> np.ndarray:
    """Masque 'encre' (contenu sombre) via Otsu — pour isoler le TEXTE du papier."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binv > 0


def _ela_separability(out_root, rows, quality=90, sample=60):
    """Séparabilité ELA par type : zone FALSIFIÉE vs TEXTE AUTHENTIQUE.

    Métrique qui compte pour un détecteur : `forged / authentic_text`, PAS
    `in_mask / out_mask`. Ce dernier ne mesure que texte-vs-papier blanc (tout
    texte a une ELA élevée sur ses bords) -> il est élevé même sans aucun signal
    de compression, donc trompeur. On compare donc l'ELA du texte falsifié à celle
    du texte AUTHENTIQUE (encre hors masque, dilaté pour exclure le halo de bord).
    Ratio >1 = la falsification ressort du texte ordinaire (exploitable) ;
    ≈1 = indiscernable (aucun signal, cas Q1==Q2).

    Renvoie {type: {"fa": [ratios forgé/auth], "fp": [ratios forgé/papier]}}.
    """
    pos = [r for r in rows if not r["is_negative"]]
    if not pos or sample <= 0:
        return {}, 0
    by_type = defaultdict(list)
    for r in pos:
        by_type[r["type"]].append(r)
    picked = []
    per = max(1, sample // max(1, len(by_type)))
    for t, lst in by_type.items():
        picked += lst[:per]
    picked = picked[:sample]

    kernel = np.ones((9, 9), np.uint8)
    agg = defaultdict(lambda: {"fa": [], "fp": []})
    for r in picked:
        img_p = os.path.join(out_root, r["path_img"])
        msk_p = os.path.join(out_root, r["path_mask"])
        rgb = np.asarray(Image.open(img_p).convert("RGB"), dtype=np.uint8)
        m = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE) > 0
        if not m.any():
            continue
        e = _ela_raw(rgb, quality)
        ink = _ink_mask(rgb)
        forged_dil = cv2.dilate(m.astype(np.uint8), kernel) > 0
        auth_text = ink & (~forged_dil)       # texte authentique (encre hors falsif.)
        paper = ~ink                          # papier (référence basse)
        if not auth_text.any():
            continue
        e_forged = float(e[m].mean())
        agg[r["type"]]["fa"].append(e_forged / max(float(e[auth_text].mean()), 1e-6))
        if paper.any():
            agg[r["type"]]["fp"].append(e_forged / max(float(e[paper].mean()), 1e-6))
    return agg, len(picked)


def _tbl(header, rows):
    """Petit tableau Markdown."""
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- rapport
def write_report(out_root: str, ela_quality: int | None = None, ela_sample: int = 60) -> str:
    rows, dist, cfg = _load(out_root)
    # Par défaut, mesurer à la qualité de sonde RÉELLE du run (≈ Q1), pas un 90 figé :
    # sinon le rapport sous-estime le signal (mesuré ailleurs qu'à l'optimum).
    if ela_quality is None:
        ela_quality = int(cfg.get("ela_preview", {}).get("ela_quality", 90)) if cfg else 90
    n = len(rows)
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    comp = cfg.get("compression", {}) if cfg else {}
    orch = cfg.get("orchestrator", {}) if cfg else {}

    L = []
    L.append(f"# Rapport de génération — `{out_root}`\n")
    L.append(f"> Auto-généré par `reporter.py`. {n} documents.\n")

    # --- 1. Source & config ---
    L.append("## 1. Source & configuration\n")
    src_dir = cfg.get("paths", {}).get("source_dir", "?") if cfg else "?"
    n_lossless = dist.get("n_lossless_kept", 0)
    src_kind = "lossless (PNG, pas de Q0)" if n_lossless else "JPEG (Q0 lu)"
    info = [
        ("Dossier source", src_dir),
        ("Sources gardées", f"{dist.get('n_jpeg_kept', '?')} (dont lossless : {n_lossless})"),
        ("Type de source", src_kind),
        ("Seed global", orch.get("seed", "?")),
        ("Q2 finale (sweep)", comp.get("quality_sweep", "-")),
        ("Écart Q1_GAP", f"Q1 = Q2 - {comp.get('q1_gap', '?')}"),
        ("Compression", "Q1 < Q2 : fond/texte authentique = Q1->Q2 ; substitution = Q2 seul (l'écart = signal ELA)"),
    ]
    if "q0_stats" in dist:
        s = dist["q0_stats"]
        info.append(("Q0 corpus", f"médiane {s['median']}, [{s['min']}–{s['max']}]"))
    if "dimensions" in dist:
        d = dist["dimensions"]
        info.append(("Dimensions", f"W {d['width']['min']}–{d['width']['max']}, "
                                    f"H {d['height']['min']}–{d['height']['max']}"))
    L.append(_tbl(["Champ", "Valeur"], info) + "\n")

    # --- 2. Composition ---
    L.append("## 2. Composition du lot\n")
    L.append(_tbl(["Catégorie", "Décompte"], [
        ("Total", n),
        ("Positifs (falsifiés)", len(pos)),
        ("Négatifs (authentiques)", f"{len(neg)} ({len(neg)/max(n,1)*100:.0f} %)"),
    ]) + "\n")
    L.append("**Types** : " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["type"] for r in rows).items())) + "  ")
    L.append("**Tailles** : " + ", ".join(f"{k} {v}" for k, v in Counter(r["size_class"] for r in rows).items()) + "  ")
    L.append("**Alignement** : " + ", ".join(f"{k} {v}" for k, v in Counter(r["alignment"] for r in rows).items()) + "  ")
    L.append("**Q2 (sauvegarde finale)** : " + ", ".join(f"{k}→{v}" for k, v in sorted(Counter(r["q2"] for r in rows).items())) + "\n")

    # --- 3. Intégrité ---
    L.append("## 3. Contrôles d'intégrité\n")
    bad_pos = sum(r["n_mask_px"] == 0 for r in pos)
    bad_neg = sum(r["n_mask_px"] > 0 for r in neg)
    ok = (bad_pos == 0 and bad_neg == 0)
    L.append(_tbl(["Contrôle", "Résultat"], [
        ("Positifs à masque vide (attendu 0)", bad_pos),
        ("Négatifs à masque non vide (attendu 0)", bad_neg),
        ("Statut global", "✅ OK" if ok else "❌ ANOMALIE"),
    ]) + "\n")
    frac = [(sc, [r["mask_frac"] * 100 for r in pos if r["size_class"] == sc]) for sc in
            ["small", "medium", "large", "very_large"]]
    L.append("Surface falsifiée moyenne par classe de taille :\n")
    L.append(_tbl(["Classe", "% page (moy)", "n"],
                  [(sc, f"{np.mean(v):.3f}" if v else "-", len(v)) for sc, v in frac]) + "\n")

    # --- 4. Séparabilité ELA ---
    L.append("## 4. Signal ELA (séparabilité échantillonnée)\n")
    agg, n_used = _ela_separability(out_root, rows, ela_quality, ela_sample)
    if agg:
        L.append(f"Ratio ELA-Q{ela_quality} de la zone **falsifiée vs texte "
                 f"AUTHENTIQUE** (échantillon de {n_used} positifs). C'est la métrique "
                 "qui compte : >1 = la falsification ressort du texte ordinaire "
                 "(exploitable) ; ≈1 = indiscernable (aucun signal, ex. Q1==Q2). "
                 "Le ratio *vs papier* est donné pour info (toujours élevé car tout "
                 "texte s'allume — trompeur seul).\n")
        types = sorted(agg) or ["substitution"]
        tbl_rows = []
        for t in types:
            fa, fp = agg[t]["fa"], agg[t]["fp"]
            fa_s = f"{np.mean(fa):.2f} (n={len(fa)})" if fa else "-"
            fp_s = f"{np.mean(fp):.2f}" if fp else "-"
            tbl_rows.append([t, fa_s, fp_s])
        L.append(_tbl(["type", f"forgé/texte-authentique (ELA-Q{ela_quality})",
                       "forgé/papier (info)"], tbl_rows) + "\n")
        best = max((np.mean(agg[t]["fa"]) for t in types if agg[t]["fa"]), default=0.0)
        if best < 1.3:
            L.append("> ⚠️ **Signal faible** (forgé/authentique < 1.3) : la falsification "
                     "est peu distinguable du texte réel. Vérifie que Q1 < Q2 (écart "
                     "`Q1_GAP`) et que `ELA_QUALITY` diffère des Q2 du sweep.\n")
    else:
        L.append("_(mesure ELA désactivée ou aucun positif)_\n")

    # --- 5. Fichiers ---
    L.append("## 5. Fichiers produits\n")
    L.append("```\n"
             f"{out_root}/images/<id>.jpg        # document falsifié (fond double-compressé à Q)\n"
             f"{out_root}/images/images.csv      # CSV du dossier (reprise facile)\n"
             f"{out_root}/masks/<id>_mask.png    # masque binaire pixel exact\n"
             f"{out_root}/masks/<id>.json        # métadonnées + grille patch 24x24\n"
             f"{out_root}/masks/masks.csv        # CSV du dossier (bboxes, n_mask_px, ...)\n"
             f"{out_root}/ela/<id>_ela.png       # ELA RGB (3 qualités ≈ Q1) sur le JPEG final\n"
             f"{out_root}/ela/ela.csv            # CSV du dossier (qualités/échelle ELA)\n"
             f"{out_root}/manifest.parquet       # table globale\n"
             f"{out_root}/distribution.json      # sonde du corpus source\n"
             f"{out_root}/run_config.yaml        # config effective (reproductibilité)\n"
             "```\n")

    text = "\n".join(L)
    path = os.path.join(out_root, "REPORT.md")
    with open(path, "w") as f:
        f.write(text)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="reporter — REPORT.md d'un lot généré.")
    ap.add_argument("--out", required=True, help="Dossier de sortie du lot (contient manifest.parquet).")
    ap.add_argument("--ela-quality", type=int, default=None,
                    help="Qualité de sonde ELA pour la séparabilité (défaut : celle du run ≈ Q1).")
    ap.add_argument("--ela-sample", type=int, default=60,
                    help="Nb de positifs échantillonnés pour la séparabilité ELA (0 = désactivé).")
    args = ap.parse_args()
    path = write_report(args.out, ela_quality=args.ela_quality, ela_sample=args.ela_sample)
    print(f"Rapport écrit : {path}")


if __name__ == "__main__":
    main()
