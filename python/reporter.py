"""reporter — Rapport lisible auto-généré par lot (REPORT.md).

Rôle
----
Après une génération, produire un `REPORT.md` dans le dossier de sortie qui
explique **les résultats du run** :
    - source, config effective, décision de mode Q1,
    - composition du lot (types / tailles / alignement / négatifs / Q1 / Q2),
    - couverture des régimes Q1/Q2 (pour l'ablation E5),
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
def _regime(q1: int, q2: int) -> str:
    return "Q1<Q2" if q1 < q2 else ("Q1=Q2" if q1 == q2 else "Q1>Q2")


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


def _ela_separability(out_root, rows, quality=90, sample=60):
    """Ratio moyen (ELA dans le masque / hors masque) par type × régime.

    Échantillonne au plus `sample` positifs (stratifié par type) pour rester
    rapide. Ratio > 1 = la zone falsifiée ressort ; < 1 = zone « anormalement
    propre » (cas substitution à fond plat).
    """
    pos = [r for r in rows if not r["is_negative"]]
    if not pos or sample <= 0:
        return {}, 0
    by_type = defaultdict(list)
    for r in pos:
        by_type[r["type"]].append(r)
    # Échantillon stratifié par type.
    picked = []
    per = max(1, sample // max(1, len(by_type)))
    for t, lst in by_type.items():
        picked += lst[:per]
    picked = picked[:sample]

    agg = defaultdict(list)
    for r in picked:
        img_p = os.path.join(out_root, r["path_img"])
        msk_p = os.path.join(out_root, r["path_mask"])
        rgb = np.asarray(Image.open(img_p).convert("RGB"), dtype=np.uint8)
        m = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE) > 0
        if not m.any():
            continue
        e = _ela_raw(rgb, quality)
        ratio = float(e[m].mean() / max(e[~m].mean(), 1e-6))
        agg[(r["type"], _regime(r["q1_effective"], r["q2"]))].append(ratio)
    return agg, len(picked)


def _tbl(header, rows):
    """Petit tableau Markdown."""
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- rapport
def write_report(out_root: str, ela_quality: int = 90, ela_sample: int = 60) -> str:
    rows, dist, cfg = _load(out_root)
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
        ("Mode Q1", comp.get("q1_mode", "?")),
        ("Q1 sweep", comp.get("q1_sweep", "-")),
        ("Q2 sweep", comp.get("q2_sweep", "-")),
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
    L.append("**Q2** : " + ", ".join(f"{k}→{v}" for k, v in sorted(Counter(r["q2"] for r in rows).items())) + "  ")
    L.append("**Q1 effectif** : " + ", ".join(f"{k}→{v}" for k, v in sorted(Counter(r["q1_effective"] for r in rows).items())) + "\n")

    # --- 3. Régimes E5 ---
    L.append("## 3. Couverture des régimes Q1/Q2 (ablation E5)\n")
    reg_all = Counter(_regime(r["q1_effective"], r["q2"]) for r in rows)
    reg_pos = Counter(_regime(r["q1_effective"], r["q2"]) for r in pos)
    L.append(_tbl(["Régime", "Tous", "Positifs"],
                  [(k, reg_all.get(k, 0), reg_pos.get(k, 0)) for k in ["Q1<Q2", "Q1=Q2", "Q1>Q2"]]) + "\n")
    missing = [k for k in ["Q1<Q2", "Q1=Q2", "Q1>Q2"] if reg_pos.get(k, 0) == 0]
    if missing:
        L.append(f"> ⚠️ Régimes non peuplés : {', '.join(missing)}. "
                 "Ajuste `q1_sweep` / `q2_sweep` pour les couvrir.\n")
    else:
        L.append("> ✅ Les trois régimes sont peuplés (E5 exploitable).\n")

    # --- 4. Intégrité ---
    L.append("## 4. Contrôles d'intégrité\n")
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

    # --- 5. Séparabilité ELA ---
    L.append("## 5. Signal ELA (séparabilité échantillonnée)\n")
    agg, n_used = _ela_separability(out_root, rows, ela_quality, ela_sample)
    if agg:
        L.append(f"Ratio ELA-Q{ela_quality} moyen **intérieur/extérieur du masque** "
                 f"(échantillon de {n_used} positifs). >1 = la zone ressort ; "
                 "<1 = zone « anormalement propre » (substitution à fond plat).\n")
        types = sorted({k[0] for k in agg}) or ["substitution", "copy_move", "splice"]
        regimes = ["Q1<Q2", "Q1=Q2", "Q1>Q2"]
        tbl_rows = []
        for t in types:
            cells = []
            for reg in regimes:
                v = agg.get((t, reg), [])
                cells.append(f"{np.mean(v):.2f} (n={len(v)})" if v else "–")
            tbl_rows.append([t] + cells)
        L.append(_tbl(["type \\ régime"] + regimes, tbl_rows) + "\n")
    else:
        L.append("_(mesure ELA désactivée ou aucun positif)_\n")

    # --- 6. Fichiers ---
    L.append("## 6. Fichiers produits\n")
    L.append("```\n"
             f"{out_root}/data/<id>.jpg          # document falsifié (fond Q1->Q2)\n"
             f"{out_root}/data/<id>_mask.png     # masque binaire pixel exact\n"
             f"{out_root}/data/<id>.json         # métadonnées + grille patch 24x24\n"
             f"{out_root}/manifest.parquet       # table globale\n"
             f"{out_root}/distribution.json      # sonde du corpus source\n"
             f"{out_root}/run_config.yaml        # config effective (reproductibilité)\n"
             f"{out_root}/ela_preview/           # planches QA (si ela_preview lancé)\n"
             "```\n")

    text = "\n".join(L)
    path = os.path.join(out_root, "REPORT.md")
    with open(path, "w") as f:
        f.write(text)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="reporter — REPORT.md d'un lot généré.")
    ap.add_argument("--out", required=True, help="Dossier de sortie du lot (contient manifest.parquet).")
    ap.add_argument("--ela-quality", type=int, default=90)
    ap.add_argument("--ela-sample", type=int, default=60,
                    help="Nb de positifs échantillonnés pour la séparabilité ELA (0 = désactivé).")
    args = ap.parse_args()
    path = write_report(args.out, ela_quality=args.ela_quality, ela_sample=args.ela_sample)
    print(f"Rapport écrit : {path}")


if __name__ == "__main__":
    main()
