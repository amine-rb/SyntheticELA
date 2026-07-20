"""reporter — Readable, auto-generated per-batch report (REPORT.md).

Role
----
After a generation, produce a `REPORT.md` in the output folder that explains
**the run's results**:
    - source, effective config, qualities (Q1 < Q2, the gap = ELA signal),
    - batch composition (types / sizes / alignment / negatives / Q),
    - integrity checks (consistent positive/negative masks),
    - sampled ELA separability (does the forged signal stand out?).

This module reads only the output-folder artifacts (`manifest.parquet`,
`distribution.json`, `run_config.yaml`): it can therefore be re-run standalone on
an existing batch.

Usage
-----
    python reporter.py --out output              # report only
    python reporter.py --out output --ela-sample 0   # without the ELA measurement
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
    """Raw ELA map (channel mean, unnormalized) for the separability."""
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
    return np.abs(rgb.astype(np.int16) - rec).mean(axis=2)


def _ink_mask(rgb: np.ndarray) -> np.ndarray:
    """'Ink' mask (dark content) via Otsu — to isolate the TEXT from the paper."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binv > 0


def _ela_separability(out_root, rows, quality=90, sample=60):
    """ELA separability per type: FORGED region vs AUTHENTIC TEXT.

    The metric that matters for a detector: `forged / authentic_text`, NOT
    `in_mask / out_mask`. The latter only measures text-vs-white-paper (any text
    has high ELA on its edges) -> it is high even without any compression signal,
    hence misleading. So we compare the forged text's ELA to that of AUTHENTIC
    text (ink outside the mask, dilated to exclude the edge halo).
    Ratio >1 = the forgery stands out from ordinary text (usable);
    ≈1 = indistinguishable (no signal, Q1==Q2 case).

    Returns {type: {"fa": [forged/auth ratios], "fp": [forged/paper ratios]}}.
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
        auth_text = ink & (~forged_dil)       # authentic text (ink outside the forgery)
        paper = ~ink                          # paper (low reference)
        if not auth_text.any():
            continue
        e_forged = float(e[m].mean())
        agg[r["type"]]["fa"].append(e_forged / max(float(e[auth_text].mean()), 1e-6))
        if paper.any():
            agg[r["type"]]["fp"].append(e_forged / max(float(e[paper].mean()), 1e-6))
    return agg, len(picked)


def _tbl(header, rows):
    """Small Markdown table."""
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- rapport
def write_report(out_root: str, ela_quality: int | None = None, ela_sample: int = 60) -> str:
    rows, dist, cfg = _load(out_root)
    # By default, measure at the run's ACTUAL probe quality (≈ Q1), not a fixed 90:
    # otherwise the report underestimates the signal (measured away from the optimum).
    if ela_quality is None:
        ela_quality = int(cfg.get("ela", cfg.get("ela_preview", {})).get("ela_quality", 90)) if cfg else 90
    n = len(rows)
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    comp = cfg.get("compression", {}) if cfg else {}
    orch = cfg.get("orchestrator", {}) if cfg else {}

    L = []
    L.append(f"# Generation report — `{out_root}`\n")
    L.append(f"> Auto-generated by `reporter.py`. {n} documents.\n")

    # --- 1. Source & config ---
    L.append("## 1. Source & configuration\n")
    src_dir = cfg.get("paths", {}).get("source_dir", "?") if cfg else "?"
    n_lossless = dist.get("n_lossless_kept", 0)
    src_kind = "lossless (PNG, no Q0)" if n_lossless else "JPEG (Q0 read)"
    info = [
        ("Source folder", src_dir),
        ("Sources kept", f"{dist.get('n_jpeg_kept', '?')} (of which lossless: {n_lossless})"),
        ("Source type", src_kind),
        ("Global seed", orch.get("seed", "?")),
        ("Final Q2 (sweep)", comp.get("quality_sweep", "-")),
        ("Q1_GAP gap", f"Q1 = Q2 - {comp.get('q1_gap', '?')}"),
        ("Compression", "Q1 < Q2: background/authentic text = Q1->Q2; substitution = Q2 only (the gap = ELA signal)"),
    ]
    if "q0_stats" in dist:
        s = dist["q0_stats"]
        info.append(("Corpus Q0", f"median {s['median']}, [{s['min']}–{s['max']}]"))
    if "dimensions" in dist:
        d = dist["dimensions"]
        info.append(("Dimensions", f"W {d['width']['min']}–{d['width']['max']}, "
                                    f"H {d['height']['min']}–{d['height']['max']}"))
    L.append(_tbl(["Field", "Value"], info) + "\n")

    # --- 2. Composition ---
    L.append("## 2. Batch composition\n")
    L.append(_tbl(["Category", "Count"], [
        ("Total", n),
        ("Positives (forged)", len(pos)),
        ("Negatives (authentic)", f"{len(neg)} ({len(neg)/max(n,1)*100:.0f} %)"),
    ]) + "\n")
    L.append("**Types**: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["type"] for r in rows).items())) + "  ")
    L.append("**Sizes**: " + ", ".join(f"{k} {v}" for k, v in Counter(r["size_class"] for r in rows).items()) + "  ")
    L.append("**Alignment**: " + ", ".join(f"{k} {v}" for k, v in Counter(r["alignment"] for r in rows).items()) + "  ")
    L.append("**Q2 (final save)**: " + ", ".join(f"{k}→{v}" for k, v in sorted(Counter(r["q2"] for r in rows).items())) + "\n")

    # --- 3. Integrity ---
    L.append("## 3. Integrity checks\n")
    bad_pos = sum(r["n_mask_px"] == 0 for r in pos)
    bad_neg = sum(r["n_mask_px"] > 0 for r in neg)
    ok = (bad_pos == 0 and bad_neg == 0)
    L.append(_tbl(["Check", "Result"], [
        ("Positives with empty mask (expected 0)", bad_pos),
        ("Negatives with non-empty mask (expected 0)", bad_neg),
        ("Overall status", "✅ OK" if ok else "❌ ANOMALY"),
    ]) + "\n")
    frac = [(sc, [r["mask_frac"] * 100 for r in pos if r["size_class"] == sc]) for sc in
            ["small", "medium", "large", "very_large"]]
    L.append("Mean forged area per size class:\n")
    L.append(_tbl(["Class", "% page (mean)", "n"],
                  [(sc, f"{np.mean(v):.3f}" if v else "-", len(v)) for sc, v in frac]) + "\n")

    # --- 4. ELA separability ---
    L.append("## 4. ELA signal (sampled separability)\n")
    agg, n_used = _ela_separability(out_root, rows, ela_quality, ela_sample)
    if agg:
        L.append(f"ELA-Q{ela_quality} ratio of the **forged region vs AUTHENTIC "
                 f"text** (sample of {n_used} positives). This is the metric that "
                 "matters: >1 = the forgery stands out from ordinary text (usable); "
                 "≈1 = indistinguishable (no signal, e.g. Q1==Q2). "
                 "The *vs paper* ratio is given for reference (always high because any "
                 "text lights up — misleading on its own).\n")
        types = sorted(agg) or ["substitution"]
        tbl_rows = []
        for t in types:
            fa, fp = agg[t]["fa"], agg[t]["fp"]
            fa_s = f"{np.mean(fa):.2f} (n={len(fa)})" if fa else "-"
            fp_s = f"{np.mean(fp):.2f}" if fp else "-"
            tbl_rows.append([t, fa_s, fp_s])
        L.append(_tbl(["type", f"forged/authentic-text (ELA-Q{ela_quality})",
                       "forged/paper (reference)"], tbl_rows) + "\n")
        best = max((np.mean(agg[t]["fa"]) for t in types if agg[t]["fa"]), default=0.0)
        if best < 1.3:
            L.append("> ⚠️ **Weak signal** (forged/authentic < 1.3): the forgery "
                     "is hard to distinguish from real text. Check that Q1 < Q2 (the "
                     "`Q1_GAP` gap) and that `ELA_QUALITY` differs from the sweep's Q2.\n")
    else:
        L.append("_(ELA measurement disabled or no positives)_\n")

    # --- 5. Files ---
    L.append("## 5. Produced files\n")
    L.append("```\n"
             f"{out_root}/images/<stem>_<n>.jpg       # forged document (background double-compressed at Q; n = number of forgeries)\n"
             f"{out_root}/images/images.csv           # folder CSV (easy loading)\n"
             f"{out_root}/masks/<stem>_mask_<n>.png   # exact pixel binary mask\n"
             f"{out_root}/masks/<stem>_<n>.json       # metadata + 24x24 patch grid\n"
             f"{out_root}/masks/masks.csv             # folder CSV (bboxes, n_mask_px, ...)\n"
             f"{out_root}/ela/<stem>_ela_<n>.png      # ELA RGB (3 qualities ≈ Q1) on the final JPEG\n"
             f"{out_root}/ela/ela.csv            # folder CSV (ELA qualities/scale)\n"
             f"{out_root}/manifest.parquet       # global table\n"
             f"{out_root}/distribution.json      # source-corpus probe\n"
             f"{out_root}/run_config.yaml        # effective config (reproducibility)\n"
             "```\n")

    text = "\n".join(L)
    path = os.path.join(out_root, "REPORT.md")
    with open(path, "w") as f:
        f.write(text)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="reporter — REPORT.md of a generated batch.")
    ap.add_argument("--out", required=True, help="Batch output folder (contains manifest.parquet).")
    ap.add_argument("--ela-quality", type=int, default=None,
                    help="ELA probe quality for the separability (default: the run's ≈ Q1).")
    ap.add_argument("--ela-sample", type=int, default=60,
                    help="Number of positives sampled for the ELA separability (0 = disabled).")
    args = ap.parse_args()
    path = write_report(args.out, ela_quality=args.ela_quality, ela_sample=args.ela_sample)
    print(f"Report written: {path}")


if __name__ == "__main__":
    main()
