"""jpeg_probe — Module 1 of the synthetic generation pipeline.

Role
----
Read, from a folder of source JPEGs (authentic Kaggle documents), the
compression information *already inscribed in each file*:

    - Q0: JPEG quality estimated from the quantization table (NEVER chosen,
          always READ — cf. instruction.md, mandatory rule on compression).
    - quantization table (luma / chroma).
    - chroma subsampling (4:4:4 / 4:2:2 / 4:2:0).
    - dimensions.

The module produces a `distribution.json` characterizing Q0 over the whole
corpus, to see the quality distribution BEFORE any generation. It keeps only
real JPEGs: a PNG (or an image without a quantization table) has no Q0, would
break the double-compression mismatch, and is therefore EXCLUDED with a log.

Q0 estimation
-------------
Robust brute-force method: for each Q in 1..100 we regenerate the standard JPEG
table (Annex K + libjpeg scaling) and keep the Q that minimizes the gap to the
real table. Validated: for a standard JPEG, the match is exact. Scanners/phones
may use custom tables: we still estimate Q0 but mark it `nonstandard` if the gap
exceeds a threshold.

Dependencies: Pillow + NumPy only.

Usage
-----
    python jpeg_probe.py --src /path/to/jpeg_kaggle --out output/distribution.json
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
# Standard JPEG quantization tables (Annex K), natural order (row-major).
# PIL returns im.quantization in this same natural order (verified empirically).
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

# Precompute the 100 scaled standard tables (Q = 1..100) for the brute force.
def _scaled_table(base: np.ndarray, quality: int) -> np.ndarray:
    """Standard table scaled to `quality`, libjpeg jpeg_quality_scaling style."""
    q = max(1, min(100, quality))
    sf = 5000.0 / q if q < 50 else 200.0 - q * 2.0
    t = np.floor((base * sf + 50.0) / 100.0)
    return np.clip(t, 1, 255)

_LUMA_TABLES = {q: _scaled_table(STD_LUMA, q) for q in range(1, 101)}


def estimate_quality(qtable: np.ndarray) -> tuple[int, float]:
    """Estimate the quality factor of a luma quantization table.

    Returns
    -------
    (best_q, absdiff)
        best_q   : Q in [1, 100] minimizing the gap to the standard table.
        absdiff  : sum of |diff| at the best Q (0 = exact standard JPEG).
                   Used to decide whether the table is "non-standard".
    """
    qtable = np.asarray(qtable, dtype=np.float64).reshape(-1)
    best_q, best_err = 100, np.inf
    for q, std in _LUMA_TABLES.items():
        err = float(np.abs(std - qtable).sum())
        if err < best_err:
            best_err, best_q = err, q
    return best_q, best_err


# -----------------------------------------------------------------------------
# Per-file record
# -----------------------------------------------------------------------------
@dataclass
class ProbeRecord:
    path: str
    filename: str
    width: int
    height: int
    q0: int                 # estimated quality (luma)
    absdiff: float          # gap to the best standard Q
    nonstandard: bool       # True if custom table (gap > threshold)
    n_qtables: int          # number of tables (1 = luma only/grayscale, 2 = luma+chroma)
    subsampling: str        # "4:4:4" / "4:2:2" / "4:2:0" / "unknown" / "none"
    mode: str               # PIL mode (RGB, L, ...)
    is_lossless: bool = False  # True if source without JPEG compression (PNG...): Q0 = -1


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
    """Probe a single file.

    Returns
    -------
    (record, None) if it is a real JPEG with a quantization table, OR a
                   lossless source (PNG...) when `allow_lossless=True` (q0=-1).
    (None, reason)  if excluded (non-JPEG format without allow_lossless, unreadable).

    `allow_lossless`: default False (historical contract: keep ONLY JPEGs, cf.
    instruction.md). When True, we also keep sources without JPEG history (PNG).
    They have NO Q0: they are usable only with a controlled Q1 (the imposed Q1
    becomes the background's sole compression history).
    """
    try:
        img = Image.open(path)
    except Exception as exc:
        return None, f"unreadable: {exc}"

    quant = getattr(img, "quantization", None)

    if img.format != "JPEG" or not quant:
        if allow_lossless:
            # Lossless source: no Q0, marked is_lossless (Q0 = -1 sentinel).
            return ProbeRecord(
                path=os.path.abspath(path), filename=os.path.basename(path),
                width=img.width, height=img.height, q0=-1, absdiff=0.0,
                nonstandard=False, n_qtables=0, subsampling="none",
                mode=img.mode, is_lossless=True,
            ), None
        if img.format != "JPEG":
            return None, f"not a JPEG (format={img.format})"
        # JPEG without a quantization table -> no usable Q0.
        return None, "no quantization table"

    # Luma table = index 0; it carries the quality factor.
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
# Folder walk + aggregation
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
    """Probe a whole folder, return an aggregated report (dict ready for JSON)."""
    if not os.path.isdir(src):
        raise NotADirectoryError(f"source_dir not found: {src}")

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
    # Lossless sources (q0=-1) have no Q0: excluded from the Q0 stats.
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
        # Q0 histogram (bins of 5), sorted.
        hist = Counter((q // 5) * 5 for q in q0_values)
        summary["q0_histogram_bin5"] = {str(k): hist[k] for k in sorted(hist)}
        summary["q0_exact_values"] = {str(k): v for k, v in sorted(Counter(q0_values).items())}
        summary["n_nonstandard_qtables"] = int(sum(r.nonstandard for r in records))

    # Q0-independent stats (valid even for a 100% lossless corpus).
    if records:
        summary["subsampling_distribution"] = dict(Counter(r.subsampling for r in records))
        dims = np.array([(r.width, r.height) for r in records])
        summary["dimensions"] = {
            "width": {"min": int(dims[:, 0].min()), "max": int(dims[:, 0].max()),
                      "median": float(np.median(dims[:, 0]))},
            "height": {"min": int(dims[:, 1].min()), "max": int(dims[:, 1].max()),
                       "median": float(np.median(dims[:, 1]))},
        }

    # Group the exclusion reasons for quick reading.
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
    print(f"  Sources kept  : {s['n_jpeg_kept']}  (of which lossless/PNG: {s.get('n_lossless_kept', 0)})")
    print(f"  Excluded      : {s['n_excluded']}  {s.get('excluded_reasons', {})}")
    if s.get("n_lossless_kept", 0) and "dimensions" in s:
        print(f"  [lossless]    : no Q0 (PNG source) -> controlled Q1 required.")
        print(f"  Dimensions    : W {s['dimensions']['width']}  H {s['dimensions']['height']}")
    if "q0_stats" in s:
        st = s["q0_stats"]
        print(f"  Q0 (luma)     : min={st['min']} p05={st['p05']} median={st['median']} "
              f"mean={st['mean']} p95={st['p95']} max={st['max']}")
        print(f"  Q0 histogram (bins of 5) : {s['q0_histogram_bin5']}")
        print(f"  Subsampling   : {s['subsampling_distribution']}")
        print(f"  Non-standard tables : {s['n_nonstandard_qtables']} / {s['n_jpeg_kept']}")
        print(f"  Dimensions    : W {s['dimensions']['width']}  H {s['dimensions']['height']}")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="jpeg_probe — Q0 distribution of the source corpus.")
    ap.add_argument("--src", required=True, help="Folder of source JPEGs (authentic Kaggle).")
    ap.add_argument("--out", default="output/distribution.json", help="Path of the JSON report.")
    ap.add_argument("--no-recursive", action="store_true", help="Do not descend into subfolders.")
    ap.add_argument("--nonstandard-threshold", type=float, default=40.0,
                    help="Gap threshold (sum |diff|) beyond which a qtable is 'non-standard'.")
    ap.add_argument("--allow-lossless", action="store_true",
                    help="Also keep lossless sources (PNG): q0=-1, usable with a controlled Q1.")
    ap.add_argument("--ext", nargs="*", default=None,
                    help="Candidate extensions (default JPEG; adds .png with --allow-lossless).")
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
    print(f"\nReport written: {args.out}  ({len(report['records'])} records)")


if __name__ == "__main__":
    main()
