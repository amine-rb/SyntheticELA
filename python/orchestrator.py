"""orchestrator — Module 5 of the pipeline.

Role
----
Batch generation, 100% scriptable, no manual intervention:
    1. probes the source corpus (jpeg_probe) and writes distribution.json,
    2. plans N deterministic jobs (per-document seed derived from the global seed),
       drawing edit type / size / alignment / Q2 / negative according to the config,
    3. runs (in parallel) forger -> recompress (single Q2 pass) -> annotator,
    4. writes <id>.jpg, <id>_mask.png, <id>.json + a global manifest.parquet.

Determinism + parallelism
--------------------------
All parameters of each job (including the seed) are drawn in the main process
via a master RNG. The worker uses only the job's seed for its internal choices
(positions, feather, text). The result is therefore independent of the workers'
execution order.

Subsampling of types / sizes / alignment
-----------------------------------------
- Negatives: proportion `negatives.ratio` (authentic Q0->Q2, empty mask).
  They prevent the model from learning GLOBAL double compression instead of
  LOCALIZING the inconsistency. Benign colored elements (logos, stamps) are
  kept by construction: we start from real receipts and never remove them.
- Quality: TWO passes Q1 < Q2 per document. Q2 (final save, high) is
  stratified over the sweep (i % len); Q1 = Q2 - q1_gap (lower base). The source
  is recompressed at Q1 (history of the "original document"), the substitution is
  painted in FRESH pixels (never seen by Q1), then a final save at Q2 -> the
  authentic text carries the Q1->Q2 history, the forged region only has Q2. In ELA
  (probed at a quality != Q2), the region stands out from ordinary text (~2.6x). The
  Q1<Q2 gap is MANDATORY: at Q1==Q2 the forgery is indistinguishable (ratio ~1.0).
- Types / sizes: drawn according to the config ratios (equiprobable sizes).
- Alignment: on copy_move + splice only, according to `aligned_ratio`.

Dependencies: Pillow, NumPy, OpenCV, PyYAML, PyArrow.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import cv2
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import jpeg_probe
from recompress import decode_source, save_q2, recompress_to_q1, DEFAULT_Q2_SUBSAMPLING
import forger as forger_mod
import annotator as ann


_SUBSAMPLING_INT2STR = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def _resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize an (H, W, 3) RGB image to a square (size, size).

    Applied to the DECODED source BEFORE the Q1->Q2 pass (see `_run_job`), so the
    whole forensic chain (Q1 history, fresh-pixel forgery, Q2 save, exact mask, ELA,
    patch grid) is produced natively at the target resolution and stays pixel-aligned.
    INTER_AREA when shrinking (the usual case: docs > 384), INTER_CUBIC when upscaling.
    """
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if (size < h or size < w) else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interp)


def _scale_bbox(b, sx: float, sy: float):
    """Scale a (x, y, w, h) bbox by (sx, sy) for the square-resized delivery. None-safe."""
    if not b:
        return b
    x, y, w, h = b
    return [int(round(x * sx)), int(round(y * sy)),
            max(1, int(round(w * sx))), max(1, int(round(h * sy)))]


# ------------------------------------------------------------------ ELA + CSV
def ela_qualities(center: int, spread: int) -> list[int]:
    """3 ELA probe qualities (R,G,B channels) = (center-spread, center, center+spread).

    The center targets ≈ Q1 (the background's fixed point) for maximum contrast; the
    `spread` gives 3 decorrelated views -> a COLOR IMAGE whose hue encodes the
    differential reaction to the 3 probes (the signal comes from quality DIVERSITY, not
    from chroma). Bounded within [40, 99]. Feeds `detection_eval` mode E2.
    """
    return [int(np.clip(center + d, 40, 99)) for d in (-spread, 0, spread)]


def compute_ela_stack(img_path: str, qualities: list[int], scale: float,
                      chroma_suppress: float = 0.0,
                      grayscale_input: bool = False) -> np.ndarray:
    """3-quality ELA stack -> (H, W, 3) uint8 RGB, on the re-read FINAL JPEG.

    Channel k = |image - recompress(image, qualities[k])| averaged over channels, at a
    fixed GLOBAL scale (`scale`), NATIVE resolution (pixel-aligned with image/mask).
    The channel order (R,G,B) = (q_low, q_mid, q_high). PIL encoder (consistent
    with `ela_scan` / `detection_eval`). The RGB "color" comes from quality DIVERSITY
    (3 probes), not from the image color -> compatible with grayscale input.

    Two colored FALSE-POSITIVE reducers (logos/stamps/seals), cumulable:
    - `grayscale_input`: converts the image to GRAYSCALE BEFORE ELA. A colored logo has
      huge PER-CHANNEL edges (the opposite channel swings 0<->255); in grayscale these
      contrasts average into a soft luminance -> the ELA of LIGHT colored furniture
      collapses (measured 74 -> ~8), the forgery (black text) keeps ~99%.
      Not enough on its own for DARK colored furniture (stays dark = contrasted).
    - `chroma_suppress` > 0: attenuates the ELA of colored pixels via a weight
      w = clip(1 - chroma/threshold, 0, 1) (chroma = (Cb,Cr) distance to gray, measured
      on the COLOR ORIGINAL even if the ELA is grayscale). Erases colored furniture
      WHATEVER its luminosity (chroma > threshold -> w=0). 0 = disabled.
    Both valid FOR SUBSTITUTION (achromatic forgery); disable if
    forging a colored region.
    """
    orig = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)  # kept for chroma
    if grayscale_input:
        g = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB).astype(np.int16)          # 3 identical gray channels
    else:
        img = orig.astype(np.int16)
    chans = []
    for q in qualities:
        buf = io.BytesIO()
        Image.fromarray(img.astype(np.uint8), "RGB").save(buf, "JPEG", quality=int(q))
        buf.seek(0)
        rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
        diff = np.abs(img - rec).mean(axis=2)          # ELA luminance at this quality
        chans.append(diff)
    ela = np.stack(chans, axis=2).astype(np.float32) * float(scale)
    if chroma_suppress and chroma_suppress > 0:
        ycc = cv2.cvtColor(orig, cv2.COLOR_RGB2YCrCb).astype(np.float32)    # chroma on the ORIGINAL
        chroma = np.sqrt((ycc[..., 1] - 128.0) ** 2 + (ycc[..., 2] - 128.0) ** 2)
        w = np.clip(1.0 - chroma / float(chroma_suppress), 0.0, 1.0)        # 1=achromatic keeps, 0=colored erases
        ela *= w[..., None]
    return np.clip(ela, 0, 255).astype(np.uint8)        # (H, W, 3) RGB


def join_board(img_rgb: np.ndarray, ela_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Stitch [image | ELA | mask] side-by-side into ONE (H, W*3, 3) RGB board.

    A per-document visual QA image (written to join/ when `ela.join` is on): lets a
    human eyeball at a glance whether the ELA lights up the forged zone and whether the
    mask covers it. Purely diagnostic — never read by the model. All three inputs are
    already pixel-aligned (native, or the same square when RESIZE_384). The mask (1
    channel) is promoted to 3 channels; a thin white separator + a labelled header bar
    are drawn on each panel.
    """
    h, w = img_rgb.shape[:2]
    panels = [("image", img_rgb),
              ("ELA", ela_rgb),
              ("mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))]
    bar = max(18, h // 30)                       # header height, scales with the doc
    sep = 3                                       # white gap between panels
    fs = max(0.4, h / 900.0)                      # font scale, scales with the doc
    out = []
    for i, (label, panel) in enumerate(panels):
        p = np.ascontiguousarray(panel, dtype=np.uint8)
        canvas = np.full((h + bar, w, 3), 255, np.uint8)
        canvas[bar:, :, :] = p
        cv2.putText(canvas, label, (6, bar - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 0, 0), 1, cv2.LINE_AA)
        out.append(canvas)
        if i < len(panels) - 1:
            out.append(np.full((h + bar, sep, 3), 255, np.uint8))
    return np.concatenate(out, axis=1)


def _bn(rel: Optional[str]) -> str:
    """Filename from a relative path (empty if None)."""
    return os.path.basename(rel) if rel else ""


_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(path: str) -> str:
    """Clean file stem from a source path (without extension).

    Used as the base for output names ({stem}_{n}.jpg, {stem}_mask_{n}.png, ...):
    we KEEP the source document name for traceability, only neutralizing the
    unsafe characters (spaces, encoded accents, /) -> FS + CSV compatible.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = _STEM_RE.sub("_", stem).strip("_")
    return stem or "doc"


def write_folder_csvs(sub_root: str, rows: list[dict]) -> None:
    """Writes one CSV per folder (images / masks / ela) to ease loading.

    Each CSV is self-contained (one row per document, filename + useful
    metadata): a folder can be loaded without reading the global Parquet manifest.
    """
    if not rows:
        return
    with open(os.path.join(sub_root, "images", "images.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "type", "is_negative", "quality",
                    "size_class", "n_forgeries", "source_id", "seed"])
        for r in rows:
            w.writerow([r["id"], _bn(r["path_img"]), r["type"], r["is_negative"],
                        r["q2"], r["size_class"], r["n_forgeries"], r["source_id"], r["seed"]])
    with open(os.path.join(sub_root, "masks", "masks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "mask", "json", "is_negative", "n_forgeries", "n_mask_px",
                    "mask_frac", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        for r in rows:
            w.writerow([r["id"], _bn(r["path_mask"]), _bn(r["path_json"]), r["is_negative"],
                        r["n_forgeries"], r["n_mask_px"], r["mask_frac"],
                        r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]])
    with open(os.path.join(sub_root, "ela", "ela.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ela", "image", "ela_qualities", "ela_scale", "is_negative", "type"])
        for r in rows:
            w.writerow([r["id"], _bn(r.get("path_ela")), _bn(r["path_img"]),
                        r.get("ela_qualities", r.get("ela_quality", "")), r.get("ela_scale", ""),
                        r["is_negative"], r["type"]])
    # dataset.csv: train/eval-ready index, ABSOLUTE paths -> aggregatable
    # by simple concatenation across subfolders AND across corpora (self-contained paths).
    #   id        = sequential integer 0..n-1 SPECIFIC TO THIS dataset.csv (not the file
    #               stem — that one stays in images.csv/masks.csv/ela.csv/r["id"]
    #               for matching by filename); not unique once concatenated
    #               with other subfolders/corpora.
    #   type      = format of the SOURCE document (png lossless vs jpeg), q0<0 => png
    #   x_path    = absolute path of the RGB ELA (the model's input X)
    #   negative  = true (authentic, empty mask) / false (forged)
    #   mask_path = absolute path of the mask IF forgery, empty otherwise
    def _abs(sub, rel):
        return os.path.abspath(os.path.join(sub_root, sub, _bn(rel))) if rel else ""
    with open(os.path.join(sub_root, "dataset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "x_path", "negative", "mask_path"])
        for i, r in enumerate(rows):
            neg = bool(r["is_negative"])
            src_fmt = "png" if int(r.get("q0", -1)) < 0 else "jpeg"
            w.writerow([i, src_fmt, _abs("ela", r.get("path_ela")),
                        str(neg).lower(), "" if neg else _abs("masks", r["path_mask"])])


# ------------------------------------------------------------------ job spec
@dataclass
class Job:
    doc_id: str
    seed: int
    q2: int
    is_negative: bool
    source_path: str
    edit_type: Optional[str]        # None if negative
    size_class: Optional[str]
    alignment: Optional[str]
    donor_path: Optional[str]       # splice only
    q1: Optional[int] = None        # None = native mode (Q0 read); otherwise controlled Q1
    n_forgeries: int = 1            # number of forgeries (same type) on this positive doc
    stem: str = ""                  # stem of the SOURCE NAME -> output names {stem}_{n}...


# ------------------------------------------------------------------ planning
KNOWN_EDIT_TYPES = ("substitution", "copy_move", "splice")


def resolve_edit_types(cfg: dict) -> list[str]:
    """Ordered, deduplicated list of types to generate (one subfolder each).

    Source: `forger.edit_types`. Back-compat: if absent, derives from the old
    `forger.edit_type_ratios` (types with weight > 0). There is NO LONGER any random
    draw between types: each type is a complete, separate batch.
    """
    fc = cfg.get("forger", {})
    types = fc.get("edit_types")
    if not types:
        ratios = fc.get("edit_type_ratios", {}) or {}
        types = [t for t, w in ratios.items() if w and float(w) > 0]
    if not types:
        raise ValueError(
            "forger.edit_types is empty: list at least one type among "
            f"{list(KNOWN_EDIT_TYPES)}.")
    unknown = [t for t in types if t not in KNOWN_EDIT_TYPES]
    if unknown:
        raise ValueError(
            f"forger.edit_types contains unknown types {unknown}; "
            f"known: {list(KNOWN_EDIT_TYPES)}.")
    seen, ordered = set(), []
    for t in types:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def parse_gap_range(cfg: dict) -> tuple[int, int]:
    """(gap_min, gap_max) from compression.q1_gap (scalar OR [min, max]).

    OPTION A (robustness to the inference Q1): a gap drawn PER DOCUMENT in
    [min, max] makes Q1 = Q2 - gap vary over a RANGE -> the model does not overfit
    a single Q1 (e.g. 67) and generalizes to documents whose base quality
    differs. gap_min == gap_max -> (quasi) fixed Q1 = the old behavior.
    """
    g = cfg["compression"].get("q1_gap", 0)
    if isinstance(g, (list, tuple)):
        gmin, gmax = int(g[0]), int(g[-1])
    else:
        gmin = gmax = int(g)
    gmax = max(gmin, gmax)
    if gmin <= 0:
        raise ValueError(
            "compression.q1_gap must be > 0: without a Q1<Q2 gap, the substitution "
            "is indistinguishable from authentic text in ELA (no exploitable signal).")
    return gmin, gmax


def plan_jobs(cfg: dict, source_paths: list[str], edit_type: str,
              doc_prefix: str, type_index: int = 0) -> list[Job]:
    """Deterministic jobs for a SINGLE edit type (one subfolder).

    Each type gets a decorrelated random stream (seed `[seed, type_index]`):
    the drawn negatives and sources differ from one type to another -> no exact duplicate
    between subfolders when aggregating. The doc_ids are prefixed with
    `doc_prefix` -> globally unique -> aggregation without collision.
    """
    master = np.random.default_rng([int(cfg["orchestrator"]["seed"]), int(type_index)])
    n_docs_cfg = cfg["orchestrator"]["n_docs"]
    # None/empty => as many documents as source images (y = x), one per source.
    n_docs = len(source_paths) if n_docs_cfg is None else int(n_docs_cfg)
    # TWO qualities per document: Q2 (FINAL save, drawn from the sweep, high) and
    # Q1 = Q2 - q1_gap (compression base of the "original document", lower).
    # It is the Q1<Q2 GAP that makes the substitution detectable: the authentic text
    # carries the Q1->Q2 history, the region painted in FRESH pixels only has Q2 -> in ELA
    # (probed at a quality != Q2) it stands out from ordinary text (~2.6x). At Q1==Q2
    # the forgery would be indistinguishable from authentic text (measured: ratio ~1.0).
    quality_sweep = [int(q) for q in cfg["compression"]["quality_sweep"]]
    if not quality_sweep:
        raise ValueError(
            "compression.quality_sweep is empty: list at least one JPEG quality.")
    gap_min, gap_max = parse_gap_range(cfg)   # OPTION A: gap (hence Q1) varies per doc
    neg_ratio = float(cfg["negatives"]["ratio"])
    aligned_ratio = float(cfg["forger"]["aligned_ratio"])
    # size_classes ORDERED from smallest to largest (config order).
    size_classes = list(cfg["size_classes"].keys())
    # Number of forgeries per positive doc: k ~ U{n_min..n_max} (same type).
    n_min, n_max = (list(cfg["forger"].get("n_forgeries", [1, 1])) + [1])[:2]
    n_min, n_max = int(n_min), int(max(int(n_min), int(n_max)))

    # Assignment of sources WITHOUT REPLACEMENT if the corpus is large enough (each source
    # doc used at most once -> unique output stems, 1:1 traceability). With replacement
    # only if n_docs > corpus size (the deduplication guard below then makes the
    # names unique via a suffix).
    n_src = len(source_paths)
    if n_docs <= n_src:
        order = master.permutation(n_src)
        doc_sources = [source_paths[int(order[i])] for i in range(n_docs)]
    else:
        doc_sources = [str(master.choice(source_paths)) for _ in range(n_docs)]
    used_ids: set[str] = set()      # guarantees unique filenames within the batch

    jobs: list[Job] = []
    for i in range(n_docs):
        # Deterministic per-document seed (independent of the workers' scheduling).
        seed = int(master.integers(0, 2**31 - 1))
        jrng = np.random.default_rng(seed)

        # Q2 STRATIFIED over the sweep (i % len) -> deterministic and
        # balanced coverage. Q1 = Q2 - gap (bounded >= 40: stays a valid JPEG). The SAME
        # (Q1, Q2) pair is applied to positives AS WELL AS negatives (no global
        # quality leak -> the model must LOCALIZE the gap, not read it at page level).
        q2 = int(quality_sweep[i % len(quality_sweep)])
        # OPTION A: gap drawn PER DOC (jrng, deterministic) -> Q1 = Q2 - gap varies over
        # [min(sweep)-gap_max, max(sweep)-gap_min]. SAME draw for positives AND
        # negatives -> no page-level quality leak, the model must localize.
        gap = int(jrng.integers(gap_min, gap_max + 1))
        q1 = max(40, q2 - gap)
        source_path = doc_sources[i]
        is_negative = bool(master.random() < neg_ratio)

        # Number of forgeries on this doc (0 if negative): goes into the
        # FILENAME (user request). k>=1 for a positive -> size cap.
        k = 0 if is_negative else int(jrng.integers(n_min, n_max + 1))

        # OUTPUT NAMES = stem of the SOURCE file + number of forgeries, for
        # source-doc -> artifacts traceability:
        #   image {stem}_{k}.jpg | mask {stem}_mask_{k}.png | ELA {stem}_ela_{k}.png
        # doc_id == stem of the image (= {stem}_{k}) -> consistent with detection_eval.
        # (The `doc_prefix`/type stays carried by the subfolder + the `type` column.)
        base_stem = _safe_stem(source_path)
        stem = base_stem
        d = 1
        while f"{stem}_{k}" in used_ids:        # source reused (with replacement) -> suffix
            d += 1
            stem = f"{base_stem}_{d}"
        used_ids.add(f"{stem}_{k}")
        doc_id = f"{stem}_{k}"

        if is_negative:
            jobs.append(Job(
                doc_id=doc_id, seed=seed, q2=q2, is_negative=True,
                source_path=source_path, edit_type=None, size_class=None,
                alignment=None, donor_path=None, q1=q1, n_forgeries=0, stem=stem,
            ))
            continue

        # SIZE CAP based on k: the more there are, the smaller they are (avoids
        # covering the whole page). k=1 -> all sizes ... k>=len(classes) -> small only.
        max_idx = min(len(size_classes) - 1, max(0, len(size_classes) - k))
        size_class = str(size_classes[int(jrng.integers(0, max_idx + 1))])

        if edit_type == "substitution":
            alignment = "N/A"
            donor_path = None
        else:
            alignment = "aligned" if jrng.random() < aligned_ratio else "misaligned"
            donor_path = None
            if edit_type == "splice":
                # Intra-corpus splice: another document from the batch.
                others = [p for p in source_paths if p != source_path] or source_paths
                donor_path = str(master.choice(others))

        jobs.append(Job(
            doc_id=doc_id, seed=seed, q2=q2, is_negative=False,
            source_path=source_path, edit_type=edit_type, size_class=size_class,
            alignment=alignment, donor_path=donor_path, q1=q1, n_forgeries=k, stem=stem,
        ))
    return jobs


# ------------------------------------------------------------------ worker
def _run_job(job: Job, cfg: dict, out_dirs: dict, nonstd_thr: float) -> dict:
    """Runs a full job: forge -> single Q2 pass -> annotation -> write."""
    rng = np.random.default_rng(job.seed)
    allow_lossless = bool(cfg["probe"].get("allow_lossless", False))
    src = decode_source(job.source_path, nonstd_thr, allow_lossless=allow_lossless)

    q2_sub_str = _SUBSAMPLING_INT2STR.get(DEFAULT_Q2_SUBSAMPLING, "4:2:0")
    ann_cfg = cfg["annotator"]

    # Optional square resize to the model's input resolution. FORENSIC ORDER MATTERS:
    # the whole chain (Q1->Q2, forgery, ELA) runs at NATIVE resolution, and the resize
    # is applied ONLY to the delivered artifacts at the very END (see below). Resizing
    # the source FIRST turns every text stroke into high frequency -> ELA saturates on
    # ALL text and the forgery no longer stands out (measured: ratio collapses). Computing
    # ELA natively THEN downscaling the ELA preserves the contrast — this is exactly what
    # detection_eval._ela_one does (native ELA -> BILINEAR resize to 384).
    resize_to = int(ann_cfg["input_res"]) if bool(ann_cfg.get("resize_square", False)) else None

    # Working base: source ALWAYS recompressed at q -> establishes the compression
    # history of the "original document" (background double-compressed after the
    # final pass at q). Q0 is still read/logged but is no longer the effective history.
    base_rgb = recompress_to_q1(src.rgb, job.q1, subsampling=DEFAULT_Q2_SUBSAMPLING)
    q1_effective = job.q1

    forgery_bboxes: list = []
    if job.is_negative:
        edited = base_rgb
        mask = np.zeros((src.height, src.width), dtype=np.uint8)
        edit_type, size_class, alignment, bbox, forge_res = (
            "authentic", "none", "N/A", None, None)
    else:
        area_range = tuple(cfg["size_classes"][job.size_class])
        feather_range = tuple(cfg["forger"]["feather_radius_px"])
        donor_rgb, donor_id, donor_q = None, None, None
        if job.edit_type == "splice":
            # The splice inserts a region with a FOREIGN compression history.
            donor = decode_source(job.donor_path, nonstd_thr, allow_lossless=allow_lossless)
            donor_rgb, donor_id = donor.rgb, donor.source_id
            if donor.q0 == -1:
                # Lossless donor (PNG): without this the region would have NO JPEG
                # history and would blend in with a substitution. We impose a
                # foreign quality Q_donor (drawn from quality_sweep) -> a real foreign grid.
                q_choices = [int(q) for q in cfg["compression"].get("quality_sweep", [job.q2])]
                donor_q = int(rng.choice(q_choices or [job.q2]))
                donor_rgb = recompress_to_q1(donor_rgb, donor_q, subsampling=DEFAULT_Q2_SUBSAMPLING)
            # (JPEG donor: native Q0 history kept, already foreign to the background.)
        # int (square) or [min_width, min_height]; normalized in forger.forge().
        min_region_px = cfg["forger"].get("min_region_px", forger_mod.JPEG_BLOCK)
        # Lever 1: placement on real content (ink) rather than in the void.
        on_content = bool(cfg["forger"].get("place_on_content", False))
        min_content_frac = float(cfg["forger"].get("min_content_frac", 0.0))
        # Fraction of substitutions rendered in COLOR (0 = all black).
        color_prob = float(cfg["forger"].get("subst_color_prob", 0.0))

        # ---- k forgeries (SAME type, SAME size class) accumulated ----
        # Each pass edits the current image and avoids the already-forged zones
        # (forbid). The final mask is the UNION of the k footprints.
        edited = base_rgb
        mask = np.zeros((src.height, src.width), dtype=np.uint8)
        forge_res = None
        for _ in range(max(1, job.n_forgeries)):
            forge_res = forger_mod.forge(
                img=edited, edit_type=job.edit_type, size_class=job.size_class,
                area_range=area_range, alignment=job.alignment,
                feather_range=feather_range, rng=rng,
                donor=donor_rgb, donor_id=donor_id,
                min_region_px=min_region_px, forbid=forgery_bboxes,
                on_content=on_content, min_content_frac=min_content_frac,
                color_prob=color_prob,
            )
            edited = forge_res.image
            mask = np.maximum(mask, forge_res.mask)
            forgery_bboxes.append(forge_res.bbox)      # avoids the next overlap
        if donor_q is not None:
            forge_res.extra["donor_q"] = donor_q   # foreign JPEG quality of the splice
        edit_type, size_class, alignment = job.edit_type, job.size_class, job.alignment
        bbox = ann.bbox_from_mask(mask)            # bounding box of the UNION

    # Output names = stem of the SOURCE file + number of forgeries (0 if negative).
    #   image {stem}_{n}.jpg (== doc_id) | mask {stem}_mask_{n}.png | ELA {stem}_ela_{n}.png
    n = 0 if job.is_negative else int(job.n_forgeries)

    # ---- SINGLE Q2 pass on the entire composite image (mandatory rule) ----
    #      -> images/ folder. Saved at NATIVE resolution so the ELA below is computed
    #      natively (the forensic order); the optional square resize happens AFTER.
    img_path = os.path.join(out_dirs["images"], f"{job.doc_id}.jpg")
    save_q2(edited, img_path, job.q2, subsampling=DEFAULT_Q2_SUBSAMPLING)

    # ---- ELA (3 qualities -> RGB) on the re-read FINAL JPEG, at NATIVE resolution ----
    ela_cfg = cfg.get("ela", cfg.get("ela_preview", {}))
    ela_center = int(ela_cfg.get("ela_quality", 90))
    ela_spread = int(ela_cfg.get("ela_spread", 8))
    ela_qs = ela_qualities(ela_center, ela_spread)
    ela_scale = float(ela_cfg.get("ela_scale", 15.0))
    ela_chroma = float(ela_cfg.get("chroma_suppress", 0.0))
    ela_gray = bool(ela_cfg.get("grayscale_input", False))
    ela_rgb = compute_ela_stack(img_path, ela_qs, ela_scale,
                                chroma_suppress=ela_chroma, grayscale_input=ela_gray)

    # ---- Optional square resize of the DELIVERED artifacts, applied AFTER native ELA ----
    # image: AREA/CUBIC · mask: NEAREST (no new labels) · ELA: BILINEAR (== detection_eval).
    # Re-saving the resized JPEG adds one compression, so its OWN ELA no longer matches;
    # the model input is the precomputed ela/ PNG (dataset.csv x_path), which is correct.
    # bboxes/mask are rescaled so mask, ELA and JSON stay pixel-aligned at (input_res)^2.
    if resize_to is not None:
        sx, sy = resize_to / float(src.width), resize_to / float(src.height)
        save_q2(_resize_square(edited, resize_to), img_path, job.q2,
                subsampling=DEFAULT_Q2_SUBSAMPLING)
        mask = cv2.resize(mask, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)
        ela_rgb = cv2.resize(ela_rgb, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        bbox = _scale_bbox(bbox, sx, sy)
        forgery_bboxes = [_scale_bbox(b, sx, sy) for b in forgery_bboxes]

    # ---- Exact pixel-level mask -> masks/ folder ----
    mask_path = os.path.join(out_dirs["masks"], f"{job.stem}_mask_{n}.png")
    cv2.imwrite(mask_path, mask)

    ela_path = os.path.join(out_dirs["ela"], f"{job.stem}_ela_{n}.png")
    Image.fromarray(ela_rgb, "RGB").save(ela_path)      # RGB: channels = (q_low, q_mid, q_high)

    # ---- Optional join/ QA board: [image | ELA | mask] side-by-side (visual check) ----
    join_path = None
    if bool(ela_cfg.get("join", False)) and out_dirs.get("join"):
        final_rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)  # exactly what is delivered
        board = join_board(final_rgb, ela_rgb, mask)
        join_path = os.path.join(out_dirs["join"], f"{job.stem}_join_{n}.png")
        Image.fromarray(board, "RGB").save(join_path)

    # ---- 24x24 patch grid ----
    labels, fracs = ann.patch_grid_labels(
        mask, input_res=ann_cfg.get("input_res", 384),
        patch_size=ann_cfg["patch_size"], grid=ann_cfg["patch_grid"],
        overlap_thr=ann_cfg["patch_positive_overlap"],
    )

    # ---- JSON metadata ----
    meta = ann.build_metadata(
        doc_id=job.doc_id, source=src, q2=job.q2, edit_type=edit_type,
        size_class=size_class, alignment=alignment, bbox=bbox, seed=job.seed,
        forge_result=forge_res, patch_size=ann_cfg["patch_size"],
        grid=ann_cfg["patch_grid"], overlap_thr=ann_cfg["patch_positive_overlap"],
        q2_subsampling=q2_sub_str,
    )
    meta["Q1_mode"] = "history_gap"            # Q1 < Q2 (gap = ELA signal)
    meta["Q1_effective"] = int(q1_effective)   # = Q1 (compression history of the background)
    # Multi-forgery: number of zones + bbox of EACH ONE (the `bbox` field is
    # the bounding box of the union, coarse; here the region-by-region ground truth).
    meta["n_forgeries"] = 0 if job.is_negative else int(job.n_forgeries)
    meta["forgery_bboxes"] = forgery_bboxes
    meta["patch_grid"] = labels.tolist()
    meta["ela"] = {"qualities": ela_qs, "center": ela_center, "spread": ela_spread,
                   "scale": ela_scale, "chroma_suppress": ela_chroma,
                   "grayscale_input": ela_gray,
                   "channels": "RGB = (q_low, q_mid, q_high)",
                   "file": os.path.basename(ela_path)}
    meta["files"] = {"image": os.path.basename(img_path),
                     "mask": os.path.basename(mask_path),
                     "ela": os.path.basename(ela_path)}
    # The .json accompanies the mask -> masks/ folder.
    json_path = os.path.join(out_dirs["masks"], f"{job.doc_id}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Manifest row ----
    n_mask_px = int((mask > 0).sum())
    mask_h, mask_w = mask.shape[:2]      # == (input_res)^2 when resized, native otherwise
    return {
        "id": job.doc_id,
        "source_id": src.source_id,
        "q0": src.q0,
        "q0_nonstandard": src.nonstandard,
        "q1_mode": "history_gap",
        "q1_effective": int(q1_effective),
        "q2": job.q2,
        "type": edit_type,
        "size_class": size_class,
        "alignment": alignment,
        "is_negative": job.is_negative,
        "n_forgeries": 0 if job.is_negative else int(job.n_forgeries),
        "bbox_x": bbox[0] if bbox else -1,
        "bbox_y": bbox[1] if bbox else -1,
        "bbox_w": bbox[2] if bbox else -1,
        "bbox_h": bbox[3] if bbox else -1,
        "n_mask_px": n_mask_px,
        "mask_frac": round(n_mask_px / float(mask_w * mask_h), 6),
        "n_pos_patches": int((labels > 0).sum()),
        "subsampling_src": src.subsampling,
        "seed": job.seed,
        "ela_quality": ela_center,
        "ela_qualities": ",".join(str(q) for q in ela_qs),
        "ela_scale": ela_scale,
        "path_img": os.path.relpath(img_path, out_dirs["root"]),
        "path_mask": os.path.relpath(mask_path, out_dirs["root"]),
        "path_json": os.path.relpath(json_path, out_dirs["root"]),
        "path_ela": os.path.relpath(ela_path, out_dirs["root"]),
    }


# module-level for picklability (ProcessPoolExecutor).
def _worker(args):
    job, cfg, out_dirs, nonstd_thr = args
    try:
        return _run_job(job, cfg, out_dirs, nonstd_thr)
    except Exception as exc:  # isolate per-document failures
        return {"id": job.doc_id, "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------ driver
def _execute_jobs(jobs: list[Job], cfg: dict, out_dirs: dict,
                  nonstd_thr: float, n_workers: int) -> tuple[list, list]:
    """Runs a list of jobs (sequential or parallel) -> (rows, errors)."""
    rows, errors = [], []
    payloads = [(j, cfg, out_dirs, nonstd_thr) for j in jobs]
    if n_workers <= 1:
        for p in payloads:
            r = _worker(p)
            (errors if "error" in r else rows).append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for r in ex.map(_worker, payloads, chunksize=4):
                (errors if "error" in r else rows).append(r)
    return rows, errors


def _write_batch(sub_root: str, sub_cfg: dict, report: dict, rows: list) -> tuple:
    """Makes a subfolder SELF-CONTAINED: distribution + config + manifest + report."""
    with open(os.path.join(sub_root, "distribution.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(sub_root, "run_config.yaml"), "w") as f:
        yaml.safe_dump(sub_cfg, f, sort_keys=False, allow_unicode=True)
    manifest_path = os.path.join(sub_root, "manifest.parquet")
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), manifest_path)
        write_folder_csvs(sub_root, rows)     # one CSV per folder (images/masks/ela)
    report_path = None
    try:
        import reporter
        report_path = reporter.write_report(sub_root)
    except Exception as exc:
        print(f"      (report not generated: {type(exc).__name__}: {exc})")
    return manifest_path, report_path


def run(cfg: dict, limit: Optional[int] = None, workers: Optional[int] = None) -> dict:
    src_dir = cfg["paths"]["source_dir"]
    if not src_dir:
        raise ValueError("paths.source_dir is empty: set the source JPEG folder.")
    out_root = cfg["paths"]["output_dir"]
    os.makedirs(out_root, exist_ok=True)

    nonstd_thr = float(cfg["probe"]["nonstandard_absdiff_threshold"])
    if limit is not None:  # n_docs PER TYPE (each subfolder will have `limit` docs)
        cfg = {**cfg, "orchestrator": {**cfg["orchestrator"], "n_docs": limit}}

    # 1) Probe the corpus ONCE (shared by all types).
    allow_lossless = bool(cfg["probe"].get("allow_lossless", True))
    print(f"[1/4] probe on {src_dir} ...")
    report = jpeg_probe.probe_dir(
        src_dir, recursive=cfg["probe"]["recursive"],
        candidate_ext=tuple(cfg["probe"]["candidate_ext"]),
        nonstandard_threshold=nonstd_thr, allow_lossless=allow_lossless,
    )
    source_paths = [r["path"] for r in report["records"]]
    if not source_paths:
        raise RuntimeError(
            f"No valid source image in {src_dir} "
            f"(exts={cfg['probe']['candidate_ext']}, allow_lossless={allow_lossless}).")
    n_lossless = report["summary"].get("n_lossless_kept", 0)
    print(f"      {len(source_paths)} sources kept "
          f"({n_lossless} of them lossless), {report['summary']['n_excluded']} excluded.")

    # Two passes Q1<Q2: source recompressed at Q1 (history of the "original
    # document"), then final save at Q2 (high). Lossless PNGs are handled
    # natively — the history comes entirely from the Q1 pass.
    sweep = [int(q) for q in cfg["compression"]["quality_sweep"]]
    gap_min, gap_max = parse_gap_range(cfg)
    ela_cfg = cfg.get("ela", cfg.get("ela_preview", {}))
    ela_q = int(ela_cfg.get("ela_quality", 90))
    ela_spread = int(ela_cfg.get("ela_spread", 8))
    # RANGE of Q1 actually generated (OPTION A: gap drawn per doc): Q1 = Q2 - gap.
    q1_lo = max(40, min(sweep) - gap_max)
    q1_hi = max(40, max(sweep) - gap_min)
    q1_reco = int(round((q1_lo + q1_hi) / 2.0))       # probe center ≈ middle of the Q1 range
    channels = ela_qualities(ela_q, ela_spread)       # 3 FIXED probes = RGB channels (inference strategy)
    # HARD GUARD: no probe must coincide with a Q2 (image at the probe's fixed
    # point -> ELA collapses to 0, no separability).
    clash = sorted(set(channels) & set(sweep))
    if clash:
        raise ValueError(
            f"ELA probe(s) {clash} coincide(s) with QUALITY_SWEEP ({sweep}): ELA "
            f"would collapse to 0. Keep ELA_QUALITY±ELA_SPREAD outside the sweep "
            f"(Q1 ∈ [{q1_lo}, {q1_hi}]).")
    # NB (measured, option A): a FIXED probe (e.g. 59/67/75) stays the best even
    # when Q1 varies over the whole range — the forgery (Q2 only) shows up at many
    # probe qualities, not only exactly at Q1. No need to widen the probe;
    # it is the DIVERSITY of Q1 in the data (gap drawn per doc) that gives
    # robustness to the inference Q1. Warn only if the probe is absurd.
    if not (40 <= ela_q < min(sweep)):
        print(f"      ⚠️  ELA_QUALITY={ela_q} outside [40, {min(sweep)}[ : probe of little use.")
    print(f"      compression: Q2 (sweep) = {sweep}, gap ∈ [{gap_min}, {gap_max}] "
          f"-> Q1 ∈ [{q1_lo}, {q1_hi}] (drawn per doc); ELA probes (RGB) = {channels}")

    # Corpus-level snapshot at the root (common reference).
    with open(os.path.join(out_root, "distribution.json"), "w") as f:
        json.dump(report, f, indent=2)

    # 2) One COMPLETE, SEPARATE batch per edit type -> one subfolder each.
    edit_types = resolve_edit_types(cfg)
    n_workers = workers if workers is not None else int(cfg["orchestrator"]["n_workers"])
    n_docs_cfg = cfg["orchestrator"]["n_docs"]
    # None/empty => as many documents as source images (y = x).
    n_per = len(source_paths) if n_docs_cfg is None else int(n_docs_cfg)
    print(f"[2/4] types: {edit_types}  (one subfolder each, {n_per} docs/type, "
          f"global seed {cfg['orchestrator']['seed']})")

    results = {}
    for ti, etype in enumerate(edit_types):
        sub_root = os.path.join(out_root, etype)
        sub_dirs = {"root": sub_root,
                    "images": os.path.join(sub_root, "images"),
                    "masks": os.path.join(sub_root, "masks"),
                    "ela": os.path.join(sub_root, "ela"),
                    "join": os.path.join(sub_root, "join")}
        _dirs = ["images", "masks", "ela"]
        if bool(cfg.get("ela", {}).get("join", False)):
            _dirs.append("join")
        for _k in _dirs:
            os.makedirs(sub_dirs[_k], exist_ok=True)
        # Effective config of the subfolder: self-describing (scalar edit_type).
        sub_cfg = {
            **cfg,
            "paths": {**cfg["paths"], "output_dir": sub_root},
            "forger": {**cfg["forger"], "edit_types": [etype], "edit_type": etype},
        }
        jobs = plan_jobs(sub_cfg, source_paths, etype, doc_prefix=etype, type_index=ti)
        print(f"[3/4] [{etype}] {len(jobs)} jobs -> {n_workers} worker(s) ...")
        rows, errors = _execute_jobs(jobs, sub_cfg, sub_dirs, nonstd_thr, n_workers)
        print(f"      [{etype}] {len(rows)} written, {len(errors)} errors.")
        for e in errors[:5]:
            print(f"        ERROR {e['id']}: {e['error']}")
        manifest_path, report_path = _write_batch(sub_root, sub_cfg, report, rows)
        _print_batch_summary(rows)
        results[etype] = {"root": sub_root, "rows": rows, "errors": errors,
                          "manifest": manifest_path, "report": report_path}

    print(f"[4/4] {len(edit_types)} subfolder(s) under {out_root}/ : "
          f"{', '.join(edit_types)}")
    print(f"      Aggregate: ./scripts/aggregate.sh")
    return {"output_dir": out_root, "edit_types": edit_types,
            "distribution": os.path.join(out_root, "distribution.json"),
            "batches": results}


def _print_batch_summary(rows: list[dict]) -> None:
    if not rows:
        return
    from collections import Counter
    print("=" * 64)
    print(f"  Documents      : {len(rows)}")
    print(f"  Negatives      : {sum(r['is_negative'] for r in rows)}")
    print(f"  Types          : {dict(Counter(r['type'] for r in rows))}")
    print(f"  Alignment      : {dict(Counter(r['alignment'] for r in rows))}")
    print(f"  Sizes          : {dict(Counter(r['size_class'] for r in rows))}")
    pos = [r for r in rows if not r["is_negative"]]
    if pos:
        print(f"  Forgeries/doc  : {dict(sorted(Counter(r['n_forgeries'] for r in pos).items()))}")
    print(f"  Q2             : {dict(Counter(r['q2'] for r in rows))}")
    print("=" * 64)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="orchestrator — batch generation of synthetic forgeries.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--src", default=None, help="Overrides paths.source_dir.")
    ap.add_argument("--out", default=None, help="Overrides paths.output_dir.")
    ap.add_argument("--n", type=int, default=None, help="Overrides orchestrator.n_docs (useful for a smoke test).")
    ap.add_argument("--workers", type=int, default=None, help="Overrides orchestrator.n_workers.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.src:
        cfg["paths"]["source_dir"] = args.src
    if args.out:
        cfg["paths"]["output_dir"] = args.out
    run(cfg, limit=args.n, workers=args.workers)


if __name__ == "__main__":
    main()
