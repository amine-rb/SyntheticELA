# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

SyntheticEla generates a **pixel-annotated synthetic dataset of document forgeries** from any
corpus of authentic document images, and ships a **detection/localization evaluation module**
(for an AnoViT-style model). It reproduces the forensic scenario of **double JPEG compression**:
a fraudster edits an already-saved document and re-saves it, creating a compression-history
inconsistency between the forged region(s) and the rest of the page — the signal that Error Level
Analysis (ELA) reveals. This is master's research code (see `markdown/plan.md`, `markdown/
instruction.md`, `markdown/RAPPORT_STAVER.md`, `markdown/SCHEMA_ROADMAP.md` for the research
design and rationale behind pipeline decisions).

The interface is deliberately minimal: **one file to edit (`config.sh`)**, **three commands to
run (`scripts/*.sh`)**. Everything else in `python/` is implementation detail not meant to be
touched for normal usage.

## Commands

No build step, no package installation of this repo itself (no `setup.py`/`pyproject.toml`) —
`python/` is a set of flat modules made importable via `PYTHONPATH` (set by the shell scripts, not
a package, so it survives macOS `spawn` multiprocessing workers).

```bash
pip install -r requirements.txt        # Pillow, NumPy, OpenCV, PyYAML, PyArrow
pip install torch scikit-learn scipy   # only needed for python/detection_eval.py
```

Edit `config.sh` (paths, sweep, edit types, sizes, etc. — every parameter lives there), then:

```bash
./scripts/run.sh          # generate: one self-contained subfolder per edit type
./scripts/aggregate.sh    # merge chosen type subfolders -> OUTPUT_DIR/_aggregated/
./scripts/preview.sh      # QA contact sheets (image | ELA | mask) for one subfolder
```

One-off CLI overrides without touching `config.sh`:

```bash
./scripts/run.sh --src OTHER/DIR --out OTHER/OUT --n 500 --workers 8
./scripts/aggregate.sh --types substitution splice --mode symlink
./scripts/preview.sh --out "$OUTPUT_DIR/_aggregated" --n 20
```

Regenerate a run's report standalone:

```bash
python python/reporter.py --out <TYPE_SUBFOLDER>
```

There is no test suite in this repo.

Each of the three shell scripts sources `scripts/_common.sh`, which loads `config.sh`, sets
`PYTHONPATH="$ROOT/python"`, and renders the shell variables into a temp `config.yaml` consumed by
the Python side (`orchestrator.load_config`) — `config.sh` is the only source of truth; nothing
else should be edited for a normal run.

## Architecture

### Forensic pipeline (what a single generated document goes through)

```
source image (decoded; Q0 read if JPEG, else lossless)
  -> recompress @Q1        (background = "original document", history at MEDIUM quality Q1)
       -> [positive] forger paints k substitutions (FRESH pixels, never saw Q1) -> save @Q2
       -> [negative]                                                                save @Q2
            -> annotator: exact mask + bbox + 24x24 patch grid + JSON metadata
```

The signal is a **compression-quality gap**: each document uses **two different qualities,
`Q1 < Q2`**. `Q2` (the final save, high) is drawn per-document from `QUALITY_SWEEP`; `Q1` (the base
pass, medium) is `Q2 - Q1_GAP`. The background and all authentic text carry the `Q1 -> Q2` history;
a substitution is painted in **fresh pixels between the two passes**, so it only ever saw `Q2`.

**Probe ELA at ≈ `Q1`** (`ELA_QUALITY` ≈ median(`Q2`) − `Q1_GAP`), the background's recompression
fixed point: this minimizes authentic-text ELA and maximizes the forgery's. Measured
`forged/authentic-text`: **≈3.2 probing at `Q1`(≈67)** vs ≈1.8 at 90 (and the region is ~2.5× brighter).
The gap also matters: `Q1_GAP` 22→~2.5, **28→~3.2**, 32 plateaus.

**The gap is mandatory.** Under `Q1 == Q2` a substitution is *indistinguishable from ordinary text*
in ELA (measured ratio ≈ 1.0: every text edge lights up regardless of history — the earlier
"3.5× in/out-mask" number only measured text-vs-white-paper and was misleading). A single quality
does **not** produce a localizable signal; that approach was tried and removed. But the old
`Q1<Q2 / Q1=Q2 / Q1>Q2` regime machinery (min-gap, `native`/`controlled`/`auto` modes) is **not**
back — the surface is a few scalars: `QUALITY_SWEEP` (the `Q2` values), `Q1_GAP`, and `ELA_QUALITY`
/`ELA_SPREAD` (probe center ≈ `Q1` and channel spread).

**ELA output is a colour RGB image**: `ela/*.png` stacks three ELA probes bracketing `Q1`
(`ELA_QUALITY ± ELA_SPREAD`, e.g. 59/67/75) as R/G/B. The colour comes from *quality diversity*, not
chroma (chroma ELA was measured anti-correlated here — it lights up authentic logos, not forgeries).
The forgery is bright in all three → white/tinted; authentic text stays dark. This gives the model 3
channels of real information (= `detection_eval`'s E2 mode).

Invariants that create the signal: (1) `Q1 < Q2` (the gap), enforced by `Q1_GAP > 0`; (2) the three
ELA probe qualities `∉ QUALITY_SWEEP` — if a probe equals a `Q2`, the whole image sits at that fixed
point and ELA collapses to ~0 (orchestrator raises on the center; keep the spread clear of `Q2` too);
and (3) probe ≈ `Q1` for contrast (orchestrator prints the recommended value and warns if far off).
Negatives get the exact same `Q1 -> Q2` double pass, so their background is indistinguishable from a
positive's → no global cue, the model must **localize**. PNG (lossless) corpora work natively — the
history comes entirely from the `Q1` pass.

`copy_move`/`splice` still exist in `forger.py` but are weaker (their region already carries a JPEG
history, unlike freshly-painted substitution); this pipeline targets substitution.

Three edit types (`forger.py`), each with different alignment implications:
- **substitution** — writes a plausible value (amount/date/quantity/code in the document's own
  format) sized to the document's *measured* text height (connected-component analysis of the ink
  mask, not a guessed fraction), over existing content. No prior 8x8 grid — `alignment = N/A`.
- **copy_move** — region copied from the same image (carries the Q1 grid); alignment depends on
  whether the copy offset is a multiple of 8.
- **splice** — region taken from a different corpus document (foreign grid); same alignment logic.

Forgeries are placed on real ink content (`PLACE_ON_CONTENT`, Otsu-based detection) rather than
blank margins, since a flat region on white margin is nearly invisible to ELA. Multiple forgeries
per document are supported (`N_FORGERIES=(min max)`); more forgeries per doc automatically caps
the allowed size class so a document is never mostly-forged. `MIN_REGION_PX` guarantees a minimum
forged rectangle (rounded up to a multiple of 8); a source too small to fit it fails the document
loudly rather than emitting a degenerate empty-mask positive.

### Module map (`python/`, one flat file per stage, no package)

| Module | Role |
| --- | --- |
| `jpeg_probe.py` | Estimates Q0 / quant table / subsampling / dims from the source corpus (or flags lossless) → feeds `distribution.json`. |
| `recompress.py` | Decodes the source; `recompress_to_q1` (base pass @`Q1`, establishes medium-quality history); `save_q2` (final save @`Q2` > `Q1`). |
| `forger.py` | Implements substitution / copy_move / splice, multi-forgery placement, edge feathering (anti-tell), minimum-size enforcement. |
| `annotator.py` | Produces the exact pixel mask, bbox, 24x24 patch-label grid, JSON metadata. |
| `orchestrator.py` | Batch driver: probes the corpus, plans deterministic per-document jobs (seed derived from the global seed, independent of worker/order), runs `forger -> recompress -> annotator` in a `ProcessPoolExecutor`, writes the manifest. Also owns `load_config` (YAML → config), consumed by other modules. |
| `aggregate.py` | Merges the per-type output subfolders into `_aggregated/` (copy/symlink/hardlink), concatenating manifests with a re-filterable `type` column. |
| `reporter.py` | Builds each run's `REPORT.md` (source/quality, composition, integrity checks, sampled ELA separability by type). |
| `ela_preview.py` | Renders image\|ELA\|mask QA contact sheets at a fixed global ELA scale, using the **same RGB 3-quality (≈`Q1`) stack** as the generated output (`orchestrator.compute_ela_stack`). |
| `main.py` | Thin CLI entry point invoked by `run.sh`; delegates to `orchestrator.main()`. |
| `detection_eval.py` | Standalone eval module meant to be **copied into the downstream training codebase** (torch/sklearn/scipy) — ELA cache building, dev/authentic datasets, best-detection-checkpoint tracking, and the final AUPRC/AUPRO/Dice/IoU evaluation protocol. Not exercised by the generation pipeline itself. |

Import direction is strictly one-way: `orchestrator` imports `jpeg_probe`, `recompress`, `forger`,
`annotator`; `ela_preview` and `reporter` read already-generated output plus `orchestrator.
load_config`; `detection_eval` has no dependency on the rest of `python/` (it's copied elsewhere).

### Output layout

Each entry in `EDIT_TYPES` produces one **self-contained** subfolder (everything needed for
downstream train/eval, no cross-references to other types):

```
<out>/distribution.json               # corpus probe, shared across types
<out>/<type>/
    images/<type>_<id>.jpg            # final document (background Q1->Q2, forged zone Q2-only)
    images/images.csv                 # per-folder CSV (id, image, type, is_negative, quality, size_class, ...)
    masks/<type>_<id>_mask.png        # exact binary mask (union of k forged zones)
    masks/<type>_<id>.json            # Q0/Q1/Q2, type, size, alignment, n_forgeries, bboxes, seed, 24x24 grid, ela
    masks/masks.csv                   # per-folder CSV (id, mask, json, n_mask_px, mask_frac, bbox_*)
    ela/<type>_<id>_ela.png           # ELA RGB (3 qualities ≈ Q1 stacked as channels, native res) on the final JPEG
    ela/ela.csv                       # per-folder CSV (id, ela, image, ela_quality, ela_scale)
    manifest.parquet                  # one row per document (adds path_ela, ela_quality, ela_scale)
    distribution.json                 # copy of the corpus probe (self-contained)
    run_config.yaml                   # frozen effective config for this run
    REPORT.md                         # human-readable run report
    ela_preview/                      # image|ELA|mask QA sheets (after ./scripts/preview.sh)
```

Three per-artifact folders each carry a self-contained CSV (`images/`, `masks/`, `ela/`), so a
folder can be loaded without the Parquet manifest. ELA is a first-class output computed at
generation time on the re-read final JPEG (RGB, 3 qualities ≈ `Q1`, native resolution, fixed global
`ELA_SCALE`, pixel-aligned with image and mask), for **every** image incl. negatives. `orchestrator.
write_folder_csvs` and `orchestrator.compute_ela_stack` produce these; `aggregate.py` reuses
`write_folder_csvs` so `_aggregated/` has the same shape. `detection_eval.py` (training side) reads
images from `images/` via `build_ela_cache`, and masks from `masks/` via `SyntheticDevDataset`.

Negative (authentic, empty-mask) documents are named `<type>_authentic_<id>` — never a
"forgery" entry with an empty mask. IDs are type-prefixed so they're globally unique across
`./scripts/aggregate.sh`.

### Reproducibility

A single global `SEED` derives one deterministic seed per document (logged in its JSON); output is
identical regardless of `N_WORKERS` since each job only depends on its own seed, not execution
order. Each edit type draws from a decorrelated random stream, so aggregating types produces no
exact duplicates.

## Working in this repo

- `config.sh` is the only file a normal user/session should edit to change behavior; treat edits
  to `python/*.py` as changes to the pipeline's implementation, not its configuration surface.
- `SOURCE_DIR`/`OUTPUT_DIR` in `config.sh` are currently absolute machine-local paths — don't
  assume they're portable when reasoning about other environments.
- PNG (lossless) and JPEG corpora are both handled with no mode to set — the source is always
  recompressed at `Q1` first, so the history comes from that pass.
- When changing anything in the forensic chain (`forger.py`, `recompress.py`, `annotator.py`,
  `orchestrator.py`'s quality sampling), preserve the invariants that create the signal:
  (1) the background/authentic text is compressed at `Q1` then `Q2` with **`Q1 < Q2`** (the gap is
  the whole signal — under `Q1 == Q2` a substitution is indistinguishable from ordinary text in
  ELA); (2) the substitution is painted in **fresh pixels between** those two passes (so it only
  ever saw `Q2`); and (3) probe ELA at **≈ `Q1`** with the three probe qualities `∉ QUALITY_SWEEP`
  (probing at `Q1` maximizes contrast; probing at a `Q2` collapses ELA to ~0 everywhere).
  Negatives must get the **same** `Q1 -> Q2` double pass as positives, or the model learns a global
  double-compression cue instead of localizing. Validate with the report's `forged/authentic-text`
  ratio (§4) — the meaningful metric; `forged/paper` is always high and misleading on its own.
