# SyntheticEla — Annotated document-forgery dataset generator + evaluation

Generates a **pixel-annotated synthetic dataset of document forgeries**
from any corpus of authentic document images, and ships the associated
**detection/localization evaluation** module (AnoViT).

Reproduces the forensic scenario of **double JPEG compression** — a fraudster edits
an already-saved document and then re-saves it — and provides, for each document,
the exact mask, a patch-label grid, and the metadata.

Deliberately minimal interface: **one file to edit (`config.sh`)**,
**three commands to run (`scripts/*.sh`)**.

---

## 0. Repository layout

```
config.sh          <- THE only file to edit (all parameters)
scripts/           <- THE only commands to run
   run.sh          - multi-corpus generation + final merged dataset
   aggregate.sh    - merge a corpus's type subfolders -> _aggregated/
   ela.sh          - ELA RGB (same parameters) on any image folder
   _common.sh      - internal (loads config.sh, renders the config, locates python)
python/            <- code (12 modules; normal usage = don't touch)
markdown/          <- research plan, roadmap schema, notes
requirements.txt
README.md          <- this file
```

No Python package: the modules in `python/` are flat files, made importable
via `PYTHONPATH` by the scripts (compatible with macOS *spawn* workers).

---

## 1. Installation

```bash
pip install -r requirements.txt   # Pillow, NumPy, OpenCV, PyYAML, PyArrow
```
For the evaluation (§11) only: `pip install torch scikit-learn scipy`.

---

## 2. Quick start

Two steps:

1. **Edit `config.sh`** — organized into 4 sections: **[COMMON]** (including the ELA
   recipe, identical between generation⇄inference), **[RUN]** (the corpora `RUN_SOURCE_DIRS`/`RUN_OUTPUT_DIRS`,
   `N_DOCS`, `N_FORGERIES`, `NEGATIVES_RATIO`…), **[ELA]** (defaults for `ela.sh`),
   **[AGGREGATE]** (defaults for `aggregate.sh`). Each parameter is used only by the
   script(s) of its section — **everything** is there.
2. **Run**:

```bash
./scripts/run.sh          # generate ALL RUN_* corpora + merge -> RUN_FINAL_DATASET_DIR
./scripts/aggregate.sh    # (optional) merge a corpus's TYPES -> OUTPUT_DIR/_aggregated/
./scripts/ela.sh --in FOLDER/IMAGES --out FOLDER/ELA   # ELA RGB on any folder
```

**Multi-corpus (new).** `run.sh` loops over the aligned pairs **`(RUN_SOURCE_DIRS[i] →
RUN_OUTPUT_DIRS[i])`** from `config.sh`: one complete dataset per corpus, **then**
automatically merges all `dataset.csv` files (all corpora + all types) into a single
index **`RUN_FINAL_DATASET_DIR/dataset.csv`** (same columns and same values as the
source CSVs, raw append, without a `corpus` column). `id` there is an integer `0..n-1`
**local to each source sub-CSV** (see `dataset.csv` below): it **repeats** in the
merged file (each corpus/type restarts at `0`) — accepted, it is no longer a unique
key. The provenance of each row (corpus, source `dataset.csv`, `id` within that
sub-CSV) is tracked separately in `RUN_FINAL_DATASET_DIR/sources.json`. A single
corpus? A single entry per list. `RUN_FINAL_DATASET_DIR=""` disables the merge.

**`dataset.csv` (training-ready index).** Each type subfolder gets a
`dataset.csv` in **ABSOLUTE paths**: `id (sequential integer 0..n-1, LOCAL to this
dataset.csv — NOT the file stem used by images.csv/masks.csv/ela.csv),
type (source png/jpeg), x_path (ELA = X input), negative (true/false), mask_path (empty
if negative)`. Self-contained → aggregatable by simple concatenation (but `id` is then
no longer unique, see above). `aggregate.sh` and `run.sh` carry it over as is.

One-off CLI override, without editing `config.sh`:

```bash
./scripts/run.sh --n 500 --workers 8              # applies to ALL RUN_* corpora
./scripts/aggregate.sh --types substitution splice --mode symlink
./scripts/ela.sh --in real_docs/ --out real_docs_ela/ --recursive   # forged OR authentic
```

JPEG **or** PNG sources: two qualities `Q1 < Q2` per document (base then final
save; the gap creates the ELA signal, §7). Each run writes a **`REPORT.md`** (§6).

---

## 3. Forensic pipeline

```
source (decoded; Q0 read if JPEG, else lossless)
   └─ recompress @Q1         ── "original document" = background + text at MEDIUM quality Q1
        ├─ POSITIVE : forger paints k substitutions (FRESH pixels, never seen by Q1)
        │       └─ save @Q2   ← background/text = Q1→Q2; substitution = Q2 only (Q1<Q2)
        └─ NEGATIVE : save @Q2 ← everything = Q1→Q2, empty mask
             └─ annotator : exact mask + bbox + 24×24 patch grid + JSON
```

Key point: **two qualities `Q1 < Q2`** per document. `Q2` (final save, high) is
drawn from `QUALITY_SWEEP`; `Q1 = Q2 − Q1_GAP` (base, medium). **It is the gap that
creates the signal**: the background and all **authentic** text carry the `Q1→Q2`
history, while the substitution, painted in **fresh pixels between the two
passes**, only ever saw `Q2`. Probed in ELA (at a quality **≠ `Q2`**), the region
that only saw `Q2` **stands out from ordinary text** (measured: *forged/authentic-text
≈ 1.8–1.9* with a varying `Q1` — cf. option A, §7). The negative undergoes exactly the same `Q1→Q2` double pass
(only the substitution is missing) → clean ELA, **indistinguishable from a positive's background**:
no global cue, the model must **localize**.

> ⚠️ **The gap is mandatory.** Under `Q1 == Q2`, a substitution is *indistinguishable
> from authentic text* in ELA (measured ratio ≈ 1.0: every text edge lights up,
> regardless of its history). The "3.5× inside/outside the mask" we used to
> measure only compared *text vs white paper* — misleading. Only two
> knobs, not the old regime machinery: `QUALITY_SWEEP` (the
> `Q2` values) and `Q1_GAP` (the gap). And `ELA_QUALITY` must differ from every `Q2` in the sweep
> (otherwise ELA collapses to 0 everywhere — the orchestrator refuses it).

Three edit types:
- **substitution** — writes a **plausible value** (amount, date, quantity, code in the
  document's format) at the **size of the document's text**, in dark ink, over
  existing content; no prior 8×8 grid (`alignment = N/A`).
- **copy_move** — region copied from the same image (carries the Q1 grid); offset ×8 → aligned, otherwise misaligned.
- **splice** — region from another corpus document (foreign grid); same alignment check.

---

## 4. Multiple forgeries & minimum size

**Several forgeries per document** (`N_FORGERIES=(min max)`). Each positive
document gets `k ~ U{min..max}` forgeries of the **same type**, with **disjoint**
footprints (no overlap), and the final mask is their **union**.

**Automatic size cap based on `k`**: the more forgeries there are, the smaller
they are, so they never cover the whole page. Concretely (4 ordered size
classes):

| `k` | allowed size classes |
| --- | --- |
| 1   | small · medium · large · very_large |
| 2   | small · medium · large |
| 3   | small · medium |
| 4   | small |
| 5   | small |

**Guaranteed minimum size** (`MIN_REGION_PX=(width height)`): floor of the
forged rectangle regardless of the class or source image, rounded up to the **next
multiple of 8** (JPEG grid) — `(10 10)` ⇒ actual minimum `16×16`. If a source is too
small to fit it on an axis, the document **fails cleanly** (error
logged by the orchestrator) instead of writing an **empty-mask** positive.

**Placement on real content** (`PLACE_ON_CONTENT=true`, `MIN_CONTENT_FRAC`): the
forgeries target a region carrying **text/digits** (Otsu-based ink detection)
instead of the white margins — a realistic scenario (a fraudster edits an existing
value) **and** an exploitable ELA signal. `MIN_CONTENT_FRAC` = the targeted ink
fraction (best-effort; falls back to the best location if the page is nearly empty). For
substitution, the **ink** is also forced dark/contrasted (real edges → real
signal), and splice/copy-move copy a **real ink region**. Without this placement,
a flat fill on a white margin produces a nearly invisible ELA region ("clean hole").

**Realistic substitution**: the injected value is **plausible** (amount/date/code in the
document's format) and rendered at the **document's ACTUAL text size** — glyph height
**measured** via connected components of the ink mask (not a guessed
fraction), lightly modulated (`×0.9–1.3`), in a **tight** box around the text.
The injected glyph is therefore ≈ **1× the document's text body** (~1 % of the page
height), instead of ~3× before. Goal: that the detection is attributable to the
**compression inconsistency** and not to a generator artifact (large text /
gibberish) — cf. the generator's "tell", `markdown/plan.md` §8.

> Multi-forgery: the manifest's `bbox_*` is the **bounding box of the union** of the k
> regions (thus large if they are scattered); the individual rectangles are in the
> JSON, field `forgery_bboxes`.

> ELA on a folder (`ELA_SCALE`): `./scripts/ela.sh --in … --out …` computes ELA at a **fixed
> global scale** (default 15, to align with `detection_eval.ELA_SCALE`) instead of a
> per-image max stretch — the preview reflects what the model "sees" and no longer
> washes out weak forgeries.

### Rotation augmentation (negatives only)

**`NEGATIVE_ROTATIONS`** (RUN) multiplies the authentic (negative) documents by
re-orienting them: **each selected negative is emitted once per angle** in the pool,
so `#negatives = (selected negatives) × len(pool)` (e.g. `N_DOCS=5`, all-negative,
`(90 180 270)` → **15** documents). Add `0` to the pool to also keep the upright
version (`(0 90 180 270)` → 20); `()` = upright only (no augmentation).

Each rotated copy is a **full standalone document** (own `image`/`mask`/`ela`/`json`),
with the **same seed / `Q1` / `Q2`** as its source — it stays a legit double-JPEG
authentic, only re-oriented. The rotation is applied to the finished (Q1-historied)
pixels **before** the `Q2` save + ELA, so `image`, `mask` and `ELA` stay pixel-aligned;
multiples of 90 are **exact** pixel permutations (the block grid is transposed, the
history preserved). Rotated files carry a `_rot{deg}` suffix; the applied angle is
logged in the JSON (`"rotation"`). **Positives are never rotated** (their mask/bbox
would have to follow) — rotation targets negatives, the orientation the model
overfits most.

**`ELA_ROTATIONS`** (ELA) is the analogous augmentation for `ela.sh`: on top of the
`0°` ELA of every image, it writes one extra ELA per rotation (rotated source → ELA
recomputed), suffixed `_rot{deg}`, with a `rotation` column in `ela.csv`. Here `0°` is
**always** emitted (it augments, it does not replace).

---

## 5. Outputs (one SELF-CONTAINED subfolder per type)

Each type in `EDIT_TYPES` produces a complete, independent subfolder (it lacks
no info needed for downstream training/evaluation):

```
<out>/distribution.json               # source corpus probe (shared across all types)
<out>/<type>/                          # e.g. substitution / copy_move / splice
     images/<stem>_<n>.jpg             # final document (background Q1→Q2; edited zone = Q2 only) — stem = SOURCE name, n = nb forgeries
     images/images.csv                 # id, image, type, is_negative, quality, size_class, n_forgeries, source_id, seed
     masks/<stem>_mask_<n>.png         # EXACT binary pixel mask (union of the k zones)
     masks/<stem>_<n>.json             # Q0/Q1/Q2, type, size, alignment, n_forgeries, bboxes, seed, 24x24 grid, ela
     masks/masks.csv                   # id, mask, json, is_negative, n_forgeries, n_mask_px, mask_frac, bbox_*
     ela/<stem>_ela_<n>.png            # ELA RGB (3 qualities ≈ Q1 stacked), native resolution, on the final JPEG
     ela/ela.csv                       # id, ela, image, ela_qualities, ela_scale, is_negative, type
     manifest.parquet                  # subfolder table (one row per document)
     distribution.json                 # corpus probe (copy, self-contained)
     run_config.yaml                   # frozen effective config
     REPORT.md                         # human-readable results report  ← §6
```

> ELA outside the pipeline: `./scripts/ela.sh --in <folder> --out <folder>` produces `*_ela.png` + `ela.csv`
> (same ≈ Q1 qualities and scale as the generation) for **any image folder**, forged
> or not — useful to apply ELA to real documents (Q1 unknown: see §7).

**Three folders, one CSV each** (`images/`, `masks/`, `ela/`): each CSV is
self-contained (one row per document, file name + metadata) → a folder can be
loaded without reading the Parquet manifest. **ELA is a first-class output**,
computed at generation time on the **re-read final JPEG** (the actual artifact), at native
resolution and fixed global scale (`ELA_SCALE`, aligned with `detection_eval.ELA_SCALE`)
→ one ELA file per image, **pixel-aligned** with image and mask. ELA is
produced for **all** images (positives AND negatives: the negative's ELA is the
"clean" reference).

**Traceable naming (`stem` = source file name, `n` = number of forgeries).**
Each artifact keeps the **source document's name** to trace output → source:
`X0001.jpg` → image `X0001_<n>.jpg`, mask `X0001_mask_<n>.png`, ELA `X0001_ela_<n>.png`,
JSON `X0001_<n>.json`. A **negative** has `n = 0` (empty mask); a positive with 3 zones has `n = 3`.
A **rotated negative** (`NEGATIVE_ROTATIONS`, §4) inserts a `_rot{deg}` suffix into the
stem: `X0001_rot90_0.jpg` (+ mask/ELA/JSON); the angle is also logged in the JSON
(`"rotation"`). The `id` (= image stem, `<stem>_<n>`) is consistent with `detection_eval` (the mask
is derived from it) — it is the `id` of `images.csv`/`masks.csv`/`ela.csv`. Sources are
drawn **without replacement** as long as the corpus is large enough (1 source doc ↔ 1 output); in
case of forced reuse, a suffix guarantees uniqueness.

**`dataset.csv` has its OWN `id`**, distinct from this one: a sequential integer `0..n-1`
local to each `dataset.csv` (see §3). When merging (`RUN_FINAL_DATASET_DIR`), each
sub-CSV restarts at `0` → `id` is **no longer unique** in the final file (no `corpus`
column added, no global renumbering); the provenance of each range
is tracked separately in `sources.json` (`corpus`, `source_dir`, `csv`, `ids` = the
original `id`s, integers too, from that sub-CSV). `./scripts/aggregate.sh`
gathers the chosen subfolders into `<out>/_aggregated/` (same structure — `images/`,
`masks/`, `ela/` + regenerated CSVs —, concatenated manifest, re-filterable `type` column).
Options: `--types t1 t2`, `--dest NAME`, `--mode copy|symlink|hardlink`
(symlink/hardlink = no disk duplication).

Manifest columns: `id, source_id, q0, q0_nonstandard, q1_mode, q1_effective,
q2, type, size_class, alignment, is_negative, n_forgeries, bbox_x/y/w/h, n_mask_px,
mask_frac, n_pos_patches, subsampling_src, seed, ela_quality, ela_qualities, ela_scale,
path_img, path_mask, path_json, path_ela`. (`bbox_*` = bounding box of the union; the individual
bboxes are in the JSON, field `forgery_bboxes`.)

---

## 6. `REPORT.md` — each run's results

Generated automatically at the end of each generation (regeneratable standalone:
`python python/reporter.py --out <SUBFOLDER>`). It contains:

1. **Source & config** (corpus, seed, `QUALITY_SWEEP`, Q0/dimensions),
2. **Composition** (types, sizes, alignment, negatives, quality `Q`),
3. **Integrity checks** (consistent positive/negative masks, area by size),
4. **Sampled ELA signal**: ELA **forged / authentic-text** ratio, by type —
   the real measure of "does the forgery stand out from ordinary text?" (and not
   text-vs-paper, which is always high and misleading on its own).

---

## 7. Compression: a quality gap `Q1 < Q2` per document

Two basic knobs: `QUALITY_SWEEP` (the `Q2` values, final save) and
`Q1_GAP` (the gap). Each document draws a `Q2` from the sweep and sets
`Q1 = Q2 − Q1_GAP`:

```
decoded source → recompress @Q1  (the "original document", MEDIUM quality Q1)
   → [positive] paint the substitution (fresh pixels) → save @Q2   (Q1 < Q2)
   → [negative]                                          save @Q2
```

- **Background + authentic text**: `Q1→Q2` history. Probed in ELA **≈ Q1** (their fixed
  point) it is **attenuated** → **dark ELA**.
- **Substitution**: painted **between** the two passes → only saw `Q2`, never `Q1` →
  **bright ELA**, it **stands out from the authentic text**.
- **Negative**: same `Q1→Q2` double pass, without substitution → **clean ELA**, background
  **identical** to a positive's → no global cue (the model localizes).

A PNG (lossless) corpus is handled natively: the history comes **entirely** from
the `Q1` pass.

### ELA probe quality: aim ≈ Q1 (crucial)

The ELA probe (`ELA_QUALITY`) must aim **≈ Q1** (= median(Q2) − `Q1_GAP`), the fixed
point of the background. That is what minimizes the authentic text's ELA and maximizes the
forgery's. Measured (StaVer corpus, `Q2∈{92,95,97}`, `Q1_GAP=28` → `Q1∈{64,67,69}`),
**forged / authentic-text** by probe:

| probe `ELA_QUALITY` | forged / **authentic-text** | region brightness |
| --- | --- | --- |
| 90 (old) | 1.8 | ×1 |
| **67 (≈ Q1, default)** | **3.2** | **×2.5** |

The `Q1_GAP` gap also matters (at a probe ≈ Q1): 22→~2.5, **28→~3.2**, 32→plateaus.
Under `Q1==Q2`: **≈ 1.0** ❌ (indistinguishable, no signal).

### `Q1_GAP` can be a RANGE — robustness to the inference Q1 (option A)

`Q1_GAP` accepts a **scalar** (nearly fixed Q1, old behavior) **or a range
`(min max)`**: a gap is then drawn **per document**, so `Q1 = Q2 − gap` **varies
over a whole band**. E.g. `QUALITY_SWEEP=(90 93 96)`, `Q1_GAP=(20 40)` → `Q1 ∈ [50, 76]`.

Why: a model trained on a single `Q1` (67) **overfits** it and fails on
an inference document whose base quality differs. Varying `Q1` in the
**data** forces the model to generalize.

> **Non-obvious (measured).** Do **not** widen the ELA probe to "cover" the `Q1`
> range. A **fixed narrow** probe (`67/8` → 59/67/75) stays best even at
> `Q1 ∈ [50, 80]` (grid: `67/8`→2.04 · `65/15`→1.74), because the forgery (Q2 only)
> shows up at **many** probe qualities, not only at `Q1` exactly. Robustness
> comes from `Q1` diversity **in the data**, not from the probe. With a varied `Q1`:
> forged/authentic-text ≈ **1.8–2.4** (vs ~3.2 at a single perfectly-probed `Q1` — the
> expected robustness/peak trade-off).

> **Scope (structural limit).** ELA detects a **re-compression** forgery
> (double JPEG). A document **never double-compressed** (single save, pristine image)
> has **no history discontinuity** → it is **not** detectable by this
> method. This is not a bug: it is ELA's validity domain.

### Reducing colored false positives (logos / stamps / seals)

Authentic **colored** furniture (logos, stamps) lights up in ELA **as strongly as a
forgery** (measured ELA ≈ 74 vs 82): it is the dominant false positive. But a substitution
is **black text** (chroma ≈ 3) whereas these elements are **colored** (chroma ≈ 40)
→ **color** is the discriminator, used **in the negative**. Two **cumulative** knobs
in `compute_ela_stack` (valid **as long as the forgery is achromatic**; disable
if forging colored regions):

- **`ELA_GRAYSCALE_INPUT`** (true/false) — grayscales the image **before** ELA. A
  colored logo has huge **per-channel** edges; grayscaling averages them → the **light**
  colored furniture's ELA collapses (74→~8), the black forgery keeps ~99 %. The RGB output survives (its
  3 channels come from the 3 **qualities**, not the image color).
- **`ELA_CHROMA_SUPPRESS`** (default 20; 0=off) — weights the ELA by
  `w = clip(1 − chroma/threshold, 0, 1)`, chroma measured on the **original color** image even if
  the ELA is grayscale → erases colored pixels **regardless of their luminance**
  (catches the **dark** colored furniture that grayscale alone leaves behind).

Cumulated (gray → chroma): logo/stamp → 0, forgery ~86 %, forged/authentic-text ratio
1.79 → 1.93. These same flags also apply to `scripts/ela.sh`
(`--grayscale-input` / `--chroma-suppress`).

### ELA output = RGB COLOR image (3 qualities)

`ela/*.png` is an **RGB image**: the 3 channels are the ELA at 3 qualities bracketing `Q1`
(`ELA_QUALITY ± ELA_SPREAD` = 59/67/75). The color comes from **quality diversity**
(not chroma — measured anti-correlated here). The forged region, bright in all 3 probes,
stands out as **white/tinted**; the authentic text stays dark. This gives the model **3 channels
of info** (= `detection_eval`'s E2 mode).

> **Two mandatory rules.** (1) `Q1 < Q2` (`Q1_GAP > 0`) — otherwise the forgery is
> indistinguishable from real text (ratio ≈ 1.0). (2) the 3 ELA qualities ≠ every `Q2` in the sweep
> — otherwise ELA **collapses to 0** (image at the probe's fixed point; the orchestrator
> refuses it). The orchestrator prints the recommended probe (≈ Q1) and warns if you
> stray from it. `copy_move`/`splice` remain generatable but weaker (region already carrying
> a JPEG history): this pipeline targets substitution.

---

## 8. Configuration (`config.sh`)

`config.sh` is the **only** file to edit. It is deliberately **minimal — one
comment per parameter**; the *why* of each setting lives here (§8) and in
§7 (compression/ELA). It is organized into 4 sections, each parameter belonging
to the script(s) that read it:

- **[COMMON]** — read by `run.sh`, `ela.sh` **and** `aggregate.sh`. Contains the **ELA
  recipe**, placed here because it **must be identical between generation and inference**
  (otherwise the model sees an out-of-distribution input).
- **[RUN]** — dataset generation (`run.sh`).
- **[ELA]** — defaults for inference ELA (`ela.sh`).
- **[AGGREGATE]** — defaults for merging types (`aggregate.sh`).

### Complete reference

| Variable(s) | Section | Role |
| --- | --- | --- |
| `PYTHON` | COMMON | Python interpreter (the one with the dependencies) — absolute path if no venv active |
| `ELA_QUALITY`, `ELA_SPREAD` | COMMON | ELA probe: center ≈ Q1 + spread → 3 RGB channels (here 52/60/68) (§7). **Identical generation⇄inference**; **fixed and narrow** probe even at varying Q1; the 3 channels **≠ every `Q2`** in the sweep (otherwise ELA→0) |
| `ELA_SCALE` | COMMON | **fixed global** ELA scale (no per-image max stretch, which would wash out weak forgeries). Aligned with `detection_eval.ELA_SCALE` (=15) |
| `ELA_GRAYSCALE_INPUT`, `ELA_CHROMA_SUPPRESS` | COMMON | colored-false-positive suppression (logos/stamps): grayscale before ELA + chroma suppression after (§7). **OFF if `SUBST_COLOR_PROB>0`** (otherwise the colored forgery is erased too) |
| `PROBE_RECURSIVE`, `CANDIDATE_EXT`, `ALLOW_LOSSLESS` | COMMON | reading a folder: recursive descent, accepted extensions, lossless support (PNG/TIFF/BMP, without Q0) |
| `NONSTANDARD_ABSDIFF_THRESHOLD` | COMMON | threshold for detecting a "non-standard" quantization table at the probe |
| `RUN_SOURCE_DIRS`, `RUN_OUTPUT_DIRS` | RUN | **aligned** (input, output) lists — one corpus per entry; `run.sh` loops over them (same length required) |
| `RUN_CORPUS_NAMES` | RUN | names (aligned) for the `corpus` label in `sources.json`; `()` → derived from the path |
| `RUN_FINAL_DATASET_DIR` | RUN | folder for the **merged** `dataset.csv` (all corpora + types, without a `corpus` column) + `sources.json` (provenance); `""` → no merge |
| `QUALITY_SWEEP` | RUN | `Q2` values (final save, high) drawn per document (§7) — **all ≠ ELA channels**. Wide range (72…96) = generalization to re-saves of varied qualities |
| `Q1_GAP` | RUN | compression gap: `Q1 = Q2 − Q1_GAP` (§7) — **the signal knob**; scalar (fixed Q1) **or range `(min max)`** → Q1 varies per doc (option A). Useful floor ≈ 15 (below that the gap is too small) |
| `EDIT_TYPES` | RUN | generated types — **one subfolder per type** (aim for `substitution`) |
| `ALIGNED_RATIO` | RUN | fraction of zones aligned on the 8×8 grid (for `copy_move` / `splice`) |
| `FEATHER_RADIUS_PX` | RUN | `(min max)` edge-feathering radius (anti-tell) |
| `SPLICE_SOURCE` | RUN | origin of the `splice` donor (`intra_corpus` = another doc from the same corpus) |
| `MIN_REGION_PX` | RUN | `(width height)` guaranteed min zone size, rounded up to the next multiple of 8 (§4) |
| `N_FORGERIES` | RUN | `(min max)` forgeries per doc — automatic size cap when k is large (§4). Enters the **file name** (§5) |
| `PLACE_ON_CONTENT`, `MIN_CONTENT_FRAC` | RUN | place the forgeries on real content (ink), with a minimum ink fraction under the zone (§4) |
| `SUBST_COLOR_PROB` | RUN | fraction of substitutions **in color** (random hue) vs black; `0`=all black. If `>0`, ELA color filters OFF (§7) |
| `SIZE_SMALL … SIZE_VERY_LARGE` | RUN | zone size classes (page fraction, `min max`) |
| `NEGATIVES_RATIO` | RUN | fraction of authentics per subfolder (high for an anomaly training set, ~0.5 for a test set) |
| `KEEP_BENIGN_COLORED` | RUN | preserve benign colored furniture (logos/stamps/headers) from real documents |
| `NEGATIVE_ROTATIONS` | RUN | rotation augmentation of **negatives** (§4): each negative emitted **once per angle** → `#neg = selected × len(pool)`. Add `0` to also keep upright; `()`=upright only. Positives never rotated |
| `INPUT_RES`, `PATCH_SIZE`, `PATCH_GRID`, `PATCH_POSITIVE_OVERLAP` | RUN | ground-truth patch grid (24×24 on 384, patch positive if overlap > threshold) |
| `ELA_N_SAMPLES` | RUN | number of QA panels `image \| ELA \| mask` |
| `SEED`, `N_DOCS`, `N_WORKERS` | RUN | reproducibility, batch **per type and per corpus**, parallelism; `N_DOCS=""` → as many documents as source images (y=x) |
| `ELA_INPUT_DIR`, `ELA_OUTPUT_DIR`, `ELA_RECURSIVE` | ELA | defaults for `ela.sh` when `--in/--out` are absent; `""` → 1st RUN corpus and `<output>/_ela_scan` |
| `ELA_ROTATIONS` | ELA | augment `ela.sh`: extra ELA per rotation angle (§4), suffixed `_rot{deg}`, `rotation` column in `ela.csv`; `0°` always emitted; `()`=none |
| `AGG_OUTPUT_DIR`, `AGG_TYPES`, `AGG_MODE`, `AGG_DEST` | AGG | defaults for `aggregate.sh`: corpus to merge (`""`→`RUN_OUTPUT_DIRS[0]`), types (`()`=all), mode `copy\|symlink\|hardlink`, subfolder name |

**Measured values (reference).** ELA probe on `Q1∈[40,81], Q2∈[72,96]` (50 docs,
color filter OFF), **forged/authentic-text** ratio: `[52,60,68]` → **3.89**
(best) · `[53,65,77]` → 3.86 · `[59,67,75]` (old) → 3.69 — hence the center `60`.
Widening the probe **serves no purpose** (the Q2-only forgery shows up at many qualities,
not only at Q1): robustness to the inference Q1 comes from **Q1 diversity
in the data** (gap drawn per doc), not from a wide probe (cf. §7).

CLI overrides (without editing `config.sh`): `--src`, `--out`, `--n`, `--workers`
(applied to all corpora); for `ela.sh`: `--in`, `--out`, `--recursive`, `--rotations`
(e.g. `--rotations 90 180 270`, or `--rotations` alone = none).

---

## 9. Code (`python/`, one file per module)

| Module | Role |
| --- | --- |
| `jpeg_probe`  | Probes Q0 / quant table / subsampling / dimensions (or marks lossless) → `distribution.json`. |
| `recompress`  | Decodes the source; `recompress_to_q1` (base @Q); `save_q2` (final save @Q, same quality). |
| `forger`      | substitution / copy_move / splice; multi-forgery; anti-tell feather; guaranteed min size. |
| `lexicon`     | Corpus of plausible values (FR/EN/date/number/code/phrase) injected by the substitution; ASCII-only (Hershey fonts). |
| `annotator`   | Exact mask, bbox, 24×24 patch grid, JSON metadata. |
| `orchestrator`| Scriptable batch, one self-contained subfolder per type, deterministic seeds, per-doc `Q1`/`Q2` draw, manifest + per-folder CSVs including `dataset.csv` (absolute paths). |
| `aggregate`   | Merges a corpus's **type** subfolders into a single dataset (`_aggregated/`). |
| `build_dataset` | Merges the `dataset.csv` files from **several corpora** into a single index + `sources.json` (provenance, without a `corpus` column) — called by `run.sh`. |
| `reporter`    | `REPORT.md` (run results + ELA separability). |
| `ela_scan`    | ELA RGB 3 qualities ≈ Q1 (same parameters as the output) on any image folder, forged or not. |
| `main`        | Entry point called by `run.sh`. |
| `detection_eval` | AnoViT detection/localization evaluation (§11) — to copy into the training codebase. |

---

## 10. Reproducibility

- Global seed → **deterministic seed per document** (logged). Output **identical**
  regardless of the number of workers (each job forges with its own seed).
- Each type gets a decorrelated random stream → no exact duplicates between
  subfolders at aggregation.
- `run_config.yaml` freezes the effective config (including the resolved `Q1_GAP` range) of each batch.

---

## 11. Detection/localization evaluation (`python/detection_eval.py`)

Standalone module to **copy into AnoViT's training codebase**. Implements the
protocol from the thesis (see `markdown/plan.md` §9.3 threshold, §9.3bis steering, §9.4
metrics). Additional dependencies: `torch, scikit-learn, scipy`.

**ELA cache (once, outside training).** The generated `ela/` folder contains the ELA
RGB (3 qualities ≈ Q1) at native resolution; training builds its own **384**
cache via `build_ela_cache`, recomputed from the **images** (`images/`): native-resolution
ELA → fixed global scale → resize 384 → grayscale PNG (never JPEG). **Aim
≈ Q1** (≈ 67 by default), NOT 90. A single pass serves E0/E1 (1 quality, 67) and E2 (3, bracketing Q1):

```python
from detection_eval import build_ela_cache
build_ela_cache("<out>/_aggregated/images", "cache/dev",  qualities=(59, 67, 75))  # ≈ Q1
build_ela_cache("path/authentics",          "cache/auth", qualities=(59, 67, 75))
```

**Loaders + best-detection steering (§9.3bis).**

```python
from torch.utils.data import DataLoader
from detection_eval import (SyntheticDevDataset, AuthenticELADataset,
                            pilot_subset, evaluate, BestDetectionTracker)

dev_ds  = SyntheticDevDataset("<out>/_aggregated/masks", "cache/dev", qualities=(67,))  # ≈Q1; 1st arg = masks/ folder
dev_ld  = DataLoader(pilot_subset(dev_ds, 400, seed=42), batch_size=48, num_workers=8)
tracker = BestDetectionTracker("experiments/E0/best_model.pt",
                               history_path="experiments/E0/auprc_curve.json", patience=15)

for epoch in range(100):
    train_one_epoch(model, train_loader)
    res = evaluate(model, dev_ld, error_mode="mae", metrics=("auprc",))
    tracker.update(epoch, res["pixel_auprc"], model)     # checkpoint = max dev AUPRC
    if tracker.should_stop:
        break
```

**Final evaluation (§9.4)** — full dev, threshold calibrated on the dev (max Dice),
**frozen** then run once on the real test:

```python
res = evaluate(model, dev_full, error_mode="mae", metrics="full", authentic_loader=auth_ld)
# {pixel_auprc, aupro, threshold, dice, iou, fpr_authentic, image_auroc, pixel_auroc}
```

Pitfalls: never binarize the scores before `evaluate` (AUPRC/AUPRO need the
continuous scores); keep the same `seed` for the pilot subset; calibrate the
threshold on the dev, never on the test.



❯ comment je trouve sur internet des données authentique images jpeg                                                                       

⏺ Pour ce pipeline, « authentique » = de vrais documents scannés/photographiés, à compression unique — donc authentiques par construction.
  Tu n'as pas besoin de vérifier quoi que ce soit si tu prends des corpus documentaires académiques connus. Voici où chercher.

  Les corpus documentaires standard (authentiques par nature)

  ┌──────────────┬────────────────────────┬─────────────┬────────────────────────────────────┐
  │   Dataset    │        Contenu         │   Format    │                 Où                 │
  ├──────────────┼────────────────────────┼─────────────┼────────────────────────────────────┤
  │ SROIE 2019   │ reçus scannés          │ JPEG        │ ICDAR / Kaggle (tu l'as déjà)      │
  ├──────────────┼────────────────────────┼─────────────┼────────────────────────────────────┤
  │ CORD         │ reçus (restaurants)    │ JPEG/PNG    │ HuggingFace naver-clova-ix/cord-v2 │
  ├──────────────┼────────────────────────┼─────────────┼────────────────────────────────────┤
  │ FUNSD        │ formulaires scannés    │ PNG         │ guillaumejaume.github.io/FUNSD     │
  ├──────────────┼────────────────────────┼─────────────┼────────────────────────────────────┤
  │ RVL-CDIP     │ 400k docs de bureau    │ TIFF (gris) │ HuggingFace aharley/rvl_cdip       │
  ├──────────────┼────────────────────────┼─────────────┼────────────────────────────────────┤
  │ DocLayNet    │ docs multi-domaines    │ PNG         │ HuggingFace ds4sd/DocLayNet        │
  ├──────────────┼────────────────────────┼─────────────┼────────────────────────────────────┤
  │ Tobacco3482  │ docs industriels       │ JPEG        │ Kaggle                             │
  ├──────────────┼────────────────────────┼─────────────┼────────────────────────────────────┤
  │ IAM / StaVer │ manuscrit / signatures │ PNG         │ (StaVer = ton 2e corpus)           │
  └──────────────┴────────────────────────┴─────────────┴────────────────────────────────────┘

  Où chercher en général

  1. HuggingFace Datasets → filtre par tâche Document, OCR. Le plus pratique (téléchargement direct, licences visibles).
  2. Kaggle → cherche « invoice », « receipt », « scanned documents », « forms ». Vérifie l'onglet License.
  3. ICDAR / concours académiques (RRC, robust reading) → jeux de docs réels, bien documentés.
  4. Roboflow Universe → beaucoup de docs annotés, export facile.

  Le critère qui compte

  - ✅ Vrais scans/photos de documents = authentiques, une seule compression. C'est ce que tu veux.
  - ❌ Évite les collections d'images retravaillées, screenshots, ou dossiers tampered/.

  Note sur le format JPEG

  Tu demandes du JPEG spécifiquement, mais rappel : le format n'est pas une contrainte ici. run.sh recompresse toujours le fond en Q1→Q2,
  donc l'historique vient de ce passage-là, pas du format source. Un corpus PNG marche tout aussi bien (le CLAUDE.md le dit). Ne te
  restreins pas au JPEG si un bon corpus PNG/TIFF est disponible.

  Attention licences

  Pour de la recherche master, prends des licences CC / recherche / academic use (SROIE, FUNSD, RVL-CDIP le sont). Vérifie avant d'inclure
  dans ton rapport.

  Si tu veux, dis-moi le type de documents que tu vises (factures ? reçus ? formulaires médicaux ? — ta config mentionne bills,
  discharge_summaries) et je te sors les 2-3 corpus les plus adaptés avec leurs liens exacts.

✻ Brewed for 29s

❯ trouvve moi 10 dataset a utiliser pour entrainer mon modele                                                                              

⏺ Je vais chercher les liens actuels pour te donner 10 corpus vérifiés et adaptés à ton pipeline (documents authentiques : factures, reçus,
  formulaires).

⏺ Web Search("SROIE FUNSD CORD RVL-CDIP document image dataset download huggingface")
  ⎿  Did 1 search in 7s
  ⎿  Allowed by auto mode classifier

⏺ Web Search("scanned invoice receipt document dataset public download research 2025")
  ⎿  Did 1 search in 7s
  ⎿  Allowed by auto mode classifier

⏺ Web Search("DocLayNet Tobacco3482 PubLayNet document image dataset huggingface download")
  ⎿  Did 1 search in 9s
  ⎿  Allowed by auto mode classifier

⏺ Voici 10 corpus de vrais documents scannés — donc authentiques par construction (compression unique, aucune forgerie pré-existante),
  exactement ce qu'il te faut comme source pour run.sh.

  Les 10 datasets

  ┌─────┬───────────────────────┬──────────────────────────────────┬──────────┬───────────────┬──────────────────────────────────────┐
  │  #  │        Dataset        │             Contenu              │  Format  │    Taille     │                Accès                 │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 1   │ SROIE 2019            │ reçus scannés                    │ JPEG     │ ~1000         │ HF darentang/sroie / Kaggle (tu l'as │
  │     │                       │                                  │          │               │  déjà)                               │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 2   │ CORD-v2               │ reçus resto/boutiques            │ PNG/JPEG │ ~1000         │ HF naver-clova-ix/cord-v2            │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 3   │ FUNSD                 │ formulaires bruités scannés      │ PNG      │ 199           │ HF nielsr/funsd                      │
  │     │                       │                                  │          │ formulaires   │                                      │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 4   │ RVL-CDIP              │ docs de bureau (16 classes)      │ TIFF     │ 400 000       │ HF aharley/rvl_cdip                  │
  │     │                       │                                  │ gris     │               │                                      │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 5   │ RVL-CDIP (invoice)    │ sous-ensemble factures           │ TIFF     │ ~invoices     │ HF chainyo/rvl-cdip-invoice          │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 6   │ DocLayNet             │ docs                             │ PNG      │ 80 863 pages  │ HF ds4sd/DocLayNet                   │
  │     │                       │ finance/science/légal/brevets    │          │               │                                      │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 7   │ PubLayNet             │ articles scientifiques           │ PNG      │ >300k pages   │ HF shunk031/PubLayNet                │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 8   │ Tobacco3482           │ docs industriels/légaux          │ JPEG     │ 3 482         │ Kaggle « tobacco3482 »               │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 9   │ Denoising Dirty       │ pages de texte scannées          │ PNG      │ ~200          │ Kaggle                               │
  │     │ Documents             │                                  │          │               │                                      │
  ├─────┼───────────────────────┼──────────────────────────────────┼──────────┼───────────────┼──────────────────────────────────────┤
  │ 10  │ BuDDIE                │ documents business multi-tâches  │ image    │ récent (2024) │ arXiv 2404.04003 / repo associé      │
  └─────┴───────────────────────┴──────────────────────────────────┴──────────┴───────────────┴──────────────────────────────────────┘

  Recommandations pour ton cas

  Ta config vise des factures / documents médicaux (bills, discharge_summaries) :

  - Meilleurs choix directs : SROIE (1), Tobacco3482 (8), RVL-CDIP invoice (5) — vrais scans JPEG, exactement le domaine.
  - Gros volume pour le train d'anomalie : RVL-CDIP (4) et DocLayNet (6) te donnent des dizaines de milliers de pages « normales ».
  - Le format n'est pas bloquant : PNG/TIFF marchent aussi bien que JPEG — run.sh recompresse toujours en Q1→Q2, l'historique vient de là
  (pas du format source). Ne te limite pas au JPEG.

  2 règles à respecter

  1. Corpus disjoints entre train et test — n'utilise pas le même dataset pour le train (ratio=1) et le test (ratio~0.5), sinon fuite. Ex.
  : train sur RVL-CDIP + DocLayNet, test sur SROIE + Tobacco3482.
  2. Licences — SROIE, FUNSD, RVL-CDIP, DocLayNet, PubLayNet sont en usage recherche/CC ; vérifie et cite-les dans ton rapport master.

  Sources

  - aharley/rvl_cdip (Hugging Face)
  - chainyo/rvl-cdip-invoice (Hugging Face)
  - ds4sd/DocLayNet (Hugging Face)
  - DS4SD/DocLayNet (GitHub)
  - shunk031/PubLayNet (Hugging Face)
  - BuDDIE: A Business Document Dataset (arXiv 2404.04003)
  - A Survey on Scanned Receipts OCR (SROIE/CORD/FUNSD overview)