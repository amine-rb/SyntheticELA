# =============================================================================
# config.sh — THE single parameter file. Details & rationale: README §7-§8.
#   ./scripts/run.sh         (multi-corpus generation + merged dataset)
#   ./scripts/ela.sh         (inference ELA on any image folder)
#   ./scripts/aggregate.sh   (merge the types of one corpus)
# Sections: [COMMON] all scripts · [RUN] run.sh · [ELA] ela.sh · [AGGREGATE] aggregate.sh
# =============================================================================


# ##################### [COMMON] — run.sh, ela.sh, aggregate.sh ###############

PYTHON="${PYTHON:-python}"                    # interpreter (the one that has the deps)

# ELA recipe — MANDATORY: IDENTICAL between generation and inference (README §7).
ELA_QUALITY=60                                # probe center (channels 52/60/68)
ELA_SPREAD=8                                  # spread of the 3 RGB channels around the center
ELA_SCALE=15                                  # fixed global scale (= detection_eval.ELA_SCALE)
ELA_GRAYSCALE_INPUT=false                     # grayscale before ELA (anti colored-FP; OFF if forging color)
ELA_CHROMA_SUPPRESS=0                         # erase colored pixels after ELA; 0=off

PROBE_RECURSIVE=true                          # probe the corpus subfolders
CANDIDATE_EXT=(.jpg .jpeg .jpe .jfif .png .tif .tiff .bmp)   # accepted extensions
ALLOW_LOSSLESS=true                           # keep PNG/TIFF/BMP (no Q0)
NONSTANDARD_ABSDIFF_THRESHOLD=40              # "non-standard quantization table" threshold


# ##################### [RUN] — run.sh (generation) ##########################

# ALIGNED (input, output) lists: one corpus per entry, same length (README §2, §8).
RUN_SOURCE_DIRS=(
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/StaVer/scans/scans"
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/SROIE2019/train/img"
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/NoisyMed/bills"
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/NoisyMed/discharge_summaries"
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/Signatures"
)
RUN_OUTPUT_DIRS=(
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/StaVer/scans/fraud"
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/SROIE2019/train/fraud"
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/NoisyMed/bills_fraud"
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/NoisyMed/discharge_summaries_fraud"
  "/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/Signatures/fraud"
)
RUN_CORPUS_NAMES=(staver sroie bills discharge)   # `corpus` column names (aligned); () = derived from the path
RUN_FINAL_DATASET_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/_dataset_final"  # "" = no merge

QUALITY_SWEEP=(72 78 84 90 96)                # Q2 values (final save, high), drawn per doc
Q1_GAP=(15 50)                                # Q1 = Q2 - gap; range (min max) => Q1 varies per doc (README §7)

EDIT_TYPES=(substitution)                     # one subfolder per type; e.g. (substitution copy_move splice)
ALIGNED_RATIO=0.5                             # fraction aligned to the 8x8 grid (copy_move + splice)
FEATHER_RADIUS_PX=(0.5 2.0)                   # edge feathering (anti-tell)
SPLICE_SOURCE=intra_corpus                    # splice donor = another doc from the same corpus
MIN_REGION_PX=(10 10)                         # guaranteed min region size (width height), rounded up to x8
N_FORGERIES=(1 5)                             # number of forgeries per positive doc (min max)
PLACE_ON_CONTENT=true                         # place forgeries on real content, not blank space
MIN_CONTENT_FRAC=0.02                         # min ink fraction under a candidate region
SUBST_COLOR_PROB=0.5                          # fraction of colored substitutions; 0=all black (ELA color filter OFF)

SIZE_SMALL=(0.001 0.005)                      # region: 0.1% – 0.5% of the page
SIZE_MEDIUM=(0.005 0.02)                      # 0.5% – 2%
SIZE_LARGE=(0.02 0.06)                        # 2% – 6%
SIZE_VERY_LARGE=(0.06 0.15)                   # 6% – 15%

NEGATIVES_RATIO=1                            # fraction of authentics/subfolder (high=anomaly train, ~0.5=test)
KEEP_BENIGN_COLORED=true                      # preserve logos/stamps/headers
NEGATIVE_ROTATIONS=(0 90 180 270)               # augment: each NEGATIVE emitted once PER angle => #neg = selected * len(pool) (e.g. n_docs=5 => 15). Add 0 to also keep upright; ()=upright only. Positives never rotated

INPUT_RES=384                                 # input resolution of the downstream model
RESIZE_384=true                             # true: deliver every doc (forged or not) as INPUT_RESxINPUT_RES (square) -> images/masks/ela model-ready, CSV points to 384x384 files. Resize is applied AFTER the native Q1->Q2 + ELA (ELA computed natively then downscaled, else text saturates). false: keep native size
PATCH_SIZE=16                                 # patch size
PATCH_GRID=24                                 # patch grid (24x24)
PATCH_POSITIVE_OVERLAP=0.5                    # patch positive if overlap > threshold

ELA_N_SAMPLES=50                              # number of QA boards image | ELA | mask
JOIN=true                                     # extra join/ subfolder: image | ELA | mask stitched side-by-side per doc (visual check)

SEED=42                                       # global seed (reproducibility)
N_DOCS=2                                   # docs PER TYPE and PER corpus; "" = as many as source images (y=x, one forgery/image); else an integer y (y<x = subsample, y>x = sources reused with a different forgery)
N_WORKERS=4                                   # parallelism


# ##################### [ELA] — ela.sh (inference) ###########################
# Defaults when --in/--out are absent; "" => 1st RUN corpus and <output>/_ela_scan.
ELA_INPUT_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/Signatures"                              # folder of docs to analyze
ELA_OUTPUT_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/data/Signatures/ela"                             # output folder for the *_ela.png
ELA_RECURSIVE=true                            # walk the subfolders of --in
ELA_ROTATIONS=(90 180 270)                    # data augmentation: extra ELA per rotation (deg); 0° always emitted; ()=none


# ##################### [AGGREGATE] — aggregate.sh (merge the types) ##########
AGG_OUTPUT_DIR=""                             # corpus to aggregate; "" => RUN_OUTPUT_DIRS[0]
AGG_TYPES=()                                  # () = all types; e.g. (substitution splice)
AGG_MODE=copy                                 # copy | symlink | hardlink
AGG_DEST=_aggregated                          # name of the merged subfolder
