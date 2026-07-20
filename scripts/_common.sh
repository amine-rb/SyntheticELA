#!/usr/bin/env bash
# _common.sh — sourced by run.sh / aggregate.sh / ela.sh.
# Loads config.sh, locates Python + the code, and generates the internal YAML config
# (implementation detail: the only file the user edits is config.sh).
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$_HERE/.." && pwd)"
PY_DIR="$ROOT/python"
CONFIG_SH="$ROOT/config.sh"

[[ -f "$CONFIG_SH" ]] || { echo "ERROR: config.sh not found ($CONFIG_SH)" >&2; exit 1; }
[[ -d "$PY_DIR"    ]] || { echo "ERROR: python/ folder not found ($PY_DIR)" >&2; exit 1; }

# shellcheck disable=SC1090
source "$CONFIG_SH"

PYTHON="${PYTHON:-python}"
# Make the code importable, including for spawn workers (macOS).
export PYTHONPATH="$PY_DIR${PYTHONPATH:+:$PYTHONPATH}"

# gen_config references SOURCE_DIR/OUTPUT_DIR (the paths of the CURRENT corpus).
# run.sh reassigns them at each iteration of its multi-corpus loop; for ela.sh
# and aggregate.sh (single-corpus), we fall back to the 1st corpus of the RUN lists.
SOURCE_DIR="${SOURCE_DIR:-${RUN_SOURCE_DIRS[0]:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_OUTPUT_DIRS[0]:-}}"

# --- Generate a complete config.yaml from the config.sh variables ------------
gen_config() {
    local out="$1" e
    {
        echo "paths:"
        echo "  source_dir: \"${SOURCE_DIR}\""
        echo "  output_dir: \"${OUTPUT_DIR}\""
        echo "  probe_report: \"${OUTPUT_DIR}/distribution.json\""

        echo "probe:"
        echo "  recursive: ${PROBE_RECURSIVE}"
        echo "  candidate_ext:"
        for e in "${CANDIDATE_EXT[@]}"; do echo "    - \"${e}\""; done
        echo "  allow_lossless: ${ALLOW_LOSSLESS}"
        echo "  nonstandard_absdiff_threshold: ${NONSTANDARD_ABSDIFF_THRESHOLD}"

        echo "compression:"
        echo "  quality_sweep: [$(IFS=', '; echo "${QUALITY_SWEEP[*]}")]"
        # Q1_GAP can be a scalar (Q1 fixed) OR an array (min max) -> Q1 varies
        # per doc (option A). We always emit a [min, max] list on the YAML side.
        if [[ ${#Q1_GAP[@]} -gt 1 ]]; then
            echo "  q1_gap: [${Q1_GAP[0]}, ${Q1_GAP[1]}]"
        else
            echo "  q1_gap: ${Q1_GAP[0]}"
        fi

        echo "forger:"
        echo "  edit_types:"
        for e in "${EDIT_TYPES[@]}"; do echo "    - ${e}"; done
        echo "  aligned_ratio: ${ALIGNED_RATIO}"
        echo "  feather_radius_px: [${FEATHER_RADIUS_PX[0]}, ${FEATHER_RADIUS_PX[1]}]"
        echo "  splice_source: \"${SPLICE_SOURCE}\""
        echo "  min_region_px: [${MIN_REGION_PX[0]}, ${MIN_REGION_PX[1]}]"
        echo "  n_forgeries: [${N_FORGERIES[0]}, ${N_FORGERIES[1]}]"
        echo "  place_on_content: ${PLACE_ON_CONTENT}"
        echo "  min_content_frac: ${MIN_CONTENT_FRAC}"
        echo "  subst_color_prob: ${SUBST_COLOR_PROB:-0.0}"

        echo "size_classes:"
        echo "  small:      [${SIZE_SMALL[0]}, ${SIZE_SMALL[1]}]"
        echo "  medium:     [${SIZE_MEDIUM[0]}, ${SIZE_MEDIUM[1]}]"
        echo "  large:      [${SIZE_LARGE[0]}, ${SIZE_LARGE[1]}]"
        echo "  very_large: [${SIZE_VERY_LARGE[0]}, ${SIZE_VERY_LARGE[1]}]"

        echo "negatives:"
        echo "  ratio: ${NEGATIVES_RATIO}"
        echo "  keep_benign_colored: ${KEEP_BENIGN_COLORED}"

        echo "annotator:"
        echo "  input_res: ${INPUT_RES}"
        echo "  resize_square: ${RESIZE_384}"
        echo "  patch_size: ${PATCH_SIZE}"
        echo "  patch_grid: ${PATCH_GRID}"
        echo "  patch_positive_overlap: ${PATCH_POSITIVE_OVERLAP}"

        echo "ela:"
        echo "  ela_quality: ${ELA_QUALITY}"
        echo "  ela_spread: ${ELA_SPREAD}"
        echo "  n_samples: ${ELA_N_SAMPLES}"
        echo "  ela_scale: ${ELA_SCALE}"
        echo "  chroma_suppress: ${ELA_CHROMA_SUPPRESS}"
        echo "  grayscale_input: ${ELA_GRAYSCALE_INPUT}"
        echo "  join: ${JOIN:-false}"

        echo "orchestrator:"
        echo "  seed: ${SEED}"
        echo "  n_docs: ${N_DOCS}"
        echo "  n_workers: ${N_WORKERS}"
    } > "$out"
}

# Create a self-cleaning temporary YAML (returns its path in $CFG).
make_config() {
    CFG="$(mktemp -t syntheticela.XXXXXX.yaml)"
    trap 'rm -f "$CFG"' EXIT
    gen_config "$CFG"
}
