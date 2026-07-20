#!/usr/bin/env bash
# _common.sh — sourcé par run.sh / aggregate.sh / ela.sh.
# Charge config.sh, localise Python + le code, et génère la config YAML interne
# (détail d'implémentation : le seul fichier que l'utilisateur édite est config.sh).
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$_HERE/.." && pwd)"
PY_DIR="$ROOT/python"
CONFIG_SH="$ROOT/config.sh"

[[ -f "$CONFIG_SH" ]] || { echo "ERREUR: config.sh introuvable ($CONFIG_SH)" >&2; exit 1; }
[[ -d "$PY_DIR"    ]] || { echo "ERREUR: dossier python/ introuvable ($PY_DIR)" >&2; exit 1; }

# shellcheck disable=SC1090
source "$CONFIG_SH"

PYTHON="${PYTHON:-python}"
# Rend le code importable, y compris pour les workers spawn (macOS).
export PYTHONPATH="$PY_DIR${PYTHONPATH:+:$PYTHONPATH}"

# --- Génère un config.yaml complet à partir des variables de config.sh -------
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
        # Q1_GAP peut être un scalaire (Q1 fixe) OU un tableau (min max) -> Q1 varie
        # par doc (option A). On rend toujours une liste [min, max] côté YAML.
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

        echo "orchestrator:"
        echo "  seed: ${SEED}"
        echo "  n_docs: ${N_DOCS}"
        echo "  n_workers: ${N_WORKERS}"
    } > "$out"
}

# Crée un YAML temporaire auto-nettoyé (retourne son chemin dans $CFG).
make_config() {
    CFG="$(mktemp -t syntheticela.XXXXXX.yaml)"
    trap 'rm -f "$CFG"' EXIT
    gen_config "$CFG"
}
