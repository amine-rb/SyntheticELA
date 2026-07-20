#!/usr/bin/env bash
# run.sh — generate the dataset for ONE OR MORE corpora, then merge.
#
# Multi-corpus: run.sh loops over the (RUN_SOURCE_DIRS[i] -> RUN_OUTPUT_DIRS[i])
# pairs from config.sh, generates a full dataset per corpus, THEN automatically
# merges all dataset.csv (all corpora + all types) into RUN_FINAL_DATASET_DIR.
# A single corpus? Put a single entry in each list.
#
# Arguments passed here override the config for ALL corpora
# (e.g. ./scripts/run.sh --n 500 --workers 8).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Corpus lists (fall back to SOURCE_DIR/OUTPUT_DIR if the lists are empty).
if [[ ${#RUN_SOURCE_DIRS[@]} -gt 0 ]]; then
    srcs=("${RUN_SOURCE_DIRS[@]}")
    outs=("${RUN_OUTPUT_DIRS[@]}")
else
    srcs=("$SOURCE_DIR")
    outs=("$OUTPUT_DIR")
fi
if [[ ${#srcs[@]} -ne ${#outs[@]} ]]; then
    echo "ERROR: RUN_SOURCE_DIRS (${#srcs[@]}) and RUN_OUTPUT_DIRS (${#outs[@]}) must have the same length." >&2
    exit 1
fi

make_config                                   # create the temporary YAML (CFG) + cleanup trap

n=${#srcs[@]}
for i in "${!srcs[@]}"; do
    SOURCE_DIR="${srcs[$i]}"
    OUTPUT_DIR="${outs[$i]}"
    echo ">>> [$((i + 1))/$n] generating: $SOURCE_DIR -> $OUTPUT_DIR"
    # Each run starts from scratch: wipe the corpus output folder before
    # regenerating (no accumulation/mixing with a previous run).
    [[ -n "$OUTPUT_DIR" ]] && rm -rf "$OUTPUT_DIR"
    gen_config "$CFG"                         # regenerate the config for THIS corpus
    "$PYTHON" "$PY_DIR/main.py" --config "$CFG" "$@"
done

# --- Final dataset merge (all corpora + all types) ---------------------------
if [[ -n "${RUN_FINAL_DATASET_DIR:-}" ]]; then
    names=()
    if [[ ${#RUN_CORPUS_NAMES[@]} -eq $n ]]; then
        for nm in "${RUN_CORPUS_NAMES[@]}"; do names+=(--name "$nm"); done
    elif [[ ${#RUN_CORPUS_NAMES[@]} -gt 0 ]]; then
        echo "WARNING: RUN_CORPUS_NAMES (${#RUN_CORPUS_NAMES[@]}) != number of corpora ($n) -> names derived from the path." >&2
    fi
    echo ">>> merging the final dataset -> $RUN_FINAL_DATASET_DIR"
    "$PYTHON" "$PY_DIR/build_dataset.py" --out "$RUN_FINAL_DATASET_DIR" \
        "${names[@]}" "${outs[@]}"
fi
