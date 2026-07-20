#!/usr/bin/env bash
# aggregate.sh — merge the TYPE SUBFOLDERS of ONE corpus into a single dataset
# (<corpus>/_aggregated/). Defaults taken from the [AGGREGATE] section of config.sh;
# everything is overridable on the CLI, e.g.:
#   ./scripts/aggregate.sh --types substitution splice --mode symlink
# (To merge several CORPORA together: that is run.sh -> RUN_FINAL_DATASET_DIR.)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

out="${AGG_OUTPUT_DIR:-$OUTPUT_DIR}"
extra=()
[[ ${#AGG_TYPES[@]} -gt 0 ]] && extra+=(--types "${AGG_TYPES[@]}")
[[ -n "${AGG_MODE:-}" ]]     && extra+=(--mode "$AGG_MODE")
[[ -n "${AGG_DEST:-}" ]]     && extra+=(--dest "$AGG_DEST")

"$PYTHON" "$PY_DIR/aggregate.py" --out "$out" "${extra[@]}" "$@"
