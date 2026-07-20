#!/usr/bin/env bash
# ela.sh — ELA RGB (the [COMMON] recipe from config.sh, IDENTICAL to generation) on
# any image folder (forged or not), WITHOUT recompression: a real forgery already
# carries its Qa->Qb history; recompressing would mask the signal.
#
#   ./scripts/ela.sh --in FOLDER/IMAGES --out FOLDER/ELA_OUTPUT [--recursive]
#   ./scripts/ela.sh --in real_docs/ --out real_docs_ela/ --ela-quality 72
#
# Without --in/--out: use the [ELA] defaults from config.sh (ELA_INPUT_DIR /
# ELA_OUTPUT_DIR / ELA_RECURSIVE), otherwise fall back to the 1st RUN corpus and _ela_scan.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
make_config

# Practical defaults if the user does not pass --in/--out/--recursive.
if [[ "$*" != *"--in"* ]]; then
    set -- --in "${ELA_INPUT_DIR:-$SOURCE_DIR}" "$@"
fi
if [[ "$*" != *"--out"* ]]; then
    set -- "$@" --out "${ELA_OUTPUT_DIR:-$OUTPUT_DIR/_ela_scan}"
fi
if [[ "${ELA_RECURSIVE:-false}" == "true" && "$*" != *"--recursive"* ]]; then
    set -- "$@" --recursive
fi

"$PYTHON" "$PY_DIR/ela_scan.py" --config "$CFG" "$@"
