#!/usr/bin/env bash
# aggregate.sh — fusionne les sous-dossiers de types en un dataset unique.
# Cible OUTPUT_DIR (config.sh) par défaut ; surcharge possible en CLI, ex. :
#   ./scripts/aggregate.sh --types substitution splice --mode symlink
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
"$PYTHON" "$PY_DIR/aggregate.py" --out "$OUTPUT_DIR" "$@"
