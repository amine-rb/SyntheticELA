#!/usr/bin/env bash
# preview.sh — planches QA (image | ELA | masque) sur un lot déjà généré.
# Par défaut : le premier type de EDIT_TYPES. Surcharge avec --out, ex. :
#   ./scripts/preview.sh --out "$OUTPUT_DIR/_aggregated" --n 20
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
make_config
DEFAULT_TARGET="$OUTPUT_DIR/${EDIT_TYPES[0]}"
"$PYTHON" "$PY_DIR/ela_preview.py" --config "$CFG" --out "$DEFAULT_TARGET" "$@"
