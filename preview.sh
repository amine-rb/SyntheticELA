#!/usr/bin/env bash
# preview.sh — planches QA (image | ELA | masque) sur un lot déjà généré.
# Usage : ./preview.sh --out output/substitution   (ou --out output/_aggregated)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python ela_preview.py "$@"
