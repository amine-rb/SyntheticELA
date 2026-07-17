#!/usr/bin/env bash
# run.sh — génère le dataset (un sous-dossier autonome par type d'édition).
# Tous les paramètres viennent de config.sh ; les arguments passés ici les
# surchargent ponctuellement (ex. ./scripts/run.sh --n 500 --workers 8).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
make_config
"$PYTHON" "$PY_DIR/main.py" --config "$CFG" "$@"
