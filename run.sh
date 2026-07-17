#!/usr/bin/env bash
# run.sh — lance la génération (main.py -> orchestrator).
# Tous les paramètres viennent de config.yaml ; les arguments passés ici les
# surchargent ponctuellement (ex. ./run.sh --n 500 --workers 8).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python main.py "$@"
