#!/usr/bin/env bash
# aggregate.sh — fusionne les sous-dossiers de types en un seul dataset.
# Usage : ./aggregate.sh --out output [--types substitution splice] [--mode symlink]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python aggregate.py "$@"
