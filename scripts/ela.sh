#!/usr/bin/env bash
# ela.sh — génère l'ELA RGB (3 qualités ≈ Q1, mêmes paramètres que la génération)
# sur un dossier d'images quelconque (falsifié ou non), et l'écrit dans un dossier
# de sortie. Aucun manifeste ni masque requis.
#
#   ./scripts/ela.sh --in DOSSIER/IMAGES --out DOSSIER/SORTIE_ELA [--recursive]
#   ./scripts/ela.sh --in real_docs/ --out real_docs_ela/ --ela-quality 72
#
# Par défaut (sans --in/--out) : ELA du dossier source du config vers OUTPUT_DIR/_ela_scan.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
make_config

# Valeurs par défaut pratiques si l'utilisateur ne passe pas --in/--out.
if [[ "$*" != *"--in"* ]]; then set -- --in "$SOURCE_DIR" "$@"; fi
if [[ "$*" != *"--out"* ]]; then set -- "$@" --out "$OUTPUT_DIR/_ela_scan"; fi

"$PYTHON" "$PY_DIR/ela_scan.py" --config "$CFG" "$@"
