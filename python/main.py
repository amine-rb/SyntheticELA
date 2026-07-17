"""main — point d'entrée du pipeline de génération.

Lance la génération : un sous-dossier complet et autonome par type d'édition.
Les paramètres proviennent d'un YAML (--config), lui-même généré à partir de
`config.sh` par les scripts.

Usage recommandé
----------------
    ./scripts/run.sh                    # tout vient de config.sh
    ./scripts/run.sh --n 500 --workers 8

Invocation directe (le YAML doit être fourni ou présent) :
    python python/main.py --config <fichier.yaml> [--src ... --out ... --n ...]
"""

from __future__ import annotations

import orchestrator


def main() -> None:
    orchestrator.main()


if __name__ == "__main__":
    main()
