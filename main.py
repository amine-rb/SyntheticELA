"""main — point d'entrée UNIQUE du pipeline de génération.

Lit `config.yaml` (paths.source_dir, paths.output_dir, forger.edit_types,
forger.min_region_px, orchestrator.n_docs, ...) et lance la génération : un
sous-dossier complet et autonome par type d'édition.

Usage
-----
    python main.py                              # tout vient de config.yaml
    python main.py --src DOSSIER --out DOSSIER   # surcharge ponctuelle
    python main.py --n 500 --workers 8

Ou, plus simple : `./run.sh` (mêmes options, cf. README).
"""

from __future__ import annotations

import orchestrator


def main() -> None:
    orchestrator.main()


if __name__ == "__main__":
    main()
