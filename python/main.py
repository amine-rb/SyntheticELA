"""main — entry point of the generation pipeline.

Runs generation: one complete, self-contained subfolder per edit type.
Parameters come from a YAML (--config), itself generated from `config.sh`
by the scripts.

Recommended usage
-----------------
    ./scripts/run.sh                    # everything comes from config.sh
    ./scripts/run.sh --n 500 --workers 8

Direct invocation (the YAML must be provided or present):
    python python/main.py --config <file.yaml> [--src ... --out ... --n ...]
"""

from __future__ import annotations

import orchestrator


def main() -> None:
    orchestrator.main()


if __name__ == "__main__":
    main()
