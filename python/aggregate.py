"""aggregate — Fusionne des sous-dossiers de types en UN sous-dossier unique.

Rôle
----
L'orchestrator écrit un sous-dossier autonome par type d'édition
(`<out>/substitution`, `<out>/copy_move`, `<out>/splice`, ...). Ce module les
réunit en un seul dataset (`<out>/_aggregated` par défaut) sans rien perdre :

    - copie (ou lie) les fichiers `data/<id>.{jpg,png,json}` de chaque type,
    - concatène les manifestes en un `manifest.parquet` unique (colonne `type`
      -> re-filtrable par type à volonté),
    - reprend `distribution.json` (corpus commun) et écrit un `run_config.yaml`
      agrégé + un `REPORT.md` régénéré.

Comme les `doc_id` sont préfixés par le type (`substitution_000000`, ...), les
noms de fichiers ne collisionnent jamais : l'agrégation est un simple `union`.
Les chemins du manifeste (`data/<id>.ext`) restent valides tels quels dans le
dossier agrégé -> aucune réécriture.

Usage
-----
    # tous les sous-dossiers présents :
    python aggregate.py --out output
    # sélection explicite + lien symbolique (pas de copie disque) :
    python aggregate.py --out output --types substitution splice --mode symlink

Dépendances : PyArrow, PyYAML.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


_PATH_KEYS = ("path_img", "path_mask", "path_json")


def _place(src: str, dst: str, mode: str) -> None:
    """Dépose `src` en `dst` selon `mode` (copy / symlink / hardlink), idempotent."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:  # copy
        shutil.copy2(src, dst)


def _discover_types(out_root: str) -> list[str]:
    """Sous-dossiers contenant un manifeste (hors dossiers agrégés `_*`)."""
    out = []
    for d in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, d)
        if (os.path.isdir(p) and not d.startswith("_")
                and os.path.exists(os.path.join(p, "manifest.parquet"))):
            out.append(d)
    return out


def aggregate(out_root: str, types: list[str] | None = None,
              dest: str = "_aggregated", mode: str = "copy") -> str:
    """Fusionne les sous-dossiers `types` de `out_root` dans `<out_root>/<dest>`."""
    if types is None:
        types = _discover_types(out_root)
    if not types:
        raise RuntimeError(
            f"Aucun sous-dossier de type avec manifeste sous {out_root} "
            "(lance d'abord `./scripts/run.sh`).")

    dest_root = os.path.join(out_root, dest)
    os.makedirs(os.path.join(dest_root, "data"), exist_ok=True)

    all_rows: list[dict] = []
    used_types, dist_src = [], None
    for t in types:
        sub = os.path.join(out_root, t)
        mpath = os.path.join(sub, "manifest.parquet")
        if not os.path.exists(mpath):
            print(f"  (ignoré : '{t}' sans manifest.parquet)")
            continue
        rows = pq.read_table(mpath).to_pylist()
        for r in rows:
            for key in _PATH_KEYS:
                rel = r.get(key)
                if not rel:
                    continue
                _place(os.path.join(sub, rel), os.path.join(dest_root, rel), mode)
        all_rows.extend(rows)
        used_types.append(t)
        if dist_src is None and os.path.exists(os.path.join(sub, "distribution.json")):
            dist_src = os.path.join(sub, "distribution.json")
        print(f"  + {t}: {len(rows)} docs")

    if not all_rows:
        raise RuntimeError("Rien à agréger (manifestes vides).")

    # Manifeste concaténé.
    pq.write_table(pa.Table.from_pylist(all_rows),
                   os.path.join(dest_root, "manifest.parquet"))

    # distribution.json (corpus commun) + run_config agrégé.
    if dist_src:
        shutil.copy2(dist_src, os.path.join(dest_root, "distribution.json"))
    base_cfg = {}
    rc = os.path.join(out_root, used_types[0], "run_config.yaml")
    if os.path.exists(rc):
        base_cfg = yaml.safe_load(open(rc)) or {}
    base_cfg.setdefault("forger", {})
    base_cfg["forger"]["edit_types"] = used_types
    base_cfg["forger"].pop("edit_type", None)      # plus d'un seul type ici
    base_cfg.setdefault("paths", {})["output_dir"] = dest_root
    base_cfg["_aggregated_from"] = used_types
    with open(os.path.join(dest_root, "run_config.yaml"), "w") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False, allow_unicode=True)

    # Rapport régénéré sur le lot fusionné.
    try:
        import reporter
        reporter.write_report(dest_root)
    except Exception as exc:
        print(f"  (rapport non généré : {type(exc).__name__}: {exc})")

    n_pos = sum(not r["is_negative"] for r in all_rows)
    print(f"= {dest_root} : {len(all_rows)} docs "
          f"({n_pos} positifs, {len(all_rows) - n_pos} négatifs), mode={mode}")
    return dest_root


def main() -> None:
    ap = argparse.ArgumentParser(
        description="aggregate — fusionne les sous-dossiers de types en un seul dataset.")
    ap.add_argument("--out", required=True,
                    help="Racine contenant les sous-dossiers de types (ex. output).")
    ap.add_argument("--types", nargs="*", default=None,
                    help="Types à fusionner (défaut : tous ceux trouvés).")
    ap.add_argument("--dest", default="_aggregated",
                    help="Nom du sous-dossier fusionné (défaut : _aggregated).")
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy",
                    help="copy (portable, défaut) | symlink/hardlink (sans doubler le disque).")
    args = ap.parse_args()
    aggregate(args.out, args.types, dest=args.dest, mode=args.mode)


if __name__ == "__main__":
    main()
