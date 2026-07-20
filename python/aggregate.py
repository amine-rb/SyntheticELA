"""aggregate — Merge type subfolders into ONE single subfolder.

Role
----
The orchestrator writes a self-contained subfolder per edit type
(`<out>/substitution`, `<out>/copy_move`, `<out>/splice`, ...). This module
merges them into a single dataset (`<out>/_aggregated` by default) losing nothing:

    - copies (or links) the `images/<id>.jpg`, `masks/<id>_mask.png`,
      `masks/<id>.json`, `ela/<id>_ela.png` files of each type,
    - concatenates the manifests into a single `manifest.parquet` (a `type`
      column -> re-filterable by type at will) + one CSV per folder (images/masks/ela),
    - reuses `distribution.json` (shared corpus) and writes an aggregated
      `run_config.yaml` + a regenerated `REPORT.md`.

Since the `doc_id`s are prefixed by the type (`substitution_000000`, ...), file
names never collide: aggregation is a simple `union`. The manifest paths
(`images/<id>.jpg`, ...) stay valid as-is in the aggregated folder -> no rewrite.

Usage
-----
    # all present subfolders:
    python aggregate.py --out output
    # explicit selection + symlink (no disk copy):
    python aggregate.py --out output --types substitution splice --mode symlink

Dependencies: PyArrow, PyYAML.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


_PATH_KEYS = ("path_img", "path_mask", "path_json", "path_ela")


def _place(src: str, dst: str, mode: str) -> None:
    """Place `src` at `dst` according to `mode` (copy / symlink / hardlink), idempotent."""
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
    """Subfolders that contain a manifest (excluding aggregated `_*` folders)."""
    out = []
    for d in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, d)
        if (os.path.isdir(p) and not d.startswith("_")
                and os.path.exists(os.path.join(p, "manifest.parquet"))):
            out.append(d)
    return out


def aggregate(out_root: str, types: list[str] | None = None,
              dest: str = "_aggregated", mode: str = "copy") -> str:
    """Merge the `types` subfolders of `out_root` into `<out_root>/<dest>`."""
    if types is None:
        types = _discover_types(out_root)
    if not types:
        raise RuntimeError(
            f"No type subfolder with a manifest under {out_root} "
            "(run `./scripts/run.sh` first).")

    dest_root = os.path.join(out_root, dest)
    # The images/ masks/ ela/ folders are created on the fly by `_place`
    # (each relative manifest path already points into them).
    os.makedirs(dest_root, exist_ok=True)

    all_rows: list[dict] = []
    used_types, dist_src = [], None
    for t in types:
        sub = os.path.join(out_root, t)
        mpath = os.path.join(sub, "manifest.parquet")
        if not os.path.exists(mpath):
            print(f"  (skipped: '{t}' without manifest.parquet)")
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
        raise RuntimeError("Nothing to aggregate (empty manifests).")

    # Concatenated manifest + one CSV per folder (images/masks/ela), like a native batch.
    pq.write_table(pa.Table.from_pylist(all_rows),
                   os.path.join(dest_root, "manifest.parquet"))
    try:
        from orchestrator import write_folder_csvs
        write_folder_csvs(dest_root, all_rows)
    except Exception as exc:
        print(f"  (per-folder CSVs not generated: {type(exc).__name__}: {exc})")

    # distribution.json (shared corpus) + aggregated run_config.
    if dist_src:
        shutil.copy2(dist_src, os.path.join(dest_root, "distribution.json"))
    base_cfg = {}
    rc = os.path.join(out_root, used_types[0], "run_config.yaml")
    if os.path.exists(rc):
        base_cfg = yaml.safe_load(open(rc)) or {}
    base_cfg.setdefault("forger", {})
    base_cfg["forger"]["edit_types"] = used_types
    base_cfg["forger"].pop("edit_type", None)      # more than one type here
    base_cfg.setdefault("paths", {})["output_dir"] = dest_root
    base_cfg["_aggregated_from"] = used_types
    with open(os.path.join(dest_root, "run_config.yaml"), "w") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False, allow_unicode=True)

    # Report regenerated on the merged batch.
    try:
        import reporter
        reporter.write_report(dest_root)
    except Exception as exc:
        print(f"  (report not generated: {type(exc).__name__}: {exc})")

    n_pos = sum(not r["is_negative"] for r in all_rows)
    print(f"= {dest_root}: {len(all_rows)} docs "
          f"({n_pos} positives, {len(all_rows) - n_pos} negatives), mode={mode}")
    return dest_root


def main() -> None:
    ap = argparse.ArgumentParser(
        description="aggregate — merge the type subfolders into a single dataset.")
    ap.add_argument("--out", required=True,
                    help="Root containing the type subfolders (e.g. output).")
    ap.add_argument("--types", nargs="*", default=None,
                    help="Types to merge (default: all those found).")
    ap.add_argument("--dest", default="_aggregated",
                    help="Name of the merged subfolder (default: _aggregated).")
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy",
                    help="copy (portable, default) | symlink/hardlink (without doubling disk).")
    args = ap.parse_args()
    aggregate(args.out, args.types, dest=args.dest, mode=args.mode)


if __name__ == "__main__":
    main()
