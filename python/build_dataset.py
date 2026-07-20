#!/usr/bin/env python3
"""build_dataset.py — merge the `dataset.csv` of several corpora into ONE index.

Each corpus (output folder of a `run.sh`) contains a `dataset.csv` per type
subfolder (`<out>/<type>/dataset.csv`), with columns:
    id, type, x_path, negative, mask_path   (x_path/mask_path = ABSOLUTE paths)
`id` there is a sequential integer `0..n-1` LOCAL to that `dataset.csv` (not a file
stem — cf. `orchestrator.write_folder_csvs`).

This script concatenates all those CSVs — across SEVERAL corpora and SEVERAL types —
into a single `<final>/dataset.csv`, WITHOUT a `corpus` column (the final CSV keeps
exactly the source columns and values, id INCLUDED: no renumbering). Each sub-CSV
restarts at `0`, so **`id` is NOT unique in the merged file** (several rows may
share the same integer) — this is accepted; the provenance of each range (which
source `dataset.csv`, which corpus, which original `id`s) is traced separately in
`<final>/sources.json`, not in a column. Since the paths are absolute, the merged
index stays self-contained despite the duplicated `id`.

    python build_dataset.py --out FINAL_DIR CORPUS_OUT_1 CORPUS_OUT_2 ...
    python build_dataset.py --out FINAL_DIR --name staver --name sroie OUT1 OUT2

Called automatically by `run.sh` at the end of a multi-corpus run (see config.sh
`RUN_FINAL_DATASET_DIR`). No dependency on the rest of `python/`.
"""
import argparse
import csv
import glob
import json
import os


def _default_tag(outdir: str) -> str:
    """Readable, reasonably unique corpus tag derived from the path.

    `<parent>_<basename>` (e.g. .../NoisyMed/bills_fraud -> NoisyMed_bills_fraud);
    prefer passing --name for a clean name.
    """
    p = outdir.rstrip(os.sep)
    parent = os.path.basename(os.path.dirname(p))
    base = os.path.basename(p)
    return f"{parent}_{base}" if parent else base


def build(final_dir: str, outputs: list[str], names: list[str] | None = None) -> str:
    if names and len(names) != len(outputs):
        raise ValueError(
            f"--name given {len(names)} times for {len(outputs)} corpora: "
            "there must be as many as output folders (or none).")
    os.makedirs(final_dir, exist_ok=True)
    dst = os.path.join(final_dir, "dataset.csv")
    sources_path = os.path.join(final_dir, "sources.json")
    n_rows = 0
    n_corpus = 0
    header_written: list[str] | None = None
    sources: list[dict] = []
    seen_tags: set[str] = set()
    with open(dst, "w", newline="") as fo:
        writer = None
        for idx, outdir in enumerate(outputs):
            tag = names[idx] if names else _default_tag(outdir)
            # guarantees uniqueness even if two folders give the same tag
            base_tag, k = tag, 1
            while tag in seen_tags:
                k += 1
                tag = f"{base_tag}_{k}"
            seen_tags.add(tag)
            csvs = sorted(glob.glob(os.path.join(outdir, "*", "dataset.csv")))
            if not csvs:
                print(f"  (no dataset.csv under {outdir} — corpus skipped)")
                continue
            n_corpus += 1
            for csvp in csvs:
                with open(csvp, newline="") as fi:
                    reader = csv.reader(fi)
                    header = next(reader, None)
                    if header is None:
                        continue
                    if writer is None:
                        writer = csv.writer(fo)
                        writer.writerow(header)
                        header_written = header
                    id_col = header.index("id") if "id" in header else 0
                    ids: list[str] = []
                    for row in reader:
                        ids.append(row[id_col])
                        writer.writerow(row)
                        n_rows += 1
                    sources.append({
                        "corpus": tag,
                        "source_dir": os.path.abspath(outdir),
                        "csv": os.path.abspath(csvp),
                        "n_rows": len(ids),
                        "ids": ids,
                    })
    with open(sources_path, "w") as f:
        json.dump({
            "final_csv": os.path.abspath(dst),
            "columns": header_written,
            "n_rows": n_rows,
            "n_corpus": n_corpus,
            "sources": sources,
        }, f, indent=2)
    print(f"merged dataset -> {dst}  ({n_rows} rows, {n_corpus} corpora)")
    print(f"sources -> {sources_path}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output folder of the merged dataset.")
    ap.add_argument("--name", action="append", default=None,
                    help="Corpus name (repeat, aligned with the folder order). "
                         "Optional: derived from the path otherwise.")
    ap.add_argument("outputs", nargs="+", help="Corpus output folders (run.sh).")
    args = ap.parse_args()
    build(args.out, args.outputs, args.name)


if __name__ == "__main__":
    main()
