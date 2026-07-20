"""ela_scan — generate ELA (RGB, 3 qualities ≈ Q1) on a raw image FOLDER.

Role
----
Applies EXACTLY the same ELA as the forgery pipeline
(`orchestrator.compute_ela_stack`, same qualities/scale as `config.sh`) to any
image folder — forged or not — and writes one RGB ELA image per document into an
output folder, plus an `ela.csv`.

Serves two purposes:
  1. produce a model's ELA input on images that do not come from the generator
     (no manifest, no mask required);
  2. visually inspect what the model "sees" on real documents.

WARNING (Q1 unknown): the probe quality (`ELA_QUALITY` ≈ Q1) is the one chosen
for the SYNTHETIC corpus. On real images whose compression history is unknown,
this probe is not guaranteed to sit at the background's fixed point — the
contrast may be lower. Use `--ela-quality` to sweep several probes if needed.

ELA = |image - recompressed_at_q|, stacked over 3 qualities -> R/G/B channels,
fixed global scale (readability + inter-image comparability).

Dependencies: Pillow, NumPy, PyYAML (via orchestrator.load_config).
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from PIL import Image

from orchestrator import load_config, compute_ela_stack, ela_qualities

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def _iter_images(src: str, recursive: bool):
    """Image paths under `src` (recursive or not), sorted, known extensions."""
    if recursive:
        for root, _, files in os.walk(src):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXT):
                    yield os.path.join(root, f)
    else:
        for f in sorted(os.listdir(src)):
            p = os.path.join(src, f)
            if os.path.isfile(p) and f.lower().endswith(IMG_EXT):
                yield p


def run(src_dir: str, out_dir: str, ela_quality: int, ela_spread: int,
        scale: float, recursive: bool = False, chroma_suppress: float = 0.0,
        grayscale_input: bool = False) -> str:
    if not os.path.isdir(src_dir):
        raise NotADirectoryError(f"Source folder not found: {src_dir}")
    qs = ela_qualities(int(ela_quality), int(ela_spread))   # SAME stack as the pipeline output
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    n = 0
    for img_path in _iter_images(src_dir, recursive):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ela_name = f"{stem}_ela.png"
        try:
            ela_rgb = compute_ela_stack(img_path, qs, scale,
                                        chroma_suppress=chroma_suppress,
                                        grayscale_input=grayscale_input)   # (H,W,3) RGB
        except Exception as e:                                  # unreadable image -> skip, report
            print(f"[ela_scan] ⚠️  skipped ({e}): {img_path}")
            continue
        Image.fromarray(ela_rgb, "RGB").save(os.path.join(out_dir, ela_name))
        rows.append({
            "id": stem,
            "ela": ela_name,
            "source_image": os.path.relpath(img_path, src_dir),
            "ela_qualities": ",".join(str(q) for q in qs),
            "ela_scale": scale,
        })
        n += 1

    with open(os.path.join(out_dir, "ela.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "ela", "source_image",
                                          "ela_qualities", "ela_scale"])
        w.writeheader()
        w.writerows(rows)

    cs = f", chroma_suppress={chroma_suppress:g}" if chroma_suppress else ""
    gs = ", grayscale_input" if grayscale_input else ""
    print(f"[ela_scan] {n} RGB ELA written to {out_dir} "
          f"(qualities {qs}, global scale ×{scale:g}{gs}{cs}).")
    if n == 0:
        print(f"[ela_scan] ⚠️  no {IMG_EXT} image found under {src_dir}"
              f"{' (try --recursive)' if not recursive else ''}.")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ela_scan — ELA RGB (3 qualities ≈ Q1) on an image folder, "
                    "with the same parameters as generation (config.sh).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--in", dest="src", required=True, help="Source image folder.")
    ap.add_argument("--out", dest="out", required=True, help="ELA output folder.")
    ap.add_argument("--recursive", action="store_true", help="Walk the subfolders.")
    ap.add_argument("--ela-quality", type=int, default=None,
                    help="Probe center (default = config ela.ela_quality ≈ Q1).")
    ap.add_argument("--ela-spread", type=int, default=None,
                    help="Spread of the 3 channels (default = config ela.ela_spread).")
    ap.add_argument("--ela-scale", type=float, default=None,
                    help="Fixed global scale (default = config ela.ela_scale=15).")
    ap.add_argument("--chroma-suppress", type=float, default=None,
                    help="Attenuate the ELA of colored pixels (logos/stamps); 0 = off "
                         "(default = config ela.chroma_suppress).")
    ap.add_argument("--grayscale-input", dest="grayscale_input", action="store_true", default=None,
                    help="Grayscale the image BEFORE ELA (default = config ela.grayscale_input).")
    ap.add_argument("--no-grayscale-input", dest="grayscale_input", action="store_false",
                    help="Force ELA on the color image (disables ela.grayscale_input).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ela_cfg = cfg.get("ela", cfg.get("ela_preview", {}))
    q = args.ela_quality if args.ela_quality is not None else int(ela_cfg["ela_quality"])
    spread = args.ela_spread if args.ela_spread is not None else int(ela_cfg.get("ela_spread", 8))
    scale = args.ela_scale if args.ela_scale is not None else float(ela_cfg.get("ela_scale", 15.0))
    chroma = args.chroma_suppress if args.chroma_suppress is not None else float(ela_cfg.get("chroma_suppress", 0.0))
    gray = args.grayscale_input if args.grayscale_input is not None else bool(ela_cfg.get("grayscale_input", False))
    run(args.src, args.out, ela_quality=q, ela_spread=spread,
        scale=scale, recursive=args.recursive, chroma_suppress=chroma, grayscale_input=gray)


if __name__ == "__main__":
    main()
