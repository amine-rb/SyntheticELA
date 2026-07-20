"""recompress — Pipeline module 3.

Role
----
1. Decode a source JPEG into a uint8 RGB array (the Q0 blocking artifacts are
   then *baked into the pixels*: this is the fingerprint the forger will
   displace to create the inconsistency).
2. Re-save the final image (background Q0->Q2 + region with inconsistent
   history) as JPEG at quality Q2, in a **single pass**, for both forgeries AND
   authentic negatives.

MANDATORY RULE (instruction.md)
-------------------------------
The edited region undergoes the SAME Q2 pass as the background. We NEVER paste a
region after this save. Concretely: the forger works in pixel space on the
DECODED image, then `save_q2` compresses everything in a single call. The whole
forensic consistency of the dataset rests on this.

Q0 is never chosen (read by jpeg_probe). Q2 is the only swept parameter.

Dependencies: Pillow + NumPy.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from jpeg_probe import estimate_quality, _read_subsampling


# Chroma subsampling used for the final Q2 pass.
# DEFAULT = 2 (4:2:0): this is the dominant subsampling of the SROIE corpus (623/626).
# A realistic recompression (re-export) uses the encoder default.
DEFAULT_Q2_SUBSAMPLING = 2  # 0=4:4:4, 1=4:2:2, 2=4:2:0


@dataclass
class SourceImage:
    """Decoded source image + compression metadata read from the file."""
    rgb: np.ndarray          # (H, W, 3) uint8, decoded image
    q0: int                  # estimated quality (luma) — READ, never chosen
    absdiff: float
    nonstandard: bool
    subsampling: str         # subsampling of the source file
    qtable_luma: list        # luma quantization table (for the JSON)
    width: int
    height: int
    source_id: str           # file name without extension
    path: str


def decode_source(
    path: str,
    nonstandard_threshold: float = 40.0,
    allow_lossless: bool = False,
) -> SourceImage:
    """Decode a source and read its Q0 compression parameters.

    JPEG     -> Q0 READ from the quantization table (never chosen).
    lossless (PNG, if `allow_lossless=True`) -> no JPEG history: q0=-1,
              empty table, subsampling "none". Usable ONLY with a controlled Q1
              (the imposed Q1 becomes the background's sole history).

    Forces RGB mode to homogenize the downstream pipeline.
    """
    img = Image.open(path)
    quant = getattr(img, "quantization", None)

    if img.format == "JPEG" and quant:
        luma = np.array(quant[0], dtype=np.float64)
        q0, absdiff = estimate_quality(luma)
        subsampling = _read_subsampling(img)
        qtable_luma = [int(x) for x in luma.reshape(-1)]
        nonstandard = bool(absdiff > nonstandard_threshold)
    elif allow_lossless:
        q0, absdiff, subsampling, qtable_luma, nonstandard = -1, 0.0, "none", [], False
    elif img.format != "JPEG":
        raise ValueError(f"{path} is not a JPEG (format={img.format}); "
                         f"enable allow_lossless for a PNG source.")
    else:
        raise ValueError(f"{path} has no quantization table (Q0 unreadable)")

    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    return SourceImage(
        rgb=rgb,
        q0=int(q0),
        absdiff=round(float(absdiff), 2),
        nonstandard=nonstandard,
        subsampling=subsampling,
        qtable_luma=qtable_luma,
        width=int(w),
        height=int(h),
        source_id=os.path.splitext(os.path.basename(path))[0],
        path=os.path.abspath(path),
    )


def recompress_to_q1(
    rgb: np.ndarray,
    q1: int,
    subsampling: int = DEFAULT_Q2_SUBSAMPLING,
) -> np.ndarray:
    """Re-encode an image to JPEG Q1 then re-decode it, in memory.

    Role (controlled Q1 — cf. plan.md §Step 0: "Q1 and Q2 independently
    tunable, central parameter for E5"). Since the SROIE source corpus is
    near-lossless (Q0≈100), a single Q1 pass imposes an EFFECTIVE compression
    history specific to the Q1 level: this pass becomes the reference
    grid/quantization of the "original document" before forgery.

    The forged document's background then carries the Q1->Q2 history (double
    compression that is actually detectable), instead of Q0(≈100)->Q2 (weak
    fingerprint). Q0 stays READ and logged; Q1 is an explicit experimental parameter.
    """
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buf, format="JPEG", quality=int(q1), subsampling=subsampling, optimize=False)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def save_q2(
    rgb: np.ndarray,
    out_path: str,
    q2: int,
    subsampling: int = DEFAULT_Q2_SUBSAMPLING,
) -> None:
    """Save the image (uint8 RGB) as JPEG at quality Q2, in a single pass.

    This is the only point where a compression is written. Used both for
    forgeries (background + inconsistent region already composited) and for
    authentic negatives (decoded image as-is).
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(
        out_path,
        format="JPEG",
        quality=int(q2),
        subsampling=subsampling,
        optimize=False,
    )
