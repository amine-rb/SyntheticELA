"""annotator — Pipeline module 4.

Role
----
Produce the ground truth of a forged (or negative) document:
    - EXACT pixel-level binary mask (geometric footprint, no dilation),
    - bbox of the edited region(s),
    - 24x24 patch-level label grid (16 px patch on the model's 384 input),
      positive label if the mask overlap exceeds a threshold (default 0.5),
    - metadata dictionary ready to serialize to JSON.

Note on the patch grid
----------------------
The downstream model resizes the input to 384x384 then splits it into 16 px
patches -> 24x24 grid. So we compute the patch labels by resizing the mask to
384x384 (nearest, to stay binary) then measuring the per-patch overlap. This
reflects exactly what the model "sees".

Dependencies: NumPy + OpenCV (resize).
"""

from __future__ import annotations

import numpy as np
import cv2


def bbox_from_mask(mask: np.ndarray):
    """Bounding box [x, y, w, h] of the non-zero pixels, or None if the mask is empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def patch_grid_labels(
    mask: np.ndarray,
    input_res: int = 384,
    patch_size: int = 16,
    grid: int = 24,
    overlap_thr: float = 0.5,
):
    """Patch-level label grid (grid x grid) + overlap fractions.

    Returns (labels[int8], overlap_fracs[float32]), both of shape (grid, grid).
    """
    # Mask -> 384x384 binary (nearest preserves the binary character).
    m = (mask > 0).astype(np.uint8) * 255
    m384 = cv2.resize(m, (input_res, input_res), interpolation=cv2.INTER_NEAREST)
    m384 = (m384 > 127).astype(np.float32)

    # Mean per-patch overlap = fraction of forged pixels in the patch.
    fracs = m384.reshape(grid, patch_size, grid, patch_size).mean(axis=(1, 3))
    labels = (fracs >= overlap_thr).astype(np.int8)
    return labels, fracs.astype(np.float32)


def build_metadata(
    *,
    doc_id: str,
    source: "SourceImage",   # noqa: F821 (recompress.SourceImage)
    q2: int,
    edit_type: str,
    size_class: str,
    alignment: str,
    bbox,
    seed: int,
    forge_result=None,
    input_res: int = 384,
    patch_size: int = 16,
    grid: int = 24,
    overlap_thr: float = 0.5,
    q2_subsampling: str = "4:2:0",
) -> dict:
    """Assemble the per-document JSON dict (instruction.md schema + useful extras)."""
    meta = {
        "id": doc_id,
        "source_id": source.source_id,
        # --- Compression (Q0 READ, never chosen; Q2 the only swept parameter) ---
        "Q0_lu": source.q0,
        "q0_absdiff": source.absdiff,
        "q0_nonstandard": source.nonstandard,
        "table_quant": source.qtable_luma,     # luma table read from the source
        "subsampling": source.subsampling,     # source subsampling
        "Q2": int(q2),
        "Q2_subsampling": q2_subsampling,
        # --- Forgery -------------------------------------------------------
        "type": edit_type,                     # substitution / copy_move / splice / authentic
        "size_class": size_class,
        "alignment": alignment,                # aligned / misaligned / N/A
        "bbox": bbox,                          # [x,y,w,h] or None (negative)
        # --- Reproducibility ----------------------------------------------
        "seed": int(seed),
        "width": source.width,
        "height": source.height,
    }
    if forge_result is not None:
        meta["src_bbox"] = forge_result.src_bbox
        meta["area_frac"] = forge_result.area_frac
        meta["feather_radius"] = forge_result.feather_radius
        meta["donor_source_id"] = forge_result.donor_source_id
        if forge_result.extra:
            meta["extra"] = forge_result.extra
    return meta
