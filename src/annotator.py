"""annotator — Module 4 du pipeline.

Rôle
----
Produire la vérité terrain d'un document falsifié (ou négatif) :
    - masque binaire pixel-level EXACT (empreinte géométrique, sans dilatation),
    - bbox de la (des) zone(s) éditée(s),
    - grille de labels patch-level 24x24 (patch 16 px sur l'entrée 384 du modèle),
      label positif si le recouvrement du masque dépasse un seuil (défaut 0.5),
    - dictionnaire de métadonnées prêt à sérialiser en JSON.

Note sur la grille patch
------------------------
Le modèle en aval redimensionne l'entrée en 384x384 puis la découpe en patchs
16 px -> grille 24x24. On calcule donc les labels patch en ramenant le masque à
384x384 (au plus proche, pour rester binaire) puis en mesurant le recouvrement
par patch. Cela reflète exactement ce que "voit" le modèle.

Dépendances : NumPy + OpenCV (resize).
"""

from __future__ import annotations

import numpy as np
import cv2


def bbox_from_mask(mask: np.ndarray):
    """Bbox englobante [x, y, w, h] des pixels non nuls, ou None si masque vide."""
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
    """Grille de labels patch-level (grid x grid) + fractions de recouvrement.

    Retourne (labels[int8], overlap_fracs[float32]) tous deux de forme (grid, grid).
    """
    # Masque -> 384x384 binaire (nearest préserve le caractère binaire).
    m = (mask > 0).astype(np.uint8) * 255
    m384 = cv2.resize(m, (input_res, input_res), interpolation=cv2.INTER_NEAREST)
    m384 = (m384 > 127).astype(np.float32)

    # Recouvrement moyen par patch = fraction de pixels falsifiés dans le patch.
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
    """Assemble le dict JSON par document (schéma instruction.md + extras utiles)."""
    meta = {
        "id": doc_id,
        "source_id": source.source_id,
        # --- Compression (Q0 LU, jamais choisi ; Q2 seul paramètre balayé) ---
        "Q0_lu": source.q0,
        "q0_absdiff": source.absdiff,
        "q0_nonstandard": source.nonstandard,
        "table_quant": source.qtable_luma,     # table luma lue dans la source
        "subsampling": source.subsampling,     # subsampling source
        "Q2": int(q2),
        "Q2_subsampling": q2_subsampling,
        # --- Falsification -------------------------------------------------
        "type": edit_type,                     # substitution / copy_move / splice / authentic
        "size_class": size_class,
        "alignment": alignment,                # aligned / misaligned / N/A
        "bbox": bbox,                          # [x,y,w,h] ou None (négatif)
        # --- Reproductibilité ---------------------------------------------
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
