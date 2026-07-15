"""recompress — Module 3 du pipeline.

Rôle
----
1. Décoder une JPEG source vers un tableau RGB uint8 (les artefacts de blocking
   Q0 sont alors *cuits dans les pixels* : c'est cette empreinte que le forger
   déplacera pour créer l'incohérence).
2. Resauvegarder l'image finale (fond Q0->Q2 + zone à historique incohérent) en
   JPEG à la qualité Q2, en **une seule passe**, faux ET négatifs authentiques.

RÈGLE IMPÉRATIVE (instruction.md)
---------------------------------
La zone éditée subit la MÊME passe Q2 que le fond. On ne colle JAMAIS une zone
après ce save. Concrètement : le forger travaille en espace pixel sur l'image
DÉCODÉE, puis `save_q2` compresse l'ensemble en un seul appel. Toute la
cohérence forensique du dataset repose là-dessus.

Q0 n'est jamais choisi (lu par jpeg_probe). Q2 est le seul paramètre balayé.

Dépendances : Pillow + NumPy.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .jpeg_probe import estimate_quality, _read_subsampling


# Sous-échantillonnage chroma utilisé pour la passe Q2 finale.
# DÉFAUT = 2 (4:2:0) : c'est le subsampling dominant du corpus SROIE (623/626).
# Une recompression réaliste (ré-export) utilise le défaut de l'encodeur.
DEFAULT_Q2_SUBSAMPLING = 2  # 0=4:4:4, 1=4:2:2, 2=4:2:0


@dataclass
class SourceImage:
    """Image source décodée + métadonnées de compression lues dans le fichier."""
    rgb: np.ndarray          # (H, W, 3) uint8, image décodée
    q0: int                  # qualité estimée (luma) — LUE, jamais choisie
    absdiff: float
    nonstandard: bool
    subsampling: str         # subsampling du fichier source
    qtable_luma: list        # table de quantification luma (pour le JSON)
    width: int
    height: int
    source_id: str           # nom de fichier sans extension
    path: str


def decode_source(
    path: str,
    nonstandard_threshold: float = 40.0,
    allow_lossless: bool = False,
) -> SourceImage:
    """Décode une source et lit ses paramètres de compression Q0.

    JPEG    -> Q0 LU dans la table de quantification (jamais choisi).
    lossless (PNG, si `allow_lossless=True`) -> pas d'historique JPEG : q0=-1,
              table vide, subsampling "none". Exploitable UNIQUEMENT en mode Q1
              contrôlé (le Q1 imposé devient l'unique historique du fond).

    Force le mode RGB pour homogénéiser le pipeline en aval.
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
        raise ValueError(f"{path} n'est pas un JPEG (format={img.format}) ; "
                         f"active allow_lossless pour une source PNG.")
    else:
        raise ValueError(f"{path} n'a pas de table de quantification (Q0 illisible)")

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
    """Ré-encode une image en JPEG Q1 puis la re-décode, en mémoire.

    Rôle (mode Q1 contrôlé — cf. plan.md §Étape 0 : "Q1 et Q2 paramétrables
    indépendamment, paramètre central pour E5"). Le corpus source SROIE étant
    quasi sans perte (Q0≈100), une passe unique à Q1 impose un historique de
    compression EFFECTIF propre au niveau Q1 : c'est cette passe qui devient la
    grille/quantification de référence du "document original" avant falsification.

    Le fond du document falsifié aura alors l'historique Q1->Q2 (double
    compression réellement détectable), au lieu de Q0(≈100)->Q2 (fingerprint
    faible). Q0 reste LU et journalisé ; Q1 est un paramètre expérimental explicite.
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
    """Sauve l'image (uint8 RGB) en JPEG qualité Q2, en une seule passe.

    C'est l'unique point où une compression est écrite. Utilisé aussi bien pour
    les faux (fond + zone incohérente déjà composités) que pour les négatifs
    authentiques (image décodée telle quelle).
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
