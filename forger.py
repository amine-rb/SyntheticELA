"""forger — Module 2 du pipeline (cœur de la génération).

Rôle
----
À partir d'une image DÉCODÉE (RGB uint8, artefacts Q0 cuits dans les pixels),
appliquer une falsification locale et retourner :
    - l'image éditée (toujours en espace pixel, AVANT la passe Q2),
    - le masque binaire pixel-level = empreinte géométrique EXACTE (sans dilatation),
    - les métadonnées géométriques (type, taille, alignement, bbox).

Trois types d'édition (instruction.md)
--------------------------------------
1. substitution : pixels neufs redessinés (texte). La zone n'a PAS de grille 8x8
   Q0 antérieure -> après Q2 elle est simplement compressée (single) tandis que
   le fond est double-compressé (Q0->Q2). alignment = "N/A".
2. copy_move    : région recopiée depuis la MÊME image (porte la grille Q0).
   L'offset de collage contrôle l'alignement : multiple de 8 -> aligné (difficile,
   grille cohérente) ; non multiple de 8 -> désaligné (facile).
3. splice       : région venue d'un AUTRE document du corpus (porte SA grille Q0').
   Même contrôle d'alignement via l'offset.

Anti-"tell" du générateur
--------------------------
- Feather gaussien léger des bords (rayon tiré dans [0.5, 2] px) : composite
  alpha doux, mais le MASQUE reste l'empreinte dure (le feather n'est pas de la
  falsification en plus).
- Substitution : couleur de fond échantillonnée localement, couleur du texte
  dérivée des pixels sombres locaux (cohérence photométrique), police cohérente.

Taille minimale garantie (min_region_px)
-----------------------------------------
Chaque rectangle falsifié respecte un plancher [largeur_min, hauteur_min] en
pixels (config `forger.min_region_px`), quel que soit `size_class` ou la taille
de l'image. Le plancher est arrondi AU MULTIPLE DE 8 SUPÉRIEUR (grille JPEG) :
demander 10 garantit >= 16, jamais moins. Si une image source ne peut pas
accueillir ce minimum sur un axe, une erreur explicite est levée (le job est
alors journalisé en erreur par l'orchestrator, JAMAIS écrit en positif à masque
vide).

IMPORTANT : le forger ne compresse RIEN. La passe Q2 unique est faite ensuite par
`recompress.save_q2` sur l'image composite entière.

Dépendances : NumPy + OpenCV.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Optional, Union

import cv2
import numpy as np


# ------------------------------------------------------------------ constantes
# Grille DCT JPEG : les blocs font 8x8. L'alignement se joue modulo 8.
JPEG_BLOCK = 8

# Alphabet pour le texte synthétique de substitution (montants/dates plausibles).
_TEXT_ALPHABET = string.digits + string.ascii_uppercase + " .,:/-$"

# Polices Hershey d'OpenCV (scalables, aucun fichier externe requis) : on couvre
# un rendu proche d'une police de reçu.
_CV_FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN]


@dataclass
class ForgeResult:
    image: np.ndarray                     # (H, W, 3) uint8, éditée, AVANT Q2
    mask: np.ndarray                      # (H, W) uint8 {0,255}, empreinte exacte
    edit_type: str                        # substitution / copy_move / splice
    size_class: str
    alignment: str                        # aligned / misaligned / N/A
    bbox: list                            # [x, y, w, h] de la zone éditée (dest)
    src_bbox: Optional[list] = None       # bbox source (copy_move / splice)
    area_frac: float = 0.0
    feather_radius: float = 0.0
    donor_source_id: Optional[str] = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------- helpers géom.
def _snap8(v: int) -> int:
    """Arrondit au multiple de 8 le plus proche (>= 8). Pour le TIRAGE de taille."""
    return max(JPEG_BLOCK, int(round(v / JPEG_BLOCK)) * JPEG_BLOCK)


def _ceil8(v: int) -> int:
    """Arrondit au multiple de 8 SUPÉRIEUR (>= 8). Pour un PLANCHER minimal :
    ne redescend jamais sous la valeur demandée (10 -> 16, jamais 8)."""
    return max(JPEG_BLOCK, int(np.ceil(v / JPEG_BLOCK)) * JPEG_BLOCK)


def normalize_min_region(min_region_px: Union[int, tuple, list]) -> tuple[int, int]:
    """Normalise `min_region_px` (entier OU [largeur_min, hauteur_min]) en
    (min_w_px, min_h_px), tous deux multiples de 8 (arrondi supérieur).

    Un entier scalaire s'applique aux deux axes.
    """
    if isinstance(min_region_px, (list, tuple)):
        min_w, min_h = min_region_px
    else:
        min_w = min_h = min_region_px
    return _ceil8(int(min_w)), _ceil8(int(min_h))


def _usable_dim(size: int) -> int:
    """Plus grand multiple de 8 tenant dans `size` (0 si size < 8)."""
    return (size // JPEG_BLOCK) * JPEG_BLOCK


def _clip_to_min(value: int, min_size_px: int, usable: int) -> int:
    """Borne `value` (déjà multiple de 8) dans [min_size_px, usable].

    `usable` >= min_size_px est un PRÉREQUIS (vérifié par l'appelant via
    `_check_min_fits`) : le clip ne peut donc jamais redescendre sous le
    plancher -> plus aucun rectangle 0px silencieux.
    """
    return min(max(value, min_size_px), usable)


def _check_min_fits(img_h: int, img_w: int, min_h_px: int, min_w_px: int,
                    context: str = "") -> tuple[int, int]:
    """Vérifie que l'image peut accueillir (min_w_px x min_h_px).

    Retourne (usable_h, usable_w). Lève une erreur EXPLICITE sinon (image
    dégénérée/trop petite) : mieux qu'un masque vide écrit silencieusement.
    """
    usable_h, usable_w = _usable_dim(img_h), _usable_dim(img_w)
    if usable_h < min_h_px or usable_w < min_w_px:
        raise ValueError(
            f"image trop petite ({img_w}x{img_h}){' ' + context if context else ''} "
            f"pour min_region_px=({min_w_px}x{min_h_px}) : aucun rectangle "
            "non-dégénéré possible (réduis forger.min_region_px, ou filtre "
            "cette source en amont).")
    return usable_h, usable_w


def _sample_size(rng, img_h, img_w, area_range,
                 min_w_px: int = JPEG_BLOCK, min_h_px: int = JPEG_BLOCK) -> tuple[int, int, float]:
    """Tire (h, w) mult. de 8 pour une fraction d'aire dans area_range.

    (min_w_px, min_h_px) est un plancher GARANTI (jamais de rectangle plus
    petit, jamais de rectangle qui dépasse l'image) : voir `_check_min_fits`.
    """
    usable_h, usable_w = _check_min_fits(img_h, img_w, min_h_px, min_w_px)

    frac = float(rng.uniform(area_range[0], area_range[1]))
    target_area = frac * img_h * img_w
    # Aspect ratio modéré (les zones de reçu sont souvent oblongues).
    ar = float(rng.uniform(0.3, 3.0))
    w = _snap8(int(np.sqrt(target_area * ar)))
    h = _snap8(int(np.sqrt(target_area / ar)))
    w = _clip_to_min(w, min_w_px, usable_w)
    h = _clip_to_min(h, min_h_px, usable_h)
    real_frac = (h * w) / (img_h * img_w)
    return h, w, real_frac


def _feather_alpha(h, w, radius, dtype=np.float32) -> np.ndarray:
    """Alpha rectangulaire (1 au centre) adouci par flou gaussien (anti-tell)."""
    alpha = np.ones((h, w), dtype=dtype)
    if radius and radius > 0:
        k = int(max(1, round(radius * 3)) | 1)  # noyau impair
        alpha = cv2.GaussianBlur(alpha, (k, k), sigmaX=float(radius),
                                 borderType=cv2.BORDER_CONSTANT)
    return np.clip(alpha, 0.0, 1.0)


def _composite(base, patch, y, x, h, w, radius) -> np.ndarray:
    """Colle `patch` (h,w,3) sur `base` en (y,x) avec bords feathered.

    Retourne une copie éditée de `base`. Le feather n'affecte QUE le composite
    visuel, pas le masque (qui reste le rectangle dur).
    """
    out = base.copy()
    alpha = _feather_alpha(h, w, radius)[..., None]  # (h,w,1)
    region = out[y:y + h, x:x + w].astype(np.float32)
    blended = patch.astype(np.float32) * alpha + region * (1.0 - alpha)
    out[y:y + h, x:x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _place_dest(rng, img_h, img_w, h, w, alignment,
                forbid=None, max_tries=60) -> tuple[int, int]:
    """Choisit la position de destination (y, x) selon l'alignement demandé.

    - aligned    : (y, x) multiples de 8  -> le décalage source->dest est mult. 8,
                   donc la grille Q0/Q0' recopiée reste en phase avec la grille
                   globale (cas difficile).
    - misaligned : au moins un axe non multiple de 8 (cas facile).
    - N/A        : position quelconque (substitution : pas de grille antérieure).

    `forbid` = bbox source [x,y,w,h] à éviter (pour le copy-move, ne pas coller
    sur soi-même).
    """
    for _ in range(max_tries):
        if alignment == "aligned":
            y = _snap8(int(rng.integers(0, max(1, img_h - h))))
            x = _snap8(int(rng.integers(0, max(1, img_w - w))))
            y = min(y, ((img_h - h) // JPEG_BLOCK) * JPEG_BLOCK)
            x = min(x, ((img_w - w) // JPEG_BLOCK) * JPEG_BLOCK)
            y, x = max(0, y), max(0, x)
        elif alignment == "misaligned":
            y = int(rng.integers(0, max(1, img_h - h)))
            x = int(rng.integers(0, max(1, img_w - w)))
            # Force un déphasage non nul sur au moins un axe.
            if y % JPEG_BLOCK == 0 and x % JPEG_BLOCK == 0:
                x = min(x + int(rng.integers(1, JPEG_BLOCK)), img_w - w)
        else:  # N/A
            y = int(rng.integers(0, max(1, img_h - h)))
            x = int(rng.integers(0, max(1, img_w - w)))

        if forbid is None:
            return y, x
        fx, fy, fw, fh = forbid
        # Rejet si chevauchement avec la région source.
        overlap = not (x + w <= fx or fx + fw <= x or y + h <= fy or fy + fh <= y)
        if not overlap:
            return y, x
    return y, x  # dernier tirage même si imparfait


def _hard_mask(img_h, img_w, y, x, h, w) -> np.ndarray:
    m = np.zeros((img_h, img_w), dtype=np.uint8)
    m[y:y + h, x:x + w] = 255
    return m


# ---------------------------------------------------------------- éditions
def _forge_substitution(rng, img, size_class, area_range, feather,
                        min_w_px: int, min_h_px: int) -> ForgeResult:
    """Peint des pixels neufs (texte) : aucune grille 8x8 Q0 antérieure."""
    H, W = img.shape[:2]
    h, w, frac = _sample_size(rng, H, W, area_range, min_w_px, min_h_px)
    y, x = _place_dest(rng, H, W, h, w, "N/A")

    region = img[y:y + h, x:x + w]
    # Fond échantillonné localement (médiane) -> cohérence photométrique.
    bg = np.median(region.reshape(-1, 3), axis=0)
    # Couleur du texte = pixels sombres locaux (l'encre est plus sombre que le fond).
    gray = region.reshape(-1, 3).mean(axis=1)
    dark_idx = gray <= np.percentile(gray, 15)
    if dark_idx.any():
        ink = region.reshape(-1, 3)[dark_idx].mean(axis=0)
    else:
        ink = np.clip(bg - 80, 0, 255)

    patch = np.empty((h, w, 3), dtype=np.uint8)
    patch[:] = bg.astype(np.uint8)

    # Texte synthétique plausible (montants/dates) rendu à l'échelle de la zone.
    n_chars = max(2, int(rng.integers(3, 9)))
    text = "".join(rng.choice(list(_TEXT_ALPHABET)) for _ in range(n_chars))
    font = _CV_FONTS[int(rng.integers(0, len(_CV_FONTS)))]
    # Ajuste la taille de police pour remplir ~70% de la hauteur de la zone.
    scale = max(0.3, (h * 0.6) / 22.0)
    thickness = max(1, int(round(h / 40)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    org_x = max(2, (w - tw) // 2)
    org_y = min(h - 2, (h + th) // 2)
    cv2.putText(patch, text, (org_x, org_y), font, scale,
                tuple(float(c) for c in ink), thickness, cv2.LINE_AA)

    edited = _composite(img, patch, y, x, h, w, feather)
    mask = _hard_mask(H, W, y, x, h, w)
    return ForgeResult(
        image=edited, mask=mask, edit_type="substitution", size_class=size_class,
        alignment="N/A", bbox=[x, y, w, h], area_frac=round(frac, 6),
        feather_radius=round(feather, 3), extra={"text": text},
    )


def _forge_copy_move(rng, img, size_class, area_range, alignment, feather,
                     min_w_px: int, min_h_px: int) -> ForgeResult:
    """Recopie une région de la MÊME image (porte la grille Q0)."""
    H, W = img.shape[:2]
    h, w, frac = _sample_size(rng, H, W, area_range, min_w_px, min_h_px)
    # Source snappée sur la grille 8 -> on copie des blocs Q0 entiers, phase propre.
    sy = _snap8(int(rng.integers(0, max(1, H - h))))
    sx = _snap8(int(rng.integers(0, max(1, W - w))))
    sy = max(0, min(sy, ((H - h) // JPEG_BLOCK) * JPEG_BLOCK))
    sx = max(0, min(sx, ((W - w) // JPEG_BLOCK) * JPEG_BLOCK))
    patch = img[sy:sy + h, sx:sx + w].copy()

    y, x = _place_dest(rng, H, W, h, w, alignment, forbid=[sx, sy, w, h])
    edited = _composite(img, patch, y, x, h, w, feather)
    mask = _hard_mask(H, W, y, x, h, w)
    return ForgeResult(
        image=edited, mask=mask, edit_type="copy_move", size_class=size_class,
        alignment=alignment, bbox=[x, y, w, h], src_bbox=[sx, sy, w, h],
        area_frac=round(frac, 6), feather_radius=round(feather, 3),
    )


def _forge_splice(rng, img, donor, donor_id, size_class, area_range,
                  alignment, feather, min_w_px: int, min_h_px: int) -> ForgeResult:
    """Insère une région d'un AUTRE document (porte la grille Q0' du donneur)."""
    H, W = img.shape[:2]
    Hd, Wd = donor.shape[:2]
    # Taille limitée par la plus petite des deux images (dest ET donneur).
    h, w, frac = _sample_size(rng, min(H, Hd), min(W, Wd), area_range, min_w_px, min_h_px)
    # Reclip strict contre le donneur seul (peut être plus petit que dest) :
    # même garantie que _sample_size, jamais de rectangle 0px silencieux.
    usable_hd, usable_wd = _check_min_fits(Hd, Wd, min_h_px, min_w_px, context="(donneur)")
    h = _clip_to_min(h, min_h_px, usable_hd)
    w = _clip_to_min(w, min_w_px, usable_wd)

    sy = _snap8(int(rng.integers(0, max(1, Hd - h))))
    sx = _snap8(int(rng.integers(0, max(1, Wd - w))))
    sy = max(0, min(sy, ((Hd - h) // JPEG_BLOCK) * JPEG_BLOCK))
    sx = max(0, min(sx, ((Wd - w) // JPEG_BLOCK) * JPEG_BLOCK))
    patch = donor[sy:sy + h, sx:sx + w].copy()

    y, x = _place_dest(rng, H, W, h, w, alignment)
    edited = _composite(img, patch, y, x, h, w, feather)
    mask = _hard_mask(H, W, y, x, h, w)
    return ForgeResult(
        image=edited, mask=mask, edit_type="splice", size_class=size_class,
        alignment=alignment, bbox=[x, y, w, h], src_bbox=[sx, sy, w, h],
        area_frac=round(frac, 6), feather_radius=round(feather, 3),
        donor_source_id=donor_id,
    )


# ------------------------------------------------------------------- API
def forge(
    img: np.ndarray,
    edit_type: str,
    size_class: str,
    area_range: tuple[float, float],
    alignment: str,
    feather_range: tuple[float, float],
    rng: np.random.Generator,
    donor: Optional[np.ndarray] = None,
    donor_id: Optional[str] = None,
    min_region_px: Union[int, tuple, list] = JPEG_BLOCK,
) -> ForgeResult:
    """Point d'entrée : applique une falsification et renvoie image+masque+méta.

    `alignment` est ignoré pour la substitution (forcé "N/A"). `donor` est requis
    pour le splice. `min_region_px` = taille MINIMALE garantie du rectangle
    falsifié : un entier (carré) ou `[largeur_min, hauteur_min]` en pixels,
    arrondi au multiple de 8 SUPÉRIEUR (grille JPEG). Si l'image source est trop
    petite pour l'accueillir, une erreur explicite est levée (le job est alors
    journalisé en erreur par l'orchestrator, jamais écrit en positif à masque
    vide).
    """
    feather = float(rng.uniform(feather_range[0], feather_range[1]))
    min_w_px, min_h_px = normalize_min_region(min_region_px)

    if edit_type == "substitution":
        result = _forge_substitution(rng, img, size_class, area_range, feather, min_w_px, min_h_px)
    elif edit_type == "copy_move":
        result = _forge_copy_move(rng, img, size_class, area_range, alignment, feather, min_w_px, min_h_px)
    elif edit_type == "splice":
        if donor is None:
            raise ValueError("splice requiert une image donneuse (donor).")
        result = _forge_splice(rng, img, donor, donor_id, size_class, area_range,
                               alignment, feather, min_w_px, min_h_px)
    else:
        raise ValueError(f"type d'édition inconnu : {edit_type}")

    # Filet de sécurité : un positif ne doit JAMAIS sortir avec un masque vide.
    if not result.mask.any():
        raise RuntimeError(
            f"masque vide produit pour edit_type={edit_type} (bbox={result.bbox}) "
            "malgré le plancher min_region_px : signaler ce cas, il ne devrait "
            "plus être possible.")
    return result
