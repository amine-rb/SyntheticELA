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

# Polices Hershey d'OpenCV (scalables, aucun fichier externe requis) : on couvre
# un rendu proche d'une police de reçu.
_CV_FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN]

# Lever 2 — encre : contraste minimal (luminance) fond↔encre pour garantir de
# VRAIS bords (donc un vrai signal ELA), et luminance cible d'une encre sombre
# de secours quand la zone n'a pas de pixels naturellement foncés.
_MIN_INK_CONTRAST = 90.0
_INK_TARGET_LUM = 55.0

# Réalisme substitution : la hauteur du texte injecté est CALÉE SUR LE TEXTE RÉEL
# du document (mesuré par composantes connexes, cf. `_estimate_text_height`), et
# NON sur une fraction devinée -> plus de « gros chiffres » qui trahissent le
# générateur (un vrai faussaire respecte la taille du corps de texte).
#   _SUBST_EMPHASIS : facteur vs le texte du document (1.0 = identique ; léger
#     renforcement possible, comme un montant total mis en avant).
#   _LINE_H_FRAC : REPLI seulement, si la page a trop peu de texte pour une mesure
#     fiable (calé ~0.6–1.1% de page, ordre de grandeur d'un corps de facture).
#   _SIZE_NCHARS : longueur de la valeur plausible selon la classe de taille.
_LINE_H_FRAC = (0.006, 0.011)
_SUBST_EMPHASIS = (0.9, 1.3)
_SIZE_NCHARS = {"small": (2, 6), "medium": (4, 9), "large": (6, 13), "very_large": (9, 18)}


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


def _content_mask(img: np.ndarray) -> np.ndarray:
    """Masque booléen (H, W) des pixels 'encre' (contenu sombre) via Otsu.

    Lever 1 : sert à placer les falsifications SUR du contenu réel (texte,
    chiffres) plutôt que dans les marges blanches -> zone réaliste et porteuse de
    signal ELA. Sur une page quasi vide, le masque est presque vide et le
    placement retombe gracieusement sur le meilleur candidat (cf. `_place_dest`).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Otsu -> seuil bimodal texte/fond ; INV => True là où c'est sombre (encre).
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binv > 0


def _content_frac(content, y, x, h, w) -> float:
    """Fraction de pixels 'encre' dans le rectangle (y,x,h,w)."""
    return float(content[y:y + h, x:x + w].mean()) if content is not None else 0.0


def _estimate_text_height(img: np.ndarray, content: Optional[np.ndarray] = None) -> Optional[float]:
    """Hauteur médiane des glyphes du DOCUMENT (px), via composantes connexes du
    masque d'encre.

    Sert à caler la substitution sur la taille du texte RÉEL (anti-tell : plus de
    texte géant). On ne garde que des blobs « glyphe » : hauteur dans une bande
    relative à la page (exclut le bruit/points en bas, logos et gros titres en
    haut) et largeur bornée (exclut filets et lignes pleine largeur). Renvoie None
    si trop peu de texte fiable -> l'appelant retombe sur une fraction de page.
    """
    H, W = img.shape[:2]
    ink = content if content is not None else _content_mask(img)
    ink_u8 = (ink.astype(np.uint8) * 255)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink_u8, connectivity=8)
    h_lo = max(3, int(round(0.0015 * H)))         # exclut bruit / points / accents
    h_hi = max(h_lo + 1, int(round(0.05 * H)))    # exclut logos / gros titres
    w_hi = max(1, int(round(0.35 * W)))           # exclut filets / lignes pleine largeur
    heights = []
    for i in range(1, n):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        ba = stats[i, cv2.CC_STAT_AREA]
        if h_lo <= bh <= h_hi and bw <= w_hi and ba >= 6:
            heights.append(bh)
    if len(heights) < 10:                          # page quasi vide / atypique
        return None
    return float(np.median(heights))


def _pick_source(rng, img_h, img_w, h, w, content=None,
                 min_content_frac=0.0, max_tries=40) -> tuple[int, int]:
    """Position SOURCE (snappée grille 8) pour copy_move/splice, préférant le
    contenu si `content` fourni. Dégradation gracieuse : meilleur candidat vu."""
    max_sy = ((img_h - h) // JPEG_BLOCK) * JPEG_BLOCK
    max_sx = ((img_w - w) // JPEG_BLOCK) * JPEG_BLOCK
    last, best = (0, 0), None
    for _ in range(max_tries):
        sy = max(0, min(_snap8(int(rng.integers(0, max(1, img_h - h)))), max_sy))
        sx = max(0, min(_snap8(int(rng.integers(0, max(1, img_w - w)))), max_sx))
        last = (sy, sx)
        if content is None or min_content_frac <= 0.0:
            return sy, sx
        frac = _content_frac(content, sy, sx, h, w)
        if frac >= min_content_frac:
            return sy, sx
        if best is None or frac > best[0]:
            best = (frac, sy, sx)
    return (best[1], best[2]) if best else last


def _place_dest(rng, img_h, img_w, h, w, alignment, forbid=None,
                content=None, min_content_frac=0.0, max_tries=60) -> tuple[int, int]:
    """Choisit la position de destination (y, x) selon l'alignement demandé.

    - aligned    : (y, x) multiples de 8  -> le décalage source->dest est mult. 8,
                   donc la grille Q0/Q0' recopiée reste en phase avec la grille
                   globale (cas difficile).
    - misaligned : au moins un axe non multiple de 8 (cas facile).
    - N/A        : position quelconque (substitution : pas de grille antérieure).

    `forbid` = liste de bbox [x,y,w,h] à ne PAS chevaucher (contrainte DURE) :
    région source d'un copy-move ET zones déjà falsifiées (multi-falsification).
    `content` + `min_content_frac` (Lever 1) = préfère les positions couvrant du
    contenu réel (contrainte SOUPLE, best-effort). Après `max_tries` échecs, on
    renvoie le meilleur candidat respectant `forbid` (ou le dernier tiré) :
    dégradation gracieuse, jamais de blocage ni de masque vide.
    """
    forbid = forbid or []
    last, best = (0, 0), None
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

        last = (y, x)
        # Contrainte DURE : pas de chevauchement avec une zone interdite.
        clash = False
        for fx, fy, fw, fh in forbid:
            if not (x + w <= fx or fx + fw <= x or y + h <= fy or fy + fh <= y):
                clash = True
                break
        if clash:
            continue
        # Contrainte SOUPLE : préférer le contenu.
        if content is None or min_content_frac <= 0.0:
            return y, x
        frac = _content_frac(content, y, x, h, w)
        if frac >= min_content_frac:
            return y, x
        if best is None or frac > best[0]:
            best = (frac, y, x)
    if best is not None:
        return best[1], best[2]
    return last  # aucun candidat sans chevauchement : dernier tirage


def _hard_mask(img_h, img_w, y, x, h, w) -> np.ndarray:
    m = np.zeros((img_h, img_w), dtype=np.uint8)
    m[y:y + h, x:x + w] = 255
    return m


# ---------------------------------------------------------------- éditions
def _plausible_token(rng, size_class) -> str:
    """Valeur PLAUSIBLE au format document (montant, date, entier, code), dont la
    longueur dépend de la classe de taille. Remplace le charabia aléatoire :
    la falsification ressemble à une vraie valeur éditée (montant/date/quantité)."""
    lo, hi = _SIZE_NCHARS.get(size_class, (3, 8))
    target = int(rng.integers(lo, hi + 1))
    kind = str(rng.choice(["amount", "amount", "amount", "date", "int", "code"]))
    if kind == "amount":
        digits = max(1, min(6, target - 3))
        whole = int(rng.integers(1, 10 ** digits))
        s = f"{whole:,}".replace(",", ".")               # séparateur milliers style EU
        return f"{s},{int(rng.integers(0, 100)):02d} €"
    if kind == "date":
        return (f"{int(rng.integers(1, 29)):02d}.{int(rng.integers(1, 13)):02d}."
                f"{int(rng.integers(1995, 2025))}")
    if kind == "int":
        d = max(1, min(6, target))
        return str(int(rng.integers(10 ** (d - 1), 10 ** d)))
    letters = "".join(str(rng.choice(list(string.ascii_uppercase)))
                      for _ in range(max(2, target // 3)))
    return f"{letters}-{int(rng.integers(100, 9999))}"


def _forge_substitution(rng, img, size_class, area_range, feather,
                        min_w_px: int, min_h_px: int, forbid=None,
                        content=None, min_content_frac=0.0) -> ForgeResult:
    """Écrit une VALEUR PLAUSIBLE à la taille du texte du document.

    Réalisme (isole le signal de compression, évite le « tell » du générateur) :
    - police calée sur une hauteur de LIGNE réaliste (pas sur une grosse boîte) ;
    - contenu = montant/date/quantité/code au format document (pas de charabia) ;
    - édition SERRÉE : la boîte (= le masque) épouse le texte, pas de grand aplat.
    + Lever 1 : placée sur du contenu ; + Lever 2 : encre sombre contrastée.
    """
    H, W = img.shape[:2]
    _check_min_fits(H, W, min_h_px, min_w_px, context="(substitution)")

    # Hauteur de ligne = celle du TEXTE RÉEL du document (anti-tell), légèrement
    # modulée (emphasis). Repli sur une fraction de page si trop peu de texte
    # mesurable. Indépendante de toute grosse boîte -> plus de charabia géant.
    emphasis = float(rng.uniform(*_SUBST_EMPHASIS))
    doc_text_h = _estimate_text_height(img, content)
    if doc_text_h is not None:
        line_px = int(round(doc_text_h * emphasis))
    else:
        line_px = int(round(H * float(rng.uniform(*_LINE_H_FRAC)) * emphasis))
    # Plancher renderable (8px) ; le plancher min_region_px est garanti par la
    # boîte plus bas (`h = max(min_h_px, ...)`). Plafond de sécurité H//4.
    line_px = max(8, min(line_px, H // 4))

    # Valeur plausible + police, échelle calibrée pour atteindre `line_px`.
    token = _plausible_token(rng, size_class)
    font = _CV_FONTS[int(rng.integers(0, len(_CV_FONTS)))]
    (_, th1), _ = cv2.getTextSize(token, font, 1.0, 1)
    scale = max(0.3, line_px / max(th1, 1))
    thickness = max(1, int(round(line_px / 14)))
    (tw, th), base = cv2.getTextSize(token, font, scale, thickness)

    pad = max(2, line_px // 6)
    # Réduire l'échelle si le texte déborde en largeur de l'image.
    if tw + 2 * pad > W - 2:
        scale = max(0.3, scale * (W - 2) / float(tw + 2 * pad))
        (tw, th), base = cv2.getTextSize(token, font, scale, thickness)

    # Boîte SERRÉE autour du texte (= masque). Respecte le plancher min et l'image.
    h = max(min_h_px, min(th + base + 2 * pad, H - 1))
    w = max(min_w_px, min(tw + 2 * pad, W - 1))

    y, x = _place_dest(rng, H, W, h, w, "N/A", forbid=forbid,
                       content=content, min_content_frac=min_content_frac)

    region = img[y:y + h, x:x + w]
    # Fond échantillonné localement (médiane) -> cohérence photométrique.
    bg = np.median(region.reshape(-1, 3), axis=0).astype(np.float64)
    # Encre = pixels sombres locaux (l'encre est plus sombre que le fond).
    gray = region.reshape(-1, 3).mean(axis=1)
    dark_idx = gray <= np.percentile(gray, 15)
    ink = (region.reshape(-1, 3)[dark_idx].mean(axis=0).astype(np.float64)
           if dark_idx.any() else bg.copy())
    # Lever 2 : garantir une encre SOMBRE contrastée si la zone est claire/plate.
    bg_lum, ink_lum = float(bg.mean()), float(ink.mean())
    if bg_lum > 100.0 and (bg_lum - ink_lum) < _MIN_INK_CONTRAST:
        ink = np.clip(bg * (_INK_TARGET_LUM / max(bg_lum, 1e-6)), 0, 255)

    patch = np.empty((h, w, 3), dtype=np.uint8)
    patch[:] = bg.astype(np.uint8)
    org = (pad, pad + th)                       # origine putText (bas-gauche)
    cv2.putText(patch, token, org, font, scale,
                tuple(float(c) for c in ink), thickness, cv2.LINE_AA)

    edited = _composite(img, patch, y, x, h, w, feather)
    mask = _hard_mask(H, W, y, x, h, w)
    frac = (h * w) / (H * W)
    return ForgeResult(
        image=edited, mask=mask, edit_type="substitution", size_class=size_class,
        alignment="N/A", bbox=[x, y, w, h], area_frac=round(frac, 6),
        feather_radius=round(feather, 3), extra={"text": token},
    )


def _forge_copy_move(rng, img, size_class, area_range, alignment, feather,
                     min_w_px: int, min_h_px: int, forbid=None,
                     content=None, min_content_frac=0.0) -> ForgeResult:
    """Recopie une région de la MÊME image (porte la grille Q0).

    Lever 1 : SOURCE et DESTINATION préfèrent du contenu réel -> on copie une
    vraie zone d'encre et on la colle sur une zone qui porte du contenu.
    """
    H, W = img.shape[:2]
    h, w, frac = _sample_size(rng, H, W, area_range, min_w_px, min_h_px)
    # Source snappée sur la grille 8 -> on copie des blocs Q0 entiers, phase propre.
    sy, sx = _pick_source(rng, H, W, h, w, content=content,
                          min_content_frac=min_content_frac)
    patch = img[sy:sy + h, sx:sx + w].copy()

    # Éviter la région source ET les zones déjà falsifiées.
    y, x = _place_dest(rng, H, W, h, w, alignment,
                       forbid=[[sx, sy, w, h], *(forbid or [])],
                       content=content, min_content_frac=min_content_frac)
    edited = _composite(img, patch, y, x, h, w, feather)
    mask = _hard_mask(H, W, y, x, h, w)
    return ForgeResult(
        image=edited, mask=mask, edit_type="copy_move", size_class=size_class,
        alignment=alignment, bbox=[x, y, w, h], src_bbox=[sx, sy, w, h],
        area_frac=round(frac, 6), feather_radius=round(feather, 3),
    )


def _forge_splice(rng, img, donor, donor_id, size_class, area_range,
                  alignment, feather, min_w_px: int, min_h_px: int, forbid=None,
                  content=None, min_content_frac=0.0, donor_content=None) -> ForgeResult:
    """Insère une région d'un AUTRE document (porte la grille Q0' du donneur).

    Lever 1 : la région SOURCE (dans le donneur) et la DESTINATION préfèrent du
    contenu -> on insère de l'encre étrangère sur une zone porteuse de contenu.
    """
    H, W = img.shape[:2]
    Hd, Wd = donor.shape[:2]
    # Taille limitée par la plus petite des deux images (dest ET donneur).
    h, w, frac = _sample_size(rng, min(H, Hd), min(W, Wd), area_range, min_w_px, min_h_px)
    # Reclip strict contre le donneur seul (peut être plus petit que dest) :
    # même garantie que _sample_size, jamais de rectangle 0px silencieux.
    usable_hd, usable_wd = _check_min_fits(Hd, Wd, min_h_px, min_w_px, context="(donneur)")
    h = _clip_to_min(h, min_h_px, usable_hd)
    w = _clip_to_min(w, min_w_px, usable_wd)

    sy, sx = _pick_source(rng, Hd, Wd, h, w, content=donor_content,
                          min_content_frac=min_content_frac)
    patch = donor[sy:sy + h, sx:sx + w].copy()

    y, x = _place_dest(rng, H, W, h, w, alignment, forbid=forbid,
                       content=content, min_content_frac=min_content_frac)
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
    forbid: Optional[list] = None,
    on_content: bool = False,
    min_content_frac: float = 0.0,
) -> ForgeResult:
    """Point d'entrée : applique UNE falsification et renvoie image+masque+méta.

    `alignment` est ignoré pour la substitution (forcé "N/A"). `donor` est requis
    pour le splice. `min_region_px` = taille MINIMALE garantie du rectangle
    falsifié : un entier (carré) ou `[largeur_min, hauteur_min]` en pixels,
    arrondi au multiple de 8 SUPÉRIEUR (grille JPEG). Si l'image source est trop
    petite pour l'accueillir, une erreur explicite est levée (le job est alors
    journalisé en erreur par l'orchestrator, jamais écrit en positif à masque
    vide). `forbid` = liste de bbox [x,y,w,h] déjà falsifiées à ne pas chevaucher
    (multi-falsification : appeler `forge` k fois en accumulant les bbox rendues).
    `on_content` (Lever 1) : place la falsification sur du contenu réel (encre)
    plutôt que dans le vide, avec `min_content_frac` = fraction min d'encre visée.
    """
    feather = float(rng.uniform(feather_range[0], feather_range[1]))
    min_w_px, min_h_px = normalize_min_region(min_region_px)

    # Lever 1 : masques de contenu calculés une fois (best-effort au placement).
    content = _content_mask(img) if on_content else None
    mcf = float(min_content_frac) if on_content else 0.0

    if edit_type == "substitution":
        result = _forge_substitution(rng, img, size_class, area_range, feather,
                                     min_w_px, min_h_px, forbid,
                                     content=content, min_content_frac=mcf)
    elif edit_type == "copy_move":
        result = _forge_copy_move(rng, img, size_class, area_range, alignment, feather,
                                  min_w_px, min_h_px, forbid,
                                  content=content, min_content_frac=mcf)
    elif edit_type == "splice":
        if donor is None:
            raise ValueError("splice requiert une image donneuse (donor).")
        donor_content = _content_mask(donor) if on_content else None
        result = _forge_splice(rng, img, donor, donor_id, size_class, area_range,
                               alignment, feather, min_w_px, min_h_px, forbid,
                               content=content, min_content_frac=mcf,
                               donor_content=donor_content)
    else:
        raise ValueError(f"type d'édition inconnu : {edit_type}")

    # Filet de sécurité : un positif ne doit JAMAIS sortir avec un masque vide.
    if not result.mask.any():
        raise RuntimeError(
            f"masque vide produit pour edit_type={edit_type} (bbox={result.bbox}) "
            "malgré le plancher min_region_px : signaler ce cas, il ne devrait "
            "plus être possible.")
    return result
