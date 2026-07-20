"""forger — Pipeline module 2 (core of the generation).

Role
----
From a DECODED image (RGB uint8, Q0 artifacts baked into the pixels), apply a
local forgery and return:
    - the edited image (still in pixel space, BEFORE the Q2 pass),
    - the pixel-level binary mask = EXACT geometric footprint (no dilation),
    - the geometric metadata (type, size, alignment, bbox).

Three edit types (instruction.md)
---------------------------------
1. substitution : fresh pixels redrawn (text). The region has NO prior 8x8 Q0
   grid -> after Q2 it is simply compressed (single) while the background is
   double-compressed (Q0->Q2). alignment = "N/A".
2. copy_move    : region copied from the SAME image (carries the Q0 grid).
   The paste offset controls the alignment: multiple of 8 -> aligned (hard,
   coherent grid); not a multiple of 8 -> misaligned (easy).
3. splice       : region taken from ANOTHER corpus document (carries ITS Q0'
   grid). Same alignment control via the offset.

Generator anti-"tell"
----------------------
- Light gaussian feather of the edges (radius drawn in [0.5, 2] px): soft alpha
  composite, but the MASK stays the hard footprint (the feather is not extra
  forgery).
- Substitution: background color sampled locally, text color derived from local
  dark pixels (photometric consistency), coherent font.

Guaranteed minimum size (min_region_px)
---------------------------------------
Every forged rectangle respects a floor [min_width, min_height] in pixels
(config `forger.min_region_px`), regardless of `size_class` or image size. The
floor is rounded UP TO THE NEXT MULTIPLE OF 8 (JPEG grid): asking for 10
guarantees >= 16, never less. If a source image cannot fit this minimum on an
axis, an explicit error is raised (the job is then logged as an error by the
orchestrator, NEVER written as an empty-mask positive).

IMPORTANT: the forger compresses NOTHING. The single Q2 pass is done afterwards
by `recompress.save_q2` on the whole composite image.

Dependencies: NumPy + OpenCV.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Optional, Union

import cv2
import numpy as np

import lexicon


# ------------------------------------------------------------------ constants
# JPEG DCT grid: blocks are 8x8. Alignment plays out modulo 8.
JPEG_BLOCK = 8

# OpenCV Hershey fonts (scalable, no external file required): they approximate a
# receipt-like font rendering.
_CV_FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN]

# Lever 2 — ink: minimum background<->ink contrast (luminance) to guarantee REAL
# edges (hence a real ELA signal), and target luminance of a fallback dark ink
# when the region has no naturally dark pixels.
_MIN_INK_CONTRAST = 90.0
_INK_TARGET_LUM = 55.0

# Substitution realism: the height of the injected text is MATCHED TO THE ACTUAL
# TEXT of the document (measured by connected components, cf.
# `_estimate_text_height`), and NOT to a guessed fraction -> no more "big
# digits" that give away the generator (a real fraudster respects the body-text
# size).
#   _SUBST_EMPHASIS : factor vs the document text (1.0 = identical; a slight
#     boost is possible, like a highlighted total amount).
#   _LINE_H_FRAC : FALLBACK only, if the page has too little text for a reliable
#     measurement (calibrated ~0.6-1.1% of the page, order of magnitude of an
#     invoice body).
# (The corpus of injected values and the per-class lengths live in lexicon.py.)
_LINE_H_FRAC = (0.006, 0.011)
_SUBST_EMPHASIS = (0.9, 1.3)


@dataclass
class ForgeResult:
    image: np.ndarray                     # (H, W, 3) uint8, edited, BEFORE Q2
    mask: np.ndarray                      # (H, W) uint8 {0,255}, exact footprint
    edit_type: str                        # substitution / copy_move / splice
    size_class: str
    alignment: str                        # aligned / misaligned / N/A
    bbox: list                            # [x, y, w, h] of the edited region (dest)
    src_bbox: Optional[list] = None       # source bbox (copy_move / splice)
    area_frac: float = 0.0
    feather_radius: float = 0.0
    donor_source_id: Optional[str] = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------- geom. helpers
def _snap8(v: int) -> int:
    """Round to the nearest multiple of 8 (>= 8). For size SAMPLING."""
    return max(JPEG_BLOCK, int(round(v / JPEG_BLOCK)) * JPEG_BLOCK)


def _ceil8(v: int) -> int:
    """Round UP to the next multiple of 8 (>= 8). For a minimum FLOOR:
    never drops below the requested value (10 -> 16, never 8)."""
    return max(JPEG_BLOCK, int(np.ceil(v / JPEG_BLOCK)) * JPEG_BLOCK)


def normalize_min_region(min_region_px: Union[int, tuple, list]) -> tuple[int, int]:
    """Normalize `min_region_px` (integer OR [min_width, min_height]) into
    (min_w_px, min_h_px), both multiples of 8 (rounded up).

    A scalar integer applies to both axes.
    """
    if isinstance(min_region_px, (list, tuple)):
        min_w, min_h = min_region_px
    else:
        min_w = min_h = min_region_px
    return _ceil8(int(min_w)), _ceil8(int(min_h))


def _usable_dim(size: int) -> int:
    """Largest multiple of 8 fitting in `size` (0 if size < 8)."""
    return (size // JPEG_BLOCK) * JPEG_BLOCK


def _clip_to_min(value: int, min_size_px: int, usable: int) -> int:
    """Bound `value` (already a multiple of 8) within [min_size_px, usable].

    `usable` >= min_size_px is a PREREQUISITE (checked by the caller via
    `_check_min_fits`): the clip can therefore never drop below the floor ->
    no more silent 0px rectangles.
    """
    return min(max(value, min_size_px), usable)


def _check_min_fits(img_h: int, img_w: int, min_h_px: int, min_w_px: int,
                    context: str = "") -> tuple[int, int]:
    """Check that the image can fit (min_w_px x min_h_px).

    Returns (usable_h, usable_w). Raises an EXPLICIT error otherwise
    (degenerate/too-small image): better than a silently written empty mask.
    """
    usable_h, usable_w = _usable_dim(img_h), _usable_dim(img_w)
    if usable_h < min_h_px or usable_w < min_w_px:
        raise ValueError(
            f"image too small ({img_w}x{img_h}){' ' + context if context else ''} "
            f"for min_region_px=({min_w_px}x{min_h_px}): no non-degenerate "
            "rectangle possible (reduce forger.min_region_px, or filter "
            "out this source upstream).")
    return usable_h, usable_w


def _sample_size(rng, img_h, img_w, area_range,
                 min_w_px: int = JPEG_BLOCK, min_h_px: int = JPEG_BLOCK) -> tuple[int, int, float]:
    """Draw (h, w) as multiples of 8 for an area fraction within area_range.

    (min_w_px, min_h_px) is a GUARANTEED floor (never a smaller rectangle,
    never a rectangle exceeding the image): see `_check_min_fits`.
    """
    usable_h, usable_w = _check_min_fits(img_h, img_w, min_h_px, min_w_px)

    frac = float(rng.uniform(area_range[0], area_range[1]))
    target_area = frac * img_h * img_w
    # Moderate aspect ratio (receipt regions are often oblong).
    ar = float(rng.uniform(0.3, 3.0))
    w = _snap8(int(np.sqrt(target_area * ar)))
    h = _snap8(int(np.sqrt(target_area / ar)))
    w = _clip_to_min(w, min_w_px, usable_w)
    h = _clip_to_min(h, min_h_px, usable_h)
    real_frac = (h * w) / (img_h * img_w)
    return h, w, real_frac


def _feather_alpha(h, w, radius, dtype=np.float32) -> np.ndarray:
    """Rectangular alpha (1 at center) softened by gaussian blur (anti-tell)."""
    alpha = np.ones((h, w), dtype=dtype)
    if radius and radius > 0:
        k = int(max(1, round(radius * 3)) | 1)  # odd kernel
        alpha = cv2.GaussianBlur(alpha, (k, k), sigmaX=float(radius),
                                 borderType=cv2.BORDER_CONSTANT)
    return np.clip(alpha, 0.0, 1.0)


def _composite(base, patch, y, x, h, w, radius) -> np.ndarray:
    """Paste `patch` (h,w,3) onto `base` at (y,x) with feathered edges.

    Returns an edited copy of `base`. The feather affects ONLY the visual
    composite, not the mask (which stays the hard rectangle).
    """
    out = base.copy()
    alpha = _feather_alpha(h, w, radius)[..., None]  # (h,w,1)
    region = out[y:y + h, x:x + w].astype(np.float32)
    blended = patch.astype(np.float32) * alpha + region * (1.0 - alpha)
    out[y:y + h, x:x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _content_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask (H, W) of the 'ink' pixels (dark content) via Otsu.

    Lever 1: used to place forgeries ON real content (text, digits) rather than
    in blank margins -> a realistic region carrying an ELA signal. On a nearly
    empty page, the mask is almost empty and placement falls back gracefully to
    the best candidate (cf. `_place_dest`).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Otsu -> bimodal text/background threshold; INV => True where it is dark (ink).
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binv > 0


def _content_frac(content, y, x, h, w) -> float:
    """Fraction of 'ink' pixels in the rectangle (y,x,h,w)."""
    return float(content[y:y + h, x:x + w].mean()) if content is not None else 0.0


def _estimate_text_height(img: np.ndarray, content: Optional[np.ndarray] = None) -> Optional[float]:
    """Median glyph height of the DOCUMENT (px), via connected components of the
    ink mask.

    Used to match the substitution to the ACTUAL text size (anti-tell: no more
    giant text). We keep only 'glyph' blobs: height within a band relative to
    the page (excludes noise/dots at the bottom, logos and large titles at the
    top) and bounded width (excludes rules and full-width lines). Returns None
    if there is too little reliable text -> the caller falls back to a page
    fraction.
    """
    H, W = img.shape[:2]
    ink = content if content is not None else _content_mask(img)
    ink_u8 = (ink.astype(np.uint8) * 255)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink_u8, connectivity=8)
    h_lo = max(3, int(round(0.0015 * H)))         # excludes noise / dots / accents
    h_hi = max(h_lo + 1, int(round(0.05 * H)))    # excludes logos / large titles
    w_hi = max(1, int(round(0.35 * W)))           # excludes rules / full-width lines
    heights = []
    for i in range(1, n):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        ba = stats[i, cv2.CC_STAT_AREA]
        if h_lo <= bh <= h_hi and bw <= w_hi and ba >= 6:
            heights.append(bh)
    if len(heights) < 10:                          # nearly empty / atypical page
        return None
    return float(np.median(heights))


def _pick_source(rng, img_h, img_w, h, w, content=None,
                 min_content_frac=0.0, max_tries=40) -> tuple[int, int]:
    """SOURCE position (snapped to grid 8) for copy_move/splice, preferring
    content if `content` is provided. Graceful degradation: best candidate seen."""
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
    """Choose the destination position (y, x) according to the requested alignment.

    - aligned    : (y, x) multiples of 8 -> the source->dest offset is a
                   multiple of 8, so the copied Q0/Q0' grid stays in phase with
                   the global grid (hard case).
    - misaligned : at least one axis not a multiple of 8 (easy case).
    - N/A        : arbitrary position (substitution: no prior grid).

    `forbid` = list of bboxes [x,y,w,h] NOT to overlap (HARD constraint): the
    source region of a copy-move AND regions already forged (multi-forgery).
    `content` + `min_content_frac` (Lever 1) = prefer positions covering real
    content (SOFT constraint, best-effort). After `max_tries` failures, we
    return the best candidate respecting `forbid` (or the last draw): graceful
    degradation, never a deadlock nor an empty mask.
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
            # Force a non-zero phase shift on at least one axis.
            if y % JPEG_BLOCK == 0 and x % JPEG_BLOCK == 0:
                x = min(x + int(rng.integers(1, JPEG_BLOCK)), img_w - w)
        else:  # N/A
            y = int(rng.integers(0, max(1, img_h - h)))
            x = int(rng.integers(0, max(1, img_w - w)))

        last = (y, x)
        # HARD constraint: no overlap with a forbidden region.
        clash = False
        for fx, fy, fw, fh in forbid:
            if not (x + w <= fx or fx + fw <= x or y + h <= fy or fy + fh <= y):
                clash = True
                break
        if clash:
            continue
        # SOFT constraint: prefer content.
        if content is None or min_content_frac <= 0.0:
            return y, x
        frac = _content_frac(content, y, x, h, w)
        if frac >= min_content_frac:
            return y, x
        if best is None or frac > best[0]:
            best = (frac, y, x)
    if best is not None:
        return best[1], best[2]
    return last  # no non-overlapping candidate: last draw


def _hard_mask(img_h, img_w, y, x, h, w) -> np.ndarray:
    m = np.zeros((img_h, img_w), dtype=np.uint8)
    m[y:y + h, x:x + w] = 255
    return m


# ---------------------------------------------------------------- edits
def _plausible_token(rng, size_class) -> str:
    """PLAUSIBLE value written by the substitution, drawn from a LARGE varied
    corpus (FR/EN/date/digits/code/character/sentence) — see `lexicon.py`. The
    content diversity prevents the model from learning the TEXT as a shortcut and
    forces it to rely on the ELA compression signal. Output guaranteed ASCII (no
    '???': cv2.putText/Hershey only renders ASCII; the lexicon transliterates
    e->e, etc.)."""
    return lexicon.plausible_token(rng, size_class)


def _random_ink_color(rng) -> np.ndarray:
    """SATURATED ink color drawn at random (RGB float64, pipeline order). Uniform
    hue on the circle, strong saturation, medium value -> plausible ink/stamp
    (blue, red, green, purple...) and VISIBLE on light paper. The hue changes on
    each call -> color diversity of the forgeries."""
    hue = float(rng.random())
    sat = float(rng.uniform(0.55, 1.0))
    val = float(rng.uniform(0.40, 0.80))
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return np.array([r * 255.0, g * 255.0, b * 255.0], dtype=np.float64)


def _forge_substitution(rng, img, size_class, area_range, feather,
                        min_w_px: int, min_h_px: int, forbid=None,
                        content=None, min_content_frac=0.0,
                        color_prob=0.0) -> ForgeResult:
    """Write a PLAUSIBLE VALUE at the document's text size.

    Realism (isolates the compression signal, avoids the generator "tell"):
    - font matched to a realistic LINE height (not to a large box);
    - content = amount/date/quantity/code in the document format (no gibberish);
    - TIGHT edit: the box (= the mask) hugs the text, no large flat fill.
    + Lever 1: placed on content; + Lever 2: contrasted dark ink.
    """
    H, W = img.shape[:2]
    _check_min_fits(H, W, min_h_px, min_w_px, context="(substitution)")

    # Line height = that of the document's ACTUAL text (anti-tell), slightly
    # modulated (emphasis). Fall back to a page fraction if there is too little
    # measurable text. Independent of any large box -> no more giant gibberish.
    emphasis = float(rng.uniform(*_SUBST_EMPHASIS))
    doc_text_h = _estimate_text_height(img, content)
    if doc_text_h is not None:
        line_px = int(round(doc_text_h * emphasis))
    else:
        line_px = int(round(H * float(rng.uniform(*_LINE_H_FRAC)) * emphasis))
    # Renderable floor (8px); the min_region_px floor is guaranteed by the box
    # below (`h = max(min_h_px, ...)`). Safety cap of H//4.
    line_px = max(8, min(line_px, H // 4))

    # Plausible value + font, scale calibrated to reach `line_px`.
    token = _plausible_token(rng, size_class)
    font = _CV_FONTS[int(rng.integers(0, len(_CV_FONTS)))]
    (_, th1), _ = cv2.getTextSize(token, font, 1.0, 1)
    scale = max(0.3, line_px / max(th1, 1))
    thickness = max(1, int(round(line_px / 14)))
    (tw, th), base = cv2.getTextSize(token, font, scale, thickness)

    pad = max(2, line_px // 6)
    # Reduce the scale if the text overflows the image width.
    if tw + 2 * pad > W - 2:
        scale = max(0.3, scale * (W - 2) / float(tw + 2 * pad))
        (tw, th), base = cv2.getTextSize(token, font, scale, thickness)

    # TIGHT box around the text (= mask). Respects the min floor and the image.
    h = max(min_h_px, min(th + base + 2 * pad, H - 1))
    w = max(min_w_px, min(tw + 2 * pad, W - 1))

    y, x = _place_dest(rng, H, W, h, w, "N/A", forbid=forbid,
                       content=content, min_content_frac=min_content_frac)

    region = img[y:y + h, x:x + w]
    # Background sampled locally (median) -> photometric consistency.
    bg = np.median(region.reshape(-1, 3), axis=0).astype(np.float64)
    # Ink = local dark pixels (the ink is darker than the background).
    gray = region.reshape(-1, 3).mean(axis=1)
    dark_idx = gray <= np.percentile(gray, 15)
    ink = (region.reshape(-1, 3)[dark_idx].mean(axis=0).astype(np.float64)
           if dark_idx.any() else bg.copy())
    # Lever 2: guarantee a contrasted DARK ink if the region is light/flat.
    bg_lum, ink_lum = float(bg.mean()), float(ink.mean())
    if bg_lum > 100.0 and (bg_lum - ink_lum) < _MIN_INK_CONTRAST:
        ink = np.clip(bg * (_INK_TARGET_LUM / max(bg_lum, 1e-6)), 0, 255)

    # COLORED ink (probability color_prob): saturated color drawn at random,
    # DIFFERENT for each substitution, so the model learns the forgery whatever
    # its hue (not only black text). We keep a minimum luminance contrast with
    # the background (real edges -> real compression signal).
    ink_colored = bool(color_prob) and (float(rng.random()) < float(color_prob))
    if ink_colored:
        ink = _random_ink_color(rng)
        ink_lum = float(ink.mean())
        if bg_lum > 100.0 and (bg_lum - ink_lum) < _MIN_INK_CONTRAST:
            ink = np.clip(ink * (max(0.0, bg_lum - _MIN_INK_CONTRAST) / max(ink_lum, 1e-6)),
                          0, 255)

    patch = np.empty((h, w, 3), dtype=np.uint8)
    patch[:] = bg.astype(np.uint8)
    org = (pad, pad + th)                       # putText origin (bottom-left)
    cv2.putText(patch, token, org, font, scale,
                tuple(float(c) for c in ink), thickness, cv2.LINE_AA)

    edited = _composite(img, patch, y, x, h, w, feather)
    mask = _hard_mask(H, W, y, x, h, w)
    frac = (h * w) / (H * W)
    return ForgeResult(
        image=edited, mask=mask, edit_type="substitution", size_class=size_class,
        alignment="N/A", bbox=[x, y, w, h], area_frac=round(frac, 6),
        feather_radius=round(feather, 3),
        extra={"text": token, "ink_colored": ink_colored,
               "ink_rgb": [int(round(c)) for c in ink]},
    )


def _forge_copy_move(rng, img, size_class, area_range, alignment, feather,
                     min_w_px: int, min_h_px: int, forbid=None,
                     content=None, min_content_frac=0.0) -> ForgeResult:
    """Copy a region of the SAME image (carries the Q0 grid).

    Lever 1: SOURCE and DESTINATION prefer real content -> we copy a real ink
    region and paste it onto a region that carries content.
    """
    H, W = img.shape[:2]
    h, w, frac = _sample_size(rng, H, W, area_range, min_w_px, min_h_px)
    # Source snapped to grid 8 -> we copy whole Q0 blocks, clean phase.
    sy, sx = _pick_source(rng, H, W, h, w, content=content,
                          min_content_frac=min_content_frac)
    patch = img[sy:sy + h, sx:sx + w].copy()

    # Avoid the source region AND the regions already forged.
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
    """Insert a region from ANOTHER document (carries the donor's Q0' grid).

    Lever 1: the SOURCE region (in the donor) and the DESTINATION prefer
    content -> we insert foreign ink onto a region carrying content.
    """
    H, W = img.shape[:2]
    Hd, Wd = donor.shape[:2]
    # Size limited by the smaller of the two images (dest AND donor).
    h, w, frac = _sample_size(rng, min(H, Hd), min(W, Wd), area_range, min_w_px, min_h_px)
    # Strict reclip against the donor alone (may be smaller than dest):
    # same guarantee as _sample_size, never a silent 0px rectangle.
    usable_hd, usable_wd = _check_min_fits(Hd, Wd, min_h_px, min_w_px, context="(donor)")
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
    color_prob: float = 0.0,
) -> ForgeResult:
    """Entry point: apply ONE forgery and return image+mask+metadata.

    `alignment` is ignored for substitution (forced to "N/A"). `donor` is
    required for splice. `min_region_px` = GUARANTEED minimum size of the forged
    rectangle: an integer (square) or `[min_width, min_height]` in pixels,
    rounded UP to the next multiple of 8 (JPEG grid). If the source image is too
    small to fit it, an explicit error is raised (the job is then logged as an
    error by the orchestrator, never written as an empty-mask positive). `forbid`
    = list of already-forged bboxes [x,y,w,h] not to overlap (multi-forgery:
    call `forge` k times accumulating the returned bboxes). `on_content`
    (Lever 1): place the forgery on real content (ink) rather than in empty
    space, with `min_content_frac` = minimum ink fraction targeted.
    """
    feather = float(rng.uniform(feather_range[0], feather_range[1]))
    min_w_px, min_h_px = normalize_min_region(min_region_px)

    # Lever 1: content masks computed once (best-effort at placement).
    content = _content_mask(img) if on_content else None
    mcf = float(min_content_frac) if on_content else 0.0

    if edit_type == "substitution":
        result = _forge_substitution(rng, img, size_class, area_range, feather,
                                     min_w_px, min_h_px, forbid,
                                     content=content, min_content_frac=mcf,
                                     color_prob=color_prob)
    elif edit_type == "copy_move":
        result = _forge_copy_move(rng, img, size_class, area_range, alignment, feather,
                                  min_w_px, min_h_px, forbid,
                                  content=content, min_content_frac=mcf)
    elif edit_type == "splice":
        if donor is None:
            raise ValueError("splice requires a donor image (donor).")
        donor_content = _content_mask(donor) if on_content else None
        result = _forge_splice(rng, img, donor, donor_id, size_class, area_range,
                               alignment, feather, min_w_px, min_h_px, forbid,
                               content=content, min_content_frac=mcf,
                               donor_content=donor_content)
    else:
        raise ValueError(f"unknown edit type: {edit_type}")

    # Safety net: a positive must NEVER come out with an empty mask.
    if not result.mask.any():
        raise RuntimeError(
            f"empty mask produced for edit_type={edit_type} (bbox={result.bbox}) "
            "despite the min_region_px floor: report this case, it should no "
            "longer be possible.")
    return result
