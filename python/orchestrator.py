"""orchestrator — Module 5 du pipeline.

Rôle
----
Génération par lot, 100% scriptable, sans intervention manuelle :
    1. sonde le corpus source (jpeg_probe) et écrit distribution.json,
    2. planifie N jobs déterministes (seed par document dérivé du seed global),
       en tirant type d'édition / taille / alignement / Q2 / négatif selon la config,
    3. exécute (en parallèle) forger -> recompress (passe Q2 unique) -> annotator,
    4. écrit <id>.jpg, <id>_mask.png, <id>.json + manifest.parquet global.

Déterminisme + parallélisme
---------------------------
Tous les paramètres de chaque job (dont le seed) sont tirés dans le processus
principal via un RNG maître. Le worker n'utilise que le seed du job pour ses
choix internes (positions, feather, texte). Le résultat est donc indépendant de
l'ordre d'exécution des workers.

Sous-échantillonnage des types / tailles / alignement
-----------------------------------------------------
- Négatifs : proportion `negatives.ratio` (authentiques Q0->Q2, masque vide).
  Ils empêchent le modèle d'apprendre la double compression GLOBALE au lieu de
  LOCALISER l'incohérence. Les éléments bénins colorés (logos, tampons) sont
  conservés par construction : on part de vrais reçus, on ne les retire jamais.
- Qualité : DEUX passes Q1 < Q2 par document. Q2 (sauvegarde finale, haute) est
  stratifiée sur le sweep (i % len) ; Q1 = Q2 - q1_gap (base plus basse). La source
  est recompressée à Q1 (historique du "document original"), la substitution est
  peinte en pixels NEUFS (jamais vus par Q1), puis sauvegarde finale à Q2 -> le
  texte authentique porte l'historique Q1->Q2, la zone falsifiée n'a que Q2. En ELA
  (sondée à une qualité != Q2), la zone ressort du texte ordinaire (~2.6x). L'écart
  Q1<Q2 est IMPÉRATIF : en Q1==Q2 la falsification est indiscernable (ratio ~1.0).
- Types / tailles : tirés selon les ratios de la config (tailles équiprobables).
- Alignement : sur copy_move + splice uniquement, selon `aligned_ratio`.

Dépendances : Pillow, NumPy, OpenCV, PyYAML, PyArrow.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import cv2
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import jpeg_probe
from recompress import decode_source, save_q2, recompress_to_q1, DEFAULT_Q2_SUBSAMPLING
import forger as forger_mod
import annotator as ann


_SUBSAMPLING_INT2STR = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


# ------------------------------------------------------------------ ELA + CSV
def ela_qualities(center: int, spread: int) -> list[int]:
    """3 qualités de sonde ELA (canaux R,G,B) = (centre-spread, centre, centre+spread).

    Le centre vise ≈ Q1 (point fixe du fond) pour le contraste max ; l'écart `spread`
    donne 3 vues décorrélées -> une IMAGE COULEUR dont la teinte encode la réaction
    différentielle aux 3 sondes (le signal vient de la DIVERSITÉ de qualité, pas de
    la chroma). Bornées dans [40, 99]. Alimente `detection_eval` mode E2.
    """
    return [int(np.clip(center + d, 40, 99)) for d in (-spread, 0, spread)]


def compute_ela_stack(img_path: str, qualities: list[int], scale: float) -> np.ndarray:
    """Pile ELA 3 qualités -> (H, W, 3) uint8 RGB, sur le JPEG FINAL re-lu.

    Canal k = |image - recompress(image, qualities[k])| moyennée sur canaux, à
    échelle GLOBALE fixe (`scale`), résolution NATIVE (aligné pixel avec image/masque).
    L'ordre des canaux (R,G,B) = (q_bas, q_moyen, q_haut). Encodeur PIL (cohérent
    avec `ela_scan` / `detection_eval`).
    """
    rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.int16)
    chans = []
    for q in qualities:
        buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8), "RGB").save(buf, "JPEG", quality=int(q))
        buf.seek(0)
        rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
        diff = np.abs(rgb - rec).mean(axis=2)          # ELA luminance à cette qualité
        chans.append(np.clip(diff * float(scale), 0, 255).astype(np.uint8))
    return np.stack(chans, axis=2)                      # (H, W, 3) RGB


def _bn(rel: Optional[str]) -> str:
    """Nom de fichier depuis un chemin relatif (vide si None)."""
    return os.path.basename(rel) if rel else ""


def write_folder_csvs(sub_root: str, rows: list[dict]) -> None:
    """Écrit un CSV par dossier (images / masks / ela) pour faciliter la reprise.

    Chaque CSV est autonome (une ligne par document, nom de fichier + métadonnées
    utiles) : on peut charger un dossier sans lire le manifeste Parquet global.
    """
    if not rows:
        return
    with open(os.path.join(sub_root, "images", "images.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "type", "is_negative", "quality",
                    "size_class", "n_forgeries", "source_id", "seed"])
        for r in rows:
            w.writerow([r["id"], _bn(r["path_img"]), r["type"], r["is_negative"],
                        r["q2"], r["size_class"], r["n_forgeries"], r["source_id"], r["seed"]])
    with open(os.path.join(sub_root, "masks", "masks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "mask", "json", "is_negative", "n_forgeries", "n_mask_px",
                    "mask_frac", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        for r in rows:
            w.writerow([r["id"], _bn(r["path_mask"]), _bn(r["path_json"]), r["is_negative"],
                        r["n_forgeries"], r["n_mask_px"], r["mask_frac"],
                        r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]])
    with open(os.path.join(sub_root, "ela", "ela.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ela", "image", "ela_qualities", "ela_scale", "is_negative", "type"])
        for r in rows:
            w.writerow([r["id"], _bn(r.get("path_ela")), _bn(r["path_img"]),
                        r.get("ela_qualities", r.get("ela_quality", "")), r.get("ela_scale", ""),
                        r["is_negative"], r["type"]])


# ------------------------------------------------------------------ job spec
@dataclass
class Job:
    doc_id: str
    seed: int
    q2: int
    is_negative: bool
    source_path: str
    edit_type: Optional[str]        # None si négatif
    size_class: Optional[str]
    alignment: Optional[str]
    donor_path: Optional[str]       # splice uniquement
    q1: Optional[int] = None        # None = mode natif (Q0 lu) ; sinon Q1 contrôlé
    n_forgeries: int = 1            # nb de falsifications (même type) sur ce doc positif


# ------------------------------------------------------------------ planning
KNOWN_EDIT_TYPES = ("substitution", "copy_move", "splice")


def resolve_edit_types(cfg: dict) -> list[str]:
    """Liste ordonnée et dédupliquée des types à générer (un sous-dossier chacun).

    Source : `forger.edit_types`. Rétro-compat : si absent, dérive de l'ancien
    `forger.edit_type_ratios` (types à poids > 0). Il n'y a PLUS aucun tirage
    aléatoire entre types : chaque type est un lot complet et séparé.
    """
    fc = cfg.get("forger", {})
    types = fc.get("edit_types")
    if not types:
        ratios = fc.get("edit_type_ratios", {}) or {}
        types = [t for t, w in ratios.items() if w and float(w) > 0]
    if not types:
        raise ValueError(
            "forger.edit_types est vide : liste au moins un type parmi "
            f"{list(KNOWN_EDIT_TYPES)}.")
    unknown = [t for t in types if t not in KNOWN_EDIT_TYPES]
    if unknown:
        raise ValueError(
            f"forger.edit_types contient des types inconnus {unknown} ; "
            f"connus : {list(KNOWN_EDIT_TYPES)}.")
    seen, ordered = set(), []
    for t in types:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def plan_jobs(cfg: dict, source_paths: list[str], edit_type: str,
              doc_prefix: str, type_index: int = 0) -> list[Job]:
    """Jobs déterministes pour UN SEUL type d'édition (un sous-dossier).

    Chaque type reçoit un flux aléatoire décorrélé (seed `[seed, type_index]`) :
    négatifs et sources tirés diffèrent d'un type à l'autre -> aucun doublon exact
    entre sous-dossiers lors de l'agrégation. Les doc_id sont préfixés par
    `doc_prefix` -> uniques globalement -> agrégation sans collision.
    """
    master = np.random.default_rng([int(cfg["orchestrator"]["seed"]), int(type_index)])
    n_docs = int(cfg["orchestrator"]["n_docs"])
    # DEUX qualités par document : Q2 (sauvegarde FINALE, tirée du sweep, haute) et
    # Q1 = Q2 - q1_gap (base de compression du "document original", plus basse).
    # C'est l'ÉCART Q1<Q2 qui rend la substitution détectable : le texte authentique
    # porte l'historique Q1->Q2, la zone peinte en pixels NEUFS n'a que Q2 -> en ELA
    # (sondée à une qualité != Q2) elle ressort du texte ordinaire (~2.6x). En Q1==Q2
    # la falsification serait indiscernable du texte authentique (mesuré : ratio ~1.0).
    quality_sweep = [int(q) for q in cfg["compression"]["quality_sweep"]]
    if not quality_sweep:
        raise ValueError(
            "compression.quality_sweep est vide : liste au moins une qualité JPEG.")
    q1_gap = int(cfg["compression"].get("q1_gap", 0))
    if q1_gap <= 0:
        raise ValueError(
            "compression.q1_gap doit être > 0 : sans écart Q1<Q2, la substitution "
            "est indiscernable du texte authentique en ELA (aucun signal exploitable).")
    neg_ratio = float(cfg["negatives"]["ratio"])
    aligned_ratio = float(cfg["forger"]["aligned_ratio"])
    # size_classes ORDONNÉES du plus petit au plus grand (ordre du config).
    size_classes = list(cfg["size_classes"].keys())
    # Nombre de falsifications par doc positif : k ~ U{n_min..n_max} (même type).
    n_min, n_max = (list(cfg["forger"].get("n_forgeries", [1, 1])) + [1])[:2]
    n_min, n_max = int(n_min), int(max(int(n_min), int(n_max)))

    jobs: list[Job] = []
    for i in range(n_docs):
        # Seed déterministe par document (indépendant du planning des workers).
        seed = int(master.integers(0, 2**31 - 1))
        jrng = np.random.default_rng(seed)

        # Q2 STRATIFIÉE sur le sweep (i % len) -> couverture déterministe et
        # équilibrée. Q1 = Q2 - gap (borné >= 40 : reste une JPEG valide). Le MÊME
        # couple (Q1, Q2) est appliqué aux positifs COMME aux négatifs (aucune fuite
        # qualité globale -> le modèle doit LOCALISER l'écart, pas le lire au niveau page).
        q2 = int(quality_sweep[i % len(quality_sweep)])
        q1 = max(40, q2 - q1_gap)
        source_path = str(master.choice(source_paths))
        is_negative = bool(master.random() < neg_ratio)
        # Nommage HONNÊTE : un négatif n'est PAS une falsification du type courant.
        # Il porte le marqueur `authentic` (mais garde le préfixe du dossier pour
        # rester unique à l'agrégation) -> plus de "substitution_XXXX" à masque vide.
        doc_id = (f"{doc_prefix}_authentic_{i:06d}" if is_negative
                  else f"{doc_prefix}_{i:06d}")

        if is_negative:
            jobs.append(Job(
                doc_id=doc_id, seed=seed, q2=q2, is_negative=True,
                source_path=source_path, edit_type=None, size_class=None,
                alignment=None, donor_path=None, q1=q1,
            ))
            continue

        # Nombre de falsifications sur ce doc, puis PLAFOND de taille selon k :
        # plus il y en a, plus elles sont petites (évite de couvrir toute la page).
        #   k=1 -> toutes tailles ... k>=len(classes) -> small only.
        k = int(jrng.integers(n_min, n_max + 1))
        max_idx = min(len(size_classes) - 1, max(0, len(size_classes) - k))
        size_class = str(size_classes[int(jrng.integers(0, max_idx + 1))])

        if edit_type == "substitution":
            alignment = "N/A"
            donor_path = None
        else:
            alignment = "aligned" if jrng.random() < aligned_ratio else "misaligned"
            donor_path = None
            if edit_type == "splice":
                # Splice intra-corpus : autre document du lot.
                others = [p for p in source_paths if p != source_path] or source_paths
                donor_path = str(master.choice(others))

        jobs.append(Job(
            doc_id=doc_id, seed=seed, q2=q2, is_negative=False,
            source_path=source_path, edit_type=edit_type, size_class=size_class,
            alignment=alignment, donor_path=donor_path, q1=q1, n_forgeries=k,
        ))
    return jobs


# ------------------------------------------------------------------ worker
def _run_job(job: Job, cfg: dict, out_dirs: dict, nonstd_thr: float) -> dict:
    """Exécute un job complet : forge -> passe Q2 unique -> annotation -> écriture."""
    rng = np.random.default_rng(job.seed)
    allow_lossless = bool(cfg["probe"].get("allow_lossless", False))
    src = decode_source(job.source_path, nonstd_thr, allow_lossless=allow_lossless)

    q2_sub_str = _SUBSAMPLING_INT2STR.get(DEFAULT_Q2_SUBSAMPLING, "4:2:0")
    ann_cfg = cfg["annotator"]

    # Base de travail : source TOUJOURS recompressée à q -> établit l'historique
    # de compression du "document original" (fond double-compressé après la passe
    # finale à q). Q0 reste lu/journalisé mais n'est plus l'historique effectif.
    base_rgb = recompress_to_q1(src.rgb, job.q1, subsampling=DEFAULT_Q2_SUBSAMPLING)
    q1_effective = job.q1

    forgery_bboxes: list = []
    if job.is_negative:
        edited = base_rgb
        mask = np.zeros((src.height, src.width), dtype=np.uint8)
        edit_type, size_class, alignment, bbox, forge_res = (
            "authentic", "none", "N/A", None, None)
    else:
        area_range = tuple(cfg["size_classes"][job.size_class])
        feather_range = tuple(cfg["forger"]["feather_radius_px"])
        donor_rgb, donor_id, donor_q = None, None, None
        if job.edit_type == "splice":
            # Le splice insère une région à historique de compression ÉTRANGER.
            donor = decode_source(job.donor_path, nonstd_thr, allow_lossless=allow_lossless)
            donor_rgb, donor_id = donor.rgb, donor.source_id
            if donor.q0 == -1:
                # Donneur lossless (PNG) : sans ça la région n'aurait AUCUN historique
                # JPEG et se confondrait avec une substitution. On lui impose une
                # qualité étrangère Q_donor (tirée dans quality_sweep) -> vraie grille étrangère.
                q_choices = [int(q) for q in cfg["compression"].get("quality_sweep", [job.q2])]
                donor_q = int(rng.choice(q_choices or [job.q2]))
                donor_rgb = recompress_to_q1(donor_rgb, donor_q, subsampling=DEFAULT_Q2_SUBSAMPLING)
            # (donneur JPEG : historique natif Q0 conservé, déjà étranger au fond.)
        # int (carré) ou [largeur_min, hauteur_min] ; normalisé dans forger.forge().
        min_region_px = cfg["forger"].get("min_region_px", forger_mod.JPEG_BLOCK)
        # Lever 1 : placement sur du contenu réel (encre) plutôt que dans le vide.
        on_content = bool(cfg["forger"].get("place_on_content", False))
        min_content_frac = float(cfg["forger"].get("min_content_frac", 0.0))

        # ---- k falsifications (MÊME type, MÊME classe de taille) accumulées ----
        # Chaque passe édite l'image courante et évite les zones déjà falsifiées
        # (forbid). Le masque final est l'UNION des k empreintes.
        edited = base_rgb
        mask = np.zeros((src.height, src.width), dtype=np.uint8)
        forge_res = None
        for _ in range(max(1, job.n_forgeries)):
            forge_res = forger_mod.forge(
                img=edited, edit_type=job.edit_type, size_class=job.size_class,
                area_range=area_range, alignment=job.alignment,
                feather_range=feather_range, rng=rng,
                donor=donor_rgb, donor_id=donor_id,
                min_region_px=min_region_px, forbid=forgery_bboxes,
                on_content=on_content, min_content_frac=min_content_frac,
            )
            edited = forge_res.image
            mask = np.maximum(mask, forge_res.mask)
            forgery_bboxes.append(forge_res.bbox)      # évite le chevauchement suivant
        if donor_q is not None:
            forge_res.extra["donor_q"] = donor_q   # qualité JPEG étrangère du splice
        edit_type, size_class, alignment = job.edit_type, job.size_class, job.alignment
        bbox = ann.bbox_from_mask(mask)            # bbox englobante de l'UNION

    # ---- Passe Q2 UNIQUE sur l'image composite entière (règle impérative) ----
    #      -> dossier images/
    img_path = os.path.join(out_dirs["images"], f"{job.doc_id}.jpg")
    save_q2(edited, img_path, job.q2, subsampling=DEFAULT_Q2_SUBSAMPLING)

    # ---- Masque pixel-level exact -> dossier masks/ ----
    mask_path = os.path.join(out_dirs["masks"], f"{job.doc_id}_mask.png")
    cv2.imwrite(mask_path, mask)

    # ---- ELA (3 qualités -> RGB) sur le JPEG FINAL re-lu -> dossier ela/ ----
    ela_cfg = cfg.get("ela", cfg.get("ela_preview", {}))
    ela_center = int(ela_cfg.get("ela_quality", 90))
    ela_spread = int(ela_cfg.get("ela_spread", 8))
    ela_qs = ela_qualities(ela_center, ela_spread)
    ela_scale = float(ela_cfg.get("ela_scale", 15.0))
    ela_rgb = compute_ela_stack(img_path, ela_qs, ela_scale)
    ela_path = os.path.join(out_dirs["ela"], f"{job.doc_id}_ela.png")
    Image.fromarray(ela_rgb, "RGB").save(ela_path)      # RGB : canaux = (q_bas, q_moyen, q_haut)

    # ---- Grille patch 24x24 ----
    labels, fracs = ann.patch_grid_labels(
        mask, input_res=ann_cfg.get("input_res", 384),
        patch_size=ann_cfg["patch_size"], grid=ann_cfg["patch_grid"],
        overlap_thr=ann_cfg["patch_positive_overlap"],
    )

    # ---- Métadonnées JSON ----
    meta = ann.build_metadata(
        doc_id=job.doc_id, source=src, q2=job.q2, edit_type=edit_type,
        size_class=size_class, alignment=alignment, bbox=bbox, seed=job.seed,
        forge_result=forge_res, patch_size=ann_cfg["patch_size"],
        grid=ann_cfg["patch_grid"], overlap_thr=ann_cfg["patch_positive_overlap"],
        q2_subsampling=q2_sub_str,
    )
    meta["Q1_mode"] = "history_gap"            # Q1 < Q2 (écart = signal ELA)
    meta["Q1_effective"] = int(q1_effective)   # = Q1 (historique de compression du fond)
    # Multi-falsification : nb de zones + bbox de CHACUNE (le champ `bbox` est
    # l'englobante de l'union, coarse ; ici la vérité région par région).
    meta["n_forgeries"] = 0 if job.is_negative else int(job.n_forgeries)
    meta["forgery_bboxes"] = forgery_bboxes
    meta["patch_grid"] = labels.tolist()
    meta["ela"] = {"qualities": ela_qs, "center": ela_center, "spread": ela_spread,
                   "scale": ela_scale, "channels": "RGB = (q_bas, q_moyen, q_haut)",
                   "file": os.path.basename(ela_path)}
    meta["files"] = {"image": os.path.basename(img_path),
                     "mask": os.path.basename(mask_path),
                     "ela": os.path.basename(ela_path)}
    # Le .json accompagne le masque -> dossier masks/.
    json_path = os.path.join(out_dirs["masks"], f"{job.doc_id}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Ligne de manifeste ----
    n_mask_px = int((mask > 0).sum())
    return {
        "id": job.doc_id,
        "source_id": src.source_id,
        "q0": src.q0,
        "q0_nonstandard": src.nonstandard,
        "q1_mode": "history_gap",
        "q1_effective": int(q1_effective),
        "q2": job.q2,
        "type": edit_type,
        "size_class": size_class,
        "alignment": alignment,
        "is_negative": job.is_negative,
        "n_forgeries": 0 if job.is_negative else int(job.n_forgeries),
        "bbox_x": bbox[0] if bbox else -1,
        "bbox_y": bbox[1] if bbox else -1,
        "bbox_w": bbox[2] if bbox else -1,
        "bbox_h": bbox[3] if bbox else -1,
        "n_mask_px": n_mask_px,
        "mask_frac": round(n_mask_px / (src.width * src.height), 6),
        "n_pos_patches": int((labels > 0).sum()),
        "subsampling_src": src.subsampling,
        "seed": job.seed,
        "ela_quality": ela_center,
        "ela_qualities": ",".join(str(q) for q in ela_qs),
        "ela_scale": ela_scale,
        "path_img": os.path.relpath(img_path, out_dirs["root"]),
        "path_mask": os.path.relpath(mask_path, out_dirs["root"]),
        "path_json": os.path.relpath(json_path, out_dirs["root"]),
        "path_ela": os.path.relpath(ela_path, out_dirs["root"]),
    }


# module-level pour picklabilité (ProcessPoolExecutor).
def _worker(args):
    job, cfg, out_dirs, nonstd_thr = args
    try:
        return _run_job(job, cfg, out_dirs, nonstd_thr)
    except Exception as exc:  # on isole les échecs par document
        return {"id": job.doc_id, "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------ driver
def _execute_jobs(jobs: list[Job], cfg: dict, out_dirs: dict,
                  nonstd_thr: float, n_workers: int) -> tuple[list, list]:
    """Exécute une liste de jobs (séquentiel ou parallèle) -> (rows, errors)."""
    rows, errors = [], []
    payloads = [(j, cfg, out_dirs, nonstd_thr) for j in jobs]
    if n_workers <= 1:
        for p in payloads:
            r = _worker(p)
            (errors if "error" in r else rows).append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for r in ex.map(_worker, payloads, chunksize=4):
                (errors if "error" in r else rows).append(r)
    return rows, errors


def _write_batch(sub_root: str, sub_cfg: dict, report: dict, rows: list) -> tuple:
    """Rend un sous-dossier AUTONOME : distribution + config + manifeste + rapport."""
    with open(os.path.join(sub_root, "distribution.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(sub_root, "run_config.yaml"), "w") as f:
        yaml.safe_dump(sub_cfg, f, sort_keys=False, allow_unicode=True)
    manifest_path = os.path.join(sub_root, "manifest.parquet")
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), manifest_path)
        write_folder_csvs(sub_root, rows)     # un CSV par dossier (images/masks/ela)
    report_path = None
    try:
        import reporter
        report_path = reporter.write_report(sub_root)
    except Exception as exc:
        print(f"      (rapport non généré : {type(exc).__name__}: {exc})")
    return manifest_path, report_path


def run(cfg: dict, limit: Optional[int] = None, workers: Optional[int] = None) -> dict:
    src_dir = cfg["paths"]["source_dir"]
    if not src_dir:
        raise ValueError("paths.source_dir est vide : renseigne le dossier des JPEG sources.")
    out_root = cfg["paths"]["output_dir"]
    os.makedirs(out_root, exist_ok=True)

    nonstd_thr = float(cfg["probe"]["nonstandard_absdiff_threshold"])
    if limit is not None:  # n_docs PAR TYPE (chaque sous-dossier aura `limit` docs)
        cfg = {**cfg, "orchestrator": {**cfg["orchestrator"], "n_docs": limit}}

    # 1) Sonde du corpus UNE SEULE FOIS (partagée par tous les types).
    allow_lossless = bool(cfg["probe"].get("allow_lossless", True))
    print(f"[1/4] probe sur {src_dir} ...")
    report = jpeg_probe.probe_dir(
        src_dir, recursive=cfg["probe"]["recursive"],
        candidate_ext=tuple(cfg["probe"]["candidate_ext"]),
        nonstandard_threshold=nonstd_thr, allow_lossless=allow_lossless,
    )
    source_paths = [r["path"] for r in report["records"]]
    if not source_paths:
        raise RuntimeError(
            f"Aucune source image valide dans {src_dir} "
            f"(exts={cfg['probe']['candidate_ext']}, allow_lossless={allow_lossless}).")
    n_lossless = report["summary"].get("n_lossless_kept", 0)
    print(f"      {len(source_paths)} sources gardées "
          f"(dont {n_lossless} lossless), {report['summary']['n_excluded']} exclues.")

    # Deux passes Q1<Q2 : source recompressée à Q1 (historique du "document
    # original"), puis sauvegarde finale à Q2 (haute). Les PNG lossless sont gérés
    # nativement — l'historique vient entièrement de la passe Q1.
    sweep = [int(q) for q in cfg["compression"]["quality_sweep"]]
    gap = int(cfg["compression"].get("q1_gap", 0))
    ela_q = int(cfg.get("ela", cfg.get("ela_preview", {})).get("ela_quality", 90))
    q1_vals = sorted({max(40, q - gap) for q in sweep})
    q1_reco = int(round(float(np.median(q1_vals))))   # sonde optimale ≈ Q1 (point fixe du fond)
    # GARDE-FOU DUR : si une Q2 du sweep == qualité de sonde ELA, toute l'image est au
    # point fixe de la sonde et l'ELA s'effondre à 0 (aucune séparabilité).
    if ela_q in sweep:
        raise ValueError(
            f"ELA_QUALITY={ela_q} coïncide avec une valeur de QUALITY_SWEEP ({sweep}) : "
            "l'ELA s'effondrerait à 0 (image au point fixe de la sonde). Choisis "
            f"ELA_QUALITY ≈ Q1 (recommandé : {q1_reco}).")
    # ALERTE SOUPLE : la sonde doit viser Q1 (fond au point fixe -> contraste max).
    # Trop loin de Q1 => signal faible (on l'a mesuré : @90 donne ~1.8 vs @Q1 ~3.2).
    if abs(ela_q - q1_reco) > 8:
        print(f"      ⚠️  ELA_QUALITY={ela_q} est loin de Q1≈{q1_reco} "
              f"(Q1∈{q1_vals}) : contraste ELA sous-optimal. Recommandé : ELA_QUALITY={q1_reco}.")
    print(f"      compression : Q2 (sweep) = {sweep}, Q1 = Q2-{gap} = {q1_vals}, "
          f"sonde ELA = {ela_q} (optimum ≈ Q1 = {q1_reco})")

    # Snapshot corpus-level à la racine (référence commune).
    with open(os.path.join(out_root, "distribution.json"), "w") as f:
        json.dump(report, f, indent=2)

    # 2) Un lot COMPLET ET SÉPARÉ par type d'édition -> un sous-dossier chacun.
    edit_types = resolve_edit_types(cfg)
    n_workers = workers if workers is not None else int(cfg["orchestrator"]["n_workers"])
    n_per = int(cfg["orchestrator"]["n_docs"])
    print(f"[2/4] types : {edit_types}  (un sous-dossier chacun, {n_per} docs/type, "
          f"seed global {cfg['orchestrator']['seed']})")

    results = {}
    for ti, etype in enumerate(edit_types):
        sub_root = os.path.join(out_root, etype)
        sub_dirs = {"root": sub_root,
                    "images": os.path.join(sub_root, "images"),
                    "masks": os.path.join(sub_root, "masks"),
                    "ela": os.path.join(sub_root, "ela")}
        for _k in ("images", "masks", "ela"):
            os.makedirs(sub_dirs[_k], exist_ok=True)
        # Config effective du sous-dossier : self-describing (edit_type scalaire).
        sub_cfg = {
            **cfg,
            "paths": {**cfg["paths"], "output_dir": sub_root},
            "forger": {**cfg["forger"], "edit_types": [etype], "edit_type": etype},
        }
        jobs = plan_jobs(sub_cfg, source_paths, etype, doc_prefix=etype, type_index=ti)
        print(f"[3/4] [{etype}] {len(jobs)} jobs -> {n_workers} worker(s) ...")
        rows, errors = _execute_jobs(jobs, sub_cfg, sub_dirs, nonstd_thr, n_workers)
        print(f"      [{etype}] {len(rows)} écrits, {len(errors)} erreurs.")
        for e in errors[:5]:
            print(f"        ERREUR {e['id']}: {e['error']}")
        manifest_path, report_path = _write_batch(sub_root, sub_cfg, report, rows)
        _print_batch_summary(rows)
        results[etype] = {"root": sub_root, "rows": rows, "errors": errors,
                          "manifest": manifest_path, "report": report_path}

    print(f"[4/4] {len(edit_types)} sous-dossier(s) sous {out_root}/ : "
          f"{', '.join(edit_types)}")
    print(f"      Agréger : ./scripts/aggregate.sh")
    return {"output_dir": out_root, "edit_types": edit_types,
            "distribution": os.path.join(out_root, "distribution.json"),
            "batches": results}


def _print_batch_summary(rows: list[dict]) -> None:
    if not rows:
        return
    from collections import Counter
    print("=" * 64)
    print(f"  Documents      : {len(rows)}")
    print(f"  Négatifs       : {sum(r['is_negative'] for r in rows)}")
    print(f"  Types          : {dict(Counter(r['type'] for r in rows))}")
    print(f"  Alignement     : {dict(Counter(r['alignment'] for r in rows))}")
    print(f"  Tailles        : {dict(Counter(r['size_class'] for r in rows))}")
    pos = [r for r in rows if not r["is_negative"]]
    if pos:
        print(f"  Falsif./doc    : {dict(sorted(Counter(r['n_forgeries'] for r in pos).items()))}")
    print(f"  Q2             : {dict(Counter(r['q2'] for r in rows))}")
    print("=" * 64)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="orchestrator — génération batch de falsifications synthétiques.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--src", default=None, help="Écrase paths.source_dir.")
    ap.add_argument("--out", default=None, help="Écrase paths.output_dir.")
    ap.add_argument("--n", type=int, default=None, help="Écrase orchestrator.n_docs (utile pour un smoke test).")
    ap.add_argument("--workers", type=int, default=None, help="Écrase orchestrator.n_workers.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.src:
        cfg["paths"]["source_dir"] = args.src
    if args.out:
        cfg["paths"]["output_dir"] = args.out
    run(cfg, limit=args.n, workers=args.workers)


if __name__ == "__main__":
    main()
