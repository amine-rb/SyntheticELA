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
- Q2 : STRATIFIÉ sur le sweep (i % len(sweep)) pour garantir la couverture de
  Q2<Q0 / Q2≈Q0 / Q2>Q0 nécessaire à l'ablation E5.
- Types / tailles : tirés selon les ratios de la config (tailles équiprobables).
- Alignement : sur copy_move + splice uniquement, selon `aligned_ratio`.

Dépendances : Pillow, NumPy, OpenCV, PyYAML, PyArrow.
"""

from __future__ import annotations

import argparse
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

import jpeg_probe
from recompress import decode_source, save_q2, recompress_to_q1, DEFAULT_Q2_SUBSAMPLING
import forger as forger_mod
import annotator as ann


_SUBSAMPLING_INT2STR = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


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
    q2_sweep = cfg["compression"]["q2_sweep"]
    # Mode Q1 : "native" (Q0 lu, scénario réaliste, défaut) ou "controlled"
    # (Q1 imposé, échantillonné dans q1_sweep -> alimente les régimes de E5).
    q1_mode = cfg["compression"].get("q1_mode", "native")
    q1_sweep = cfg["compression"].get("q1_sweep", [])
    neg_ratio = float(cfg["negatives"]["ratio"])
    aligned_ratio = float(cfg["forger"]["aligned_ratio"])
    size_classes = list(cfg["size_classes"].keys())

    jobs: list[Job] = []
    for i in range(n_docs):
        # Seed déterministe par document (indépendant du planning des workers).
        seed = int(master.integers(0, 2**31 - 1))
        jrng = np.random.default_rng(seed)

        q2 = int(q2_sweep[i % len(q2_sweep)])          # stratifié -> couverture
        # Q1 contrôlé : grille cartésienne Q1×Q2 (couvre Q1<Q2, Q1=Q2, Q1>Q2 pour E5).
        q1 = None
        if q1_mode == "controlled" and q1_sweep:
            q1 = int(q1_sweep[(i // len(q2_sweep)) % len(q1_sweep)])
        source_path = str(master.choice(source_paths))
        is_negative = bool(master.random() < neg_ratio)
        doc_id = f"{doc_prefix}_{i:06d}"

        if is_negative:
            jobs.append(Job(
                doc_id=doc_id, seed=seed, q2=q2, is_negative=True,
                source_path=source_path, edit_type=None, size_class=None,
                alignment=None, donor_path=None, q1=q1,
            ))
            continue

        size_class = str(size_classes[int(jrng.integers(0, len(size_classes)))])

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
            alignment=alignment, donor_path=donor_path, q1=q1,
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

    # Base de travail : image native (Q0 lu) ou recompressée à Q1 (mode contrôlé).
    # Q1 devient alors l'historique EFFECTIF du "document original" (fond = Q1->Q2).
    base_rgb = src.rgb
    if job.q1 is not None:
        base_rgb = recompress_to_q1(src.rgb, job.q1, subsampling=DEFAULT_Q2_SUBSAMPLING)
    q1_effective = job.q1 if job.q1 is not None else src.q0

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
                # qualité étrangère Q_donor (tirée dans q1_sweep) -> vraie grille étrangère.
                q1_sweep = cfg["compression"].get("q1_sweep", [job.q2])
                donor_q = int(rng.choice(q1_sweep))
                donor_rgb = recompress_to_q1(donor_rgb, donor_q, subsampling=DEFAULT_Q2_SUBSAMPLING)
            # (donneur JPEG : historique natif Q0 conservé, déjà étranger au fond.)
        # int (carré) ou [largeur_min, hauteur_min] ; normalisé dans forger.forge().
        min_region_px = cfg["forger"].get("min_region_px", forger_mod.JPEG_BLOCK)
        forge_res = forger_mod.forge(
            img=base_rgb, edit_type=job.edit_type, size_class=job.size_class,
            area_range=area_range, alignment=job.alignment,
            feather_range=feather_range, rng=rng,
            donor=donor_rgb, donor_id=donor_id,
            min_region_px=min_region_px,
        )
        if donor_q is not None:
            forge_res.extra["donor_q"] = donor_q   # qualité JPEG étrangère du splice
        edited, mask = forge_res.image, forge_res.mask
        edit_type, size_class, alignment = job.edit_type, job.size_class, job.alignment
        bbox = ann.bbox_from_mask(mask)

    # ---- Passe Q2 UNIQUE sur l'image composite entière (règle impérative) ----
    img_path = os.path.join(out_dirs["data"], f"{job.doc_id}.jpg")
    save_q2(edited, img_path, job.q2, subsampling=DEFAULT_Q2_SUBSAMPLING)

    # ---- Masque pixel-level exact ----
    mask_path = os.path.join(out_dirs["data"], f"{job.doc_id}_mask.png")
    cv2.imwrite(mask_path, mask)

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
    meta["Q1_mode"] = "controlled" if job.q1 is not None else "native"
    meta["Q1_effective"] = int(q1_effective)   # historique effectif du fond (vs Q2)
    meta["patch_grid"] = labels.tolist()
    meta["files"] = {"image": os.path.basename(img_path),
                     "mask": os.path.basename(mask_path)}
    json_path = os.path.join(out_dirs["data"], f"{job.doc_id}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Ligne de manifeste ----
    n_mask_px = int((mask > 0).sum())
    return {
        "id": job.doc_id,
        "source_id": src.source_id,
        "q0": src.q0,
        "q0_nonstandard": src.nonstandard,
        "q1_mode": "controlled" if job.q1 is not None else "native",
        "q1_effective": int(q1_effective),
        "q2": job.q2,
        "type": edit_type,
        "size_class": size_class,
        "alignment": alignment,
        "is_negative": job.is_negative,
        "bbox_x": bbox[0] if bbox else -1,
        "bbox_y": bbox[1] if bbox else -1,
        "bbox_w": bbox[2] if bbox else -1,
        "bbox_h": bbox[3] if bbox else -1,
        "n_mask_px": n_mask_px,
        "mask_frac": round(n_mask_px / (src.width * src.height), 6),
        "n_pos_patches": int((labels > 0).sum()),
        "subsampling_src": src.subsampling,
        "seed": job.seed,
        "path_img": os.path.relpath(img_path, out_dirs["root"]),
        "path_mask": os.path.relpath(mask_path, out_dirs["root"]),
        "path_json": os.path.relpath(json_path, out_dirs["root"]),
    }


# module-level pour picklabilité (ProcessPoolExecutor).
def _worker(args):
    job, cfg, out_dirs, nonstd_thr = args
    try:
        return _run_job(job, cfg, out_dirs, nonstd_thr)
    except Exception as exc:  # on isole les échecs par document
        return {"id": job.doc_id, "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------ auto mode
def _resolve_q1_mode(cfg: dict, report: dict) -> tuple[str, str]:
    """Résout `compression.q1_mode`. 'native'/'controlled' -> tels quels.

    'auto' décide d'après le corpus (mesures §README) :
      - sources lossless (PNG, pas de Q0)                 -> controlled (obligatoire)
      - JPEG avec Q0 médian >= seuil (défaut 95)          -> controlled (double
        compression native trop faible : fingerprint quasi nul)
      - JPEG avec Q0 médian < seuil                       -> native (historique réel)
    """
    mode = cfg["compression"].get("q1_mode", "auto")
    if mode != "auto":
        return mode, "fixé dans la config"
    s = report["summary"]
    if s.get("n_lossless_kept", 0) > 0:
        return "controlled", "corpus lossless (PNG) : pas de Q0"
    thr = cfg["compression"].get("q1_auto_q0_threshold", 95)
    med = s.get("q0_stats", {}).get("median")
    if med is not None and med >= thr:
        return "controlled", f"Q0 médian {med} >= {thr} (double compression native faible)"
    return "native", f"Q0 médian {med} < {thr} (historique JPEG réel exploitable)"


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

    # Résolution du mode Q1 (auto -> décidé d'après le corpus), figée pour tous les types.
    resolved_mode, reason = _resolve_q1_mode(cfg, report)
    cfg = {**cfg, "compression": {**cfg["compression"], "q1_mode": resolved_mode}}
    print(f"      q1_mode = {resolved_mode}  ({reason})")
    if n_lossless and resolved_mode != "controlled":
        raise ValueError(
            f"{n_lossless} sources lossless (pas de Q0) mais q1_mode='{resolved_mode}'. "
            "En native le fond n'est pas double-compressé -> aucun signal. Utilise "
            "q1_mode: controlled (ou auto).")

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
        sub_dirs = {"root": sub_root, "data": os.path.join(sub_root, "data")}
        os.makedirs(sub_dirs["data"], exist_ok=True)
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
    print(f"      Agréger : ./aggregate.sh --out {out_root}")
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
