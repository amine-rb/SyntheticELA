# SyntheticEla — Générateur de falsifications documentaires annotées + évaluation

Génère un dataset **synthétique de falsifications documentaires annotées au pixel**
à partir de n'importe quel corpus d'images de documents authentiques, et fournit le
module d'**évaluation détection/localisation** (AnoViT) associé.

Reproduit le scénario forensique de **double compression JPEG** — un fraudeur modifie
un document déjà enregistré, puis le resauvegarde — et fournit, pour chaque document,
le masque exact, une grille de labels patch, et les métadonnées.

Interface volontairement minimale : **un seul fichier à éditer (`config.sh`)**,
**trois commandes à lancer (`scripts/*.sh`)**.

---

## 0. Organisation du dépôt

```
config.sh          <- LE seul fichier à éditer (tous les paramètres)
scripts/           <- LES seules commandes à lancer
   run.sh          - génération (un sous-dossier autonome par type)
   aggregate.sh    - fusion des sous-dossiers de types -> _aggregated/
   preview.sh      - planches QA image | ELA | masque
   _common.sh      - interne (charge config.sh, génère la config, localise python)
python/            <- code (10 modules ; usage normal = ne pas toucher)
markdown/          <- plan de recherche, schéma roadmap, notes
requirements.txt
README.md          <- ce fichier
```

Aucun package Python : les modules de `python/` sont des fichiers plats, rendus
importables via `PYTHONPATH` par les scripts (compatible workers *spawn* macOS).

---

## 1. Installation

```bash
pip install -r requirements.txt   # Pillow, NumPy, OpenCV, PyYAML, PyArrow
```
Pour l'évaluation (§11) uniquement : `pip install torch scikit-learn scipy`.

---

## 2. Démarrage rapide

Deux étapes :

1. **Édite `config.sh`** — au minimum `SOURCE_DIR` et `OUTPUT_DIR` ; puis `EDIT_TYPES`,
   `N_DOCS`, `N_FORGERIES`, `MIN_REGION_PX`, `NEGATIVES_RATIO`… **tout** y est.
2. **Lance** :

```bash
./scripts/run.sh          # génère un sous-dossier autonome par type d'édition
./scripts/aggregate.sh    # (option) fusionne -> OUTPUT_DIR/_aggregated/
./scripts/preview.sh      # (option) planches QA du 1er type
```

Surcharge ponctuelle en CLI, sans éditer `config.sh` :

```bash
./scripts/run.sh --src AUTRE/DOSSIER --out AUTRE/SORTIE --n 500 --workers 8
./scripts/aggregate.sh --types substitution splice --mode symlink
./scripts/preview.sh --out "$OUTPUT_DIR/_aggregated" --n 20
```

Sources JPEG **ou** PNG : le mode de compression est choisi automatiquement (§7).
Chaque run écrit un **`REPORT.md`** qui explique ses résultats (§6).

---

## 3. Chaîne forensique

```
source (Q0 lu si JPEG, sinon lossless)
   └─[si controlled] recompression unique à Q1   ── fond = historique Q1
        └─ forger : k falsifications (même type) en espace pixel
             └─ recompress : UNE SEULE passe JPEG Q2   ← unique compression finale
                  └─ annotator : masque exact + bbox + grille patch 24×24 + JSON
```

Point clé : la (les) zone(s) éditée(s) subissent **la même passe Q2 que le fond** —
rien n'est collé après le save final. C'est ce qui crée une vraie **incohérence
d'historique de compression** entre les zones falsifiées et le reste du document,
que l'ELA révèle.

Trois types d'édition :
- **substitution** — pixels neufs peints (texte) : pas de grille 8×8 antérieure (`alignment = N/A`).
- **copy_move** — région recopiée de la même image (porte la grille Q1) ; offset ×8 → aligné, sinon désaligné.
- **splice** — région d'un autre document du corpus (grille étrangère) ; même contrôle d'alignement.

---

## 4. Falsifications multiples & taille minimale

**Plusieurs falsifications par document** (`N_FORGERIES=(min max)`). Chaque document
positif reçoit `k ~ U{min..max}` falsifications du **même type**, aux empreintes
**disjointes** (pas de chevauchement), et le masque final est leur **union**.

**Plafond de taille automatique selon `k`** : plus il y a de falsifications, plus
elles sont petites, pour ne jamais couvrir toute la page. Concrètement (4 classes de
taille ordonnées) :

| `k` | classes de taille autorisées |
| --- | --- |
| 1   | small · medium · large · very_large |
| 2   | small · medium · large |
| 3   | small · medium |
| 4   | small |
| 5   | small |

**Taille minimale garantie** (`MIN_REGION_PX=(largeur hauteur)`) : plancher du
rectangle falsifié quel que soit la classe ou l'image source, arrondi au **multiple de
8 supérieur** (grille JPEG) — `(10 10)` ⇒ minimum réel `16×16`. Si une source est trop
petite pour l'accueillir sur un axe, le document **échoue proprement** (erreur
journalisée par l'orchestrator) au lieu d'écrire un positif à **masque vide**.

---

## 5. Sorties (un sous-dossier AUTONOME par type)

Chaque type de `EDIT_TYPES` produit un sous-dossier complet et indépendant (il ne
manque aucune info pour l'entraînement/évaluation en aval) :

```
<out>/distribution.json          # sonde du corpus source (commune à tous les types)
<out>/<type>/                     # ex. substitution / copy_move / splice
     data/<type>_<id>.jpg         # document final (fond Q1->Q2, zones incohérentes)
     data/<type>_<id>_mask.png    # masque binaire pixel EXACT (union des k zones)
     data/<type>_<id>.json        # Q0/Q1/Q2, type, taille, alignement, n_forgeries, bboxes, seed, grille 24x24
     manifest.parquet             # table du sous-dossier (une ligne par document)
     distribution.json            # sonde du corpus (copie, self-contained)
     run_config.yaml              # config effective figée
     REPORT.md                    # rapport lisible des résultats  ← §6
     ela_preview/                 # planches QA (après ./scripts/preview.sh)
```

Les négatifs (authentiques, masque vide) portent le marqueur `authentic` dans leur
nom (`<type>_authentic_<id>`) : jamais de « falsification » à masque vide.

Les `id` sont **préfixés par le type** → uniques globalement. `./scripts/aggregate.sh`
réunit les sous-dossiers choisis dans `<out>/_aggregated/` (même structure, manifeste
concaténé, colonne `type` re-filtrable). Options : `--types t1 t2`, `--dest NOM`,
`--mode copy|symlink|hardlink` (symlink/hardlink = pas de duplication disque).

Colonnes du manifeste : `id, source_id, q0, q0_nonstandard, q1_mode, q1_effective,
q2, type, size_class, alignment, is_negative, n_forgeries, bbox_x/y/w/h, n_mask_px,
mask_frac, n_pos_patches, subsampling_src, seed, path_img, path_mask, path_json`.
(`bbox_*` = englobante de l'union ; les bbox individuelles sont dans le JSON,
champ `forgery_bboxes`.)

---

## 6. `REPORT.md` — les résultats de chaque run

Généré automatiquement à la fin de chaque génération (regénérable seul :
`python python/reporter.py --out <SOUS-DOSSIER>`). Il contient :

1. **Source & config** (corpus, seed, mode Q1 choisi, sweeps, Q0/dimensions),
2. **Composition** (types, tailles, alignement, négatifs, Q1, Q2),
3. **Couverture des régimes Q1/Q2** pour l'ablation robustesse (Q1<Q2 / Q1=Q2 / Q1>Q2),
4. **Contrôles d'intégrité** (masques positifs/négatifs cohérents, surface par taille),
5. **Signal ELA échantillonné** : ratio moyen ELA intérieur/extérieur du masque, par
   type × régime — la mesure « le signal falsifié ressort-il ? ».

---

## 7. Mode de compression Q1 (auto, adaptatif)

`Q1_MODE` (config.sh) :

| Valeur | Comportement |
| --- | --- |
| `native` | garde Q0 (scénario réaliste). Adapté à un corpus JPEG à Q0 varié/modéré. |
| `controlled` | impose Q1 par recompression → fond réellement double-compressé Q1→Q2. |
| `auto` *(défaut)* | décide seul et **journalise** : lossless **ou** Q0 médian ≥ `Q1_AUTO_Q0_THRESHOLD` (95) → `controlled` ; sinon `native`. |

Pourquoi `auto` bascule en `controlled` pour un corpus lossless (PNG) ou quasi sans
perte (Q0≈100) — **mesuré**, pas supposé (ratio ELA dans/hors masque) :

| type | native (Q0≈100) | controlled Q1<Q2 | controlled Q1>Q2 |
| --- | --- | --- | --- |
| splice    | 1.41 | **5.68** | 1.91 |
| copy_move | 1.40 | **1.69** | 1.18 |

En native sur Q0≈100, le fond n'est quasi pas double-compressé → signal faible. En
controlled, le fond devient réellement double-compressé et le signal net **et dépendant
du régime** (Q2<Q1 écrase le signal, limite physique attendue). `Q1_SWEEP ⊆ Q2_SWEEP`
garantit la couverture des trois régimes. Q0 reste toujours **lu et journalisé**.

> Un corpus PNG **doit** utiliser `controlled` (pas de Q0) : le pipeline le force et
> lève une erreur explicite si on demande `native`.

---

## 8. Configuration (`config.sh`)

**Tous** les paramètres sont là (variables shell commentées) — seul fichier à éditer.

| Variable(s) | Rôle |
| --- | --- |
| `SOURCE_DIR`, `OUTPUT_DIR` | corpus source & racine de sortie |
| `CANDIDATE_EXT`, `ALLOW_LOSSLESS` | extensions acceptées, prise en charge lossless (PNG) |
| `Q2_SWEEP`, `Q1_MODE`, `Q1_SWEEP`, `Q1_AUTO_Q0_THRESHOLD` | balayage de compression |
| `EDIT_TYPES` | types générés — **un sous-dossier par type** |
| `N_FORGERIES` | `(min max)` falsifications par doc — plafond de taille auto (§4) |
| `MIN_REGION_PX` | `(largeur hauteur)` taille min garantie de zone (§4) |
| `ALIGNED_RATIO`, `FEATHER_RADIUS_PX` | alignement grille 8×8, adoucissement anti-tell |
| `SIZE_SMALL … SIZE_VERY_LARGE` | tailles de zone (fraction de page, min max) |
| `NEGATIVES_RATIO` | part d'authentiques par sous-dossier (`0.0` = que des fraudes) |
| `INPUT_RES`, `PATCH_SIZE`, `PATCH_GRID`, `PATCH_POSITIVE_OVERLAP` | grille patch |
| `ELA_QUALITY`, `ELA_N_SAMPLES` | QA visuel |
| `SEED`, `N_DOCS`, `N_WORKERS` | reproductibilité, lot **par type**, parallélisme |
| `PYTHON` | interpréteur (celui qui a les dépendances) |

Surcharges CLI (sans éditer `config.sh`) : `--src`, `--out`, `--n`, `--workers`.

---

## 9. Code (`python/`, un fichier par module)

| Module | Rôle |
| --- | --- |
| `jpeg_probe`  | Sonde Q0 / table quant / subsampling / dimensions (ou marque lossless) → `distribution.json`. |
| `recompress`  | Décode la source ; `recompress_to_q1` (mode contrôlé) ; `save_q2` (passe Q2 unique). |
| `forger`      | substitution / copy_move / splice ; multi-falsification ; feather anti-tell ; taille min garantie. |
| `annotator`   | Masque exact, bbox, grille patch 24×24, métadonnées JSON. |
| `orchestrator`| Batch scriptable, un sous-dossier autonome par type, seeds déterministes, mode Q1 auto, manifeste. |
| `aggregate`   | Fusionne les sous-dossiers de types en un dataset unique (`_aggregated/`). |
| `reporter`    | `REPORT.md` (résultats du run + séparabilité ELA). |
| `ela_preview` | Planches QA image \| ELA \| masque (ELA à une qualité distincte de Q2). |
| `main`        | Point d'entrée appelé par `run.sh`. |
| `detection_eval` | Évaluation détection/localisation AnoViT (§11) — à copier dans le codebase d'entraînement. |

---

## 10. Reproductibilité

- Seed global → **seed déterministe par document** (journalisé). Sortie **identique**
  quel que soit le nombre de workers (chaque job forge avec son propre seed).
- Chaque type reçoit un flux aléatoire décorrélé → aucun doublon exact entre
  sous-dossiers à l'agrégation.
- `run_config.yaml` fige la config effective (dont le mode Q1 résolu) de chaque lot.

---

## 11. Évaluation détection/localisation (`python/detection_eval.py`)

Module autonome à **copier dans le codebase d'entraînement** d'AnoViT. Implémente le
protocole du mémoire (voir `markdown/plan.md` §9.3 seuil, §9.3bis pilotage, §9.4
métriques). Dépendances supplémentaires : `torch, scikit-learn, scipy`.

**Cache ELA (une fois, hors entraînement).** ELA à résolution native → échelle globale
fixe → resize 384 → PNG gris (jamais JPEG). Une passe sert E0/E1 (1 qualité) et E2 (3) :

```python
from detection_eval import build_ela_cache
build_ela_cache("<out>/_aggregated/data", "cache/dev",  qualities=(75, 85, 90, 95))
build_ela_cache("chemin/authentiques",    "cache/auth", qualities=(75, 85, 90, 95))
```

**Loaders + pilotage best-detection (§9.3bis).**

```python
from torch.utils.data import DataLoader
from detection_eval import (SyntheticDevDataset, AuthenticELADataset,
                            pilot_subset, evaluate, BestDetectionTracker)

dev_ds  = SyntheticDevDataset("<out>/_aggregated/data", "cache/dev", qualities=(90,))
dev_ld  = DataLoader(pilot_subset(dev_ds, 400, seed=42), batch_size=48, num_workers=8)
tracker = BestDetectionTracker("experiments/E0/best_model.pt",
                               history_path="experiments/E0/auprc_curve.json", patience=15)

for epoch in range(100):
    train_one_epoch(model, train_loader)
    res = evaluate(model, dev_ld, error_mode="mae", metrics=("auprc",))
    tracker.update(epoch, res["pixel_auprc"], model)     # checkpoint = max AUPRC dev
    if tracker.should_stop:
        break
```

**Évaluation finale (§9.4)** — dev complet, seuil calibré sur le dev (max Dice),
**figé** puis repassé une seule fois au test réel :

```python
res = evaluate(model, dev_full, error_mode="mae", metrics="full", authentic_loader=auth_ld)
# {pixel_auprc, aupro, threshold, dice, iou, fpr_authentic, image_auroc, pixel_auroc}
```

Pièges : ne jamais binariser les scores avant `evaluate` (AUPRC/AUPRO ont besoin des
scores continus) ; garder le même `seed` pour le sous-échantillon pilote ; calibrer le
seuil sur le dev, jamais sur le test.
