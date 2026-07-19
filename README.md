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

Sources JPEG **ou** PNG : deux qualités `Q1 < Q2` par document (base puis save
final ; l'écart crée le signal ELA, §7). Chaque run écrit un **`REPORT.md`** (§6).

---

## 3. Chaîne forensique

```
source (décodée ; Q0 lu si JPEG, sinon lossless)
   └─ recompress @Q1         ── "document original" = fond + texte à qualité MOYENNE Q1
        ├─ POSITIF : forger peint k substitutions (pixels NEUFS, jamais vus par Q1)
        │       └─ save @Q2   ← fond/texte = Q1→Q2 ; substitution = Q2 seul (Q1<Q2)
        └─ NÉGATIF : save @Q2 ← tout = Q1→Q2, masque vide
             └─ annotator : masque exact + bbox + grille patch 24×24 + JSON
```

Point clé : **deux qualités `Q1 < Q2`** par document. `Q2` (save final, haute) est
tiré de `QUALITY_SWEEP` ; `Q1 = Q2 − Q1_GAP` (base, moyenne). **C'est l'écart qui
crée le signal** : le fond et tout le texte **authentique** portent l'historique
`Q1→Q2`, tandis que la substitution, peinte en pixels **neufs entre les deux
passes**, n'a subi que `Q2`. Sondée en ELA (à une qualité **≠ `Q2`**), la zone
n'ayant vu que `Q2` **ressort du texte ordinaire** (mesuré : *forgé/texte-authentique
≈ 1,8* à `Q1_GAP=28`). Le négatif subit exactement le même double-passage `Q1→Q2`
(seule la substitution manque) → ELA propre, **indiscernable du fond d'un positif** :
aucun indice global, le modèle doit **localiser**.

> ⚠️ **L'écart est impératif.** En `Q1 == Q2`, une substitution est *indiscernable
> du texte authentique* en ELA (ratio mesuré ≈ 1,0 : tout bord de texte s'allume,
> quel que soit son historique). Le « 3,5× intérieur/extérieur du masque » qu'on
> mesurait auparavant ne comparait que *texte vs papier blanc* — trompeur. Deux
> boutons seulement, pas l'ancienne machinerie de régimes : `QUALITY_SWEEP` (les
> `Q2`) et `Q1_GAP` (l'écart). Et `ELA_QUALITY` doit différer de tout `Q2` du sweep
> (sinon l'ELA s'effondre à 0 partout — l'orchestrator le refuse).

Trois types d'édition :
- **substitution** — écrit une **valeur plausible** (montant, date, quantité, code au
  format document) à la **taille du texte du document**, en encre sombre, sur du
  contenu existant ; pas de grille 8×8 antérieure (`alignment = N/A`).
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

**Placement sur contenu réel** (`PLACE_ON_CONTENT=true`, `MIN_CONTENT_FRAC`) : les
falsifications visent une zone portant du **texte/chiffres** (détection d'encre par
Otsu) au lieu des marges blanches — scénario réaliste (un fraudeur modifie une valeur
existante) **et** signal ELA exploitable. `MIN_CONTENT_FRAC` = fraction d'encre visée
(best-effort ; retombe sur le meilleur emplacement si la page est presque vide). Pour
la substitution, l'**encre** est en outre forcée sombre/contrastée (vrais bords → vrai
signal), et le splice/copy-move copient une **vraie zone d'encre**. Sans ce placement,
un aplat sur marge blanche produit une zone quasi invisible en ELA (« trou propre »).

**Substitution réaliste** : la valeur injectée est **plausible** (montant/date/code au
format document) et rendue à la **taille du texte RÉEL du document** — hauteur des
glyphes **mesurée** par composantes connexes du masque d'encre (pas une fraction
devinée), légèrement modulée (`×0,9–1,3`), dans une boîte **serrée** autour du texte.
Le glyphe injecté fait donc ≈ **1× le corps de texte** du document (~1 % de la hauteur
de page), au lieu de ~3× auparavant. But : que la détection soit attribuable à
l'**incohérence de compression** et non à un artefact du générateur (gros texte /
charabia) — cf. le « tell » du générateur, `markdown/plan.md` §8.

> Multi-falsification : `bbox_*` du manifeste est l'**englobante de l'union** des k
> zones (donc large si elles sont dispersées) ; les rectangles individuels sont dans
> le JSON, champ `forgery_bboxes`.

> Aperçu QA (`ELA_SCALE`) : `./scripts/preview.sh` calcule l'ELA à **échelle globale
> fixe** (défaut 15, à aligner sur `detection_eval.ELA_SCALE`) au lieu d'un étirement
> par le max de chaque image — l'aperçu reflète ce que « voit » le modèle et n'écrase
> plus les fraudes faibles.

---

## 5. Sorties (un sous-dossier AUTONOME par type)

Chaque type de `EDIT_TYPES` produit un sous-dossier complet et indépendant (il ne
manque aucune info pour l'entraînement/évaluation en aval) :

```
<out>/distribution.json               # sonde du corpus source (commune à tous les types)
<out>/<type>/                          # ex. substitution / copy_move / splice
     images/<type>_<id>.jpg            # document final (fond Q1→Q2 ; zone éditée = Q2 seul)
     images/images.csv                 # id, image, type, is_negative, quality, size_class, n_forgeries, source_id, seed
     masks/<type>_<id>_mask.png        # masque binaire pixel EXACT (union des k zones)
     masks/<type>_<id>.json            # Q0/Q1/Q2, type, taille, alignement, n_forgeries, bboxes, seed, grille 24x24, ela
     masks/masks.csv                   # id, mask, json, is_negative, n_forgeries, n_mask_px, mask_frac, bbox_*
     ela/<type>_<id>_ela.png           # ELA RGB (3 qualités ≈ Q1 empilées), résolution native, sur le JPEG final
     ela/ela.csv                       # id, ela, image, ela_qualities, ela_scale, is_negative, type
     manifest.parquet                  # table du sous-dossier (une ligne par document)
     distribution.json                 # sonde du corpus (copie, self-contained)
     run_config.yaml                   # config effective figée
     REPORT.md                         # rapport lisible des résultats  ← §6
     ela_preview/                      # planches QA image|ELA|masque (après ./scripts/preview.sh)
```

**Trois dossiers, un CSV chacun** (`images/`, `masks/`, `ela/`) : chaque CSV est
autonome (une ligne par document, nom de fichier + métadonnées) → on charge un
dossier sans lire le manifeste Parquet. L'**ELA est une sortie de première classe**,
calculée à la génération sur le **JPEG final re-lu** (l'artefact réel), à résolution
native et échelle globale fixe (`ELA_SCALE`, alignée sur `detection_eval.ELA_SCALE`)
→ un fichier ELA par image, **aligné pixel-à-pixel** avec image et masque. L'ELA est
produite pour **toutes** les images (positifs ET négatifs : l'ELA du négatif est la
référence « propre »).

Les négatifs (authentiques, masque vide) portent le marqueur `authentic` dans leur
nom (`<type>_authentic_<id>`) : jamais de « falsification » à masque vide.

Les `id` sont **préfixés par le type** → uniques globalement. `./scripts/aggregate.sh`
réunit les sous-dossiers choisis dans `<out>/_aggregated/` (même structure — `images/`,
`masks/`, `ela/` + CSV régénérés —, manifeste concaténé, colonne `type` re-filtrable).
Options : `--types t1 t2`, `--dest NOM`, `--mode copy|symlink|hardlink`
(symlink/hardlink = pas de duplication disque).

Colonnes du manifeste : `id, source_id, q0, q0_nonstandard, q1_mode, q1_effective,
q2, type, size_class, alignment, is_negative, n_forgeries, bbox_x/y/w/h, n_mask_px,
mask_frac, n_pos_patches, subsampling_src, seed, ela_quality, ela_qualities, ela_scale,
path_img, path_mask, path_json, path_ela`. (`bbox_*` = englobante de l'union ; les bbox
individuelles sont dans le JSON, champ `forgery_bboxes`.)

---

## 6. `REPORT.md` — les résultats de chaque run

Généré automatiquement à la fin de chaque génération (regénérable seul :
`python python/reporter.py --out <SOUS-DOSSIER>`). Il contient :

1. **Source & config** (corpus, seed, `QUALITY_SWEEP`, Q0/dimensions),
2. **Composition** (types, tailles, alignement, négatifs, qualité `Q`),
3. **Contrôles d'intégrité** (masques positifs/négatifs cohérents, surface par taille),
4. **Signal ELA échantillonné** : ratio ELA **forgé / texte authentique**, par type —
   la vraie mesure « la falsification ressort-elle du texte ordinaire ? » (et non
   texte-vs-papier, qui est toujours élevé et trompeur seul).

---

## 7. Compression : un écart de qualité `Q1 < Q2` par document

Deux boutons seulement : `QUALITY_SWEEP` (les valeurs de `Q2`, save final) et
`Q1_GAP` (l'écart). Chaque document tire un `Q2` dans le sweep et pose
`Q1 = Q2 − Q1_GAP` :

```
source décodée → recompress @Q1  (le "document original", qualité MOYENNE Q1)
   → [positif] peindre la substitution (pixels neufs) → save @Q2   (Q1 < Q2)
   → [négatif]                                          save @Q2
```

- **Fond + texte authentique** : historique `Q1→Q2`. Sondé en ELA **≈ Q1** (leur point
  fixe) il est **atténué** → **ELA sombre**.
- **Substitution** : peinte **entre** les deux passes → n'a vu que `Q2`, jamais `Q1` →
  **ELA vif**, elle **ressort du texte authentique**.
- **Négatif** : même double-passage `Q1→Q2`, sans substitution → **ELA propre**, fond
  **identique** à celui d'un positif → pas d'indice global (le modèle localise).

Un corpus PNG (lossless) est géré nativement : l'historique vient **entièrement** de
la passe `Q1`.

### Qualité de sonde ELA : viser ≈ Q1 (crucial)

La sonde ELA (`ELA_QUALITY`) doit viser **≈ Q1** (= médiane(Q2) − `Q1_GAP`), le point
fixe du fond. C'est ce qui minimise l'ELA du texte authentique et maximise celle de la
falsification. Mesuré (corpus StaVer, `Q2∈{92,95,97}`, `Q1_GAP=28` → `Q1∈{64,67,69}`),
**forgé / texte-authentique** selon la sonde :

| sonde `ELA_QUALITY` | forgé / **texte-authentique** | luminosité zone |
| --- | --- | --- |
| 90 (ancien) | 1.8 | ×1 |
| **67 (≈ Q1, défaut)** | **3.2** | **×2.5** |

L'écart `Q1_GAP` compte aussi (à sonde ≈ Q1) : 22→~2.5, **28→~3.2**, 32→plafonne.
En `Q1==Q2` : **≈ 1.0** ❌ (indiscernable, aucun signal).

### ELA de sortie = image COULEUR RGB (3 qualités)

`ela/*.png` est une **image RGB** : les 3 canaux sont l'ELA à 3 qualités encadrant `Q1`
(`ELA_QUALITY ± ELA_SPREAD` = 59/67/75). La couleur vient de la **diversité de qualité**
(pas de la chroma — mesurée anti-corrélée ici). La zone falsifiée, vive dans les 3 sondes,
ressort en **blanc/teinté** ; le texte authentique reste sombre. Cela donne **3 canaux
d'info** au modèle (= mode E2 de `detection_eval`).

> **Deux règles impératives.** (1) `Q1 < Q2` (`Q1_GAP > 0`) — sinon la falsification est
> indiscernable du texte réel (ratio ≈ 1,0). (2) les 3 qualités ELA ≠ tout `Q2` du sweep
> — sinon l'ELA **s'effondre à 0** (image au point fixe de la sonde ; l'orchestrator le
> refuse). L'orchestrator affiche la sonde recommandée (≈ Q1) et alerte si tu t'en
> éloignes. `copy_move`/`splice` restent générables mais plus faibles (zone déjà porteuse
> d'un historique JPEG) : ce pipeline vise la substitution.

---

## 8. Configuration (`config.sh`)

**Tous** les paramètres sont là (variables shell commentées) — seul fichier à éditer.

| Variable(s) | Rôle |
| --- | --- |
| `SOURCE_DIR`, `OUTPUT_DIR` | corpus source & racine de sortie |
| `CANDIDATE_EXT`, `ALLOW_LOSSLESS` | extensions acceptées, prise en charge lossless (PNG) |
| `QUALITY_SWEEP` | valeurs de `Q2` (save final, hautes) tirées par document (§7) — **toutes ≠ `ELA_QUALITY`** |
| `Q1_GAP` | écart de compression : `Q1 = Q2 − Q1_GAP` (§7) — **le bouton du signal**, défaut `28` |
| `EDIT_TYPES` | types générés — **un sous-dossier par type** (viser `substitution`) |
| `N_FORGERIES` | `(min max)` falsifications par doc — plafond de taille auto (§4) |
| `MIN_REGION_PX` | `(largeur hauteur)` taille min garantie de zone (§4) |
| `PLACE_ON_CONTENT`, `MIN_CONTENT_FRAC` | placer les fraudes sur du contenu réel (§4) |
| `ALIGNED_RATIO`, `FEATHER_RADIUS_PX` | alignement grille 8×8, adoucissement anti-tell |
| `SIZE_SMALL … SIZE_VERY_LARGE` | tailles de zone (fraction de page, min max) |
| `NEGATIVES_RATIO` | part d'authentiques par sous-dossier (`0.0` = que des fraudes) |
| `INPUT_RES`, `PATCH_SIZE`, `PATCH_GRID`, `PATCH_POSITIVE_OVERLAP` | grille patch |
| `ELA_QUALITY`, `ELA_SPREAD` | sonde ELA : centre ≈ Q1 + écart des 3 canaux RGB (§7) |
| `ELA_N_SAMPLES`, `ELA_SCALE` | nb de planches QA ; échelle ELA globale fixe |
| `SEED`, `N_DOCS`, `N_WORKERS` | reproductibilité, lot **par type**, parallélisme |
| `PYTHON` | interpréteur (celui qui a les dépendances) |

Surcharges CLI (sans éditer `config.sh`) : `--src`, `--out`, `--n`, `--workers`.

---

## 9. Code (`python/`, un fichier par module)

| Module | Rôle |
| --- | --- |
| `jpeg_probe`  | Sonde Q0 / table quant / subsampling / dimensions (ou marque lossless) → `distribution.json`. |
| `recompress`  | Décode la source ; `recompress_to_q1` (base @Q) ; `save_q2` (save final @Q, même qualité). |
| `forger`      | substitution / copy_move / splice ; multi-falsification ; feather anti-tell ; taille min garantie. |
| `annotator`   | Masque exact, bbox, grille patch 24×24, métadonnées JSON. |
| `orchestrator`| Batch scriptable, un sous-dossier autonome par type, seeds déterministes, mode Q1 auto, manifeste. |
| `aggregate`   | Fusionne les sous-dossiers de types en un dataset unique (`_aggregated/`). |
| `reporter`    | `REPORT.md` (résultats du run + séparabilité ELA). |
| `ela_preview` | Planches QA image \| ELA \| masque (même ELA RGB 3 qualités ≈ Q1 que la sortie). |
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

**Cache ELA (une fois, hors entraînement).** Le dossier `ela/` généré contient l'ELA
RGB (3 qualités ≈ Q1) à résolution native ; l'entraînement construit son propre cache
**384** via `build_ela_cache`, recalculé depuis les **images** (`images/`) : ELA
résolution native → échelle globale fixe → resize 384 → PNG gris (jamais JPEG). **Viser
≈ Q1** (≈ 67 par défaut), PAS 90. Une passe sert E0/E1 (1 qualité, 67) et E2 (3, encadrant Q1) :

```python
from detection_eval import build_ela_cache
build_ela_cache("<out>/_aggregated/images", "cache/dev",  qualities=(59, 67, 75))  # ≈ Q1
build_ela_cache("chemin/authentiques",      "cache/auth", qualities=(59, 67, 75))
```

**Loaders + pilotage best-detection (§9.3bis).**

```python
from torch.utils.data import DataLoader
from detection_eval import (SyntheticDevDataset, AuthenticELADataset,
                            pilot_subset, evaluate, BestDetectionTracker)

dev_ds  = SyntheticDevDataset("<out>/_aggregated/masks", "cache/dev", qualities=(67,))  # ≈Q1 ; 1er arg = dossier masks/
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
