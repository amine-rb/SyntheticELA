# SyntheticEla — Générateur de falsifications documentaires annotées

Génère un dataset **synthétique de falsifications documentaires annotées au pixel**
à partir de n'importe quel corpus d'images de documents authentiques. Reproduit le
scénario forensique de **double compression JPEG** (un fraudeur modifie un document
déjà enregistré, puis le resauvegarde) et fournit, pour chaque document, le masque
exact, une grille de labels patch, et les métadonnées.

Conçu pour être **réutilisable et configurable** : un seul fichier de config,
des scripts plats à la racine (pas de package Python), et le pipeline s'adapte
au type de corpus (JPEG ou lossless).

---

## 1. Installation

```bash
pip install -r requirements.txt   # Pillow, NumPy, OpenCV, PyYAML, PyArrow
```

## 2. Démarrage rapide (n'importe quel dataset)

Tout se pilote depuis `config.yaml` (`paths.source_dir`, `paths.output_dir`,
`forger.edit_types`, `orchestrator.n_docs`, ...). Édite-le, puis lance :

```bash
./run.sh                                    # tout vient de config.yaml
```

Ou surcharge ponctuellement en ligne de commande (sans toucher au fichier) :

```bash
# Génération : UN SOUS-DOSSIER par type d'édition (config forger.edit_types).
# Le pipeline sonde le corpus et choisit tout seul le mode de compression.
./run.sh --src <DOSSIER_IMAGES> --out <DOSSIER_SORTIE> --n 1000
#   -> <DOSSIER_SORTIE>/substitution/ , /copy_move/ , /splice/  (chacun autonome)

# (option) Fusionner les sous-dossiers en un dataset unique
./aggregate.sh --out <DOSSIER_SORTIE>                # -> <...>/_aggregated/

# Planches de contrôle visuel (image | ELA | masque) sur un sous-dossier donné
./preview.sh   --out <DOSSIER_SORTIE>/_aggregated
```

Exemples :
```bash
./run.sh --src SROIE2019/train/img --out output          --n 2000
./run.sh --src StaVer/scans/scans  --out output_staver   --n 1000
```

Rien d'autre à configurer : sources JPEG **ou** PNG, le mode de compression est
choisi automatiquement (§6). Chaque run écrit un **`REPORT.md`** qui explique ses
résultats (§5).

> Équivalent direct sans les `.sh` (même effet, un seul point d'entrée
> `main.py`) : `python main.py --src ... --out ... --n ...`.

## 3. Ce que fait le pipeline (chaîne forensique)

```
source (Q0 lu si JPEG, sinon lossless)
   └─[si controlled] recompression unique à Q1   ── fond = historique Q1
        └─ forger : substitution / copy_move / splice   (édition en espace pixel)
             └─ recompress : UNE SEULE passe JPEG Q2     ← unique compression finale
                  └─ annotator : masque exact + bbox + grille patch 24×24 + JSON
```

Point clé : la zone éditée subit **la même passe Q2 que le fond** — rien n'est
collé après le save final. C'est ce qui crée une vraie **incohérence d'historique
de compression** entre la zone falsifiée et le reste du document, que l'ELA révèle.

Trois types d'édition :
- **substitution** — pixels neufs peints (texte) : pas de grille 8×8 antérieure (`alignment = N/A`).
- **copy_move** — région recopiée de la même image (porte la grille Q1) ; offset ×8 → aligné, sinon désaligné.
- **splice** — région d'un autre document du corpus (grille étrangère) ; même contrôle d'alignement.

## 4. Sorties (un sous-dossier AUTONOME par type)

Chaque type demandé dans `forger.edit_types` produit un sous-dossier complet et
indépendant (il ne manque aucune info pour l'entraînement/évaluation en aval) :

```
<out>/distribution.json          # sonde du corpus source (commune à tous les types)
<out>/<type>/                     # ex. substitution / copy_move / splice
     data/<type>_<id>.jpg         # document final (fond Q1->Q2, zone incohérente)
     data/<type>_<id>_mask.png    # masque binaire pixel EXACT (sans dilatation)
     data/<type>_<id>.json        # Q0, Q1, Q2, type, taille, alignement, bbox, seed, grille 24x24
     manifest.parquet             # table du sous-dossier (une ligne par document)
     distribution.json            # sonde du corpus (copie, self-contained)
     run_config.yaml              # config effective figée (dont edit_type)
     REPORT.md                    # rapport lisible des résultats  ← §5
     ela_preview/                 # planches QA (après `./preview.sh --out <sous-dossier>`)
```

Les `id` sont **préfixés par le type** → uniques globalement. `./aggregate.sh`
réunit les sous-dossiers choisis dans `<out>/_aggregated/` (même structure, manifeste
concaténé, colonne `type` re-filtrable). Options : `--types t1 t2`, `--dest NOM`,
`--mode copy|symlink|hardlink` (symlink/hardlink = pas de duplication disque).

Colonnes du manifeste : `id, source_id, q0, q0_nonstandard, q1_mode, q1_effective,
q2, type, size_class, alignment, is_negative, bbox_x/y/w/h, n_mask_px, mask_frac,
n_pos_patches, subsampling_src, seed, path_img, path_mask, path_json`.

## 5. `REPORT.md` — les résultats de chaque run

Généré automatiquement à la fin de chaque génération (et regénérable seul :
`python reporter.py --out <DOSSIER_SORTIE>`). Il contient :

1. **Source & config** (corpus, seed, mode Q1 choisi, sweeps, Q0/dimensions),
2. **Composition** (types, tailles, alignement, négatifs, Q1, Q2),
3. **Couverture des régimes Q1/Q2** pour l'ablation robustesse (Q1<Q2 / Q1=Q2 / Q1>Q2),
4. **Contrôles d'intégrité** (masques positifs/négatifs cohérents, surface par taille),
5. **Signal ELA échantillonné** : ratio moyen ELA intérieur/extérieur du masque, par
   type × régime — c'est la mesure « le signal falsifié ressort-il ? ».

## 6. Mode de compression Q1 (auto, adaptatif)

Le paramètre `compression.q1_mode` :

| Valeur | Comportement |
| --- | --- |
| `native` | garde Q0 (scénario réaliste). Adapté à un corpus JPEG à Q0 varié/modéré. |
| `controlled` | impose Q1 par recompression → fond réellement double-compressé Q1→Q2. |
| `auto` *(défaut)* | décide seul et **journalise** le choix : lossless **ou** Q0 médian ≥ `q1_auto_q0_threshold` (95) → `controlled` ; sinon `native`. |

Pourquoi `auto` bascule en `controlled` pour les corpus lossless (PNG) ou quasi
sans perte (Q0≈100) — **mesuré**, pas supposé (ratio ELA dans/hors masque) :

| type | native (Q0≈100) | controlled Q1<Q2 | controlled Q1>Q2 |
| --- | --- | --- | --- |
| splice        | 1.41 | **5.68** | 1.91 |
| copy_move     | 1.40 | **1.69** | 1.18 |

En native sur un corpus Q0≈100, le fond n'est quasi pas double-compressé → signal
copy_move/splice faible. En controlled, le fond devient réellement double-compressé
et le signal devient net **et dépendant du régime** (le cas Q2<Q1 écrase le signal,
limite physique attendue). `q1_sweep ⊆ q2_sweep` garantit la couverture des trois
régimes. Q0 reste toujours **lu et journalisé**, jamais réinventé.

> Un corpus PNG **doit** utiliser `controlled` (pas de Q0) : le pipeline le force et
> lève une erreur explicite si on demande `native`.

## 7. Configuration (`config.yaml`)

Tous les défauts sont dans `config.yaml`, commentés. Principaux réglages :

| Section | Clés | Rôle |
| --- | --- | --- |
| `paths` | `source_dir`, `output_dir` | corpus & sortie (surchargés par `--src` / `--out`) |
| `probe` | `candidate_ext`, `allow_lossless` | extensions acceptées, prise en charge lossless |
| `compression` | `q2_sweep`, `q1_mode`, `q1_sweep`, `q1_auto_q0_threshold` | balayage de compression |
| `forger` | `edit_types`, `aligned_ratio`, `feather_radius_px`, `min_region_px` | types générés (1 sous-dossier/type), taille min garantie |
| `size_classes` | small…very_large | tailles de zone (fraction de page, ×8 px) |
| `negatives` | `ratio` | part d'authentiques (masque vide), **par sous-dossier** |
| `annotator` | `patch_size`, `patch_grid`, `patch_positive_overlap` | grille patch |
| `orchestrator` | `seed`, `n_docs`, `n_workers` | lot **par type** & parallélisme |

Surcharges CLI : `--src`, `--out`, `--n`, `--workers`, `--config`.

### `forger.min_region_px` — taille minimale garantie

```yaml
forger:
  min_region_px: [10, 10]   # [largeur_min, hauteur_min] px ; ou un entier (carré)
```

Plancher **garanti** du rectangle falsifié (substitution/copy_move/splice), quel
que soit `size_class` ou la taille de l'image source. Arrondi au **multiple de 8
supérieur** (grille JPEG) : `[10, 10]` donne donc un minimum réel de `16×16`,
jamais moins que demandé. Empêche les masques à 0 pixel sur des corpus à images
très petites (miniatures, crops) : si une source ne peut pas accueillir ce
minimum sur un axe, le document échoue proprement (erreur journalisée par
`orchestrator`, jamais écrit comme positif à masque vide).

## 8. Modules (racine du projet, un fichier par module — pas de package)

Aucun `import src...` : tous les scripts sont des fichiers plats à la racine,
appelables directement (`python module.py`) ou via `python -m module` (compatible).

| Module | Rôle |
| --- | --- |
| `jpeg_probe`  | Sonde Q0 / table quant / subsampling / dimensions (ou marque lossless) → `distribution.json`. |
| `recompress`  | Décode la source ; `recompress_to_q1` (mode contrôlé) ; `save_q2` (passe Q2 unique). |
| `forger`      | substitution / copy_move / splice ; feather anti-tell ; cohérence photométrique ; taille min garantie. |
| `annotator`   | Masque exact, bbox, grille patch 24×24, métadonnées JSON. |
| `orchestrator`| Batch scriptable, un sous-dossier autonome par type, seeds déterministes, mode Q1 auto, manifeste. |
| `aggregate`   | Fusionne les sous-dossiers de types en un dataset unique (`_aggregated/`), copy/symlink/hardlink. |
| `reporter`    | `REPORT.md` (résultats du run + séparabilité ELA). |
| `ela_preview` | Planches QA image \| ELA \| masque (ELA à une qualité distincte de Q2). |
| `main`        | Point d'entrée unique (`python main.py` = `./run.sh`). |

Wrappers shell (mêmes options que les scripts Python, lisent `config.yaml`) :
`run.sh` (génération), `aggregate.sh` (fusion), `preview.sh` (QA visuel).

## 9. Reproductibilité

- Seed global → **seed déterministe par document** (journalisé). Sortie **identique**
  quel que soit le nombre de workers (vérifié bit-à-bit).
- `run_config.yaml` fige la config effective (dont le mode Q1 résolu) de chaque lot.
