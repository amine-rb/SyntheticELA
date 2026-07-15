# SyntheticEla — Générateur de falsifications documentaires annotées

Génère un dataset **synthétique de falsifications documentaires annotées au pixel**
à partir de n'importe quel corpus d'images de documents authentiques. Reproduit le
scénario forensique de **double compression JPEG** (un fraudeur modifie un document
déjà enregistré, puis le resauvegarde) et fournit, pour chaque document, le masque
exact, une grille de labels patch, et les métadonnées.

Conçu pour être **réutilisable et configurable** : un seul fichier de config, une
ligne de commande, et le pipeline s'adapte au type de corpus (JPEG ou lossless).

---

## 1. Installation

```bash
pip install -r requirements.txt   # Pillow, NumPy, OpenCV, PyYAML, PyArrow
```

## 2. Démarrage rapide (n'importe quel dataset)

```bash
# Génération (le pipeline sonde le corpus et choisit tout seul le mode adapté)
python -m src.orchestrator --src <DOSSIER_IMAGES> --out <DOSSIER_SORTIE> --n 1000

# Planches de contrôle visuel (image | ELA | masque)
python -m src.ela_preview  --out <DOSSIER_SORTIE>
```

Exemples :
```bash
python -m src.orchestrator --src SROIE2019/train/img --out output          --n 2000
python -m src.orchestrator --src StaVer/scans/scans  --out output_staver   --n 1000
```

Rien d'autre à configurer : sources JPEG **ou** PNG, le mode de compression est
choisi automatiquement (§6). Chaque run écrit un **`REPORT.md`** qui explique ses
résultats (§5).

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

## 4. Sorties (par run)

```
<out>/data/<id>.jpg          # document final (fond Q1->Q2, zone incohérente)
<out>/data/<id>_mask.png     # masque binaire pixel EXACT (sans dilatation)
<out>/data/<id>.json         # Q0, Q1, Q2, type, taille, alignement, bbox, seed, grille patch 24x24
<out>/manifest.parquet       # table globale (une ligne par document)
<out>/distribution.json      # sonde du corpus source (Q0 / lossless / dimensions)
<out>/run_config.yaml        # config effective figée (reproductibilité)
<out>/REPORT.md              # rapport lisible des résultats du run  ← §5
<out>/ela_preview/           # planches QA (après `python -m src.ela_preview`)
```

Colonnes du manifeste : `id, source_id, q0, q0_nonstandard, q1_mode, q1_effective,
q2, type, size_class, alignment, is_negative, bbox_x/y/w/h, n_mask_px, mask_frac,
n_pos_patches, subsampling_src, seed, path_img, path_mask, path_json`.

## 5. `REPORT.md` — les résultats de chaque run

Généré automatiquement à la fin de chaque génération (et regénérable seul :
`python -m src.reporter --out <DOSSIER_SORTIE>`). Il contient :

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
| `forger` | `edit_type_ratios`, `aligned_ratio`, `feather_radius_px` | mix des falsifications |
| `size_classes` | small…very_large | tailles de zone (fraction de page, ×8 px) |
| `negatives` | `ratio` | part d'authentiques (masque vide) |
| `annotator` | `patch_size`, `patch_grid`, `patch_positive_overlap` | grille patch |
| `orchestrator` | `seed`, `n_docs`, `n_workers` | lot & parallélisme |

Surcharges CLI : `--src`, `--out`, `--n`, `--workers`, `--config`.

## 8. Modules (`src/`, un module par fichier)

| Module | Rôle |
| --- | --- |
| `jpeg_probe`  | Sonde Q0 / table quant / subsampling / dimensions (ou marque lossless) → `distribution.json`. |
| `recompress`  | Décode la source ; `recompress_to_q1` (mode contrôlé) ; `save_q2` (passe Q2 unique). |
| `forger`      | substitution / copy_move / splice ; feather anti-tell ; cohérence photométrique. |
| `annotator`   | Masque exact, bbox, grille patch 24×24, métadonnées JSON. |
| `orchestrator`| Batch scriptable, seeds déterministes, mode Q1 auto, parallélisme, manifeste. |
| `reporter`    | `REPORT.md` (résultats du run + séparabilité ELA). |
| `ela_preview` | Planches QA image \| ELA \| masque (ELA à une qualité distincte de Q2). |

## 9. Reproductibilité

- Seed global → **seed déterministe par document** (journalisé). Sortie **identique**
  quel que soit le nombre de workers (vérifié bit-à-bit).
- `run_config.yaml` fige la config effective (dont le mode Q1 résolu) de chaque lot.
