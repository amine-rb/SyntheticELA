# Mini-rapport — Intégration du corpus StaVer (stampDS)

**Date :** 2026-07-15 · **Seed :** 42 · **Sortie :** `output_staver/` (indépendante de `output/` SROIE)

## 1. Ce qu'est le corpus StaVer

Exploré `StaVer/` : ce n'est **pas** un dataset de falsification, mais un dataset
de **tampons/signatures** (stampDS) — 427 scans d'**invoices authentiques**
(1632×2302, RGB, **PNG**) accompagnés de ground-truth de localisation des tampons
(`ground-truth-maps/`, `ground-truth-pixel/`) et d'un `info/*.txt`
(`#signature textOverlap numStamps bwStamp1`).

→ Je l'ai donc traité comme un **corpus de documents authentiques source**
(comme SROIE), pour générer un **second dev synthétique falsifié indépendant**.
Les ground-truth de tampons ne sont pas utilisés pour la génération (ils
annotent des tampons, pas des falsifications) ; les tampons/signatures restent
présents dans le fond des documents générés — c'est précisément leur intérêt (§4).

## 2. Décision clé : PNG → mode Q1 contrôlé obligatoire (justifié)

Un PNG est **lossless** : aucune compression JPEG antérieure, donc **pas de Q0**.
L'instruction excluait les PNG car ils « cassent le mismatch de double
compression » — vrai **en mode native uniquement**. En mode **Q1 contrôlé**, on
impose Q1 par une recompression : le fond devient réellement double-compressé
Q1→Q2. Et comme le PNG est parfaitement sans perte, la passe Q1 crée un
historique **exactement** à Q1 (aucun résidu Q0, contrairement à SROIE Q0≈100) —
le régime Q1/Q2 est donc encore **mieux contrôlé** ici que sur SROIE.

Garde-fou ajouté : si le corpus contient des sources lossless et que
`q1_mode != controlled`, l'orchestrateur **s'arrête avec une erreur explicite**
(en native, le fond ne serait pas double-compressé → aucun signal à localiser).
Vérifié : le garde-fou se déclenche bien.

## 3. Modifications du pipeline (rétro-compatibles, SROIE inchangé)

| Module | Changement |
| --- | --- |
| `jpeg_probe` | `allow_lossless` : garde les PNG avec `q0=-1`, `is_lossless=True` ; stats Q0 séparées du décompte lossless ; CLI `--allow-lossless`. |
| `recompress` | `decode_source` accepte les sources lossless (q0=-1, table vide, subsampling "none"). |
| `orchestrator` | lit `probe.allow_lossless` ; **garde-fou** lossless→controlled ; **splice à donneur lossless** : on impose au donneur une qualité JPEG étrangère `Q_donor` (tirée dans `q1_sweep`) pour lui donner une vraie grille étrangère (sinon un splice PNG se confondrait avec une substitution). `donor_q` journalisé. |
| `config.yaml` | `q1_mode: auto` : détecte le corpus lossless et bascule seul en `controlled` (plus besoin de config dédiée). |

Le chemin JPEG (SROIE) est inchangé : le garde-fou et la recompression du
donneur ne se déclenchent que pour des sources/donneurs lossless. `output/`
(SROIE, 2000 docs) est resté intact.

## 4. Pourquoi ce corpus est précieux (au-delà de l'indépendance)

Les scans StaVer contiennent **tampons, signatures et logos colorés** conservés
par construction (on part du scan complet). C'est exactement le matériau de
l'**analyse de faux positifs E7** (plan §E7) et de l'**hypothèse secondaire**
(§4 du plan : le modèle ne doit pas apprendre « coloré = anomalie »). Ce corpus
complète SROIE (reçus) par un **type de document différent** (factures) →
utile pour un contrôle **cross-corpus**.

## 5. Lot généré (`output_staver/`)

- **1000 documents**, 0 erreur, ~79 s (4 workers). 427 sources uniques → ~2,3
  variantes/source (édition, Q1, Q2, position variés).
- Négatifs **311** (31 %) · types splice 233 / substitution 225 / copy_move 231 /
  authentic 311 · alignement aligné 235 / désaligné 229 / N/A 536.
- Q2 stratifié (250 chacun) · Q1 effectif ∈ {55,70,85} équilibré.
- **Régimes E5 (positifs)** : Q1<Q2 341 · Q1=Q2 179 · Q1>Q2 169 → les 3 régimes peuplés.
- Intégrité : 0 positif à masque vide, 0 négatif à masque non vide.
- 233/233 splices avec `donor_q` (qualité JPEG étrangère) journalisée.

Sorties : `output_staver/data/<id>.{jpg,_mask.png,json}`, `manifest.parquet`,
`distribution.json`, `run_config.yaml` (snapshot reproductible), `ela_preview/`
(**50 planches** image | ELA Q90 | masque).

## 6. Vérification visuelle

Contrôle ELA (Q90, distincte de Q2) : la région falsifiée ressort nettement sur
le masque (splice/substitution). Les tampons/logos colorés répondent aussi en ELA
— confirmation qualitative que ce corpus est adapté à l'étude des faux positifs.

## 7. Réserve à documenter (honnêteté méthodologique)

Idéalement, Q1 devrait suivre la distribution réelle des qualités JPEG des
documents cibles EY (plan §Étape 0). StaVer étant lossless, le sweep Q1 est un
choix expérimental assumé (comme pour SROIE Q0≈100). Les 50 réels EY restent le
juge final ; ce second corpus sert de contrôle de généralisation cross-corpus.

## 8. Reproduire

```bash
python -m src.orchestrator --src StaVer/scans/scans --out output_staver --n 1000 --workers 4
python -m src.ela_preview  --out output_staver
```
Le mode Q1 est choisi automatiquement (`auto` → `controlled` car corpus PNG).
