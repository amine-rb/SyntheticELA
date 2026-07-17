# plan.md — Roadmap mémoire de recherche (v2)

> Révision v2 : ajout de l'étape 0 (génération d'un jeu falsifié synthétique annoté),
> design de données à trois niveaux, protocole d'entraînement piloté par l'AUPRC de
> localisation, correction du data leakage de sélection, recalibrage des attentes
> statistiques (test réel = 50/50), priorisation des métriques threshold-free.

## 1. Contexte

Ce mémoire porte sur la **localisation non supervisée de falsifications documentaires**
dans des documents JPEG scannés.

Le scénario étudié est le suivant : un fraudeur part d'un document déjà compressé en JPEG,
modifie localement une zone sensible (montant, date, signature, texte), puis resauvegarde le
document. Cette opération crée une **incohérence locale d'historique de compression** entre les
régions modifiées et les régions non modifiées.

L'Error Level Analysis (ELA) rend partiellement visibles ces différences de compression.
L'objectif n'est pas simplement de détecter une anomalie visuelle, mais de localiser une région
dont le comportement de recompression est incohérent avec le reste du document.

---

## 2. Positionnement du travail

L'équipe dispose déjà d'une **baseline CNN** pour la localisation de falsifications sur cartes ELA.
Ma contribution est l'introduction et l'adaptation d'une architecture **AnoViT encoder-decoder**,
motivée par l'hypothèse que le contexte global des mécanismes d'attention réduit les faux positifs
sur les éléments saillants (logos, tampons, zones colorées) là où un encodeur convolutionnel à champ
réceptif local échoue.

La baseline CNN sert donc de **point de comparaison de référence** (état de l'art interne).
AnoViT, dans sa **configuration optimisée finale**, est le système central du mémoire (**E0**).

Le décodeur d'AnoViT a été modifié par rapport au papier original (§5.2). Cette modification est
traitée comme une **contribution architecturale à part entière** et fait l'objet d'une ablation
dédiée (E8), évaluée en localisation et non en simple qualité de reconstruction.

> Question : une architecture Transformer (AnoViT) exploite-t-elle mieux les incohérences ELA d'une
> falsification JPEG que la baseline CNN de l'équipe, pour localiser les zones modifiées ?

---

## 3. Question de recherche

> Dans quelle mesure une falsification documentaire JPEG peut-elle être détectée comme un événement
> hors-distribution dans l'espace ELA appris par un autoencodeur AnoViT entraîné sur documents
> authentiques, et cet apport est-il supérieur à celui de la baseline CNN de l'équipe ?

---

## 4. Hypothèse principale

Les documents authentiques présentent une distribution relativement cohérente dans l'espace ELA.
Une zone falsifiée puis resauvegardée possède un historique de compression différent, produisant une
erreur de reconstruction plus élevée lorsqu'un modèle entraîné uniquement sur documents authentiques
tente de la reconstruire.

**Hypothèse secondaire :** le contexte global des mécanismes d'attention (AnoViT) réduit les faux
positifs sur les éléments naturellement saillants (logos, tampons, zones colorées) par rapport à un
encodeur purement convolutionnel à champ réceptif local.

**Réserve théorique à garder en tête (paradoxe de l'autoencodeur) :** une meilleure reconstruction
des authentiques n'implique pas mécaniquement une meilleure localisation. Un décodeur trop expressif
peut reconstruire correctement des motifs jamais vus, y compris les falsifications, et refermer
l'écart normal/anormal. Tout gain de reconstruction doit donc être validé en gain de localisation
(AUPRC), jamais présenté seul.

---

## 5. Modèle utilisé : AnoViT encoder-decoder

### 5.1 Encoder

**Backbone :** `vit_base_patch16_384.augreg_in21k_ft_in1k`

- Type : ViT-Base/16
- Préentraînement : ImageNet-21k → ImageNet-1k
- Input : `(3 × 384 × 384)`
- Patch size : `16 × 16`
- Nombre de patchs : `24 × 24 = 576`
- Tokens latents : `(577 × 768)` — 576 patch tokens + 1 CLS token
- Output encoder : `(577 × 768)`

### 5.2 Decoder (modifié par rapport au papier)

Le décodeur prend les tokens produits par le ViT, retire le token CLS, reconstruit une carte spatiale
`24 × 24`, puis reconstruit progressivement une image `384 × 384`.

```text
Input tokens: (577 × 768)
        ↓ Remove CLS token
(576 × 768)
        ↓ Reshape
(768 × 24 × 24)
        ↓ Conv2D(768 → 512, 3×3) + ELU
        ↓ Conv2D(512 → 512, 3×3) + ELU
        ↓ Deconv2D(512 → 256, 4×4, stride=2) + BN + ELU
        ↓ Deconv2D(256 → 128, 4×4, stride=2) + BN + ELU
        ↓ Deconv2D(128 → 64,  4×4, stride=2) + BN + ELU
        ↓ Deconv2D(64  → 32,  4×4, stride=2) + BN + ELU
        ↓ Conv2D(32 → 3, 1×1)
        ↓ Sigmoid
Output: (3 × 384 × 384)
```

Le décodeur original d'AnoViT sert de point de comparaison dans l'ablation E8.

### 5.3 Schéma global

```text
Input ELA image (3 × 384 × 384)
        ↓ ViT-Base/16 encoder
Tokens (577 × 768)
        ↓ CNN decoder
Reconstructed ELA image (3 × 384 × 384)
        ↓ Reconstruction error map
Anomaly localization map
```

---

## 5bis. Méthodologie de sélection du modèle (travail réalisé + correction)

Cette section documente **a posteriori** le processus ayant abouti à la config d'AnoViT, puis la
correction apportée en v2.

**Ce qui a été fait initialement.** Le critère de sélection réel était la **qualité de localisation**
(le masque généré collait-il aux zones falsifiées ?), observée directement sur le **jeu de test réel**
falsifié. La reconstruction des authentiques n'était qu'un signal intermédiaire ; la décision finale
sur l'architecture du décodeur, la loss et les hyperparamètres regardait la localisation sur le test.

**Problème identifié.** Sélectionner l'architecture, la loss et les hyperparamètres en observant la
localisation sur le test réel constitue un **data leakage** : ce test n'est plus vierge, et les
métriques finales mesurées dessus deviennent une **borne optimiste**, pas une estimation de
généralisation. Le nombre d'époques choisi (1000) est concerné par le même biais : retenir le
checkpoint qui gagne sur le test revient à sélectionner sur le test le long de l'axe temporel.

**Correction (v2).** Toute la sélection de modèle et le pilotage de l'entraînement (dont le nombre
d'époques) sont désormais faits sur un **jeu falsifié synthétique annoté** (dev synthétique, §5ter).
Le test réel 50/50 est **regelé** et n'est évalué **qu'une seule fois**, config figée, en fin de
parcours. Cette correction est documentée honnêtement en §8.

- **Loss** : comparaison MAE / MSE / SSIM / MAE+SSIM → à re-trancher sur le dev synthétique (§E4).
- **Learning rate & scheduler** : `1e-4` + cosine + warmup.
- **Architecture décodeur** : config §5.2, comparée au décodeur original en E8.
- **Dataset** : ~10 800 documents authentiques (dont 20 % validation).

---

## 5ter. Design des données à trois niveaux

Le dispositif de données repose sur trois jeux aux rôles disjoints. Cette séparation résout à la fois
le leakage (la sélection ne touche plus le test réel) et le manque de puissance statistique (le
synthétique fournit le volume).

| Jeu | Rôle | Utilisation |
| --- | --- | --- |
| **Synthétique falsifié annoté** (étape 0, gros volume) | Sélection de modèle, ablations fines, early stopping | Utilisé librement et répété |
| **Réel 50/50 EY** (domaine cible exact) | Test de généralisation | Gelé, évalué **une seule fois** |
| **DocTamper** (falsifications documentaires externes) | Test de généralisation indépendant | Bonus, sous-échantillon, hors chemin critique |

Le dev synthétique remplit **deux fonctions simultanées** : la sélection de modèle et le pilotage de
l'early stopping par l'AUPRC de localisation. Il doit donc être prêt et fiable **avant tout
entraînement de la roadmap**. C'est le sens de l'étape 0.

> Validité externe : le triple ancrage (synthétique → réel EY → DocTamper) neutralise l'objection
> « synthétique et test réel partagent peut-être le même biais ». Si les trois concordent, le résultat
> est robuste ; s'ils divergent, la divergence est elle-même un résultat sur le domain gap.

---

# 6. Roadmap expérimentale

## Étape 0 (préalable, avant tout entraînement) — Jeu falsifié synthétique annoté

### Objectif
Produire un dev falsifié synthétique reproduisant fidèlement le scénario de double compression JPEG,
avec masques exacts, afin de servir de base à la sélection et au pilotage de l'entraînement.

### Contenu
- Pipeline de génération respectant la chaîne : rendu propre → JPEG **Q1** → réédition locale →
  JPEG **Q2** (détails techniques traités dans la discussion dédiée à la génération).
- **Q1 échantillonné selon la distribution réelle** des qualités JPEG mesurée sur les documents EY
  (authentiques d'entraînement + 50 négatifs de test).
- **Q1 et Q2 paramétrables indépendamment** (paramètre central pour E5).
- Masques de vérité terrain exacts au pixel.
- Précautions anti-artefact (alignement de grille 8×8 couvrant les cas aligné/non aligné, feather des
  bords, cohérence police/couleur) pour éviter que le modèle apprenne un « tell » du générateur.

### Livrable
Dev synthétique versionné + histogramme des qualités JPEG (réel vs synthétique) prouvant l'alignement
des domaines côté compression.

---

## Semaine 1 — Baselines, validation de l'hypothèse, calage de l'entraînement

### Objectif
Poser la comparaison de référence (B-CNN équipe vs AnoViT E0), vérifier que l'erreur de reconstruction
sépare zones authentiques et falsifiées, et **caler empiriquement le protocole d'entraînement**.

### B-CNN — Baseline équipe (référence fixe)
- Encodeur CNN existant de l'équipe, **non ré-optimisé**.
- Rôle : matérialise le « vs CNN » de la problématique. Évaluée avec le **même protocole** qu'E0.

### E0-naive — Borne inférieure sans modèle
- Seuillage statistique direct de l'ELA Q90 brute.
- Rôle : prouver que le modèle apporte un gain réel par rapport à l'ELA seule.

### E0 — AnoViT ELA Q90, config finale (contribution principale)
- Entrée/cible : ELA Q90 · entraînement : authentiques uniquement.
- Score anomalie : erreur de reconstruction.

### Calage de l'entraînement (nouveau, à faire une fois)
- Tracer la **courbe AUPRC de localisation (sur dev synthétique) en fonction des époques**.
- Attendu : montée puis stabilisation ou redescente (phénomène de sur-reconstruction, §4).
- Double vertu : justifie empiriquement le plafond d'époques, et **démontre** le paradoxe de
  l'autoencodeur sur le problème réel (petit résultat scientifique exploitable en soutenance).

### Métriques
Pixel AUPRC · AUPRO (principales) · Dice · IoU · FPR authentiques · Image AUROC.
Pixel AUROC reporté à titre indicatif seulement (gonflé par le déséquilibre, cf. §9.4).

### Livrable
Comparaison E0-naive vs B-CNN vs E0 → **première réponse à la problématique** + courbe détection vs
époques.

---

## Semaine 2 — Étude du signal ELA

### Objectif
Comprendre si la performance dépend fortement du choix de la qualité ELA.

### E1 — Ablation qualité ELA
Tester `Q75, Q85, Q90, Q95`. Pour chaque qualité : même modèle, même split, même protocole ;
seule la représentation ELA change.
> Sélection et comparaison faites sur le **dev synthétique**. Les écarts fins seront probablement dans
> le bruit sur le test réel (50 docs) : présenter comme **tendances**, pas comme un vainqueur net.

### E2 — Multi-ELA
```text
Input  = [ELA Q75, ELA Q85, ELA Q95]
Target = même représentation multi-canal
```

### Questions scientifiques
- Quelle qualité ELA donne la meilleure séparabilité ?
- Le multi-qualité est-il plus robuste qu'une qualité unique ?
- Les zones falsifiées sont-elles visibles de façon stable ou seulement à certaines qualités ?

### Livrable
Tableau comparatif : `Expérience | Représentation | AUPRC | AUPRO | Dice | IoU | FPR authentique | Commentaire`

---

## Semaine 3 — Ablations AnoViT utiles

### Objectif
Justifier les choix principaux du modèle sans transformer le mémoire en travail d'architecture.

### E3 — Régime de préentraînement (à faire)
```text
ViT-Base ImageNet — frozen
ViT-Base ImageNet — fine-tuné
ViT-Small        — from scratch
ViT-Small        — frozen / fine-tuné   (pour isoler l'effet préentraînement, voir remarque)
```
> **Confound à traiter :** comparer Base-pretrained vs Base-scratch mélange deux variables
> (préentraînement + taille). Faire au minimum **ViT-Small dans les trois régimes** pour isoler
> l'effet du seul préentraînement, ou assumer explicitement le confound comme limite.
> Question : le préentraînement ImageNet aide-t-il en domaine ELA non naturel ? Frozen suffit-il ?

### E4 — Loss et carte d'anomalie
Comparer MAE / MSE / SSIM / MAE+SSIM, **sélection sur le dev synthétique en AUPRC de localisation**
(et non plus sur la reconstruction des authentiques ni sur le test réel).
> Question scientifique : quelle erreur de reconstruction donne la meilleure localisation des
> falsifications ?

### E8 — Ablation décodeur (nouveau, contribution architecturale)
Comparer le **décodeur original AnoViT** vs le **décodeur modifié** (§5.2), à encodeur et protocole
identiques, évalués en **AUPRC de localisation** (pas en reconstruction).
> Question : la modification du décodeur améliore-t-elle réellement la localisation, ou seulement la
> reconstruction ? C'est le test qui valide (ou non) la contribution architecturale.

### Livrable
Choix final justifié : backbone / régime (E3), loss et carte d'anomalie (E4), décodeur (E8).

---

## Semaine 4 — Robustesse forensique et analyse critique

### Objectif
Tester les limites physiques de l'approche face à différents historiques JPEG.
> Note : E5, E6 et E7 sont des **évaluations du E0 déjà entraîné**, donc quasi gratuites en temps GPU.
> Elles constituent le cœur scientifique du mémoire et ne doivent jamais être sacrifiées au temps.

### E5 — Robustesse Q1/Q2
```text
Q1 > Q2   (recompression plus forte)
Q1 < Q2   (recompression plus légère)
Q1 = Q2   (cas dégénéré)
```
où Q1 = qualité du document original, Q2 = qualité de recompression après falsification.
> Cas central : dans quels scénarios l'ELA distingue-t-elle réellement une zone falsifiée ?
> Le cas Q2 < Q1 écrase souvent le signal : à générer et à présenter comme limite physique.

### E6 — Taille de falsification
`petite | moyenne | grande | très grande`
> Le bottleneck 24×24 d'AnoViT limite-t-il la localisation des petites falsifications ?

### E7 — Analyse des faux positifs
Analyser les erreurs sur authentiques : logos · tampons · signatures · zones colorées · texte dense ·
bruit de scan · bordures · artefacts de compression naturels.
> Lien direct avec l'hypothèse secondaire (§4) : AnoViT réduit-il ces FP vs B-CNN ?

### Livrable
Tableau robustesse Q1/Q2 · tableau perf par taille · analyse qualitative des FP · discussion honnête des limites.

---

# 7. Expériences principales retenues

| ID       | Expérience                   | Rôle                                      |
| -------- | ---------------------------- | ----------------------------------------- |
| B-CNN    | Baseline CNN équipe          | Référence fixe (« vs CNN »)               |
| E0-naive | Seuillage ELA brute          | Borne inférieure (le modèle sert-il ?)    |
| **E0**   | **AnoViT ELA Q90 (config finale)** | **Contribution principale**         |
| E1       | ELA Q75/Q85/Q90/Q95          | Sensibilité à la qualité ELA              |
| E2       | Multi-ELA                    | Représentation forensique enrichie        |
| E3       | Frozen / fine-tuné / scratch | Justification du backbone                 |
| E4       | MAE/MSE/SSIM/MAE+SSIM        | Choix de la carte d'anomalie              |
| **E8**   | **Décodeur original vs modifié** | **Contribution architecturale**       |
| E5       | Robustesse Q1/Q2             | Test forensique central                   |
| E6       | Taille de falsification      | Limite de résolution                      |
| E7       | Faux positifs                | Analyse industrielle et critique          |

---

# 8. Expériences non prioritaires et limites méthodologiques

**Non prioritaires** (section *Limites et perspectives*, pas nécessaires pour défendre le mémoire) :
implémentation de TruFor · CAT-Net · U-Net supervisé complet · fusion RGB avancée · late fusion
complexe · ablations lourdes du décodeur · dégradations combinées avancées.

**Limites méthodologiques à assumer explicitement :**

1. **Data leakage de sélection (corrigé).** La première phase de sélection (décodeur, loss, époques) a
   observé la localisation sur le test réel. Correction : toute sélection déplacée sur le dev
   synthétique, test réel regelé et évalué une seule fois (§5bis, §5ter). À présenter honnêtement.
2. **Puissance statistique du test réel.** 50 falsifiés annotés = unité statistique document faible ;
   IC bootstrap larges. Les comparaisons fines (qualité ELA) sont des tendances, pas des verdicts.
   L'image-AUROC (100 points) est la métrique la plus solide sur le test réel.
3. **Réalisme du synthétique.** Les falsifications synthétiques peuvent être moins réalistes que les
   vraies. C'est précisément pourquoi les 50 réels restent le juge final, et DocTamper un contrôle
   externe.
4. **Confound frozen/scratch** si ViT-Small n'est pas testé dans les trois régimes (§E3).

---

# 9. Protocole expérimental figé

## 9.1 Configuration globale
```yaml
seed: 42
seeds_key_comparisons: [42, 1337, 2024]   # multi-seed sur B-CNN vs E0 (comparaison centrale)
epochs_max: 100                            # plafond ; arrêt piloté par AUPRC dev (9.3bis)
early_stopping_patience: 15
batch_size: 48
optimizer: AdamW
lr: 1.0e-4
weight_decay: 1.0e-2
scheduler: cosine
warmup_epochs: 5
input_res: 384
patch_size: 16
latent_grid: 24x24
ela_quality_baseline: 90
mixed_precision: true
```

## 9.2 Règle de split (prioritaire sur tout le reste)
```text
- Documents AUTHENTIQUES  -> train / val = 80 / 20 (split au niveau document source).
- DEV SYNTHÉTIQUE falsifié annoté -> sélection de modèle, ablations, pilotage early stopping.
- TEST RÉEL = 50 falsifiés annotés + 50 authentiques tenus à part (FPR et image-AUROC).
             -> gelé, évalué UNE SEULE FOIS, config figée.
- TEST EXTERNE (bonus) = sous-ensemble DocTamper.
- Aucune version d'un même document source ne doit apparaître dans plusieurs splits.
```

## 9.3 Sélection du seuil
Le dev synthétique contient des pixels falsifiés : le seuil peut donc être **calibré en localisation**
(p. ex. seuil maximisant le Dice, ou point de fonctionnement cible sur la courbe précision/rappel)
sur le dev synthétique, puis figé.
```text
1. Entraîner sur train (authentiques).
2. Calibrer le seuil sur le DEV SYNTHÉTIQUE falsifié (jamais sur le test réel).
3. Sauvegarder le seuil.
4. Évaluer UNE SEULE FOIS sur le test réel.
```
> Complément possible : seuil statistique sur authentiques (percentile p99 / p99.5 ou mean + k·σ) comme
> point de comparaison. Le seuil n'est **jamais** choisi sur le test réel.

## 9.3bis Pilotage de l'entraînement (nouveau)
```text
- Métrique de suivi = AUPRC de localisation sur le DEV SYNTHÉTIQUE (à chaque époque).
- Early stopping et sélection du checkpoint = "best-detection" (max AUPRC dev), PAS le dernier.
- Ne PAS piloter sur la loss de reconstruction des authentiques seule
  (converge ≠ détection maximale, cf. paradoxe §4).
```

## 9.4 Métriques finales
```text
PRINCIPALES (threshold-free, robustes au déséquilibre) :
  Pixel AUPRC
  AUPRO           # per-region overlap, standard localisation d'anomalies
OPÉRATIONNELLES (au seuil figé, pour l'usage réel) :
  Dice
  IoU
  FPR sur documents authentiques
GLOBALE :
  Image AUROC     # 50 pos / 50 neg au test réel : métrique la plus solide statistiquement
INDICATIVE SEULEMENT :
  Pixel AUROC     # gonflé par le déséquilibre pixel, ne pas en faire l'argument central
```

### Rôle de chaque métrique

| Catégorie | Métrique | Rôle |
| --- | --- | --- |
| Principale (threshold-free) | **Pixel AUPRC** | Robuste au déséquilibre pixel (zones falsifiées minuscules vs document). Métrique de pilotage de l'early stopping (§9.3bis) et de sélection dans toutes les ablations. |
| Principale (threshold-free) | **AUPRO** | Per-region overlap : évite qu'une grosse région détectée compense de nombreuses petites régions ratées. |
| Opérationnelle (seuil figé) | **Dice** | Qualité du masque binaire prédit. |
| Opérationnelle (seuil figé) | **IoU** | Complémentaire du Dice sur le masque binaire. |
| Opérationnelle (seuil figé) | **FPR authentiques** | Point critique industriel : faux positifs sur logos/tampons/éléments saillants (problème d'origine du B-CNN, cf. §14 et E7). |
| Globale | **Image AUROC** | Décision document falsifié/authentique. Sur le test réel 50/50, la plus solide statistiquement (100 points au niveau document). |
| Indicative | **Pixel AUROC** | Gonflée mécaniquement par le déséquilibre pixel. Reportée par transparence, jamais comme argument central. |

> Usage pratique : sélection/comparaison en **Pixel AUPRC** (dev synthétique) ; contribution défendue
> avec **AUPRC + AUPRO + IC** ; utilisabilité montrée avec **Dice/IoU/FPR au seuil figé** ; robustesse
> statistique ancrée par l'**Image AUROC** sur le test réel. Format du tableau livrable :
> `Expérience | Représentation | AUPRC | AUPRO | Dice | IoU | FPR authentique | Commentaire`.

## 9.5 Intervalles de confiance
```text
bootstrap non paramétrique au niveau document
IC 95% sur AUPRC, AUPRO, Dice et FPR
multi-seed (9.1) prioritairement sur B-CNN vs E0 ; mono-seed ailleurs si le temps manque
Attendu : IC larges sur le test réel (50 docs) -> conclusions calibrées en conséquence
```

---

# 10. Structure des résultats

```text
experiments/{ID}/config.yaml
experiments/{ID}/training.log
experiments/{ID}/best_model.pt
experiments/{ID}/metrics.json
experiments/{ID}/threshold.json
experiments/{ID}/anomaly_maps/
experiments/{ID}/qualitative/
results/master_table.csv
```

Colonnes minimales de `master_table.csv` :
```text
ID, input, backbone, regime, loss, decoder, anomaly_map,
pixel_AUPRC, AUPRO, Dice, IoU, FPR_authentic, image_AUROC, pixel_AUROC,
eval_set, seed, CI95_AUPRC, notes
```
> `eval_set` distingue dev synthétique / test réel / DocTamper pour éviter toute confusion de source.

---

# 11. Contribution attendue du mémoire

La contribution n'est pas une nouvelle famille d'architectures, mais une étude rigoureuse de :

1. la capacité d'un autoencodeur ViT à apprendre la distribution normale des cartes ELA ;
2. **le gain réel d'AnoViT face à la baseline CNN de l'équipe et à un seuillage ELA naïf** ;
3. **l'apport de la modification du décodeur (E8), validé en localisation et non en reconstruction** ;
4. l'impact de la qualité ELA (et du multi-ELA) sur la localisation des falsifications ;
5. l'effet du régime de préentraînement (frozen / fine-tuné / scratch) en domaine forensique non naturel ;
6. la robustesse face aux différents scénarios de recompression JPEG (Q1/Q2) et tailles de falsification ;
7. les limites pratiques : faux positifs et petites falsifications ;
8. **un protocole de validation à trois niveaux** (synthétique contrôlé, réel annoté, DocTamper externe)
   garantissant la validité externe des conclusions.

---

# 12. Priorités si le temps manque

Ne jamais sacrifier :
```text
1. splits propres + test réel gelé (9.2)
2. étape 0 (dev synthétique)                <-- débloque sélection, seuil ET pilotage époques
3. E0 (AnoViT) + B-CNN (baseline équipe)    <-- répond à la problématique
4. E8 (décodeur original vs modifié)        <-- valide la contribution architecturale
5. ablation qualité ELA (E1)
6. régime backbone : au minimum frozen vs scratch (E3)
7. robustesse Q1/Q2 (E5) + faux positifs (E7)   <-- quasi gratuits, coeur scientifique
```
Multi-ELA (E2) et multi-seed complet sont importants mais peuvent passer après si le temps est
vraiment limité. E5/E6/E7 étant des évaluations du E0 déjà entraîné, elles ne consomment presque pas
de temps GPU : les protéger.

---

# 13. Conclusion

Le mémoire est une étude de **localisation forensique JPEG basée sur ELA**, avec AnoViT (décodeur
modifié) comme modèle principal d'anomalie non supervisée, **comparé rigoureusement à la baseline CNN
de l'équipe**, sélectionné sans leakage sur un dev synthétique, et validé à trois niveaux (synthétique,
réel EY annoté, DocTamper). Ce cadrage est réaliste, cohérent avec le temps disponible, et
scientifiquement défendable.

---

# 14. Contexte du mémoire

Je travaille sur un projet de détection de modifications dans des documents numérisés en exploitant les différences de niveaux de compression JPEG. Le scénario étudié est le suivant : un fraudeur modifie un document déjà scanné puis le réenregistre, ce qui introduit une nouvelle compression JPEG sur tout ou partie de l’image. Cette recompression crée des incohérences dans les artefacts de compression entre les régions authentiques et les régions altérées. L’objectif est d’exploiter ces différences, notamment à travers l’Error Level Analysis (ELA), afin de localiser automatiquement les zones modifiées.

La problématique initiale reposait sur une approche classique utilisant des cartes ELA associées à un modèle CNN. Cependant, cette méthode générait de nombreux faux positifs, notamment sur certains éléments naturellement saillants du document (couleurs, logos, tampons ou éléments graphiques). L’hypothèse de recherche a donc été d’étudier l’apport des architectures Transformer, capables de modéliser des dépendances contextuelles à plus grande échelle grâce aux mécanismes d’attention, afin de mieux distinguer les véritables anomalies des variations normales présentes dans les documents.

Après une étude de la littérature, le choix s’est porté sur ANO-ViT, une approche basée sur un encodeur Transformer. Le modèle a été implémenté puis adapté à la problématique spécifique de la détection de falsifications documentaires à partir de cartes ELA. En parallèle, un pipeline complet de préparation des données a été développé ainsi qu’un jeu de données dédié construit à partir de multiples sources publiques. Le modèle est entraîné uniquement sur environ 10 800 documents authentiques (dont 20 % réservés à la validation), conformément au principe de détection d’anomalies d’ANO-ViT.

L’évaluation est réalisée sur un jeu de test indépendant composé d’images falsifiées et annotées manuellement, pour lesquelles les zones modifiées sont localisées avec précision. Le principe consiste à reconstruire les cartes ELA à l’aide de l’architecture encodeur-décodeur puis à comparer l’entrée et la reconstruction afin de produire une carte d’erreur. Une méthode est considérée pertinente si les zones présentant les plus fortes erreurs de reconstruction coïncident majoritairement avec les régions réellement falsifiées. Cette approche permet ainsi d’évaluer quantitativement la capacité du modèle à localiser les altérations liées aux différences de compression.

Bien que l’implémentation du modèle, la construction du jeu de données et le pipeline de traitement soient terminés, un mémoire de recherche ne peut pas se limiter à la présentation d’un système fonctionnel. Il est nécessaire d’apporter une validation expérimentale structurée permettant de répondre scientifiquement à la problématique initiale, à savoir l’intérêt des architectures Transformer pour la localisation d’altérations de compression dans les documents par rapport à une approche CNN plus classique. Afin d’obtenir des résultats suffisamment solides pour être analysés et discutés dans le mémoire, un plan expérimental sur quatre semaines a été défini. Ce plan vise à étudier de manière méthodique l’impact de plusieurs choix architecturaux et méthodologiques, notamment le type d’encodeur (entraînement depuis zéro ou pré-entraînement ImageNet), l’apport réel du contexte global fourni par les Transformers, la pertinence des signaux ELA utilisés en entrée ainsi que la robustesse du modèle face à différents scénarios de compression et de falsification. Les résultats obtenus constitueront la base de l’analyse scientifique et des conclusions du mémoire.