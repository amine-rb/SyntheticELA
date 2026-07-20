# =============================================================================
# config.sh — UNIQUE fichier de paramètres du pipeline.
# =============================================================================
# Édite CE fichier, puis lance :
#     ./scripts/run.sh                 # génération (un sous-dossier par type)
#     ./scripts/aggregate.sh           # fusion des sous-dossiers -> _aggregated/
#     ./scripts/preview.sh             # planches QA image | ELA | masque
#
# Aucun autre fichier à toucher : les scripts .sh génèrent la config interne à
# partir des variables ci-dessous. Tout override ponctuel se passe en CLI
# (ex. ./scripts/run.sh --n 500 --workers 8).
# =============================================================================

# --- Chemins -----------------------------------------------------------------
# SOURCE_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/StaVer/scans/scans"   # dossier des images sources
# OUTPUT_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/StaVer/scans/fraud"   # dossier des images sources
# SOURCE_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/SROIE2019/train/img"   # dossier des images sources
# OUTPUT_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/SROIE2019/train/fraud"  
# SOURCE_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/NoisyMed/bills"   # dossier des images sources
# OUTPUT_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/NoisyMed/bills_fraud"    # racine des sorties
SOURCE_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/NoisyMed/discharge_summaries"   # dossier des images sources
OUTPUT_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/NoisyMed/discharge_summaries_fraud"    # racine des sorties

# --- Sonde du corpus ---------------------------------------------------------
PROBE_RECURSIVE=true                         # parcourt les sous-dossiers
CANDIDATE_EXT=(.jpg .jpeg .jpe .jfif .png .tif .tiff .bmp)   # extensions acceptées
ALLOW_LOSSLESS=true                          # garder PNG/TIFF/BMP (pas de Q0)
NONSTANDARD_ABSDIFF_THRESHOLD=40             # seuil "qtable non standard"

# --- Compression (qualité JPEG UNIQUE par document) --------------------------
# DEUX passes de qualités DIFFÉRENTES par document (Q1 < Q2). C'est l'écart qui
# crée le signal ELA de la substitution :
#   1) recompression de la source à Q1 (qualité MOYENNE) -> "document original"
#      (fond + texte authentique portent la grille/quantification Q1),
#   2) substitution peinte en pixels NEUFS (n'a JAMAIS vu Q1),
#   3) sauvegarde finale de TOUT à Q2 (qualité HAUTE).
# => Le texte AUTHENTIQUE porte l'historique Q1->Q2 ; la zone falsifiée n'a que Q2.
#    Sondée en ELA (qualité != Q2), la zone falsifiée RESSORT nettement du texte
#    ordinaire (mesuré : forgé/authentique ~2.6x). ATTENTION : en Q1==Q2 la
#    falsification est INDISCERNABLE du texte authentique (ratio ~1.0) -> pas de
#    signal exploitable. L'écart Q1<Q2 est donc IMPÉRATIF.
# QUALITY_SWEEP = valeurs de Q2 (sauvegarde FINALE, hautes). Chaque doc en tire une.
# Elles DOIVENT toutes être != ELA_QUALITY (sinon toute l'image est au point fixe
# de la sonde et l'ELA s'effondre à 0 partout).
QUALITY_SWEEP=(90 93 96)
# Écart de compression : Q1 = Q2 - Q1_GAP (borné >= 40). C'est LE bouton du signal.
# OPTION A (robustesse au Q1 d'inférence) : Q1_GAP est une PLAGE (min max) -> un gap
# est tiré PAR DOCUMENT -> Q1 varie sur toute une bande au lieu d'une valeur unique.
# Le modèle n'apprend donc PAS un Q1 fixe (ex. 67) et généralise à des documents
# reçus à l'inférence dont la qualité de base diffère. Avec sweep (90 93 96) :
#   Q1 ∈ [min(sweep)-max_gap, max(sweep)-min_gap] = [90-40, 96-16] = [50, 80].
# Un scalaire (ex. Q1_GAP=28) reste accepté = Q1 quasi fixe (ancien comportement).
# Plancher à 20 : sous ~20 le signal est trop faible (mesuré 22->~1.5). Avec sweep
# (90 93 96) et gap ∈ [20,40] -> Q1 ∈ [90-40, 96-20] = [50, 76].
Q1_GAP=(20 40)

# --- Falsification -----------------------------------------------------------
# Types à générer : UN SOUS-DOSSIER COMPLET ET SÉPARÉ par type (aucun tirage
# aléatoire entre types). Ajoute/retire des entrées librement.
EDIT_TYPES=(substitution)                    # ex. (substitution copy_move splice)
ALIGNED_RATIO=0.5                            # part alignée grille 8x8 (copy_move + splice)
FEATHER_RADIUS_PX=(0.5 2.0)                  # adoucissement des bords (anti-tell)
SPLICE_SOURCE=intra_corpus                   # donneur = autre doc du même corpus
# Taille MINIMALE garantie du rectangle falsifié : (largeur_min hauteur_min) px,
# arrondie au multiple de 8 SUPÉRIEUR -> (10 10) garantit >= 16x16. Empêche tout
# masque à 0 pixel ; une source trop petite échoue proprement (erreur journalisée).
MIN_REGION_PX=(10 10)
# Nombre de falsifications PAR DOCUMENT positif : (min max). Chaque doc reçoit
# k ~ tirage uniforme dans {min..max}, puis k falsifications du MÊME type. Plus k
# est grand, plus les zones sont PETITES (plafond de taille auto) pour ne pas
# couvrir toute la page : ex. k=5 -> uniquement des zones "small". (1 1) = une seule.
N_FORGERIES=(1 5)
# Placer les falsifications SUR du contenu réel (texte/chiffres) au lieu du vide
# -> zones réalistes et porteuses de signal ELA (pas d'aplat blanc invisible).
# MIN_CONTENT_FRAC = fraction min de pixels "encre" visée dans la zone (best-effort ;
# retombe sur le meilleur emplacement si la page est presque vide).
PLACE_ON_CONTENT=true
MIN_CONTENT_FRAC=0.02
# Couleur de l'encre injectée par la substitution. Fraction des substitutions
# rendues EN COULEUR (couleur saturée tirée au hasard, différente à chaque fois) ;
# le reste est en encre sombre (quasi-noire) comme le texte du document.
#   0.0 = tout en noir (défaut historique)  |  0.5 = moitié couleur / moitié noir.
# But : que le modèle apprenne à détecter la fraude quelle que soit sa couleur.
# ATTENTION : une fraude colorée est EFFACÉE par le filtre anti-couleur de l'ELA
# (ELA_GRAYSCALE_INPUT / ELA_CHROMA_SUPPRESS). Pour détecter les fraudes colorées,
# METS CE FILTRE À OFF (ELA_GRAYSCALE_INPUT=false, ELA_CHROMA_SUPPRESS=0) et laisse
# l'autoencodeur gérer les logos via la nouveauté (cf. §7 README).
SUBST_COLOR_PROB=0.5

# --- Classes de taille de zone (fraction de surface page : MIN MAX) ----------
SIZE_SMALL=(0.001 0.005)                     # ~0.1% – 0.5% de la page
SIZE_MEDIUM=(0.005 0.02)                     # 0.5% – 2%
SIZE_LARGE=(0.02 0.06)                       # 2% – 6%
SIZE_VERY_LARGE=(0.06 0.15)                  # 6% – 15%

# --- Négatifs (authentiques, masque vide) : part DANS CHAQUE sous-dossier -----
NEGATIVES_RATIO=0                        # 0.0 = uniquement des falsifications
KEEP_BENIGN_COLORED=true                     # préserve logos/tampons/en-têtes

# --- Annotation (grille patch pour le modèle en aval) ------------------------
INPUT_RES=384
PATCH_SIZE=16
PATCH_GRID=24
PATCH_POSITIVE_OVERLAP=0.5                   # patch positif si recouvrement > seuil

# --- ELA (sortie 1re classe RGB + planches QA) -------------------------------
# L'ELA de sortie (ela/*.png) est une IMAGE COULEUR RGB : 3 canaux = 3 qualités de
# sonde (ELA_QUALITY-ELA_SPREAD, ELA_QUALITY, ELA_QUALITY+ELA_SPREAD). La "couleur"
# vient de la DIVERSITÉ de qualité (pas de la chroma) -> 3 canaux d'info pour le
# modèle (mode E2 de detection_eval). La zone falsifiée réagit fortement aux 3 sondes.
#
# ELA_QUALITY / ELA_SPREAD = sonde FIXE (mêmes 3 qualités pour TOUS les docs, comme à
# l'inférence où l'on ignore le Q1 du document). Les 3 canaux RGB = (ELA_QUALITY-SPREAD,
# ELA_QUALITY, ELA_QUALITY+SPREAD), tous DISTINCTS de tout Q2 du sweep (sinon ELA -> 0).
# IMPORTANT (mesuré, option A) : même quand Q1 varie sur [50,76], une sonde ÉTROITE à
# 67 reste la MEILLEURE — la fraude (Q2 seul) ressort à beaucoup de qualités, pas
# seulement à Q1 pile. Inutile d'élargir la sonde ; la robustesse au Q1 d'inférence
# vient de la DIVERSITÉ de Q1 dans les DONNÉES (Q1_GAP plage), pas de la sonde.
#   Grille sur données Q1∈[50,80] : 59/67/75 -> moy 2.04 | 57/65/73 -> 1.97 |
#   53/63/73 -> 1.86 | 55/67/79 -> 1.86 | 50/65/80 -> 1.74. => 67/8 gagne.
ELA_QUALITY=67                               # sonde fixe optimale (grille) même à Q1 variable
ELA_SPREAD=8                                 # canaux 59/67/75
ELA_N_SAMPLES=50                             # nb de planches image | ELA | masque
# Échelle GLOBALE FIXE de l'ELA d'aperçu (pas d'étirement par max d'image, qui
# écrasait les fraudes faibles). Aligne-la sur detection_eval.ELA_SCALE (=15).
ELA_SCALE=15
# --- Réduction des FAUX POSITIFS colorés (logos/tampons/cachets) --------------
# Deux leviers CUMULABLES. La substitution est du texte NOIR ; le mobilier
# authentique coloré brille autant en ELA -> faux positifs. Valides tant que la
# fraude est achromatique (substitution) ; à désactiver si on falsifie du coloré.
#
# 1) Gris AVANT l'ELA : effondre l'ELA du mobilier coloré CLAIR (mesuré 74 -> ~8),
#    la fraude garde ~99 %. Ne suffit pas seul sur le coloré FONCÉ. true/false.
ELA_GRAYSCALE_INPUT=false
# 2) Suppression chroma APRÈS l'ELA : efface les pixels colorés quelle que soit leur
#    luminosité (chroma > seuil -> 0), donc rattrape le coloré foncé. Seuil ~20 :
#    logo/tampon -> 0, fraude garde ~87 %. 0 = désactivé.
ELA_CHROMA_SUPPRESS=0

# --- Orchestration -----------------------------------------------------------
SEED=42                                      # seed global (reproductibilité)
N_DOCS=20                                  # nb de documents PAR TYPE
N_WORKERS=4                                  # parallélisme

# --- Interpréteur Python (celui qui a les dépendances) -----------------------
# Laisse "python" si conda/venv est déjà activé. Mets un chemin absolu sinon.
PYTHON="${PYTHON:-python}"
