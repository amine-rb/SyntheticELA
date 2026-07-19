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
SOURCE_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/StaVer/scans/scans"   # dossier des images sources
OUTPUT_DIR="/Users/amine_rb/Desktop/Master IASD/coding/SyntheticEla/StaVer/scans/fraud"    # racine des sorties

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
QUALITY_SWEEP=(92 95 97)
# Écart de compression : Q1 = Q2 - Q1_GAP (borné >= 40). C'est LE bouton du signal.
# Mesuré sur ce corpus (forgé/texte-authentique, la métrique qui compte) :
#   gap 22 -> ~1.54 | gap 28 -> ~1.76 | gap 32 -> ~1.73 (plafonne).
# Trop petit -> signal faible ; trop grand -> blocking Q1 visible sur tout le
# document (le "document original" paraît dégradé). 28 = bon compromis.
Q1_GAP=28

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

# --- Classes de taille de zone (fraction de surface page : MIN MAX) ----------
SIZE_SMALL=(0.001 0.005)                     # ~0.1% – 0.5% de la page
SIZE_MEDIUM=(0.005 0.02)                     # 0.5% – 2%
SIZE_LARGE=(0.02 0.06)                       # 2% – 6%
SIZE_VERY_LARGE=(0.06 0.15)                  # 6% – 15%

# --- Négatifs (authentiques, masque vide) : part DANS CHAQUE sous-dossier -----
NEGATIVES_RATIO=0.0                          # 0.0 = uniquement des falsifications
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
# ELA_QUALITY = qualité CENTRALE. Choix CRITIQUE : viser ≈ Q1 (= médiane(Q2)−Q1_GAP),
# le point fixe de compression du fond. Sonder à Q1 met le texte AUTHENTIQUE (qui a
# l'historique Q1) à son minimum d'ELA tandis que la zone falsifiée (jamais vue à Q1)
# explose -> contraste ~2× meilleur ET zone plus vive qu'à Q90.
#   Mesuré : centre @90 -> forgé/auth 1.8 | centre @Q1(67) -> 3.2, zone 2.5× plus vive.
# Les 3 qualités doivent rester DISTINCTES de tout Q2 du sweep (sinon ELA nulle). Ici
# Q2∈{92,95,97}, Q1_GAP=28 -> Q1∈{64,67,69} -> centre 67. Si tu changes sweep/gap,
# l'orchestrator affiche la valeur recommandée et alerte si tu es trop loin de Q1.
ELA_QUALITY=67                               # centre ≈ Q1 (point fixe du fond) -> contraste max
ELA_SPREAD=8                                 # écart des 3 canaux : (67-8, 67, 67+8) = 59/67/75
ELA_N_SAMPLES=50                             # nb de planches image | ELA | masque
# Échelle GLOBALE FIXE de l'ELA d'aperçu (pas d'étirement par max d'image, qui
# écrasait les fraudes faibles). Aligne-la sur detection_eval.ELA_SCALE (=15).
ELA_SCALE=15

# --- Orchestration -----------------------------------------------------------
SEED=42                                      # seed global (reproductibilité)
N_DOCS=20                                  # nb de documents PAR TYPE
N_WORKERS=4                                  # parallélisme

# --- Interpréteur Python (celui qui a les dépendances) -----------------------
# Laisse "python" si conda/venv est déjà activé. Mets un chemin absolu sinon.
PYTHON="${PYTHON:-python}"
