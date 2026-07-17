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

# --- Compression (Q1 = historique avant fraude ; Q2 = compression finale) ----
Q2_SWEEP=(55 70 85 95)                       # couvre Q2<Q1, Q2≈Q1, Q2>Q1
Q1_MODE=auto                                 # native | controlled | auto
Q1_AUTO_Q0_THRESHOLD=95                      # Q0 médian >= seuil -> controlled (mode auto)
Q1_SWEEP=(55 70 85)                          # utilisé si controlled ; ⊆ Q2_SWEEP

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

# --- Classes de taille de zone (fraction de surface page : MIN MAX) ----------
SIZE_SMALL=(0.001 0.005)                     # ~0.1% – 0.5% de la page
SIZE_MEDIUM=(0.005 0.02)                     # 0.5% – 2%
SIZE_LARGE=(0.02 0.06)                       # 2% – 6%
SIZE_VERY_LARGE=(0.06 0.15)                  # 6% – 15%

# --- Négatifs (authentiques, masque vide) : part DANS CHAQUE sous-dossier -----
NEGATIVES_RATIO=0.3                          # 0.0 = uniquement des falsifications
KEEP_BENIGN_COLORED=true                     # préserve logos/tampons/en-têtes

# --- Annotation (grille patch pour le modèle en aval) ------------------------
INPUT_RES=384
PATCH_SIZE=16
PATCH_GRID=24
PATCH_POSITIVE_OVERLAP=0.5                   # patch positif si recouvrement > seuil

# --- QA visuel (planches ELA) ------------------------------------------------
ELA_QUALITY=90                               # qualité ELA d'aperçu, DISTINCTE de Q2
ELA_N_SAMPLES=50                             # nb de planches image | ELA | masque

# --- Orchestration -----------------------------------------------------------
SEED=42                                      # seed global (reproductibilité)
N_DOCS=2000                                  # nb de documents PAR TYPE
N_WORKERS=4                                  # parallélisme

# --- Interpréteur Python (celui qui a les dépendances) -----------------------
# Laisse "python" si conda/venv est déjà activé. Mets un chemin absolu sinon.
PYTHON="${PYTHON:-python}"
