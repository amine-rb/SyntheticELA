CONTEXTE
Je construis un dataset synthétique de falsifications documentaires pour un mémoire de
recherche en détection de falsifications par Error Level Analysis (ELA) + Vision
Transformer. Le modèle en aval est un autoencodeur (encodeur ViT-Base/16, décodeur
convolutionnel) entraîné en détection d'anomalie ; il prend une JPEG en entrée. Ta tâche
ici n'est PAS le modèle : c'est UNIQUEMENT le pipeline de génération de données.

Je pars de JPEG AUTHENTIQUES récupérées sur Kaggle (documents réels non falsifiés).
À partir d'elles, je veux générer des documents falsifiés annotés au pixel, reproduisant
le vrai scénario forensique.

INSIGHT CENTRAL À NE PAS CASSER
Une JPEG Kaggle est DÉJÀ compressée une fois, à une qualité Q0 inconnue mais inscrite
dans le fichier (table de quantification). Le scénario est donc :
- Fond du document : Q0 (lu dans le fichier) -> resauvegardé en Q2 = DOUBLE COMPRESSION.
- Zone éditée : historique de compression DIFFÉRENT du fond selon le type d'édition.
Le signal que l'ELA doit révéler est cette INCOHÉRENCE D'HISTORIQUE DE COMPRESSION entre
la zone falsifiée et le fond. Tout le reste (plausibilité sémantique du texte) est
secondaire.

RÈGLES IMPÉRATIVES SUR LA COMPRESSION
- Q0 n'est JAMAIS choisi : il est LU dans le fichier source (PIL Image.quantization,
  subsampling, dimensions) et journalisé. Ne le réinvente pas.
- Q2 est le SEUL paramètre de compression que je fais varier. Balayage par défaut :
  {55, 70, 85, 95}, exposé en config, pour couvrir Q2<Q0, Q2≈Q0, Q2>Q0.
- La zone éditée subit la MÊME passe Q2 que le fond. Elle n'est JAMAIS collée après le
  save final. Si tu colles une zone après la compression finale, le dataset est faux.

TYPES D'ÉDITION ET GRILLE JPEG 8x8 (déterminant pour l'ablation alignement)
- Substitution de texte peinte (pixels neufs redessinés sur l'image décodée) : zone vue
  UNE SEULE FOIS au save Q2, pas de grille 8x8 antérieure -> alignment = "N/A".
- Copy-move interne (zone recopiée depuis la MÊME image Q0) : porte la grille Q0.
  Offset de collage NON multiple de 8 -> désaligné (cas facile) ; multiple de 8 ->
  aligné (cas difficile).
- Splice (zone venue d'un AUTRE JPEG) : porte la grille de sa source ; même contrôle
  d'alignement via l'offset. Par défaut splice INTRA-CORPUS (autre doc Kaggle du lot),
  pour garantir un historique JPEG propre côté source sans dépendance externe.
L'ablation aligné/non-aligné repose donc sur copy-move et splice ; la substitution reste
en "N/A". Je veux les deux régimes équilibrés, pas un biais vers l'un.

ANTI-"TELL" DU GÉNÉRATEUR (le modèle doit apprendre le signal de compression, pas ma
signature de générateur)
- Feather gaussien léger des bords de la zone éditée (rayon tiré dans [0.5, 2] px).
- Couleur/texture de la zone échantillonnée localement sous la zone éditée.
- Police cohérente avec le document pour la substitution ; pas de bordure nette
  systématique ; cohérence résolution/couleur zone insérée vs reste.

VÉRITÉ TERRAIN
- Masque binaire pixel-level = empreinte géométrique EXACTE, SANS dilatation (le feather
  est un anti-tell, pas "plus de falsification").
- Fournir aussi une grille de labels patch-level : patch 16 px -> 24x24, label positif
  si recouvrement du masque au-dessus d'un seuil (expose le seuil, défaut 0.5).

NÉGATIFS (important pour une tâche de LOCALISATION)
- Générer aussi des AUTHENTIQUES resauvegardés Q0->Q2 SANS édition (mêmes Q2), pour
  empêcher le modèle d'apprendre la double compression GLOBALE au lieu de LOCALISER
  l'incohérence. Masque vide.
- Conserver/injecter des éléments bénins colorés (logos, tampons, en-têtes) dans le fond
  des faux ET des négatifs : c'est un problème structurel connu (l'ELA d'une zone
  colorée authentique est hétérogène). Sans ça le modèle apprendra "coloré = anomalie".

FORMAT DE SORTIE (par document)
  <id>.jpg          doc final (fond Q0->Q2, zone à historique incohérent)
  <id>_mask.png     masque binaire pixel-level exact
  <id>.json         {Q0_lu, table_quant, subsampling, Q2, type, size_class,
                     alignment, bbox, seed, source_id}
Plus : manifest.parquet global + distribution.json (caractérisation de Q0 sur le corpus).

ARCHITECTURE ATTENDUE (modules découplés, orchestrés par config)
1. jpeg_probe    : lit Q0/table quant/subsampling/dimensions sur le dossier source,
                   produit distribution.json. Filtre pour ne garder QUE des sources JPEG
                   (une PNG n'a pas de Q0 : elle casserait le mismatch -> exclue, avec log).
2. forger        : applique substitution / copy-move / splice avec classe de taille,
                   alignement, feather, cohérence photométrique. Retourne image éditée +
                   empreinte géométrique.
3. recompress    : resauve en JPEG Q2 (faux ET négatifs authentiques).
4. annotator     : masque pixel + bbox + grille patch 24x24 + métadonnées JSON.
5. orchestrator  : boucle batch, seeds déterministes, tirage des paramètres,
                   parallélisation, manifeste global. Génération par lot 100% scriptable,
                   sans intervention manuelle (cible : centaines à quelques milliers de docs).
6. ela_preview   : QA hors entraînement, calcule l'ELA sur quelques échantillons pour
                   contrôle visuel, à une qualité ELA DOCUMENTÉE et DISTINCTE de Q2.

CLASSES DE TAILLE DE ZONE
Quatre classes (petite / moyenne / grande / très grande), définies en fraction de surface
page ET calées sur des multiples de 8 px. Expose les seuils exacts en config.

EXIGENCES D'INGÉNIERIE
- Config-driven (un fichier de config : chemins, Q2, ratios de types d'édition, ratio
  aligné/non-aligné, distribution des tailles, proportion de négatifs, seed global).
- Reproductible : seed déterministe par document, journalisé dans le JSON.
- Bibliothèques par défaut : Pillow (I/O JPEG, accepte qtables/subsampling), NumPy,
  OpenCV pour le compositing. Signale-moi tout autre choix.
- Code structuré, commenté, un module par fichier.

CE QUE J'ATTENDS
- Commence par jpeg_probe (je veux voir la distribution de Q0 de mon corpus avant tout),
  puis forger, puis les autres modules.
- Signale EXPLICITEMENT chaque valeur par défaut (seuils, tailles de patch, ratios) pour
  que je la valide.
- Ne suppose PAS l'existence de fichiers que je n'ai pas donnés. AVANT de lancer un batch,
  demande-moi le chemin du dossier des JPEG Kaggle et confirme le filtrage PNG. Pose une
  question si un détail manque plutôt que d'inventer.