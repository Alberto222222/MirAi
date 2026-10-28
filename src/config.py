"""Configurazione centrale del progetto MirAi.

Tutti i parametri che influenzano i risultati stanno qui, così sono
documentati in un solo punto e si possono citare nel report.
"""

# --- Classi -----------------------------------------------------------------
# Ordine fisso: è anche l'ordine degli indici della softmax.
CLASSES = ["Romanesque", "Gothic", "Renaissance", "Baroque"]

# Nomi in italiano per l'app e le figure.
CLASS_NAMES_IT = {
    "Romanesque": "Romanico",
    "Gothic": "Gotico",
    "Renaissance": "Rinascimentale",
    "Baroque": "Barocco",
    "Classical": "Classico antico",
    "Neoclassical": "Neoclassico",
}

# Estensione a 6 stili (notebook 01c e 03e): monumenti di ogni tipo da Wikidata.
# I notebook a 6 stili impostano config.CLASSES = CLASSES6 all'avvio.
CLASSES6 = CLASSES + ["Classical", "Neoclassical"]

# Identificativi Wikidata degli stili, usati da churches.json di WikiChurches.
# Vengono comunque verificati contro style_names.txt durante l'ispezione.
STYLE_QIDS = {
    "Romanesque": "Q46261",    # Romanesque architecture
    "Gothic": "Q176483",       # Gothic architecture
    "Renaissance": "Q236122",  # Renaissance architecture
    "Baroque": "Q840829",      # Baroque architecture
}

# --- Dataset ----------------------------------------------------------------
ZENODO_RECORD = "5166987"  # WikiChurches v1.0, licenza CC BY-SA 4.0
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# Lato corto delle immagini dopo il ridimensionamento (poi crop 224x224).
IMG_SHORT_SIDE = 256
JPEG_QUALITY = 90

# Split ufficiale WikiChurches-4 (file labels/wc4_{train,val,test}.txt di labels.zip):
# divisione per chiesa decisa dagli autori, sotto-stili già ricondotti ai 4 stili.
OFFICIAL_SPLIT = "wc4"

# Usato solo se si costruisce uno split proprio (build_index + make_splits):
# divisione per CHIESA (non per foto) train / validation / test.
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Seme unico per tutti gli esperimenti (riproducibilità).
SEED = 42

# --- Addestramento ----------------------------------------------------------
IMG_SIZE = 224          # ingresso delle reti (ritaglio dalle immagini a 256 px)
BATCH_SIZE = 32

# Pesi di classe nella cross-entropy: "balanced" = n / (k * n_classe),
# "sqrt" = radice dei pesi bilanciati (più morbidi), "none" = nessun peso.
# Con lo split ufficiale (train: Romanico 1653, Gotico 3041, Rinascimentale 64, Barocco 1080)
# "balanced" darebbe al Rinascimentale peso 22.8; "sqrt" lo porta a 4.8 (Gotico 0.69).
CLASS_WEIGHT_MODE = "sqrt"

# Baseline: piccola CNN addestrata da zero.
# Iterazione 1 (30 epoche, pazienza 4): fermata all'epoca 6 senza aver imparato
# (predice una sola classe) -> più epoche, più pazienza, BatchNorm più reattiva.
CNN_EPOCHS = 40
CNN_LR = 1e-3
CNN_PATIENCE = 8
CNN_BN_MOMENTUM = 0.9

# M1: VGG-16 pre-addestrata e congelata + nuova testa
#     GAP -> BatchNorm -> Dense(256, ReLU) -> Dropout -> Dense softmax.
# Iterazione 1 (testa GAP -> Dropout -> Dense): loss iniziale 4.5 e accuracy di training 0.68
# (underfitting) -> normalizzazione delle feature e uno strato nascosto nella testa.
HEAD_EPOCHS = 30
HEAD_LR = 1e-3
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.5
HEAD_PATIENCE = 6

# M2: fine-tuning. Si parte da M1 e si sblocca l'ultimo blocco convoluzionale della VGG-16
# (block5_conv1..3, ~7 milioni di pesi). Learning rate 100 volte più basso di quello della
# testa, per adattare i filtri senza distruggere quanto appreso su ImageNet.
FT_UNFREEZE_PREFIX = "block5"
FT_LR = 1e-5
FT_EPOCHS = 20
FT_PATIENCE = 5

# --- Miglioramenti (notebook 03b) -------------------------------------------
# Validation più grande: il 15% delle chiese del training passa al validation
# (206 immagini erano troppo poche per scegliere l'epoca migliore). Test ufficiale invariato.
VAL_EXTRA_FRAC = 0.15
# Sovra-campionamento: ogni stile ha almeno OVERSAMPLE_MIN immagini per epoca (Rinascimentale).
OVERSAMPLE_MIN = 400
# Augmentation forte: ritaglio casuale tra il 60% e il 100% dell'area dell'immagine.
CROP_SCALE = (0.6, 1.0)

# M3: VGG-16 con fine-tuning più profondo (blocchi 4 e 5), partendo da M1
M3_UNFREEZE = ("block4", "block5")
M3_LR = 1e-5
M3_EPOCHS = 25
M3_PATIENCE = 6

# Confronto con un'architettura moderna: ConvNeXt-Tiny (2022) pre-addestrata su ImageNet.
# Fase 1: solo testa (backbone congelato); fase 2: sblocco dell'ultimo stadio.
CNX_HEAD_EPOCHS = 6
CNX_HEAD_LR = 1e-3
CNX_UNFREEZE = "stage_3"
CNX_FT_LR = 2e-5
CNX_FT_EPOCHS = 20
CNX_PATIENCE = 6

# Dati esterni (notebook 03c): MonuMAI, solo nel training, stili in comune con WikiChurches
MONUMAI_REPO = "https://github.com/ari-dasci/OD-MonuMAI.git"
MONUMAI_STYLES = ("Gothic", "Renaissance", "Baroque")
# Distanza di Hamming massima (su 64 bit) tra hash percettivi per considerare due foto duplicate
PHASH_MAX_DIST = 8
# Miglioramento minimo della macro-F1 di validation per accettare il modello con dati esterni
DELTA_MIN = 0.01

# Mappa di Kohonen (notebook 04) sulle 512 caratteristiche del modello finale (strato GAP).
# Griglia ~ 5·sqrt(N) neuroni per N ≈ 5.000 immagini di training -> 18 x 18 = 324 neuroni.
SOM_ROWS = 18
SOM_COLS = 18
SOM_ITER_PER_SAMPLE = 20      # iterazioni totali = 20 x numero di immagini di training
SOM_ALPHA0 = 0.5              # tasso di apprendimento iniziale
SOM_SIGMA0 = 9.0              # raggio iniziale del vicinato (metà del lato della griglia)
KNN_K = 5                     # chiese simili mostrate dall'app

# Esperimento ad alta risoluzione (notebook 01b e 03d): immagini a 512 px sul lato corto,
# ritaglio 448x448 (stesso rapporto 0,875 di 224/256), batch ridotto per la memoria della GPU.
IMG_SHORT_SIDE_HR = 512
IMG_SIZE_HR = 448
BATCH_SIZE_HR = 16
# Il modello a 448 px si accetta se la media della macro-F1 di validation di due addestramenti
# supera di almeno DELTA_HR la media dei due addestramenti a 224 px (03b e controllo del 03c).
DELTA_HR = 0.03

# Dataset di monumenti da Wikidata (notebook 01c): stili cercati per nome, sotto-stili inclusi
WIKIDATA_STYLE_SEARCH = {
    "Classical": ["ancient Roman architecture", "ancient Greek architecture"],
    "Neoclassical": ["Neoclassical architecture"],
}
# I monumenti antichi su Wikidata raramente hanno la proprietà «stile»: per il Classico antico si aggiungono gli
# edifici di questi tipi (e sotto-tipi)
WIKIDATA_TYPE_SEARCH = {
    "Classical": ["Roman temple", "ancient Greek temple", "Roman theatre", "ancient Greek theatre",
                  "Roman amphitheatre", "Roman triumphal arch", "Roman aqueduct"],
}
# Area geografica: Europa e Mediterraneo (la guida è pensata per monumenti europei; esclude ad es. le case
# coloniali nordamericane etichettate come neoclassiche)
WD_REGION_LAT = (30.0, 72.0)
WD_REGION_LON = (-25.0, 45.0)
WD_CAP_PER_CLASS = 1500       # immagini massime per stile
WD_MIN_PER_CLASS = 300        # sotto questa soglia lo stile non è utilizzabile
WD_THUMB_WIDTH = 960          # larghezza della miniatura scaricata da Wikimedia Commons
WD_MIN_SHORT_SIDE = 400       # immagini più piccole scartate
WD_MAX_ASPECT = 2.5           # panorami e dettagli molto allungati scartati
WD_ARCH_QUANTILE = 0.05       # filtro «è un edificio?»: soglia = 5° percentile sulle foto WikiChurches
WD_LABEL_MIN_PROB = 0.05      # pulizia delle etichette del training (probabilità fuori fold minima)
WD_SPLIT = (0.70, 0.15, 0.15)
WD_TAR_NAME = "wikidata6_512.tar"
WD_SPLITS_CSV = "data/splits/splits6_wikidata.csv"
# Somiglianza delle caratteristiche ImageNet (coseno) oltre la quale due foto mostrano lo stesso soggetto
WD_FEAT_DUP = 0.95
# Elementi architettonici diagnostici di MonuMAI (nomi delle annotazioni) e stile associato
ELEMENTI_STILE = {
    "arco-apuntado": "Gothic", "arco-conopial": "Gothic", "arco-trilobulado": "Gothic", "pinaculo-gotico": "Gothic",
    "fronton": "Renaissance", "fronton-curvo": "Renaissance", "serliana": "Renaissance",
    "fronton-partido": "Baroque", "columna-salomonica": "Baroque",
}
ELEMENTI_NOMI_IT = {
    "arco-apuntado": "arco a sesto acuto", "arco-conopial": "arco inflesso", "arco-trilobulado": "arco trilobato",
    "pinaculo-gotico": "pinnacolo gotico", "fronton": "timpano triangolare", "fronton-curvo": "timpano curvo",
    "serliana": "serliana", "fronton-partido": "timpano spezzato", "columna-salomonica": "colonna tortile (salomonica)",
    "arco-medio-punto": "arco a tutto sesto", "vano-adintelado": "portale architravato", "ojo-de-buey": "oculo",
}
ELEMENTI_CAP = 400            # ritagli massimi per stile nel training
ELEMENTI_SPLITS_CSV = "data/splits/elementi_monumai.csv"
# Modello a 6 stili: fase 1 solo testa, fase 2 blocchi 4-5 (come M3)
S6_HEAD_EPOCHS = 6
S6_HEAD_LR = 1e-3
S6_HEAD_PATIENCE = 3
# Varianti confrontate nel notebook 03e (stessa ricetta, dati diversi)
S6_VARIANTI = ("base", "elementi")
S6_AUGMENT = "light"          # augmentation forte + variazioni di luce del giorno
# Accettazione: sul validation WikiChurches perde al massimo S6_MAX_LOSS di macro-F1 rispetto al modello a 4
# stili, e sulle foto Wikidata dei 4 stili originali non va peggio.
S6_MAX_LOSS = 0.03

# Early stopping sulla loss di validazione (valore predefinito)
PATIENCE = 4

# --- Percorsi Colab ---------------------------------------------------------
COLAB_REPO_DIR = "/content/mirai"
COLAB_RAW_DIR = "/content/raw"              # disco locale, veloce, temporaneo
COLAB_PROCESSED_DIR = "/content/processed"  # immagini ridimensionate
DRIVE_DATA_DIR = "/content/drive/MyDrive/mirai_data"  # persistente
DRIVE_MODELS_DIR = f"{DRIVE_DATA_DIR}/models"         # modelli addestrati (.keras)
PROCESSED_TAR_NAME = "wikichurches4_256.tar"
COLAB_MONUMAI_RAW = "/content/monumai_raw"   # repository MonuMAI originale
COLAB_MONUMAI_DIR = "/content/monumai"       # immagini MonuMAI ridimensionate
MONUMAI_TAR_NAME = "monumai_256.tar"
COLAB_PROCESSED_DIR_HR = "/content/processed512"  # immagini a 512 px (esperimento 03d)
PROCESSED_TAR_NAME_HR = "wikichurches4_512.tar"
SPLITS_CSV = "data/splits/splits.csv"
