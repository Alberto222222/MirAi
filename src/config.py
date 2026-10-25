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
}

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

# Early stopping sulla loss di validazione (valore predefinito)
PATIENCE = 4

# --- Percorsi Colab ---------------------------------------------------------
COLAB_REPO_DIR = "/content/mirai"
COLAB_RAW_DIR = "/content/raw"              # disco locale, veloce, temporaneo
COLAB_PROCESSED_DIR = "/content/processed"  # immagini ridimensionate
DRIVE_DATA_DIR = "/content/drive/MyDrive/mirai_data"  # persistente
DRIVE_MODELS_DIR = f"{DRIVE_DATA_DIR}/models"         # modelli addestrati (.keras)
PROCESSED_TAR_NAME = "wikichurches4_256.tar"
SPLITS_CSV = "data/splits/splits.csv"
