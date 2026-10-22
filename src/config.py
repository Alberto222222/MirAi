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

# --- Percorsi Colab ---------------------------------------------------------
COLAB_REPO_DIR = "/content/mirai"
COLAB_RAW_DIR = "/content/raw"              # disco locale, veloce, temporaneo
COLAB_PROCESSED_DIR = "/content/processed"  # immagini ridimensionate
DRIVE_DATA_DIR = "/content/drive/MyDrive/mirai_data"  # persistente
PROCESSED_TAR_NAME = "wikichurches4_256.tar"
