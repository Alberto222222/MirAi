"""Caricamento delle immagini per l'addestramento (tf.data).

Flusso per ogni immagine:
    file JPEG -> decodifica -> lato corto 256 px (uint8, in cache)
      training:   ritaglio casuale 224x224 + ribaltamento + luminosità/contrasto
      val/test:   ritaglio centrale 224x224
    -> pre-elaborazione richiesta dal modello ("vgg" oppure "raw")

La data augmentation mostra alla rete lo stesso edificio in inquadrature e
condizioni di luce diverse, senza raccogliere nuove foto.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src import config

AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = len(config.CLASSES)
# Media dei canali di ImageNet in ordine BGR: è la pre-elaborazione "caffe" della VGG-16
VGG_MEAN_BGR = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)


# ---------------------------------------------------------------------------
# Tabella degli split
# ---------------------------------------------------------------------------
def load_splits(csv_path: str | Path = config.SPLITS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["label"] = df["style"].map({c: i for i, c in enumerate(config.CLASSES)})
    if df["label"].isna().any():
        raise ValueError("splits.csv contiene stili non presenti in config.CLASSES")
    df["label"] = df["label"].astype(int)
    return df


def stratified_subset(df: pd.DataFrame, per_class: int, seed: int = config.SEED) -> pd.DataFrame:
    """Sottoinsieme piccolo e bilanciato, usato per la prova rapida della pipeline."""
    parts = [g.sample(min(per_class, len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).reset_index(drop=True)


def class_weights(train_df: pd.DataFrame, mode: str = config.CLASS_WEIGHT_MODE) -> dict[int, float]:
    """Pesi di classe inversamente proporzionali alla frequenza nel training set."""
    counts = (train_df["label"].value_counts()
              .reindex(range(NUM_CLASSES), fill_value=0).to_numpy(dtype=float))
    w = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    if mode == "sqrt":
        w = np.sqrt(w)
    elif mode == "none":
        w = np.ones(NUM_CLASSES)
    elif mode != "balanced":
        raise ValueError(f"CLASS_WEIGHT_MODE sconosciuto: {mode}")
    return {i: float(w[i]) for i in range(NUM_CLASSES)}


def enlarge_validation(df: pd.DataFrame, frac: float = config.VAL_EXTRA_FRAC,
                       seed: int = config.SEED) -> pd.DataFrame:
    """Sposta nel validation una frazione delle CHIESE del training (stratificata per stile).

    Il test ufficiale non viene toccato; la colonna 'split_ufficiale' conserva lo split originale.
    """
    from sklearn.model_selection import train_test_split

    df = df.copy()
    df["split_ufficiale"] = df["split"]
    ch = df[df["split"] == "train"].groupby("church_id")["style"].first().reset_index()
    _, extra = train_test_split(ch, test_size=frac, stratify=ch["style"], random_state=seed)
    df.loc[df["church_id"].isin(set(extra["church_id"])) & (df["split"] == "train"), "split"] = "val"
    return df


def oversample(train_df: pd.DataFrame, min_count: int = config.OVERSAMPLE_MIN,
               seed: int = config.SEED) -> pd.DataFrame:
    """Ripete le immagini delle classi con meno di min_count esempi fino a raggiungerlo.
    Ogni ripetizione riceve una data augmentation diversa durante l'addestramento."""
    parts = []
    for _, g in train_df.groupby("label"):
        if len(g) < min_count:
            extra = g.sample(min_count - len(g), replace=True, random_state=seed)
            g = pd.concat([g, extra])
        parts.append(g)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Operazioni sulle immagini
# ---------------------------------------------------------------------------
def _load(path: tf.Tensor) -> tf.Tensor:
    """Legge un JPEG e porta il lato corto a IMG_SHORT_SIDE (uint8, adatto alla cache)."""
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    hw = tf.cast(tf.shape(img)[:2], tf.float32)
    scale = config.IMG_SHORT_SIDE / tf.reduce_min(hw)
    new_hw = tf.cast(tf.math.round(hw * scale), tf.int32)
    img = tf.image.resize(img, new_hw, antialias=True)
    return tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)


def _augment(img: tf.Tensor) -> tf.Tensor:
    s = config.IMG_SIZE
    img = tf.image.random_crop(img, [s, s, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.cast(img, tf.float32)
    img = tf.image.random_brightness(img, max_delta=0.15 * 255.0)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 255.0)


def _augment_strong(img: tf.Tensor) -> tf.Tensor:
    """Augmentation più forte: ritaglio di area e proporzioni variabili (poi 224x224),
    ribaltamento, luminosità, contrasto, saturazione e tonalità."""
    s = config.IMG_SIZE
    hw = tf.cast(tf.shape(img)[:2], tf.float32)
    area = hw[0] * hw[1] * tf.random.uniform([], config.CROP_SCALE[0], config.CROP_SCALE[1])
    ratio = tf.exp(tf.random.uniform([], tf.math.log(3 / 4), tf.math.log(4 / 3)))
    ch = tf.cast(tf.minimum(tf.sqrt(area / ratio), hw[0]), tf.int32)
    cw = tf.cast(tf.minimum(tf.sqrt(area * ratio), hw[1]), tf.int32)
    img = tf.image.random_crop(img, tf.stack([ch, cw, 3]))
    img = tf.image.resize(tf.cast(img, tf.float32), [s, s], antialias=True) / 255.0
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(tf.clip_by_value(img, 0.0, 1.0), 0.8, 1.2)
    img = tf.image.random_hue(img, 0.03)
    return tf.clip_by_value(img, 0.0, 1.0) * 255.0


def _center(img: tf.Tensor) -> tf.Tensor:
    s = config.IMG_SIZE
    return tf.cast(tf.image.resize_with_crop_or_pad(img, s, s), tf.float32)


def preprocess(img: tf.Tensor, kind: str) -> tf.Tensor:
    """img: float32 RGB in [0, 255].

    "vgg": RGB -> BGR e sottrazione della media di ImageNet (come keras vgg16.preprocess_input)
    "raw": nessuna trasformazione (la rete ha il suo strato di Rescaling)
    """
    if kind == "vgg":
        return img[..., ::-1] - VGG_MEAN_BGR
    if kind == "raw":
        return img
    raise ValueError(f"pre-elaborazione sconosciuta: {kind}")


def load_image_for_model(path: str | Path, kind: str = "vgg") -> tf.Tensor:
    """Una singola immagine pronta per il modello (ritaglio centrale), forma (224, 224, 3)."""
    return preprocess(_center(_load(tf.constant(str(path)))), kind)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def make_dataset(df: pd.DataFrame, root: str | Path, training: bool, kind: str = "vgg",
                 class_weight: dict[int, float] | None = None,
                 batch_size: int = config.BATCH_SIZE, seed: int = config.SEED,
                 augment: str = "basic", view: str = "center") -> tf.data.Dataset:
    """Restituisce batch (x, y) oppure (x, y, peso) se class_weight è indicato.

    augment (solo training): "basic" oppure "strong".
    view (solo valutazione): "center" = ritaglio centrale, "flip" = ritaglio centrale ribaltato
    (le due viste servono al test-time augmentation).
    """
    paths = [str(Path(root) / p) for p in df["path"]]
    labels = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (_load(p), y), num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(min(len(paths), 2000), seed=seed, reshuffle_each_iteration=True)
        aug = _augment_strong if augment == "strong" else _augment
        ds = ds.map(lambda x, y: (aug(x), y), num_parallel_calls=AUTOTUNE)
    elif view == "flip":
        ds = ds.map(lambda x, y: (tf.image.flip_left_right(_center(x)), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (_center(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess(x, kind), y), num_parallel_calls=AUTOTUNE)
    if class_weight is not None:
        w = tf.constant([class_weight[i] for i in range(NUM_CLASSES)], dtype=tf.float32)
        ds = ds.map(lambda x, y: (x, y, tf.gather(w, y)), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)
