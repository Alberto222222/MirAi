"""Architetture usate nel progetto.

- build_cnn_baseline: piccola CNN addestrata da zero, catena
  convoluzione -> pooling ripetuta -> GAP -> Dropout -> softmax.
- build_vgg16_head (M1): VGG-16 pre-addestrata su ImageNet e congelata, con una
  nuova testa GAP -> BatchNorm -> Dense(256, ReLU) -> Dropout -> Dense softmax.
  La BatchNorm porta le feature della VGG (valori grandi, non normalizzati) su una
  scala adatta all'addestramento; lo strato nascosto dà alla testa capacità non lineare.

Il modello VGG è costruito "piatto" (gli strati della VGG sono strati del modello
stesso): così Grad-CAM può accedere direttamente a 'block5_conv3'.
"""
from __future__ import annotations

import numpy as np
from tensorflow import keras

from src import config

layers = keras.layers
NUM_CLASSES = len(config.CLASSES)


def build_cnn_baseline(size: int = config.IMG_SIZE, num_classes: int = NUM_CLASSES,
                       filters=(32, 64, 128, 256), dropout: float = 0.3,
                       bn_momentum: float = config.CNN_BN_MOMENTUM) -> keras.Model:
    inp = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inp)
    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(momentum=bn_momentum, name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inp, out, name="cnn_baseline")


def build_vgg16_head(size: int = config.IMG_SIZE, num_classes: int = NUM_CLASSES,
                     hidden: int = config.HEAD_HIDDEN,
                     dropout: float = config.HEAD_DROPOUT) -> keras.Model:
    base = keras.applications.VGG16(include_top=False, weights="imagenet",
                                    input_shape=(size, size, 3))
    for layer in base.layers:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(hidden, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(base.input, out, name="vgg16_m1")


def vgg16_at_size(model: keras.Model, size: int) -> keras.Model:
    """Stessa rete VGG-16 con testa, ma con ingresso size x size: gli strati convoluzionali e la
    testa (dopo il Global Average Pooling) non dipendono dalla dimensione dell'immagine,
    quindi i pesi si copiano senza modifiche."""
    new = build_vgg16_head(size=size)
    new.set_weights(model.get_weights())
    return new


def load(path) -> keras.Model:
    """Carica un modello salvato in formato .keras."""
    return keras.models.load_model(str(path))


HEAD_LAYERS = {"gap", "head_bn", "head_dense", "dropout", "predictions"}


def _add_head(x, num_classes: int, hidden: int, dropout: float):
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(hidden, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    return layers.Dense(num_classes, activation="softmax", name="predictions")(x)


def build_convnext_head(size: int = config.IMG_SIZE, num_classes: int = NUM_CLASSES,
                        hidden: int = config.HEAD_HIDDEN,
                        dropout: float = config.HEAD_DROPOUT) -> keras.Model:
    """ConvNeXt-Tiny pre-addestrata su ImageNet (backbone congelato) con la stessa testa di M1.
    La rete contiene già la sua pre-elaborazione: l'ingresso sono pixel RGB in [0, 255]."""
    base = keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet",
                                           input_shape=(size, size, 3), include_preprocessing=True)
    for layer in base.layers:
        layer.trainable = False
    out = _add_head(base.output, num_classes, hidden, dropout)
    return keras.Model(base.input, out, name="convnext_tiny_head")


def load_convnext(weights_path) -> keras.Model:
    """Ricostruisce la ConvNeXt con testa e carica i pesi (più robusto del caricamento .keras)."""
    model = build_convnext_head()
    model.load_weights(str(weights_path))
    return model


def unfreeze_containing(model: keras.Model, token: str) -> list[str]:
    """Sblocca gli strati il cui nome contiene token (es. 'stage_3'); gli altri strati del
    backbone restano congelati, la testa resta addestrabile."""
    sbloccati = []
    for layer in model.layers:
        if layer.name in HEAD_LAYERS:
            layer.trainable = True
        elif token in layer.name:
            layer.trainable = True
            if layer.weights:
                sbloccati.append(layer.name)
        else:
            layer.trainable = False
    return sbloccati


def unfreeze(model: keras.Model, prefix=config.FT_UNFREEZE_PREFIX) -> list[str]:
    """Fine-tuning: rende addestrabili gli strati della VGG il cui nome inizia con prefix
    (es. 'block5', oppure una tupla come ('block4', 'block5')); gli altri strati
    convoluzionali restano congelati, la testa resta addestrabile."""
    head = HEAD_LAYERS
    prefix = tuple(prefix) if isinstance(prefix, (list, tuple)) else prefix
    sbloccati = []
    for layer in model.layers:
        if layer.name.startswith(prefix):
            layer.trainable = True
            if layer.weights:
                sbloccati.append(layer.name)
        elif layer.name not in head:
            layer.trainable = False
    return sbloccati


def count_params(model: keras.Model) -> dict[str, int]:
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    return {"addestrabili": trainable, "congelati": frozen, "totali": trainable + frozen}
