"""Architetture usate nel progetto.

- build_cnn_baseline: piccola CNN addestrata da zero, catena
  convoluzione -> pooling ripetuta -> GAP -> Dropout -> softmax
  (slide 10-- DEEP_L p31-62; Neural Networks.pdf p162-174).
- build_vgg16_head (M1): VGG-16 pre-addestrata su ImageNet e congelata, con una
  nuova testa GAP -> Dropout -> Dense softmax (11-- Explainability p4, p15-17).
  Il Global Average Pooling prima del Dense rende possibile il CAM (p34, p38).

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
                       filters=(32, 64, 128, 256), dropout: float = 0.3) -> keras.Model:
    inp = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inp)
    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inp, out, name="cnn_baseline")


def build_vgg16_head(size: int = config.IMG_SIZE, num_classes: int = NUM_CLASSES,
                     dropout: float = config.HEAD_DROPOUT) -> keras.Model:
    base = keras.applications.VGG16(include_top=False, weights="imagenet",
                                    input_shape=(size, size, 3))
    for layer in base.layers:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(base.input, out, name="vgg16_m1")


def count_params(model: keras.Model) -> dict[str, int]:
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    return {"addestrabili": trainable, "congelati": frozen, "totali": trainable + frozen}
