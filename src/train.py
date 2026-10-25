"""Addestramento con cross-entropy, Adam, early stopping e salvataggio dei risultati."""
from __future__ import annotations

import json
import time
from pathlib import Path

from tensorflow import keras


def train_model(model: keras.Model, train_ds, val_ds, name: str, epochs: int, lr: float,
                models_dir: str | Path, reports_dir: str | Path, patience: int) -> dict:
    """Addestra il modello e restituisce la storia dell'addestramento.

    - loss: cross-entropy categoriale sparsa (§17), pesata per classe tramite i pesi nel dataset;
    - ottimizzatore: Adam (discesa del gradiente con momento adattivo, §16);
    - early stopping sulla loss di validazione con ripristino dei pesi migliori (§17);
    - learning rate dimezzato quando la validazione smette di migliorare;
    - a fine addestramento il modello torna ai pesi dell'epoca migliore e viene salvato
      in models_dir come <name>.keras.
    """
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    (reports_dir / "logs").mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    best_weights = models_dir / f"{name}.best.weights.h5"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(1, patience // 2), verbose=1),
        keras.callbacks.ModelCheckpoint(str(best_weights), monitor="val_loss",
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.CSVLogger(str(reports_dir / "logs" / f"{name}.csv")),
    ]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    # Ripristina i pesi con la loss di validazione minima e salva il modello completo
    model.load_weights(str(best_weights))
    model.save(str(models_dir / f"{name}.keras"))
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    history["tempo_secondi"] = round(time.time() - t0, 1)
    history["epoche_eseguite"] = len(history.get("loss", []))
    history["epoca_migliore"] = int(min(range(len(history["val_loss"])),
                                        key=history["val_loss"].__getitem__)) + 1
    with open(reports_dir / "metrics" / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"{name}: {history['epoche_eseguite']} epoche in {history['tempo_secondi']} s, "
          f"migliore all'epoca {history['epoca_migliore']}")
    return history
