"""Metriche, grafici e test statistici (non dipende da TensorFlow).

Metriche: accuracy, macro-F1 (media non pesata sulle classi: non premia le classi
numerose), balanced accuracy, precision/recall/F1 per classe, matrice di confusione.
Confronto tra due modelli sullo stesso test set: test di McNemar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src import config

LABELS_IT = [config.CLASS_NAMES_IT[c] for c in config.CLASSES]


def compute_metrics(y_true, y_pred, name: str) -> dict:
    k = len(config.CLASSES)
    rep = classification_report(y_true, y_pred, labels=list(range(k)), target_names=config.CLASSES,
                                output_dict=True, zero_division=0)
    return {
        "modello": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(k)), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_classe": {c: {m: float(rep[c][m]) for m in ("precision", "recall", "f1-score", "support")}
                       for c in config.CLASSES},
        "matrice_confusione": confusion_matrix(y_true, y_pred, labels=list(range(k))).tolist(),
    }


def print_metrics(m: dict) -> pd.DataFrame:
    print(f"{m['modello']}: accuracy {m['accuracy']:.3f} | macro-F1 {m['macro_f1']:.3f} | "
          f"balanced accuracy {m['balanced_accuracy']:.3f}")
    df = pd.DataFrame(m["per_classe"]).T.rename(index=config.CLASS_NAMES_IT)
    df["support"] = df["support"].astype(int)
    print(df.round(3).to_string())
    return df


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_predictions(df: pd.DataFrame, probs: np.ndarray, path: str | Path) -> pd.DataFrame:
    """Salva per ogni immagine di test etichetta, predizione e probabilità (servono a McNemar e alla XAI)."""
    out = df[["path", "label", "style", "church_id"]].copy().reset_index(drop=True)
    out["pred"] = probs.argmax(axis=1)
    for i, c in enumerate(config.CLASSES):
        out[f"p_{c}"] = probs[:, i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def majority_baseline(train_labels, test_labels) -> np.ndarray:
    """Predice sempre la classe più frequente nel training set."""
    majority = int(pd.Series(train_labels).mode()[0])
    return np.full(len(test_labels), majority)


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------
def plot_history(history: dict, title: str, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = range(1, len(history["loss"]) + 1)
    for ax, key, lab in ((axes[0], "loss", "loss (cross-entropy)"), (axes[1], "accuracy", "accuracy")):
        ax.plot(ep, history[key], label="training")
        ax.plot(ep, history[f"val_{key}"], label="validation")
        if "epoca_migliore" in history:
            ax.axvline(history["epoca_migliore"], color="grey", ls="--", lw=1, label="epoca migliore")
        ax.set_xlabel("epoca")
        ax.set_ylabel(lab)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, title: str, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(LABELS_IT)), LABELS_IT, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS_IT)), LABELS_IT)
    ax.set_xlabel("stile predetto")
    ax.set_ylabel("stile reale")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confronto tra modelli
# ---------------------------------------------------------------------------
def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Test di McNemar sulle immagini in cui i due modelli non sono d'accordo.

    b = A corretto e B sbagliato, c = A sbagliato e B corretto.
    Con b + c < 25 si usa il test binomiale esatto, altrimenti il chi-quadro
    con correzione di continuità.
    """
    from scipy.stats import binomtest, chi2

    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    b, c = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
    if b + c == 0:
        return {"b": b, "c": c, "metodo": "nessun disaccordo", "statistica": 0.0, "p_value": 1.0}
    if b + c < 25:
        p = binomtest(b, b + c, 0.5).pvalue
        return {"b": b, "c": c, "metodo": "binomiale esatto", "statistica": float(b), "p_value": float(p)}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "metodo": "chi-quadro con correzione", "statistica": float(stat),
            "p_value": float(chi2.sf(stat, 1))}


def comparison_table(metrics_list: list[dict]) -> pd.DataFrame:
    rows = [{"modello": m["modello"], "accuracy": m["accuracy"], "macro-F1": m["macro_f1"],
             "balanced accuracy": m["balanced_accuracy"],
             **{f"F1 {config.CLASS_NAMES_IT[c]}": m["per_classe"][c]["f1-score"] for c in config.CLASSES}}
            for m in metrics_list]
    return pd.DataFrame(rows).set_index("modello").round(3)
