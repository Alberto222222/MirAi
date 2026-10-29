"""Risposta gerarchica: famiglia storico-architettonica (livello 1) e stile (livello 2).

La probabilità di una famiglia è la somma delle probabilità dei suoi stili. Si sceglie prima la famiglia più
probabile, poi lo stile più probabile all'interno della famiglia scelta.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src import config


def family_names() -> list[str]:
    return list(config.FAMIGLIE)


def style_to_family() -> np.ndarray:
    """Indice della famiglia per ogni stile di config.CLASSES."""
    out = np.zeros(len(config.CLASSES), dtype=int)
    for f, (_, stili) in enumerate(config.FAMIGLIE.items()):
        for s in stili:
            out[config.CLASSES.index(s)] = f
    return out


def family_probs(p: np.ndarray) -> np.ndarray:
    s2f = style_to_family()
    return np.stack([p[:, s2f == f].sum(1) for f in range(len(config.FAMIGLIE))], axis=1)


def predict(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(famiglia predetta, stile predetto dentro la famiglia, probabilità delle famiglie)."""
    pf = family_probs(p)
    fam = pf.argmax(1)
    s2f = style_to_family()
    masked = np.where(s2f[None, :] == fam[:, None], p, -1.0)
    return fam, masked.argmax(1), pf


def family_metrics(y_style: np.ndarray, p: np.ndarray, name: str) -> dict:
    """Metriche a livello di famiglia (solo sulle famiglie presenti nell'insieme)."""
    s2f = style_to_family()
    yf = s2f[np.asarray(y_style)]
    fam, _, pf = predict(p)
    present = sorted(set(yf.tolist()))
    names = family_names()
    rec = {names[f]: float(np.mean(fam[yf == f] == f)) for f in present}
    return {"modello": name, "accuracy": float(np.mean(fam == yf)),
            "macro_f1": float(f1_score(yf, fam, labels=present, average="macro", zero_division=0)),
            "balanced_accuracy": float(np.mean(list(rec.values()))),
            "recall": rec,
            "matrice_confusione": confusion_matrix(yf, fam, labels=list(range(len(names)))).tolist()}


def top_k(y: np.ndarray, p: np.ndarray, k: int = 2) -> float:
    return float(np.mean([yy in np.argsort(-pp)[:k] for yy, pp in zip(y, p)]))


def coverage_curve(conf: np.ndarray, ok: np.ndarray, soglie=np.round(np.arange(0.30, 0.991, 0.05), 2)):
    rows = []
    for t in soglie:
        m = conf >= t
        rows.append({"soglia": float(t), "copertura": float(m.mean()),
                     "accuratezza": float(ok[m].mean()) if m.any() else float("nan")})
    return rows


def safe_threshold(conf: np.ndarray, ok: np.ndarray, target: float) -> float:
    """Soglia di confidenza più bassa con accuratezza ≥ target (calcolata sul validation)."""
    for r in coverage_curve(conf, ok):
        if r["accuratezza"] == r["accuratezza"] and r["accuratezza"] >= target:
            return r["soglia"]
    return 0.99
