"""Analisi degli errori usando i metadati delle chiese (stili Wikidata, Paese, coordinate).

- substyle_errors: per le immagini di uno stile (es. Gotico), quanto spesso vengono
  scambiate con un altro stile (es. Romanico), separatamente per sotto-stile
  (es. «Brick Gothic», «Italian Gothic»).
- errors_by_country: accuracy per Paese.
- plot_error_map: posizione delle chiese del test, corrette ed errate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data.wikichurches import parse_style_names


def church_style_names(meta_dir: str | Path) -> dict[str, str]:
    """{QID chiesa: stili Wikidata della chiesa in chiaro, separati da ' + '}."""
    meta_dir = Path(meta_dir)
    churches = json.loads((meta_dir / "churches.json").read_text(encoding="utf-8"))
    names = parse_style_names(meta_dir / "style_names.txt")
    out = {}
    for cid, ch in churches.items():
        styles = ch.get("styles") or []
        out[cid] = " + ".join(sorted(names.get(q, q) for q in styles)) or "?"
    return out


def with_metadata(pred_df: pd.DataFrame, splits_df: pd.DataFrame,
                  style_names: dict[str, str] | None = None) -> pd.DataFrame:
    cols = [c for c in ("path", "church_name", "country", "lat", "lon") if c in splits_df.columns]
    df = pred_df.merge(splits_df[cols], on="path", how="left")
    df["correct"] = df["pred"] == df["label"]
    df["pred_style"] = df["pred"].map(dict(enumerate(config.CLASSES)))
    if style_names is not None:
        df["wikidata_styles"] = df["church_id"].map(style_names).fillna("?")
    return df


def substyle_errors(df: pd.DataFrame, true_style: str = "Gothic", confused_with: str = "Romanesque",
                    min_images: int = 8) -> pd.DataFrame:
    sub = df[df["style"] == true_style]
    g = sub.groupby("wikidata_styles").agg(
        immagini=("path", "size"),
        corrette=("correct", "mean"),
        **{f"come_{confused_with}": ("pred_style", lambda s: float((s == confused_with).mean()))})
    g = g[g["immagini"] >= min_images].sort_values(f"come_{confused_with}", ascending=False)
    return g.round(3)


def errors_by_country(df: pd.DataFrame, min_images: int = 20) -> pd.DataFrame:
    g = df.groupby("country").agg(immagini=("path", "size"), accuracy=("correct", "mean"))
    return g[g["immagini"] >= min_images].sort_values("immagini", ascending=False).round(3)


def plot_error_map(df: pd.DataFrame, title: str, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["lat", "lon"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ok, ko = d[d["correct"]], d[~d["correct"]]
    ax.scatter(ok["lon"], ok["lat"], s=6, c="#4f9d69", alpha=0.55, label=f"corrette ({len(ok)})")
    ax.scatter(ko["lon"], ko["lat"], s=6, c="#d1495b", alpha=0.65, label=f"errate ({len(ko)})")
    ax.set_xlim(-11, 32)
    ax.set_ylim(35, 62)
    ax.set_xlabel("longitudine")
    ax.set_ylabel("latitudine")
    ax.set_title(title)
    ax.legend(loc="lower left", markerscale=3)
    ax.set_aspect(1.4)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confidence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Probabilità media assegnata alla classe predetta, per immagini corrette ed errate."""
    p = df[[f"p_{c}" for c in config.CLASSES]].to_numpy()
    df = df.assign(confidenza=p.max(axis=1))
    return df.groupby("correct")["confidenza"].describe()[["count", "mean", "50%"]].round(3)
