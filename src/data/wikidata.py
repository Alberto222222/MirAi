"""Dataset di monumenti da Wikidata e Wikimedia Commons (estensione a 6 stili).

Ogni edificio su Wikidata può avere uno «stile architettonico» (proprietà P149) e un'immagine (P18).
Per ogni stile si raccolgono tutti gli edifici con quello stile o con un suo sotto-stile (P279*), per esempio
«Italian Gothic» per il Gotico. Si scarica un'immagine per edificio da Wikimedia Commons.

Pulizia (le prime tre su tutte le immagini, l'ultima solo sul training):
1. etichette ambigue: edifici con stili di classi diverse (es. romanico e gotico) vengono scartati;
2. formato: immagini piccole, panorami e dettagli molto allungati vengono scartati;
3. duplicati: foto quasi identiche tra loro o a foto di WikiChurches (hash percettivo) vengono scartate;
4. «è un edificio?»: una VGG-16 ImageNet stima la probabilità che la foto mostri un edificio; le foto sotto la
   soglia (5° percentile sulle foto di WikiChurches) – interni non architettonici, mappe, ritratti – vengono scartate;
5. coerenza dell'etichetta (solo training): un classificatore lineare addestrato sulle caratteristiche delle altre
   foto (validazione incrociata a 5 fold) stima la probabilità dello stile dichiarato; le foto con probabilità
   molto bassa sono probabilmente etichettate male e vengono scartate.
"""
from __future__ import annotations

import io
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps

from src import config

SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
UA = {"User-Agent": "MirAi-student-project/1.0 (https://github.com/Alberto222222/MirAi)"}
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _sparql(query: str, tries: int = 5) -> list[dict]:
    """Interroga Wikidata; riprova con attese crescenti se il servizio è lento o non risponde."""
    err = ""
    for t in range(tries):
        try:
            r = requests.get(SPARQL, params={"query": query, "format": "json"}, headers=UA, timeout=90)
            if r.status_code == 200:
                return r.json()["results"]["bindings"]
            err = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:
            err = type(e).__name__
        time.sleep(10 * (t + 1))
    raise RuntimeError(f"Wikidata non risponde ({err}) – riprovare tra qualche minuto")


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# 1. Stili e conteggi
# ---------------------------------------------------------------------------
def search_style(label: str) -> dict:
    """Cerca uno stile per nome inglese e restituisce il primo risultato con la descrizione da verificare.

    Prima la ricerca di Wikidata (wbsearchentities); se non risponde, la ricerca esatta dell'etichetta
    via SPARQL (con e senza iniziale maiuscola), preferendo gli elementi che sono stili architettonici."""
    try:
        r = requests.get(API, params={"action": "wbsearchentities", "search": label, "language": "en",
                                      "uselang": "en", "type": "item", "limit": 1, "format": "json"},
                         headers=UA, timeout=30)
        js = r.json()
        if js.get("search"):
            hit = js["search"][0]
            return {"qid": hit["id"], "label": hit.get("label", ""), "description": hit.get("description", "")}
        print(f"  ricerca Wikidata senza risultati per «{label}» (HTTP {r.status_code}): {str(js)[:200]}")
    except Exception as e:  # noqa: BLE001
        print(f"  ricerca Wikidata non disponibile ({type(e).__name__}), uso SPARQL")
    varianti = " ".join(f'"{v}"@en' for v in {label, label.lower(), label[:1].upper() + label[1:]})
    q = f"""SELECT ?s ?d (COUNT(?x) AS ?n) WHERE {{ VALUES ?l {{ {varianti} }} ?s rdfs:label ?l .
            OPTIONAL {{ ?s schema:description ?d FILTER(lang(?d) = "en") }}
            OPTIONAL {{ ?x wdt:P149 ?s }} }} GROUP BY ?s ?d ORDER BY DESC(?n)"""
    b = _sparql(q)
    if not b:
        raise ValueError(f"stile «{label}» non trovato su Wikidata")
    return {"qid": _qid(b[0]["s"]["value"]), "label": label, "description": b[0].get("d", {}).get("value", "")}


def resolve_roots() -> dict[str, list[str]]:
    """QID radice di ogni stile: i 4 di WikiChurches e i 2 nuovi, fissati in config."""
    roots = {c: [config.STYLE_QIDS[c]] for c in config.CLASSES[:4]}
    roots.update({c: list(q) for c, q in config.WIKIDATA_STYLE_QIDS_NEW.items()})
    for c, qs in roots.items():
        print(f"{c:13s} → {', '.join(qs)}")
    return roots


def substyles(roots: dict[str, list[str]]) -> pd.DataFrame:
    """Tutti i sotto-stili (P279*) di ogni radice, con il nome."""
    rows = []
    for c, qs in roots.items():
        vals = " ".join(f"wd:{q}" for q in qs)
        q = f"""SELECT DISTINCT ?s ?sLabel WHERE {{ VALUES ?root {{ {vals} }} ?s wdt:P279* ?root .
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}"""
        rows += [{"style": c, "qid": _qid(b["s"]["value"]), "name": b.get("sLabel", {}).get("value", "")}
                 for b in _sparql(q)]
    return pd.DataFrame(rows)


def items(subs: pd.DataFrame) -> pd.DataFrame:
    """Edifici con stile (P149) in uno dei sotto-stili e con immagine (P18). Una riga per (edificio, classe)."""
    rows = []
    for c, g in subs.groupby("style"):
        qids = g["qid"].tolist()
        for i in range(0, len(qids), 150):
            vals = " ".join(f"wd:{q}" for q in qids[i:i + 150])
            q = f"""SELECT ?item ?st (SAMPLE(?img) AS ?image) (SAMPLE(?co) AS ?coord) WHERE {{ VALUES ?st {{ {vals} }}
                    ?item wdt:P149 ?st ; wdt:P18 ?img . OPTIONAL {{ ?item wdt:P625 ?co }} }} GROUP BY ?item ?st"""
            rows += [_row(b, c, _qid(b["st"]["value"])) for b in _sparql(q)]
    return pd.DataFrame(rows)


def _row(b: dict, style: str, substyle: str) -> dict:
    co = pd.Series([b.get("coord", {}).get("value", "")]).str.extract(r"Point\(([-\d.]+) ([-\d.]+)\)").astype(float)
    return {"item": _qid(b["item"]["value"]), "style": style, "substyle": substyle,
            "image": b["image"]["value"], "lon": co.iloc[0, 0], "lat": co.iloc[0, 1]}


def items_by_type(style: str, types: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Edifici riconosciuti dal TIPO invece che dalla proprietà «stile» (es. tempio romano, anfiteatro romano):
    su Wikidata i monumenti antichi raramente hanno P149. Tipi inclusi anche nei sotto-tipi (P279*)."""
    tipi, rows = [], []
    for qid, lab in types.items():
        q = f"""SELECT ?item (SAMPLE(?img) AS ?image) (SAMPLE(?co) AS ?coord) WHERE {{
                ?item wdt:P31/wdt:P279* wd:{qid} ; wdt:P18 ?img . OPTIONAL {{ ?item wdt:P625 ?co }} }}
                GROUP BY ?item"""
        b = _sparql(q)
        print(f"{style:13s} tipo {qid} «{lab}»: {len(b)} edifici con immagine")
        tipi.append({"style": style, "qid": qid, "name": f"[tipo] {lab}"})
        rows += [_row(x, style, qid) for x in b]
    return pd.DataFrame(rows), pd.DataFrame(tipi)


def resolve_items(raw: pd.DataFrame, exclude: set[str]) -> tuple[pd.DataFrame, dict]:
    """Un edificio per riga. Scarta gli edifici già in WikiChurches e quelli con stili di classi diverse."""
    stats = {"edifici_trovati": raw["item"].nunique()}
    raw = raw[~raw["item"].isin(exclude)]
    stats["dopo_esclusione_wikichurches"] = raw["item"].nunique()
    (la0, la1), (lo0, lo1) = config.WD_REGION_LAT, config.WD_REGION_LON
    raw = raw[raw["lat"].between(la0, la1) & raw["lon"].between(lo0, lo1)]
    stats["in_europa_e_mediterraneo"] = raw["item"].nunique()
    n_cls = raw.groupby("item")["style"].nunique()
    ambigui = set(n_cls[n_cls > 1].index)
    stats["ambigui_scartati"] = len(ambigui)
    df = (raw[~raw["item"].isin(ambigui)].groupby("item")
          .agg(style=("style", "first"), substyle=("substyle", "first"), image=("image", "first"),
               lat=("lat", "first"), lon=("lon", "first")).reset_index())
    df = df[df["image"].str.lower().str.endswith(IMG_EXT)]
    stats["con_immagine_raster"] = len(df)
    return df, stats


def add_labels(df: pd.DataFrame, chunk: int = 80) -> pd.DataFrame:
    """Nome dell'edificio, Paese e tipo (es. palazzo, castello), in italiano se disponibile.
    Informazioni accessorie: se Wikidata non risponde per un gruppo, quei campi restano vuoti."""
    rows, ids = [], df["item"].tolist()
    for i in range(0, len(ids), chunk):
        vals = " ".join(f"wd:{q}" for q in ids[i:i + chunk])
        q = f"""SELECT ?item ?itemLabel ?cLabel ?tLabel WHERE {{ VALUES ?item {{ {vals} }}
                OPTIONAL {{ ?item wdt:P17 ?c }} OPTIONAL {{ ?item wdt:P31 ?t }}
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "it,en". }} }}"""
        try:
            for b in _sparql(q, tries=3):
                rows.append({"item": _qid(b["item"]["value"]), "name": b.get("itemLabel", {}).get("value", ""),
                             "country": b.get("cLabel", {}).get("value", ""), "type": b.get("tLabel", {}).get("value", "")})
        except RuntimeError:
            print(f"  nomi non disponibili per il gruppo {i // chunk + 1}: proseguo")
    lab = pd.DataFrame(rows, columns=["item", "name", "country", "type"]).drop_duplicates("item")
    return df.merge(lab, on="item", how="left")


# ---------------------------------------------------------------------------
# 2. Download e ridimensionamento
# ---------------------------------------------------------------------------
def _thumb_url(commons_url: str, width: int) -> str:
    name = unquote(commons_url.rsplit("/", 1)[-1])
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(name)}?width={width}"


def _download_one(row, out_dir: Path, width: int) -> dict:
    dest = out_dir / row.style / f"{row.item}.jpg"
    res = {"item": row.item, "path": f"wikidata/{row.style}/{row.item}.jpg", "ok": False, "motivo": ""}
    if dest.exists():
        with Image.open(dest) as im:
            res.update(ok=True, w0=im.width, h0=im.height)
        return res
    for t in range(5):
        try:
            r = requests.get(_thumb_url(row.image, width), headers=UA, timeout=60)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 5 * (t + 1))))
                continue
            r.raise_for_status()
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
            w0, h0 = im.size
            s = config.IMG_SHORT_SIDE_HR / min(w0, h0)
            if s < 1:
                im = im.resize((round(w0 * s), round(h0 * s)), Image.Resampling.LANCZOS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest, "JPEG", quality=config.JPEG_QUALITY)
            res.update(ok=True, w0=w0, h0=h0)
            return res
        except Exception as e:  # noqa: BLE001 – rete o immagine non leggibile
            res["motivo"] = type(e).__name__
            time.sleep(2 * (t + 1))
    return res


def download(df: pd.DataFrame, root: str | Path = config.COLAB_PROCESSED_DIR_HR,
             width: int = config.WD_THUMB_WIDTH, workers: int = 6,
             checkpoint_tar: str | Path | None = None, batch: int = 1000) -> pd.DataFrame:
    """Scarica e ridimensiona (lato corto 512 px) in <root>/wikidata/<stile>/<QID>.jpg.

    Con checkpoint_tar (un file su Google Drive) il download è riprendibile: all'avvio si ripristinano le foto
    già salvate, e ogni `batch` foto le nuove vengono aggiunte all'archivio. Se il runtime si interrompe,
    basta rieseguire: si riparte dall'ultimo salvataggio."""
    root = Path(root)
    out_dir = root / "wikidata"
    if checkpoint_tar and Path(checkpoint_tar).exists():
        with tarfile.open(checkpoint_tar) as tar:
            try:
                tar.extractall(root.parent, filter="data")
            except TypeError:
                tar.extractall(root.parent)
        print(f"  ripristinate dal salvataggio intermedio: {sum(1 for _ in out_dir.rglob('*.jpg'))} foto")
    results = []
    rows = list(df.itertuples())
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        nuove = [r for r in part if not (out_dir / r.style / f"{r.item}.jpg").exists()]
        with ThreadPoolExecutor(workers) as ex:
            results += [f.result() for f in as_completed([ex.submit(_download_one, r, out_dir, width) for r in part])]
        if checkpoint_tar and nuove:
            Path(checkpoint_tar).parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(checkpoint_tar, "a") as tar:
                for r in nuove:
                    f = out_dir / r.style / f"{r.item}.jpg"
                    if f.exists():
                        tar.add(f, arcname=f"{root.name}/wikidata/{r.style}/{r.item}.jpg")
        print(f"  scaricate {min(start + batch, len(rows))}/{len(rows)}"
              + (" – salvate su Drive" if checkpoint_tar else ""))
    res = pd.DataFrame(results)
    return df.merge(res, on="item", how="left")


def format_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Scarta download falliti, immagini piccole e proporzioni estreme."""
    short = df[["w0", "h0"]].min(axis=1)
    aspect = df[["w0", "h0"]].max(axis=1) / short.clip(lower=1)
    df = df.assign(motivo=np.where(~df["ok"].fillna(False).astype(bool), "download fallito",
                          np.where(short < config.WD_MIN_SHORT_SIDE, "immagine piccola",
                          np.where(aspect > config.WD_MAX_ASPECT, "proporzioni estreme", ""))))
    return df


# ---------------------------------------------------------------------------
# 3. Salvataggio
# ---------------------------------------------------------------------------
def save_to_drive(root: str | Path = config.COLAB_PROCESSED_DIR_HR,
                  drive_dir: str | Path = config.DRIVE_DATA_DIR, folders=("wikidata", "elementi")) -> Path:
    """Archivio delle cartelle <root>/wikidata e <root>/elementi come processed512/... (si ripristinano
    accanto alle immagini di WikiChurches)."""
    tar_path = Path(drive_dir) / config.WD_TAR_NAME
    with tarfile.open(tar_path, "w") as tar:
        for f in folders:
            if (Path(root) / f).is_dir():
                tar.add(Path(root) / f, arcname=f"{Path(root).name}/{f}")
    print(f"Salvato {tar_path} ({tar_path.stat().st_size / 1e6:.0f} MB)")
    return tar_path


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False))
