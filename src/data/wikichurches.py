"""Preparazione del dataset WikiChurches (4 stili) per il progetto MirAi.

Passi (ognuno è una funzione e un sotto-comando da riga di comando):

    inspect      stampa la struttura dei file di metadati (verifica)
    preview      conta chiese e immagini per stile, PRIMA del download grosso
    build-index  collega ogni immagine dello zip alla sua chiesa e al suo stile
    extract      estrae solo le immagini utili e le ridimensiona (lato corto 256)
    split        divide in train/val/test PER CHIESA (niente foto dello stesso
                 edificio in due insiemi diversi)
    report       tabelle dei conteggi, grafico a barre, griglia di esempi

Esempio:
    python -m src.data.wikichurches inspect --raw /content/raw

Il modulo non conosce in anticipo il formato esatto di image_meta.json e
dello zip: lo riconosce da solo e stampa che cosa ha capito, così ogni
ipotesi è verificabile.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image, ImageOps

from src import config

try:  # barra di avanzamento se disponibile (su Colab c'è)
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):
        return it

QID_RE = re.compile(r"Q\d+")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

METADATA_FILES = [
    "churches.json",
    "image_meta.json",
    "style_names.txt",
    "parent_child_rel.txt",
    "labels.zip",
    "README.md",
    "LICENSE",
]
IMAGES_ZIP = "images.zip"


# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------
def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _short(obj, n: int = 300) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + " …"


def _qids_in(value) -> list[str]:
    """Estrae gli identificativi Wikidata (Q123) da stringhe, URL o liste."""
    vals = value if isinstance(value, (list, tuple)) else [value]
    out = []
    for v in vals:
        if isinstance(v, str):
            out.extend(QID_RE.findall(v))
        elif isinstance(v, dict):
            for key in ("id", "qid"):
                if isinstance(v.get(key), str):
                    out.extend(QID_RE.findall(v[key]))
    return out


def url_for(filename: str) -> str:
    return f"{config.ZENODO_BASE}/{filename}?download=1"


# ---------------------------------------------------------------------------
# Stili
# ---------------------------------------------------------------------------
def parse_style_names(path: Path) -> dict[str, str]:
    """Legge style_names.txt e restituisce {QID: nome}, se il formato lo consente."""
    names = {}
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        qids = QID_RE.findall(line)
        if len(qids) != 1:
            continue
        name = line.replace(qids[0], " ")
        name = re.sub(r"[\t,;:|]+", " ", name).strip(" -")
        names[qids[0]] = re.sub(r"\s+", " ", name)
    return names


def parse_hierarchy(path: Path, targets: dict[str, str]) -> dict[str, str]:
    """Da parent_child_rel.txt ricava {QID_sottostile: classe}.

    La direzione (genitore-figlio o figlio-genitore) viene dedotta: gli stili
    principali compaiono più spesso nella posizione del genitore.
    """
    if not path.exists():
        return {}
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        q = QID_RE.findall(line)
        if len(q) == 2:
            pairs.append((q[0], q[1]))
    target_q = {q: c for c, q in targets.items()}
    first = sum(a in target_q for a, _ in pairs)
    second = sum(b in target_q for _, b in pairs)
    if first == 0 and second == 0:
        return {}
    if first < second:  # formato figlio -> genitore
        pairs = [(b, a) for a, b in pairs]
    children = defaultdict(set)
    for parent, child in pairs:
        children[parent].add(child)
    mapping = {}
    for q, cls in target_q.items():  # chiusura transitiva
        stack = list(children.get(q, ()))
        while stack:
            c = stack.pop()
            if c in mapping or c in target_q:
                continue
            mapping[c] = cls
            stack.extend(children.get(c, ()))
    return mapping


def assign_class(styles: list[str], targets: dict[str, str],
                 sub_map: dict[str, str] | None = None) -> str | None:
    """Restituisce la classe di una chiesa, oppure None se non è 'pura'.

    Pura = tutti i suoi stili appartengono a UNA sola delle 4 classi.
    Se sub_map è None, sono ammessi solo i 4 stili principali (default).
    """
    if not styles:
        return None
    q2c = {q: c for c, q in targets.items()}
    if sub_map:
        q2c = {**sub_map, **q2c}
    classes = set()
    for q in styles:
        if q not in q2c:
            return None  # ha uno stile estraneo: chiesa mista, la scartiamo
        classes.add(q2c[q])
    return classes.pop() if len(classes) == 1 else None


# ---------------------------------------------------------------------------
# Collegamento immagini -> chiese
# ---------------------------------------------------------------------------
def map_images_to_churches(meta, churches: dict) -> tuple[list, str]:
    """Riconosce la struttura di image_meta.json.

    Restituisce (righe, descrizione) con righe = [(chiave_immagine, qid_chiesa, record)].
    """
    rows = []
    if isinstance(meta, dict):
        keys = list(meta.keys())[:2000]
        frac = sum(k in churches for k in keys) / max(1, len(keys))
        if frac > 0.5:  # struttura chiesa -> immagini
            for cid, imgs in meta.items():
                if cid not in churches:
                    continue
                for im in imgs if isinstance(imgs, list) else [imgs]:
                    if isinstance(im, dict):
                        rows.append((None, cid, im))
                    else:
                        rows.append((str(im), cid, None))
            return rows, "chiesa -> lista di immagini (chiavi = QID delle chiese)"
        # Formato WikiChurches: chiave = nome del file, che inizia con il QID della chiesa
        # (es. "Q16550866_wd0.jpg" = foto 0 della chiesa Q16550866)
        key_hits = sum(any(q in churches for q in QID_RE.findall(k)[:1]) for k in keys)
        if key_hits / max(1, len(keys)) > 0.5:
            for k, rec in meta.items():
                q = QID_RE.findall(k)[:1]
                if q and q[0] in churches:
                    rows.append((k, q[0], rec))
            return rows, "nome del file -> chiesa (il nome inizia con il QID della chiesa)"
        records = list(meta.items())
    elif isinstance(meta, list):
        records = [(None, r) for r in meta]
    else:
        raise ValueError("image_meta.json ha una struttura non prevista")

    field_hits = Counter()
    for _, rec in records[:2000]:
        if isinstance(rec, dict):
            for f, v in rec.items():
                if any(q in churches for q in _qids_in(v)):
                    field_hits[f] += 1
    if not field_hits:
        return [], "nessun campo di image_meta.json contiene QID di chiese"
    field = field_hits.most_common(1)[0][0]
    for key, rec in records:
        if not isinstance(rec, dict):
            continue
        cid = next((q for q in _qids_in(rec.get(field)) if q in churches), None)
        if cid:
            rows.append((key, cid, rec))
    return rows, f"immagine -> chiesa tramite il campo '{field}'"


def _zip_image_members(zf: zipfile.ZipFile) -> list[str]:
    return [n for n in zf.namelist()
            if n.lower().endswith(IMG_EXT) and not n.startswith("__MACOSX")]


def _scalar_candidates(key, rec) -> dict[str, str]:
    cands = {}
    if key is not None:
        cands["<chiave>"] = str(key)
    if isinstance(rec, dict):
        for f, v in rec.items():
            if isinstance(v, (str, int)):
                cands[f] = str(v)
    return cands


def _attrib(rec) -> dict:
    """Campi utili per l'attribuzione (licenza CC BY-SA delle immagini)."""
    out = {"author": "", "license": "", "source_url": ""}
    if not isinstance(rec, dict):
        return out
    for f, v in rec.items():
        if not isinstance(v, (str, int)):
            continue
        fl = f.lower()
        if not out["license"] and "licen" in fl:
            out["license"] = str(v)
        elif not out["author"] and any(k in fl for k in ("author", "artist", "creator", "user")):
            out["author"] = str(v)
        elif not out["source_url"] and "url" in fl:
            out["source_url"] = str(v)
    return out


# ---------------------------------------------------------------------------
# 1. Ispezione
# ---------------------------------------------------------------------------
def inspect_metadata(raw_dir: str | Path) -> None:
    raw = Path(raw_dir)
    print("=" * 70, "\nFILE PRESENTI in", raw)
    for p in sorted(raw.iterdir()):
        print(f"  {p.name:28s} {p.stat().st_size / 1e6:10.1f} MB")

    churches = _load_json(raw / "churches.json")
    print("=" * 70, "\nchurches.json:", type(churches).__name__, "con", len(churches), "voci")
    for k in list(churches)[:2]:
        print("  ", k, "->", _short(churches[k]))

    meta = _load_json(raw / "image_meta.json")
    print("=" * 70, "\nimage_meta.json:", type(meta).__name__, "con", len(meta), "voci")
    sample = list(meta.items())[:2] if isinstance(meta, dict) else list(enumerate(meta[:2]))
    for k, v in sample:
        print("  ", k, "->", _short(v, 500))
    rows, how = map_images_to_churches(meta, churches)
    print(f"  STRUTTURA RICONOSCIUTA: {how}  ({len(rows)} immagini collegate)")

    names = parse_style_names(raw / "style_names.txt")
    print("=" * 70, "\nstyle_names.txt:", len(names), "stili letti")
    for line in (raw / "style_names.txt").read_text(encoding="utf-8").splitlines()[:5]:
        print("   |", line)
    print("  Verifica dei 4 QID usati dal progetto:")
    for cls, q in config.STYLE_QIDS.items():
        print(f"    {cls:12s} {q:10s} -> nome nel dataset: {names.get(q, '??? NON TROVATO')}")

    hier = parse_hierarchy(raw / "parent_child_rel.txt", config.STYLE_QIDS)
    print("=" * 70, "\nparent_child_rel.txt (prime righe):")
    for line in (raw / "parent_child_rel.txt").read_text(encoding="utf-8").splitlines()[:6]:
        print("   |", line)
    print(f"  Sotto-stili collegati ai 4 stili: {len(hier)}")
    for q, cls in sorted(hier.items(), key=lambda x: x[1]):
        print(f"    {q:10s} {names.get(q, ''):45s} -> {cls}")

    lab = raw / "labels.zip"
    if lab.exists():
        print("=" * 70, "\nlabels.zip:")
        with zipfile.ZipFile(lab) as zf:
            for n in zf.namelist()[:15]:
                print("   ", n)
            for n in zf.namelist():
                if re.search(r"wc4_(train|val|test)\.txt$", n):
                    lines = zf.read(n).decode("utf-8", "replace").splitlines()
                    print(f"  {n}: {len(lines)} righe, esempi: {lines[:3]}")
    print("=" * 70, "\nFine ispezione.")


# ---------------------------------------------------------------------------
# 2. Anteprima dei conteggi (senza le immagini)
# ---------------------------------------------------------------------------
def preview_counts(raw_dir: str | Path) -> pd.DataFrame:
    raw = Path(raw_dir)
    churches = _load_json(raw / "churches.json")
    meta = _load_json(raw / "image_meta.json")
    rows, _ = map_images_to_churches(meta, churches)
    img_per_church = Counter(cid for _, cid, _ in rows)
    sub_map = parse_hierarchy(raw / "parent_child_rel.txt", config.STYLE_QIDS)

    table = []
    for label, sm in (("solo 4 stili principali", None), ("+ sotto-stili", sub_map)):
        per_cls_ch, per_cls_img = Counter(), Counter()
        for cid, ch in churches.items():
            cls = assign_class(_qids_in(ch.get("styles", [])), config.STYLE_QIDS, sm)
            if cls:
                per_cls_ch[cls] += 1
                per_cls_img[cls] += img_per_church.get(cid, 0)
        for cls in config.CLASSES:
            table.append({"criterio": label, "stile": cls,
                          "chiese": per_cls_ch[cls], "immagini": per_cls_img[cls]})
    df = pd.DataFrame(table)
    print(df.pivot(index="stile", columns="criterio", values=["chiese", "immagini"])
            .loc[config.CLASSES].to_string())
    return df


# ---------------------------------------------------------------------------
# 3. Indice immagine -> chiesa -> stile
# ---------------------------------------------------------------------------
def build_index(raw_dir: str | Path, out_csv: str | Path,
                use_substyles: bool = False) -> pd.DataFrame:
    raw = Path(raw_dir)
    churches = _load_json(raw / "churches.json")
    meta = _load_json(raw / "image_meta.json")
    sub_map = parse_hierarchy(raw / "parent_child_rel.txt", config.STYLE_QIDS) if use_substyles else None

    church_cls = {}
    for cid, ch in churches.items():
        cls = assign_class(_qids_in(ch.get("styles", [])), config.STYLE_QIDS, sub_map)
        if cls:
            church_cls[cid] = cls

    with zipfile.ZipFile(raw / IMAGES_ZIP) as zf:
        members = _zip_image_members(zf)
    print(f"Immagini nello zip: {len(members)}")
    by_stem, by_name = defaultdict(list), defaultdict(list)
    for m in members:
        p = PurePosixPath(m)
        by_stem[p.stem].append(m)
        by_name[p.name].append(m)

    rows, how = map_images_to_churches(meta, churches)
    print("Struttura di image_meta.json:", how)
    rows = [r for r in rows if r[1] in church_cls]

    # Quale campo di image_meta corrisponde al nome del file nello zip?
    hits = Counter()
    for key, cid, rec in rows[:3000]:
        for f, v in _scalar_candidates(key, rec).items():
            if v in by_stem or v in by_name or PurePosixPath(v).name in by_name:
                hits[f] += 1
    records = []
    if hits:
        field = hits.most_common(1)[0][0]
        print(f"Campo che identifica il file nello zip: '{field}'")
        for key, cid, rec in rows:
            v = _scalar_candidates(key, rec).get(field)
            if v is None:
                continue
            cands = by_stem.get(v) or by_name.get(v) or by_name.get(PurePosixPath(v).name) or []
            if len(cands) > 1:  # stesso nome in cartelle diverse: preferisci quella della chiesa
                cands = [c for c in cands if cid in c] or cands[:1]
            if cands:
                records.append({"member": cands[0], "church_id": cid, **_attrib(rec)})
    else:
        # Ripiego: la chiesa è scritta nel percorso dello zip (es. images/Q1389/x.jpg)
        print("Nessun campo coincide coi nomi dei file: provo a leggere la chiesa dal percorso.")
        for m in members:
            cid = next((q for q in QID_RE.findall(m) if q in church_cls), None)
            if cid:
                records.append({"member": m, "church_id": cid, "author": "", "license": "", "source_url": ""})

    df = pd.DataFrame(records).drop_duplicates("member")
    if df.empty:
        raise RuntimeError("Nessuna immagine collegata: controllare l'output di inspect_metadata().")
    df["style"] = df["church_id"].map(church_cls)
    df["church_name"] = df["church_id"].map(lambda c: churches[c].get("name", ""))
    df["country"] = df["church_id"].map(lambda c: (churches[c].get("country") or {}).get("name", ""))
    df["lat"] = df["church_id"].map(lambda c: (churches[c].get("location") or {}).get("lat"))
    df["lon"] = df["church_id"].map(lambda c: (churches[c].get("location") or {}).get("lon"))

    expected = sum(1 for _, cid, _ in rows)
    print(f"Immagini collegate: {len(df)} su {expected} attese ({len(df) / max(1, expected):.1%})")
    print(df.groupby("style").agg(immagini=("member", "size"), chiese=("church_id", "nunique"))
            .loc[config.CLASSES].to_string())
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Indice salvato in", out_csv)
    return df


# ---------------------------------------------------------------------------
# 4. Estrazione e ridimensionamento
# ---------------------------------------------------------------------------
def _resize_short_side(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    s = short / min(w, h)
    if s >= 1:  # non ingrandiamo le immagini piccole
        return img
    return img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)


def extract_resize(index_csv: str | Path, zip_path: str | Path, out_dir: str | Path,
                   short_side: int = config.IMG_SHORT_SIDE) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    out = Path(out_dir)
    for cls in config.CLASSES:
        (out / cls).mkdir(parents=True, exist_ok=True)
    paths, md5s, sizes, errors = [], [], [], []
    with zipfile.ZipFile(zip_path) as zf:
        for row in tqdm(df.itertuples(), total=len(df), desc="Estrazione"):
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", PurePosixPath(row.member).stem)[:80]
            name = stem if stem.startswith(row.church_id) else f"{row.church_id}_{stem}"
            rel = f"{row.style}/{name}.jpg"
            try:
                data = zf.read(row.member)
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                img = _resize_short_side(img, short_side)
                img.save(out / rel, "JPEG", quality=config.JPEG_QUALITY)
                paths.append(rel)
                md5s.append(hashlib.md5(data).hexdigest())
                sizes.append(f"{img.size[0]}x{img.size[1]}")
            except Exception as e:  # immagine corrotta o formato strano
                errors.append((row.member, repr(e)[:120]))
                paths.append(None); md5s.append(None); sizes.append(None)
    df["path"], df["md5"], df["size"] = paths, md5s, sizes
    df = df.dropna(subset=["path"])
    df.to_csv(out / "index_processed.csv", index=False)
    print(f"Immagini salvate: {len(df)}  |  errori: {len(errors)}")
    for m, e in errors[:10]:
        print("   errore:", m, e)
    return df


def save_to_drive(processed_dir: str | Path, drive_dir: str | Path) -> Path:
    """Impacchetta le immagini in un solo .tar su Drive (molti file piccoli su Drive sono lenti)."""
    drive = Path(drive_dir)
    drive.mkdir(parents=True, exist_ok=True)
    tar_path = drive / config.PROCESSED_TAR_NAME
    src = Path(processed_dir)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(src, arcname=src.name)
    print(f"Salvato {tar_path} ({tar_path.stat().st_size / 1e6:.0f} MB)")
    return tar_path


def restore_from_drive(drive_dir: str | Path = config.DRIVE_DATA_DIR,
                       dest: str | Path = "/content") -> Path:
    """Da usare nei notebook successivi: ripristina /content/processed in pochi secondi."""
    tar_path = Path(drive_dir) / config.PROCESSED_TAR_NAME
    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # Python vecchio senza il parametro filter
            tar.extractall(dest)
    out = Path(dest) / "processed"
    print("Immagini ripristinate in", out)
    return out


# ---------------------------------------------------------------------------
# 5. Split per chiesa
# ---------------------------------------------------------------------------
def make_splits(processed_csv: str | Path, out_csv: str | Path,
                ratios=config.SPLIT_RATIOS, seed: int = config.SEED) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(processed_csv)
    ch = df.groupby("church_id")["style"].first().reset_index()
    tr, va, te = ratios
    ch_train, ch_rest = train_test_split(ch, test_size=va + te, stratify=ch["style"], random_state=seed)
    ch_val, ch_test = train_test_split(ch_rest, test_size=te / (va + te),
                                       stratify=ch_rest["style"], random_state=seed)
    split_of = {**{c: "train" for c in ch_train.church_id},
                **{c: "val" for c in ch_val.church_id},
                **{c: "test" for c in ch_test.church_id}}
    df["split"] = df["church_id"].map(split_of)

    # Controllo 1: nessuna chiesa in due split
    assert df.groupby("church_id")["split"].nunique().max() == 1, "chiesa in più split!"
    # Controllo 2: foto identiche (stesso md5) in split diversi -> le togliamo da val/test
    dup = df[df.duplicated("md5", keep=False)]
    leak = dup.groupby("md5")["split"].nunique()
    leak_md5 = set(leak[leak > 1].index)
    removed = df["md5"].isin(leak_md5) & (df["split"] != "train")
    df = df[~removed]
    print(f"Duplicati esatti tra split rimossi da val/test: {int(removed.sum())}")

    cols = ["path", "style", "split", "church_id", "church_name", "country",
            "lat", "lon", "author", "license", "source_url", "md5"]
    df = df[[c for c in cols if c in df.columns]].sort_values(["split", "style", "path"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Split salvato in", out_csv)
    return df


# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
def report(splits_csv: str | Path, processed_dir: str | Path, fig_dir: str | Path) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(splits_csv)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = ["train", "val", "test"]

    imgs = pd.crosstab(df["style"], df["split"]).reindex(index=config.CLASSES, columns=order).fillna(0).astype(int)
    chs = (df.groupby(["style", "split"])["church_id"].nunique().unstack()
             .reindex(index=config.CLASSES, columns=order).fillna(0).astype(int))
    imgs["totale"], chs["totale"] = imgs.sum(axis=1), chs.sum(axis=1)
    table = pd.concat({"immagini": imgs, "chiese": chs}, axis=1)
    table.loc["totale"] = table.sum()
    print(table.to_string())
    table.to_csv(fig_dir.parent / "conteggi_dataset.csv")

    per_ch = df.groupby("church_id").size()
    print(f"\nImmagini per chiesa: media {per_ch.mean():.2f}, mediana {per_ch.median():.0f}, "
          f"massimo {per_ch.max()} ({df[df.church_id == per_ch.idxmax()].church_name.iloc[0]})")
    print("Squilibrio tra classi (train, max/min):",
          f"{imgs['train'].max() / max(1, imgs['train'].min()):.2f}")

    # Grafico a barre
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(config.CLASSES))
    w = 0.27
    for i, s in enumerate(order):
        ax.bar([xi + (i - 1) * w for xi in x], imgs[s], width=w, label=s)
    ax.set_xticks(list(x), [config.CLASS_NAMES_IT[c] for c in config.CLASSES])
    ax.set_ylabel("immagini")
    ax.set_title("WikiChurches – immagini per stile e split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "01_conteggi_per_split.png", dpi=150)
    plt.close(fig)

    # Griglia di esempi (6 per classe, dal train)
    k = 6
    fig, axes = plt.subplots(len(config.CLASSES), k, figsize=(2 * k, 2.2 * len(config.CLASSES)))
    for r, cls in enumerate(config.CLASSES):
        sub = df[(df["style"] == cls) & (df["split"] == "train")]
        sub = sub.sample(min(k, len(sub)), random_state=config.SEED)
        for c in range(k):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(sub):
                ax.imshow(Image.open(Path(processed_dir) / sub.iloc[c]["path"]))
        axes[r, 0].set_title(config.CLASS_NAMES_IT[cls], loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_esempi_per_stile.png", dpi=110)
    plt.close(fig)
    print("Figure salvate in", fig_dir)
    return table


# ---------------------------------------------------------------------------
# Riga di comando
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "preview"):
        s = sub.add_parser(name)
        s.add_argument("--raw", default=config.COLAB_RAW_DIR)
    s = sub.add_parser("build-index")
    s.add_argument("--raw", default=config.COLAB_RAW_DIR)
    s.add_argument("--out", default=f"{config.COLAB_RAW_DIR}/index.csv")
    s.add_argument("--use-substyles", action="store_true")
    s = sub.add_parser("extract")
    s.add_argument("--index", default=f"{config.COLAB_RAW_DIR}/index.csv")
    s.add_argument("--zip", default=f"{config.COLAB_RAW_DIR}/{IMAGES_ZIP}")
    s.add_argument("--out", default=config.COLAB_PROCESSED_DIR)
    s = sub.add_parser("split")
    s.add_argument("--processed-csv", default=f"{config.COLAB_PROCESSED_DIR}/index_processed.csv")
    s.add_argument("--out", default="data/splits/splits.csv")
    s = sub.add_parser("report")
    s.add_argument("--splits", default="data/splits/splits.csv")
    s.add_argument("--processed", default=config.COLAB_PROCESSED_DIR)
    s.add_argument("--figs", default="reports/figures")
    a = p.parse_args(argv)

    if a.cmd == "inspect":
        inspect_metadata(a.raw)
    elif a.cmd == "preview":
        preview_counts(a.raw)
    elif a.cmd == "build-index":
        build_index(a.raw, a.out, a.use_substyles)
    elif a.cmd == "extract":
        extract_resize(a.index, a.zip, a.out)
    elif a.cmd == "split":
        make_splits(a.processed_csv, a.out)
    elif a.cmd == "report":
        report(a.splits, a.processed, a.figs)


if __name__ == "__main__":
    main()
