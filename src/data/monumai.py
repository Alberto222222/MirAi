"""Dataset esterno MonuMAI (Lamas et al., 2021) usato per integrare il training.

MonuMAI contiene 1.514 fotografie di facciate di monumenti in quattro stili (Ispano-musulmano,
Gotico, Rinascimentale, Barocco), con riquadri annotati da esperti su 15 elementi architettonici
(timpano triangolare, serliana, colonna salomonica, arco a sesto acuto, ...).
Licenza CC BY 4.0 – https://github.com/ari-dasci/OD-MonuMAI

Regole d'uso nel progetto:
- solo stili presenti anche in WikiChurches (Gotico, Rinascimentale, Barocco);
- le immagini entrano SOLO nel training: validation e test restano quelli di WikiChurches;
- prima dell'uso si eliminano le foto quasi identiche a immagini di validation o test
  (hash percettivo), per evitare che il test contenga immagini già viste in addestramento.
"""
from __future__ import annotations

import re
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src import config

IMG_EXT = {".jpg", ".jpeg", ".png"}


def clone(dest: str | Path = config.COLAB_MONUMAI_RAW) -> Path:
    """Scarica il repository pubblico di MonuMAI (solo l'ultima versione)."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", config.MONUMAI_REPO, str(dest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
    root = next(dest.glob("MonuMAI_dataset*"), dest)
    print("MonuMAI in", root)
    return root


def _monument(stem: str) -> str:
    """Nome del monumento dal nome del file: 'san_pedro_cordoba7' -> 'san_pedro_cordoba'."""
    return re.sub(r"[\s_\-()\d]+$", "", stem.lower()) or stem.lower()


def build_index(root: str | Path, styles=config.MONUMAI_STYLES) -> pd.DataFrame:
    """Una riga per immagine: percorso originale, stile, etichetta (-1 per stili fuori dalle 4 classi), monumento."""
    rows = []
    label_of = {c: i for i, c in enumerate(config.CLASSES)}
    for style in styles:
        folder = Path(root) / style
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                rows.append({"src": str(f), "file": f.name, "style": style, "label": label_of.get(style, -1),
                             "monument": _monument(f.stem), "source": "monumai"})
    df = pd.DataFrame(rows)
    print("Immagini MonuMAI per stile:", df["style"].value_counts().reindex(styles).to_dict())
    return df


def resize(df: pd.DataFrame, out_dir: str | Path = config.COLAB_MONUMAI_DIR) -> pd.DataFrame:
    """Lato corto a IMG_SHORT_SIDE px, JPEG; scarta i file illeggibili.
    La colonna 'path' contiene il percorso assoluto dell'immagine ridimensionata."""
    out_dir = Path(out_dir)
    keep, paths = [], []
    for i, r in df.iterrows():
        dest = out_dir / r["style"] / (Path(r["file"]).stem + ".jpg")
        try:
            if not dest.exists():
                im = ImageOps.exif_transpose(Image.open(r["src"])).convert("RGB")
                s = config.IMG_SHORT_SIDE / min(im.size)
                im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
                dest.parent.mkdir(parents=True, exist_ok=True)
                im.save(dest, "JPEG", quality=92)
            keep.append(i)
            paths.append(str(dest))
        except Exception as e:  # file corrotto o formato non leggibile
            print("scartata:", r["file"], type(e).__name__)
    out = df.loc[keep].copy()
    out["path"] = paths
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Controllo dei duplicati con hash percettivo
# ---------------------------------------------------------------------------
def phash_bits(paths) -> np.ndarray:
    """Hash percettivo a 64 bit di ogni immagine, come matrice booleana (n, 64).
    Due foto uguali ma ridimensionate o ricompresse hanno hash quasi identici."""
    # Algoritmo pHash: scala di grigi 32x32 -> DCT 2D -> blocco 8x8 delle basse frequenze
    # -> bit = coefficiente sopra la mediana del blocco
    from scipy.fft import dct
    out = np.zeros((len(paths), 64), dtype=bool)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
        low = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
        out[i] = (low > np.median(low)).flatten()
    return out


def nearest(ext_bits: np.ndarray, ref_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per ogni immagine esterna: distanza di Hamming minima da un'immagine di riferimento e indice."""
    a, b = ext_bits.astype(np.int16), ref_bits.astype(np.int16)
    # bit diversi = n_bit - concordanze sui bit a 1 - concordanze sui bit a 0
    dist = 64 - (a @ b.T) - ((1 - a) @ (1 - b).T)
    return dist.min(axis=1), dist.argmin(axis=1)


def remove_duplicates(ext: pd.DataFrame, ref: pd.DataFrame, ref_root: str | Path,
                      max_dist: int = config.PHASH_MAX_DIST) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elimina dalle immagini esterne quelle quasi identiche a un'immagine di riferimento
    (validation + test). Restituisce (immagini tenute, coppie di duplicati trovate)."""
    ext_bits = phash_bits(ext["path"])
    ref_bits = phash_bits([str(Path(ref_root) / p) for p in ref["path"]])
    d, j = nearest(ext_bits, ref_bits)
    ext = ext.assign(phash_dist=d, ref_path=ref["path"].to_numpy()[j],
                     ref_split=ref["split"].to_numpy()[j], ref_style=ref["style"].to_numpy()[j])
    dup = ext[ext["phash_dist"] <= max_dist]
    print(f"Duplicati (distanza ≤ {max_dist} bit su 64): {len(dup)} su {len(ext)}")
    return ext[ext["phash_dist"] > max_dist].reset_index(drop=True), dup.reset_index(drop=True)


def balance(df: pd.DataFrame, per_class: int | None = None, seed: int = config.SEED) -> pd.DataFrame:
    """Stesso numero di immagini per stile (di default quello del Rinascimentale), così che la
    provenienza 'MonuMAI' non diventi un indizio dello stile."""
    n = per_class or int((df["style"] == "Renaissance").sum())
    parts = [g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("style")]
    return pd.concat(parts).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Annotazioni degli elementi architettonici (usate dalla spiegabilità e dall'app)
# ---------------------------------------------------------------------------
def read_annotations(root: str | Path, styles=config.MONUMAI_STYLES) -> pd.DataFrame:
    """Riquadri degli elementi (formato Pascal VOC): file, stile, elemento, xmin, ymin, xmax, ymax,
    larghezza e altezza dell'immagine originale."""
    rows = []
    for style in styles:
        for x in sorted((Path(root) / style / "xml").glob("*.xml")):
            try:
                t = ET.parse(x).getroot()
            except ET.ParseError:
                continue
            size = t.find("size")
            w = int(float(size.findtext("width", "0"))) if size is not None else 0
            h = int(float(size.findtext("height", "0"))) if size is not None else 0
            for o in t.iter("object"):
                bb = o.find("bndbox")
                rows.append({"xml": x.name, "file": t.findtext("filename", x.stem + ".jpg"),
                             "style": style, "element": o.findtext("name", "").strip(),
                             "xmin": float(bb.findtext("xmin")), "ymin": float(bb.findtext("ymin")),
                             "xmax": float(bb.findtext("xmax")), "ymax": float(bb.findtext("ymax")),
                             "width": w, "height": h})
    return pd.DataFrame(rows)


def save_to_drive(local_dir: str | Path = config.COLAB_MONUMAI_DIR,
                  drive_dir: str | Path = config.DRIVE_DATA_DIR) -> Path:
    tar_path = Path(drive_dir) / config.MONUMAI_TAR_NAME
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(local_dir, arcname=Path(local_dir).name)
    print(f"Salvato {tar_path} ({tar_path.stat().st_size / 1e6:.0f} MB)")
    return tar_path


def element_crops(ann: pd.DataFrame, root: str | Path, out_dir: str | Path, element_style: dict[str, str],
                  short_side: int, min_side: int = 100, margin: float = 0.15) -> pd.DataFrame:
    """Ritagli degli elementi architettonici diagnostici (es. arco a sesto acuto → Gotico).

    Si tengono solo gli elementi associati a uno stile e coerenti con lo stile dell'edificio, con lato
    minimo min_side px nell'immagine originale; margine del 15% attorno al riquadro; lato corto portato
    a short_side px. Restituisce una riga per ritaglio (percorso relativo a out_dir/.., stile, elemento)."""
    out_dir = Path(out_dir)
    label_of = {c: i for i, c in enumerate(config.CLASSES)}
    ann = ann[ann["element"].map(element_style).eq(ann["style"])]
    rows = []
    for (style, fname), g in ann.groupby(["style", "file"]):
        src = Path(root) / style / fname
        if not src.exists():
            alt = list((Path(root) / style).glob(Path(fname).stem + ".*"))
            if not alt:
                continue
            src = alt[0]
        try:
            im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        for k, r in enumerate(g.itertuples()):
            sx = im.width / r.width if r.width else 1.0
            sy = im.height / r.height if r.height else 1.0
            x0, x1, y0, y1 = r.xmin * sx, r.xmax * sx, r.ymin * sy, r.ymax * sy
            w, h = x1 - x0, y1 - y0
            if min(w, h) < min_side:
                continue
            box = (max(0, x0 - margin * w), max(0, y0 - margin * h), min(im.width, x1 + margin * w),
                   min(im.height, y1 + margin * h))
            crop = im.crop(tuple(round(v) for v in box))
            s = short_side / min(crop.size)
            crop = crop.resize((round(crop.width * s), round(crop.height * s)), Image.LANCZOS)
            rel = f"{out_dir.name}/{style}/{Path(fname).stem}_{k}.jpg"
            (out_dir / style).mkdir(parents=True, exist_ok=True)
            crop.save(out_dir.parent / rel, "JPEG", quality=92)
            rows.append({"path": rel, "style": style, "label": label_of[style], "element": r.element,
                         "monument": _monument(Path(fname).stem), "source_file": f"{style}/{fname}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Zone «tipo LIME» dalle foto MonuMAI annotate (verifica dei nomi degli elementi nelle condizioni dell'app)
# ---------------------------------------------------------------------------
def image_path(root: str | Path, style: str, fname: str) -> Path | None:
    src = Path(root) / style / fname
    if src.exists():
        return src
    alt = list((Path(root) / style).glob(Path(fname).stem + ".*"))
    return alt[0] if alt else None


def _inter(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def lime_like_regions(ann: pd.DataFrame, root: str | Path, element_style: dict[str, str], max_pos: int = 3,
                      max_neg: int = 3, size: int = 448, short: int = 512, seed: int = config.SEED):
    """Per ogni foto MonuMAI: stessa geometria dell'app (lato corto 512, ritaglio centrale 448), super-pixel SLIC e
    riquadri costruiti come le zone LIME (xai.box_of_segment). Ogni riquadro diventa:
    - esempio dell'elemento diagnostico e (coerente con lo stile dell'edificio) se contiene almeno il 60% del
      riquadro annotato di e e l'elemento occupa almeno il 15% del riquadro (elementi piccoli), oppure se l'elemento
      occupa almeno metà del riquadro (elementi grandi, di cui la zona mostra una parte);
    - esempio «altro» se non tocca nessun elemento annotato (di qualsiasi tipo: < 10% di ogni elemento e < 5%
      del riquadro);
    - scartato negli altri casi (ambiguo).
    Restituisce, una foto alla volta, (ritagli, etichette, monumento, file)."""
    from src import xai
    rng = np.random.default_rng(seed)
    for (style, fname), g in ann.groupby(["style", "file"]):
        src = image_path(root, style, fname)
        if src is None:
            continue
        try:
            im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        W0, H0 = im.size
        s = short / min(W0, H0)
        W1, H1 = max(size, round(W0 * s)), max(size, round(H0 * s))
        l, t = (W1 - size) // 2, (H1 - size) // 2
        img = np.asarray(im.resize((W1, H1), Image.LANCZOS).crop((l, t, l + size, t + size)))
        boxes_ann = []
        for r in g.itertuples():
            sx = W0 / r.width if r.width else 1.0
            sy = H0 / r.height if r.height else 1.0
            b = (r.xmin * sx * W1 / W0 - l, r.ymin * sy * H1 / H0 - t, r.xmax * sx * W1 / W0 - l, r.ymax * sy * H1 / H0 - t)
            diag = element_style.get(r.element) == style
            boxes_ann.append((b, r.element if diag else None))
        seg = xai.superpixels(img)
        pos, neg = {}, []
        for sp in rng.permutation(seg.max() + 1):
            bx = xai.box_of_segment(seg, int(sp))
            ab = _area(bx)
            best, best_frac, tocca = None, 0.0, False
            for b, el in boxes_ann:
                aa = _area(b)
                if aa <= 0:
                    continue
                cov_el, cov_box = _inter(bx, b) / aa, _inter(bx, b) / ab
                if cov_el >= 0.10 or cov_box >= 0.05:
                    tocca = True
                if el and ((cov_el >= 0.60 and cov_box >= 0.15) or cov_box >= 0.50) and cov_box > best_frac:
                    best, best_frac = el, cov_box
            if best is not None:
                if len(pos.setdefault(best, [])) < max_pos:
                    pos[best].append(bx)
            elif not tocca and len(neg) < max_neg:
                neg.append(bx)
        crops, labels = [], []
        for el, bxs in pos.items():
            for bx in bxs:
                crops.append(np.ascontiguousarray(img[bx[1]:bx[3], bx[0]:bx[2]])); labels.append(el)
        for bx in neg:
            crops.append(np.ascontiguousarray(img[bx[1]:bx[3], bx[0]:bx[2]])); labels.append(None)
        if crops:
            yield crops, labels, _monument(Path(fname).stem), f"{style}/{fname}"
