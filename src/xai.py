"""Spiegabilità del modello (notebook 05 e app).

- LIME (Local Interpretable Model-agnostic Explanations), implementato come nelle slide:
  1. l'immagine è divisa in super-pixel (SLIC);
  2. si generano N versioni perturbate spegnendo a caso dei super-pixel (sostituiti dal loro colore medio);
  3. il modello predice ogni versione;
  4. ogni versione è pesata con un kernel esponenziale sulla distanza del coseno dall'immagine originale
     (località);
  5. una regressione lineare pesata (Ridge) approssima il modello attorno all'immagine: il coefficiente di ogni
     super-pixel è la sua importanza; l'R² della regressione misura la fedeltà locale della spiegazione.
  Si applica a qualsiasi modello: qui spiega il modello definitivo (CLIP + DINOv2 + testa MLP).
- Grad-CAM sulla VGG-16 (rete convoluzionale): gradiente dell'uscita scelta rispetto alle mappe di
  'block5_conv3', media spaziale come peso di ogni mappa, somma pesata e ReLU.
- Tre uscite spiegabili: stile, famiglia (somma degli stili della famiglia), contrasto (stile − stile rivale).
- Deletion test: si cancellano (sfocano) i pixel dal più al meno importante e si misura la caduta della
  probabilità; una spiegazione fedele fa cadere la probabilità più in fretta di un ordine casuale.
- Nome degli elementi: CLIP confronta il ritaglio di una zona importante con frasi come «a pointed gothic arch».
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from src import config, hierarchy as hi

# (frase per CLIP, nome italiano, stile tipico o None)
ELEMENTI = [
    ("a pointed gothic arch", "arco a sesto acuto", "Gothic"),
    ("an ogee arch", "arco inflesso", "Gothic"),
    ("a trefoil arch", "arco trilobato", "Gothic"),
    ("a gothic pinnacle", "pinnacolo gotico", "Gothic"),
    ("a gothic rose window", "rosone", "Gothic"),
    ("flying buttresses", "archi rampanti", "Gothic"),
    ("a round romanesque arch", "arco a tutto sesto", "Romanesque"),
    ("a romanesque blind arcade with small arches", "archetti pensili", "Romanesque"),
    ("a thick stone wall with small windows", "muratura massiccia con piccole finestre", "Romanesque"),
    ("a triangular pediment", "timpano triangolare", "Renaissance"),
    ("a curved segmental pediment", "timpano curvo", "Renaissance"),
    ("a serliana window", "serliana", "Renaissance"),
    ("a broken pediment", "timpano spezzato", "Baroque"),
    ("a twisted solomonic column", "colonna tortile (salomonica)", "Baroque"),
    ("baroque scrolls and volutes", "volute", "Baroque"),
    ("a statue in a niche", "statua in nicchia", "Baroque"),
    ("an ancient roman ruin", "rovina romana", "Classical"),
    ("the arches of a roman amphitheatre", "arcate di anfiteatro", "Classical"),
    ("a greek temple with columns", "tempio con colonne", "Classical"),
    ("a neoclassical portico with columns and a pediment", "pronao neoclassico", "Neoclassical"),
    ("a dome", "cupola", None),
    ("an oval porthole window", "oculo", None),
    ("a row of classical columns", "colonnato", None),
    ("a bell tower", "campanile", None),
    ("a window", "finestra", None),
    ("a door", "portale", None),
    ("a roof", "tetto", None),
    ("a plain wall", "muro", None),
    ("the sky", "cielo", None),
    ("trees", "alberi", None),
    ("people", "persone", None),
]
FRASI = [e[0] for e in ELEMENTI]
NOMI_IT = [e[1] for e in ELEMENTI]
STILE_ELEMENTO = [e[2] for e in ELEMENTI]
# Elementi annotati da MonuMAI → nome italiano dell'elenco (verifica del riconoscimento con CLIP)
MONUMAI_A_NOME = {
    "arco-apuntado": "arco a sesto acuto", "arco-conopial": "arco inflesso", "arco-trilobulado": "arco trilobato",
    "pinaculo-gotico": "pinnacolo gotico", "fronton": "timpano triangolare", "fronton-curvo": "timpano curvo",
    "serliana": "serliana", "fronton-partido": "timpano spezzato", "columna-salomonica": "colonna tortile (salomonica)",
}


# ---------------------------------------------------------------------------
# Immagini
# ---------------------------------------------------------------------------
def load_square(path, size: int = config.IMG_SIZE_HR, short: int = config.IMG_SHORT_SIDE_HR) -> np.ndarray:
    """Lato corto a `short` px, poi ritaglio centrale size × size (lo stesso usato dalla VGG-16 in valutazione)."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = short / min(im.size)
    im = im.resize((max(size, round(im.width * s)), max(size, round(im.height * s))), Image.LANCZOS)
    l, t = (im.width - size) // 2, (im.height - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)))


def output_columns(p: np.ndarray, rival: tuple[int, int] | None = None) -> np.ndarray:
    """Colonne spiegabili: 6 stili, 3 famiglie, (contrasto stile − rivale)."""
    cols = [p, hi.family_probs(p)]
    if rival is not None:
        a, b = rival
        cols.append((p[:, a] - p[:, b])[:, None])
    return np.concatenate(cols, axis=1)


def col_family(f: int) -> int:
    return len(config.CLASSES) + f


COL_CONTRASTO = len(config.CLASSES) + len(config.FAMIGLIE)


# ---------------------------------------------------------------------------
# Modello definitivo
# ---------------------------------------------------------------------------
class FinalModel:
    """CLIP + DINOv2 congelati + testa MLP, con predizione su immagini numpy (H, W, 3) uint8."""

    def __init__(self, head_path):
        import torch
        from src import foundation as fd
        self.torch, self.fd = torch, fd
        self.head, self.std, self.meta = fd.load_head(head_path)
        self.nets = {n: fd.load_backbone(n) for n in self.meta["rete"].split("+")}
        self._txt = None

    def _embed(self, name, images, batch=32):
        torch = self.torch
        model, proc = self.nets[name]
        out = []
        for i in range(0, len(images), batch):
            pil = [Image.fromarray(np.asarray(x).astype(np.uint8)) for x in images[i:i + batch]]
            x = proc(images=pil, return_tensors="pt")["pixel_values"].to(self.fd.DEVICE, dtype=torch.float16)
            with torch.no_grad():
                if name == "clip":
                    f = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
                else:
                    h = model(pixel_values=x).last_hidden_state
                    f = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
            out.append(f.float().cpu().numpy())
        return np.concatenate(out)

    def features(self, images):
        return np.concatenate([self._embed(n, images) for n in self.nets], axis=1)

    def predict(self, images, tta: bool = False) -> np.ndarray:
        p = self.fd.predict_proba(self.head, self.std(self.features(images)).astype(np.float32))
        if tta:
            p = (p + self.predict([np.ascontiguousarray(np.asarray(x)[:, ::-1]) for x in images])) / 2
        return p

    def name_elements(self, crops) -> np.ndarray:
        """Probabilità (n_ritagli, n_elementi) che ogni ritaglio mostri ciascun elemento (CLIP zero-shot)."""
        torch = self.torch
        model, _ = self.nets["clip"]
        if self._txt is None:
            from transformers import CLIPTokenizer
            tok = CLIPTokenizer.from_pretrained(config.BACKBONES["clip"])
            t = tok([f"a photo of {f}, architectural detail" for f in FRASI], padding=True, return_tensors="pt")
            t = {k: v.to(self.fd.DEVICE) for k, v in t.items()}
            with torch.no_grad():
                e = model.text_projection(model.text_model(**t).pooler_output).float().cpu().numpy()
            self._txt = e / np.linalg.norm(e, axis=1, keepdims=True)
        im = self._embed("clip", crops)
        im = im / np.linalg.norm(im, axis=1, keepdims=True)
        z = 100.0 * im @ self._txt.T
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------
def superpixels(img: np.ndarray, n_segments: int = 60) -> np.ndarray:
    from skimage.segmentation import slic
    seg = slic(img, n_segments=n_segments, compactness=20, start_label=0)
    if seg.max() + 1 < n_segments // 4:                    # immagine quasi uniforme: griglia regolare
        g = int(np.sqrt(n_segments))
        r = np.minimum(np.arange(img.shape[0]) * g // img.shape[0], g - 1)
        c = np.minimum(np.arange(img.shape[1]) * g // img.shape[1], g - 1)
        seg = r[:, None] * g + c[None, :]
    return seg


def lime_explain(predict_columns, img: np.ndarray, columns, num_samples: int = 500, n_segments: int = 60,
                 kernel_width: float = 0.25, seed: int = config.SEED, batch: int = 100):
    """predict_columns(lista di immagini) -> matrice (n, colonne). Restituisce segmenti, pesi per colonna e R²."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(seed)
    seg = superpixels(img, n_segments)
    n = seg.max() + 1
    mean_col = np.zeros((n, 3), dtype=np.float32)
    for s in range(n):
        mean_col[s] = img[seg == s].mean(0)
    Z = rng.integers(0, 2, size=(num_samples, n)).astype(bool)
    Z[0] = True                                            # prima riga = immagine originale
    Y = []
    for i in range(0, num_samples, batch):
        imgs = []
        for z in Z[i:i + batch]:
            x = img.copy()
            off = ~z[seg]
            x[off] = mean_col[seg[off]].astype(np.uint8)
            imgs.append(x)
        Y.append(predict_columns(imgs))
    Y = np.concatenate(Y)
    zf = Z.astype(float)
    cos = zf @ zf[0] / (np.linalg.norm(zf, axis=1) * np.linalg.norm(zf[0]) + 1e-12)
    w = np.sqrt(np.exp(-((1 - cos) ** 2) / kernel_width ** 2))
    pesi, r2 = {}, {}
    for c in columns:
        reg = Ridge(alpha=1.0, random_state=seed).fit(zf, Y[:, c], sample_weight=w)
        pesi[c] = reg.coef_
        r2[c] = float(reg.score(zf, Y[:, c], sample_weight=w))
    return {"segments": seg, "pesi": pesi, "r2": r2, "pred": Y[0]}


def top_regions(seg: np.ndarray, weights: np.ndarray, k: int = 3, min_side: int = 128, margin: float = 0.2):
    """Riquadri dei k super-pixel con peso positivo più alto, allargati ad almeno min_side px."""
    H, W = seg.shape
    boxes = []
    for s in np.argsort(-weights)[:k]:
        if weights[s] <= 0:
            break
        ys, xs = np.nonzero(seg == s)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        h = max(min_side, (ys.max() - ys.min()) * (1 + 2 * margin))
        w = max(min_side, (xs.max() - xs.min()) * (1 + 2 * margin))
        y0, x0 = int(max(0, cy - h / 2)), int(max(0, cx - w / 2))
        y1, x1 = int(min(H, y0 + h)), int(min(W, x0 + w))
        boxes.append((x0, y0, x1, y1, float(weights[s])))
    return boxes


def crop(img: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box[:4]
    return np.ascontiguousarray(img[y0:y1, x0:x1])


# ---------------------------------------------------------------------------
# Grad-CAM (VGG-16)
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, layer: str = "block5_conv3"):
        from tensorflow import keras
        self.model = model
        self.grad_model = keras.Model(model.inputs, [model.get_layer(layer).output, model.output])

    def preprocess(self, imgs):
        from src.data.dataset import VGG_MEAN_BGR
        x = np.asarray(imgs, dtype=np.float32)
        return x[..., ::-1] - VGG_MEAN_BGR.numpy()

    def predict(self, imgs, batch: int = 4) -> np.ndarray:
        x = self.preprocess(imgs)
        return np.concatenate([self.model.predict(x[i:i + batch], verbose=0) for i in range(0, len(x), batch)])

    def __call__(self, img: np.ndarray, target) -> np.ndarray:
        """target: ('stile', k) | ('famiglia', f) | ('contrasto', (a, b)). Mappa normalizzata in [0, 1]."""
        import tensorflow as tf
        x = tf.constant(self.preprocess([img]))
        kind, val = target
        with tf.GradientTape() as tape:
            conv, p = self.grad_model(x, training=False)
            if kind == "stile":
                t = p[:, val]
            elif kind == "famiglia":
                idx = [i for i, f in enumerate(hi.style_to_family()) if f == val]
                t = tf.reduce_sum(tf.gather(p, idx, axis=1), axis=1)
            else:
                t = p[:, val[0]] - p[:, val[1]]
        g = tape.gradient(t, conv)[0]
        w = tf.reduce_mean(g, axis=(0, 1))
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * w, axis=-1)).numpy()
        return cam / (cam.max() + 1e-8)


def upsample(m: np.ndarray, shape) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.float32), mode="F").resize((shape[1], shape[0]), Image.BILINEAR))


# ---------------------------------------------------------------------------
# Deletion test
# ---------------------------------------------------------------------------
def deletion_curve(predict_target, img: np.ndarray, importance: np.ndarray, steps: int = 10, blur: int = 16):
    """Probabilità del bersaglio cancellando (sfocando) 0%, 10%, …, 100% dei pixel in ordine di importanza."""
    blurred = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(blur)))
    order = np.argsort(-importance.ravel(), kind="stable")
    imgs = []
    for i in range(steps + 1):
        m = np.zeros(order.size, dtype=bool)
        m[order[:int(round(order.size * i / steps))]] = True
        m = m.reshape(importance.shape)
        x = img.copy()
        x[m] = blurred[m]
        imgs.append(x)
    return np.asarray(predict_target(imgs), dtype=float)


def auc(curve: np.ndarray) -> float:
    """Area sotto la curva di cancellazione (asse x da 0 a 1): più bassa = spiegazione più fedele."""
    c = np.asarray(curve, dtype=float)
    return float(((c[1:] + c[:-1]) / 2).mean())


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def overlay_lime(img: np.ndarray, seg: np.ndarray, w: np.ndarray, top: int = 8) -> np.ndarray:
    """Super-pixel più importanti in verde (a favore) e rossi (contro), il resto attenuato."""
    from skimage.segmentation import mark_boundaries
    out = img.astype(np.float32) * 0.45
    m = np.abs(w).max() + 1e-12
    for s in np.argsort(-np.abs(w))[:top]:
        mask = seg == s
        a = 0.55 + 0.45 * abs(w[s]) / m
        col = np.array([60, 200, 90]) if w[s] > 0 else np.array([220, 60, 60])
        out[mask] = img[mask] * a + col * (1 - a) * 0.6
    sel = np.isin(seg, np.argsort(-np.abs(w))[:top])
    out = mark_boundaries(out.clip(0, 255).astype(np.uint8), np.where(sel, seg, -1), color=(1, 1, 0.3))
    return (out * 255).astype(np.uint8) if out.max() <= 1.0 else out.astype(np.uint8)


def overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm
    heat = cm.jet(upsample(cam, img.shape[:2]))[..., :3] * 255
    return (img * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
