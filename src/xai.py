"""Spiegabilità del modello definitivo (notebook 05, 05c e app).

Schema della spiegazione di una foto (funzione `spiega`, usata dall'app):

    foto 448 × 448 ──► modello definitivo ──► famiglia, stile, stile rivale
          │
          └──► LIME ──► peso di ogni super-pixel per 3 uscite: famiglia, stile, contrasto (stile − rivale)
                  │
                  └──► zone più importanti per lo stile ──► ritaglio ingrandito
                                                              └──► nome dell'elemento (se affidabile)

- LIME (Local Interpretable Model-agnostic Explanations), implementato come nelle slide:
  1. l'immagine è divisa in super-pixel (SLIC, ~60);
  2. si generano N versioni perturbate spegnendo a caso dei super-pixel (sostituiti dal loro colore medio);
  3. il modello predice ogni versione;
  4. ogni versione è pesata con un kernel esponenziale sulla distanza del coseno dall'immagine originale
     (località);
  5. una regressione lineare pesata (Ridge) approssima il modello attorno all'immagine: il coefficiente di ogni
     super-pixel è la sua importanza; l'R² della regressione misura la fedeltà locale della spiegazione.
  LIME tratta il modello come una scatola nera: spiega esattamente il modello dell'app (CLIP + DINOv2 + testa MLP).
- Deletion test: si cancellano (sfocano) i pixel dal più al meno importante e si misura la caduta della
  probabilità; una spiegazione fedele fa cadere la probabilità più in fretta di un ordine casuale.
- Nomi degli elementi (ElementNamer): classificatore lineare sui vettori CLIP dei ritagli, addestrato su zone
  costruite come le zone LIME delle foto MonuMAI annotate (notebook 05c); si mostrano solo i nomi con precisione
  misurata ≥ 0,75 e sopra la soglia di confidenza, altrimenti la zona resta un «dettaglio decisivo».
- Grad-CAM sulla VGG-16 (rete convoluzionale, solo confronto nel notebook 05 e nel paper): gradiente dell'uscita
  rispetto alle mappe di 'block5_conv3', media spaziale come peso di ogni mappa, somma pesata e ReLU.
- Controllo di qualità: probabilità che un'immagine sia un disegno o un'incisione invece di una foto.
- Il vocabolario ELEMENTI (31 nomi, zero-shot) è quello del notebook 05, conservato per riproducibilità.
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

    def text_embed(self, groups, templates=("{}",)) -> np.ndarray:
        """Un vettore di testo CLIP per ogni gruppo di frasi: media normalizzata su frasi × modelli di frase."""
        torch = self.torch
        model, _ = self.nets["clip"]
        if not hasattr(self, "_tok"):
            from transformers import CLIPTokenizer
            self._tok = CLIPTokenizer.from_pretrained(config.BACKBONES["clip"])
        out = []
        for frasi in groups:
            frasi = [frasi] if isinstance(frasi, str) else list(frasi)
            t = self._tok([m.format(f) for f in frasi for m in templates], padding=True, return_tensors="pt")
            t = {k: v.to(self.fd.DEVICE) for k, v in t.items()}
            with torch.no_grad():
                e = model.text_projection(model.text_model(**t).pooler_output).float().cpu().numpy()
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            m = e.mean(0)
            out.append(m / np.linalg.norm(m))
        return np.stack(out)

    def image_embed(self, images) -> np.ndarray:
        e = self._embed("clip", images)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

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
# Nomi degli elementi, versione 2 (notebook 05b e app)
# ---------------------------------------------------------------------------
# Più frasi per elemento e più modelli di frase (media dei vettori di testo): riduce la dipendenza da una
# singola formulazione. Il vocabolario comprende elementi che nella versione 1 mancavano (bugnato, finestra con
# timpanino, merlatura…) e zone di sfondo da scartare.
TEMPLATE = ("a photo of {}.", "a close-up photo of {}, architectural detail.",
            "{} on a historic building.", "a detail of a facade showing {}.")
# (nome italiano, tipico, tipo, frasi).
# tipico: stile (es. "Gothic"), famiglia (es. "Classicismo moderno") o None se l'elemento compare in più famiglie
# (es. il timpano triangolare: templi antichi, Rinascimento, Barocco, Neoclassico).
# tipo: "diagnostico" = elemento annotato da MonuMAI (riconosciuto dal classificatore addestrato sui ritagli),
# "elemento" = solo zero-shot, "sfondo" = zona scartata.
VOCABOLARIO = [
    ("arco a sesto acuto", "Gothic", "diagnostico", ["a pointed gothic arch", "a gothic lancet arch window"]),
    ("arco inflesso", "Gothic", "diagnostico", ["an ogee arch", "a late gothic ogee arch over a doorway"]),
    ("arco trilobato", "Gothic", "diagnostico", ["a trefoil arch", "a three-lobed gothic arch"]),
    ("pinnacolo gotico", "Gothic", "diagnostico", ["a gothic pinnacle", "small gothic spires with crockets"]),
    ("timpano triangolare", None, "diagnostico", ["a triangular pediment", "a classical triangular gable"]),
    ("timpano curvo", "Classicismo moderno", "diagnostico", ["a curved segmental pediment", "a rounded arched pediment"]),
    ("serliana", "Renaissance", "diagnostico",
     ["a serliana window", "a palladian window with a central arch and two side openings"]),
    ("timpano spezzato", "Baroque", "diagnostico", ["a broken pediment", "a baroque pediment broken open at the top"]),
    ("colonna tortile (salomonica)", "Baroque", "diagnostico", ["a twisted solomonic column", "a spiral twisted column"]),
    ("rosone", "Gothic", "elemento", ["a rose window", "a large circular stained glass window"]),
    ("archi rampanti", "Gothic", "elemento", ["flying buttresses"]),
    ("archetti pensili", "Romanesque", "elemento",
     ["a row of small blind arches under the eaves", "romanesque lombard bands"]),
    ("portale strombato", "Romanesque", "elemento",
     ["a deep recessed romanesque portal with concentric round arches"]),
    ("muratura massiccia con piccole finestre", "Romanesque", "elemento", ["a thick stone wall with small windows"]),
    ("bugnato", "Classicismo moderno", "elemento",
     ["rusticated stone masonry", "a facade of large rough-cut stone blocks", "diamond-point rustication"]),
    ("volute", "Baroque", "elemento", ["baroque scrolls and volutes", "large curved scrolls on a facade"]),
    ("facciata concava e convessa", "Baroque", "elemento", ["a curved concave and convex baroque facade"]),
    ("rovina romana", "Classical", "elemento", ["an ancient roman ruin", "ancient stone ruins"]),
    ("arcate di anfiteatro", "Classical", "elemento", ["the arches of a roman amphitheatre"]),
    ("gradinate di teatro antico", "Classical", "elemento", ["the stone seats of an ancient theatre"]),
    ("tempio con colonne", "Classical", "elemento", ["a greek temple with columns", "an ancient temple"]),
    ("pronao neoclassico", "Neoclassical", "elemento", ["a neoclassical portico with columns and a pediment"]),
    ("arco a tutto sesto", None, "elemento", ["a round arch", "a semicircular arched doorway"]),
    ("bifora", None, "elemento", ["a mullioned window with two arched lights"]),
    ("finestra con cornice", None, "elemento", ["a window with a decorated stone frame"]),
    ("lesene", None, "elemento", ["flat pilasters on a facade"]),
    ("loggia ad archi", None, "elemento", ["an arcade of arches on columns", "a loggia"]),
    ("statua in nicchia", None, "elemento", ["a statue in a niche"]),
    ("merlatura", None, "elemento", ["battlements on top of a wall", "crenellations of a castle"]),
    ("balaustra", None, "elemento", ["a stone balustrade"]),
    ("cornicione", None, "elemento", ["a projecting cornice under the roof"]),
    ("cupola", None, "elemento", ["a dome"]),
    ("oculo", None, "elemento", ["an oval porthole window", "a round oculus window"]),
    ("colonnato", None, "elemento", ["a row of classical columns"]),
    ("campanile o torre", None, "elemento", ["a bell tower", "a tower"]),
    ("finestra", None, "elemento", ["a plain window"]),
    ("portale", None, "elemento", ["a door", "an entrance doorway"]),
    ("tetto", None, "elemento", ["a roof", "roof tiles"]),
    ("muro in mattoni", None, "elemento", ["a brick wall"]),
    ("muro intonacato", None, "elemento", ["a plain plastered wall"]),
    ("cielo", None, "sfondo", ["the sky", "clouds"]),
    ("alberi", None, "sfondo", ["trees", "leaves and branches"]),
    ("persone", None, "sfondo", ["people", "tourists"]),
    ("automobili", None, "sfondo", ["cars", "a street with parked cars"]),
    ("prato o terreno", None, "sfondo", ["grass", "the ground"]),
    ("acqua", None, "sfondo", ["water", "a river"]),
]
V_NOMI = [v[0] for v in VOCABOLARIO]
V_TIPICO = [v[1] for v in VOCABOLARIO]
V_TIPO = [v[2] for v in VOCABOLARIO]
V_FRASI = [v[3] for v in VOCABOLARIO]
DIAGNOSTICI = [n for n, t in zip(V_NOMI, V_TIPO) if t == "diagnostico"]
assert sorted(DIAGNOSTICI) == sorted(MONUMAI_A_NOME.values())
ALTRO = "altro"            # classe «nessun elemento diagnostico» del classificatore


def famiglia_tipica(tipico) -> int | None:
    """Indice della famiglia a cui rimanda uno stile o una famiglia tipica (None = nessuna)."""
    if tipico is None:
        return None
    if tipico in config.FAMIGLIE:
        return list(config.FAMIGLIE).index(tipico)
    return int(hi.style_to_family()[config.CLASSES.index(tipico)])


def nome_tipico(tipico) -> str:
    return config.CLASS_NAMES_IT.get(tipico, tipico)


# Foto o disegno? (controllo di qualità dei dati e avviso nell'app)
FRASI_FOTO = ["a photograph of a building", "a color photo of a church", "a black and white photograph of a building",
              "a photo of a monument", "a photo of ancient ruins"]
FRASI_NON_FOTO = ["an old engraving of a city", "a pen drawing of a building", "a painting of a building",
                  "an architectural drawing or plan", "a vintage illustration of a castle", "a map",
                  "a watercolor of a church"]


def prob_non_foto(fm: FinalModel, E: np.ndarray) -> np.ndarray:
    """Probabilità (CLIP zero-shot) che l'immagine sia un disegno, un'incisione, un dipinto o una pianta.
    E: vettori immagine CLIP normalizzati."""
    T = fm.text_embed(FRASI_FOTO + FRASI_NON_FOTO)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    z = 100.0 * E @ T.T
    z = np.exp(z - z.max(1, keepdims=True))
    p = z / z.sum(1, keepdims=True)
    return p[:, len(FRASI_FOTO):].sum(1)


def fit_probe(E: np.ndarray, nomi, groups, seed: int = config.SEED, Cs=(0.001, 0.01, 0.1, 1.0)):
    """Classificatore lineare (regressione logistica multinomiale = un neurone softmax per classe) sui vettori CLIP
    dei ritagli: 9 elementi diagnostici + classe ALTRO (ritagli casuali senza elementi diagnostici).
    Regolarizzazione scelta con validazione incrociata raggruppata per monumento."""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    nomi, groups = np.asarray(nomi), np.asarray(groups)
    try:
        cv = list(GroupKFold(min(3, len(np.unique(groups)))).split(E, nomi, groups))
        lr = LogisticRegressionCV(Cs=list(Cs), cv=cv, class_weight="balanced", max_iter=3000,
                                  random_state=seed, n_jobs=-1)
        return make_pipeline(StandardScaler(), lr).fit(E, nomi)
    except ValueError:                                     # pochi esempi: regolarizzazione fissa
        lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000, random_state=seed)
        return make_pipeline(StandardScaler(), lr).fit(E, nomi)


SOGLIA_NON_FOTO = 0.9      # sopra questa probabilità l'immagine è quasi certamente un disegno o un'incisione
ZS_SPENTO = 1.01           # soglia zero-shot impossibile: CLIP zero-shot serve solo a scartare lo sfondo
# Nomi con precisione ≥ 0,75 sulle zone tipo LIME (notebook 05c, validazione incrociata per monumento):
# arco a sesto acuto 0,88 · pinnacolo gotico 0,89 · arco inflesso 0,82 · colonna tortile 0,98 · timpano spezzato 0,79.
# Esclusi: timpano triangolare 0,74 · arco trilobato 0,65 · timpano curvo 0,60 · serliana 0,60.
NOMI_AFFIDABILI = ("arco a sesto acuto", "pinnacolo gotico", "arco inflesso", "colonna tortile (salomonica)",
                   "timpano spezzato")


def relazione_stile(tipico, stile_pred: str) -> str | None:
    """Rapporto tra lo stile/famiglia tipico di un elemento e lo stile predetto:
    'stile' (coincide), 'famiglia' (stessa famiglia, stile diverso o elemento tipico di una famiglia),
    'altra famiglia', oppure None se l'elemento non è tipico di nulla."""
    if tipico is None:
        return None
    if tipico == stile_pred:
        return "stile"
    f_pred = famiglia_tipica(stile_pred)
    if tipico in config.FAMIGLIE:
        return "famiglia" if famiglia_tipica(tipico) == f_pred else "altra famiglia"
    return "famiglia" if famiglia_tipica(tipico) == f_pred else "altra famiglia"


class ElementNamer:
    """Nome dell'elemento architettonico in un ritaglio.

    1. Classificatore addestrato sui ritagli MonuMAI (9 elementi diagnostici + «altro»): se indica un elemento
       diagnostico con confidenza ≥ soglia_probe, quello è il nome.
    2. CLIP zero-shot sul vocabolario (frasi multiple per nome) serve a riconoscere le zone di sfondo (cielo,
       alberi, persone…), che vengono scartate. Con soglia_zs ≤ 1 dà anche i nomi degli altri elementi; con la
       soglia predefinita (ZS_SPENTO) non dà nomi: sui ritagli annotati non raggiunge l'80% di precisione.
    3. Astensione: sotto la soglia la zona resta un «dettaglio decisivo» senza nome.
    """

    def __init__(self, fm: FinalModel, probe=None, soglia_probe: float = 0.0, soglia_zs: float = ZS_SPENTO,
                 affidabili=None):
        self.fm, self.probe = fm, probe
        self.affidabili = None if affidabili is None else set(affidabili)    # None = tutti i nomi
        self.soglia_probe, self.soglia_zs = soglia_probe, soglia_zs
        self.T = fm.text_embed(V_FRASI, TEMPLATE)
        self.diag = np.array([t == "diagnostico" for t in V_TIPO])

    def zero_shot(self, E: np.ndarray) -> np.ndarray:
        z = 100.0 * E @ self.T.T
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    @staticmethod
    def _out(nome, conf, ok, via):
        j = V_NOMI.index(nome)
        return {"nome": nome if ok else None, "candidato": nome, "conf": float(conf),
                "tipico": V_TIPICO[j] if ok else None, "tipo": V_TIPO[j], "via": via}

    def decide(self, z: np.ndarray, pp: np.ndarray | None = None) -> dict:
        cand_diag = None
        if pp is not None:
            k = int(pp.argmax())
            nome = str(self.probe.classes_[k])
            if nome != ALTRO:
                if pp[k] >= self.soglia_probe:
                    ok = self.affidabili is None or nome in self.affidabili
                    return self._out(nome, pp[k], ok, "classificatore")
                cand_diag = (nome, pp[k])
            z = np.where(self.diag, 0.0, z)                # gli elementi diagnostici li giudica il classificatore
        j = int(z.argmax())
        if V_TIPO[j] == "sfondo":
            return {"nome": None, "candidato": V_NOMI[j], "conf": float(z[j]), "tipico": None, "tipo": "sfondo",
                    "via": "zero-shot"}
        if z[j] >= self.soglia_zs:
            return self._out(V_NOMI[j], z[j], True, "zero-shot")
        if cand_diag is not None:
            return self._out(cand_diag[0], cand_diag[1], False, "classificatore")
        return {"nome": None, "candidato": None, "conf": float(z[j]), "tipico": None, "tipo": "elemento",
                "via": "zero-shot"}

    def name(self, crops=None, E: np.ndarray | None = None) -> list[dict]:
        E = self.fm.image_embed(crops) if E is None else E
        Z = self.zero_shot(E)
        PP = self.probe.predict_proba(E) if self.probe is not None else [None] * len(E)
        return [self.decide(z, pp) for z, pp in zip(Z, PP)]

    def save(self, path) -> None:
        import joblib
        joblib.dump({"probe": self.probe, "soglia_probe": self.soglia_probe, "soglia_zs": self.soglia_zs,
                     "vocabolario": V_NOMI}, path)

    @classmethod
    def load(cls, fm: FinalModel, path, affidabili=None) -> "ElementNamer":
        """affidabili: nomi che si possono mostrare (None = tutti; l'app passa NOMI_AFFIDABILI)."""
        import joblib
        d = joblib.load(path)
        assert d["vocabolario"] == V_NOMI, "vocabolario cambiato: rieseguire il notebook 05c"
        return cls(fm, d["probe"], d["soglia_probe"], d.get("soglia_zs", ZS_SPENTO), affidabili)


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


def box_of_segment(seg: np.ndarray, s: int, min_side: int = 128, margin: float = 0.2):
    """Riquadro (x0, y0, x1, y1) attorno al super-pixel s, con margine e lato minimo min_side px."""
    H, W = seg.shape
    ys, xs = np.nonzero(seg == s)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    h = max(min_side, (ys.max() - ys.min()) * (1 + 2 * margin))
    w = max(min_side, (xs.max() - xs.min()) * (1 + 2 * margin))
    y0, x0 = int(max(0, cy - h / 2)), int(max(0, cx - w / 2))
    y1, x1 = int(min(H, y0 + h)), int(min(W, x0 + w))
    return x0, y0, x1, y1


def top_regions(seg: np.ndarray, weights: np.ndarray, k: int = 3, min_side: int = 128, margin: float = 0.2):
    """Riquadri dei k super-pixel con peso positivo più alto, allargati ad almeno min_side px."""
    boxes = []
    for s in np.argsort(-weights)[:k]:
        if weights[s] <= 0:
            break
        boxes.append((*box_of_segment(seg, int(s), min_side, margin), float(weights[s])))
    return boxes


def crop(img: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box[:4]
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def spiega(fm: FinalModel, img: np.ndarray, namer: ElementNamer | None = None, num_samples: int = 600,
           k: int = 3, seed: int = config.SEED) -> dict:
    """Spiegazione completa di una foto 448 × 448 (uint8), come nell'app.

    Restituisce: probabilità degli stili e delle famiglie, famiglia e stile (risposta gerarchica), stile rivale,
    risultato LIME (super-pixel, pesi per famiglia/stile/contrasto, R²) e fino a k zone a favore dello stile, ognuna
    con riquadro, peso LIME, ritaglio, nome dell'elemento (o None = «dettaglio decisivo») e rapporto con lo stile
    predetto. Le zone di sfondo (cielo, alberi, persone…) vengono saltate."""
    p = fm.predict([img])[0]
    fam, stile, pf = hi.predict(p[None])
    fam, stile, pf = int(fam[0]), int(stile[0]), pf[0]
    rivale = int(next(j for j in np.argsort(-p) if j != stile))
    cols = lambda X: output_columns(fm.predict(X), (stile, rivale))
    res = lime_explain(cols, img, [stile, col_family(fam), COL_CONTRASTO], num_samples=num_samples, seed=seed)
    zone = []
    cand = top_regions(res["segments"], res["pesi"][stile], k=2 * k)
    if cand:
        crops = [crop(img, b) for b in cand]
        nomi = namer.name(crops) if namer is not None else [None] * len(cand)
        for b, c, d in zip(cand, crops, nomi):
            if d is not None and d["tipo"] == "sfondo":
                continue
            nome = d["nome"] if d else None
            zone.append({"box": b[:4], "peso": b[4], "ritaglio": c, "nome": nome,
                         "conf": d["conf"] if d and nome else None,
                         "tipico": d["tipico"] if d and nome else None,
                         "relazione": relazione_stile(d["tipico"], config.CLASSES[stile]) if d and nome else None})
            if len(zone) == k:
                break
    return {"p": p, "p_famiglie": pf, "famiglia": fam, "stile": stile, "rivale": rivale, "lime": res,
            "r2_stile": res["r2"][stile], "zone": zone}


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
