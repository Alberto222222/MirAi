"""Logica dell'app MirAi, separata dall'interfaccia (si può provare da un notebook senza Gradio).

Flusso di una foto:
1. `prepara`: orientamento EXIF, lato corto 512 px, ritaglio centrale 448 × 448;
2. `analizza` (≈ 1 s): caratteristiche CLIP + DINOv2 (media foto originale + ribaltata, come nel test), testa MLP,
   risposta gerarchica, affidabilità, controlli (fuori dominio, disegno), edifici simili, posizione sulla SOM;
3. `spiega` (≈ 20 s): LIME sul modello definitivo e nomi degli elementi (xai.spiega);
4. `salva_riscontro`: foto, predizione e correzione dell'utente su Drive, per un futuro riaddestramento verificato.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src import config, hierarchy as hi, xai


def prepara(foto, size: int = config.IMG_SIZE_HR, short: int = config.IMG_SHORT_SIDE_HR) -> np.ndarray:
    """PIL o numpy → numpy uint8 448 × 448 (stessa geometria di xai.load_square)."""
    im = foto if isinstance(foto, Image.Image) else Image.fromarray(np.asarray(foto).astype(np.uint8))
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = short / min(im.size)
    im = im.resize((max(size, round(im.width * s)), max(size, round(im.height * s))), Image.LANCZOS)
    l, t = (im.width - size) // 2, (im.height - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)))


def _norm(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


class MiraiApp:
    """fm: xai.FinalModel; namer: xai.ElementNamer (o None);
    db: dict con F (caratteristiche di training + validation del notebook 03f), path, label;
    is_train: maschera delle righe di training (riferimento per il filtro fuori dominio);
    som: contenuto di som_finale.npz (o None); nomi: {path: nome dell'edificio}; info: modello_finale.json."""

    def __init__(self, fm, namer, db: dict, is_train: np.ndarray, som: dict | None, nomi: dict, info: dict,
                 img_root: str = config.COLAB_PROCESSED_DIR_HR, dir_riscontri: str | None = None, k: int = config.KNN_K):
        import torch
        self.fm, self.namer, self.som, self.nomi, self.k = fm, namer, som, nomi, k
        self.img_root = img_root
        self.soglia_stile = float(info.get("soglia_stile", 0.80))
        self.soglia_fam = float(info.get("soglia_famiglia", 0.60))
        F = np.asarray(db["F"], dtype=np.float32)
        self.path = np.asarray(db["path"]).astype(str)
        self.label = np.asarray(db["label"]).astype(int)
        self.foto_intere = ~np.char.startswith(self.path, "elementi/")      # niente ritagli tra gli edifici simili
        # spazio della testa (512 attivazioni dello strato nascosto): edifici simili
        self._torch = torch
        self.H = _norm(self._testa(F))
        # filtro fuori dominio (notebook 04b): distanza media dai 5 vicini di training, caratteristiche standardizzate
        if som is not None:
            self.mu, self.sd = np.asarray(som["mu"]), np.asarray(som["sd"])
        else:
            self.mu, self.sd = F[is_train].mean(0), F[is_train].std(0) + 1e-6
        self.D_tr = _norm((F[is_train] - self.mu) / self.sd).astype(np.float32)
        va = _norm((F[~is_train] - self.mu) / self.sd).astype(np.float32)
        self.soglia_ood = float(np.quantile(self._dist_vicini(va), 0.95)) if len(va) else 1.0
        self.dir_riscontri = Path(dir_riscontri) if dir_riscontri else None

    # ------------------------------------------------------------------
    def _testa(self, F: np.ndarray, batch: int = 4096) -> np.ndarray:
        torch, head = self._torch, self.fm.head
        out = []
        with torch.no_grad():
            for i in range(0, len(F), batch):
                x = torch.tensor(self.fm.std(F[i:i + batch]).astype(np.float32))
                out.append(torch.relu(head[0](x)).numpy() if hasattr(head, "__getitem__") else x.numpy())
        return np.concatenate(out)

    def _dist_vicini(self, Q: np.ndarray, batch: int = 512) -> np.ndarray:
        d = []
        for i in range(0, len(Q), batch):
            S = Q[i:i + batch] @ self.D_tr.T
            d.append(1 - np.sort(S, axis=1)[:, -self.k:].mean(1))
        return np.concatenate(d)

    def caratteristiche(self, img: np.ndarray) -> np.ndarray:
        """Media delle caratteristiche di foto originale e ribaltata (come validation e test del 03f)."""
        F = self.fm.features([img, np.ascontiguousarray(img[:, ::-1])])
        return F.mean(0, keepdims=True).astype(np.float32)

    # ------------------------------------------------------------------
    def analizza(self, foto) -> dict:
        img = prepara(foto)
        F = self.caratteristiche(img)
        p = self.fm.fd.predict_proba(self.fm.head, self.fm.std(F).astype(np.float32))[0]
        fam, stile, pf = hi.predict(p[None])
        fam, stile, pf = int(fam[0]), int(stile[0]), pf[0]
        ordine = [int(j) for j in np.argsort(-p)]
        rivale = next(j for j in ordine if j != stile)
        # controlli
        d_ood = float(self._dist_vicini(_norm((F - self.mu) / self.sd).astype(np.float32))[0])
        p_disegno = float(xai.prob_non_foto(self.fm, self.fm.image_embed([img]))[0])
        # edifici simili: stesso stile predetto, spazio della testa
        h = _norm(self._testa(F))[0]
        cand = np.flatnonzero((self.label == stile) & self.foto_intere)
        top = cand[np.argsort(-(self.H[cand] @ h))[:self.k]]
        simili = [(f"{self.img_root}/{self.path[j]}", self.nomi.get(self.path[j], "")) for j in top]
        # mappa di Kohonen
        bmu = None
        if self.som is not None:
            z = (((F - self.som["mu"]) / self.som["sd"]) - self.som["centro"]) @ self.som["pca"].T
            bmu = int(((np.asarray(self.som["W"]) - z) ** 2).sum(1).argmin())
        return {"img": img, "F": F, "p": p, "pf": pf, "famiglia": fam, "stile": stile, "rivale": rivale,
                "conf_stile": float(p.max()), "conf_famiglia": float(pf.max()),
                "sicura_stile": bool(p.max() >= self.soglia_stile), "sicura_famiglia": bool(pf.max() >= self.soglia_fam),
                "d_ood": d_ood, "fuori_dominio": bool(d_ood > self.soglia_ood),
                "p_disegno": p_disegno, "disegno": bool(p_disegno >= xai.SOGLIA_NON_FOTO),
                "simili": simili, "bmu": bmu}

    def spiega(self, R: dict, num_samples: int = 600, progress=None) -> dict:
        return xai.spiega(self.fm, R["img"], self.namer, num_samples=num_samples, p=R["p"], progress=progress)

    # ------------------------------------------------------------------
    def salva_riscontro(self, R: dict, stile_corretto: str, commento: str, lingua: str) -> Path | None:
        if self.dir_riscontri is None:
            return None
        d = self.dir_riscontri
        (d / "foto").mkdir(parents=True, exist_ok=True)
        ora = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        nome = f"{ora}_{np.random.randint(1e6):06d}.jpg"
        Image.fromarray(R["img"]).save(d / "foto" / nome, quality=90)
        f = d / "riscontri.csv"
        nuovo = not f.exists()
        with open(f, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if nuovo:
                w.writerow(["data", "foto", "lingua", "famiglia_predetta", "stile_predetto", "confidenza_stile",
                            "stile_corretto", "commento", "fuori_dominio", "disegno", "verificato"])
            w.writerow([ora, nome, lingua, list(config.FAMIGLIE)[R["famiglia"]], config.CLASSES[R["stile"]],
                        round(R["conf_stile"], 3), stile_corretto, commento.replace("\n", " ").strip(),
                        R["fuori_dominio"], R["disegno"], ""])
        return f
