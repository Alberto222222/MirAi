"""Reti moderne pre-addestrate (Transformer) usate come estrattori di caratteristiche congelati, e testa
addestrata sopra (notebook 03f).

- CLIP ViT-L/14 (OpenAI, 2021): Vision Transformer addestrato su 400 milioni di coppie immagine-testo;
  caratteristica = embedding dell'immagine (768 valori).
- DINOv2 ViT-L/14 (Meta, 2023): Vision Transformer auto-supervisionato su 142 milioni di immagini;
  caratteristica = token CLS concatenato alla media dei token delle patch (2 × 1024 valori).

La rete resta congelata: ogni immagine si legge una sola volta e si addestra solo la testa (pochi parametri,
basso rischio di overfitting).

Teste:
- lineare: un solo strato softmax (regressione logistica multinomiale) con cross-entropy pesata per classe e
  regolarizzazione L2, ottimizzato con L-BFGS;
- MLP: Dense(512, ReLU) → Dropout → Dense softmax, Adam, early stopping su un 10% del training separato
  per edificio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Condizioni di luce simulate (stessi valori di dataset.LIGHT_VIEWS)
# ---------------------------------------------------------------------------
def apply_light(img: Image.Image, view: str | None) -> Image.Image:
    if not view:
        return img
    a = np.asarray(img).astype(np.float32)
    if view.startswith("sera"):
        a = a * 0.55
    elif view.startswith("sole"):
        a = 255.0 * (a / 255.0) ** 0.6
    elif view.startswith("tramonto"):
        a = a * np.array([1.18, 1.0, 0.78], dtype=np.float32)
    elif view.startswith("cielo"):
        a = 0.5 * a + 0.5 * a.mean()
    else:
        raise ValueError(view)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Estrazione delle caratteristiche
# ---------------------------------------------------------------------------
def load_backbone(name: str):
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPImageProcessor
    hub = config.BACKBONES[name]
    if name == "clip":
        model, proc = CLIPModel.from_pretrained(hub, torch_dtype=torch.float16), CLIPImageProcessor.from_pretrained(hub)
    else:
        model, proc = AutoModel.from_pretrained(hub, torch_dtype=torch.float16), AutoImageProcessor.from_pretrained(hub)
    return model.to(DEVICE).eval(), proc


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths, proc, flip=False, view=None):
        self.paths, self.proc, self.flip, self.view = list(paths), proc, flip, view

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.flip:
            img = ImageOps.mirror(img)
        img = apply_light(img, self.view)
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def extract(name: str, model, proc, paths, flip: bool = False, view: str | None = None,
            batch: int = 64, workers: int = 4) -> np.ndarray:
    dl = torch.utils.data.DataLoader(_Images(paths, proc, flip, view), batch_size=batch,
                                     num_workers=workers, pin_memory=True)
    out = []
    for x in dl:
        x = x.to(DEVICE, dtype=torch.float16)
        if name == "clip":
            # embedding dell'immagine = proiezione del token CLS dell'encoder visivo (indipendente dalla versione
            # di transformers, in cui get_image_features può restituire un oggetto invece di un tensore)
            f = model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
        else:
            h = model(pixel_values=x).last_hidden_state
            f = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Teste
# ---------------------------------------------------------------------------
class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6

    def __call__(self, X):
        return (X - self.mu) / self.sd


def _cw(y: np.ndarray, k: int) -> torch.Tensor:
    n = np.bincount(y, minlength=k).astype(np.float32)
    w = n.sum() / (k * np.maximum(n, 1))
    return torch.tensor(w, device=DEVICE)


def fit_linear(X: np.ndarray, y: np.ndarray, l2: float, k: int, iters: int = 300) -> torch.nn.Linear:
    torch.manual_seed(config.SEED)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    lin = torch.nn.Linear(X.shape[1], k).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=_cw(y, k))
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn(lin(Xt), yt) + l2 * (lin.weight ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lin.eval()


def fit_mlp(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int, p=config.FOUND_MLP):
    """MLP con early stopping su un 10% del training separato per edificio (mai il validation)."""
    from sklearn.model_selection import GroupShuffleSplit
    tr, ho = next(GroupShuffleSplit(1, test_size=0.1, random_state=config.SEED).split(X, y, groups))
    torch.manual_seed(config.SEED)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], p["hidden"]), torch.nn.ReLU(),
                              torch.nn.Dropout(p["dropout"]), torch.nn.Linear(p["hidden"], k)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=p["l2"])
    loss_fn = torch.nn.CrossEntropyLoss(weight=_cw(y[tr], k))
    f32, i64 = dict(dtype=torch.float32, device=DEVICE), dict(dtype=torch.long, device=DEVICE)
    Xtr, ytr = torch.tensor(X[tr], **f32), torch.tensor(y[tr], **i64)
    Xho, yho = torch.tensor(X[ho], **f32), torch.tensor(y[ho], **i64)
    best, best_state, wait, storia = 1e9, None, 0, []
    g = torch.Generator(device="cpu").manual_seed(config.SEED)
    for ep in range(p["epochs"]):
        net.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(256):
            idx = idx.to(DEVICE)
            opt.zero_grad()
            loss_fn(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lt = float(loss_fn(net(Xtr), ytr))
            lh = float(loss_fn(net(Xho), yho))
        storia.append((lt, lh))
        if lh < best - 1e-4:
            best, wait = lh, 0
            best_state = {k_: v.clone() for k_, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= p["patience"]:
                break
    net.load_state_dict(best_state)
    return net.eval(), storia


@torch.no_grad()
def predict_proba(head, X: np.ndarray) -> np.ndarray:
    return torch.softmax(head(torch.tensor(X, dtype=torch.float32, device=DEVICE)), dim=1).cpu().numpy()


def save_head(path, head, std: Standardizer, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "mu": std.mu, "sd": std.sd, "meta": meta}, path)
