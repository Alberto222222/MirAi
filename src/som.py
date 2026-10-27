"""Mappa auto-organizzante di Kohonen (SOM), implementata in NumPy.

Algoritmo (versione online, una presentazione alla volta):
1. pesi dei neuroni inizializzati (qui: lungo le prime due componenti principali dei dati);
2. si sceglie a caso un vettore di ingresso x;
3. si trova il neurone vincitore (BMU, Best Matching Unit): quello con i pesi più vicini a x;
4. il raggio del vicinato sigma(t) e il tasso di apprendimento alpha(t) decrescono nel tempo:
       sigma(t) = sigma0 * exp(-t / lambda),  alpha(t) = alpha0 * exp(-t / lambda),
       lambda = n_iter / ln(sigma0);
5. ogni neurone j si avvicina a x in misura dipendente dalla distanza sulla griglia dal vincitore:
       w_j <- w_j + alpha(t) * theta_j(t) * (x - w_j),  theta_j = exp(-d_j^2 / (2 sigma(t)^2));
6. si ripete dal punto 2 per n_iter iterazioni.

L'apprendimento è non supervisionato: le etichette di stile servono solo dopo, per colorare
la mappa e interpretarla.
"""
from __future__ import annotations

import numpy as np


class SOM:
    def __init__(self, rows: int, cols: int, dim: int, seed: int = 42):
        self.rows, self.cols, self.dim = rows, cols, dim
        self.rng = np.random.default_rng(seed)
        r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self.grid = np.stack([r.ravel(), c.ravel()], axis=1).astype(float)   # (n_neuroni, 2)
        self.W = np.zeros((rows * cols, dim))
        self.history: list[tuple[int, float]] = []

    # ------------------------------------------------------------------
    def init_pca(self, X: np.ndarray) -> None:
        """Pesi iniziali disposti sul piano delle prime due componenti principali:
        la mappa parte già «distesa» sui dati e converge più in fretta."""
        mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
        scale = s[:2] / np.sqrt(len(X))
        a = np.linspace(-1, 1, self.rows)[self.grid[:, 0].astype(int)]
        b = np.linspace(-1, 1, self.cols)[self.grid[:, 1].astype(int)]
        self.W = mean + 2 * (a[:, None] * scale[0] * vt[0] + b[:, None] * scale[1] * vt[1])

    def _dist2(self, X: np.ndarray) -> np.ndarray:
        """Distanze euclidee al quadrato tra ogni vettore e ogni neurone, forma (n, n_neuroni)."""
        return (X ** 2).sum(1)[:, None] - 2 * X @ self.W.T + (self.W ** 2).sum(1)[None, :]

    def bmu(self, X: np.ndarray, batch: int = 2048) -> np.ndarray:
        """Indice del neurone vincitore per ogni vettore."""
        return np.concatenate([self._dist2(X[i:i + batch]).argmin(1) for i in range(0, len(X), batch)])

    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, n_iter: int, alpha0: float = 0.5, sigma0: float | None = None,
              log_every: int = 2000, X_monitor: np.ndarray | None = None) -> "SOM":
        sigma0 = sigma0 or max(self.rows, self.cols) / 2
        lam = n_iter / np.log(sigma0)
        X_monitor = X if X_monitor is None else X_monitor
        for t in range(n_iter):
            x = X[self.rng.integers(len(X))]
            b = np.argmin(((self.W - x) ** 2).sum(1))
            sigma = sigma0 * np.exp(-t / lam)
            alpha = alpha0 * np.exp(-t / lam)
            d2 = ((self.grid - self.grid[b]) ** 2).sum(1)
            near = d2 <= (3 * sigma) ** 2          # oltre 3 sigma l'aggiornamento è trascurabile
            theta = np.exp(-d2[near] / (2 * sigma ** 2))
            self.W[near] += (alpha * theta)[:, None] * (x - self.W[near])
            if log_every and (t % log_every == 0 or t == n_iter - 1):
                self.history.append((t, self.quantization_error(X_monitor)))
        return self

    # ------------------------------------------------------------------
    def quantization_error(self, X: np.ndarray) -> float:
        """Distanza media tra ogni vettore e il suo neurone vincitore (qualità della quantizzazione)."""
        return float(np.mean([np.sqrt(np.maximum(self._dist2(X[i:i + 2048]).min(1), 0)).mean()
                              for i in range(0, len(X), 2048)]))

    def topographic_error(self, X: np.ndarray) -> float:
        """Frazione di vettori i cui due neuroni più vicini NON sono adiacenti sulla griglia
        (misura quanto la mappa conserva la topologia dei dati)."""
        errs = []
        for i in range(0, len(X), 2048):
            two = np.argsort(self._dist2(X[i:i + 2048]), axis=1)[:, :2]
            d = np.abs(self.grid[two[:, 0]] - self.grid[two[:, 1]]).max(1)
            errs.append(d > 1)
        return float(np.concatenate(errs).mean())

    def umatrix(self) -> np.ndarray:
        """Distanza media di ogni neurone dai vicini sulla griglia: valori alti = confini tra gruppi."""
        W = self.W.reshape(self.rows, self.cols, self.dim)
        U = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                d = [np.linalg.norm(W[r, c] - W[rr, cc])
                     for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                     if 0 <= rr < self.rows and 0 <= cc < self.cols]
                U[r, c] = np.mean(d)
        return U

    # ------------------------------------------------------------------
    def hits(self, X: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
        """Conteggio dei vettori per neurone e per classe, forma (rows, cols, n_classes)."""
        H = np.zeros((self.rows * self.cols, n_classes), dtype=int)
        np.add.at(H, (self.bmu(X), labels), 1)
        return H.reshape(self.rows, self.cols, n_classes)

    def label_neurons(self, X: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
        """Etichetta di ogni neurone = classe più frequente tra i vettori che vince.
        I neuroni senza vettori prendono l'etichetta del neurone etichettato più vicino (nei pesi)."""
        H = self.hits(X, labels, n_classes).reshape(-1, n_classes)
        lab = H.argmax(1)
        empty = H.sum(1) == 0
        if empty.any() and (~empty).any():
            Wl, idx = self.W[~empty], np.flatnonzero(~empty)
            for j in np.flatnonzero(empty):
                lab[j] = lab[idx[np.argmin(((Wl - self.W[j]) ** 2).sum(1))]]
        return lab

    def save(self, path, **extra) -> None:
        np.savez_compressed(path, W=self.W, rows=self.rows, cols=self.cols, **extra)
