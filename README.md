# MirAi – Guida Turistica Architettonica Automatica

Progetto d'esame per il corso **IARTS** (6 CFU). L'utente fotografa una chiesa o un
monumento e l'app ne riconosce lo stile architettonico — **Romanico, Gotico,
Rinascimentale o Barocco** — mostrando *dove* la rete ha guardato per decidere
(Grad-CAM e LIME).

## Pipeline

```
Foto ──► VGG-16 pre-addestrata (fine-tuning blocco 5) ──► softmax a 4 stili
                     │
                     └──► Grad-CAM / LIME ──► zone decisive (portali, rosoni, archi…)
```

| Fase | Contenuto | Stato |
|---|---|---|
| 1. Dataset | WikiChurches-4, split ufficiale per chiesa (7.526 immagini) | ✅ |
| 2. Baseline | classe maggioritaria + piccola CNN da zero | ⏳ |
| 3. Modello M1 | VGG-16 congelata + testa GAP → Dense softmax | ⏳ |
| 4. Modello M2 | fine-tuning del blocco 5 | ☐ |
| 5. Valutazione | accuracy, P/R/F1 per classe, matrice di confusione, McNemar | ☐ |
| 6. Explainable AI | Grad-CAM, LIME, deletion test | ☐ |
| 7. App | Gradio: stile + heatmap + descrizione di stile, elementi e monumento | ☐ |

## Struttura della repository

```
MirAi/
├── notebooks/            notebook Colab, uno per fase (01_dataset, 02_baseline, …)
├── src/
│   ├── config.py         TUTTI i parametri (classi, seed, split, addestramento)
│   ├── models.py         CNN baseline e VGG-16 con nuova testa (M1)
│   ├── train.py          addestramento: cross-entropy pesata, Adam, early stopping
│   ├── evaluation.py     metriche, matrici di confusione, curve, test di McNemar
│   ├── colab_git.py      commit e push da Colab senza esporre il token
│   └── data/
│       ├── wikichurches.py   download, indice, ridimensionamento, split, report
│       └── dataset.py        caricamento tf.data, data augmentation, pesi di classe
├── data/splits/          splits.csv (solo percorsi ed etichette, niente immagini)
├── reports/              metriche (JSON/CSV), predizioni sul test, figure
└── docs/decisioni.md     registro delle scelte progettuali con la motivazione
```

Le immagini **non** stanno su GitHub: sono su Google Drive
(`MyDrive/mirai_data/wikichurches4_256.tar`).

## Come riprodurre

1. Nei *Secrets* di Colab crea `GITHUB_TOKEN` (token con accesso a questo repository).
2. `notebooks/01_dataset.ipynb` (CPU): prepara il dataset e lo salva su Google Drive.
3. `notebooks/02_baseline_m1.ipynb` (GPU): baseline, modello M1 e confronto.

Ogni notebook si esegue dall'alto in basso.

Tutti i risultati usano `SEED = 42` (in `src/config.py`).

## Dataset e licenze

- **WikiChurches** – B. Barz, J. Denzler, *WikiChurches: A Fine-Grained Dataset of
  Architectural Styles with Real-World Challenges*, NeurIPS 2021 Datasets and Benchmarks.
  Dati: <https://doi.org/10.5281/zenodo.5166986> – licenza **CC BY-SA 4.0**.
  Le immagini provengono da Wikimedia Commons; autore e licenza di ogni immagine sono
  nelle colonne `author` e `license` di `data/splits/splits.csv`.
- Il codice di questa repository è distribuito a scopo didattico.
