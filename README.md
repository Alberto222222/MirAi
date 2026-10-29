# MirAi – Guida Turistica Architettonica Automatica

Progetto d'esame per il corso **IARTS** (6 CFU). L'utente fotografa una chiesa o un
monumento e l'app ne riconosce lo stile architettonico — **Romanico, Gotico,
Rinascimentale o Barocco** — mostrando *dove* la rete ha guardato per decidere
(Grad-CAM e LIME).

## Pipeline

```
Foto 448×448 ──► VGG-16 pre-addestrata (fine-tuning blocchi 4-5) ──► softmax a 4 stili
                     │
                     └──► Grad-CAM / LIME ──► zone decisive (portali, rosoni, archi…)
```

| Fase | Contenuto | Stato |
|---|---|---|
| 1. Dataset | WikiChurches-4, split ufficiale per chiesa (7.526 immagini) | ✅ |
| 2. Baseline | classe maggioritaria + piccola CNN da zero | ✅ |
| 3. Modello M1 | VGG-16 congelata + testa GAP → BatchNorm → Dense(256) → softmax | ✅ |
| 4. Modello M2 | fine-tuning del blocco 5 + analisi degli errori | ✅ |
| 4b. Miglioramenti | M3 (VGG-16 blocchi 4-5), ConvNeXt-Tiny, scelta del modello finale | ✅ |
| 4c. Dati esterni | MonuMAI nel training per il Rinascimentale: esperimento respinto sul validation | ✅ |
| 4d. Mappa di Kohonen | SOM sulle 512 caratteristiche del modello finale, chiese simili | ✅ |
| 4e. Alta risoluzione | M3 con immagini 448 × 448: nuovo modello finale | ✅ |
| 4f. Sei stili | monumenti di ogni tipo da Wikidata (+ Classico antico, Neoclassico), elementi MonuMAI | ✅ |
| 4g. Transformer e gerarchia | CLIP / DINOv2 congelati, risposta famiglia → stile, confronto con VGG-16 | ⏳ |
| 5. Valutazione | accuracy, P/R/F1 per classe, matrice di confusione, McNemar | ☐ |
| 6. Explainable AI | Grad-CAM, LIME, deletion test | ☐ |
| 7. App | Gradio: stile + heatmap + descrizione di stile, elementi e monumento | ☐ |

## Risultati sul test ufficiale (1.481 immagini)

| Modello | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---|
| Classe maggioritaria | 0,342 | 0,127 | 0,250 |
| CNN da zero | 0,454 | 0,437 | 0,477 |
| M1 VGG-16 congelata | 0,569 | 0,511 | 0,511 |
| M2 VGG-16 fine-tuning blocco 5 | 0,653 | 0,585 | 0,573 |
| M3 VGG-16 fine-tuning blocchi 4-5 + TTA (224 px) | 0,677 | 0,588 | 0,577 |
| **M3 a 448 px + TTA (finale)** | **0,698** | **0,630** | **0,617** |
| ConvNeXt-Tiny | 0,631 | 0,536 | 0,535 |

Dettagli, matrici di confusione e test statistici in `docs/decisioni.md` e `reports/`.

## Struttura della repository

```
MirAi/
├── notebooks/            notebook Colab, uno per fase (01_dataset, 02_baseline, …)
├── src/
│   ├── config.py         TUTTI i parametri (classi, seed, split, addestramento)
│   ├── models.py         CNN baseline e VGG-16 con nuova testa (M1)
│   ├── train.py          addestramento: cross-entropy pesata, Adam, early stopping
│   ├── evaluation.py     metriche, matrici di confusione, curve, test di McNemar
│   ├── analysis.py       analisi degli errori per sotto-stile, Paese e posizione
│   ├── som.py            mappa auto-organizzante di Kohonen (NumPy)
│   ├── foundation.py     CLIP / DINOv2 congelati: estrazione caratteristiche e teste
│   ├── hierarchy.py      risposta gerarchica famiglia → stile e relative metriche
│   ├── colab_git.py      commit e push da Colab senza esporre il token
│   └── data/
│       ├── wikichurches.py   download, indice, ridimensionamento, split, report
│       ├── dataset.py        caricamento tf.data, data augmentation, pesi di classe
│       ├── wikidata.py       monumenti da Wikidata/Commons: interrogazione, download, pulizia
│                             (+ monumai.py: ritagli degli elementi architettonici)
│       └── monumai.py        dataset esterno MonuMAI: download, duplicati (hash percettivo), annotazioni
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
4. `notebooks/03_finetuning_errori.ipynb` (GPU): fine-tuning (M2) e analisi degli errori.
5. `notebooks/03b_miglioramenti.ipynb` (GPU): M3, ConvNeXt-Tiny e scelta del modello finale.
6. `notebooks/03c_dati_esterni.ipynb` (GPU): M3 con immagini MonuMAI aggiunte al training.
7. `notebooks/01b_alta_risoluzione.ipynb` (CPU): immagini a 512 px sul lato corto.
8. `notebooks/03d_alta_risoluzione.ipynb` (GPU): M3 a 448 × 448 e decisione sul modello finale.
9. `notebooks/01c_monumenti_wikidata.ipynb` (GPU): dataset di monumenti a 6 stili da Wikidata, con pulizia.
10. `notebooks/03e_sei_stili.ipynb` (GPU): modello a 6 stili e decisione.
11. `notebooks/03f_transformer_gerarchico.ipynb` (GPU): CLIP e DINOv2 congelati, risposta gerarchica.
12. `notebooks/04_som.ipynb` (GPU): mappa di Kohonen dello spazio degli stili e chiese simili
   (usa la risoluzione indicata in `reports/metrics/modello_finale.json`).

Ogni notebook si esegue dall'alto in basso.

Tutti i risultati usano `SEED = 42` (in `src/config.py`).

## Dataset e licenze

- **WikiChurches** – B. Barz, J. Denzler, *WikiChurches: A Fine-Grained Dataset of
  Architectural Styles with Real-World Challenges*, NeurIPS 2021 Datasets and Benchmarks.
  Dati: <https://doi.org/10.5281/zenodo.5166986> – licenza **CC BY-SA 4.0**.
  Le immagini provengono da Wikimedia Commons; autore e licenza di ogni immagine sono
  nelle colonne `author` e `license` di `data/splits/splits.csv`.
- **MonuMAI** – A. Lamas et al., *MonuMAI: Dataset, deep learning pipeline and citizen science
  based app for monumental heritage taxonomy and classification*, Neurocomputing 420 (2021).
  Dati: <https://github.com/ari-dasci/OD-MonuMAI> – licenza **CC BY 4.0**. Usato solo per il
  training (stili Gotico, Rinascimentale, Barocco); le immagini non sono incluse in questa repository.
- **Wikidata / Wikimedia Commons** – dati Wikidata CC0; immagini con la licenza indicata su Commons
  (colonna `source_url` di `data/splits/splits6_wikidata.csv`). Le immagini non sono incluse in questa repository.
- Il codice di questa repository è distribuito a scopo didattico.
