# MirAi – Guida Turistica Architettonica Automatica

Progetto d'esame per il corso **IARTS** (6 CFU). L'utente fotografa un monumento — una chiesa, un palazzo, un
castello, un teatro, un tempio — e l'app ne riconosce lo stile architettonico, mostrando **quali parti della foto
hanno deciso la risposta** (LIME) e, quando è affidabile, il **nome dell'elemento** ingrandito (arco a sesto acuto,
timpano spezzato…).

**6 stili in 3 famiglie:** Medievale (Romanico, Gotico) · Classicismo moderno (Rinascimentale, Barocco,
Neoclassico) · Classico antico.

## Pipeline del modello definitivo

```
foto ─► lato corto 512, ritaglio 448 × 448
          ├─► CLIP ViT-L/14 (congelato)  ── 768 valori ──┐
          └─► DINOv2 ViT-L/14 (congelato) ── 2.048 valori ┴─► testa MLP (2.816 → 512 → 6) ─► softmax a 6 stili
                                                                                              │
                     famiglia = somma degli stili della famiglia ◄─────────────────────────────┘
                     risposta gerarchica: prima la famiglia, poi lo stile al suo interno

spiegazione ─► LIME sul modello definitivo (famiglia, stile, contrasto con lo stile rivale)
                └─► zone decisive ingrandite ─► nome dell'elemento (classificatore sui vettori CLIP, con astensione)
```

La **VGG-16** a 6 stili (fine-tuning, 448 px) con **Grad-CAM** resta come termine di confronto nel paper.

## Scheda del modello definitivo

**Nome:** CLIP + DINOv2 congelati con testa MLP e risposta gerarchica (notebook `03f`, scelto fra 18
configurazioni e la VGG-16 sul validation; registro in `reports/metrics/modello_finale.json`).

| Parte | Scelta |
|---|---|
| Ingresso | foto RGB, orientamento EXIF corretto, lato corto 512 px e ritaglio centrale 448 × 448 (formato dell'app e di LIME); ogni rete la porta a 224 × 224 con la propria normalizzazione |
| Rete 1 – CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) | Vision Transformer addestrato su coppie immagine-testo; **congelato**; vettore = proiezione del token CLS, 768 valori |
| Rete 2 – DINOv2 ViT-L/14 (`facebook/dinov2-large`) | Vision Transformer auto-supervisionato; **congelato**; vettore = token CLS + media dei token delle patch, 2.048 valori |
| Unione | concatenazione (2.816 valori) e standardizzazione (media e deviazione del training) |
| Testa (unica parte addestrata) | MLP: Dense 2.816 → 512, ReLU, Dropout 0,5, Dense 512 → 6, softmax (≈ 1,45 M parametri) |
| Addestramento della testa | cross-entropy pesata per classe, Adam (lr 1e-3, weight decay 1e-4), batch 256, max 60 epoche, early stopping (pazienza 8) su un 10% del training separato **per edificio** |
| Dati di training | 10.888 immagini: WikiChurches (split ufficiale), monumenti Wikidata, ritagli degli elementi MonuMAI; ogni immagine anche ribaltata orizzontalmente |
| Validation / test | 1.082 + 1.214 / 1.481 + 1.214 (chiese + monumenti); in valutazione media tra foto originale e ribaltata |
| Uscita | probabilità dei 6 stili; famiglia = somma degli stili; risposta gerarchica (prima la famiglia, poi lo stile al suo interno) |
| Affidabilità | risposta «sicura» se confidenza stile ≥ 0,80 (88% corretta sul 56% delle foto) o famiglia ≥ 0,60 (91% sul 95%), soglie scelte sul validation |
| Controlli sulla foto | fuori dominio: distanza media dai 5 edifici di training più vicini (AUROC 0,96, soglia al 95° percentile del validation); disegno/incisione: CLIP zero-shot, probabilità ≥ 0,9 |
| Edifici simili | 5 vicini nello spazio della testa, solo edifici dello stile predetto |

**Spiegazione LIME (`xai.spiega`)**

| Passo | Scelta |
|---|---|
| Super-pixel | SLIC, circa 60 per foto (compattezza 20) |
| Perturbazioni | 600 versioni con super-pixel spenti a caso (sostituiti dal loro colore medio), predette dal modello definitivo |
| Località | kernel esponenziale sulla distanza del coseno dalla foto originale (larghezza 0,25) |
| Modello locale | regressione Ridge pesata: peso di ogni super-pixel; R² = fedeltà locale |
| Uscite spiegate | famiglia (perché medievale?), stile (perché gotico?), contrasto (perché gotico e non romanico?) |
| Zone decisive | super-pixel con peso positivo più alto per lo stile, riquadro con margine del 20% e lato ≥ 128 px; zone di sfondo scartate; le prime 3 ingrandite |
| Nome dell'elemento | regressione logistica sui vettori CLIP del ritaglio (9 elementi MonuMAI + «altro»), soglia 0,65; mostrati solo i 5 nomi con precisione ≥ 0,75, altrimenti «dettaglio decisivo» |

## Risultati

**Test ufficiale (2.695 foto mai usate per scelte o addestramento), modello definitivo contro VGG-16:**

| Test | Famiglia | Stile: accuracy | Stile: macro-F1 | VGG-16: famiglia / stile |
|---|---|---|---|---|
| Chiese (WikiChurches, 1.481) | **0,912** | **0,777** | **0,724** | 0,895 / 0,698 |
| Monumenti (Wikidata, 1.214) | **0,862** | **0,680** | **0,692** | 0,815 / 0,560 |

- Differenza significativa sullo stile (McNemar: chiese p ≈ 1·10⁻¹¹, monumenti p ≈ 2·10⁻¹⁶).
- **Risposte sicure** (soglia scelta sul validation): stile corretto all'88% sul 56% delle foto; famiglia corretta al
  91% sul 95% delle foto.
- Nessun overfitting evidente: validation e test quasi identici (macro-F1 stile 0,743 → 0,724 e 0,682 → 0,692).
- **Classe problematica:** chiese rinascimentali (F1 0,49; il 35% classificato come barocco) — confine stilistico
  graduale e etichette riferite all'edificio, non alla parte fotografata.

**Spiegabilità (LIME sul modello definitivo, 48 foto, 600 perturbazioni):**

| Verifica | Risultato |
|---|---|
| Fedeltà locale (R² della regressione LIME) | stile 0,71 · contrasto 0,71 · famiglia 0,63 |
| Deletion test: AUC LIME vs ordine casuale | 0,485 vs 0,686 – LIME migliore nel **100%** delle foto (Wilcoxon p = 7·10⁻¹⁵) |
| Probabilità dello stile cancellando il 20% della foto | 0,79 → **0,58** (zone LIME) contro 0,79 → 0,76 (zone casuali) |
| Grad-CAM sulla VGG-16 (confronto) | AUC 0,234 vs 0,404 casuale – migliore nel 96% (p = 9·10⁻¹¹) |
| Nomi degli elementi sulle zone tipo LIME | precisione **0,82**, falsi allarmi 0,07 (contro 0,46 e 0,40 prima della correzione del dominio) |

Altri risultati: mappa di Kohonen (le 3 famiglie formano regioni con purezza 0,88), filtro per foto fuori dominio
(AUROC 0,96), controllo dei dati (l'1% delle immagini Wikidata sono disegni o incisioni; escluderle non cambia le
metriche). Dettagli e motivazioni di ogni scelta in `docs/decisioni.md`.

## Fasi

| Fase | Contenuto | Notebook |
|---|---|---|
| 1. Dati | WikiChurches (split ufficiale per chiesa), monumenti da Wikidata, elementi MonuMAI, immagini a 512 px | 01, 01b, 01c |
| 2. Primi modelli | baseline, CNN da zero, VGG-16 congelata (M1) | 02 |
| 3. CNN | fine-tuning VGG-16 (M2, M3), ConvNeXt-Tiny, dati esterni, alta risoluzione, 6 stili | 03, 03b, 03c, 03d, 03e |
| 4. Modello definitivo | CLIP + DINOv2 congelati, testa MLP, risposta gerarchica, test finale | 03f |
| 5. Mappa di Kohonen | spazio degli stili, edifici simili, foto fuori dominio | 04, 04b |
| 6. Spiegabilità | LIME, Grad-CAM, deletion test | 05 |
| 6b. Nomi degli elementi | classificatore con astensione, tarato sulle zone LIME | 05c (05b: passaggio intermedio) |
| 7. App | Gradio: foto → famiglia e stile, zone LIME ingrandite, descrizione; schermata tecnica | in corso |

<details><summary>Benchmark a 4 stili sulle chiese (modelli CNN, fasi 2–3)</summary>

| Modello | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---|
| Classe maggioritaria | 0,342 | 0,127 | 0,250 |
| CNN da zero | 0,454 | 0,437 | 0,477 |
| M1 VGG-16 congelata | 0,569 | 0,511 | 0,511 |
| M2 VGG-16 fine-tuning blocco 5 | 0,653 | 0,585 | 0,573 |
| M3 VGG-16 fine-tuning blocchi 4-5 + TTA (224 px) | 0,677 | 0,588 | 0,577 |
| M3 a 448 px + TTA | 0,698 | 0,630 | 0,617 |
| ConvNeXt-Tiny | 0,631 | 0,536 | 0,535 |
| **Modello definitivo (6 stili, valutato sulle 4 classi)** | **0,777** | **0,724** | **0,712** |

</details>

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
│   ├── xai.py            LIME (spiega: pipeline dell'app), deletion test, nomi degli elementi, Grad-CAM
│   ├── colab_git.py      commit e push da Colab senza esporre il token
│   └── data/
│       ├── wikichurches.py   download, indice, ridimensionamento, split, report
│       ├── dataset.py        caricamento tf.data, data augmentation, pesi di classe
│       ├── wikidata.py       monumenti da Wikidata/Commons: interrogazione, download, pulizia
│       └── monumai.py        MonuMAI: download, duplicati (hash percettivo), annotazioni, zone tipo LIME
├── data/splits/          splits.csv (solo percorsi ed etichette, niente immagini)
├── reports/              metriche (JSON/CSV), predizioni sul test, figure
└── docs/decisioni.md     registro delle scelte progettuali con la motivazione
```

Le immagini **non** stanno su GitHub: sono su Google Drive
(`MyDrive/mirai_data/`: archivi delle immagini, caratteristiche estratte, testa del modello definitivo e
classificatore degli elementi).

## Come riprodurre

1. Nei *Secrets* di Colab crea `GITHUB_TOKEN` (token con accesso a questo repository).
2. Dati: `01_dataset` (CPU) → `01b_alta_risoluzione` (CPU) → `01c_monumenti_wikidata` (GPU).
3. Modelli CNN: `02_baseline_m1` → `03_finetuning_errori` → `03b_miglioramenti` → `03c_dati_esterni` →
   `03d_alta_risoluzione` → `03e_sei_stili` (GPU).
4. Modello definitivo: `03f_transformer_gerarchico` (GPU).
5. Mappa di Kohonen: `04_som`, `04b_som_modello_finale` (GPU T4, quasi tutto CPU).
6. Spiegabilità: `05_spiegabilita` (LIME, Grad-CAM, deletion test), `05c_nomi_elementi` (nomi degli elementi).
   `05b_spiegabilita_nomi` documenta il passaggio intermedio e non è necessario.

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
