# Scelte progettuali

Ogni scelta è valutata secondo tre criteri: **resa** (qualità del risultato),
**tempo** di realizzazione e **difficoltà** implementativa.

## D1 – Dataset: WikiChurches
L'*Architectural Heritage Elements Dataset* (Llamas et al., 2017) raccoglie immagini
di singoli elementi architettonici (cupole, colonne, gargoyle…), senza edifici interi
e senza etichette di stile: non è adatto alla classificazione degli stili.

È stato quindi adottato **WikiChurches** (Barz & Denzler, NeurIPS 2021): fotografie di
chiese da Wikimedia Commons, con lo stile di ciascun edificio ricavato da Wikidata,
licenza CC BY-SA 4.0.

- Resa: alta – foto reali e identificativo dell'edificio per ogni immagine.
- Tempo: medio – archivio da 12,4 GB, ridotto una sola volta a circa 150 MB.
- Difficoltà: media.

## D2 – Classi: Romanico, Gotico, Rinascimentale, Barocco
Sono i quattro stili più rappresentati nel dataset. La coppia Rinascimentale/Barocco,
stilisticamente vicina, rende significativa l'analisi della matrice di confusione.

## D3 – Split ufficiale WikiChurches-4
Conteggi preliminari (immagini per stile):

| Stile | Solo stile principale | Con sotto-stili |
|---|---|---|
| Romanico | 1.591 | 2.209 |
| Gotico | 1.581 | 3.649 |
| Rinascimentale | 187 | 217 |
| Barocco | 1.392 | 1.508 |

Si adotta lo split ufficiale *WikiChurches-4* fornito dagli autori (7.526 immagini:
5.838 training, 206 validazione, 1.482 test), che riconduce i sotto-stili (es. *Italian
Gothic*, *Rococo*) ai quattro stili principali.

- Resa: alta – risultati confrontabili con quelli pubblicati sullo stesso benchmark.
- Tempo: basso – etichette e divisione già pronte.
- Difficoltà: bassa.

Le chiese con stili appartenenti a classi diverse (es. romanico e gotico) non fanno parte
del benchmark; possono servire come casi di studio qualitativi per la lettura «fuzzy»
delle probabilità.

## D4 – Divisione per chiesa
Nello split ufficiale tutte le foto di uno stesso edificio appartengono a un solo insieme.
Uno split per foto porterebbe immagini dello stesso edificio sia in addestramento sia in
test, e la valutazione misurerebbe il riconoscimento del singolo monumento anziché dello
stile. La proprietà viene verificata automaticamente, insieme all'assenza di
foto identiche (stesso hash MD5) tra training e validazione/test.

## D4b – Classe minoritaria: Rinascimentale
Nello split ufficiale il training set contiene 64 immagini rinascimentali contro 3.041
gotiche (rapporto 1:47); validazione e test sono invece più bilanciati. Contromisure:

- **pesi di classe** nella cross-entropy. I pesi «bilanciati» n / (k · n_classe)
  darebbero al Rinascimentale peso 22,8: ogni sua immagine conterebbe quanto 47 immagini
  gotiche e, con soli 64 esempi, il modello tenderebbe a sovra-predire la classe.
  Si usa la **radice** dei pesi bilanciati (Romanico 0,94 · Gotico 0,69 ·
  Rinascimentale 4,78 · Barocco 1,16), che riduce lo squilibrio senza esasperarlo;
- **data augmentation** (ritaglio casuale, ribaltamento orizzontale, luminosità e
  contrasto), che moltiplica le varianti delle poche immagini disponibili;
- **macro-F1** come metrica principale accanto all'accuracy, che da sola premierebbe
  le classi più numerose;
- analisi della confusione Rinascimentale/Barocco nella matrice di confusione.

## D5 – Immagini ridimensionate a 256 px sul lato corto
Formato standard per le reti pre-addestrate su ImageNet, con ritaglio 224×224
in addestramento. Le immagini sono archiviate in un unico file `.tar` su Google
Drive, più rapido da trasferire di migliaia di file singoli.

## D6 – Perimetro del progetto
Per privilegiare un sistema completo e valutato in modo rigoroso sono stati esclusi:

- **ricerca del monumento tramite GPS/EXIF**: i metadati di posizione sono spesso assenti
  nelle foto condivise e indicano la posizione del fotografo, non del monumento;
- **ottimizzazione degli iperparametri con algoritmo genetico**: ogni valutazione
  richiederebbe un addestramento della VGG-16, con un costo non compatibile con le
  risorse disponibili; si usano valori di riferimento ed early stopping;
- **generazione del testo con un LLM**: sostituita da testi descrittivi per stile,
  che non rischiano di introdurre informazioni errate.

## D7 – Baseline e modello M1: dalla prima alla seconda iterazione
Test set ufficiale: 1.481 immagini (Romanico 503, Gotico 506, Rinascimentale 100, Barocco 372).

**Iterazione 1**

| Modello | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---|
| Classe maggioritaria (sempre Gotico) | 0,342 | 0,127 | 0,250 |
| CNN da zero (30 epoche max, pazienza 4) | 0,340 | 0,127 | 0,250 |
| M1: VGG-16 congelata + GAP → Dropout → Dense | 0,574 | 0,503 | 0,491 |

McNemar CNN vs M1: b = 99, c = 446, p ≈ 1·10⁻⁴⁹.

Diagnosi:
- **CNN**: loss di training ferma a ~0,99 e accuracy di validazione costante a 0,24; l'early
  stopping l'ha interrotta all'epoca 6 (migliore: 2) e il modello predice una sola classe
  (Romanico). Non è una baseline significativa ma un addestramento interrotto troppo presto:
  una rete addestrata da zero richiede più epoche, e la validazione (206 immagini) è rumorosa.
- **M1**: loss iniziale 4,48 (il valore atteso per 4 classi equiprobabili è ln 4 ≈ 1,39), segno
  di feature VGG con valori grandi e non normalizzati; accuracy di training ferma a 0,68
  (underfitting della testa lineare); forte sovra-predizione del Romanico (41% dei gotici e
  38% dei barocchi classificati come romanici).

**Correzioni (iterazione 2)**
- CNN: epoche 30 → 40, pazienza 4 → 8, momentum della BatchNorm 0,99 → 0,9 (statistiche usate
  in valutazione più rapide a stabilizzarsi).
- M1: testa GAP → **BatchNorm** → **Dense(256, ReLU)** → Dropout(0,5) → Dense softmax;
  epoche 20 → 30, pazienza 4 → 6. La VGG-16 resta congelata.

**Risultati dell'iterazione 2 (configurazione definitiva del notebook 02)**

| Modello | Accuracy | Macro-F1 | Balanced accuracy | F1 Rom. | F1 Got. | F1 Rin. | F1 Bar. |
|---|---|---|---|---|---|---|---|
| Classe maggioritaria | 0,342 | 0,127 | 0,250 | 0,000 | 0,509 | 0,000 | 0,000 |
| CNN da zero | 0,454 | 0,437 | 0,477 | 0,466 | 0,523 | 0,365 | 0,396 |
| M1 VGG-16 congelata | 0,569 | 0,511 | 0,511 | 0,631 | 0,566 | 0,333 | 0,513 |

McNemar CNN vs M1: b = 206, c = 376, p ≈ 2,5·10⁻¹². La CNN ora impara (25 epoche, migliore la 17);
M1 parte da una loss di 1,03 ma va in overfitting dopo 2 epoche (loss di training in calo, di
validazione in salita): le feature congelate di ImageNet sono il limite. Errore principale di M1:
il 47% delle immagini gotiche è classificato come romanico.

## D8 – Fine-tuning dell'ultimo blocco (M2)
- Si parte dai pesi di M1 e si sblocca solo `block5` (3 strati convoluzionali, ~7 milioni di pesi):
  gli strati iniziali rappresentano bordi e texture, generici e già adatti; quelli finali forme e
  parti, da specializzare su archi, finestre e facciate.
- Learning rate 1·10⁻⁵ (100 volte più basso di quello della testa), per non distruggere le
  feature pre-addestrate; massimo 20 epoche, pazienza 5.
- Confronto M1 vs M2 con il test di McNemar sullo stesso test set.

## D9 – Analisi degli errori con i metadati delle chiese
Oltre alla matrice di confusione, gli errori vengono analizzati con informazioni che il modello
non vede: sotto-stile Wikidata (per verificare se l'errore Gotico → Romanico è concentrato in
sotto-stili come il *Brick Gothic*), Paese e posizione geografica, confidenza della predizione.
