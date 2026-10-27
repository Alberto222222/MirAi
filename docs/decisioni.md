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

**Risultati del notebook 03 (test ufficiale, 1.481 immagini)**

| Modello | Accuracy | Macro-F1 | Balanced acc. | F1 Rom. | F1 Got. | F1 Rin. | F1 Bar. |
|---|---|---|---|---|---|---|---|
| M1 VGG-16 congelata | 0,569 | 0,511 | 0,511 | 0,631 | 0,566 | 0,333 | 0,513 |
| M2 VGG-16 blocco 5 | 0,653 | 0,585 | 0,573 | 0,662 | 0,670 | 0,327 | 0,681 |

McNemar M1 vs M2: b = 96, c = 220, p ≈ 4,5·10⁻¹². L'errore Gotico → Romanico scende dal 47% al
27%; il Barocco riconosciuto sale dal 38% al 69%. Il Rinascimentale resta debole: il 56% delle sue
immagini è classificato come Barocco.

Analisi per sotto-stile (M2, immagini gotiche classificate come romaniche): «Gothic architecture»
38,8% (224 immagini), Brick Gothic 22,6% (31), English Gothic 21,4% (14), Gothic Revival 15,8%
(228). **L'ipotesi Brick Gothic è smentita**: l'errore si concentra sulle chiese con etichetta
generica, spesso edifici medievali di transizione. Accuracy per Paese: Francia 0,74, Italia 0,49.
Confidenza media: 0,72 sulle predizioni corrette, 0,59 su quelle errate.

## D10 – Mappa di Kohonen (SOM) al posto dell'algoritmo genetico
L'algoritmo genetico avrebbe cercato gli iperparametri della testa (learning rate, dropout,
neuroni, L2, esponente dei pesi di classe) con fitness = macro-F1 di validazione. Guadagno atteso
piccolo e rischio di sovradattamento a un validation ridotto. La SOM, addestrata senza etichette
sui vettori di 512 caratteristiche del modello finale, risponde invece a una domanda emersa dai
risultati: gli stili sono separati nello spazio appreso dalla rete, e dove si sovrappongono
(Rinascimento/Barocco, Gotico/Romanico)?

## D11 – Miglioramenti del modello (notebook 03b)
Riferimento: sul benchmark a 6 stili gli autori di WikiChurches riportano una balanced accuracy
del 60,2% (ResNet-50, 224 px) e del 69,5% nella configurazione migliore (448 px, pre-addestramento
ImageNet-21k, bilanciamento); anche lì il Rinascimentale è lo stile più confuso con il Barocco.

Interventi a basso costo, tutti scelti sul validation:
- validation allargato con il 15% delle chiese del training (il test ufficiale resta invariato);
- sovra-campionamento (almeno 400 immagini per stile per epoca) e augmentation forte;
- **M3**: VGG-16 con fine-tuning dei blocchi 4 e 5, partendo da M1;
- test-time augmentation (media tra immagine e immagine ribaltata);
- confronto con **ConvNeXt-Tiny** (2022), stesso protocollo e stessa testa.

Il modello finale è scelto sulla macro-F1 di validation; il test serve solo a riportare i
risultati. Escluso per tempi: aumento della risoluzione a 448 px (richiederebbe di rigenerare il
dataset), indicato come sviluppo futuro.

**Dati del notebook 03b.** Validation allargato: 1.082 immagini (Romanico 295, Gotico 512,
Rinascimentale 61, Barocco 214) contro le 206 dello split ufficiale. Training: 4.962 immagini,
5.307 dopo il sovra-campionamento (Rinascimentale da 55 a 400, ottenute ripetendo le stesse 55 foto).
Pesi di classe ricalcolati: Romanico 0,97 · Gotico 0,72 · Rinascimentale 1,82 · Barocco 1,20.

**Addestramento.**
- M3: migliore all'epoca 5 su 11 (250 s su GPU L4). Dopo l'epoca 5 la loss di training continua a
  scendere (0,57 → 0,41) mentre quella di validation risale (0,77 → 0,81): sovradattamento,
  contenuto dall'early stopping.
- ConvNeXt-Tiny: fase 1 (solo testa) migliore all'epoca 2; nella fase 2 (sblocco di `stage_3`) la
  loss di validation non migliora mai (epoca migliore 1). Il fine-tuning non porta beneficio.

**Validation (scelta del modello)**

| Modello | Accuracy | Macro-F1 | Balanced acc. | F1 Rinasc. |
|---|---|---|---|---|
| M3 VGG-16 blocchi 4-5 | 0,713 | 0,637 | 0,624 | 0,414 |
| **M3 + TTA (scelto)** | 0,719 | **0,639** | 0,627 | 0,395 |
| M2 blocco 5 (ottimistico: parte del validation era nel suo training) | 0,701 | 0,638 | 0,633 | 0,424 |
| ConvNeXt-Tiny + TTA | 0,665 | 0,578 | 0,564 | 0,360 |

**Test ufficiale (1.481 immagini)**

| Modello | Accuracy | Macro-F1 | Balanced acc. | F1 Rom. | F1 Got. | F1 Rin. | F1 Bar. |
|---|---|---|---|---|---|---|---|
| M2 VGG-16 blocco 5 | 0,653 | 0,585 | 0,573 | 0,662 | 0,670 | 0,327 | 0,681 |
| M3 VGG-16 blocchi 4-5 | 0,676 | 0,594 | 0,581 | 0,678 | 0,708 | 0,299 | 0,691 |
| **M3 + TTA (modello finale)** | **0,677** | 0,588 | 0,577 | 0,678 | 0,711 | 0,272 | 0,692 |
| ConvNeXt-Tiny | 0,631 | 0,536 | 0,535 | 0,594 | 0,677 | 0,205 | 0,668 |
| ConvNeXt-Tiny + TTA | 0,630 | 0,532 | 0,532 | 0,591 | 0,675 | 0,192 | 0,671 |

McNemar M2 vs modello finale: b = 111 (solo M2 corretto), c = 146 (solo M3 + TTA corretto),
chi-quadro con correzione di continuità, p ≈ 0,034: il miglioramento è statisticamente
significativo al 5%, ma di entità modesta (+2,4 punti di accuracy).

**Lettura dei risultati**
- L'errore Gotico → Romanico scende dal 27% (M2) al 16%; la confusione inversa Romanico → Gotico
  sale al 27%. Le chiese di transizione romanico-gotica, con una sola etichetta Wikidata per
  edificio, restano il caso ambiguo principale.
- La test-time augmentation ha un effetto nullo (+0,002 di macro-F1 sul validation, −0,006 sul
  test). La scelta del modello resta quella fatta sul validation.
- ConvNeXt-Tiny, architettura più recente, è inferiore alla VGG-16 su questo problema: con dati
  scarsi e rumorosi un backbone più moderno non garantisce un guadagno.
- **Il Rinascimentale resta il limite principale**: recall 0,17 e precisione 0,68. Il modello lo
  predice raramente (25 volte su 1.481) e il 56% delle immagini rinascimentali è classificato come
  Barocco. Pesi di classe, sovra-campionamento, augmentation e cambio di architettura non lo hanno
  migliorato: il limite sono le 55 fotografie distinte disponibili nel training, non il modello.

## D12 – Dati esterni per il Rinascimentale (notebook 03c)
Dall'analisi del notebook 03b il limite del Rinascimentale sono i dati (55 fotografie distinte nel
training), non il modello. Si integra il training con **MonuMAI** (Lamas et al., 2021): 1.514 facciate di
monumenti in quattro stili, di cui 312 rinascimentali, con riquadri annotati da esperti su 15 elementi
architettonici; licenza CC BY 4.0.

Alternativa scartata: l'*Architectural Style Dataset* (Xu et al., 2014, 25 stili da Wikimedia Commons)
non ha una classe dedicata al Rinascimento.

Regole fissate prima di eseguire l'esperimento:
- MonuMAI entra solo nel training; validation e test restano quelli di WikiChurches;
- solo Gotico, Rinascimentale e Barocco, con lo stesso numero di immagini per stile: se MonuMAI
  fornisse solo immagini rinascimentali, la rete potrebbe imparare a riconoscere la provenienza della
  foto (inquadratura frontale, monumenti spagnoli) invece dello stile;
- eliminazione delle immagini quasi identiche a quelle di validation o test, con hash percettivo
  (pHash a 64 bit, distanza di Hamming ≤ 8): le due raccolte provengono entrambe dal web;
- stessa configurazione di M3; un secondo M3 senza dati esterni, con un altro seme, stima la
  variabilità dovuta al caso;
- il nuovo modello si accetta solo se la macro-F1 di validation supera quella di M3 + TTA di almeno
  max(0,01; variabilità stimata). Il test si usa solo dopo la decisione.

Le annotazioni degli elementi di MonuMAI servono anche alle fasi successive: verifica delle mappe di
salienza (la rete guarda gli elementi indicati dagli esperti?) e descrizione degli elementi nell'app.

**Risultati del notebook 03c**
- Duplicati: nessuna immagine MonuMAI è quasi identica a immagini di validation o test
  (0 su 1.187; distanza minima 10 bit, mediana 18).
- Training integrato: 312 immagini MonuMAI per stile (Gotico, Rinascimentale, Barocco). Il
  Rinascimentale passa da 55 a 367 immagini distinte, di cui l'85% da MonuMAI.

Validation (1.082 immagini):

| Modello | Accuracy | Macro-F1 | F1 Rinasc. |
|---|---|---|---|
| M3 (03b) + TTA – modello finale | 0,719 | **0,639** | 0,395 |
| M3 controllo (stessi dati, altro seme) + TTA | 0,689 | 0,597 | 0,300 |
| M3 + MonuMAI + TTA | 0,694 | 0,573 | 0,187 |

Variabilità stimata tra due addestramenti identici: 0,042 di macro-F1, quindi soglia 0,042.
M3 + MonuMAI: Δ = −0,066 → **respinto** secondo la regola fissata; il modello finale resta M3 (03b) + TTA.

Test ufficiale, riportato per completezza dopo la decisione: M3 + MonuMAI + TTA ottiene accuracy
0,689, macro-F1 0,610, balanced accuracy 0,595 (F1 Rom. 0,690 · Got. 0,722 · Rin. 0,321 · Bar. 0,706),
contro 0,677 / 0,588 / 0,577 del modello finale; recall del Rinascimentale 0,21 contro 0,17.
McNemar: b = 72, c = 91, p ≈ 0,16, differenza **non significativa**.

**Lettura**
- Validation e test danno indicazioni opposte (−0,066 contro +0,022). Le differenze sono dello stesso
  ordine della variabilità tra due addestramenti identici (0,04): con questi dati, scarti di qualche
  centesimo di macro-F1 tra varianti di M3 non sono distinguibili dal caso. La regola decisa prima
  dell'esperimento evita di scegliere il modello in base al test.
- Il valore di validation di M3 (03b) è probabilmente ottimistico: M3 è stato scelto come migliore
  tra più candidati sullo stesso validation (effetto del «vincitore»), e il controllo con un altro seme
  ottiene 0,597.
- MonuMAI non risolve il Rinascimentale (recall sul test da 0,17 a 0,21): sono facciate andaluse
  fotografate da vicino, un dominio diverso dalle chiese di WikiChurches, spesso fotografate per intero. Il limite resta la quantità di dati rinascimentali dello
  stesso dominio.
- MonuMAI resta utile per le fasi successive: le annotazioni dei 15 elementi architettonici servono a
  verificare le mappe di salienza e a descrivere gli elementi nell'app.

## D13 – Il modello si congela; mappa di Kohonen sulle caratteristiche (notebook 04)
Dopo il notebook 03c il modello finale resta **M3 (03b) + TTA**. Ensemble, soglie per classe e
calibrazione delle probabilità sono stati valutati e scartati: i guadagni attesi sono dello stesso ordine
della variabilità tra addestramenti (0,04 di macro-F1) e l'incertezza tra stili vicini viene gestita
nell'applicazione con una risposta graduata (logica fuzzy).

La SOM (Kohonen) proietta le 512 caratteristiche dello strato GAP del modello finale (media tra immagine e
immagine ribaltata, standardizzate sul training) su una griglia 18 × 18, circa 5·√N neuroni per N ≈ 5.000
immagini di training. Addestramento online non supervisionato: 20 presentazioni per immagine, raggio del
vicinato da 9 neuroni e tasso di apprendimento da 0,5 in decrescita esponenziale, pesi iniziali sul piano
delle prime due componenti principali.

Uso dei risultati:
- mappa degli stili, mappe per stile e U-matrix: dove gli stili sono separati e dove si sovrappongono;
- SOM come classificatore (etichetta del neurone vincitore) confrontata con la rete;
- posizione degli errori sulla mappa (confini, regioni miste);
- «chiese simili»: le 5 immagini più vicine nello spazio delle caratteristiche, mostrate dall'app.
