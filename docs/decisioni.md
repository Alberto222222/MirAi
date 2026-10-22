# Scelte progettuali

Ogni scelta è valutata secondo tre criteri: **resa** (qualità del risultato),
**tempo** di realizzazione e **difficoltà** implementativa. I riferimenti «§»
rimandano ai paragrafi della dispensa del corso IARTS.

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

## D3 – Solo chiese a stile singolo
Vengono usate solo le chiese i cui stili appartengono tutti a una delle quattro classi,
in linea con i sottoinsiemi di riferimento di WikiChurches. Le chiese con stili misti
(es. romanico e gotico) sono escluse dall'addestramento e possono servire come casi di
studio qualitativi per la lettura «fuzzy» delle probabilità (§30, §50).

## D4 – Split per chiesa (70/15/15, seed 42)
Tutte le foto di uno stesso edificio appartengono a un solo insieme. Uno split per foto
porterebbe immagini dello stesso edificio sia in addestramento sia in test, e la
valutazione misurerebbe il riconoscimento del singolo monumento anziché dello stile
(§17, §50). Lo split è stratificato per stile; i duplicati esatti (stesso hash MD5)
presenti in insiemi diversi vengono rimossi da validazione e test.

## D5 – Immagini ridimensionate a 256 px sul lato corto
Formato standard per le reti pre-addestrate su ImageNet, con ritaglio 224×224
in addestramento (§44). Le immagini sono archiviate in un unico file `.tar` su Google
Drive, più rapido da trasferire di migliaia di file singoli.

## D6 – Perimetro del progetto
Per privilegiare un sistema completo e valutato in modo rigoroso sono stati esclusi:

- **ricerca del monumento tramite GPS/EXIF**: i metadati di posizione sono spesso assenti
  nelle foto condivise e indicano la posizione del fotografo, non del monumento;
- **ottimizzazione degli iperparametri con algoritmo genetico**: ogni valutazione
  richiederebbe un addestramento della VGG-16, con un costo non compatibile con le
  risorse disponibili; si usano valori di riferimento ed early stopping (§17, §18);
- **generazione del testo con un LLM**: sostituita da testi descrittivi per stile,
  che non rischiano di introdurre informazioni errate.
