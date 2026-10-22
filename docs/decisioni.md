# Registro delle decisioni

Ogni scelta è valutata con tre criteri: **Resa** (qualità e impatto all'esame),
**Tempo**, **Difficoltà**. I riferimenti «§» rimandano ai paragrafi della
*Dispensa IARTS discorsiva*.

## D1 – Dataset: WikiChurches al posto di Architectural Heritage Elements
- **Problema:** AHE contiene elementi architettonici (cupole, colonne, gargoyle…) ma
  nessuna etichetta di stile e nessun edificio intero: non permette di addestrare un
  classificatore di stili.
- **Scelta:** WikiChurches, chiese con stile da Wikidata, licenza CC BY-SA 4.0.
- **Resa:** alta (foto reali, 4 stili, identificativo dell'edificio per ogni foto).
  **Tempo:** medio (zip da 12,4 GB, ridotto una volta sola a circa 150 MB).
  **Difficoltà:** media.

## D2 – Quattro classi: Romanico, Gotico, Rinascimentale, Barocco
- Il Rinascimentale aggiunge la coppia difficile Rinascimento/Barocco: la matrice di
  confusione diventa più interessante da commentare.

## D3 – Solo chiese «pure»
- Si tengono solo le chiese i cui stili appartengono tutti a una sola delle 4 classi
  (come i sottoinsiemi ufficiali di WikiChurches, che usano chiese a etichetta singola).
- Le chiese miste (es. romanico + gotico) sono escluse dall'addestramento: potranno
  servire come casi di test qualitativi per la lettura fuzzy delle probabilità (§30, §50).

## D4 – Split per chiesa, 70/15/15, seed 42
- Tutte le foto di uno stesso edificio finiscono nello stesso insieme: altrimenti il test
  misurerebbe la memoria del singolo edificio e non lo stile (§17, §50).
- Split stratificato per stile; controllo automatico dei duplicati esatti (md5) tra insiemi.

## D5 – Immagini ridimensionate a lato corto 256 px
- Formato standard per reti ImageNet: ritaglio 224×224 al momento dell'addestramento (§44).
- Salvate come un unico `.tar` su Drive: molti file piccoli su Drive sono lenti.

## D6 – Tagli per rispettare la scadenza del 5 novembre
- Niente GPS/Geopy, niente algoritmo genetico, LLM solo se avanza tempo.
- Motivazioni nella revisione della pipeline (affidabilità, costo, assenza dalla dispensa).
