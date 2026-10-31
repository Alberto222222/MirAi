"""Testi dell'app MirAi in italiano e inglese: stili, famiglie, elementi architettonici, interfaccia."""
from __future__ import annotations

LINGUE = {"it": "Italiano", "en": "English"}

STILI = {
    "Romanesque": {
        "it": {"nome": "Romanico", "epoca": "XI – XII secolo",
               "testo": "Nasce in Europa intorno all'anno Mille, quando si torna a costruire in pietra su larga scala. "
                        "Gli edifici sono solidi e raccolti: muri spessi che reggono il peso delle volte, poche e piccole "
                        "finestre, interni in penombra.",
               "cerca": "archi a tutto sesto, muri massicci, archetti pensili sotto i tetti, portali strombati con "
                        "archi concentrici, campanili a pianta quadrata",
               "esempi": "Basilica di Sant'Ambrogio (Milano), Duomo di Modena, Cattedrale di Pisa"},
        "en": {"nome": "Romanesque", "epoca": "11th – 12th century",
               "testo": "It spreads across Europe around the year 1000, when large stone buildings return. Buildings "
                        "are solid and compact: thick walls carry the weight of the vaults, windows are few and small, "
                        "interiors are dim.",
               "cerca": "round arches, massive walls, small blind arches under the eaves, deep portals with "
                        "concentric arches, square bell towers",
               "esempi": "Sant'Ambrogio (Milan), Modena Cathedral, Pisa Cathedral"},
    },
    "Gothic": {
        "it": {"nome": "Gotico", "epoca": "metà XII – XV secolo",
               "testo": "Nasce in Francia (Saint-Denis, 1140 circa). L'arco acuto e gli archi rampanti scaricano il "
                        "peso verso l'esterno: i muri si alleggeriscono, le chiese salgono verso l'alto e si riempiono "
                        "di vetrate colorate. Lo stile vale anche per palazzi e municipi.",
               "cerca": "archi a sesto acuto, pinnacoli, guglie, rosoni, archi rampanti, grandi vetrate, verticalità",
               "esempi": "Notre-Dame de Paris, Duomo di Milano, Palazzo Ducale di Venezia"},
        "en": {"nome": "Gothic", "epoca": "mid-12th – 15th century",
               "testo": "Born in France (Saint-Denis, around 1140). Pointed arches and flying buttresses push the "
                        "weight outwards: walls become lighter, churches rise upwards and fill with stained glass. "
                        "The style is also used for palaces and town halls.",
               "cerca": "pointed arches, pinnacles, spires, rose windows, flying buttresses, large windows, verticality",
               "esempi": "Notre-Dame de Paris, Milan Cathedral, Doge's Palace (Venice)"},
    },
    "Renaissance": {
        "it": {"nome": "Rinascimentale", "epoca": "XV – XVI secolo",
               "testo": "Nasce a Firenze con Brunelleschi: si riscoprono le proporzioni e gli ordini dell'architettura "
                        "romana. Tutto è misurato, simmetrico e armonioso; la facciata si legge come una griglia "
                        "ordinata di piani e finestre.",
               "cerca": "simmetria, colonne e lesene con capitelli classici, finestre con timpani triangolari o "
                        "curvi, bugnato, cupole, serliane",
               "esempi": "Spedale degli Innocenti e Cupola del Duomo (Firenze), Palazzo Medici Riccardi, "
                         "Tempietto di San Pietro in Montorio (Roma)"},
        "en": {"nome": "Renaissance", "epoca": "15th – 16th century",
               "testo": "Born in Florence with Brunelleschi: the proportions and orders of Roman architecture are "
                        "rediscovered. Everything is measured, symmetrical and harmonious; the facade reads as an "
                        "orderly grid of floors and windows.",
               "cerca": "symmetry, columns and pilasters with classical capitals, windows with triangular or curved "
                        "pediments, rusticated stone, domes, serlian windows",
               "esempi": "Ospedale degli Innocenti and the Duomo dome (Florence), Palazzo Medici Riccardi, "
                         "Tempietto of San Pietro in Montorio (Rome)"},
    },
    "Baroque": {
        "it": {"nome": "Barocco", "epoca": "XVII – metà XVIII secolo",
               "testo": "Nasce a Roma con Bernini e Borromini. Usa lo stesso vocabolario classico del Rinascimento "
                        "ma lo mette in movimento: facciate curve, effetti di luce e ombra, decorazione ricca, "
                        "architettura come teatro che deve stupire.",
               "cerca": "facciate concave e convesse, timpani spezzati, volute, colonne tortili, statue, "
                        "forte chiaroscuro",
               "esempi": "San Carlo alle Quattro Fontane e Piazza San Pietro (Roma), Basilica di Santa Croce (Lecce)"},
        "en": {"nome": "Baroque", "epoca": "17th – mid-18th century",
               "testo": "Born in Rome with Bernini and Borromini. It uses the same classical vocabulary as the "
                        "Renaissance but sets it in motion: curved facades, light and shadow, rich decoration, "
                        "architecture as a theatre meant to amaze.",
               "cerca": "concave and convex facades, broken pediments, scrolls, twisted columns, statues, "
                        "strong contrast of light and shade",
               "esempi": "San Carlo alle Quattro Fontane and St Peter's Square (Rome), Santa Croce (Lecce)"},
    },
    "Classical": {
        "it": {"nome": "Classico antico", "epoca": "VII sec. a.C. – V sec. d.C.",
               "testo": "L'architettura greca e romana. I Greci fissano gli ordini (dorico, ionico, corinzio) e il "
                        "tempio con colonnato; i Romani aggiungono arco, volta, cupola e calcestruzzo per costruire "
                        "anfiteatri, terme e acquedotti.",
               "cerca": "colonne con capitelli dorici, ionici o corinzi, frontoni, arcate sovrapposte, "
                        "gradinate di teatri, rovine in pietra o mattone",
               "esempi": "Partenone (Atene), Colosseo e Pantheon (Roma), Teatro di Taormina"},
        "en": {"nome": "Classical antiquity", "epoca": "7th century BC – 5th century AD",
               "testo": "Greek and Roman architecture. The Greeks define the orders (Doric, Ionic, Corinthian) and the "
                        "colonnaded temple; the Romans add the arch, vault, dome and concrete to build amphitheatres, "
                        "baths and aqueducts.",
               "cerca": "columns with Doric, Ionic or Corinthian capitals, pediments, stacked arcades, theatre "
                        "seating, stone or brick ruins",
               "esempi": "Parthenon (Athens), Colosseum and Pantheon (Rome), Theatre of Taormina"},
    },
    "Neoclassical": {
        "it": {"nome": "Neoclassico", "epoca": "metà XVIII – metà XIX secolo",
               "testo": "Dopo gli scavi di Ercolano e Pompei l'Europa torna a imitare direttamente Grecia e Roma, "
                        "contro gli eccessi del Barocco. Linee pulite, superfici lisce, pochi ornamenti: è lo stile di "
                        "teatri, musei, banche e palazzi del potere.",
               "cerca": "pronao con colonne e frontone triangolare, simmetria rigorosa, superfici lisce, "
                        "decorazione sobria",
               "esempi": "Teatro alla Scala (Milano), San Francesco di Paola (Napoli), Porta di Brandeburgo (Berlino)"},
        "en": {"nome": "Neoclassical", "epoca": "mid-18th – mid-19th century",
               "testo": "After the excavations of Herculaneum and Pompeii, Europe goes back to imitating Greece and "
                        "Rome directly, against Baroque excess. Clean lines, smooth surfaces, little ornament: the "
                        "style of theatres, museums, banks and seats of power.",
               "cerca": "portico with columns and a triangular pediment, strict symmetry, smooth surfaces, "
                        "restrained decoration",
               "esempi": "La Scala (Milan), San Francesco di Paola (Naples), Brandenburg Gate (Berlin)"},
    },
}

FAMIGLIE = {
    "Medievale": {
        "it": {"nome": "Medievale", "testo": "Romanico e Gotico: il linguaggio delle masse murarie, degli archi e "
                                            "delle volte, prima della riscoperta degli ordini classici."},
        "en": {"nome": "Medieval", "testo": "Romanesque and Gothic: a language of masonry, arches and vaults, before "
                                           "the classical orders were rediscovered."},
    },
    "Classicismo moderno": {
        "it": {"nome": "Classicismo moderno", "testo": "Rinascimento, Barocco e Neoclassico: tre modi diversi di usare "
                                                      "colonne, timpani e proporzioni dell'antichità."},
        "en": {"nome": "Modern classicism", "testo": "Renaissance, Baroque and Neoclassical: three different ways of "
                                                    "using the columns, pediments and proportions of antiquity."},
    },
    "Classico antico": {
        "it": {"nome": "Classico antico", "testo": "Gli edifici originali di Grecia e Roma."},
        "en": {"nome": "Classical antiquity", "testo": "The original buildings of Greece and Rome."},
    },
}

# Elementi che l'app può nominare (xai.NOMI_AFFIDABILI) e testo per gli stili/famiglie tipiche
ELEMENTI = {
    "arco a sesto acuto": {
        "it": ("Arco a sesto acuto", "Due archi di cerchio che si incontrano in una punta: spinge il peso verso il "
                                     "basso e permette aperture più alte. Il simbolo del Gotico."),
        "en": ("Pointed arch", "Two circular arcs meeting at a point: it directs the weight downwards and allows taller "
                               "openings. The symbol of Gothic architecture.")},
    "pinnacolo gotico": {
        "it": ("Pinnacolo gotico", "Piccola guglia sopra un contrafforte: aggiunge peso che stabilizza la struttura "
                                   "e accentua lo slancio verticale."),
        "en": ("Gothic pinnacle", "A small spire on top of a buttress: its weight stabilises the structure and "
                                  "stresses the upward movement.")},
    "arco inflesso": {
        "it": ("Arco inflesso", "Arco con i lati a doppia curva (a «S») che si chiudono in una punta: tipico del "
                                "Gotico tardo, per esempio a Venezia."),
        "en": ("Ogee arch", "An arch with S-shaped sides meeting in a point: typical of late Gothic, for example in "
                            "Venice.")},
    "colonna tortile (salomonica)": {
        "it": ("Colonna tortile", "Colonna con il fusto avvolto a spirale: trasmette movimento, come nel baldacchino "
                                  "di Bernini in San Pietro. Tipica del Barocco."),
        "en": ("Twisted column", "A column with a spiral shaft that conveys movement, as in Bernini's baldachin in "
                                 "St Peter's. Typical of the Baroque.")},
    "timpano spezzato": {
        "it": ("Timpano spezzato", "Frontone aperto in alto o sui lati: rompe la forma chiusa classica per creare "
                                   "movimento. Tipico del Barocco."),
        "en": ("Broken pediment", "A pediment left open at the top or sides: it breaks the closed classical shape to "
                                  "create movement. Typical of the Baroque.")},
}

# Interfaccia
T = {
    "it": {
        "tab_guida": "🏛️ Guida", "tab_lab": "🔬 Laboratorio",
        "titolo": "MirAi", "sottotitolo": "La tua guida tascabile agli stili dell'architettura",
        "carica": "Scatta o carica la foto di un monumento", "analizza": "Che stile è?",
        "esempi": "Oppure prova con una di queste foto",
        "mira_attesa": "Ciao, sono <b>Mira</b>! Mostrami un monumento e ti dirò a che stile appartiene.",
        "mira_pensa": "Sto osservando i dettagli… dammi una ventina di secondi.",
        "mira_fatto": "Ecco cosa ho visto! Scorri i dettagli per capire perché.",
        "mira_dubbio": "Hmm, non sono molto sicura: guarda anche il secondo stile possibile.",
        "mira_ood": "Questa foto non somiglia agli edifici che conosco: prendi la risposta con cautela.",
        "mira_disegno": "Sembra un disegno o un'incisione: funziono meglio con le fotografie.",
        "mira_vuoto": "Prima scatta o carica una foto!",
        "famiglia": "Famiglia", "stile": "Stile", "secondo": "Seconda ipotesi",
        "sicuro": "Ne sono sicura", "probabile": "Probabilmente", "incerto": "Non ne sono sicura",
        "affid_stile": "Quando sono così sicura dello stile, ho ragione 88 volte su 100.",
        "affid_fam": "Quando sono così sicura della famiglia, ho ragione 91 volte su 100.",
        "affid_no": "Con questa confidenza sbaglio più spesso: considera anche la seconda ipotesi.",
        "affid_solo_fam": "Sulla famiglia sono affidabile (ho ragione 91 volte su 100); sullo stile meno: "
                          "considera anche la seconda ipotesi.",
        "epoca": "Epoca", "cerca": "Cosa cercare", "esempi_celebri": "Esempi celebri",
        "perche": "Perché? I dettagli che hanno deciso", "mappa": "Zone a favore (verde) e contro (rosso)",
        "dettaglio": "Dettaglio decisivo",
        "dettaglio_testo": "Questa zona ha pesato molto sulla decisione, ma non so dare un nome sicuro all'elemento.",
        "tipico": "tipico dello stile", "di_solito": "di solito", "stessa_fam": "stessa famiglia",
        "simili": "Edifici simili nella mia collezione", "in_arrivo": "I dettagli stanno arrivando…",
        "lab_titolo": "Dentro il modello",
        "lab_probs": "Probabilità per stile e famiglia", "lab_lime": "LIME: famiglia · stile · contrasto",
        "lab_som": "Dove cade la foto sulla mappa di Kohonen",
        "lab_metriche": "Metriche sul test (2.695 foto mai viste)", "lab_modello": "Il modello",
        "lab_vuoto": "Analizza una foto nella scheda Guida per vedere qui l'interno del modello.",
        "riscontro": "Il tuo riscontro", "riscontro_info": "Lo stile era giusto? Le correzioni verificate potranno "
                                                          "essere usate per riaddestrare la testa del modello.",
        "giusto": "Stile corretto secondo te", "commento": "Commento (facoltativo)", "invia": "Invia riscontro",
        "grazie": "Grazie! Riscontro salvato.", "no_foto": "Analizza prima una foto.",
        "non_so": "Non lo so",
        "tua_foto": "La mia foto", "msg_risposta": "Secondo me è <b>{stile}</b> ({ps}), famiglia <b>{fam}</b> ({pf}).",
        "msg_scorri": "Scorri i riquadri qui sotto: ognuno è un pezzo della foto che mi ha convinta.",
        "msg_inizio": "Tocca il riquadro per scattare una foto o sceglierne una dalla galleria, poi premi il pulsante.",
        "perche_di": "Perché {stile}?", "la_mappa": "La mappa di LIME",
        "mappa_testo": "verde: zone a favore · rosso: zone contro", "in_breve": "Lo stile in breve",
        "lab_lime_riga": "LIME sulle tre domande", "pezzo": "Pezzo",
    },
    "en": {
        "tab_guida": "🏛️ Guide", "tab_lab": "🔬 Lab",
        "titolo": "MirAi", "sottotitolo": "Your pocket guide to architectural styles",
        "carica": "Take or upload a photo of a monument", "analizza": "Which style is it?",
        "esempi": "Or try one of these photos",
        "mira_attesa": "Hi, I'm <b>Mira</b>! Show me a monument and I'll tell you its style.",
        "mira_pensa": "I'm looking at the details… give me about twenty seconds.",
        "mira_fatto": "Here's what I saw! Scroll through the details to see why.",
        "mira_dubbio": "Hmm, I'm not very sure: have a look at the second possible style too.",
        "mira_ood": "This photo doesn't look like the buildings I know: take my answer with caution.",
        "mira_disegno": "This looks like a drawing or an engraving: I work best with photographs.",
        "mira_vuoto": "Take or upload a photo first!",
        "famiglia": "Family", "stile": "Style", "secondo": "Second guess",
        "sicuro": "I'm sure", "probabile": "Probably", "incerto": "I'm not sure",
        "affid_stile": "When I'm this sure about the style, I'm right 88 times out of 100.",
        "affid_fam": "When I'm this sure about the family, I'm right 91 times out of 100.",
        "affid_no": "At this confidence I'm wrong more often: consider the second guess too.",
        "affid_solo_fam": "I'm reliable on the family (right 91 times out of 100), less so on the style: "
                          "consider the second guess too.",
        "epoca": "Period", "cerca": "What to look for", "esempi_celebri": "Famous examples",
        "perche": "Why? The details that decided", "mappa": "Areas in favour (green) and against (red)",
        "dettaglio": "Decisive detail",
        "dettaglio_testo": "This area weighed a lot in the decision, but I can't name the element with confidence.",
        "tipico": "typical of the style", "di_solito": "usually", "stessa_fam": "same family",
        "simili": "Similar buildings in my collection", "in_arrivo": "The details are on their way…",
        "lab_titolo": "Inside the model",
        "lab_probs": "Probability per style and family", "lab_lime": "LIME: family · style · contrast",
        "lab_som": "Where the photo falls on the Kohonen map",
        "lab_metriche": "Test metrics (2,695 unseen photos)", "lab_modello": "The model",
        "lab_vuoto": "Analyse a photo in the Guide tab to see the inside of the model here.",
        "riscontro": "Your feedback", "riscontro_info": "Was the style right? Verified corrections can be used to "
                                                       "retrain the head of the model.",
        "giusto": "Correct style in your opinion", "commento": "Comment (optional)", "invia": "Send feedback",
        "grazie": "Thank you! Feedback saved.", "no_foto": "Analyse a photo first.",
        "non_so": "I don't know",
        "tua_foto": "My photo", "msg_risposta": "I think it's <b>{stile}</b> ({ps}), family <b>{fam}</b> ({pf}).",
        "msg_scorri": "Scroll through the cards below: each one is a piece of the photo that convinced me.",
        "msg_inizio": "Tap the box to take a photo or pick one from your gallery, then press the button.",
        "perche_di": "Why {stile}?", "la_mappa": "The LIME map",
        "mappa_testo": "green: areas in favour · red: areas against", "in_breve": "The style in brief",
        "lab_lime_riga": "LIME on the three questions", "pezzo": "Piece",
    },
}


def t(lang: str, chiave: str) -> str:
    return T.get(lang, T["it"]).get(chiave, T["it"][chiave])


def nome_stile(stile: str, lang: str) -> str:
    return STILI[stile][lang]["nome"]


def nome_famiglia(fam: str, lang: str) -> str:
    return FAMIGLIE[fam][lang]["nome"]
