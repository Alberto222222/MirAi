"""Interfaccia Gradio di MirAi: «Guida» (vetrina in stile streaming, toni caldi) e «Laboratorio» (blu), mascotte Mira.

La logica sta in src/app_logic.py; qui solo presentazione. Avvio dal notebook 06_app:
    demo = app_ui.costruisci(app, esempi); demo.queue().launch(share=True)
"""
from __future__ import annotations

import base64
import html
import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import config, hierarchy as hi, xai
from src import descrizioni as D
from src.descrizioni import t

N_LIME = 600

# Colori
CALDI = {"notte": "#160b14", "notte2": "#24101f", "porpora": "#8a2a6b", "porpora_chiaro": "#c2508f", "pesca": "#ffb08a",
         "pesca_chiaro": "#ffd9c4", "bianco": "#ffffff", "inchiostro": "#2a1424", "grigio": "#7a6672",
         "oro": "#ffc93c", "oro_scuro": "#a86d00", "crema": "#fff7f2"}
BLU = {"notte": "#0b1d3a", "blocco": "#12305c", "bordo": "#2a5a9c", "accento": "#4da3ff", "testo": "#e6f0ff"}
COL_FAM = ["#b3261e", "#6d1a4f", "#d4a017"]          # Medievale, Classicismo moderno, Classico antico
COL_FAM_LAB = ["#4da3ff", "#a78bfa", "#f6c453"]


# ---------------------------------------------------------------------------
# Mira: mascotte originale (statuetta di marmo con corona d'alloro, toga porpora e compasso d'architetto)
# ---------------------------------------------------------------------------
def _alloro() -> str:
    foglie = []
    for lato in (-1, 1):
        for i in range(6):
            a = math.radians(205 + i * 22) if lato < 0 else math.radians(335 - i * 22)
            x, y = 110 + 52 * math.cos(a), 104 + 50 * math.sin(a)
            rot = math.degrees(a) + (90 if lato < 0 else -90)
            foglie.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="5" ry="11" transform="rotate({rot:.0f} {x:.1f} {y:.1f})" '
                          f'fill="url(#oro)"/>')
    return "".join(foglie)


def mira_svg(stato: str = "attesa") -> str:
    """Mira: figura morbida di marmo (testa e corpo in un'unica forma), occhi a punto, corona d'alloro,
    drappo porpora e compasso d'architetto, su un piccolo capitello."""
    oy = 112 if stato != "pensa" else 108
    ox = 4 if stato == "pensa" else 0
    # sempre sorridente: cambia solo l'ampiezza del sorriso (e un sopracciglio alzato quando è in dubbio)
    bocca = {"attesa": "M98 130 Q110 143 122 130", "fatto": "M96 128 Q110 147 124 128",
             "pensa": "M101 131 Q110 140 119 131", "dubbio": "M101 131 Q110 139 119 131"}.get(stato, "M98 130 Q110 143 122 130")
    extra = ""
    if stato == "dubbio":
        extra = '<path d="M118 97 Q126 92 134 96" stroke="#2a1622" stroke-width="2.6" fill="none" stroke-linecap="round"/>'
    if stato == "pensa":
        extra = ('<g class="mira-idea"><circle cx="160" cy="48" r="4" fill="#ffc93c"/><circle cx="174" cy="40" r="4" '
                 'fill="#ffc93c" opacity=".7"/><circle cx="188" cy="32" r="4" fill="#ffc93c" opacity=".45"/></g>')
    elif stato == "fatto":
        extra += ('<g class="mira-idea"><circle cx="170" cy="44" r="3.5" fill="#ffc93c"/><circle cx="184" cy="56" r="2.5" '
                 'fill="#ffc93c"/><circle cx="182" cy="34" r="2" fill="#ffc93c"/></g>')
    return f"""<svg class="mira mira-{stato}" viewBox="0 0 220 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mira">
<defs>
 <radialGradient id="marmo" cx="38%" cy="30%" r="75%"><stop offset="0" stop-color="#ffffff"/><stop offset=".7" stop-color="#f3ece6"/><stop offset="1" stop-color="#ddd1c6"/></radialGradient>
 <linearGradient id="oro" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffe07a"/><stop offset="1" stop-color="#d99a00"/></linearGradient>
 <linearGradient id="toga" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#a3367a"/><stop offset="1" stop-color="#5e1846"/></linearGradient>
</defs>
<ellipse cx="110" cy="244" rx="54" ry="6" fill="#000" opacity="0.18"/>
<g class="mira-corpo">
 <rect x="66" y="218" width="88" height="9" rx="3" fill="#efe6dc"/>
 <rect x="74" y="227" width="72" height="11" rx="4" fill="#e4d8cb"/>
 <circle cx="72" cy="232" r="7" fill="#efe6dc"/><circle cx="72" cy="232" r="2.5" fill="#bfa98f"/>
 <circle cx="148" cy="232" r="7" fill="#efe6dc"/><circle cx="148" cy="232" r="2.5" fill="#bfa98f"/>
 <path d="M110 52 C150 52 164 86 164 120 C164 160 162 196 152 214 Q110 224 68 214 C58 196 56 160 56 120 C56 86 70 52 110 52 Z" fill="url(#marmo)"/>
 <path d="M60 168 Q104 178 160 150 L162 176 Q110 200 62 194 Z" fill="url(#toga)" opacity=".95"/>
 <circle cx="80" cy="182" r="4" fill="#ffc93c"/>
 {_alloro()}
 <g class="mira-compasso">
  <ellipse cx="166" cy="170" rx="10" ry="15" transform="rotate(-30 166 170)" fill="url(#marmo)"/>
  <line x1="172" y1="160" x2="180" y2="122" stroke="#c48a00" stroke-width="3.5" stroke-linecap="round"/>
  <line x1="172" y1="160" x2="196" y2="132" stroke="#c48a00" stroke-width="3.5" stroke-linecap="round"/>
  <circle cx="172" cy="160" r="4.5" fill="url(#oro)"/>
 </g>
 <g class="mira-occhi">
  <circle cx="{94 + ox}" cy="{oy}" r="7.5" fill="#2a1622"/><circle cx="{126 + ox}" cy="{oy}" r="7.5" fill="#2a1622"/>
  <circle cx="{96.5 + ox}" cy="{oy - 2.5}" r="2.2" fill="#fff"/><circle cx="{128.5 + ox}" cy="{oy - 2.5}" r="2.2" fill="#fff"/>
 </g>
 <circle cx="82" cy="128" r="6" fill="#f6a98c" opacity=".55"/><circle cx="138" cy="128" r="6" fill="#f6a98c" opacity=".55"/>
 <path d="{bocca}" stroke="#7b1f5c" stroke-width="3.2" fill="none" stroke-linecap="round"/>
 {extra}
</g>
</svg>"""


def mira_mini(stato: str = "attesa") -> str:
    return f'<div class="avatar">{mira_svg(stato)}</div>'


def _b64(img, lato: int | None = None, q: int = 85) -> str:
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img).astype(np.uint8))
    im = im.convert("RGB")
    if lato and max(im.size) > lato:
        im.thumbnail((lato, lato))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# Caratteri (Google Fonts): si possono cambiare dal notebook, es. app_ui.FONT_TITOLI = "Cinzel"
FONT_TITOLI = "Poppins"
FONT_TESTO = "Poppins"
CSS_EXTRA = ""          # regole CSS aggiuntive, applicate per ultime (per prove dal notebook)


def _url_font() -> str:
    fam = []
    for f in dict.fromkeys([FONT_TITOLI, FONT_TESTO]):
        fam.append(f"family={f.replace(' ', '+')}:wght@400;500;600;700;800")
    return "https://fonts.googleapis.com/css2?" + "&".join(fam) + "&display=swap"


def crea_css() -> str:
    """Foglio di stile, ricostruito con i valori attuali di colori e caratteri."""
    C, B = CALDI, BLU
    grad = f"linear-gradient(135deg, {C['porpora']} 0%, {C['porpora_chiaro']} 45%, {C['pesca']} 100%)"
    return f"""
@import url('{_url_font()}');
.gradio-container {{
  --body-background-fill: transparent; --background-fill-primary: transparent;
  --block-background-fill: rgba(255,255,255,0.05); --block-border-color: rgba(255,255,255,0.12);
  --block-label-background-fill: transparent; --block-label-text-color: {C['pesca_chiaro']};
  --block-title-text-color: {C['pesca_chiaro']}; --body-text-color: {C['crema']};
  --body-text-color-subdued: #cdb3c0; --input-background-fill: rgba(255,255,255,0.06);
  --border-color-primary: rgba(255,255,255,0.14); --color-accent-soft: rgba(255,176,138,0.15);
  font-family: '{FONT_TESTO}', sans-serif; max-width: 1180px !important; color: {C['crema']};
  background:
    radial-gradient(rgba(255,255,255,0.07) 1px, transparent 1.6px) 0 0 / 22px 22px,
    radial-gradient(520px 520px at 92% 4%, rgba(255,176,138,0.30), transparent 70%),
    radial-gradient(620px 620px at 0% 30%, rgba(138,42,107,0.45), transparent 70%),
    radial-gradient(500px 500px at 70% 95%, rgba(194,80,143,0.25), transparent 70%),
    linear-gradient(180deg, {C['notte2']} 0%, {C['notte']} 100%) !important;
}}
body {{ background: {C['notte']} !important; }}
.gradio-container * {{ font-family: '{FONT_TESTO}', sans-serif; }}
.titolo-app {{ display:flex; align-items:center; gap:14px; }}
.titolo-app h1 {{ font-family:'{FONT_TITOLI}', sans-serif; font-weight:800; font-size:2.6rem; margin:0; letter-spacing:-.02em;
  background:{grad}; -webkit-background-clip:text; background-clip:text; color:transparent; }}
.titolo-app p {{ margin:0; color:{C['pesca_chiaro']}; font-weight:500; }}
.orb {{ width:34px; height:34px; border-radius:50%; background:{grad}; box-shadow: 0 0 24px rgba(255,176,138,.55);
  animation: respira 4s ease-in-out infinite; }}
@keyframes respira {{ 50% {{ transform: scale(1.12); filter: hue-rotate(-12deg); }} }}
.tab-nav button {{ font-weight:600 !important; font-size:1rem !important; color:{C['pesca_chiaro']} !important; }}
.tab-nav button.selected {{ color:{C['bianco']} !important; border-color:{C['pesca']} !important; }}
/* pulsante: pillola con sfumatura porpora → pesca */
.bottone-stile {{ font-weight:700 !important; font-size:1.15rem !important; color:#fff !important; border:none !important;
  border-radius:999px !important; background:{grad} !important; padding:14px 22px !important;
  box-shadow: 0 8px 26px rgba(194,80,143,.45), inset 0 1px 0 rgba(255,255,255,.35) !important; transition: transform .12s, box-shadow .12s; }}
.bottone-stile:hover {{ transform: translateY(-1px); box-shadow: 0 12px 30px rgba(255,176,138,.5) !important; }}
.bottone-stile:active {{ transform: scale(.98); }}
/* chat: bolle bianche */
.chat {{ display:flex; flex-direction:column; gap:14px; padding:6px 2px; }}
.msg {{ display:flex; gap:12px; align-items:flex-end; animation: entra .35s ease-out; }}
.msg.utente {{ justify-content:flex-end; }}
.chat-host {{ display:flex; gap:18px; align-items:flex-start; }}
.host {{ position:sticky; top:12px; width:170px; min-width:150px; text-align:center; }}
.host svg {{ width:100%; filter: drop-shadow(0 10px 18px rgba(0,0,0,.4)); }}
.host-nome {{ display:inline-block; margin-top:2px; padding:3px 14px; border-radius:999px; background:{grad}; color:#fff;
  font-weight:700; font-size:.85rem; }}
.chat {{ flex:1; }}
.bolla.larga {{ max-width:100%; }}
.bolla {{ background:{C['bianco']}; color:{C['inchiostro']}; border-radius:22px 22px 22px 6px; padding:13px 17px; max-width:88%;
  line-height:1.5; font-size:.97rem; box-shadow: 0 6px 20px rgba(0,0,0,.25); }}
.bolla b {{ color:{C['porpora']}; }}
.msg.utente .bolla {{ background:{grad}; color:#fff; border-radius:22px 22px 6px 22px; }}
.msg.utente .bolla b {{ color:#fff; }}
.bolla img.mini {{ display:block; width:160px; border-radius:14px; margin-top:8px; }}
.bolla.avviso {{ background:{C['pesca_chiaro']}; }}
.bolla h4 {{ font-weight:700; color:{C['porpora']}; margin:0 0 6px; font-size:1.1rem; }}
.puntini {{ display:inline-flex; gap:5px; vertical-align:middle; margin-left:6px; }}
.puntini i {{ width:8px; height:8px; border-radius:50%; background:{grad}; animation: salta 1s infinite ease-in-out; }}
.puntini i:nth-child(2) {{ animation-delay:.15s; }} .puntini i:nth-child(3) {{ animation-delay:.3s; }}
@keyframes salta {{ 0%, 80%, 100% {{ transform: translateY(0); opacity:.5; }} 40% {{ transform: translateY(-5px); opacity:1; }} }}
.dots {{ display:inline-flex; gap:6px; margin:12px 0 4px; }}
.dots i {{ width:12px; height:12px; border-radius:50%; background:rgba(255,255,255,.22); }}
.dots i.on {{ box-shadow: 0 0 10px rgba(255,176,138,.55); }}
@keyframes entra {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform:none; }} }}
/* vetrina */
.hero {{ position:relative; height:440px; border-radius:28px; overflow:hidden; box-shadow: 0 20px 50px rgba(0,0,0,.5);
  margin-bottom:26px; background:{C['notte']}; }}
.hero-sfondo {{ position:absolute; inset:-30px; background-size:cover; background-position:center; filter: blur(26px) brightness(.4) saturate(1.2); }}
.hero-foto {{ position:absolute; right:0; top:0; height:100%; aspect-ratio:1/1; object-fit:cover;
  -webkit-mask-image: linear-gradient(90deg, transparent 0%, #000 30%); mask-image: linear-gradient(90deg, transparent 0%, #000 30%); }}
.hero::after {{ content:''; position:absolute; inset:0; pointer-events:none;
  background: linear-gradient(0deg, rgba(22,11,20,.8) 0%, rgba(22,11,20,0) 42%); }}
.hero-testo {{ position:absolute; left:32px; bottom:30px; z-index:1; max-width:56%; }}
.pill {{ display:inline-block; padding:5px 13px; border-radius:999px; font-size:.78rem; font-weight:600;
  background:{C['bianco']}; color:{C['porpora']}; }}
.pill.rossa {{ background:{C['pesca_chiaro']}; color:#a3243a; }}
.hero-titolo {{ font-family:'{FONT_TITOLI}', sans-serif; font-weight:800; font-size:4.2rem; line-height:1.02; margin:12px 0 6px;
  color:#fff; letter-spacing:-.03em; text-shadow: 0 6px 30px rgba(0,0,0,.45); }}
.hero-meta {{ color:#f5dfe9; font-size:1rem; }}
.hero-meta b {{ color:{C['pesca']}; }}
.riga h3 {{ font-family:'{FONT_TITOLI}', sans-serif; font-weight:700; font-size:1.35rem; color:#fff; margin:22px 0 12px; }}
.riga h3 span {{ font-size:.75rem; font-weight:600; color:{C['porpora']}; background:{C['bianco']}; border-radius:999px;
  padding:3px 10px; margin-left:10px; vertical-align:middle; }}
.scorri {{ display:flex; gap:16px; overflow-x:auto; scroll-snap-type:x mandatory; padding:4px 2px 16px; }}
.scorri::-webkit-scrollbar {{ height:8px; }} .scorri::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,.25); border-radius:4px; }}
.carta {{ flex:0 0 230px; scroll-snap-align:start; position:relative; border-radius:22px; overflow:hidden;
  background:{C['bianco']}; color:{C['inchiostro']}; box-shadow: 0 8px 24px rgba(0,0,0,.3); transition: transform .2s, box-shadow .2s; }}
.carta:hover {{ transform: translateY(-4px) scale(1.02); box-shadow: 0 16px 34px rgba(194,80,143,.35); }}
.carta.larga {{ flex-basis:300px; }}
.carta img {{ width:100%; aspect-ratio:1/1; object-fit:cover; display:block; }}
.carta .corpo {{ padding:12px 14px 14px; }}
.carta .nome {{ font-weight:700; color:{C['inchiostro']}; }}
.carta .testo {{ font-size:.83rem; color:{C['grigio']}; margin-top:4px; line-height:1.4; }}
.carta .tag {{ display:inline-block; margin-top:9px; font-size:.74rem; font-weight:600; padding:4px 10px; border-radius:999px;
  background:{C['pesca_chiaro']}; color:{C['porpora']}; }}
.num {{ position:absolute; top:10px; left:10px; width:30px; height:30px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-weight:700; color:#fff; background:{grad}; box-shadow:0 4px 12px rgba(0,0,0,.3); }}
.scheletro {{ height:310px; background: linear-gradient(90deg, #f4ecef, #ffffff, #f4ecef); background-size:200% 100%;
  animation: luccica 1.4s infinite; }}
@keyframes luccica {{ from {{ background-position:200% 0; }} to {{ background-position:-200% 0; }} }}
.mira-corpo {{ animation: galleggia 3.6s ease-in-out infinite; }}
.mira-occhi {{ transform-box: view-box; transform-origin: 110px 112px; animation: battito 5s infinite; }}
.mira-pensa .mira-compasso {{ transform-box: view-box; transform-origin: 172px 160px; animation: oscilla 1.2s ease-in-out infinite; }}
.mira-idea {{ animation: brilla 1.4s ease-in-out infinite; }}
@keyframes galleggia {{ 50% {{ transform: translateY(-6px); }} }}
@keyframes battito {{ 0%, 93%, 100% {{ transform: scaleY(1); }} 96% {{ transform: scaleY(0.1); }} }}
@keyframes oscilla {{ 50% {{ transform: rotate(-12deg); }} }}
@keyframes brilla {{ 50% {{ opacity: 0.4; }} }}
.benvenuto {{ display:flex; align-items:center; gap:28px; padding:30px 34px; border-radius:28px; background:{C['bianco']};
  color:{C['inchiostro']}; box-shadow: 0 16px 40px rgba(0,0,0,.35); position:relative; overflow:hidden; }}
.benvenuto::before {{ content:''; position:absolute; width:340px; height:340px; left:-60px; top:-90px; border-radius:50%;
  background: radial-gradient(circle, rgba(255,176,138,.55), rgba(194,80,143,.25) 55%, transparent 72%); filter: blur(10px); }}
.benvenuto svg {{ width:250px; min-width:180px; position:relative; }}
.benvenuto h2 {{ font-family:'{FONT_TITOLI}', sans-serif; font-weight:800; font-size:2.2rem; margin:0 0 6px; letter-spacing:-.02em;
  background:{grad}; -webkit-background-clip:text; background-clip:text; color:transparent; }}
.benvenuto p {{ color:{C['grigio']}; margin:4px 0; }}
/* laboratorio */
.lab {{ --block-background-fill: {B['blocco']}; --block-border-color: {B['bordo']}; --body-text-color: {B['testo']};
  --block-label-text-color: {B['accento']}; --block-title-text-color: {B['accento']}; --input-background-fill: #0f2547;
  --border-color-primary: {B['bordo']}; background: radial-gradient(500px 400px at 90% 0%, rgba(77,163,255,.25), transparent 70%),
  linear-gradient(170deg, {B['notte']}, #10284f) !important; border-radius:24px; padding:16px; color:{B['testo']}; }}
.carta-lab {{ background:{B['blocco']}; border:1px solid {B['bordo']}; border-radius:18px; padding:16px 20px; color:{B['testo']}; overflow-x:auto; }}
.lab h3 {{ font-weight:700; color:#fff; margin:4px 0 10px; }}
.lab table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
.lab th, .lab td {{ border-bottom:1px solid {B['bordo']}; padding:6px 8px; text-align:left; color:{B['testo']}; }}
.lab th {{ color:{B['accento']}; }}
.lab .carta {{ background:#fff; }}
.lab .riga h3 span {{ color:#1d4f91; }}
@media (max-width: 680px) {{
  .hero {{ height:430px; }} .hero-foto {{ width:100%; height:auto; aspect-ratio:1/1;
    -webkit-mask-image: linear-gradient(180deg, #000 55%, transparent 100%); mask-image: linear-gradient(180deg, #000 55%, transparent 100%); }}
  .hero::after {{ background: linear-gradient(0deg, rgba(22,11,20,.97) 0%, rgba(22,11,20,.85) 35%, rgba(22,11,20,0) 65%); }}
  .hero-testo {{ left:18px; right:18px; bottom:18px; max-width:none; }}
  .hero-titolo {{ font-size:2.7rem; }} .carta {{ flex-basis:200px; }} .carta.larga {{ flex-basis:260px; }}
  .benvenuto {{ flex-direction:column; text-align:center; padding:24px; }} .titolo-app h1 {{ font-size:2.1rem; }}
  .host {{ width:110px; min-width:100px; }} .chat-host {{ gap:10px; }}
}}
"""


CSS = crea_css()


# ---------------------------------------------------------------------------
# Guida: chat con Mira e vetrina
# ---------------------------------------------------------------------------
def dots(v: float, n: int = 10) -> str:
    """Livello di confidenza come fila di puntini (v in [0, 1])."""
    k = int(round(v * n))
    a, b = np.array([0x8a, 0x2a, 0x6b]), np.array([0xff, 0xb0, 0x8a])
    out = []
    for i in range(n):
        if i < k:
            c = (a + (b - a) * i / max(1, n - 1)).astype(int)
            out.append(f'<i class="on" style="background:rgb({c[0]},{c[1]},{c[2]})"></i>')
        else:
            out.append("<i></i>")
    return '<div class="dots">' + "".join(out) + "</div>"


PUNTINI = '<span class="puntini"><i></i><i></i><i></i></span>'


def _msg(testo: str, classe: str = "") -> str:
    return f'<div class="msg"><div class="bolla {classe}">{testo}</div></div>'


def _chat(stato: str, messaggi: list[str]) -> str:
    """Mira grande a sinistra (con l'espressione dell'ultimo messaggio), bolle bianche a destra."""
    return (f'<div class="chat-host"><div class="host">{mira_svg(stato)}<div class="host-nome">Mira</div></div>'
            f'<div class="chat">{"".join(messaggi)}</div></div>')


def chat_html(lang: str, R: dict | None = None, S: dict | None = None, foto_utente=None, avviso: str | None = None) -> str:
    if R is None:
        m = [_msg(t(lang, "mira_attesa")), _msg(t(lang, "msg_inizio"))]
        if avviso:
            m.append(_msg(avviso, "avviso"))
        return _chat("dubbio" if avviso else "attesa", m)
    fams = list(config.FAMIGLIE)
    m = [_msg(t(lang, "mira_attesa"))]
    if foto_utente is not None:
        m.append(f'<div class="msg utente"><div class="bolla">{t(lang, "tua_foto")}<img class="mini" src="{_b64(foto_utente, 300)}"></div></div>')
    risposta = t(lang, "msg_risposta").format(
        stile=D.nome_stile(config.CLASSES[R["stile"]], lang), ps=f"{R['p'][R['stile']]:.0%}",
        fam=D.nome_famiglia(fams[R["famiglia"]], lang), pf=f"{R['pf'][R['famiglia']]:.0%}")
    if R["sicura_stile"]:
        aff = t(lang, "affid_stile")
    elif R["sicura_famiglia"]:
        aff = t(lang, "affid_solo_fam")
    else:
        aff = t(lang, "affid_no")
    rv = D.nome_stile(config.CLASSES[R["rivale"]], lang)
    m.append(_msg(f'{risposta}<br><span style="opacity:.8">{aff} {t(lang, "secondo")}: {rv} ({R["p"][R["rivale"]]:.0%}).</span>'))
    if R["disegno"]:
        m.append(_msg(t(lang, "mira_disegno"), "avviso"))
    if R["fuori_dominio"]:
        m.append(_msg(t(lang, "mira_ood"), "avviso"))
    m.append(_msg(t(lang, "msg_scorri") if S else t(lang, "mira_pensa") + PUNTINI))
    if not S:
        stato = "pensa"
    elif R["sicura_famiglia"] and not (R["fuori_dominio"] or R["disegno"]):
        stato = "fatto"
    else:
        stato = "dubbio"
    return _chat(stato, m)


def benvenuto_html(lang: str) -> str:
    return (f'<div class="benvenuto">{mira_svg("attesa")}<div><h2>{t(lang, "titolo")}</h2>'
            f'<p>{t(lang, "sottotitolo")}</p><p style="opacity:.8">{t(lang, "msg_inizio")}</p></div></div>')


def _carta(img_uri: str, nome: str, testo: str = "", tag: str = "", num: int | None = None, larga: bool = False) -> str:
    n = f'<div class="num">{num}</div>' if num else ""
    tg = f'<span class="tag">{tag}</span>' if tag else ""
    tx = f'<div class="testo">{testo}</div>' if testo else ""
    return (f'<div class="carta{" larga" if larga else ""}">{n}<img src="{img_uri}" loading="lazy">'
            f'<div class="corpo"><div class="nome">{nome}</div>{tx}{tg}</div></div>')


def _zona_info(z: dict, lang: str):
    if z["nome"] and z["nome"] in D.ELEMENTI:
        nome, testo = D.ELEMENTI[z["nome"]][lang]
        tip, rel = z["tipico"], z["relazione"]
        tip_nome = (D.nome_stile(tip, lang) if tip in D.STILI else D.nome_famiglia(tip, lang)) if tip else ""
        if rel == "stile":
            tag = f"✓ {t(lang, 'tipico')} {tip_nome}"
        elif rel:
            tag = f"{t(lang, 'di_solito')}: {tip_nome}" + (f" · {t(lang, 'stessa_fam')}" if rel == "famiglia" else "")
        else:
            tag = ""
        return f"{nome} · {z['conf']:.0%}", testo, tag
    return t(lang, "dettaglio"), t(lang, "dettaglio_testo"), ""


def vetrina_html(R: dict | None, S: dict | None, lang: str, simili=None) -> str:
    if R is None:
        return benvenuto_html(lang)
    fams = list(config.FAMIGLIE)
    st, fam = config.CLASSES[R["stile"]], fams[R["famiglia"]]
    d = D.STILI[st][lang]
    lab_s = t(lang, "sicuro") if R["sicura_stile"] else t(lang, "probabile")
    uri = _b64(R["img"], 900, 88)
    hero = (f'<div class="hero"><div class="hero-sfondo" style="background-image:url({uri})"></div>'
            f'<img class="hero-foto" src="{uri}"><div class="hero-testo">'
            f'<span class="pill">{t(lang, "famiglia")} · {D.nome_famiglia(fam, lang)} · {R["pf"][R["famiglia"]]:.0%}</span>'
            + (' <span class="pill rossa">⚠</span>' if (R["fuori_dominio"] or R["disegno"]) else "") +
            f'<div class="hero-titolo">{d["nome"]}</div>'
            f'<div class="hero-meta">{d["epoca"]} · <b>{lab_s}</b> · {R["p"][R["stile"]]:.0%}</div>'
            f'{dots(R["p"][R["stile"]])}</div></div>')
    if S:
        carte = [_carta(_b64(mappa_lime(R, S), 600), t(lang, "la_mappa"), t(lang, "mappa_testo"), larga=True)]
        for i, z in enumerate(S["zone"], 1):
            nome, testo, tag = _zona_info(z, lang)
            carte.append(_carta(_b64(z["ritaglio"], 400), nome, testo, tag, num=i))
    else:
        carte = ['<div class="carta larga"><div class="scheletro"></div></div>'] + \
                ['<div class="carta"><div class="scheletro"></div></div>'] * 3
    perche = (f'<div class="riga"><h3>{t(lang, "perche_di").format(stile=d["nome"])}<span>LIME</span></h3>'
              f'<div class="scorri">{"".join(carte)}</div></div>')
    breve = (f'<div class="riga"><h3>{t(lang, "in_breve")}</h3>'
             + _msg(f'<h4>{d["nome"]} · {d["epoca"]}</h4>{d["testo"]}<br><br><b>{t(lang, "cerca")}:</b> {d["cerca"]}.'
                    f'<br><b>{t(lang, "esempi_celebri")}:</b> {d["esempi"]}.<br><br>'
                    f'<span style="opacity:.8">{D.FAMIGLIE[fam][lang]["nome"]}: {D.FAMIGLIE[fam][lang]["testo"]}</span>',
                    "larga") + '</div>')
    sim = simili if simili is not None else simili_carte(R)
    righe_sim = (f'<div class="riga"><h3>{t(lang, "simili")}</h3><div class="scorri">{sim}</div></div>' if sim else "")
    return hero + perche + breve + righe_sim


def simili_carte(R: dict) -> str:
    out = []
    for p, nome in R["simili"]:
        try:
            out.append(_carta(_b64(Image.open(p), 400), html.escape(nome[:40]) or "—"))
        except OSError:
            continue
    return "".join(out)


def mappa_lime(R: dict, S: dict) -> np.ndarray:
    """Foto con le zone LIME dello stile (verde a favore, rosso contro) e i riquadri numerati delle zone ingrandite."""
    img = R["img"]
    base = xai.overlay_lime(img, S["lime"]["segments"], S["lime"]["pesi"][S["stile"]])
    im = Image.fromarray(base)
    dr = ImageDraw.Draw(im)
    f = _font(20)
    for i, z in enumerate(S["zone"], 1):
        x0, y0, x1, y1 = z["box"]
        dr.rectangle([x0, y0, x1 - 1, y1 - 1], outline="#ffffff", width=3)
        dr.ellipse([x0 + 4, y0 + 4, x0 + 32, y0 + 32], fill=CALDI["porpora"], outline="#ffffff", width=2)
        dr.text((x0 + 18, y0 + 18), str(i), fill="#ffffff", font=f, anchor="mm")
    return np.asarray(im)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()



# ---------------------------------------------------------------------------
# Laboratorio
# ---------------------------------------------------------------------------
def _stile_lab(ax, fig):
    fig.patch.set_facecolor(BLU["notte"])
    ax.set_facecolor(BLU["notte"])
    ax.tick_params(colors=BLU["testo"])
    for s in ax.spines.values():
        s.set_color(BLU["bordo"])
    ax.title.set_color(BLU["accento"])


def grafico_probabilita(R: dict, lang: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s2f = hi.style_to_family()
    nomi_s = [D.nome_stile(c, lang) for c in config.CLASSES]
    nomi_f = [D.nome_famiglia(f, lang) for f in config.FAMIGLIE]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [2, 1.2]})
    for ax in (a1, a2):
        _stile_lab(ax, fig)
    o = np.argsort(R["p"])
    a1.barh([nomi_s[i] for i in o], R["p"][o], color=[COL_FAM_LAB[s2f[i]] for i in o])
    a1.set_xlim(0, 1); a1.set_title(t(lang, "stile"))
    a2.barh(nomi_f, R["pf"], color=COL_FAM_LAB)
    a2.set_xlim(0, 1); a2.set_title(t(lang, "famiglia"))
    for ax in (a1, a2):
        for b in ax.patches:
            ax.text(min(b.get_width() + 0.02, 0.85), b.get_y() + b.get_height() / 2, f"{b.get_width():.0%}",
                    va="center", color=BLU["testo"], fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def lime_lab_html(R: dict, S: dict | None, lang: str) -> str:
    """Le tre spiegazioni LIME (famiglia, stile, contrasto) come carte da scorrere, con l'R² di ciascuna."""
    if not S:
        return ""
    carte = "".join(_carta(_b64(im, 500), cap) for im, cap in lime_lab(R, S, lang))
    return f'<div class="riga"><h3>{t(lang, "lab_lime_riga")}</h3><div class="scorri">{carte}</div></div>'


def lime_lab(R: dict, S: dict, lang: str):
    seg, W = S["lime"]["segments"], S["lime"]["pesi"]
    st, rv, fam = S["stile"], S["rivale"], S["famiglia"]
    r2 = S["lime"]["r2"]
    nf = D.nome_famiglia(list(config.FAMIGLIE)[fam], lang)
    ns, nr = D.nome_stile(config.CLASSES[st], lang), D.nome_stile(config.CLASSES[rv], lang)
    vs = "vs" if lang == "en" else "e non"
    return [(xai.overlay_lime(R["img"], seg, W[xai.col_family(fam)]), f"{t(lang, 'famiglia')}: {nf} · R² {r2[xai.col_family(fam)]:.2f}"),
            (xai.overlay_lime(R["img"], seg, W[st]), f"{t(lang, 'stile')}: {ns} · R² {r2[st]:.2f}"),
            (xai.overlay_lime(R["img"], seg, W[xai.COL_CONTRASTO]), f"{ns} {vs} {nr} · R² {r2[xai.COL_CONTRASTO]:.2f}")]


def grafico_som(app, R: dict, lang: str):
    if app.som is None or R.get("bmu") is None:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    rows, cols = int(app.som["rows"]), int(app.som["cols"])
    lab = np.asarray(app.som["lab_famiglia"]).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(5, 5 * rows / cols))
    _stile_lab(ax, fig)
    ax.imshow(lab, cmap=ListedColormap(COL_FAM_LAB), vmin=0, vmax=2, alpha=0.85)
    if "umatrix" in app.som:
        U = np.asarray(app.som["umatrix"]).reshape(rows, cols)
        ax.imshow(U, cmap="gray_r", alpha=0.25)
    r, c = divmod(R["bmu"], cols)
    ax.scatter([c], [r], marker="*", s=420, c="white", edgecolors="black", linewidths=1.5, zorder=3)
    for i, f in enumerate(config.FAMIGLIE):
        ax.scatter([], [], c=COL_FAM_LAB[i], s=80, label=D.nome_famiglia(f, lang))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, labelcolor=BLU["testo"], fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t(lang, "lab_som"))
    fig.tight_layout()
    plt.close(fig)
    return fig


def info_lab(R: dict, S: dict | None, lang: str, app) -> str:
    fams = list(config.FAMIGLIE)
    righe = [(t(lang, "famiglia"), f"{D.nome_famiglia(fams[R['famiglia']], lang)} ({R['conf_famiglia']:.3f}; "
                                   f"{'≥' if R['sicura_famiglia'] else '<'} {app.soglia_fam:.2f})"),
             (t(lang, "stile"), f"{D.nome_stile(config.CLASSES[R['stile']], lang)} ({R['conf_stile']:.3f}; "
                                f"{'≥' if R['sicura_stile'] else '<'} {app.soglia_stile:.2f})"),
             (t(lang, "secondo"), D.nome_stile(config.CLASSES[R["rivale"]], lang)),
             ("OOD: 1 − sim. media 5 vicini" if lang == "en" else "Fuori dominio: 1 − simil. media 5 vicini",
              f"{R['d_ood']:.3f} (soglia {app.soglia_ood:.3f}) → {'⚠️' if R['fuori_dominio'] else '✓'}"),
             ("P(drawing)" if lang == "en" else "P(disegno)", f"{R['p_disegno']:.2f} (soglia {xai.SOGLIA_NON_FOTO})")]
    if S is not None:
        righe += [("LIME", f"{N_LIME} perturbazioni, {int(S['lime']['segments'].max()) + 1} super-pixel, "
                           f"R² stile {S['r2_stile']:.2f}")]
    corpo = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>" for a, b in righe)
    return f'<div class="carta-lab"><h3>{t(lang, "lab_titolo")}</h3><table>{corpo}</table></div>'


def metriche_html(lang: str) -> str:
    en = lang == "en"
    return f'''<div class="carta-lab"><h3>{t(lang, "lab_metriche")}</h3>
<table><tr><th>{"Test" if en else "Test"}</th><th>{"Family" if en else "Famiglia"}</th><th>{"Style accuracy" if en else "Stile: accuracy"}</th><th>Macro-F1</th><th>VGG-16 ({"family / style" if en else "famiglia / stile"})</th></tr>
<tr><td>{"Churches" if en else "Chiese"} (1.481)</td><td><b>0,912</b></td><td><b>0,777</b></td><td>0,724</td><td>0,895 / 0,698</td></tr>
<tr><td>{"Monuments" if en else "Monumenti"} (1.214)</td><td><b>0,862</b></td><td><b>0,680</b></td><td>0,692</td><td>0,815 / 0,560</td></tr></table>
<p>{"Safe answers: style correct 88% of the time on 56% of photos; family 91% on 95%." if en else
    "Risposte sicure: stile corretto all'88% sul 56% delle foto; famiglia al 91% sul 95%."}
{"McNemar vs VGG-16 (style): p ≈ 10⁻¹¹ churches, 10⁻¹⁶ monuments." if en else
 "McNemar contro VGG-16 (stile): p ≈ 10⁻¹¹ chiese, 10⁻¹⁶ monumenti."}</p>
<h3>LIME</h3><table>
<tr><td>{"Local fidelity (R²)" if en else "Fedeltà locale (R²)"}</td><td>{"style" if en else "stile"} 0,71 · {"contrast" if en else "contrasto"} 0,71 · {"family" if en else "famiglia"} 0,63</td></tr>
<tr><td>Deletion test (AUC)</td><td>LIME 0,485 vs {"random" if en else "casuale"} 0,686 · {"better in 100% of photos" if en else "migliore nel 100% delle foto"} (p = 7·10⁻¹⁵)</td></tr>
<tr><td>{"20% of the photo deleted" if en else "20% della foto cancellato"}</td><td>0,79 → 0,58 (LIME) · 0,79 → 0,76 ({"random" if en else "casuale"})</td></tr>
<tr><td>Grad-CAM {"on VGG-16 (comparison)" if en else "sulla VGG-16 (confronto)"}</td><td>AUC 0,234 vs 0,404 · {"better in 96%" if en else "migliore nel 96%"}</td></tr>
<tr><td>{"Element names" if en else "Nomi degli elementi"}</td><td>{"precision" if en else "precisione"} 0,82 · {"false alarms" if en else "falsi allarmi"} 0,07</td></tr></table></div>'''


def modello_html(lang: str) -> str:
    en = lang == "en"
    return f'''<div class="carta-lab"><h3>{t(lang, "lab_modello")}</h3>
<pre style="white-space:pre-wrap;color:#e6f0ff;background:transparent;font-size:.85rem">{"photo" if en else "foto"} 448×448
 ├─► CLIP ViT-L/14   ({"frozen" if en else "congelato"}) ─► 768
 └─► DINOv2 ViT-L/14 ({"frozen" if en else "congelato"}) ─► 2.048
        ─► {"standardisation" if en else "standardizzazione"} ─► Dense 2.816→512 · ReLU · Dropout 0,5 ─► Dense 512→6 · softmax
        ─► {"family = sum of its styles → hierarchical answer" if en else "famiglia = somma dei suoi stili → risposta gerarchica"}
LIME: SLIC ~60 super-pixel · {N_LIME} {"perturbations" if en else "perturbazioni"} · kernel {"cosine" if en else "coseno"} 0,25 · Ridge</pre>
<p>{"Only the MLP head (≈1.45 M parameters) is trained; the two Transformers (≈300 M parameters each) are frozen." if en else
    "Si addestra solo la testa MLP (≈1,45 M parametri); i due Transformer (≈300 M parametri ciascuno) sono congelati."}</p></div>'''



# ---------------------------------------------------------------------------
# Interfaccia
# ---------------------------------------------------------------------------
def intestazione(lang: str) -> str:
    return f'<div class="titolo-app"><div class="orb"></div><h1>{t(lang, "titolo")}</h1><p>{t(lang, "sottotitolo")}</p></div>'


def scelte_stile(lang: str):
    return [(D.nome_stile(c, lang), c) for c in config.CLASSES] + [(t(lang, "non_so"), "?")]


def costruisci(app, esempi: list[str] | None = None):
    import gradio as gr

    def lab_vuoto(lang):
        return f'<div class="carta-lab">{t(lang, "lab_vuoto")}</div>'

    def analizza(foto, lang, progress=gr.Progress()):
        if foto is None:
            yield (chat_html(lang, avviso=t(lang, "mira_vuoto")), vetrina_html(None, None, lang), lab_vuoto(lang),
                   None, "", None, None)
            return
        R = app.analizza(foto)
        sim = simili_carte(R)
        yield (chat_html(lang, R, None, R["img"]), vetrina_html(R, None, lang, sim), info_lab(R, None, lang, app),
               grafico_probabilita(R, lang), "", grafico_som(app, R, lang), {"R": R, "S": None, "sim": sim})
        S = app.spiega(R, num_samples=N_LIME, progress=lambda f: progress(f, desc="LIME"))
        yield (chat_html(lang, R, S, R["img"]), vetrina_html(R, S, lang, sim), info_lab(R, S, lang, app),
               grafico_probabilita(R, lang), lime_lab_html(R, S, lang), grafico_som(app, R, lang),
               {"R": R, "S": S, "sim": sim})

    def cambia_lingua(lang, stato):
        """Ritraduce l'interfaccia e, se c'è già un'analisi, i risultati (senza ripetere LIME)."""
        fissi = [intestazione(lang), gr.update(label=t(lang, "carica")), gr.update(value=t(lang, "analizza")),
                 metriche_html(lang), modello_html(lang),
                 gr.update(choices=scelte_stile(lang), label=t(lang, "giusto")), gr.update(label=t(lang, "commento")),
                 gr.update(value=t(lang, "invia")), f"**{t(lang, 'riscontro')}** — {t(lang, 'riscontro_info')}"]
        if not stato:
            return fissi + [chat_html(lang), vetrina_html(None, None, lang), lab_vuoto(lang),
                            gr.update(), "", gr.update()]
        R, S = stato["R"], stato["S"]
        return fissi + [chat_html(lang, R, S, R["img"]), vetrina_html(R, S, lang, stato["sim"]), info_lab(R, S, lang, app),
                        grafico_probabilita(R, lang), lime_lab_html(R, S, lang), grafico_som(app, R, lang)]

    def invia(stato, stile_ok, commento, lang):
        if not stato:
            return t(lang, "no_foto")
        app.salva_riscontro(stato["R"], stile_ok or "?", commento or "", lang)
        return t(lang, "grazie")

    with gr.Blocks(title="MirAi", css=crea_css() + CSS_EXTRA, theme=gr.themes.Base(primary_hue="pink", neutral_hue="stone")) as demo:
        stato = gr.State(None)
        with gr.Row(equal_height=True):
            testa = gr.HTML(intestazione("it"))
            lingua = gr.Radio([("Italiano", "it"), ("English", "en")], value="it", show_label=False, scale=0, min_width=220)
        with gr.Tabs():
            with gr.Tab("🏛️ Guida · Guide"):
                with gr.Row():
                    with gr.Column(scale=4, min_width=280):
                        foto = gr.Image(sources=["upload", "webcam"], type="pil", label=t("it", "carica"), height=320)
                        bottone = gr.Button(t("it", "analizza"), elem_classes=["bottone-stile"])
                        if esempi:
                            gr.Examples(examples=[[e] for e in esempi], inputs=[foto], label="✨")
                    with gr.Column(scale=6, min_width=280):
                        chat = gr.HTML(chat_html("it"))
                vetrina = gr.HTML(vetrina_html(None, None, "it"))
            with gr.Tab("🔬 Laboratorio · Lab", elem_classes=["lab"]):
                info = gr.HTML(lab_vuoto("it"))
                probs = gr.Plot(show_label=False)
                lime3 = gr.HTML()
                with gr.Row():
                    som = gr.Plot(show_label=False)
                    metriche = gr.HTML(metriche_html("it"))
                with gr.Row():
                    modello = gr.HTML(modello_html("it"))
                    with gr.Column():
                        titolo_risc = gr.Markdown(f"**{t('it', 'riscontro')}** — {t('it', 'riscontro_info')}")
                        stile_ok = gr.Dropdown(scelte_stile("it"), label=t("it", "giusto"))
                        commento = gr.Textbox(label=t("it", "commento"), lines=2)
                        bottone_risc = gr.Button(t("it", "invia"), elem_classes=["bottone-stile"])
                        esito = gr.Markdown()

        bottone.click(analizza, [foto, lingua], [chat, vetrina, info, probs, lime3, som, stato])
        lingua.change(cambia_lingua, [lingua, stato],
                      [testa, foto, bottone, metriche, modello, stile_ok, commento, bottone_risc, titolo_risc,
                       chat, vetrina, info, probs, lime3, som])
        bottone_risc.click(invia, [stato, stile_ok, commento, lingua], esito)
    return demo
