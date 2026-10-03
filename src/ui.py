"""Theme for the app: dark gradient with floating light orbs. Pure CSS."""
import base64
import html
from functools import lru_cache
from pathlib import Path

import streamlit as st

# Optional AI-generated ambient background video. If assets/bg-calm.mp4 exists
# it plays silently behind everything (muted, looping); otherwise the theme
# falls back to the animated gradient alone.
BG_VIDEO = Path(__file__).resolve().parent.parent / "assets" / "bg-calm.mp4"

# Optional custom dustbin icon for the sidebar session-delete buttons.
# Streamlit buttons only accept text/emoji, so the icon is injected as a CSS
# background image (see _trash_icon_css). If the file is missing, the emoji
# label shows as before.
TRASH_ICON = Path(__file__).resolve().parent.parent / "assets" / "trash.png"


def _trash_icon_css() -> str:
    """CSS that swaps the delete-button emoji for the custom dustbin icon."""
    if not TRASH_ICON.exists():
        return ""
    b64 = base64.b64encode(TRASH_ICON.read_bytes()).decode("ascii")
    return f"""
<style>
/* crisp dustbin icon on the sidebar session-delete buttons */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
div[data-testid="column"]:nth-child(2) button {{
    font-size: 0 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
div[data-testid="column"]:nth-child(2) button p {{
    display: none;
}}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
div[data-testid="column"]:nth-child(2) button::after {{
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    background: url("data:image/png;base64,{b64}") center / contain no-repeat;
    vertical-align: middle;
}}
</style>
"""

THEME_CSS = """
<style>
/* animated deep-teal/violet background */
.stApp {
    background: linear-gradient(-50deg, #07131f, #0e2a3a, #14224d, #0a1a2e);
    background-size: 380% 380%;
    animation: psycheDrift 26s ease infinite;
}
@keyframes psycheDrift {
    0% { background-position: 0% 45%; }
    50% { background-position: 100% 55%; }
    100% { background-position: 0% 45%; }
}
/* AI ambient background video: fixed, subtle, never intercepts clicks */
#psyche-bg-video {
    position: fixed; inset: 0;
    width: 100vw; height: 100vh;
    object-fit: cover;
    z-index: 0;
    opacity: .35;
    pointer-events: none;
}
/* soft floating light orbs */
.stApp::before {
    content: "";
    position: fixed; inset: 0;
    pointer-events: none; z-index: 0;
    background:
        radial-gradient(420px 420px at 15% 20%, rgba(94,234,212,.13), transparent 60%),
        radial-gradient(520px 520px at 85% 80%, rgba(129,140,248,.12), transparent 60%),
        radial-gradient(380px 380px at 70% 12%, rgba(45,212,191,.10), transparent 60%);
    animation: orbRise 18s ease-in-out infinite alternate;
}
@keyframes orbRise {
    from { transform: translate3d(0,-24px,0); }
    to { transform: translate3d(0,24px,0); }
}
.stApp > header, .stApp [data-testid="stAppViewContainer"] {
    position: relative; z-index: 1;
}
/* hero title shimmer */
.hero-title {
    font-size: 2.6rem; font-weight: 800; text-align: center;
    background: linear-gradient(90deg, #5eead4, #a5b4fc, #5eead4);
    background-size: 220% auto;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: shimmer 5s linear infinite;
}
@keyframes shimmer { to { background-position: 220% center; } }
.hero-sub {
    text-align: center; color: #b8c4d8; font-size: 1.05rem; margin-top: .4rem;
}
/* gentle entrance for content blocks */
.fade-up { animation: fadeUp .7s ease both; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}
/* glowing primary buttons */
.stButton > button[kind="primary"] {
    box-shadow: 0 0 18px rgba(94,234,212,.35);
    transition: box-shadow .25s ease, transform .15s ease;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 28px rgba(94,234,212,.55);
    transform: translateY(-1px);
}
/* perspective cards */
.persp-card {
    border: 1px solid rgba(148,163,184,.25);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .8rem;
    background: rgba(15,23,42,.55);
}
.disclaimer-box {
    border-left: 4px solid #f59e0b; border-radius: 8px;
    background: rgba(245,158,11,.08); padding: .8rem 1rem; margin: 1rem 0;
    color: #d6c9a8; font-size: .92rem;
}
.crisis-box {
    border: 2px solid #ef4444; border-radius: 12px;
    background: rgba(239,68,68,.10); padding: 1.2rem; margin: 1rem 0;
}
/* ---------- landing: mini perspective cards ---------- */
.persp-mini { text-align: center; font-size: .95rem; }
/* ---------- Q&A progress bar ---------- */
.qa-progress-wrap { margin: .6rem 0 1rem; }
.qa-progress-label { font-size: .9rem; color: #c3cbdf; margin-bottom: .35rem;
    font-weight: 600; }
.qa-progress-track { height: 8px; border-radius: 999px;
    background: rgba(148,163,184,.18); overflow: hidden; }
.qa-progress-fill { height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #2dd4bf, #818cf8);
    box-shadow: 0 0 12px rgba(94,234,212,.55); transition: width .4s ease; }
/* ---------- confidence meters ---------- */
.conf-meter { height: 10px; border-radius: 999px;
    background: rgba(148,163,184,.16); overflow: hidden; margin: .5rem 0 .25rem; }
.conf-fill { height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #2dd4bf, #a5b4fc);
    box-shadow: 0 0 10px rgba(94,234,212,.5); }
.conf-label { font-size: .8rem; color: #94a3b8; }
.persp-card h4 { margin-top: 0; color: #e2e8f0; }
/* ---------- consensus banner ---------- */
.consensus-banner { border: 1px solid rgba(94,234,212,.35); border-radius: 12px;
    padding: 1rem 1.2rem; background: rgba(45,212,191,.07);
    margin-bottom: 1.1rem; color: #d7e3f4; }
/* ---------- breathing coach: pure CSS, 16s cycle (4-4-4-4) ---------- */
.breath-wrap { text-align: center; padding: 1.2rem 0 .6rem; }
.breath-circle { width: 170px; height: 170px; margin: 0 auto; border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, rgba(153,246,228,.9),
        rgba(45,212,191,.35) 55%, rgba(45,212,191,.06) 75%);
    box-shadow: 0 0 70px rgba(94,234,212,.45), inset 0 0 45px rgba(94,234,212,.28);
    animation: breatheCycle 16s ease-in-out infinite; }
@keyframes breatheCycle {
    0%, 100% { transform: scale(.7); }
    25% { transform: scale(1.15); }
    50% { transform: scale(1.15); }
    75% { transform: scale(.7); }
}
.breath-labels { position: relative; height: 2.4rem; margin-top: 1rem;
    font-size: 1.35rem; font-weight: 700; color: #5eead4; }
.bphase { position: absolute; left: 0; right: 0; opacity: 0;
    animation: phaseCycle 16s linear infinite; }
@keyframes phaseCycle {
    0% { opacity: 0; } 4% { opacity: 1; } 21% { opacity: 1; }
    25%, 100% { opacity: 0; }
}
.bchips { display: flex; gap: .5rem; justify-content: center; margin-top: .9rem;
    flex-wrap: wrap; }
.bchip { padding: .32rem .85rem; border-radius: 999px; font-size: .85rem;
    border: 1px solid rgba(148,163,184,.3); color: #94a3b8;
    animation: chipCycle 16s linear infinite; }
@keyframes chipCycle {
    0% { background: rgba(94,234,212,.16); border-color: #5eead4; color: #5eead4;
         box-shadow: 0 0 12px rgba(94,234,212,.35); }
    21% { background: rgba(94,234,212,.16); border-color: #5eead4; color: #5eead4;
          box-shadow: 0 0 12px rgba(94,234,212,.35); }
    25%, 100% { background: transparent; border-color: rgba(148,163,184,.3);
                color: #94a3b8; box-shadow: none; }
}
.breath-note { margin-top: 1rem; color: #8b95ad; font-size: .9rem; }
/* ---- sidebar: kill top dead space, keep every row reachable ---- */
section[data-testid="stSidebar"] {
    overflow-y: auto !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 2.5rem !important;   /* clears the « button, nothing more */
    padding-bottom: 1rem !important;
}
section[data-testid="stSidebar"] h1 {
    margin-top: 0 !important;
    padding-top: 0 !important;
    font-size: 1.4rem !important;
}
section[data-testid="stSidebar"] hr {
    margin: .6rem 0 !important;
}
/* slimmer session rows */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button {
    padding-top: .35rem !important;
    padding-bottom: .35rem !important;
}
/* compact crisis contacts so they fit above the fold */
.side-sos { font-size: .85rem; line-height: 1.55; }
.side-sos a { color: #7dd3fc; }
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    _trash_css = _trash_icon_css()
    if _trash_css:
        st.markdown(_trash_css, unsafe_allow_html=True)
    if BG_VIDEO.exists():
        b64 = base64.b64encode(BG_VIDEO.read_bytes()).decode("ascii")
        st.markdown(
            '<video id="psyche-bg-video" autoplay muted loop playsinline '
            'aria-hidden="true">'
            f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
            "</video>",
            unsafe_allow_html=True,
        )


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero-title fade-up">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub fade-up">{subtitle}</div>', unsafe_allow_html=True)


def disclaimer_box(text: str) -> None:
    st.markdown(f'<div class="disclaimer-box">⚠️ {text}</div>', unsafe_allow_html=True)


def crisis_box(text: str) -> None:
    # escape first so no markup in the source text can ever render raw;
    # the findahelpline URL contains no escapable characters, so the link
    # replacement stays safe afterwards
    body = html.escape(text).replace("\n", "<br>")
    body = body.replace(
        "https://findahelpline.org",
        '<a href="https://findahelpline.org" target="_blank">findahelpline.org</a>',
    )
    st.markdown(
        '<div class="crisis-box">🆘 <b>Please read this first</b><br><br>'
        + body + "</div>",
        unsafe_allow_html=True,
    )


def confidence_meter(pct: int) -> str:
    """Glowing horizontal confidence bar (pct clamped to 0–100)."""
    try:
        pct = int(pct)
    except (TypeError, ValueError):
        pct = 50
    pct = max(0, min(100, pct))
    return (
        '<div class="conf-meter">'
        f'<div class="conf-fill" style="width:{pct}%"></div></div>'
        f'<div class="conf-label">Confidence: {pct}%</div>'
    )


def qa_progress(current: int, total: int) -> str:
    """Slim glowing 'Question X of N' progress bar for the Q&A flow."""
    pct = round(100 * current / max(1, total))
    return (
        '<div class="qa-progress-wrap">'
        f'<div class="qa-progress-label">Question {current} of {total}</div>'
        '<div class="qa-progress-track">'
        f'<div class="qa-progress-fill" style="width:{pct}%"></div>'
        "</div></div>"
    )


def consensus_banner(text: str) -> str:
    """Highlighted 'what the perspectives agree on' banner."""
    return ('<div class="consensus-banner">🤝 <b>What the perspectives agree on</b>'
            f"<br>{html.escape(text)}</div>")


def breathing_coach() -> str:
    """Pure-CSS box-breathing coach: 16s loop (inhale 4 / hold 4 / exhale 4 /
    rest 4). No timers, no blocking — the browser animates it."""
    phases = [("Breathe in…", "Inhale 4", 0), ("Hold…", "Hold 4", 4),
              ("Breathe out…", "Exhale 4", 8), ("Rest…", "Rest 4", 12)]
    labels = "".join(
        f'<span class="bphase" style="animation-delay:{d}s">{t}</span>'
        for t, _, d in phases)
    chips = "".join(
        f'<span class="bchip" style="animation-delay:{d}s">{c}</span>'
        for _, c, d in phases)
    return (
        '<div class="breath-wrap">'
        '<div class="breath-circle"></div>'
        f'<div class="breath-labels">{labels}</div>'
        f'<div class="bchips">{chips}</div>'
        '<div class="breath-note">Follow the circle — loop it a few times, '
        "about a minute of calm. 🌿</div>"
        "</div>"
    )


def _is_valid_image(p: Path) -> bool:
    """True only if the file is a genuinely readable image.

    A half-downloaded or mangled file can exist on disk but still crash
    Streamlit's avatar loader — this check keeps that from ever breaking
    the app.
    """
    try:
        from PIL import Image
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=8)
def _cached_avatar(mtime_ns: int, size: int) -> str:
    p = Path(__file__).resolve().parent.parent / "assets" / "avatar.png"
    if _is_valid_image(p):
        return str(p)
    return "🧠"


def assistant_avatar() -> str:
    """Custom psychologist avatar for chat messages.

    Uses assets/avatar.png when it exists AND is a readable image; falls back
    to 🧠 so the chat never shows the default robot icon — and a corrupt
    download degrades gracefully instead of crashing the app.
    """
    p = Path(__file__).resolve().parent.parent / "assets" / "avatar.png"
    if not p.exists():
        return "🧠"
    st_ = p.stat()
    return _cached_avatar(st_.st_mtime_ns, st_.st_size)
