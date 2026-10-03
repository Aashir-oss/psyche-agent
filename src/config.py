"""Central configuration: Groq models, temperatures, paths, safety text."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Override the DB location for tests: PSYCHE_DB_PATH=/tmp/test.db
DB_PATH = Path(os.getenv("PSYCHE_DB_PATH", DATA_DIR / "psyche.db"))

CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(exist_ok=True)
GRAPHS_DIR = DATA_DIR / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# FREE Groq models, quality-first assignment.
# 120b = reasoning (perspectives, skeptic, RAG researcher, planner).
# 20b  = mechanical (intake structuring, plain-language editing, distillation).
# ---------------------------------------------------------------------------
MODELS = {
    "reasoning": "groq/openai/gpt-oss-120b",
    "mechanical": "groq/openai/gpt-oss-20b",
}

TEMPERATURE_ANALYTICAL = 0.25  # analysis: precise, grounded
TEMPERATURE_CREATIVE = 0.7     # reframing: imaginative but plausible

APP_TITLE = "Your Personal Psych — AI Psychology Assistant"

DISCLAIMER = (
    "Your Personal Psych offers exploratory psychological perspectives for self-reflection "
    "and education only. It is not a psychological diagnosis, therapy, or "
    "medical advice, and it cannot replace a qualified mental-health "
    "professional. If you are in crisis, contact a helpline or emergency "
    "services immediately."
)

# ---- identity & history policy (demo mode: no login page yet) ----
DEMO_USERNAME = "demo"
MAX_SESSIONS_PER_USER = 5

# ---- crisis safety ----
# NOTE: helpline numbers can change — verify locally before any real deployment.
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "killing myself", "end my life",
    "want to die", "wish i were dead", "wish i was dead",
    "self-harm", "self harm", "hurt myself", "harm myself",
    "cutting myself", "no reason to live", "better off dead",
    # Roman Urdu (Urdu in Latin script) crisis signals
    "khudkushi", "khud kushi", "marna chahta", "marna chahti",
    "marna chahte", "mar jana", "mar jaun", "jeene ka dil nahi",
    "jeena nahi chahta", "jeena nahi chahti", "marne ka dil",
    # Urdu script crisis signals
    "خودکشی", "مرنا چاہتا", "مرنا چاہتی", "جینے کا دل نہیں",
]

HELPLINES = [
    ("Pakistan — Umang Mental Health Helpline", "0311 7786264"),
    ("Pakistan — Emergency / Rescue", "1122"),
    ("International — Find a Helpline", "https://findahelpline.org"),
]

CRISIS_RESPONSE = (
    "I'm really glad you told me this. What you're feeling matters, and you "
    "deserve support from a real person right now — I'm an AI and I can't "
    "provide the help you need in this moment.\n\n"
    "Please reach out immediately:\n"
    + "\n".join(f"- {name}: {contact}" for name, contact in HELPLINES)
    + "\n\nIf you feel you may act on these thoughts, call your local emergency "
    "number now. You don't have to face this alone."
)

# Phrasing the final safety scan blocks: definitive-diagnosis language.
DIAGNOSIS_PATTERNS = [
    r"\byou have ([a-z\- ]{2,40}) ?disorder\b",
    r"\byou (?:are|were) diagnosed with\b",
    r"\bdiagnosis:\s*[a-z]",
    r"\bmy diagnosis is\b",
    r"\bi diagnose you with\b",
]
