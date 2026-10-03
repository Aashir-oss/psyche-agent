"""CrewAI tools the agents can call. Scoped per-user via ContextVar (set per run)."""
from contextvars import ContextVar

from crewai.tools import tool

from .rag import query_knowledge, query_history

_current_user_id: ContextVar[int | None] = ContextVar("rag_user_id", default=None)


def set_search_user_id(user_id: int | None) -> None:
    _current_user_id.set(user_id)


def _fmt_cards(cards: list[dict]) -> str:
    if not cards:
        return "No relevant entries found."
    out = []
    for c in cards:
        src = f"Source: {c['source']}" if c.get("source") else ""
        out.append(
            f"--- CARD {c['id']} ---\n"
            f"Title: {c.get('title', '')}\n"
            f"{c['text']}\n"
            f"{src}".strip()
        )
    return "\n\n".join(out)


@tool("Search knowledge base")
def search_knowledge_base(query: str) -> str:
    """Search the curated psychology technique library (vetted sources).
    Input: keywords describing the user's symptoms or needs,
    e.g. 'panic anxiety grounding' or 'low mood inactivity'.
    Returns matching technique cards with their CARD IDs and source lines.
    IMPORTANT: cite the CARD IDs you use in your final output.
    """
    try:
        return _fmt_cards(query_knowledge(query, n=5))
    except RuntimeError as e:
        return f"Knowledge base unavailable: {e}"


@tool("Search own history")
def search_own_history(query: str) -> str:
    """Search THIS user's own past assessments for similar situations.
    Input: keywords about the current situation, e.g. 'exam stress insomnia'.
    Returns the user's own similar past sessions (newest relevant first).
    Only ever returns this user's history — never anyone else's.
    """
    uid = _current_user_id.get()
    if uid is None:
        return "No user context set — skipping history search."
    try:
        cards = query_history(query, uid, n=5)
    except RuntimeError as e:
        return f"History search unavailable: {e}"
    if not cards:
        return "No similar past sessions found for this user (first session or no match)."
    out = []
    for c in cards:
        out.append(f"--- Past session ---\n{c['text']}")
    return "\n\n".join(out)
