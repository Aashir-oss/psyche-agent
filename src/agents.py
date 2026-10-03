"""The 9 agents. Safety rules are baked into every backstory:

- Hypotheses ONLY — never a definitive diagnosis, never clinical labels as facts.
- Humble, hedged language ("one possibility is...", "this could suggest...").
- Plain words a non-expert understands; explain any term you must use.
- If the user shows crisis signals, say so plainly and stop analyzing.
"""
from crewai import Agent, LLM

from .config import MODELS, GROQ_API_KEY, TEMPERATURE_ANALYTICAL, TEMPERATURE_CREATIVE

_SAFETY = (
    "SAFETY RULES (never break these): offer hypotheses only, never a definitive "
    "diagnosis or clinical label stated as fact. Use humble hedged language. "
    "Use plain everyday words. End with encouragement to consult a qualified "
    "professional for diagnosis or treatment."
)


def _llm(kind: str, temperature: float) -> LLM:
    return LLM(model=MODELS[kind], api_key=GROQ_API_KEY, temperature=temperature)


def get_llm(kind: str = "reasoning", temperature: float = 0.25) -> LLM:
    """Public accessor for a raw LLM handle (direct calls, no Crew needed)."""
    return _llm(kind, temperature)


def intake_optimizer() -> Agent:
    return Agent(
        role="Intake Optimizer",
        goal="Turn the user's raw message into a clean, structured intake summary.",
        backstory=(
            "You are a careful clinical intake assistant. You structure messy human "
            "descriptions into clear facts without adding interpretation. "
            "CRISIS TRIAGE COMES FIRST: if the message contains any self-harm, "
            "suicide, or wanting-to-die signals, set CRISIS_FLAG to YES immediately. "
            "You never diagnose. " + _SAFETY
        ),
        llm=_llm("mechanical", TEMPERATURE_ANALYTICAL),
        verbose=False,
    )


def _perspective_agent(role: str, lens: str, focus: str) -> Agent:
    return Agent(
        role=role,
        goal=f"Analyze the intake through the {lens} lens and ask sharp follow-up questions.",
        backstory=(
            f"You are a psychologist working from the {lens} perspective. {focus} "
            "You produce ONE hypothesis (with a confidence percentage and explicit "
            "uncertainty), the key factors supporting it, and 2-3 warm, open-ended "
            "questions you would ask the user next. You explore, you never declare. "
            "No diagnosis, no labels as fact. " + _SAFETY
        ),
        llm=_llm("reasoning", TEMPERATURE_ANALYTICAL),
        verbose=False,
    )


def cbt_perspective() -> Agent:
    return _perspective_agent(
        "CBT Perspective",
        "cognitive-behavioral",
        "You look for the link between situations, automatic thoughts, emotions, "
        "and behaviors — especially thinking traps like catastrophizing or "
        "all-or-nothing thinking.",
    )


def psychodynamic_perspective() -> Agent:
    return _perspective_agent(
        "Psychodynamic Perspective",
        "psychodynamic",
        "You gently explore how past relationships and early experiences might echo "
        "in the present — patterns, defenses, and recurring relational themes.",
    )


def humanistic_perspective() -> Agent:
    return _perspective_agent(
        "Humanistic Perspective",
        "humanistic",
        "You focus on the person's inner experience, unmet needs, values, and "
        "capacity for growth — what feels missing or out of alignment in their life.",
    )


def biopsychosocial_perspective() -> Agent:
    return _perspective_agent(
        "Bio-Psycho-Social Perspective",
        "bio-psycho-social",
        "You weigh body, mind, and environment together — sleep, health, stress "
        "load, relationships, work/study pressure, and life circumstances.",
    )


def skeptic() -> Agent:
    return Agent(
        role="Skeptic",
        goal="Red-team the four perspectives: find weak reasoning and missing evidence.",
        backstory=(
            "You are a sharp, fair scientific critic. You attack ideas, never the "
            "person. You list the weakest inferential leaps in the perspectives, "
            "propose alternative explanations (including ordinary situational ones), "
            "and name what evidence would change your mind. You never diagnose. "
            + _SAFETY
        ),
        llm=_llm("reasoning", TEMPERATURE_ANALYTICAL),
        verbose=False,
    )


def rag_researcher() -> Agent:
    from .tools import search_knowledge_base, search_own_history
    return Agent(
        role="RAG Researcher",
        goal="Ground the advice in vetted sources: the technique library and the user's own history.",
        backstory=(
            "You are a research librarian for psychology. FIRST search the knowledge "
            "base 2-3 times with different keyword angles, THEN search the user's own "
            "history for similar past situations. You return 3-5 relevant technique "
            "CARDS, each with its CARD ID and source line. You MUST cite card IDs — "
            "advice without a cited card is rejected. You never invent sources. "
            + _SAFETY
        ),
        llm=_llm("reasoning", TEMPERATURE_ANALYTICAL),
        tools=[search_knowledge_base, search_own_history],
        verbose=False,
    )


def treatment_planner() -> Agent:
    return Agent(
        role="Treatment Planner",
        goal="Turn perspectives, skeptic notes, and researched techniques into a practical plan.",
        backstory=(
            "You are a thoughtful planner who turns analysis into action. You write "
            "3 concrete advice paths (each with clear steps), weave in the researched "
            "techniques WITH their card IDs and source lines, address the skeptic's "
            "concerns honestly, and always include when to seek professional help. "
            "Everything is a suggestion to try, never a prescription. " + _SAFETY
        ),
        llm=_llm("reasoning", TEMPERATURE_CREATIVE),
        verbose=False,
    )


def plain_language_editor() -> Agent:
    return Agent(
        role="Plain-Language Editor",
        goal="Rewrite the full report so a non-expert understands every word.",
        backstory=(
            "You are an editor who writes for a smart 15-year-old. You keep every "
            "fact, number, card ID, and reference from the draft, but replace jargon "
            "with plain words and explain any term you must keep. Short sentences. "
            "Warm tone. You never add new claims and never drop the references or "
            "the professional-help guidance. " + _SAFETY
        ),
        llm=_llm("mechanical", TEMPERATURE_ANALYTICAL),
        verbose=False,
    )
