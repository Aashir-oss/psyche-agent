"""Task definitions with labeled output formats the app can parse reliably."""
import re

from crewai import Task

# ---------------------------------------------------------------- formats ---

INTAKE_FORMAT = """\
SYMPTOMS: <main feelings/symptoms in plain words>
DURATION: <how long this has been going on>
SEVERITY_1_10: <single number 1-10>
TRIGGERS: <what seems to set it off>
CONTEXT: <life situation: work/study, relationships, health, sleep>
LANGUAGE: <the language of the user's message — e.g. 'English', 'Urdu', 'Punjabi', 'Arabic'. IMPORTANT: if it is Urdu written in Latin/Roman letters (e.g. 'mujhe neend nahi aati', 'tension ho rahi hai'), write EXACTLY 'Roman Urdu'>
CRISIS_FLAG: <YES or NO>
"""

PERSPECTIVE_FORMAT = """\
HYPOTHESIS: <one careful hypothesis, hedged — never a diagnosis>
CONFIDENCE: <percentage, e.g. 60%>
KEY_FACTORS: <2-4 supporting factors, comma-separated>
QUESTIONS:
- <warm, open-ended question 1>
- <warm, open-ended question 2>
- <warm, open-ended question 3>
"""

PLAN_FORMAT = """\
SUMMARY: <3-4 sentence plain-language summary of what the perspectives found>
ADVICE_PATHS:
(write exactly "PATH 1:", "PATH 2:", "PATH 3:" on their own lines — no bullets,
no bold, no dashes on the PATH lines themselves; steps below each stay as "- " bullets)
PATH 1: <path title>
- <concrete step 1>
- <concrete step 2>
TECHNIQUE_IDS: <card IDs used, e.g. box_breathing, worry_time>
PATH 2: <path title>
- <concrete step 1>
- <concrete step 2>
TECHNIQUE_IDS: <card IDs used>
PATH 3: <path title>
- <concrete step 1>
- <concrete step 2>
TECHNIQUE_IDS: <card IDs used>
SKEPTIC_ANSWERED: <how the plan addresses the skeptic's main concern, 2-3 sentences>
WHEN_TO_SEEK_HELP: <concrete signs it's time to see a professional, 2-4 bullets>
REFERENCES:
- <CARD ID> — <technique title> — <source line>
"""

SKEPTIC_FORMAT = """\
WEAK_LEAPS: <weakest inferential leaps across the perspectives>
ALTERNATIVES: <2-3 alternative explanations, including ordinary situational ones>
MISSING_EVIDENCE: <what would change your mind, phrased as questions>
"""

RESEARCH_FORMAT = """\
CARDS_USED:
- <CARD ID> — <why it fits this user, one line>
- <CARD ID> — <why it fits this user, one line>
HISTORY_MATCHES: <similar past sessions of THIS user, or "none — first session">
"""


# ------------------------------------------------------ language rule ---

def language_directive(language: str) -> str:
    """Instruction appended to task prompts so agents reply in the user's language.

    Section labels (HYPOTHESIS:, PATH 1:, ...) stay in English — only the
    content is written in the user's language, so parsing never breaks.
    """
    if not language or language.strip().lower() == "english":
        return ""
    lang = language.strip()
    extra = ""
    if lang.lower() == "roman urdu":
        extra = (" Write in Latin/Roman script exactly like the user does — "
                 "do NOT switch to Urdu (Arabic) script and do NOT switch to English.")
    return (
        f"\nLANGUAGE RULE: the user communicates in {lang}. Write ALL "
        f"user-facing content (questions, hypotheses, advice, summaries) in "
        f"{lang}.{extra} Keep every labeled section header (e.g. HYPOTHESIS:, "
        f"CONFIDENCE:, PATH 1:, REFERENCES:) in English EXACTLY as specified — "
        f"translate only the content, never the labels."
    )


# ---------------------------------------------------------------- parsing ---

def _label_pattern(lbl: str) -> "re.Pattern":
    # Tolerant label match: optional quote/bullet prefix, optional **bold**
    # (before the label and/or after the colon), any case, flexible space
    # before the colon. Catches "**CONSENSUS:**", "**Consensus**:",
    # "- hypothesis:", "> PATH 1 :" etc. without breaking exact matches.
    return re.compile(
        rf"(?m)^\s*(?:>\s*)?[*\-•]?\s*\*{{0,2}}{re.escape(lbl)}\*{{0,2}}\s*:\*{{0,2}}",
        re.IGNORECASE,
    )


def parse_sections(text: str, labels: list[str]) -> dict:
    """Split labeled output (LABEL: value) into a dict. Missing labels -> ''."""
    out = {lbl: "" for lbl in labels}
    if not text:
        return out
    # find each label's start position
    positions = []
    for lbl in labels:
        m = _label_pattern(lbl).search(text)
        if m:
            positions.append((m.start(), lbl, m.end()))
    positions.sort()
    for i, (start, lbl, end) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        value = text[end:stop].strip()
        # the model sometimes emits markdown separators ("---") between
        # sections — strip a trailing one so it doesn't leak into the value
        value = re.sub(r"\s*(\n\s*)?(-{3,}|\*{3,}|_{3,})\s*$", "", value)
        out[lbl] = value
    return out


def parse_questions(questions_block: str) -> list[str]:
    lines = []
    for line in (questions_block or "").splitlines():
        line = re.sub(r"^([-*•]|\d+[.)])\s+", "", line.strip())
        if line:
            lines.append(line)
    # prefer lines with a question mark (Latin "?" or Urdu "؟"), but never
    # drop a question just because the model omitted it (common in Roman
    # Urdu) — an empty queue would silently skip phase-2 input entirely.
    with_q = [l for l in lines if "?" in l or "؟" in l]
    qs = with_q + [l for l in lines if l not in with_q]
    return qs[:3]


# ---------------------------------------------------------------- phase 1 ---
# Phase 1 runs as TWO crews: intake first (it also detects the user's language
# at zero extra API cost), then the four perspectives, which are instructed to
# reply in that language. The user's message / intake text is embedded directly
# in the task descriptions so nothing depends on CrewAI's input plumbing.

def build_intake_task(a: dict, user_text: str) -> Task:
    return Task(
        description=(
            "Read the user's message below and structure it. Output EXACTLY this "
            "labeled format and nothing else:\n\n" + INTAKE_FORMAT +
            f"\nUSER'S MESSAGE:\n{(user_text or '')[:2000]}\n"
            "\nRules: be faithful to what the user said, never invent details. "
            "SEVERITY_1_10 is your best estimate from their words. "
            "LANGUAGE is the language the user wrote in (note the Roman Urdu rule "
            "in the format). "
            "CRISIS_FLAG is YES if there is ANY self-harm/suicide/wanting-to-die "
            "signal, otherwise NO."
        ),
        expected_output="Structured intake in the exact labeled format.",
        agent=a["intake"],
    )


def build_perspective_tasks(a: dict, intake_text: str, past_qa_text: str = "",
                            language: str = "English") -> dict:
    # Anti-repeat: questions already covered in earlier sessions must not be
    # asked again — the perspectives build on them instead. Empty on the
    # first session, so nothing changes there.
    anti_repeat = ""
    if past_qa_text.strip():
        anti_repeat = (
            "\nThe user was ALREADY asked these questions in earlier sessions "
            "(with their answers). Do NOT ask them again — treat the answers "
            "as known background, and ask NEW questions that build on them or "
            "explore uncovered ground:\n" + past_qa_text
        )
    lang_rule = language_directive(language)
    perspectives = {}
    for key, label in (("cbt", "CBT"), ("psychodynamic", "Psychodynamic"),
                       ("humanistic", "Humanistic"), ("biopsychosocial", "Bio-Psycho-Social")):
        perspectives[key] = Task(
            description=(
                f"Here is the structured intake:\n{intake_text}\n\n"
                f"Using the {label} lens, analyze the intake. "
                "Output EXACTLY this labeled format and nothing else:\n\n"
                + PERSPECTIVE_FORMAT +
                "\nRules: HYPOTHESIS is one careful hedged hypothesis — never a "
                "diagnosis. CONFIDENCE is honest (most hypotheses are 40-70%). "
                "QUESTIONS must be warm, open-ended, and answerable by the user."
                + anti_repeat
                + lang_rule
            ),
            expected_output=f"{label} perspective in the exact labeled format.",
            agent=a[key],
        )
    return perspectives


# ---------------------------------------------------------------- phase 2 ---

def build_phase2_tasks(a: dict, intake_text: str, perspectives_text: str,
                       user_answers: str, language: str = "English") -> dict:
    lang_rule = language_directive(language)
    t_skeptic = Task(
        description=(
            "Red-team the four perspectives below. Output EXACTLY this labeled "
            "format:\n\n" + SKEPTIC_FORMAT +
            "\nBe sharp but fair. Attack ideas, never the person."
            + lang_rule
        ),
        expected_output="Skeptic critique in the exact labeled format.",
        agent=a["skeptic"],
    )
    t_research = Task(
        description=(
            "Ground the upcoming advice in vetted sources.\n"
            "FIRST use the 'Search knowledge base' tool 2-3 times with different "
            "keyword angles (symptoms, needs, situation). THEN use 'Search own "
            "history' once with the core symptoms.\n"
            "Output EXACTLY this labeled format:\n\n" + RESEARCH_FORMAT +
            "\nRules: every card you list MUST be a real CARD ID from the tool "
            "results. Never invent a source."
            + lang_rule
        ),
        expected_output="Researched cards with IDs + history matches.",
        agent=a["researcher"],
    )
    t_plan = Task(
        description=(
            "Write the treatment plan using the intake, perspectives, skeptic "
            "critique, researched cards, and the user's answers below.\n\n"
            f"INTAKE:\n{intake_text}\n\n"
            f"PERSPECTIVES:\n{perspectives_text}\n\n"
            f"USER'S ANSWERS TO THE PERSPECTIVES' QUESTIONS:\n{user_answers}\n\n"
            "Output EXACTLY this labeled format:\n\n" + PLAN_FORMAT +
            "\nRules: 3 concrete advice paths a common person can follow. Every "
            "technique named must carry its real CARD ID and appear in REFERENCES "
            "with its source line. WHEN_TO_SEEK_HELP must be concrete."
            + lang_rule
        ),
        expected_output="Treatment plan in the exact labeled format.",
        agent=a["planner"],
        context=[t_skeptic, t_research],
    )
    t_edit = Task(
        description=(
            "Rewrite the plan below in plain language for a non-expert. KEEP every "
            "fact, number, card ID, and reference EXACTLY as written — change only "
            "the wording to be simpler. Keep all labeled sections and their order. "
            "Do not add new claims.\n\n" + t_plan.description
            + lang_rule
        ),
        expected_output="Plain-language plan in the exact labeled format.",
        agent=a["editor"],
        context=[t_plan],
    )
    return {"skeptic": t_skeptic, "research": t_research,
            "plan": t_plan, "edit": t_edit}
