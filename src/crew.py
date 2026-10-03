"""Two-phase orchestration + deterministic safety guard (pure code, not LLM).

Phase 1: intake optimizer + 4 perspectives (each asks 2-3 questions).
Phase 2: skeptic -> RAG researcher -> treatment planner -> plain-language editor.
"""
import re
import time

from crewai import Crew, Process, Agent, Task, LLM

from . import agents as A
from .config import (MODELS, GROQ_API_KEY, TEMPERATURE_ANALYTICAL, DISCLAIMER,
                     CRISIS_KEYWORDS, CRISIS_RESPONSE, DIAGNOSIS_PATTERNS)
from .database import save_feedback, save_learning, get_learnings, get_assessments
from .tasks import (build_intake_task, build_perspective_tasks, build_phase2_tasks,
                    parse_sections, parse_questions,
                    language_directive)
from .tools import set_search_user_id

_PERSPECTIVE_LABELS = ["HYPOTHESIS", "CONFIDENCE", "KEY_FACTORS", "QUESTIONS"]
_PERSPECTIVE_NAMES = {
    "cbt": "CBT", "psychodynamic": "Psychodynamic",
    "humanistic": "Humanistic", "biopsychosocial": "Bio-Psycho-Social",
}


# ------------------------------------------------------- safety guard ---

def check_crisis(text: str) -> bool:
    """Deterministic crisis triage on the RAW user message. Runs before any LLM call."""
    low = text.lower()
    return any(k in low for k in CRISIS_KEYWORDS)


def scan_output_safety(text: str) -> str:
    """Deterministic post-LLM scan: block definitive-diagnosis phrasing, ensure disclaimer."""
    cleaned = text
    for pat in DIAGNOSIS_PATTERNS:
        if re.search(pat, cleaned, flags=re.IGNORECASE):
            cleaned = (
                "> ⚠️ Safety note: an earlier draft used definitive diagnostic language, "
                "which this assistant must not do. What follows are hypotheses to "
                "discuss with a qualified professional — not a diagnosis.\n\n"
            ) + cleaned
            break
    if "not a psychological diagnosis" not in cleaned.lower():
        cleaned = cleaned.rstrip() + "\n\n---\n" + DISCLAIMER
    return cleaned


def get_past_qa_text(user_id: int, max_sessions: int = 3,
                     max_chars: int = 1500) -> str:
    """Compact text of questions already asked in earlier sessions.

    The perspectives get this so they don't re-ask the same questions —
    they build on past answers instead. Empty string for first sessions
    (no behavior change). Pure DB read, no LLM.
    """
    parts = []
    for a in get_assessments(user_id, limit=max_sessions):
        for qa in ((a.get("answers") or {}).get("qa") or []):
            q = (qa.get("question") or "").strip()
            if not q:
                continue
            ans = (qa.get("answer") or "").strip() or "(skipped)"
            parts.append(f"[{qa.get('perspective', 'Perspective')}] "
                         f"Q: {q} A: {ans}")
    return "\n".join(parts)[:max_chars]


# ------------------------------------------------------------- phase 1 ---

def _normalize_language(raw: str) -> str:
    """Clean the intake's LANGUAGE field into a short language name."""
    lang = (raw or "").strip().split("\n")[0].strip()[:40]
    if not lang:
        return "English"
    low = lang.lower()
    # the model sometimes writes a sentence ("the user wrote in Roman Urdu")
    if "roman" in low and "urdu" in low:
        return "Roman Urdu"
    return lang


def _kickoff_with_retry(crew, attempts: int = 2, pause: int = 10) -> None:
    """crew.kickoff() with one retry after a pause — covers transient Groq
    522s. Raises the last exception if all attempts fail; callers show a
    friendly message and keep user data safe. Retries only happen on
    failure, so the normal API budget is unchanged."""
    for attempt in range(1, attempts + 1):
        try:
            crew.kickoff()
            return
        except Exception:
            if attempt < attempts:
                time.sleep(pause)
            else:
                raise


def run_phase1(user_text: str, user_id: int) -> dict:
    """Intake + 4 perspectives. Returns structured data or a crisis response.

    Runs as two crews: intake first (its LANGUAGE field detects the user's
    language at zero extra API cost), then the four perspectives, instructed
    to reply in that language.

    Crisis path: returns {"crisis": True, "crisis_response": ...} and makes
    ZERO LLM calls.
    """
    if check_crisis(user_text):
        return {"crisis": True, "crisis_response": CRISIS_RESPONSE,
                "language": "English"}

    agents = {
        "intake": A.intake_optimizer(),
        "cbt": A.cbt_perspective(),
        "psychodynamic": A.psychodynamic_perspective(),
        "humanistic": A.humanistic_perspective(),
        "biopsychosocial": A.biopsychosocial_perspective(),
    }
    # Crew 1: intake alone.
    t_intake = build_intake_task(agents, user_text)
    _kickoff_with_retry(Crew(agents=[agents["intake"]], tasks=[t_intake],
                                process=Process.sequential, verbose=False))
    intake_raw = t_intake.output.raw
    intake = parse_sections(
        intake_raw,
        ["SYMPTOMS", "DURATION", "SEVERITY_1_10", "TRIGGERS", "CONTEXT",
         "LANGUAGE", "CRISIS_FLAG"],
    )
    # belt-and-braces: LLM-side crisis flag also triggers the crisis path
    _flag = intake["CRISIS_FLAG"].strip()
    if _flag.upper().startswith("Y") or _flag == "\u06c1\u0627\u06ba":
        return {"crisis": True, "crisis_response": CRISIS_RESPONSE,
                "language": "English"}
    language = _normalize_language(intake["LANGUAGE"])

    # Crew 2: perspectives, replying in the user's language.
    ptasks = build_perspective_tasks(
        agents, intake_raw,
        past_qa_text=get_past_qa_text(user_id), language=language)
    _kickoff_with_retry(Crew(
        agents=[agents[k] for k in ("cbt", "psychodynamic", "humanistic",
                                    "biopsychosocial")],
        tasks=list(ptasks.values()),
        process=Process.sequential, verbose=False))

    perspectives = []
    for key, name in _PERSPECTIVE_NAMES.items():
        raw = ptasks[key].output.raw
        p = parse_sections(raw, _PERSPECTIVE_LABELS)
        perspectives.append({
            "key": key, "name": name, "raw": raw,
            "hypothesis": p["HYPOTHESIS"], "confidence": p["CONFIDENCE"],
            "key_factors": p["KEY_FACTORS"],
            "questions": parse_questions(p["QUESTIONS"]),
        })

    try:
        severity = int(re.search(r"\d+", intake["SEVERITY_1_10"]).group())
        severity = max(1, min(10, severity))
    except Exception:
        severity = 5

    return {"crisis": False, "intake": intake, "intake_raw": intake_raw,
            "perspectives": perspectives, "severity": severity,
            "language": language}


# ------------------------------------------------------------- phase 2 ---

_REFRESH_SECTIONS = [
    "UPDATED_CBT", "UPDATED_PSYCHODYNAMIC", "UPDATED_HUMANISTIC",
    "UPDATED_BIOPSYCHOSOCIAL", "CONSENSUS",
]
_REFRESH_KEY_MAP = {
    "UPDATED_CBT": "cbt",
    "UPDATED_PSYCHODYNAMIC": "psychodynamic",
    "UPDATED_HUMANISTIC": "humanistic",
    "UPDATED_BIOPSYCHOSOCIAL": "biopsychosocial",
}


def refresh_hypotheses(perspectives: list[dict], answers_text: str,
                       language: str = "English") -> dict:
    """Revise each phase-1 hypothesis in light of the user's actual answers.

    One direct LLM call (no full Crew). Resilient: one retry after a short
    pause covers transient Groq hiccups/empty responses. Fail-safe: if both
    attempts fail, the original perspectives are returned with an empty
    consensus — the flow never crashes because of this step.
    """
    for attempt in (1, 2):
        try:
            return _refresh_hypotheses_inner(perspectives, answers_text, language)
        except Exception:  # noqa: BLE001
            if attempt == 1:
                time.sleep(10)
    return {"perspectives": perspectives, "consensus": ""}


def _refresh_hypotheses_inner(perspectives: list[dict], answers_text: str,
                              language: str = "English") -> dict:
    llm = A.get_llm("reasoning", TEMPERATURE_ANALYTICAL)

    originals = []
    for p in perspectives:
        originals.append(
            f"[{p.get('name', 'Perspective')}] (key: {p.get('key', '')})\n"
            f"ORIGINAL HYPOTHESIS: {p.get('hypothesis', '')}\n"
            f"ORIGINAL CONFIDENCE: {p.get('confidence', '')}\n"
            f"ORIGINAL KEY FACTORS: {p.get('key_factors', '')}"
        )

    prompt = (
        "You are revising four psychological hypotheses now that the user has "
        "answered the follow-up questions. Read the original hypotheses and the "
        "user's actual answers carefully, then update each hypothesis: keep what "
        "the answers support, drop what they contradict, and adjust the "
        "confidence honestly.\n\n"
        "ORIGINAL PERSPECTIVES:\n" + "\n\n".join(originals) + "\n\n"
        "THE USER'S ACTUAL ANSWERS:\n" + (answers_text or "(no answers given)") +
        "\n\n"
        "RULES: hypotheses only — never a definitive diagnosis or clinical label "
        "stated as fact. Humble, hedged language. Plain everyday words. Connect "
        "each update to something the user actually said.\n\n"
        "OUTPUT FORMAT — exactly these sections, in this order:\n"
        "UPDATED_CBT:\nHYPOTHESIS: <revised hypothesis>\nCONFIDENCE: <0-100%>\n"
        "KEY_FACTORS: <what the answers support>\n\n"
        "UPDATED_PSYCHODYNAMIC:\nHYPOTHESIS: ...\nCONFIDENCE: ...\nKEY_FACTORS: ...\n\n"
        "UPDATED_HUMANISTIC:\nHYPOTHESIS: ...\nCONFIDENCE: ...\nKEY_FACTORS: ...\n\n"
        "UPDATED_BIOPSYCHOSOCIAL:\nHYPOTHESIS: ...\nCONFIDENCE: ...\nKEY_FACTORS: ...\n\n"
        "CONSENSUS: <one plain-language paragraph on what all four perspectives "
        "agree on after seeing the answers>"
        + language_directive(language)
    )
    raw = llm.call([{"role": "user", "content": prompt}])
    safe = scan_output_safety(raw if isinstance(raw, str) else str(raw))
    return _parse_refresh(safe, perspectives, language)


def _repair_consensus(perspectives: list[dict], language: str = "English") -> str:
    """Last-resort consensus: if the refresh model skipped the CONSENSUS section
    (or it failed to parse), ask for just that paragraph directly. One tiny
    call; returns '' on any failure so the flow never breaks."""
    try:
        llm = A.get_llm("mechanical", TEMPERATURE_ANALYTICAL)
        bullets = "\n".join(
            f"- [{p.get('name', 'Perspective')}] {p.get('hypothesis', '')[:400]}"
            for p in perspectives
        )
        prompt = (
            "Four psychological perspectives analyzed a user and were revised "
            "after the user's answers. Write ONE plain-language paragraph on "
            "what all four perspectives agree on. No diagnosis, hedged language."
            + language_directive(language)
            + "\n\nPERSPECTIVES:\n" + bullets
        )
        out = llm.call([{"role": "user", "content": prompt}])
        safe = scan_output_safety(out if isinstance(out, str) else str(out))
        # scan appends the disclaimer after "\n---\n" — strip it back off
        return safe.split("\n---\n")[0].strip()
    except Exception:  # noqa: BLE001
        return ""


def _parse_refresh(text: str, originals: list[dict],
                   language: str = "English") -> dict:
    """Parse the refresh output; fall back to the original perspective per section."""
    sections = parse_sections(text, _REFRESH_SECTIONS)
    consensus = sections.get("CONSENSUS", "").strip()
    # scan_output_safety appends the disclaimer after the last section —
    # strip it back off so the consensus stays a clean paragraph.
    if "\n---\n" in consensus:
        consensus = consensus.split("\n---\n")[0].strip()
    if not consensus:
        # generation-side repair: the model skipped/mangled the section —
        # ask for just the consensus paragraph directly.
        consensus = _repair_consensus(originals, language)

    refreshed = []
    for p in originals:
        key = p.get("key", "")
        section_name = next(
            (s for s, k in _REFRESH_KEY_MAP.items() if k == key), None
        )
        upd = sections.get(section_name, "") if section_name else ""
        fields = parse_sections(upd, ["HYPOTHESIS", "CONFIDENCE", "KEY_FACTORS"])
        if fields["HYPOTHESIS"].strip():
            refreshed.append({
                **p,
                "hypothesis": fields["HYPOTHESIS"].strip(),
                "confidence": fields["CONFIDENCE"].strip() or p.get("confidence", ""),
                "key_factors": fields["KEY_FACTORS"].strip() or p.get("key_factors", ""),
            })
        else:
            refreshed.append(p)
    return {"perspectives": refreshed, "consensus": consensus}

def _perspectives_text(perspectives: list[dict]) -> str:
    parts = []
    for p in perspectives:
        parts.append(
            f"## {p['name']} perspective\n"
            f"HYPOTHESIS: {p['hypothesis']}\n"
            f"CONFIDENCE: {p['confidence']}\n"
            f"KEY_FACTORS: {p['key_factors']}"
        )
    return "\n\n".join(parts)


def run_phase2(intake: dict, intake_raw: str, perspectives: list[dict],
               user_answers: str, user_id: int, language: str = "English") -> dict:
    """Skeptic -> RAG researcher -> planner -> plain-language editor -> safety scan."""
    agents = {
        "skeptic": A.skeptic(),
        "researcher": A.rag_researcher(),
        "planner": A.treatment_planner(),
        "editor": A.plain_language_editor(),
    }
    tasks = build_phase2_tasks(
        agents,
        intake_text=intake_raw,
        perspectives_text=_perspectives_text(perspectives),
        user_answers=user_answers or "(user skipped the questions)",
        language=language,
    )
    crew = Crew(agents=list(agents.values()), tasks=list(tasks.values()),
                process=Process.sequential, verbose=False)
    set_search_user_id(user_id)
    try:
        _kickoff_with_retry(crew)
    finally:
        set_search_user_id(None)

    skeptic_raw = tasks["skeptic"].output.raw or ""
    research_raw = tasks["research"].output.raw or ""
    final_raw = tasks["edit"].output.raw or ""
    final_report = scan_output_safety(final_raw)

    plan = parse_sections(final_raw, [
        "SUMMARY", "ADVICE_PATHS", "SKEPTIC_ANSWERED",
        "WHEN_TO_SEEK_HELP", "REFERENCES",
    ])
    references = _parse_references(plan["REFERENCES"])
    technique_ids = _parse_technique_ids(final_raw)

    return {
        "skeptic_raw": skeptic_raw,
        "research_raw": research_raw,
        "report": final_report,
        "summary": plan["SUMMARY"],
        "advice_paths": plan["ADVICE_PATHS"],
        "advice_path_list": _parse_advice_paths(plan["ADVICE_PATHS"]) or
        _repair_advice_paths(plan["SUMMARY"], technique_ids, language),
        "skeptic_answered": plan["SKEPTIC_ANSWERED"],
        "when_to_seek_help": plan["WHEN_TO_SEEK_HELP"],
        "references": references,
        "technique_ids": technique_ids,
    }


def _parse_advice_paths(block: str) -> list[dict]:
    """Split the ADVICE_PATHS block into [{title, steps, technique_ids}].

    Line-scanner, tolerant of the many ways the planner formats path markers:
    plain ("PATH 1:"), any dash ("PATH 2 \u2013"), bullet prefixes ("- PATH 2 \u2013"),
    bold/emphasis ("**PATH 1:**", "- **Path 2:**"), any case, and repeated
    markers in the title ("PATH 1: PATH 1 \u2013 Calm\u2026"). A path marker always
    starts a new path, so paths can never be swallowed into each other.
    """
    _tids = re.compile(r"TECHNIQUE_IDS:\s*[*_]*\s*(.+)$", re.IGNORECASE)
    _bullet = re.compile(r"^([-*\u2022]|\d+[.)])\s+")
    _marker = re.compile(r"PATH\s*(\d+)\s*[:\-\u2013\u2014]\s*(.*)$",
                         re.IGNORECASE)
    _restrip = re.compile(r"^PATH\s*\d+\s*[:\-\u2013\u2014]\s*",
                          re.IGNORECASE)

    def _probe(ls: str) -> str:
        # strip leading markdown markers (bullets, quotes, bold/italic) so
        # "**PATH 1:**", "- **Path 2:**", "> PATH 3:" all match as markers
        return re.sub(r"^[\s>*_\u2022#-]+", "", ls)

    paths: list[dict] = []
    current: dict | None = None
    for raw_line in (block or "").splitlines():
        ls = raw_line.strip()
        if not ls:
            continue
        if re.fullmatch(r"[-*_]{3,}", ls):
            continue  # stray markdown separator, not a step
        m = _marker.match(_probe(ls))
        if m:
            title = _probe(m.group(2)).strip().strip("*_").strip()
            # strip a repeated marker inside the title ("PATH 1: PATH 1 \u2013 X")
            title = _restrip.sub("", title).strip("*_").strip()
            current = {"title": title, "steps": [], "technique_ids": []}
            paths.append(current)
            continue
        if current is None:
            continue  # ignore preamble before the first PATH marker
        m2 = _tids.search(_probe(ls))
        if m2:
            current["technique_ids"] = [
                t.strip().strip("`*_") for t in m2.group(1).split(",")
                if t.strip().strip("`*_")]
            continue
        step = _bullet.sub("", ls)
        if step:
            current["steps"].append(step)
    # drop a path that ended up with no usable content at all
    return [p for p in paths if p["title"] or p["steps"]]


def _repair_advice_paths(summary: str, technique_ids: list[str],
                         language: str = "English") -> list[dict]:
    """Last-resort advice paths: if the planner's ADVICE_PATHS block came back
    empty or unparseable, ask for just the three paths directly. One tiny
    call; returns [] on any failure so the flow never breaks."""
    if not summary and not technique_ids:
        return []
    try:
        llm = A.get_llm("mechanical", TEMPERATURE_ANALYTICAL)
        prompt = (
            "A psychology assessment produced the summary below"
            + (" and identified these support techniques: "
               + ", ".join(technique_ids) + "."
               if technique_ids else
               " (no specific techniques identified; use only well-established "
               "CBT, grounding, and mindfulness techniques). ")
            + "Write exactly 3 practical advice paths in this EXACT format "
            "(plain lines only \u2014 no bullets, no bold, no extra text):\n"
            "PATH 1: <short path title>\n"
            "- <concrete step>\n"
            "- <concrete step>\n"
            "TECHNIQUE_IDS: <relevant ids from the list>\n"
            "PATH 2: <short path title>\n"
            "- <concrete step>\n"
            "- <concrete step>\n"
            "TECHNIQUE_IDS: <relevant ids from the list>\n"
            "PATH 3: <short path title>\n"
            "- <concrete step>\n"
            "- <concrete step>\n"
            "TECHNIQUE_IDS: <relevant ids from the list>"
            + language_directive(language)
            + "\n\nSUMMARY:\n" + (summary or "")[:1500]
        )
        out = llm.call([{"role": "user", "content": prompt}])
        safe = scan_output_safety(out if isinstance(out, str) else str(out))
        paths = _parse_advice_paths(safe.split("\n---\n")[0])
        # keep only well-formed card ids; drop anything invented as prose
        for q in paths:
            q["technique_ids"] = [t for t in q["technique_ids"]
                                  if re.fullmatch(r"[a-z0-9_]+", t)]
        return paths
    except Exception:  # noqa: BLE001
        return []


def _parse_references(block: str) -> list[dict]:
    """Split reference lines on em-dash, en-dash, or spaced hyphen; strip
    bold/emphasis markers from fields."""
    refs = []
    for line in (block or "").splitlines():
        line = re.sub(r"^([-*•]|\d+[.)])\s+", "", line.strip()).strip("*_").strip()
        if not line:
            continue
        parts = [p.strip().strip("*_") for p in
                 re.split(r"[—–]|\s+-\s+", line, maxsplit=2)]
        refs.append({
            "id": parts[0] if len(parts) > 0 else "",
            "title": parts[1] if len(parts) > 1 else "",
            "source": parts[2] if len(parts) > 2 else "",
            "license": "", "url": "",
        })
    return refs


def _parse_technique_ids(plan_text: str) -> list[str]:
    """Collect card IDs from TECHNIQUE_IDS lines, tolerating bullets, bold,
    and case variants ("- TECHNIQUE_IDS:", "**Technique_Ids:**")."""
    ids = []
    for line in (plan_text or "").splitlines():
        ls = re.sub(r"^[\s>*_•-]+", "", line.strip())
        m = re.match(r"TECHNIQUE_IDS:\s*[*_]*\s*(.+)$", ls, re.IGNORECASE)
        if not m:
            continue
        for part in m.group(1).split(","):
            tid = part.strip().strip("`*_")
            if tid and tid not in ids:
                ids.append(tid)
    return ids


# ------------------------------------------------------- learning loop ---

def regenerate_advice_paths(summary: str, technique_ids: list[str],
                              language: str = "English") -> list[dict]:
    """Rebuild a session's three advice paths (one tiny LLM call). Used by
    the Plan tab's one-tap button for older sessions whose plan lost them."""
    return _repair_advice_paths(summary, technique_ids, language)


def learn_from_feedback(assessment_id: int, user_id: int, rating: int,
                        helpful: str, correction: str) -> list[str]:
    """Save feedback, distill 1-3 durable lessons about what helps THIS user."""
    save_feedback(assessment_id, rating, helpful, correction)
    agent = Agent(
        role="Learning Distiller",
        goal="Turn user feedback into short, durable lessons about what helps this user.",
        backstory=(
            "You compress feedback into memory. Each lesson is one sentence, "
            "specific to THIS user — e.g. which advice paths they found doable. "
            "Never invent facts. Never diagnose."
        ),
        llm=LLM(model=MODELS["mechanical"], api_key=GROQ_API_KEY,
                temperature=TEMPERATURE_ANALYTICAL),
        verbose=False,
    )
    task = Task(
        description=(
            f"User rating: {rating}/5\nWhat helped: {helpful or '(not specified)'}\n"
            f"Correction: {correction or '(none)'}\n\n"
            "Distill 1-3 durable lessons about what helps THIS user. "
            "One per line starting with '- '. If nothing learnable, output: NONE"
        ),
        expected_output="1-3 lessons, one per line starting with '- ', or NONE.",
        agent=agent,
    )
    try:
        Crew(agents=[agent], tasks=[task], process=Process.sequential,
             verbose=False).kickoff()
        raw = task.output.raw or ""
    except Exception:  # noqa: BLE001
        return []  # rating already saved above; lessons are best-effort
    saved = []
    existing = [e.lower() for e in get_learnings(user_id, 200)]
    for line in raw.splitlines():
        line = re.sub(r"^([-*•]|\d+[.)])\s+", "", line.strip())
        if len(line) > 12 and line.upper() != "NONE":
            if not any(line.lower() in e or e in line.lower() for e in existing):
                save_learning(line, user_id)
                saved.append(line)
    return saved
