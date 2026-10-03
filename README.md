# Your Personal Psych — Build Plan

## What this is
**Your Personal Psych** is a multi-agent psychology assistant (Streamlit + CrewAI + free Groq
models). It examines what the user shares from four psychological perspectives,
asks follow-up questions, grounds its advice in a vetted technique library,
and produces a plain-language plan + downloadable report. It offers hypotheses
only — never a diagnosis.

## Agent roster (9)

| # | Agent | Model | Job |
|---|-------|-------|-----|
| 1 | Intake Optimizer | 20b | Structures raw input → SYMPTOMS/DURATION/SEVERITY/TRIGGERS/CONTEXT. **Crisis triage first.** |
| 2 | CBT Perspective | 120b | Thoughts↔emotions↔behaviors; thinking traps |
| 3 | Psychodynamic Perspective | 120b | Past relationships echoing in the present |
| 4 | Humanistic Perspective | 120b | Inner experience, unmet needs, values |
| 5 | Bio-Psycho-Social Perspective | 120b | Body, mind, environment together |
| 6 | Skeptic | 120b | Red-teams all four perspectives |
| 7 | RAG Researcher | 120b | Searches knowledge base (2-3 angles) + own history; **must cite card IDs** |
| 8 | Treatment Planner | 120b | 3 advice paths + technique sources + when-to-seek-help |
| 9 | Plain-Language Editor | 120b→20b | Rewrites for a non-expert; keeps every fact/ID/reference |

Each perspective outputs exactly one hypothesis + confidence % + key factors +
2–3 warm questions (see `PERSPECTIVE_FORMAT` in `src/tasks.py`).

## Two-phase workflow

```
Phase 1:  user text → [crisis check] → intake → 4 perspectives (parallel-ready)
                          │                    each asks 2-3 questions
                          ▼
          App shows questions → user answers in chat (skippable)
                          │
Phase 2:                  ▼
          skeptic → RAG researcher → treatment planner → plain-language editor
                          │
                          ▼
          deterministic safety scan → report → UI tabs + DOCX + graphs
```

Perspective tasks have **zero dependencies** between them (logically simultaneous;
`async_execution` can be enabled later). On Groq's free tier we run sequentially
to protect rate limits (~30 req/min, 8k tokens/min).

## Safety design (architecture, not just a disclaimer)

1. **Deterministic crisis triage** (`check_crisis`) runs on the raw message
   *before any LLM call*. Hit → zero LLM calls, immediate helpline resources.
2. Intake agent also sets `CRISIS_FLAG` (belt-and-braces second check).
3. **Deterministic output scan** (`scan_output_safety`) blocks definitive-diagnosis
   phrasing (`DIAGNOSIS_PATTERNS` in config) and appends the disclaimer if missing.
4. Every agent backstory enforces: hypotheses only, hedged language, plain words.
5. Every report ends with **when-to-seek-help** guidance + helplines.

## RAG: two corpora

- **`knowledge`**: 12 curated, original technique cards (`src/knowledge/*.md`).
  Vetted source layer — works from session one. See `docs/REFERENCES.md`.
- **`history`**: the *same user's* past assessments only, embedded for similarity
  search ("last time you felt like this, X helped"). Never cross-user.

Stack: ChromaDB (local, persistent at `data/chroma`) + sentence-transformers
`all-MiniLM-L6-v2` on CPU. Zero API cost; first run downloads the model once.

## Data model (all per-user)

`users → sessions(user_id) → messages | assessments(user_id)` (FK cascades),
`learnings(user_id)`, `feedback → assessments`. Sidebar shows the 5 most recent
sessions (`MAX_SESSIONS_PER_USER`); creating a 6th prunes the oldest —
**learnings survive pruning**.

## Identity seam (demo mode)

No login page. `get_current_user()` in `app.py` returns the demo user, but
already checks `st.session_state["authenticated_user_id"]` and the `?uid=`
query param — the future auth handoff. The auth system hands over one stable
identifier; we map it via `get_or_create_user()` and key everything off our
local id.

## Free-tier quality levers

- 120b for reasoning, 20b for mechanical work; low temperature (0.25) for analysis.
- Labeled output formats (`INTAKE_FORMAT`, `PERSPECTIVE_FORMAT`, `PLAN_FORMAT`)
  so the UI, graphs, and DOCX parse reliably.
- Feedback → distilled learnings → injected into future runs (per user).
- "Quick mode" (2 perspectives) vs "deep mode" (4) can be added to manage latency.

## Milestone 2 — measurable benefit (2026-10-01)

Goal: make the assistant measurably beneficial instead of just chatty — patients
see their numbers move, hypotheses stay honest after answers arrive, and help
is available in the moment. All free, no new dependencies.

1. **Post-answer hypothesis refresh** (`src/crew.py::refresh_hypotheses`)
   - One direct LLM call after the one-by-one questions are answered, before the
     phase-2 planner runs. Each of the 4 phase-1 hypotheses is revised in light
     of the user's actual answers (keeps what's supported, drops what's
     contradicted, adjusts confidence honestly), plus a **CONSENSUS** paragraph:
     "what all four perspectives agree on".
   - Fail-safe: on ANY exception the originals are returned with an empty
     consensus — the flow never crashes because of this step.
   - The report shows the refreshed hypotheses and the consensus section.

2. **GAD-7 / PHQ-9 check-ins** (`src/scales.py`, 📊 Check-in tab)
   - Verbatim public-domain items; scoring is pure code (no LLM, no network).
   - Bands: GAD-7 0–4 Minimal / 5–9 Mild / 10–14 Moderate / 15–21 Severe;
     PHQ-9 0–4 Minimal / 5–9 Mild / 10–14 Moderate / 15–19 Moderately severe /
     20–27 Severe. Every band note ends with "This is a screening signal, not
     a diagnosis." — scores are screening signals, never diagnoses.
   - **Item-9 safety path**: PHQ-9 item 9 (thoughts of being better off dead or
     hurting yourself) is checked deterministically. Any score > 0 shows the
     crisis resources FIRST, then a caring message, and only then saves the
     check-in and shows the score.
   - Stored per user (`checkins` table); 📈 Progress draws a trend chart with
     shaded severity bands once a scale has 2+ check-ins. The report's
     "Screening scores" section lists the latest score/band/date per scale.
   - Gentle nudge in the Chat tab when no check-in exists in the last 7 days.

3. **Coping toolkit tab** (🧰 Toolkit)
   - Box breathing: 4 rounds of 4s-in / 4s-hold / 4s-out / 4s-hold with a live
     progress bar and countdown (stays in-session, no LLM calls).
   - 5-4-3-2-1 grounding walkthrough: 5 text inputs + encouraging completion
     message (session-only, nothing stored).
   - Thought-record worksheet: situation → automatic thought → emotion +
     intensity (0–100) → evidence for/against → balanced thought → new
     intensity, saved to `thought_records` (per user) and reviewable in
     expanders, newest first.
