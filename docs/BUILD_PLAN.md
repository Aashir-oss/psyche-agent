# Psyche — Build Plan

## What this is
**Psyche** is a multi-agent psychology assistant (Streamlit + CrewAI + free Groq
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
