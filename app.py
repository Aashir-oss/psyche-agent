"""Your Personal Psych — AI Psychology Assistant. Two-phase multi-agent flow with safety guard."""
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src import database as db
from src import scales
from src.config import (DISCLAIMER, GROQ_API_KEY, DEMO_USERNAME,
                        MAX_SESSIONS_PER_USER, KNOWLEDGE_DIR,
                        CRISIS_RESPONSE)
from src.crew import (run_phase1, run_phase2, learn_from_feedback,
                      refresh_hypotheses, regenerate_advice_paths)
from src.ui import (inject_theme, hero, disclaimer_box, crisis_box,
                   confidence_meter, qa_progress, consensus_banner,
                   breathing_coach, assistant_avatar)
from src.graphs import (severity_trend, perspective_agreement, parse_confidence,
                        scale_trend)
from src.report import build_docx
from src.rag import parse_card

# Friendly, leak-free error text — never interpolate the raw exception (Groq
# outages return HTML pages, which must not be shown to the user).
_AI_DOWN_MSG = ("The AI service is having a rough moment — this is usually "
                "temporary. Your answers are saved, so please wait a minute "
                "and try again.")
# After a phase-2 failure (~5 LLM calls), block rapid retries so the user
# can't burn the free API quota hammering the button.
_PHASE2_COOLDOWN_S = 60


def _phase2_guard(sid: str) -> bool:
    """False (with a warning shown) when a phase-2 attempt failed recently."""
    last = st.session_state.get(f"phase2_fail_{sid}")
    if last and time.monotonic() - last < _PHASE2_COOLDOWN_S:
        wait = int(_PHASE2_COOLDOWN_S - (time.monotonic() - last)) + 1
        st.warning(f"That analysis just failed a moment ago — waiting ~{wait}s "
                   "before retrying protects the free AI quota. "
                   "Your answers are saved.")
        return False
    return True


def _phase2_failed(sid: str) -> None:
    st.session_state[f"phase2_fail_{sid}"] = time.monotonic()
    st.error(_AI_DOWN_MSG)
    st.stop()

st.set_page_config(page_title="Your Personal Psych", page_icon="🧠", layout="wide")
db.init_db()
inject_theme()

if not GROQ_API_KEY or GROQ_API_KEY == "paste_your_key_here":
    st.error(
        "GROQ_API_KEY is missing. Copy `.env.example` to `.env` and paste your free key "
        "from https://console.groq.com (API Keys → Create API Key)."
    )
    st.stop()


# ---- identity: THE seam for future auth integration ----
# No login page yet: the app runs as a local demo user. After the auth pages
# exist, they hand over the user id via st.session_state["authenticated_user_id"]
# or the ?uid= query param (see docs/BUILD_PLAN.md). Everything keys off this id.
def get_current_user() -> dict:
    handoff = st.session_state.get("authenticated_user_id") or st.query_params.get("uid")
    username = str(handoff).strip() if handoff else DEMO_USERNAME
    user = db.get_or_create_user(username or DEMO_USERNAME)
    db.claim_orphan_sessions(user["id"])
    return user


user = get_current_user()
uid = user["id"]
demo_mode = not (st.session_state.get("authenticated_user_id")
                 or st.query_params.get("uid"))

if "session_id" not in st.session_state:
    st.session_state.session_id = None


@st.cache_data
def _card_lookup() -> dict:
    lookup = {}
    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        card = parse_card(md)
        lookup[card.get("ID", md.stem)] = card
    return lookup


def _enrich_reference(ref: dict) -> dict:
    """Attach license/URL from the knowledge card to a reference row.

    Trust rule: when the technique ID matches a vetted knowledge card, the
    CARD's title/source/license always win — the LLM sometimes hallucinates
    reference text, and invented sources must never reach the report.
    The LLM's text is only used for IDs with no matching card.
    """
    raw_id = ref.get("id", "") or ""
    # strip markdown artifacts only — card IDs legitimately contain underscores
    clean_id = re.sub(r"[\\*`]", "", raw_id).strip().strip("_")
    card = _card_lookup().get(clean_id, {})
    if card:
        title = card.get("TITLE", "")
        source = card.get("SOURCE", "")
        lic = card.get("LICENSE", "")
    else:
        title = re.sub(r"[\\*`]", "", ref.get("title", "") or "")
        source = re.sub(r"[\\*`]", "", ref.get("source", "") or "")
        lic = ""
    url_m = re.search(r"https?://\S+", source)
    return {
        "id": clean_id or raw_id,
        "title": title,
        "source": re.sub(r"https?://\S+", "", source).strip(" ,;()"),
        "license": lic,
        "url": url_m.group(0) if url_m else "",
        "accessed": card.get("ACCESSED", ""),
    }


def _norm_phase2(p2: dict) -> dict:
    """Normalize fresh or DB-restored phase-2 output to one shape."""
    return {
        "summary": p2.get("summary", ""),
        "advice_path_list": p2.get("advice_path_list", []),
        "skeptic_answered": p2.get("skeptic_answered", ""),
        "when_to_seek_help": p2.get("when_to_seek_help", ""),
        "references": [_enrich_reference(r) for r in p2.get("references", [])],
        "technique_ids": p2.get("technique_ids", []),
        "perspectives": p2.get("perspectives", []),
        "consensus": p2.get("consensus", ""),
        "report": p2.get("report", ""),
    }


def _intake_text(p1: dict) -> str:
    if p1.get("intake_raw"):
        return p1["intake_raw"]
    return "\n".join(f"{k}: {v}" for k, v in (p1.get("intake") or {}).items())


def _screening_summary(uid_: int) -> list[dict]:
    """Latest check-in per scale: [{scale, score, band, date}]."""
    out = []
    for code, label in (("GAD7", "GAD-7"), ("PHQ9", "PHQ-9")):
        latest = db.get_latest_checkin(uid_, code)
        if latest:
            band, _ = (scales.gad7_band(latest["score"]) if code == "GAD7"
                       else scales.phq9_band(latest["score"]))
            out.append({"scale": label, "score": latest["score"], "band": band,
                        "date": (latest["created_at"] or "")[:10]})
    return out


# --------------------------------- one-by-one question flow -----------------
_Q_PREFIX = "❓ **"  # marks perspective-question messages in the chat history
_SKIP_LABEL = "⏭ Skipped"


def _build_queue(p1: dict) -> list[dict]:
    """Flatten the four perspectives' questions into one ordered queue."""
    queue = []
    for p in p1.get("perspectives", []):
        for q in p.get("questions", []):
            queue.append({"perspective": p.get("name", "Perspective"), "question": q})
    return queue


def _format_question(item: dict, i: int, n: int) -> str:
    return f"❓ **{item['perspective']}** — question {i + 1} of {n}\n\n{item['question']}"


def _qa_state(sid: str, p1: dict):
    """Return (queue, answered_count, answers).

    Cached in session state; on a fresh restore the progress is replayed
    from the persisted chat messages so a refresh never loses your place.
    """
    qk, ik, ak = f"qqueue_{sid}", f"qidx_{sid}", f"qans_{sid}"
    if qk not in st.session_state:
        queue = _build_queue(p1)
        qidx, qanswers = 0, []
        msgs = db.get_messages(sid)
        i = 0
        while i < len(msgs) and qidx < len(queue):
            m = msgs[i]
            if m["role"] == "assistant" and m["content"].startswith(_Q_PREFIX):
                # a question only counts as done when the user's answer follows it
                if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
                    ans = msgs[i + 1]["content"]
                    if ans == _SKIP_LABEL:
                        ans = "(skipped)"
                    qanswers.append({"perspective": queue[qidx]["perspective"],
                                     "question": queue[qidx]["question"],
                                     "answer": ans})
                    qidx += 1
                    i += 1
                else:
                    break  # asked but not answered yet — this is the current one
            i += 1
        st.session_state[qk] = queue
        st.session_state[ik] = qidx
        st.session_state[ak] = qanswers
    return st.session_state[qk], st.session_state[ik], st.session_state[ak]


def _record_answer(sid: str, answer: str) -> None:
    qk, ik, ak = f"qqueue_{sid}", f"qidx_{sid}", f"qans_{sid}"
    item = st.session_state[qk][st.session_state[ik]]
    st.session_state[ak].append({"perspective": item["perspective"],
                                 "question": item["question"],
                                 "answer": answer})
    st.session_state[ik] = st.session_state[ik] + 1


def _answers_text(qanswers: list[dict]) -> str:
    """Structured Q&A text for the phase-2 planner."""
    if not qanswers:
        return "(no questions were asked)"
    parts = []
    for qa in qanswers:
        ans = qa["answer"] or "(skipped)"
        parts.append(f"[{qa['perspective']}] Q: {qa['question']}\nA: {ans}")
    return "\n\n".join(parts)


def _finish_phase2(sid: str, uid_: int, p1: dict, qanswers: list[dict],
                   flow_key: str) -> None:
    """Run phase 2, persist everything, mark the flow done.

    Milestone 2: before the planner runs, each phase-1 hypothesis is revised
    in light of the user's actual answers (refresh_hypotheses), and the four
    perspectives' common ground is captured as a consensus paragraph.
    """
    user_answers = _answers_text(qanswers)
    # multilingual: the whole phase-2 pipeline follows the session's language
    _lang = (st.session_state.get(f"lang_{sid}")
             or db.get_session_language(sid) or "English")
    with st.spinner("All questions answered — revising the hypotheses, then "
                    "the skeptic, researcher and planner are building your plan…"):
        refreshed = refresh_hypotheses(p1.get("perspectives", []), user_answers,
                                       language=_lang)
        p2 = run_phase2(p1.get("intake", {}), _intake_text(p1),
                        refreshed["perspectives"], user_answers, uid_,
                        language=_lang)
    norm = _norm_phase2({**p2,
                         "perspectives": refreshed["perspectives"],
                         "consensus": refreshed["consensus"]})
    aid = db.save_assessment(
        sid, uid_, "phase2", intake=p1.get("intake"),
        perspectives=refreshed["perspectives"], answers={"qa": qanswers},
        report=p2["report"], references=norm["references"],
        result=norm, severity=p1.get("severity"),
    )
    st.session_state[f"phase2_{sid}"] = norm
    st.session_state[f"aid_{sid}"] = aid
    st.session_state[flow_key] = "done"
    db.add_message(sid, "assistant",
                   "✅ Your plan is ready — open the **📋 Plan** tab.")
    try:  # index own history for future similarity search (best effort)
        from src.rag import index_user_history
        index_user_history(uid_, db.get_assessments(uid_))
    except Exception:
        pass


_CARD_ID_RE = re.compile(r"\s*\(card [a-z0-9_]+\)")


def _clean_step(step: str) -> str:
    """Remove raw '(card xyz)' markers from advice steps — the technique names
    and references already carry that information for the user."""
    return _CARD_ID_RE.sub("", step or "")


def _install_tab_jump_poller() -> None:
    """Install a tiny persistent JS poller that jumps to a tab on demand.

    Why a poller instead of a one-shot click script: Streamlit reuses an
    identically-rendered components.html iframe across reruns WITHOUT
    re-executing its script — so the old "click the tab" script injected on
    button press only ever ran the first time and silently did nothing on
    later clicks. Instead, the "Open Plan tab" buttons just set
    ?tab=plan_<n> in the URL; this poller (rendered identically on every run,
    so Streamlit keeps the same iframe alive across reruns) sees the
    parameter and clicks the matching tab, then clears it.
    No race with the rerun render — works on every click.
    """
    components.html(
        """<script>
        (function () {
          function wantTab() {
            var q = "";
            try { q = window.parent.location.search || ""; } catch (e) { return null; }
            var m = q.match(/[?&]tab=([a-z]+)/i);
            return m ? m[1].toLowerCase() : null;
          }
          function clearTab() {
            try {
              var url = new URL(window.parent.location.href);
              url.searchParams.delete("tab");
              window.parent.history.replaceState(null, "", url.toString());
            } catch (e) {}
          }
          function go() {
            var want = wantTab();
            if (!want) return;
            var tabs = window.parent.document.querySelectorAll(
              'button[data-baseweb="tab"]');
            for (var i = 0; i < tabs.length; i++) {
              var label = (tabs[i].innerText || tabs[i].textContent || "").toLowerCase();
              if (label.indexOf(want) !== -1) {
                if (tabs[i].getAttribute("aria-selected") === "true") {
                  clearTab();  // already there — done
                } else {
                  tabs[i].click();
                  // param stays until a later tick confirms the switch,
                  // so a click lost mid-rerun is retried, not silently dropped
                }
                return;
              }
            }
          }
          setInterval(go, 400);
        })();
        </script>""",
        height=0,
    )


def _request_plan_tab() -> None:
    """Signal the tab-jump poller: switch to the Plan tab on this click."""
    _n = st.session_state.get("_tabjump_n", 0) + 1
    st.session_state["_tabjump_n"] = _n
    st.query_params["tab"] = f"plan_{_n}"


# Persistent tab-jump helper: rendered identically on every script run, so
# Streamlit reuses the same iframe and its JS stays alive across reruns.
_install_tab_jump_poller()


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.title("🧠 Assessments")
    if st.button("＋ New assessment", use_container_width=True):
        st.session_state.session_id = db.create_session(user_id=uid)
        db.prune_old_sessions(uid, keep=MAX_SESSIONS_PER_USER)
        st.rerun()
    st.divider()

    sessions = db.list_sessions(uid, limit=MAX_SESSIONS_PER_USER)
    if not sessions:
        st.caption("No assessments yet — start one above.")
    for s in sessions:
        c1, c2 = st.columns([5, 1])
        with c1:
            label = ("▶ " if s["id"] == st.session_state.session_id else "") + s["title"][:34]
            if st.button(label, key=f"open_{s['id']}", use_container_width=True):
                st.session_state.session_id = s["id"]
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"del_{s['id']}"):
                db.delete_session(s["id"])
                if st.session_state.session_id == s["id"]:
                    st.session_state.session_id = None
                st.rerun()
    st.divider()
    # compact, always-visible crisis contacts: Pakistani numbers + international
    # link. Deliberately NOT a dropdown — one glance away, no click needed.
    st.markdown(
        '<div class="side-sos">'
        "🆘 <b>Need help right now?</b><br>"
        "🇵🇰 Umang: <b>0311 7786264</b> · Rescue: <b>1122</b><br>"
        '🌍 <a href="https://findahelpline.org" target="_blank">findahelpline.org</a>'
        " <span style='color:#8b95ad'>(international)</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"👤 {user['username']}" + (" · demo" if demo_mode else "")
               + " · AI support, not a diagnosis.")

# ---------------------------------------------------------------- main ------
if not st.session_state.session_id:
    hero("🧠 Your Personal Psych", "Four psychological perspectives. One clear, plain-language plan.")
    st.markdown(
        '<div class="fade-up" style="text-align:center;color:#c3cbdf;'
        'max-width:640px;margin:0 auto">Tell me what\'s been on your mind — '
        "a crew of AI psychology agents will examine it from four angles, ask you "
        "thoughtful questions one at a time, research vetted techniques, and build "
        "a practical plan you can actually follow.</div>",
        unsafe_allow_html=True,
    )
    _hc1, _hc2, _hc3 = st.columns([1, 2, 1])
    with _hc2:
        if st.button("🚀 Begin your assessment", type="primary",
                      use_container_width=True, key="hero_start"):
            st.session_state.session_id = db.create_session(user_id=uid)
            db.prune_old_sessions(uid, keep=MAX_SESSIONS_PER_USER)
            st.rerun()
    st.markdown(
        '<div class="fade-up" style="text-align:center;color:#8b95ad;margin:.6rem 0 1.2rem">'
        "1️⃣ Share what's on your mind &nbsp;→&nbsp; 2️⃣ Answer a few questions "
        "&nbsp;→&nbsp; 3️⃣ Get your plan</div>",
        unsafe_allow_html=True,
    )
    _pcols = st.columns(4)
    for _col, (_e, _n, _d) in zip(_pcols, [
        ("🧠", "CBT", "Thoughts, behaviors & the loops between them."),
        ("🕰️", "Psychodynamic", "Past experiences & hidden patterns."),
        ("🌱", "Humanistic", "Your values, growth & self-compassion."),
        ("🧬", "Bio-Psycho-Social", "Body, mind & environment together."),
    ]):
        with _col:
            st.markdown(
                f'<div class="persp-card persp-mini fade-up">{_e}<br><b>{_n}</b><br>'
                f'<span style="color:#8b95ad;font-size:.85rem">{_d}</span></div>',
                unsafe_allow_html=True,
            )
    disclaimer_box(DISCLAIMER)
    st.stop()

sid = st.session_state.session_id
try:
    db.assert_session_owner(sid, uid)
except PermissionError:
    st.session_state.session_id = None
    st.rerun()
session = db.get_session(sid)
if not session:  # deleted in another tab
    st.session_state.session_id = None
    st.rerun()

# ---- restore two-phase flow state from the DB ----
_flow_key = f"flow_{sid}"
if _flow_key not in st.session_state:
    _latest = db.get_latest_assessment(sid)
    if _latest and _latest.get("phase") == "phase2" and _latest.get("result"):
        st.session_state[_flow_key] = "done"
        st.session_state[f"phase2_{sid}"] = _norm_phase2(_latest["result"])
        st.session_state[f"aid_{sid}"] = _latest["id"]
        _ans = _latest.get("answers") or {}
        st.session_state[f"qans_{sid}"] = _ans.get("qa", [])
    elif _latest and _latest.get("phase") == "phase1":
        st.session_state[_flow_key] = "awaiting_answers"
        st.session_state[f"phase1_{sid}"] = {
            "intake": _latest.get("intake"),
            "perspectives": _latest.get("perspectives") or [],
            "severity": _latest.get("severity"),
        }
    else:
        st.session_state[_flow_key] = "idle"
flow = st.session_state[_flow_key]

st.title(session["title"])
disclaimer_box(DISCLAIMER)

tab_chat, tab_persp, tab_plan, tab_toolkit, tab_checkin, tab_prog, tab_report = st.tabs(
    ["💬 Chat", "🔍 Perspectives", "📋 Plan", "🧰 Toolkit", "📊 Check-in",
     "📈 Progress", "📄 Report"]
)

# ------------------------------------------------------------ chat tab ------
with tab_chat:
    if st.session_state.get(f"crisis_{sid}"):
        crisis_box(st.session_state[f"crisis_{sid}"])
    else:
        # one-by-one flow: make sure the current question is in the history
        if flow == "awaiting_answers":
            _p1x = st.session_state.get(f"phase1_{sid}", {})
            _qx, _ix, _ = _qa_state(sid, _p1x)
            if _ix < len(_qx):
                _qt = _format_question(_qx[_ix], _ix, len(_qx))
                _hm = db.get_messages(sid)
                if not _hm or _hm[-1]["content"] != _qt:
                    db.add_message(sid, "assistant", _qt)
        for m in db.get_messages(sid):
            _av = assistant_avatar() if m["role"] == "assistant" else None
            with st.chat_message(m["role"], avatar=_av):
                st.markdown(m["content"])

    if flow == "idle" and not st.session_state.get(f"crisis_{sid}"):
        # gentle nudge: a check-in every week or so keeps the progress charts useful
        _recent_ci = db.get_checkins(uid, limit=1)
        _stale = True
        if _recent_ci:
            try:
                _last = datetime.fromisoformat(_recent_ci[0]["created_at"])
                _stale = (datetime.now() - _last) >= timedelta(days=7)
            except Exception:  # noqa: BLE001
                _stale = False
        if _stale:
            st.info("📊 Haven't checked in lately? The **📊 Check-in** tab has a quick "
                    "GAD-7 / PHQ-9 questionnaire — your scores are plotted over time "
                    "in **📈 Progress**.")
        prompt = st.chat_input("What's on your mind? Describe what's been happening…")
        if prompt:
            with st.spinner("Intake assistant and the four perspectives are working…"):
                try:
                    p1 = run_phase1(prompt, uid)
                except Exception:  # noqa: BLE001
                    st.error(_AI_DOWN_MSG)
                    st.stop()
            # record the message only now — a failed phase 1 leaves no orphan
            # user message sitting in the chat with no reply
            db.add_message(sid, "user", prompt)
            if session["title"] == "New assessment":
                db.rename_session(sid, (prompt[:45] + "…") if len(prompt) > 45 else prompt)
            # multilingual: the intake detected the user's language at zero
            # extra API cost — persist it for phase 2 and the report.
            _lang = p1.get("language") or "English"
            db.set_session_language(sid, _lang)
            st.session_state[f"lang_{sid}"] = _lang
            if p1.get("crisis"):
                db.add_message(sid, "assistant", "🆘 Shared crisis resources.")
                st.session_state[f"crisis_{sid}"] = p1["crisis_response"]
                st.rerun()
            db.save_assessment(sid, uid, "phase1", intake=p1["intake"],
                               perspectives=p1["perspectives"], severity=p1["severity"])
            st.session_state[f"phase1_{sid}"] = p1
            if not _build_queue(p1):
                if not _phase2_guard(sid):
                    st.stop()
                try:
                    _finish_phase2(sid, uid, p1, [], _flow_key)
                except Exception:  # noqa: BLE001
                    _phase2_failed(sid)
            else:
                st.session_state[_flow_key] = "awaiting_answers"
                db.add_message(
                    sid, "assistant",
                    "I've got it. The four perspectives have formed their hypotheses — "
                    "you can read them in the 🔍 **Perspectives** tab.\n\n"
                    "Now I'll ask their questions **one at a time** below. "
                    "Just answer in the chat, or skip the ones you'd rather not answer. ⏭",
                )
            st.rerun()

    elif flow == "awaiting_answers":
        p1 = st.session_state.get(f"phase1_{sid}", {})
        queue, qidx, qanswers = _qa_state(sid, p1)

        def _advance(answer: str, chat_text: str) -> None:
            db.add_message(sid, "user", chat_text)
            _record_answer(sid, answer)
            _q2, _i2, _a2 = (st.session_state[f"qqueue_{sid}"],
                             st.session_state[f"qidx_{sid}"],
                             st.session_state[f"qans_{sid}"])
            if _i2 >= len(_q2):
                if not _phase2_guard(sid):
                    st.stop()
                try:
                    _finish_phase2(sid, uid, p1, _a2, _flow_key)
                except Exception:  # noqa: BLE001
                    # roll back the just-added answer so it doesn't sit in the
                    # chat with no reply; answers themselves are preserved
                    db.delete_last_user_message(sid)
                    _phase2_failed(sid)
            st.rerun()

        if qidx < len(queue):
            st.markdown(qa_progress(qidx + 1, len(queue)), unsafe_allow_html=True)
            st.caption("Answer below, or skip any question you'd rather not answer.")
            if st.button("⏭ Skip this question", key=f"skipq_{sid}_{qidx}"):
                _advance("(skipped)", _SKIP_LABEL)
            answer = st.chat_input("Your answer…")
            if answer:
                _advance(answer, answer)
        else:
            # queue exhausted but phase 2 hasn't run (e.g. restored edge case)
            st.info("All questions are answered — build your plan when ready.")
            if st.button("🔄 Build my plan", key=f"retry_p2_{sid}"):
                if not _phase2_guard(sid):
                    st.stop()
                try:
                    _finish_phase2(sid, uid, p1, qanswers, _flow_key)
                except Exception:  # noqa: BLE001
                    _phase2_failed(sid)
                st.rerun()
    elif flow == "done":
        st.success("Assessment complete. Explore **Perspectives**, **Plan**, **Progress** "
                   "and **Report**. Start a new assessment from the sidebar for a new topic.")
        # One-tap jump to the Plan tab (signals the poller installed at
        # startup: Streamlit tabs have no URL anchors).
        if st.button("Open 📋 Plan tab →", key=f"openplan_{sid}"):
            _request_plan_tab()

# ---------------------------------------------------- perspectives tab ------
with tab_persp:
    p1 = st.session_state.get(f"phase1_{sid}")
    if not p1 or not p1.get("perspectives"):
        st.info("Run an assessment first — the four perspectives will appear here.")
    else:
        p2 = st.session_state.get(f"phase2_{sid}")
        # prefer refreshed post-answer hypotheses once the plan is built
        persps = (p2.get("perspectives") if p2 and p2.get("perspectives")
                  else p1["perspectives"])
        if p2 and p2.get("consensus"):
            st.markdown(consensus_banner(p2["consensus"]), unsafe_allow_html=True)
        for _r in range(0, len(persps), 2):
            _cols = st.columns(2)
            for _col, p in zip(_cols, persps[_r:_r + 2]):
                with _col:
                    _pct = parse_confidence(p.get("confidence", ""))
                    st.markdown(
                        '<div class="persp-card fade-up">'
                        f"<h4>🔍 {p.get('name') or 'Perspective'}</h4>"
                        f"<b>Hypothesis:</b> {p.get('hypothesis') or '—'}<br>"
                        + (f"<b>Key factors:</b> {p['key_factors']}<br>"
                           if p.get("key_factors") else "")
                        + confidence_meter(_pct)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    if p.get("questions"):
                        with st.expander("Questions asked"):
                            for q in p["questions"]:
                                st.markdown(f"- {q}")
        if p2 and p2.get("advice_path_list"):
            if st.button("Open 📋 Plan tab →", key=f"openplan2_{sid}"):
                _request_plan_tab()

# ------------------------------------------------------------ plan tab ------
with tab_plan:
    p2 = st.session_state.get(f"phase2_{sid}")
    if not p2 or not p2.get("advice_path_list"):
        st.info("Answer the perspectives' questions in the Chat tab to build your plan.")
        # older restored sessions can have a summary but no paths (the planner's
        # block came back empty) — one-tap rebuild instead of a dead end
        if p2 and p2.get("summary"):
            if st.button("🔄 Regenerate advice plan", key=f"regenplan_{sid}"):
                with st.spinner("Rebuilding your three advice paths…"):
                    try:
                        _new_paths = regenerate_advice_paths(
                            p2.get("summary") or "",
                            p2.get("technique_ids") or [],
                            language=(st.session_state.get(f"lang_{sid}")
                                      or db.get_session_language(sid)
                                      or "English"))
                    except Exception:  # noqa: BLE001
                        st.error(_AI_DOWN_MSG)
                        st.stop()
                if _new_paths:
                    p2 = {**p2, "advice_path_list": _new_paths}
                    st.session_state[f"phase2_{sid}"] = p2
                    _aid = st.session_state.get(f"aid_{sid}")
                    if _aid:
                        db.update_assessment_result(_aid, p2)
                    st.success("Plan rebuilt — your three advice paths are below.")
                    st.rerun()
                else:
                    st.error(_AI_DOWN_MSG)
    else:
        st.markdown("## Summary")
        st.markdown(p2["summary"])
        for i, path in enumerate(p2["advice_path_list"], 1):
            st.markdown(f"## Path {i}: {path['title']}")
            for step in path["steps"]:
                st.markdown(f"- {_clean_step(step)}")
            for tid in path["technique_ids"]:
                card = _card_lookup().get(tid)
                if card:
                    st.caption(f"📚 Technique: {card.get('TITLE','')} — {card.get('SOURCE','')}")
        if p2.get("skeptic_answered"):
            st.markdown("## Answering the skeptic")
            st.markdown(p2["skeptic_answered"])
        st.markdown("## When to see a professional")
        st.markdown(p2["when_to_seek_help"])

# -------------------------------------------------------- toolkit tab ------
with tab_toolkit:
    st.markdown("## 🧰 Coping toolkit")
    st.caption("Small, practical tools you can use right now — in a hard moment, "
               "not just inside reports.")

    # ---- box breathing ----
    st.markdown("### 🫁 Box breathing")
    st.caption("Follow the circle — breathe in as it grows, hold at its widest, "
               "breathe out as it shrinks, rest. One full cycle is 16 seconds.")
    st.markdown(breathing_coach(), unsafe_allow_html=True)

    # ---- 5-4-3-2-1 grounding ----
    st.markdown("### 🌿 5-4-3-2-1 grounding")
    st.caption("Name things around you right now. Slow and specific beats fast and vague.")
    _g_see = st.text_input("👀 5 things you can SEE", key="g_see")
    _g_hear = st.text_input("👂 4 things you can HEAR", key="g_hear")
    _g_touch = st.text_input("✋ 3 things you can TOUCH", key="g_touch")
    _g_smell = st.text_input("👃 2 things you can SMELL", key="g_smell")
    _g_taste = st.text_input("👅 1 thing you can TASTE", key="g_taste")
    if st.button("✅ Complete grounding", key="ground_done"):
        _filled = [x for x in (_g_see, _g_hear, _g_touch, _g_smell, _g_taste)
                   if x and x.strip()]
        if len(_filled) == 5:
            st.success("🌿 Beautiful — you just anchored yourself back in this moment. "
                       "Well done.")
        else:
            st.warning(f"You filled {len(_filled)}/5 — try to fill all five when you "
                       "can. Each one counts.")

    # ---- thought-record worksheet ----
    st.markdown("### 📝 Thought record")
    st.caption("Catch a difficult thought, examine the evidence, and find a fairer "
               "way to see it. Saved to your private records.")
    with st.form("thought_record_form"):
        _tr_situation = st.text_input("Situation — what happened?", key="tr_sit")
        _tr_thought = st.text_area("Automatic thought — what went through your mind?",
                                   key="tr_thought")
        _tr_emotion = st.text_input("Emotion (e.g. anxious, sad, angry)", key="tr_emo")
        _tr_ib = st.slider("Emotion intensity BEFORE (0–100)", 0, 100, 50, key="tr_ib")
        _tr_for = st.text_area("Evidence FOR the thought", key="tr_for")
        _tr_against = st.text_area("Evidence AGAINST the thought", key="tr_against")
        _tr_balanced = st.text_area("Balanced thought — a fairer way to see it",
                                    key="tr_bal")
        _tr_ia = st.slider("Emotion intensity AFTER (0–100)", 0, 100, 50, key="tr_ia")
        _tr_submit = st.form_submit_button("💾 Save thought record", type="primary")
    if _tr_submit:
        if not _tr_thought.strip():
            st.warning("Write down the automatic thought first — that's the heart of "
                       "the record.")
        else:
            # Enter inside a form re-submits it: refuse to save an exact
            # duplicate of the most recent record.
            _last = db.get_thought_records(uid, limit=1)
            _is_dup = bool(
                _last
                and (_last[0].get("thought") or "").strip() == _tr_thought.strip()
                and (_last[0].get("situation") or "").strip() == _tr_situation.strip()
            )
            if _is_dup:
                st.session_state["_tr_msg"] = (
                    "info", "Already saved — no duplicate created.")
            else:
                db.save_thought_record(uid, {
                    "situation": _tr_situation, "thought": _tr_thought,
                    "emotion": _tr_emotion, "intensity_before": _tr_ib,
                    "intensity_after": _tr_ia, "evidence_for": _tr_for,
                    "evidence_against": _tr_against, "balanced_thought": _tr_balanced,
                })
                st.session_state["_tr_msg"] = (
                    "success", "Saved! Noticing the thought is already half the work. 💪")
            # Clear the form so a stray Enter afterwards submits empty fields
            # (blocked by the guard above) instead of re-saving this record.
            for _k in ("tr_sit", "tr_thought", "tr_emo", "tr_ib",
                       "tr_for", "tr_against", "tr_bal", "tr_ia"):
                st.session_state.pop(_k, None)
            st.rerun()
    _tr_msg = st.session_state.pop("_tr_msg", None)
    if _tr_msg:
        (st.info if _tr_msg[0] == "info" else st.success)(_tr_msg[1])

    st.markdown("### Your past thought records")
    _trs = db.get_thought_records(uid, limit=10)
    if not _trs:
        st.caption("No thought records yet.")
    for _tr in _trs:
        _ttl = re.sub(r"[*_`#\[\]>]", "", _tr["thought"] or "Untitled")[:50]
        with st.expander(f"{(_tr['created_at'] or '')[:10]} — {_ttl}"):
            st.markdown(f"**Situation:** {_tr['situation'] or '—'}")
            st.markdown(f"**Automatic thought:** {_tr['thought'] or '—'}")
            st.markdown(f"**Emotion:** {_tr['emotion'] or '—'} "
                        f"({_tr['intensity_before']} → {_tr['intensity_after']})")
            if _tr["evidence_for"]:
                st.markdown(f"**Evidence for:** {_tr['evidence_for']}")
            if _tr["evidence_against"]:
                st.markdown(f"**Evidence against:** {_tr['evidence_against']}")
            if _tr["balanced_thought"]:
                st.markdown(f"**Balanced thought:** {_tr['balanced_thought']}")

# -------------------------------------------------------- check-in tab ------
with tab_checkin:
    st.markdown("## 📊 Symptom check-in")
    st.caption("GAD-7 and PHQ-9 are short, standard questionnaires used to track "
               "anxiety and mood symptoms over time. They're screening tools — "
               "your scores are **screening signals, not diagnoses**.")
    _scale_label = st.radio("Choose a scale", ["GAD-7", "PHQ-9"], horizontal=True,
                            key="ci_scale")
    _scale = "GAD7" if _scale_label == "GAD-7" else "PHQ9"
    _items = scales.GAD7_ITEMS if _scale == "GAD7" else scales.PHQ9_ITEMS
    st.markdown(f"**{scales.GAD7_STEM}**")
    with st.form("checkin_form"):
        _raw = []
        for _i, _item in enumerate(_items):
            _raw.append(st.radio(f"{_i + 1}. {_item}", scales.CHOICES, index=None,
                                 horizontal=True, key=f"ci_{_scale}_{_i}"))
        _impairment = None
        if _scale == "PHQ9":
            _impairment = st.radio(scales.IMPAIRMENT_QUESTION, scales.IMPAIRMENT_CHOICES,
                                   index=None, horizontal=True, key="ci_impairment")
        _submitted = st.form_submit_button("✅ Submit check-in", type="primary")

    if _submitted:
        _answers = [scales.CHOICES.index(v) if v in scales.CHOICES else None
                    for v in _raw]
        _missing_impairment = _scale == "PHQ9" and _impairment is None
        if any(a is None for a in _answers) or _missing_impairment:
            st.warning("Please answer every question before submitting.")
        else:
            _score = scales.score_scale(_answers)
            # DETERMINISTIC item-9 path FIRST — before saving, before the score.
            if scales.item9_positive(_scale, _answers):
                crisis_box(CRISIS_RESPONSE)
                st.markdown(
                    "Thank you for being honest about that last question — that takes "
                    "courage, and it matters that you said it. What you're feeling "
                    "deserves support from a real person, so please reach out using "
                    "the resources above. I'm an AI and I can't give you the help "
                    "you need in this moment."
                )
            _dup = db.get_latest_checkin(uid, _scale)
            _is_dup = (_dup and _dup.get("score") == _score
                       and (_dup.get("answers") or []) == _answers
                       and (_dup.get("impairment") or "") == (_impairment or ""))
            if _is_dup:
                st.info("That exact check-in is already saved — no duplicate created.")
            else:
                db.save_checkin(uid, _scale, _score, _answers, _impairment)
                _band, _note = (scales.gad7_band(_score) if _scale == "GAD7"
                                else scales.phq9_band(_score))
                st.success(f"Saved! Your {_scale_label} score is **{_score}** — {_band}.")
                st.caption(_note)
            # clear the form radios + rerun so a stray Enter can't resubmit
            for _k in [k for k in st.session_state.keys()
                       if k.startswith(f"ci_{_scale}_") or k == "ci_impairment"]:
                del st.session_state[_k]
            st.rerun()

    st.markdown("### Your recent check-ins")
    _recent = db.get_checkins(uid, limit=5)
    if not _recent:
        st.caption("No check-ins yet — your history will appear here.")
    for _ci in _recent:
        _b, _ = (scales.gad7_band(_ci["score"]) if _ci["scale"] == "GAD7"
                 else scales.phq9_band(_ci["score"]))
        _lbl = "GAD-7" if _ci["scale"] == "GAD7" else "PHQ-9"
        st.markdown(f"- {(_ci['created_at'] or '')[:10]} · {_lbl} · "
                    f"**{_ci['score']}** ({_b})")

# -------------------------------------------------------- progress tab ------
with tab_prog:
    hist = db.get_severity_history(uid)
    if len(hist) < 2:
        st.info("Severity trends appear here after 2+ assessments.")
    else:
        dates = [d for d, _ in hist]
        scores = [s for _, s in hist]
        path = severity_trend(dates, scores, out_name=f"severity_{uid}.png")
        st.image(path)

    # ---- GAD-7 / PHQ-9 trend charts (need 2+ check-ins per scale) ----
    for _scode, _slabel, _sbands in (
        ("GAD7", "GAD-7", scales.GAD7_GRAPH_BANDS),
        ("PHQ9", "PHQ-9", scales.PHQ9_GRAPH_BANDS),
    ):
        _cis = db.get_checkins(uid, scale=_scode, limit=50)
        if len(_cis) >= 2:
            _cis_sorted = sorted(_cis, key=lambda c: c["created_at"] or "")
            _p = scale_trend(
                [c["created_at"][:10] for c in _cis_sorted],
                [c["score"] for c in _cis_sorted],
                _slabel, _sbands,
                out_name=f"scale_{_scode}_{uid}.png",
            )
            st.markdown(f"### {_slabel} over time")
            st.image(_p)
            st.caption("Shaded bands show the standard severity ranges. "
                       "Scores are screening signals, not diagnoses.")
        elif _cis:
            st.caption(f"📊 One {_slabel} check-in so far — add one more to see your "
                       "trend chart.")
    st.markdown("### Past assessments")
    for a in db.get_assessments(uid, limit=20):
        intake = a.get("intake") or {}
        with st.expander(
            f"{(a.get('created_at') or '')[:10]} — severity "
            f"{a.get('severity') or '?'} — {(intake.get('SYMPTOMS') or '')[:60]}"
        ):
            st.markdown((a.get("report") or "")[:2000])

# ---------------------------------------------------------- report tab ------
with tab_report:
    p2 = st.session_state.get(f"phase2_{sid}")
    p1 = st.session_state.get(f"phase1_{sid}")
    if not p2:
        st.info("Your downloadable report appears here after the plan is built.")
    else:
        st.markdown("## References")
        st.caption("Every technique below comes from the vetted knowledge library — "
                   "no borrowed knowledge without a visible source.")
        for r in p2.get("references", []):
            line = f"- **{r.get('id','')}** — {r.get('title','')} — {r.get('source','')}"
            if r.get("license"):
                line += f" [{r['license']}]"
            if r.get("url"):
                line += f" ([link]({r['url']}))"
            if r.get("accessed"):
                line += f" — accessed {r['accessed']}"
            st.markdown(line)

        if p2.get("consensus"):
            st.markdown("## What the perspectives agree on")
            st.markdown(p2["consensus"])

        _screening = _screening_summary(uid)
        if _screening:
            st.markdown("## Screening scores")
            for _s in _screening:
                st.markdown(f"- **{_s['scale']}**: {_s['score']} ({_s['band']}) — "
                            f"checked {_s['date']}")
            st.caption("Screening signals, not diagnoses.")

        st.markdown("---")
        if st.button("📥 Build my .docx report", type="primary"):
            with st.spinner("Building your formatted report…"):
                gpaths = []
                hist = db.get_severity_history(uid)
                if len(hist) >= 2:
                    gpaths.append(severity_trend(
                        [d for d, _ in hist], [s for _, s in hist],
                        out_name=f"severity_{uid}.png"))
                persps = p2.get("perspectives") or (p1.get("perspectives") if p1 else [])
                if persps:
                    gpaths.append(perspective_agreement(
                        [(p.get("name") or "Perspective",
                          parse_confidence(p.get("confidence") or ""))
                         for p in persps],
                        out_name=f"agree_{sid}.png"))
                # human-readable technique names for the report
                card_lookup = _card_lookup()
                advice_paths = []
                for path in p2["advice_path_list"]:
                    advice_paths.append({
                        **path,
                        "technique_names": [
                            (card_lookup.get(t) or {}).get("TITLE")
                            or t.replace("_", " ").title()
                            for t in path.get("technique_ids", [])
                        ],
                    })
                refs = p2["references"]
                if not refs:
                    # Researcher returned nothing usable: fall back to citing
                    # the vetted cards behind the techniques actually used,
                    # so the report never shows a dangling references intro.
                    tids = dict.fromkeys(
                        t for path in p2["advice_path_list"]
                        for t in path.get("technique_ids", []))
                    refs = [_enrich_reference({"id": t, "title": "", "source": ""})
                            for t in tids]
                data = {
                    "username": user["username"],
                    "summary": p2["summary"],
                    "perspectives": [
                        {"name": p["name"], "hypothesis": p["hypothesis"],
                         "confidence": p["confidence"],
                         "key_factors": p.get("key_factors", "")}
                        for p in persps
                    ],
                    "consensus": p2.get("consensus", ""),
                    "qa_pairs": st.session_state.get(f"qans_{sid}", []),
                    "advice_paths": advice_paths,
                    "when_to_seek_help": p2["when_to_seek_help"],
                    "references": refs,
                    "screening": _screening_summary(uid),
                }
                docx_path = build_docx(data, gpaths, out_name=f"psyche_report_{sid}.docx")
            with open(docx_path, "rb") as f:
                st.download_button("⬇️ Download .docx", f,
                                   file_name=Path(docx_path).name,
                                   mime="application/vnd.openxmlformats-officedocument."
                                        "wordprocessingml.document")

        # ---- learning loop ----
        aid = st.session_state.get(f"aid_{sid}")
        if aid and not db.has_feedback(aid):
            with st.expander("🧠 Was this helpful? Teach the assistant", expanded=False):
                _rating = st.radio("Helpfulness", [1, 2, 3, 4, 5], index=4,
                                   horizontal=True, key=f"rt_{aid}")
                _helpful = st.text_input("What helped most?", key=f"he_{aid}")
                _correction = st.text_area("What should it do differently next time?",
                                           key=f"co_{aid}")
                if st.button("💾 Save feedback", key=f"fb_{aid}"):
                    with st.spinner("Learning…"):
                        try:
                            _new = learn_from_feedback(aid, uid, _rating, _helpful,
                                                       _correction)
                        except Exception:  # noqa: BLE001
                            _new = None
                            st.warning("Feedback saved — the learning step hit the "
                                       "AI service, so it will just apply next time.")
                    if _new:
                        st.success("Learned — the assistant will apply this next time:")
                        for _l in _new:
                            st.markdown(f"- {_l}")
                    else:
                        st.success("Feedback saved.")
