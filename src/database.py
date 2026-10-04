"""SQLite storage: users, sessions, messages, assessments, learnings, feedback.

All reads/writes are scoped by user_id — no cross-user access anywhere.
"""
import json
import sqlite3
import uuid
from datetime import datetime

from .config import DB_PATH


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id INTEGER REFERENCES users(id),
                phase TEXT NOT NULL,          -- 'phase1' | 'phase2'
                intake_json TEXT,             -- structured intake (phase 1)
                perspectives_json TEXT,      -- 4 perspectives + questions (phase 1)
                answers_json TEXT,            -- user's answers to the questions
                report TEXT,                  -- plain-language final report (phase 2)
                references_json TEXT,        -- [{id, title, source, license, url}]
                result_json TEXT,            -- structured phase-2 output (paths, perspectives)
                severity INTEGER,             -- 1-10 from intake
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id INTEGER NOT NULL,
                rating INTEGER,
                helpful TEXT,
                correction TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                scale TEXT NOT NULL,        -- 'GAD7' | 'PHQ9'
                score INTEGER NOT NULL,
                answers_json TEXT,          -- list of 0-3 answers
                impairment TEXT,            -- PHQ-9 functional-impairment answer (unscored)
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS thought_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                situation TEXT,
                thought TEXT,
                emotion TEXT,
                intensity_before INTEGER,
                intensity_after INTEGER,
                evidence_for TEXT,
                evidence_against TEXT,
                balanced_thought TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        # migrations for DBs created before a column existed
        with _conn() as c:
            needed = {
                "sessions": ["user_id", "language"],
                "assessments": ["user_id", "result_json"],
                "learnings": ["user_id"],
                "users": ["display_name"],
            }
            for table, cols_needed in needed.items():
                cols = [r["name"] for r in
                        c.execute(f"PRAGMA table_info({table})").fetchall()]
                for col in cols_needed:
                    if col not in cols:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
            # backfill learnings that predate per-user scoping
            c.execute(
                "UPDATE learnings SET user_id=(SELECT MIN(id) FROM users)"
                " WHERE user_id IS NULL"
            )


# ---------------- users ----------------

def get_or_create_user(username: str, display_name: str | None = None) -> dict:
    """Identity anchor: return the existing profile or create a new one.

    username is the stable website user id (identity key, never changes);
    display_name is the friendly label shown in the UI (from the website's
    ?name= parameter) so users never see a raw id.
    """
    username = username.strip()
    display_name = (display_name or "").strip() or None
    with _conn() as c:
        r = c.execute(
            "SELECT id, username, display_name FROM users WHERE username=?", (username,)
        ).fetchone()
        if r:
            d = dict(r)
            if display_name and display_name != (d.get("display_name") or ""):
                c.execute("UPDATE users SET display_name=? WHERE id=?",
                          (display_name, d["id"]))
                d["display_name"] = display_name
            return d
        cur = c.execute(
            "INSERT INTO users (username, display_name, created_at) VALUES (?,?,?)",
            (username, display_name, _now()),
        )
        return {"id": cur.lastrowid, "username": username,
                "display_name": display_name}


def claim_orphan_sessions(user_id: int) -> None:
    """Adopt sessions created before per-user scoping existed (same device)."""
    with _conn() as c:
        c.execute("UPDATE sessions SET user_id=? WHERE user_id IS NULL", (user_id,))


# ---------------- sessions ----------------

def create_session(title: str = "New assessment", user_id: int | None = None) -> str:
    sid = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (id, user_id, title, created_at) VALUES (?,?,?,?)",
            (sid, user_id, title, _now()),
        )
    return sid


def list_sessions(user_id: int, limit: int = 5) -> list[dict]:
    """Most recent sessions for this user, newest first (sidebar history)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM sessions WHERE user_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(sid: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    return dict(r) if r else None


def rename_session(sid: str, title: str) -> None:
    with _conn() as c:
        c.execute("UPDATE sessions SET title=? WHERE id=?", (title, sid))


def set_session_language(sid: str, language: str) -> None:
    """Persist the detected chat language for a session (multilingual agents)."""
    with _conn() as c:
        c.execute("UPDATE sessions SET language=? WHERE id=?",
                  (language or "English", sid))


def get_session_language(sid: str) -> str:
    with _conn() as c:
        r = c.execute("SELECT language FROM sessions WHERE id=?",
                      (sid,)).fetchone()
    return (r["language"] if r and r["language"] else "English")


def delete_session(sid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE id=?", (sid,))


def assert_session_owner(sid: str, user_id: int) -> None:
    """Defense-in-depth: raise PermissionError if the session doesn't belong
    to this user. Call before acting on a session id that came from UI state
    (buttons, restored session state) rather than a fresh user-scoped query."""
    s = get_session(sid)
    if not s or s.get("user_id") != user_id:
        raise PermissionError("session does not belong to this user")


def prune_old_sessions(user_id: int, keep: int = 5) -> int:
    """Delete this user's sessions beyond the most recent `keep`.

    Messages/assessments/feedback cascade via foreign keys. Learnings are NOT
    deleted — raw chats rotate out, but what was learned persists.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT id FROM sessions WHERE user_id=? ORDER BY created_at DESC, rowid DESC"
            " LIMIT -1 OFFSET ?",
            (user_id, keep),
        ).fetchall()
        for r in rows:
            c.execute("DELETE FROM sessions WHERE id=?", (r["id"],))
    return len(rows)


# ---------------- messages ----------------

def add_message(sid: str, role: str, content: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (sid, role, content, _now()),
        )


def delete_last_user_message(sid: str) -> None:
    """Delete the session's most recent message, but only if it's a user
    message. Used to roll back an orphaned user message when the AI call
    that should have answered it fails."""
    with _conn() as c:
        r = c.execute(
            "SELECT id, role FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
        if r and r["role"] == "user":
            c.execute("DELETE FROM messages WHERE id=?", (r["id"],))


def get_messages(sid: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id", (sid,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------- assessments ----------------

def save_assessment(sid: str, user_id: int, phase: str, intake: dict | None = None,
                    perspectives: list | None = None, answers: dict | None = None,
                    report: str | None = None, references: list | None = None,
                    result: dict | None = None,
                    severity: int | None = None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO assessments (session_id, user_id, phase, intake_json,"
            " perspectives_json, answers_json, report, references_json, result_json,"
            " severity, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sid, user_id, phase,
             json.dumps(intake) if intake is not None else None,
             json.dumps(perspectives) if perspectives is not None else None,
             json.dumps(answers) if answers is not None else None,
             report, json.dumps(references) if references is not None else None,
             json.dumps(result) if result is not None else None,
             severity, _now()),
        )
        return cur.lastrowid


def get_latest_assessment(sid: str) -> dict | None:
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM assessments WHERE session_id=? ORDER BY id DESC LIMIT 1",
            (sid,),
        ).fetchone()
    return _decode_assessment(r)


def get_assessments(user_id: int, limit: int = 50) -> list[dict]:
    """All phase-2 assessments for the Progress tab, newest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM assessments WHERE user_id=? AND phase='phase2'"
            " ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [_decode_assessment(r) for r in rows]


def get_severity_history(user_id: int) -> list[tuple[str, int]]:
    """(date, severity) pairs for the progress graph. Same user only."""
    with _conn() as c:
        rows = c.execute(
            "SELECT created_at, severity FROM assessments"
            " WHERE user_id=? AND phase='phase2' AND severity IS NOT NULL"
            " ORDER BY id",
            (user_id,),
        ).fetchall()
    return [(r["created_at"][:10], r["severity"]) for r in rows]


def _decode_assessment(r) -> dict | None:
    if not r:
        return None
    d = dict(r)
    for key in ("intake_json", "perspectives_json", "answers_json",
                "references_json", "result_json"):
        d[key.replace("_json", "")] = json.loads(d[key]) if d.get(key) else None
    return d


# ---------------- learnings & feedback ----------------

# ---------------- check-ins (GAD-7 / PHQ-9) ----------------

def save_checkin(user_id: int, scale: str, score: int,
                 answers: list[int] | None = None,
                 impairment: str | None = None) -> int:
    """Save one screening check-in. scale is 'GAD7' or 'PHQ9'."""
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO checkins (user_id, scale, score, answers_json, impairment,"
            " created_at) VALUES (?,?,?,?,?,?)",
            (user_id, scale, int(score),
             json.dumps(answers) if answers is not None else None,
             impairment, _now()),
        )
        return cur.lastrowid


def get_checkins(user_id: int, scale: str | None = None,
                 limit: int = 50) -> list[dict]:
    """Check-ins for this user, newest first. Same user only."""
    with _conn() as c:
        if scale:
            rows = c.execute(
                "SELECT * FROM checkins WHERE user_id=? AND scale=?"
                " ORDER BY id DESC LIMIT ?",
                (user_id, scale, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM checkins WHERE user_id=?"
                " ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["answers"] = json.loads(d["answers_json"]) if d.get("answers_json") else None
        out.append(d)
    return out


def get_latest_checkin(user_id: int, scale: str) -> dict | None:
    """Most recent check-in for this user on one scale (or None)."""
    rows = get_checkins(user_id, scale=scale, limit=1)
    return rows[0] if rows else None


# ---------------- thought records (coping toolkit) ----------------

def save_thought_record(user_id: int, data: dict) -> int:
    """Save one CBT thought record from the toolkit tab."""
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO thought_records (user_id, situation, thought, emotion,"
            " intensity_before, intensity_after, evidence_for, evidence_against,"
            " balanced_thought, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (user_id, (data.get("situation") or "").strip(),
             (data.get("thought") or "").strip(),
             (data.get("emotion") or "").strip(),
             data.get("intensity_before"), data.get("intensity_after"),
             (data.get("evidence_for") or "").strip(),
             (data.get("evidence_against") or "").strip(),
             (data.get("balanced_thought") or "").strip(),
             _now()),
        )
        return cur.lastrowid


def get_thought_records(user_id: int, limit: int = 20) -> list[dict]:
    """Thought records for this user, newest first. Same user only."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM thought_records WHERE user_id=?"
            " ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------- learnings & feedback ----------------

def save_learning(text: str, user_id: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO learnings (user_id, text, created_at) VALUES (?,?,?)",
            (user_id, text.strip(), _now()),
        )


def get_learnings(user_id: int, limit: int = 20) -> list[str]:
    with _conn() as c:
        rows = c.execute(
            "SELECT text FROM learnings WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [r["text"] for r in rows]


def save_feedback(assessment_id: int, rating: int, helpful: str, correction: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO feedback (assessment_id, rating, helpful, correction, created_at)"
            " VALUES (?,?,?,?,?)",
            (assessment_id, rating, helpful or "", correction or "", _now()),
        )


def update_assessment_result(aid: int, result: dict) -> None:
    """Overwrite a saved assessment's result JSON (e.g. after regenerating
    advice paths for an older session)."""
    with _conn() as c:
        c.execute("UPDATE assessments SET result_json=? WHERE id=?",
                  (json.dumps(result), aid))


def has_feedback(assessment_id: int) -> bool:
    with _conn() as c:
        r = c.execute(
            "SELECT 1 FROM feedback WHERE assessment_id=? LIMIT 1", (assessment_id,)
        ).fetchone()
    return r is not None
