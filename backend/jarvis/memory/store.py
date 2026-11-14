"""SQLite persistence: conversation, tasks, checkpoints, profile facts.

All long-lived state survives restarts so multi-day tasks resume cleanly.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config as cfg

_conn: Optional[sqlite3.Connection] = None
_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,          -- 'voice' | 'console' | 'task:<id>'
    role TEXT NOT NULL,             -- user | assistant | system | tool
    content TEXT NOT NULL,
    meta TEXT,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel, created);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,           -- queued running waiting_user paused done failed cancelled
    progress INTEGER DEFAULT 0,
    plan TEXT,
    result TEXT,
    max_steps INTEGER,
    workspace TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS task_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    thought TEXT,
    kind TEXT NOT NULL,             -- act decompose ask_user finish wait
    actions TEXT,
    outcome TEXT,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_task ON task_steps(task_id, id);

CREATE TABLE IF NOT EXISTS task_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    asked REAL NOT NULL,
    answered REAL
);

CREATE TABLE IF NOT EXISTS profile_facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT,
    updated REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    tags TEXT,
    created REAL NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    db = cfg.db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            _conn = _connect()
        return _conn


def _now() -> float:
    return time.time()


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


# ------------------------------------------------------------------ messages

def add_message(channel: str, role: str, content: str,
                meta: Optional[dict] = None) -> str:
    mid = _uuid()
    with _lock:
        conn().execute(
            "INSERT INTO messages(id, channel, role, content, meta, created) "
            "VALUES(?,?,?,?,?,?)",
            (mid, channel, role, content,
             json.dumps(meta or {}), _now()))
        conn().commit()
    return mid


def recent_messages(channel: str, limit: int = 40) -> List[Dict[str, Any]]:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM messages WHERE channel=? ORDER BY created DESC LIMIT ?",
            (channel, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]


def clear_channel(channel: str) -> None:
    with _lock:
        conn().execute("DELETE FROM messages WHERE channel=?", (channel,))
        conn().commit()


# ------------------------------------------------------------------ tasks

def create_task(title: str, objective: str, workspace: str,
                max_steps: int, plan: Optional[list] = None) -> str:
    tid = _uuid()
    now = _now()
    with _lock:
        conn().execute(
            "INSERT INTO tasks(id, title, objective, status, progress, plan,"
            " max_steps, workspace, created, updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (tid, title, objective, "queued", 0,
             json.dumps(plan or []), max_steps, workspace, now, now))
        conn().commit()
    return tid


def update_task(tid: str, **fields: Any) -> None:
    cols = []
    vals: List[Any] = []
    for k, v in fields.items():
        if k in ("title", "objective", "status", "progress", "plan",
                 "result", "max_steps", "workspace"):
            cols.append(f"{k}=?")
            vals.append(json.dumps(v) if isinstance(v, (list, dict)) else v)
    if not cols:
        return
    with _lock:
        conn().execute(
            f"UPDATE tasks SET {', '.join(cols)}, updated=? WHERE id=?",
            (*vals, _now(), tid))
        conn().commit()


def get_task(tid: str) -> Optional[Dict[str, Any]]:
    with _lock:
        row = conn().execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
    return _row_to_task(row) if row else None


def all_tasks() -> List[Dict[str, Any]]:
    with _lock:
        rows = conn().execute("SELECT * FROM tasks ORDER BY created DESC").fetchall()
    return [_row_to_task(r) for r in rows]


def open_tasks() -> List[Dict[str, Any]]:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM tasks WHERE status IN ('queued','running','waiting_user','paused')"
            " ORDER BY created").fetchall()
    return [_row_to_task(r) for r in rows]


def _row_to_task(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["plan"] = json.loads(d.get("plan") or "[]")
    return d


def add_step(tid: str, thought: str, kind: str, actions: list,
             outcome: Optional[str]) -> int:
    with _lock:
        cur = conn().execute(
            "INSERT INTO task_steps(task_id, thought, kind, actions, outcome, created)"
            " VALUES(?,?,?,?,?,?)",
            (tid, thought, kind, json.dumps(actions), outcome, _now()))
        conn().commit()
        return int(cur.lastrowid or 0)


def recent_steps(tid: str, limit: int = 12) -> List[Dict[str, Any]]:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM task_steps WHERE task_id=? ORDER BY id DESC LIMIT ?",
            (tid, limit)).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        d["actions"] = json.loads(d.get("actions") or "[]")
        out.append(d)
    return out


def add_question(tid: str, question: str) -> int:
    with _lock:
        cur = conn().execute(
            "INSERT INTO task_questions(task_id, question, asked) VALUES(?,?,?)",
            (tid, question, _now()))
        conn().commit()
        return int(cur.lastrowid or 0)


def answer_question(qid: int, answer: str) -> None:
    with _lock:
        conn().execute(
            "UPDATE task_questions SET answer=?, answered=? WHERE id=?",
            (answer, _now(), qid))
        conn().commit()


def pending_questions(task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with _lock:
        if task_id:
            rows = conn().execute(
                "SELECT * FROM task_questions WHERE task_id=? AND answer IS NULL"
                " ORDER BY asked", (task_id,)).fetchall()
        else:
            rows = conn().execute(
                "SELECT * FROM task_questions WHERE answer IS NULL ORDER BY asked"
            ).fetchall()
    return [dict(r) for r in rows]


def latest_answered_question(tid: str) -> Optional[Dict[str, Any]]:
    with _lock:
        row = conn().execute(
            "SELECT * FROM task_questions WHERE task_id=? AND answer IS NOT NULL"
            " ORDER BY answered DESC LIMIT 1", (tid,)).fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------------ profile

def set_fact(key: str, value: str, source: str = "onboarding",
             confidence: float = 1.0) -> None:
    with _lock:
        conn().execute(
            "INSERT INTO profile_facts(key, value, confidence, source, updated)"
            " VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET"
            " value=excluded.value, confidence=excluded.confidence,"
            " source=excluded.source, updated=excluded.updated",
            (key, value, confidence, source, _now()))
        conn().commit()


def get_fact(key: str) -> Optional[str]:
    with _lock:
        row = conn().execute(
            "SELECT value FROM profile_facts WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def all_facts() -> Dict[str, Any]:
    with _lock:
        rows = conn().execute("SELECT * FROM profile_facts").fetchall()
    return {r["key"]: {"value": r["value"], "confidence": r["confidence"],
                       "source": r["source"], "updated": r["updated"]}
            for r in rows}


# ------------------------------------------------------------------ notes

def add_note(content: str, tags: Optional[list] = None) -> int:
    with _lock:
        cur = conn().execute(
            "INSERT INTO notes(content, tags, created) VALUES(?,?,?)",
            (content, json.dumps(tags or []), _now()))
        conn().commit()
        return int(cur.lastrowid or 0)


def recent_notes(limit: int = 20) -> List[Dict[str, Any]]:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM notes ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
