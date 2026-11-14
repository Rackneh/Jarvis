"""Background task engine backed by the Dory IBA SQL runtime.

Each task runs in parallel via an independent LM Studio / OpenAI-compatible
agent session, where EVERY step is a fresh stateless conversation built from
durable SQLite state (SELF-HANDOFF briefing + authoritative STATE JSON in
``jarvis_home/iba_runtime/control.sqlite``) while maintaining a coherent session
for the task across steps.

Questions, completions, and problems are relayed back through the bridge to the
Conscious layer + GUI + TTS voice.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config as cfg
from ..events import publish
from ..iba.engine import Runtime as IBARuntime
from ..iba.manager import generate_step_title, generate_task_title
from ..iba.provider import JarvisIBAProvider
from ..iba.sandbox import DANGEROUS_COMMAND_SUBSTRINGS, WorkspaceSandbox  # noqa: F401
from ..iba.store import Store as IBAStore, encode as iba_encode
from ..memory import store

# Compatibility constants referenced by existing tests/diagnostics
DANGEROUS_PATTERNS = list(DANGEROUS_COMMAND_SUBSTRINGS)


class TaskError(Exception):
    pass


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").lower()).strip("_")
    return s or "task"


class TaskEngine:
    OWNER = "jarvis@local"
    SESSION = "jarvis"

    def __init__(self) -> None:
        max_p = max(2, int(cfg.get("tasks.max_parallel", 2) or 2))
        self._executor = ThreadPoolExecutor(max_workers=max(4, max_p), thread_name_prefix="jarvis-iba")
        self._workers: Dict[str, Any] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopping = threading.Event()
        self._started = False
        self._bridge = None
        self._iba_store: Optional[IBAStore] = None
        self._provider = JarvisIBAProvider()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup

    def iba_store(self) -> IBAStore:
        root = Path(cfg.home()) / "iba_runtime"
        if self._iba_store is None or self._iba_store.root != root.resolve():
            self._iba_store = IBAStore(root)
        return self._iba_store

    def set_provider(self, provider: Any) -> None:
        """Override the inference provider (primarily for automated tests)."""
        self._provider = provider

    def start(self, bridge=None) -> None:
        if self._started:
            if bridge is not None:
                self._bridge = bridge
            return
        self._bridge = bridge
        self._started = True
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        self._stopping.clear()

        iba = self.iba_store()
        try:
            iba.recover(all_leases=True)
        except Exception:
            pass

        # Resume any previously queued/running tasks stored in MemoryStore
        resumed = 0
        for task in store.open_tasks():
            tstatus = task.get("status")
            if tstatus == "running":
                tstatus = "queued"
                store.update_task(task["id"], status="queued")
            if tstatus in ("queued",):
                try:
                    # Ensure corresponding IBA task is ready if it exists
                    with iba.connect(True) as db:
                        row = db.execute("SELECT status FROM task WHERE id=?", (task["id"],)).fetchone()
                        if row and row["status"] == "paused":
                            db.execute("UPDATE task SET status='ready', reason='', lease=NULL, expires=NULL WHERE id=?", (task["id"],))
                except Exception:
                    pass
                self.enqueue(task["id"])
                resumed += 1
        publish("task_engine_started", resumed=resumed, runtime="iba_sql")

    def stop(self) -> None:
        self._stopping.set()
        for ev in list(self._cancel_flags.values()):
            ev.set()

    # ------------------------------------------------------------- public API

    @staticmethod
    def _resolve_workspace(workspace: Optional[str], objective: str) -> Path:
        candidates: list[str] = []
        if workspace and str(workspace).strip():
            candidates.append(str(workspace).strip().strip('"').strip("'"))
        for m in re.finditer(r'([A-Za-z]:[\\/][^"\'<>\r\n]+)', objective or ""):
            raw = m.group(1).rstrip(".,;:!?)\"'")
            # Also try trimming trailing natural-language tokens if unquoted path has spaces
            candidates.append(raw)
            if " " in raw:
                candidates.append(raw.split(" ", 1)[0].rstrip(".,;:!?)\"'"))
        for raw in candidates:
            try:
                p = Path(raw).expanduser()
                if p.is_dir():
                    return p.resolve()
                if p.is_file():
                    return p.parent.resolve()
            except Exception:
                continue
        ws = Path(cfg.tasks_root()) / ("task_" + _slug(objective)[:32])
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def create(
        self,
        objective: str,
        title: Optional[str] = None,
        max_steps: Optional[int] = None,
        workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        objective = (objective or "").strip()
        if not objective:
            raise TaskError("empty objective")
        max_steps = int(max_steps or cfg.get("tasks.max_steps_default", 200) or 200)
        iba = self.iba_store()
        ws = self._resolve_workspace(workspace, objective)
        title = (title or "").strip() or generate_task_title(objective)
        iba_objective = (
            f"{title} — {objective}"
            if title and title.lower() not in objective.lower()
            else objective
        )

        rt = IBARuntime(iba, self._provider, ws, max_steps=max_steps)
        tid = rt.submit(self.OWNER, self.SESSION, iba_objective, allow_decompose=False)

        # Mirror in MemoryStore using the exact same tid so GUI & REST share one ID
        now = time.time()
        with store._lock:  # noqa: SLF001
            store.conn().execute(
                "INSERT OR REPLACE INTO tasks(id, title, objective, status, progress, plan,"
                " max_steps, workspace, created, updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (tid, title, objective, "queued", 0, json.dumps([]), max_steps, str(ws), now, now),
            )
            store.conn().commit()
        store.add_step(tid, "IBA SQL task session initialized.", "act", [], "queued")
        publish("task_created", task_id=tid, title=title, objective=objective, runtime="iba_sql")
        self.enqueue(tid)
        return store.get_task(tid) or {"id": tid, "title": title, "objective": objective, "status": "queued"}

    def spawn(self, goal: str, priority: int = 5, parent_task_id: Optional[str] = None) -> str:
        task = self.create(objective=goal)
        return task["id"]

    def enqueue(self, tid: str) -> None:
        with self._lock:
            existing = self._workers.get(tid)
            if existing is not None and not existing.done():
                return
            cancel_ev = threading.Event()
            self._cancel_flags[tid] = cancel_ev
            fut = self._executor.submit(self._run_iba_task_sync, tid, cancel_ev)
            self._workers[tid] = fut

    def cancel(self, tid: str) -> None:
        store.update_task(tid, status="cancelled")
        self._cancel_flags.setdefault(tid, threading.Event()).set()
        try:
            self.iba_store().control(self.OWNER, self.SESSION, tid, "cancel", reason="User cancel")
        except Exception:
            pass
        publish("task_cancelled", task_id=tid)

    def pause(self, tid: str) -> None:
        store.update_task(tid, status="paused")
        self._cancel_flags.setdefault(tid, threading.Event()).set()
        try:
            self.iba_store().control(self.OWNER, self.SESSION, tid, "pause", reason="User pause")
        except Exception:
            pass
        publish("task_paused", task_id=tid)

    def resume(self, tid: str) -> None:
        self._cancel_flags.pop(tid, None)
        try:
            iba = self.iba_store()
            with iba.connect(True) as db:
                db.execute(
                    "UPDATE task SET status='ready', reason='', lease=NULL, expires=NULL WHERE id=?",
                    (tid,),
                )
        except Exception:
            pass
        store.update_task(tid, status="queued")
        self.enqueue(tid)
        publish("task_resumed", task_id=tid)

    def deliver_answer(self, tid: str, answer: str) -> bool:
        qs = store.pending_questions(tid)
        delivered = False
        if qs:
            store.answer_question(qs[0]["id"], answer)
            delivered = True
        try:
            iba = self.iba_store()
            row = iba.task(self.OWNER, self.SESSION, tid)
            if row["status"] == "waiting":
                iba.reply(self.OWNER, self.SESSION, tid, answer)
                delivered = True
            elif not qs:
                return False
        except Exception:
            if not qs:
                return False
        publish("task_answered", task_id=tid, answer=answer)
        store.update_task(tid, status="queued")
        self._cancel_flags.pop(tid, None)
        self.enqueue(tid)
        return delivered

    def answer_task(self, tid: str, answer: str) -> bool:
        return self.deliver_answer(tid, answer)

    def snapshot(self) -> List[Dict[str, Any]]:
        out = []
        for t in store.all_tasks():
            qs = store.pending_questions(t["id"])
            steps = store.recent_steps(t["id"], limit=1)
            out.append({
                **t,
                "pending_question": qs[0]["question"] if qs else None,
                "last_step": steps[-1]["thought"] if steps else None,
            })
        return out

    def get_task_detail(self, tid: str) -> Dict[str, Any]:
        task = store.get_task(tid) or {}
        try:
            iba_steps = self.iba_store().steps(self.OWNER, self.SESSION, tid)
        except Exception:
            iba_steps = []
        return {
            "task": task,
            "steps": store.recent_steps(tid, limit=50),
            "iba_steps": iba_steps,
        }

    # -------------------------------------------------------- worker & relay

    def _bridge_call(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        bridge = self._bridge
        if bridge is None:
            return
        fn = getattr(bridge, method_name, None)
        if fn is None:
            return
        try:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(lambda: fn(*args, **kwargs))
            else:
                fn(*args, **kwargs)
        except Exception:
            pass

    def _run_iba_task_sync(self, tid: str, cancel_ev: threading.Event) -> None:
        task = store.get_task(tid) or {}
        title = task.get("title") or "Task"
        ws = Path(task.get("workspace") or (Path(cfg.tasks_root()) / f"task_{tid[:8]}"))
        ws.mkdir(parents=True, exist_ok=True)
        max_steps = int(task.get("max_steps") or 40)

        store.update_task(tid, status="running", progress=max(int(task.get("progress") or 0), 5))
        publish("task_started", task_id=tid, title=title)

        iba = self.iba_store()
        rt = IBARuntime(iba, self._provider, ws, max_steps=max_steps)

        step_counter = [0]
        thought_buf: List[str] = []
        step_thought_accum: List[str] = []
        thought_last_flush = [time.time()]
        notified_wait = [False]

        def flush_thought(force: bool = False) -> None:
            if not thought_buf:
                return
            delta = "".join(thought_buf)
            now = time.time()
            if not delta.strip():
                thought_buf.clear()
                return
            if not force and len(delta) < 110 and (now - thought_last_flush[0]) < 1.5:
                return
            thought_buf.clear()
            thought_last_flush[0] = now
            accum = "".join(step_thought_accum).strip()
            publish(
                "task_thought",
                task_id=tid,
                step=step_counter[0],
                delta=delta,
                text=accum if accum else delta.strip(),
            )

        def on_event(ev: Dict[str, Any]) -> None:
            ename = ev.get("event")
            if ename == "step_start":
                step_counter[0] += 1
                thought_buf.clear()
                step_thought_accum.clear()
                pct = min(90, 10 + step_counter[0] * 12)
                store.update_task(tid, progress=pct)
                publish(
                    "task_step_start",
                    task_id=tid,
                    title=title,
                    step_number=step_counter[0],
                    step_id=ev.get("step"),
                )
                publish("task_progress", task_id=tid, progress=pct, step=step_counter[0])

            elif ename == "thought":
                tok = ev.get("token") or ""
                if tok:
                    thought_buf.append(tok)
                    step_thought_accum.append(tok)
                    flush_thought(force=False)

            elif ename == "tool_start":
                flush_thought(force=True)
                tool = ev.get("tool") or ""
                args = ev.get("args") or {}
                step_title = generate_step_title(tool, args)
                publish(
                    "task_tool_call",
                    task_id=tid,
                    title=title,
                    step=step_counter[0],
                    tool=tool,
                    args=args,
                    step_title=step_title,
                )

            elif ename == "tool_result":
                tool = ev.get("tool") or ""
                args = ev.get("args") or {}
                status = ev.get("status") or "completed"
                ok = status == "completed"
                res = ev.get("result")
                step_title = generate_step_title(tool, args)
                preview = iba_encode(res)[:240] if res is not None else ""
                store.add_step(
                    tid,
                    step_title,
                    "act",
                    [{"tool": tool, "args": args, "ok": ok}],
                    f"{status}: {preview}",
                )
                publish(
                    "task_tool_result",
                    task_id=tid,
                    title=title,
                    step=step_counter[0],
                    tool=tool,
                    ok=ok,
                    preview=preview,
                )

            elif ename == "step_completed":
                flush_thought(force=True)
                summary = ev.get("text") or f"Step {step_counter[0]} completed"
                publish(
                    "task_step_done",
                    task_id=tid,
                    title=title,
                    step=step_counter[0],
                    summary=summary[:240],
                )

            elif ename == "task_waiting":
                flush_thought(force=True)
                q = ev.get("text") or "Need user input."
                opts = ev.get("options") or []
                notified_wait[0] = True
                store.add_question(tid, q)
                store.update_task(tid, status="waiting_user")
                publish("task_asks_user", task_id=tid, title=title, question=q, options=opts)
                self._bridge_call("on_task_question", tid, title, q, options=opts)

        try:
            # Ensure runnable state inside IBA SQLite
            with iba.connect(True) as db:
                row = db.execute("SELECT status FROM task WHERE id=?", (tid,)).fetchone()
                if row and row["status"] == "paused":
                    db.execute("UPDATE task SET status='ready', reason='', lease=NULL, expires=NULL WHERE id=?", (tid,))
                db.execute("UPDATE task SET status='ready', reason='', lease=NULL, expires=NULL WHERE parent=? AND status='paused'", (tid,))

            sys_role = (
                "You are Jarvis's autonomous IBA background task worker. "
                "Complete the user's task directly and accurately using the available tools. "
                "If asked to create or modify files on the Desktop or Documents, pass "
                "path='Desktop/<filename>' (or 'Documents/<filename>') to write_file/read_file "
                "with expected_hash='absent' for new files. When finished, return kind='finish' "
                "with a clear summary of exactly what you created or verified."
            )

            # variable_newconv=False ensures EVERY step runs as a separate stateless
            # conversation assembled from SQLite (SELF-HANDOFF + STATE JSON).
            final_task = rt.run(
                self.OWNER,
                self.SESSION,
                tid,
                system_prompt=sys_role,
                callback=on_event,
                variable_newconv=False,
                cancel_event=cancel_ev,
            )
            flush_thought(force=True)

            cur_mem = store.get_task(tid) or {}
            if cur_mem.get("status") in ("cancelled", "paused") or cancel_ev.is_set():
                return

            fstatus = final_task.get("status")
            if fstatus == "completed":
                res_body = ""
                if final_task.get("result"):
                    try:
                        res_body = iba.artifact(self.OWNER, self.SESSION, final_task["result"], 0, 8192).get("body", "")
                    except Exception:
                        res_body = ""
                if not res_body:
                    res_body = rt.response(self.OWNER, self.SESSION, final_task)
                store.add_step(tid, "Task finished.", "finish", [], res_body[:500])
                store.update_task(tid, status="done", progress=100, result=res_body)
                publish("task_progress", task_id=tid, progress=100, step=step_counter[0])
                publish("task_finished", task_id=tid, title=title, result=res_body)
                self._bridge_call("on_task_completed", tid, title, res_body)

            elif fstatus == "waiting":
                if not notified_wait[0]:
                    q = final_task.get("reason") or "The task needs clarification to proceed."
                    store.add_question(tid, q)
                    store.update_task(tid, status="waiting_user")
                    publish("task_asks_user", task_id=tid, title=title, question=q, options=[])
                    self._bridge_call("on_task_question", tid, title, q, options=[])

            else:
                child_err = None
                try:
                    with iba.connect() as db:
                        child_err = db.execute(
                            "SELECT reason FROM task WHERE parent=? AND status IN ('paused','failed') ORDER BY updated DESC LIMIT 1",
                            (tid,),
                        ).fetchone()
                except Exception:
                    pass
                reason = final_task.get("reason") or (child_err["reason"] if child_err else "") or f"Task stopped with status '{fstatus}'"
                mem_status = "paused" if fstatus == "paused" and reason.startswith("User ") else "failed"
                store.add_step(tid, f"Stopped: {reason[:200]}", "wait", [], reason[:400])
                store.update_task(tid, status=mem_status, result=reason)
                if mem_status == "failed":
                    publish("task_failed", task_id=tid, title=title, error=reason)
                    self._bridge_call("on_task_problem", tid, title, reason)

        except Exception as exc:  # noqa: BLE001
            flush_thought(force=True)
            err = str(exc)
            store.add_step(tid, f"Error: {err[:200]}", "wait", [], err[:400])
            store.update_task(tid, status="failed", result=err)
            publish("task_failed", task_id=tid, title=title, error=err)
            self._bridge_call("on_task_problem", tid, title, err)
        finally:
            self._cancel_flags.pop(tid, None)


engine = TaskEngine()
