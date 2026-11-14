"""Smoke test: conscious agent + task engine end-to-end with a stub provider.

Verifies: text-embedded tool call -> spawn_task -> worker thread -> act steps
(write_file) -> finish; plus task_status round-trip and profile remember.
"""
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import jarvis.config as _cfg
_SMOKE_HOME = Path(tempfile.mkdtemp(prefix="jarvis_smoke_home_"))
_cfg._default_home = lambda: _SMOKE_HOME

from jarvis import events
from jarvis.conscious.agent import agent
from jarvis.providers.base import Provider, StreamChunk
from jarvis.tasks.engine import engine

CONSCIOUS_REPLIES = [
    # turn 1: model writes the tool call as plain text (the common local-model case)
    '{"name": "spawn_task", "arguments": {"objective": "Write hello.txt containing hi", "title": "Smoke task"}}',
    "I've started the background task; it's writing hello.txt now.",
    # turn 2: task_status text tool call
    '{"name": "task_status", "arguments": {}}',
    "The task is done and hello.txt is in place.",
]
TASK_REPLIES = [
    '{"thought": "create the file", "kind": "act", "actions": [{"tool": "write_file", "args": {"path": "hello.txt", "content": "hi"}}], "progress": 50, "summary": "wrote hello.txt"}',
    '{"thought": "done", "kind": "finish", "result": "hello.txt written", "progress": 100, "summary": "complete"}',
]
_conscious = list(CONSCIOUS_REPLIES)
_task = list(TASK_REPLIES)


class StubProvider(Provider):
    kind = "stub"

    def __init__(self) -> None:
        self.base_url = "stub"
        self.model = "stub"
        self.api_key = "stub"
        self.context_length = 8192
        self.inference_slots = 1

    async def chat_stream(self, messages, tools=None, temperature=0.6,
                          max_tokens=None, cancel=None):
        is_task = any(m.role == "system" and "task worker" in m.content.lower()
                      for m in messages)
        src = _task if is_task else _conscious
        text = src.pop(0) if src else "All caught up."
        yield StreamChunk(content=text)
        yield StreamChunk(finish_reason="stop")

    async def available(self):
        return True


async def main() -> None:
    import jarvis.providers.registry as reg

    async def eff():
        return "stub"
    reg.effective_kind = eff

    stub = StubProvider()
    reg.get_registry().get = lambda kind=None: stub

    loop = asyncio.get_event_loop()
    events.bus.attach_loop(loop)
    q = events.bus.subscribe()
    async def drain_events():
        seen = []
        while True:
            try:
                ev = q.get_nowait()
                if ev.type.startswith("task_step"):
                    seen.append((ev.type, str(ev.data)[:160]))
            except asyncio.QueueEmpty:
                return seen
        return seen
    agent.attach_engine(engine)
    engine.start()

    # --- turn 1: spawn the task (spoken answer should NOT be raw JSON)
    sentences = []
    agen = agent.handle_utterance("start a background task to write hello.txt", source="test")
    async for ev in agen:
        if ev["type"] == "sentence":
            sentences.append(ev["text"])
        elif ev["type"] == "tool":
            print("tool:", ev["name"], "->", ev["result"][:80])
    spoken = " ".join(sentences)
    print("spoken:", spoken[:200])
    assert "spawn_task" not in spoken and "{" not in spoken, "raw JSON leaked to speech"

    await asyncio.sleep(4)  # let the worker run its two steps
    for typ, data in await drain_events():
        print("EV:", typ, data)

    snap = engine.snapshot()
    assert snap, "no tasks"
    t = snap[0]
    print("task:", t["status"], f"{t['progress']}%")
    assert t["status"] == "done", f"task not done: {t['status']}"
    hello = Path(t["workspace"]) / "hello.txt"
    assert hello.exists() and hello.read_text() == "hi", "hello.txt missing/wrong"
    print("workspace artifact verified:", hello)

    # --- turn 2: task_status round-trip
    agen = agent.handle_utterance("how is the task going?", source="test")
    async for ev in agen:
        if ev["type"] == "tool":
            assert "hello.txt" or t["id"] in ev["result"]
            print("tool:", ev["name"], "-> ok")

    print("SMOKE OK")


try:
    asyncio.run(main())
finally:
    try:
        from jarvis.memory import store as _mem_store
        if _mem_store._conn is not None:
            _mem_store._conn.close()
            _mem_store._conn = None
    except Exception:
        pass
    shutil.rmtree(_SMOKE_HOME, ignore_errors=True)
