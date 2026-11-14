"""End-to-end verification for:
1. ConsciousAgent thought coalescing (eliminates per-token '◌ word' spam).
2. ConsciousAgent embedded / reasoning-tail / direct-action tool call escalation to spawn_task.
3. Full IBA SQL TaskEngine (step-as-separate-conversation in control.sqlite + Desktop resolution).
4. Bidirectional Bridge relays: task_asks_user -> deliver_answer -> resume, task_completed, task_failed.
5. Optional live LM Studio probe when LM Studio is running on port 1234.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from jarvis import config as cfg  # noqa: E402
from jarvis.conscious.agent import ConsciousAgent  # noqa: E402
from jarvis.events import bus  # noqa: E402
from jarvis.iba.tools import _user_desktop  # noqa: E402
from jarvis.memory import store as mem_store  # noqa: E402
from jarvis.providers.base import StreamChunk, ToolCall  # noqa: E402
from jarvis.tasks.engine import TaskEngine  # noqa: E402


class _MockModel:
    def __init__(self, script):
        self._script = script
        self.identifier = "mock-gemma-iba"
        self.context_length = 32768

    def count(self, messages):
        return max(1, sum(len(m.get("content", "")) for m in messages) // 3)

    def predict(self, messages, max_tokens, on_thought=None, on_content=None, **_kwargs):
        return self._script(messages, on_thought=on_thought, on_content=on_content)


class _MockIBAProvider:
    def __init__(self, script):
        self._script = script

    def inference_kind(self):
        return "lmstudio"

    def inference_slots(self):
        return 2

    @contextmanager
    def session(self):
        yield _MockModel(self._script)


class _DummyEngine:
    def __init__(self):
        self.created = []

    def create(self, objective, title=None, max_steps=None):
        tid = f"t_dummy_{len(self.created) + 1}"
        rec = {"id": tid, "title": title or "Create Hello Txt", "objective": objective, "status": "queued"}
        self.created.append(rec)
        return rec


async def test_conscious_thought_coalescing_and_tool_rescue():
    ca = ConsciousAgent()
    dummy = _DummyEngine()
    ca.attach_engine(dummy)

    tokens = ["The", " file", " should", " be", " \"", "hello", ".", "txt", "\"", " on", " the", " Desktop", ".",
              " I", " will", " respond", " with", " the", " tool", " call", " JSON", " object", ".",
              ' {"name": "write_file", "arguments": {"path": "Desktop/hello.txt", "text": "Hello"}}']

    round_num = [0]

    async def fake_stream(provider, messages, tools=None, temperature=0.6, max_tokens=None, cancel=None):
        round_num[0] += 1
        if round_num[0] == 1:
            async def gen():
                for t in tokens:
                    yield StreamChunk(reasoning=t)
                yield StreamChunk(finish_reason="stop", tool_calls=[])
            return gen()
        else:
            async def gen2():
                yield StreamChunk(content="Right away, sir. ")
                yield StreamChunk(content="I have started a background task to create hello.txt on your Desktop.")
                yield StreamChunk(finish_reason="stop", tool_calls=[])
            return gen2()

    events = []
    with patch("jarvis.conscious.agent.cfg.get", side_effect=lambda k, d=None: True if k == "conscious.reasoning_enabled" else d), \
         patch("jarvis.conscious.agent.open_chat_stream", side_effect=fake_stream), \
         patch("jarvis.conscious.agent.store.recent_messages", return_value=[]), \
         patch("jarvis.conscious.agent.store.add_message"), \
         patch("jarvis.conscious.agent.store.open_tasks", return_value=[]), \
         patch("jarvis.conscious.agent.store.pending_questions", return_value=[]):
        async for ev in ca.handle_utterance("Make a file on the desktop named hello.txt", source="voice"):
            events.append(ev)

    thought_events = [e for e in events if e["type"] == "thought"]
    tool_events = [e for e in events if e["type"] == "tool"]
    sentence_events = [e for e in events if e["type"] == "sentence"]

    # When reasoning_enabled=True: must NOT emit 24 single-word thoughts and must emit at most 1 summarized thought per round
    assert len(thought_events) == 1, f"Expected 1 summarized thought for round 1, got {len(thought_events)}: {thought_events}"
    assert len(thought_events[0]["text"]) <= 165, f"Thought not capped: {thought_events[0]}"
    # Must have rescued write_file from reasoning_content and escalated to spawn_task
    assert len(tool_events) == 1 and tool_events[0]["name"] == "spawn_task", f"Tool rescue failed: {tool_events}"
    assert len(dummy.created) == 1, f"Task not spawned: {dummy.created}"
    assert any("hello.txt" in s["text"] for s in sentence_events), f"No spoken reply: {sentence_events}"

    # Now verify reasoning_enabled=False (default): zero thought events, prompt says REASONING DISABLED, tool rescue still works
    round_num[0] = 0
    events_off = []
    captured_sys = []

    async def fake_stream_off(provider, messages, tools=None, temperature=0.6, max_tokens=None, cancel=None):
        captured_sys.append(messages[0].content if messages else "")
        return await fake_stream(provider, messages, tools=tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel)

    with patch("jarvis.conscious.agent.cfg.get", side_effect=lambda k, d=None: False if k == "conscious.reasoning_enabled" else d), \
         patch("jarvis.conscious.agent.open_chat_stream", side_effect=fake_stream_off), \
         patch("jarvis.conscious.agent.store.recent_messages", return_value=[]), \
         patch("jarvis.conscious.agent.store.add_message"), \
         patch("jarvis.conscious.agent.store.open_tasks", return_value=[]), \
         patch("jarvis.conscious.agent.store.pending_questions", return_value=[]):
        async for ev in ca.handle_utterance("Make a file on the desktop named hello.txt", source="voice"):
            events_off.append(ev)

    assert [e for e in events_off if e["type"] == "thought"] == [], f"Unexpected thought events when disabled: {events_off}"
    assert any("REASONING DISABLED" in s for s in captured_sys), "Expected REASONING DISABLED in system prompt"
    assert len(dummy.created) == 2, f"Tool rescue failed when reasoning disabled: {dummy.created}"
    print("[PASS] test_conscious_thought_coalescing_and_tool_rescue")


async def test_thought_brevity_hex_cleanup_and_reasoning_delegation():
    # 1) _summarize_thought strips Current State Observation + hex IDs and caps at 2 short sentences
    verbose = (
        "Let's think carefully in response to the user's request. "
        "Current State Observation: `[c8d58524f9634fafa9eb49f8ee93f67c] Creating Hello Worlds File — running, 34%`. "
        "Check the active task progress briefly. "
        "Then summarize clearly without extra fluff. "
        "And make sure we don't talk forever."
    )
    summary = ConsciousAgent._summarize_thought(verbose)
    assert "c8d58524" not in summary
    assert "Current State Observation" not in summary
    assert len(summary) <= 160
    assert "don't talk forever" not in summary, f"Expected <=2 sentences, got: {summary}"

    # 2) _clean_spoken_text removes raw 32-char hex IDs so TTS never reads them aloud
    spoken = "The file creation task (`c8d58524f9634fafa9eb49f8ee93f67c`) is currently running and shows 34% progress."
    cleaned = ConsciousAgent._clean_spoken_text(spoken)
    assert cleaned == "The file creation task is currently running and shows 34% progress.", f"Got: {cleaned!r}"

    # 3) Long multi-sentence reasoning yields at most 1 concise thought event per round,
    #    and analytical requests with heavy reasoning auto-delegate to IBA spawn_task
    ca = ConsciousAgent()
    dummy = _DummyEngine()
    ca.attach_engine(dummy)
    round_num = [0]

    async def long_reasoning_stream(provider, messages, tools=None, temperature=0.6, max_tokens=None, cancel=None):
        round_num[0] += 1
        if round_num[0] == 1:
            async def gen():
                yield StreamChunk(reasoning="Check what the codebase does first. ")
                yield StreamChunk(reasoning="This is a very long multi-step analysis " + ("about modules and callers " * 14) + ". ")
                yield StreamChunk(reasoning="Sentence three should never spam the HUD. Sentence four either.")
                yield StreamChunk(content="", finish_reason="stop", tool_calls=[])
            return gen()
        else:
            async def gen2():
                yield StreamChunk(content="Right away, sir. I've started an IBA session to investigate that.", finish_reason="stop")
            return gen2()

    events = []
    with patch("jarvis.conscious.agent.cfg.get", side_effect=lambda k, d=None: True if k == "conscious.reasoning_enabled" else d), \
         patch("jarvis.conscious.agent.open_chat_stream", side_effect=long_reasoning_stream), \
         patch("jarvis.conscious.agent.store.recent_messages", return_value=[]), \
         patch("jarvis.conscious.agent.store.add_message"), \
         patch("jarvis.conscious.agent.store.open_tasks", return_value=[]), \
         patch("jarvis.conscious.agent.store.pending_questions", return_value=[]):
        async for ev in ca.handle_utterance("Please analyze and debug why the pipeline stalls", source="voice"):
            events.append(ev)

    thought_events = [e for e in events if e["type"] == "thought"]
    tool_events = [e for e in events if e["type"] == "tool"]
    assert len(thought_events) == 1, f"Expected 1 capped thought event, got {len(thought_events)}: {thought_events}"
    assert len(thought_events[0]["text"]) <= 160
    assert len(tool_events) == 1 and tool_events[0]["name"] == "spawn_task", f"Expected IBA delegation: {tool_events}"
    print("[PASS] test_thought_brevity_hex_cleanup_and_reasoning_delegation")


class _RecordingBridge:
    def __init__(self):
        self.questions = []
        self.completions = []
        self.problems = []
        self.done_ev = threading.Event()
        self.wait_ev = threading.Event()

    def on_task_question(self, task_id, title, question, options=None):
        self.questions.append((task_id, title, question, options))
        self.wait_ev.set()

    def on_task_completed(self, task_id, title, result):
        self.completions.append((task_id, title, result))
        self.done_ev.set()

    def on_task_problem(self, task_id, title, reason):
        self.problems.append((task_id, title, reason))
        self.done_ev.set()


def test_iba_sql_desktop_file_and_wait_resume():
    tmp_home = Path(tempfile.mkdtemp(prefix="jarvis_iba_test_"))
    desktop = _user_desktop()
    test_file = desktop / "_jarvis_iba_test_hello.txt"
    if test_file.exists():
        test_file.unlink()

    orig_conn = mem_store._conn
    mem_store._conn = None
    packets_seen = []

    try:
        # Redirect cfg.home(), cfg.tasks_root(), and cfg.db_path() to temp dir cleanly
        with patch("jarvis.tasks.engine.cfg.home", return_value=tmp_home), \
             patch("jarvis.tasks.engine.cfg.tasks_root", return_value=tmp_home / "tasks"), \
             patch("jarvis.memory.store.cfg.db_path", return_value=tmp_home / "jarvis.db"):

            # 1) Create Desktop file using 2 separate stateless IBA conversations
            def desktop_script(messages, on_thought=None, on_content=None):
                packets_seen.append(messages)
                assert len(messages) == 3, f"IBA step must receive 3 messages [sys, briefing, state], got {len(messages)}"
                assert "SELF-HANDOFF" in messages[1]["content"]
                state = json.loads(messages[2]["content"])
                if not state.get("recent_actions"):
                    if on_thought:
                        on_thought("Planning Desktop file creation.")
                    return {
                        "content": json.dumps({
                            "kind": "action",
                            "text": "Create _jarvis_iba_test_hello.txt on Desktop",
                            "tool": "write_file",
                            "tool_arguments": {
                                "path": "Desktop/_jarvis_iba_test_hello.txt",
                                "text": "Hello from Jarvis IBA SQL!\n",
                            },
                            "children": [], "evidence": [], "coverage": [],
                        }),
                        "reasoning": "Planning Desktop file creation.",
                        "stop_reason": "stop",
                        "output_tokens": 35,
                    }
                else:
                    return {
                        "content": json.dumps({
                            "kind": "finish",
                            "text": "Created _jarvis_iba_test_hello.txt on the Desktop.",
                            "coverage": [], "tool": "", "tool_arguments": {}, "children": [], "evidence": [],
                        }),
                        "reasoning": "File written cleanly.",
                        "stop_reason": "stop",
                        "output_tokens": 25,
                    }

            eng = TaskEngine()
            eng.set_provider(_MockIBAProvider(desktop_script))
            bridge = _RecordingBridge()
            eng._bridge = bridge
            eng._started = True

            t = eng.create(objective="Make a file on desktop named _jarvis_iba_test_hello.txt")
            tid = t["id"]
            assert bridge.done_ev.wait(timeout=15), "IBA task did not finish within 15s"
            assert test_file.exists(), f"Expected {test_file} to exist!"
            assert test_file.read_text(encoding="utf-8") == "Hello from Jarvis IBA SQL!\n"
            assert len(bridge.completions) == 1, f"Expected completion relay, got {bridge.completions}"
            assert len(packets_seen) == 2, f"Expected 2 stateless step packets, got {len(packets_seen)}"
            assert "WHAT PREVIOUS STEPS DID" in packets_seen[1][1]["content"]

            # Inspect IBA SQLite tables directly
            iba = eng.iba_store()
            assert iba.path.name == "control.sqlite"
            steps = iba.steps(eng.OWNER, eng.SESSION, tid)
            assert len(steps) == 2, f"Expected 2 durable IBA steps in SQLite, got {len(steps)}"
            assert steps[0]["action_name"] == "write_file" and steps[0]["action_state"] == "completed"
            task_row = iba.task(eng.OWNER, eng.SESSION, tid)
            assert task_row["status"] == "completed"
            print("[PASS] test_iba_sql_desktop_file (2 separate conversations recorded in control.sqlite)")

            # 2) Test IBA 'wait' -> bridge question relay -> deliver_answer -> finish
            def wait_then_finish_script(messages, on_thought=None, on_content=None):
                state = json.loads(messages[2]["content"])
                if not state.get("user_replies"):
                    return {
                        "content": json.dumps({
                            "kind": "wait",
                            "text": "What text should go inside the file?",
                            "options": ["Hello World", "Jarvis Status"],
                            "tool": "", "tool_arguments": {}, "children": [], "evidence": [], "coverage": [],
                        }),
                        "reasoning": "Need text from user.",
                        "stop_reason": "stop",
                        "output_tokens": 20,
                    }
                return {
                    "content": json.dumps({
                        "kind": "finish",
                        "text": "Received user reply: " + state["user_replies"][0].get("body", ""),
                        "coverage": [], "tool": "", "tool_arguments": {}, "children": [], "evidence": [],
                    }),
                    "reasoning": "Acting on user reply.",
                    "stop_reason": "stop",
                    "output_tokens": 25,
                }

            bridge2 = _RecordingBridge()
            eng._bridge = bridge2
            eng.set_provider(_MockIBAProvider(wait_then_finish_script))
            t2 = eng.create(objective="Ask user before finishing")
            tid2 = t2["id"]
            assert bridge2.wait_ev.wait(timeout=15), "Task did not enter waiting state"
            assert len(bridge2.questions) == 1
            assert bridge2.questions[0][2] == "What text should go inside the file?"

            ok = eng.deliver_answer(tid2, "Hello World")
            assert ok, "deliver_answer returned False"
            assert bridge2.done_ev.wait(timeout=15), "Resumed task did not complete after answer"
            assert len(bridge2.completions) == 1
            assert "Hello World" in bridge2.completions[0][2]
            print("[PASS] test_iba_wait_and_answer_resume")

    finally:
        try:
            if mem_store._conn is not None:
                mem_store._conn.close()
        except Exception:
            pass
        mem_store._conn = orig_conn
        if test_file.exists():
            try:
                test_file.unlink()
            except Exception:
                pass


def test_spawn_task_context_and_iba_file_robustness():
    from jarvis.iba.tools import Tools
    from jarvis.iba.store import Store as IBAStore
    from jarvis.iba.engine import Runtime as IBARuntime

    # 1) _normalize_tool_call preserves title & user_text inside spawn_task objective
    name, args = ConsciousAgent._normalize_tool_call(
        "spawn_task",
        {"title": "Creating Hello Worlds File", "objective": "Write content into the specified file path."},
        user_text='Create a file named "Hello Worlds" and put it on the desktop.',
    )
    assert name == "spawn_task"
    assert "Creating Hello Worlds File" in args["objective"]
    assert "Hello Worlds" in args["objective"]
    assert "desktop" in args["objective"].lower()

    tmp_root = Path(tempfile.mkdtemp(prefix="jarvis_robust_"))
    ws = tmp_root / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    iba = IBAStore(tmp_root / "iba_rt")
    desktop = _user_desktop()
    desk_probe = desktop / "_jarvis_robust_probe.txt"
    missing_name = "_jarvis_definitely_missing_12345.txt"
    missing_ws = ws / missing_name
    if missing_ws.exists():
        missing_ws.unlink()

    try:
        # 2) Tools.read_file falls back to Desktop when relative path isn't in empty workspace
        desk_probe.write_text("desktop hello\n", encoding="utf-8")
        tools = Tools(iba, "u", "s", ws)
        res = tools.execute("read_file", {"path": "_jarvis_robust_probe.txt"})
        assert "desktop hello" in res["text"], f"Desktop fallback failed: {res}"
        assert res["resolved_path"] == "Desktop/_jarvis_robust_probe.txt", f"Got resolved_path={res.get('resolved_path')}"

        # 3) Missing file anywhere gives explicit write_file guidance and no-retry warning
        try:
            tools.execute("read_file", {"path": missing_name})
            assert False, "Expected ValueError on missing file"
        except ValueError as exc:
            msg = str(exc)
            assert "Do NOT call read_file again" in msg, msg
            assert "write_file" in msg, msg

        # 4) write_file creates a new file even if stale expected_hash was supplied,
        #    and overwrites an existing small file without requiring a fresh hash
        w1 = tools.execute("write_file", {
            "path": missing_name,
            "text": "v1",
            "expected_hash": "0" * 64,
        })
        assert missing_ws.read_text(encoding="utf-8") == "v1"
        assert w1["bytes"] == 2

        w2 = tools.execute("write_file", {
            "path": missing_name,
            "text": "v2",
        })
        assert missing_ws.read_text(encoding="utf-8") == "v2"
        assert w2["hash"] != w1["hash"]

        # 4b) copy_path, move_path, and run_shell file copy emulation
        sub_dir = ws / "_test_copy_src"
        sub_dir.mkdir(exist_ok=True)
        (sub_dir / "asset1.bin").write_bytes(b"\x00\x01\x02\x03")
        (sub_dir / "sub").mkdir(exist_ok=True)
        (sub_dir / "sub" / "asset2.bin").write_bytes(b"\x04\x05\x06")

        # First-class copy_path (tree)
        c_res = tools.execute("copy_path", {
            "source": str(sub_dir),
            "destination": str(ws / "_test_copy_dst"),
        })
        assert c_res["status"] == "completed" and c_res["files_copied"] == 2
        assert (ws / "_test_copy_dst" / "asset1.bin").exists()
        assert (ws / "_test_copy_dst" / "sub" / "asset2.bin").exists()

        # run_shell Unix cp -r emulation
        sh_res = tools.execute("run_shell", {
            "command": f'cp -r "{sub_dir}" "{ws / "_test_cp_dst"}"',
            "cwd": str(ws),
        })
        assert sh_res["status"] == "completed" and sh_res["exit_code"] == 0
        assert (ws / "_test_cp_dst" / "sub" / "asset2.bin").exists()

        # run_shell Windows copy /E emulation
        sh_res2 = tools.execute("run_shell", {
            "command": f'copy /E "{sub_dir}" "{ws / "_test_win_copy_dst"}"',
            "cwd": str(ws),
        })
        assert sh_res2["status"] == "completed" and sh_res2["exit_code"] == 0
        assert (ws / "_test_win_copy_dst" / "sub" / "asset2.bin").exists()

        # 5) IBARuntime immediately rejects a second read_file on the same missing path
        #    even with changed line/limit args, and surfaces last_error in the next SELF-HANDOFF
        packets = []
        step_i = [0]

        def retry_read_script(messages, on_thought=None, on_content=None):
            packets.append(messages)
            step_i[0] += 1
            if step_i[0] == 1:
                return {
                    "content": json.dumps({
                        "kind": "action",
                        "text": "Read missing file",
                        "tool": "read_file",
                        "tool_arguments": {"path": "_missing_in_runtime.txt"},
                    }),
                    "reasoning": "Try reading missing file.",
                    "stop_reason": "stop",
                    "output_tokens": 20,
                }
            elif step_i[0] == 2:
                return {
                    "content": json.dumps({
                        "kind": "action",
                        "text": "Retry missing file with line and limit",
                        "tool": "read_file",
                        "tool_arguments": {"path": "_missing_in_runtime.txt", "line": 1, "limit": 80},
                    }),
                    "reasoning": "Retry with line and limit.",
                    "stop_reason": "stop",
                    "output_tokens": 20,
                }
            else:
                return {
                    "content": json.dumps({
                        "kind": "action",
                        "text": "Write the file instead",
                        "tool": "write_file",
                        "tool_arguments": {"path": "_missing_in_runtime.txt", "text": "created"},
                    }),
                    "reasoning": "Create the missing file instead.",
                    "stop_reason": "stop",
                    "output_tokens": 20,
                }

        rt = IBARuntime(iba, _MockIBAProvider(retry_read_script), ws, max_steps=10)
        tid = rt.submit("u", "s", "Create _missing_in_runtime.txt")
        # Step 1 (fails) + Step 2 (rejected immediately by _accept; correction 1 re-runs step 3 which writes)
        # Wait: make script finish on step 4 so runtime terminates cleanly
        orig_script = retry_read_script

        def bounded_script(messages, on_thought=None, on_content=None):
            if step_i[0] >= 3:
                packets.append(messages)
                return {
                    "content": json.dumps({
                        "kind": "finish",
                        "text": "Done.",
                        "coverage": [], "tool": "", "tool_arguments": {}, "children": [], "evidence": [],
                    }),
                    "reasoning": "Done.",
                    "stop_reason": "stop",
                    "output_tokens": 15,
                }
            return orig_script(messages, on_thought=on_thought, on_content=on_content)

        rt.provider = _MockIBAProvider(bounded_script)
        final_task = rt.run("u", "s", tid)
        assert final_task["status"] == "completed", f"Task failed: {final_task}"
        assert (ws / "_missing_in_runtime.txt").read_text(encoding="utf-8") == "created"
        # Packet at step 3 (after step 2 OutcomeRejected) must carry last_error in SELF-HANDOFF briefing
        assert len(packets) >= 3
        briefing_after_reject = packets[2][1]["content"]
        assert "do not retry read_file" in briefing_after_reject.lower(), f"Missing last_error in briefing:\n{briefing_after_reject}"
        print("[PASS] test_spawn_task_context_and_iba_file_robustness")
    finally:
        if desk_probe.exists():
            try:
                desk_probe.unlink()
            except Exception:
                pass


def test_live_lmstudio_probe_if_up():
    from jarvis.iba.provider import JarvisIBAProvider
    prov = JarvisIBAProvider()
    if prov._active_kind() != "lmstudio":
        print("[SKIP] live LM Studio probe (LM Studio not active)")
        return
    try:
        with prov.session() as model:
            assert prov.inference_slots() >= 2
            c = model.count([{"role": "user", "content": "ping"}])
            assert c >= 1
            print(f"[PASS] live LM Studio IBA probe (model={model.identifier}, ctx={model.context_length}, slots={prov.inference_slots()})")
    except Exception as exc:
        print(f"[SKIP] live LM Studio IBA probe ({exc})")


def test_external_workspace_agents_md_discovery_rea_and_binary_probe():
    import struct
    from pathlib import Path
    from jarvis.iba import mcp_bridge
    from jarvis.iba.engine import Runtime as IBARuntime, _compose_briefing, OutcomeRejected
    from jarvis.iba.store import Store as IBAStore
    from jarvis.iba.tools import Tools
    from jarvis.tasks.engine import TaskEngine

    waspo_ws = Path(r"D:\MGS3WaspoTool")
    # 1. Workspace auto-detection + AGENTS.md briefing injection
    resolved = TaskEngine._resolve_workspace(None, "Inspect D:\\MGS3WaspoTool\\bin\\Gcx.exe imports and MIPS tools")
    if waspo_ws.is_dir():
        assert resolved == (waspo_ws / "bin").resolve() or resolved == waspo_ws.resolve()
        ws_resolved = TaskEngine._resolve_workspace(None, "Work in D:\\MGS3WaspoTool on Waspo")
        assert ws_resolved == waspo_ws.resolve(), f"Expected D:\\MGS3WaspoTool, got {ws_resolved}"
        mem = mcp_bridge.extract_project_memory(waspo_ws)
        assert "MGS3" in mem and ("MIPS:LE:64:64-32addr" in mem or "Checkpoint" in mem), mem[:300]
        brief = _compose_briefing({"generation": 1}, "Check Waspo", {"allowed_kinds": ["action"]}, [], workspace=waspo_ws)
        assert "PROJECT MEMORY (AGENTS.md):" in brief

    # 2. Cross-drive path resolution + synthetic MIPS binary_probe (always offline/hermetic)
    tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_probe_test_"))
    try:
        target_va = 0x00123456
        base_va = 0x00100000
        # Build little-endian MIPS instructions referencing 0x00123456:
        # offset 0: lui $v0, 0x0012    -> (0x0F << 26) | (2 << 16) | 0x0012
        # offset 4: ori $v0, $v0, 0x3456 -> (0x0D << 26) | (2 << 21) | (2 << 16) | 0x3456
        # offset 8: jal 0x00123456     -> (0x03 << 26) | ((target_va >> 2) & 0x03FFFFFF)
        # offset 12: raw 32-bit ptr    -> 0x00123456
        # offset 16: ASCII string "WASPO_TEST\x00"
        blob = bytearray()
        blob += struct.pack("<I", (0x0F << 26) | (2 << 16) | 0x0012)
        blob += struct.pack("<I", (0x0D << 26) | (2 << 21) | (2 << 16) | 0x3456)
        blob += struct.pack("<I", (0x03 << 26) | ((target_va >> 2) & 0x03FFFFFF))
        blob += struct.pack("<I", target_va)
        blob += b"WASPO_TEST\x00\x00"
        bin_file = tmp_dir / "sample_mips.bin"
        bin_file.write_bytes(bytes(blob))

        store = IBAStore(str(tmp_dir / "iba.sqlite"))
        tools = Tools(store, "u", "s", waspo_ws if waspo_ws.is_dir() else tmp_dir)

        xrefs = tools.execute("binary_probe", {
            "path": str(bin_file),
            "mode": "mips_xrefs",
            "target_addr": hex(target_va),
            "base_addr": hex(base_va),
        })
        assert hex(base_va + 12) in xrefs["pointer_hits"], xrefs
        assert hex(base_va + 8) in xrefs["jal_hits"], xrefs
        assert any(h["lui_va"] == hex(base_va) for h in xrefs["lui_pair_hits"]), xrefs

        strings_res = tools.execute("binary_probe", {"path": str(bin_file), "mode": "strings", "pattern": "WASPO"})
        assert any(h["text"] == "WASPO_TEST" for h in strings_res["hits"]), strings_res

        # 3. Tool discovery & inspection (workspace + skill + live REA MCP if installed)
        if waspo_ws.is_dir():
            ws_disc = tools.execute("discover_tools", {"query": "mgs3_probe", "source": "workspace"})
            assert any("mgs3_probe" in t["name"] for t in ws_disc["tools"]), ws_disc
            ws_insp = tools.execute("inspect_tool", {"name": "tools/mgs3_probe.py"})
            assert "address_references" in ws_insp.get("preview", ""), ws_insp

        sk_disc = tools.execute("discover_tools", {"query": "reverse-engineer", "source": "skills"})
        if sk_disc["tools"]:
            sk_insp = tools.execute("inspect_tool", {"name": sk_disc["tools"][0]["name"]})
            assert "content" in sk_insp and len(sk_insp["content"]) > 100

        rea_cfg = mcp_bridge.load_rea_server_config()
        rea_script = (rea_cfg.get("args") or [""])[0]
        if Path(rea_cfg.get("command") or "").exists() and Path(rea_script).exists():
            rea_disc = tools.execute("discover_tools", {"query": "inspect_pe_imports", "source": "rea", "limit": 10})
            assert any(t["name"] == "inspect_pe_imports" for t in rea_disc["tools"]), rea_disc
            rea_insp = tools.execute("inspect_tool", {"name": "inspect_pe_imports"})
            assert "inputSchema" in rea_insp and "path" in (rea_insp["inputSchema"].get("properties") or {}), rea_insp

            # Outcome schema validation must reject missing required args on direct MCP tools
            bad_rejected = False
            try:
                IBARuntime._outcome(json.dumps({
                    "kind": "action",
                    "text": "Missing path check",
                    "tool": "inspect_pe_imports",
                    "tool_arguments": {},
                }))
            except OutcomeRejected as rej:
                bad_rejected = "path" in str(rej).lower()
            assert bad_rejected, "Expected _outcome to reject missing required MCP tool arg"

            gcx = waspo_ws / "bin" / "Gcx.exe"
            if gcx.is_file():
                # Both explicit `rea` wrapper and direct MCP tool invocation work
                direct_res = tools.execute("inspect_pe_imports", {"path": str(gcx)})
                assert direct_res.get("ok") is True and "evidence_id" in str(direct_res), direct_res
                wrapped_res = tools.execute("rea", {"tool": "inspect_pe_imports", "arguments": {"path": str(gcx)}})
                assert wrapped_res.get("ok") is True, wrapped_res

        print("[PASS] test_external_workspace_agents_md_discovery_rea_and_binary_probe")
    finally:
        mcp_bridge.get_mcp_bridge().close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def test_retrospective_task_query_redirects_hallucinated_spawn_to_status():
    from jarvis.providers.base import ToolCall
    completed_tasks = [
        {
            "id": "t_1",
            "title": "Writing Hello World File",
            "objective": "Write hello world into hello.txt on the Desktop",
            "status": "done",
            "progress": 100,
            "result": "Wrote 'hello world' into Desktop/hello.txt.",
            "pending_question": None,
            "last_step": "done",
        },
        {
            "id": "t_2",
            "title": "Inspecting Gcx Imports",
            "objective": "Inspect PE imports of Gcx.exe",
            "status": "done",
            "progress": 100,
            "result": "Summarized imported DLLs for Gcx.exe.",
            "pending_question": None,
            "last_step": "done",
        },
    ]

    class _MockStatusEngine(_DummyEngine):
        def snapshot(self):
            return list(completed_tasks)

    ca = ConsciousAgent()
    dummy = _MockStatusEngine()
    ca.attach_engine(dummy)

    seen_tools_per_round = []
    seen_sys_prompts = []

    async def fake_stream(provider, messages, tools=None, temperature=0.6, max_tokens=None, cancel=None):
        seen_tools_per_round.append(tools is not None)
        if messages and messages[0].role == "system":
            seen_sys_prompts.append(messages[0].content)
        idx = len(seen_tools_per_round)
        if idx == 1:
            # Model mistakenly tries to re-spawn the completed hello.txt task from history
            async def gen1():
                yield StreamChunk(
                    tool_calls=[ToolCall(
                        id="tc1",
                        name="spawn_task",
                        arguments=json.dumps({
                            "title": "Writing Hello World File",
                            "objective": "Write \"hello world\" content into a file named \"hello.txt\".",
                        }),
                    )],
                    finish_reason="tool_calls",
                )
            return gen1()
        else:
            async def gen2():
                yield StreamChunk(
                    content="Your last two tasks were Writing Hello World File and Inspecting Gcx Imports; both completed cleanly.",
                    finish_reason="stop",
                )
            return gen2()

    events = []
    with patch("jarvis.conscious.agent.open_chat_stream", side_effect=fake_stream), \
         patch("jarvis.conscious.agent.store.recent_messages", return_value=[]), \
         patch("jarvis.conscious.agent.store.add_message"), \
         patch("jarvis.conscious.agent.store.all_tasks", return_value=completed_tasks), \
         patch("jarvis.conscious.agent.store.pending_questions", return_value=[]):
        async for ev in ca.handle_utterance("Please go over the last two tasks you've done", source="voice"):
            events.append(ev)

    tool_events = [e for e in events if e["type"] == "tool"]
    assert len(dummy.created) == 0, f"Retrospective query should NOT recreate task: {dummy.created}"
    assert len(tool_events) == 1 and tool_events[0]["name"] == "task_status", f"Expected redirect to task_status: {tool_events}"
    assert seen_tools_per_round == [True, False], f"Round 2 should pass tools=None: {seen_tools_per_round}"
    assert any("Recent completed tasks:" in sp and "Writing Hello World File" in sp for sp in seen_sys_prompts)

    # Now verify legitimate new task request still spawns normally
    seen_tools_per_round.clear()

    async def fake_real_spawn(provider, messages, tools=None, temperature=0.6, max_tokens=None, cancel=None):
        seen_tools_per_round.append(tools is not None)
        if len(seen_tools_per_round) == 1:
            async def g1():
                yield StreamChunk(
                    tool_calls=[ToolCall(
                        id="tc2",
                        name="spawn_task",
                        arguments=json.dumps({
                            "title": "Creating Notes File",
                            "objective": "Create notes.txt on the Desktop with meeting reminders",
                        }),
                    )],
                    finish_reason="tool_calls",
                )
            return g1()
        else:
            async def g2():
                yield StreamChunk(content="Started creating notes.txt on your Desktop.", finish_reason="stop")
            return g2()

    with patch("jarvis.conscious.agent.open_chat_stream", side_effect=fake_real_spawn), \
         patch("jarvis.conscious.agent.store.recent_messages", return_value=[]), \
         patch("jarvis.conscious.agent.store.add_message"), \
         patch("jarvis.conscious.agent.store.all_tasks", return_value=completed_tasks), \
         patch("jarvis.conscious.agent.store.pending_questions", return_value=[]):
        events2 = [ev async for ev in ca.handle_utterance("Make a file on the desktop named notes.txt", source="voice")]

    assert len(dummy.created) == 1, f"Legitimate task should have spawned: {dummy.created}"
    print("[PASS] test_retrospective_task_query_redirects_hallucinated_spawn_to_status")


async def test_deobfuscation_iso_binary_guards_ghidra_quoting_and_continue_redirect() -> None:
    """Verify Mode2/2352 ISO extraction, binary guards on read_file/search_text,
    Ghidra invocation with spaces in paths, and Conscious continue -> resume redirect."""
    import struct
    from jarvis.iba import mcp_bridge
    from jarvis.iba.tools import Tools
    from jarvis.iba.store import Store as IBAStore

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td_path = Path(td)
        # 1) Build a Mode 2 / 2352 ISO9660 image with spaces in the parent folder
        spaced_dir = td_path / "Game Tools [PAL]"
        spaced_dir.mkdir(parents=True, exist_ok=True)
        bin_path = spaced_dir / "Test Game.bin"

        # Construct sectors 0..25 (Mode 2 / 2352 bytes per sector, +24 data offset)
        total_sectors = 26
        buf = bytearray(total_sectors * 2352)

        def _write_sec(lba: int, payload: bytes) -> None:
            off = lba * 2352
            buf[off : off + 12] = b"\x00" + b"\xff" * 10 + b"\x00"
            buf[off + 12 : off + 16] = b"\x00\x02\x16\x02"
            p = payload[:2048].ljust(2048, b"\x00")
            buf[off + 24 : off + 24 + 2048] = p

        # Minimal ELF32 MIPS LE at LBA 24 (512 bytes)
        elf_payload = (
            b"\x7fELF\x01\x01\x01" + b"\x00" * 9
            + struct.pack("<HHIIIIIHHHHHH", 2, 8, 1, 0x100000, 52, 0, 0x20000000, 52, 32, 1, 0, 0, 0)
            + struct.pack("<IIIIIIII", 1, 0x100, 0x100000, 0x100000, 256, 256, 7, 0x1000)
        ).ljust(512, b"\x00")
        cnf_payload = b"BOOT2 = cdrom0:\\SLES_999.99;1\r\nVER = 1.00\r\n"

        # Directory at LBA 22
        def _dir_rec(name_bytes: bytes, lba: int, size: int, flags: int = 0) -> bytes:
            rec_len = 33 + len(name_bytes) + (1 if (33 + len(name_bytes)) % 2 else 0)
            r = bytearray(rec_len)
            r[0] = rec_len
            struct.pack_into("<I", r, 2, lba)
            struct.pack_into("<I", r, 10, size)
            r[25] = flags
            r[32] = len(name_bytes)
            r[33 : 33 + len(name_bytes)] = name_bytes
            return bytes(r)

        dir_data = (
            _dir_rec(b"\x00", 22, 2048, 2)
            + _dir_rec(b"\x01", 22, 2048, 2)
            + _dir_rec(b"SYSTEM.CNF;1", 23, len(cnf_payload), 0)
            + _dir_rec(b"SLES_999.99;1", 24, len(elf_payload), 0)
        )
        pvd = bytearray(2048)
        pvd[0:6] = b"\x01CD001"
        pvd[8:40] = b"PLAYSTATION".ljust(32, b" ")
        pvd[156:190] = _dir_rec(b"\x00", 22, 2048, 2)[:34]
        _write_sec(16, bytes(pvd))
        _write_sec(22, dir_data)
        _write_sec(23, cnf_payload)
        _write_sec(24, elf_payload)
        bin_path.write_bytes(bytes(buf))

        # Check binary_probe(mode='info') & binary_probe(mode='iso_extract')
        iba_s = IBAStore(td_path / "iba_s")
        t = Tools(iba_s, "jarvis@local", "jarvis", spaced_dir)
        info = t.execute("binary_probe", {"path": "Test Game.bin", "mode": "info"})
        assert info["format"] == "ISO9660_Mode2_2352", info
        assert info["boot_executable"] == "SLES_999.99", info

        ext = t.execute("binary_probe", {"path": "Test Game.bin", "mode": "iso_extract"})
        assert ext["extracted_count"] == 2, ext
        elf_rec = [x for x in ext["extracted"] if x["iso_path"] == "SLES_999.99"][0]
        assert elf_rec["format"] == "ELF32" and elf_rec["machine"] == "MIPS", elf_rec
        assert Path(elf_rec["extracted_path"]).is_file()

        # Check ghidra_analyze refuses raw ISO .bin image gracefully with status='failed'
        g_iso = t.execute("ghidra_analyze", {"path": "Test Game.bin"})
        assert g_iso["ok"] is False and g_iso["status"] == "failed" and "iso_extract" in g_iso["error"]

        # 2) read_file and search_text reject binary files cleanly
        for bad_tool, bad_args in [
            ("read_file", {"path": "Test Game.bin", "line": 1, "limit": 80}),
            ("search_text", {"path": "Test Game.bin", "pattern": "rrrr"}),
            ("read_file", {"path": "Test Game_extracted/SLES_999.99", "line": 1, "limit": 80}),
        ]:
            raised = False
            try:
                t.execute(bad_tool, bad_args)
            except ValueError as exc:
                raised = "binary file" in str(exc).lower()
            assert raised, f"{bad_tool} should fail fast on binary file"

        ws_zero = t.execute("search_text", {"pattern": "rrrr"})
        assert ws_zero["count"] == 0 and "binary files" in ws_zero["hint"].lower(), ws_zero

        # 3) ghidra_analyze invokes analyzeHeadless.bat directly (not cmd.exe /c)
        fake_ghidra = td_path / "MGS3 Tools With Spaces" / "support" / "analyzeHeadless.bat"
        fake_ghidra.parent.mkdir(parents=True, exist_ok=True)
        fake_ghidra.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
        seen_cmds = []

        class _FakeProc:
            returncode = 0
            stdout = "Ghidra OK"
            stderr = ""

        def _fake_run(cmd, **kwargs):
            seen_cmds.append(cmd)
            return _FakeProc()

        with patch.dict(os.environ, {"GHIDRA_INSTALL_DIR": str(fake_ghidra.parent.parent)}), \
             patch("jarvis.iba.mcp_bridge.subprocess.run", side_effect=_fake_run):
            g_ok = mcp_bridge.ghidra_analyze(Path(elf_rec["extracted_path"]))
        assert g_ok["ok"] is True and g_ok["status"] == "completed", g_ok
        assert seen_cmds and seen_cmds[0][0] == str(fake_ghidra) and "cmd.exe" not in seen_cmds[0][0].lower(), seen_cmds

    # 4) ConsciousAgent redirects "make it continue" hallucinated spawn_task -> control_task(resume)
    ca = ConsciousAgent()
    ca.channel = "voice"

    class _DummyResumeEngine:
        def __init__(self) -> None:
            self.created = []
            self.resumed = []

        def create(self, objective: str, title=None, workspace=None):
            self.created.append(objective)
            return {"id": "new_tid", "title": title or "New", "status": "queued"}

        def resume(self, tid: str) -> None:
            self.resumed.append(tid)

        def snapshot(self):
            return []

    dummy = _DummyResumeEngine()
    ca.attach_engine(dummy)

    paused_task = [{
        "id": "tid_deobf_1",
        "title": "Deobfuscate Myth Makers Orbs of Doom",
        "objective": "Deobfuscate Myth Makers Orbs of Doom.bin",
        "status": "paused",
        "progress": 22,
        "result": None,
    }]
    calls = 0

    async def fake_continue_stream(provider, messages, tools=None, temperature=0.6, max_tokens=None, cancel=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            async def g1():
                yield StreamChunk(
                    tool_calls=[ToolCall(
                        id="tc_cont",
                        name="spawn_task",
                        arguments=json.dumps({
                            "title": "Deobfuscation Continuation",
                            "objective": "Continue deobfuscation",
                        }),
                    )],
                    finish_reason="tool_calls",
                )
            return g1()
        else:
            async def g2():
                yield StreamChunk(content="Resumed your deobfuscation task, sir.", finish_reason="stop")
            return g2()

    events = []
    with patch("jarvis.conscious.agent.open_chat_stream", side_effect=fake_continue_stream), \
         patch("jarvis.conscious.agent.store.recent_messages", return_value=[]), \
         patch("jarvis.conscious.agent.store.add_message"), \
         patch("jarvis.conscious.agent.store.all_tasks", return_value=paused_task), \
         patch("jarvis.conscious.agent.store.get_task", return_value=paused_task[0]), \
         patch("jarvis.conscious.agent.store.pending_questions", return_value=[]):
        async for ev in ca.handle_utterance("That's fantastic can you make it continue", source="voice"):
            events.append(ev)

    assert len(dummy.created) == 0, f"Should NOT spawn duplicate task: {dummy.created}"
    assert dummy.resumed == ["tid_deobf_1"], f"Should resume paused task: {dummy.resumed}"
    print("[PASS] test_deobfuscation_iso_binary_guards_ghidra_quoting_and_continue_redirect")


def test_outcome_rea_hoisting_and_iso_root_first() -> None:
    import struct
    from jarvis.iba import mcp_bridge
    from jarvis.iba.engine import Runtime as IBARuntime
    from jarvis.iba.store import Store as IBAStore

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td_path = Path(td)
        iba_s = IBAStore(td_path / "iba_s")
        rt = IBARuntime(iba_s, None, td_path)

        # 1) Runtime._outcome must not pop top-level "tool" for rea, and redirects rea({"path": ...}) -> binary_probe
        parsed = rt._outcome(json.dumps({
            "kind": "action",
            "text": "Inspect LIBSD.IRX",
            "tool": "rea",
            "tool_arguments": {"path": "D:/fake/LIBSD.IRX"},
        }))
        assert parsed["tool"] == "binary_probe"
        assert parsed["arguments"]["path"] == "D:/fake/LIBSD.IRX"

        parsed_valid_rea = rt._outcome(json.dumps({
            "kind": "action",
            "text": "Call valid REA tool",
            "tool": "rea",
            "tool_arguments": {"tool": "analyze_ps2_elf", "arguments": {"path": "x"}},
        }))
        assert parsed_valid_rea["tool"] == "rea"
        assert parsed_valid_rea["arguments"]["tool"] == "analyze_ps2_elf"

        # 2) ISO9660 with >40 subdir files BEFORE root SYSTEM.CNF & SLES_999.99 still extracts root boot files
        bin_path = td_path / "ManyModules.bin"
        buf = bytearray(30 * 2352)

        def _write_sec(lba: int, payload: bytes) -> None:
            off = lba * 2352
            buf[off : off + 12] = b"\x00" + b"\xff" * 10 + b"\x00"
            buf[off + 12 : off + 16] = b"\x00\x02\x16\x02"
            buf[off + 24 : off + 24 + 2048] = payload[:2048].ljust(2048, b"\x00")

        def _dir_rec(name_bytes: bytes, lba: int, size: int, flags: int = 0) -> bytes:
            rec_len = 33 + len(name_bytes) + (1 if (33 + len(name_bytes)) % 2 else 0)
            r = bytearray(rec_len)
            r[0] = rec_len
            struct.pack_into("<I", r, 2, lba)
            struct.pack_into("<I", r, 10, size)
            r[25] = flags
            r[32] = len(name_bytes)
            r[33 : 33 + len(name_bytes)] = name_bytes
            return bytes(r)

        # Subdir at LBA 25..26 with 45 dummy files
        sub_data = _dir_rec(b"\x00", 25, 4096, 2) + _dir_rec(b"\x01", 22, 2048, 2)
        for idx in range(45):
            entry = _dir_rec(f"MOD_{idx:02d}.IRX;1".encode(), 28, 32, 0)
            if (len(sub_data) // 2048) != ((len(sub_data) + len(entry) - 1) // 2048):
                sub_data = sub_data.ljust(((len(sub_data) // 2048) + 1) * 2048, b"\x00")
            sub_data += entry
        _write_sec(25, sub_data[:2048])
        _write_sec(26, sub_data[2048:4096])

        cnf_payload = b"BOOT2 = cdrom0:\\SLES_999.99;1\r\n"
        elf_payload = b"\x7fELF\x01\x01\x01" + b"\x00" * 57
        # Put MODULES dir FIRST in root directory before SLES_999.99 & SYSTEM.CNF
        root_dir = (
            _dir_rec(b"\x00", 22, 2048, 2)
            + _dir_rec(b"\x01", 22, 2048, 2)
            + _dir_rec(b"MODULES", 25, 4096, 2)
            + _dir_rec(b"SLES_999.99;1", 27, len(elf_payload), 0)
            + _dir_rec(b"SYSTEM.CNF;1", 28, len(cnf_payload), 0)
        )
        pvd = bytearray(2048)
        pvd[0:6] = b"\x01CD001"
        pvd[156:190] = _dir_rec(b"\x00", 22, 2048, 2)[:34]
        _write_sec(16, bytes(pvd))
        _write_sec(22, root_dir)
        _write_sec(27, elf_payload)
        _write_sec(28, cnf_payload)
        bin_path.write_bytes(bytes(buf))

        ext = mcp_bridge._extract_iso9660_files(bin_path, td_path / "ext_out")
        extracted_names = {x["iso_path"] for x in ext["extracted"]}
        assert "SLES_999.99" in extracted_names and "SYSTEM.CNF" in extracted_names, ext

        # 3) Oversized read_file batch (9 .IRX files) + trailing <tool_call|> truncates to 4 & redirects to binary_probe
        batch_payload = {
            "kind": "action",
            "text": "Batch read extracted modules",
            "tool": "read_file",
            "tool_arguments": [
                {"path": f"D:/fake/MODULES/MOD_{i}.IRX", "line": 1, "limit": 80}
                for i in range(9)
            ],
        }
        parsed_batch = rt._outcome(json.dumps(batch_payload) + "<tool_call|>")
        assert parsed_batch["tool"] == "binary_probe", parsed_batch
        assert isinstance(parsed_batch["arguments"], list) and len(parsed_batch["arguments"]) == 4, parsed_batch
        assert all(a["mode"] == "info" for a in parsed_batch["arguments"]), parsed_batch

        for bin_target in ("D:/fake/SLES_522.41", "D:/fake/ORBSPS2.WAD"):
            parsed_bin = rt._outcome(json.dumps({
                "kind": "action",
                "text": "Read binary",
                "tool": "read_file",
                "tool_arguments": {"path": bin_target, "line": 1, "limit": 80},
            }))
            assert parsed_bin["tool"] == "binary_probe" and parsed_bin["arguments"] == {"path": bin_target, "mode": "info"}

        # 4) _parse_wad_summary / binary_probe(mode="info") identifies WADH archives & extracts sample_entries
        str_tbl = b"DATA\x00test.bmp\x00"
        entries_raw = (
            struct.pack("<6i", 0, 0, 0, 1, 1, -1)
            + struct.pack("<6i", 5, 16, 128, 0, -1, -1)
        )
        data_off = 16 + len(entries_raw) + len(str_tbl)
        wad_bytes = (
            b"WADH"
            + struct.pack("<3I", data_off, 2, len(str_tbl))
            + entries_raw
            + str_tbl
            + b"\x00" * 128
        )
        wad_path = td_path / "TEST.WAD"
        wad_path.write_bytes(wad_bytes)
        wad_info = mcp_bridge.binary_probe(wad_path, mode="info")
        assert wad_info["format"] == "WADH_Archive", wad_info
        assert wad_info["entry_count"] == 2, wad_info
        assert [e["name"] for e in wad_info["sample_entries"]] == ["DATA", "test.bmp"], wad_info
        assert wad_info["sample_entries"][1]["data_offset"] == data_off + 16, wad_info

        # 5) Malformed model JSON repair (invalid escape \A, comma for colon, missing colon before object)
        malformed = (
            '{"kind","action","text":"Read ALCHEMY.INI content to verify hash and prepare for copying into '
            'Coliseum work folder","tool","read_file", "tool_arguments" '
            '{"path":"D:\\\\Games\\\\Emulation\\\\PS2\\\\Games\\\\ISO&Rar\\\\Yu-Gi-Oh! Capsule Monster Coliseum (USA)_extracted\\ALCHEMY.INI","line":1}}'
        )
        parsed_malformed = rt._outcome(malformed)
        assert parsed_malformed["kind"] == "action", parsed_malformed
        assert parsed_malformed["tool"] == "read_file", parsed_malformed
        assert parsed_malformed["arguments"]["path"] == "D:\\Games\\Emulation\\PS2\\Games\\ISO&Rar\\Yu-Gi-Oh! Capsule Monster Coliseum (USA)_extracted\\ALCHEMY.INI", parsed_malformed
        assert parsed_malformed["arguments"]["line"] == 1, parsed_malformed

        # rea({"artifact": "D_ANM.BIN"}) redirects to binary_probe
        parsed_art_rea = rt._outcome(json.dumps({
            "kind": "action",
            "text": "Analyze data",
            "tool": "rea",
            "tool_arguments": {"artifact": "D_ANM.BIN"},
        }))
        assert parsed_art_rea["tool"] == "binary_probe" and parsed_art_rea["arguments"]["path"] == "D_ANM.BIN"

        # Single quoted JSON outcome parses cleanly
        parsed_sq = rt._outcome("{'kind': 'action', 'text': 'Single quote test', 'tool': 'list_directory', 'tool_arguments': {'path': 'Desktop'}}")
        assert parsed_sq["kind"] == "action" and parsed_sq["tool"] == "list_directory"
    print("[PASS] test_outcome_rea_hoisting_and_iso_root_first")


if __name__ == "__main__":
    asyncio.run(test_conscious_thought_coalescing_and_tool_rescue())
    asyncio.run(test_thought_brevity_hex_cleanup_and_reasoning_delegation())
    asyncio.run(test_retrospective_task_query_redirects_hallucinated_spawn_to_status())
    asyncio.run(test_deobfuscation_iso_binary_guards_ghidra_quoting_and_continue_redirect())
    test_outcome_rea_hoisting_and_iso_root_first()
    test_spawn_task_context_and_iba_file_robustness()
    test_iba_sql_desktop_file_and_wait_resume()
    test_external_workspace_agents_md_discovery_rea_and_binary_probe()
    test_live_lmstudio_probe_if_up()
    print("ALL IBA & CONSCIOUS TESTS PASSED")
