"""The conscious agent — the foreground "self" that listens, thinks, speaks.

- Streams from the dynamic provider (Dory <-> LM Studio re-resolved per turn).
- Emits sentence-level deltas so TTS can start speaking before the full
  answer exists.
- Holds OpenAI-style tools (spawn tasks, relay answers, remember, switch
  backends, rename itself).
- Knows the profile, open tasks, pending task questions, and onboarding state.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import AsyncIterator, List, Optional

from .. import config as cfg
from ..events import publish
from ..memory import store
from ..memory.profile import get_profile
from ..providers.base import ChatMessage, open_chat_stream
from ..providers.registry import get_registry
from .tools import TOOLS, ToolHost, machine_summary

SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])(?<!\d\.)(?<!\.\.\.)(?<!\b[A-Za-z]\.)\s+(?=[A-Z0-9\"'(\[])"
)

# Trivial facts small local models reliably hallucinate — answer directly.
# Every alternative must reach the end of the utterance so scheduling
# questions ("what time does the store close") don't false-trigger.
_DIRECT_TIME = re.compile(
    r"(?:"
    r"\bwhat(?:'s| is| are)?\s+(?:the\s+)?(?:time|date|day)"
    r"(?:\s+(?:is\s+)?(?:it|today|now|right now))?"
    r"|\bcurrent\s+(?:time|date)"
    r"|\b(?:do you know|tell me)\s+(?:the\s+)?(?:time|date|day)"
    r"|\bwhat day is (?:it|today)"
    r"|\btime right now"
    r")\s*[?.!]*\s*$", re.I)


class ConsciousAgent:
    def __init__(self) -> None:
        self.channel = "voice"
        self._busy = False
        self._cancel = asyncio.Event()
        self._engine = None
        self._host: Optional[ToolHost] = None
        self._turn_lock: Optional[asyncio.Lock] = None
        self._turn_lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_turn_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._turn_lock is None or self._turn_lock_loop is not loop:
            self._turn_lock = asyncio.Lock()
            self._turn_lock_loop = loop
        return self._turn_lock

    @staticmethod
    def reasoning_enabled() -> bool:
        return bool(cfg.get("conscious.reasoning_enabled", False))

    # ------------------------------------------------------------------ context

    def _system_prompt(self) -> str:
        name = cfg.assistant_name()
        profile = get_profile()
        all_tasks = store.all_tasks()[:8]
        active = []
        recent_done = []
        for t in all_tasks:
            st = str(t.get("status") or "").lower()
            if st in ("queued", "running", "waiting_user", "paused"):
                q = store.pending_questions(t["id"])
                active.append(
                    f"- \"{t['title']}\" — {st}, {t.get('progress', 0)}%"
                    + (f" (NEEDS ANSWER: {q[0]['question']})" if q else "")
                )
            elif len(recent_done) < 5:
                res = " ".join(str(t.get("result") or "").split())
                if len(res) > 120:
                    res = res[:117].rstrip() + "..."
                recent_done.append(
                    f"- \"{t['title']}\" — {st}" + (f": {res}" if res else "")
                )
        pending = [
            f"- task \"{t['title']}\" asks: {q['question']}"
            for t in all_tasks if t.get("status") == "waiting_user"
            for q in store.pending_questions(t["id"])
        ]
        reasoning_rule = (
            'REASONING BUDGET: at most 1-2 short sentences (under 30 words). Never repeat SYSTEM STATE.'
            if self.reasoning_enabled()
            else 'REASONING DISABLED: Do NOT emit <think> blocks or chain-of-thought. Reply directly.'
        )
        status = f"""SYSTEM STATE:
- Name: {name} | Machine: {machine_summary()} | Onboarding: {"complete" if profile.is_complete() else f"{int(profile.progress()*100)}%"}
- Open background tasks:
{chr(10).join(active) if active else "  (none)"}
- Recent completed tasks:
{chr(10).join(recent_done) if recent_done else "  (none)"}
- Pending task questions:
{chr(10).join(pending) if pending else "  (none)"}

Rules:
- Keep spoken answers concise (1-3 sentences) and natural; never say hex IDs or raw JSON aloud.
- {reasoning_rule}
- Use spawn_task ONLY for NEW multi-step work/files/code/research requested now — NEVER to inspect, review, or summarize past/active tasks (use task_status for that).
- Use get_current_time for time/date, system_info for PC hardware specs, answer_task_question when the user answers a pending task question.
- Call tools via tool calling or a single JSON object {{"name": "<tool>", "arguments": {{...}}}}. After a tool result returns, answer in one short plain sentence."""
        interview = ""
        if not profile.is_complete():
            missing = profile.missing_slots()
            interview = f"""
ONBOARDING MODE: You are running the first-boot interview naturally, one
question at a time, woven into conversation. Do NOT interrogate mechanically.
Remaining slots: {', '.join(missing)}.
After each answer the profile is updated automatically; move to the next topic
when you sense the answer is complete. Once all slots are filled, say the
interview is done and switch to normal operation.
"""
        return f"""{cfg.get('assistant.persona', '')}

{profile.summary()}

{status}{interview}"""

    def _history(self) -> List[ChatMessage]:
        rows = store.recent_messages(self.channel,
                                     limit=int(cfg.get("conscious.history_messages", 40) or 40))
        out = []
        for r in rows:
            role = r["role"]
            if role == "tool":
                out.append(ChatMessage(role="assistant", content=f"(tool result) {r['content']}"))
            else:
                out.append(ChatMessage(role=role, content=r["content"]))
        return out

    # ------------------------------------------------------------------ turns

    async def handle_utterance(
        self, text: str, source: str = "voice"
    ) -> AsyncIterator[dict]:
        """Process one user utterance. Yields events:
            {type: 'sentence', text}   — visible speech, ready for TTS
            {type: 'thought', text}    — chain-of-thought (GUI only)
            {type: 'tool', name, args} — tool usage (GUI only)
            {type: 'done', full}       — final full text
        """
        async with self._get_turn_lock():
            self._busy = True
            self._cancel = asyncio.Event()
            publish("conscious_thinking", input=text, source=source)
            try:
                async for ev in self._run_turn(text, source):
                    yield ev
            finally:
                self._busy = False
                publish("conscious_idle")

    async def interrupt(self) -> None:
        self._cancel.set()

    async def _run_turn(self, text: str, source: str) -> AsyncIterator[dict]:
        store.add_message(self.channel, "user", text)

        direct = self._direct_fact_answer(text)
        if direct is not None:
            store.add_message(self.channel, "assistant", direct)
            yield {"type": "sentence", "text": direct}
            yield {"type": "done", "full": direct}
            return

        profile = get_profile()
        if not profile.is_complete():
            current_key = profile.missing_slots()[0] if profile.missing_slots() else None
        else:
            current_key = None

        if self._host is None:
            self._host = ToolHost(None, get_profile)

        messages: List[ChatMessage] = [ChatMessage(role="system",
                                                   content=self._system_prompt())]
        messages += self._history()
        if messages[-1].content == text and messages[-1].role == "user":
            pass  # history already includes it

        full_parts: List[str] = []
        buffer = ""
        rounds = 0
        executed_tool_names: List[str] = []
        while rounds < 4:
            rounds += 1
            reasoning_on = self.reasoning_enabled()
            provider = get_registry().get()
            parts: List[str] = []
            tool_calls = {}
            json_candidate = False
            reasoning_parts: List[str] = []
            thought_emitted = False
            try:
                agen = await open_chat_stream(
                    provider, messages,
                    tools=None if executed_tool_names else TOOLS,
                    temperature=0.6,
                    max_tokens=int(cfg.get("conscious.response_token_budget", 2048) or 2048),
                    cancel=self._cancel)
                async for chunk in agen:
                    if chunk.reasoning:
                        reasoning_parts.append(chunk.reasoning)
                        if reasoning_on and not thought_emitted:
                            so_far = "".join(reasoning_parts)
                            # Wait briefly for 1-2 clean sentences before emitting once per round
                            if len(so_far) >= 160 or len(SENTENCE_SPLIT.findall(so_far)) >= 1 and len(so_far) >= 45:
                                summary = self._summarize_thought(so_far)
                                if summary:
                                    thought_emitted = True
                                    yield {"type": "thought", "text": summary}
                    if chunk.content:
                        if reasoning_on and reasoning_parts and not thought_emitted:
                            summary = self._summarize_thought("".join(reasoning_parts))
                            thought_emitted = True
                            if summary:
                                yield {"type": "thought", "text": summary}
                        parts.append(chunk.content)
                        buffer += chunk.content
                        raw = "".join(parts)
                        if not json_candidate and raw.lstrip().startswith("{"):
                            json_candidate = True
                        # sentence-level streaming to TTS (held back while the
                        # reply might still be an embedded tool-call JSON blob)
                        if not json_candidate:
                            while True:
                                m = SENTENCE_SPLIT.search(buffer)
                                if not m:
                                    break
                                sentence = self._clean_spoken_text(buffer[:m.end()])
                                buffer = buffer[m.end():]
                                if sentence:
                                    full_parts.append(sentence)
                                    yield {"type": "sentence", "text": sentence}
                    for tc in chunk.tool_calls:
                        tool_calls[tc.id or tc.name] = tc
                    if self._cancel.is_set():
                        break
            except Exception as exc:  # noqa: BLE001
                msg = f"My reasoning engine hit an error: {exc}"
                yield {"type": "sentence", "text": msg}
                yield {"type": "done", "full": msg}
                store.add_message(self.channel, "assistant", msg)
                return

            content = "".join(parts).strip()
            reasoning_full = "".join(reasoning_parts).strip()
            if reasoning_on and reasoning_full and not thought_emitted:
                summary = self._summarize_thought(reasoning_full)
                if summary:
                    yield {"type": "thought", "text": summary}
            if self._cancel.is_set():
                yield {"type": "done", "full": "(interrupted)"}
                return

            if tool_calls:
                # execute native tool calls, then continue with a short reply
                normalized_tcs = []
                for tc in tool_calls.values():
                    try:
                        args = json.loads(tc.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    raw_name, raw_args = tc.name, args
                    if raw_name == "spawn_task":
                        redir = self._resolve_spawn_redirect(raw_args, text)
                        if redir is not None:
                            raw_name, raw_args = redir
                    name, args = self._normalize_tool_call(raw_name, raw_args, text)
                    normalized_tcs.append((name, args))
                messages.append(ChatMessage(
                    role="assistant",
                    content=content or
                    f"[tool calls: {', '.join(n for n, _ in normalized_tcs)}]"))
                tool_results = []
                for name, args in normalized_tcs:
                    executed_tool_names.append(name)
                    publish("conscious_tool", name=name, args=args)
                    result = await self._host.execute(name, args)
                    tool_results.append(result)
                    yield {"type": "tool", "name": name, "args": args,
                           "result": result[:400]}
                for tr in tool_results:
                    messages.append(ChatMessage(role="user", content=f"[tool result] {tr}"))
                messages.append(ChatMessage(
                    role="user",
                    content="Using the tool result above, answer the user's "
                            "question in ONE short natural sentence — give them "
                            "the actual information, not a report about the "
                            "tool. Plain language only, never JSON or tool syntax."))
                buffer = ""
                full_parts = []
                continue

            # text answer: local models sometimes write tool calls as plain
            # JSON (inside content OR at the tail of reasoning_content) — detect
            # and execute instead of dropping it or speaking raw JSON.
            embedded = (
                self._parse_embedded_tool_call(content, text)
                or self._parse_embedded_tool_call(reasoning_full, text)
            )
            if not embedded and not executed_tool_names and self._should_auto_spawn(text, content, reasoning_full):
                embedded = ("spawn_task", {"objective": text})
            if embedded:
                name, args = embedded
                if name == "spawn_task":
                    redir = self._resolve_spawn_redirect(args, text)
                    if redir is not None:
                        name, args = redir
                name, args = self._normalize_tool_call(name, args, text)
                executed_tool_names.append(name)
                publish("conscious_tool", name=name, args=args)
                result = await self._host.execute(name, args)
                yield {"type": "tool", "name": name, "args": args,
                       "result": result[:400]}
                messages.append(ChatMessage(role="assistant",
                                            content=f"[called {name}]"))
                messages.append(ChatMessage(role="user",
                                            content=f"[tool result] {result}"))
                messages.append(ChatMessage(
                    role="user",
                    content="Using the tool result above, answer the user's "
                            "question in ONE short natural sentence — give them "
                            "the actual information, not a report about the "
                            "tool. Plain language only, never JSON or tool syntax."))
                buffer = ""
                full_parts = []
                continue

            # text answer — flush the sentence buffer tail
            buffer_final = buffer.strip()
            # never speak raw JSON at the user
            if buffer_final.lstrip().startswith("{") and \
                    buffer_final.rstrip().endswith("}"):
                buffer_final = ""
            buffer_final = self._clean_spoken_text(buffer_final)
            if buffer_final:
                yield {"type": "sentence", "text": buffer_final}
                full_parts.append(buffer_final)
            final = self._clean_spoken_text(" ".join(p for p in full_parts if p))
            if not final and executed_tool_names:
                if "spawn_task" in executed_tool_names:
                    final = "Right away, sir. I've launched a background agent session for that task and will report back when it finishes."
                else:
                    final = "Done, sir."
                yield {"type": "sentence", "text": final}
            if final:
                store.add_message(self.channel, "assistant", final)
            yield {"type": "done", "full": final}
            return

    # ------------------------------------------------------------------ wiring

    @staticmethod
    def _direct_fact_answer(text: str) -> Optional[str]:
        """Instant answer for trivial facts (time/date) the LLM would guess."""
        if not _DIRECT_TIME.search(text):
            return None
        now = datetime.now()
        low = text.lower()
        if "date" in low or "day" in low:
            return now.strftime("It's %A, %B %d, %Y.")
        clock = now.strftime("%I:%M %p")
        if clock.startswith("0"):
            clock = clock[1:]
        return f"It's {clock} on {now.strftime('%A, %B %d')}."

    TOOL_NAMES = {"spawn_task", "task_status", "answer_task_question",
                  "control_task", "remember", "switch_backend",
                  "rename_assistant", "system_info", "get_current_time"}
    WORKER_TOOL_NAMES = {
        "write_file", "create_file", "edit_file", "read_file", "delete_file",
        "list_directory", "find_files", "search_text", "run_shell", "shell",
        "run_command", "execute_command", "python", "web_search", "fetch_url",
    }

    _AUTO_TASK_RE = re.compile(
        r"\b(?:make|create|write|save|edit|modify|update|delete|remove|build|"
        r"run|execute|generate|search\s+for|find\s+the)\s+(?:a\s+|an\s+|the\s+)?"
        r"(?:file|text\s+file|script|folder|directory|document|note|code|program|"
        r"desktop\b|files\b)"
        r"|\bon\s+(?:the|my)\s+desktop\b",
        re.IGNORECASE,
    )
    _DEEP_REASONING_RE = re.compile(
        r"\b(?:analyze|analyse|investigate|debug|diagnose|refactor|design|architect|"
        r"research|compare\s+in\s+detail|solve|calculate|plan\s+out|write\s+a\s+report|"
        r"audit|review\s+the\s+codebase)\b",
        re.IGNORECASE,
    )
    _SPOKEN_HEX_ID_RE = re.compile(
        r"\s*\(\s*`?[0-9a-f]{16,36}`?\s*\)"
        r"|\s*\[`?[0-9a-f]{16,36}`?\]"
        r"|\s*`[0-9a-f]{16,36}`"
        r"|\b[0-9a-f]{24,36}\b",
        re.IGNORECASE,
    )
    _THOUGHT_OBS_RE = re.compile(
        r"(?:Current\s+State\s+Observation\s*:.*?(`[^`]*`|\.[^.]*\.|$))",
        re.IGNORECASE | re.DOTALL,
    )

    @classmethod
    def _clean_spoken_text(cls, text: str) -> str:
        """Strip raw task hex IDs, markdown markers, and extra whitespace before TTS/GUI."""
        if not text:
            return ""
        cleaned = cls._SPOKEN_HEX_ID_RE.sub("", text)
        cleaned = re.sub(r"\*\*|__|`|~~", "", cleaned)
        cleaned = re.sub(r"\s+\*\s+", ", ", cleaned)
        cleaned = cleaned.replace("*", "")
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    @classmethod
    def _summarize_thought(cls, text: str) -> str:
        """Keep at most 1-2 short sentences (<=160 chars) of clean foreground thinking."""
        if not text:
            return ""
        # Drop embedded JSON tool call tail and raw state observation dumps
        cleaned = re.sub(r"\{.*\}\s*$", "", text, flags=re.DOTALL)
        cleaned = cls._THOUGHT_OBS_RE.sub(" ", cleaned)
        cleaned = cls._SPOKEN_HEX_ID_RE.sub("", cleaned)
        cleaned = " ".join(cleaned.replace("`", "").split()).strip()
        if not cleaned:
            return ""
        # Extract up to 2 short sentences
        parts = SENTENCE_SPLIT.split(cleaned)
        picked = " ".join(p.strip() for p in parts[:2] if p.strip()).strip()
        if not picked:
            picked = cleaned
        if len(picked) > 160:
            cut = picked.rfind(" ", 60, 157)
            if cut == -1:
                cut = 157
            picked = picked[:cut].rstrip(" ,;:") + "…"
        return picked

    def _should_auto_spawn(self, user_text: str, content: str, reasoning_full: str) -> bool:
        if self._host is None or self._host.engine is None:
            return False
        if self._AUTO_TASK_RE.search(user_text or ""):
            if not content or (content.lstrip().startswith("{") and content.rstrip().endswith("}")):
                return True
            if "tool call" in (reasoning_full or "").lower() or "json" in (reasoning_full or "").lower():
                return True
        if self._DEEP_REASONING_RE.search(user_text or "") and len(reasoning_full or "") > 280:
            return True
        if "tool call json" in (reasoning_full or "").lower() and not content:
            return True
        return False

    _META_TASK_STOPWORDS = frozenset({
        "task", "tasks", "background", "start", "spawn", "run", "please",
        "can", "you", "your", "the", "last", "two", "three", "few", "recent",
        "previous", "done", "did", "what", "how", "status", "report",
        "summary", "summarize", "over", "about", "tell", "check", "going",
        "finished", "completed", "have", "ive", "youve", "that", "those",
        "them", "for", "with", "into", "from", "are", "were", "was", "been",
        "one", "ones", "thing", "things", "just", "all",
        "continue", "continuing", "continuation", "resume", "keep", "fantastic",
        "great", "make", "proceed", "unpause", "carry",
    })
    _RETRY_INTENT_RE = re.compile(
        r"\b(?:again|retry|redo|rerun|re-run|recreate|another|new\s+task|repeat)\b",
        re.IGNORECASE,
    )
    _CONTINUE_INTENT_RE = re.compile(
        r"\b(?:continue|resume|keep\s+going|unpause|carry\s+on)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        return {t for t in tokens if t not in cls._META_TASK_STOPWORDS}

    @classmethod
    def _resolve_spawn_redirect(cls, raw_args: dict, user_text: str) -> Optional[tuple[str, dict]]:
        """Redirect continue/status requests mistaken for spawn_task to control_task or task_status."""
        if cls._CONTINUE_INTENT_RE.search(user_text or "") and not cls._RETRY_INTENT_RE.search(user_text or ""):
            user_tokens = cls._content_tokens(user_text)
            try:
                recent = store.all_tasks()[:8]
            except Exception:
                recent = []
            matched_task = None
            for t in recent:
                if t.get("status") in ("done", "cancelled"):
                    continue
                prior_tokens = cls._content_tokens(f"{t.get('title', '')} {t.get('objective', '')}")
                if not user_tokens or (user_tokens & prior_tokens):
                    matched_task = t
                    break
            if matched_task is None and recent and not user_tokens:
                if recent[0].get("status") in ("paused", "queued", "running", "waiting_user"):
                    matched_task = recent[0]
            if matched_task is not None:
                st = str(matched_task.get("status") or "").lower()
                if st in ("paused", "queued"):
                    return ("control_task", {"task_id": matched_task["id"], "action": "resume"})
                if st == "running":
                    return ("task_status", {"task_id": matched_task["id"]})
        if cls._should_redirect_spawn_to_status(raw_args, user_text):
            return ("task_status", {})
        return None

    @classmethod
    def _should_redirect_spawn_to_status(cls, raw_args: dict, user_text: str) -> bool:
        """Redirect hallucinated/retrospective spawn_task calls back to task_status."""
        if cls._RETRY_INTENT_RE.search(user_text or ""):
            return False
        user_tokens = cls._content_tokens(user_text)
        if not user_tokens:
            return True
        obj_raw = f"{(raw_args or {}).get('title', '')} {(raw_args or {}).get('objective', '')}"
        obj_tokens = cls._content_tokens(obj_raw)
        if not obj_tokens:
            return False
        if obj_tokens.isdisjoint(user_tokens):
            return True
        try:
            recent = store.all_tasks()[:8]
        except Exception:
            recent = []
        for t in recent:
            prior_tokens = cls._content_tokens(f"{t.get('title', '')} {t.get('objective', '')}")
            if not prior_tokens:
                continue
            matched_prior = obj_tokens & prior_tokens
            if len(matched_prior) >= 0.6 * len(obj_tokens):
                # Same objective as an existing recent task without new target words in user_text
                if len(user_tokens & matched_prior) <= 1 and not ((obj_tokens - prior_tokens) & user_tokens):
                    return True
        return False

    @classmethod
    def _normalize_tool_call(cls, name: str, args: dict, user_text: str = "") -> tuple[str, dict]:
        n = (name or "").strip()
        a = dict(args or {})
        if n in cls.WORKER_TOOL_NAMES:
            obj = (user_text or "").strip()
            hint = f"{n}({json.dumps(a, ensure_ascii=False)})"
            objective = f"{obj}\nRequested action: {hint}" if obj else f"Perform action: {hint}"
            return "spawn_task", {"objective": objective}
        if n == "spawn_task":
            if "objective" not in a and isinstance(a.get("goal"), str):
                a["objective"] = a.pop("goal")
            if "objective" not in a and isinstance(a.get("task"), str):
                a["objective"] = a.pop("task")
            title_str = str(a.get("title") or "").strip()
            obj_str = str(a.get("objective") or "").strip()
            u_str = (user_text or "").strip()
            if title_str and title_str.lower() not in obj_str.lower():
                obj_str = f"{title_str}: {obj_str}" if obj_str else title_str
            if u_str and u_str.lower() not in obj_str.lower():
                obj_str = f"{obj_str}\nUser request: {u_str}" if obj_str else u_str
            a["objective"] = obj_str
        return n, a

    @classmethod
    def _parse_embedded_tool_call(cls, text: str, user_text: str = ""):
        """Detect embedded tool-call JSON written inside content or reasoning."""
        if not text or "{" not in text:
            return None
        cleaned = text.strip()
        # Strip markdown fences if present
        m_fence = re.search(r"```(?:json)?\s*\r?\n?(.*?)\r?\n?\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if m_fence:
            cleaned = m_fence.group(1).strip()

        decoder = json.JSONDecoder()
        candidates = []
        idx = 0
        while idx < len(cleaned):
            nb = cleaned.find("{", idx)
            if nb == -1:
                break
            try:
                obj, end_idx = decoder.raw_decode(cleaned, nb)
                if isinstance(obj, dict):
                    name = obj.get("name") or obj.get("tool") or obj.get("function")
                    if isinstance(name, str) and (name in cls.TOOL_NAMES or name in cls.WORKER_TOOL_NAMES):
                        args = obj.get("arguments")
                        if args is None:
                            args = obj.get("tool_arguments")
                        if args is None:
                            args = obj.get("args")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args or "{}")
                            except json.JSONDecodeError:
                                args = {}
                        if not isinstance(args, dict):
                            args = {
                                k: v for k, v in obj.items()
                                if k not in ("name", "tool", "function", "arguments", "tool_arguments", "args", "kind", "text")
                            }
                        candidates.append((name, args))
                idx = max(nb + 1, end_idx)
            except Exception:
                idx = nb + 1
        if not candidates:
            return None
        raw_name, raw_args = candidates[-1]
        if raw_name == "spawn_task" and cls._should_redirect_spawn_to_status(raw_args, user_text):
            return ("task_status", {})
        return cls._normalize_tool_call(raw_name, raw_args, user_text)

    def attach_engine(self, engine) -> None:
        self._engine = engine
        self._host = ToolHost(engine, get_profile)


agent = ConsciousAgent()
