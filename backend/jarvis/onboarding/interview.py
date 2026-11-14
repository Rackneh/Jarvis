"""Onboarding: guided interview + background extraction + environment scan.

Flow on first boot:
1. The conscious agent sees `interview` mode and asks the REQUIRED_SLOTS
   questions one at a time, conversationally.
2. Every user answer is ALSO queued to a background parser thread that calls
   the LLM to extract/normalize facts into the profile while the conversation
   keeps flowing.
3. In parallel, an environment scan walks the user's stated project paths and
   produces a compact summary into profile 'environment'.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .. import config as cfg
from ..events import publish
from ..memory.profile import Profile, get_profile
from ..memory import store

EXTRACT_SYSTEM = """You extract structured profile facts from a user's answer to an
onboarding question. Reply with ONLY a JSON object mapping profile keys to values.
Known keys: call_name (name to call the user), user_role, assistant_name_pref,
active_projects (comma-separated list incl. any paths mentioned), preferences,
permissions, working_hours, personality, plus any other durable facts as new
snake_case keys. Omit keys the answer says nothing about. Keep values terse."""


class Onboarder:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.profile: Profile = get_profile()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._parser_thread: Optional[threading.Thread] = None
        self._scan_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._parser_thread is None:
            self._parser_thread = threading.Thread(
                target=self._parser_loop, name="jarvis-onboard-parser", daemon=True)
            self._parser_thread.start()
        # background environment scan (first boot or whenever asked)
        self.scan_environment()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ interview

    def is_needed(self) -> bool:
        return not self.profile.is_complete()

    def next_question(self) -> Optional[str]:
        for key, question in self.profile_missing():
            return question
        return None

    def profile_missing(self):
        from ..memory.profile import REQUIRED_SLOTS
        return [(k, q) for k, q in REQUIRED_SLOTS if not self.profile.get(k)]

    def ingest_answer(self, answer: str, question_key: str) -> None:
        """Record the direct answer immediately, and queue background extraction."""
        self.profile.set(question_key, answer, source="onboarding")
        self._queue.put(answer)
        publish("onboard_answer", key=question_key)

    def interview_progress(self) -> float:
        return self.profile.progress()

    # ------------------------------------------------------------------ parser

    def _parser_loop(self) -> None:
        while not self._stop.is_set():
            try:
                answer = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                asyncio.run_coroutine_threadsafe(
                    self._extract(answer), self.loop).result(timeout=120)
            except Exception:
                pass

    async def _extract(self, answer: str) -> None:
        from ..providers.registry import get_registry
        from ..providers.base import ChatMessage
        provider = get_registry().get()
        messages = [
            ChatMessage(role="system", content=EXTRACT_SYSTEM),
            ChatMessage(role="user", content=f"User's answer: {answer}\nJSON:"),
        ]
        parts: List[str] = []
        try:
            from ..providers.base import ChatMessage, open_chat_stream
            agen = await open_chat_stream(provider, messages, temperature=0.1,
                                          max_tokens=600)
            async for chunk in agen:
                if chunk.content:
                    parts.append(chunk.content)
        except Exception:
            return
        text = "".join(parts).strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return
        merged = self.profile.merge_extracted(data, source="onboarding_parser")
        if merged:
            publish("onboard_extracted", merged=merged)

    # ------------------------------------------------------------------ env scan

    def scan_environment(self) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            return
        self._scan_thread = threading.Thread(
            target=self._scan_loop, name="jarvis-env-scan", daemon=True)
        self._scan_thread.start()

    def _scan_loop(self) -> None:
        projects = self.profile.get("active_projects", "")
        home = str(Path.home())
        candidates: List[str] = []
        for part in projects.replace(";", ",").split(","):
            part = part.strip()
            if part and (os.path.isdir(part) or os.path.isdir(Path(home) / part)):
                candidates.append(part if os.path.isabs(part) else str(Path(home) / part))
        # always include the default dev folder if present
        dev = Path(home) / "Documents" / "Github"
        if dev.is_dir():
            candidates.append(str(dev))

        lines = ["ENVIRONMENT SCAN:"]
        seen = set()
        for root in candidates:
            rp = Path(root)
            if not rp.is_dir() or str(rp) in seen:
                continue
            seen.add(str(rp))
            lines.append(f"\n[{rp}]")
            try:
                entries = sorted(rp.iterdir())[:25]
                for e in entries:
                    kind = "dir" if e.is_dir() else "file"
                    lines.append(f"  ({kind}) {e.name}")
            except OSError:
                lines.append("  (unreadable)")
        self.profile.set_environment_summary("\n".join(lines)[:4000])
        publish("environment_scanned", roots=len(seen))
