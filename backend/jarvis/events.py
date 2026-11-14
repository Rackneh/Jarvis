"""Async event bus.

Every subsystem publishes structured events here; the FastAPI websocket/SSE layers
fan them out to the GUI. One channel, one vocabulary — no polling endpoints.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List


@dataclass
class Event:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "ts": self.ts}


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        # Recent events keep late-joining GUIs coherent.
        self._recent: List[Event] = []
        self._recent_limit = 200

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subscribers.add(q)
        for ev in list(self._recent):
            q.put_nowait(ev)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, type_: str, **data) -> None:
        ev = Event(type=type_, data=data)
        if self._recent and self._recent[-1].ts == ev.ts:
            self._recent[-1] = ev
        self._recent.append(ev)
        if len(self._recent) > self._recent_limit:
            del self._recent[: len(self._recent) - self._recent_limit]

        loop = self._loop
        for q in list(self._subscribers):
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(self._put, q, ev)
            else:
                self._put(q, ev)

    @staticmethod
    def _put(q: asyncio.Queue, ev: Event) -> None:
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait(ev)
            except Exception:
                pass

    async def stream(self) -> AsyncIterator[Event]:
        q = self.subscribe()
        try:
            while True:
                ev = await q.get()
                yield ev
        finally:
            self.unsubscribe(q)


bus = EventBus()


def publish(type_: str, **data) -> None:
    bus.publish(type_, **data)
