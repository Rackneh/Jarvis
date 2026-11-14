"""OpenAI-compatible streaming chat against llama-server (Dory) or LM Studio.

Both engines speak /v1/chat/completions; the difference is housekeeping:
- Dory: `delta.reasoning_content` carries chain-of-thought; we surface it as
  reasoning chunks so the GUI can show the thought stream.
- LM Studio: model may need JIT load; requests include an explicit model name
  ("" lets LM Studio use its loaded/default model).
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Dict, List, Optional

import httpx

from .. import config as cfg
from .base import ChatMessage, Provider, StreamChunk, ToolCall


def _tool_call_buffers() -> Dict[str, ToolCall]:
    return {}


class OpenAICompatProvider(Provider):
    reasoning_support: bool = False

    async def _list_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                r = await client.get(f"{self.base_url}/models", headers=self._headers())
                r.raise_for_status()
                return [m.get("id", "") for m in r.json().get("data", [])]
        except Exception:
            return []

    async def available(self) -> bool:
        return bool(await self._list_models())

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        cancel=None,
        reasoning: Optional[bool] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Implemented as a plain async generator — open_chat_stream() normalizes."""
        reasoning_on = bool(cfg.get("conscious.reasoning_enabled", False)) if reasoning is None else bool(reasoning)
        payload: Dict = {
            "model": self.model or "local-model",
            "messages": [
                {"role": m.role, "content": m.content, **({"name": m.name} if m.name else {})}
                for m in messages
            ],
            "temperature": temperature,
            "stream": True,
        }
        if not reasoning_on:
            payload["reasoning"] = {"effort": "none"}
            payload["reasoning_effort"] = "none"
            payload["reasoning_budget_tokens"] = 0
            payload["chat_template_kwargs"] = {
                "enable_thinking": False,
                "thinking": False,
                "medium_effort": False,
            }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        tool_bufs: Dict[int, ToolCall] = {}
        flushed_tools = False
        in_think = False
        timeout = httpx.Timeout(connect=10, read=None, write=30, pool=30)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers(), timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if cancel is not None and cancel.is_set():
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta", {}) or {}
                    if rc := (delta.get("reasoning_content") or delta.get("reasoning")):
                        yield StreamChunk(reasoning=rc)
                    if c := delta.get("content"):
                        if "<think>" in c:
                            pre, post = c.split("<think>", 1)
                            if pre:
                                yield StreamChunk(content=pre)
                            in_think = True
                            c = post
                        if "</think>" in c:
                            pre, post = c.split("</think>", 1)
                            if pre:
                                yield StreamChunk(reasoning=pre)
                            in_think = False
                            if post:
                                yield StreamChunk(content=post)
                            continue
                        if in_think:
                            yield StreamChunk(reasoning=c)
                        else:
                            yield StreamChunk(content=c)
                    for tc in delta.get("tool_calls", []) or []:
                        idx = tc.get("index", 0)
                        buf = tool_bufs.setdefault(idx, ToolCall(id="", name="", arguments=""))
                        if tid := tc.get("id"):
                            buf.id = tid
                        fn = tc.get("function", {}) or {}
                        if n := fn.get("name"):
                            buf.name = n
                        if a := fn.get("arguments"):
                            buf.arguments += a
                    if fr := choice.get("finish_reason"):
                        flushed_tools = True
                        yield StreamChunk(finish_reason=fr,
                                          tool_calls=list(tool_bufs.values()))
                if tool_bufs and not flushed_tools:
                    yield StreamChunk(finish_reason="tool_calls",
                                      tool_calls=list(tool_bufs.values()))
