"""Chat provider abstraction shared by the conscious loop and task workers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str
    content: str
    name: Optional[str] = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class StreamChunk:
    """A single delta from the model.

    content -> visible text; reasoning -> chain-of-thought (Dory/Nemotron).
    tool_calls accumulate into ToolCall objects.
    """
    content: str = ""
    reasoning: str = ""
    finish_reason: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


async def open_chat_stream(provider: "Provider", messages: List[ChatMessage],
                           **kwargs) -> AsyncIterator[StreamChunk]:
    """Normalize across implementations: some providers implement chat_stream
    as a coroutine returning a generator, others as an async generator."""
    result = provider.chat_stream(messages, **kwargs)
    if hasattr(result, "__anext__"):        # already an async generator
        return result
    return await result                     # coroutine returning one


class Provider:
    """Base class for OpenAI-compatible chat providers."""

    kind: str = "base"
    display_name: str = "base"

    def __init__(self, base_url: str, model: str, api_key: str = "x",
                 context_length: int = 32768):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.context_length = context_length
        self.inference_slots = 1

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        cancel=None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream one assistant turn. `cancel` is an asyncio.Event or None."""
        raise NotImplementedError

    async def count_tokens(self, text: str) -> int:
        # Cheap approximation; Dory exposes /tokenize but the delta rarely matters.
        return max(1, len(text) // 3)

    async def available(self) -> bool:
        raise NotImplementedError
