import asyncio, sys
sys.path.insert(0, r"C:\Users\Rackneh\Documents\Github\Repositories\Rackneh's Jarvis\backend")
from jarvis.providers.base import ChatMessage, open_chat_stream
from jarvis.providers.registry import LMStudioProvider
from jarvis.conscious.tools import TOOLS

SYSTEM = """You are Jarvis, a calm, precise, proactive AI assistant.

AVAILABLE TOOLS - to use one, respond with ONLY a JSON object (no prose):
{"name": "spawn_task", "arguments": {"objective": "<what to do>", "title": "<short title>"}}
Starts a background task. Use when the user asks for work to be done in the background.

VOICE mode rules:
- Keep spoken answers SHORT (2-5 sentences).

When you start a task via spawn_task, confirm to the user in one sentence."""

async def main():
    p = LMStudioProvider()
    msgs = [ChatMessage(role="system", content=SYSTEM),
            ChatMessage(role="user",
                        content="Jarvis, spawn a background task with objective: "
                                "create a file named jarvis_test.txt in your workspace "
                                "containing the word operational. Confirm in one sentence.")]
    agen = await open_chat_stream(p, msgs, tools=TOOLS, temperature=0.6, max_tokens=8192)
    content, reasoning = [], []
    tool_calls = {}
    async for ch in agen:
        if ch.reasoning:
            reasoning.append(ch.reasoning)
        if ch.content:
            content.append(ch.content)
        for tc in ch.tool_calls:
            tool_calls[tc.id or tc.name] = tc
    print("REASONING len:", len("".join(reasoning)))
    print("CONTENT:", "".join(content)[:600])
    print("TOOL_CALLS:", [(tc.name, tc.arguments[:200]) for tc in tool_calls.values()])

asyncio.run(main())
