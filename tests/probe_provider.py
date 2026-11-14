import asyncio, sys
sys.path.insert(0, r"C:\Users\Rackneh\Documents\Github\Repositories\Rackneh's Jarvis\backend")
from jarvis.providers.base import ChatMessage, open_chat_stream
from jarvis.providers.registry import LMStudioProvider, _provider_state

async def main():
    p = LMStudioProvider()
    print("base_url:", p.base_url, "| model:", p.model or "(state: %s)" % _provider_state().get("lm_model"))
    msgs = [ChatMessage(role="system", content="You are Jarvis."),
            ChatMessage(role="user", content="Say hello in one short sentence.")]
    from jarvis.conscious.tools import TOOLS
    agen = await open_chat_stream(p, msgs, tools=TOOLS, temperature=0.6, max_tokens=4096)
    n = 0
    async for chunk in agen:
        n += 1
        if chunk.content or chunk.tool_calls or chunk.finish_reason:
            print(f"chunk#{n}: content={chunk.content!r} finish={chunk.finish_reason} tool_calls={chunk.tool_calls}")
    print("total chunks:", n)

asyncio.run(main())
