import asyncio, sys, json
sys.path.insert(0, r"C:\Users\Rackneh\Documents\Github\Repositories\Rackneh's Jarvis\backend")
import httpx
from jarvis import config as cfg

d = cfg.section("providers")["lmstudio"]
payload = {
    "model": d["model"],
    "messages": [
        {"role": "system", "content": "You are Jarvis."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ],
    "temperature": 0.6,
    "stream": True,
    "tools": [{"type": "function", "function": {"name": "remember", "description": "remember a fact", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}}],
    "tool_choice": "auto",
}

async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=None, write=30, pool=30)) as client:
        async with client.stream("POST", f"{d['base_url']}/chat/completions",
                                 json=payload,
                                 headers={"Authorization": f"Bearer {d['api_key']}"},
                                 ) as resp:
            print("HTTP", resp.status_code)
            n = 0
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                n += 1
                if n <= 12:
                    print("LINE:", line[:300])
            print("total lines:", n)

asyncio.run(main())
