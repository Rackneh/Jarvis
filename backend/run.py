"""Entrypoint: python run.py [--cli]

--cli: console chat (no voice) for headless/testing use.
"""
import argparse
import threading

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true",
                        help="console chat loop instead of server")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.cli:
        _run_cli()
        return

    port = args.port or int(__import__("jarvis.config", fromlist=["config"])
                           .get("backend.port", 8010) or 8010)
    uvicorn.run("jarvis.server:app", host="127.0.0.1", port=port, log_level="info")


def _run_cli() -> None:
    import asyncio

    from jarvis import config as cfg
    from jarvis.conscious.agent import agent
    from jarvis.events import bus, publish
    from jarvis.memory import store
    from jarvis.providers.registry import get_registry
    from jarvis.tasks.engine import engine

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bus.attach_loop(loop)
    agent.attach_engine(engine)
    engine.start()

    name = cfg.assistant_name()
    print(f"=== {name} (console mode) — 'exit' to quit, 'tasks' for status ===")
    while True:
        try:
            text = input(f"{name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break
        if text.lower() == "tasks":
            for t in engine.snapshot():
                print(f"[{t['status']:>12}] {t['title']} ({t['progress']}%) id={t['id']}")
            continue
        store.add_message("console", "user", text)
        parts = []

        def pump(agen):
            async def _run():
                async for ev in agen:
                    if ev["type"] == "sentence":
                        parts.append(ev["text"])
            return loop.run_until_complete(_run())

        pump(agent.handle_utterance(text, source="console"))
        print(" ".join(parts) or "(no output)")


if __name__ == "__main__":
    main()
