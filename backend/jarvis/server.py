"""Jarvis server: WebSocket event channel + REST control surface.

Everything the GUI needs flows through one WebSocket (/ws) fed by the event
bus; REST endpoints are for actions (send text, manage tasks, switch backend,
push-to-talk, rename).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config as cfg
from .conscious.agent import agent
from .events import EventBus, bus, publish
from .memory import store
from .memory.profile import get_profile
from .onboarding.interview import Onboarder
from .providers import registry
from .tasks.engine import engine
from .voice.audio import MicSource, SD_OK, list_output_devices
from .voice.session import session

app = FastAPI(title="Jarvis", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])


class Utterance(BaseModel):
    text: str
    speak: bool = True


class TaskCreate(BaseModel):
    objective: str
    title: str | None = None


class TaskAnswer(BaseModel):
    task_id: str
    answer: str


class BackendSwitch(BaseModel):
    target: str


class Rename(BaseModel):
    name: str


class PTT(BaseModel):
    action: str  # start | stop


class ConfigUpdate(BaseModel):
    settings: dict


@app.on_event("startup")
async def startup() -> None:
    loop = asyncio.get_event_loop()
    bus.attach_loop(loop)
    agent.attach_engine(engine)
    session.attach(agent, loop)
    onboarder = Onboarder(loop)
    app.state.onboarder = onboarder
    onboarder.start()
    engine.start(bridge=_Bridge())
    if registry.selected_kind() in ("lmstudio", "auto"):
        target_model = str(cfg.get("providers.lmstudio.model", "gemma-4-e4b-uncensored-hauhaucs-aggressive"))
        loop.run_in_executor(None, registry.ensure_fast_lmstudio_model, target_model)
    publish("server_started", version="0.1.0",
            assistant=cfg.assistant_name())
    try:
        session.start()
    except Exception as exc:  # voice unavailable (no mic lib) — console still works
        publish("voice_disabled", error=str(exc))


class _Bridge:
    """Relays task questions, completions, and problems into the conscious layer, GUI & voice."""

    @staticmethod
    def _concise(text: str, limit: int = 240) -> str:
        clean = " ".join((text or "").split())
        if len(clean) <= limit:
            return clean
        cut = clean[:limit]
        for sep in (". ", "! ", "? "):
            idx = cut.rfind(sep)
            if idx >= 40:
                return cut[: idx + 1]
        return cut.rstrip() + "..."

    def _relay_speech(self, task_id: str, text: str) -> None:
        try:
            store.add_message(agent.channel, "assistant", text)
        except Exception:
            pass
        publish("assistant_sentence", text=text, from_task=task_id)
        try:
            if getattr(session, "speak_text", None) is not None:
                session.speak_text(text)
            elif getattr(session, "speech", None) is not None:
                session.speech.speak(text)
        except Exception:
            pass

    def on_task_question(self, task_id: str, title: str, question: str, options: list | None = None) -> None:
        publish("task_question_relay", task_id=task_id, title=title,
                question=question, options=options or [])
        opts_hint = ""
        if options:
            opts_hint = " Options are: " + ", ".join(str(o) for o in options[:4]) + "."
        spoken = f"Sir, the agent working on {title} has a question: {self._concise(question, 220)}{opts_hint}"
        self._relay_speech(task_id, spoken)

    def on_task_completed(self, task_id: str, title: str, result: str) -> None:
        publish("task_completion_relay", task_id=task_id, title=title, result=result)
        summary = self._concise(result, 240)
        spoken = f"Sir, the task {title} is complete. {summary}" if summary else f"Sir, the task {title} is complete."
        self._relay_speech(task_id, spoken)

    def on_task_problem(self, task_id: str, title: str, reason: str) -> None:
        publish("task_problem_relay", task_id=task_id, title=title, error=reason)
        spoken = f"Sir, the agent working on {title} ran into a problem: {self._concise(reason, 220)}"
        self._relay_speech(task_id, spoken)


@app.on_event("shutdown")
async def shutdown() -> None:
    session.stop()
    try:
        session.tts_engine.close()
    except Exception:
        pass
    engine.stop()


# ------------------------------------------------------------------ websocket

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    q: asyncio.Queue = bus.subscribe()
    try:
        # initial snapshot for late joiners
        await ws.send_json({"type": "snapshot", "data": await _snapshot()})
        getter = asyncio.create_task(ws.receive_text())
        sender = asyncio.create_task(_pump(q, ws))
        done, pending = await asyncio.wait(
            {getter, sender}, return_when=asyncio.FIRST_COMPLETED)
        for p in pending:
            p.cancel()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        bus.unsubscribe(q)


async def _pump(q: asyncio.Queue, ws: WebSocket) -> None:
    while True:
        ev = await q.get()
        await ws.send_json(ev.as_dict())


async def _snapshot() -> dict:
    return {
        "assistant": cfg.assistant_name(),
        "provider": await registry.provider_status(),
        "conscious_reasoning": bool(cfg.get("conscious.reasoning_enabled", False)),
        "tasks": engine.snapshot(),
        "profile_progress": get_profile().progress(),
        "voice_state": session.state.value,
        "voice_engines": {
            "tts": session.tts_engine.engine_name,
            "tts_provider": getattr(session.tts_engine, "provider_name", "CPU"),
            "tts_cache": session.tts_engine.cache.stats() if getattr(session.tts_engine, "cache", None) else None,
            "stt": session.stt.engine_name,
            "wake": session.wakeword.available,
            "mic_frames": getattr(session, "_mic_frames", 0),
            "process_frames": getattr(session, "_process_frames", 0),
            "speech_speaking": bool(getattr(session.speech, "speaking", False)),
            "tts_pending": int(getattr(session.speech, "pending", 0)),
            "tts_last_error": getattr(session.tts_engine, "last_error", None),
        },
    }


CONFIG_SCHEMA = {
    "assistant.name": str,
    "assistant.persona": str,
    "providers.default": str,
    "providers.lmstudio.model": str,
    "providers.dory.model": str,
    "voice.activation_mode": str,
    "voice.conversation_idle_seconds": float,
    "voice.conversation_exit_phrases": str,
    "voice.wake_word": str,
    "voice.wake_threshold": float,
    "voice.wake_cooldown_seconds": float,
    "voice.stt_engine": str,
    "voice.stt_model": str,
    "voice.stt_moonshine_size": str,
    "voice.stt_partial_interval_ms": int,
    "voice.vad_threshold": float,
    "voice.silence_end_ms": int,
    "voice.min_speech_ms": int,
    "voice.max_utterance_ms": int,
    "voice.tts_engine": str,
    "voice.tts_voice": str,
    "voice.tts_speed": float,
    "voice.tts_clone_reference": str,
    "voice.tts_gpu": bool,
    "voice.jarvis_clone_quality": str,
    "voice.cached_conversation": bool,
    "voice.breeze_server_url": str,
    "voice.breeze_server_exe": str,
    "voice.breeze_model": str,
    "voice.breeze_voices_dir": str,
    "voice.breeze_voice_id": str,
    "voice.breeze_ref_audio": str,
    "voice.breeze_ref_text": str,
    "voice.breeze_instruction": str,
    "voice.breeze_cfg_scale": float,
    "voice.breeze_seed": int,
    "voice.breeze_split_chars": int,
    "voice.breeze_auto_start": bool,
    "voice.breeze_startup_timeout": int,
    "voice.breeze_retry_seconds": int,
    "voice.breeze_cpu": bool,
    "voice.breeze_te_cpu": bool,
    "voice.breeze_parallel": int,
    "voice.echo_guard": bool,
    "voice.push_to_talk_key": str,
    "voice.input_device": int,
    "voice.output_device": int,
    "tasks.max_parallel": int,
    "tasks.max_steps_default": int,
    "conscious.history_messages": int,
    "conscious.response_token_budget": int,
    "conscious.reasoning_enabled": bool,
}

CONFIG_ENUMS = {
    "providers.default": {"auto", "lmstudio", "dory"},
    "voice.activation_mode": {"wake", "ptt", "always", "off", "conversation", "cached_conversation"},
    "voice.stt_engine": {"auto", "faster-whisper", "moonshine"},
    "voice.tts_engine": {"auto", "jarvis_clone", "jarvis_onnx", "xtts", "breeze", "kokoro", "piper", "sapi"},
    "voice.jarvis_clone_quality": {"high", "medium"},
}
CONFIG_NULLABLE = {"voice.input_device", "voice.output_device"}


def _public_config() -> dict:
    return {
        "assistant": {
            "name": cfg.assistant_name(),
            "persona": cfg.get("assistant.persona", ""),
            "first_run": bool(cfg.get("assistant.first_run", True)),
        },
        "providers": {
            "default": cfg.get("providers.default", "lmstudio"),
            "lmstudio_model": cfg.get("providers.lmstudio.model", ""),
            "dory_model": cfg.get("providers.dory.model", ""),
        },
        "voice": {
            "activation_mode": cfg.get("voice.activation_mode", "wake"),
            "conversation_idle_seconds": cfg.get("voice.conversation_idle_seconds", 60),
            "conversation_exit_phrases": cfg.get("voice.conversation_exit_phrases", ""),
            "wake_word": cfg.get("voice.wake_word", ""),
            "wake_threshold": cfg.get("voice.wake_threshold", 0.5),
            "wake_cooldown_seconds": cfg.get("voice.wake_cooldown_seconds", 2.0),
            "stt_engine": cfg.get("voice.stt_engine", "auto"),
            "stt_model": cfg.get("voice.stt_model", "auto"),
            "stt_moonshine_size": cfg.get("voice.stt_moonshine_size", "tiny"),
            "stt_partial_interval_ms": cfg.get("voice.stt_partial_interval_ms", 300),
            "vad_threshold": cfg.get("voice.vad_threshold", 0.45),
            "silence_end_ms": cfg.get("voice.silence_end_ms", 800),
            "min_speech_ms": cfg.get("voice.min_speech_ms", 200),
            "max_utterance_ms": cfg.get("voice.max_utterance_ms", 60000),
            "tts_engine": cfg.get("voice.tts_engine", "kokoro"),
            "tts_voice": cfg.get("voice.tts_voice", "am_michael"),
            "tts_speed": cfg.get("voice.tts_speed", 1.0),
            "tts_clone_reference": cfg.get("voice.tts_clone_reference", ""),
            "tts_gpu": bool(cfg.get("voice.tts_gpu", True)),
            "jarvis_clone_quality": cfg.get("voice.jarvis_clone_quality", "high"),
            "cached_conversation": bool(cfg.get("voice.cached_conversation", True)),
            "breeze_server_url": cfg.get("voice.breeze_server_url",
                                       "http://127.0.0.1:8137"),
            "breeze_server_exe": cfg.get("voice.breeze_server_exe", ""),
            "breeze_model": cfg.get("voice.breeze_model", ""),
            "breeze_voices_dir": cfg.get("voice.breeze_voices_dir", ""),
            "breeze_voice_id": cfg.get("voice.breeze_voice_id", "jarvis"),
            "breeze_ref_audio": cfg.get("voice.breeze_ref_audio", ""),
            "breeze_ref_text": cfg.get("voice.breeze_ref_text", ""),
            "breeze_instruction": cfg.get("voice.breeze_instruction", ""),
            "breeze_cfg_scale": cfg.get("voice.breeze_cfg_scale", 1.0),
            "breeze_seed": cfg.get("voice.breeze_seed", 42),
            "breeze_split_chars": cfg.get("voice.breeze_split_chars"),
            "breeze_auto_start": bool(cfg.get("voice.breeze_auto_start", True)),
            "breeze_startup_timeout": int(cfg.get("voice.breeze_startup_timeout", 180) or 180),
            "breeze_retry_seconds": int(cfg.get("voice.breeze_retry_seconds", 30) or 30),
            "breeze_cpu": bool(cfg.get("voice.breeze_cpu", False)),
            "breeze_te_cpu": bool(cfg.get("voice.breeze_te_cpu", False)),
            "breeze_parallel": int(cfg.get("voice.breeze_parallel", 1) or 1),
            "echo_guard": bool(cfg.get("voice.echo_guard", True)),
            "push_to_talk_key": cfg.get("voice.push_to_talk_key", "F2"),
            "input_device": cfg.get("voice.input_device"),
            "output_device": cfg.get("voice.output_device"),
        },
        "tasks": {
            "max_parallel": int(cfg.get("tasks.max_parallel", 2) or 2),
            "max_steps_default": int(cfg.get("tasks.max_steps_default", 200) or 200),
        },
        "conscious": {
            "history_messages": int(cfg.get("conscious.history_messages", 40) or 40),
            "response_token_budget": int(cfg.get("conscious.response_token_budget", 2048) or 2048),
            "reasoning_enabled": bool(cfg.get("conscious.reasoning_enabled", False)),
        },
    }


def _coerce_setting(key: str, value) -> object:
    expected = CONFIG_SCHEMA[key]
    if expected is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if expected is int:
        return int(float(value))
    if expected is float:
        return float(value)
    return str(value).strip()


# ------------------------------------------------------------------ REST

@app.get("/api/status")
async def status():
    return await _snapshot()


@app.post("/api/say")
async def say(u: Utterance):
    """Text mode: same conscious path as voice, also queueing ordered TTS when voice is active."""
    events = []
    agen = agent.handle_utterance(u.text, source="console")
    async for ev in agen:
        data = {k: v for k, v in ev.items() if k != "type"}
        events.append({"type": ev.get("type", "event"), "data": data, **data})
        if u.speak and ev.get("type") == "sentence" and ev.get("text"):
            try:
                if getattr(session, "speak_text", None) is not None:
                    session.speak_text(ev["text"])
            except Exception:
                pass
    return {"events": events}


@app.get("/api/history")
async def history():
    rows = store.recent_messages("voice", limit=100)
    return rows


@app.get("/api/tasks")
async def tasks():
    return engine.snapshot()


@app.post("/api/tasks")
async def create_task(t: TaskCreate):
    return engine.create(objective=t.objective, title=t.title)


@app.post("/api/tasks/answer")
async def answer_task(a: TaskAnswer):
    ok = engine.deliver_answer(a.task_id, a.answer)
    return {"ok": ok}


@app.post("/api/tasks/{tid}/{action}")
async def task_action(tid: str, action: str):
    if action == "pause":
        engine.pause(tid)
    elif action == "resume":
        engine.resume(tid)
    elif action == "cancel":
        engine.cancel(tid)
    else:
        return {"ok": False, "error": "unknown action"}
    return {"ok": True}


@app.get("/api/provider")
async def get_provider():
    return await registry.provider_status()


@app.post("/api/provider")
async def set_provider(b: BackendSwitch):
    receipt = await registry.switch_backend(b.target, reason="GUI")
    return receipt


@app.get("/api/config")
async def get_config():
    return _public_config()


@app.post("/api/config")
async def update_config(u: ConfigUpdate):
    changed = []
    for key, value in u.settings.items():
        if key not in CONFIG_SCHEMA:
            raise HTTPException(400, f"unknown setting: {key}")
        if value is None and key in CONFIG_NULLABLE:
            coerced = None
        else:
            coerced = _coerce_setting(key, value)
            if key in CONFIG_NULLABLE and coerced == -1:
                coerced = None
        enum = CONFIG_ENUMS.get(key)
        if enum is not None and coerced not in enum:
            raise HTTPException(400, f"invalid value for {key}: {coerced}")
        cfg.update(key, coerced)
        changed.append(key)
    if "assistant.name" in changed:
        publish("assistant_renamed", name=cfg.assistant_name())
    if "providers.default" in changed:
        try:
            registry.set_selected_kind(str(cfg.get("providers.default", "lmstudio")))
        except Exception:
            pass
    if "conscious.reasoning_enabled" in changed:
        publish("conscious_reasoning_changed",
                enabled=bool(cfg.get("conscious.reasoning_enabled", False)))
    session.apply_settings(set(changed))
    return {"ok": True, "changed": changed, "config": _public_config()}


@app.post("/api/rename")
async def rename(r: Rename):
    cfg.set_assistant_name(r.name)
    publish("assistant_renamed", name=r.name)
    return {"ok": True, "name": cfg.assistant_name()}


@app.post("/api/voice/ptt")
async def push_to_talk(p: PTT):
    if p.action == "start":
        return {"ok": session.ptt_start()}
    session.ptt_stop()
    return {"ok": True}


class CacheAction(BaseModel):
    action: str = "seed"  # seed | clear


@app.post("/api/voice/say")
async def voice_say(u: Utterance):
    """Queue text straight into the TTS pipeline (voice-path test)."""
    session.speech.speak(u.text)
    return {"ok": True, "pending": session.speech.pending}


@app.get("/api/voice/cache")
async def voice_cache_status():
    cache = getattr(session.tts_engine, "cache", None)
    return {
        "enabled": session.tts_engine.cached_conversation_enabled(),
        "engine": session.tts_engine.engine_name,
        "provider": getattr(session.tts_engine, "provider_name", "CPU"),
        "stats": cache.stats() if cache is not None else None,
    }


@app.post("/api/voice/cache")
async def voice_cache_action(body: CacheAction):
    cache = getattr(session.tts_engine, "cache", None)
    act = (body.action or "").strip().lower()
    if act in ("off", "disable", "stop"):
        cfg.update("voice.cached_conversation", False)
        session.apply_settings({"voice.cached_conversation"})
        return {"ok": True, "action": act, "enabled": False, "stats": cache.stats() if cache is not None else None}
    if act in ("on", "enable", "start"):
        cfg.update("voice.cached_conversation", True)
        session.apply_settings({"voice.cached_conversation"})
        return {"ok": True, "action": act, "enabled": True, "stats": cache.stats() if cache is not None else None}
    if act == "toggle":
        curr = session.tts_engine.cached_conversation_enabled()
        new_state = not curr
        cfg.update("voice.cached_conversation", new_state)
        session.apply_settings({"voice.cached_conversation"})
        return {"ok": True, "action": act, "enabled": new_state, "stats": cache.stats() if cache is not None else None}
    if cache is None:
        return {"ok": False, "error": "cache unavailable"}
    if act == "clear":
        cache.clear()
        return {"ok": True, "action": "clear", "stats": cache.stats()}
    if act == "seed":
        cache.seed_defaults_async()
        return {"ok": True, "action": "seed", "stats": cache.stats()}
    raise HTTPException(400, f"unknown cache action: {body.action}")


@app.post("/api/voice/mute")
async def mute(on: bool):
    session.set_mic_muted(on)
    return {"ok": True}


@app.get("/api/voice/devices")
async def voice_devices():
    return {
        "ok": SD_OK,
        "input": MicSource.list_devices(),
        "output": list_output_devices(),
    }


@app.get("/api/profile")
async def profile():
    p = get_profile()
    return {"facts": p.facts, "progress": p.progress(),
            "missing": p.missing_slots()}


@app.post("/api/profile/rescan")
async def rescan():
    app.state.onboarder.scan_environment()
    return {"ok": True}


@app.get("/")
async def index():
    return HTMLResponse("<h1>Jarvis backend running</h1>"
                        "<p>GUI: connect via WebSocket /ws</p>")
