"""Conscious agent tools — what the foreground assistant can *do*."""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List

import psutil

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "spawn_task",
            "description": ("Start a long-running background task (multi-hour or "
                            "multi-day is fine). You stay conversational while "
                            "it works; report its id and that it is running."),
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string",
                                  "description": "Full, self-contained objective."},
                    "title": {"type": "string"},
                    "workspace": {"type": "string",
                                  "description": "Optional external project workspace directory path (e.g. D:\\MGS3WaspoTool)."},
                },
                "required": ["objective"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_status",
            "description": ("Get status, progress, and results of active and recent completed "
                            "background tasks. Pass task_id for one task, or omit for recent tasks."),
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_task_question",
            "description": ("Deliver the user's answer to a question a background "
                            "task asked. Only call after the user has answered."),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["task_id", "answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_task",
            "description": "Pause, resume or cancel a background task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "action": {"type": "string", "enum": ["pause", "resume", "cancel"]},
                },
                "required": ["task_id", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": ("Persist a durable fact about the user or environment "
                            "into their profile."),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_backend",
            "description": ("Switch the LLM backend between 'dory' (deep local "
                            "reasoning, llama.cpp) and 'lmstudio' (faster general "
                            "model). Backends are GPU-exclusive; switching handles "
                            "load/unload automatically."),
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "enum": ["dory", "lmstudio"]}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_assistant",
            "description": "Change your own name. Default is Jarvis.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": ("Get this machine's hardware and OS details: CPU, "
                            "GPU, RAM, storage, uptime. Use when the user asks "
                            "what computer/system you are running on, or for "
                            "specs/hardware details."),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ------------------------------------------------------------------ machine info

_CPU_GPU_CACHE: Dict[str, Any] = {"at": 0.0, "cpu": "", "gpu": ""}
_ENV_CACHE: Dict[str, Any] = {"at": 0.0, "text": ""}


def _cpu_gpu_names() -> tuple:
    now = time.time()
    if now - float(_CPU_GPU_CACHE["at"]) < 3600 and _CPU_GPU_CACHE["cpu"]:
        return _CPU_GPU_CACHE["cpu"], _CPU_GPU_CACHE["gpu"]
    cpu = gpu = ""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1 "
             "-ExpandProperty Name); "
             "(Get-CimInstance Win32_VideoController | "
             "Select-Object -ExpandProperty Name)"],
            capture_output=True, text=True, timeout=15)
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        cpu = lines[0] if lines else ""
        gpus = lines[1:]
        # skip virtual/display-link adapters (Meta Virtual Monitor, Parsec,
        # RemoteFX, Indirect Display, basic mirror drivers)
        bad = ("virtual", "remote", "indirect", "parsec", "meta",
               "usb", "basic", "mirror", "idd")
        real = [g for g in gpus
                if not any(b in g.lower() for b in bad)]
        gpu = (real or gpus or [""])[0]
    except Exception:
        pass
    _CPU_GPU_CACHE.update({"at": now, "cpu": cpu or "unknown",
                           "gpu": gpu or "unknown"})
    return _CPU_GPU_CACHE["cpu"], _CPU_GPU_CACHE["gpu"]


def _os_name() -> str:
    try:
        return f"{platform.system()} {platform.release()}".strip()
    except Exception:
        return platform.system() or "unknown OS"


def machine_summary() -> str:
    """One-line machine description for the system prompt (cached ~60 s)."""
    now = time.time()
    if now - float(_ENV_CACHE["at"]) < 60 and _ENV_CACHE["text"]:
        return _ENV_CACHE["text"]
    cpu, gpu = _cpu_gpu_names()
    try:
        cores = psutil.cpu_count(logical=True) or "?"
    except Exception:
        cores = "?"
    try:
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        ram_gb = "?"
    text = (f"{_os_name()}, {cpu} ({cores} threads), {ram_gb} GB RAM, GPU: {gpu}")
    _ENV_CACHE.update({"at": now, "text": text})
    return text


def system_info_text() -> str:
    cpu, gpu = _cpu_gpu_names()
    lines = [
        f"OS: {_os_name()} ({platform.machine()})",
        f"CPU: {cpu}",
        f"GPU: {gpu}",
    ]
    try:
        vm = psutil.virtual_memory()
        lines.append(f"RAM: {round(vm.total / (1024**3), 1)} GB total, "
                     f"{round(vm.available / (1024**3), 1)} GB available")
    except Exception:
        pass
    try:
        for part in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(part.mountpoint)
                lines.append(f"Disk {part.mountpoint}: "
                             f"{round(u.free / (1024**3), 1)} GB free of "
                             f"{round(u.total / (1024**3), 1)} GB")
            except Exception:
                continue
    except Exception:
        pass
    try:
        lines.append(f"Uptime: {str(datetime.now() - datetime.fromtimestamp(
            psutil.boot_time())).split('.')[0]}")
    except Exception:
        pass
    return "\n".join(lines)


class ToolHost:
    """Executes conscious-tool calls; wired to engine/profile/providers."""

    def __init__(self, engine, profile_getter, on_notice=None):
        self.engine = engine
        self.profile = profile_getter
        self.on_notice = on_notice or (lambda text: None)

    async def execute(self, name: str, args: dict) -> str:
        try:
            if name == "spawn_task":
                kwargs = {"objective": args["objective"], "title": args.get("title")}
                if args.get("workspace"):
                    kwargs["workspace"] = args["workspace"]
                task = self.engine.create(**kwargs)
                return (f"Task created: id={task['id']}, title={task['title']!r}. "
                        "It is running in the background. Tell the user it has started.")
            if name == "task_status":
                snap = self.engine.snapshot()
                if args.get("task_id"):
                    snap = [t for t in snap if t["id"] == args["task_id"]]
                else:
                    snap = snap[:8]
                compact = [
                    {
                        **({"id": t.get("id")} if args.get("task_id") else {}),
                        "title": t.get("title"),
                        "status": t.get("status"),
                        "progress": t.get("progress"),
                        "result": t.get("result") or t.get("last_step"),
                        **({"pending_question": t["pending_question"]} if t.get("pending_question") else {}),
                    }
                    for t in snap
                ]
                return json.dumps(compact, indent=1)[:3000]
            if name == "answer_task_question":
                ok = self.engine.deliver_answer(args["task_id"], args["answer"])
                return "delivered." if ok else "no pending question for that task."
            if name == "control_task":
                from ..memory import store as _mem_store
                raw_tid = str(args.get("task_id") or "").strip()
                action = str(args.get("action") or "resume").strip().lower()
                tid = ""
                if raw_tid and raw_tid.lower() not in ("last", "latest", "current"):
                    if _mem_store.get_task(raw_tid):
                        tid = raw_tid
                    else:
                        for t in _mem_store.all_tasks():
                            if t["id"].startswith(raw_tid) or raw_tid.lower() in str(t.get("title") or "").lower():
                                tid = t["id"]
                                break
                if not tid:
                    all_t = _mem_store.all_tasks()
                    open_t = [t for t in all_t if t.get("status") in ("paused", "queued", "running", "waiting_user")]
                    picked = (open_t or all_t or [None])[0]
                    if picked:
                        tid = picked["id"]
                if not tid:
                    return "no matching task found."
                if action == "pause":
                    self.engine.pause(tid)
                elif action == "resume":
                    self.engine.resume(tid)
                else:
                    self.engine.cancel(tid)
                tinfo = _mem_store.get_task(tid) or {}
                return f"task '{tinfo.get('title', tid)}' ({tid}) {action}d (status: {tinfo.get('status', 'unknown')})."
            if name == "remember":
                self.profile().set(args["key"], args["value"], source="conversation")
                return f"remembered {args['key']}."
            if name == "switch_backend":
                from ..providers.registry import switch_backend
                receipt = await switch_backend(args["target"],
                                               reason="conscious tool call")
                return json.dumps(receipt)
            if name == "rename_assistant":
                from .. import config as cfg
                cfg.set_assistant_name(args["name"])
                self.on_notice(f"I'm now called {args['name']}.")
                return f"name set to {args['name']}."
            if name == "system_info":
                return system_info_text()
            if name == "get_current_time":
                return datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
            return f"unknown tool {name}"
        except Exception as exc:  # noqa: BLE001
            return f"tool error: {exc}"
