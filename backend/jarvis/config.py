"""Configuration loading, saving and paths.

The assistant's name lives here: config.json -> assistant.name (default "Jarvis").
Everything that must survive restarts (profile, tasks, memory) lives under home_dir.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.json"

_lock = threading.Lock()


def _default_home() -> Path:
    cfg = _read_raw()
    custom = cfg.get("backend", {}).get("home_dir", "")
    if custom:
        return Path(custom)
    return REPO_ROOT / "jarvis_home"


def _read_raw() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def home() -> Path:
    p = _default_home()
    p.mkdir(parents=True, exist_ok=True)
    return p


def tasks_root() -> Path:
    cfg = _read_raw()
    custom = cfg.get("tasks", {}).get("workspace_root", "")
    root = Path(custom) if custom else home() / "tasks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_path() -> Path:
    return home() / "jarvis.sqlite"


def config() -> dict:
    return _read_raw()


def section(name: str) -> dict:
    return config().get(name, {})


def get(key_path: str, default=None):
    """Dot-separated lookup, e.g. get('voice.tts_engine')."""
    node = config()
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def update(key_path: str, value) -> None:
    """Set a dot-separated key and persist to disk atomically."""
    with _lock:
        cfg = _read_raw()
        node = cfg
        parts = key_path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        import tempfile, time
        fd, tmp_name = tempfile.mkstemp(prefix="config.", suffix=".tmp",
                                        dir=str(CONFIG_PATH.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            last = None
            for _ in range(40):
                try:
                    os.replace(tmp_name, CONFIG_PATH)
                    break
                except PermissionError as exc:
                    last = exc
                    time.sleep(0.025)
            else:
                raise last
        finally:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def assistant_name() -> str:
    return str(get("assistant.name", "Jarvis")) or "Jarvis"


def set_assistant_name(name: str) -> None:
    update("assistant.name", name.strip() or "Jarvis")


def first_run() -> bool:
    return bool(get("assistant.first_run", True))


def complete_first_run() -> None:
    update("assistant.first_run", False)
