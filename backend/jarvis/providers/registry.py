"""Provider registry: Dory (llama.cpp, :8080) <-> LM Studio (:1234).

Reuses the proven pattern from the IBA backend:
- config persisted at ~/.jarvis/provider.json (+ env overrides)
- config.json providers.default is "lmstudio"; "auto" resolves LM Studio-first
  with health probing
- switch_backend() makes the two backends GPU-exclusive: unloading LM Studio
  models when starting the Dory engine, and stopping the Dory engine when
  loading LM Studio. Engine lifecycle goes through the Dory GUI wrapper
  (:5000) so its logs/status stay authoritative.
- DynamicProvider re-resolves on every session() so a switch takes effect at
  the next turn without restarts.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .. import config as cfg
from ..events import publish
from .base import Provider
from .openai_compat import OpenAICompatProvider

PROVIDER_CFG_PATH = Path.home() / ".jarvis" / "provider.json"
_lock = threading.Lock()


# ---------------------------------------------------------------- persisted state

def _provider_state() -> dict:
    try:
        with open(PROVIDER_CFG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"provider": cfg.get("providers.default", "lmstudio"), "lm_model": ""}


def _write_provider_state(state: dict) -> None:
    PROVIDER_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROVIDER_CFG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, PROVIDER_CFG_PATH)


def selected_kind() -> str:
    env = os.environ.get("JARVIS_PROVIDER")
    if env:
        return env
    return _provider_state().get(
        "provider", cfg.get("providers.default", "lmstudio"))


def set_selected_kind(kind: str) -> None:
    kind = kind.lower().strip()
    if kind not in ("auto", "dory", "lmstudio"):
        raise ValueError(kind)
    with _lock:
        st = _provider_state()
        st["provider"] = kind
        _write_provider_state(st)
    publish("provider_changed", provider=kind)


# ---------------------------------------------------------------- concrete providers

class DoryProvider(OpenAICompatProvider):
    kind = "dory"
    display_name = "Dory 1.0 (llama.cpp)"
    reasoning_support = True

    def __init__(self) -> None:
        d = cfg.section("providers").get("dory", {})
        super().__init__(
            base_url=d.get("base_url", "http://127.0.0.1:8080"),
            model=d.get("model", "dory"),
            api_key=d.get("api_key", "dory"),
        )


class LMStudioProvider(OpenAICompatProvider):
    kind = "lmstudio"
    display_name = "LM Studio"
    reasoning_support = False

    def __init__(self) -> None:
        d = cfg.section("providers").get("lmstudio", {})
        st = _provider_state()
        super().__init__(
            base_url=d.get("base_url", "http://127.0.0.1:1234"),
            model=d.get("model") or st.get("lm_model") or "",
            api_key=d.get("api_key", "lm-studio"),
        )
        self.inference_slots = 1


# ---------------------------------------------------------------- probing / switching

def _root(base_url: str) -> str:
    """Strip a trailing /v1 so server-specific endpoints (props, api/v0) build correctly."""
    return base_url.rstrip("/").removesuffix("/v1").removesuffix("/v1") or base_url


def _auth(d: dict) -> dict:
    key = d.get("api_key")
    return {"Authorization": f"Bearer {key}"} if key else {}


async def _dory_up() -> bool:
    d = cfg.section("providers").get("dory", {})
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{_root(d.get('base_url', 'http://127.0.0.1:8080'))}/props",
                                 headers=_auth(d), timeout=5)
            return r.status_code == 200
    except Exception:
        return False


async def _lmstudio_up() -> bool:
    d = cfg.section("providers").get("lmstudio", {})
    root = _root(d.get("base_url", "http://127.0.0.1:1234"))
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{root}/api/v0/models", headers=_auth(d))
            return r.status_code == 200
    except Exception:
        # v0 not present on old versions; fall back to v1 / OpenAI route
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{root}/v1/models", headers=_auth(d))
                return r.status_code == 200
        except Exception:
            return False


async def effective_kind() -> str:
    want = selected_kind()
    dory = await _dory_up()
    lm = await _lmstudio_up()
    if want == "dory":
        return "dory" if dory else ("lmstudio" if lm else "none")
    if want == "lmstudio":
        return "lmstudio" if lm else ("dory" if dory else "none")
    # auto: LM Studio first (the configured default), Dory as fallback
    if lm:
        return "lmstudio"
    if dory:
        return "dory"
    return "none"


def _make(kind: str) -> Provider:
    if kind == "dory":
        return DoryProvider()
    if kind == "lmstudio":
        return LMStudioProvider()
    raise RuntimeError(f"no provider available (wanted {kind!r})")


class DynamicProvider(Provider):
    """Facade that re-resolves the concrete backend on every chat_stream call."""

    kind = "dynamic"
    display_name = "dynamic"

    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(4)

    async def chat_stream(self, messages, tools=None, temperature=0.6,
                          max_tokens=None, cancel=None):
        kind = await effective_kind()
        if kind == "none":
            await asyncio.sleep(1.0)
            kind = await effective_kind()
        provider = _make(kind)
        from .base import open_chat_stream
        async with self._sem:
            agen = await open_chat_stream(
                provider, messages, tools=tools, temperature=temperature,
                max_tokens=max_tokens, cancel=cancel)
            async for chunk in agen:
                yield chunk

    async def available(self) -> bool:
        return (await effective_kind()) != "none"


# ---------------------------------------------------------------- engine lifecycle

async def _gui_call(method: str, path: str, body: Optional[dict] = None) -> tuple[int, dict]:
    url = cfg.get("providers.dory_gui_url", "http://127.0.0.1:5000")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if method == "GET":
                r = await client.get(f"{url}{path}")
            else:
                r = await client.post(f"{url}{path}", json=body or {})
            try:
                return r.status_code, r.json()
            except Exception:
                return r.status_code, {}
    except Exception:
        return 0, {}


async def _unload_lmstudio_models() -> None:
    """Ask LM Studio (JIT loader) to release every loaded model."""
    d = cfg.section("providers").get("lmstudio", {})
    base = _root(d.get("base_url", "http://127.0.0.1:1234"))
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.get(f"{base}/api/v0/models", headers=_auth(d))
            models = r.json().get("data", [])
        except Exception:
            return
        for m in models:
            if m.get("state") == "loaded":
                try:
                    await client.post(f"{base}/api/v0/model/unload",
                                      json={"model": m.get("id")},
                                      headers=_auth(d))
                except Exception:
                    pass


async def _stop_dory_engine() -> None:
    await _gui_call("POST", "/api/server/stop")


async def _start_dory_engine() -> bool:
    status, _ = await _gui_call("GET", "/api/status")
    if status == 200:
        return True
    status, _ = await _gui_call("POST", "/api/server/start")
    if status in (200, 202):
        for _ in range(60):
            await asyncio.sleep(1)
            if await _dory_up():
                return True
    return False


def ensure_fast_lmstudio_model(
    model_name: str = "gemma-4-e4b-uncensored-hauhaucs-aggressive",
    ctx_length: int = 8192,
) -> bool:
    """Ensure LM Studio has Gemma 4 loaded with optimal performance flags
    (max GPU offload, 8192 context, parallel 1, no vision projector) to prevent
    VRAM spillover and KV-cache thrashing.
    """
    import shutil
    import subprocess

    lms_bin = shutil.which("lms") or r"C:\Users\Rackneh\.lmstudio\bin\lms.exe"
    if not os.path.exists(lms_bin) and not shutil.which("lms"):
        return False

    try:
        res = subprocess.run([lms_bin, "ps", "--json"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            try:
                loaded_models = json.loads(res.stdout)
                for m in loaded_models:
                    if m.get("modelKey") == model_name or m.get("identifier") == model_name:
                        if m.get("contextLength") == ctx_length and m.get("parallel") == 1:
                            return True
                        subprocess.run([lms_bin, "unload", model_name], capture_output=True, timeout=10)
                        break
            except Exception:
                pass

        load_res = subprocess.run(
            [lms_bin, "load", model_name, "--gpu", "max", "-c", str(ctx_length), "--parallel", "1", "-y"],
            capture_output=True, text=True, timeout=45
        )
        return load_res.returncode == 0
    except Exception:
        return False


async def switch_backend(target: str, reason: str = "") -> dict:
    """GPU-exclusive switch: 'dory' or 'lmstudio'.

    Returns a receipt; the GUI and conscious loop surface this to the user.
    """
    target = target.lower().strip()
    if target not in ("dory", "lmstudio"):
        return {"ok": False, "error": f"unknown backend {target!r}"}

    publish("backend_switching", target=target, reason=reason)
    if target == "dory":
        await _unload_lmstudio_models()
        ok = await _start_dory_engine()
        if ok:
            set_selected_kind("dory")
            receipt = {"ok": True, "backend": "dory", "reason": reason}
        else:
            receipt = {"ok": False, "error": "Dory engine failed to start"}
    else:
        await _stop_dory_engine()
        # Give the driver a moment to release VRAM.
        for _ in range(5):
            if not await _dory_up():
                break
            await asyncio.sleep(1)
        kind = "lmstudio" if await _lmstudio_up() else "none"
        if kind == "lmstudio":
            target_model = str(cfg.get("providers.lmstudio.model", "gemma-4-e4b-uncensored-hauhaucs-aggressive"))
            await asyncio.to_thread(ensure_fast_lmstudio_model, target_model)
            set_selected_kind("lmstudio")
            receipt = {"ok": True, "backend": "lmstudio", "reason": reason}
        else:
            receipt = {"ok": False,
                       "error": "LM Studio server not reachable on :1234"}
    publish("backend_switched", **receipt)
    return receipt


# ---------------------------------------------------------------- registry API

class ProviderRegistry:
    def __init__(self) -> None:
        self._cache: Dict[str, Provider] = {}

    def get(self, kind: Optional[str] = None) -> Provider:
        if kind is None or kind == "auto":
            return DynamicProvider()
        if kind not in self._cache:
            self._cache[kind] = _make(kind)
        return self._cache[kind]


_registry = ProviderRegistry()


def get_registry() -> ProviderRegistry:
    return _registry


async def provider_status() -> dict:
    dory = await _dory_up()
    lm = await _lmstudio_up()
    eff = await effective_kind()
    lm_models: List[str] = []
    if lm:
        d = cfg.section("providers").get("lmstudio", {})
        base = _root(d.get("base_url", "http://127.0.0.1:1234"))
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                r = await client.get(f"{base}/api/v0/models", headers=_auth(d))
                lm_models = [m.get("id", "?") for m in r.json().get("data", [])
                             if m.get("state") == "loaded"]
        except Exception:
            pass
    return {
        "selected": selected_kind(),
        "effective": eff,
        "dory_up": dory,
        "lmstudio_up": lm,
        "lmstudio_loaded": lm_models,
    }
