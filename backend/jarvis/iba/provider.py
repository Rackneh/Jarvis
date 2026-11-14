"""Exact token accounting and stateless inference against Dory llama-server or LM Studio."""
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse
import json
import os
import re
import httpx
import time
from .protocol import outcome_schema


def is_local_url(url):
    return urlparse(url or "http://127.0.0.1:8080").hostname in ("localhost", "127.0.0.1", "::1")


def provider_config_path():
    return Path.home() / ".dory_llm" / "provider.json"


def provider_config():
    """Resolve which inference backend IBA should use."""
    cfg = {"provider": "auto", "lm_model": "", "lm_api_key": ""}
    cfg_path = provider_config_path()
    try:
        if cfg_path.exists():
            cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))
    except Exception:
        pass
    if os.environ.get("IBA_PROVIDER"):
        cfg["provider"] = os.environ["IBA_PROVIDER"].strip().lower()
    if os.environ.get("IBA_LM_MODEL"):
        cfg["lm_model"] = os.environ["IBA_LM_MODEL"].strip()
    if os.environ.get("LM_STUDIO_API_KEY"):
        cfg["lm_api_key"] = os.environ["LM_STUDIO_API_KEY"].strip()
    return cfg


def write_provider_config(provider, lm_model=None, lm_api_key=None):
    cfg = provider_config()
    cfg["provider"] = provider
    if lm_model is not None:
        cfg["lm_model"] = lm_model
    if lm_api_key is not None:
        cfg["lm_api_key"] = lm_api_key
    path = provider_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def effective_provider_kind():
    kind = provider_config().get("provider", "auto")
    if kind in ("dory", "lmstudio"):
        return kind
    try:
        with httpx.Client(timeout=1.0) as client:
            if client.get("http://127.0.0.1:1234/api/v1/models").status_code in (200, 401):
                return "lmstudio"
    except Exception:
        pass
    try:
        with httpx.Client(timeout=1.0) as client:
            if client.get("http://127.0.0.1:8080/props").status_code == 200:
                return "dory"
    except Exception:
        pass
    return "lmstudio"


class DoryLlamaServer:
    """Direct provider connecting to Dory llama-server on port 8080."""
    def __init__(self, base_url="http://127.0.0.1:8080", model="dory", api_key=""):
        self.base_url = (base_url or "http://127.0.0.1:8080").rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model or "dory"
        self.api_key = api_key

    def inference_kind(self):
        return "dory"

    def inference_slots(self):
        return 1

    @contextmanager
    def session(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        props_url = f"{self.base_url}/props"
        ctx_size = 32768
        alias = self.model
        try:
            with httpx.Client(timeout=4.0) as client:
                resp = client.get(props_url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    ctx_size = data.get("default_generation_settings", {}).get("n_ctx", 32768)
                    alias = data.get("model_alias") or self.model
        except Exception:
            pass

        yield _LoadedDoryServer(self.base_url, alias, ctx_size, self.api_key)


class _LoadedDoryServer:
    def __init__(self, base_url: str, identifier: str, context_length: int, api_key: str = ""):
        self.base_url = base_url
        self.identifier = identifier
        self.context_length = context_length
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _format_messages_text(self, messages: list) -> str:
        text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        return text

    def count(self, messages: list) -> int:
        formatted = self._format_messages_text(messages)
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    f"{self.base_url}/tokenize",
                    json={"content": formatted},
                    headers=self.headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return len(data.get("tokens", []))
        except Exception:
            pass
        return max(1, len(formatted) // 3)

    def predict(self, messages: list, max_tokens: int, on_thought=None, on_content=None,
                reasoning_budget=None, reasoning_budget_message=None) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.identifier,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "stream": True,
            "chat_template_kwargs": {"medium_effort": True},
        }
        if reasoning_budget is not None:
            payload["reasoning_budget_tokens"] = max(32, int(reasoning_budget))
            if reasoning_budget_message:
                payload["reasoning_budget_message"] = reasoning_budget_message

        full_content = []
        full_reasoning = []
        in_think_tag = False
        think_buffer = ""
        token_count = 0
        finish_reason = "stop"

        with httpx.Client(timeout=300.0) as client:
            with client.stream("POST", url, json=payload, headers=self.headers) as resp:
                if resp.status_code != 200:
                    err = resp.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Engine inference error ({resp.status_code}): {err}")

                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choice = chunk.get("choices", [{}])[0]
                        finish = choice.get("finish_reason")
                        if finish:
                            finish_reason = finish
                        delta = choice.get("delta", {})

                        reasoning_chunk = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or ""
                        )
                        content_chunk = delta.get("content") or ""

                        if reasoning_chunk:
                            token_count += 1
                            full_reasoning.append(reasoning_chunk)
                            if on_thought:
                                on_thought(reasoning_chunk)

                        if content_chunk:
                            token_count += 1
                            combined = think_buffer + content_chunk
                            if "<think>" in combined and not in_think_tag:
                                in_think_tag = True
                                parts = combined.split("<think>", 1)
                                if parts[0]:
                                    full_content.append(parts[0])
                                    if on_content:
                                        on_content(parts[0])
                                think_buffer = ""
                                content_chunk = parts[1]

                            if "</think>" in content_chunk:
                                in_think_tag = False
                                parts = content_chunk.split("</think>", 1)
                                if parts[0]:
                                    full_reasoning.append(parts[0])
                                    if on_thought:
                                        on_thought(parts[0])
                                if full_content:
                                    leaked = "".join(full_content)
                                    full_reasoning.append(leaked)
                                    full_content.clear()
                                if parts[1]:
                                    full_content.append(parts[1])
                                    if on_content:
                                        on_content(parts[1])
                                continue

                            if in_think_tag:
                                full_reasoning.append(content_chunk)
                                if on_thought:
                                    on_thought(content_chunk)
                            else:
                                full_content.append(content_chunk)
                                if on_content:
                                    on_content(content_chunk)
                    except Exception:
                        continue

        content_str = "".join(full_content)
        if "</think>" in content_str:
            pre_t, post_t = content_str.rsplit("</think>", 1)
            if pre_t:
                full_reasoning.append(pre_t)
            content_str = post_t.strip()
        reasoning_str = "".join(full_reasoning)
        stop_reason = "stop" if finish_reason in ("stop", "stopStringFound", "eosFound") else finish_reason

        return {
            "content": content_str,
            "reasoning": reasoning_str,
            "stop_reason": stop_reason,
            "output_tokens": token_count,
        }


class LMStudioHTTP:
    """Stateless OpenAI-compatible provider for LM Studio's REST server.

    Each predict() is an independent HTTP request so background agent tasks run
    in parallel with the conscious foreground assistant.
    """
    def __init__(self, base_url="http://127.0.0.1:1234", model="", api_key=""):
        raw_url = (base_url or "http://127.0.0.1:1234").rstrip("/")
        if raw_url.endswith("/v1"):
            raw_url = raw_url[:-3]
        parsed = urlparse(raw_url)
        if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Durable local IBA requires a loopback LM Studio address")
        self.base_url = raw_url
        self.model = model
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def inference_kind(self):
        return "lmstudio"

    def inference_slots(self):
        return 1

    def _find_model(self):
        with httpx.Client(timeout=5.0) as client:
            try:
                resp = client.get(f"{self.base_url}/api/v1/models", headers=self.headers)
                if resp.status_code == 200:
                    wanted = (self.model or "").lower()
                    llms = [m for m in resp.json().get("models", []) if m.get("type") == "llm"]
                    # Prefer loaded instances first when model name is unspecified
                    if not wanted:
                        loaded = [m for m in llms if m.get("loaded_instances")]
                        if loaded:
                            return loaded[0]
                    for m in llms:
                        key = (m.get("key") or "").lower()
                        if not wanted or key == wanted or wanted in key or key in wanted:
                            return m
            except Exception:
                pass
            # Fallback to OpenAI /v1/models
            resp2 = client.get(f"{self.base_url}/v1/models", headers=self.headers)
            resp2.raise_for_status()
            data = resp2.json().get("data", [])
            wanted = (self.model or "").lower()
            for m in data:
                mid = (m.get("id") or "")
                if not wanted or mid.lower() == wanted or wanted in mid.lower() or mid.lower() in wanted:
                    return {"key": mid, "max_context_length": 32768}
            if data:
                return {"key": data[0].get("id") or self.model or "local-model", "max_context_length": 32768}
        return None

    @contextmanager
    def session(self):
        info = self._find_model()
        if not info:
            raise RuntimeError("LM Studio model not found: " + (self.model or "(any)"))
        ctx = int(info.get("max_context_length") or 32768)
        for inst in info.get("loaded_instances") or []:
            ic = inst.get("config", {}).get("context_length") if isinstance(inst, dict) else None
            if ic:
                ctx = min(ctx, int(ic))
        yield _LoadedLMStudioHTTP(self.base_url, info["key"], min(ctx, 32768), self.api_key)


LMStudio = LMStudioHTTP


class _LoadedLMStudioHTTP:
    def __init__(self, base_url, identifier, context_length, api_key=""):
        self.base_url = base_url
        self.identifier = identifier
        self.context_length = context_length
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def count(self, messages):
        total = sum(len(m.get("content") or "") for m in messages)
        return max(1, total // 3)

    def predict(self, messages, max_tokens, on_thought=None, on_content=None,
                reasoning_budget=None, reasoning_budget_message=None):
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.identifier,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if reasoning_budget is not None and int(reasoning_budget) <= 128:
            payload["reasoning"] = {"effort": "none"}
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        full_content = []
        full_reasoning = []
        token_count = 0
        finish_reason = "stop"
        in_think_tag = False
        with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0)) as client:
            with client.stream("POST", url, json=payload, headers=self.headers) as resp:
                if resp.status_code != 200:
                    err = resp.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"LM Studio inference error ({resp.status_code}): {err[:500]}")
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    usage = chunk.get("usage") or {}
                    if usage.get("completion_tokens"):
                        token_count = usage["completion_tokens"]
                    r = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    c = delta.get("content") or ""
                    if r:
                        token_count += 1
                        full_reasoning.append(r)
                        if on_thought:
                            on_thought(r)
                    if c:
                        token_count += 1
                        if "<think>" in c:
                            pre, post = c.split("<think>", 1)
                            if pre:
                                full_content.append(pre)
                                if on_content:
                                    on_content(pre)
                            in_think_tag = True
                            c = post
                        if "</think>" in c:
                            pre, post = c.split("</think>", 1)
                            if pre:
                                full_reasoning.append(pre)
                                if on_thought:
                                    on_thought(pre)
                            in_think_tag = False
                            if post:
                                full_content.append(post)
                                if on_content:
                                    on_content(post)
                            continue
                        if in_think_tag:
                            full_reasoning.append(c)
                            if on_thought:
                                on_thought(c)
                        else:
                            full_content.append(c)
                            if on_content:
                                on_content(c)
        return {
            "content": "".join(full_content),
            "reasoning": "".join(full_reasoning),
            "stop_reason": finish_reason if finish_reason in ("stop", "stopStringFound", "eosFound", "length") else "stop",
            "output_tokens": token_count,
        }


class DynamicProvider:
    """Provider facade that re-resolves the configured backend on every session()."""
    def __init__(self, port: int = 8080, api_key: str = ""):
        self.port = port
        self.api_key = api_key
        self._kind_cache = (0.0, "lmstudio")

    def inference_kind(self):
        now = time.time()
        if now - self._kind_cache[0] > 5.0:
            try:
                self._kind_cache = (now, effective_provider_kind())
            except Exception:
                self._kind_cache = (now, "lmstudio")
        return self._kind_cache[1]

    def inference_slots(self):
        return 1 if self.inference_kind() == "lmstudio" else 1

    @contextmanager
    def session(self):
        with _resolve_provider(self.port, self.api_key).session() as model:
            yield model


class _AsyncRegistryProviderAdapter:
    """Adapter exposing a Jarvis async Provider (e.g. test stubs) to synchronous IBA Runtime."""

    def __init__(self, provider) -> None:
        self._prov = provider
        self.identifier = getattr(provider, "model", "custom")
        self.context_length = int(getattr(provider, "context_length", 32768) or 32768)

    def count(self, messages) -> int:
        return max(1, sum(len(m.get("content") or "") for m in messages) // 3)

    def predict(self, messages, max_tokens, on_thought=None, on_content=None, **_kwargs) -> dict:
        import asyncio
        from ..providers.base import ChatMessage, open_chat_stream

        async def _run():
            msgs = [ChatMessage(role=m["role"], content=m.get("content", "")) for m in messages]
            agen = await open_chat_stream(self._prov, msgs, max_tokens=max_tokens)
            content_parts = []
            reasoning_parts = []
            finish = "stop"
            async for ch in agen:
                if getattr(ch, "reasoning", None):
                    reasoning_parts.append(ch.reasoning)
                    if on_thought:
                        on_thought(ch.reasoning)
                if getattr(ch, "content", None):
                    content_parts.append(ch.content)
                    if on_content:
                        on_content(ch.content)
                if getattr(ch, "finish_reason", None):
                    finish = ch.finish_reason
            return {
                "content": "".join(content_parts),
                "reasoning": "".join(reasoning_parts),
                "stop_reason": finish or "stop",
                "output_tokens": max(1, len("".join(content_parts)) // 4),
            }

        return asyncio.run(_run())


class JarvisIBAProvider:
    """Jarvis-integrated IBA provider that reads Jarvis config.json dynamically
    and executes parallel step requests against LM Studio (or Dory fallback)."""

    def __init__(self) -> None:
        self._kind_cache = (0.0, "lmstudio")

    def _active_kind(self) -> str:
        from .. import config as jcfg
        want = str(jcfg.get("providers.default", "lmstudio") or "lmstudio").lower()
        if want in ("lmstudio", "dory"):
            # Quick check if preferred backend is reachable; fallback cleanly
            lm_cfg = jcfg.section("providers").get("lmstudio", {})
            dory_cfg = jcfg.section("providers").get("dory", {})
            if want == "lmstudio":
                if self._ping_lm(lm_cfg):
                    return "lmstudio"
                if self._ping_dory(dory_cfg):
                    return "dory"
                return "lmstudio"
            else:
                if self._ping_dory(dory_cfg):
                    return "dory"
                if self._ping_lm(lm_cfg):
                    return "lmstudio"
                return "dory"
        return "lmstudio"

    @staticmethod
    def _ping_lm(lm_cfg: dict) -> bool:
        base = (lm_cfg.get("base_url") or "http://127.0.0.1:1234").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        headers = {}
        key = lm_cfg.get("api_key")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            with httpx.Client(timeout=1.5) as client:
                r = client.get(f"{base}/v1/models", headers=headers)
                return r.status_code in (200, 401)
        except Exception:
            return False

    @staticmethod
    def _ping_dory(dory_cfg: dict) -> bool:
        base = (dory_cfg.get("base_url") or "http://127.0.0.1:8080").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        try:
            with httpx.Client(timeout=1.5) as client:
                r = client.get(f"{base}/props")
                return r.status_code == 200
        except Exception:
            return False

    def inference_kind(self):
        now = time.time()
        if now - self._kind_cache[0] > 3.0:
            try:
                self._kind_cache = (now, self._active_kind())
            except Exception:
                self._kind_cache = (now, "lmstudio")
        return self._kind_cache[1]

    def inference_slots(self):
        return 1 if self.inference_kind() == "lmstudio" else 1

    @contextmanager
    def session(self):
        from .. import config as jcfg
        custom = None
        try:
            from ..providers.registry import get_registry
            reg_prov = get_registry().get()
            if getattr(reg_prov, "kind", "dynamic") not in ("dynamic", "lmstudio", "dory"):
                custom = _AsyncRegistryProviderAdapter(reg_prov)
        except Exception:
            custom = None
        if custom is not None:
            yield custom
            return
        kind = self._active_kind()
        self._kind_cache = (time.time(), kind)
        provs = jcfg.section("providers")
        if kind == "dory":
            dc = provs.get("dory", {})
            prov = DoryLlamaServer(
                base_url=dc.get("base_url", "http://127.0.0.1:8080"),
                model=dc.get("model", "dory"),
                api_key=dc.get("api_key", ""),
            )
        else:
            lc = provs.get("lmstudio", {})
            prov = LMStudioHTTP(
                base_url=lc.get("base_url", "http://127.0.0.1:1234"),
                model=lc.get("model", ""),
                api_key=lc.get("api_key", ""),
            )
        with prov.session() as model:
            yield model


def get_default_provider(port: int = 8080, api_key: str = ""):
    return DynamicProvider(port, api_key)


def _resolve_provider(port: int = 8080, api_key: str = ""):
    cfg = provider_config()
    kind = cfg.get("provider", "auto")
    if kind == "lmstudio":
        token = cfg.get("lm_api_key") or (api_key if api_key.startswith("sk-lm") else "")
        return LMStudioHTTP("http://127.0.0.1:1234", model=cfg.get("lm_model", ""), api_key=token)
    try:
        with httpx.Client(timeout=1.0) as client:
            r = client.get("http://127.0.0.1:1234/api/v1/models")
            if r.status_code in (200, 401):
                token = cfg.get("lm_api_key") or (api_key if api_key.startswith("sk-lm") else "")
                return LMStudioHTTP("http://127.0.0.1:1234", model=cfg.get("lm_model", ""), api_key=token)
    except Exception:
        pass
    return DoryLlamaServer(f"http://127.0.0.1:{port}", "dory", api_key)
