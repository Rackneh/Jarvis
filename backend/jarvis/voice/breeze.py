"""Breeze TTS 2 client for the Breeze-TTS-2.cpp GGUF server.

The official BreezeBlue/Breeze-TTS-2 checkpoint wants a CUDA NVIDIA GPU and
~8-14 GiB VRAM. The GGUF reimplementation (HoppouAI/Breeze-TTS-2.cpp) runs the
same model through ggml/Vulkan, so it works on AMD/Intel GPUs and CPU. This
client talks to its local HTTP API (`breeze-server`), streams 24 kHz s16le PCM,
and resamples to Jarvis's 16 kHz pipeline rate.

It can also auto-start `breeze-server` when `voice.breeze_server_exe` and
`voice.breeze_model` point at a local GGUF + binary (see scripts/setup_breeze.py).
"""
from __future__ import annotations

import atexit
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import numpy as np

from .. import config as cfg
from .audio import SAMPLE_RATE

log = logging.getLogger("jarvis.voice.breeze")


class BreezeError(RuntimeError):
    pass


class _Resampler:
    """Streaming linear resampler int16 -> int16 between arbitrary rates."""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self.step = max(1e-6, float(in_rate) / float(out_rate))
        self.pos = 0.0
        self.buf = np.empty(0, dtype=np.float32)

    def feed(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        if pcm.size == 0:
            return None
        x = pcm.astype(np.float32) / 32768.0
        self.buf = x if self.buf.size == 0 else np.concatenate([self.buf, x])
        vals = []
        while self.pos + 1.0 < len(self.buf):
            i = int(self.pos)
            frac = self.pos - i
            vals.append(self.buf[i] * (1.0 - frac) + self.buf[i + 1] * frac)
            self.pos += self.step
            if self.pos > 8192:
                cut = int(self.pos) - 1
                if cut > 0:
                    self.buf = self.buf[cut:]
                    self.pos -= cut
        if not vals:
            return None
        arr = np.asarray(vals, dtype=np.float32)
        return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)

    def flush(self) -> Optional[np.ndarray]:
        if self.buf.size == 0:
            return None
        limit = float(len(self.buf))
        pad = int(np.ceil(self.step)) + 2
        padded = np.concatenate(
            [self.buf, np.full(pad, float(self.buf[-1]), dtype=np.float32)])
        vals = []
        while self.pos < limit:
            i = int(self.pos)
            if i + 1 >= len(padded):
                break
            frac = self.pos - i
            vals.append(padded[i] * (1.0 - frac) + padded[i + 1] * frac)
            self.pos += self.step
        if not vals:
            return None
        arr = np.asarray(vals, dtype=np.float32)
        return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)


class BreezeClient:
    def __init__(self) -> None:
        self._ensure_lock = threading.Lock()
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=10.0),
            trust_env=False,
        )
        self._proc: Optional[subprocess.Popen] = None
        self._ready = False
        self._voice_id: Optional[str] = None
        self._use_ref_upload = False
        self._has_ref = False
        self._ref_path: Optional[Path] = None
        self._ref_text = ""
        atexit.register(self.close)

    # ------------------------------------------------------------------ config

    @property
    def base_url(self) -> str:
        return str(cfg.get("voice.breeze_server_url",
                           "http://127.0.0.1:8137") or
                   "http://127.0.0.1:8137").rstrip("/")

    @property
    def _parsed(self):
        return urlparse(self.base_url)

    @property
    def _port(self) -> int:
        p = self._parsed.port
        if p:
            return p
        return 443 if self._parsed.scheme == "https" else 80

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ------------------------------------------------------------------ server

    def _health(self, timeout: float = 2.0) -> Optional[dict]:
        try:
            r = self._client.get(self._url("/health"), timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def _server_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        exe = str(cfg.get("voice.breeze_server_exe", "") or "").strip()
        model = str(cfg.get("voice.breeze_model", "") or "").strip()
        exe_path = model_path = None
        if exe:
            p = Path(exe)
            if not p.is_absolute():
                p = cfg.REPO_ROOT / p
            if p.is_dir():
                p = p / "breeze-server.exe"
            elif p.suffix == "":
                p = p.with_suffix(".exe")
            exe_path = p if p.exists() else None
        if model:
            p = Path(model)
            if not p.is_absolute():
                p = cfg.REPO_ROOT / p
            model_path = p if p.exists() else None
        return exe_path, model_path

    def _launch_server(self, exe: Path, model: Path) -> bool:
        voices_dir = str(cfg.get("voice.breeze_voices_dir", "") or "").strip()
        if voices_dir:
            vd = Path(voices_dir)
            if not vd.is_absolute():
                vd = cfg.REPO_ROOT / vd
        else:
            vd = cfg.home() / "models" / "breeze" / "voices"
        vd.mkdir(parents=True, exist_ok=True)

        host = self._parsed.hostname or "127.0.0.1"
        args = [str(exe), str(model), "--host", host, "--port", str(self._port),
                "--voices-dir", str(vd)]
        if cfg.get("voice.breeze_cpu", False):
            args.append("--cpu")
        if cfg.get("voice.breeze_te_cpu", False):
            args.append("--te-cpu")
        parallel = int(cfg.get("voice.breeze_parallel", 1) or 1)
        if parallel > 1:
            args += ["--parallel", str(parallel)]

        log_dir = cfg.home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "breeze-server.log"
        try:
            logf = open(log_path, "ab")
        except Exception:
            logf = subprocess.DEVNULL
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                args, stdout=logf, stderr=subprocess.STDOUT,
                creationflags=creationflags)
        except Exception as exc:
            log.warning("breeze-server launch failed: %s", exc)
            try:
                if logf is not subprocess.DEVNULL:
                    logf.close()
            except Exception:
                pass
            return False
        log.info("breeze-server started pid=%s model=%s", self._proc.pid, model)
        return True

    def _wait_health(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._health(timeout=2.0):
                return True
            if self._proc is not None and self._proc.poll() is not None:
                log.warning("breeze-server exited early rc=%s", self._proc.returncode)
                return False
            time.sleep(0.5)
        return False

    # ------------------------------------------------------------------ voice

    def _resolve_ref(self) -> tuple[Optional[Path], str]:
        configured = str(cfg.get("voice.breeze_ref_audio", "") or "").strip()
        candidates = []
        if configured:
            p = Path(configured)
            candidates.append(p if p.is_absolute() else cfg.REPO_ROOT / p)
        voice_dir = cfg.home() / "models" / "voice"
        candidates += [voice_dir / "breeze_reference.wav",
                       voice_dir / "reference.wav"]
        audio = next((p for p in candidates if p.exists()), None)
        if audio is None:
            return None, ""
        texts = [audio.with_suffix(".txt"),
                 audio.parent / "breeze_reference.txt",
                 audio.parent / "reference.txt"]
        if configured:
            texts.insert(0, Path(configured).with_suffix(".txt"))
        text = ""
        override = str(cfg.get("voice.breeze_ref_text", "") or "").strip()
        if override:
            text = override
        else:
            for tp in texts:
                try:
                    if tp.exists():
                        text = tp.read_text(encoding="utf-8").strip()
                        break
                except Exception:
                    continue
        return audio, text

    def _voice_exists(self, voice_id: str) -> bool:
        try:
            r = self._client.get(self._url("/v1/voices"), timeout=10.0)
            if r.status_code != 200:
                return False
            data = r.json()
            items = data if isinstance(data, list) else data.get("voices", [])
            return any(v.get("id") == voice_id for v in items
                       if isinstance(v, dict))
        except Exception:
            return False

    def _register_voice(self, voice_id: str, ref: Path, ref_text: str) -> bool:
        try:
            r = self._client.post(
                self._url("/v1/voices"),
                data={"name": voice_id, "ref_text": ref_text,
                      "description": "Jarvis clone voice"},
                files={"ref_audio": (ref.name, ref.read_bytes(), "audio/wav")},
                timeout=60.0)
            return r.status_code in (200, 201)
        except Exception as exc:
            log.warning("breeze voice registration failed: %s", exc)
            return False

    def _ensure_voice(self) -> None:
        self._ref_path, self._ref_text = self._resolve_ref()
        self._has_ref = bool(self._ref_path and self._ref_text)
        want_id = str(cfg.get("voice.breeze_voice_id", "jarvis") or "").strip()
        self._voice_id = None
        self._use_ref_upload = False
        if want_id:
            if self._voice_exists(want_id):
                self._voice_id = want_id
                return
            if self._has_ref and self._register_voice(want_id, self._ref_path,
                                                      self._ref_text):
                self._voice_id = want_id
                return
            if self._has_ref:
                self._use_ref_upload = True
                return
            self._voice_id = want_id  # let the request 404 -> design fallback
            return
        if self._has_ref:
            self._use_ref_upload = True

    # ------------------------------------------------------------------ ensure

    def ensure(self) -> bool:
        with self._ensure_lock:
            if self._ready:
                return True
            if self._health(timeout=2.0):
                self._ensure_voice()
                self._ready = True
                return True
            if not cfg.get("voice.breeze_auto_start", True):
                return False
            exe, model = self._server_paths()
            if exe is None or model is None:
                log.info("breeze auto-start unavailable (exe=%s model=%s)",
                         exe, model)
                return False
            if not self._launch_server(exe, model):
                return False
            timeout = float(cfg.get("voice.breeze_startup_timeout", 180) or 180)
            if not self._wait_health(timeout):
                return False
            self._ensure_voice()
            self._ready = True
            return True

    # ------------------------------------------------------------------ synth

    def _request_parts(self, text: str):
        data = {"text": text}
        instruction = str(cfg.get("voice.breeze_instruction", "") or "").strip()
        if instruction:
            data["instruction"] = instruction
        try:
            data["cfg_scale"] = str(float(
                cfg.get("voice.breeze_cfg_scale", 1.0) or 1.0))
        except Exception:
            pass
        try:
            seed = int(cfg.get("voice.breeze_seed", 42) or 0)
            if seed:
                data["seed"] = str(seed)
        except Exception:
            pass
        split = cfg.get("voice.breeze_split_chars", None)
        if split is not None:
            try:
                data["split_chars"] = str(int(split))
            except Exception:
                pass
        files = None
        if self._voice_id:
            data["voice_id"] = self._voice_id
        elif self._use_ref_upload and self._ref_path and self._ref_text:
            data["ref_text"] = self._ref_text
            files = {"ref_audio": (self._ref_path.name,
                                   self._ref_path.read_bytes(), "audio/wav")}
        return data, files

    def synthesize_stream(self, text: str):
        """Yield int16 mono 16 kHz PCM blocks as the server streams 24 kHz."""
        if not text.strip():
            return
        if not self.ensure():
            raise BreezeError("breeze server unavailable")
        produced = False
        for attempt in range(3):
            data, files = self._request_parts(text)
            try:
                with self._client.stream(
                        "POST", self._url("/v1/audio/speech"),
                        data=data, files=files) as r:
                    if r.status_code == 404 and self._voice_id:
                        # saved voice went missing: fall back to ref upload/design
                        self._voice_id = None
                        self._use_ref_upload = bool(
                            self._ref_path and self._ref_text)
                        continue
                    if r.status_code == 409 and attempt < 2:
                        time.sleep(0.4 * (attempt + 1))
                        continue
                    if r.status_code != 200:
                        body = r.read()[:200]
                        raise BreezeError(
                            f"breeze speech HTTP {r.status_code}: {body!r}")
                    try:
                        sr = int(r.headers.get("x-sample-rate", "24000") or 24000)
                    except Exception:
                        sr = 24000
                    resampler = _Resampler(sr, SAMPLE_RATE)
                    pending = bytearray()
                    for chunk in r.iter_bytes(16384):
                        pending += chunk
                        n = (len(pending) // 2) * 2
                        if n == 0:
                            continue
                        pcm = np.frombuffer(bytes(pending[:n]), dtype=np.int16)
                        del pending[:n]
                        block = resampler.feed(pcm)
                        if block is not None and len(block):
                            produced = True
                            yield block
                    tail = resampler.flush()
                    if tail is not None and len(tail):
                        produced = True
                        yield tail
                    return
            except httpx.HTTPError as exc:
                if produced:
                    raise BreezeError(f"breeze speech stream broke: {exc}") from exc
                if attempt < 2:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                raise BreezeError(f"breeze speech failed: {exc}") from exc
        raise BreezeError("breeze speech failed after retries")

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            self._client.close()
        except Exception:
            pass
