"""Text-to-speech: sentence-streamed synthesis + Cached Conversation mode.

Engine is chosen by voice.tts_engine:
- jarvis_clone (default): GPU-accelerated (DirectML on AMD/Intel/NVIDIA) Jarvis
  VITS ONNX model (jarvis-high / jarvis-medium), ~0.15x RTF (~0.3-0.5s/sentence),
  falling back to XTTS-v2 / Kokoro / Piper / SAPI if ONNX weights are absent.
- jarvis_onnx: explicit fast Jarvis VITS ONNX engine.
- xtts: Coqui XTTS-v2 zero-shot voice cloning (with disk-cached speaker latents).
- breeze: Breeze-TTS-2.cpp GGUF/Vulkan server.
- kokoro: Kokoro-82M ONNX (includes blended 'jarvis' voice).
- piper / sapi: local fallbacks.

When Cached Conversation mode is enabled (voice.cached_conversation or
activation_mode == "cached_conversation"), sentences, clauses/phrases, and
words are cached per voice signature and automatically reused when covered.
"""
from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config as cfg
from .audio import SAMPLE_RATE, SpeakerSink
from .breeze import BreezeClient
from .cache import TTSConversationCache
from .jarvis_onnx import JarvisOnnxEngine


def _log(*args) -> None:
    print("[tts]", *args, file=sys.stderr, flush=True)


class TTSEngine:
    def __init__(self) -> None:
        self._breeze: Optional[BreezeClient] = None
        self._breeze_failed = False
        self._breeze_retry_after = 0.0
        self._jarvis_onnx: Optional[JarvisOnnxEngine] = None
        self._jarvis_onnx_failed = False
        self._xtts = None
        self._xtts_ref = None
        self._xtts_ref_path = None
        self._xtts_latents = None
        self._xtts_wanted = False
        self._xtts_failed = False
        self._xtts_loading = False
        self._xtts_lock = threading.Lock()
        self._xtts_done = threading.Event()
        self._kokoro = None
        self._voices = None
        self._piper_voice = None
        self._sapi_lock = threading.Lock()
        self._synthesizing = 0
        self._synth_busy_fn = None
        self.last_error: Optional[str] = None
        self.cache: Optional[TTSConversationCache] = None
        self._init_backends()
        self._init_cache()

    def _init_backends(self) -> None:
        wanted = str(cfg.get("voice.tts_engine", "jarvis_clone") or "jarvis_clone").lower()
        if wanted == "breeze":
            self._breeze = BreezeClient()
        if wanted in ("jarvis_clone", "jarvis_onnx", "auto"):
            self._init_jarvis_onnx()
        if wanted == "xtts" or (wanted in ("jarvis_clone", "auto") and self._jarvis_onnx is None):
            self._prepare_xtts()
        if wanted in ("kokoro", "auto", "jarvis_clone", "jarvis_onnx", "breeze", "xtts"):
            self._init_kokoro()
        if wanted in ("piper", "auto", "jarvis_clone", "jarvis_onnx", "kokoro", "breeze", "xtts"):
            self._init_piper()

    def _init_jarvis_onnx(self) -> None:
        try:
            self._jarvis_onnx = JarvisOnnxEngine()
            self._jarvis_onnx_failed = False
        except Exception as exc:
            self._jarvis_onnx = None
            self._jarvis_onnx_failed = True
            _log(f"jarvis_onnx unavailable ({exc})")

    def _cache_signature(self) -> str:
        speed = round(float(cfg.get("voice.tts_speed", 1.0) or 1.0), 2)
        eng = self.engine_name
        if eng in ("jarvis_clone", "jarvis_onnx"):
            q = str(cfg.get("voice.jarvis_clone_quality", "high") or "high").lower()
            return f"jarvis_onnx_{q}_s{speed}"
        if eng == "kokoro":
            v = str(cfg.get("voice.tts_voice", "jarvis") or "jarvis")
            return f"kokoro_{v}_s{speed}"
        if eng == "breeze":
            v = str(cfg.get("voice.breeze_voice_id", "jarvis") or "jarvis")
            return f"breeze_{v}_s{speed}"
        return f"{eng}_s{speed}"

    def _init_cache(self) -> None:
        try:
            self.cache = TTSConversationCache(self._cache_signature())
            if self.engine_name in ("jarvis_clone", "jarvis_onnx", "kokoro", "piper"):
                self.cache.start_background_worker(
                    self._synthesize_raw_single,
                    is_busy_fn=self._is_busy,
                )
        except Exception as exc:
            _log(f"cache init failed: {exc}")
            self.cache = None

    def set_busy_probe(self, fn) -> None:
        self._synth_busy_fn = fn

    @property
    def synthesizing(self) -> bool:
        return self._synthesizing > 0

    def _is_busy(self) -> bool:
        try:
            return self._synthesizing > 0 or bool(self._synth_busy_fn and self._synth_busy_fn())
        except Exception:
            return self._synthesizing > 0

    @staticmethod
    def cached_conversation_enabled() -> bool:
        val = cfg.get("voice.cached_conversation", None)
        if val is not None:
            return bool(val)
        mode = str(cfg.get("voice.activation_mode", "wake") or "wake").lower()
        return mode == "cached_conversation"

    def _xtts_deps_ok(self) -> bool:
        try:
            import importlib.util
            return all(importlib.util.find_spec(m) for m in
                       ("TTS", "torch", "torchaudio"))
        except Exception:
            return False

    def _resolve_xtts_ref(self) -> Optional[Path]:
        configured = cfg.get("voice.tts_clone_reference", "")
        candidates = []
        if configured:
            ref = Path(configured)
            candidates.append(ref if ref.is_absolute() else cfg.REPO_ROOT / ref)
        candidates.append(cfg.home() / "models" / "voice" / "reference.wav")
        ref = next((p for p in candidates if p.exists()), candidates[-1])
        return ref if ref.exists() else None

    def _prepare_xtts(self) -> None:
        """Mark XTTS as available without loading torch/TTS yet."""
        self._xtts_ref_path = self._resolve_xtts_ref()
        self._xtts_wanted = bool(self._xtts_ref_path and self._xtts_deps_ok())
        self._xtts_failed = not self._xtts_wanted

    def _load_or_compute_xtts_latents(self, model) -> Optional[tuple]:
        """Cache XTTS speaker conditioning latents on disk so sentences skip re-encoding."""
        try:
            import torch
            tts_model = getattr(getattr(model, "synthesizer", None), "tts_model", None)
            if tts_model is None or not hasattr(tts_model, "get_conditioning_latents"):
                return None
            ref_path = Path(self._xtts_ref_path)
            ref_mtime = ref_path.stat().st_mtime
            latents_path = cfg.home() / "models" / "voice" / "xtts_latents.pt"
            if latents_path.exists():
                data = torch.load(str(latents_path), map_location="cpu", weights_only=False)
                if data.get("mtime") == ref_mtime and data.get("ref") == str(ref_path):
                    return data["gpt_cond_latent"], data["speaker_embedding"]
            gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
                audio_path=[str(ref_path)]
            )
            torch.save(
                {
                    "mtime": ref_mtime,
                    "ref": str(ref_path),
                    "gpt_cond_latent": gpt_cond_latent.cpu(),
                    "speaker_embedding": speaker_embedding.cpu(),
                },
                str(latents_path),
            )
            return gpt_cond_latent, speaker_embedding
        except Exception as exc:
            _log(f"xtts latent cache skipped: {exc}")
            return None

    def _ensure_xtts(self) -> bool:
        with self._xtts_lock:
            if self._xtts is not None or self._xtts_failed:
                return self._xtts is not None
            if self._xtts_loading:
                waiter = True
            else:
                if self._xtts_ref_path is None or not self._xtts_deps_ok():
                    self._xtts_failed = True
                    return False
                self._xtts_loading = True
                self._xtts_done.clear()
                waiter = False
        if waiter:
            self._xtts_done.wait()
            return self._xtts is not None
        try:
            import os
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            from TTS.api import TTS
            model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            latents = self._load_or_compute_xtts_latents(model)
            with self._xtts_lock:
                self._xtts = model
                self._xtts_ref = str(self._xtts_ref_path)
                self._xtts_latents = latents
        except Exception:
            with self._xtts_lock:
                self._xtts_failed = True
        finally:
            with self._xtts_lock:
                self._xtts_loading = False
                self._xtts_done.set()
        return self._xtts is not None

    def warm_xtts(self) -> None:
        if (not self._xtts_wanted or self._xtts_failed or
                self._xtts is not None or self._xtts_loading):
            return
        threading.Thread(target=self._ensure_xtts,
                         name="jarvis-tts-warm", daemon=True).start()

    def _ensure_breeze(self) -> bool:
        if self._breeze is None:
            return False
        now = time.time()
        if self._breeze_failed and now < self._breeze_retry_after:
            return False
        try:
            ok = self._breeze.ensure()
        except Exception:
            ok = False
        self._breeze_failed = not ok
        if not ok:
            retry = float(cfg.get("voice.breeze_retry_seconds", 30) or 30)
            self._breeze_retry_after = now + max(5.0, retry)
        return ok

    def warm(self) -> None:
        """Pre-warm the preferred engine on wake/listen without blocking."""
        if self._breeze is not None and not self._breeze_failed:
            threading.Thread(target=self._ensure_breeze,
                             name="jarvis-breeze-warm", daemon=True).start()
        elif self._jarvis_onnx is not None:
            if self.cache is not None and self.cached_conversation_enabled():
                self.cache.seed_defaults_async()
        elif self._xtts_wanted and not self._xtts_failed:
            self.warm_xtts()

    def close(self) -> None:
        if self.cache is not None:
            try:
                self.cache.close()
            except Exception:
                pass
        if self._breeze is not None:
            try:
                self._breeze.close()
            except Exception:
                pass

    def _init_kokoro(self) -> None:
        try:
            from kokoro_onnx import Kokoro
            import onnxruntime as ort
            models_dir = cfg.home() / "models"
            model = models_dir / "kokoro-v1.0.onnx"
            voices = models_dir / "voices.bin"
            if model.exists() and voices.exists():
                # Kokoro hits a DML ConvTranspose parameter bug on AMD DirectML;
                # force CPUExecutionProvider explicitly so it never fails.
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                sess = ort.InferenceSession(
                    str(model), sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._kokoro = Kokoro.from_session(sess, str(voices))
        except Exception:
            self._kokoro = None

    def _init_piper(self) -> None:
        try:
            from piper import PiperVoice  # noqa: F401
            self._piper_voice = "available"
        except Exception:
            self._piper_voice = None

    @property
    def engine_name(self) -> str:
        wanted = str(cfg.get("voice.tts_engine", "jarvis_clone") or "jarvis_clone").lower()
        if self._breeze is not None and not self._breeze_failed:
            return "breeze"
        if self._jarvis_onnx is not None and not self._jarvis_onnx_failed:
            return "jarvis_onnx" if wanted == "jarvis_onnx" else "jarvis_clone"
        if self._xtts is not None or (self._xtts_wanted and not self._xtts_failed):
            return "xtts" if wanted == "xtts" else "jarvis_clone"
        if self._kokoro:
            return "kokoro"
        if self._piper_voice:
            return "piper"
        return "sapi"

    @property
    def provider_name(self) -> str:
        eng = self.engine_name
        if eng in ("jarvis_clone", "jarvis_onnx") and self._jarvis_onnx is not None:
            return "DirectML GPU" if self._jarvis_onnx.provider == "DmlExecutionProvider" else "CPU"
        if eng == "breeze":
            return "CPU" if cfg.get("voice.breeze_cpu", False) else "Vulkan GPU"
        return "CPU"

    # ------------------------------------------------------------------ synthesize

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Return int16 mono 16k PCM or None (compatibility helper)."""
        blocks = [b for b in self.synthesize_stream(text)
                  if b is not None and len(b)]
        if not blocks:
            return None
        if len(blocks) == 1:
            return blocks[0]
        return np.concatenate(blocks)

    def synthesize_stream(self, text: str):
        """Yield int16 mono 16k PCM blocks, using Cached Conversation when enabled."""
        if not text.strip():
            return
        if self.cached_conversation_enabled() and self.cache is not None:
            from ..events import publish

            def _on_hit(kind: str, sentence: str) -> None:
                publish("tts_cache_hit", kind=kind, text=sentence[:120],
                        stats=self.cache.stats())

            yield from self.cache.resolve_stream(
                text,
                self._synthesize_raw_stream,
                on_hit=_on_hit,
            )
            return
        yield from self._synthesize_raw_stream(text)

    def _synthesize_raw_stream(self, text: str, background: bool = False):
        """Yield raw int16 mono 16k PCM blocks directly from the active TTS backend."""
        if not text.strip():
            return
        if not background:
            self._synthesizing += 1
        try:
            if self._breeze is not None and not self._breeze_failed:
                try:
                    if self._ensure_breeze():
                        stream = self._breeze.synthesize_stream(text)
                        first = next(stream)
                        yield first
                        yield from stream
                        return
                except StopIteration:
                    self._breeze_failed = True
                    self._breeze_retry_after = time.time() + 30.0
                except Exception:
                    self._breeze_failed = True
                    self._breeze_retry_after = time.time() + 30.0
            pcm = self._synthesize_other(text)
            if pcm is not None and len(pcm):
                yield pcm
        finally:
            if not background:
                self._synthesizing = max(0, self._synthesizing - 1)

    def _synthesize_raw_single(self, text: str) -> Optional[np.ndarray]:
        blocks = [b for b in self._synthesize_raw_stream(text, background=True)
                  if b is not None and len(b)]
        if not blocks:
            return None
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    def _synthesize_other(self, text: str) -> Optional[np.ndarray]:
        if self._jarvis_onnx is not None and not self._jarvis_onnx_failed:
            pcm = self._jarvis_onnx_speak(text)
            if pcm is not None:
                return pcm
        if self._xtts_wanted and not self._xtts_failed and self._ensure_xtts():
            return self._xtts_speak(text)
        if self._kokoro:
            return self._kokoro_speak(text)
        if self._piper_voice:
            return self._piper_speak(text)
        return self._sapi_speak(text)

    def _jarvis_onnx_speak(self, text: str) -> Optional[np.ndarray]:
        try:
            speed = float(cfg.get("voice.tts_speed", 1.0) or 1.0)
            return self._jarvis_onnx.synthesize(text, speed=speed)
        except Exception:
            self.last_error = "jarvis_onnx: " + traceback.format_exc(limit=3)
            _log("jarvis_onnx failed:", self.last_error)
            self._jarvis_onnx_failed = True
            return None

    def _xtts_speak(self, text: str) -> Optional[np.ndarray]:
        t0 = time.time()
        try:
            _log("xtts synth start:", text[:60])
            if self._xtts_latents is not None:
                tts_model = self._xtts.synthesizer.tts_model
                gpt_cond, spk_emb = self._xtts_latents
                out = tts_model.inference(
                    text=text,
                    language="en",
                    gpt_cond_latent=gpt_cond,
                    speaker_embedding=spk_emb,
                )
                wav = np.asarray(out["wav"], dtype=np.float32)
                pcm = np.clip(wav * 32767.0, -32768.0, 32767.0).astype(np.int16)
                rate = 24000
            else:
                import tempfile, os, wave as _wave, io as _io
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    out_path = f.name
                try:
                    self._xtts.tts_to_file(
                        text=text,
                        speaker_wav=self._xtts_ref,
                        language="en",
                        file_path=out_path,
                        split_sentences=False,
                    )
                    buf = _io.BytesIO(open(out_path, "rb").read())
                    with _wave.open(buf, "rb") as w:
                        rate = w.getframerate()
                        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                finally:
                    os.unlink(out_path)
            if rate != SAMPLE_RATE:
                n_out = int(len(pcm) * SAMPLE_RATE / rate)
                idx = np.linspace(0, len(pcm) - 1, n_out)
                pcm = np.interp(idx, np.arange(len(pcm)),
                                pcm.astype(np.float32)).astype(np.int16)
            _log("xtts synth done dt=%.1f samples=%d" % (time.time() - t0, len(pcm)))
            return pcm
        except Exception:
            self.last_error = "xtts: " + traceback.format_exc(limit=3)
            _log("xtts failed:", self.last_error)
            with self._xtts_lock:
                self._xtts = None
                self._xtts_failed = True  # don't retry a broken engine mid-conversation
            if self._kokoro:
                return self._kokoro_speak(text)
            if self._piper_voice:
                return self._piper_speak(text)
            return self._sapi_speak(text)

    def _kokoro_speak(self, text: str) -> Optional[np.ndarray]:
        try:
            voice = cfg.get("voice.tts_voice", "am_michael")
            speed = float(cfg.get("voice.tts_speed", 1.0) or 1.0)
            audio, sr = self._kokoro.create(text, voice=voice, speed=speed, lang="en-us")
            arr = np.asarray(audio, dtype=np.float32)
            if sr != SAMPLE_RATE:
                n_out = int(len(arr) * SAMPLE_RATE / sr)
                idx = np.linspace(0, len(arr) - 1, n_out)
                arr = np.interp(idx, np.arange(len(arr)), arr)
            return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
        except Exception:
            self.last_error = "kokoro: " + traceback.format_exc(limit=3)
            _log("kokoro failed:", self.last_error)
            self._kokoro = None
            if self._piper_voice:
                return self._piper_speak(text)
            return self._sapi_speak(text)

    def _piper_speak(self, text: str) -> Optional[np.ndarray]:
        try:
            from piper import PiperVoice, SynthesisConfig
            import wave as _wave
            import io as _io
            voice_path = cfg.get("voice.piper_voice_path", "")
            if not voice_path:
                return self._sapi_speak(text)
            pv = PiperVoice.load(voice_path)
            cfgobj = SynthesisConfig(length_scale=1.0 / max(0.5, float(cfg.get("voice.tts_speed", 1.0))))
            buf = _io.BytesIO()
            with _wave.Wave_write(buf) as w:
                pv.synthesize(text, w, syn_config=cfgobj)
            data = buf.getvalue()
            pcm = np.frombuffer(data[44:], dtype=np.int16)
            if pv.config.sample_rate != SAMPLE_RATE:
                n_out = int(len(pcm) * SAMPLE_RATE / pv.config.sample_rate)
                idx = np.linspace(0, len(pcm) - 1, n_out)
                pcm = np.interp(idx, np.arange(len(pcm)), pcm.astype(np.float32)).astype(np.int16)
            return pcm
        except Exception:
            self.last_error = "piper: " + traceback.format_exc(limit=3)
            _log("piper failed:", self.last_error)
            return self._sapi_speak(text)

    def _sapi_speak(self, text: str) -> Optional[np.ndarray]:
        """Windows SAPI through PowerShell -> WAV bytes -> PCM."""
        import io as _io
        import wave as _wave
        import tempfile, os
        try:
            with self._sapi_lock:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Rate = {int(max(-2, min(4, (float(cfg.get('voice.tts_speed', 1.0)) - 1) * 3)))}; "
                    f"$s.SetOutputToWaveFile('{wav_path}'); "
                    f"$s.Speak('{text.replace(chr(39), chr(39)*2)}'); "
                    "$s.Dispose()")
                subprocess.run(["powershell", "-NoProfile", "-Command", script],
                               capture_output=True, timeout=60, check=True)
            data = open(wav_path, "rb").read()
            os.unlink(wav_path)
            buf = _io.BytesIO(data)
            with _wave.open(buf, "rb") as w:
                rate = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if rate != SAMPLE_RATE:
                n_out = int(len(pcm) * SAMPLE_RATE / rate)
                idx = np.linspace(0, len(pcm) - 1, n_out)
                pcm = np.interp(idx, np.arange(len(pcm)), pcm.astype(np.float32)).astype(np.int16)
            return pcm
        except Exception:
            self.last_error = "sapi: " + traceback.format_exc(limit=3)
            _log("sapi failed:", self.last_error)
            return None


_CLEAN_MD_RE = re.compile(r"\*\*|__|\*|`|~~|^#+\s*|^>\s*|^\s*[-•]\s+", re.M)
_SENTENCE_BREAK_RE = re.compile(
    r"(?:(?<=[.!?])(?<!\d\.)(?<!\.\.\.)(?<!\b[A-Za-z]\.)\s+(?=[A-Z0-9\"'(\[])|\n+)"
)
_COMPLETE_END_RE = re.compile(r"(?:(?<=[.!?])(?<!\d\.)(?<!\.\.\.)(?<!\b[A-Za-z]\.)\s*$|\n\s*$)")


def clean_for_speech(text: str) -> str:
    """Normalize markdown/quotes/whitespace into clean natural text for TTS."""
    if not text:
        return ""
    cleaned = text.replace('""', '"')
    cleaned = _CLEAN_MD_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+\*\s+", ", ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return re.sub(r"[ \t\r\f\v]+", " ", cleaned).strip()


def split_spoken_sentences(text: str) -> list[str]:
    """Split text into ordered speakable sentences without breaking '1.' or '.bin'."""
    cleaned = clean_for_speech(text)
    if not cleaned:
        return []
    parts = [p.strip() for p in _SENTENCE_BREAK_RE.split(cleaned) if p and p.strip()]
    return parts or ([cleaned] if cleaned else [])


class SpeechQueue:
    """Speaks a stream of sentences strictly in FIFO order; interruptible."""

    def __init__(self, engine: TTSEngine, sink: SpeakerSink) -> None:
        self.engine = engine
        self.sink = sink
        self._q: "queue.Queue[Optional[tuple[int, str]]]" = queue.Queue(maxsize=256)
        self._lock = threading.Lock()
        self._gen = 0
        self._in_flight = 0
        self.speaking = False
        self.engine.set_busy_probe(lambda: self.is_active)
        self._thread = threading.Thread(target=self._worker, name="jarvis-tts",
                                        daemon=True)
        self._thread.start()

    @property
    def pending(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def is_active(self) -> bool:
        with self._lock:
            return bool(
                self.speaking
                or self._in_flight > 0
                or self._q.qsize() > 0
                or self.engine.synthesizing
            )

    def speak(self, sentence: str) -> None:
        parts = split_spoken_sentences(sentence)
        if not parts:
            return
        with self._lock:
            gen = self._gen
            for part in parts:
                try:
                    self._q.put_nowait((gen, part))
                    self._in_flight += 1
                    self.speaking = True
                except queue.Full:
                    _log("SpeechQueue full; dropping:", part[:60])

    def stop(self) -> None:
        """Barge-in: drop everything pending, abort playback."""
        with self._lock:
            self._gen += 1
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
            self._in_flight = 0
            self.speaking = False
        self.sink.stop()

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                continue
            gen, sentence = item
            with self._lock:
                if gen != self._gen:
                    self._in_flight = max(0, self._in_flight - 1)
                    self.speaking = self._in_flight > 0
                    continue
                self.speaking = True
            from ..events import publish
            publish("tts_start", text=sentence[:120])
            t0 = time.time()
            try:
                for block in self.engine.synthesize_stream(sentence):
                    if block is None or len(block) == 0:
                        continue
                    if gen != self._gen:
                        break
                    self.sink.play(block)
                _log("sentence done dt=%.2f engine=%s: %s" % (
                    time.time() - t0, self.engine.engine_name, sentence[:60]))
            except Exception:
                self.engine.last_error = "queue: " + traceback.format_exc(limit=3)
                _log("sentence failed:", self.engine.last_error)
            finally:
                with self._lock:
                    if gen == self._gen:
                        self._in_flight = max(0, self._in_flight - 1)
                    self.speaking = self._in_flight > 0 or self._q.qsize() > 0
                publish("tts_end")


class SpeechStreamer:
    """Feeds raw assistant text into the queue sentence-by-sentence in order."""

    def __init__(self, speech: SpeechQueue) -> None:
        self.speech = speech
        self.buffer = ""

    def feed(self, text: str) -> None:
        cleaned = clean_for_speech(text)
        if not cleaned:
            return
        if self.buffer:
            self.buffer = f"{self.buffer.rstrip()} {cleaned}"
        else:
            self.buffer = cleaned
        parts = split_spoken_sentences(self.buffer)
        if not parts:
            return
        if _COMPLETE_END_RE.search(self.buffer):
            for part in parts:
                self.speech.speak(part)
            self.buffer = ""
        elif len(parts) > 1:
            for part in parts[:-1]:
                self.speech.speak(part)
            self.buffer = parts[-1]

    def flush(self) -> None:
        if self.buffer.strip():
            self.speech.speak(self.buffer.strip())
        self.buffer = ""

    def abort(self) -> None:
        self.buffer = ""
        self.speech.stop()
