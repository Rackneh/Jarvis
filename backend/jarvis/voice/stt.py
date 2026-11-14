"""Speech-to-text: fast finals + rolling partials.

Primary: moonshine-voice (ONNX Runtime) — the tiny-streaming English model
transcribes a 4 s utterance in ~0.4 s on CPU, roughly an order of magnitude
faster than faster-whisper. Fallback: faster-whisper (CTranslate2).

Finals and partials use SEPARATE model instances and are dispatched on
separate executors (see session.py) so a final transcription never queues
behind an in-flight partial.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from .. import config as cfg

SAMPLE_RATE = 16000

_MOONSHINE_ARCHES = {
    "tiny": "TINY_STREAMING",
    "small": "SMALL_STREAMING",
    "medium": "MEDIUM_STREAMING",
    "base": "BASE",
}


class STTEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mv: dict = {}          # slot -> moonshine Transcriber
        self._mv_loading: set = set()
        self._mv_failed = False
        self._mv_ok = False
        self._fw: dict = {}          # slot -> WhisperModel
        self._fw_loading: set = set()
        self._fw_failed = False
        self._fw_name = None
        self._fw_ok = False
        self._init_backends()

    def _init_backends(self) -> None:
        wanted = str(cfg.get("voice.stt_engine", "auto") or "auto").lower()
        if wanted in ("moonshine", "moonshine-voice", "auto"):
            try:
                import moonshine_voice  # noqa: F401
                self._mv_ok = True
            except Exception:
                self._mv_ok = False
        if wanted in ("faster-whisper", "whisper", "auto", "moonshine",
                      "moonshine-voice"):
            try:
                import faster_whisper  # noqa: F401
                self._fw_ok = True
            except Exception:
                self._fw_ok = False

    # ------------------------------------------------------------- loaders

    def _ensure_mv(self, slot: str) -> None:
        if (not self._mv_ok or self._mv_failed or slot in self._mv or
                slot in self._mv_loading):
            return
        with self._lock:
            if slot in self._mv or slot in self._mv_loading:
                return
            self._mv_loading.add(slot)
        try:
            from moonshine_voice import get_model_for_language
            from moonshine_voice.moonshine_api import ModelArch
            from moonshine_voice.transcriber import Transcriber
            size = str(cfg.get("voice.stt_moonshine_size", "tiny") or "tiny").lower()
            arch_name = _MOONSHINE_ARCHES.get(size, "TINY_STREAMING")
            arch = getattr(ModelArch, arch_name, ModelArch.TINY_STREAMING)
            path, resolved = get_model_for_language("en", arch)
            tr = Transcriber(path, resolved)
            with self._lock:
                self._mv[slot] = tr
        except Exception:
            self._mv_failed = True
        finally:
            with self._lock:
                self._mv_loading.discard(slot)

    def _ensure_fw(self, slot: str = "final") -> None:
        if (not self._fw_ok or self._fw_failed or slot in self._fw or
                slot in self._fw_loading):
            return
        with self._lock:
            if slot in self._fw or slot in self._fw_loading:
                return
            self._fw_loading.add(slot)
        try:
            from faster_whisper import WhisperModel
            name = cfg.get("voice.stt_model", "auto")
            if name in (None, "", "auto"):
                name = "base.en"
            model = WhisperModel(name, device="auto", compute_type="auto")
            with self._lock:
                self._fw[slot] = model
                self._fw_name = name
        except Exception:
            self._fw_failed = True
        finally:
            with self._lock:
                self._fw_loading.discard(slot)

    def preload(self) -> None:
        def _warm() -> None:
            self._ensure_mv("final")
            self._ensure_mv("partial")
        threading.Thread(target=_warm, name="jarvis-stt-preload",
                         daemon=True).start()

    @property
    def available(self) -> bool:
        return bool(self._mv or self._mv_ok or self._fw or self._fw_ok)

    @property
    def engine_name(self) -> str:
        if self._mv:
            size = str(cfg.get("voice.stt_moonshine_size", "tiny") or "tiny")
            return f"moonshine-voice:{size}"
        if self._mv_ok:
            return "moonshine-voice:loading"
        if self._fw:
            return f"faster-whisper:{self._fw_name or 'lazy'}"
        if self._fw_ok:
            return "faster-whisper:lazy"
        return "none"

    # ------------------------------------------------------------- moonshine

    def _mv_transcribe(self, pcm: bytes, slot: str) -> Optional[str]:
        """Returns transcript, or None if moonshine is unavailable/failed."""
        if not self._mv_ok or self._mv_failed:
            return None
        self._ensure_mv(slot)
        with self._lock:
            tr = self._mv.get(slot)
        if tr is None:
            return None
        try:
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            res = tr.transcribe_without_streaming(arr.tolist(), SAMPLE_RATE)
            lines = getattr(res, "lines", None) or []
            text = " ".join(getattr(l, "text", "") or "" for l in lines)
            return " ".join(text.split())
        except Exception:
            return None

    # ------------------------------------------------------------- whisper

    def _fw_transcribe(self, pcm: bytes, window: Optional[int] = None,
                       slot: str = "final") -> str:
        if window and len(pcm) > window * 2:
            pcm = pcm[-window * 2:]
        self._ensure_fw(slot)
        with self._lock:
            model = self._fw.get(slot)
        if model is None:
            return ""
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            segments, _info = model.transcribe(
                arr, language="en", beam_size=1, vad_filter=False,
                condition_on_previous_text=False)
            return " ".join(s.text for s in segments).strip()
        except Exception:
            return ""

    # ------------------------------------------------------------- public

    def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        txt = self._mv_transcribe(pcm, "final")
        if txt is not None:
            return txt
        return self._fw_transcribe(pcm, window=None, slot="final")

    def partial(self, rolling: bytes, window_seconds: float = 8.0) -> str:
        window = int(window_seconds * SAMPLE_RATE * 2)
        if len(rolling) > window:
            rolling = rolling[-window:]
        txt = self._mv_transcribe(rolling, "partial")
        if txt is not None:
            return txt
        return self._fw_transcribe(rolling, window=int(window_seconds),
                                   slot="partial")
