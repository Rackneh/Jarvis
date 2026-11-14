"""Fast Jarvis-cloned VITS ONNX TTS engine with DirectML GPU acceleration.

Runs the pre-trained MCU J.A.R.V.I.S. VITS ONNX voice models (`jarvis-high.onnx`
/ `jarvis-medium.onnx` from jgkawell/jarvis) via `onnxruntime-directml`
(`DmlExecutionProvider` on AMD/Intel/NVIDIA GPUs, with `CPUExecutionProvider`
fallback) and `espeak-ng` phonemization (`espeakng_loader` + `phonemizer`).

Applies studio Jarvis acoustic conditioning (9.5 kHz warm low-pass filter,
active RMS loudness leveling, soft peak limiting) and resamples to 16 kHz PCM.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config as cfg
from .audio import SAMPLE_RATE


def _log(*args) -> None:
    print("[jarvis-onnx]", *args, file=sys.stderr, flush=True)


def _get_espeak_lock() -> threading.Lock:
    try:
        from kokoro_onnx.tokenizer import _espeak_lock
        return _espeak_lock
    except Exception:
        return _FALLBACK_ESPEAK_LOCK


_FALLBACK_ESPEAK_LOCK = threading.Lock()
_ESPEAK_INIT_LOCK = threading.Lock()
_ESPEAK_INITIALIZED = False
_ORT_LOCK = threading.RLock()
_SESSION_CACHE: dict[tuple[str, bool], tuple[object, str]] = {}


def _ensure_espeak() -> None:
    global _ESPEAK_INITIALIZED
    if _ESPEAK_INITIALIZED:
        return
    with _ESPEAK_INIT_LOCK:
        if _ESPEAK_INITIALIZED:
            return
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        data_path = espeakng_loader.get_data_path()
        lib_path = os.getenv("PHONEMIZER_ESPEAK_LIBRARY") or espeakng_loader.get_library_path()
        with _get_espeak_lock():
            EspeakWrapper.set_data_path(data_path)
            EspeakWrapper.set_library(lib_path)
        _ESPEAK_INITIALIZED = True


class JarvisOnnxEngine:
    """Fast GPU/CPU Jarvis VITS ONNX synthesizer."""

    def __init__(self, quality: Optional[str] = None, use_gpu: Optional[bool] = None) -> None:
        self.quality = (quality or str(cfg.get("voice.jarvis_clone_quality", "high") or "high")).lower()
        self.use_gpu = bool(cfg.get("voice.tts_gpu", True) if use_gpu is None else use_gpu)
        self.model_path, self.config_path = self._resolve_model_paths()
        if self.model_path is None or self.config_path is None:
            raise FileNotFoundError("Jarvis ONNX model files not found in jarvis_home/models/voice")

        with open(self.config_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.sample_rate = int(meta.get("audio", {}).get("sample_rate", 22050))
        self.espeak_voice = str(meta.get("espeak", {}).get("voice", "en-gb-x-rp"))
        self.phoneme_id_map: dict[str, list[int]] = meta["phoneme_id_map"]
        inf = meta.get("inference", {})
        self.noise_scale = float(inf.get("noise_scale", 0.55))
        self.length_scale = float(inf.get("length_scale", 1.10))
        self.noise_w = float(inf.get("noise_w", 0.7))

        _ensure_espeak()
        self._session = None
        self.provider = "CPUExecutionProvider"
        self._synth_lock = threading.Lock()
        self._init_session()

    def _resolve_model_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        voice_dir = cfg.home() / "models" / "voice"
        order = (
            ("jarvis-medium.onnx", "jarvis-high.onnx")
            if self.quality == "medium"
            else ("jarvis-high.onnx", "jarvis-medium.onnx")
        )
        for name in order:
            model = voice_dir / name
            conf = voice_dir / f"{name}.json"
            if model.exists() and model.stat().st_size > 0 and conf.exists():
                return model, conf
        return None, None

    def _make_session(self, providers: list[str]):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if "DmlExecutionProvider" in providers:
            # DirectML handles dynamic VITS shapes best without static memory pattern assumptions
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        else:
            opts.intra_op_num_threads = max(2, min(6, (os.cpu_count() or 4)))
        return ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)

    def _init_session(self) -> None:
        import onnxruntime as ort

        key = (str(self.model_path), bool(self.use_gpu))
        with _ORT_LOCK:
            cached = _SESSION_CACHE.get(key)
            if cached is not None:
                self._session, self.provider = cached
                return

            avail = ort.get_available_providers()
            if self.use_gpu and "DmlExecutionProvider" in avail:
                try:
                    dml_sess = self._make_session(["DmlExecutionProvider", "CPUExecutionProvider"])
                    active = dml_sess.get_providers()
                    if "DmlExecutionProvider" in active:
                        # Verify DirectML runs dynamic VITS inference cleanly and check latency
                        t0 = time.perf_counter()
                        self._run_session(dml_sess, self._text_to_ids("Systems online."), speed=1.0)
                        dml_dt = time.perf_counter() - t0
                        self._session = dml_sess
                        self.provider = "DmlExecutionProvider"
                        _SESSION_CACHE[key] = (self._session, self.provider)
                        _log(f"loaded {self.model_path.name} on DmlExecutionProvider (warmup={dml_dt*1000:.0f} ms)")
                        return
                except Exception as exc:
                    _log(f"DmlExecutionProvider failed ({exc}); falling back to CPUExecutionProvider")

            self._session = self._make_session(["CPUExecutionProvider"])
            self.provider = "CPUExecutionProvider"
            _SESSION_CACHE[key] = (self._session, self.provider)
            _log(f"loaded {self.model_path.name} on CPUExecutionProvider")

    # ------------------------------------------------------------------ phonemize

    def _phonemize(self, text: str) -> str:
        import phonemizer

        clean = re.sub(r"\s+", " ", (text or "").strip())
        if not clean:
            return ""
        _ensure_espeak()
        with _get_espeak_lock():
            ph = phonemizer.phonemize(
                clean,
                language=self.espeak_voice,
                backend="espeak",
                preserve_punctuation=True,
                with_stress=True,
                strip=True,
            )
        return ph.strip()

    def _text_to_ids(self, text: str) -> list[int]:
        phonemes = self._phonemize(text)
        if not phonemes:
            return []
        bos = self.phoneme_id_map.get("^", [1])[0]
        eos = self.phoneme_id_map.get("$", [2])[0]
        pad = self.phoneme_id_map.get("_", [0])[0]

        ids = [bos, pad]
        for ch in unicodedata.normalize("NFD", phonemes):
            mapped = self.phoneme_id_map.get(ch)
            if mapped is None:
                continue
            ids.extend(mapped)
            ids.append(pad)
        ids.append(eos)
        return ids

    # ------------------------------------------------------------------ inference

    def _run_session(self, session, ids: list[int], speed: float = 1.0) -> np.ndarray:
        if not ids:
            return np.zeros(0, dtype=np.float32)
        inp = np.asarray([ids], dtype=np.int64)
        inp_len = np.asarray([len(ids)], dtype=np.int64)
        length_scale = self.length_scale / max(0.5, min(2.0, float(speed or 1.0)))
        scales = np.asarray([self.noise_scale, length_scale, self.noise_w], dtype=np.float32)
        outputs = session.run(
            None,
            {
                "input": inp,
                "input_lengths": inp_len,
                "scales": scales,
            },
        )
        audio = np.asarray(outputs[0], dtype=np.float32).squeeze()
        return audio

    def synthesize(self, text: str, speed: Optional[float] = None) -> Optional[np.ndarray]:
        """Synthesize text to 16 kHz mono int16 PCM."""
        if not text or not text.strip():
            return None
        if speed is None:
            speed = float(cfg.get("voice.tts_speed", 1.0) or 1.0)

        t0 = time.perf_counter()
        ids = self._text_to_ids(text)
        if not ids:
            return None

        with _ORT_LOCK:
            audio = self._run_session(self._session, ids, speed=speed)

        if audio.size == 0:
            return None

        pcm = self._postprocess(audio)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        dur_s = len(pcm) / float(SAMPLE_RATE)
        rtf = (dt_ms / 1000.0) / max(dur_s, 1e-4)
        _log(f"synth {dt_ms:.0f} ms for {dur_s:.2f}s audio (RTF={rtf:.3f}x, {self.provider}): {text[:50]}")
        return pcm

    def _postprocess(self, audio: np.ndarray) -> np.ndarray:
        """Apply Jarvis warm studio conditioning and resample to 16 kHz int16."""
        samples = audio.astype(np.float32) * 32767.0

        try:
            from scipy import signal

            if self.sample_rate > 19500:
                nyq = self.sample_rate * 0.5
                b, a = signal.butter(2, min(9500.0 / nyq, 0.95), btype="low")
                samples = signal.lfilter(b, a, samples).astype(np.float32)

            # Consistent active speech loudness leveling (-20 dBFS target)
            active = samples[np.abs(samples) > 350.0]
            if active.size > 0:
                cur_rms = float(np.sqrt(np.mean(active * active)))
                target_rms = 32768.0 * (10.0 ** (-20.0 / 20.0))
                gain = float(np.clip(target_rms / (cur_rms + 1e-5), 0.5, 2.2))
                samples = samples * gain

            # Soft tanh limiter
            max_val = 30000.0
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            if peak > max_val:
                samples = (np.tanh(samples / max_val) * max_val).astype(np.float32)

            if self.sample_rate != SAMPLE_RATE:
                import math
                g = math.gcd(SAMPLE_RATE, self.sample_rate)
                samples = signal.resample_poly(
                    samples, SAMPLE_RATE // g, self.sample_rate // g
                ).astype(np.float32)
        except Exception:
            if self.sample_rate != SAMPLE_RATE and samples.size > 1:
                n_out = int(len(samples) * SAMPLE_RATE / self.sample_rate)
                idx = np.linspace(0, len(samples) - 1, max(1, n_out))
                samples = np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)

        return np.clip(samples, -32768.0, 32767.0).astype(np.int16)
