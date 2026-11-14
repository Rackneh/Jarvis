"""Microphone capture: 16 kHz mono int16, frame-based, thread -> asyncio bridge."""
from __future__ import annotations

import asyncio
import re
import threading
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
    SD_OK = True
except Exception:  # noqa: BLE001
    sd = None
    SD_OK = False

SAMPLE_RATE = 16000
FRAME_MS = 32  # 512 samples — Silero VAD frame size
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


class MicSource:
    """Continuous capture. `on_frame(bytes)` is called from the audio thread."""

    def __init__(self, on_frame: Callable[[bytes], None],
                 device: Optional[int] = None) -> None:
        self.on_frame = on_frame
        self.device = device
        self._stream: Optional["sd.InputStream"] = None
        self._tail = b""

    def start(self) -> None:
        if not SD_OK:
            raise RuntimeError("sounddevice not available")

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                pass
            pcm = self._tail + indata[:, 0].tobytes()
            need = FRAME_SAMPLES * 2
            while len(pcm) >= need:
                self.on_frame(pcm[:need])
                pcm = pcm[need:]
            self._tail = pcm

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=FRAME_SAMPLES, device=self.device, callback=callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    @staticmethod
    def list_devices() -> list:
        return list_input_devices()


class SpeakerSink:
    """PCM playback (16k mono int16) with level reporting and hard stop."""

    def __init__(self, device: Optional[int] = None) -> None:
        self.device = device
        self._stream: Optional["sd.OutputStream"] = None
        self._lock = threading.Lock()
        self._stop_flag = False
        self.level = 0.0

    def _ensure(self) -> None:
        if self._stream is not None:
            try:
                if getattr(self._stream, "active", False):
                    return
                self._stream.start()
                return
            except Exception:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=512, device=self.device)
        self._stream.start()

    def play(self, pcm16: np.ndarray) -> None:
        with self._lock:
            self._stop_flag = False
            try:
                self._ensure()
                chunk = 2048
                for i in range(0, len(pcm16), chunk):
                    if self._stop_flag:
                        break
                    block = pcm16[i:i + chunk]
                    self.level = float(np.abs(block.astype(np.float32)).mean()) / 32768.0
                    try:
                        self._stream.write(block)
                    except Exception:
                        if self._stop_flag:
                            break
                        try:
                            if self._stream is not None:
                                self._stream.close()
                        except Exception:
                            pass
                        self._stream = None
                        self._ensure()
                        self._stream.write(block)
            finally:
                self.level = 0.0

    def stop(self) -> None:
        self._stop_flag = True
        self.level = 0.0
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def close(self) -> None:
        self._stop_flag = True
        self.level = 0.0
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def _match_score(pa_name: str, endpoint_name: str) -> float:
    a = _norm(pa_name)
    b = _norm(endpoint_name)
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    if a.startswith(b) or b.startswith(a):
        return 80.0
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb) * 50.0


def _portaudio_devices():
    if not SD_OK:
        return [], {}
    try:
        return list(sd.query_devices()), sd.query_hostapis()
    except Exception:
        return [], {}


def _hostapi_name(apis, device) -> str:
    try:
        return apis[device["hostapi"]]["name"]
    except Exception:
        return ""


def _default_indices():
    try:
        di, do = sd.default.device
        return int(di), int(do)
    except Exception:
        try:
            v = int(sd.default.device)
            return v, v
        except Exception:
            return None, None


_HOSTAPI_ORDER = ["Windows WASAPI", "Windows DirectSound", "MME", "Windows WDM-KS"]


def _hostapi_rank(host: str, default_host: Optional[str]) -> int:
    if host == default_host:
        return -1
    try:
        return _HOSTAPI_ORDER.index(host)
    except ValueError:
        return len(_HOSTAPI_ORDER)


def _active_endpoints(kind: str) -> list:
    try:
        from pycaw.constants import AudioDeviceState
        from pycaw.pycaw import AudioUtilities
    except Exception:
        return []
    prefix = "{0.0.0." if kind == "render" else "{0.0.1."
    out = []
    try:
        devices = AudioUtilities.GetAllDevices()
    except Exception:
        return []
    for d in devices:
        try:
            if getattr(d, "state", None) != AudioDeviceState.Active:
                continue
            did = getattr(d, "id", "") or ""
            if not did.startswith(prefix):
                continue
            name = getattr(d, "FriendlyName", "") or ""
            if name:
                out.append({"name": name, "id": did})
        except Exception:
            continue
    return out


def _list_audio_devices(kind: str) -> list:
    if not SD_OK:
        return []
    devs, apis = _portaudio_devices()
    if not devs:
        return []
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    candidates = [(i, d) for i, d in enumerate(devs) if d.get(channel_key, 0) > 0]
    endpoints = _active_endpoints("render" if kind == "output" else "capture")
    default_idx = _default_indices()[0 if kind == "input" else 1]
    default_host = None
    if default_idx is not None and 0 <= default_idx < len(devs):
        default_host = _hostapi_name(apis, devs[default_idx])

    results = []
    used = set()
    for ep in endpoints:
        best = None
        best_score = -1e9
        for i, d in candidates:
            if i in used:
                continue
            match = _match_score(d.get("name", ""), ep["name"])
            if match < 40.0:
                continue
            host = _hostapi_name(apis, d)
            rank = _hostapi_rank(host, default_host)
            score = match + (50.0 if rank == -1 else 0.0) - (rank if rank >= 0 else 0.0) * 0.01
            if score > best_score:
                best = (i, d)
                best_score = score
        if best is not None:
            used.add(best[0])
            results.append({"index": best[0], "name": ep["name"]})

    if not results and default_idx is not None and 0 <= default_idx < len(devs):
        d = devs[default_idx]
        if d.get(channel_key, 0) > 0:
            results.append({"index": default_idx, "name": d.get("name", "")})
    return results


def list_input_devices() -> list:
    return _list_audio_devices("input")


def list_output_devices() -> list:
    return _list_audio_devices("output")
