"""Silero VAD (ONNX, ~2 ms/frame) with an RMS fallback.

Downloads the 2 MB ONNX once into jarvis_home/models if pip silero-vad
(or bundled copies) is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
    ORT_OK = True
except Exception:  # noqa: BLE001
    ort = None
    ORT_OK = False

from .. import config as cfg

MODEL_URL = ("https://github.com/snakers4/silero-vad/raw/master/"
             "src/silero_vad/data/silero_vad.onnx")


def _model_path() -> Path:
    p = cfg.home() / "models" / "silero_vad.onnx"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class SileroVAD:
    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold
        self._sess = None
        self._state = None
        self._context = np.zeros(64, dtype=np.float32)
        self._rms_floor = 120.0
        self.last_prob = 0.0
        if ORT_OK:
            self._load()

    def _load(self) -> None:
        path = _model_path()
        if not path.exists():
            try:
                import urllib.request
                urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310
            except Exception:
                return
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self._sess = ort.InferenceSession(str(path), sess_options=opts,
                                              providers=["CPUExecutionProvider"])
            self._reset()
        except Exception:
            self._sess = None

    def _reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)
        self.last_prob = 0.0

    def reset(self) -> None:
        if self._sess:
            self._reset()
        else:
            self._context = np.zeros(64, dtype=np.float32)
            self.last_prob = 0.0
        self._rms_floor = 120.0

    def is_speech(self, frame: bytes) -> bool:
        """frame: 512 int16 little-endian mono samples."""
        x_int16 = np.frombuffer(frame, dtype=np.int16)
        if len(x_int16) == 0:
            self.last_prob = 0.0
            return False
        rms = float(np.sqrt(np.mean(x_int16.astype(np.float32) ** 2)))
        if self._sess is None:
            return self._rms_speech(frame)
        x = x_int16.astype(np.float32) / 32768.0
        try:
            # Silero VAD v5 requires 64 context samples prepended to the 512-sample frame
            inp = np.concatenate([self._context, x]).reshape(1, -1)
            if len(x) >= 64:
                self._context = x[-64:].copy()
            out = self._sess.run(None, {
                "input": inp,
                "state": self._state,
                "sr": np.array(16000, dtype=np.int64),
            })
            self._state = out[1]
            prob = float(out[0][0][0])
            self.last_prob = prob
            if rms < 30.0:
                self._rms_floor = 0.95 * self._rms_floor + 0.05 * rms
                return False
            if prob < 0.25:
                self._rms_floor = 0.95 * self._rms_floor + 0.05 * rms
            return (
                prob > self.threshold
                or (prob > max(0.20, self.threshold * 0.5)
                    and rms > max(self._rms_floor * 3.2, 220.0))
            )
        except Exception:
            return self._rms_speech(frame)

    def _rms_speech(self, frame: bytes) -> bool:
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt((x ** 2).mean()))
        self._rms_floor = 0.95 * self._rms_floor + 0.05 * rms
        return rms > max(self._rms_floor * 3.0, 250.0)
