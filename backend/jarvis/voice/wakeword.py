"""Wake word via openWakeWord (pretrained "hey jarvis" ships with the package).

If the assistant has a custom name with no pretrained model, wakeword=None and
the session falls back to "wake on speech, verify the name in the transcript".
"""
from __future__ import annotations

import threading
from typing import Optional

try:
    from openwakeword.model import Model as OWWModel
    OWW_OK = True
except Exception:  # noqa: BLE001
    OWWModel = None
    OWW_OK = False


def pretrained_for_name(name: str) -> Optional[str]:
    n = name.lower().strip()
    if "jarvis" in n:
        return "hey_jarvis"
    if "alexa" in n:
        return "alexa"
    if "siri" in n:
        return "hey_siri"
    if "hey rhasspy" in n or n == "rhasspy":
        return "hey_rhasspy"
    return None


def pretrained_for_wake(wake: str) -> Optional[str]:
    w = (wake or "").lower().strip()
    if not w:
        return None
    if "jarvis" in w:
        return "hey_jarvis"
    if "alexa" in w:
        return "alexa"
    if "mycroft" in w:
        return "hey_mycroft"
    if "rhasspy" in w or "raspberry" in w:
        return "hey_rhasspy"
    if "timer" in w:
        return "timer"
    if "weather" in w:
        return "weather"
    return None


class WakeWordEngine:
    def __init__(self, assistant_name: str, threshold: float = 0.5,
                 wake_word: str = "") -> None:
        self.threshold = threshold
        self.wake_word = (wake_word or "").strip().lower()
        self.model_name = (pretrained_for_wake(self.wake_word)
                           if self.wake_word else pretrained_for_name(assistant_name))
        self._model = None
        self._lock = threading.Lock()
        if OWW_OK and self.model_name:
            try:
                self._model = OWWModel(wakeword_models=[self.model_name],
                                       vad_threshold=0.5, ncpu=1)
            except Exception:
                self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def phrase(self) -> str:
        if self.wake_word:
            return self.wake_word
        return self.model_name.replace("_", " ") if self.model_name else ""

    def detect(self, frame: bytes) -> float:
        """frame: 1280 samples (80 ms) int16 mono. Returns score 0..1."""
        if self._model is None:
            return 0.0
        with self._lock:
            try:
                import numpy as np
                scores = self._model.predict(np.frombuffer(frame, dtype=np.int16))
                key = list(scores.keys())[0] if scores else None
                return float(scores.get(key, 0.0)) if key else 0.0
            except Exception:
                return 0.0

    def reset(self) -> None:
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass
