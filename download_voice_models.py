"""Pre-fetch local voice models so first boot is instant.

- openWakeWord: pretrained wake-word models (includes "hey jarvis")
- Silero VAD onnx
- Kokoro-82M TTS (onnx + voices)
faster-whisper / Moonshine download lazily on first use, that's fine.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from jarvis import config as cfg  # noqa: E402

MODELS = cfg.home() / "models"
MODELS.mkdir(parents=True, exist_ok=True)

KOKORO_MODEL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                "model-files-v1.1/kokoro-v1.0.fp16.onnx")
KOKORO_VOICES = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                 "model-files-v1.1/voices-v1.0.bin")
SILERO = ("https://github.com/snakers4/silero-vad/raw/master/"
          "src/silero_vad/data/silero_vad.onnx")

JARVIS_ONNX_BASE = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis"
JARVIS_ONNX_FILES = (
    ("high/jarvis-high.onnx", "jarvis-high.onnx"),
    ("high/jarvis-high.onnx.json", "jarvis-high.onnx.json"),
    ("medium/jarvis-medium.onnx", "jarvis-medium.onnx"),
    ("medium/jarvis-medium.onnx.json", "jarvis-medium.onnx.json"),
)


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ok] {dest.name} ({dest.stat().st_size // 1024 // 1024} MB)")
        return
    print(f"[..] downloading {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
        print(f"[ok] {dest.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"[!!] {dest.name}: {exc}")


def ensure_kokoro_jarvis_voice(voices_path: Path) -> None:
    """Add a blended British 'jarvis' voice embedding to Kokoro's voices.bin."""
    if not voices_path.exists():
        return
    try:
        import numpy as np
        with np.load(voices_path, allow_pickle=False) as data:
            if "jarvis" in data.files:
                return
            if not all(k in data.files for k in ("bm_george", "bm_fable", "bm_lewis")):
                return
            voices = {k: data[k] for k in data.files}
        voices["jarvis"] = (
            0.60 * voices["bm_george"]
            + 0.25 * voices["bm_fable"]
            + 0.15 * voices["bm_lewis"]
        ).astype(voices["bm_george"].dtype)
        tmp = voices_path.with_suffix(".tmp.npz")
        np.savez(tmp, **voices)
        tmp.replace(voices_path)
        print("[ok] added blended 'jarvis' voice to voices.bin")
    except Exception as exc:  # noqa: BLE001
        print(f"[!!] kokoro jarvis blend: {exc}")


def main() -> None:
    # wake word models
    try:
        import openwakeword.utils
        print("[..] openwakeword pretrained models (hey jarvis, alexa, …)")
        openwakeword.utils.download_models()
        print("[ok] openwakeword models")
    except Exception as exc:  # noqa: BLE001
        print(f"[!!] openwakeword: {exc}")

    fetch(SILERO, MODELS / "silero_vad.onnx")
    fetch(KOKORO_MODEL, MODELS / "kokoro-v1.0.onnx")
    fetch(KOKORO_VOICES, MODELS / "voices.bin")
    ensure_kokoro_jarvis_voice(MODELS / "voices.bin")

    voice_dir = MODELS / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    for rel, local_name in JARVIS_ONNX_FILES:
        fetch(f"{JARVIS_ONNX_BASE}/{rel}", voice_dir / local_name)
    print("done.")


if __name__ == "__main__":
    main()
