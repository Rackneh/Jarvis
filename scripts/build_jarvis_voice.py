"""Build Jarvis clone voices from the HuggingFace jarvis-voice-samples dataset.

Downloads the parquet corpus (720 Jarvis-style clips), extracts the audio, and
constructs reference clips + transcripts for the TTS engines:

- reference.wav / reference.txt      (~30 s, XTTS-v2 zero-shot cloning)
- breeze_reference.wav / .txt        (~8 s, Breeze TTS 2 voice cloning)

Results live under jarvis_home/models/voice/.
"""
import io
import sys
import wave
from pathlib import Path

import numpy as np

PARQUET_URL = ("https://huggingface.co/datasets/derekurban2001/jarvis-voice-samples"
               "/resolve/main/data/train-00000-of-00001.parquet")
JARVIS_ONNX_BASE = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis"
JARVIS_ONNX_FILES = (
    ("high/jarvis-high.onnx", "jarvis-high.onnx"),
    ("high/jarvis-high.onnx.json", "jarvis-high.onnx.json"),
    ("medium/jarvis-medium.onnx", "jarvis-medium.onnx"),
    ("medium/jarvis-medium.onnx.json", "jarvis-medium.onnx.json"),
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from jarvis import config as cfg  # noqa: E402

VOICE_DIR = cfg.home() / "models" / "voice"
REFERENCE_TARGET_SEC = 30.0
BREEZE_TARGET_SEC = 8.0
MAX_SAMPLES = 12  # keep a handful of individual clips for inspection


def main() -> None:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- fetch
    pq_path = VOICE_DIR / "jarvis-samples.parquet"
    if not pq_path.exists():
        print("[..] downloading jarvis-voice-samples corpus (~153 MB)…")
        import urllib.request
        urllib.request.urlretrieve(PARQUET_URL, pq_path)  # noqa: S310
    print(f"[ok] corpus at {pq_path}")

    # ---------------------------------------------------------------- read
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[!!] pyarrow missing: pip install pyarrow")
        return

    table = pq.read_table(str(pq_path), columns=["audio", "text"])
    audio_col = table.column("audio")
    texts = table.column("text").to_pylist()
    print(f"[ok] {table.num_rows} clips")

    clips = []
    for i in range(table.num_rows):
        rec = audio_col[i].as_py()
        if not rec:
            continue
        data = rec.get("bytes")
        if not data:
            continue
        try:
            pcm, sr, dur = _decode(data)
        except Exception:
            continue
        clips.append({"pcm": pcm, "sr": sr, "dur": dur,
                      "text": texts[i] if i < len(texts) else ""})

    if not clips:
        print("[!!] no decodable audio found")
        return

    # keep a few sample wavs
    clips.sort(key=lambda c: c["dur"], reverse=True)
    for i, c in enumerate(clips[:MAX_SAMPLES]):
        _write_wav(VOICE_DIR / f"sample_{i:02d}.wav", c["pcm"], c["sr"])

    _ensure_reference(clips)
    _ensure_breeze_reference(clips)
    _ensure_jarvis_onnx()


def _ensure_jarvis_onnx() -> None:
    import urllib.request
    for rel, local_name in JARVIS_ONNX_FILES:
        dest = VOICE_DIR / local_name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[ok] {dest.name}")
            continue
        url = f"{JARVIS_ONNX_BASE}/{rel}"
        print(f"[..] downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)  # noqa: S310
            print(f"[ok] {dest.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[!!] {dest.name}: {exc}")


def _ensure_reference(clips: list) -> None:
    ref_path = VOICE_DIR / "reference.wav"
    txt_path = VOICE_DIR / "reference.txt"
    if ref_path.exists():
        print(f"[ok] reference voice already exists: {ref_path}")
        return
    # pick clear mid-length clips (1.5-3.5 s) until we hit ~30 s — avoids any
    # corrupted long tail and gives XTTS good phoneme coverage.
    chosen, total = [], 0.0
    for c in clips:
        if 1.5 <= c["dur"] <= 3.5 and total < REFERENCE_TARGET_SEC:
            chosen.append(c)
            total += c["dur"]
    if not chosen:
        chosen = clips[:8]
        total = sum(c["dur"] for c in chosen)
    combined = np.concatenate([c["pcm"] for c in chosen])
    _write_wav(ref_path, combined, chosen[0]["sr"])
    _write_transcript(txt_path, chosen)
    print(f"[ok] reference voice built: {ref_path} ({total:.1f}s, {len(chosen)} clips)")


def _ensure_breeze_reference(clips: list) -> None:
    wav_path = VOICE_DIR / "breeze_reference.wav"
    txt_path = VOICE_DIR / "breeze_reference.txt"
    if wav_path.exists():
        print(f"[ok] breeze reference already exists: {wav_path}")
        return
    # Breeze clones best from a short, clean clip (~5-10 s). Prefer 2-4 s clips.
    chosen, total = [], 0.0
    for c in clips:
        if 1.5 <= c["dur"] <= 4.5 and total < BREEZE_TARGET_SEC:
            chosen.append(c)
            total += c["dur"]
    if not chosen:
        chosen = clips[:4]
        total = sum(c["dur"] for c in chosen)
    combined = np.concatenate([c["pcm"] for c in chosen])
    _write_wav(wav_path, combined, chosen[0]["sr"])
    _write_transcript(txt_path, chosen)
    print(f"[ok] breeze reference built: {wav_path} ({total:.1f}s, {len(chosen)} clips)")


def _write_transcript(path: Path, clips: list) -> None:
    text = " ".join(c.get("text", "").strip() for c in clips if c.get("text"))
    path.write_text(" ".join(text.split()), encoding="utf-8")
    print(f"     wrote {path.name} ({len(text.split())} words)")


def _decode(data: bytes):
    import soundfile as sf  # soundfile handles wav/flac/ogg inside parquet
    audio, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
    mono = audio[:, 0]
    return mono, sr, len(mono) / sr


def _write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.astype(np.int16).tobytes())
    print(f"     wrote {path.name} ({len(pcm)/sr:.1f}s)")


if __name__ == "__main__":
    main()
