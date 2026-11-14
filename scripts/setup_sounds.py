"""Fetch/generate futuristic sci-fi HUD sound effects into jarvis_home/sounds/.

Tries free online CC0/open-source sci-fi audio URLs first where available, and
synthesizes clean 44.1 kHz 16-bit stereo sci-fi HUD sound effects for any cue
not downloaded so all 12 futuristic UI cues are guaranteed present and crisp.
"""
from __future__ import annotations

import math
import struct
import sys
import urllib.request
import wave
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from jarvis import config as cfg  # noqa: E402

SR = 44100

# Optional free CC0/MIT online sci-fi UI sound sources (checked with short timeout).
# Each download is validated as a non-empty playable WAV file before keeping.
ONLINE_SOURCES: Dict[str, List[str]] = {
    "boot": [
        "https://upload.wikimedia.org/wikipedia/commons/3/34/Sound_Effect_-_Sci-Fi_Computer_Startup.wav",
    ],
    "pulse": [
        "https://upload.wikimedia.org/wikipedia/commons/4/45/Sci-fi_sonar_ping.wav",
    ],
}


def _adsr(t: float, dur: float, a: float = 0.01, r: float = 0.08) -> float:
    if t < 0 or t > dur:
        return 0.0
    env = 1.0
    if t < a and a > 0:
        env *= t / a
    rem = dur - t
    if rem < r and r > 0:
        env *= max(0.0, rem / r)
    return env


def _write_stereo_wav(path: Path, samples: List[Tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = max((max(abs(l), abs(r)) for l, r in samples), default=1.0)
    scale = min(1.0, 0.82 / peak) if peak > 0 else 1.0
    frames = bytearray()
    for l, r in samples:
        il = max(-32767, min(32767, int(l * scale * 32767)))
        ir = max(-32767, min(32767, int(r * scale * 32767)))
        frames.extend(struct.pack("<hh", il, ir))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(bytes(frames))


def _render(dur: float, fn: Callable[[float], Tuple[float, float]]) -> List[Tuple[float, float]]:
    n = int(SR * dur)
    return [fn(i / SR) for i in range(n)]


def gen_boot() -> List[Tuple[float, float]]:
    # Warm sub-bass reactor swell -> shimmering crystalline futuristic triad
    dur = 0.85
    def s(t: float) -> Tuple[float, float]:
        e = _adsr(t, dur, a=0.04, r=0.35)
        f_sweep = 95.0 + 360.0 * (1.0 - math.exp(-t * 4.5))
        bass = 0.38 * math.sin(2 * math.pi * (f_sweep * 0.5) * t) * math.exp(-t * 2.4)
        chime1 = 0.28 * math.sin(2 * math.pi * 523.25 * t) * max(0.0, min(1.0, (t - 0.12) * 14))
        chime2 = 0.24 * math.sin(2 * math.pi * 659.25 * t) * max(0.0, min(1.0, (t - 0.20) * 14))
        chime3 = 0.26 * math.sin(2 * math.pi * 1046.50 * t) * max(0.0, min(1.0, (t - 0.28) * 12))
        shimmer = 0.10 * math.sin(2 * math.pi * 2093.0 * t + 0.5 * math.sin(2 * math.pi * 7 * t))
        l = e * (bass + chime1 + 0.7 * chime2 + 0.9 * chime3 + shimmer)
        r = e * (bass + 0.7 * chime1 + chime2 + 0.95 * chime3 - 0.8 * shimmer)
        return l, r
    return _render(dur, s)


def gen_wake() -> List[Tuple[float, float]]:
    # Crisp iris wake chime (E5 -> B5) with subtle harmonic tail
    dur = 0.34
    def s(t: float) -> Tuple[float, float]:
        p1 = _adsr(t, 0.16, 0.006, 0.07) * math.sin(2 * math.pi * 659.25 * t)
        p2 = _adsr(t - 0.08, 0.26, 0.006, 0.16) * (
            0.75 * math.sin(2 * math.pi * 987.77 * t) + 0.25 * math.sin(2 * math.pi * 1975.53 * t)
        )
        pan = 0.15 * math.sin(2 * math.pi * 6 * t)
        v = 0.55 * p1 + 0.65 * p2
        return v * (1.0 - pan), v * (1.0 + pan)
    return _render(dur, s)


def gen_listen_start() -> List[Tuple[float, float]]:
    dur = 0.18
    def s(t: float) -> Tuple[float, float]:
        e = _adsr(t, dur, 0.008, 0.06)
        freq = 540.0 + 480.0 * (t / dur)
        v = e * (0.65 * math.sin(2 * math.pi * freq * t) + 0.2 * math.sin(2 * math.pi * freq * 2 * t))
        return v * 0.9, v
    return _render(dur, s)


def gen_listen_end() -> List[Tuple[float, float]]:
    dur = 0.18
    def s(t: float) -> Tuple[float, float]:
        e = _adsr(t, dur, 0.008, 0.07)
        freq = 880.0 - 380.0 * (t / dur)
        v = e * (0.65 * math.sin(2 * math.pi * freq * t) + 0.2 * math.sin(2 * math.pi * freq * 1.5 * t))
        return v, v * 0.9
    return _render(dur, s)


def gen_thought() -> List[Tuple[float, float]]:
    # Ultra-soft holographic neural shimmer
    dur = 0.26
    def s(t: float) -> Tuple[float, float]:
        e = _adsr(t, dur, 0.03, 0.14)
        f1 = 740.0 + 40.0 * math.sin(2 * math.pi * 5 * t)
        l = 0.35 * e * (math.sin(2 * math.pi * f1 * t) + 0.4 * math.sin(2 * math.pi * 1110.0 * t))
        r = 0.35 * e * (math.sin(2 * math.pi * (f1 * 1.005) * t) + 0.4 * math.sin(2 * math.pi * 1115.0 * t))
        return l, r
    return _render(dur, s)


def gen_tool() -> List[Tuple[float, float]]:
    # Telemetry data lock double-chirp
    dur = 0.22
    def s(t: float) -> Tuple[float, float]:
        c1 = _adsr(t, 0.08, 0.004, 0.03) * math.sin(2 * math.pi * 1174.66 * t)
        c2 = _adsr(t - 0.07, 0.14, 0.004, 0.08) * math.sin(2 * math.pi * 1567.98 * t)
        return 0.55 * c1 + 0.45 * c2, 0.45 * c1 + 0.6 * c2
    return _render(dur, s)


def gen_cache_hit() -> List[Tuple[float, float]]:
    # Crisp crystalline flash chime
    dur = 0.20
    def s(t: float) -> Tuple[float, float]:
        e = _adsr(t, dur, 0.004, 0.11)
        v = e * (
            0.5 * math.sin(2 * math.pi * 1318.51 * t)
            + 0.35 * math.sin(2 * math.pi * 1975.53 * t)
            + 0.2 * math.sin(2 * math.pi * 2637.02 * t)
        )
        return v, v * 0.92
    return _render(dur, s)


def gen_task_spawn() -> List[Tuple[float, float]]:
    # Futuristic orbital agent launch sweep + harmonic fifth lock
    dur = 0.48
    def s(t: float) -> Tuple[float, float]:
        e = _adsr(t, dur, 0.015, 0.22)
        sub = 0.35 * math.sin(2 * math.pi * (110.0 + 180.0 * t) * t) * math.exp(-t * 4.0)
        tone1 = _adsr(t - 0.06, 0.38, 0.01, 0.22) * math.sin(2 * math.pi * 587.33 * t)
        tone2 = _adsr(t - 0.13, 0.35, 0.01, 0.22) * math.sin(2 * math.pi * 880.00 * t)
        return e * (sub + 0.5 * tone1 + 0.45 * tone2), e * (sub + 0.42 * tone1 + 0.55 * tone2)
    return _render(dur, s)


def gen_task_step() -> List[Tuple[float, float]]:
    # Subtle soft telemetry tick
    dur = 0.11
    def s(t: float) -> Tuple[float, float]:
        e = _adsr(t, dur, 0.004, 0.06)
        v = 0.42 * e * (math.sin(2 * math.pi * 920.0 * t) + 0.3 * math.sin(2 * math.pi * 1840.0 * t))
        return v * 0.85, v
    return _render(dur, s)


def gen_task_done() -> List[Tuple[float, float]]:
    # Uplifting three-note futuristic major9 chord arpeggio (C5-G5-D6)
    dur = 0.55
    def s(t: float) -> Tuple[float, float]:
        n1 = _adsr(t, 0.35, 0.008, 0.22) * math.sin(2 * math.pi * 523.25 * t)
        n2 = _adsr(t - 0.08, 0.38, 0.008, 0.24) * math.sin(2 * math.pi * 783.99 * t)
        n3 = _adsr(t - 0.16, 0.39, 0.008, 0.26) * (
            0.7 * math.sin(2 * math.pi * 1174.66 * t) + 0.25 * math.sin(2 * math.pi * 1567.98 * t)
        )
        l = 0.55 * n1 + 0.45 * n2 + 0.55 * n3
        r = 0.45 * n1 + 0.55 * n2 + 0.60 * n3
        return l, r
    return _render(dur, s)


def gen_alert() -> List[Tuple[float, float]]:
    # Sci-fi dual-tone attention sonar pulse for questions/problems
    dur = 0.40
    def s(t: float) -> Tuple[float, float]:
        b1 = _adsr(t, 0.14, 0.006, 0.05) * (math.sin(2 * math.pi * 784.0 * t) + 0.3 * math.sin(2 * math.pi * 1176.0 * t))
        b2 = _adsr(t - 0.17, 0.22, 0.006, 0.12) * (math.sin(2 * math.pi * 659.25 * t) + 0.3 * math.sin(2 * math.pi * 988.0 * t))
        return 0.55 * (b1 + b2), 0.55 * (b1 * 0.9 + b2 * 1.05)
    return _render(dur, s)


def gen_pulse() -> List[Tuple[float, float]]:
    # Holographic shockwave pulse
    dur = 0.32
    def s(t: float) -> Tuple[float, float]:
        e = _adsr(t, dur, 0.01, 0.20)
        sweep = 480.0 * math.exp(-t * 5.0) + 180.0
        v = e * (0.6 * math.sin(2 * math.pi * sweep * t) + 0.3 * math.sin(2 * math.pi * 1046.5 * t) * math.exp(-t * 9.0))
        return v, v
    return _render(dur, s)


GENERATORS: Dict[str, Callable[[], List[Tuple[float, float]]]] = {
    "boot": gen_boot,
    "wake": gen_wake,
    "listen_start": gen_listen_start,
    "listen_end": gen_listen_end,
    "thought": gen_thought,
    "tool": gen_tool,
    "cache_hit": gen_cache_hit,
    "task_spawn": gen_task_spawn,
    "task_step": gen_task_step,
    "task_done": gen_task_done,
    "alert": gen_alert,
    "pulse": gen_pulse,
}


def try_fetch_wav(urls: List[str], dest: Path) -> bool:
    for url in urls:
        if not url.lower().endswith(".wav"):
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "JarvisSoundSetup/1.0"})
            with urllib.request.urlopen(req, timeout=4) as r:
                data = r.read()
            if len(data) > 1024 and data[:4] == b"RIFF":
                dest.write_bytes(data)
                return True
        except Exception:
            continue
    return False


def ensure_sound_bank(force: bool = False) -> Path:
    sounds_dir = cfg.home() / "sounds"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in GENERATORS.items():
        dest = sounds_dir / f"{name}.wav"
        if dest.exists() and dest.stat().st_size > 1024 and not force:
            continue
        if name in ONLINE_SOURCES and try_fetch_wav(ONLINE_SOURCES[name], dest):
            print(f"[sounds] downloaded {dest.name}")
            continue
        _write_stereo_wav(dest, fn())
        print(f"[sounds] generated {dest.name} ({dest.stat().st_size} bytes)")
    return sounds_dir


if __name__ == "__main__":
    sdir = ensure_sound_bank(force=("--force" in sys.argv))
    print(f"Sound bank ready at: {sdir}")
