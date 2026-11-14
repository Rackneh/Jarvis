"""VoiceSession — the always-on state machine.

    IDLE (wake-word listening)
      -> wake detected / push-to-talk
    LISTENING (Silero VAD endpointing + streaming partial transcripts)
      -> end of speech
    THINKING (conscious agent streaming; sentences flow to TTS)
      -> done / interrupted
    SPEAKING (Kokoro/Piper/SAPI sentence queue; barge-in via wake word or speech)
      -> back to IDLE

Barge-in while speaking uses the wake word when available, otherwise a
VAD/energy guard above an adaptive echo level (the mic hears our own TTS;
without AEC we require speech clearly louder than our playback echo).
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional

import numpy as np

from .. import config as cfg
from ..events import publish
from .audio import FRAME_MS, MicSource, SAMPLE_RATE, SpeakerSink
from .stt import STTEngine
from .tts import SpeechQueue, SpeechStreamer, TTSEngine
from .vad import SileroVAD
from .wakeword import WakeWordEngine


class VoiceState(str, Enum):
    OFF = "off"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceSession:
    def __init__(self) -> None:
        self.state = VoiceState.OFF
        self._q: asyncio.Queue = asyncio.Queue(maxsize=400)
        self._task: Optional[asyncio.Task] = None
        self._mic: Optional[MicSource] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.vad = SileroVAD(threshold=float(cfg.get("voice.vad_threshold", 0.45) or 0.45))
        self.wakeword = WakeWordEngine(
            cfg.assistant_name(),
            float(cfg.get("voice.wake_threshold", 0.5) or 0.5),
            wake_word=cfg.get("voice.wake_word", ""))
        self.stt = STTEngine()
        self.tts_engine = TTSEngine()
        self.sink = SpeakerSink(device=cfg.get("voice.output_device"))
        self.speech = SpeechQueue(self.tts_engine, self.sink)
        self._stt_pool = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="jarvis-stt-partial")
        self._stt_final_pool = ThreadPoolExecutor(max_workers=2,
                                                  thread_name_prefix="jarvis-stt-final")
        self._partial_task: Optional[asyncio.Task] = None
        self._partial_running = False
        self._watchdog_task: Optional[asyncio.Task] = None
        self._last_frame_ts = 0.0

        self._utterance = bytearray()
        self._rolling = bytearray()
        self._speeching = False
        self._silence_ms = 0.0
        self._soft_silence_ms = 0.0
        self._speech_ms = 0.0
        self._speech_peak_rms = 0.0
        self._utterance_ms = 0.0
        self._listen_elapsed_ms = 0.0
        self._mute_ms = 0.0
        self._noise_rms = 80.0
        self._last_partial = 0.0
        self._echo_level = 0.0
        self._barge_ms = 0.0
        self._settle_until = 0.0
        self._ww_window = bytearray()
        self._activated = False  # push-to-talk or wake trigger
        self._in_conversation = False
        self._conv_deadline = 0.0
        self._wake_cooldown_until = 0.0
        self._mic_level_acc = 0.0
        self._mic_level_peak = 0.0
        self._mic_level_count = 0
        self._mic_level_speech = False
        self._mic_level_last = 0.0
        self._mic_frames = 0
        self._process_frames = 0
        self._agent = None
        self._dispatch_lock: Optional[asyncio.Lock] = None
        self._dispatch_lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._drain_task: Optional[asyncio.Task] = None
        self._mic_muted: bool = False

    def _get_dispatch_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._dispatch_lock is None or self._dispatch_lock_loop is not loop:
            self._dispatch_lock = asyncio.Lock()
            self._dispatch_lock_loop = loop
        return self._dispatch_lock

    # ------------------------------------------------------------ lifecycle

    def attach(self, agent, loop) -> None:
        self._agent = agent
        self._loop = loop

    def start(self) -> None:
        self._mic_muted = False
        if self._task is not None:
            return
        if str(cfg.get("voice.activation_mode", "wake") or "wake").lower() == "off":
            self._set_state(VoiceState.OFF)
            return
        self._mic = MicSource(self._on_audio_frame,
                              device=cfg.get("voice.input_device"))
        self._mic.start()
        self._task = asyncio.get_event_loop().create_task(self._run())
        self._watchdog_task = asyncio.get_event_loop().create_task(
            self._watchdog())
        self.stt.preload()
        self.tts_engine.warm()
        self._set_state(VoiceState.IDLE)

    def set_mic_muted(self, muted: bool) -> None:
        """Mute or unmute microphone capture without killing outbound TTS playback."""
        if not muted:
            self._mic_muted = False
            self.start()
            return
        self._mic_muted = True
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        if self._mic is not None:
            self._mic.stop()
            self._mic = None
        if not self.speech.is_active:
            self._set_state(VoiceState.OFF)

    def stop(self) -> None:
        self._mic_muted = True
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        if self._mic is not None:
            self._mic.stop()
            self._mic = None
        self.speech.stop()
        self._set_state(VoiceState.OFF)

    def apply_settings(self, changed: set) -> None:
        if "voice.activation_mode" in changed:
            self._apply_activation_mode()
            self._in_conversation = False
        if changed & {"voice.input_device", "voice.output_device"}:
            self._apply_audio_devices()
        if changed & {"assistant.name", "voice.wake_word", "voice.wake_threshold"}:
            self._reload_wakeword()
        if "voice.vad_threshold" in changed:
            self.vad.threshold = float(cfg.get("voice.vad_threshold", 0.45) or 0.45)
        if changed & {"voice.stt_engine", "voice.stt_model"}:
            self.stt = STTEngine()
        if changed & {"voice.tts_engine", "voice.tts_voice", "voice.tts_speed",
                      "voice.tts_clone_reference", "voice.jarvis_clone_quality",
                      "voice.tts_gpu"}:
            threading.Thread(target=self._reload_tts, name="jarvis-tts-reload",
                             daemon=True).start()
        elif "voice.cached_conversation" in changed or "voice.activation_mode" in changed:
            if self.tts_engine.cached_conversation_enabled() and self.tts_engine.cache is not None:
                self.tts_engine.cache.seed_defaults_async()
        publish("config_applied", changed=sorted(changed))

    def _apply_activation_mode(self) -> None:
        mode = str(cfg.get("voice.activation_mode", "wake") or "wake").lower()
        if mode == "off":
            if self.state != VoiceState.OFF or self._task is not None or self._mic is not None:
                self.stop()
            return
        if self._task is None:
            self.start()
        elif self.state == VoiceState.OFF:
            self._set_state(VoiceState.IDLE)

    def _apply_audio_devices(self) -> None:
        active = self.state != VoiceState.OFF
        self.stop()
        old_sink = self.sink
        self.sink = SpeakerSink(device=cfg.get("voice.output_device"))
        self.speech.sink = self.sink
        try:
            old_sink.close()
        except Exception:
            pass
        if active:
            self.start()

    def _reload_wakeword(self) -> None:
        self.wakeword = WakeWordEngine(
            cfg.assistant_name(),
            float(cfg.get("voice.wake_threshold", 0.5) or 0.5),
            wake_word=cfg.get("voice.wake_word", ""))
        publish("wakeword_changed", available=self.wakeword.available,
                phrase=self.wakeword.phrase, model=self.wakeword.model_name or "")

    def _reload_tts(self) -> None:
        old_engine = self.tts_engine
        try:
            new_engine = TTSEngine()
        except Exception:
            return
        self.tts_engine = new_engine
        self.speech.engine = new_engine
        new_engine.set_busy_probe(lambda: self.speech.is_active)
        try:
            old_engine.close()
        except Exception:
            pass
        publish("voice_state", state=self.state.value,
                tts=new_engine.engine_name, stt=self.stt.engine_name,
                wake=self.wakeword.available)

    def _set_state(self, s: VoiceState) -> None:
        if self.state != s:
            self.state = s
            publish("voice_state", state=s.value, tts=self.tts_engine.engine_name,
                    stt=self.stt.engine_name, wake=self.wakeword.available)

    # ------------------------------------------------------------ audio thread

    def _on_audio_frame(self, frame: bytes) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        loop.call_soon_threadsafe(self._push_frame, frame)

    def _push_frame(self, frame: bytes) -> None:
        self._mic_frames += 1
        self._last_frame_ts = time.time()
        try:
            self._q.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._q.get_nowait()
                self._q.put_nowait(frame)
            except Exception:
                pass

    # ------------------------------------------------------------ main loop

    async def _run(self) -> None:
        while True:
            frame = await self._q.get()
            await self._process_frame(frame)

    async def _process_frame(self, frame: bytes) -> None:
        self._process_frames += 1
        now = time.time()
        if self.state == VoiceState.OFF:
            return
        is_speech = self.vad.is_speech(frame)

        rms = _rms(frame)
        self._mic_level_acc += rms
        self._mic_level_peak = max(self._mic_level_peak, rms)
        self._mic_level_count += 1
        self._mic_level_speech = self._mic_level_speech or is_speech
        if now - self._mic_level_last >= 0.1:
            count = max(1, self._mic_level_count)
            avg = self._mic_level_acc / count
            peak = self._mic_level_peak
            level = min(1.0, (avg * 0.6 + peak * 0.4) / 3500.0)
            publish("mic_level", level=round(level, 3), speech=self._mic_level_speech)
            self._mic_level_acc = 0.0
            self._mic_level_peak = 0.0
            self._mic_level_count = 0
            self._mic_level_speech = False
            self._mic_level_last = now

        if self.state == VoiceState.IDLE:
            mode = str(cfg.get("voice.activation_mode", "wake") or "wake").lower()
            if mode == "ptt":
                self._ww_window.clear()
                return
            if mode == "always":
                self._ww_window.clear()
                if is_speech:
                    self._begin_listening()
                return
            if mode == "off":
                return
            if now < self._wake_cooldown_until:
                self._ww_window.clear()
                return
            self._ww_window += frame
            if len(self._ww_window) >= 1280 * 2:
                window = bytes(self._ww_window[:1280 * 2])
                self._ww_window = self._ww_window[1280 * 2 // 1:]
                score = self.wakeword.detect(window)
                if score >= float(cfg.get("voice.wake_threshold", 0.5) or 0.5):
                    publish("wake_detected", score=round(score, 3),
                            phrase=self.wakeword.phrase)
                    self.wakeword.reset()
                    self._wake_cooldown_until = now + float(
                        cfg.get("voice.wake_cooldown_seconds", 2.0) or 2.0)
                    self._begin_listening()
            return

        if self.state == VoiceState.LISTENING:
            self._utterance += frame
            self._rolling += frame
            if len(self._rolling) > 12 * SAMPLE_RATE * 2:
                del self._rolling[:-12 * SAMPLE_RATE * 2]
            self._utterance_ms += FRAME_MS
            self._listen_elapsed_ms += FRAME_MS

            min_speech_ms = max(96.0, float(cfg.get("voice.min_speech_ms", 160) or 160))
            silence_limit = max(350.0, float(cfg.get("voice.silence_end_ms", 650) or 650))
            partial_interval_s = max(
                0.25, float(cfg.get("voice.stt_partial_interval_ms", 350) or 350) / 1000.0
            )

            if not self._speeching and self._speech_ms == 0 and len(self._utterance) > 20480:
                del self._utterance[:-20480]
                self._utterance_ms = min(self._utterance_ms, 640.0)

            strong_speech = is_speech and rms > self._strong_speech_threshold()
            if is_speech:
                self._speech_ms += FRAME_MS
                self._silence_ms = 0.0
                self._mute_ms = 0.0
                self._speech_peak_rms = max(self._speech_peak_rms, rms)
                if self._in_conversation:
                    self._conv_deadline = now + self._conversation_idle_seconds()
                if not self._speeching and self._speech_ms >= min_speech_ms:
                    self._speeching = True
                    publish("user_speech_start")
            else:
                self._silence_ms += FRAME_MS
                self._noise_rms = max(30.0, min(600.0, 0.96 * self._noise_rms + 0.04 * rms))

            if self._speeching:
                if strong_speech:
                    self._soft_silence_ms = 0.0
                else:
                    self._soft_silence_ms += FRAME_MS
                if rms < 18.0 and not is_speech:
                    self._mute_ms += FRAME_MS
                else:
                    self._mute_ms = 0.0
                if (self._soft_silence_ms >= silence_limit or
                        self._silence_ms >= silence_limit or
                        self._mute_ms >= 320.0):
                    await self._end_utterance()
                    return
            elif self._speech_ms >= 64.0 and self._silence_ms >= silence_limit:
                await self._end_utterance()
                return

            if (self._speeching and not self._partial_running and
                    now - self._last_partial >= partial_interval_s):
                self._last_partial = now
                tail = bytes(self._rolling[-4 * SAMPLE_RATE * 2:])
                self._partial_running = True
                self._partial_task = asyncio.get_event_loop().create_task(
                    self._partial_async(tail))
            if (not self._in_conversation and not self._speeching and
                    self._speech_ms == 0 and self._listen_elapsed_ms >= 8000.0):
                self._set_state(VoiceState.IDLE)
                return
            if (self._in_conversation and not self._speeching and
                    now > self._conv_deadline):
                self._end_conversation("idle")
                return
            if self._utterance_ms >= float(
                    cfg.get("voice.max_utterance_ms", 60000) or 60000):
                await self._end_utterance()
            return

        if self.state in (VoiceState.THINKING, VoiceState.SPEAKING):
            # barge-in detection
            self._echo_level = 0.9 * self._echo_level + 0.1 * _rms(frame)
            threshold = max(1300.0, self._echo_level * 2.2) if cfg.get(
                "voice.echo_guard", True) else 250.0
            if self.wakeword.available:
                self._ww_window += frame
                if len(self._ww_window) >= 2560:
                    window = bytes(self._ww_window[:2560])
                    self._ww_window = self._ww_window[2560:]
                    if self.wakeword.detect(window) >= 0.5:
                        self.wakeword.reset()
                        self._barge_in("wake word")
                        return
            elif is_speech and bool(cfg.get("voice.allow_speech_barge_in", False)):
                x = np.frombuffer(frame, dtype=np.int16)
                if float(np.abs(x.astype(np.float32)).mean()) > threshold:
                    self._barge_ms += FRAME_MS
                    if self._barge_ms >= 200:
                        self._barge_in("speech")
                        return
                else:
                    self._barge_ms = 0.0

    # ------------------------------------------------------------ transitions

    def _strong_speech_threshold(self) -> float:
        return max(self._noise_rms * 1.6, self._speech_peak_rms * 0.15, 55.0)

    def _conversation_idle_seconds(self) -> float:
        return float(cfg.get("voice.conversation_idle_seconds", 60) or 60)

    def _exit_phrases(self) -> list:
        default = ("goodbye,bye,that's all,thats all,never mind,nevermind,"
                   "i'm done,im done,that's it,thats it,go to sleep,"
                   "end conversation")
        raw = str(cfg.get("voice.conversation_exit_phrases", "") or "") or default
        return [p.strip().lower() for p in raw.split(",") if p.strip()]

    def _is_exit_phrase(self, text: str) -> bool:
        import re
        norm = re.sub(r"[^a-z0-9' ]", " ", text.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        return any(re.search(r"\b" + re.escape(p) + r"\b", norm)
                   for p in self._exit_phrases())

    def _end_conversation(self, reason: str) -> None:
        self._in_conversation = False
        self._conv_deadline = 0.0
        publish("conversation_ended", reason=reason)
        self._set_state(VoiceState.IDLE)

    def _begin_listening(self) -> None:
        self._utterance.clear()
        self._rolling.clear()
        self._speeching = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._soft_silence_ms = 0.0
        self._speech_peak_rms = 0.0
        self._utterance_ms = 0.0
        self._listen_elapsed_ms = 0.0
        self._mute_ms = 0.0
        self.vad.reset()
        self._activated = True
        mode = str(cfg.get("voice.activation_mode", "wake") or "wake").lower()
        self._in_conversation = mode in ("conversation", "cached_conversation")
        self._conv_deadline = (time.time() + self._conversation_idle_seconds()
                               if self._in_conversation else 0.0)
        self.stt.preload()
        self.tts_engine.warm()
        self._set_state(VoiceState.LISTENING)

    async def _partial_async(self, pcm: bytes) -> None:
        try:
            text = await asyncio.get_running_loop().run_in_executor(
                self._stt_pool, self.stt.partial, pcm)
            if text and self.state == VoiceState.LISTENING and self._speeching:
                publish("partial_transcript", text=text)
        except Exception:
            pass
        finally:
            self._partial_running = False

    async def _watchdog(self) -> None:
        # If mic frames stop arriving while we think we're mid-utterance
        # (device disabled, stream stalled), force-finalize instead of
        # waiting for VAD silence that will never come.
        while True:
            await asyncio.sleep(0.5)
            try:
                if (self.state == VoiceState.LISTENING and self._speeching
                        and self._last_frame_ts
                        and time.time() - self._last_frame_ts > 1.0):
                    await self._end_utterance()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    async def _end_utterance(self) -> None:
        pcm = bytes(self._utterance)
        self._speeching = False
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()
        min_speech = max(96.0, float(cfg.get("voice.min_speech_ms", 160) or 160))
        if self._utterance_ms < min_speech:
            if self._in_conversation:
                self._begin_listening()
            else:
                self._set_state(VoiceState.IDLE)
            return
        # Show THINKING immediately so the UI doesn't sit in "listening"
        # for the whole STT + queue wait.
        self._set_state(VoiceState.THINKING)
        publish("stt_transcribing", ms=round(self._utterance_ms))
        text = await asyncio.get_running_loop().run_in_executor(
            self._stt_final_pool, self.stt.transcribe, pcm)
        publish("final_transcript", text=text)
        if not text:
            asyncio.get_event_loop().create_task(self._fallback_idle())
            return
        if self._in_conversation and self._is_exit_phrase(text):
            self._in_conversation = False
            self._conv_deadline = 0.0
            publish("conversation_ended", reason="phrase", transcript=text)
            asyncio.get_event_loop().create_task(
                self._speak_then_idle("Conversation ended."))
            return
        asyncio.get_event_loop().create_task(self._dispatch(text))

    def speak_text(self, text: str) -> None:
        """Queue text to be spoken in FIFO order from any caller (/api/say, task relay)."""
        if not text or not text.strip():
            return
        try:
            self.speech.speak(text)
        except Exception:
            return
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                self._on_external_speech_queued()
            else:
                loop.call_soon_threadsafe(self._on_external_speech_queued)

    def _on_external_speech_queued(self) -> None:
        # If _dispatch owns the lock, it is already managing SPEAKING -> LISTENING/IDLE/OFF.
        if self._dispatch_lock is not None and self._dispatch_lock.locked():
            self._set_state(VoiceState.SPEAKING)
            return
        self._set_state(VoiceState.SPEAKING)
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.get_running_loop().create_task(self._drain_speech())

    async def _drain_speech(self) -> None:
        per = 90.0 if self._tts_slow() else 30.0
        deadline = time.time() + (120.0 if self._tts_slow() else 45.0) + max(0, self.speech.pending) * per
        last_pending = self.speech.pending
        while self.speech.is_active and time.time() < deadline:
            cur_pending = self.speech.pending
            if cur_pending < last_pending:
                deadline = max(deadline, time.time() + per)
            last_pending = cur_pending
            await asyncio.sleep(0.15)
        if self._dispatch_lock is not None and self._dispatch_lock.locked():
            return
        if self.state == VoiceState.SPEAKING:
            if (
                self._mic_muted
                or self._task is None
                or str(cfg.get("voice.activation_mode", "wake") or "wake").lower() == "off"
            ):
                self._set_state(VoiceState.OFF)
                return
            if self._in_conversation or str(cfg.get("voice.activation_mode", "wake") or "wake").lower() in ("conversation", "cached_conversation"):
                if self._in_conversation:
                    self._begin_listening()
                    return
            self._set_state(VoiceState.IDLE)

    async def _speak_then_idle(self, text: str) -> None:
        self._set_state(VoiceState.SPEAKING)
        streamer = SpeechStreamer(self.speech)
        streamer.feed(text)
        streamer.flush()
        deadline = time.time() + (240.0 if self._tts_slow() else 60.0)
        while self.speech.is_active and time.time() < deadline:
            await asyncio.sleep(0.2)
        if self.state == VoiceState.SPEAKING:
            self._set_state(VoiceState.IDLE)

    async def _fallback_idle(self) -> None:
        await asyncio.sleep(0.5)
        if self.state == VoiceState.THINKING:
            if self._in_conversation:
                self._begin_listening()
            else:
                self._set_state(VoiceState.IDLE)

    async def _dispatch(self, text: str) -> None:
        # Custom-name fallback: if no pretrained wake word, verify the name
        if not self.wakeword.available and not self._activated_by_ptt:
            name = cfg.assistant_name().lower()
            first = text.lower()[:len(name) + 12]
            if name not in first and name.split()[-1] not in first:
                publish("wake_ignored", transcript=text)
                self._set_state(VoiceState.IDLE)
                return
        self._activated_by_ptt = False
        async with self._get_dispatch_lock():
            self._cancel_requested = False
            streamer = SpeechStreamer(self.speech)
            try:
                agen = self._agent.handle_utterance(text, source="voice")
                async for ev in agen:
                    if self._cancel_requested:
                        break
                    if ev["type"] == "sentence":
                        publish("assistant_sentence", text=ev["text"])
                        self._set_state(VoiceState.SPEAKING)
                        streamer.feed(ev["text"])
                    elif ev["type"] == "thought":
                        publish("assistant_thought", text=ev["text"][:300])
                    elif ev["type"] == "tool":
                        pass  # conscious_tool is already published by ConsciousAgent._run_turn
                    elif ev["type"] == "done":
                        streamer.flush()
            finally:
                if not self._cancel_requested:
                    streamer.flush()
                self._cancel_requested = False
                await asyncio.sleep(0.15)
                if self.state in (VoiceState.SPEAKING, VoiceState.THINKING) or self.speech.is_active:
                    per = 90.0 if self._tts_slow() else 30.0
                    timeout = (120.0 if self._tts_slow() else 45.0) + \
                        max(0, self.speech.pending) * per
                    deadline = time.time() + timeout
                    last_pending = self.speech.pending
                    while self.speech.is_active and time.time() < deadline:
                        cur_pending = self.speech.pending
                        if cur_pending < last_pending:
                            deadline = max(deadline, time.time() + per)
                        last_pending = cur_pending
                        await asyncio.sleep(0.15)
                    if self.speech.is_active:
                        publish("tts_timeout", waited=round(timeout, 1))
                        self.speech.stop()
                if self._in_conversation and self.state != VoiceState.OFF:
                    self._begin_listening()
                elif self.state in (VoiceState.SPEAKING, VoiceState.THINKING):
                    self._set_state(VoiceState.IDLE)

    def _tts_slow(self) -> bool:
        try:
            return self.speech.engine.engine_name in ("xtts", "breeze")
        except Exception:
            return False

    _cancel_requested = False
    _activated_by_ptt = False

    def _barge_in(self, why: str) -> None:
        self._barge_ms = 0.0
        self._settle_until = time.time() + 0.4
        publish("barge_in", reason=why)
        self.speech.stop()
        if self._agent is not None:
            asyncio.get_event_loop().create_task(self._agent.interrupt())
        self._cancel_requested = True
        self._set_state(VoiceState.IDLE)

    # ------------------------------------------------------------ push-to-talk

    def ptt_start(self) -> bool:
        if self.state not in (VoiceState.OFF, VoiceState.IDLE):
            return False
        self._activated_by_ptt = True
        self._begin_listening()
        return True

    def ptt_stop(self) -> None:
        if self.state == VoiceState.LISTENING:
            asyncio.get_event_loop().create_task(self._end_utterance())


def _rms(frame: bytes) -> float:
    x = np.frombuffer(frame, dtype=np.int16)
    if len(x) == 0:
        return 0.0
    return float(np.sqrt((x.astype(np.float32) ** 2).mean()))


session = VoiceSession()
