"""Verification & benchmark for GPU Jarvis ONNX TTS and Cached Conversation mode."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import onnxruntime as ort
from jarvis import config as cfg
from jarvis.voice.cache import TTSConversationCache
from jarvis.voice.tts import (
    SpeechQueue,
    SpeechStreamer,
    TTSEngine,
    clean_for_speech,
    split_spoken_sentences,
)


def test_directml_available() -> None:
    providers = ort.get_available_providers()
    print("ORT providers:", providers)
    assert "DmlExecutionProvider" in providers, f"DmlExecutionProvider missing: {providers}"


def test_cached_conversation_unit() -> None:
    """Verify sentence, clause/phrase, word-coverage, and partial hybrid caching."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        orig_home = cfg.home
        cfg.home = lambda: Path(td)  # type: ignore[assignment]
        cache = None
        try:
            cache = TTSConversationCache("unit_test_voice")
            synth_calls: list[str] = []

            def fake_synth(text: str):
                synth_calls.append(text)
                # Return non-silent deterministic 16kHz audio proportional to char count
                n = max(1600, len(text) * 400)
                t = np.arange(n, dtype=np.float32)
                pcm = (np.sin(2.0 * np.pi * 220.0 * t / 16000.0) * 6000.0).astype(np.int16)
                yield pcm

            # 1) Exact sentence miss -> populate -> instant sentence hit (0 synth calls)
            blocks1 = list(cache.resolve_stream("At your service, sir.", fake_synth))
            assert blocks1 and len(synth_calls) == 1
            synth_calls.clear()
            t0 = time.perf_counter()
            blocks2 = list(cache.resolve_stream("At your service, sir.", fake_synth))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert blocks2 and len(synth_calls) == 0, "Exact sentence should come from cache"
            print(f"[ok] Test 1 exact sentence hit in {dt_ms:.2f} ms (0 synth calls)")

            # 2) Clause / part-of-sentence hit
            list(cache.resolve_stream("Good evening, sir.", fake_synth))
            list(cache.resolve_stream("All systems are online.", fake_synth))
            synth_calls.clear()
            combined = "Good evening, sir, all systems are online."
            assert cache.is_covered(combined), "Combined sentence should be 100% covered by cached clauses"
            blocks3 = list(cache.resolve_stream(combined, fake_synth))
            assert blocks3 and len(synth_calls) == 0, f"Expected 0 synth calls, got {synth_calls}"
            print("[ok] Test 2 clause-assembled sentence hit (0 synth calls)")

            # 3) Word-level coverage hit
            for w in ("yes", "sir", "ready", "now"):
                pcm = next(fake_synth(w))
                cache.put(w, pcm)
            synth_calls.clear()
            word_sentence = "Yes sir, ready now."
            assert cache.is_covered(word_sentence), "Sentence made of cached words must be covered"
            blocks4 = list(cache.resolve_stream(word_sentence, fake_synth))
            assert blocks4 and len(synth_calls) == 0, f"Expected 0 synth calls, got {synth_calls}"
            print("[ok] Test 3 word-level coverage hit (0 synth calls)")

            # 4) Partial cache + generation hybrid (1 cached clause + 1 uncached clause)
            synth_calls.clear()
            hybrid = "Good evening, sir, reactor diagnostics look nominal."
            assert not cache.is_covered(hybrid)
            blocks5 = list(cache.resolve_stream(hybrid, fake_synth))
            assert blocks5 and synth_calls == ["reactor diagnostics look nominal"], (
                f"Expected only uncached clause to synthesize, got {synth_calls}"
            )
            print("[ok] Test 4 partial cache + uncached clause hybrid:", synth_calls)

            stats = cache.stats()
            print("Cache stats:", stats)
            assert stats["hits_sentence"] >= 1
            assert stats["hits_stitched"] >= 2
            assert stats["hits_partial"] >= 1
        finally:
            if cache is not None:
                cache.close()
            cfg.home = orig_home  # type: ignore[assignment]


def test_real_jarvis_engine_and_cache() -> None:
    """Benchmark live TTSEngine (jarvis_clone on GPU) + live Cached Conversation."""
    old_cached = cfg.get("voice.cached_conversation", None)
    cfg.update("voice.cached_conversation", True)
    engine = TTSEngine()
    try:
        print(f"Active TTS engine: {engine.engine_name} ({engine.provider_name})")
        assert engine.engine_name == "jarvis_clone", f"Unexpected engine: {engine.engine_name}"
        assert engine.provider_name == "DirectML GPU", f"Expected DirectML GPU, got {engine.provider_name}"

        # Cold generation (unique phrase unlikely to be in cache yet)
        s1 = f"Telemetry sweep confirms nominal thermal headroom across cycle {int(time.time()) % 1000}."
        t0 = time.perf_counter()
        pcm1 = engine.synthesize(s1)
        gen_ms = (time.perf_counter() - t0) * 1000.0
        assert pcm1 is not None and len(pcm1) > 8000
        print(f"[ok] Live GPU Jarvis synth: {gen_ms:.1f} ms for {len(pcm1)/16000:.2f}s audio")

        # Repeat exact sentence -> instant cache hit
        t1 = time.perf_counter()
        pcm2 = engine.synthesize(s1)
        hit_ms = (time.perf_counter() - t1) * 1000.0
        assert pcm2 is not None and len(pcm2) > 8000
        assert len(pcm2) >= int(len(pcm1) * 0.75) and len(pcm2) <= len(pcm1)
        assert hit_ms < 60.0, f"Cache hit took too long: {hit_ms:.2f} ms"
        pcm3 = engine.synthesize(s1)
        assert pcm3 is not None and len(pcm3) == len(pcm2)
        print(f"[ok] Live sentence cache hit: {hit_ms:.2f} ms ({len(pcm2)} samples)")

        # Verify turning cache OFF disables cache hits
        cfg.update("voice.cached_conversation", False)
        assert not engine.cached_conversation_enabled()
        print("[ok] Explicit cache disable confirmed")
    finally:
        if old_cached is not None:
            cfg.update("voice.cached_conversation", old_cached)
        engine.close()


def test_speech_queue_and_streamer_ordering() -> None:
    """Verify numbered-list splitting, SpeechStreamer FIFO queueing, SpeakerSink recovery, and agent turn serialization."""
    import asyncio
    from unittest.mock import patch
    from jarvis.conscious.agent import SENTENCE_SPLIT, ConsciousAgent
    from jarvis.memory import store
    from jarvis.providers.base import StreamChunk
    from jarvis.voice.audio import SpeakerSink

    # 1) Numbered lists, markdown, and Windows paths must not break mid-item
    sample = (
        'Certainly; the last two tasks performed were: 1. **Task:** Writing Hello World File (Completed) '
        '* *Result:* Content written to `Desktop/hello.txt`. '
        '2. The preceding task was checking ""D:\\Games\\Myth Makers Orbs of Doom.bin"".'
    )
    cleaned = clean_for_speech(sample)
    assert "**" not in cleaned and "`" not in cleaned and '""' not in cleaned
    parts = split_spoken_sentences(sample)
    assert len(parts) == 2, f"Expected 2 clean sentences, got {parts}"
    assert parts[0].startswith("Certainly; the last two tasks performed were: 1. Task:")
    assert parts[1].startswith("2. The preceding task was checking")
    # SENTENCE_SPLIT in agent.py must also not isolate "1." or "2."
    agent_parts = [p.strip() for p in SENTENCE_SPLIT.split(cleaned) if p.strip()]
    assert agent_parts == parts, f"Agent split mismatch: {agent_parts} vs {parts}"

    # 2) SpeechStreamer.feed + SpeechQueue FIFO playback order (including no trailing space)
    class _FakeEngine:
        engine_name = "fake"
        last_error = None
        synthesizing = False

        def set_busy_probe(self, fn):
            self._probe = fn

        def synthesize_stream(self, sentence: str):
            # Vary synthesis latency so slower first sentence must still play before faster second sentence
            time.sleep(0.05 if "First" in sentence else 0.005)
            yield np.full(800, 1000 if "First" in sentence else 2000, dtype=np.int16)

    played: list[int] = []

    class _FakeSink:
        def __init__(self) -> None:
            self.stopped = 0

        def play(self, block: np.ndarray) -> None:
            played.append(int(block[0]))
            time.sleep(0.02)

        def stop(self) -> None:
            self.stopped += 1

    fake_sink = _FakeSink()
    sq = SpeechQueue(_FakeEngine(), fake_sink)  # type: ignore[arg-type]
    streamer = SpeechStreamer(sq)
    streamer.feed("First sentence without trailing space.")
    streamer.feed("Second sentence queued right after.")
    streamer.feed("Third fragment")
    assert sq.is_active, "Queue should be active immediately after feeding complete sentences"
    streamer.flush()
    deadline = time.time() + 5.0
    while sq.is_active and time.time() < deadline:
        time.sleep(0.02)
    assert not sq.is_active
    assert played == [1000, 2000, 2000], f"Out-of-order or dropped playback: {played}"

    # 3) SpeakerSink recovery after stop()
    class _FakeStream:
        def __init__(self) -> None:
            self.active = True
            self.writes = 0
            self.aborted = 0
            self.closed = 0

        def start(self) -> None:
            self.active = True

        def write(self, block) -> None:
            if not self.active:
                raise RuntimeError("PortAudio stream stopped")
            self.writes += 1

        def abort(self) -> None:
            self.active = False
            self.aborted += 1

        def close(self) -> None:
            self.active = False
            self.closed += 1

    created_streams: list[_FakeStream] = []
    with patch("jarvis.voice.audio.sd.OutputStream", side_effect=lambda **kw: created_streams.append(_FakeStream()) or created_streams[-1]):
        sink = SpeakerSink()
        sink.play(np.ones(1024, dtype=np.int16))
        sink.stop()
        sink.play(np.ones(1024, dtype=np.int16))
        assert len(created_streams) == 2 and created_streams[0].aborted == 1 and created_streams[1].writes == 1

    # 4) ConsciousAgent.handle_utterance serializes overlapping calls instead of dropping without done
    class _DoneProfile:
        def is_complete(self) -> bool:
            return True

        def progress(self) -> float:
            return 1.0

        def summary(self) -> str:
            return ""

    agent = ConsciousAgent()
    turn_idx = 0

    async def _fake_stream(provider, messages, tools=None, temperature=0.6, max_tokens=None, cancel=None):
        nonlocal turn_idx
        turn_idx += 1
        my_idx = turn_idx

        async def _gen():
            await asyncio.sleep(0.05 if my_idx == 1 else 0.01)
            yield StreamChunk(content=f"Reply number {my_idx}.")
            yield StreamChunk(finish_reason="stop", tool_calls=[])

        return _gen()

    async def _run_both():
        with patch("jarvis.conscious.agent.open_chat_stream", side_effect=_fake_stream), \
             patch("jarvis.conscious.agent.store.recent_messages", return_value=[]), \
             patch("jarvis.conscious.agent.store.add_message"), \
             patch("jarvis.conscious.agent.store.all_tasks", return_value=[]), \
             patch("jarvis.conscious.agent.store.pending_questions", return_value=[]), \
             patch("jarvis.conscious.agent.get_profile", return_value=_DoneProfile()):
            async def _collect(prompt: str):
                evs = []
                async for ev in agent.handle_utterance(prompt, source="console"):
                    evs.append(ev)
                return evs

            return await asyncio.gather(
                _collect("First prompt"),
                _collect("Second prompt"),
            )

    res1, res2 = asyncio.run(_run_both())
    s1 = [e["text"] for e in res1 if e["type"] == "sentence"]
    s2 = [e["text"] for e in res2 if e["type"] == "sentence"]
    assert s1 == ["Reply number 1."] and any(e["type"] == "done" for e in res1), res1
    assert s2 == ["Reply number 2."] and any(e["type"] == "done" for e in res2), res2

    # 5) _Bridge._relay_speech and /api/say speak even while VoiceSession.state == LISTENING
    import jarvis.server as srv
    from jarvis.voice.session import VoiceState

    spoken_calls: list[str] = []

    class _FakeSession:
        state = VoiceState.LISTENING
        speech = sq

        def speak_text(self, text: str) -> None:
            spoken_calls.append(text)

    async def _fake_say_turn(text: str, source: str = "console"):
        yield {"type": "sentence", "text": "Typed response one."}
        yield {"type": "sentence", "text": "Typed response two."}
        yield {"type": "done", "full": "Typed response one. Typed response two."}

    with patch.object(srv, "session", _FakeSession()), \
         patch.object(srv.store, "add_message"), \
         patch.object(srv.agent, "handle_utterance", side_effect=_fake_say_turn):
        srv._Bridge()._relay_speech("tid1", "Background task finished.")
        asyncio.run(srv.say(srv.Utterance(text="Status please")))

    assert spoken_calls == [
        "Background task finished.",
        "Typed response one.",
        "Typed response two.",
    ], f"Expected all relay + /api/say sentences spoken in order, got {spoken_calls}"

    print("[ok] Speech queue, streamer, SpeakerSink recovery, agent serialization, and server speech routing verified")


def test_mic_mute_and_background_cache_tts() -> None:
    """Verify TTS plays even when mic is muted (VoiceState.OFF) and bg cache doesn't stall queue."""
    import asyncio
    from unittest.mock import patch
    import jarvis.server as srv
    from jarvis.voice.session import VoiceSession, VoiceState

    class _FastFakeEngine:
        engine_name = "fake"
        last_error = None
        synthesizing = False

        def set_busy_probe(self, fn):
            self._probe = fn

        def warm(self):
            pass

        def synthesize_stream(self, sentence: str):
            yield np.full(800, 1200, dtype=np.int16)

    class _RecordingSink:
        def __init__(self) -> None:
            self.played = 0
            self.stopped = 0

        def play(self, block: np.ndarray) -> None:
            self.played += 1
            time.sleep(0.02)

        def stop(self) -> None:
            self.stopped += 1

    async def _run_mute_test() -> None:
        vs = VoiceSession()
        vs._loop = asyncio.get_running_loop()
        vs.tts_engine = _FastFakeEngine()  # type: ignore[assignment]
        sink = _RecordingSink()
        vs.sink = sink  # type: ignore[assignment]
        vs.speech = SpeechQueue(vs.tts_engine, sink)  # type: ignore[arg-type]

        # Mute microphone without killing TTS
        vs.set_mic_muted(True)
        assert vs._mic_muted is True and vs.state == VoiceState.OFF

        with patch.object(srv, "session", vs), patch.object(srv.store, "add_message"):
            srv._Bridge().on_task_problem("tid_mute", "Test Task", "Simulated error")

        assert sink.stopped == 0, "set_mic_muted(True) must not abort playback"
        # Wait for drain task to complete and return state to OFF
        deadline = time.time() + 5.0
        while (vs.speech.is_active or (vs._drain_task and not vs._drain_task.done())) and time.time() < deadline:
            await asyncio.sleep(0.02)
        assert sink.played >= 1, f"Expected TTS playback even with mic muted, played={sink.played}"
        assert vs.state == VoiceState.OFF, f"Expected return to OFF when muted, got {vs.state}"

    asyncio.run(_run_mute_test())

    # Background cache synthesis (_synthesize_raw_single) must not raise TTSEngine.synthesizing
    eng = object.__new__(TTSEngine)
    eng._synthesizing = 0
    eng._breeze = None
    eng._breeze_failed = True
    eng._synth_busy_fn = None
    eng._synthesize_other = lambda text: np.full(400, 500, dtype=np.int16)  # type: ignore[method-assign]

    observed_during_bg: list[bool] = []
    orig_synth = eng._synthesize_other

    def _spy(text: str):
        observed_during_bg.append(eng.synthesizing)
        return orig_synth(text)

    eng._synthesize_other = _spy  # type: ignore[method-assign]
    pcm = eng._synthesize_raw_single("background check")
    assert pcm is not None and observed_during_bg == [False], (
        f"Background synthesis must not mark engine.synthesizing=True: {observed_during_bg}"
    )
    list(eng._synthesize_raw_stream("foreground check", background=False))
    assert observed_during_bg[-1] is True
    assert eng.synthesizing is False
    print("[ok] Mic mute TTS decoupling & background cache isolation verified")


if __name__ == "__main__":
    test_directml_available()
    test_cached_conversation_unit()
    test_speech_queue_and_streamer_ordering()
    test_mic_mute_and_background_cache_tts()
    test_real_jarvis_engine_and_cache()
    print("ALL TTS & CACHE TESTS PASSED")
