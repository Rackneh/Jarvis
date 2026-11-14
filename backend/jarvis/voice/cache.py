"""Cached Conversation TTS: sentence, clause/phrase, and word-level audio cache.

When Cached Conversation mode is enabled, Jarvis caches synthesized PCM audio at
three granularities — full sentences, clauses/phrases (parts of sentences), and
individual words — namespaced by the active voice signature so cached clips
always match the current engine/voice/speed.

Before generating audio for a sentence:
1. Check for an exact normalized sentence match.
2. Otherwise split into clauses (parts of sentences) and match each clause
   against cached phrases or a dynamic-programming longest-match sequence of
   cached sub-phrases and words.
3. If the entire sentence is covered by cached parts/words, stitch them with
   silence trimming, raised-cosine cross-fade edges, and natural micro-pauses —
   automatically serving from cache with zero TTS generation.
4. If only some clauses/parts are covered, reuse the cached parts and generate
   only the uncached parts, while learning new words/phrases in the background.
"""
from __future__ import annotations

import queue
import re
import sqlite3
import sys
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from .. import config as cfg
from .audio import SAMPLE_RATE

_MD_RE = re.compile(r"\*\*|__|`|^#+\s*|^>\s*", re.M)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+|(?<=[.!?])\s+|\s+(?:—|--|-)\s+|\.{3,}\s*")
_WORD_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", re.I)

MAX_PHRASE_WORDS = 12


def _log(*args) -> None:
    print("[tts-cache]", *args, file=sys.stderr, flush=True)


def norm_sentence(text: str) -> str:
    """Normalize a full sentence while keeping terminal punctuation for prosody."""
    s = _MD_RE.sub("", text or "")
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def norm_phrase(text: str) -> str:
    """Normalize a clause/phrase/word key (punctuation-insensitive)."""
    words = tokenize_words(text)
    return " ".join(words)


def tokenize_words(text: str) -> list[str]:
    """Extract normalized word tokens, preserving contractions like i'm / don't."""
    s = _MD_RE.sub("", text or "")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    return [m.group(0).lower() for m in _WORD_RE.finditer(s)]


def split_clauses(sentence: str) -> list[str]:
    """Split a sentence into natural prosodic clauses / parts."""
    clean = re.sub(r"\s+", " ", _MD_RE.sub("", sentence or "")).strip()
    if not clean:
        return []
    parts = [p.strip(" ,;:-—") for p in _CLAUSE_SPLIT_RE.split(clean)]
    return [p for p in parts if p and tokenize_words(p)]


def trim_silence(pcm: np.ndarray, threshold: int = 180, margin_ms: float = 10.0) -> np.ndarray:
    """Trim dead leading/trailing silence while keeping a small natural margin."""
    if pcm.size == 0:
        return pcm
    idx = np.flatnonzero(np.abs(pcm) > threshold)
    if idx.size == 0:
        return pcm[: min(len(pcm), int(SAMPLE_RATE * 0.05))]
    margin = int(SAMPLE_RATE * margin_ms / 1000.0)
    start = max(0, int(idx[0]) - margin)
    end = min(len(pcm), int(idx[-1]) + 1 + margin)
    return pcm[start:end]


def apply_fade(pcm: np.ndarray, fade_ms: float = 6.0) -> np.ndarray:
    """Apply short raised-cosine fade-in/out to prevent clicks when stitching."""
    if pcm.size == 0:
        return pcm
    fade = min(int(SAMPLE_RATE * fade_ms / 1000.0), pcm.size // 4)
    if fade <= 1:
        return pcm
    out = pcm.astype(np.float32, copy=True)
    t = np.linspace(0.0, np.pi, fade, dtype=np.float32)
    win = 0.5 * (1.0 - np.cos(t))
    out[:fade] *= win
    out[-fade:] *= win[::-1]
    return np.clip(out, -32768.0, 32767.0).astype(np.int16)


def stitch_clips(clips: list[np.ndarray], gap_ms: float = 28.0, trim: bool = True) -> np.ndarray:
    """Concatenate PCM clips with trimmed edges, cosine fades, and micro-pauses."""
    clean = []
    for c in clips:
        if c is None or c.size == 0:
            continue
        seg = trim_silence(c) if trim else c
        seg = apply_fade(seg, fade_ms=6.0)
        if seg.size > 0:
            clean.append(seg)
    if not clean:
        return np.zeros(0, dtype=np.int16)
    if len(clean) == 1:
        return clean[0]
    gap = np.zeros(max(0, int(SAMPLE_RATE * gap_ms / 1000.0)), dtype=np.int16)
    pieces = []
    for i, seg in enumerate(clean):
        if i > 0 and gap.size > 0:
            pieces.append(gap)
        pieces.append(seg)
    return np.concatenate(pieces)


@dataclass
class CacheSegment:
    kind: str  # "cached_sentence" | "cached_clause" | "cached_stitched" | "uncached"
    text: str
    pcm: Optional[np.ndarray] = None
    match_detail: str = ""


class TTSConversationCache:
    """Persistent + in-memory multi-tier TTS audio cache."""

    CORE_PHRASES = (
        "Good morning, sir.",
        "Good evening, sir.",
        "At your service, sir.",
        "Right away, sir.",
        "Certainly, sir.",
        "One moment, sir.",
        "Conversation ended.",
        "All systems are online.",
        "All systems are ready.",
        "I have started the background task.",
        "The task is complete.",
        "By the way,",
        "In progress.",
        "Yes, sir.",
        "No, sir.",
        "Thank you, sir.",
    )
    CORE_WORDS = (
        "yes", "no", "okay", "sure", "sir", "jarvis", "ready", "done",
        "complete", "online", "offline", "started", "running", "paused",
        "cancelled", "failed", "good", "morning", "afternoon", "evening",
        "hello", "hi", "bye", "goodbye", "thanks", "please", "sorry",
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "now", "later", "today", "tomorrow", "here", "there",
        "all", "systems", "are", "is", "the", "a", "an", "and", "or", "but",
        "i", "you", "we", "it", "this", "that", "your", "my", "to", "for",
        "of", "in", "on", "at", "with", "from", "by", "as", "not", "task",
        "tasks", "background", "file", "files", "system", "status", "progress",
        "minute", "minutes", "second", "seconds", "hour", "hours", "percent",
        "check", "checking", "working", "waiting", "finished", "updated",
    )

    def __init__(self, signature: str) -> None:
        self.signature = re.sub(r"[^a-zA-Z0-9._-]+", "_", signature or "default")
        self._dir = cfg.home() / "cache" / "tts" / self.signature
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "cache.sqlite"
        self._lock = threading.RLock()

        self._sentences: dict[str, np.ndarray] = {}
        self._phrases: dict[str, np.ndarray] = {}
        self._words: dict[str, np.ndarray] = {}

        self._hits_sentence = 0
        self._hits_clause = 0
        self._hits_stitched = 0
        self._hits_partial = 0
        self._misses = 0

        self._bg_q: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=256)
        self._bg_synth: Optional[Callable[[str], Optional[np.ndarray]]] = None
        self._bg_busy: Optional[Callable[[], bool]] = None
        self._bg_started = False
        self._closed = False

        self._init_db()
        self._load_all()

    # ------------------------------------------------------------------ sqlite

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tts_cache (
                        tier TEXT NOT NULL,
                        key TEXT NOT NULL,
                        pcm BLOB NOT NULL,
                        samples INTEGER NOT NULL,
                        hits INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        last_used REAL NOT NULL,
                        PRIMARY KEY (tier, key)
                    )
                    """
                )

    def _load_all(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute("SELECT tier, key, pcm FROM tts_cache").fetchall()
        for tier, key, blob in rows:
            arr = np.frombuffer(blob, dtype=np.int16).copy()
            if arr.size == 0:
                continue
            if tier == "sentence":
                self._sentences[key] = arr
            elif tier == "phrase":
                self._phrases[key] = arr
            elif tier == "word":
                self._words[key] = arr

    def _store_entry(self, tier: str, key: str, pcm: np.ndarray) -> None:
        if not key or pcm is None or pcm.size == 0:
            return
        arr = np.asarray(pcm, dtype=np.int16)
        now = time.time()
        with self._lock:
            if tier == "sentence":
                self._sentences[key] = arr
            elif tier == "phrase":
                self._phrases[key] = arr
            elif tier == "word":
                self._words[key] = arr
            else:
                return
            try:
                with closing(self._connect()) as conn, conn:
                    conn.execute(
                        """
                        INSERT INTO tts_cache(tier, key, pcm, samples, hits, created_at, last_used)
                        VALUES (?, ?, ?, ?, 1, ?, ?)
                        ON CONFLICT(tier, key) DO UPDATE SET
                            pcm=excluded.pcm,
                            samples=excluded.samples,
                            last_used=excluded.last_used
                        """,
                        (tier, key, arr.tobytes(), int(arr.size), now, now),
                    )
            except Exception as exc:
                _log(f"sqlite write failed ({tier}/{key}): {exc}")

    def _bump_hit(self, tier: str, key: str) -> None:
        try:
            with self._lock, closing(self._connect()) as conn, conn:
                conn.execute(
                    "UPDATE tts_cache SET hits = hits + 1, last_used = ? WHERE tier = ? AND key = ?",
                    (time.time(), tier, key),
                )
        except Exception:
            pass

    # ------------------------------------------------------------------ lookups

    def put(self, text: str, pcm: np.ndarray) -> None:
        """Store synthesized PCM into the appropriate tier(s)."""
        if pcm is None or pcm.size == 0:
            return
        skey = norm_sentence(text)
        pkey = norm_phrase(text)
        words = tokenize_words(text)
        if not words:
            return
        if len(words) == 1:
            self._store_entry("word", words[0], trim_silence(pcm))
        else:
            if skey:
                self._store_entry("sentence", skey, pcm)
            if pkey and len(words) <= MAX_PHRASE_WORDS:
                self._store_entry("phrase", pkey, trim_silence(pcm))

    def _lookup_exact(self, text: str) -> Optional[tuple[str, np.ndarray]]:
        skey = norm_sentence(text)
        pkey = norm_phrase(text)
        words = tokenize_words(text)
        with self._lock:
            if skey and skey in self._sentences:
                return "sentence", self._sentences[skey]
            if pkey and pkey in self._phrases:
                return "phrase", self._phrases[pkey]
            if len(words) == 1 and words[0] in self._words:
                return "word", self._words[words[0]]
        return None

    def _dp_cover_words(self, words: list[str]) -> Optional[list[tuple[str, np.ndarray]]]:
        """Find minimum-segment 100% cache cover of `words` from phrases + words."""
        n = len(words)
        if n == 0:
            return []
        with self._lock:
            # dp[i] = (best_segment_count, next_index, matched_tier, pcm)
            dp: list[Optional[tuple[int, int, str, np.ndarray]]] = [None] * (n + 1)
            dp[n] = (0, n, "", np.zeros(0, dtype=np.int16))
            for i in range(n - 1, -1, -1):
                best = None
                max_len = min(MAX_PHRASE_WORDS, n - i)
                for span in range(max_len, 1, -1):
                    if dp[i + span] is None:
                        continue
                    cand = " ".join(words[i : i + span])
                    pcm = self._phrases.get(cand)
                    if pcm is not None:
                        cand_cost = 1 + dp[i + span][0]
                        if best is None or cand_cost < best[0]:
                            best = (cand_cost, i + span, "phrase", pcm)
                            break  # longest span with same/lower cost preferred
                if dp[i + 1] is not None:
                    w_pcm = self._words.get(words[i])
                    if w_pcm is not None:
                        cand_cost = 1 + dp[i + 1][0]
                        if best is None or cand_cost < best[0]:
                            best = (cand_cost, i + 1, "word", w_pcm)
                dp[i] = best
            if dp[0] is None:
                return None
            out: list[tuple[str, np.ndarray]] = []
            idx = 0
            while idx < n:
                _, nxt, tier, pcm = dp[idx]  # type: ignore[misc]
                out.append((tier, pcm))
                idx = nxt
            return out

    def _split_partial_clause(self, clause: str) -> list[CacheSegment]:
        """Extract cached prefix/suffix phrases or words around an uncached span."""
        words = tokenize_words(clause)
        n = len(words)
        if n <= 1:
            return [CacheSegment("uncached", clause)]

        with self._lock:
            # Find longest cached prefix (phrase >= 3 words, or contiguous cached words >= 3)
            prefix_end = 0
            prefix_pcm: Optional[np.ndarray] = None
            for span in range(min(MAX_PHRASE_WORDS, n - 1), 2, -1):
                cand = " ".join(words[:span])
                if cand in self._phrases:
                    prefix_end = span
                    prefix_pcm = self._phrases[cand]
                    break
            if prefix_end == 0 and n >= 4 and words[0] in self._words:
                # Chain contiguous leading cached words if >= 3 words
                clips = []
                k = 0
                while k < n - 1 and words[k] in self._words:
                    clips.append(self._words[words[k]])
                    k += 1
                if k >= 3:
                    prefix_end = k
                    prefix_pcm = stitch_clips(clips, gap_ms=26.0, trim=True)

            # Find longest cached suffix after prefix_end (phrase >= 3 words)
            suffix_start = n
            suffix_pcm: Optional[np.ndarray] = None
            rem = n - prefix_end
            for span in range(min(MAX_PHRASE_WORDS, rem - 1), 2, -1):
                cand = " ".join(words[n - span :])
                if cand in self._phrases:
                    suffix_start = n - span
                    suffix_pcm = self._phrases[cand]
                    break
            if suffix_start == n and rem >= 4:
                clips = []
                k = n
                while k - 1 > prefix_end and words[k - 1] in self._words:
                    clips.append(self._words[words[k - 1]])
                    k -= 1
                if n - k >= 3:
                    suffix_start = k
                    suffix_pcm = stitch_clips(list(reversed(clips)), gap_ms=26.0, trim=True)

        if prefix_end == 0 and suffix_start == n:
            return [CacheSegment("uncached", clause)]

        segs: list[CacheSegment] = []
        if prefix_pcm is not None and prefix_end > 0:
            segs.append(CacheSegment("cached_phrase", " ".join(words[:prefix_end]), prefix_pcm, "prefix"))
        if prefix_end < suffix_start:
            mid_text = " ".join(words[prefix_end:suffix_start])
            segs.append(CacheSegment("uncached", mid_text))
        if suffix_pcm is not None and suffix_start < n:
            segs.append(CacheSegment("cached_phrase", " ".join(words[suffix_start:]), suffix_pcm, "suffix"))
        return segs

    def plan_sentence(self, sentence: str) -> tuple[bool, list[CacheSegment]]:
        """Plan how to realize `sentence` from cache and/or generation.

        Returns `(fully_covered, segments)`. When `fully_covered` is True,
        every segment has `.pcm` populated and zero TTS generation is needed.
        """
        exact = self._lookup_exact(sentence)
        if exact is not None:
            tier, pcm = exact
            return True, [CacheSegment(f"cached_{tier}", sentence, pcm, "exact")]

        clauses = split_clauses(sentence)
        if not clauses:
            return False, []

        segments: list[CacheSegment] = []
        fully_covered = True
        for clause in clauses:
            c_exact = self._lookup_exact(clause)
            if c_exact is not None:
                tier, pcm = c_exact
                segments.append(CacheSegment(f"cached_{tier}", clause, pcm, "clause"))
                continue

            words = tokenize_words(clause)
            cover = self._dp_cover_words(words)
            if cover is not None and (len(cover) <= 3 or len(words) <= 4):
                stitched = stitch_clips([pcm for _, pcm in cover], gap_ms=26.0, trim=True)
                if stitched.size > 0:
                    tiers = "+".join(t for t, _ in cover)
                    segments.append(CacheSegment("cached_stitched", clause, stitched, tiers))
                    continue

            fully_covered = False
            segments.extend(self._split_partial_clause(clause))

        if not fully_covered:
            full_words = tokenize_words(sentence)
            full_cover = self._dp_cover_words(full_words)
            if full_cover is not None and (len(full_cover) <= 3 or len(full_words) <= 4):
                stitched = stitch_clips([pcm for _, pcm in full_cover], gap_ms=28.0, trim=True)
                if stitched.size > 0:
                    tiers = "+".join(t for t, _ in full_cover)
                    return True, [CacheSegment("cached_stitched", sentence, stitched, tiers)]

        # Coalesce adjacent uncached segments so cold text synthesizes in one natural pass
        merged: list[CacheSegment] = []
        for seg in segments:
            if seg.pcm is None and merged and merged[-1].pcm is None:
                merged[-1] = CacheSegment("uncached", f"{merged[-1].text}, {seg.text}")
            else:
                merged.append(seg)
        if not fully_covered:
            partial = self._split_partial_clause(sentence)
            if any(s.pcm is not None for s in partial):
                clause_cached_words = sum(
                    len(tokenize_words(s.text)) for s in merged if s.pcm is not None
                )
                partial_cached_words = sum(
                    len(tokenize_words(s.text)) for s in partial if s.pcm is not None
                )
                if partial_cached_words > clause_cached_words:
                    return False, partial
        return fully_covered, merged

    def is_covered(self, sentence: str) -> bool:
        """Return True when `sentence` can be spoken 100% from cache."""
        covered, _ = self.plan_sentence(sentence)
        return covered

    # ------------------------------------------------------------------ resolve

    def resolve_stream(
        self,
        sentence: str,
        synth_stream_fn: Callable[[str], Iterator[np.ndarray]],
        on_hit: Optional[Callable[[str, str], None]] = None,
    ) -> Iterator[np.ndarray]:
        """Yield 16 kHz int16 PCM blocks for `sentence`, using cache automatically."""
        if not sentence or not sentence.strip():
            return

        fully_covered, segments = self.plan_sentence(sentence)
        if not segments:
            return

        if fully_covered:
            kind = segments[0].kind if len(segments) == 1 else "cached_assembled"
            with self._lock:
                if kind == "cached_sentence":
                    self._hits_sentence += 1
                    self._bump_hit("sentence", norm_sentence(sentence))
                elif kind in ("cached_clause", "cached_phrase", "cached_word"):
                    self._hits_clause += 1
                else:
                    self._hits_stitched += 1
            if on_hit:
                try:
                    on_hit(kind, sentence)
                except Exception:
                    pass
            _log(f"cache HIT ({kind}, {len(segments)} part(s)): {sentence[:60]}")
            gap = np.zeros(int(SAMPLE_RATE * 0.075), dtype=np.int16)
            for idx, seg in enumerate(segments):
                if seg.pcm is None or seg.pcm.size == 0:
                    continue
                if idx > 0 and gap.size > 0:
                    yield gap
                yield apply_fade(seg.pcm, fade_ms=4.0)
            return

        # Partial cache hit or full cache miss
        had_cached = any(seg.pcm is not None for seg in segments)
        with self._lock:
            if had_cached:
                self._hits_partial += 1
            else:
                self._misses += 1
        if had_cached and on_hit:
            try:
                on_hit("cached_partial", sentence)
            except Exception:
                pass

        assembled_parts: list[np.ndarray] = []
        gap = np.zeros(int(SAMPLE_RATE * 0.045), dtype=np.int16)

        for idx, seg in enumerate(segments):
            if idx > 0 and gap.size > 0:
                yield gap
            if seg.pcm is not None and seg.pcm.size > 0:
                pcm = apply_fade(seg.pcm, fade_ms=4.0)
                assembled_parts.append(pcm)
                yield pcm
                continue

            # Synthesize only this uncached segment
            seg_blocks: list[np.ndarray] = []
            for block in synth_stream_fn(seg.text):
                if block is None or block.size == 0:
                    continue
                seg_blocks.append(block)
                yield block
            if seg_blocks:
                seg_pcm = seg_blocks[0] if len(seg_blocks) == 1 else np.concatenate(seg_blocks)
                self.put(seg.text, seg_pcm)
                assembled_parts.append(trim_silence(seg_pcm))

        # Store full sentence + clauses for instant future reuse
        if assembled_parts:
            full_pcm = stitch_clips(assembled_parts, gap_ms=45.0, trim=False)
            self.put(sentence, full_pcm)
            self._enqueue_learning(sentence)

    # ------------------------------------------------------------------ background learner

    def start_background_worker(
        self,
        synth_fn: Callable[[str], Optional[np.ndarray]],
        is_busy_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._bg_synth = synth_fn
        self._bg_busy = is_busy_fn
        if self._bg_started:
            return
        self._bg_started = True
        threading.Thread(target=self._bg_worker, name="jarvis-tts-cache-learner", daemon=True).start()

    def _enqueue_learning(self, sentence: str) -> None:
        if not self._bg_started:
            return
        clauses = split_clauses(sentence)
        words = tokenize_words(sentence)
        with self._lock:
            missing_clauses = [
                c for c in clauses
                if 2 <= len(tokenize_words(c)) <= MAX_PHRASE_WORDS
                and norm_phrase(c) not in self._phrases
            ]
            missing_words = [w for w in dict.fromkeys(words) if w not in self._words]
            missing_bigrams = []
            for i in range(len(words) - 1):
                bg = f"{words[i]} {words[i+1]}"
                if bg not in self._phrases:
                    missing_bigrams.append(bg)
        for cl in missing_clauses[:8]:
            try:
                self._bg_q.put_nowait(("phrase", cl))
            except queue.Full:
                break
        for w in missing_words[:24]:
            try:
                self._bg_q.put_nowait(("word", w))
            except queue.Full:
                break
        for bg in missing_bigrams[:12]:
            try:
                self._bg_q.put_nowait(("phrase", bg))
            except queue.Full:
                break

    def seed_defaults_async(self) -> None:
        """Queue core Jarvis phrases and words for background pre-caching."""
        if not self._bg_started:
            return
        with self._lock:
            for ph in self.CORE_PHRASES:
                if norm_sentence(ph) not in self._sentences:
                    try:
                        self._bg_q.put_nowait(("sentence", ph))
                    except queue.Full:
                        break
            for w in self.CORE_WORDS:
                if w not in self._words:
                    try:
                        self._bg_q.put_nowait(("word", w))
                    except queue.Full:
                        break

    def _bg_worker(self) -> None:
        while not self._closed:
            try:
                tier, text = self._bg_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if self._closed or self._bg_synth is None:
                continue
            # Yield to live speech
            while not self._closed and self._bg_busy is not None and self._bg_busy():
                time.sleep(0.15)
            key = norm_sentence(text) if tier == "sentence" else norm_phrase(text)
            with self._lock:
                already = (
                    (tier == "sentence" and key in self._sentences)
                    or (tier == "phrase" and key in self._phrases)
                    or (tier == "word" and key in self._words)
                )
            if already:
                continue
            try:
                pcm = self._bg_synth(text)
                if pcm is not None and pcm.size > 0:
                    if tier == "word":
                        self._store_entry("word", key, trim_silence(pcm))
                    elif tier == "phrase":
                        self._store_entry("phrase", key, trim_silence(pcm))
                    else:
                        self.put(text, pcm)
            except Exception:
                pass
            time.sleep(0.01)

    def clear(self) -> None:
        with self._lock:
            self._sentences.clear()
            self._phrases.clear()
            self._words.clear()
            self._hits_sentence = 0
            self._hits_clause = 0
            self._hits_stitched = 0
            self._hits_partial = 0
            self._misses = 0
            with closing(self._connect()) as conn, conn:
                conn.execute("DELETE FROM tts_cache")

    def stats(self) -> dict:
        with self._lock:
            total_hits = self._hits_sentence + self._hits_clause + self._hits_stitched
            return {
                "signature": self.signature,
                "sentences": len(self._sentences),
                "phrases": len(self._phrases),
                "words": len(self._words),
                "hits_sentence": self._hits_sentence,
                "hits_clause": self._hits_clause,
                "hits_stitched": self._hits_stitched,
                "hits_partial": self._hits_partial,
                "hits_total": total_hits,
                "misses": self._misses,
            }

    def close(self) -> None:
        self._closed = True
