#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""TTS caching mixin for reducing API costs on repeated phrases.

Designed for Pipecat's ``TTSService`` contract (``pipecat-ai>=1.5.0``). The mixin
aligns with the framework's audio-context model: synthesis is keyed by the
``context_id`` that Pipecat assigns to each request, audio is captured as it flows
through ``push_frame`` (the single point every audio frame passes through, whether a
service yields it synchronously or delivers it asynchronously from a websocket receive
loop), and cached audio is replayed by yielding it back through the same audio context
so it obeys the same playback ordering as live audio.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

from .backends.base import CacheBackend
from .key_generator import generate_cache_key
from .models import CachedAudioChunk, CachedTTSResponse, CachedWordTimestamp

_CACHE_ORIGIN_ATTR = "_tts_cache_origin"


@dataclass
class _PendingTask:
    """One text pending a cache write within an audio context."""

    text: str
    cache_key: str
    word_count: int


@dataclass
class _ContextCapture:
    """Audio and word timestamps captured for a single ``context_id``.

    A context may hold more than one pending task when the TTS service reuses one
    audio context across multiple sentences within an LLM turn (Pipecat's default
    ``reuse_context_id_within_turn=True``).
    """

    tasks: List[_PendingTask] = field(default_factory=list)
    audio: List[CachedAudioChunk] = field(default_factory=list)
    word_timestamps: List[Tuple[str, float]] = field(default_factory=list)


class TTSCacheMixin:
    """Mixin that adds caching to any Pipecat ``TTSService`` subclass.

    Usage: ``class CachedTTS(TTSCacheMixin, SomeTTSService): pass``
    """

    def __init__(
        self,
        *args,
        cache_backend: Optional[CacheBackend] = None,
        cache_ttl: Optional[int] = 86400,
        cache_namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize TTS cache mixin.

        Args:
            *args: Positional arguments passed to parent class.
            cache_backend: Cache backend instance. If None, caching is disabled.
            cache_ttl: Time-to-live for cache entries in seconds.
            cache_namespace: Optional namespace prefix for cache keys.
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self._cache_backend = cache_backend
        self._cache_ttl = cache_ttl
        self._cache_namespace = cache_namespace
        self._enable_cache = cache_backend is not None

        self._cache_hits = 0
        self._cache_misses = 0
        # Per-context capture state, keyed by Pipecat's context_id.
        self._contexts: Dict[str, _ContextCapture] = {}

        if self._enable_cache:
            logger.info(
                f"TTS caching enabled: backend={type(cache_backend).__name__}, "
                f"ttl={cache_ttl}s, namespace={cache_namespace or 'default'}"
            )
        else:
            logger.debug("TTS caching disabled: no backend provided")

    def _generate_cache_key(self, text: str) -> str:
        """Generate a cache key for the current TTS request.

        Reads the request-identifying state from the service's ``TTSSettings`` store
        (``voice``, ``model`` and the remaining runtime settings) plus the resolved
        output sample rate. ``given_fields()`` returns only the settings that have a
        value, so new provider settings are captured automatically.
        """
        given = self._settings.given_fields()
        extra_settings = {k: v for k, v in given.items() if k not in ("voice", "model")}
        return generate_cache_key(
            text=text,
            voice_id=given.get("voice"),
            model=given.get("model"),
            sample_rate=self.sample_rate,
            settings=extra_settings,
            namespace=self._cache_namespace,
            provider=self._wrapped_service_id(),
        )

    def _wrapped_service_id(self) -> Optional[str]:
        """Stable identifier of the wrapped TTS service class.

        Folded into the cache key so two different services that happen to share the
        same voice/model never collide on a shared backend. Resolved as the first real
        ``TTSService`` in the MRO, so an intermediate mixin in the composition does not
        mask (and alias) the actual provider.
        """
        for cls in type(self).__mro__:
            if issubclass(cls, TTSCacheMixin):
                continue
            if issubclass(cls, TTSService):
                return f"{cls.__module__}.{cls.__qualname__}"
        return None

    def _parse_words_from_text(self, text: str) -> List[str]:
        """Parse words from text to match TTS word segmentation."""
        cleaned = re.sub(r"[^\w\s']", "", text)
        return [w for w in cleaned.split() if w]

    async def run_tts(self, text: str, context_id: str):
        """Run TTS with caching support.

        On a cache hit, replays stored audio (and word timestamps) into the audio
        context. On a miss, delegates to the wrapped service while registering the
        request so its audio can be captured in ``push_frame``.
        """
        if not self._enable_cache:
            async for frame in super().run_tts(text, context_id):
                yield frame
            return

        try:
            cache_key = self._generate_cache_key(text)
        except Exception as e:
            # Fail-safe: never let cache-key generation break synthesis. Drop any capture
            # context for this id first so this un-keyed request's audio is not appended to
            # a sibling that reuses the same context within the turn, then synthesize.
            logger.warning(f"Cache key generation failed, bypassing cache: {e}")
            self._contexts.pop(context_id, None)
            async for frame in super().run_tts(text, context_id):
                yield frame
            return

        cached_response = await self._safe_cache_get(cache_key)

        if cached_response:
            self._cache_hits += 1
            logger.debug(f"Cache hit: '{text[:50]}' ({len(cached_response.audio_chunks)} chunks)")
            async for frame in self._yield_cached_frames(cached_response, context_id):
                yield frame
            return

        self._cache_misses += 1
        logger.debug(f"Cache miss: '{text[:50]}'")

        context = self._contexts.setdefault(context_id, _ContextCapture())
        context.tasks.append(
            _PendingTask(
                text=text,
                cache_key=cache_key,
                word_count=len(self._parse_words_from_text(text)),
            )
        )

        try:
            async for frame in super().run_tts(text, context_id):
                yield frame
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            self._contexts.pop(context_id, None)
            raise

    async def _safe_cache_get(self, key: str) -> Optional[CachedTTSResponse]:
        """Get from cache with error handling."""
        assert self._cache_backend is not None  # only called when caching is enabled
        try:
            return await self._cache_backend.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    async def _yield_cached_frames(self, cached: CachedTTSResponse, context_id: str):
        """Replay cached audio (and word timestamps) into the audio context.

        Word timestamps are replayed first so the framework regenerates the
        ``TTSTextFrame``s (transcript / word alignment) before the audio, mirroring how
        a word-timestamp-capable service behaves live. Audio frames are stamped with the
        active ``context_id`` and tagged as cache-origin so they are not re-captured.
        Start/stop framing is left to the base class, which brackets every audio context.
        """
        if cached.word_timestamps:
            word_times = [(wt.word, wt.timestamp) for wt in cached.word_timestamps]
            if hasattr(super(), "add_word_timestamps"):
                await super().add_word_timestamps(word_times, context_id=context_id)

        for chunk in cached.audio_chunks:
            frame = TTSAudioRawFrame(
                audio=chunk.audio,
                sample_rate=chunk.sample_rate,
                num_channels=chunk.num_channels,
                context_id=context_id,
            )
            setattr(frame, _CACHE_ORIGIN_ATTR, True)
            yield frame

    def _is_from_cache(self, frame: Frame) -> bool:
        """Check if a frame originated from cache replay."""
        return getattr(frame, _CACHE_ORIGIN_ATTR, False)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Intercept audio frames for caching, keyed by their audio context."""
        if self._enable_cache and not self._is_from_cache(frame):
            context_id = getattr(frame, "context_id", None)
            if context_id is not None and context_id in self._contexts:
                if isinstance(frame, TTSAudioRawFrame):
                    self._contexts[context_id].audio.append(
                        CachedAudioChunk(
                            audio=frame.audio,
                            sample_rate=frame.sample_rate,
                            num_channels=frame.num_channels,
                            pts=getattr(frame, "pts", None),
                        )
                    )
                elif isinstance(frame, TTSStoppedFrame):
                    await self._finalize_context(context_id)

        await super().push_frame(frame, direction)

    async def add_word_timestamps(
        self,
        word_times: List[Tuple[str, float]],
        context_id: Optional[str] = None,
        includes_inter_frame_spaces: Optional[bool] = None,
        pre_merge_tokens: bool = False,
    ):
        """Intercept word timestamps for caching, then forward them unchanged."""
        if self._enable_cache and context_id is not None and context_id in self._contexts:
            self._contexts[context_id].word_timestamps.extend(word_times)

        if hasattr(super(), "add_word_timestamps"):
            await super().add_word_timestamps(
                word_times,
                context_id=context_id,
                includes_inter_frame_spaces=includes_inter_frame_spaces,
                pre_merge_tokens=pre_merge_tokens,
            )

    async def _finalize_context(self, context_id: str) -> None:
        """Store the audio captured for a completed audio context."""
        context = self._contexts.pop(context_id, None)
        if context is None or not context.tasks:
            return

        if not context.audio:
            logger.warning(
                f"No audio captured for {len(context.tasks)} task(s) in context "
                f"{context_id}, skipping cache"
            )
            return

        all_audio = b"".join(chunk.audio for chunk in context.audio)
        sample_rate = context.audio[0].sample_rate
        num_channels = context.audio[0].num_channels

        if len(context.tasks) == 1:
            await self._store_task(
                context.tasks[0], all_audio, sample_rate, num_channels, context.word_timestamps
            )
        else:
            await self._store_split_tasks(context, all_audio, sample_rate, num_channels)

    async def _store_task(
        self,
        task: _PendingTask,
        audio: bytes,
        sample_rate: int,
        num_channels: int,
        word_timestamps: List[Tuple[str, float]],
    ) -> None:
        """Store a single task's full audio (with word timestamps if any were emitted)."""
        duration = self._audio_duration(audio, sample_rate, num_channels)
        timestamps = [CachedWordTimestamp(word=w, timestamp=t) for w, t in word_timestamps]
        await self._store_response(task, audio, sample_rate, num_channels, timestamps, duration)

    async def _store_split_tasks(
        self, context: _ContextCapture, all_audio: bytes, sample_rate: int, num_channels: int
    ) -> None:
        """Split one audio context across multiple tasks at word boundaries.

        Only possible when word timestamps were emitted and the locally-parsed word
        count matches the number of timestamps received; otherwise the batch is skipped
        (the audio cannot be reliably attributed to individual texts).
        """
        if not context.word_timestamps:
            logger.debug(
                f"Cannot split {len(context.tasks)} cached texts without word "
                "timestamps, skipping cache"
            )
            return

        total_expected = sum(t.word_count for t in context.tasks)
        actual = len(context.word_timestamps)
        if total_expected != actual:
            logger.debug(
                f"Word count mismatch for context (expected {total_expected}, got "
                f"{actual}), cannot split reliably, skipping cache"
            )
            return

        # Integrity guard: slicing audio by word-timestamp boundaries is only sound when
        # the timestamps are non-decreasing. If a provider ever emits non-monotonic times
        # (e.g. a mid-turn cumulative reset), skip the whole batch rather than risk
        # mis-attributing audio to the wrong sentence (silent corruption -> safe skip).
        times = [t for _, t in context.word_timestamps]
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            logger.debug("Non-monotonic word timestamps for context, skipping split cache")
            return

        bytes_per_sample = 2 * num_channels  # 16-bit PCM
        total_duration = self._audio_duration(all_audio, sample_rate, num_channels)
        word_idx = 0

        for task in context.tasks:
            task_timestamps = context.word_timestamps[word_idx : word_idx + task.word_count]
            if not task_timestamps:
                word_idx += task.word_count
                continue

            start_time = task_timestamps[0][1]
            next_idx = word_idx + task.word_count
            if next_idx < len(context.word_timestamps):
                end_time = context.word_timestamps[next_idx][1]
            else:
                end_time = total_duration

            start_byte = (
                int(start_time * sample_rate * bytes_per_sample) // bytes_per_sample
            ) * bytes_per_sample
            end_byte = (
                int(end_time * sample_rate * bytes_per_sample) // bytes_per_sample
            ) * bytes_per_sample
            start_byte = max(0, start_byte)
            end_byte = min(len(all_audio), end_byte)

            task_audio = all_audio[start_byte:end_byte]
            if not task_audio:
                logger.warning(f"Empty audio slice for '{task.text[:30]}', skipping")
                word_idx += task.word_count
                continue

            normalized = [
                CachedWordTimestamp(word=w, timestamp=t - start_time) for w, t in task_timestamps
            ]
            await self._store_response(
                task, task_audio, sample_rate, num_channels, normalized, end_time - start_time
            )
            word_idx += task.word_count

    @staticmethod
    def _audio_duration(audio: bytes, sample_rate: int, num_channels: int) -> float:
        """Duration in seconds for 16-bit PCM audio."""
        bytes_per_sample = 2 * num_channels
        denom = sample_rate * bytes_per_sample
        return len(audio) / denom if denom else 0.0

    async def _store_response(
        self,
        task: _PendingTask,
        audio: bytes,
        sample_rate: int,
        num_channels: int,
        timestamps: List[CachedWordTimestamp],
        duration: float,
    ) -> None:
        """Write a completed response to the cache backend (fail-safe)."""
        assert self._cache_backend is not None  # only reached on a cache miss (enabled)
        try:
            cached_response = CachedTTSResponse(
                audio_chunks=[CachedAudioChunk(audio, sample_rate, num_channels)],
                sample_rate=sample_rate,
                num_channels=num_channels,
                word_timestamps=timestamps,
                total_duration_s=duration,
                metadata={
                    "text": task.text,
                    "audio_bytes": len(audio),
                    "word_count": len(timestamps),
                },
            )
            success = await self._cache_backend.set(
                task.cache_key, cached_response, ttl=self._cache_ttl
            )
            if success:
                logger.debug(f"Cached: '{task.text[:50]}' ({len(audio)} bytes)")
        except Exception as e:
            logger.error(f"Error caching '{task.text[:30]}': {e}")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Discard in-flight cache captures on interruption."""
        if self._contexts:
            logger.debug(f"Interruption - clearing {len(self._contexts)} pending cache context(s)")
            self._contexts.clear()

        if hasattr(super(), "_handle_interruption"):
            await super()._handle_interruption(frame, direction)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        stats = {
            "enabled": self._enable_cache,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total,
        }

        if self._cache_backend:
            try:
                stats["backend"] = await self._cache_backend.get_stats()
            except Exception as e:
                logger.error(f"Error getting backend stats: {e}")
                stats["backend"] = {"error": str(e)}

        return stats

    async def clear_cache(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries."""
        if not self._cache_backend:
            logger.warning("Cannot clear cache: no backend configured")
            return 0

        try:
            cleared = await self._cache_backend.clear(namespace)
            logger.info(f"Cleared {cleared} cache entries")
            return cleared
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
