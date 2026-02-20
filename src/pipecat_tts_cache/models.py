#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Data models for TTS caching."""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CachedAudioChunk:
    """Single chunk of cached TTS audio."""

    audio: bytes
    sample_rate: int
    num_channels: int
    pts: Optional[int] = None


@dataclass
class CachedWordTimestamp:
    """Word with timing information for replay."""

    word: str
    timestamp: float


@dataclass
class CachedTTSResponse:
    """Complete cached TTS response."""

    audio_chunks: List[CachedAudioChunk]
    sample_rate: int
    num_channels: int
    word_timestamps: Optional[List[CachedWordTimestamp]] = None
    total_duration_s: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_audio_bytes(self) -> int:
        """Calculate total audio size in bytes."""
        return sum(len(chunk.audio) for chunk in self.audio_chunks)


@dataclass
class CacheStats:
    """Standardized metrics for TTS cache backends.

    Attributes:
        hits (int): Number of times a requested TTS phrase was successfully found in the cache.
            Shows how often your cache is actively intercepting requests and preventing API calls.
        misses (int): Number of times a requested TTS phrase was not found (or expired) and had
            to be generated. High misses could indicate a short TTL or unique phrases.
        hit_rate (float): Ratio of cache hits to total requests (hits / (hits + misses)).
         evictions (int): Number of items forcibly removed from the cache to make room for new
            entries. High counts indicate cache thrashing (the cache is too small for the traffic).
        eviction_ratio (float): Ratio of evictions to total cache insertions (evictions / total_sets).
            If this approaches 1.0, old data is constantly being replaced by new data.
        memory_usage_bytes (int): Total memory usage of the active cached audio and metadata
            in bytes. Helps monitor capacity and prevent Out-Of-Memory (OOM) crashes.
        number_of_keys (int): Total count of active, unexpired items currently stored in the cache.
            Gauges cache saturation against your maximum size limits.
        usage_saved_characters (int): Cumulative length of text characters served from cache,
            representing API token savings.
    """

    hits: int
    misses: int
    hit_rate: float
    eviction_count: int
    eviction_ratio: float
    memory_usage_bytes: int
    number_of_keys: int
    usage_saved_characters: int


@dataclass
class RedisCacheStats(CacheStats):
    """Extended metrics specific to Redis backends.

    This class extends the base cache statistics with Redis-specific global
    metrics to help distinguish application-specific cache performance from
    the overall health of the shared Redis server.

    Attributes:
        redis_url (str): Sanitized connection string of the Redis instance (passwords removed).
            Helps identify exactly which Redis instance these stats belong to.
        key_prefix (str): The namespace prefix used to isolate TTS cache keys in the Redis database.
    """

    redis_url: str
    key_prefix: str
