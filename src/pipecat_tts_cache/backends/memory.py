#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""In-memory LRU cache backend for TTS caching."""

import asyncio
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from pipecat_tts_cache.backends.base import CacheBackend
from pipecat_tts_cache.models import CachedTTSResponse, CacheStats


class MemoryCacheBackend(CacheBackend):
    """In-memory LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000):
        """Initialize in-memory cache backend.

        Args:
            max_size: Maximum number of cache entries to store.
        """
        self._cache: OrderedDict[str, Tuple[CachedTTSResponse, float]] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_sets = 0
        self._usage_saved_chars = 0
        self._size_bytes = 0

        logger.debug(f"Initialized MemoryCacheBackend: max_size={max_size}")

    async def get(self, key: str) -> Optional[CachedTTSResponse]:
        """Retrieve cached response, or None if not found/expired."""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            response, expiry = self._cache[key]

            if expiry > 0 and time.time() > expiry:
                self._size_bytes -= response.total_audio_bytes
                del self._cache[key]
                self._misses += 1
                return None

            self._cache.move_to_end(key)
            self._hits += 1

            text_len = len(response.metadata.get("text", "")) if response.metadata else 0
            self._usage_saved_chars += text_len

            return response

    async def set(self, key: str, response: CachedTTSResponse, ttl: Optional[int] = None) -> bool:
        """Store a TTS response in cache. Returns True on success."""
        async with self._lock:
            try:
                self._total_sets += 1
                expiry = time.time() + ttl if ttl else 0.0

                if len(self._cache) >= self._max_size and key not in self._cache:
                    _, (popped_response, _) = self._cache.popitem(last=False)
                    self._size_bytes -= popped_response.total_audio_bytes
                    self._evictions += 1

                if key in self._cache:
                    old_response, _ = self._cache[key]
                    self._size_bytes -= old_response.total_audio_bytes

                self._cache[key] = (response, expiry)
                self._cache.move_to_end(key)
                self._size_bytes += response.total_audio_bytes

                return True

            except Exception as e:
                logger.error(f"Memory cache set error: {e}")
                return False

    async def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if deleted."""
        async with self._lock:
            if key in self._cache:
                response, _ = self._cache[key]
                self._size_bytes -= response.total_audio_bytes
                del self._cache[key]
                return True
            return False

    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries. Returns number of entries deleted."""
        async with self._lock:
            if namespace is None:
                count = len(self._cache)
                self._cache.clear()
                self._size_bytes = 0
                return count

            to_delete = [k for k in self._cache.keys() if k.startswith(namespace)]
            for key in to_delete:
                response, _ = self._cache[key]
                self._size_bytes -= response.total_audio_bytes
                del self._cache[key]
            return len(to_delete)

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        async with self._lock:
            if key not in self._cache:
                return False

            _, expiry = self._cache[key]
            if expiry > 0 and time.time() > expiry:
                response, _ = self._cache[key]
                self._size_bytes -= response.total_audio_bytes
                del self._cache[key]
                return False
            return True

    async def get_stats(self) -> CacheStats:
        """Get cache statistics for monitoring."""
        async with self._lock:
            total_reads = self._hits + self._misses
            hit_rate = (self._hits / total_reads) if total_reads > 0 else 0.0
            eviction_rate = (self._evictions / self._total_sets) if self._total_sets > 0 else 0.0

            stats = CacheStats(
                hits=self._hits,
                misses=self._misses,
                hit_rate=round(hit_rate, 4),
                usage_saved_characters=self._usage_saved_chars,
                memory_usage_bytes=self._size_bytes,
                number_of_keys=len(self._cache),
                eviction_count=self._evictions,
                eviction_ratio=round(eviction_rate, 4),
            )
            return stats

    async def close(self) -> None:
        """Close backend connections and cleanup resources."""
        async with self._lock:
            self._cache.clear()
            self._size_bytes = 0
