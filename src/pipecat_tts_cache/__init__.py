#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""TTS caching module for reducing API costs on repeated phrases."""

from pipecat_tts_cache.backends import (
    REDIS_AVAILABLE,
    CacheBackend,
    MemoryCacheBackend,
    RedisCacheBackend,
)
from pipecat_tts_cache.key_generator import generate_cache_key
from pipecat_tts_cache.mixin import TTSCacheMixin
from pipecat_tts_cache.models import (
    CachedAudioChunk,
    CachedTTSResponse,
    CachedWordTimestamp,
)

__all__ = [
    "TTSCacheMixin",
    "CacheBackend",
    "MemoryCacheBackend",
    "RedisCacheBackend",
    "REDIS_AVAILABLE",
    "CachedAudioChunk",
    "CachedWordTimestamp",
    "CachedTTSResponse",
    "generate_cache_key",
]
