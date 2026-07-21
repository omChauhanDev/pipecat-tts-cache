#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Cache backend implementations for TTS caching."""

from pipecat_tts_cache.backends.base import CacheBackend
from pipecat_tts_cache.backends.memory import MemoryCacheBackend

try:
    # REDIS_AVAILABLE is sourced from the redis module (which reflects whether redis-py
    # is importable), not set here — the module imports cleanly even without redis-py.
    from pipecat_tts_cache.backends.redis import REDIS_AVAILABLE, RedisCacheBackend
except ImportError:
    RedisCacheBackend = None  # type: ignore
    REDIS_AVAILABLE = False

__all__ = [
    "CacheBackend",
    "MemoryCacheBackend",
    "RedisCacheBackend",
    "REDIS_AVAILABLE",
]
