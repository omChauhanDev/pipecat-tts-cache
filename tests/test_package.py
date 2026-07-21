#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Public API surface smoke tests."""

import pipecat_tts_cache


def test_public_exports_are_available():
    for name in [
        "TTSCacheMixin",
        "CacheBackend",
        "MemoryCacheBackend",
        "RedisCacheBackend",
        "REDIS_AVAILABLE",
        "CachedAudioChunk",
        "CachedWordTimestamp",
        "CachedTTSResponse",
        "generate_cache_key",
    ]:
        assert hasattr(pipecat_tts_cache, name), f"missing public export: {name}"
        assert name in pipecat_tts_cache.__all__
