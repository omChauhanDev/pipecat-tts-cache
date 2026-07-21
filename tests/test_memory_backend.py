#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""MemoryCacheBackend: the default in-process LRU + TTL store.

Exercised directly (not just via the mixin) because its eviction and expiry logic is
non-trivial and regression-prone.
"""

import asyncio
import time

from pipecat_tts_cache import MemoryCacheBackend
from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse


def _response(byte_len: int = 16) -> CachedTTSResponse:
    return CachedTTSResponse(
        audio_chunks=[CachedAudioChunk(b"\x00\x01" * byte_len, 16000, 1)],
        sample_rate=16000,
        num_channels=1,
    )


async def test_set_then_get_round_trip():
    backend = MemoryCacheBackend()
    response = _response()
    assert await backend.set("k", response) is True
    assert await backend.get("k") is response


async def test_get_miss_returns_none():
    backend = MemoryCacheBackend()
    assert await backend.get("absent") is None


async def test_expired_entries_are_not_returned():
    backend = MemoryCacheBackend()
    await backend.set("k", _response(), ttl=100)
    # Deterministically expire the entry (avoids a real sleep) and confirm the observable
    # behavior: an expired key reads as a miss and is dropped.
    stored_response, _ = backend._cache["k"]
    backend._cache["k"] = (stored_response, time.time() - 1)
    assert await backend.get("k") is None
    assert "k" not in backend._cache


async def test_zero_ttl_never_expires():
    backend = MemoryCacheBackend()
    await backend.set("k", _response(), ttl=0)
    _, expiry = backend._cache["k"]
    assert expiry == 0.0
    assert await backend.get("k") is not None


async def test_lru_eviction_keeps_recently_used_entry():
    backend = MemoryCacheBackend(max_size=2)
    await backend.set("a", _response())
    await backend.set("b", _response())
    await backend.get("a")  # touch 'a' so 'b' becomes least-recently-used
    await backend.set("c", _response())  # capacity reached -> evict LRU ('b')

    assert await backend.exists("a") is True
    assert await backend.exists("b") is False
    assert await backend.exists("c") is True
    assert (await backend.get_stats())["evictions"] == 1


async def test_delete():
    backend = MemoryCacheBackend()
    await backend.set("k", _response())
    assert await backend.delete("k") is True
    assert await backend.get("k") is None
    assert await backend.delete("k") is False  # already gone


async def test_exists_lazily_expires():
    backend = MemoryCacheBackend()
    await backend.set("k", _response(), ttl=100)
    stored_response, _ = backend._cache["k"]
    backend._cache["k"] = (stored_response, time.time() - 1)
    assert await backend.exists("k") is False
    assert "k" not in backend._cache


async def test_clear_removes_everything():
    backend = MemoryCacheBackend()
    await backend.set("a", _response())
    await backend.set("b", _response())
    assert await backend.clear() == 2
    assert (await backend.get_stats())["size"] == 0


async def test_get_stats_reports_size_hits_misses_and_hit_rate():
    backend = MemoryCacheBackend(max_size=10)
    await backend.set("k", _response())
    await backend.get("k")  # hit
    await backend.get("absent")  # miss

    stats = await backend.get_stats()
    assert stats["type"] == "memory"
    assert stats["size"] == 1
    assert stats["max_size"] == 10
    assert stats["backend_hits"] == 1
    assert stats["backend_misses"] == 1
    assert stats["hit_rate"] == 0.5


async def test_close_clears_the_cache():
    backend = MemoryCacheBackend()
    await backend.set("k", _response())
    await backend.close()
    assert (await backend.get_stats())["size"] == 0


async def test_concurrent_access_is_serialized_by_the_lock():
    """The backend guards its OrderedDict with an asyncio.Lock; concurrent ops are safe."""
    backend = MemoryCacheBackend(max_size=100)

    async def writer(i: int):
        await backend.set(f"k{i}", _response())

    await asyncio.gather(*(writer(i) for i in range(50)))
    assert (await backend.get_stats())["size"] == 50


async def test_clear_by_namespace_only_removes_that_prefix():
    """Namespace-scoped clearing matches the ``{namespace}:`` key prefix precisely."""
    backend = MemoryCacheBackend()
    await backend.set("tenant_a:k1", _response())
    await backend.set("tenant_a:k2", _response())
    await backend.set("tenant_b:k1", _response())

    assert await backend.clear("tenant_a") == 2
    assert await backend.exists("tenant_b:k1") is True  # no superstring/other-tenant deletion
    assert (await backend.get_stats())["size"] == 1


async def test_max_size_is_clamped_to_at_least_one():
    backend = MemoryCacheBackend(max_size=0)
    assert backend._max_size == 1
    # Would silently fail (popitem on empty dict) if max_size were left at 0.
    assert await backend.set("k", _response()) is True
    assert await backend.get("k") is not None


async def test_non_positive_ttl_means_no_expiry():
    backend = MemoryCacheBackend()
    await backend.set("k", _response(), ttl=-5)
    _, expiry = backend._cache["k"]
    assert expiry == 0.0  # consistent with the Redis backend (no expiry), not instant-expired
    assert await backend.get("k") is not None
