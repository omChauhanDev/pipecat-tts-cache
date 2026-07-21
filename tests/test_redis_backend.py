#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""RedisCacheBackend: the distributed store.

Tested against an in-memory ``fakeredis`` server so the backend's real code path
(connection, pickle serialization, TTL, scan-based clear, error fallbacks) runs
unmodified — no mocking of our own logic. Skipped if ``fakeredis`` is unavailable.
"""

import pickle

import pytest

fakeredis = pytest.importorskip("fakeredis")
import fakeredis.aioredis  # noqa: E402

import pipecat_tts_cache.backends.redis as redis_backend  # noqa: E402
from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse  # noqa: E402


def _response(byte_len: int = 16) -> CachedTTSResponse:
    return CachedTTSResponse(
        audio_chunks=[CachedAudioChunk(b"\x00\x01" * byte_len, 16000, 1)],
        sample_rate=16000,
        num_channels=1,
    )


@pytest.fixture
def backend(monkeypatch):
    """A RedisCacheBackend whose client talks to a shared in-memory fake server."""
    server = fakeredis.FakeServer()

    def _fake_from_url(url, **kwargs):
        return fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)

    monkeypatch.setattr(redis_backend.aioredis, "from_url", _fake_from_url)
    return redis_backend.RedisCacheBackend(redis_url="redis://fake", key_prefix="test:")


async def test_set_get_round_trip_serializes_the_response(backend):
    response = _response()
    assert await backend.set("k1", response, ttl=100) is True

    fetched = await backend.get("k1")
    assert isinstance(fetched, CachedTTSResponse)
    assert fetched.total_audio_bytes == response.total_audio_bytes
    assert fetched.word_timestamps == response.word_timestamps


async def test_get_missing_returns_none(backend):
    assert await backend.get("missing") is None


async def test_exists(backend):
    await backend.set("k", _response())
    assert await backend.exists("k") is True
    assert await backend.exists("absent") is False


async def test_delete(backend):
    await backend.set("k", _response())
    assert await backend.delete("k") is True
    assert await backend.get("k") is None


async def test_clear_removes_all_prefixed_keys(backend):
    await backend.set("a", _response())
    await backend.set("b", _response())
    assert await backend.clear() == 2
    assert await backend.get("a") is None
    assert await backend.get("b") is None


async def test_corrupt_payload_is_treated_as_a_miss(backend):
    """A value that is not a CachedTTSResponse must not be returned to the caller."""
    client = await backend._get_client()
    await client.set("test:bad", pickle.dumps({"not": "a response"}))
    assert await backend.get("bad") is None


async def test_get_stats_is_fail_safe(backend):
    # fakeredis does not implement INFO; the backend must still return a dict, not raise.
    await backend.set("k", _response())
    stats = await backend.get_stats()
    assert stats["type"] == "redis"


async def test_close_is_idempotent_and_safe(backend):
    await backend.set("k", _response())
    await backend.close()
    await backend.close()  # second close must not raise


async def test_clear_by_namespace_only_removes_that_prefix(backend):
    await backend.set("tenant_a:k1", _response())
    await backend.set("tenant_a:k2", _response())
    await backend.set("tenant_b:k1", _response())

    assert await backend.clear("tenant_a") == 2
    assert await backend.exists("tenant_b:k1") is True


async def test_float_ttl_is_coerced_and_does_not_fail_the_write(backend):
    # A float TTL (e.g. from timedelta.total_seconds()) must not silently fail on redis.
    assert await backend.set("k", _response(), ttl=2.5) is True
    assert await backend.get("k") is not None


async def test_sub_second_ttl_is_clamped_and_still_caches(backend):
    # review NG3: a 0<ttl<1 must clamp to >=1s, not send ex=0 (which Redis rejects and which
    # would silently disable caching while the memory backend caches the same value fine).
    assert await backend.set("k", _response(), ttl=0.5) is True
    assert await backend.get("k") is not None
