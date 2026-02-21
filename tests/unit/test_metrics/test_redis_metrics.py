#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Unit tests for RedisCacheBackend metric collection via get_stats()."""

import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from testcontainers.redis import RedisContainer

from pipecat_tts_cache.backends.redis import RedisCacheBackend
from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse


@pytest.fixture(scope="session")
def redis_url():
    """Spin up a real Redis container for the test session."""
    with RedisContainer() as redis:
        host = redis.get_container_host_ip()
        port = redis.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


@pytest_asyncio.fixture
async def backend(redis_url):
    """Fresh backend with a fully flushed DB between tests."""
    client = aioredis.from_url(redis_url)
    await client.flushdb()
    await client.aclose()

    b = RedisCacheBackend(redis_url=redis_url, key_prefix="test:")
    yield b
    await b.close()


@pytest.fixture
def response():
    return CachedTTSResponse(
        audio_chunks=[CachedAudioChunk(audio=b"x" * 100, sample_rate=16000, num_channels=1)],
        sample_rate=16000,
        num_channels=1,
        metadata={"text": "hello world"},
    )


@pytest.mark.asyncio
async def test_hit_and_miss_counts(backend, response):
    """get() increments hits on cache hit and misses on cache miss."""
    await backend.set("key1", response)
    await backend.get("key1")  # hit
    await backend.get("missing")  # miss

    stats = await backend.get_stats()
    assert stats.hits == 1
    assert stats.misses == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hits,misses,expected_rate",
    [
        (9, 1, 0.9),
        (0, 5, 0.0),
        (1, 1, 0.5),
    ],
)
async def test_hit_rate(backend, response, hits, misses, expected_rate):
    """hit_rate is computed correctly from accumulated hits and misses."""
    await backend.set("key1", response)
    for _ in range(hits):
        await backend.get("key1")
    for i in range(misses):
        await backend.get(f"missing{i}")

    stats = await backend.get_stats()
    assert stats.hit_rate == expected_rate


@pytest.mark.asyncio
async def test_saved_characters_on_hit(backend, response):
    """usage_saved_characters accumulates the text length on each cache hit."""
    await backend.set("key1", response)
    await backend.get("key1")
    await backend.get("key1")

    stats = await backend.get_stats()
    assert stats.usage_saved_characters == len("hello world") * 2


@pytest.mark.asyncio
async def test_saved_characters_not_incremented_on_miss(backend):
    """usage_saved_characters stays 0 when only misses occur."""
    await backend.get("missing")

    stats = await backend.get_stats()
    assert stats.usage_saved_characters == 0


@pytest.mark.asyncio
async def test_number_of_keys_and_memory(backend, response):
    """number_of_keys and memory_usage_bytes reflect stored entries."""
    await backend.set("key1", response)
    await backend.set("key2", response)

    stats = await backend.get_stats()
    assert stats.number_of_keys == 2
    assert stats.memory_usage_bytes > 0
