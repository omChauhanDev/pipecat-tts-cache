import asyncio

import pytest

from pipecat_tts_cache.backends import MemoryCacheBackend
from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse


@pytest.fixture
def backend():
    """Provide an isolated memory backend instance."""
    return MemoryCacheBackend(max_size=10)


@pytest.fixture
def create_response():
    """Factory fixture to create test TTS responses with configurable size."""

    def _create(text="hello world", audio_size=16):
        return CachedTTSResponse(
            audio_chunks=[
                CachedAudioChunk(audio=b"x" * audio_size, sample_rate=16000, num_channels=1)
            ],
            sample_rate=16000,
            num_channels=1,
            metadata={"text": text},
        )

    return _create


@pytest.mark.asyncio
async def test_memory_stats_initial_state(backend):
    """Verify memory metrics return default zero values upon initialization."""
    stats = await backend.get_stats()

    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.hit_rate == 0.0
    assert stats.number_of_keys == 0
    assert stats.memory_usage_bytes == 0
    assert stats.usage_saved_characters == 0
    assert stats.eviction_count == 0
    assert stats.eviction_ratio == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "num_hits,expected_hits",
    [
        (1, 1),
        (3, 3),
        (5, 5),
    ],
)
async def test_hits_repeated_access(backend, create_response, num_hits, expected_hits):
    """Verify hits metric accumulates for repeated access to same key."""
    response = create_response()
    await backend.set("key1", response)

    for _ in range(num_hits):
        await backend.get("key1")

    stats = await backend.get_stats()
    assert stats.hits == expected_hits


@pytest.mark.asyncio
async def test_hits_different_keys(backend, create_response):
    """Verify hits metric tracks access across multiple different keys."""
    response = create_response()
    await backend.set("key1", response)
    await backend.set("key2", response)
    await backend.set("key3", response)

    await backend.get("key1")
    await backend.get("key2")
    await backend.get("key3")
    await backend.get("key1")  # Hit same key again

    stats = await backend.get_stats()
    assert stats.hits == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "num_misses,expected_misses",
    [
        (1, 1),
        (3, 3),
        (5, 5),
    ],
)
async def test_misses_nonexistent_keys(backend, num_misses, expected_misses):
    """Verify misses metric accumulates for nonexistent keys."""
    for i in range(num_misses):
        await backend.get(f"key{i}")

    stats = await backend.get_stats()
    assert stats.misses == expected_misses


@pytest.mark.asyncio
async def test_misses_expired_key(backend, create_response):
    """Verify misses metric increments when accessing expired key."""
    response = create_response()
    await backend.set("key1", response, ttl=1)  # 1 second TTL

    await asyncio.sleep(1.1)  # Wait for expiration
    result = await backend.get("key1")

    stats = await backend.get_stats()
    assert result is None
    assert stats.misses == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hits,misses,expected_rate",
    [
        (3, 0, 1.0),
        (0, 3, 0.0),
        (2, 2, 0.5),
        (2, 5, 0.2857),  # Precision test: 2/7 = 0.285714...
        (7, 3, 0.7),
    ],
)
async def test_hit_rate_calculation(backend, create_response, hits, misses, expected_rate):
    """Verify hit_rate calculation for various hit/miss combinations."""
    response = create_response()
    await backend.set("key1", response)

    for _ in range(hits):
        await backend.get("key1")

    for i in range(misses):
        await backend.get(f"nonexistent{i}")

    stats = await backend.get_stats()
    assert stats.hit_rate == expected_rate


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "num_entries,expected_keys",
    [
        (1, 1),
        (3, 3),
        (5, 5),
    ],
)
async def test_number_of_keys_multiple_entries(
    backend, create_response, num_entries, expected_keys
):
    """Verify number_of_keys tracks multiple cache entries."""
    response = create_response()
    for i in range(num_entries):
        await backend.set(f"key{i}", response)

    stats = await backend.get_stats()
    assert stats.number_of_keys == expected_keys


@pytest.mark.asyncio
async def test_number_of_keys_after_operations(backend, create_response):
    """Verify number_of_keys changes correctly with delete and overwrite operations."""
    response = create_response()

    await backend.set("key1", response)
    await backend.set("key2", response)
    stats = await backend.get_stats()
    assert stats.number_of_keys == 2

    await backend.delete("key1")
    stats = await backend.get_stats()
    assert stats.number_of_keys == 1

    await backend.set("key2", create_response("new"))
    stats = await backend.get_stats()
    assert stats.number_of_keys == 1


@pytest.mark.asyncio
async def test_number_of_keys_after_clear(backend, create_response):
    """Verify number_of_keys resets to 0 after clearing cache."""
    response = create_response()
    await backend.set("key1", response)
    await backend.set("key2", response)
    await backend.set("key3", response)

    await backend.clear()

    stats = await backend.get_stats()
    assert stats.number_of_keys == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audio_sizes,expected_bytes",
    [
        ([100], 100),
        ([100, 200], 300),
        ([100, 200, 150], 450),
    ],
)
async def test_memory_usage_bytes_accumulation(
    backend, create_response, audio_sizes, expected_bytes
):
    """Verify memory_usage_bytes accumulates correctly across entries."""
    for i, size in enumerate(audio_sizes):
        response = create_response(audio_size=size)
        await backend.set(f"key{i}", response)

    stats = await backend.get_stats()
    assert stats.memory_usage_bytes == expected_bytes


@pytest.mark.asyncio
async def test_memory_usage_bytes_operations(backend, create_response):
    """Verify memory_usage_bytes updates correctly for delete, clear, and overwrite."""
    response1 = create_response(audio_size=100)
    response2 = create_response(audio_size=200)

    await backend.set("key1", response1)
    await backend.set("key2", response2)
    stats = await backend.get_stats()
    assert stats.memory_usage_bytes == 300

    await backend.delete("key1")
    stats = await backend.get_stats()
    assert stats.memory_usage_bytes == 200

    response3 = create_response(audio_size=350)
    await backend.set("key2", response3)
    stats = await backend.get_stats()
    assert stats.memory_usage_bytes == 350

    await backend.clear()
    stats = await backend.get_stats()
    assert stats.memory_usage_bytes == 0


@pytest.mark.asyncio
async def test_memory_usage_bytes_after_eviction(create_response):
    """Verify memory_usage_bytes accounts for evicted entries."""
    small_backend = MemoryCacheBackend(max_size=2)
    response = create_response(audio_size=100)

    await small_backend.set("key1", response)
    await small_backend.set("key2", response)
    await small_backend.set("key3", response)

    stats = await small_backend.get_stats()
    assert stats.memory_usage_bytes == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,num_hits,expected_chars",
    [
        ("hello world", 1, 11),
        ("hello", 3, 15),
        ("test case", 2, 18),
    ],
)
async def test_usage_saved_characters_tracking(
    backend, create_response, text, num_hits, expected_chars
):
    """Verify usage_saved_characters tracks character savings correctly."""
    response = create_response(text)
    await backend.set("key1", response)

    for _ in range(num_hits):
        await backend.get("key1")

    stats = await backend.get_stats()
    assert stats.usage_saved_characters == expected_chars


@pytest.mark.asyncio
async def test_usage_saved_characters_different_keys(backend, create_response):
    """Verify usage_saved_characters tracks savings across different keys."""
    test_data = [
        ("hello", "key1", 1),
        ("world", "key2", 2),
        ("test case", "key3", 1),
    ]

    for text, key, hits in test_data:
        response = create_response(text)
        await backend.set(key, response)
        for _ in range(hits):
            await backend.get(key)

    stats = await backend.get_stats()
    assert stats.usage_saved_characters == 5 + 10 + 9


@pytest.mark.asyncio
async def test_usage_saved_characters_no_metadata(backend):
    """Verify usage_saved_characters handles responses without text metadata."""
    response = CachedTTSResponse(
        audio_chunks=[CachedAudioChunk(audio=b"test", sample_rate=16000, num_channels=1)],
        sample_rate=16000,
        num_channels=1,
        metadata={},  # No text field
    )
    await backend.set("key1", response)
    await backend.get("key1")

    stats = await backend.get_stats()
    assert stats.usage_saved_characters == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_size,total_sets,expected_evictions",
    [
        (2, 2, 0),
        (2, 3, 1),
        (5, 5, 0),
        (3, 7, 4),
    ],
)
async def test_eviction_count(create_response, max_size, total_sets, expected_evictions):
    """Verify eviction_count tracks LRU evictions correctly."""
    test_backend = MemoryCacheBackend(max_size=max_size)
    response = create_response()

    for i in range(total_sets):
        await test_backend.set(f"key{i}", response)

    stats = await test_backend.get_stats()
    assert stats.eviction_count == expected_evictions


@pytest.mark.asyncio
async def test_eviction_count_overwrite_no_eviction(create_response):
    """Verify eviction_count doesn't increment when overwriting existing key."""
    test_backend = MemoryCacheBackend(max_size=2)
    response1 = create_response("first")
    response2 = create_response("second")

    await test_backend.set("key1", response1)
    await test_backend.set("key1", response2)

    stats = await test_backend.get_stats()
    assert stats.eviction_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_size,total_sets,expected_evictions,expected_ratio",
    [
        (2, 2, 0, 0.0),
        (2, 3, 1, round(1 / 3, 4)),
        (2, 6, 4, round(4 / 6, 4)),
        (5, 8, 3, round(3 / 8, 4)),
    ],
)
async def test_eviction_ratio(
    create_response, max_size, total_sets, expected_evictions, expected_ratio
):
    """Verify eviction_ratio calculation for various scenarios."""
    test_backend = MemoryCacheBackend(max_size=max_size)
    response = create_response()

    for i in range(total_sets):
        await test_backend.set(f"key{i}", response)

    stats = await test_backend.get_stats()
    assert stats.eviction_count == expected_evictions
    assert stats.eviction_ratio == expected_ratio


@pytest.mark.asyncio
async def test_eviction_ratio_with_overwrites(create_response):
    """Verify eviction_ratio correctly handles overwrites that don't cause evictions."""
    test_backend = MemoryCacheBackend(max_size=2)
    response = create_response()

    await test_backend.set("key1", response)
    await test_backend.set("key2", response)
    await test_backend.set("key1", response)
    await test_backend.set("key2", response)
    await test_backend.set("key3", response)
    await test_backend.set("key4", response)
    await test_backend.set("key5", response)

    stats = await test_backend.get_stats()
    assert stats.eviction_count == 3
    assert stats.eviction_ratio == round(3 / 7, 4)


@pytest.mark.asyncio
async def test_metrics_after_ttl_expiration(backend, create_response):
    """Verify metrics handle TTL expiration correctly."""
    response = create_response("test text", audio_size=100)
    await backend.set("key1", response, ttl=1)

    await backend.get("key1")
    await asyncio.sleep(1.1)
    await backend.get("key1")

    stats = await backend.get_stats()

    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.number_of_keys == 0
    assert stats.memory_usage_bytes == 0


@pytest.mark.asyncio
async def test_metrics_after_namespace_clear(backend, create_response):
    """Verify metrics update correctly after namespace-specific clear."""
    response = create_response(audio_size=100)

    await backend.set("ns1:key1", response)
    await backend.set("ns1:key2", response)
    await backend.set("ns2:key1", response)

    await backend.clear(namespace="ns1:")

    stats = await backend.get_stats()

    assert stats.number_of_keys == 1  # Only ns2:key1 remains
    assert stats.memory_usage_bytes == 100
