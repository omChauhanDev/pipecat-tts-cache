#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""CachedTTSResponse msgpack serialization: round-trip fidelity and rejection of bad input."""

import pytest

pytest.importorskip("msgpack")

from pipecat_tts_cache.models import (  # noqa: E402
    CachedAudioChunk,
    CachedTTSResponse,
    CachedWordTimestamp,
)


def test_round_trip_preserves_every_field():
    original = CachedTTSResponse(
        audio_chunks=[
            CachedAudioChunk(b"\x00\x01\x02", 24000, 2, pts=0),
            CachedAudioChunk(b"\x03\x04", 24000, 2, pts=512),
        ],
        sample_rate=24000,
        num_channels=2,
        word_timestamps=[
            CachedWordTimestamp(word="foo", timestamp=0.0),
            CachedWordTimestamp(word="bar", timestamp=0.3),
        ],
        total_duration_s=0.6,
        created_at=1234.5,
        metadata={"text": "foo bar", "provider": "cartesia"},
    )

    restored = CachedTTSResponse.from_msgpack(original.to_msgpack())

    assert restored == original


def test_round_trip_with_no_word_timestamps():
    original = CachedTTSResponse(
        audio_chunks=[CachedAudioChunk(b"\xaa" * 10, 16000, 1)],
        sample_rate=16000,
        num_channels=1,
    )

    restored = CachedTTSResponse.from_msgpack(original.to_msgpack())

    assert restored.word_timestamps is None
    assert restored.audio_chunks[0].pts is None
    assert restored.total_audio_bytes == original.total_audio_bytes


def test_audio_bytes_stay_bytes_not_str():
    # use_bin_type=True keeps `bytes` distinct from `str` across the round-trip.
    original = CachedTTSResponse(
        audio_chunks=[CachedAudioChunk(b"\x00\xff\x10", 16000, 1)],
        sample_rate=16000,
        num_channels=1,
    )

    restored = CachedTTSResponse.from_msgpack(original.to_msgpack())

    assert isinstance(restored.audio_chunks[0].audio, bytes)
    assert restored.audio_chunks[0].audio == b"\x00\xff\x10"


def test_garbage_bytes_raise_value_error():
    with pytest.raises(Exception):
        CachedTTSResponse.from_msgpack(b"\xff\xff not msgpack \x00")


def test_non_dict_payload_is_rejected():
    import msgpack

    with pytest.raises(ValueError, match="expected dict"):
        CachedTTSResponse.from_msgpack(msgpack.packb([1, 2, 3]))


def test_unknown_schema_version_is_rejected():
    import msgpack

    raw = msgpack.packb({"v": 999, "chunks": []})
    with pytest.raises(ValueError, match="schema version"):
        CachedTTSResponse.from_msgpack(raw)


def test_missing_chunks_key_is_rejected():
    import msgpack

    raw = msgpack.packb({"v": 1, "sample_rate": 16000, "num_channels": 1})
    with pytest.raises(ValueError, match="missing 'chunks'"):
        CachedTTSResponse.from_msgpack(raw)
