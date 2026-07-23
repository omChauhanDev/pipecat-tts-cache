#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Data models for TTS caching."""

import time
from dataclasses import dataclass, field
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


#: Serialization schema version stamped into every msgpack payload. Bump on any
#: incompatible change to the on-the-wire shape; ``from_msgpack`` rejects unknown
#: versions so a stale entry is treated as a miss instead of mis-decoded.
_MSGPACK_SCHEMA_VERSION = 1


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

    def to_msgpack(self) -> bytes:
        """Serialize to msgpack bytes for storage.

        Every field is written as a plain msgpack scalar/container, so reading
        it back constructs only data. Requires the ``msgpack`` package
        (installed with the ``redis`` extra).
        """
        import msgpack

        packed = msgpack.packb(
            {
                "v": _MSGPACK_SCHEMA_VERSION,
                "chunks": [
                    {
                        "audio": c.audio,
                        "sample_rate": c.sample_rate,
                        "num_channels": c.num_channels,
                        "pts": c.pts,
                    }
                    for c in self.audio_chunks
                ],
                "sample_rate": self.sample_rate,
                "num_channels": self.num_channels,
                "word_timestamps": (
                    None
                    if self.word_timestamps is None
                    else [
                        {"word": wt.word, "timestamp": wt.timestamp} for wt in self.word_timestamps
                    ]
                ),
                "total_duration_s": self.total_duration_s,
                "created_at": self.created_at,
                "metadata": self.metadata,
            },
            use_bin_type=True,
        )
        # msgpack.packb is typed as Optional but only returns None with a custom
        # default that returns None; ours never does.
        assert packed is not None
        return packed

    @classmethod
    def from_msgpack(cls, raw: bytes) -> "CachedTTSResponse":
        """Deserialize from msgpack bytes.

        Every field is type-checked before the response is constructed, so a
        decodable-but-malformed payload (e.g. a string where an int is expected)
        raises ``ValueError`` rather than producing a corrupt response. Callers
        treat any ``ValueError`` as a cache miss. Requires ``msgpack``.
        """
        import msgpack

        payload = msgpack.unpackb(raw, raw=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid cached TTS payload: expected dict, got {type(payload)}")
        if payload.get("v") != _MSGPACK_SCHEMA_VERSION:
            raise ValueError(f"Unsupported cached TTS schema version: {payload.get('v')!r}")

        chunks_raw = _require(payload, "chunks", list)
        raw_timestamps = payload.get("word_timestamps")
        if raw_timestamps is not None and not isinstance(raw_timestamps, list):
            raise ValueError(
                f"'word_timestamps' must be a list or null, got {type(raw_timestamps)}"
            )
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"'metadata' must be a dict, got {type(metadata)}")

        return cls(
            audio_chunks=[_chunk_from(c) for c in chunks_raw],
            sample_rate=_require_int(payload, "sample_rate"),
            num_channels=_require_int(payload, "num_channels"),
            word_timestamps=(
                None if raw_timestamps is None else [_timestamp_from(wt) for wt in raw_timestamps]
            ),
            total_duration_s=_opt_float(payload, "total_duration_s", 0.0),
            created_at=_opt_float(payload, "created_at", 0.0),
            metadata=metadata,
        )


def _require(payload: Dict[str, Any], key: str, expected: type) -> Any:
    """Return ``payload[key]``, raising ``ValueError`` if absent or the wrong type."""
    if key not in payload:
        raise ValueError(f"Invalid cached TTS payload: missing {key!r}")
    value = payload[key]
    if not isinstance(value, expected):
        raise ValueError(f"{key!r} must be {expected.__name__}, got {type(value)}")
    return value


def _require_int(payload: Dict[str, Any], key: str) -> int:
    """Return an ``int`` field, rejecting ``bool`` (an ``int`` subclass) and other types."""
    value = _require(payload, key, int)
    if isinstance(value, bool):
        raise ValueError(f"{key!r} must be int, got bool")
    return value


def _opt_float(payload: Dict[str, Any], key: str, default: float) -> float:
    """Return an optional numeric field as ``float``, or ``default`` if absent."""
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key!r} must be a number, got {type(value)}")
    return float(value)


def _chunk_from(c: Any) -> CachedAudioChunk:
    """Build a ``CachedAudioChunk`` from a decoded dict, validating each field."""
    if not isinstance(c, dict):
        raise ValueError(f"Audio chunk must be a dict, got {type(c)}")
    audio = _require(c, "audio", bytes)
    pts = c.get("pts")
    if pts is not None and (isinstance(pts, bool) or not isinstance(pts, int)):
        raise ValueError(f"'pts' must be int or null, got {type(pts)}")
    return CachedAudioChunk(
        audio=audio,
        sample_rate=_require_int(c, "sample_rate"),
        num_channels=_require_int(c, "num_channels"),
        pts=pts,
    )


def _timestamp_from(wt: Any) -> CachedWordTimestamp:
    """Build a ``CachedWordTimestamp`` from a decoded dict, validating each field."""
    if not isinstance(wt, dict):
        raise ValueError(f"Word timestamp must be a dict, got {type(wt)}")
    word = _require(wt, "word", str)
    ts = wt.get("timestamp")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        raise ValueError(f"'timestamp' must be a number, got {type(ts)}")
    return CachedWordTimestamp(word=word, timestamp=float(ts))
