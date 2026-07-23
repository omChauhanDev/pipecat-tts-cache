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

        Raises ``ValueError`` on a malformed payload or an unrecognized schema
        version; callers treat that as a cache miss. Requires ``msgpack``.
        """
        import msgpack

        payload = msgpack.unpackb(raw, raw=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid cached TTS payload: expected dict, got {type(payload)}")
        if payload.get("v") != _MSGPACK_SCHEMA_VERSION:
            raise ValueError(f"Unsupported cached TTS schema version: {payload.get('v')!r}")
        if "chunks" not in payload:
            raise ValueError("Invalid cached TTS payload: missing 'chunks'")

        raw_timestamps = payload.get("word_timestamps")
        return cls(
            audio_chunks=[
                CachedAudioChunk(
                    audio=c["audio"],
                    sample_rate=c["sample_rate"],
                    num_channels=c["num_channels"],
                    pts=c.get("pts"),
                )
                for c in payload["chunks"]
            ],
            sample_rate=payload["sample_rate"],
            num_channels=payload["num_channels"],
            word_timestamps=(
                None
                if raw_timestamps is None
                else [
                    CachedWordTimestamp(word=wt["word"], timestamp=wt["timestamp"])
                    for wt in raw_timestamps
                ]
            ),
            total_duration_s=payload.get("total_duration_s", 0.0),
            created_at=payload.get("created_at", 0.0),
            metadata=payload.get("metadata", {}),
        )
