#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Cache key generation for TTS caching."""

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional

# Cache-key schema version. Bump when the key layout changes so that entries written
# by an older version are cleanly ignored (they hash differently) instead of colliding.
CACHE_KEY_SCHEMA = "v2"

# Settings to exclude from cache key (sensitive or non-deterministic)
EXCLUDED_SETTINGS: FrozenSet[str] = frozenset(
    {
        "api_key",
        "api_secret",
        "auth_token",
        "authorization",
        "credentials",
        "timeout",
        "connect_timeout",
        "read_timeout",
        "retry_count",
        "max_retries",
        "log_level",
        "debug",
        "verbose",
        "random_seed",
        "seed",
        "session_id",
        "request_id",
    }
)


def normalize_text(text: str) -> str:
    """Normalize text for consistent cache key generation."""
    return " ".join(text.strip().split())


def normalize_value(value: Any) -> Any:
    """Normalize a setting value for consistent, JSON-serializable hashing.

    Settings can surface structured, non-primitive values (e.g. Pipecat providers
    expose pydantic models such as Cartesia's ``GenerationConfig``). These are
    coerced to their primitive representation so the key both serializes and
    reflects the underlying values (so speed/emotion/etc. isolate the cache).
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, Enum):
        return normalize_value(value.value)
    # pydantic BaseModel (duck-typed to avoid importing pydantic here).
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return normalize_value(model_dump())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return normalize_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        # Sort so the key never depends on set iteration order (which is hash-seed
        # dependent and would otherwise differ across processes).
        return sorted(normalize_value(v) for v in value)
    # Anything else exotic is left as-is; json.dumps raises on it and the caller bypasses
    # the cache rather than mint a non-deterministic key (see generate_cache_key).
    return value


def filter_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out sensitive/non-deterministic settings."""
    return {
        k: normalize_value(v)
        for k, v in settings.items()
        if k not in EXCLUDED_SETTINGS and v is not None
    }


def generate_cache_key(
    text: str,
    voice_id: Optional[str],
    model: Optional[str],
    sample_rate: int,
    settings: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Generate a deterministic SHA-256 cache key from TTS parameters.

    Args:
        text: The text to synthesize (normalized before hashing).
        voice_id: The voice identifier.
        model: The model identifier.
        sample_rate: The output sample rate in Hz.
        settings: Additional service settings (filtered and normalized).
        namespace: Optional isolation namespace. Applied as a **key prefix** so that
            namespace-scoped clearing (prefix match) works and distinct namespaces
            never collide.
        provider: Optional provider discriminator (e.g. the wrapped service's class
            path) so different services with the same voice/model do not collide on
            a shared cache.

    Returns:
        A cache key: ``"{namespace}:{sha256}"`` when a namespace is given, else the
        bare ``sha256`` hex digest.

    Raises:
        ValueError: If ``text`` normalizes to empty.
        TypeError: If a settings value cannot be normalized to a JSON-serializable form.
            Callers (the mixin) treat this as a cache bypass rather than risk a
            non-deterministic key.
    """
    normalized_text = normalize_text(text)
    if not normalized_text:
        raise ValueError("Cannot generate cache key for empty text")

    filtered_settings = filter_settings(settings) if settings else {}

    key_data: Dict[str, Any] = {
        "schema": CACHE_KEY_SCHEMA,
        "text": normalized_text,
        "voice_id": voice_id,
        "model": model,
        "sample_rate": sample_rate,
    }

    if provider:
        key_data["provider"] = provider

    if filtered_settings:
        key_data["settings"] = filtered_settings

    # No ``default=`` fallback: a value that survived normalization unserialized is
    # allowed to raise so the caller bypasses the cache, rather than be coerced to a
    # non-deterministic ``str()`` that would silently never hit across processes.
    key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(key_string.encode("utf-8")).hexdigest()

    # Namespace is a real key prefix (not hashed in) so both per-key isolation and
    # namespace-scoped clearing work.
    return f"{namespace}:{digest}" if namespace else digest
