#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Cache-key generation: determinism, isolation, normalization, and Pipecat 1.5.0 wiring."""

import pytest
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from pipecat_tts_cache import MemoryCacheBackend, TTSCacheMixin, generate_cache_key


def test_key_is_deterministic_sha256():
    a = generate_cache_key("hello", "voice", "model", 16000, {"speed": 1.0})
    b = generate_cache_key("hello", "voice", "model", 16000, {"speed": 1.0})
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_text_whitespace_is_normalized():
    assert generate_cache_key("  hello   world ", "v", "m", 16000) == generate_cache_key(
        "hello world", "v", "m", 16000
    )


def test_voice_model_sample_rate_namespace_isolate_the_key():
    base = generate_cache_key("hello", "v", "m", 16000)
    assert base != generate_cache_key("hello", "v2", "m", 16000)
    assert base != generate_cache_key("hello", "v", "m2", 16000)
    assert base != generate_cache_key("hello", "v", "m", 24000)
    assert base != generate_cache_key("hello", "v", "m", 16000, namespace="user-123")


def test_secrets_and_non_deterministic_settings_are_excluded():
    plain = generate_cache_key("hello", "v", "m", 16000, {"speed": 1.0})
    with_secrets = generate_cache_key(
        "hello", "v", "m", 16000, {"speed": 1.0, "api_key": "SECRET", "timeout": 5}
    )
    assert plain == with_secrets


def test_float_settings_round_to_six_places():
    assert generate_cache_key("hello", "v", "m", 16000, {"speed": 1.0}) == generate_cache_key(
        "hello", "v", "m", 16000, {"speed": 1.0000000004}
    )


def test_settings_dict_order_does_not_change_the_key():
    a = generate_cache_key("hello", "v", "m", 16000, {"speed": 1.0, "pitch": 2.0})
    b = generate_cache_key("hello", "v", "m", 16000, {"pitch": 2.0, "speed": 1.0})
    assert a == b


def test_nested_settings_are_normalized_deterministically():
    a = generate_cache_key("hello", "v", "m", 16000, {"opts": {"b": 2, "a": 1}})
    b = generate_cache_key("hello", "v", "m", 16000, {"opts": {"a": 1, "b": 2}})
    assert a == b


def test_none_voice_and_model_are_supported():
    # Pipecat 1.5.0 store-mode settings can legitimately carry voice=None / model=None.
    key = generate_cache_key("hello", None, None, 16000)
    assert len(key) == 64


def test_empty_text_raises():
    with pytest.raises(ValueError):
        generate_cache_key("   ", "v", "m", 16000)


def test_schema_version_participates_in_key():
    """A schema bump must change all keys so old cache entries are cleanly ignored."""
    import pipecat_tts_cache.key_generator as kg

    original = kg.CACHE_KEY_SCHEMA
    baseline = generate_cache_key("hello", "v", "m", 16000)
    try:
        kg.CACHE_KEY_SCHEMA = "vNEXT"
        bumped = generate_cache_key("hello", "v", "m", 16000)
    finally:
        kg.CACHE_KEY_SCHEMA = original
    assert baseline != bumped


class _FakeTTS(TTSService):
    async def run_tts(self, text, context_id):  # pragma: no cover - not exercised here
        yield None


class _CachedFakeTTS(TTSCacheMixin, _FakeTTS):
    pass


def _cached_service() -> _CachedFakeTTS:
    service = _CachedFakeTTS(cache_backend=MemoryCacheBackend())
    service._sample_rate = 16000
    return service


def test_mixin_reads_voice_and_model_from_tts_settings():
    """The mixin must derive the key from TTSSettings (voice/model/given_fields), not
    the removed 0.0.x attributes (_voice_id / model_name)."""
    service = _cached_service()

    service._settings = TTSSettings(voice="alice", model="m1", language=None)
    key_alice = service._generate_cache_key("hello")

    service._settings = TTSSettings(voice="bob", model="m1", language=None)
    assert service._generate_cache_key("hello") != key_alice

    service._settings = TTSSettings(voice="alice", model="m2", language=None)
    assert service._generate_cache_key("hello") != key_alice


def test_mixin_key_gen_does_not_crash_on_tts_settings():
    """Regression for the AttributeError('TTSSettings' has no 'items') breakage (§4)."""
    service = _cached_service()
    service._settings = TTSSettings(voice="v", model="m", language=None)
    # Would raise AttributeError under the old dict-based implementation.
    assert len(service._generate_cache_key("hello world")) == 64


def test_non_json_serializable_setting_is_coerced_not_crashed():
    """An object-valued setting (e.g. a pydantic model) is coerced, not crashed on
    (review C1), and its fields still isolate the key."""

    class _GenCfg:  # dataclass-free object with a model_dump (mirrors pydantic)
        def __init__(self, speed):
            self._speed = speed

        def model_dump(self):
            return {"speed": self._speed}

    fast = generate_cache_key("hi", "v", "m", 16000, {"generation_config": _GenCfg(1.2)})
    slow = generate_cache_key("hi", "v", "m", 16000, {"generation_config": _GenCfg(0.8)})
    assert len(fast) == 64
    assert fast != slow  # speed variants isolate the cache
    # Two equal-valued instances must fold to the SAME key (determinism, not an address repr).
    assert fast == generate_cache_key("hi", "v", "m", 16000, {"generation_config": _GenCfg(1.2)})


def test_set_valued_setting_normalizes_to_a_sorted_list():
    """review NG2: set/frozenset settings normalize to a sorted list so the key never depends
    on hash-seed-dependent set iteration order (which differs across processes)."""
    from pipecat_tts_cache.key_generator import normalize_value

    assert normalize_value({"c", "a", "b"}) == ["a", "b", "c"]
    assert normalize_value(frozenset({3, 1, 2})) == [1, 2, 3]


def test_unserializable_setting_raises_instead_of_minting_a_nondeterministic_key():
    """review NG2: an opaque value that survives normalization must raise (so the mixin
    bypasses the cache), not be coerced to a non-deterministic ``str()``."""

    class _Opaque:
        pass

    with pytest.raises(TypeError):
        generate_cache_key("hi", "v", "m", 16000, {"weird": _Opaque()})


def test_provider_isolates_the_key():
    base = generate_cache_key("hi", "v", "m", 16000, provider="pkg.ProviderA")
    assert base != generate_cache_key("hi", "v", "m", 16000, provider="pkg.ProviderB")
    assert base == generate_cache_key("hi", "v", "m", 16000, provider="pkg.ProviderA")


def test_namespace_is_applied_as_a_key_prefix():
    key = generate_cache_key("hi", "v", "m", 16000, namespace="tenant_a")
    assert key.startswith("tenant_a:")
    # The digest after the prefix is a bare sha256 (namespace is not hashed in).
    assert len(key.split(":", 1)[1]) == 64
