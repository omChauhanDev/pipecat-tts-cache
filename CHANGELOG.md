# Changelog

All notable changes to this **Pipecat TTS Cache** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-20

Compatibility release for the current Pipecat line. Pipecat `1.x` reorganized the TTS
service contract (notably `run_tts()` and the word-timestamp / audio-context system),
which broke the mixin against any recent Pipecat. This release adapts to that contract.

### Changed

- **BREAKING:** now requires `pipecat-ai>=1.5.0` and **Python >=3.11** (previously
  `pipecat-ai>=0.0.91` / Python 3.10). Pin `pipecat-tts-cache==0.0.3` if you must stay on
  Pipecat `0.0.91`–`0.0.101`.
- Cache keys are now derived from the service's `TTSSettings` store (`voice`, `model`, and
  remaining runtime settings via `given_fields()`) instead of the removed `_voice_id` /
  `model_name` attributes. The key also folds in a provider discriminator, so two different
  services that share a voice/model never collide on a shared backend. The key format
  changed (schema `v2`), so entries written by older versions are ignored and re-synthesized
  once — no manual cache migration needed.
- `cache_namespace` is now applied as a key **prefix** rather than hashed into the digest,
  so distinct namespaces isolate and namespace-scoped clearing works (see Fixed).
- Word-timestamp capability is now detected from actual runtime emission rather than the
  service's class, so services that don't emit word timestamps (Google, OpenAI, Deepgram,
  Sarvam, …) are cached again (audio-only) instead of being silently skipped.
- The mixin never closes a cache backend you pass in — you own its lifecycle, so one backend
  can be shared across services and processes safely.
- Modernized `examples/basic_caching.py` for Pipecat 1.x: uses `PipelineWorker` /
  `WorkerRunner` (replacing the deprecated `PipelineTask` / `PipelineRunner`) and configures
  VAD on the user aggregator (`LLMUserAggregatorParams`) instead of the removed
  `TransportParams(vad_analyzer=...)`.

### Fixed

- `TTSCacheMixin.run_tts()` now accepts the `context_id` argument Pipecat passes
  (`>=0.0.102`), fixing `run_tts() takes 2 positional arguments but 3 were given`
  ([#1], [#5]).
- Cache-key generation no longer raises `AttributeError: 'TTSSettings' object has no
  attribute 'items'`, and voice/model are no longer collapsed to `"default"`.
- Object-valued settings (e.g. Cartesia's `generation_config` pydantic model) are normalized
  into the cache key instead of breaking key generation, so speed / emotion variants isolate
  correctly.
- `clear_cache(namespace=...)` now clears only the requested namespace; previously the
  namespace was hashed into the key and scoped clearing was a silent no-op.
- Float and sub-second `cache_ttl` values (e.g. from `timedelta.total_seconds()`) are coerced
  to a valid Redis expiry instead of silently disabling caching.
- Multi-sentence turns are split at word boundaries only when the provider's timestamps are
  monotonic, preventing mis-attributed (corrupted) cached audio.
- `add_word_timestamps()` now matches Pipecat's widened signature
  (`context_id` / `includes_inter_frame_spaces` / `pre_merge_tokens`).
- Cache hits now replay audio through the service's audio context, so a cached sentence can
  no longer interleave with earlier live audio in the same turn ([#2]).
- Word timestamps are captured and replayed on cache hits, keeping assistant transcripts
  correct — including on interruption ([#6]).

### Added

- Per-`context_id` capture/replay aligned with Pipecat's audio-context model.
- A real test suite (cache-key behavior, end-to-end cache miss/hit/interruption via
  Pipecat's `run_test`, and a Pipecat-contract guard that fails CI if these integration
  points change again), run on Python 3.11, 3.12 and 3.13 ([#3]).

### Security

- The Redis backend serializes entries with `pickle`, so a cache entry is trusted code when
  it is read back. Point `RedisCacheBackend` only at a Redis instance you control
  (single-tenant, authenticated, network-isolated) — see [SECURITY.md](SECURITY.md) and the
  README. Cache entries that are absent or of an unexpected type are now deleted on read
  (self-heal) rather than retried on every lookup.

[#1]: https://github.com/omChauhanDev/pipecat-tts-cache/issues/1
[#2]: https://github.com/omChauhanDev/pipecat-tts-cache/issues/2
[#3]: https://github.com/omChauhanDev/pipecat-tts-cache/issues/3
[#5]: https://github.com/omChauhanDev/pipecat-tts-cache/issues/5
[#6]: https://github.com/omChauhanDev/pipecat-tts-cache/issues/6

---

## [0.0.3] - 2026-01-18

### Changed

- Updated minimum `pipecat-ai` version requirement from `>=0.0.90` to `>=0.0.91` for improved compatibility.

### Fixed

- Added check to only await for coroutine objects, improving async handling.

---

## [0.0.2] - 2026-01-17

### Added

- Initial release of `pipecat-tts-cache`. A TTS caching layer implemented as a non-invasive Mixin.

- Added `TTSCacheMixin` to intercept `run_tts()` checks and `push_frame()` audio collection.  
  It supports all four `Pipecat TTS` service patterns:

  - **AudioContextWordTTSService**: Full batch caching with word-boundary splitting (e.g., Cartesia).
  - **WordTTSService**: Full caching with timestamp preservation (e.g., ElevenLabs).
  - **InterruptibleTTSService**: Single-sentence caching (e.g., Deepgram WS).
  - **TTSService**: Standard audio caching (e.g., Google HTTP).

- Added two cache backend implementations:

  - `MemoryCacheBackend`: Async in-memory LRU cache.  
  - `RedisCacheBackend`: Distributed cache using redis-py (asyncio).

- Added deterministic cache key generation:  
  Keys are derived from normalized text, voice ID, model, sample rate, and voice settings (speed, pitch)  
  while excluding sensitive credentials.

- Added word timestamp preservation:  
  Timestamps are stored alongside audio and replayed with relative offsets to ensure frontend  
  lip-sync and transcription alignment.

- Added interruption safety:  
  `InterruptionFrame` events immediately clear pending cache buffers to prevent corruption.