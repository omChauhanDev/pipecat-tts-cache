# Changelog

All notable changes to this **Pipecat TTS Cache** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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