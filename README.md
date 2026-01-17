
<h1><div align="center">
  <img alt="Pipecat TTS Cache" width="300px" height="auto" src="https://raw.githubusercontent.com/omChauhanDev/pipecat-tts-cache/main/assets/pipecat-tts-cache.png">
</div></h1>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/pipecat-tts-cache)](https://pypi.org/project/pipecat-tts-cache)
![Tests](https://github.com/omChauhanDev/pipecat-tts-cache/actions/workflows/ci.yaml/badge.svg)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Redis](https://img.shields.io/badge/Backend-Redis-red)](https://redis.io)

</div>

# Pipecat TTS Cache: Zero-Latency Audio Synthesis

**Pipecat TTS Cache** is a lightweight caching layer for the Pipecat ecosystem. It transparently wraps existing TTS services to eliminate API costs for repeated phrases and reduce response latency to **<5ms**.

> **See it in action:** [Watch the Demo Video](https://drive.google.com/file/d/1jZRZVPNVrcrbslyKDRhww2qEXkj29b9F/view?usp=sharing)

## 🚀 Key Features

- **Ultra-Low Latency** – Delivers cached audio in ~0.1ms (Memory) or ~1-5ms (Redis).
- **Cost Reduction** – Stop paying your TTS provider for common phrases like "Hello," "One moment," or "I didn't catch that."
- **Universal Compatibility** – Works as a Mixin with **all** Pipecat TTS services (Cartesia, ElevenLabs, Deepgram, Google, etc.).
- **Smart Interruption** – Automatically clears pending cache tasks and resets state when users interrupt the bot.
- **Precision Alignment** – Preserves word-level timestamps for perfect lip-syncing and subtitles, even on cached replays.

## 📦 Installation

```bash
# Standard installation (Memory backend only)
pip install pipecat-tts-cache

# Production installation (with Redis support)
pip install "pipecat-tts-cache[redis]"

```

## 🧩 Service Compatibility

The caching layer intelligently handles different TTS architectures to ensure smooth playback regardless of the provider.

| **Service Type**            | **Caching Strategy**                                   | **Supported Providers (Examples)**        |
|----------------------------|--------------------------------------------------------|-------------------------------------------|
| **AudioContextWordTTS**    | **Batch Caching**  <br> Splits audio at word boundaries and caches individual sentences. | Cartesia, Rime |
| **WordTTSService**         | **Full Caching w/ Timestamps**  <br> Caches the full response and preserves alignment data. | ElevenLabs, Hume |
| **TTSService**             | **Standard Caching**  <br> Caches the full audio response (no alignment data). | Google, OpenAI, Deepgram (HTTP) |
| **InterruptibleTTS**       | **Sentence Caching**  <br> Caches single-sentence responses only. | Sarvam, Deepgram (WebSocket) |
## 🛠️ Usage

### 1. Basic In-Memory Cache (Development)

The `MemoryCacheBackend` is perfect for local development or single-process bots. It uses an LRU (Least Recently Used) eviction policy.

```python
from pipecat_tts_cache import TTSCacheMixin, MemoryCacheBackend
from pipecat.services.google.tts import GoogleHttpTTSService

# 1. Create a cached class using the Mixin
class CachedGoogleTTS(TTSCacheMixin, GoogleHttpTTSService):
    pass

# 2. Initialize with memory backend
tts = CachedGoogleTTS(
    voice_id="en-US-Chirp3-HD-Charon",
    cache_backend=MemoryCacheBackend(max_size=1000),
    cache_ttl=86400,  # Cache for 24 hours
)

```

### 2. Distributed Redis Cache (Production)

For production deployments, use `RedisCacheBackend`. This allows the cache to persist across restarts and be shared among multiple bot instances.

```python
from pipecat_tts_cache.backends import RedisCacheBackend

tts = CachedGoogleTTS(
    voice_id="en-US-Chirp3-HD-Charon",
    cache_backend=RedisCacheBackend(
        redis_url="redis://localhost:6379/0",
        key_prefix="pipecat:tts:",
    ),
    cache_ttl=604800, # Cache for 1 week
)

```

## 🧠 How It Works

The system utilizes a **Frame Interception Architecture** to seamlessly integrate with the Pipecat pipeline:

1. **Deterministic Key Gen**: Before requesting audio, a unique key is generated based on the normalized text, voice ID, model, speed, and pitch. Sensitive data (API keys) is excluded.
2. **Cache Check (`run_tts`)**:
* **Hit:** The system immediately pushes cached audio frames and timestamps to the pipeline.
* **Miss:** The system calls the parent TTS service.


3. **Collection (`push_frame`)**: As the parent service generates audio, the Mixin intercepts the frames, aggregates them, and stores them in the backend for future use.

### Interruption Handling

When an `InterruptionFrame` is received, the cache mixin immediately:

* Clears all pending cache write tasks.
* Resets the internal batch state.
* Ensures no partial or cut-off audio is committed to the pipeline.

## 📊 Management & Stats

You can monitor cache performance or clear entries programmatically.

```python
# Check performance
stats = await tts.get_cache_stats()
print(f"Hit Rate: {stats['hit_rate']:.1%}")
print(f"Total Saved Calls: {stats['hits']}")

# Maintenance
await tts.clear_cache() # Clear all
await tts.clear_cache(namespace="user_123") # Clear specific namespace

```

## ⚡ Performance

| Metric | Direct API | Memory Cache | Redis Cache |
| --- | --- | --- | --- |
| **Latency** | 200ms - 1500ms | **~0.1ms** | **~2ms** |
| **Cost** | $ per character | **$0** | **$0** |
| **Consistency** | Variable | Deterministic | Deterministic |

## 🛟 Getting help

➡️ [Reach out via mail](mailto:omchauhan64408@gmail.com)

➡️ [Connect on LinkedIn](https://www.linkedin.com/in/om-chauhan-dev/)





