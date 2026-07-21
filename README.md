
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

The mixin works with **any** Pipecat `TTSService` — HTTP or WebSocket. It captures audio as it
flows through the pipeline (keyed by Pipecat's per-request audio context) and replays cached audio
back through that same audio context, so playback ordering is preserved even when cache hits and
live synthesis are mixed in a single turn.

How much detail is preserved on a cache hit depends on what the underlying service produces:

| **What the service emits** | **What is cached & replayed** | **Providers (examples)** |
|----------------------------|-------------------------------|--------------------------|
| **Word timestamps** | Audio **plus** word-level timestamps → `TTSTextFrame`s are regenerated on replay, so transcripts and word alignment stay correct (even on interruption). | Cartesia, Rime, ElevenLabs, Hume |
| **Audio only** | The full audio response is cached and replayed; transcript text is preserved via the framework's own text frames. | Google, OpenAI, Deepgram, Sarvam |

> Since Pipecat `1.0`, word-timestamp support lives on the base `TTSService`, so provider class
> names are no longer meaningful for caching — the mixin adapts to whatever each service emits at
> runtime.

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
    settings=CachedGoogleTTS.Settings(voice="en-US-Chirp3-HD-Charon"),
    cache_backend=MemoryCacheBackend(max_size=1000),
    cache_ttl=86400,  # Cache for 24 hours
)

```

### 2. Distributed Redis Cache (Production)

For production deployments, use `RedisCacheBackend`. This allows the cache to persist across restarts and be shared among multiple bot instances.

```python
from pipecat_tts_cache.backends import RedisCacheBackend

tts = CachedGoogleTTS(
    settings=CachedGoogleTTS.Settings(voice="en-US-Chirp3-HD-Charon"),
    cache_backend=RedisCacheBackend(
        redis_url="redis://localhost:6379/0",
        key_prefix="pipecat:tts:",
    ),
    cache_ttl=604800, # Cache for 1 week
)

```

> ⚠️ **Security — Redis trust boundary.** The Redis backend serializes cached audio with
> `pickle`, so it must be treated as trusted: use a **single-tenant, authenticated,
> network-isolated** Redis instance. Never point it at a shared/untrusted Redis — anyone who
> can write the keyspace could achieve code execution when an entry is read. See `SECURITY.md`.
>
> **Backend lifecycle.** You own the backend you pass in — reuse a single instance across
> sessions (recommended for Redis, so its connection pool is shared) and call
> `await backend.close()` when your app shuts down. The package never closes an injected
> backend, so a shared one is safe.

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

## Running the Example

### Prerequisites

```bash
# Install with example dependencies
pip install "pipecat-tts-cache[examples]"

# Optional: Install with Redis support
pip install "pipecat-tts-cache[examples,redis]"

# Set environment variables
export DEEPGRAM_API_KEY=your_key
export CARTESIA_API_KEY=your_key
export GOOGLE_API_KEY=your_key

# Optional: For Redis backend
export USE_REDIS_CACHE=true
export REDIS_URL=redis://localhost:6379/0
```

### Option 1: Daily Bots (Recommended)

```bash
# Start the bot server
python examples/basic_caching.py --host 0.0.0.0 --port 7860

# Connect via Daily Bots or your Daily room
```

### Option 2: Local WebRTC

```bash
# Run with local WebRTC transport
python examples/basic_caching.py -t webrtc --host localhost --port 8765
```

## Compatibility

| Pipecat Version | Status |
|-----------------|--------|
| v1.5.0+         | ✅ Tested (requires Python ≥ 3.11) |
| v0.0.91 – v0.0.101 | ⚠️ Use `pipecat-tts-cache` `0.0.3` |

> Pipecat `0.0.102` changed the `TTSService.run_tts()` contract and `1.0` reorganized the TTS/
> word-timestamp architecture. This release targets the current Pipecat line (`>=1.5.0`); the cache
> key format also changed, so entries written by older versions are ignored (they re-synthesize once).

## 🛟 Getting help

➡️ [Reach out via mail](https://mail.google.com/mail/?view=cm&fs=1&to=omchauhan64408@gmail.com)

➡️ [Connect on LinkedIn](https://www.linkedin.com/in/omchauhandev/)






