# pyright: reportArgumentType=false
#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""TTS Caching Example - Reduce API costs by caching repeated phrases.

Requirements:
    pip install "pipecat-tts-cache[examples]"

    Set environment variables:
    - DEEPGRAM_API_KEY
    - CARTESIA_API_KEY
    - GOOGLE_API_KEY

This example demonstrates the TTS caching feature. To test:
1. Run the bot and let it say the intro
2. Say "please repeat that" or "say that again"
3. Watch the logs - you should see cache hits on the second request

To switch TTS providers, create a cached version like:
    class CachedMyTTS(TTSCacheMixin, MyTTSService): pass

Supported TTS types:
- WebSocket with word timestamps (Eg: CartesiaTTSService): Full batch caching
- WebSocket without timestamps (Eg: DeepgramTTSService): Single-sentence caching only
- HTTP with timestamps (Eg: ElevenLabsHttpTTSService): Full caching
- HTTP without timestamps (Eg: GoogleHttpTTSService): Full caching
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat_tts_cache import MemoryCacheBackend, TTSCacheMixin

load_dotenv(override=True)


class CachedCartesiaTTSService(TTSCacheMixin, CartesiaTTSService):
    """Cartesia WebSocket TTS with caching support."""

    pass


FIXED_INTRO = (
    "Hi, I am your friendly assistant with TTS caching enabled. Try saying: please repeat that."
)


def get_required_api_keys() -> tuple[str, str, str]:
    """Load and validate required API keys from environment."""
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
    cartesia_key = os.getenv("CARTESIA_API_KEY", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")

    if not all([deepgram_key, cartesia_key, google_key]):
        raise ValueError(
            "Missing required API keys. Set DEEPGRAM_API_KEY, CARTESIA_API_KEY, and GOOGLE_API_KEY"
        )

    return deepgram_key, cartesia_key, google_key


def create_cache_backend():
    """Create the appropriate cache backend based on environment."""
    use_redis = os.getenv("USE_REDIS_CACHE", "").lower() == "true"

    if use_redis:
        try:
            from pipecat_tts_cache.backends import RedisCacheBackend

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            safe_url = redis_url.split("@")[-1] if "@" in redis_url else redis_url
            logger.info(f"Using Redis cache backend: {safe_url}")
            return RedisCacheBackend(
                redis_url=redis_url,
                key_prefix="pipecat:tts:example:",
                max_connections=5,
            )
        except ImportError:
            logger.warning("Redis not available, falling back to memory cache")

    logger.info("Using in-memory cache backend (max_size=100)")
    return MemoryCacheBackend(max_size=100)


transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting TTS caching demo bot")

    deepgram_api_key, cartesia_api_key, google_api_key = get_required_api_keys()

    cache_backend = create_cache_backend()

    stt = DeepgramSTTService(api_key=deepgram_api_key)

    tts = CachedCartesiaTTSService(
        api_key=cartesia_api_key,
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",
        cache_backend=cache_backend,
        cache_ttl=3600,
    )

    llm = GoogleLLMService(api_key=google_api_key)

    system_prompt = f"""\
You are a helpful assistant demonstrating TTS caching.

IMPORTANT RULES:
1. When asked to introduce yourself or greet the user, you MUST say EXACTLY this message, word for word:
"{FIXED_INTRO}"

2. When the user says anything like "repeat", "say that again", "repeat that",
"I didn't hear you", or "can you say that again", you MUST repeat EXACTLY
the same intro message above, word for word.

3. For other questions, respond naturally but keep responses short.

4. Your responses will be spoken aloud, so avoid special characters.
"""

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected - starting TTS cache demo")
        messages.append({"role": "user", "content": "Please introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")

        # Display cache statistics
        stats = await cache_backend.get_stats()
        logger.info("TTS Cache Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        await cache_backend.close()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
