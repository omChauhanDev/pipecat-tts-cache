import pytest

from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse


@pytest.fixture
def dummy_response():
    """Provide a standard CachedTTSResponse instance for testing."""
    return CachedTTSResponse(
        audio_chunks=[
            CachedAudioChunk(audio=b"16_bytes_of_data", sample_rate=16000, num_channels=1)
        ],
        sample_rate=16000,
        num_channels=1,
        metadata={"text": "hello world"},
    )
