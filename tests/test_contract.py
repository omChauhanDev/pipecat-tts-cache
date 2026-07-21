#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Pipecat-contract guard tests.

These pin the exact integration points ``TTSCacheMixin`` depends on. If a future
Pipecat release changes any of them, these fail loudly at CI time instead of shipping a
silently-broken package — the failure mode that caused this whole maintenance cycle
(GitHub #1: ``run_tts()`` gaining ``context_id``).
"""

import inspect

from pipecat.frames.frames import (
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


def test_run_tts_takes_text_and_context_id():
    params = list(inspect.signature(TTSService.run_tts).parameters)
    assert params == ["self", "text", "context_id"]


def test_add_word_timestamps_has_every_kwarg_the_mixin_forwards():
    # The mixin forwards all of these to super(); a rename/removal must fail CI.
    params = inspect.signature(TTSService.add_word_timestamps).parameters
    for name in ("word_times", "context_id", "includes_inter_frame_spaces", "pre_merge_tokens"):
        assert name in params, f"TTSService.add_word_timestamps is missing '{name}'"


def test_push_frame_signature_is_stable():
    params = list(inspect.signature(TTSService.push_frame).parameters)
    assert params[:3] == ["self", "frame", "direction"]


def test_handle_interruption_signature_is_stable():
    # The mixin overrides this and relies on the base dispatch signature.
    params = list(inspect.signature(TTSService._handle_interruption).parameters)
    assert params[:3] == ["self", "frame", "direction"]


def test_sample_rate_is_a_readable_property():
    # The mixin reads self.sample_rate when building the cache key.
    assert isinstance(getattr(TTSService, "sample_rate", None), property)


def test_tts_settings_exposes_voice_model_and_given_fields():
    settings = TTSSettings(voice="v", model="m", language=None)
    assert hasattr(settings, "given_fields")
    given = settings.given_fields()
    assert given.get("voice") == "v"
    assert given.get("model") == "m"


def test_frames_construct_and_accept_dynamic_attributes():
    frame = TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
    # The mixin tags frames with a dynamic attribute; frames must not be slotted/frozen.
    setattr(frame, "_tts_cache_origin", True)
    assert getattr(frame, "_tts_cache_origin") is True
    assert TTSStartedFrame() is not None
    assert TTSStoppedFrame() is not None
    assert InterruptionFrame() is not None


def test_frame_direction_downstream_exists():
    assert FrameDirection.DOWNSTREAM is not None
