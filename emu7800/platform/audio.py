"""
Audio output device for EMU7800.
Uses pygame.mixer to play sound samples produced by the emulated machine.

Each emulated frame writes one 16-bit (signed, little-endian) audio sample
per scanline into the FrameBuffer's sound_buffer.  This module reads those
samples after every frame, upsamples them to match the machine's declared
audio sample rate, and feeds the result into pygame's audio mixer.

Sample-rate reconciliation
--------------------------
The FrameBuffer produces exactly ``scanlines`` samples per frame (one per
scanline).  The *native* rate is therefore ``scanlines * frame_hz`` (e.g.
262 * 60 = 15 720 Hz for NTSC).  However the machine may declare a higher
``sound_sample_frequency`` (e.g. 31 440 Hz for the 2600 NTSC, which is 2x
the native rate).  In that case each raw sample is duplicated by a small
integer factor so the output chunk has exactly
``sound_sample_frequency / frame_hz`` samples -- keeping audio and video
in perfect lock-step.

Approach
--------
1. ``pygame.mixer`` is initialised at the machine's declared sample rate.
2. After each :meth:`submit_frame` call the scanline samples are extracted,
   upsampled if needed, wrapped in a ``pygame.mixer.Sound``, and queued on
   a dedicated channel.
3. A small ring of pre-allocated ``Sound`` objects is reused to avoid
   per-frame allocation overhead and to keep a shallow playback queue that
   smooths over scheduling jitter.
"""

from __future__ import annotations

import logging
import struct
from typing import Optional

import pygame

logger = logging.getLogger(__name__)

# Number of Sound objects to keep in the ring buffer.  Having more than one
# allows us to queue the *next* frame's audio while the *current* one is
# still playing, preventing gaps.
_RING_SIZE: int = 4

# Minimum pygame mixer buffer size (in samples).  Smaller values reduce
# latency but may cause underruns on slower machines.
_MIXER_BUFFER_SAMPLES: int = 512


class AudioDevice:
    """Stream audio from the emulated machine to the speakers.

    Parameters
    ----------
    machine:
        The emulated machine.  Expected attributes:

        * ``frame_buffer`` -- :class:`~emu7800.core.frame_buffer.FrameBuffer`
        * ``frame_hz`` -- ``int``, frames per second (60 for NTSC, 50 for PAL)

        Optional:

        * ``sound_sample_frequency`` -- ``int``, the declared audio sample
          rate in Hz (e.g. 31 440).  When absent it falls back to
          ``scanlines * frame_hz``.
    enabled:
        Set to ``False`` to create the device in a silent / no-op mode.
    """

    def __init__(self, machine: object, *, enabled: bool = True) -> None:
        self._machine = machine
        self._enabled: bool = enabled
        self._channel: Optional[pygame.mixer.Channel] = None
        self._ring: list[Optional[pygame.mixer.Sound]] = [None] * _RING_SIZE
        self._ring_idx: int = 0

        fb = machine.frame_buffer  # type: ignore[attr-defined]
        self._scanlines: int = fb.scanlines
        frame_hz: int = getattr(machine, "frame_hz", 60)

        # Native rate: the number of samples the FrameBuffer actually has.
        self._native_rate: int = self._scanlines * frame_hz

        # Declared rate from the machine (may be higher than native).
        self._sample_rate: int = getattr(
            machine,
            "sound_sample_frequency",
            self._native_rate,
        )

        # Compute the integer upsample factor.  For NTSC 2600 this is
        # 31440 / 15720 = 2.  For machines where the rates match it is 1.
        if self._sample_rate >= self._native_rate and self._native_rate > 0:
            self._upsample: int = max(1, self._sample_rate // self._native_rate)
        else:
            self._upsample = 1

        # Number of raw bytes in the FrameBuffer sound_buffer that we read
        # each frame (one 16-bit sample per scanline = 2 bytes each).
        self._raw_byte_count: int = self._scanlines * 2

        # The actual mixer rate (after rounding the upsample factor) so that
        # each frame produces an exact integer number of samples.
        self._mixer_rate: int = self._native_rate * self._upsample

        if not self._enabled:
            logger.info("AudioDevice: disabled (silent mode)")
            return

        self._init_mixer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """The effective audio sample rate in Hz used by the mixer."""
        return self._mixer_rate

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if value and not self._enabled:
            self._init_mixer()
        elif not value and self._enabled:
            self._shutdown_mixer()
        self._enabled = value

    def submit_frame(self) -> None:
        """Read the current frame's audio samples from the FrameBuffer and
        queue them for playback.

        Call this once per frame, **after** :meth:`machine.compute_next_frame`.
        """
        if not self._enabled or self._channel is None:
            return

        fb = self._machine.frame_buffer  # type: ignore[attr-defined]
        sound_buf = fb.sound_buffer

        # Extract the raw bytes for this frame.
        raw = bytes(sound_buf[: self._raw_byte_count])

        # Upsample if needed (e.g. duplicate each sample 2x for 31440 Hz).
        if self._upsample > 1:
            raw = self._upsample_bytes(raw, self._upsample)

        # Wrap in a pygame Sound and queue it.
        try:
            snd = pygame.mixer.Sound(buffer=raw)
        except Exception:
            logger.debug("AudioDevice: Sound(buffer=) failed; skipping frame")
            return

        # Place the new sound in the ring (keeps a reference alive so pygame
        # does not garbage-collect the buffer while it is still playing).
        self._ring[self._ring_idx] = snd
        self._ring_idx = (self._ring_idx + 1) % _RING_SIZE

        # Queue the sound.  If the channel is idle it starts immediately;
        # if something is already playing it will start as soon as the
        # current sound finishes.
        if not self._channel.get_busy():
            self._channel.play(snd)
        else:
            self._channel.queue(snd)

    def shutdown(self) -> None:
        """Stop playback and release the mixer."""
        self._shutdown_mixer()

    # ------------------------------------------------------------------
    # Volume control
    # ------------------------------------------------------------------

    def get_volume(self) -> float:
        """Return the current volume (0.0 .. 1.0)."""
        if self._channel is not None:
            return self._channel.get_volume()
        return 1.0

    def set_volume(self, volume: float) -> None:
        """Set the playback volume (0.0 = mute, 1.0 = full)."""
        volume = max(0.0, min(1.0, volume))
        if self._channel is not None:
            self._channel.set_volume(volume)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _upsample_bytes(raw: bytes, factor: int) -> bytes:
        """Duplicate each 16-bit sample *factor* times.

        Parameters
        ----------
        raw:
            Byte string of 16-bit little-endian samples.
        factor:
            Duplication count (2 means each sample appears twice).

        Returns
        -------
        bytes
            The upsampled byte string.
        """
        if factor == 1:
            return raw
        out = bytearray(len(raw) * factor)
        sample_size = 2  # 16-bit
        num_samples = len(raw) // sample_size
        dst = 0
        for i in range(num_samples):
            src = i * sample_size
            sample_bytes = raw[src : src + sample_size]
            for _ in range(factor):
                out[dst : dst + sample_size] = sample_bytes
                dst += sample_size
        return bytes(out)

    def _init_mixer(self) -> None:
        """Initialise (or re-initialise) the pygame mixer."""
        # pygame.mixer may already be initialised by the display layer.
        # Re-init with our desired parameters -- pygame handles this
        # gracefully (it quits and re-inits if already running).
        try:
            pygame.mixer.quit()
        except pygame.error:
            pass

        try:
            pygame.mixer.init(
                frequency=self._mixer_rate,
                size=-16,       # signed 16-bit
                channels=1,     # mono
                buffer=_MIXER_BUFFER_SAMPLES,
            )
        except pygame.error as exc:
            logger.warning(
                "AudioDevice: pygame.mixer.init failed (%s); "
                "trying default frequency",
                exc,
            )
            try:
                pygame.mixer.init(
                    frequency=44100,
                    size=-16,
                    channels=1,
                    buffer=_MIXER_BUFFER_SAMPLES,
                )
                self._mixer_rate = 44100
            except pygame.error as exc2:
                logger.error("AudioDevice: mixer init failed entirely: %s", exc2)
                self._enabled = False
                return

        # Reserve a channel for emulator audio.
        pygame.mixer.set_num_channels(8)
        self._channel = pygame.mixer.Channel(0)

        actual_freq, actual_size, actual_channels = pygame.mixer.get_init()
        logger.info(
            "AudioDevice: mixer ready at %d Hz, %d-bit, %d ch "
            "(native=%d Hz, upsample=%dx, mixer_rate=%d Hz)",
            actual_freq,
            abs(actual_size),
            actual_channels,
            self._native_rate,
            self._upsample,
            self._mixer_rate,
        )

    def _shutdown_mixer(self) -> None:
        """Stop the mixer channel and release resources."""
        if self._channel is not None:
            try:
                self._channel.stop()
            except pygame.error:
                pass
            self._channel = None

        # Clear the ring to release Sound references.
        for i in range(_RING_SIZE):
            self._ring[i] = None

        try:
            pygame.mixer.quit()
        except pygame.error:
            pass
