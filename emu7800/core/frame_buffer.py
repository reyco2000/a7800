"""
FrameBuffer -- video and audio output buffers for EMU7800.
Ported from the C# FrameBuffer class.

Each emulated frame produces:

* A **video buffer** -- one byte per pixel, using the Atari palette index
  (0..255).  The buffer is laid out in scanline order:
  ``video_buffer[scanline * visible_pitch + x]``

* A **sound buffer** -- one 16-bit (little-endian) sample per scanline,
  stored as two bytes per entry:
  ``sound_buffer[scanline * 2]`` (low byte) and
  ``sound_buffer[scanline * 2 + 1]`` (high byte).

Standard dimensions
-------------------

=======  ==============  =========  =============
Console  visible_pitch   scanlines  video (bytes)
=======  ==============  =========  =============
2600     160             262        41 920
2600 PAL 160             312        49 920
7800     320             262        83 840
7800 PAL 320             312        99 840
=======  ==============  =========  =============
"""

from __future__ import annotations


class FrameBuffer:
    """Holds one frame of video output and the accompanying audio samples.

    Parameters
    ----------
    visible_pitch:
        Horizontal pixel count per scanline.  Typically 160 for the 2600
        or 320 for the 7800.
    scanlines:
        Total number of scanlines per frame.  262 for NTSC, 312 for PAL.
    """

    # Common constants so callers do not need to hard-code magic numbers.
    NTSC_SCANLINES: int = 262
    PAL_SCANLINES: int = 312
    VISIBLE_PITCH_2600: int = 160
    VISIBLE_PITCH_7800: int = 320

    # Sound: one 16-bit sample per scanline -> 2 bytes each.
    SOUND_BYTES_PER_SCANLINE: int = 2

    def __init__(self, visible_pitch: int, scanlines: int) -> None:
        if visible_pitch <= 0:
            raise ValueError(f"visible_pitch must be positive, got {visible_pitch}")
        if scanlines <= 0:
            raise ValueError(f"scanlines must be positive, got {scanlines}")

        self.visible_pitch: int = visible_pitch
        self.scanlines: int = scanlines

        # Video buffer: one byte per pixel (palette index).
        self._video_buffer_size: int = visible_pitch * scanlines
        self.video_buffer: bytearray = bytearray(self._video_buffer_size)

        # Sound buffer: two bytes (one 16-bit LE sample) per scanline.
        self._sound_buffer_size: int = scanlines * self.SOUND_BYTES_PER_SCANLINE
        self.sound_buffer: bytearray = bytearray(self._sound_buffer_size)

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def for_2600_ntsc(cls) -> FrameBuffer:
        """Create a FrameBuffer sized for Atari 2600 NTSC."""
        return cls(cls.VISIBLE_PITCH_2600, cls.NTSC_SCANLINES)

    @classmethod
    def for_2600_pal(cls) -> FrameBuffer:
        """Create a FrameBuffer sized for Atari 2600 PAL."""
        return cls(cls.VISIBLE_PITCH_2600, cls.PAL_SCANLINES)

    @classmethod
    def for_7800_ntsc(cls) -> FrameBuffer:
        """Create a FrameBuffer sized for Atari 7800 NTSC."""
        return cls(cls.VISIBLE_PITCH_7800, cls.NTSC_SCANLINES)

    @classmethod
    def for_7800_pal(cls) -> FrameBuffer:
        """Create a FrameBuffer sized for Atari 7800 PAL."""
        return cls(cls.VISIBLE_PITCH_7800, cls.PAL_SCANLINES)

    # ------------------------------------------------------------------
    # Video helpers
    # ------------------------------------------------------------------

    @property
    def video_buffer_size(self) -> int:
        """Total number of bytes in the video buffer."""
        return self._video_buffer_size

    def video_offset(self, scanline: int) -> int:
        """Return the byte offset into :attr:`video_buffer` for *scanline*.

        Raises:
            IndexError: If *scanline* is out of range.
        """
        if not 0 <= scanline < self.scanlines:
            raise IndexError(
                f"scanline {scanline} out of range [0, {self.scanlines})"
            )
        return scanline * self.visible_pitch

    def write_pixel(self, scanline: int, x: int, palette_index: int) -> None:
        """Write a single pixel to the video buffer.

        Args:
            scanline: The scanline number (0-based).
            x: The horizontal pixel position within the scanline.
            palette_index: The Atari palette index (0..255).
        """
        offset = scanline * self.visible_pitch + x
        self.video_buffer[offset] = palette_index & 0xFF

    def read_pixel(self, scanline: int, x: int) -> int:
        """Read a single pixel from the video buffer.

        Returns:
            The palette index at the given position.
        """
        offset = scanline * self.visible_pitch + x
        return self.video_buffer[offset]

    # ------------------------------------------------------------------
    # Sound helpers
    # ------------------------------------------------------------------

    @property
    def sound_buffer_size(self) -> int:
        """Total number of bytes in the sound buffer."""
        return self._sound_buffer_size

    def write_sound_sample(self, scanline: int, sample: int) -> None:
        """Write a 16-bit audio sample for the given scanline.

        The sample is stored in little-endian byte order.

        Args:
            scanline: The scanline number (0-based).
            sample: A signed or unsigned 16-bit value.  Only the low 16 bits
                    are kept.
        """
        offset = scanline * self.SOUND_BYTES_PER_SCANLINE
        sample &= 0xFFFF
        self.sound_buffer[offset] = sample & 0xFF
        self.sound_buffer[offset + 1] = (sample >> 8) & 0xFF

    def read_sound_sample(self, scanline: int) -> int:
        """Read the 16-bit audio sample for the given scanline.

        Returns:
            The unsigned 16-bit sample stored for that scanline.
        """
        offset = scanline * self.SOUND_BYTES_PER_SCANLINE
        return self.sound_buffer[offset] | (self.sound_buffer[offset + 1] << 8)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Zero out both the video and sound buffers."""
        for i in range(self._video_buffer_size):
            self.video_buffer[i] = 0
        for i in range(self._sound_buffer_size):
            self.sound_buffer[i] = 0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FrameBuffer("
            f"visible_pitch={self.visible_pitch}, "
            f"scanlines={self.scanlines}, "
            f"video_bytes={self._video_buffer_size}, "
            f"sound_bytes={self._sound_buffer_size})"
        )
