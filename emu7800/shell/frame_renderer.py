"""
Frame renderer for EMU7800.
Converts the machine's palette-indexed FrameBuffer into an RGB pygame Surface.

The emulation core produces one byte per pixel (an Atari palette index in
0..255).  This module looks up each index in the appropriate NTSC or PAL
colour table and writes the result into a pygame Surface suitable for
blitting to the display.

Performance notes
-----------------
The inner loop uses **numpy** for bulk palette look-up, which is orders of
magnitude faster than per-pixel Python iteration.  A pure-Python fallback
is provided for environments where numpy is not available.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import pygame

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

from emu7800.core.types import MachineType

logger = logging.getLogger(__name__)


# ========================================================================
# Standard Atari colour palettes (256 entries each, 0xRRGGBB)
# ========================================================================
# The TIA produces 128 unique colours (16 hues x 8 luminance levels).
# Palette indices 0-255 are used, but the low bit is ignored by hardware,
# so entries come in identical pairs: [0]==[1], [2]==[3], etc.
#
# The values below are the widely-used palette from the Stella emulator
# project and match the C# EMU7800 ``TIATables`` palettes closely.
# ========================================================================

# fmt: off
NTSC_PALETTE: list[int] = [
    # Hue 0 -- grey
    0x000000, 0x000000, 0x4A4A4A, 0x4A4A4A, 0x6C6C6C, 0x6C6C6C, 0x909090, 0x909090,
    0xB0B0B0, 0xB0B0B0, 0xC8C8C8, 0xC8C8C8, 0xDCDCDC, 0xDCDCDC, 0xECECEC, 0xECECEC,
    # Hue 1 -- gold
    0x484800, 0x484800, 0x636200, 0x636200, 0x7C7C00, 0x7C7C00, 0x959500, 0x959500,
    0xABAB00, 0xABAB00, 0xC1C100, 0xC1C100, 0xD7D700, 0xD7D700, 0xEDED00, 0xEDED00,
    # Hue 2 -- orange
    0x702800, 0x702800, 0x844414, 0x844414, 0x985C28, 0x985C28, 0xAC783C, 0xAC783C,
    0xBC8C4C, 0xBC8C4C, 0xCCA05C, 0xCCA05C, 0xDCB468, 0xDCB468, 0xECC878, 0xECC878,
    # Hue 3 -- red-orange
    0x841800, 0x841800, 0x983418, 0x983418, 0xAC5030, 0xAC5030, 0xC06848, 0xC06848,
    0xD0805C, 0xD0805C, 0xE09470, 0xE09470, 0xECA880, 0xECA880, 0xFCBC94, 0xFCBC94,
    # Hue 4 -- pink
    0x880000, 0x880000, 0x9C2020, 0x9C2020, 0xB03C3C, 0xB03C3C, 0xC05858, 0xC05858,
    0xD07070, 0xD07070, 0xE08888, 0xE08888, 0xECA0A0, 0xECA0A0, 0xFCB4B4, 0xFCB4B4,
    # Hue 5 -- purple
    0x78005C, 0x78005C, 0x8C2074, 0x8C2074, 0xA03C88, 0xA03C88, 0xB0589C, 0xB0589C,
    0xC070B0, 0xC070B0, 0xD084C0, 0xD084C0, 0xDC9CD0, 0xDC9CD0, 0xECB0E0, 0xECB0E0,
    # Hue 6 -- purple-blue
    0x480078, 0x480078, 0x602090, 0x602090, 0x783CA4, 0x783CA4, 0x8C58B8, 0x8C58B8,
    0xA070CC, 0xA070CC, 0xB484DC, 0xB484DC, 0xC49CEC, 0xC49CEC, 0xD4B0FC, 0xD4B0FC,
    # Hue 7 -- blue
    0x140084, 0x140084, 0x302098, 0x302098, 0x4C3CAC, 0x4C3CAC, 0x6858C0, 0x6858C0,
    0x7C70D0, 0x7C70D0, 0x9488E0, 0x9488E0, 0xA8A0EC, 0xA8A0EC, 0xBCB4FC, 0xBCB4FC,
    # Hue 8 -- blue
    0x000088, 0x000088, 0x1C209C, 0x1C209C, 0x3840B0, 0x3840B0, 0x505CC0, 0x505CC0,
    0x6874D0, 0x6874D0, 0x7C8CE0, 0x7C8CE0, 0x90A4EC, 0x90A4EC, 0xA4B8FC, 0xA4B8FC,
    # Hue 9 -- light blue
    0x00187C, 0x00187C, 0x1C3890, 0x1C3890, 0x3854A8, 0x3854A8, 0x5070BC, 0x5070BC,
    0x6888CC, 0x6888CC, 0x7C9CDC, 0x7C9CDC, 0x90B4EC, 0x90B4EC, 0xA4C8FC, 0xA4C8FC,
    # Hue 10 -- turquoise
    0x002C5C, 0x002C5C, 0x1C4C78, 0x1C4C78, 0x386890, 0x386890, 0x5084AC, 0x5084AC,
    0x689CC0, 0x689CC0, 0x7CB4D4, 0x7CB4D4, 0x90CCE8, 0x90CCE8, 0xA4E0FC, 0xA4E0FC,
    # Hue 11 -- green-blue
    0x003C2C, 0x003C2C, 0x1C5C48, 0x1C5C48, 0x387C64, 0x387C64, 0x509C80, 0x509C80,
    0x68B494, 0x68B494, 0x7CD0AC, 0x7CD0AC, 0x90E4C0, 0x90E4C0, 0xA4FCD4, 0xA4FCD4,
    # Hue 12 -- green
    0x003C00, 0x003C00, 0x205C20, 0x205C20, 0x407C40, 0x407C40, 0x5C9C5C, 0x5C9C5C,
    0x74B474, 0x74B474, 0x8CD08C, 0x8CD08C, 0xA4E4A4, 0xA4E4A4, 0xB8FCB8, 0xB8FCB8,
    # Hue 13 -- yellow-green
    0x143800, 0x143800, 0x345C1C, 0x345C1C, 0x507C38, 0x507C38, 0x6C9850, 0x6C9850,
    0x84B468, 0x84B468, 0x9CCC7C, 0x9CCC7C, 0xB4E490, 0xB4E490, 0xC8FCA4, 0xC8FCA4,
    # Hue 14 -- orange-green
    0x2C3000, 0x2C3000, 0x4C501C, 0x4C501C, 0x687034, 0x687034, 0x848C4C, 0x848C4C,
    0x9CA864, 0x9CA864, 0xB4C078, 0xB4C078, 0xCCD488, 0xCCD488, 0xE0EC9C, 0xE0EC9C,
    # Hue 15 -- light orange
    0x442800, 0x442800, 0x644818, 0x644818, 0x846830, 0x846830, 0xA08444, 0xA08444,
    0xB89C58, 0xB89C58, 0xD0B46C, 0xD0B46C, 0xE8CC7C, 0xE8CC7C, 0xFCE08C, 0xFCE08C,
]

PAL_PALETTE: list[int] = [
    # Hue 0 -- grey
    0x000000, 0x000000, 0x2B2B2B, 0x2B2B2B, 0x525252, 0x525252, 0x767676, 0x767676,
    0x979797, 0x979797, 0xB6B6B6, 0xB6B6B6, 0xD2D2D2, 0xD2D2D2, 0xECECEC, 0xECECEC,
    # Hue 1
    0x000000, 0x000000, 0x2B2B2B, 0x2B2B2B, 0x525252, 0x525252, 0x767676, 0x767676,
    0x979797, 0x979797, 0xB6B6B6, 0xB6B6B6, 0xD2D2D2, 0xD2D2D2, 0xECECEC, 0xECECEC,
    # Hue 2 -- yellow
    0x805800, 0x805800, 0x967020, 0x967020, 0xAB8840, 0xAB8840, 0xBFA05C, 0xBFA05C,
    0xD2B878, 0xD2B878, 0xE3CE94, 0xE3CE94, 0xF3E4AC, 0xF3E4AC, 0xFFF8C4, 0xFFF8C4,
    # Hue 3 -- orange-yellow
    0x445C00, 0x445C00, 0x607420, 0x607420, 0x7C8C40, 0x7C8C40, 0x97A45C, 0x97A45C,
    0xB1BC78, 0xB1BC78, 0xC9D294, 0xC9D294, 0xE0E8AC, 0xE0E8AC, 0xF6FCC4, 0xF6FCC4,
    # Hue 4 -- green
    0x144C00, 0x144C00, 0x346820, 0x346820, 0x548440, 0x548440, 0x739C5C, 0x739C5C,
    0x90B478, 0x90B478, 0xABCC94, 0xABCC94, 0xC4E4AC, 0xC4E4AC, 0xDBFCC4, 0xDBFCC4,
    # Hue 5 -- green
    0x003C14, 0x003C14, 0x1C5C34, 0x1C5C34, 0x3C7C54, 0x3C7C54, 0x5C9C70, 0x5C9C70,
    0x78B48C, 0x78B48C, 0x94CCA8, 0x94CCA8, 0xACE4C0, 0xACE4C0, 0xC4FCD4, 0xC4FCD4,
    # Hue 6 -- cyan
    0x003C40, 0x003C40, 0x1C5C5C, 0x1C5C5C, 0x3C7C7C, 0x3C7C7C, 0x5C9C98, 0x5C9C98,
    0x78B4B4, 0x78B4B4, 0x94CCCC, 0x94CCCC, 0xACE4E4, 0xACE4E4, 0xC4FCFC, 0xC4FCFC,
    # Hue 7 -- cyan-blue
    0x002C5C, 0x002C5C, 0x1C4C78, 0x1C4C78, 0x3C6C98, 0x3C6C98, 0x5C8CB4, 0x5C8CB4,
    0x78A4D0, 0x78A4D0, 0x94BCE8, 0x94BCE8, 0xACD4FC, 0xACD4FC, 0xC4E8FC, 0xC4E8FC,
    # Hue 8 -- blue
    0x001470, 0x001470, 0x20348C, 0x20348C, 0x4054A8, 0x4054A8, 0x5C74C0, 0x5C74C0,
    0x788CD8, 0x788CD8, 0x94A4EC, 0x94A4EC, 0xACBCFC, 0xACBCFC, 0xC4D4FC, 0xC4D4FC,
    # Hue 9 -- blue-purple
    0x2C0078, 0x2C0078, 0x4C2090, 0x4C2090, 0x6840A8, 0x6840A8, 0x845CC0, 0x845CC0,
    0x9C78D8, 0x9C78D8, 0xB494EC, 0xB494EC, 0xCCACFC, 0xCCACFC, 0xE0C4FC, 0xE0C4FC,
    # Hue 10 -- purple
    0x6C0060, 0x6C0060, 0x842078, 0x842078, 0x9C4090, 0x9C4090, 0xB45CA8, 0xB45CA8,
    0xC878C0, 0xC878C0, 0xDC94D8, 0xDC94D8, 0xECACEC, 0xECACEC, 0xFCC4FC, 0xFCC4FC,
    # Hue 11 -- pink-purple
    0x6C0020, 0x6C0020, 0x842040, 0x842040, 0x9C4060, 0x9C4060, 0xB45C7C, 0xB45C7C,
    0xC87898, 0xC87898, 0xDC94B0, 0xDC94B0, 0xECACCC, 0xECACCC, 0xFCC4E0, 0xFCC4E0,
    # Hue 12 -- red
    0x7C2000, 0x7C2000, 0x943C20, 0x943C20, 0xAC5C40, 0xAC5C40, 0xC0785C, 0xC0785C,
    0xD49478, 0xD49478, 0xE4AC94, 0xE4AC94, 0xF4C4AC, 0xF4C4AC, 0xFCD8C4, 0xFCD8C4,
    # Hue 13 -- orange-red
    0x7C3C00, 0x7C3C00, 0x945820, 0x945820, 0xAC7440, 0xAC7440, 0xC08C5C, 0xC08C5C,
    0xD4A478, 0xD4A478, 0xE4BC94, 0xE4BC94, 0xF4D0AC, 0xF4D0AC, 0xFCE4C4, 0xFCE4C4,
    # Hue 14 -- brown
    0x583000, 0x583000, 0x744C1C, 0x744C1C, 0x906838, 0x906838, 0xA88454, 0xA88454,
    0xC09C70, 0xC09C70, 0xD4B48C, 0xD4B48C, 0xE8CCA4, 0xE8CCA4, 0xFCE0BC, 0xFCE0BC,
    # Hue 15 -- yellow-brown
    0x402C00, 0x402C00, 0x5C4C1C, 0x5C4C1C, 0x786838, 0x786838, 0x948454, 0x948454,
    0xAC9C70, 0xAC9C70, 0xC4B48C, 0xC4B48C, 0xD8CCA4, 0xD8CCA4, 0xECE0BC, 0xECE0BC,
]
# fmt: on

assert len(NTSC_PALETTE) == 256, f"NTSC palette must have 256 entries, got {len(NTSC_PALETTE)}"
assert len(PAL_PALETTE) == 256, f"PAL palette must have 256 entries, got {len(PAL_PALETTE)}"


def get_default_palette(machine_type: MachineType) -> list[int]:
    """Return the standard palette for the given machine type."""
    if MachineType.is_pal(machine_type):
        return list(PAL_PALETTE)
    return list(NTSC_PALETTE)


# ========================================================================
# FrameRenderer
# ========================================================================

class FrameRenderer:
    """Convert a machine's palette-indexed :class:`FrameBuffer` into an RGB
    :class:`pygame.Surface` each frame.

    Parameters
    ----------
    machine:
        The emulated machine.  Expected attributes:

        * ``frame_buffer`` -- a :class:`~emu7800.core.frame_buffer.FrameBuffer`
        * ``first_scanline`` -- ``int``, the first *visible* scanline
        * ``palette`` -- ``list[int]`` of 256 ``0xRRGGBB`` values
          (falls back to a default NTSC/PAL palette if not present)
        * ``machine_type`` -- :class:`MachineType`
    """

    def __init__(self, machine: object) -> None:
        self._machine = machine
        fb = machine.frame_buffer  # type: ignore[attr-defined]

        # Determine the first visible scanline.  The top portion of the frame
        # is vertical blank / overscan and should not be displayed.
        self._first_scanline: int = getattr(machine, "first_scanline", 0)

        self._visible_pitch: int = fb.visible_pitch
        self._total_scanlines: int = fb.scanlines
        self._visible_lines: int = self._total_scanlines - self._first_scanline

        # Resolve the palette.
        self._palette: list[int] = self._resolve_palette(machine)

        # Pre-build an RGB look-up table: palette index -> (R, G, B).
        # Used by both the numpy path and the pure-Python fallback.
        self._lut_r: list[int] = [(c >> 16) & 0xFF for c in self._palette]
        self._lut_g: list[int] = [(c >> 8) & 0xFF for c in self._palette]
        self._lut_b: list[int] = [c & 0xFF for c in self._palette]

        if _HAS_NUMPY:
            # Build a numpy LUT of shape (256, 3) for vectorised look-up.
            self._np_lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                self._np_lut[i, 0] = self._lut_r[i]
                self._np_lut[i, 1] = self._lut_g[i]
                self._np_lut[i, 2] = self._lut_b[i]

        # Create the output surface (RGB, no alpha needed).
        self._surface: pygame.Surface = pygame.Surface(
            (self._visible_pitch, self._visible_lines)
        )

        logger.info(
            "FrameRenderer: %dx%d visible (first_scanline=%d, pitch=%d)",
            self._visible_pitch,
            self._visible_lines,
            self._first_scanline,
            self._visible_pitch,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Width of the rendered surface in pixels."""
        return self._visible_pitch

    @property
    def height(self) -> int:
        """Height of the rendered surface in pixels."""
        return self._visible_lines

    @property
    def surface(self) -> pygame.Surface:
        """The internal pygame Surface (updated on each :meth:`render` call)."""
        return self._surface

    def render(self) -> pygame.Surface:
        """Render the current frame and return the surface.

        Reads ``machine.frame_buffer.video_buffer``, applies the palette
        look-up, and writes RGB pixels into the internal surface.  The
        same :class:`pygame.Surface` object is reused each frame to avoid
        allocation churn.

        Returns:
            The updated :class:`pygame.Surface`.
        """
        fb = self._machine.frame_buffer  # type: ignore[attr-defined]
        video_buf = fb.video_buffer

        if _HAS_NUMPY:
            self._render_numpy(video_buf)
        else:
            self._render_python(video_buf)

        return self._surface

    def update_palette(self, palette: Sequence[int]) -> None:
        """Replace the colour palette at runtime.

        Parameters:
            palette: 256 ``0xRRGGBB`` values.
        """
        if len(palette) != 256:
            raise ValueError(f"Palette must have 256 entries, got {len(palette)}")
        self._palette = list(palette)
        self._lut_r = [(c >> 16) & 0xFF for c in self._palette]
        self._lut_g = [(c >> 8) & 0xFF for c in self._palette]
        self._lut_b = [c & 0xFF for c in self._palette]
        if _HAS_NUMPY:
            for i in range(256):
                self._np_lut[i, 0] = self._lut_r[i]
                self._np_lut[i, 1] = self._lut_g[i]
                self._np_lut[i, 2] = self._lut_b[i]

    # ------------------------------------------------------------------
    # Rendering back-ends
    # ------------------------------------------------------------------

    def _render_numpy(self, video_buf: bytearray) -> None:
        """Fast rendering path using numpy array indexing."""
        pitch = self._visible_pitch
        first = self._first_scanline
        vis = self._visible_lines

        # Wrap the video buffer in a numpy array (no copy) and reshape.
        raw = np.frombuffer(video_buf, dtype=np.uint8)
        frame = raw.reshape((self._total_scanlines, pitch))

        # Slice to visible region.
        visible = frame[first : first + vis, :]

        # Palette look-up: (H, W) uint8 -> (H, W, 3) uint8
        rgb = self._np_lut[visible]

        # pygame surfarray expects (W, H, 3) -- transpose width and height.
        pygame.surfarray.blit_array(self._surface, rgb.transpose(1, 0, 2))

    def _render_python(self, video_buf: bytearray) -> None:
        """Pure-Python fallback -- slow but always available."""
        pitch = self._visible_pitch
        first = self._first_scanline
        vis = self._visible_lines
        lut_r = self._lut_r
        lut_g = self._lut_g
        lut_b = self._lut_b

        pix_array = pygame.PixelArray(self._surface)  # type: ignore[arg-type]
        try:
            for y in range(vis):
                src_scanline = first + y
                offset = src_scanline * pitch
                for x in range(pitch):
                    idx = video_buf[offset + x]
                    pix_array[x, y] = (lut_r[idx], lut_g[idx], lut_b[idx])
        finally:
            pix_array.close()

    # ------------------------------------------------------------------
    # Palette resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_palette(machine: object) -> list[int]:
        """Obtain a 256-entry palette from the machine, or fall back to a
        default based on the machine type."""
        palette = getattr(machine, "palette", None)
        if palette is not None and len(palette) == 256:
            return list(palette)

        # Fall back to default palette based on machine_type.
        mt = getattr(machine, "machine_type", MachineType.A2600NTSC)
        return get_default_palette(mt)
