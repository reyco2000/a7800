"""
MARIA graphics chip palette tables for the Atari 7800.

The MARIA chip uses the same colour encoding as the TIA (8-bit colour
registers: high nibble = hue, bits 3-1 = luminance, bit 0 unused).
The actual RGB values are produced by the television encoder and differ
between NTSC and PAL systems.

These palettes are used by the MARIA renderer to convert 7800 colour
register values into ARGB pixel values for display output.  Each palette
contains 256 uint32 entries in 0xAARRGGBB format.
"""

from __future__ import annotations

import math
from typing import List


# ---------------------------------------------------------------------------
# NTSC palette (256 entries, 0xAARRGGBB)
# ---------------------------------------------------------------------------
# The Atari 7800 NTSC colour output is very similar to the 2600 but uses a
# slightly warmer colour temperature.  The palette is generated from the
# YIQ colour model with parameters tuned to match real hardware captures.

def _generate_maria_ntsc_palette() -> List[int]:
    """Build the 256-entry MARIA NTSC palette."""
    # Luminance ramp (8 levels, normalised 0..1)
    luma = [0.0, 0.1046, 0.2093, 0.3186, 0.4372, 0.5651, 0.7070, 0.8605]

    palette: List[int] = [0] * 256
    for idx in range(256):
        hue = (idx >> 4) & 0x0F
        lum = (idx >> 1) & 0x07
        y = luma[lum]

        if hue == 0:
            r = g = b = int(y * 255 + 0.5)
        else:
            # Phase angle per hue step, shifted for warmer tones
            angle = math.radians((hue - 1) * 25.7 - 55.0)
            saturation = 0.26 * (0.65 + y * 0.35)
            i_comp = saturation * math.cos(angle)
            q_comp = saturation * math.sin(angle)

            # YIQ -> sRGB
            r = int((y + 0.956 * i_comp + 0.621 * q_comp) * 255 + 0.5)
            g = int((y - 0.272 * i_comp - 0.647 * q_comp) * 255 + 0.5)
            b = int((y - 1.107 * i_comp + 1.704 * q_comp) * 255 + 0.5)
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

        palette[idx] = 0xFF000000 | (r << 16) | (g << 8) | b
    return palette


# ---------------------------------------------------------------------------
# PAL palette (256 entries, 0xAARRGGBB)
# ---------------------------------------------------------------------------
# PAL Atari 7800 encodes colour with alternating subcarrier phase on
# successive scanlines; the net result is the "average" colour from both
# phases.  Hues 0 and 1 are both grey.

def _generate_maria_pal_palette() -> List[int]:
    """Build the 256-entry MARIA PAL palette."""
    luma = [0.0, 0.1046, 0.2093, 0.3186, 0.4372, 0.5651, 0.7070, 0.8605]

    palette: List[int] = [0] * 256
    for idx in range(256):
        hue = (idx >> 4) & 0x0F
        lum = (idx >> 1) & 0x07
        y = luma[lum]

        if hue < 2:
            r = g = b = int(y * 255 + 0.5)
        else:
            angle = math.radians((hue - 2) * 25.7 - 15.0)
            saturation = 0.30 * (0.65 + y * 0.35)
            u_comp = saturation * math.cos(angle)
            v_comp = saturation * math.sin(angle)

            # YUV -> sRGB
            r = int((y + 1.140 * v_comp) * 255 + 0.5)
            g = int((y - 0.396 * u_comp - 0.581 * v_comp) * 255 + 0.5)
            b = int((y + 2.029 * u_comp) * 255 + 0.5)
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

        palette[idx] = 0xFF000000 | (r << 16) | (g << 8) | b
    return palette


# ---------------------------------------------------------------------------
# Module-level pre-computed palettes
# ---------------------------------------------------------------------------

ntsc_palette: List[int] = _generate_maria_ntsc_palette()
"""256-entry MARIA NTSC colour palette (0xAARRGGBB)."""

pal_palette: List[int] = _generate_maria_pal_palette()
"""256-entry MARIA PAL colour palette (0xAARRGGBB)."""
