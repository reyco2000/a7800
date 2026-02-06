"""
TIA (Television Interface Adaptor) lookup tables for graphics rendering.

Ported from EMU7800 C# TIATables. Provides pre-computed collision masks,
playfield/ball/missile/player graphic masks, bit-reversal tables, and
NTSC/PAL color palettes.
"""

from __future__ import annotations

import math
from typing import List

from .types import TIACxFlags, TIACxPairFlags


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_lfsr(bits: int, tap_a: int, tap_b: int) -> List[int]:
    """Generate a maximal-length LFSR sequence.

    Returns a list of 0/1 values with length (2**bits - 1).
    """
    size = (1 << bits) - 1
    reg = (1 << bits) - 1  # seed: all ones
    seq: List[int] = []
    for _ in range(size):
        seq.append(reg & 1)
        feedback = ((reg >> tap_a) ^ (reg >> tap_b)) & 1
        reg = (reg >> 1) | (feedback << (bits - 1))
    return seq


# ---------------------------------------------------------------------------
# Collision mask  (64 entries)
# ---------------------------------------------------------------------------

def _build_collision_mask() -> List[int]:
    """Map every 6-bit combination of TIACxFlags to TIACxPairFlags."""
    table: List[int] = [0] * 64
    for i in range(64):
        pf = bool(i & TIACxFlags.PF)
        bl = bool(i & TIACxFlags.BL)
        m0 = bool(i & TIACxFlags.M0)
        m1 = bool(i & TIACxFlags.M1)
        p0 = bool(i & TIACxFlags.P0)
        p1 = bool(i & TIACxFlags.P1)

        v = 0
        if m0 and p1:
            v |= TIACxPairFlags.M0P1
        if m0 and p0:
            v |= TIACxPairFlags.M0P0
        if m1 and p0:
            v |= TIACxPairFlags.M1P0
        if m1 and p1:
            v |= TIACxPairFlags.M1P1
        if p0 and pf:
            v |= TIACxPairFlags.P0PF
        if p0 and bl:
            v |= TIACxPairFlags.P0BL
        if p1 and pf:
            v |= TIACxPairFlags.P1PF
        if p1 and bl:
            v |= TIACxPairFlags.P1BL
        if m0 and pf:
            v |= TIACxPairFlags.M0PF
        if m0 and bl:
            v |= TIACxPairFlags.M0BL
        if m1 and pf:
            v |= TIACxPairFlags.M1PF
        if m1 and bl:
            v |= TIACxPairFlags.M1BL
        if bl and pf:
            v |= TIACxPairFlags.BLPF
        if p0 and p1:
            v |= TIACxPairFlags.P0P1
        if m0 and m1:
            v |= TIACxPairFlags.M0M1
        table[i] = v
    return table


# ---------------------------------------------------------------------------
# Playfield mask  [reflection_state 0|1][hpos 0..159] -> uint bitmask
# ---------------------------------------------------------------------------
# PF registers are packed as:  pf_packed = PF0 | (PF1 << 8) | (PF2 << 16)
# The 20 playfield columns (each 4 clocks wide) map to individual bits
# of that packed word.

# Left half columns 0-19:
#   col  0 (hpos  0- 3) -> PF0 bit4    col 12 (hpos 48-51) -> PF2 bit0
#   col  1 (hpos  4- 7) -> PF0 bit5    col 13 (hpos 52-55) -> PF2 bit1
#   col  2 (hpos  8-11) -> PF0 bit6    col 14 (hpos 56-59) -> PF2 bit2
#   col  3 (hpos 12-15) -> PF0 bit7    col 15 (hpos 60-63) -> PF2 bit3
#   col  4 (hpos 16-19) -> PF1 bit7    col 16 (hpos 64-67) -> PF2 bit4
#   col  5 (hpos 20-23) -> PF1 bit6    col 17 (hpos 68-71) -> PF2 bit5
#   col  6 (hpos 24-27) -> PF1 bit5    col 18 (hpos 72-75) -> PF2 bit6
#   col  7 (hpos 28-31) -> PF1 bit4    col 19 (hpos 76-79) -> PF2 bit7
#   col  8 (hpos 32-35) -> PF1 bit3
#   col  9 (hpos 36-39) -> PF1 bit2
#   col 10 (hpos 40-43) -> PF1 bit1
#   col 11 (hpos 44-47) -> PF1 bit0

_PF_COLUMN_BIT: List[int] = [
    1 << 4,   # col 0  -> PF0 bit 4
    1 << 5,   # col 1  -> PF0 bit 5
    1 << 6,   # col 2  -> PF0 bit 6
    1 << 7,   # col 3  -> PF0 bit 7
    1 << 15,  # col 4  -> PF1 bit 7
    1 << 14,  # col 5  -> PF1 bit 6
    1 << 13,  # col 6  -> PF1 bit 5
    1 << 12,  # col 7  -> PF1 bit 4
    1 << 11,  # col 8  -> PF1 bit 3
    1 << 10,  # col 9  -> PF1 bit 2
    1 << 9,   # col 10 -> PF1 bit 1
    1 << 8,   # col 11 -> PF1 bit 0
    1 << 16,  # col 12 -> PF2 bit 0
    1 << 17,  # col 13 -> PF2 bit 1
    1 << 18,  # col 14 -> PF2 bit 2
    1 << 19,  # col 15 -> PF2 bit 3
    1 << 20,  # col 16 -> PF2 bit 4
    1 << 21,  # col 17 -> PF2 bit 5
    1 << 22,  # col 18 -> PF2 bit 6
    1 << 23,  # col 19 -> PF2 bit 7
]


def _build_pf_mask() -> List[List[int]]:
    """Build playfield bit-mask table.

    Returns ``table[reflection][hpos]`` where *reflection* is 0 for normal
    (right half repeats left) or 1 for reflected (right half mirrors left).
    Each entry is a single-bit mask into the packed PF register word.
    """
    table: List[List[int]] = [[0] * 160, [0] * 160]

    for hpos in range(160):
        if hpos < 80:
            col = hpos >> 2  # hpos // 4
            mask = _PF_COLUMN_BIT[col]
            table[0][hpos] = mask
            table[1][hpos] = mask
        else:
            # Normal: repeat left half
            left_col = (hpos - 80) >> 2
            table[0][hpos] = _PF_COLUMN_BIT[left_col]
            # Reflected: mirror left half
            reflected_col = 19 - ((hpos - 80) >> 2)
            table[1][hpos] = _PF_COLUMN_BIT[reflected_col]

    return table


# ---------------------------------------------------------------------------
# Ball mask  [size 0..3][pos 0..159] -> bool
# ---------------------------------------------------------------------------
# size: 0=1px, 1=2px, 2=4px, 3=8px

def _build_bl_mask() -> List[List[bool]]:
    table: List[List[bool]] = [[False] * 160 for _ in range(4)]
    for size in range(4):
        width = 1 << size  # 1, 2, 4, 8
        for pos in range(160):
            table[size][pos] = pos < width
    return table


# ---------------------------------------------------------------------------
# Missile mask  [size 0..3][nusiz 0..7][pos 0..159] -> bool
# ---------------------------------------------------------------------------
# NUSIZ copy offsets (same as player):
#   0: [0]              4: [0, 64]
#   1: [0, 16]          5: [0]  (double-size player, single missile)
#   2: [0, 32]          6: [0, 32, 64]
#   3: [0, 16, 32]      7: [0]  (quad-size player, single missile)
# Missile width is always determined by *size*, not NUSIZ stretch.

_NUSIZ_COPY_OFFSETS: List[List[int]] = [
    [0],            # 0: one copy
    [0, 16],        # 1: two close
    [0, 32],        # 2: two medium
    [0, 16, 32],    # 3: three close
    [0, 64],        # 4: two wide
    [0],            # 5: one copy (double-size player)
    [0, 32, 64],    # 6: three medium
    [0],            # 7: one copy (quad-size player)
]


def _build_mx_mask() -> List[List[List[bool]]]:
    table: List[List[List[bool]]] = [
        [[False] * 160 for _ in range(8)] for _ in range(4)
    ]
    for size in range(4):
        width = 1 << size
        for nusiz in range(8):
            for offset in _NUSIZ_COPY_OFFSETS[nusiz]:
                for px in range(width):
                    p = (offset + px) % 160
                    table[size][nusiz][p] = True
    return table


# ---------------------------------------------------------------------------
# Player mask  [suppress 0|1][nusiz 0..7][pos 0..159] -> uint8 GRP mask
# ---------------------------------------------------------------------------
# Returns a single-bit mask (0x80..0x01) indicating which bit of the GRP
# register is visible at *pos* pixels from the player reset position,
# or 0 if nothing is visible.
#
# NUSIZ stretch:  5 -> double (each GRP pixel is 2 clocks)
#                 7 -> quad   (each GRP pixel is 4 clocks)
#                 all others -> normal (1 clock per pixel)

_NUSIZ_PIXEL_WIDTH: List[int] = [1, 1, 1, 1, 1, 2, 1, 4]


def _build_px_mask() -> List[List[List[int]]]:
    """Build player graphic mask table.

    ``table[suppress][nusiz][pos]``
    *suppress* = 1 suppresses all output (returns 0 everywhere).
    """
    table: List[List[List[int]]] = [
        [[0] * 160 for _ in range(8)] for _ in range(2)
    ]

    for nusiz in range(8):
        pixel_w = _NUSIZ_PIXEL_WIDTH[nusiz]
        copy_width = 8 * pixel_w  # total clocks per copy
        for offset in _NUSIZ_COPY_OFFSETS[nusiz]:
            for clock in range(copy_width):
                bit_index = 7 - (clock // pixel_w)  # 7 down to 0
                mask = 1 << bit_index
                p = (offset + clock) % 160
                table[0][nusiz][p] = mask

    # suppress == 1 -> all zeros (already initialised to 0)
    return table


# ---------------------------------------------------------------------------
# GRP bit-reverse table  (256 entries)
# ---------------------------------------------------------------------------

def _build_grp_reflect() -> List[int]:
    table: List[int] = [0] * 256
    for i in range(256):
        v = 0
        for bit in range(8):
            if i & (1 << bit):
                v |= 1 << (7 - bit)
        table[i] = v
    return table


# ---------------------------------------------------------------------------
# Colour palettes – NTSC and PAL  (256 x uint32 ARGB)
# ---------------------------------------------------------------------------
# Generated with a YIQ (NTSC) / YUV (PAL) model that closely matches the
# Atari TIA / MARIA colour output.  128 unique colours (hue * luminance),
# each duplicated for even/odd indices (bit 0 of the colour register is
# unused by the hardware).

def _generate_ntsc_palette() -> List[int]:
    """Generate a 256-entry NTSC ARGB palette."""
    # Eight luminance levels (normalised 0-1)
    luma = [0.0, 0.1046, 0.2093, 0.3186, 0.4372, 0.5651, 0.7070, 0.8605]
    palette: List[int] = [0] * 256

    for idx in range(256):
        hue = (idx >> 4) & 0x0F
        lum = (idx >> 1) & 0x07
        y = luma[lum]

        if hue == 0:
            r = g = b = int(y * 255 + 0.5)
        else:
            # NTSC colour burst angle per hue step (~25.7 degrees)
            angle = math.radians((hue - 1) * 25.7 - 58.0)
            saturation = 0.27 * (0.7 + y * 0.3)
            i_comp = saturation * math.cos(angle)
            q_comp = saturation * math.sin(angle)

            # YIQ -> RGB
            r = int((y + 0.956 * i_comp + 0.621 * q_comp) * 255 + 0.5)
            g = int((y - 0.272 * i_comp - 0.647 * q_comp) * 255 + 0.5)
            b = int((y - 1.107 * i_comp + 1.704 * q_comp) * 255 + 0.5)

            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

        palette[idx] = 0xFF000000 | (r << 16) | (g << 8) | b
    return palette


def _generate_pal_palette() -> List[int]:
    """Generate a 256-entry PAL ARGB palette.

    PAL uses alternating colour-burst phase on successive lines.  The lookup
    here provides the "average" colour seen on a real PAL display, where
    hue 0 and 1 are both grey and the remaining 14 hues are spaced evenly
    around the YUV colour wheel with a slightly different angle spacing
    than NTSC.
    """
    luma = [0.0, 0.1046, 0.2093, 0.3186, 0.4372, 0.5651, 0.7070, 0.8605]
    palette: List[int] = [0] * 256

    for idx in range(256):
        hue = (idx >> 4) & 0x0F
        lum = (idx >> 1) & 0x07
        y = luma[lum]

        if hue < 2:
            # Hues 0 and 1 are greyscale in PAL
            r = g = b = int(y * 255 + 0.5)
        else:
            angle = math.radians((hue - 2) * 25.7 - 15.0)
            saturation = 0.30 * (0.7 + y * 0.3)
            u_comp = saturation * math.cos(angle)
            v_comp = saturation * math.sin(angle)

            # YUV -> RGB
            r = int((y + 1.140 * v_comp) * 255 + 0.5)
            g = int((y - 0.396 * u_comp - 0.581 * v_comp) * 255 + 0.5)
            b = int((y + 2.029 * u_comp) * 255 + 0.5)

            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

        palette[idx] = 0xFF000000 | (r << 16) | (g << 8) | b
    return palette


# ---------------------------------------------------------------------------
# Module-level pre-computed tables
# ---------------------------------------------------------------------------

collision_mask: List[int] = _build_collision_mask()
"""Maps 6-bit TIACxFlags combination to TIACxPairFlags collision result."""

_pf_mask: List[List[int]] = _build_pf_mask()
_bl_mask: List[List[bool]] = _build_bl_mask()
_mx_mask: List[List[List[bool]]] = _build_mx_mask()
_px_mask: List[List[List[int]]] = _build_px_mask()

grp_reflect: List[int] = _build_grp_reflect()
"""Bit-reversed byte table (256 entries). ``grp_reflect[v]`` has the 8 bits
of *v* in reversed order."""

ntsc_palette: List[int] = _generate_ntsc_palette()
"""256-entry NTSC colour palette (0xAARRGGBB)."""

pal_palette: List[int] = _generate_pal_palette()
"""256-entry PAL colour palette (0xAARRGGBB)."""


# ---------------------------------------------------------------------------
# Public fetch helpers
# ---------------------------------------------------------------------------

def pf_mask_fetch(reflection_state: int, hpos: int) -> int:
    """Return playfield bit-mask for *hpos* (0-159).

    *reflection_state*: 0 = normal (right half repeats), 1 = reflected.
    The returned value is a single-bit mask into the packed PF register word
    ``PF0 | (PF1 << 8) | (PF2 << 16)``.
    """
    return _pf_mask[reflection_state][hpos]


def bl_mask_fetch(size: int, pos: int) -> bool:
    """Return ``True`` if the ball is enabled at *pos* clocks from reset.

    *size*: 0-3 (1, 2, 4, 8 pixels wide).
    """
    return _bl_mask[size][pos % 160]


def mx_mask_fetch(size: int, nusiz_type: int, pos: int) -> bool:
    """Return ``True`` if a missile pixel is enabled.

    *size*: 0-3 (1, 2, 4, 8 pixels wide).
    *nusiz_type*: 0-7 (copy/spacing mode from NUSIZx register).
    *pos*: offset from missile reset position (mod 160).
    """
    return _mx_mask[size][nusiz_type][pos % 160]


def px_mask_fetch(suppress: int, nusiz_type: int, pos: int) -> int:
    """Return a single-bit GRP mask for a player pixel.

    *suppress*: 1 to suppress all player output.
    *nusiz_type*: 0-7 (copy/spacing/stretch mode from NUSIZx).
    *pos*: offset from player reset position (mod 160).

    Returns 0x00-0x80 (one bit set) or 0 if nothing visible.
    """
    return _px_mask[suppress][nusiz_type][pos % 160]
