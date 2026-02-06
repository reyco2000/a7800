"""
TIA -- Television Interface Adaptor chip emulation.
Ported from the C# EMU7800 TIA implementation.

The TIA is the Atari 2600's custom graphics and sound chip.  It generates
the composite video signal by rendering five movable graphic objects
(two players, two missiles, one ball) on top of a 20-bit playfield,
handles collision detection between all object pairs, reads the console
input ports, and drives two independent audio channels.

Timing
------
The TIA colour clock runs at exactly 3x the CPU clock.  Each scanline is
228 colour clocks wide: 68 clocks of horizontal blank followed by 160
visible clocks.  NTSC frames are 262 scanlines; PAL frames are 312.

Rendering model
---------------
Rendering is performed lazily.  Every ``peek`` (register read) and
``poke`` (register write) first calls ``render_from_start_clock_to`` to
catch up the video output to the current CPU clock before reading or
modifying internal state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emu7800.core.types import TIACxFlags, TIACxPairFlags
from emu7800.core.sound.tia_sound import TIASound

if TYPE_CHECKING:
    pass  # forward refs for Machine, FrameBuffer, etc.

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_HBLANK = 68
_SCANLINE_CLOCKS = 228
_VISIBLE_PIXELS = 160

# Player copy offsets indexed by NUSIZ (bits 0-2)
_COPY_OFFSETS = (
    (0,),            # 0: one copy
    (0, 16),         # 1: two copies, close
    (0, 32),         # 2: two copies, medium
    (0, 16, 32),     # 3: three copies, close
    (0, 64),         # 4: two copies, wide
    (0,),            # 5: one copy, double-size
    (0, 32, 64),     # 6: three copies, medium
    (0,),            # 7: one copy, quad-size
)

# Player pixel width per copy
_PLAYER_WIDTH = (8, 8, 8, 8, 8, 16, 8, 32)

# How many screen pixels per graphic bit (stretch factor)
_PLAYER_STRETCH = (1, 1, 1, 1, 1, 2, 1, 4)

# ---------------------------------------------------------------------------
# Collision lookup table  (64 entries, indexed by TIACxFlags bitmask)
# ---------------------------------------------------------------------------

def _build_cx_table():
    """Pre-compute object-mask -> collision-pair-flags table."""
    table = [0] * 64
    for mask in range(64):
        pf = bool(mask & TIACxFlags.PF)
        bl = bool(mask & TIACxFlags.BL)
        m0 = bool(mask & TIACxFlags.M0)
        m1 = bool(mask & TIACxFlags.M1)
        p0 = bool(mask & TIACxFlags.P0)
        p1 = bool(mask & TIACxFlags.P1)
        cx = 0
        if m0 and p1: cx |= TIACxPairFlags.M0P1
        if m0 and p0: cx |= TIACxPairFlags.M0P0
        if m1 and p0: cx |= TIACxPairFlags.M1P0
        if m1 and p1: cx |= TIACxPairFlags.M1P1
        if p0 and pf: cx |= TIACxPairFlags.P0PF
        if p0 and bl: cx |= TIACxPairFlags.P0BL
        if p1 and pf: cx |= TIACxPairFlags.P1PF
        if p1 and bl: cx |= TIACxPairFlags.P1BL
        if m0 and pf: cx |= TIACxPairFlags.M0PF
        if m0 and bl: cx |= TIACxPairFlags.M0BL
        if m1 and pf: cx |= TIACxPairFlags.M1PF
        if m1 and bl: cx |= TIACxPairFlags.M1BL
        if bl and pf: cx |= TIACxPairFlags.BLPF
        if p0 and p1: cx |= TIACxPairFlags.P0P1
        if m0 and m1: cx |= TIACxPairFlags.M0M1
        table[mask] = cx
    return table


_CX_TABLE: list[int] = _build_cx_table()

# ---------------------------------------------------------------------------
# Signed-nibble helper (HMxx register upper nibble -> -8..+7)
# ---------------------------------------------------------------------------

def _signed_nibble(val: int) -> int:
    """Convert bits 7-4 of a register byte to a signed motion value -8..+7.

    Positive values move the object to the LEFT; negative to the RIGHT.
    """
    n = (val >> 4) & 0x0F
    return n - 16 if n >= 8 else n

# ---------------------------------------------------------------------------
# Object-pixel test helpers
# ---------------------------------------------------------------------------

def _pf_bit_on(pf20: int, pixel_x: int, reflect: bool) -> bool:
    """Return True when the playfield is set at visible pixel *pixel_x*.

    *pf20* is a 20-bit integer where bit 19 is the leftmost playfield pixel.
    The left half (pixels 0-79) always maps directly; the right half
    (pixels 80-159) is either a repeat or a reflection of the left half.
    """
    pf_idx = pixel_x >> 2  # 0..39  (each PF bit covers 4 pixels)
    if pf_idx < 20:
        return bool(pf20 & (1 << (19 - pf_idx)))
    else:
        if reflect:
            return bool(pf20 & (1 << (pf_idx - 20)))
        else:
            return bool(pf20 & (1 << (39 - pf_idx)))


def _player_pixel_on(pixel_x: int, pos: int, grp: int,
                     nusiz: int, reflected: bool) -> bool:
    """Return True when a player graphic is visible at *pixel_x*.

    *pos* is the player's horizontal position (0-159).
    *grp* is the 8-bit graphic pattern.
    *nusiz* bits 0-2 select copy count and stretch.
    *reflected* mirrors the graphic left-right.
    """
    if grp == 0:
        return False
    nt = nusiz & 0x07
    width = _PLAYER_WIDTH[nt]
    stretch = _PLAYER_STRETCH[nt]
    for off in _COPY_OFFSETS[nt]:
        d = (pixel_x - (pos + off)) % 160
        if d < width:
            bit = d // stretch
            if reflected:
                bit_idx = bit
            else:
                bit_idx = 7 - bit
            if grp & (1 << bit_idx):
                return True
    return False


def _missile_pixel_on(pixel_x: int, pos: int, nusiz: int, size: int) -> bool:
    """Return True when a missile is visible at *pixel_x*.

    Missiles use the same copy-offset table as the parent player (from
    NUSIZ bits 0-2).  *size* is the missile width in pixels (1/2/4/8,
    derived from NUSIZ bits 4-5).
    """
    nt = nusiz & 0x07
    # Missiles always use single-copy offsets for the multi-copy modes
    # but use the same copy pattern as their player.
    for off in _COPY_OFFSETS[nt]:
        d = (pixel_x - (pos + off)) % 160
        if d < size:
            return True
    return False


def _ball_pixel_on(pixel_x: int, pos: int, size: int) -> bool:
    """Return True when the ball is visible at *pixel_x*.

    *size* is the ball width (1/2/4/8, from CTRLPF bits 4-5).
    """
    d = (pixel_x - pos) % 160
    return d < size


# ===================================================================
# TIA class
# ===================================================================

class TIA:
    """Television Interface Adaptor (TIA) chip."""

    # ---- Write-register symbolic addresses (0x00 .. 0x2C) ----
    VSYNC  = 0x00; VBLANK = 0x01; WSYNC  = 0x02; RSYNC  = 0x03
    NUSIZ0 = 0x04; NUSIZ1 = 0x05; COLUP0 = 0x06; COLUP1 = 0x07
    COLUPF = 0x08; COLUBK = 0x09; CTRLPF = 0x0A; REFP0  = 0x0B
    REFP1  = 0x0C; PF0    = 0x0D; PF1    = 0x0E; PF2    = 0x0F
    RESP0  = 0x10; RESP1  = 0x11; RESM0  = 0x12; RESM1  = 0x13
    RESBL  = 0x14; AUDC0  = 0x15; AUDC1  = 0x16; AUDF0  = 0x17
    AUDF1  = 0x18; AUDV0  = 0x19; AUDV1  = 0x1A; GRP0   = 0x1B
    GRP1   = 0x1C; ENAM0  = 0x1D; ENAM1  = 0x1E; ENABL  = 0x1F
    HMP0   = 0x20; HMP1   = 0x21; HMM0   = 0x22; HMM1   = 0x23
    HMBL   = 0x24; VDELP0 = 0x25; VDELP1 = 0x26; VDELBL = 0x27
    RESMP0 = 0x28; RESMP1 = 0x29; HMOVE  = 0x2A; HMCLR  = 0x2B
    CXCLR  = 0x2C

    # ---- Read-register symbolic addresses (0x00 .. 0x0D) ----
    CXM0P  = 0x00; CXM1P  = 0x01; CXP0FB = 0x02; CXP1FB = 0x03
    CXM0FB = 0x04; CXM1FB = 0x05; CXBLPF = 0x06; CXPPMM = 0x07
    INPT0  = 0x08; INPT1  = 0x09; INPT2  = 0x0A; INPT3  = 0x0B
    INPT4  = 0x0C; INPT5  = 0x0D

    # ------------------------------------------------------------------
    # Construction / reset
    # ------------------------------------------------------------------

    def __init__(self, machine, scanlines: int):
        self.m = machine
        self.scanlines = scanlines

        # Write-register backing store (0x00..0x3F; upper mirrors ignored)
        self.reg_w: bytearray = bytearray(0x40)

        # Horizontal sync counter (0..227 per scanline)
        self.hsync: int = 0

        # Object position counters (0..159, visible pixel coordinates)
        self.p0: int = 0
        self.p1: int = 0
        self.m0: int = 0
        self.m1: int = 0
        self.bl: int = 0

        # Player graphics: current / old / effective
        self.grp0: int = 0
        self.grp1: int = 0
        self.old_grp0: int = 0
        self.old_grp1: int = 0
        self.eff_grp0: int = 0
        self.eff_grp1: int = 0

        # Ball enable: current / old / effective
        self.enabl_new: bool = False
        self.enabl_old: bool = False
        self.eff_enabl: bool = False

        # Missile enables
        self.enam0: bool = False
        self.enam1: bool = False

        # Decoded colours
        self.colubk: int = 0
        self.colupf: int = 0
        self.colup0: int = 0
        self.colup1: int = 0

        # Decoded CTRLPF
        self.pf_reflection: bool = False
        self.scoreon: bool = False
        self.pfpriority: bool = False
        self.ball_size: int = 1

        # Decoded NUSIZx
        self.missile0_size: int = 1
        self.missile1_size: int = 1

        # Decoded reflect flags
        self.refp0: bool = False
        self.refp1: bool = False

        # VDEL flags
        self.vdelp0: bool = False
        self.vdelp1: bool = False
        self.vdelbl: bool = False

        # RESMP flags
        self.resmp0: bool = False
        self.resmp1: bool = False

        # Display flags
        self.vblankon: bool = False
        self.vsync_enabled: bool = False

        # Playfield combined 20-bit register
        self.pf210: int = 0

        # Collision accumulator (TIACxPairFlags)
        self.collisions: int = 0

        # HMOVE state
        self.hmove_latch: bool = False
        self.start_hmove_clock: int = -1
        self.hmove_counter: int = 0
        self.hm_p0_latch: int = 0
        self.hm_p1_latch: int = 0
        self.hm_m0_latch: int = 0
        self.hm_m1_latch: int = 0
        self.hm_bl_latch: int = 0
        self.late_hmove_blank_on: bool = False

        # Timing
        self.start_clock: int = 0       # last-rendered frame-relative TIA clock
        self.frame_cpu_clock_start: int = 0
        self.wsync_delay_clocks: int = 0
        self.end_of_frame: bool = False

        # Input latches
        self.inpt_latch_enabled: bool = False   # VBLANK D6
        self.inpt_dump: bool = False             # VBLANK D7
        self.inpt4_latch: int = 0x80             # bit-7 = not-pressed
        self.inpt5_latch: int = 0x80

        # Sound
        self.tia_sound: TIASound = TIASound()

        # Frame buffer (set externally by the machine before first frame)
        self.frame_buffer = None

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Hard-reset all TIA state."""
        self.reg_w = bytearray(0x40)
        self.hsync = 0
        self.p0 = self.p1 = self.m0 = self.m1 = self.bl = 0
        self.grp0 = self.grp1 = self.old_grp0 = self.old_grp1 = 0
        self.eff_grp0 = self.eff_grp1 = 0
        self.enabl_new = self.enabl_old = self.eff_enabl = False
        self.enam0 = self.enam1 = False
        self.colubk = self.colupf = self.colup0 = self.colup1 = 0
        self.pf_reflection = self.scoreon = self.pfpriority = False
        self.ball_size = 1
        self.missile0_size = self.missile1_size = 1
        self.refp0 = self.refp1 = False
        self.vdelp0 = self.vdelp1 = self.vdelbl = False
        self.resmp0 = self.resmp1 = False
        self.vblankon = False
        self.vsync_enabled = False
        self.pf210 = 0
        self.collisions = 0
        self.hmove_latch = False
        self.start_hmove_clock = -1
        self.hmove_counter = 0
        self.hm_p0_latch = self.hm_p1_latch = 0
        self.hm_m0_latch = self.hm_m1_latch = self.hm_bl_latch = 0
        self.late_hmove_blank_on = False
        self.start_clock = 0
        self.frame_cpu_clock_start = 0
        self.wsync_delay_clocks = 0
        self.end_of_frame = False
        self.inpt_latch_enabled = False
        self.inpt_dump = False
        self.inpt4_latch = 0x80
        self.inpt5_latch = 0x80
        self.tia_sound.reset()

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def start_of_frame(self) -> None:
        """Prepare for a new display frame."""
        self.frame_cpu_clock_start = self.m.cpu.clock
        self.start_clock = 0
        self.end_of_frame = False
        self.wsync_delay_clocks = 0

    def end_of_scanline(self) -> None:
        """Called internally at each hsync wrap (clock 228)."""
        self.hsync = 0
        # HMOVE blank latch is valid only for the line on which HMOVE fires;
        # it is cleared at the start of the next line.
        self.late_hmove_blank_on = False

    # ------------------------------------------------------------------
    # Current-clock helpers
    # ------------------------------------------------------------------

    def _tia_clock(self) -> int:
        """Frame-relative TIA colour clock derived from the CPU clock."""
        return (self.m.cpu.clock - self.frame_cpu_clock_start) * 3

    def _render_to_current_clock(self) -> None:
        self.render_from_start_clock_to(self._tia_clock())

    # ------------------------------------------------------------------
    # Playfield construction
    # ------------------------------------------------------------------

    def _update_pf(self) -> None:
        """Rebuild the 20-bit playfield word from PF0, PF1, PF2."""
        pf0 = self.reg_w[self.PF0]
        pf1 = self.reg_w[self.PF1]
        pf2 = self.reg_w[self.PF2]

        word = 0
        # PF0 bits 4,5,6,7 -> playfield positions 0-3 (left to right)
        for i in range(4):
            if pf0 & (1 << (4 + i)):
                word |= (1 << (19 - i))
        # PF1 bits 7,6,...,0 -> positions 4-11
        for i in range(8):
            if pf1 & (1 << (7 - i)):
                word |= (1 << (15 - i))
        # PF2 bits 0,1,...,7 -> positions 12-19
        for i in range(8):
            if pf2 & (1 << i):
                word |= (1 << (7 - i))

        self.pf210 = word

    # ------------------------------------------------------------------
    # Effective graphics helpers (VDEL)
    # ------------------------------------------------------------------

    def _update_eff_grp0(self) -> None:
        self.eff_grp0 = self.old_grp0 if self.vdelp0 else self.grp0

    def _update_eff_grp1(self) -> None:
        self.eff_grp1 = self.old_grp1 if self.vdelp1 else self.grp1

    def _update_eff_enabl(self) -> None:
        self.eff_enabl = self.enabl_old if self.vdelbl else self.enabl_new

    # ------------------------------------------------------------------
    # Position reset helpers
    # ------------------------------------------------------------------

    def _object_reset_pos(self) -> int:
        """Compute the screen-pixel position for a RESPx/RESMx/RESBL strobe.

        Returns a value in [0..159].  During HBLANK the position is clamped
        to 3 (the leftmost visible result after the internal 4-5 clock
        pipeline delay).  During the visible region the position is
        ``(hsync - 68 + 5) % 160``, modelling the ~5-clock reset delay.
        """
        if self.hsync < _HBLANK:
            return 3
        return (self.hsync - _HBLANK + 5) % 160

    # ------------------------------------------------------------------
    # Core rendering engine
    # ------------------------------------------------------------------

    def render_from_start_clock_to(self, end_clock: int) -> None:
        """Advance TIA output from *start_clock* to *end_clock*.

        Both values are frame-relative TIA colour clocks.  Every colour
        clock: the hsync counter is incremented, HMOVE motion is applied
        if pending, and -- in the visible region -- a pixel colour is
        determined and written to the frame buffer.
        """
        if end_clock <= self.start_clock:
            return

        # Clamp to the end of the frame
        max_clock = self.scanlines * _SCANLINE_CLOCKS
        if end_clock > max_clock:
            end_clock = max_clock

        fb = self.frame_buffer
        if fb is None:
            self.start_clock = end_clock
            return

        video_buffer = fb.video_buffer
        buf_len = len(video_buffer)

        # Cache instance attributes in locals for the hot loop
        hsync = self.hsync
        p0 = self.p0; p1 = self.p1
        m0 = self.m0; m1 = self.m1; bl = self.bl

        colubk = self.colubk; colupf = self.colupf
        colup0 = self.colup0; colup1 = self.colup1
        vblankon = self.vblankon
        pf_reflect = self.pf_reflection
        scoreon = self.scoreon
        pfpriority = self.pfpriority
        pf210 = self.pf210
        ball_size = self.ball_size
        m0_size = self.missile0_size
        m1_size = self.missile1_size

        eff_grp0 = self.eff_grp0; eff_grp1 = self.eff_grp1
        refp0 = self.refp0; refp1 = self.refp1
        nusiz0 = self.reg_w[self.NUSIZ0]
        nusiz1 = self.reg_w[self.NUSIZ1]

        enam0 = self.enam0; enam1 = self.enam1
        eff_enabl = self.eff_enabl

        resmp0 = self.resmp0; resmp1 = self.resmp1

        collisions = self.collisions

        hmove_latch = self.hmove_latch
        late_blank = self.late_hmove_blank_on
        start_hmove_clock = self.start_hmove_clock
        hmove_counter = self.hmove_counter
        hm_p0 = self.hm_p0_latch
        hm_p1 = self.hm_p1_latch
        hm_m0 = self.hm_m0_latch
        hm_m1 = self.hm_m1_latch
        hm_bl = self.hm_bl_latch

        clock = self.start_clock

        while clock < end_clock:
            # ---- scanline wrap ----
            if hsync >= _SCANLINE_CLOCKS:
                hsync = 0
                late_blank = False

            # ---- HMOVE processing ----
            # The HMOVE mechanism processes one comparison per colour clock
            # for 15 clocks after the strobe.  Each object whose motion
            # requirement has not yet been fulfilled is moved one position.
            if hmove_latch and hmove_counter < 16 and start_hmove_clock >= 0:
                elapsed = clock - start_hmove_clock
                if 0 < elapsed <= 16:
                    cmp_val = 15 - (elapsed - 1)  # 15 down to 0
                    # Move objects whose |HMxx| > (15 - cmp_val)
                    # Simplified: object moves while counter < motionReq
                    # motionReq = (HMxx ^ 8) where HMxx is the unsigned
                    # upper-nibble (0..15).  An object moves one clock left
                    # for each comparison where cmp_val >= 0 AND
                    # the compare counter hasn't matched yet.
                    # For correctness we just check if the counter is still
                    # above the XOR-ed threshold.
                    if cmp_val < hm_p0:
                        p0 = (p0 - 1) % 160
                    if cmp_val < hm_p1:
                        p1 = (p1 - 1) % 160
                    if cmp_val < hm_m0:
                        m0 = (m0 - 1) % 160
                    if cmp_val < hm_m1:
                        m1 = (m1 - 1) % 160
                    if cmp_val < hm_bl:
                        bl = (bl - 1) % 160
                if elapsed >= 16:
                    hmove_latch = False
                    start_hmove_clock = -1

            # ---- visible region ----
            if hsync >= _HBLANK:
                pixel_x = hsync - _HBLANK  # 0..159
                scanline = clock // _SCANLINE_CLOCKS

                fb_idx = scanline * _VISIBLE_PIXELS + pixel_x

                if vblankon:
                    # VBLANK is active: emit black (colour 0)
                    if fb_idx < buf_len:
                        video_buffer[fb_idx] = 0
                elif late_blank and pixel_x < 8:
                    # HMOVE blank region: forced background
                    if fb_idx < buf_len:
                        video_buffer[fb_idx] = colubk
                else:
                    # --- missile position lock ---
                    m0x = m0 if not resmp0 else (p0 + 4) % 160
                    m1x = m1 if not resmp1 else (p1 + 4) % 160

                    # --- determine which objects are present ---
                    cx = 0

                    if _pf_bit_on(pf210, pixel_x, pf_reflect):
                        cx |= TIACxFlags.PF

                    if eff_enabl and _ball_pixel_on(pixel_x, bl, ball_size):
                        cx |= TIACxFlags.BL

                    if enam0 and (not resmp0) and \
                       _missile_pixel_on(pixel_x, m0x, nusiz0, m0_size):
                        cx |= TIACxFlags.M0

                    if enam1 and (not resmp1) and \
                       _missile_pixel_on(pixel_x, m1x, nusiz1, m1_size):
                        cx |= TIACxFlags.M1

                    if _player_pixel_on(pixel_x, p0, eff_grp0,
                                        nusiz0, refp0):
                        cx |= TIACxFlags.P0

                    if _player_pixel_on(pixel_x, p1, eff_grp1,
                                        nusiz1, refp1):
                        cx |= TIACxFlags.P1

                    # --- collisions ---
                    collisions |= _CX_TABLE[cx]

                    # --- colour priority ---
                    pfbl = cx & (TIACxFlags.PF | TIACxFlags.BL)
                    p0m0 = cx & (TIACxFlags.P0 | TIACxFlags.M0)
                    p1m1 = cx & (TIACxFlags.P1 | TIACxFlags.M1)

                    if pfpriority:
                        if pfbl:
                            if scoreon and (cx & TIACxFlags.PF):
                                colour = colup0 if pixel_x < 80 else colup1
                            else:
                                colour = colupf
                        elif p0m0:
                            colour = colup0
                        elif p1m1:
                            colour = colup1
                        else:
                            colour = colubk
                    else:
                        if p0m0:
                            colour = colup0
                        elif p1m1:
                            colour = colup1
                        elif pfbl:
                            if scoreon and (cx & TIACxFlags.PF):
                                colour = colup0 if pixel_x < 80 else colup1
                            else:
                                colour = colupf
                        else:
                            colour = colubk

                    if fb_idx < buf_len:
                        video_buffer[fb_idx] = colour

            hsync += 1
            clock += 1

        # ---- write locals back ----
        self.hsync = hsync
        self.p0 = p0; self.p1 = p1
        self.m0 = m0; self.m1 = m1; self.bl = bl
        self.collisions = collisions
        self.hmove_latch = hmove_latch
        self.late_hmove_blank_on = late_blank
        self.start_hmove_clock = start_hmove_clock
        self.hmove_counter = hmove_counter
        self.start_clock = end_clock

        # End-of-frame detection
        if end_clock >= max_clock:
            self.end_of_frame = True

    # ==================================================================
    # Peek -- register read  (addresses 0x00 .. 0x0D)
    # ==================================================================

    def peek(self, addr: int) -> int:
        """Read a TIA register.  Only bits 7-6 (collision) or bit 7
        (input ports) are valid; the remaining bits are open bus.

        *addr* is the raw bus address; only the low 4 bits are used.
        """
        self._render_to_current_clock()

        reg = addr & 0x0F

        # ---- Collision registers (0x00 .. 0x07) ----
        if reg == self.CXM0P:   # 0x00
            val = 0
            if self.collisions & TIACxPairFlags.M0P1:
                val |= 0x80
            if self.collisions & TIACxPairFlags.M0P0:
                val |= 0x40
            return val

        if reg == self.CXM1P:   # 0x01
            val = 0
            if self.collisions & TIACxPairFlags.M1P0:
                val |= 0x80
            if self.collisions & TIACxPairFlags.M1P1:
                val |= 0x40
            return val

        if reg == self.CXP0FB:  # 0x02
            val = 0
            if self.collisions & TIACxPairFlags.P0PF:
                val |= 0x80
            if self.collisions & TIACxPairFlags.P0BL:
                val |= 0x40
            return val

        if reg == self.CXP1FB:  # 0x03
            val = 0
            if self.collisions & TIACxPairFlags.P1PF:
                val |= 0x80
            if self.collisions & TIACxPairFlags.P1BL:
                val |= 0x40
            return val

        if reg == self.CXM0FB:  # 0x04
            val = 0
            if self.collisions & TIACxPairFlags.M0PF:
                val |= 0x80
            if self.collisions & TIACxPairFlags.M0BL:
                val |= 0x40
            return val

        if reg == self.CXM1FB:  # 0x05
            val = 0
            if self.collisions & TIACxPairFlags.M1PF:
                val |= 0x80
            if self.collisions & TIACxPairFlags.M1BL:
                val |= 0x40
            return val

        if reg == self.CXBLPF:  # 0x06
            val = 0
            if self.collisions & TIACxPairFlags.BLPF:
                val |= 0x80
            # D6 is unused (always 0)
            return val

        if reg == self.CXPPMM:  # 0x07
            val = 0
            if self.collisions & TIACxPairFlags.P0P1:
                val |= 0x80
            if self.collisions & TIACxPairFlags.M0M1:
                val |= 0x40
            return val

        # ---- Analog input ports (0x08 .. 0x0B) ----
        if reg == self.INPT0:   # 0x08
            return self._read_paddle_port(0)
        if reg == self.INPT1:   # 0x09
            return self._read_paddle_port(1)
        if reg == self.INPT2:   # 0x0A
            return self._read_paddle_port(2)
        if reg == self.INPT3:   # 0x0B
            return self._read_paddle_port(3)

        # ---- Digital input ports (0x0C .. 0x0D) ----
        if reg == self.INPT4:   # 0x0C
            return self._read_digital_port(0)
        if reg == self.INPT5:   # 0x0D
            return self._read_digital_port(1)

        # Addresses 0x0E..0x0F mirror 0x06..0x07 on real hardware,
        # but we already masked with 0x0F so they can't appear here.
        return 0

    # ------------------------------------------------------------------
    # Input-port helpers
    # ------------------------------------------------------------------

    def _read_paddle_port(self, port: int) -> int:
        """Read INPT0-INPT3 (analog paddle ports).

        When VBLANK D7 (dump) is set the capacitor is grounded and the
        port always reads low (0x00).  Otherwise the port reads 0x80
        after a timing delay proportional to the paddle position.
        """
        if self.inpt_dump:
            return 0x00
        # Simplified: always return 0x80 (charged) when not dumping.
        # A full implementation would integrate the RC charge curve
        # using the paddle position from the input adapter.
        return 0x80

    def _read_digital_port(self, port: int) -> int:
        """Read INPT4 or INPT5 (trigger / keypad).

        In latch mode (VBLANK D6) the input is latched on the low-going
        transition and stays low until the latch is re-enabled.
        """
        # Read the raw trigger state from the machine's input adapter.
        # Bit 7: 0x80 = not pressed, 0x00 = pressed.
        raw = self._raw_trigger(port)

        if self.inpt_latch_enabled:
            # Latch mode: once the button is pressed (raw bit7 == 0)
            # the latch is cleared and stays cleared until VBLANK D6 is
            # re-asserted (which re-arms the latch to 0x80).
            if port == 0:
                self.inpt4_latch &= raw
                return self.inpt4_latch
            else:
                self.inpt5_latch &= raw
                return self.inpt5_latch
        return raw

    def _raw_trigger(self, port: int) -> int:
        """Return 0x80 (not pressed) or 0x00 (pressed) for trigger *port*.

        *port* 0 = left controller, 1 = right controller.
        """
        m = self.m
        if not hasattr(m, 'input_state'):
            return 0x80
        inp = m.input_state
        if inp is None:
            return 0x80

        # Check for a keypad scan (PIA SWCHA column selection)
        if hasattr(m, 'pia') and m.pia is not None:
            swcha = m.pia.read_output_port_a() if hasattr(m.pia, 'read_output_port_a') else 0xFF
            # Keypad scanning: the PIA drives columns low one at a time.
            # If any column is driven low we are in keypad mode.
            if port == 0 and (swcha & 0xF0) != 0xF0:
                return self._keypad_scan(port, swcha)
            if port == 1 and (swcha & 0x0F) != 0x0F:
                return self._keypad_scan(port, swcha)

        # Normal trigger read
        if hasattr(inp, 'get_trigger'):
            return 0x00 if inp.get_trigger(port) else 0x80
        return 0x80

    def _keypad_scan(self, port: int, swcha: int) -> int:
        """Handle Atari keypad matrix scanning for INPT4/INPT5."""
        # Simplified keypad scan stub.  A full implementation reads the
        # PIA output lines (SWCHA) to determine which column is being
        # scanned and returns the appropriate row bit for INPT4/INPT5.
        return 0x80

    # ==================================================================
    # Poke -- register write  (addresses 0x00 .. 0x2C)
    # ==================================================================

    def poke(self, addr: int, val: int) -> None:
        """Write a TIA register.

        *addr* is the raw bus address; only the low 6 bits are used.
        """
        reg = addr & 0x3F

        # Store raw value in the register file
        self.reg_w[reg] = val

        # Most writes require catching up rendering first.  A few
        # (GRP0, GRP1, PF regs) add a small delay so the write takes
        # effect one pixel later -- this is modelled by rendering to
        # ``current_clock + delay`` before applying the state change.

        # ---- VSYNC (0x00) ----
        if reg == self.VSYNC:
            self._render_to_current_clock()
            self.vsync_enabled = bool(val & 0x02)
            return

        # ---- VBLANK (0x01) ----
        if reg == self.VBLANK:
            self._render_to_current_clock()
            self.vblankon = bool(val & 0x02)
            # D6: latch enable for INPT4-5
            if val & 0x40:
                self.inpt_latch_enabled = True
                # Re-arm latches
                self.inpt4_latch = 0x80
                self.inpt5_latch = 0x80
            else:
                self.inpt_latch_enabled = False
            # D7: dump paddle caps
            self.inpt_dump = bool(val & 0x80)
            return

        # ---- WSYNC (0x02) ----
        if reg == self.WSYNC:
            self._render_to_current_clock()
            # Halt CPU until end of current scanline
            tia_clock = self._tia_clock()
            hsync_pos = tia_clock % _SCANLINE_CLOCKS
            if hsync_pos > 0:
                remaining = _SCANLINE_CLOCKS - hsync_pos
                cpu_delay = (remaining + 2) // 3
                self.m.cpu.clock += cpu_delay
            return

        # ---- RSYNC (0x03) ----
        if reg == self.RSYNC:
            self._render_to_current_clock()
            # Reset the horizontal sync counter.  Rarely used by games.
            self.hsync = 0
            return

        # ---- NUSIZ0 (0x04) ----
        if reg == self.NUSIZ0:
            self._render_to_current_clock()
            self.missile0_size = 1 << ((val >> 4) & 0x03)
            return

        # ---- NUSIZ1 (0x05) ----
        if reg == self.NUSIZ1:
            self._render_to_current_clock()
            self.missile1_size = 1 << ((val >> 4) & 0x03)
            return

        # ---- COLUP0 (0x06) ----
        if reg == self.COLUP0:
            self._render_to_current_clock()
            self.colup0 = val & 0xFE
            return

        # ---- COLUP1 (0x07) ----
        if reg == self.COLUP1:
            self._render_to_current_clock()
            self.colup1 = val & 0xFE
            return

        # ---- COLUPF (0x08) ----
        if reg == self.COLUPF:
            self._render_to_current_clock()
            self.colupf = val & 0xFE
            return

        # ---- COLUBK (0x09) ----
        if reg == self.COLUBK:
            self._render_to_current_clock()
            self.colubk = val & 0xFE
            return

        # ---- CTRLPF (0x0A) ----
        if reg == self.CTRLPF:
            self._render_to_current_clock()
            self.pf_reflection = bool(val & 0x01)
            self.scoreon = bool(val & 0x02)
            self.pfpriority = bool(val & 0x04)
            self.ball_size = 1 << ((val >> 4) & 0x03)
            return

        # ---- REFP0 (0x0B) ----
        if reg == self.REFP0:
            self._render_to_current_clock()
            self.refp0 = bool(val & 0x08)
            return

        # ---- REFP1 (0x0C) ----
        if reg == self.REFP1:
            self._render_to_current_clock()
            self.refp1 = bool(val & 0x08)
            return

        # ---- PF0 (0x0D) ----
        if reg == self.PF0:
            # PF updates have a 1-clock delay
            self.render_from_start_clock_to(self._tia_clock() + 1)
            self.reg_w[self.PF0] = val
            self._update_pf()
            return

        # ---- PF1 (0x0E) ----
        if reg == self.PF1:
            self.render_from_start_clock_to(self._tia_clock() + 1)
            self.reg_w[self.PF1] = val
            self._update_pf()
            return

        # ---- PF2 (0x0F) ----
        if reg == self.PF2:
            self.render_from_start_clock_to(self._tia_clock() + 1)
            self.reg_w[self.PF2] = val
            self._update_pf()
            return

        # ---- RESP0 (0x10) ----
        if reg == self.RESP0:
            self._render_to_current_clock()
            self.p0 = self._object_reset_pos()
            return

        # ---- RESP1 (0x11) ----
        if reg == self.RESP1:
            self._render_to_current_clock()
            self.p1 = self._object_reset_pos()
            return

        # ---- RESM0 (0x12) ----
        if reg == self.RESM0:
            self._render_to_current_clock()
            self.m0 = self._object_reset_pos()
            return

        # ---- RESM1 (0x13) ----
        if reg == self.RESM1:
            self._render_to_current_clock()
            self.m1 = self._object_reset_pos()
            return

        # ---- RESBL (0x14) ----
        if reg == self.RESBL:
            self._render_to_current_clock()
            self.bl = self._object_reset_pos()
            return

        # ---- Audio registers (0x15..0x1A) ----
        if 0x15 <= reg <= 0x1A:
            self._render_to_current_clock()
            self.tia_sound.update(reg, val)
            return

        # ---- GRP0 (0x1B) ----
        if reg == self.GRP0:
            self.render_from_start_clock_to(self._tia_clock() + 1)
            # Latch old GRP1
            self.old_grp1 = self.grp1
            self.grp0 = val
            self._update_eff_grp0()
            self._update_eff_grp1()
            return

        # ---- GRP1 (0x1C) ----
        if reg == self.GRP1:
            self.render_from_start_clock_to(self._tia_clock() + 1)
            # Latch old GRP0 and old ENABL
            self.old_grp0 = self.grp0
            self.enabl_old = self.enabl_new
            self.grp1 = val
            self._update_eff_grp0()
            self._update_eff_grp1()
            self._update_eff_enabl()
            return

        # ---- ENAM0 (0x1D) ----
        if reg == self.ENAM0:
            self._render_to_current_clock()
            self.enam0 = bool(val & 0x02)
            return

        # ---- ENAM1 (0x1E) ----
        if reg == self.ENAM1:
            self._render_to_current_clock()
            self.enam1 = bool(val & 0x02)
            return

        # ---- ENABL (0x1F) ----
        if reg == self.ENABL:
            self._render_to_current_clock()
            self.enabl_new = bool(val & 0x02)
            self._update_eff_enabl()
            return

        # ---- HMP0 (0x20) ----
        if reg == self.HMP0:
            self._render_to_current_clock()
            return

        # ---- HMP1 (0x21) ----
        if reg == self.HMP1:
            self._render_to_current_clock()
            return

        # ---- HMM0 (0x22) ----
        if reg == self.HMM0:
            self._render_to_current_clock()
            return

        # ---- HMM1 (0x23) ----
        if reg == self.HMM1:
            self._render_to_current_clock()
            return

        # ---- HMBL (0x24) ----
        if reg == self.HMBL:
            self._render_to_current_clock()
            return

        # ---- VDELP0 (0x25) ----
        if reg == self.VDELP0:
            self._render_to_current_clock()
            self.vdelp0 = bool(val & 0x01)
            self._update_eff_grp0()
            return

        # ---- VDELP1 (0x26) ----
        if reg == self.VDELP1:
            self._render_to_current_clock()
            self.vdelp1 = bool(val & 0x01)
            self._update_eff_grp1()
            return

        # ---- VDELBL (0x27) ----
        if reg == self.VDELBL:
            self._render_to_current_clock()
            self.vdelbl = bool(val & 0x01)
            self._update_eff_enabl()
            return

        # ---- RESMP0 (0x28) ----
        if reg == self.RESMP0:
            self._render_to_current_clock()
            old_resmp0 = self.resmp0
            self.resmp0 = bool(val & 0x02)
            # When RESMP0 transitions from set to clear, missile keeps
            # its last locked position.
            if self.resmp0:
                # Lock missile to player centre
                self.m0 = (self.p0 + 4) % 160
            return

        # ---- RESMP1 (0x29) ----
        if reg == self.RESMP1:
            self._render_to_current_clock()
            old_resmp1 = self.resmp1
            self.resmp1 = bool(val & 0x02)
            if self.resmp1:
                self.m1 = (self.p1 + 4) % 160
            return

        # ---- HMOVE (0x2A) ----
        if reg == self.HMOVE:
            self._render_to_current_clock()
            # Latch motion values (upper nibble XOR 8 gives the
            # comparison threshold used by the hardware counter).
            self.hm_p0_latch = ((self.reg_w[self.HMP0] >> 4) & 0x0F) ^ 0x08
            self.hm_p1_latch = ((self.reg_w[self.HMP1] >> 4) & 0x0F) ^ 0x08
            self.hm_m0_latch = ((self.reg_w[self.HMM0] >> 4) & 0x0F) ^ 0x08
            self.hm_m1_latch = ((self.reg_w[self.HMM1] >> 4) & 0x0F) ^ 0x08
            self.hm_bl_latch = ((self.reg_w[self.HMBL] >> 4) & 0x0F) ^ 0x08
            self.start_hmove_clock = self._tia_clock()
            self.hmove_latch = True
            self.hmove_counter = 0
            # Enable the 8-clock late-HBLANK blanking
            self.late_hmove_blank_on = True
            return

        # ---- HMCLR (0x2B) ----
        if reg == self.HMCLR:
            self._render_to_current_clock()
            self.reg_w[self.HMP0] = 0
            self.reg_w[self.HMP1] = 0
            self.reg_w[self.HMM0] = 0
            self.reg_w[self.HMM1] = 0
            self.reg_w[self.HMBL] = 0
            return

        # ---- CXCLR (0x2C) ----
        if reg == self.CXCLR:
            self._render_to_current_clock()
            self.collisions = 0
            return

        # Registers 0x2D..0x3F: unmapped / mirrors.  Store in reg_w
        # (already done above) but take no further action.

    # ------------------------------------------------------------------
    # Convenience aliases used by the address space mapper
    # ------------------------------------------------------------------

    def __getitem__(self, addr: int) -> int:
        return self.peek(addr)

    def __setitem__(self, addr: int, val: int) -> None:
        self.poke(addr, val)
