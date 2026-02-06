"""
TIA sound emulation -- two-channel audio based on Ron Fries' TIASound.

Each channel has three registers:
  AUDC  (4 bits) -- audio control / distortion selector
  AUDF  (5 bits) -- frequency divider
  AUDV  (4 bits) -- volume (0-15)

Sound is generated through polynomial counters (4-bit, 5-bit, 9-bit)
combined with frequency dividers.  The base TIA audio clock is approximately
31.4 kHz (NTSC CPU clock / 114).

AUDC distortion modes:
  0x00       Set to 1 (volume only)
  0x01       4-bit poly
  0x02       div31 -> 4-bit poly
  0x03       5-bit poly -> 4-bit poly
  0x04/0x05  Pure tone (div2 toggle)
  0x06/0x0A  div31 -> pure tone
  0x07       5-bit poly -> pure tone
  0x08       9-bit poly (white noise)
  0x09       5-bit poly
  0x0B       Set to 1 (volume only)
  0x0C/0x0D  div6 (div3 -> toggle)
  0x0E       div93 (div31 -> div3 -> toggle)
  0x0F       5-bit poly -> div6

TIA register addresses used by :meth:`update`:
  0x15 AUDC0   0x16 AUDC1
  0x17 AUDF0   0x18 AUDF1
  0x19 AUDV0   0x1A AUDV1

Ported from EMU7800 C# TIASound (originally by Ron Fries).
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Polynomial counter sizes
# ---------------------------------------------------------------------------
POLY4_SIZE: int = 15
POLY5_SIZE: int = 31
POLY9_SIZE: int = 511

# TIA audio register addresses
AUDC0: int = 0x15
AUDC1: int = 0x16
AUDF0: int = 0x17
AUDF1: int = 0x18
AUDV0: int = 0x19
AUDV1: int = 0x1A


# ---------------------------------------------------------------------------
# Polynomial table generation
# ---------------------------------------------------------------------------

def _generate_lfsr(bits: int, tap_a: int, tap_b: int) -> List[int]:
    """Return the full period of an LFSR as a list of 0/1 values.

    Uses a Fibonacci LFSR: ``feedback = bit[tap_a] XOR bit[tap_b]``.
    """
    size = (1 << bits) - 1
    reg = (1 << bits) - 1  # seed with all-ones
    seq: List[int] = []
    for _ in range(size):
        seq.append(reg & 1)
        feedback = ((reg >> tap_a) ^ (reg >> tap_b)) & 1
        reg = (reg >> 1) | (feedback << (bits - 1))
    return seq


def _generate_div31() -> List[int]:
    """Generate a div-by-31 pulse table.

    Exactly one element is 1 (the pulse); the rest are 0.  Indexed by
    the poly-5 counter position to implement a divide-by-31 gate.
    """
    table = [0] * POLY5_SIZE
    table[0] = 1
    return table


# Pre-generated polynomial tables (module-level constants).
_POLY4: List[int] = _generate_lfsr(4, 0, 1)   # x^4 + x + 1,  period 15
_POLY5: List[int] = _generate_lfsr(5, 0, 2)   # x^5 + x^2 + 1, period 31
_POLY9: List[int] = _generate_lfsr(9, 0, 4)   # x^9 + x^4 + 1, period 511
_DIV31: List[int] = _generate_div31()          # pulse every 31 ticks


class TIASound:
    """Two-channel TIA sound generator.

    Usage::

        snd = TIASound()
        snd.start_frame()
        # ... for each register write during the frame ...
        snd.update(addr, data)
        # ... at end of frame ...
        snd.render_samples(samples_per_frame)
        pcm = snd.end_frame()
    """

    # ----- construction / reset -------------------------------------------

    def __init__(self) -> None:
        # Polynomial counter positions (shared between both channels)
        self._p4: int = 0
        self._p5: int = 0
        self._p9: int = 0

        # Per-channel registers
        self._audc: List[int] = [0, 0]
        self._audf: List[int] = [0, 0]
        self._audv: List[int] = [0, 0]

        # Per-channel divider state
        self._div_n_cnt: List[int] = [0, 0]
        self._div_n_max: List[int] = [0, 0]

        # Per-channel output toggle state (0 or 1)
        self._output: List[int] = [0, 0]
        # Per-channel current output volume (0..15)
        self._outvol: List[int] = [0, 0]

        # Frame buffer (filled by render_samples, read by end_frame)
        self._buffer: List[int] = []
        self._buffer_index: int = 0

    def reset(self) -> None:
        """Reset all state to power-on defaults."""
        self._p4 = 0
        self._p5 = 0
        self._p9 = 0
        self._audc[:] = [0, 0]
        self._audf[:] = [0, 0]
        self._audv[:] = [0, 0]
        self._div_n_cnt[:] = [0, 0]
        self._div_n_max[:] = [0, 0]
        self._output[:] = [0, 0]
        self._outvol[:] = [0, 0]
        self._buffer.clear()
        self._buffer_index = 0

    # ----- register writes -----------------------------------------------

    def update(self, addr: int, data: int) -> None:
        """Process a TIA audio register write.

        Parameters
        ----------
        addr : int
            Register address (0x15-0x1A).
        data : int
            Byte value written.
        """
        addr &= 0xFF
        data &= 0xFF

        if addr == AUDC0:
            self._audc[0] = data & 0x0F
            self._recalc(0)
        elif addr == AUDC1:
            self._audc[1] = data & 0x0F
            self._recalc(1)
        elif addr == AUDF0:
            self._audf[0] = data & 0x1F
            self._recalc(0)
        elif addr == AUDF1:
            self._audf[1] = data & 0x1F
            self._recalc(1)
        elif addr == AUDV0:
            self._audv[0] = data & 0x0F
            # In volume-only mode the output tracks AUDV immediately
            if self._div_n_max[0] == 0:
                self._outvol[0] = self._audv[0]
        elif addr == AUDV1:
            self._audv[1] = data & 0x0F
            if self._div_n_max[1] == 0:
                self._outvol[1] = self._audv[1]

    # ----- frame lifecycle -----------------------------------------------

    def start_frame(self) -> None:
        """Prepare internal buffer for a new video frame."""
        self._buffer_index = 0

    def render_samples(self, count: int) -> None:
        """Generate *count* PCM samples into the internal frame buffer.

        Each sample represents one TIA audio clock tick (~31.4 kHz for
        NTSC, ~31.2 kHz for PAL).  The output value per sample is in the
        range 0..30 (two channels mixed, each contributing 0..15).
        """
        # Ensure the buffer has enough room
        needed = self._buffer_index + count
        if len(self._buffer) < needed:
            self._buffer.extend(0 for _ in range(needed - len(self._buffer)))

        # Pull state into locals for speed
        p4 = self._p4
        p5 = self._p5
        p9 = self._p9

        audc = self._audc
        audv = self._audv
        div_cnt = self._div_n_cnt
        div_max = self._div_n_max
        output = self._output
        outvol = self._outvol
        buf = self._buffer
        bi = self._buffer_index

        for _ in range(count):
            # ----- advance shared polynomial counters -----
            p4 = p4 + 1
            if p4 >= POLY4_SIZE:
                p4 = 0
            p5 = p5 + 1
            if p5 >= POLY5_SIZE:
                p5 = 0
            p9 = p9 + 1
            if p9 >= POLY9_SIZE:
                p9 = 0

            # ----- process channel 0 -----
            if div_max[0] != 0:
                div_cnt[0] -= 1
                if div_cnt[0] <= 0:
                    div_cnt[0] = div_max[0]
                    ac = audc[0]

                    if ac <= 0x00:
                        outvol[0] = audv[0]
                    elif ac == 0x01:
                        outvol[0] = audv[0] if _POLY4[p4] else 0
                    elif ac == 0x02:
                        outvol[0] = audv[0] if _POLY4[p4] else 0
                    elif ac == 0x03:
                        if _POLY5[p5]:
                            outvol[0] = audv[0] if _POLY4[p4] else 0
                    elif ac == 0x04 or ac == 0x05:
                        output[0] ^= 1
                        outvol[0] = audv[0] if output[0] else 0
                    elif ac == 0x06:
                        output[0] ^= 1
                        outvol[0] = audv[0] if output[0] else 0
                    elif ac == 0x07:
                        if _POLY5[p5]:
                            output[0] ^= 1
                            outvol[0] = audv[0] if output[0] else 0
                    elif ac == 0x08:
                        outvol[0] = audv[0] if _POLY9[p9] else 0
                    elif ac == 0x09:
                        outvol[0] = audv[0] if _POLY5[p5] else 0
                    elif ac == 0x0A:
                        output[0] ^= 1
                        outvol[0] = audv[0] if output[0] else 0
                    elif ac == 0x0B:
                        outvol[0] = audv[0]
                    elif ac == 0x0C or ac == 0x0D:
                        output[0] ^= 1
                        outvol[0] = audv[0] if output[0] else 0
                    elif ac == 0x0E:
                        output[0] ^= 1
                        outvol[0] = audv[0] if output[0] else 0
                    elif ac == 0x0F:
                        if _POLY5[p5]:
                            output[0] ^= 1
                            outvol[0] = audv[0] if output[0] else 0

            # ----- process channel 1 -----
            if div_max[1] != 0:
                div_cnt[1] -= 1
                if div_cnt[1] <= 0:
                    div_cnt[1] = div_max[1]
                    ac = audc[1]

                    if ac <= 0x00:
                        outvol[1] = audv[1]
                    elif ac == 0x01:
                        outvol[1] = audv[1] if _POLY4[p4] else 0
                    elif ac == 0x02:
                        outvol[1] = audv[1] if _POLY4[p4] else 0
                    elif ac == 0x03:
                        if _POLY5[p5]:
                            outvol[1] = audv[1] if _POLY4[p4] else 0
                    elif ac == 0x04 or ac == 0x05:
                        output[1] ^= 1
                        outvol[1] = audv[1] if output[1] else 0
                    elif ac == 0x06:
                        output[1] ^= 1
                        outvol[1] = audv[1] if output[1] else 0
                    elif ac == 0x07:
                        if _POLY5[p5]:
                            output[1] ^= 1
                            outvol[1] = audv[1] if output[1] else 0
                    elif ac == 0x08:
                        outvol[1] = audv[1] if _POLY9[p9] else 0
                    elif ac == 0x09:
                        outvol[1] = audv[1] if _POLY5[p5] else 0
                    elif ac == 0x0A:
                        output[1] ^= 1
                        outvol[1] = audv[1] if output[1] else 0
                    elif ac == 0x0B:
                        outvol[1] = audv[1]
                    elif ac == 0x0C or ac == 0x0D:
                        output[1] ^= 1
                        outvol[1] = audv[1] if output[1] else 0
                    elif ac == 0x0E:
                        output[1] ^= 1
                        outvol[1] = audv[1] if output[1] else 0
                    elif ac == 0x0F:
                        if _POLY5[p5]:
                            output[1] ^= 1
                            outvol[1] = audv[1] if output[1] else 0

            # Mix both channels
            buf[bi] = outvol[0] + outvol[1]
            bi += 1

        # Store back poly positions
        self._p4 = p4
        self._p5 = p5
        self._p9 = p9
        self._buffer_index = bi

    def end_frame(self) -> List[int]:
        """Finish the current frame and return the PCM sample buffer.

        Returns a list of ``buffer_index`` unsigned samples in range 0..30.
        The caller is responsible for any resampling to the final output
        rate (e.g. 44100 Hz).
        """
        return self._buffer[:self._buffer_index]

    # ----- internals ------------------------------------------------------

    def _recalc(self, ch: int) -> None:
        """Recompute *div_n_max* for channel *ch* after a register change.

        Pre-dividers (div31 = 31x, div3 = 3x) are folded into the maximum
        counter value so the inner sample loop only needs a simple
        countdown.

        Division chain by AUDC value:
          0x00/0x0B  volume only (max = 0)
          0x01       (AUDF+1)
          0x02       (AUDF+1) * 31      div31 -> poly4
          0x03       (AUDF+1)           5bit-gated poly4
          0x04/0x05  (AUDF+1)           toggle
          0x06/0x0A  (AUDF+1) * 31      div31 -> toggle
          0x07       (AUDF+1)           5bit-gated toggle
          0x08       (AUDF+1)           poly9
          0x09       (AUDF+1)           poly5
          0x0C/0x0D  (AUDF+1) * 3       div3 -> toggle (= /6)
          0x0E       (AUDF+1) * 93      div31*3 -> toggle (= /186)
          0x0F       (AUDF+1) * 3       5bit-gated div3 -> toggle
        """
        ac = self._audc[ch]
        af = self._audf[ch]

        if ac == 0x00 or ac == 0x0B:
            # Volume-only: disable divider, just output volume
            self._div_n_max[ch] = 0
            self._outvol[ch] = self._audv[ch]
            return

        base = af + 1

        if ac in (0x02, 0x06, 0x0A):
            new_max = base * 31
        elif ac in (0x0C, 0x0D, 0x0F):
            new_max = base * 3
        elif ac == 0x0E:
            new_max = base * 93
        else:
            new_max = base

        self._div_n_max[ch] = new_max

        # Restart the counter when the new period is shorter than the
        # current count (avoids stale long waits after a register change).
        if self._div_n_cnt[ch] <= 0 or self._div_n_cnt[ch] > new_max:
            self._div_n_cnt[ch] = new_max
