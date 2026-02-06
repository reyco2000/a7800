"""
Machine2600 -- Atari 2600 (VCS) machine emulation.
Ported from the C# Machine2600, Machine2600NTSC, and Machine2600PAL classes.

The Atari 2600 hardware consists of:

* **MOS 6507** (6502 with 13 address lines) -- CPU.
* **TIA** (Television Interface Adapter) -- video and audio.
* **PIA / RIOT** (6532) -- RAM, I/O, timer.
* **Cartridge** -- ROM with optional bank-switching.

Address-space layout (13-bit, 8 KB, 64-byte pages, 128 pages):

=========  ============  ==========  ================================
Range      Size          Device      Notes
=========  ============  ==========  ================================
0x0000-    0x0080 each   TIA         Repeated every 0x100 in 0x0000-
 0x007F    (2 pages)                 0x0FFF (where A7=0)
0x0080-    0x0080 each   PIA (RIOT)  Repeated every 0x100 in 0x0000-
 0x00FF    (2 pages)                 0x0FFF (where A7=1)
0x1000-    0x1000        Cartridge   ROM / bank-switched mapper
 0x1FFF    (64 pages)
=========  ============  ==========  ================================

Frame loop
----------
Each frame, the TIA drives the scanline timing.  The CPU runs until
the TIA signals end-of-frame.  WSYNC delays are handled by stealing
clocks from the CPU.  One CPU clock equals three TIA (colour) clocks,
so ``RunClocksMultiple = 1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from emu7800.core.machine_base import MachineBase

if TYPE_CHECKING:
    from emu7800.core.address_space import AddressSpace
    from emu7800.core.devices import IDevice
    from emu7800.core.m6502 import M6502
    from emu7800.core.pia import PIA
    from emu7800.core.tia import TIA


class Machine2600(MachineBase):
    """Atari 2600 machine emulation (base for NTSC and PAL variants).

    This class wires up the 2600 hardware and implements the TIA-driven
    frame loop.  The NTSC/PAL subclasses supply the timing parameters
    and palette.

    Parameters
    ----------
    cart:
        The cartridge device to install.
    p1:
        Controller type for player 1.
    p2:
        Controller type for player 2.
    scanlines:
        Total scanlines per frame.
    first_scanline:
        First visible scanline.
    frame_hz:
        Frame rate in Hz.
    sound_sample_freq:
        Audio sample rate in Hz.
    palette:
        256-entry ARGB palette.
    """

    # 2600 visible resolution: 160 pixels wide.
    VISIBLE_PITCH: int = 160

    def __init__(
        self,
        cart: IDevice,
        p1: int,
        p2: int,
        scanlines: int,
        first_scanline: int,
        frame_hz: int,
        sound_sample_freq: int,
        palette: List[int],
    ) -> None:
        super().__init__(
            scanlines=scanlines,
            first_scanline=first_scanline,
            frame_hz=frame_hz,
            sound_sample_freq=sound_sample_freq,
            palette=palette,
            visible_pitch=self.VISIBLE_PITCH,
        )

        # Store controller configuration.
        self.p1: int = p1
        self.p2: int = p2

        # -- Build the hardware --

        # 13-bit address space (8 KB), 64-byte pages -> 128 pages.
        from emu7800.core.address_space import AddressSpace
        self.mem: AddressSpace = AddressSpace(
            machine=self,
            addr_space_shift=13,
            page_shift=6,
        )

        # CPU: MOS 6507 (6502 core, RunClocksMultiple=1).
        from emu7800.core.m6502 import M6502
        self.cpu: M6502 = M6502(self, run_clocks_multiple=1)

        # TIA: Television Interface Adapter (video + audio).
        from emu7800.core.tia import TIA
        self._tia: TIA = TIA(self, scanlines)

        # PIA / RIOT: 6532 (128 bytes RAM, I/O ports, timer).
        from emu7800.core.pia import PIA
        self.pia: PIA = PIA(self)

        # Cartridge.
        self.cart: IDevice = cart

        # -- Wire the address space --
        self._map_devices()

    # ------------------------------------------------------------------
    # Address-space mapping
    # ------------------------------------------------------------------

    def _map_devices(self) -> None:
        """Map all devices into the 13-bit address space.

        Within the lower 4 KB (0x0000-0x0FFF), every 256-byte block is
        divided:
        * Low half  (A7=0, 0x00-0x7F) -> TIA  (2 pages of 64 bytes)
        * High half (A7=1, 0x80-0xFF) -> PIA  (2 pages of 64 bytes)

        The upper 4 KB (0x1000-0x1FFF) is the cartridge ROM window.
        """
        for base in range(0x0000, 0x1000, 0x0100):
            self.mem.map(base, 0x0080, self._tia)
            self.mem.map(base + 0x0080, 0x0080, self.pia)

        # Map the cartridge.  If the cart provides its own map() method
        # (for bank-switching), let it handle the mapping.  Otherwise
        # use the default 0x1000-0x1FFF window.
        if hasattr(self.cart, "map") and callable(self.cart.map):
            self.cart.map(self.mem)
        else:
            self.mem.map(0x1000, 0x1000, self.cart)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tia(self) -> TIA:
        """The TIA chip instance."""
        return self._tia

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the 2600 to power-on state."""
        super().reset()
        self.cpu.reset()
        self._tia.reset()
        self.pia.reset()
        if hasattr(self.cart, "reset") and callable(self.cart.reset):
            self.cart.reset()

    def compute_next_frame(self) -> None:
        """Run one NTSC/PAL frame of 2600 emulation.

        The TIA is the master timing device.  The loop gives the CPU
        enough clocks for ``scanlines + 3`` lines (with some margin),
        then runs cycles until the TIA signals end-of-frame or the CPU
        jams.

        WSYNC handling: when a game writes to TIA WSYNC, the TIA sets
        ``wsync_delay_clocks`` to the number of TIA colour clocks
        remaining on the current scanline.  We convert that to CPU
        clocks (divide by 3) and steal them from the CPU budget.
        """
        super().compute_next_frame()
        if self.machine_halt:
            return

        tia = self._tia
        cpu = self.cpu
        fb = self.frame_buffer

        tia.start_of_frame()

        # Budget: enough CPU clocks for the full frame plus a few extra
        # scanlines of margin.  76 colour clocks per scanline / 3 = ~25
        # CPU clocks, but the C# code uses 76 directly because
        # RunClocksMultiple=1 and the CPU internally counts in single-
        # clock steps.
        cpu.run_clocks = (fb.scanlines + 3) * 76

        while cpu.run_clocks > 0 and not cpu.jammed:
            # If the TIA is asserting WSYNC, steal the appropriate
            # number of CPU clocks (TIA colour clocks / 3).
            if tia.wsync_delay_clocks > 0:
                stolen = tia.wsync_delay_clocks // 3
                cpu.clock += stolen
                cpu.run_clocks -= stolen
                tia.wsync_delay_clocks = 0

            # Stop early if the TIA has finished the frame.
            if tia.end_of_frame:
                break

            cpu.execute()

        # Frame done - end_of_frame flag was set by TIA during rendering

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return a serialisable snapshot of the full 2600 state."""
        snap = super().get_snapshot()
        snap["machine_class"] = self.__class__.__name__
        return snap

    def restore_snapshot(self, snapshot: dict) -> None:
        """Restore 2600 state from a previous snapshot."""
        super().restore_snapshot(snapshot)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scanlines={self.frame_buffer.scanlines}, "
            f"first_scanline={self.first_scanline}, "
            f"frame_hz={self.frame_hz}, "
            f"frame={self.frame_number})"
        )


# ======================================================================
# NTSC / PAL concrete variants
# ======================================================================


class Machine2600NTSC(Machine2600):
    """Atari 2600 configured for NTSC timing and palette.

    * 262 scanlines per frame
    * First visible scanline: 16
    * 60 Hz frame rate
    * 31 440 Hz audio sample rate
    * NTSC colour palette (128 hues x 8 luminances = 256 entries using
      the TIA NTSC palette table, indexed 0-255 but only even indices
      produce unique colours; odd indices duplicate the preceding even
      entry)
    """

    SCANLINES: int = 262
    FIRST_SCANLINE: int = 16
    FRAME_HZ: int = 60
    SOUND_SAMPLE_FREQ: int = 31440

    def __init__(self, cart: IDevice, p1: int, p2: int) -> None:
        from emu7800.core.tia_tables import ntsc_palette as NTSC_PALETTE

        super().__init__(
            cart=cart,
            p1=p1,
            p2=p2,
            scanlines=self.SCANLINES,
            first_scanline=self.FIRST_SCANLINE,
            frame_hz=self.FRAME_HZ,
            sound_sample_freq=self.SOUND_SAMPLE_FREQ,
            palette=NTSC_PALETTE,
        )


class Machine2600PAL(Machine2600):
    """Atari 2600 configured for PAL timing and palette.

    * 312 scanlines per frame
    * First visible scanline: 32
    * 50 Hz frame rate
    * 31 200 Hz audio sample rate
    * PAL colour palette
    """

    SCANLINES: int = 312
    FIRST_SCANLINE: int = 32
    FRAME_HZ: int = 50
    SOUND_SAMPLE_FREQ: int = 31200

    def __init__(self, cart: IDevice, p1: int, p2: int) -> None:
        from emu7800.core.tia_tables import pal_palette as PAL_PALETTE

        super().__init__(
            cart=cart,
            p1=p1,
            p2=p2,
            scanlines=self.SCANLINES,
            first_scanline=self.FIRST_SCANLINE,
            frame_hz=self.FRAME_HZ,
            sound_sample_freq=self.SOUND_SAMPLE_FREQ,
            palette=PAL_PALETTE,
        )
