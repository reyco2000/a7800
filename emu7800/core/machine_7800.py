"""
Machine7800 -- Atari 7800 ProSystem machine emulation.
Ported from the C# Machine7800, Machine7800NTSC, and Machine7800PAL classes.

The Atari 7800 hardware consists of:

* **6502C** (Sally) -- CPU, with ``RunClocksMultiple=4`` because the
  Maria graphics chip operates at 4x the CPU clock rate (7.16 MHz
  colour clock vs. 1.79 MHz CPU clock).
* **Maria** -- custom graphics chip producing the video output.
* **TIA** -- legacy 2600 TIA for audio only (no video output on the
  7800).  Attached as the address-space *snooper* so it sees bus
  traffic from Maria DMA.
* **PIA / RIOT** (6532) -- RAM, I/O, timer.
* **RAM0** (2 KB, 6116 SRAM) -- at 0x1800-0x1FFF.
* **RAM1** (2 KB, 6116 SRAM) -- at 0x2000-0x27FF with mirrors.
* **Cartridge** -- ROM with optional bank-switching.
* **BIOS** (optional) -- 4 KB or 16 KB ROM mapped into the upper
  address space during boot.

Address-space layout (16-bit, 64 KB, 64-byte pages, 1024 pages):

===========  ============  ===========  ================================
Range        Size          Device       Notes
===========  ============  ===========  ================================
0x0000-003F  0x0040        Maria        Repeated at 0x0100, 0x0200,
             (1 page)                   0x0300
0x0040-00FF  0x00C0        RAM1 mirror  Zero-page RAM (3 pages)
0x0140-01FF  0x00C0        RAM1 mirror  Stack-page RAM (3 pages)
0x0280-02FF  0x0080        PIA          Repeated at 0x0480, 0x0580
             (2 pages)
0x1800-1FFF  0x0800        RAM0         2 KB on-board RAM (32 pages)
0x2000-27FF  0x0800        RAM1         2 KB on-board RAM (32 pages)
0x2040-20FF  0x00C0        RAM1 mirror  (3 pages, within primary)
0x2140-21FF  0x00C0        RAM1 mirror  (3 pages, within primary)
0x2800-2FFF  0x0800        RAM1 mirror  Full 2 KB mirror
0x3000-37FF  0x0800        RAM1 mirror  Full 2 KB mirror
0x3800-3FFF  0x0800        RAM1 mirror  Full 2 KB mirror
0x4000-FFFF  0xC000        Cartridge    ROM / bank-switched mapper
===========  ============  ===========  ================================

BIOS mapping
------------
When a BIOS ROM is present, it is temporarily mapped over the top of
the cartridge space during reset so the CPU reads the reset vector from
the BIOS.  ``swap_in_bios()`` / ``swap_out_bios()`` handle this.

* 4 KB BIOS  -> mapped at 0xF000-0xFFFF
* 16 KB BIOS -> mapped at 0xC000-0xFFFF

Frame loop
----------
Each scanline is 114 colour clocks wide (= 454 Maria clocks / 4).
Maria DMA steals clocks from the CPU; the remaining clocks are given
to the CPU for instruction execution.  ``RunClocksMultiple=4`` means
``cpu.run_clocks`` is counted in Maria-clock units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from emu7800.core.machine_base import MachineBase
from emu7800.core.devices import RAM6116
from emu7800.core.types import Controller

if TYPE_CHECKING:
    from emu7800.core.address_space import AddressSpace
    from emu7800.core.devices import Bios7800, IDevice
    from emu7800.core.m6502 import M6502
    from emu7800.core.maria import Maria
    from emu7800.core.pia import PIA
    from emu7800.core.types import MachineType


class Machine7800(MachineBase):
    """Atari 7800 ProSystem machine emulation (base for NTSC/PAL).

    Parameters
    ----------
    cart:
        The cartridge device to install.
    bios:
        Optional BIOS ROM (4 KB or 16 KB).
    machine_type:
        The specific 7800 variant (with/without HSC, XM, etc.).
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

    # 7800 visible resolution: 320 pixels wide (Maria).
    VISIBLE_PITCH: int = 320

    # Colour clocks per scanline.
    CLOCKS_PER_SCANLINE: int = 114

    def __init__(
        self,
        cart: IDevice,
        bios: Optional[Bios7800],
        machine_type: MachineType,
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

        # Store configuration.
        self.machine_type: MachineType = machine_type
        self.p1: int = p1
        self.p2: int = p2

        # -- Build the hardware --

        # 16-bit address space (64 KB), 64-byte pages -> 1024 pages.
        from emu7800.core.address_space import AddressSpace
        self.mem: AddressSpace = AddressSpace(
            machine=self,
            addr_space_shift=16,
            page_shift=6,
        )

        # CPU: 6502C (Sally), RunClocksMultiple=4 because Maria runs
        # at 4x the CPU clock.
        from emu7800.core.m6502 import M6502
        self.cpu: M6502 = M6502(self, run_clocks_multiple=4)

        # Maria: custom graphics chip.
        from emu7800.core.maria import Maria
        self._maria: Maria = Maria(self)

        # PIA / RIOT: 6532.
        from emu7800.core.pia import PIA
        self.pia: PIA = PIA(self)

        # On-board RAM.
        self._ram0: RAM6116 = RAM6116()  # 2 KB at 0x1800
        self._ram1: RAM6116 = RAM6116()  # 2 KB at 0x2000 (+ mirrors)

        # Cartridge.
        self.cart: IDevice = cart

        # BIOS ROM (may be None if running without BIOS).
        self._bios: Optional[Bios7800] = bios

        # -- Wire the address space --
        self._map_devices()

    # ------------------------------------------------------------------
    # Address-space mapping
    # ------------------------------------------------------------------

    def _map_devices(self) -> None:
        """Map all devices into the 16-bit address space.

        See the module docstring for the full memory map.
        """
        # Maria registers at 0x0000-0x003F, repeated at 0x0100, 0x0200,
        # 0x0300.  Each mapping is 1 page (64 bytes).
        self.mem.map(0x0000, 0x0040, self._maria)
        self.mem.map(0x0100, 0x0040, self._maria)
        self.mem.map(0x0200, 0x0040, self._maria)
        self.mem.map(0x0300, 0x0040, self._maria)

        # PIA / RIOT at 0x0280-0x02FF, repeated at 0x0480, 0x0580.
        # Each mapping is 2 pages (128 bytes).
        self.mem.map(0x0280, 0x0080, self.pia)
        self.mem.map(0x0480, 0x0080, self.pia)
        self.mem.map(0x0580, 0x0080, self.pia)

        # RAM0: 2 KB at 0x1800-0x1FFF (32 pages).
        self.mem.map(0x1800, 0x0800, self._ram0)

        # RAM1: 2 KB primary at 0x2000-0x27FF (32 pages).
        self.mem.map(0x2000, 0x0800, self._ram1)

        # RAM1 mirrors into zero-page / stack-page area.
        # 0x0040-0x00FF: 192 bytes (3 pages) of RAM1 mirrored into
        #   the zero-page region (above the Maria registers).
        self.mem.map(0x0040, 0x00C0, self._ram1)
        # 0x0140-0x01FF: 192 bytes (3 pages) of RAM1 mirrored into
        #   the stack-page region.
        self.mem.map(0x0140, 0x00C0, self._ram1)

        # RAM1 mirrors within the primary 0x2000 block (these overlap
        # the primary mapping but ensure correct page resolution).
        self.mem.map(0x2040, 0x00C0, self._ram1)
        self.mem.map(0x2140, 0x00C0, self._ram1)

        # RAM1 full 2 KB mirrors at 0x2800, 0x3000, 0x3800.
        self.mem.map(0x2800, 0x0800, self._ram1)
        self.mem.map(0x3000, 0x0800, self._ram1)
        self.mem.map(0x3800, 0x0800, self._ram1)

        # Cartridge: default window is 0x4000-0xFFFF (48 KB).
        # If the cart provides its own map() method (for bank-switching),
        # let it handle the mapping.
        if hasattr(self.cart, "map") and callable(self.cart.map):
            self.cart.map(self.mem)
        else:
            self.mem.map(0x4000, 0xC000, self.cart)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def maria(self) -> Maria:
        """The Maria graphics chip instance."""
        return self._maria

    @property
    def ram0(self) -> RAM6116:
        """The 2 KB RAM0 at 0x1800."""
        return self._ram0

    @property
    def ram1(self) -> RAM6116:
        """The 2 KB RAM1 at 0x2000."""
        return self._ram1

    @property
    def bios(self) -> Optional[Bios7800]:
        """The BIOS ROM, or None if running without BIOS."""
        return self._bios

    # ------------------------------------------------------------------
    # BIOS swap
    # ------------------------------------------------------------------

    def swap_in_bios(self) -> None:
        """Map the BIOS ROM over the upper address space.

        * 4 KB BIOS  -> 0xF000-0xFFFF
        * 16 KB BIOS -> 0xC000-0xFFFF

        Called during :meth:`reset` so the CPU reads the reset vector
        from the BIOS instead of the cartridge.
        """
        if self._bios is None:
            return
        bios_size: int = self._bios.size
        if bios_size == 4096:
            self.mem.map(0xF000, 0x1000, self._bios)
        elif bios_size == 16384:
            self.mem.map(0xC000, 0x4000, self._bios)

    def swap_out_bios(self) -> None:
        """Remove the BIOS mapping and restore the cartridge.

        Re-maps the cartridge into the 0x4000-0xFFFF window, either
        via the cart's own ``map()`` method or the default linear
        mapping.
        """
        if self.cart is not None:
            if hasattr(self.cart, "map") and callable(self.cart.map):
                self.cart.map(self.mem)
            else:
                self.mem.map(0x4000, 0xC000, self.cart)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the 7800 to power-on state.

        If a BIOS is present it is swapped in before the CPU reset so
        the CPU fetches the reset vector from the BIOS.  The BIOS code
        will eventually hand off to the cartridge.
        """
        super().reset()

        # Swap BIOS in before CPU reset so the reset vector comes from
        # the BIOS ROM.
        if self._bios is not None:
            self.swap_in_bios()

        # Connect controllers so PIA can read them.
        self.input_state.left_controller_jack = Controller(self.p1)
        self.input_state.right_controller_jack = Controller(self.p2)

        self.cpu.reset()
        self._maria.reset()
        self.pia.reset()
        self._ram0.reset()
        self._ram1.reset()
        if hasattr(self.cart, "reset") and callable(self.cart.reset):
            self.cart.reset()

    def compute_next_frame(self) -> None:
        """Run one NTSC/PAL frame of 7800 emulation.

        Each scanline is 114 colour clocks (28.5 CPU clocks at 4x
        multiple).  Maria DMA steals clocks from the CPU during each
        scanline.

        The scanline loop:

        1. Give the CPU 7 clocks (times ``run_clocks_multiple``) for
           the horizontal blank interval.
        2. Execute CPU instructions.
        3. If the CPU requested a preempt (e.g. WSYNC), let Maria do
           DMA, pad the rest of the scanline, and continue.
        4. Otherwise, let Maria do DMA.  The DMA return value is the
           number of colour clocks consumed; these are stolen from the
           CPU's budget.  DMA clocks are rounded up to the next
           ``run_clocks_multiple`` boundary.
        5. Give the CPU the remaining scanline clocks and execute.
        6. Align to the 114-clock scanline boundary.
        """
        super().compute_next_frame()
        if self.machine_halt:
            return

        maria = self._maria
        cart = self.cart
        cpu = self.cpu
        fb = self.frame_buffer
        rcm = cpu.run_clocks_multiple  # 4

        maria.start_frame()
        if hasattr(cart, "start_frame") and callable(cart.start_frame):
            cart.start_frame()

        for _scanline in range(fb.scanlines):
            if cpu.jammed:
                break

            # Compute the effective CPU clock at the start of this
            # scanline.  run_clocks may be negative (leftover debt from
            # the previous scanline), so we must account for it.
            start_of_scanline: int = cpu.clock + (cpu.run_clocks // cpu.run_clocks_multiple)

            # -- Phase 1: HBLANK (7 CPU clocks) --
            cpu.run_clocks += 7 * rcm
            remaining: int = (self.CLOCKS_PER_SCANLINE - 7) * rcm
            cpu.execute()

            if cpu.jammed:
                break

            # -- Phase 2: Check for CPU preempt request (WSYNC) --
            if cpu.emulator_preempt_request:
                cpu.emulator_preempt_request = False
                maria.do_dma_processing()
                remaining_cpu_clocks = self.CLOCKS_PER_SCANLINE - (cpu.clock - start_of_scanline)
                cpu.clock += remaining_cpu_clocks
                cpu.run_clocks = 0
                continue

            # -- Phase 3: Maria DMA --
            dma_clocks: int = maria.do_dma_processing()

            # Ace of Aces title-screen flicker workaround (from C#).
            if (_scanline == 203 and fb.scanlines == 262) or \
               (_scanline == 228 and fb.scanlines == 312):
                if dma_clocks == 152 and remaining == 428 and \
                   cpu.run_clocks in (-4, -8):
                    dma_clocks -= 4

            # KLAX safety valve: if Maria DMA exceeds the available
            # scanline budget, halve it repeatedly until it fits.
            while cpu.run_clocks + remaining < dma_clocks:
                dma_clocks >>= 1

            if dma_clocks > 0:
                # Round DMA clocks up to the next div-4 boundary so the
                # CPU clock stays aligned with RunClocksMultiple.
                if (dma_clocks & 3) != 0:
                    dma_clocks += 4
                    dma_clocks -= dma_clocks & 3

                cpu.clock += dma_clocks // rcm
                cpu.run_clocks -= dma_clocks

            # -- Phase 4: Give CPU the remaining scanline clocks --
            cpu.run_clocks += remaining
            cpu.execute()

            # -- Phase 5: Align to scanline boundary --
            if cpu.emulator_preempt_request:
                cpu.emulator_preempt_request = False
            remaining_cpu_clocks = self.CLOCKS_PER_SCANLINE - (cpu.clock - start_of_scanline)
            if remaining_cpu_clocks > 0:
                cpu.clock += remaining_cpu_clocks
            cpu.run_clocks = 0

        if hasattr(cart, "end_frame") and callable(cart.end_frame):
            cart.end_frame()
        maria.end_frame()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return a serialisable snapshot of the full 7800 state."""
        snap = super().get_snapshot()
        snap["machine_class"] = self.__class__.__name__
        snap["ram0"] = self._ram0.get_snapshot()
        snap["ram1"] = self._ram1.get_snapshot()
        return snap

    def restore_snapshot(self, snapshot: dict) -> None:
        """Restore 7800 state from a previous snapshot."""
        super().restore_snapshot(snapshot)
        if "ram0" in snapshot:
            self._ram0.restore_snapshot(snapshot["ram0"])
        if "ram1" in snapshot:
            self._ram1.restore_snapshot(snapshot["ram1"])

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"machine_type={self.machine_type!r}, "
            f"scanlines={self.frame_buffer.scanlines}, "
            f"first_scanline={self.first_scanline}, "
            f"frame_hz={self.frame_hz}, "
            f"bios={'yes' if self._bios else 'no'}, "
            f"frame={self.frame_number})"
        )


# ======================================================================
# NTSC / PAL concrete variants
# ======================================================================


class Machine7800NTSC(Machine7800):
    """Atari 7800 configured for NTSC timing and palette.

    * 262 scanlines per frame
    * First visible scanline: 16
    * 60 Hz frame rate
    * 31 440 Hz audio sample rate
    * NTSC colour palette from the Maria palette table (256 entries)
    """

    SCANLINES: int = 262
    FIRST_SCANLINE: int = 16
    FRAME_HZ: int = 60
    SOUND_SAMPLE_FREQ: int = 31440

    def __init__(
        self,
        cart: IDevice,
        bios: Optional[Bios7800],
        machine_type: MachineType,
        p1: int,
        p2: int,
    ) -> None:
        from emu7800.core.maria_tables import ntsc_palette as NTSC_PALETTE

        super().__init__(
            cart=cart,
            bios=bios,
            machine_type=machine_type,
            p1=p1,
            p2=p2,
            scanlines=self.SCANLINES,
            first_scanline=self.FIRST_SCANLINE,
            frame_hz=self.FRAME_HZ,
            sound_sample_freq=self.SOUND_SAMPLE_FREQ,
            palette=NTSC_PALETTE,
        )


class Machine7800PAL(Machine7800):
    """Atari 7800 configured for PAL timing and palette.

    * 312 scanlines per frame
    * First visible scanline: 34
    * 50 Hz frame rate
    * 31 200 Hz audio sample rate
    * PAL colour palette from the Maria palette table (256 entries)
    """

    SCANLINES: int = 312
    FIRST_SCANLINE: int = 34
    FRAME_HZ: int = 50
    SOUND_SAMPLE_FREQ: int = 31200

    def __init__(
        self,
        cart: IDevice,
        bios: Optional[Bios7800],
        machine_type: MachineType,
        p1: int,
        p2: int,
    ) -> None:
        from emu7800.core.maria_tables import pal_palette as PAL_PALETTE

        super().__init__(
            cart=cart,
            bios=bios,
            machine_type=machine_type,
            p1=p1,
            p2=p2,
            scanlines=self.SCANLINES,
            first_scanline=self.FIRST_SCANLINE,
            frame_hz=self.FRAME_HZ,
            sound_sample_freq=self.SOUND_SAMPLE_FREQ,
            palette=PAL_PALETTE,
        )
