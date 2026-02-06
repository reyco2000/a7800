"""
Maria -- MARIA display device for the Atari 7800.

The MARIA chip is the 7800's custom graphics processor.  It handles:

* **DMA** -- fetching display-list lists (DLL), display lists (DL), and
  graphic data from the cartridge / RAM address space.
* **Line-RAM rendering** -- decoding graphic data into a 512-byte line
  buffer using one of six pixel modes (160A, 160B, 320A, 320B, 320C,
  320D).
* **Palette colour mapping** -- converting line-RAM colour indices to
  palette-register values and writing them to the video frame buffer.
* **Register I/O** -- responding to CPU reads/writes at addresses
  0x00-0x3F (CTRL, WSYNC, MSTAT, palette colours, DPPH/DPPL,
  CHARBASE, INPTCTRL, INPTx, and TIA audio registers).

MARIA is memory-mapped at 0x0000-0x003F (with mirrors at 0x0100,
0x0200, 0x0300) in the 7800 address space.

Register map (0x00 - 0x3F):

    0x01  INPTCTRL   Input / BIOS control
    0x08  INPT0      Input port 0  (left trigger)
    0x09  INPT1      Input port 1  (left trigger2)
    0x0A  INPT2      Input port 2  (right trigger)
    0x0B  INPT3      Input port 3  (right trigger2)
    0x0C  INPT4      Input port 4  (left fire, latched)
    0x0D  INPT5      Input port 5  (right fire, latched)
    0x15  AUDC0      Audio control channel 0  (forwarded to TIASound)
    0x16  AUDC1      Audio control channel 1
    0x17  AUDF0      Audio frequency channel 0
    0x18  AUDF1      Audio frequency channel 1
    0x19  AUDV0      Audio volume channel 0
    0x1A  AUDV1      Audio volume channel 1
    0x20  BACKGRND   Background colour
    0x21-0x23  P0C1-P0C3   Palette 0, colours 1-3
    0x24  WSYNC      Wait for horizontal sync (halts CPU)
    0x25-0x27  P1C1-P1C3   Palette 1, colours 1-3
    0x28  MSTAT      Maria status (bit 7 = VBLANK)
    0x29-0x2B  P2C1-P2C3   Palette 2, colours 1-3
    0x2C  DPPH       Display list list pointer high
    0x2D-0x2F  P3C1-P3C3   Palette 3, colours 1-3
    0x30  DPPL       Display list list pointer low
    0x31-0x33  P4C1-P4C3   Palette 4, colours 1-3
    0x34  CHARBASE   Character base address (high byte)
    0x35-0x37  P5C1-P5C3   Palette 5, colours 1-3
    0x38  OFFSET     Reserved for future expansion
    0x39-0x3B  P6C1-P6C3   Palette 6, colours 1-3
    0x3C  CTRL       Maria control register
    0x3D-0x3F  P7C1-P7C3   Palette 7, colours 1-3

CTRL register bits:
    7    ColorKill   (force monochrome)
    6:5  DMA         (01 = enabled)
    4    CWidth      (character width: 0 = 1-byte, 1 = 2-byte)
    3    BCntl       (border control)
    2    Kangaroo    (1 = transparent pixels do NOT overwrite line RAM)
    1:0  RM          (render mode 0-3)

Six render modes selected by RM and WM (write-mode flag per DL entry):
    RM=0  WM=0  160A   2bpp, 4 pixels/byte, doubled horizontally
    RM=0  WM=1  160B   per-pixel palette, 2 pixels/byte, doubled
    RM=2  WM=0  320D   2bpp interleaved palette bits, 8 pixels/byte
    RM=2  WM=1  320B   2bpp interleaved, 4 pixels/byte
    RM=3  WM=0  320A   1bpp, 8 pixels/byte
    RM=3  WM=1  320C   per-pixel palette, 4 pixels/byte
    RM=1  (skip -- DL entry is ignored)

Derived from Dan Boris' work with 7800 emulation within the MESS
emulator.  Thanks to Matthias Luedtke for corrections to the 320B
render mode (credited to Eckhard Stolberg on Atari Age, June 2005).

Ported from the EMU7800 C# Maria class (Copyright 2004-2012 Mike Murphy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emu7800.core.devices import IDevice
from emu7800.core.sound.tia_sound import TIASound
from emu7800.core.types import Controller, ControllerAction

if TYPE_CHECKING:
    from emu7800.core.machine_7800 import Machine7800

# ---------------------------------------------------------------------------
# Register address constants
# ---------------------------------------------------------------------------

INPTCTRL = 0x01   # Write: input port control (VBLANK in TIA)
INPT0    = 0x08   # Read pot port: D7
INPT1    = 0x09   # Read pot port: D7
INPT2    = 0x0A   # Read pot port: D7
INPT3    = 0x0B   # Read pot port: D7
INPT4    = 0x0C   # Read P1 joystick trigger: D7
INPT5    = 0x0D   # Read P2 joystick trigger: D7

AUDC0    = 0x15   # Write: audio control 0 (D3-0)
AUDC1    = 0x16   # Write: audio control 1 (D4-0)
AUDF0    = 0x17   # Write: audio frequency 0 (D4-0)
AUDF1    = 0x18   # Write: audio frequency 1 (D3-0)
AUDV0    = 0x19   # Write: audio volume 0 (D3-0)
AUDV1    = 0x1A   # Write: audio volume 1 (D3-0)

BACKGRND = 0x20   # Background color
P0C1     = 0x21   # Palette 0 - color 1
P0C2     = 0x22   # Palette 0 - color 2
P0C3     = 0x23   # Palette 0 - color 3
WSYNC    = 0x24   # Wait for sync
P1C1     = 0x25   # Palette 1 - color 1
P1C2     = 0x26   # Palette 1 - color 2
P1C3     = 0x27   # Palette 1 - color 3
MSTAT    = 0x28   # Maria status
P2C1     = 0x29   # Palette 2 - color 1
P2C2     = 0x2A   # Palette 2 - color 2
P2C3     = 0x2B   # Palette 2 - color 3
DPPH     = 0x2C   # Display list list pointer high
P3C1     = 0x2D   # Palette 3 - color 1
P3C2     = 0x2E   # Palette 3 - color 2
P3C3     = 0x2F   # Palette 3 - color 3
DPPL     = 0x30   # Display list list pointer low
P4C1     = 0x31   # Palette 4 - color 1
P4C2     = 0x32   # Palette 4 - color 2
P4C3     = 0x33   # Palette 4 - color 3
CHARBASE = 0x34   # Character base address
P5C1     = 0x35   # Palette 5 - color 1
P5C2     = 0x36   # Palette 5 - color 2
P5C3     = 0x37   # Palette 5 - color 3
OFFSET   = 0x38   # Future expansion (store zero here)
P6C1     = 0x39   # Palette 6 - color 1
P6C2     = 0x3A   # Palette 6 - color 2
P6C3     = 0x3B   # Palette 6 - color 3
CTRL     = 0x3C   # Maria control register
P7C1     = 0x3D   # Palette 7 - color 1
P7C2     = 0x3E   # Palette 7 - color 2
P7C3     = 0x3F   # Palette 7 - color 3

# Set of all palette/color register addresses for fast membership test.
_PALETTE_ADDRS = frozenset((
    BACKGRND,
    P0C1, P0C2, P0C3, P1C1, P1C2, P1C3,
    P2C1, P2C2, P2C3, P3C1, P3C2, P3C3,
    P4C1, P4C2, P4C3, P5C1, P5C2, P5C3,
    P6C1, P6C2, P6C3, P7C1, P7C2, P7C3,
))

# Set of audio register addresses.
_AUDIO_ADDRS = frozenset((AUDC0, AUDC1, AUDF0, AUDF1, AUDV0, AUDV1))

CPU_TICKS_PER_AUDIO_SAMPLE = 57


# ---------------------------------------------------------------------------
# Maria display device
# ---------------------------------------------------------------------------

class Maria(IDevice):
    """MARIA display device for the Atari 7800.

    Handles DMA-driven rendering of display lists into a line-RAM buffer,
    converting graphic data through one of six pixel modes and outputting
    palette-mapped colour values to the video frame buffer.

    Parameters
    ----------
    machine :
        The owning :class:`~emu7800.core.machine_7800.Machine7800` instance.
        Maria accesses the CPU, address space, frame buffer, input state,
        PIA, and BIOS-swap methods through this reference.
    """

    def __init__(self, machine: Machine7800) -> None:
        self._m: Machine7800 = machine

        # Determine visible scanline range from the machine's frame buffer.
        scanlines: int = machine.frame_buffer.scanlines
        if scanlines not in (262, 312):
            raise ValueError(
                f"scanlines must be 262 (NTSC) or 312 (PAL), got {scanlines}"
            )
        self._first_visible_scanline: int = 11
        self._last_visible_scanline: int = (
            self._first_visible_scanline + 242 + (50 if scanlines == 312 else 0)
        )
        self._is_pal: bool = scanlines == 312

        # Line RAM: 512-byte buffer used for scanline composition.
        self._line_ram: bytearray = bytearray(0x200)

        # Register file: 64 bytes mapped to 0x00-0x3F.
        self._registers: bytearray = bytearray(0x40)

        # TIA sound (two-channel audio routed through Maria register space).
        self._tia_sound: TIASound = TIASound()

        # Frame timing -- derived from CPU clock.
        self._start_of_frame_cpu_clock: int = 0
        self._dma_clocks: int = 0

        # Lightgun emulation state (transient, not serialised).
        self._lightgun_first_sample_cpu_clock: int = 0
        self._lightgun_frame_samples: int = 0
        self._lightgun_sampled_scanline: int = 0
        self._lightgun_sampled_visible_hpos: int = 0

        # DMA / display-list state.
        self._wm: bool = False              # Write mode flag from DL header
        self._dll: int = 0                  # Display List List pointer (16-bit)
        self._dl: int = 0                   # Current Display List pointer (16-bit)
        self._offset: int = 0               # Zone line offset counter
        self._holey: int = 0                # Holey DMA mode (0, 1, or 2)
        self._width: int = 0                # Object width from DL header
        self._hpos: int = 0                 # Horizontal position from DL header
        self._palette_no: int = 0           # Palette number (pre-shifted)
        self._ind_mode: bool = False        # Indirect (character) mode flag

        # INPTCTRL lock flag.
        self._ctrl_lock: bool = False

        # MARIA CTRL register decoded fields.
        self._dma_enabled: bool = False
        self._color_kill: bool = False
        self._cwidth: bool = False
        self._bcntl: bool = False
        self._kangaroo: bool = False
        self._rm: int = 0                   # Render mode (0-3)

    # ------------------------------------------------------------------
    # IDevice interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset Maria to power-on state."""
        self._ctrl_lock = False
        self._dma_enabled = False
        self._color_kill = False
        self._cwidth = False
        self._bcntl = False
        self._kangaroo = False
        self._rm = 0
        self._tia_sound.reset()

    def __getitem__(self, addr: int) -> int:
        """Read (peek) a Maria register.

        Used by the CPU to read MSTAT, input ports, etc.
        """
        return self._peek(addr)

    def __setitem__(self, addr: int, value: int) -> None:
        """Write (poke) a Maria register.

        Used by the CPU to write CTRL, WSYNC, palette colours, audio
        registers, INPTCTRL, etc.
        """
        self._poke(addr, value)

    # ------------------------------------------------------------------
    # Public frame lifecycle
    # ------------------------------------------------------------------

    def start_frame(self) -> None:
        """Prepare Maria for a new video frame.

        Records the CPU clock at the start of the frame so that the
        current scanline can be derived at any point during the frame.
        """
        cpu = self._m.cpu
        self._start_of_frame_cpu_clock = (
            cpu.clock + (cpu.run_clocks // cpu.run_clocks_multiple)
        )
        self._lightgun_first_sample_cpu_clock = 0
        self._tia_sound.start_frame()

    def do_dma_processing(self) -> int:
        """Perform Maria DMA for the current scanline.

        First outputs the previous scanline's line-RAM contents to the
        frame buffer, then (if DMA is enabled and within the visible
        area) fetches the next display list and builds line-RAM data.

        Returns
        -------
        int
            The number of DMA colour clocks consumed.  These clocks are
            stolen from the CPU's budget for this scanline.
        """
        self._output_line_ram()

        sl = self._scanline

        if (not self._dma_enabled
                or sl < self._first_visible_scanline
                or sl >= self._last_visible_scanline):
            return 0

        self._dma_clocks = 0

        if self._dma_enabled and sl == self._first_visible_scanline:
            # DMA TIMING: End of VBLANK: DMA startup + long shutdown
            self._dma_clocks += 15

            self._dll = self._word(
                self._registers[DPPL], self._registers[DPPH]
            )
            self._consume_next_dll_entry()

        # DMA TIMING: DMA Startup, 5-9 cycles
        self._dma_clocks += 5

        self._build_line_ram()

        self._offset -= 1
        if self._offset < 0:
            self._consume_next_dll_entry()
            # DMA TIMING: DMA Shutdown: Last line of zone, 10-13 cycles
            self._dma_clocks += 10
        else:
            # DMA TIMING: DMA Shutdown: Other line of zone, 4-7 cycles
            self._dma_clocks += 4

        return self._dma_clocks

    def end_frame(self) -> None:
        """Finalise the current video frame."""
        self._tia_sound.end_frame()

    # ------------------------------------------------------------------
    # Scanline property (derived from CPU clock)
    # ------------------------------------------------------------------

    @property
    def _scanline(self) -> int:
        """Current scanline number, derived from the CPU clock.

        Each scanline is 114 colour clocks wide.  The scanline number
        is the integer quotient of the elapsed colour clocks since the
        start of the frame divided by 114.
        """
        return (self._m.cpu.clock - self._start_of_frame_cpu_clock) // 114

    # ------------------------------------------------------------------
    # Display-list / line-RAM builders
    # ------------------------------------------------------------------

    def _build_line_ram(self) -> None:
        """Walk the current display list and render all objects into
        line-RAM according to the active render mode.

        Each DL entry is either 4 bytes (normal) or 5 bytes (extended /
        indirect):

        Normal (4-byte) header:
            byte 0: graphic data address low
            byte 1: palette(7-5) / width(4-0)
            byte 2: graphic data address high
            byte 3: horizontal position

        Extended (5-byte) header:
            byte 0: graphic data address low
            byte 1: mode byte (WM(7), IND(5), lower 5 bits = 0)
            byte 2: graphic data address high
            byte 3: palette(7-5) / width(4-0)
            byte 4: horizontal position

        A DL entry with (byte1 & 0x5F) == 0 marks the end of the list.
        """
        dl = self._dl

        while True:
            mode_byte = self._dma_read(dl + 1)
            if (mode_byte & 0x5F) == 0:
                break

            self._ind_mode = False

            if (mode_byte & 0x1F) == 0:
                # Extended DL header (5 bytes)
                dl0 = self._dma_read(dl)
                dl += 1
                dl1 = self._dma_read(dl)
                dl += 1
                dl2 = self._dma_read(dl)
                dl += 1
                dl3 = self._dma_read(dl)
                dl += 1
                dl4 = self._dma_read(dl)
                dl += 1

                graphaddr = self._word(dl0, dl2)
                self._wm = (dl1 & 0x80) != 0
                self._ind_mode = (dl1 & 0x20) != 0
                self._palette_no = (dl3 & 0xE0) >> 3
                self._width = (~dl3 & 0x1F) + 1
                self._hpos = dl4

                # DMA TIMING: DL 5 byte header
                self._dma_clocks += 10
            else:
                # Normal DL header (4 bytes)
                dl0 = self._dma_read(dl)
                dl += 1
                dl1 = self._dma_read(dl)
                dl += 1
                dl2 = self._dma_read(dl)
                dl += 1
                dl3 = self._dma_read(dl)
                dl += 1

                graphaddr = self._word(dl0, dl2)
                self._palette_no = (dl1 & 0xE0) >> 3
                self._width = (~dl1 & 0x1F) + 1
                self._hpos = dl3

                # DMA TIMING: DL 4 byte header
                self._dma_clocks += 8

            # DMA TIMING: Graphic reads
            if self._rm != 1:
                if self._ind_mode:
                    gfx_clocks = self._width * (9 if self._cwidth else 6)
                else:
                    gfx_clocks = self._width * 3
                self._dma_clocks += gfx_clocks

            rm = self._rm
            if rm == 0:
                if self._wm:
                    self._build_line_ram_160b(graphaddr)
                else:
                    self._build_line_ram_160a(graphaddr)
            elif rm == 1:
                # RM=1: skip this DL entry (blank mode)
                continue
            elif rm == 2:
                if self._wm:
                    self._build_line_ram_320b(graphaddr)
                else:
                    self._build_line_ram_320d(graphaddr)
            elif rm == 3:
                if self._wm:
                    self._build_line_ram_320c(graphaddr)
                else:
                    self._build_line_ram_320a(graphaddr)

    # ------------------------------------------------------------------
    # 160A: 2bpp, 4 pixels per byte, each pixel doubled horizontally
    # ------------------------------------------------------------------

    def _build_line_ram_160a(self, graphaddr: int) -> None:
        """160A mode: 4 colour pixels from DL-header palette.

        Each byte yields 4 pixels (2 bits each), doubled to occupy 8
        line-RAM entries.  Non-zero colour values write
        ``palette_no | colour`` to line-RAM.
        """
        indbytes = 2 if (self._ind_mode and self._cwidth) else 1
        hpos = self._hpos << 1
        dataaddr = (graphaddr + (self._offset << 8)) & 0xFFFF

        line_ram = self._line_ram
        palette_no = self._palette_no
        holey = self._holey
        width = self._width
        ind_mode = self._ind_mode
        offset = self._offset
        charbase = self._registers[CHARBASE]

        for i in range(width):
            if ind_mode:
                dataaddr = self._word(
                    self._dma_read(graphaddr + i),
                    charbase + offset,
                )

            for _j in range(indbytes):
                if ((holey == 0x02 and (dataaddr & 0x9000) == 0x9000)
                        or (holey == 0x01
                            and (dataaddr & 0x8800) == 0x8800)):
                    hpos += 8
                    dataaddr = (dataaddr + 1) & 0xFFFF
                    continue

                d = self._dma_read(dataaddr)
                dataaddr = (dataaddr + 1) & 0xFFFF

                # Pixel 0 (bits 7-6)
                c = (d & 0xC0) >> 6
                if c != 0:
                    val = palette_no | c
                    line_ram[hpos & 0x1FF] = val
                    line_ram[(hpos + 1) & 0x1FF] = val
                hpos += 2

                # Pixel 1 (bits 5-4)
                c = (d & 0x30) >> 4
                if c != 0:
                    val = palette_no | c
                    line_ram[hpos & 0x1FF] = val
                    line_ram[(hpos + 1) & 0x1FF] = val
                hpos += 2

                # Pixel 2 (bits 3-2)
                c = (d & 0x0C) >> 2
                if c != 0:
                    val = palette_no | c
                    line_ram[hpos & 0x1FF] = val
                    line_ram[(hpos + 1) & 0x1FF] = val
                hpos += 2

                # Pixel 3 (bits 1-0)
                c = d & 0x03
                if c != 0:
                    val = palette_no | c
                    line_ram[hpos & 0x1FF] = val
                    line_ram[(hpos + 1) & 0x1FF] = val
                hpos += 2

    # ------------------------------------------------------------------
    # 160B: per-pixel palette, 2 pixels per byte, doubled
    # ------------------------------------------------------------------

    def _build_line_ram_160b(self, graphaddr: int) -> None:
        """160B write-mode: 2 pixels per byte with per-pixel palette
        selection.

        Each byte is split into two halves.  The upper pixel uses bits
        7-6 for colour and bits 3-2 for the low palette bits.  The
        lower pixel uses bits 5-4 for colour and bits 1-0 for the low
        palette bits.  The high palette bit comes from the DL header.

        Kangaroo mode: when colour is zero, write zero to line-RAM
        (erase).
        """
        indbytes = 2 if (self._ind_mode and self._cwidth) else 1
        hpos = self._hpos << 1
        dataaddr = (graphaddr + (self._offset << 8)) & 0xFFFF

        line_ram = self._line_ram
        palette_no = self._palette_no
        kangaroo = self._kangaroo
        holey = self._holey
        width = self._width
        ind_mode = self._ind_mode
        offset = self._offset
        charbase = self._registers[CHARBASE]

        for i in range(width):
            if ind_mode:
                dataaddr = self._word(
                    self._dma_read(graphaddr + i),
                    charbase + offset,
                )

            for _j in range(indbytes):
                if ((holey == 0x02 and (dataaddr & 0x9000) == 0x9000)
                        or (holey == 0x01
                            and (dataaddr & 0x8800) == 0x8800)):
                    hpos += 4
                    dataaddr = (dataaddr + 1) & 0xFFFF
                    continue

                d = self._dma_read(dataaddr)
                dataaddr = (dataaddr + 1) & 0xFFFF

                # Upper pixel: colour from bits 7-6, palette from bits 3-2
                c = (d & 0xC0) >> 6
                if c != 0:
                    p = ((palette_no >> 2) & 0x04) | ((d & 0x0C) >> 2)
                    val = (p << 2) | c
                    line_ram[hpos & 0x1FF] = val
                    line_ram[(hpos + 1) & 0x1FF] = val
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                    line_ram[(hpos + 1) & 0x1FF] = 0
                hpos += 2

                # Lower pixel: colour from bits 5-4, palette from bits 1-0
                c = (d & 0x30) >> 4
                if c != 0:
                    p = ((palette_no >> 2) & 0x04) | (d & 0x03)
                    val = (p << 2) | c
                    line_ram[hpos & 0x1FF] = val
                    line_ram[(hpos + 1) & 0x1FF] = val
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                    line_ram[(hpos + 1) & 0x1FF] = 0
                hpos += 2

    # ------------------------------------------------------------------
    # 320A: 1bpp, 8 pixels per byte, single colour from palette
    # ------------------------------------------------------------------

    def _build_line_ram_320a(self, graphaddr: int) -> None:
        """320A mode: 1 bit per pixel, 8 pixels per byte.

        Set bits write ``palette_no | 2`` (palette colour 2) to
        line-RAM.  Clear bits are transparent unless Kangaroo mode is
        active, in which case they write zero.
        """
        color = self._palette_no | 2
        hpos = self._hpos << 1
        dataaddr = (graphaddr + (self._offset << 8)) & 0xFFFF

        line_ram = self._line_ram
        kangaroo = self._kangaroo
        holey = self._holey
        width = self._width
        ind_mode = self._ind_mode
        offset = self._offset
        charbase = self._registers[CHARBASE]

        for i in range(width):
            if ind_mode:
                dataaddr = self._word(
                    self._dma_read(graphaddr + i),
                    charbase + offset,
                )

            if ((holey == 0x02 and (dataaddr & 0x9000) == 0x9000)
                    or (holey == 0x01
                        and (dataaddr & 0x8800) == 0x8800)):
                hpos += 8
                dataaddr = (dataaddr + 1) & 0xFFFF
                continue

            d = self._dma_read(dataaddr)
            dataaddr = (dataaddr + 1) & 0xFFFF

            if (d & 0x80) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            if (d & 0x40) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            if (d & 0x20) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            if (d & 0x10) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            if (d & 0x08) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            if (d & 0x04) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            if (d & 0x02) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            if (d & 0x01) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

    # ------------------------------------------------------------------
    # 320B: 2bpp interleaved colour bits, 4 pixels per byte
    # ------------------------------------------------------------------

    def _build_line_ram_320b(self, graphaddr: int) -> None:
        """320B write-mode: 2-bit colour with interleaved bit encoding.

        Each byte encodes 4 pixels.  The colour bits for each pixel
        are interleaved across the upper and lower nibbles:

            pixel 0 = (d7 << 1) | d3
            pixel 1 = (d6 << 1) | d2
            pixel 2 = (d5 << 1) | d1
            pixel 3 = (d4 << 1) | d0

        Transparency and Kangaroo handling follows special rules tied
        to the high/low nibble groupings.  Thanks to Matthias Luedtke
        and Eckhard Stolberg for the correct behaviour here.
        """
        indbytes = 2 if (self._ind_mode and self._cwidth) else 1
        hpos = self._hpos << 1
        dataaddr = (graphaddr + (self._offset << 8)) & 0xFFFF

        line_ram = self._line_ram
        palette_no = self._palette_no
        kangaroo = self._kangaroo
        holey = self._holey
        width = self._width
        ind_mode = self._ind_mode
        offset = self._offset
        charbase = self._registers[CHARBASE]

        for i in range(width):
            if ind_mode:
                dataaddr = self._word(
                    self._dma_read(graphaddr + i),
                    charbase + offset,
                )

            for _j in range(indbytes):
                if ((holey == 0x02 and (dataaddr & 0x9000) == 0x9000)
                        or (holey == 0x01
                            and (dataaddr & 0x8800) == 0x8800)):
                    hpos += 4
                    dataaddr = (dataaddr + 1) & 0xFFFF
                    continue

                d = self._dma_read(dataaddr)
                dataaddr = (dataaddr + 1) & 0xFFFF

                # Pixel 0: colour from d7,d3 -- group check uses d7,d6
                c = ((d & 0x80) >> 6) | ((d & 0x08) >> 3)
                if c != 0:
                    if (d & 0xC0) != 0 or kangaroo:
                        line_ram[hpos & 0x1FF] = palette_no | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                elif (d & 0xCC) != 0:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 1: colour from d6,d2 -- group check uses d7,d6
                c = ((d & 0x40) >> 5) | ((d & 0x04) >> 2)
                if c != 0:
                    if (d & 0xC0) != 0 or kangaroo:
                        line_ram[hpos & 0x1FF] = palette_no | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                elif (d & 0xCC) != 0:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 2: colour from d5,d1 -- group check uses d5,d4
                c = ((d & 0x20) >> 4) | ((d & 0x02) >> 1)
                if c != 0:
                    if (d & 0x30) != 0 or kangaroo:
                        line_ram[hpos & 0x1FF] = palette_no | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                elif (d & 0x33) != 0:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 3: colour from d4,d0 -- group check uses d5,d4
                c = ((d & 0x10) >> 3) | (d & 0x01)
                if c != 0:
                    if (d & 0x30) != 0 or kangaroo:
                        line_ram[hpos & 0x1FF] = palette_no | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                elif (d & 0x33) != 0:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

    # ------------------------------------------------------------------
    # 320C: write-mode, per-pixel palette, 4 pixels per byte
    # ------------------------------------------------------------------

    def _build_line_ram_320c(self, graphaddr: int) -> None:
        """320C write-mode: 1bpp with per-pixel palette selection.

        Each byte is split into two pairs of 2 pixels.  The upper pair
        (bits 7-6) uses bits 3-2 for its palette low bits; the lower
        pair (bits 5-4) uses bits 1-0.  The high palette bit comes from
        the DL header.  Set bits write ``(palette << 2) | 2``.
        """
        hpos = self._hpos << 1
        dataaddr = (graphaddr + (self._offset << 8)) & 0xFFFF

        line_ram = self._line_ram
        palette_no = self._palette_no
        kangaroo = self._kangaroo
        holey = self._holey
        width = self._width
        ind_mode = self._ind_mode
        offset = self._offset
        charbase = self._registers[CHARBASE]

        for i in range(width):
            if ind_mode:
                dataaddr = self._word(
                    self._dma_read(graphaddr + i),
                    charbase + offset,
                )

            if ((holey == 0x02 and (dataaddr & 0x9000) == 0x9000)
                    or (holey == 0x01
                        and (dataaddr & 0x8800) == 0x8800)):
                hpos += 4
                dataaddr = (dataaddr + 1) & 0xFFFF
                continue

            d = self._dma_read(dataaddr)
            dataaddr = (dataaddr + 1) & 0xFFFF

            # Upper pair: palette from bits 3-2 combined with DL palette MSB
            color = ((((d & 0x0C) >> 2)
                      | ((palette_no >> 2) & 0x04)) << 2) | 2

            if (d & 0x80) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            if (d & 0x40) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            # Lower pair: palette from bits 1-0 combined with DL palette MSB
            color = (((d & 0x03)
                      | ((palette_no >> 2) & 0x04)) << 2) | 2

            if (d & 0x20) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

            if (d & 0x10) != 0:
                line_ram[hpos & 0x1FF] = color
            elif kangaroo:
                line_ram[hpos & 0x1FF] = 0
            hpos += 1

    # ------------------------------------------------------------------
    # 320D: 2bpp interleaved palette bits, 8 pixels per byte
    # ------------------------------------------------------------------

    def _build_line_ram_320d(self, graphaddr: int) -> None:
        """320D mode: 1bpp data with 2-bit palette index interleaved
        from the DL header palette field.

        Each byte produces 8 pixels.  For each pixel, one bit comes
        from the graphic data and the other from an alternating
        selection of palette_no bits, yielding a 2-bit colour index
        combined with a reduced palette selector.
        """
        indbytes = 2 if (self._ind_mode and self._cwidth) else 1
        hpos = self._hpos << 1
        dataaddr = (graphaddr + (self._offset << 8)) & 0xFFFF

        line_ram = self._line_ram
        palette_no = self._palette_no
        kangaroo = self._kangaroo
        holey = self._holey
        width = self._width
        ind_mode = self._ind_mode
        offset = self._offset
        charbase = self._registers[CHARBASE]

        for i in range(width):
            if ind_mode:
                dataaddr = self._word(
                    self._dma_read(graphaddr + i),
                    charbase + offset,
                )

            for _j in range(indbytes):
                if ((holey == 0x02 and (dataaddr & 0x9000) == 0x9000)
                        or (holey == 0x01
                            and (dataaddr & 0x8800) == 0x8800)):
                    hpos += 8
                    dataaddr = (dataaddr + 1) & 0xFFFF
                    continue

                d = self._dma_read(dataaddr)
                dataaddr = (dataaddr + 1) & 0xFFFF

                # Pixel 0
                c = ((d & 0x80) >> 6) | (((palette_no >> 2) & 2) >> 1)
                if c != 0:
                    line_ram[hpos & 0x1FF] = (palette_no & 0x10) | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 1
                c = ((d & 0x40) >> 5) | ((palette_no >> 2) & 1)
                if c != 0:
                    line_ram[hpos & 0x1FF] = (palette_no & 0x10) | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 2
                c = ((d & 0x20) >> 4) | (((palette_no >> 2) & 2) >> 1)
                if c != 0:
                    line_ram[hpos & 0x1FF] = (palette_no & 0x10) | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 3
                c = ((d & 0x10) >> 3) | ((palette_no >> 2) & 1)
                if c != 0:
                    line_ram[hpos & 0x1FF] = (palette_no & 0x10) | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 4
                c = ((d & 0x08) >> 2) | (((palette_no >> 2) & 2) >> 1)
                if c != 0:
                    line_ram[hpos & 0x1FF] = (palette_no & 0x10) | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 5
                c = ((d & 0x04) >> 1) | ((palette_no >> 2) & 1)
                if c != 0:
                    line_ram[hpos & 0x1FF] = (palette_no & 0x10) | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 6
                c = (d & 0x02) | (((palette_no >> 2) & 2) >> 1)
                if c != 0:
                    line_ram[hpos & 0x1FF] = (palette_no & 0x10) | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

                # Pixel 7
                c = ((d & 0x01) << 1) | ((palette_no >> 2) & 1)
                if c != 0:
                    line_ram[hpos & 0x1FF] = (palette_no & 0x10) | c
                elif kangaroo:
                    line_ram[hpos & 0x1FF] = 0
                hpos += 1

    # ------------------------------------------------------------------
    # Line-RAM output to frame buffer
    # ------------------------------------------------------------------

    def _output_line_ram(self) -> None:
        """Convert line-RAM colour indices to palette register values
        and write them into the video frame buffer.

        Each line-RAM entry is an index.  When the low two bits are
        zero the pixel is treated as background
        (``registers[BACKGRND]``); otherwise the full index selects
        a palette colour register at ``registers[BACKGRND + index]``.

        The output targets scanline ``(current_scanline + 1)``,
        wrapping within the video buffer.  The line-RAM is cleared to
        zero as it is consumed.
        """
        fb = self._m.frame_buffer
        vbuf = fb.video_buffer
        vbuf_len = len(vbuf)
        visible_pitch = fb.visible_pitch
        line_ram = self._line_ram
        registers = self._registers

        fbi = ((self._scanline + 1) * visible_pitch) % vbuf_len

        for i in range(visible_pitch):
            color_index = line_ram[i]

            if (color_index & 3) == 0:
                vbuf[fbi] = registers[BACKGRND]
            else:
                vbuf[fbi] = registers[BACKGRND + color_index]

            line_ram[i] = 0

            fbi += 1
            if fbi == vbuf_len:
                fbi = 0

    # ------------------------------------------------------------------
    # Register peek (read)
    # ------------------------------------------------------------------

    def _peek(self, addr: int) -> int:
        """Handle a CPU read of a Maria register."""
        addr &= 0x3F
        mi = self._m.input_state

        if addr == MSTAT:
            sl = self._scanline
            if (sl < self._first_visible_scanline
                    or sl >= self._last_visible_scanline):
                return 0x80   # VBLANK ON
            return 0x00       # VBLANK OFF

        elif addr == INPT0:
            # player 1, button R
            return (0x80
                    if mi.sample_captured_controller_action_state(
                        0, ControllerAction.Trigger)
                    else 0x00)

        elif addr == INPT1:
            # player 1, button L
            return (0x80
                    if mi.sample_captured_controller_action_state(
                        0, ControllerAction.Trigger2)
                    else 0x00)

        elif addr == INPT2:
            # player 2, button R
            return (0x80
                    if mi.sample_captured_controller_action_state(
                        1, ControllerAction.Trigger)
                    else 0x00)

        elif addr == INPT3:
            # player 2, button L
            return (0x80
                    if mi.sample_captured_controller_action_state(
                        1, ControllerAction.Trigger2)
                    else 0x00)

        elif addr == INPT4:
            # player 1, button L/R (latched -- note inverted polarity)
            return 0x00 if self._sample_inpt_latched(4) else 0x80

        elif addr == INPT5:
            # player 2, button L/R (latched -- note inverted polarity)
            return 0x00 if self._sample_inpt_latched(5) else 0x80

        else:
            return self._registers[addr]

    # ------------------------------------------------------------------
    # Register poke (write)
    # ------------------------------------------------------------------

    def _poke(self, addr: int, data: int) -> None:
        """Handle a CPU write to a Maria register."""
        addr &= 0x3F
        data &= 0xFF

        if addr == INPTCTRL:
            # INPUT PORT CONTROL
            # D0: lock mode (once set, no further changes until power cycle)
            # D1: 0=disable MARIA; 1=enable MARIA (also enables system RAM)
            # D2: 0=enable BIOS; 1=disable BIOS and enable cartridge
            # D3: 0=MARIA video output; 1=TIA video output
            if self._ctrl_lock:
                return

            self._ctrl_lock = (data & (1 << 0)) != 0
            bios_disable = (data & (1 << 2)) != 0

            if bios_disable:
                self._m.swap_out_bios()
            else:
                self._m.swap_in_bios()

        elif addr == WSYNC:
            # Request a CPU preemption to service the WSYNC delay.
            self._m.cpu.emulator_preempt_request = True

        elif addr == CTRL:
            self._color_kill = (data & 0x80) != 0
            self._dma_enabled = (data & 0x60) == 0x40
            self._cwidth = (data & 0x10) != 0
            self._bcntl = (data & 0x08) != 0
            self._kangaroo = (data & 0x04) != 0
            self._rm = data & 0x03

        elif addr == MSTAT:
            # MSTAT writes are ignored.
            pass

        elif addr in (CHARBASE, DPPH, DPPL):
            self._registers[addr] = data

        elif addr in _PALETTE_ADDRS:
            self._registers[addr] = data

        elif addr in _AUDIO_ADDRS:
            self._tia_sound.update(addr, data)

        elif addr == OFFSET:
            # Reserved for future expansion; writes are noted but ignored.
            pass

        else:
            self._registers[addr] = data

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def _sample_inpt_latched(self, inpt: int) -> bool:
        """Sample an INPT4/INPT5 latched trigger input.

        The result depends on the controller type plugged into the
        corresponding jack:

        * **Joystick**: returns the Trigger action state.
        * **ProLineJoystick**: returns Trigger OR Trigger2, subject to
          PIA port-B direction/output gating.
        * **Lightgun**: performs scanline/hpos timing comparison against
          the sampled gun position.

        Returns ``True`` when the trigger is active (button pressed).
        """
        mi = self._m.input_state
        player_no = inpt - 4

        if player_no == 0:
            jack = mi.left_controller_jack
        else:
            jack = mi.right_controller_jack

        if jack == Controller.Joystick:
            return mi.sample_captured_controller_action_state(
                player_no, ControllerAction.Trigger
            )

        elif jack == Controller.ProLineJoystick:
            portb_line = 4 << (player_no << 1)
            if ((self._m.pia.ddrb & portb_line) != 0
                    and (self._m.pia.written_port_b & portb_line) == 0):
                return False
            return (
                mi.sample_captured_controller_action_state(
                    player_no, ControllerAction.Trigger)
                or mi.sample_captured_controller_action_state(
                    player_no, ControllerAction.Trigger2)
            )

        elif jack == Controller.Lightgun:
            # Lightgun timing comparison.
            #
            # Track the number of samples this frame, the time of the
            # first sample, and capture the lightgun location.
            cpu = self._m.cpu

            if self._lightgun_first_sample_cpu_clock == 0:
                self._lightgun_first_sample_cpu_clock = cpu.clock
                self._lightgun_frame_samples = 0
                sl, hp = mi.sample_captured_light_gun_position(player_no)
                self._lightgun_sampled_scanline = sl
                self._lightgun_sampled_visible_hpos = hp

            self._lightgun_frame_samples += 1

            # Magic adjustment factor to account for timing impact of
            # successive lightgun reads (slow memory accesses).
            # Obtained through trial-and-error.
            magic_adjustment_factor = 2.135

            first_lg_sample_maria_clock = int(
                (self._lightgun_first_sample_cpu_clock
                 - self._start_of_frame_cpu_clock) << 2
            )
            maria_clocks_since_first = int(
                (cpu.clock - self._lightgun_first_sample_cpu_clock) << 2
            )
            adjustment_maria_clocks = round(
                self._lightgun_frame_samples * magic_adjustment_factor
            )
            actual_maria_frame_clock = (
                first_lg_sample_maria_clock
                + maria_clocks_since_first
                + adjustment_maria_clocks
            )
            actual_scanline = actual_maria_frame_clock // 456
            actual_hpos = actual_maria_frame_clock % 456

            # Lightgun sampling looks intended to begin at the start of
            # the scanline.  Compensate with a magic constant since we
            # are always off by a fixed amount.
            actual_hpos -= 62
            if actual_hpos < 0:
                actual_hpos += 456
                actual_scanline -= 1

            sampled_scanline = self._lightgun_sampled_scanline
            sampled_visible_hpos = self._lightgun_sampled_visible_hpos

            # The gun sees more than a single pixel (more like a circle
            # or oval) and triggers sooner accordingly.  These
            # adjustments were obtained through trial-and-error.
            if self._is_pal:
                sampled_scanline -= 19
            else:
                sampled_scanline -= 16
                sampled_visible_hpos += 4

            # 136 = HBLANK clocks
            return (actual_scanline >= sampled_scanline
                    and actual_hpos >= sampled_visible_hpos + 136)

        return False

    # ------------------------------------------------------------------
    # DLL / DMA helpers
    # ------------------------------------------------------------------

    def _consume_next_dll_entry(self) -> None:
        """Read the next 3-byte Display List List entry.

        A DLL entry contains:
          byte 0: DLI(7) | Holey(6-5) | Offset(3-0)
          byte 1: High byte of Display List address
          byte 2: Low byte of Display List address

        Updates ``_dl``, ``_offset``, and ``_holey``.
        If the DLI flag is set, triggers an NMI on the CPU.
        """
        dll0 = self._dma_read(self._dll)
        self._dll = (self._dll + 1) & 0xFFFF
        dll1 = self._dma_read(self._dll)
        self._dll = (self._dll + 1) & 0xFFFF
        dll2 = self._dma_read(self._dll)
        self._dll = (self._dll + 1) & 0xFFFF

        dli = (dll0 & 0x80) != 0
        self._holey = (dll0 & 0x60) >> 5
        self._offset = dll0 & 0x0F

        # Update current Display List pointer
        self._dl = self._word(dll2, dll1)

        if dli:
            self._m.cpu.nmi_interrupt_request = True
            # DMA TIMING: One tick between DMA Shutdown and DLI
            self._dma_clocks += 1

    def _dma_read(self, addr: int) -> int:
        """Perform a Maria DMA read from the address space.

        Sets the ``maria_read`` latch on the address space so that the
        TIA snooper can distinguish DMA traffic from CPU traffic.

        Parameters
        ----------
        addr :
            16-bit address to read from.

        Returns
        -------
        int
            The byte value (0-255) at the given address.
        """
        addr &= 0xFFFF
        self._m.mem.maria_read = 1
        rb = self._m.mem[addr]
        self._m.mem.maria_read = 0
        return rb

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _word(lsb: int, msb: int) -> int:
        """Combine two bytes into a 16-bit word (little-endian).

        Both arguments are masked to 8 bits before combining, matching
        the C# ``WORD((byte)lsb, (byte)msb)`` semantics.
        """
        return ((msb & 0xFF) << 8) | (lsb & 0xFF)

    # ------------------------------------------------------------------
    # Serialisation support
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return a serialisable snapshot of Maria's internal state."""
        return {
            "line_ram": bytes(self._line_ram),
            "registers": bytes(self._registers),
            "wm": self._wm,
            "dll": self._dll,
            "dl": self._dl,
            "offset": self._offset,
            "holey": self._holey,
            "width": self._width,
            "hpos": self._hpos,
            "palette_no": self._palette_no,
            "ind_mode": self._ind_mode,
            "ctrl_lock": self._ctrl_lock,
            "dma_enabled": self._dma_enabled,
            "color_kill": self._color_kill,
            "cwidth": self._cwidth,
            "bcntl": self._bcntl,
            "kangaroo": self._kangaroo,
            "rm": self._rm,
        }

    def restore_snapshot(self, snap: dict) -> None:
        """Restore Maria's internal state from a previous snapshot."""
        self._line_ram[:] = snap["line_ram"]
        self._registers[:] = snap["registers"]
        self._wm = snap["wm"]
        self._dll = snap["dll"]
        self._dl = snap["dl"]
        self._offset = snap["offset"]
        self._holey = snap["holey"]
        self._width = snap["width"]
        self._hpos = snap["hpos"]
        self._palette_no = snap["palette_no"]
        self._ind_mode = snap["ind_mode"]
        self._ctrl_lock = snap["ctrl_lock"]
        self._dma_enabled = snap["dma_enabled"]
        self._color_kill = snap["color_kill"]
        self._cwidth = snap["cwidth"]
        self._bcntl = snap["bcntl"]
        self._kangaroo = snap["kangaroo"]
        self._rm = snap["rm"]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Maria("
            f"dma={'on' if self._dma_enabled else 'off'}, "
            f"rm={self._rm}, "
            f"scanline={self._scanline}, "
            f"ctrl_lock={self._ctrl_lock})"
        )
