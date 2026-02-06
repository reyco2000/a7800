"""
Atari 2600 cartridge mappers for EMU7800.
Ported from the C# Cart*K / Cart*KR classes.

All 2600 carts sit in the 4 KB window at $1000-$1FFF of the 13-bit address
space.  The 12-bit offset (``addr & 0xFFF``) is used internally to select
ROM bytes and detect bank-switch hotspot accesses.

Mappers
-------
CartA2K      -- 2 KB flat ROM (addr & 0x7FF).
CartA4K      -- 4 KB flat ROM (addr & 0xFFF).
CartA8K      -- 8 KB  / F8  bank-switch (2 x 4 KB).
CartA8KR     -- 8 KB  / F8SC + 128 B RAM.
CartA16K     -- 16 KB / F6  bank-switch (4 x 4 KB).
CartA16KR    -- 16 KB / F6SC + 128 B RAM.
CartA32K     -- 32 KB / F4  bank-switch (8 x 4 KB).
CartA32KR    -- 32 KB / F4SC + 128 B RAM.
CartDC8K     -- 8 KB  Activision FE scheme.
CartPB8K     -- 8 KB  Parker Brothers E0 scheme.
CartTV8K     -- 8 KB  Tigervision 3F scheme.
CartCBS12K   -- 12 KB CBS RAM Plus / FA scheme + 256 B RAM.
CartMN16K    -- 16 KB M-Network / E7 scheme + 1 KB RAM (4 x 256 B).
CartDPC      -- 10 KB DPC (Pitfall II) -- 8 KB ROM + 2 KB display data.
"""

from __future__ import annotations

from emu7800.core.carts.cart import Cart


# ======================================================================
# CartA2K -- 2 KB
# ======================================================================

class CartA2K(Cart):
    """2 KB standard Atari 2600 cartridge.  Address mirrored via & 0x7FF."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=2048)

    def __getitem__(self, addr: int) -> int:
        return self.rom[addr & 0x7FF]

    def __setitem__(self, addr: int, value: int) -> None:
        pass  # ROM is read-only


# ======================================================================
# CartA4K -- 4 KB
# ======================================================================

class CartA4K(Cart):
    """4 KB standard Atari 2600 cartridge.  No bank switching."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=4096)

    def __getitem__(self, addr: int) -> int:
        return self.rom[addr & 0xFFF]

    def __setitem__(self, addr: int, value: int) -> None:
        pass


# ======================================================================
# CartA8K -- 8 KB / F8
# ======================================================================

class CartA8K(Cart):
    """8 KB bank-switched cartridge (F8 scheme).

    Two 4 KB banks.  Accessing hotspot addresses in the last page triggers
    bank switching:

    * $xFF8 -> bank 0  (ROM offset 0x0000)
    * $xFF9 -> bank 1  (ROM offset 0x1000)
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=8192)
        self._bank_base_addr: int = 0

    def reset(self) -> None:
        # Power-on defaults to last bank so reset vector is correct.
        self._bank_base_addr = len(self.rom) - 0x1000

    def _check_bank_switch(self, addr12: int) -> None:
        if addr12 == 0xFF8:
            self._bank_base_addr = 0x0000
        elif addr12 == 0xFF9:
            self._bank_base_addr = 0x1000

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        return self.rom[self._bank_base_addr + addr12]

    def __setitem__(self, addr: int, value: int) -> None:
        self._check_bank_switch(addr & 0xFFF)


# ======================================================================
# CartA8KR -- 8 KB / F8SC + 128 B RAM
# ======================================================================

class CartA8KR(Cart):
    """8 KB bank-switched + 128 bytes on-cart RAM (F8SC scheme).

    Bank switching identical to :class:`CartA8K`.  Additionally, 128 bytes
    of RAM are accessible inside the cart window:

    * Write port: $x000 - $x07F  (write *value* to RAM[addr & 0x7F])
    * Read  port: $x080 - $x0FF  (read  RAM[addr & 0x7F])
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=8192)
        self._bank_base_addr: int = 0
        self._ram: bytearray = bytearray(128)

    def reset(self) -> None:
        self._bank_base_addr = len(self.rom) - 0x1000
        for i in range(128):
            self._ram[i] = 0

    def _check_bank_switch(self, addr12: int) -> None:
        if addr12 == 0xFF8:
            self._bank_base_addr = 0x0000
        elif addr12 == 0xFF9:
            self._bank_base_addr = 0x1000

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        # RAM read port
        if 0x80 <= addr12 <= 0xFF:
            return self._ram[addr12 & 0x7F]
        return self.rom[self._bank_base_addr + addr12]

    def __setitem__(self, addr: int, value: int) -> None:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        # RAM write port
        if addr12 <= 0x7F:
            self._ram[addr12] = value & 0xFF


# ======================================================================
# CartA16K -- 16 KB / F6
# ======================================================================

class CartA16K(Cart):
    """16 KB bank-switched cartridge (F6 scheme).

    Four 4 KB banks.  Hotspots:

    * $xFF6 -> bank 0
    * $xFF7 -> bank 1
    * $xFF8 -> bank 2
    * $xFF9 -> bank 3
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=16384)
        self._bank_base_addr: int = 0

    def reset(self) -> None:
        self._bank_base_addr = len(self.rom) - 0x1000

    def _check_bank_switch(self, addr12: int) -> None:
        if addr12 == 0xFF6:
            self._bank_base_addr = 0x0000
        elif addr12 == 0xFF7:
            self._bank_base_addr = 0x1000
        elif addr12 == 0xFF8:
            self._bank_base_addr = 0x2000
        elif addr12 == 0xFF9:
            self._bank_base_addr = 0x3000

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        return self.rom[self._bank_base_addr + addr12]

    def __setitem__(self, addr: int, value: int) -> None:
        self._check_bank_switch(addr & 0xFFF)


# ======================================================================
# CartA16KR -- 16 KB / F6SC + 128 B RAM
# ======================================================================

class CartA16KR(Cart):
    """16 KB bank-switched + 128 bytes RAM (F6SC scheme).

    Bank switching identical to :class:`CartA16K`.  128-byte RAM at the
    same addresses as :class:`CartA8KR`.
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=16384)
        self._bank_base_addr: int = 0
        self._ram: bytearray = bytearray(128)

    def reset(self) -> None:
        self._bank_base_addr = len(self.rom) - 0x1000
        for i in range(128):
            self._ram[i] = 0

    def _check_bank_switch(self, addr12: int) -> None:
        if addr12 == 0xFF6:
            self._bank_base_addr = 0x0000
        elif addr12 == 0xFF7:
            self._bank_base_addr = 0x1000
        elif addr12 == 0xFF8:
            self._bank_base_addr = 0x2000
        elif addr12 == 0xFF9:
            self._bank_base_addr = 0x3000

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        if 0x80 <= addr12 <= 0xFF:
            return self._ram[addr12 & 0x7F]
        return self.rom[self._bank_base_addr + addr12]

    def __setitem__(self, addr: int, value: int) -> None:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        if addr12 <= 0x7F:
            self._ram[addr12] = value & 0xFF


# ======================================================================
# CartA32K -- 32 KB / F4
# ======================================================================

class CartA32K(Cart):
    """32 KB bank-switched cartridge (F4 scheme).

    Eight 4 KB banks.  Hotspots at $xFF4-$xFFB select banks 0-7.
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=32768)
        self._bank_base_addr: int = 0

    def reset(self) -> None:
        self._bank_base_addr = len(self.rom) - 0x1000

    def _check_bank_switch(self, addr12: int) -> None:
        if 0xFF4 <= addr12 <= 0xFFB:
            self._bank_base_addr = (addr12 - 0xFF4) * 0x1000

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        return self.rom[self._bank_base_addr + addr12]

    def __setitem__(self, addr: int, value: int) -> None:
        self._check_bank_switch(addr & 0xFFF)


# ======================================================================
# CartA32KR -- 32 KB / F4SC + 128 B RAM
# ======================================================================

class CartA32KR(Cart):
    """32 KB bank-switched + 128 bytes RAM (F4SC scheme)."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=32768)
        self._bank_base_addr: int = 0
        self._ram: bytearray = bytearray(128)

    def reset(self) -> None:
        self._bank_base_addr = len(self.rom) - 0x1000
        for i in range(128):
            self._ram[i] = 0

    def _check_bank_switch(self, addr12: int) -> None:
        if 0xFF4 <= addr12 <= 0xFFB:
            self._bank_base_addr = (addr12 - 0xFF4) * 0x1000

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        if 0x80 <= addr12 <= 0xFF:
            return self._ram[addr12 & 0x7F]
        return self.rom[self._bank_base_addr + addr12]

    def __setitem__(self, addr: int, value: int) -> None:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        if addr12 <= 0x7F:
            self._ram[addr12] = value & 0xFF


# ======================================================================
# CartDC8K -- 8 KB Activision FE scheme
# ======================================================================

class CartDC8K(Cart):
    """8 KB Activision ``Display Cart'' (FE scheme).

    Two 4 KB banks.  The bank is selected by monitoring CPU accesses to
    address $01FE (stack page -- triggered by JSR / RTS).  Bit 5 (D5) of
    the data byte on the bus selects the bank:

    * D5 = 1 -> bank 0  (ROM[0x0000:0x1000])
    * D5 = 0 -> bank 1  (ROM[0x1000:0x2000])

    On a *write* to $01FE (e.g. JSR pushing the return address), the
    written byte supplies D5 and the bank switches immediately.

    On a *read* from $01FE (e.g. RTS pulling the return address), the
    byte previously stored in RAM supplies D5.  Since the cart is called
    as the snooper *before* the actual RAM device, it uses the
    ``data_bus_state`` of the address space (which still holds the value
    from the most recent bus transaction) as an approximation.

    Because the trigger address is outside the cart ROM window, the cart
    sets ``request_snooping = True`` so the machine installs it as the
    address-space snooper.
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=8192)
        self.request_snooping = True
        self._bank_base_addr: int = 0

    def reset(self) -> None:
        self._bank_base_addr = 0

    def _apply_bank_switch(self, data: int) -> None:
        """Use D5 of *data* to select the bank."""
        if data & 0x20:
            self._bank_base_addr = 0x0000
        else:
            self._bank_base_addr = 0x1000

    def __getitem__(self, addr: int) -> int:
        addr13 = addr & 0x1FFF
        if addr13 == 0x01FE:
            # Snooped read of $01FE (e.g. RTS).  Use the data bus state
            # from the machine if available; otherwise the ROM byte at the
            # current bank (which is what the snooper return value would be).
            if self.m is not None and hasattr(self.m, "addr_space"):
                self._apply_bank_switch(self.m.addr_space.data_bus_state)
        return self.rom[self._bank_base_addr + (addr & 0xFFF)]

    def __setitem__(self, addr: int, value: int) -> None:
        addr13 = addr & 0x1FFF
        if addr13 == 0x01FE:
            self._apply_bank_switch(value & 0xFF)


# ======================================================================
# CartPB8K -- 8 KB Parker Brothers E0 scheme
# ======================================================================

class CartPB8K(Cart):
    """8 KB Parker Brothers cartridge (E0 scheme).

    The ROM is divided into eight 1 KB banks.  The 4 KB cart window is
    split into four 1 KB segments:

    * Segment 0 ($x000-$x3FF): switchable via hotspots $xFE0-$xFE7
    * Segment 1 ($x400-$x7FF): switchable via hotspots $xFE8-$xFEF
    * Segment 2 ($x800-$xBFF): switchable via hotspots $xFF0-$xFF7
    * Segment 3 ($xC00-$xFFF): fixed to bank 7 (last 1 KB)

    Accessing a hotspot selects the bank whose number equals ``addr & 7``.
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=8192)
        # ROM offsets for the four 1 KB segments
        self._seg_base: list[int] = [0x0000, 0x0400, 0x0800, 0x1C00]

    def reset(self) -> None:
        self._seg_base = [0x0000, 0x0400, 0x0800, 0x1C00]

    def _check_bank_switch(self, addr12: int) -> None:
        if 0xFE0 <= addr12 <= 0xFE7:
            self._seg_base[0] = (addr12 & 7) * 0x400
        elif 0xFE8 <= addr12 <= 0xFEF:
            self._seg_base[1] = (addr12 & 7) * 0x400
        elif 0xFF0 <= addr12 <= 0xFF7:
            self._seg_base[2] = (addr12 & 7) * 0x400

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        segment = (addr12 >> 10) & 3
        offset = addr12 & 0x3FF
        return self.rom[self._seg_base[segment] + offset]

    def __setitem__(self, addr: int, value: int) -> None:
        self._check_bank_switch(addr & 0xFFF)


# ======================================================================
# CartTV8K -- 8 KB Tigervision 3F scheme
# ======================================================================

class CartTV8K(Cart):
    """8 KB Tigervision cartridge (3F scheme).

    The ROM contains up to four 2 KB banks.  The cart window is split:

    * Lower half ($x000-$x7FF): switchable 2 KB bank
    * Upper half ($x800-$xFFF): fixed to the *last* 2 KB bank

    Bank switching occurs when the CPU writes to TIA address $003F.
    Because that address is outside the cart window, the cart sets
    ``request_snooping = True``.  The low bits of the written byte
    select the bank for the lower half.
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=8192)
        self.request_snooping = True
        self._bank_base_addr: int = 0
        self._n_banks: int = 0

    def reset(self) -> None:
        self._n_banks = max(1, len(self.rom) // 2048)
        self._bank_base_addr = 0

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        if addr12 >= 0x800:
            # Upper half: fixed last 2 KB bank
            return self.rom[len(self.rom) - 2048 + (addr12 & 0x7FF)]
        # Lower half: switchable bank
        return self.rom[self._bank_base_addr + (addr12 & 0x7FF)]

    def __setitem__(self, addr: int, value: int) -> None:
        # Writes to $003F (TIA range): low bits select the bank.
        # On the 2600 bus, TIA write registers are decoded when the
        # address has bits 12 and 7 clear.  Address $3F matches that.
        if (addr & 0x1FFF) < 0x1000 and (addr & 0x3F) == 0x3F:
            self._bank_base_addr = ((value & 0xFF) % self._n_banks) * 2048


# ======================================================================
# CartCBS12K -- 12 KB CBS RAM Plus / FA scheme
# ======================================================================

class CartCBS12K(Cart):
    """12 KB CBS RAM Plus cartridge (FA scheme).

    Three 4 KB banks + 256 bytes of on-cart RAM.

    Bank switching hotspots:

    * $xFF8 -> bank 0
    * $xFF9 -> bank 1
    * $xFFA -> bank 2

    RAM is mapped inside the cart window:

    * Write port: $x000 - $x0FF  (write to RAM[addr & 0xFF])
    * Read  port: $x100 - $x1FF  (read  from RAM[addr & 0xFF])
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=12288)
        self._bank_base_addr: int = 0
        self._ram: bytearray = bytearray(256)

    def reset(self) -> None:
        self._bank_base_addr = len(self.rom) - 0x1000
        for i in range(256):
            self._ram[i] = 0

    def _check_bank_switch(self, addr12: int) -> None:
        if addr12 == 0xFF8:
            self._bank_base_addr = 0x0000
        elif addr12 == 0xFF9:
            self._bank_base_addr = 0x1000
        elif addr12 == 0xFFA:
            self._bank_base_addr = 0x2000

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        # RAM read port
        if 0x100 <= addr12 <= 0x1FF:
            return self._ram[addr12 & 0xFF]
        return self.rom[self._bank_base_addr + addr12]

    def __setitem__(self, addr: int, value: int) -> None:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        # RAM write port
        if addr12 <= 0xFF:
            self._ram[addr12] = value & 0xFF


# ======================================================================
# CartMN16K -- 16 KB M-Network / E7 scheme
# ======================================================================

class CartMN16K(Cart):
    """16 KB M-Network cartridge (E7 scheme).

    The ROM is divided into eight 2 KB banks.  The 4 KB cart window is
    split into a lower half and an upper half:

    **Lower 2 KB** ($x000-$x7FF):
        Switchable among ROM banks 0-6.  Bank 7 can also be selected but
        is primarily used in the upper half.  Hotspots $xFE0-$xFE6 select
        banks 0-6 for this region; $xFE7 selects bank 7.

    **Upper 2 KB** ($x800-$xFFF):
        * $x800-$x8FF: RAM write port (write to RAM bank[addr & 0xFF])
        * $x900-$x9FF: RAM read port  (read from RAM bank[addr & 0xFF])
        * $xA00-$xFFF: fixed ROM from the last 1.5 KB of bank 7

    There are four 256-byte RAM banks (1 KB total).  Hotspots $xFE8-$xFEB
    select the active RAM bank (0-3).
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=16384)
        self._rom_bank_base: int = 0       # switchable ROM bank offset
        self._ram_bank: int = 0            # active RAM bank (0-3)
        self._ram: bytearray = bytearray(1024)  # 4 x 256 B

    def reset(self) -> None:
        self._rom_bank_base = 0
        self._ram_bank = 0
        for i in range(1024):
            self._ram[i] = 0

    def _check_bank_switch(self, addr12: int) -> None:
        if 0xFE0 <= addr12 <= 0xFE7:
            self._rom_bank_base = (addr12 - 0xFE0) * 0x800
        elif 0xFE8 <= addr12 <= 0xFEB:
            self._ram_bank = addr12 - 0xFE8

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)

        if addr12 < 0x800:
            # Lower 2 KB: switchable ROM bank
            return self.rom[self._rom_bank_base + (addr12 & 0x7FF)]

        if 0x900 <= addr12 <= 0x9FF:
            # RAM read port
            return self._ram[self._ram_bank * 256 + (addr12 & 0xFF)]

        if addr12 >= 0xA00:
            # Fixed ROM: last 1.5 KB of bank 7
            # Bank 7 starts at ROM offset 0x3800.  The 0xA00 region
            # corresponds to the upper 1.5 KB of that bank (offset 0x200).
            return self.rom[0x3800 + 0x200 + (addr12 - 0xA00)]

        # $x800-$x8FF: RAM write zone -- reads here are undefined (return 0).
        return 0

    def __setitem__(self, addr: int, value: int) -> None:
        addr12 = addr & 0xFFF
        self._check_bank_switch(addr12)
        # RAM write port
        if 0x800 <= addr12 <= 0x8FF:
            self._ram[self._ram_bank * 256 + (addr12 & 0xFF)] = value & 0xFF


# ======================================================================
# CartDPC -- DPC chip (Pitfall II)
# ======================================================================

class CartDPC(Cart):
    """DPC (David P. Crane) chip used by *Pitfall II: Lost Caverns*.

    The cartridge contains 8 KB of program ROM (two 4 KB banks using
    standard F8 switching at $xFF8/$xFF9) plus 2 KB of display data that
    is streamed to the TIA through eight hardware data fetchers.

    **Data Fetchers** (0-7):

    Each fetcher has an 11-bit counter that points into the 2 KB display
    data ROM, plus *top* and *bottom* boundary registers and a 1-bit flag.

    Read registers ($x000-$x03F, grouped by ``addr & 0x38``):

    ======= ========================================================
    Group   Function
    ======= ========================================================
    0x00-07 Display data at counter
    0x08-0F Display data AND flag
    0x10-17 Display data AND flag, bitwise complement
    0x18-1F Display data AND flag (alternate)
    0x20-27 Display data at counter (alternate)
    0x28-2F Flag value ($00 or $FF)
    0x30-37 Music data (fetchers 5-7 only)
    0x38-3F Random number (8-bit LFSR)
    ======= ========================================================

    After each display-data read (groups 0-4) the fetcher counter is
    decremented.

    Write registers ($x040-$x07F):

    ======= ========================================================
    Group   Function
    ======= ========================================================
    0x40-47 Set counter top
    0x48-4F Set counter bottom
    0x50-57 Set counter low byte
    0x58-5F Set counter high byte (bits 2:0)
    0x60-67 Enable music mode (bit 4)
    0x68-6F Set random seed
    0x70-77 Reset counter to top value
    ======= ========================================================

    Fetchers 5, 6 and 7 support *music mode*: when enabled, their counters
    are clocked by the frame oscillator to produce audio.
    """

    DISPLAY_DATA_SIZE: int = 2048
    NUM_FETCHERS: int = 8

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        min_size = 8192 + self.DISPLAY_DATA_SIZE
        if len(rom_bytes) < min_size:
            padded = bytearray(min_size)
            padded[: len(rom_bytes)] = rom_bytes
            rom_bytes = bytes(padded)
        # First 8 KB is program ROM.
        self.rom = bytearray(rom_bytes[:8192])
        # Next 2 KB is display data.
        self._display_data: bytearray = bytearray(
            rom_bytes[8192 : 8192 + self.DISPLAY_DATA_SIZE]
        )

        self._bank_base_addr: int = 0

        # Per-fetcher state
        self._tops: list[int] = [0] * self.NUM_FETCHERS
        self._bottoms: list[int] = [0] * self.NUM_FETCHERS
        self._counters: list[int] = [0] * self.NUM_FETCHERS
        self._flags: list[int] = [0] * self.NUM_FETCHERS
        self._music_mode: list[bool] = [False] * self.NUM_FETCHERS

        # 8-bit LFSR random-number generator
        self._random: int = 1

    def reset(self) -> None:
        self._bank_base_addr = 0x1000  # start in last bank
        self._tops = [0] * self.NUM_FETCHERS
        self._bottoms = [0] * self.NUM_FETCHERS
        self._counters = [0] * self.NUM_FETCHERS
        self._flags = [0] * self.NUM_FETCHERS
        self._music_mode = [False] * self.NUM_FETCHERS
        self._random = 1

    # ---- helpers ----

    def _clock_random(self) -> None:
        """Advance the 8-bit LFSR (polynomial x^8+x^6+x^5+x^4+1)."""
        bit = (
            (self._random >> 3)
            ^ (self._random >> 4)
            ^ (self._random >> 5)
            ^ (self._random >> 7)
        ) & 1
        self._random = ((self._random << 1) | bit) & 0xFF
        if self._random == 0:
            self._random = 1

    def _read_display(self, fetcher: int) -> int:
        """Return the display-data byte at the fetcher's current pointer."""
        return self._display_data[self._counters[fetcher] & 0x7FF]

    def _update_flag(self, fetcher: int) -> None:
        """Recompute the fetcher's flag from counter vs top/bottom."""
        low = self._counters[fetcher] & 0xFF
        if low == self._tops[fetcher]:
            self._flags[fetcher] = 0xFF
        elif low == self._bottoms[fetcher]:
            self._flags[fetcher] = 0x00

    def _decrement_counter(self, fetcher: int) -> None:
        self._counters[fetcher] = (self._counters[fetcher] - 1) & 0x7FF
        self._update_flag(fetcher)

    # ---- frame hooks ----

    def start_frame(self) -> None:
        """Clock the music-mode fetchers once per frame."""
        for f in range(5, 8):
            if self._music_mode[f]:
                self._decrement_counter(f)

    # ---- read / write ----

    def __getitem__(self, addr: int) -> int:
        addr12 = addr & 0xFFF

        # F8 bank-switching hotspots
        if addr12 == 0xFF8:
            self._bank_base_addr = 0x0000
        elif addr12 == 0xFF9:
            self._bank_base_addr = 0x1000

        # DPC read registers: $x000-$x03F
        if addr12 < 0x040:
            fetcher = addr12 & 0x07
            group = (addr12 >> 3) & 0x07

            if group == 0:
                # Group 0: plain display data
                result = self._read_display(fetcher)
                self._decrement_counter(fetcher)
                return result
            elif group == 1:
                # Group 1: display data AND flag
                result = self._read_display(fetcher) & self._flags[fetcher]
                self._decrement_counter(fetcher)
                return result
            elif group == 2:
                # Group 2: display data AND flag, complemented
                result = (
                    ~(self._read_display(fetcher) & self._flags[fetcher])
                ) & 0xFF
                self._decrement_counter(fetcher)
                return result
            elif group == 3:
                # Group 3: display data AND flag (alternate)
                result = self._read_display(fetcher) & self._flags[fetcher]
                self._decrement_counter(fetcher)
                return result
            elif group == 4:
                # Group 4: plain display data (alternate)
                result = self._read_display(fetcher)
                self._decrement_counter(fetcher)
                return result
            elif group == 5:
                # Group 5: flag register ($00 or $FF)
                return self._flags[fetcher]
            elif group == 6:
                # Group 6: music data (fetchers 5-7)
                if 5 <= fetcher <= 7:
                    return self._read_display(fetcher)
                return 0
            else:
                # Group 7: random number
                self._clock_random()
                return self._random

        # Normal ROM read
        return self.rom[self._bank_base_addr + addr12]

    def __setitem__(self, addr: int, value: int) -> None:
        addr12 = addr & 0xFFF
        value &= 0xFF

        # F8 bank-switching
        if addr12 == 0xFF8:
            self._bank_base_addr = 0x0000
        elif addr12 == 0xFF9:
            self._bank_base_addr = 0x1000

        # DPC write registers: $x040-$x07F
        if 0x040 <= addr12 < 0x080:
            fetcher = addr12 & 0x07
            reg = (addr12 - 0x040) >> 3

            if reg == 0:
                # $x040-$x047: set counter top
                self._tops[fetcher] = value
                self._update_flag(fetcher)
            elif reg == 1:
                # $x048-$x04F: set counter bottom
                self._bottoms[fetcher] = value
                self._update_flag(fetcher)
            elif reg == 2:
                # $x050-$x057: set counter low byte
                self._counters[fetcher] = (
                    (self._counters[fetcher] & 0x700) | value
                )
            elif reg == 3:
                # $x058-$x05F: set counter high byte (bits 2:0)
                high = (value & 0x07) << 8
                self._counters[fetcher] = (
                    high | (self._counters[fetcher] & 0xFF)
                )
            elif reg == 4:
                # $x060-$x067: enable music mode (bit 4)
                self._music_mode[fetcher] = bool(value & 0x10)
            elif reg == 5:
                # $x068-$x06F: seed random number generator
                self._random = value if value != 0 else 1
            elif reg == 6:
                # $x070-$x077: reset counter to top
                self._counters[fetcher] = (
                    (self._counters[fetcher] & 0x700) | self._tops[fetcher]
                )
                self._flags[fetcher] = 0xFF
