"""
Atari 7800 cartridge mappers for EMU7800.
Ported from the C# Cart78* classes.

The 7800 has a full 16-bit (64 KB) address space.  Cartridge ROM is typically
mapped into the upper portion ($4000-$FFFF, 48 KB) with bank-switching logic
controlling which slices of a larger ROM image appear in the CPU's view.

Standard layout for banked 7800 carts (SuperGame and friends)::

    $4000-$7FFF   fixed or switchable 16 KB region (``low'')
    $8000-$BFFF   switchable 16 KB bank           (``mid'')
    $C000-$FFFF   fixed 16 KB region               (``high'')

Mappers
-------
Cart7808      -- 8 KB flat ($E000-$FFFF).
Cart7816      -- 16 KB flat ($C000-$FFFF).
Cart7832      -- 32 KB flat ($8000-$FFFF).
Cart7832P     -- 32 KB + POKEY @ $4000.
Cart7832PL    -- 32 KB + POKEY @ $0450.
Cart7848      -- 48 KB flat ($4000-$FFFF, two regions).
Cart78AB      -- 64 KB Absolute / F18 Hornet.
Cart78AC      -- 128 KB Activision / Double Dragon.
Cart78SG      -- 128 KB SuperGame (8 x 16 KB).
Cart78SGP     -- SuperGame + POKEY @ $4000.
Cart78SGR     -- SuperGame + 16 KB RAM @ $4000.
Cart78S9      -- 144 KB 9-bank SuperGame.
Cart78S9PL    -- 9-bank + POKEY @ $0450.
Cart78S4      -- 64 KB 4-bank SuperGame.
Cart78S4R     -- 4-bank SuperGame + 16 KB RAM.
Cart78BB*     -- BankswitchBoard variants (32 KB .. 128 KB).
"""

from __future__ import annotations

from emu7800.core.carts.cart import Cart


# ======================================================================
#  Small flat-ROM carts
# ======================================================================

class Cart7808(Cart):
    """8 KB flat cart mapped at $E000-$FFFF."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=8192)

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0xE000, 0x2000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        return self.rom[addr & 0x1FFF]

    def __setitem__(self, addr: int, value: int) -> None:
        pass


class Cart7816(Cart):
    """16 KB flat cart mapped at $C000-$FFFF."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=16384)

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0xC000, 0x4000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        return self.rom[addr & 0x3FFF]

    def __setitem__(self, addr: int, value: int) -> None:
        pass


class Cart7832(Cart):
    """32 KB flat cart mapped at $8000-$FFFF."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=32768)

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x8000, 0x8000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        return self.rom[addr & 0x7FFF]

    def __setitem__(self, addr: int, value: int) -> None:
        pass


# ======================================================================
#  32 KB + POKEY variants
# ======================================================================

class Cart7832P(Cart):
    """32 KB flat cart + POKEY sound chip at $4000.

    ROM is at $8000-$FFFF (32 KB).  POKEY is memory-mapped at $4000-$7FFF.
    When a POKEY device is available it should be assigned to ``self.pokey``
    before ``map()`` is called; ``map()`` will then install it in the
    address space.  Without a POKEY, reads from $4000-$7FFF return 0.
    """

    POKEY_BASE: int = 0x4000

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=32768)
        self.pokey = None  # optional POKEY device

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x8000, 0x8000, self)
            if self.pokey is not None:
                addr_space.map(self.POKEY_BASE, 0x4000, self.pokey)
            else:
                # Map the cart itself so reads return 0 (handled below)
                addr_space.map(0x4000, 0x4000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0x8000:
            return self.rom[addr & 0x7FFF]
        # $4000-$7FFF: POKEY range (if no real POKEY, return 0)
        if self.pokey is not None:
            return self.pokey[addr]
        return 0

    def __setitem__(self, addr: int, value: int) -> None:
        if self.pokey is not None and 0x4000 <= addr <= 0x7FFF:
            self.pokey[addr] = value


class Cart7832PL(Cart):
    """32 KB flat cart + POKEY sound chip at $0450.

    Identical to :class:`Cart7832` for ROM access.  POKEY sits at $0450
    in the TIA/RIOT area.  The machine is responsible for mapping POKEY
    at that low address; the cart only handles the ROM region.
    """

    POKEY_BASE: int = 0x0450

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=32768)
        self.pokey = None

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x8000, 0x8000, self)
            if self.pokey is not None:
                addr_space.map(self.POKEY_BASE, 0x40, self.pokey)
        return True

    def __getitem__(self, addr: int) -> int:
        return self.rom[addr & 0x7FFF]

    def __setitem__(self, addr: int, value: int) -> None:
        pass


# ======================================================================
#  48 KB flat
# ======================================================================

class Cart7848(Cart):
    """48 KB flat cart occupying $4000-$FFFF.

    ROM layout::

        ROM[0x0000 : 0x4000]  ->  $4000-$7FFF  (first 16 KB)
        ROM[0x4000 : 0xC000]  ->  $8000-$FFFF  (remaining 32 KB)

    Total: 48 KB.
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=49152)

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0x8000:
            # Upper 32 KB: ROM[0x4000 + (addr & 0x7FFF)]
            return self.rom[0x4000 + (addr & 0x7FFF)]
        # $4000-$7FFF: ROM[0x0000 + (addr & 0x3FFF)]
        return self.rom[addr & 0x3FFF]

    def __setitem__(self, addr: int, value: int) -> None:
        pass


# ======================================================================
#  Cart78AB -- Absolute / F18 Hornet (64 KB)
# ======================================================================

class Cart78AB(Cart):
    """64 KB Absolute mapper (used by *F-18 Hornet*).

    The 64 KB ROM is arranged as:

    * ROM[0x0000:0x4000] -- bank 0 (16 KB, switchable)
    * ROM[0x4000:0x8000] -- bank 1 (16 KB, switchable)
    * ROM[0x8000:0x10000] -- 32 KB fixed

    Memory map:

    * $4000-$7FFF: switchable between bank 0 and bank 1
    * $8000-$FFFF: fixed 32 KB (last 32 KB of ROM)

    Bank switching is triggered by a write to any address in $8000-$BFFF.
    Bit 1 of the written value selects the bank:

    * D1 = 0 -> bank 1  (ROM[0x4000:0x8000])
    * D1 = 1 -> bank 0  (ROM[0x0000:0x4000])
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=65536)
        self._bank_base_addr: int = 0  # offset of switchable bank

    def reset(self) -> None:
        # Default to bank 1
        self._bank_base_addr = 0x4000

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0x8000:
            # Fixed upper 32 KB
            return self.rom[0x8000 + (addr & 0x7FFF)]
        # $4000-$7FFF: switchable 16 KB bank
        return self.rom[self._bank_base_addr + (addr & 0x3FFF)]

    def __setitem__(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xBFFF:
            if value & 0x02:
                self._bank_base_addr = 0x0000  # bank 0
            else:
                self._bank_base_addr = 0x4000  # bank 1


# ======================================================================
#  Cart78AC -- Activision / Double Dragon (128 KB)
# ======================================================================

class Cart78AC(Cart):
    """128 KB Activision mapper (used by *Double Dragon*, *Rampage*).

    The 128 KB ROM is arranged as eight 16 KB banks (0-7).

    Memory map:

    * $4000-$7FFF:  bank (reg - 2), derived from the bank register
    * $8000-$BFFF:  bank (reg), selected by bank register
    * $C000-$FFFF:  fixed bank 7 (last 16 KB)

    Bank switching is triggered by writing to addresses $xFF80-$xFF87.
    The low 3 bits of the address select the bank register value.
    The region at $4000 shows bank (reg - 2) if reg >= 2, else bank 0.
    """

    BANK_SIZE: int = 0x4000  # 16 KB

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=131072)
        self._bank_reg: int = 0  # 0-7

    def reset(self) -> None:
        self._bank_reg = 0

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
        return True

    def _mid_bank_offset(self) -> int:
        return self._bank_reg * self.BANK_SIZE

    def _low_bank_offset(self) -> int:
        low_bank = max(0, self._bank_reg - 2)
        return low_bank * self.BANK_SIZE

    def __getitem__(self, addr: int) -> int:
        if addr >= 0xC000:
            # Fixed: bank 7
            return self.rom[7 * self.BANK_SIZE + (addr & 0x3FFF)]
        if addr >= 0x8000:
            # Switchable mid region
            return self.rom[self._mid_bank_offset() + (addr & 0x3FFF)]
        # $4000-$7FFF: low region
        return self.rom[self._low_bank_offset() + (addr & 0x3FFF)]

    def __setitem__(self, addr: int, value: int) -> None:
        # Hotspots at $xFF80-$xFF87 (or any mirror where low byte is 80-87)
        addr_low = addr & 0xFF
        if 0x80 <= addr_low <= 0x87:
            self._bank_reg = addr_low & 0x07


# ======================================================================
#  SuperGame family
# ======================================================================

class Cart78SG(Cart):
    """128 KB SuperGame mapper (8 x 16 KB banks).

    Memory map:

    * $4000-$7FFF:  fixed bank 6
    * $8000-$BFFF:  switchable (bank register, 0-7)
    * $C000-$FFFF:  fixed bank 7

    Bank switching is triggered by writing to any address in $8000-$BFFF.
    The low 3 bits of the written value select the bank.
    """

    BANK_SIZE: int = 0x4000  # 16 KB

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=131072)
        self._n_banks: int = max(1, len(self.rom) // self.BANK_SIZE)
        self._bank_reg: int = 0

    def reset(self) -> None:
        self._bank_reg = 0

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0xC000:
            # Fixed: last bank
            offset = (self._n_banks - 1) * self.BANK_SIZE + (addr & 0x3FFF)
            return self.rom[offset]
        if addr >= 0x8000:
            # Switchable mid region
            bank = self._bank_reg % self._n_banks
            offset = bank * self.BANK_SIZE + (addr & 0x3FFF)
            return self.rom[offset]
        # $4000-$7FFF: fixed second-to-last bank (bank 6 for 8-bank)
        offset = (self._n_banks - 2) * self.BANK_SIZE + (addr & 0x3FFF)
        return self.rom[offset]

    def __setitem__(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xBFFF:
            self._bank_reg = value & 0x07


class Cart78SGP(Cart):
    """128 KB SuperGame + POKEY at $4000.

    Identical to :class:`Cart78SG` for banking.  POKEY is mapped at
    $4000-$7FFF; POKEY addresses shadow the fixed bank 6 ROM region.
    When ``self.pokey`` is None, reads from $4000-$7FFF fall through
    to bank 6 ROM.
    """

    BANK_SIZE: int = 0x4000
    POKEY_BASE: int = 0x4000

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=131072)
        self._n_banks: int = max(1, len(self.rom) // self.BANK_SIZE)
        self._bank_reg: int = 0
        self.pokey = None

    def reset(self) -> None:
        self._bank_reg = 0

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
            if self.pokey is not None:
                addr_space.map(self.POKEY_BASE, 0x4000, self.pokey)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0xC000:
            offset = (self._n_banks - 1) * self.BANK_SIZE + (addr & 0x3FFF)
            return self.rom[offset]
        if addr >= 0x8000:
            bank = self._bank_reg % self._n_banks
            return self.rom[bank * self.BANK_SIZE + (addr & 0x3FFF)]
        # $4000-$7FFF: POKEY or bank 6
        if self.pokey is not None:
            return self.pokey[addr]
        offset = (self._n_banks - 2) * self.BANK_SIZE + (addr & 0x3FFF)
        return self.rom[offset]

    def __setitem__(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xBFFF:
            self._bank_reg = value & 0x07
        elif self.pokey is not None and 0x4000 <= addr <= 0x7FFF:
            self.pokey[addr] = value


class Cart78SGR(Cart):
    """128 KB SuperGame + 16 KB RAM at $4000.

    Banking identical to :class:`Cart78SG` for $8000-$FFFF.  At
    $4000-$7FFF, either the fixed bank 6 ROM or 16 KB of RAM is visible
    depending on bit 4 of the bank register:

    * D4 = 0  ->  bank 6 ROM at $4000
    * D4 = 1  ->  16 KB RAM at $4000
    """

    BANK_SIZE: int = 0x4000

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=131072)
        self._n_banks: int = max(1, len(self.rom) // self.BANK_SIZE)
        self._bank_reg: int = 0
        self._ram_enabled: bool = False
        self._ram: bytearray = bytearray(0x4000)  # 16 KB

    def reset(self) -> None:
        self._bank_reg = 0
        self._ram_enabled = False
        for i in range(len(self._ram)):
            self._ram[i] = 0

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0xC000:
            offset = (self._n_banks - 1) * self.BANK_SIZE + (addr & 0x3FFF)
            return self.rom[offset]
        if addr >= 0x8000:
            bank = self._bank_reg % self._n_banks
            return self.rom[bank * self.BANK_SIZE + (addr & 0x3FFF)]
        # $4000-$7FFF
        if self._ram_enabled:
            return self._ram[addr & 0x3FFF]
        offset = (self._n_banks - 2) * self.BANK_SIZE + (addr & 0x3FFF)
        return self.rom[offset]

    def __setitem__(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xBFFF:
            self._bank_reg = value & 0x07
            self._ram_enabled = bool(value & 0x10)
        elif 0x4000 <= addr <= 0x7FFF:
            if self._ram_enabled:
                self._ram[addr & 0x3FFF] = value & 0xFF


# ======================================================================
#  9-bank SuperGame (144 KB)
# ======================================================================

class Cart78S9(Cart):
    """144 KB 9-bank SuperGame (9 x 16 KB).

    Layout:

    * $4000-$7FFF:  fixed bank 8 (the extra 9th bank, first 16 KB of ROM)
    * $8000-$BFFF:  switchable among banks 0-7
    * $C000-$FFFF:  fixed bank 7 (last 16 KB of the first 128 KB)

    ROM organisation::

        ROM[0x00000:0x04000]  bank 8  (fixed at $4000)
        ROM[0x04000:0x08000]  bank 0
        ROM[0x08000:0x0C000]  bank 1
        ...
        ROM[0x20000:0x24000]  bank 7  (fixed at $C000)

    Bank switching: write to $8000-$BFFF, low 3 bits select bank 0-7.
    """

    BANK_SIZE: int = 0x4000

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=147456)  # 9 * 16384
        self._bank_reg: int = 0

    def reset(self) -> None:
        self._bank_reg = 0

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0xC000:
            # Fixed: bank 7 = ROM[(1 + 7) * 16KB]
            return self.rom[8 * self.BANK_SIZE + (addr & 0x3FFF)]
        if addr >= 0x8000:
            # Switchable: banks 0-7 start at ROM[1 * 16KB]
            bank = self._bank_reg & 0x07
            return self.rom[(1 + bank) * self.BANK_SIZE + (addr & 0x3FFF)]
        # $4000-$7FFF: fixed bank 8 = ROM[0x0000:0x4000]
        return self.rom[addr & 0x3FFF]

    def __setitem__(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xBFFF:
            self._bank_reg = value & 0x07


class Cart78S9PL(Cart):
    """144 KB 9-bank SuperGame + POKEY at $0450.

    Banking identical to :class:`Cart78S9`.  POKEY is mapped at $0450 by
    the machine; the cart itself only handles ROM reads.
    """

    BANK_SIZE: int = 0x4000
    POKEY_BASE: int = 0x0450

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=147456)
        self._bank_reg: int = 0
        self.pokey = None

    def reset(self) -> None:
        self._bank_reg = 0

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
            if self.pokey is not None:
                addr_space.map(self.POKEY_BASE, 0x40, self.pokey)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0xC000:
            return self.rom[8 * self.BANK_SIZE + (addr & 0x3FFF)]
        if addr >= 0x8000:
            bank = self._bank_reg & 0x07
            return self.rom[(1 + bank) * self.BANK_SIZE + (addr & 0x3FFF)]
        return self.rom[addr & 0x3FFF]

    def __setitem__(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xBFFF:
            self._bank_reg = value & 0x07


# ======================================================================
#  4-bank SuperGame (64 KB)
# ======================================================================

class Cart78S4(Cart):
    """64 KB 4-bank SuperGame (4 x 16 KB).

    Layout:

    * $4000-$7FFF:  fixed bank 2 (second-to-last)
    * $8000-$BFFF:  switchable among banks 0-3
    * $C000-$FFFF:  fixed bank 3 (last)

    Bank switching: write to $8000-$BFFF, low 2 bits select bank 0-3.
    """

    BANK_SIZE: int = 0x4000

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=65536)
        self._bank_reg: int = 0

    def reset(self) -> None:
        self._bank_reg = 0

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0xC000:
            # Fixed: bank 3
            return self.rom[3 * self.BANK_SIZE + (addr & 0x3FFF)]
        if addr >= 0x8000:
            bank = self._bank_reg & 0x03
            return self.rom[bank * self.BANK_SIZE + (addr & 0x3FFF)]
        # $4000-$7FFF: fixed bank 2
        return self.rom[2 * self.BANK_SIZE + (addr & 0x3FFF)]

    def __setitem__(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xBFFF:
            self._bank_reg = value & 0x03


class Cart78S4R(Cart):
    """64 KB 4-bank SuperGame + 16 KB RAM at $4000.

    Like :class:`Cart78S4` but with switchable RAM at $4000-$7FFF.
    Bit 4 of the bank register enables/disables the RAM overlay.
    """

    BANK_SIZE: int = 0x4000

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__()
        self.load_rom(rom_bytes, min_size=65536)
        self._bank_reg: int = 0
        self._ram_enabled: bool = False
        self._ram: bytearray = bytearray(0x4000)

    def reset(self) -> None:
        self._bank_reg = 0
        self._ram_enabled = False
        for i in range(len(self._ram)):
            self._ram[i] = 0

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0xC000:
            return self.rom[3 * self.BANK_SIZE + (addr & 0x3FFF)]
        if addr >= 0x8000:
            bank = self._bank_reg & 0x03
            return self.rom[bank * self.BANK_SIZE + (addr & 0x3FFF)]
        if self._ram_enabled:
            return self._ram[addr & 0x3FFF]
        return self.rom[2 * self.BANK_SIZE + (addr & 0x3FFF)]

    def __setitem__(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xBFFF:
            self._bank_reg = value & 0x03
            self._ram_enabled = bool(value & 0x10)
        elif 0x4000 <= addr <= 0x7FFF:
            if self._ram_enabled:
                self._ram[addr & 0x3FFF] = value & 0xFF


# ======================================================================
#  BankswitchBoard (BB) family
# ======================================================================

class Cart78BB(Cart):
    """Base class for 7800 BankswitchBoard cart variants.

    The BankswitchBoard carts follow the standard 7800 three-region
    layout with a bankset switching mechanism:

    * $4000-$7FFF:  ``low'' region -- contents depend on variant
    * $8000-$BFFF:  switchable 16 KB bank (bank register)
    * $C000-$FFFF:  fixed to the last 16 KB bank

    Bank switching is triggered by writing to $8000-$BFFF.  The low
    bits of the written value select the bank.

    Subclasses configure:

    * ``_n_banks``         -- total number of 16 KB banks
    * ``_bank_mask``       -- bitmask for the bank register
    * ``_low_bank_index``  -- which bank is fixed at $4000 (-1 = none)
    * ``_has_ram``         -- whether 16 KB RAM can overlay $4000
    * ``_has_pokey``       -- whether a POKEY is present
    * ``_pokey_addr``      -- POKEY base address ($4000 or $0450)
    """

    BANK_SIZE: int = 0x4000  # 16 KB

    def __init__(
        self,
        rom_bytes: bytes,
        *,
        n_banks: int,
        bank_mask: int,
        low_bank_index: int = -1,
        has_ram: bool = False,
        has_pokey: bool = False,
        pokey_addr: int = 0x4000,
        extra_rom_offset: int = -1,
        extra_rom_size: int = 0,
    ) -> None:
        super().__init__()
        min_rom = n_banks * self.BANK_SIZE
        if extra_rom_size > 0:
            min_rom = extra_rom_size + n_banks * self.BANK_SIZE
        self.load_rom(rom_bytes, min_size=min_rom)

        self._n_banks: int = n_banks
        self._bank_mask: int = bank_mask
        self._bank_reg: int = 0
        self._low_bank_index: int = low_bank_index
        self._has_ram: bool = has_ram
        self._ram_enabled: bool = False
        self._ram: bytearray = bytearray(0x4000) if has_ram else bytearray()
        self._has_pokey: bool = has_pokey
        self._pokey_addr: int = pokey_addr
        self.pokey = None  # set externally

        # For carts with extra ROM before the banked region (e.g. 52 KB).
        self._extra_rom_offset: int = extra_rom_offset
        self._extra_rom_size: int = extra_rom_size

    def reset(self) -> None:
        self._bank_reg = 0
        self._ram_enabled = False
        if self._has_ram:
            for i in range(len(self._ram)):
                self._ram[i] = 0

    @property
    def _banks_base(self) -> int:
        """ROM offset where the banked region starts."""
        if self._extra_rom_size > 0:
            return self._extra_rom_size
        return 0

    def _bank_offset(self, bank: int) -> int:
        return self._banks_base + bank * self.BANK_SIZE

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            # Always map the upper 32 KB ($8000-$FFFF)
            addr_space.map(0x8000, 0x8000, self)
            # Map $4000-$7FFF if there is content there
            if (
                self._low_bank_index >= 0
                or self._has_ram
                or self._extra_rom_size > 0
            ):
                addr_space.map(0x4000, 0x4000, self)
            # POKEY at $4000: shadow the cart mapping
            if self._has_pokey and self.pokey is not None:
                if self._pokey_addr == 0x4000:
                    addr_space.map(0x4000, 0x4000, self.pokey)
                else:
                    addr_space.map(self._pokey_addr, 0x40, self.pokey)
        return True

    def __getitem__(self, addr: int) -> int:
        if addr >= 0xC000:
            # Fixed high bank (last bank)
            off = self._bank_offset(self._n_banks - 1) + (addr & 0x3FFF)
            return self.rom[off] if off < len(self.rom) else 0
        if addr >= 0x8000:
            # Switchable mid bank
            bank = self._bank_reg & self._bank_mask
            off = self._bank_offset(bank) + (addr & 0x3FFF)
            return self.rom[off] if off < len(self.rom) else 0
        # $4000-$7FFF (low region)
        return self._read_low(addr)

    def __setitem__(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xBFFF:
            self._bank_reg = value & self._bank_mask
            if self._has_ram:
                self._ram_enabled = bool(value & 0x10)
        elif 0x4000 <= addr <= 0x7FFF:
            self._write_low(addr, value)

    def _read_low(self, addr: int) -> int:
        """Read from the $4000-$7FFF region."""
        # RAM overlay
        if self._has_ram and self._ram_enabled:
            return self._ram[addr & 0x3FFF]
        # POKEY
        if self._has_pokey and self.pokey is not None:
            return self.pokey[addr]
        # Extra ROM (for 52K-style carts)
        if self._extra_rom_size > 0:
            offset = addr & 0x3FFF
            if offset < self._extra_rom_size:
                return self.rom[offset]
            return 0
        # Fixed low bank
        if self._low_bank_index >= 0:
            off = self._bank_offset(self._low_bank_index) + (addr & 0x3FFF)
            return self.rom[off] if off < len(self.rom) else 0
        return 0

    def _write_low(self, addr: int, value: int) -> None:
        """Write to the $4000-$7FFF region."""
        if self._has_ram and self._ram_enabled:
            self._ram[addr & 0x3FFF] = value & 0xFF
        elif self._has_pokey and self.pokey is not None:
            self.pokey[addr] = value


# ------------------------------------------------------------------
#  BB 32 KB variants
# ------------------------------------------------------------------

class Cart78BB32K(Cart78BB):
    """BankswitchBoard 32 KB (2 x 16 KB).

    * $8000-$BFFF: switchable bank 0 or 1
    * $C000-$FFFF: fixed bank 1
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(rom_bytes, n_banks=2, bank_mask=0x01)


class Cart78BB32KP(Cart78BB):
    """BankswitchBoard 32 KB + POKEY at $4000."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=2, bank_mask=0x01,
            has_pokey=True, pokey_addr=0x4000,
        )


class Cart78BB32KRPL(Cart78BB):
    """BankswitchBoard 32 KB + 16 KB RAM + POKEY at $0450."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=2, bank_mask=0x01,
            has_ram=True, has_pokey=True, pokey_addr=0x0450,
        )


# ------------------------------------------------------------------
#  BB 48 KB variants
# ------------------------------------------------------------------

class Cart78BB48K(Cart78BB):
    """BankswitchBoard 48 KB (3 x 16 KB).

    * $4000-$7FFF: fixed bank 0
    * $8000-$BFFF: switchable banks 0-2
    * $C000-$FFFF: fixed bank 2
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=3, bank_mask=0x03,
            low_bank_index=0,
        )


class Cart78BB48KP(Cart78BB):
    """BankswitchBoard 48 KB + POKEY at $4000.

    Like :class:`Cart78BB48K` but POKEY overlays $4000.
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=3, bank_mask=0x03,
            low_bank_index=0, has_pokey=True, pokey_addr=0x4000,
        )


# ------------------------------------------------------------------
#  BB 52 KB variants
# ------------------------------------------------------------------

class Cart78BB52K(Cart78BB):
    """BankswitchBoard 52 KB.

    The first 4 KB of ROM sits at $4000-$4FFF (extra ROM region).
    The remaining 48 KB is arranged as three 16 KB banks:

    * $8000-$BFFF: switchable banks 0-2
    * $C000-$FFFF: fixed bank 2
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=3, bank_mask=0x03,
            extra_rom_offset=0, extra_rom_size=4096,
        )

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
        return True


class Cart78BB52KP(Cart78BB):
    """BankswitchBoard 52 KB + POKEY at $4000.

    Like :class:`Cart78BB52K` but POKEY overlays $4000-$40FF.
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=3, bank_mask=0x03,
            extra_rom_offset=0, extra_rom_size=4096,
            has_pokey=True, pokey_addr=0x4000,
        )

    def map(self, addr_space=None) -> bool:
        if addr_space is not None:
            addr_space.map(0x4000, 0xC000, self)
            if self._has_pokey and self.pokey is not None:
                addr_space.map(self._pokey_addr, 0x4000, self.pokey)
        return True


# ------------------------------------------------------------------
#  BB 128 KB variants
# ------------------------------------------------------------------

class Cart78BB128K(Cart78BB):
    """BankswitchBoard 128 KB (8 x 16 KB).

    * $4000-$7FFF: fixed bank 6
    * $8000-$BFFF: switchable banks 0-7
    * $C000-$FFFF: fixed bank 7
    """

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=8, bank_mask=0x07,
            low_bank_index=6,
        )


class Cart78BB128KR(Cart78BB):
    """BankswitchBoard 128 KB + 16 KB RAM at $4000."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=8, bank_mask=0x07,
            low_bank_index=6, has_ram=True,
        )


class Cart78BB128KP(Cart78BB):
    """BankswitchBoard 128 KB + POKEY at $4000."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=8, bank_mask=0x07,
            low_bank_index=6, has_pokey=True, pokey_addr=0x4000,
        )


class Cart78BB128KRPL(Cart78BB):
    """BankswitchBoard 128 KB + 16 KB RAM + POKEY at $0450."""

    def __init__(self, rom_bytes: bytes) -> None:
        super().__init__(
            rom_bytes, n_banks=8, bank_mask=0x07,
            low_bank_index=6, has_ram=True,
            has_pokey=True, pokey_addr=0x0450,
        )
