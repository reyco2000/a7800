"""
Core device abstractions for EMU7800.
Ported from C# classes: IDevice, NullDevice, RAM6116, Cart (BIOS portion).

IDevice is the abstract interface for all memory-mapped devices.
NullDevice is a no-op device used as a placeholder in the address space.
RAM6116 is a 2KB static RAM (used for 7800 on-board RAM, etc.).
NVRAM2k is a 2KB non-volatile RAM with save/load callbacks (used for HSC).
Bios7800 is a read-only ROM device for the 7800 BIOS (4KB or 16KB).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


class IDevice(ABC):
    """Abstract interface for all memory-mapped devices in the address space."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the device to its initial power-on state."""
        ...

    @abstractmethod
    def __getitem__(self, addr: int) -> int:
        """Read a byte from the device at the given address.

        Args:
            addr: The raw address on the bus (device applies its own masking).

        Returns:
            An integer in the range 0..255.
        """
        ...

    @abstractmethod
    def __setitem__(self, addr: int, value: int) -> None:
        """Write a byte to the device at the given address.

        Args:
            addr: The raw address on the bus (device applies its own masking).
            value: The byte value to write (0..255).
        """
        ...


class NullDevice(IDevice):
    """A device that ignores all writes and always reads as zero.

    Used as the default mapping for unmapped pages in the address space so that
    accesses to unmapped regions do not raise exceptions.
    """

    _instance: Optional[NullDevice] = None

    def __new__(cls) -> NullDevice:
        """NullDevice is a singleton -- every call returns the same instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def reset(self) -> None:
        pass

    def __getitem__(self, addr: int) -> int:
        return 0

    def __setitem__(self, addr: int, value: int) -> None:
        pass

    def __repr__(self) -> str:
        return "NullDevice()"


class RAM6116(IDevice):
    """2 KB (2048-byte) static RAM, mirroring the 6116 SRAM chip.

    The address is masked with 0x7FF so that accesses wrap within the 2 KB
    window regardless of where the device is mapped in the address space.
    """

    RAM_SIZE: int = 0x800  # 2048 bytes

    def __init__(self) -> None:
        self._data: bytearray = bytearray(self.RAM_SIZE)

    def reset(self) -> None:
        """Clear the RAM contents to all zeros."""
        for i in range(self.RAM_SIZE):
            self._data[i] = 0

    def __getitem__(self, addr: int) -> int:
        return self._data[addr & 0x7FF]

    def __setitem__(self, addr: int, value: int) -> None:
        self._data[addr & 0x7FF] = value & 0xFF

    # ------------------------------------------------------------------
    # Serialisation helpers (for save-state support)
    # ------------------------------------------------------------------

    def get_snapshot(self) -> bytes:
        """Return an immutable copy of the RAM contents."""
        return bytes(self._data)

    def restore_snapshot(self, data: bytes) -> None:
        """Restore RAM contents from a previous snapshot.

        Args:
            data: Exactly RAM_SIZE bytes captured by :meth:`get_snapshot`.

        Raises:
            ValueError: If *data* is not the expected length.
        """
        if len(data) != self.RAM_SIZE:
            raise ValueError(
                f"Snapshot size mismatch: expected {self.RAM_SIZE}, got {len(data)}"
            )
        self._data[:] = data

    def __repr__(self) -> str:
        return f"RAM6116(size={self.RAM_SIZE})"


class NVRAM2k(IDevice):
    """2 KB non-volatile RAM with optional load/save callbacks.

    This is used for the Atari 7800 High-Score Cartridge (HSC) whose 2 KB of
    battery-backed RAM persists across power cycles.  The caller supplies
    *on_load* and *on_save* callbacks that handle the actual persistence
    (file I/O, database, etc.).

    The load callback is invoked once during :meth:`reset` to populate the RAM.
    The save callback can be triggered explicitly via :meth:`save`.
    """

    RAM_SIZE: int = 0x800  # 2048 bytes

    def __init__(
        self,
        on_load: Optional[Callable[[], Optional[bytes]]] = None,
        on_save: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        """
        Args:
            on_load: Called during reset; should return 2048 bytes of saved
                     data, or ``None`` if no saved data is available.
            on_save: Called by :meth:`save`; receives the current RAM contents
                     as ``bytes``.
        """
        self._data: bytearray = bytearray(self.RAM_SIZE)
        self._on_load: Optional[Callable[[], Optional[bytes]]] = on_load
        self._on_save: Optional[Callable[[bytes], None]] = on_save

    def reset(self) -> None:
        """Reset the NVRAM.

        If an *on_load* callback was provided, it is called to restore
        previously saved contents.  Otherwise the RAM is cleared to zeros.
        """
        if self._on_load is not None:
            saved = self._on_load()
            if saved is not None and len(saved) == self.RAM_SIZE:
                self._data[:] = saved
                return
        # Fallback: clear to zero
        for i in range(self.RAM_SIZE):
            self._data[i] = 0

    def __getitem__(self, addr: int) -> int:
        return self._data[addr & 0x7FF]

    def __setitem__(self, addr: int, value: int) -> None:
        self._data[addr & 0x7FF] = value & 0xFF

    def save(self) -> None:
        """Persist the current RAM contents via the *on_save* callback."""
        if self._on_save is not None:
            self._on_save(bytes(self._data))

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_snapshot(self) -> bytes:
        """Return an immutable copy of the NVRAM contents."""
        return bytes(self._data)

    def restore_snapshot(self, data: bytes) -> None:
        """Restore NVRAM contents from a previous snapshot.

        Raises:
            ValueError: If *data* is not the expected length.
        """
        if len(data) != self.RAM_SIZE:
            raise ValueError(
                f"Snapshot size mismatch: expected {self.RAM_SIZE}, got {len(data)}"
            )
        self._data[:] = data

    def __repr__(self) -> str:
        return f"NVRAM2k(size={self.RAM_SIZE})"


class Bios7800(IDevice):
    """Read-only ROM device for the Atari 7800 BIOS.

    Accepts either a 4 KB (4096-byte) or 16 KB (16384-byte) ROM image.
    Reads are masked so they wrap within the ROM size.  Writes are silently
    ignored (ROM is read-only).
    """

    VALID_SIZES: tuple[int, ...] = (4096, 16384)

    def __init__(self, rom: bytes) -> None:
        """
        Args:
            rom: The BIOS ROM image -- must be exactly 4096 or 16384 bytes.

        Raises:
            ValueError: If the ROM size is not one of the valid sizes.
        """
        if len(rom) not in self.VALID_SIZES:
            raise ValueError(
                f"Bios7800 ROM must be {self.VALID_SIZES} bytes, got {len(rom)}"
            )
        self._rom: bytes = bytes(rom)  # immutable copy
        self._mask: int = len(self._rom) - 1

    def reset(self) -> None:
        """No-op -- ROM contents are immutable."""
        pass

    def __getitem__(self, addr: int) -> int:
        return self._rom[addr & self._mask]

    def __setitem__(self, addr: int, value: int) -> None:
        # ROM is read-only; silently ignore writes.
        pass

    @property
    def size(self) -> int:
        """The size of the ROM image in bytes."""
        return len(self._rom)

    def __repr__(self) -> str:
        return f"Bios7800(size={len(self._rom)})"
