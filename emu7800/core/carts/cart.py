"""
Base Cart class and factory for EMU7800 cartridge system.
Ported from the C# Cart class hierarchy.

The Cart abstract base class extends IDevice and provides the common interface
for all cartridge types (Atari 2600 and 7800).  The static ``create()`` factory
method maps :class:`~emu7800.core.types.CartType` enum values to concrete
cart implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from emu7800.core.devices import IDevice
from emu7800.core.types import CartType


class Cart(IDevice, ABC):
    """Abstract base class for all cartridge types (2600 and 7800).

    Subclasses must implement ``__getitem__`` and ``__setitem__`` to handle
    reads and writes from the CPU (and Maria DMA on the 7800).

    Attributes:
        rom:              The cartridge ROM data.
        m:                Back-reference to the owning machine (set by ``attach``).
        request_snooping: When True the machine should install this cart as the
                          address-space snooper so it can observe *all* bus
                          traffic (needed by Tigervision 3F, Activision FE, etc.).
    """

    def __init__(self) -> None:
        self.rom: bytearray = bytearray()
        self.m: object = None  # machine reference, set by attach()
        self.request_snooping: bool = False

    # ------------------------------------------------------------------
    # IDevice interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the cartridge to its power-on state.  Override in subclass."""
        pass

    @abstractmethod
    def __getitem__(self, addr: int) -> int:
        """Read a byte from the cartridge at *addr*."""
        ...

    @abstractmethod
    def __setitem__(self, addr: int, value: int) -> None:
        """Write a byte to the cartridge at *addr*."""
        ...

    # ------------------------------------------------------------------
    # Cart-specific interface
    # ------------------------------------------------------------------

    def attach(self, machine: object) -> None:
        """Attach this cartridge to a machine."""
        self.m = machine

    def map(self, addr_space=None) -> bool:
        """Map the cartridge into the address space.

        Called by :meth:`AddressSpace.map_cart` with the address-space instance.
        Override in subclasses that need non-default memory mapping (e.g. 7800
        carts that occupy less than the full 0x4000-0xFFFF window, or that need
        to install a POKEY device).

        Returns:
            True if the cart performed its own mapping, False to request the
            default mapping (entire cart region).
        """
        return False

    def start_frame(self) -> None:
        """Called at the start of each video frame."""
        pass

    def end_frame(self) -> None:
        """Called at the end of each video frame."""
        pass

    # ------------------------------------------------------------------
    # ROM helpers
    # ------------------------------------------------------------------

    def load_rom(self, rom_bytes: bytes, min_size: Optional[int] = None) -> None:
        """Load ROM data, optionally padding to *min_size* with zeros.

        Args:
            rom_bytes: Raw ROM image bytes.
            min_size:  If the image is smaller than this, it is zero-padded.
        """
        if min_size is not None and len(rom_bytes) < min_size:
            self.rom = bytearray(min_size)
            self.rom[: len(rom_bytes)] = rom_bytes
        else:
            self.rom = bytearray(rom_bytes)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create(rom_bytes: bytes, cart_type: CartType) -> "Cart":
        """Create the appropriate Cart subclass for *cart_type*.

        Args:
            rom_bytes: Raw ROM image.
            cart_type: A :class:`CartType` enum value that identifies the mapper.

        Returns:
            A fully constructed (but not yet attached/reset) Cart instance.

        Raises:
            ValueError: If *cart_type* is unknown.
        """
        # Lazy imports avoid circular-dependency issues and keep the module
        # import lightweight when only the base class is needed.
        from emu7800.core.carts.cart_2600 import (
            CartA2K,
            CartA4K,
            CartA8K,
            CartA8KR,
            CartA16K,
            CartA16KR,
            CartA32K,
            CartA32KR,
            CartCBS12K,
            CartDC8K,
            CartDPC,
            CartMN16K,
            CartPB8K,
            CartTV8K,
        )
        from emu7800.core.carts.cart_7800 import (
            Cart7808,
            Cart7816,
            Cart7832,
            Cart7832P,
            Cart7832PL,
            Cart7848,
            Cart78AB,
            Cart78AC,
            Cart78BB128K,
            Cart78BB128KP,
            Cart78BB128KR,
            Cart78BB128KRPL,
            Cart78BB32K,
            Cart78BB32KP,
            Cart78BB32KRPL,
            Cart78BB48K,
            Cart78BB48KP,
            Cart78BB52K,
            Cart78BB52KP,
            Cart78S4,
            Cart78S4R,
            Cart78S9,
            Cart78S9PL,
            Cart78SG,
            Cart78SGP,
            Cart78SGR,
        )

        _cart_map = {
            # ---- Atari 2600 ----
            CartType.A2K: CartA2K,
            CartType.A4K: CartA4K,
            CartType.A8K: CartA8K,
            CartType.A8KR: CartA8KR,
            CartType.A16K: CartA16K,
            CartType.A16KR: CartA16KR,
            CartType.A32K: CartA32K,
            CartType.A32KR: CartA32KR,
            CartType.DC8K: CartDC8K,
            CartType.PB8K: CartPB8K,
            CartType.TV8K: CartTV8K,
            CartType.CBS12K: CartCBS12K,
            CartType.MN16K: CartMN16K,
            CartType.DPC: CartDPC,
            # ---- Atari 7800 ----
            CartType.A7808: Cart7808,
            CartType.A7816: Cart7816,
            CartType.A7832: Cart7832,
            CartType.A7832P: Cart7832P,
            CartType.A7832PL: Cart7832PL,
            CartType.A7848: Cart7848,
            CartType.A78AB: Cart78AB,
            CartType.A78AC: Cart78AC,
            CartType.A78SG: Cart78SG,
            CartType.A78SGP: Cart78SGP,
            CartType.A78SGR: Cart78SGR,
            CartType.A78S9: Cart78S9,
            CartType.A78S9PL: Cart78S9PL,
            CartType.A78S4: Cart78S4,
            CartType.A78S4R: Cart78S4R,
            # ---- 7800 BankswitchBoard ----
            CartType.A78BB32K: Cart78BB32K,
            CartType.A78BB32KP: Cart78BB32KP,
            CartType.A78BB32KRPL: Cart78BB32KRPL,
            CartType.A78BB48K: Cart78BB48K,
            CartType.A78BB48KP: Cart78BB48KP,
            CartType.A78BB52K: Cart78BB52K,
            CartType.A78BB52KP: Cart78BB52KP,
            CartType.A78BB128K: Cart78BB128K,
            CartType.A78BB128KR: Cart78BB128KR,
            CartType.A78BB128KP: Cart78BB128KP,
            CartType.A78BB128KRPL: Cart78BB128KRPL,
        }

        cls = _cart_map.get(cart_type)
        if cls is None:
            raise ValueError(f"Unknown or unsupported cart type: {cart_type!r}")

        return cls(rom_bytes)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rom_size={len(self.rom)})"
