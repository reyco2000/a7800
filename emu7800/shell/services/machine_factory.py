"""
Machine creation factory for EMU7800.
Ported from the C# MachineFactory / GameProgramInfoService classes.

Creates fully-configured emulated machine instances from a ROM file path
and optional overrides for machine type, cart type, BIOS, and controllers.

Typical usage::

    machine = MachineFactory.create("game.a78")
    machine = MachineFactory.create("game.a26", machine_type=MachineType.A2600NTSC)
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

from emu7800.core.types import CartType, Controller, MachineType
from emu7800.core.devices import Bios7800
from emu7800.shell.services.rom_bytes_service import RomBytesService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forward-compatible imports for core components that may not be fully
# implemented yet.  Cart and MachineBase are the two main dependencies.
# We import them conditionally so that the factory module can at least be
# *loaded* without the full core in place (useful during incremental porting).
# ---------------------------------------------------------------------------
try:
    from emu7800.core.carts.cart import Cart
except ImportError:
    Cart = None  # type: ignore[assignment,misc]

try:
    from emu7800.core.machine_base import MachineBase
except ImportError:
    MachineBase = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Default 7800 BIOS stub
# ---------------------------------------------------------------------------

def _make_default_bios_7800() -> Bios7800:
    """Create a minimal 4 KB BIOS stub for the Atari 7800.

    This stub does not execute the real BIOS startup sequence; it simply
    places a ``JMP ($FFFC)`` at the reset vector so the CPU jumps straight
    to the cartridge entry point.

    Layout (4096 bytes, base address $F000):
      - $FFFC-$FFFD (offsets 0xFFC-0xFFD): Reset vector -> $FFF8
      - $FFF8-$FFFA (offsets 0xFF8-0xFFA): JMP ($FFFC) = 6C FC FF
      - Everything else is 0x00 (NOP sled on 6502 is effectively BRK,
        but we never execute any of it).

    .. note:: A real BIOS image is recommended for full compatibility.
    """
    rom = bytearray(4096)

    # Place an indirect JMP through the cart's reset vector at $FFF8.
    # 6502 opcode: JMP (abs) = 0x6C, followed by the 16-bit address $FFFC
    # in little-endian.
    rom[0xFF8] = 0x6C  # JMP indirect
    rom[0xFF9] = 0xFC  # low byte of $FFFC
    rom[0xFFA] = 0xFF  # high byte of $FFFC

    # Set the BIOS reset vector itself to point at $FFF8.
    rom[0xFFC] = 0xF8  # low byte -> $FFF8
    rom[0xFFD] = 0xFF  # high byte

    # NMI vector -> also point at $FFF8 (a safe no-op landing).
    rom[0xFFA] = 0xF8
    rom[0xFFB] = 0xFF

    # IRQ vector -> $FFF8
    rom[0xFFE] = 0xF8
    rom[0xFFF] = 0xFF

    return Bios7800(bytes(rom))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MachineFactory:
    """Create an emulated Atari machine from a ROM file."""

    @staticmethod
    def create(
        rom_path: str,
        machine_type: Optional[Union[MachineType, str]] = None,
        cart_type: Optional[Union[CartType, str]] = None,
        bios_path: Optional[str] = None,
        p1: Optional[Controller] = None,
        p2: Optional[Controller] = None,
    ) -> object:
        """Build and return a fully-wired machine ready to run.

        Parameters
        ----------
        rom_path:
            Filesystem path to the ROM file (``.a78``, ``.a26``, ``.bin``).
        machine_type:
            Override for the target machine.  Can be a :class:`MachineType`
            enum value or its name as a string (e.g. ``"A7800NTSC"``).
            When ``None``, the type is inferred from the ROM.
        cart_type:
            Override for the cartridge mapper.  Can be a :class:`CartType`
            enum value or its name as a string.  When ``None``, the type is
            inferred from the ROM size and header.
        bios_path:
            Path to an Atari 7800 BIOS ROM image.  Only meaningful for 7800
            machine types.  When ``None``, a minimal built-in stub is used.
        p1:
            Controller type for player 1.  ``None`` picks a sensible default.
        p2:
            Controller type for player 2.  ``None`` defaults to
            :attr:`Controller.Non`.

        Returns
        -------
        object
            A :class:`MachineBase` subclass instance (the concrete type
            depends on the *machine_type*).

        Raises
        ------
        FileNotFoundError
            If *rom_path* (or *bios_path*) does not exist.
        ValueError
            If an unrecognised string is passed for *machine_type* or
            *cart_type*.
        RuntimeError
            If the core emulation modules are not yet available.
        """
        # -- resolve string overrides to enum values -----------------------
        if isinstance(machine_type, str):
            machine_type = MachineType[machine_type]
        if isinstance(cart_type, str):
            cart_type = CartType[cart_type]

        # -- read ROM bytes ------------------------------------------------
        logger.info("Loading ROM: %s", rom_path)
        rom_bytes = RomBytesService.read(rom_path)
        logger.info("ROM size (without header): %d bytes", len(rom_bytes))

        # -- infer cart type -----------------------------------------------
        if cart_type is None:
            cart_type = RomBytesService.infer_cart_type(rom_bytes, rom_path)
        logger.info("Cart type: %s", cart_type.name)

        # -- infer machine type --------------------------------------------
        if machine_type is None:
            machine_type = RomBytesService.infer_machine_type(rom_path, cart_type)
        logger.info("Machine type: %s", machine_type.name)

        # -- infer controllers ---------------------------------------------
        if p1 is None or p2 is None:
            default_p1, default_p2 = RomBytesService.infer_controllers(
                rom_path, cart_type
            )
            if p1 is None:
                p1 = default_p1
            if p2 is None:
                p2 = default_p2
        logger.info("Controllers: P1=%s, P2=%s", p1.name, p2.name)

        # -- create cartridge ----------------------------------------------
        if Cart is None:
            raise RuntimeError(
                "Cart class not available.  The core cartridge module "
                "(emu7800.core.carts.cart) has not been ported yet."
            )
        cart = Cart.create(rom_bytes, cart_type)
        logger.info("Cart created: %r", cart)

        # -- load or create BIOS -------------------------------------------
        bios = MachineFactory._load_bios(bios_path, machine_type)
        logger.info("BIOS: %r", bios)

        # -- create machine ------------------------------------------------
        if MachineBase is None:
            raise RuntimeError(
                "MachineBase class not available.  The core machine module "
                "(emu7800.core.machine_base) has not been ported yet."
            )
        machine = MachineBase.create(machine_type, cart, bios, p1, p2)
        logger.info("Machine created: %r", machine)

        return machine

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _load_bios(
        bios_path: Optional[str],
        machine_type: MachineType,
    ) -> Bios7800:
        """Load a 7800 BIOS image or create a default stub.

        For 2600 machine types the BIOS is never executed, but a valid
        :class:`Bios7800` is still returned so that callers never receive
        ``None``.

        Parameters
        ----------
        bios_path:
            Optional path to a real BIOS ROM.
        machine_type:
            The target machine type (used to decide NTSC/PAL default).

        Returns
        -------
        Bios7800
        """
        if bios_path is not None:
            bios_path = os.path.expanduser(bios_path)
            if not os.path.isfile(bios_path):
                raise FileNotFoundError(
                    f"BIOS file not found: {bios_path}"
                )
            with open(bios_path, "rb") as fh:
                bios_data = fh.read()
            logger.info(
                "Loaded BIOS from %s (%d bytes)", bios_path, len(bios_data)
            )
            return Bios7800(bios_data)

        # No BIOS supplied -- use the built-in stub.
        logger.info(
            "No BIOS path supplied; using built-in 4 KB stub for %s",
            machine_type.name,
        )
        return _make_default_bios_7800()

    @staticmethod
    def describe(rom_path: str) -> dict[str, str]:
        """Return a human-readable description of a ROM file.

        Useful for UI display or logging before creating the machine.

        Returns a dict with keys: ``title``, ``cart_type``, ``machine_type``,
        ``rom_size``, ``p1``, ``p2``.
        """
        rom_bytes, header = RomBytesService.read_with_header(rom_path)
        cart_type = RomBytesService.infer_cart_type(rom_bytes, rom_path)
        machine_type = RomBytesService.infer_machine_type(rom_path, cart_type)
        p1, p2 = RomBytesService.infer_controllers(rom_path, cart_type)

        title = header.title if header is not None else os.path.basename(rom_path)

        return {
            "title": title,
            "cart_type": cart_type.name,
            "machine_type": machine_type.name,
            "rom_size": str(len(rom_bytes)),
            "p1": p1.name,
            "p2": p2.name,
        }
