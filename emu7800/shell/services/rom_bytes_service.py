"""
ROM loading and type-inference service for EMU7800.
Ported from the C# RomBytesService / RomPropertiesService classes.

Responsibilities:
  - Read ROM files from disk, stripping the .a78 header when present.
  - Parse the .a78 header to extract cart-type, controller, and TV-type metadata.
  - Infer CartType and MachineType from ROM size, file extension, and header data.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Optional

from emu7800.core.types import CartType, Controller, MachineType


# ---------------------------------------------------------------------------
# .a78 header constants
# ---------------------------------------------------------------------------

_A78_HEADER_SIZE: int = 128
_A78_SIGNATURE: bytes = b"ATARI7800"
_A78_SIGNATURE_OFFSET: int = 1
_A78_SIGNATURE_END: int = _A78_SIGNATURE_OFFSET + len(_A78_SIGNATURE)

# Cart-type bit definitions inside the 16-bit field at header[53:55]
_CT_POKEY_4000: int = 0x0001
_CT_SUPERGAME: int = 0x0002
_CT_SUPERGAME_RAM: int = 0x0004
_CT_ROM_4000: int = 0x0008
_CT_BANK6_4000: int = 0x0010
_CT_BANKED_RAM: int = 0x0020
_CT_POKEY_0450: int = 0x0040
_CT_MIRROR_RAM: int = 0x0080
_CT_ABSOLUTE: int = 0x0100
_CT_ACTIVISION: int = 0x0200
_CT_SG_9BANK: int = 0x0400

# Controller-type byte values in the .a78 header
_A78_CTRL_NONE: int = 0
_A78_CTRL_PROLINE: int = 1
_A78_CTRL_LIGHTGUN: int = 2
_A78_CTRL_PADDLES: int = 3
_A78_CTRL_TRACKBALL: int = 4
_A78_CTRL_JOYSTICK: int = 5
_A78_CTRL_KEYPAD: int = 6
_A78_CTRL_HPD: int = 7

_HEADER_CTRL_MAP: dict[int, Controller] = {
    _A78_CTRL_NONE: Controller.Non,
    _A78_CTRL_PROLINE: Controller.ProLineJoystick,
    _A78_CTRL_LIGHTGUN: Controller.Lightgun,
    _A78_CTRL_PADDLES: Controller.Paddles,
    _A78_CTRL_TRACKBALL: Controller.Joystick,
    _A78_CTRL_JOYSTICK: Controller.Joystick,
    _A78_CTRL_KEYPAD: Controller.Keypad,
    _A78_CTRL_HPD: Controller.ProLineJoystick,
}


# ---------------------------------------------------------------------------
# Parsed header data-class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class A78Header:
    """Parsed contents of a .a78 ROM header."""

    version: int
    title: str
    rom_size: int
    cart_type_bits: int
    controller1: Controller
    controller2: Controller
    is_pal: bool
    save_device: int


# ---------------------------------------------------------------------------
# Size -> CartType lookup tables
# ---------------------------------------------------------------------------

_SIZE_TO_CART_2600: dict[int, CartType] = {
    2048: CartType.A2K,
    4096: CartType.A4K,
    8192: CartType.A8K,
    12288: CartType.CBS12K,
    16384: CartType.A16K,
    32768: CartType.A32K,
}

_SIZE_TO_CART_7800: dict[int, CartType] = {
    8192: CartType.A7808,
    16384: CartType.A7816,
    32768: CartType.A7832,
    49152: CartType.A7848,
}

# Extensions that strongly suggest a 2600 ROM
_EXT_2600: frozenset[str] = frozenset({".a26", ".bin", ".2k", ".4k"})

# Extensions that strongly suggest a 7800 ROM
_EXT_7800: frozenset[str] = frozenset({".a78"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RomBytesService:
    """Static utility for loading ROM files and inferring metadata."""

    # -- reading -----------------------------------------------------------

    @staticmethod
    def read(path: str) -> bytes:
        """Read a ROM file from *path*, stripping the .a78 header if present.

        Returns:
            The raw ROM bytes (without the 128-byte header).

        Raises:
            FileNotFoundError: If *path* does not exist.
            OSError: On general I/O failure.
        """
        with open(path, "rb") as fh:
            data = fh.read()

        if RomBytesService._has_a78_header(data):
            return data[_A78_HEADER_SIZE:]
        return data

    @staticmethod
    def read_with_header(path: str) -> tuple[bytes, Optional[A78Header]]:
        """Read a ROM file and return ``(rom_bytes, header)``.

        If the file contains a valid .a78 header, *header* is a populated
        :class:`A78Header` and *rom_bytes* has the header stripped.
        Otherwise *header* is ``None`` and *rom_bytes* is the full file
        contents.
        """
        with open(path, "rb") as fh:
            data = fh.read()

        if RomBytesService._has_a78_header(data):
            header = RomBytesService._parse_header(data[:_A78_HEADER_SIZE])
            return data[_A78_HEADER_SIZE:], header
        return data, None

    # -- type inference ----------------------------------------------------

    @staticmethod
    def infer_cart_type(rom_bytes: bytes, path: str) -> CartType:
        """Determine the :class:`CartType` for a ROM.

        The decision is based on:

        1. The .a78 header (if present in the original file).
        2. The file extension.
        3. The ROM size.

        Parameters:
            rom_bytes: The ROM data **after** any header has been stripped
                       (i.e. as returned by :meth:`read`).
            path: Original file path -- used for extension-based heuristics.

        Returns:
            The best-guess :class:`CartType`.
        """
        # Try header first -- re-read the raw file for header bytes.
        header = RomBytesService._try_read_header(path)
        if header is not None:
            cart = RomBytesService._cart_type_from_header(
                header.cart_type_bits, len(rom_bytes)
            )
            if cart != CartType.Unknown:
                return cart

        ext = os.path.splitext(path)[1].lower()
        size = len(rom_bytes)

        # Explicit 2600 extension
        if ext in _EXT_2600:
            return _SIZE_TO_CART_2600.get(size, CartType.A4K)

        # Explicit 7800 extension (but no usable header)
        if ext in _EXT_7800:
            return _SIZE_TO_CART_7800.get(size, CartType.A7832)

        # No extension hint -- try 2600 sizes first (they are more common
        # for headerless ROMs), then 7800 sizes.
        if size in _SIZE_TO_CART_2600:
            return _SIZE_TO_CART_2600[size]
        if size in _SIZE_TO_CART_7800:
            return _SIZE_TO_CART_7800[size]

        # Large ROMs are almost certainly 7800 SuperGame.
        if size >= 65536:
            return CartType.A78SG
        if size > 32768:
            return CartType.A7848

        # Final fallback
        return CartType.A4K

    @staticmethod
    def infer_machine_type(path: str, cart_type: CartType) -> MachineType:
        """Determine the :class:`MachineType` from the cart type and file.

        Uses the .a78 header's TV-type byte (NTSC/PAL) when available;
        otherwise defaults to NTSC for the appropriate console.

        Parameters:
            path: Original file path -- used to re-read the .a78 header.
            cart_type: Previously inferred cart type.

        Returns:
            The best-guess :class:`MachineType`.
        """
        is_7800 = RomBytesService._is_7800_cart(cart_type)

        # Check the header for PAL indication
        header = RomBytesService._try_read_header(path)
        is_pal = header.is_pal if header is not None else False

        if is_7800:
            return MachineType.A7800PAL if is_pal else MachineType.A7800NTSC
        return MachineType.A2600PAL if is_pal else MachineType.A2600NTSC

    @staticmethod
    def infer_controllers(path: str, cart_type: CartType) -> tuple[Controller, Controller]:
        """Return the best-guess controllers for player 1 and player 2.

        Uses the .a78 header when present; otherwise picks sensible defaults
        based on the machine type.
        """
        header = RomBytesService._try_read_header(path)
        if header is not None:
            return header.controller1, header.controller2

        is_7800 = RomBytesService._is_7800_cart(cart_type)
        if is_7800:
            return Controller.ProLineJoystick, Controller.Non
        return Controller.Joystick, Controller.Non

    # -- private helpers ---------------------------------------------------

    @staticmethod
    def _has_a78_header(data: bytes) -> bool:
        """Return ``True`` if *data* begins with a valid .a78 header."""
        if len(data) < _A78_HEADER_SIZE + 1:
            return False
        return data[_A78_SIGNATURE_OFFSET:_A78_SIGNATURE_END] == _A78_SIGNATURE

    @staticmethod
    def _parse_header(header_bytes: bytes) -> A78Header:
        """Parse the first 128 bytes of a .a78 file into an :class:`A78Header`."""
        version = header_bytes[0]
        title = (
            header_bytes[17:49]
            .rstrip(b"\x00")
            .decode("ascii", errors="replace")
            .strip()
        )
        rom_size = struct.unpack_from(">I", header_bytes, 49)[0]
        cart_type_bits = struct.unpack_from(">H", header_bytes, 53)[0]

        ctrl1_byte = header_bytes[55]
        ctrl2_byte = header_bytes[56]
        controller1 = _HEADER_CTRL_MAP.get(ctrl1_byte, Controller.ProLineJoystick)
        controller2 = _HEADER_CTRL_MAP.get(ctrl2_byte, Controller.Non)

        is_pal = header_bytes[57] != 0
        save_device = header_bytes[58]

        return A78Header(
            version=version,
            title=title,
            rom_size=rom_size,
            cart_type_bits=cart_type_bits,
            controller1=controller1,
            controller2=controller2,
            is_pal=is_pal,
            save_device=save_device,
        )

    @staticmethod
    def _try_read_header(path: str) -> Optional[A78Header]:
        """Attempt to read and parse a .a78 header from *path*.

        Returns ``None`` if the file cannot be read or has no valid header.
        """
        try:
            with open(path, "rb") as fh:
                data = fh.read(_A78_HEADER_SIZE + 16)
        except OSError:
            return None

        if RomBytesService._has_a78_header(data):
            return RomBytesService._parse_header(data[:_A78_HEADER_SIZE])
        return None

    @staticmethod
    def _cart_type_from_header(cart_type_bits: int, rom_size: int) -> CartType:
        """Map the .a78 header cart-type bitfield to a :class:`CartType`.

        Parameters:
            cart_type_bits: The 16-bit value from header[53:55].
            rom_size: The ROM data size (header stripped).

        Returns:
            A :class:`CartType`, or ``CartType.Unknown`` if the bits cannot be
            decoded.
        """
        is_abs = bool(cart_type_bits & _CT_ABSOLUTE)
        is_act = bool(cart_type_bits & _CT_ACTIVISION)
        is_sg = bool(cart_type_bits & _CT_SUPERGAME)
        has_ram = bool(cart_type_bits & _CT_SUPERGAME_RAM)
        has_pokey_4000 = bool(cart_type_bits & _CT_POKEY_4000)
        has_pokey_0450 = bool(cart_type_bits & _CT_POKEY_0450)
        has_rom_4000 = bool(cart_type_bits & _CT_ROM_4000)
        has_bank6 = bool(cart_type_bits & _CT_BANK6_4000)
        has_banked_ram = bool(cart_type_bits & _CT_BANKED_RAM)
        is_sg9 = bool(cart_type_bits & _CT_SG_9BANK)

        # Absolute / Activision special mappers
        if is_abs:
            return CartType.A78AB
        if is_act:
            return CartType.A78AC

        # SuperGame variants
        if is_sg:
            if is_sg9:
                if has_pokey_4000 or has_pokey_0450:
                    return CartType.A78S9PL
                return CartType.A78S9
            if has_ram and (has_pokey_4000 or has_pokey_0450):
                return CartType.A78SGP
            if has_ram:
                return CartType.A78SGR
            if has_pokey_4000 or has_pokey_0450:
                return CartType.A78SGP
            return CartType.A78SG

        # Bankset (large ROMs) variants
        if rom_size > 49152:
            if rom_size <= 65536:
                if has_ram and (has_pokey_4000 or has_pokey_0450):
                    return CartType.A78BB52KP
                return CartType.A78BB52K
            if rom_size <= 131072:
                if has_ram and (has_pokey_4000 or has_pokey_0450):
                    return CartType.A78BB128KP
                if has_ram:
                    return CartType.A78BB128KR
                return CartType.A78BB128K

        # Standard (non-banked) sizes
        if rom_size <= 8192:
            return CartType.A7808
        if rom_size <= 16384:
            return CartType.A7816
        if rom_size <= 32768:
            if has_pokey_4000 or has_pokey_0450:
                return CartType.A7832P
            return CartType.A7832
        if rom_size <= 49152:
            return CartType.A7848

        return CartType.Unknown

    @staticmethod
    def _is_7800_cart(cart_type: CartType) -> bool:
        """Return ``True`` if *cart_type* is an Atari 7800 cartridge type."""
        return cart_type.value >= CartType.A7808.value
