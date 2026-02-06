#!/usr/bin/env python3
"""
EMU7800 -- Atari 7800 / 2600 Emulator (Python Port)

Main entry point.  Parses command-line arguments, creates the emulated
machine from a ROM file, and launches the pygame display window.

Usage examples::

    # Run a ROM with auto-detected settings
    python main.py roms/game.a78

    # Force machine type and scale
    python main.py roms/game.bin --machine A2600NTSC --scale 4

    # Supply a 7800 BIOS image
    python main.py roms/game.a78 --bios roms/7800BIOS_U.rom

    # List ROM metadata without launching
    python main.py roms/game.a78 --info

    # Disable audio
    python main.py roms/game.a78 --no-audio
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``emu7800`` can be imported
# regardless of how the script is invoked.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from emu7800.core.types import CartType, Controller, MachineType
from emu7800.shell.services.machine_factory import MachineFactory
from emu7800.shell.services.rom_bytes_service import RomBytesService
from emu7800.platform.window import Window


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="emu7800",
        description=(
            "EMU7800 -- Atari 7800 / 2600 Emulator (Python port).  "
            "Load a ROM file and play it in a pygame window."
        ),
    )

    parser.add_argument(
        "rom",
        help="Path to the ROM file (.a78, .a26, .bin)",
    )

    # Machine / cart overrides
    machine_names = [mt.name for mt in MachineType if mt != MachineType.Unknown]
    parser.add_argument(
        "--machine", "-m",
        choices=machine_names,
        default=None,
        metavar="TYPE",
        help=(
            "Override the machine type (auto-detected from the ROM if not "
            "given).  Valid values: " + ", ".join(machine_names)
        ),
    )

    cart_names = [ct.name for ct in CartType if ct != CartType.Unknown]
    parser.add_argument(
        "--cart", "-c",
        choices=cart_names,
        default=None,
        metavar="TYPE",
        help=(
            "Override the cartridge mapper type (auto-detected if not given).  "
            "Valid values: " + ", ".join(cart_names)
        ),
    )

    # BIOS
    parser.add_argument(
        "--bios", "-b",
        default=None,
        metavar="PATH",
        help="Path to an Atari 7800 BIOS ROM image (optional).",
    )

    # Controllers
    ctrl_names = [c.name for c in Controller]
    parser.add_argument(
        "--p1",
        choices=ctrl_names,
        default=None,
        metavar="CTRL",
        help="Player 1 controller type.  Default: auto.",
    )
    parser.add_argument(
        "--p2",
        choices=ctrl_names,
        default=None,
        metavar="CTRL",
        help="Player 2 controller type.  Default: Non (unplugged).",
    )

    # Display
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=3,
        help="Display scale factor (1-8).  Default: 3.",
    )

    # Audio
    parser.add_argument(
        "--no-audio",
        action="store_true",
        default=False,
        help="Disable audio output.",
    )

    # Debugging / info
    parser.add_argument(
        "--info",
        action="store_true",
        default=False,
        help="Print ROM metadata and exit without launching the emulator.",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase log verbosity (-v for INFO, -vv for DEBUG).",
    )

    return parser


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbosity: int) -> None:
    """Set up the root logger based on requested verbosity."""
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Info mode
# ---------------------------------------------------------------------------

def _print_rom_info(rom_path: str) -> None:
    """Print human-readable metadata for a ROM and exit."""
    try:
        info = MachineFactory.describe(rom_path)
    except FileNotFoundError:
        print(f"Error: ROM file not found: {rom_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error reading ROM: {exc}", file=sys.stderr)
        sys.exit(1)

    print("EMU7800 ROM Information")
    print("=" * 40)
    for key, value in info.items():
        label = key.replace("_", " ").title()
        print(f"  {label:20s}: {value}")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """Application entry point.

    Parameters
    ----------
    argv:
        Command-line arguments.  ``None`` to use ``sys.argv``.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    logger = logging.getLogger("emu7800.main")

    # Validate the ROM path early.
    rom_path: str = os.path.expanduser(args.rom)
    if not os.path.isfile(rom_path):
        print(f"Error: ROM file not found: {rom_path}", file=sys.stderr)
        return 1

    # Info-only mode.
    if args.info:
        _print_rom_info(rom_path)
        return 0

    # Resolve optional controller overrides from strings.
    p1: Optional[Controller] = Controller[args.p1] if args.p1 else None
    p2: Optional[Controller] = Controller[args.p2] if args.p2 else None

    # Create the emulated machine.
    try:
        machine = MachineFactory.create(
            rom_path=rom_path,
            machine_type=args.machine,
            cart_type=args.cart,
            bios_path=args.bios,
            p1=p1,
            p2=p2,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Failed to create machine")
        print(f"Error creating machine: {exc}", file=sys.stderr)
        return 1

    # Launch the window.
    logger.info("Starting emulation ...")
    try:
        window = Window(
            machine,
            scale=args.scale,
            enable_audio=not args.no_audio,
        )
        window.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception("Fatal error during emulation")
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1

    logger.info("Exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
