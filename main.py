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

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print Maria/TIA diagnostic info for the first 5 frames and exit.",
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

def _run_debug(machine) -> int:
    """Run a few frames and print Maria/TIA diagnostic information."""
    print("=" * 60)
    print("EMU7800 Debug Diagnostics")
    print("=" * 60)
    print(f"Machine: {machine}")
    print(f"CPU PC after reset: ${machine.cpu.pc:04X}")
    print(f"CPU jammed: {machine.cpu.jammed}")

    is_7800 = hasattr(machine, '_maria')

    machine.reset()
    print(f"CPU PC after reset: ${machine.cpu.pc:04X}")

    for frame_no in range(5):
        machine.compute_next_frame()

        fb = machine.frame_buffer
        vbuf = fb.video_buffer
        non_zero = sum(1 for b in vbuf if b != 0)
        unique_vals = set(vbuf)

        print(f"\n--- Frame {frame_no + 1} ---")
        print(f"  CPU: PC=${machine.cpu.pc:04X} jammed={machine.cpu.jammed} clock={machine.cpu.clock}")

        if is_7800:
            maria = machine._maria
            regs = maria._registers
            print(f"  Maria: dma_enabled={maria._dma_enabled} rm={maria._rm}")
            print(f"  Maria: color_kill={maria._color_kill} cwidth={maria._cwidth} kangaroo={maria._kangaroo}")
            print(f"  Maria: ctrl_lock={maria._ctrl_lock}")
            print(f"  Maria DPPL=${regs[0x30]:02X} DPPH=${regs[0x2C]:02X} -> DLL=${regs[0x2C]:02X}{regs[0x30]:02X}")
            print(f"  Maria BACKGRND=${regs[0x20]:02X} CHARBASE=${regs[0x34]:02X}")
            print(f"  Palettes: P0=[{regs[0x21]:02X},{regs[0x22]:02X},{regs[0x23]:02X}] "
                  f"P1=[{regs[0x25]:02X},{regs[0x26]:02X},{regs[0x27]:02X}]")
        else:
            tia = machine._tia
            print(f"  TIA: end_of_frame={tia.end_of_frame}")
            print(f"  TIA: COLUBK=${tia.colubk:02X} COLUPF=${tia.colupf:02X} "
                  f"COLUP0=${tia.colup0:02X} COLUP1=${tia.colup1:02X}")

        print(f"  Video buffer: {non_zero}/{len(vbuf)} non-zero pixels")
        print(f"  Unique palette indices: {sorted(unique_vals)[:20]}"
              + ("..." if len(unique_vals) > 20 else ""))

        # Sample a few scanlines
        pitch = fb.visible_pitch
        for sl in [0, 16, 50, 100, 130]:
            start = sl * pitch
            end = start + pitch
            line_data = vbuf[start:end]
            line_nz = sum(1 for b in line_data if b != 0)
            line_unique = sorted(set(line_data))
            if line_nz > 0:
                print(f"  Scanline {sl}: {line_nz} non-zero, vals={line_unique[:10]}")

    print("\n" + "=" * 60)
    print("Debug complete.")
    return 0


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

    # Debug mode: run a few frames and print diagnostics.
    if args.debug:
        return _run_debug(machine)

    # Launch the window.
    logger.info("Starting emulation ...")
    machine.reset()
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
