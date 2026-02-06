"""
MachineBase -- abstract base class for all emulated Atari machines.
Ported from the C# MachineBase class.

Every emulated system (2600 NTSC, 2600 PAL, 7800 NTSC, 7800 PAL, etc.)
inherits from MachineBase.  The base class holds the components shared by
every variant:

* **CPU** -- the MOS 6502 (or 6502C for the 7800).
* **Address space** -- page-granularity memory map.
* **PIA** -- the 6532 RIOT (RAM-I/O-Timer).
* **Cart** -- the cartridge ROM / mapper.
* **FrameBuffer** -- video and audio output buffers.
* **InputState** -- controller / console-switch input.

Subclasses wire up the system-specific video chip (TIA for the 2600,
Maria for the 7800) and implement :meth:`compute_next_frame`.

The static :meth:`create` factory builds the correct subclass for a
given :class:`~emu7800.core.types.MachineType`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from emu7800.core.frame_buffer import FrameBuffer

if TYPE_CHECKING:
    from emu7800.core.address_space import AddressSpace
    from emu7800.core.devices import Bios7800, IDevice
    from emu7800.core.input_state import InputState
    from emu7800.core.m6502 import M6502
    from emu7800.core.pia import PIA
    from emu7800.core.types import MachineType


class MachineBase(ABC):
    """Abstract base class for all emulated Atari machines.

    Parameters
    ----------
    scanlines:
        Total number of scanlines per frame (262 for NTSC, 312 for PAL).
    first_scanline:
        The first *visible* scanline (used by the video chip to know
        where the active picture begins).
    frame_hz:
        Frame rate in Hz (60 for NTSC, 50 for PAL).  Clamped to >= 1.
    sound_sample_freq:
        Audio sample rate in Hz (e.g. 31440 for NTSC, 31200 for PAL).
    palette:
        A list of 256 ``uint32`` ARGB colour values for the system.
    visible_pitch:
        Horizontal pixel count per scanline (160 for 2600, 320 for 7800).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        scanlines: int,
        first_scanline: int,
        frame_hz: int,
        sound_sample_freq: int,
        palette: List[int],
        visible_pitch: int,
    ) -> None:
        # Core hardware components -- populated by subclass constructors.
        self.cpu: Optional[M6502] = None
        self.mem: Optional[AddressSpace] = None
        self.pia: Optional[PIA] = None
        self.cart: Optional[IDevice] = None

        # Output buffers.
        self.frame_buffer: FrameBuffer = FrameBuffer(visible_pitch, scanlines)

        # Input handling -- constructed eagerly so callers can register
        # inputs before the first frame.
        from emu7800.core.input_state import InputState
        self.input_state: InputState = InputState()

        # Machine run-state.
        self.machine_halt: bool = False
        self.frame_number: int = 0

        # Timing / display parameters.
        self.first_scanline: int = first_scanline
        self.frame_hz: int = max(1, frame_hz)
        self.sound_sample_frequency: int = sound_sample_freq

        # Palette: 256 ARGB uint32 values used to render the indexed
        # video buffer into true-colour output.
        self.palette: List[int] = list(palette)

        # When True, reads of NOP / unmapped TIA registers return the
        # data-bus state instead of zero.  Some games depend on this
        # behaviour.
        self.nop_register_dumping: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the machine to its power-on state.

        Subclasses **must** call ``super().reset()`` first, then reset
        their own chip instances (TIA/Maria, PIA, Cart, RAM, etc.).
        """
        self.frame_number = 0
        self.machine_halt = False
        self.input_state.clear_all_input()

    def compute_next_frame(self) -> None:
        """Advance the emulation by one video frame.

        The base implementation captures input, increments the frame
        counter, and clears the sound buffer.  Subclasses **must** call
        ``super().compute_next_frame()`` first, then drive the
        chip-specific scanline loop.

        If :attr:`machine_halt` is ``True`` the method returns
        immediately without doing any work.
        """
        if self.machine_halt:
            return
        self.input_state.capture_input_state()
        self.frame_number += 1
        # Clear the sound buffer for the upcoming frame.
        sb = self.frame_buffer.sound_buffer
        for i in range(len(sb)):
            sb[i] = 0

    # ------------------------------------------------------------------
    # Serialisation helpers (save-state support)
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return a serialisable snapshot of the base machine state.

        Subclasses should call ``super().get_snapshot()`` and merge the
        result with their own chip snapshots.
        """
        return {
            "machine_halt": self.machine_halt,
            "frame_number": self.frame_number,
        }

    def restore_snapshot(self, snapshot: dict) -> None:
        """Restore base machine state from a previous snapshot.

        Subclasses should call ``super().restore_snapshot(snapshot)``
        and then restore their own chip state.
        """
        self.machine_halt = snapshot.get("machine_halt", False)
        self.frame_number = snapshot.get("frame_number", 0)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        machine_type: MachineType,
        cart: IDevice,
        bios: Optional[Bios7800],
        p1: int,
        p2: int,
    ) -> MachineBase:
        """Create the appropriate machine variant for *machine_type*.

        This is the primary entry point for constructing a fully-wired
        emulated system.

        Parameters
        ----------
        machine_type:
            The specific Atari system variant to emulate (see
            :class:`~emu7800.core.types.MachineType`).
        cart:
            The cartridge device to install.
        bios:
            Optional 7800 BIOS ROM.  Ignored for 2600 machine types.
            Required for 7800 ``*bios`` / ``*hsc`` / ``*xm`` variants.
        p1:
            Controller type for player 1 (a
            :class:`~emu7800.core.types.Controller` value).
        p2:
            Controller type for player 2 (a
            :class:`~emu7800.core.types.Controller` value).

        Returns
        -------
        MachineBase
            A fully-wired machine ready for :meth:`reset` and
            :meth:`compute_next_frame`.

        Raises
        ------
        ValueError
            If *machine_type* is not a recognised variant.
        """
        # Local imports to avoid circular dependencies and to defer
        # loading the concrete subclasses until they are actually needed.
        from emu7800.core.types import MachineType as MT
        from emu7800.core.machine_2600 import Machine2600NTSC, Machine2600PAL
        from emu7800.core.machine_7800 import Machine7800NTSC, Machine7800PAL

        if machine_type == MT.A2600NTSC:
            return Machine2600NTSC(cart, p1, p2)
        elif machine_type == MT.A2600PAL:
            return Machine2600PAL(cart, p1, p2)
        elif MT.is_7800_ntsc(machine_type):
            return Machine7800NTSC(cart, bios, machine_type, p1, p2)
        elif MT.is_7800_pal(machine_type):
            return Machine7800PAL(cart, bios, machine_type, p1, p2)
        else:
            raise ValueError(f"Unsupported machine type: {machine_type}")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"frame_hz={self.frame_hz}, "
            f"scanlines={self.frame_buffer.scanlines}, "
            f"first_scanline={self.first_scanline}, "
            f"frame={self.frame_number})"
        )
