"""
PIA (6532 RIOT) - Peripheral Interface Adapter.

Ported from the EMU7800 C# PIA class.  The 6532 RIOT provides:

* 128 bytes of static RAM
* A programmable interval timer with four selectable clock dividers
  (1, 8, 64, 1024 CPU clocks per tick)
* Port A (SWCHA) -- directly connected to the two controller jacks;
  active-low joystick directions, driving gray codes, paddle buttons
* Port B (SWCHB) -- active-low console switches (Reset, Select,
  Color/BW, P0/P1 Difficulty)
* Data direction registers DDRA and DDRB that control which bits of
  each port are inputs vs. outputs

Address decoding
----------------
Peek (read):
    addr & 0x200 == 0  ->  RAM[addr & 0x7f]
    addr & 0x200 != 0  ->  I/O register selected by (addr & 7):
        0 : Port A  (SWCHA, after DDRA masking)
        1 : DDRA
        2 : Port B  (SWCHB, after DDRB masking)
        3 : 0       (DDRB -- reads as 0 on the 2600)
        4,6 : Timer value
        5,7 : Interrupt flag register

Poke (write):
    addr & 0x200 == 0  ->  RAM[addr & 0x7f] = data
    addr & 0x200 != 0:
        addr & 0x04 set:
            addr & 0x10 set -> write timer:
                timer_shift = {0:0, 1:3, 2:6, 3:10}[addr & 3]
                timer_target = cpu.clock + (data << timer_shift)
            (addr & 0x10 clear -> edge-detect control, ignored)
        addr & 0x04 clear:
            addr & 3: 0=Port A, 1=DDRA, 2=Port B, 3=DDRB
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emu7800.core.types import (
    Controller,
    ControllerAction,
    ConsoleSwitch,
)

if TYPE_CHECKING:
    from emu7800.core.input_state import InputState


class PIA:
    """6532 RIOT (RAM-I/O-Timer) chip."""

    # addr & 3  ->  timer shift (dividers 1, 8, 64, 1024)
    _TIMER_SHIFTS: list[int] = [0, 3, 6, 10]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, machine: object) -> None:
        self._machine = machine

        # 128 bytes of on-chip RAM
        self.ram: list[int] = [0] * 128

        # Data-direction registers (0 = input, 1 = output)
        self._ddra: int = 0
        self._ddrb: int = 0

        # Latched values written by the CPU to ports A and B
        self._written_port_a: int = 0
        self._written_port_b: int = 0

        # Timer state --------------------------------------------------
        # timer_target is an absolute CPU-clock value.  The timer value
        # at any moment equals  (timer_target - cpu.clock) >> timer_shift
        # while the result is non-negative.  After underflow the timer
        # keeps counting at the 1x rate (shift = 0) and wraps at 0xff.
        self._timer_target: int = 0
        self._timer_shift: int = 10       # power-on default: 1024 divider
        self._timer_interrupt_flag: bool = False
        self._irq_enabled: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the PIA to power-on state."""
        for i in range(128):
            self.ram[i] = 0
        self._ddra = 0
        self._ddrb = 0
        self._written_port_a = 0
        self._written_port_b = 0
        self._timer_target = 0
        self._timer_shift = 10
        self._timer_interrupt_flag = False
        self._irq_enabled = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def written_port_a(self) -> int:
        """Last value the CPU wrote to Port A."""
        return self._written_port_a

    @property
    def written_port_b(self) -> int:
        """Last value the CPU wrote to Port B."""
        return self._written_port_b

    @property
    def ddra(self) -> int:
        """Data-direction register A (1 = output bit)."""
        return self._ddra

    @property
    def ddrb(self) -> int:
        """Data-direction register B (1 = output bit)."""
        return self._ddrb

    # ------------------------------------------------------------------
    # Bus interface
    # ------------------------------------------------------------------

    def __getitem__(self, addr: int) -> int:
        """Read a byte from the PIA address space (supports [] syntax)."""
        return self.peek(addr)

    def __setitem__(self, addr: int, value: int) -> None:
        """Write a byte to the PIA address space (supports [] syntax)."""
        self.poke(addr, value)

    def peek(self, addr: int) -> int:
        """Read a byte from the PIA address space."""
        if (addr & 0x200) == 0:
            return self.ram[addr & 0x7f]

        reg = addr & 0x07
        if reg == 0:
            return self._read_port_a()
        if reg == 1:
            return self._ddra
        if reg == 2:
            return self._read_port_b()
        if reg == 3:
            return 0  # DDRB reads as 0 on the Atari 2600
        if reg in (4, 6):
            return self._read_timer()
        # reg in (5, 7)
        return self._read_interrupt_flag()

    def poke(self, addr: int, data: int) -> None:
        """Write a byte to the PIA address space."""
        data &= 0xff

        if (addr & 0x200) == 0:
            self.ram[addr & 0x7f] = data
            return

        if addr & 0x04:
            # Timer / edge-detect region
            if addr & 0x10:
                # Write a new timer value
                self._timer_shift = self._TIMER_SHIFTS[addr & 0x03]
                self._timer_target = (
                    self._machine.cpu.clock + (data << self._timer_shift)
                )
                self._timer_interrupt_flag = False
                self._irq_enabled = bool(addr & 0x08)
            # else: edge-detect control -- not used on the 2600; ignored
        else:
            # I/O registers
            reg = addr & 0x03
            if reg == 0:
                self._written_port_a = data
            elif reg == 1:
                self._ddra = data
            elif reg == 2:
                self._written_port_b = data
            else:  # reg == 3
                self._ddrb = data

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _read_timer(self) -> int:
        """Return the current timer value (INTIM)."""
        delta = self._timer_target - self._machine.cpu.clock

        if delta >= 0:
            # Timer is still counting down at the programmed rate.
            return (delta >> self._timer_shift) & 0xff

        # Timer has underflowed.  After reaching 0 it continues to
        # decrement at the 1x clock rate, wrapping at 0xff.
        self._timer_interrupt_flag = True
        return delta & 0xff

    def _read_interrupt_flag(self) -> int:
        """Return the timer interrupt-flag register (TIMINT).

        Bit 7 is set if the timer has underflowed since the last
        timer-write.  The flag is also re-evaluated lazily here so it
        stays current even if no one read the timer itself.
        """
        delta = self._timer_target - self._machine.cpu.clock
        if delta < 0:
            self._timer_interrupt_flag = True
        return 0x80 if self._timer_interrupt_flag else 0x00

    # ------------------------------------------------------------------
    # Port A  (SWCHA -- controller directions / driving / paddles)
    # ------------------------------------------------------------------

    def _read_port_a(self) -> int:
        """Read SWCHA.

        Upper nybble = player 0 (left jack).
        Lower nybble = player 1 (right jack).

        For a standard joystick the bits are active-low::

            bit 7 : P0 Right    bit 3 : P1 Right
            bit 6 : P0 Left     bit 2 : P1 Left
            bit 5 : P0 Down     bit 1 : P1 Down
            bit 4 : P0 Up       bit 0 : P1 Up

        The final value mixes input bits with the written output
        according to DDRA:
        ``(written_port_a & ddra) | (input_bits & ~ddra)``
        """
        input_bits = 0xff
        input_state: InputState = self._machine.input_state

        jack0 = input_state.left_controller_jack
        jack1 = input_state.right_controller_jack

        # ---- Player 0 (upper nybble) --------------------------------

        if jack0 in (
            Controller.Joystick,
            Controller.ProLineJoystick,
            Controller.BoosterGrip,
            Controller.Lightgun,
        ):
            if input_state.sample_captured_controller_action_state(
                0, ControllerAction.Up
            ):
                input_bits &= ~0x10
            if input_state.sample_captured_controller_action_state(
                0, ControllerAction.Down
            ):
                input_bits &= ~0x20
            if input_state.sample_captured_controller_action_state(
                0, ControllerAction.Left
            ):
                input_bits &= ~0x40
            if input_state.sample_captured_controller_action_state(
                0, ControllerAction.Right
            ):
                input_bits &= ~0x80

        elif jack0 == Controller.Driving:
            gray = input_state.sample_captured_driving_state(0)
            input_bits = (input_bits & 0x0f) | ((gray & 0x0f) << 4)

        elif jack0 == Controller.Paddles:
            # Paddle 0 button -> bit 7 (right-direction line)
            # Paddle 1 button -> bit 6 (left-direction line)
            if input_state.sample_captured_controller_action_state(
                0, ControllerAction.Trigger
            ):
                input_bits &= ~0x80
            if input_state.sample_captured_controller_action_state(
                1, ControllerAction.Trigger
            ):
                input_bits &= ~0x40

        # Controller.Keypad and Controller.Non leave the nybble at 0xf0

        # ---- Player 1 (lower nybble) --------------------------------

        if jack1 in (
            Controller.Joystick,
            Controller.ProLineJoystick,
            Controller.BoosterGrip,
            Controller.Lightgun,
        ):
            if input_state.sample_captured_controller_action_state(
                1, ControllerAction.Up
            ):
                input_bits &= ~0x01
            if input_state.sample_captured_controller_action_state(
                1, ControllerAction.Down
            ):
                input_bits &= ~0x02
            if input_state.sample_captured_controller_action_state(
                1, ControllerAction.Left
            ):
                input_bits &= ~0x04
            if input_state.sample_captured_controller_action_state(
                1, ControllerAction.Right
            ):
                input_bits &= ~0x08

        elif jack1 == Controller.Driving:
            gray = input_state.sample_captured_driving_state(1)
            input_bits = (input_bits & 0xf0) | (gray & 0x0f)

        elif jack1 == Controller.Paddles:
            # Paddle 2 button -> bit 3 (right-direction line)
            # Paddle 3 button -> bit 2 (left-direction line)
            if input_state.sample_captured_controller_action_state(
                2, ControllerAction.Trigger
            ):
                input_bits &= ~0x08
            if input_state.sample_captured_controller_action_state(
                3, ControllerAction.Trigger
            ):
                input_bits &= ~0x04

        input_bits &= 0xff
        return (self._written_port_a & self._ddra) | (
            input_bits & (~self._ddra & 0xff)
        )

    # ------------------------------------------------------------------
    # Port B  (SWCHB -- console switches)
    # ------------------------------------------------------------------

    def _read_port_b(self) -> int:
        """Read SWCHB.

        All switches are active-low (0 = pressed / active)::

            bit 0 : Reset
            bit 1 : Select
            bit 2 : (unused, always 1)
            bit 3 : Color/BW  (1 = Color, 0 = BW)
            bit 4 : (unused, always 1)
            bit 5 : (unused, always 1)
            bit 6 : P0 Difficulty  (1 = B / Beginner, 0 = A / Advanced)
            bit 7 : P1 Difficulty  (1 = B / Beginner, 0 = A / Advanced)
        """
        # Start with all bits high (nothing pressed, Color mode,
        # both difficulties on B/easy).
        input_bits = 0xff
        input_state: InputState = self._machine.input_state

        # Active switches pull their bit low.
        if input_state.sample_captured_console_switch_state(
            ConsoleSwitch.GameReset
        ):
            input_bits &= ~0x01

        if input_state.sample_captured_console_switch_state(
            ConsoleSwitch.GameSelect
        ):
            input_bits &= ~0x02

        if input_state.sample_captured_console_switch_state(
            ConsoleSwitch.GameBW
        ):
            input_bits &= ~0x08

        if input_state.sample_captured_console_switch_state(
            ConsoleSwitch.LeftDifficultyA
        ):
            input_bits &= ~0x40

        if input_state.sample_captured_console_switch_state(
            ConsoleSwitch.RightDifficultyA
        ):
            input_bits &= ~0x80

        return (self._written_port_b & self._ddrb) | (
            input_bits & (~self._ddrb & 0xff)
        )
