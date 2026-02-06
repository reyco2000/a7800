"""
InputState - Manages controller and console switch state with double-buffering.

Ported from the EMU7800 C# InputState class. Provides a staging buffer
(_next_input_state) where host code writes input events, and a captured
buffer (_input_state) that the emulation core reads during a frame.

Index layout of the state arrays (size INPUT_STATE_SIZE = 15):
  [0]     left_controller_jack   (Controller enum value)
  [1]     right_controller_jack  (Controller enum value)
  [2]     console_switch         (bitmask: 1 << ConsoleSwitch)
  [3..6]  controller_action_state per player 0-3  (bitmask: 1 << ControllerAction)
  [7..10] ohms per player 0-3  /  lightgun position (overloaded)
  [11..14] reserved
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from emu7800.core.types import (
    Controller,
    ControllerAction,
    ConsoleSwitch,
    MachineInput,
)

if TYPE_CHECKING:
    pass  # Machine forward reference would go here

INPUT_STATE_SIZE: int = 15


# ---------------------------------------------------------------------------
# MachineInput -> ControllerAction lookup (built once at module level)
# ---------------------------------------------------------------------------

_MACHINE_INPUT_TO_CONTROLLER_ACTION: dict[MachineInput, ControllerAction] = {
    MachineInput.Fire:      ControllerAction.Trigger,
    MachineInput.Fire2:     ControllerAction.Trigger2,
    MachineInput.Up:        ControllerAction.Up,
    MachineInput.Down:      ControllerAction.Down,
    MachineInput.Left:      ControllerAction.Left,
    MachineInput.Right:     ControllerAction.Right,
    MachineInput.NumPad7:   ControllerAction.Keypad7,
    MachineInput.NumPad8:   ControllerAction.Keypad8,
    MachineInput.NumPad9:   ControllerAction.Keypad9,
    MachineInput.NumPad4:   ControllerAction.Keypad4,
    MachineInput.NumPad5:   ControllerAction.Keypad5,
    MachineInput.NumPad6:   ControllerAction.Keypad6,
    MachineInput.NumPad1:   ControllerAction.Keypad1,
    MachineInput.NumPad2:   ControllerAction.Keypad2,
    MachineInput.NumPad3:   ControllerAction.Keypad3,
    MachineInput.NumPadMult: ControllerAction.KeypadA,
    MachineInput.NumPad0:   ControllerAction.Keypad0,
    MachineInput.NumPadHash: ControllerAction.KeypadP,
    MachineInput.Driving0:  ControllerAction.Driving0,
    MachineInput.Driving1:  ControllerAction.Driving1,
    MachineInput.Driving2:  ControllerAction.Driving2,
    MachineInput.Driving3:  ControllerAction.Driving3,
}

# MachineInput -> ConsoleSwitch lookup
_MACHINE_INPUT_TO_CONSOLE_SWITCH: dict[MachineInput, ConsoleSwitch] = {
    MachineInput.Pause:           ConsoleSwitch.Pause,
    MachineInput.Reset:           ConsoleSwitch.GameReset,
    MachineInput.Select:          ConsoleSwitch.GameSelect,
    MachineInput.Color:           ConsoleSwitch.GameBW,
    MachineInput.LeftDifficulty:  ConsoleSwitch.LeftDifficultyA,
    MachineInput.RightDifficulty: ConsoleSwitch.RightDifficultyA,
}


class InputState:
    """Manages controller and console-switch input state with double-buffering.

    Host code writes into *_next_input_state* via :meth:`raise_input`,
    :meth:`set_ohm_state`, and :meth:`set_light_gun_position`.

    At every frame boundary the emulation core calls :meth:`capture_input_state`
    which snapshots the staging buffer into *_input_state*.  The emulation
    hardware (PIA, TIA) then reads from the captured buffer through the
    various ``sample_captured_*`` methods.
    """

    # Gray codes for the four driving-controller rotation positions.
    # Only the low two bits carry the gray-code value; the upper two bits
    # are set to 1 (inactive-high, matching SWCHA convention).
    _rot_gray_codes: list[int] = [0x0f, 0x0d, 0x0c, 0x0e]

    def __init__(self, machine: object = None) -> None:
        self._machine = machine
        self._next_input_state: list[int] = [0] * INPUT_STATE_SIZE
        self._input_state: list[int] = [0] * INPUT_STATE_SIZE
        # Per-player driving-controller rotation position (0-3).
        # Only player 0 and 1 are meaningful (one per jack).
        self._rot_state: list[int] = [0, 0]

    # ------------------------------------------------------------------
    # Controller-jack properties (read/write the *staging* buffer)
    # ------------------------------------------------------------------

    @property
    def left_controller_jack(self) -> Controller:
        """Controller type plugged into the left jack."""
        return Controller(self._next_input_state[0])

    @left_controller_jack.setter
    def left_controller_jack(self, value: Controller) -> None:
        self._next_input_state[0] = int(value)

    @property
    def right_controller_jack(self) -> Controller:
        """Controller type plugged into the right jack."""
        return Controller(self._next_input_state[1])

    @right_controller_jack.setter
    def right_controller_jack(self, value: Controller) -> None:
        self._next_input_state[1] = int(value)

    # ------------------------------------------------------------------
    # Frame-boundary snapshot
    # ------------------------------------------------------------------

    def capture_input_state(self) -> None:
        """Copy the staging buffer to the captured buffer.

        Called once per frame, before the emulation core begins reading
        controller / switch state for that frame.
        """
        self._input_state[:] = self._next_input_state[:]

    # ------------------------------------------------------------------
    # Host-side input event injection
    # ------------------------------------------------------------------

    def raise_input(self, player_no: int, machine_input: MachineInput, down: bool) -> None:
        """Translate a high-level *MachineInput* event into the internal
        controller-action or console-switch bitmask.

        Parameters
        ----------
        player_no : int
            Player index (0-3).  Ignored for console-switch inputs.
        machine_input : MachineInput
            The logical input that was pressed or released.
        down : bool
            ``True`` if the input is now active (pressed / on),
            ``False`` if released / off.
        """
        if player_no < 0 or player_no > 3:
            return

        action = _MACHINE_INPUT_TO_CONTROLLER_ACTION.get(machine_input)
        if action is not None:
            self._set_controller_action_state(player_no, action, down)
            return

        switch = _MACHINE_INPUT_TO_CONSOLE_SWITCH.get(machine_input)
        if switch is not None:
            self._set_console_switch_state(switch, down)
            return

        # MachineInput.End and any unknown values are silently ignored.

    # ------------------------------------------------------------------
    # Ohm / lightgun position helpers  (write staging buffer)
    # ------------------------------------------------------------------

    def set_ohm_state(self, player_no: int, ohms: int) -> None:
        """Set the paddle potentiometer resistance for *player_no*."""
        if 0 <= player_no <= 3:
            self._next_input_state[7 + player_no] = ohms

    def set_light_gun_position(self, player_no: int, scanline: int, hpos: int) -> None:
        """Set the light-gun screen position for *player_no*.

        The value is packed as ``(scanline << 16) | (hpos & 0xffff)``.
        """
        if 0 <= player_no <= 3:
            self._next_input_state[7 + player_no] = (scanline << 16) | (hpos & 0xffff)

    # ------------------------------------------------------------------
    # Sampling (the emulation core reads from the *captured* buffer)
    # ------------------------------------------------------------------

    def sample_captured_console_switch_state(self, switch: ConsoleSwitch) -> bool:
        """Return ``True`` if *switch* is active in the captured state."""
        return (self._input_state[2] & (1 << int(switch))) != 0

    def sample_captured_controller_action_state(self, player_no: int,
                                                action: ControllerAction) -> bool:
        """Return ``True`` if *action* is active for *player_no* in the
        captured state."""
        if player_no < 0 or player_no > 3:
            return False
        return (self._input_state[3 + player_no] & (1 << int(action))) != 0

    def sample_captured_ohm_state(self, player_no: int) -> int:
        """Return the paddle resistance (ohms) for *player_no*."""
        if player_no < 0 or player_no > 3:
            return 0
        return self._input_state[7 + player_no]

    def sample_captured_light_gun_position(self, player_no: int) -> Tuple[int, int]:
        """Return ``(scanline, hpos)`` for the light gun of *player_no*."""
        if player_no < 0 or player_no > 3:
            return (0, 0)
        packed = self._input_state[7 + player_no]
        return (packed >> 16, packed & 0xffff)

    def sample_captured_driving_state(self, player_no: int) -> int:
        """Return the 4-bit driving-controller gray code for *player_no*.

        If any ``Driving0``-``Driving3`` action is currently active, the
        internal rotation position is updated to the latest one found.
        The gray code for that position is then returned.  If no driving
        action is active the previously latched position is used.
        """
        if player_no < 0 or player_no > 1:
            return 0
        action_bits = self._input_state[3 + player_no]
        for i in range(4):
            if action_bits & (1 << int(ControllerAction.Driving0 + i)):
                self._rot_state[player_no] = i
        return self._rot_gray_codes[self._rot_state[player_no]]

    # ------------------------------------------------------------------
    # Bulk clear helpers
    # ------------------------------------------------------------------

    def clear_all_input(self) -> None:
        """Clear all controller-action state, console switches, and
        auxiliary (ohm / lightgun) state.  Controller-jack assignments
        (indices 0-1) are preserved."""
        for i in range(2, INPUT_STATE_SIZE):
            self._next_input_state[i] = 0

    def clear_input_by_player(self, player_no: int) -> None:
        """Clear the controller-action and auxiliary state for a single player."""
        if 0 <= player_no <= 3:
            self._next_input_state[3 + player_no] = 0
            self._next_input_state[7 + player_no] = 0

    def clear_console_switches(self) -> None:
        """Clear all console-switch state."""
        self._next_input_state[2] = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_controller_action_state(self, player_no: int,
                                     action: ControllerAction,
                                     down: bool) -> None:
        """Set or clear a single controller-action bit for *player_no*
        in the staging buffer."""
        bit = 1 << int(action)
        if down:
            self._next_input_state[3 + player_no] |= bit
        else:
            self._next_input_state[3 + player_no] &= ~bit

    def _set_console_switch_state(self, switch: ConsoleSwitch,
                                  down: bool) -> None:
        """Set or clear a single console-switch bit in the staging buffer."""
        bit = 1 << int(switch)
        if down:
            self._next_input_state[2] |= bit
        else:
            self._next_input_state[2] &= ~bit
