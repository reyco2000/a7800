"""
Input handler for EMU7800.
Maps keyboard keys and gamepad buttons to emulated :class:`MachineInput` events.

Keyboard layout
---------------

===================  ============================  =======
Key                  Action                        Player
===================  ============================  =======
Arrow Up / W         Joystick Up                   0
Arrow Down / S       Joystick Down                 0
Arrow Left / A       Joystick Left                 0
Arrow Right / D      Joystick Right                0
Z / Space            Fire (trigger)                0
X                    Fire2 (trigger 2)             0
1-9, 0, ``*``, #    Keypad digits / symbols       0
F1                   Console Reset                 --
F2                   Console Select                --
F3                   Colour / B&W toggle           --
F4                   Left Difficulty toggle         --
F5                   Right Difficulty toggle        --
Escape               Quit                          --
===================  ============================  =======

When a pygame joystick / gamepad is connected the first controller is
automatically mapped to player 0.  D-pad maps to directions, buttons 0/1
map to Fire/Fire2.
"""

from __future__ import annotations

import logging
from typing import Optional

import pygame

from emu7800.core.types import MachineInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyboard -> (player, MachineInput) mappings
# ---------------------------------------------------------------------------
# Each entry is ``(pygame key constant) -> (player_number, MachineInput)``.
# Console switches (Reset, Select, ...) use player -1 by convention; the
# handler sends them with ``player=0`` which is how the C# EMU7800 works
# (the machine routes them regardless of player number).

_KEY_MAP: dict[int, tuple[int, MachineInput]] = {
    # -- Player 0 directions -----------------------------------------------
    pygame.K_UP:     (0, MachineInput.Up),
    pygame.K_DOWN:   (0, MachineInput.Down),
    pygame.K_LEFT:   (0, MachineInput.Left),
    pygame.K_RIGHT:  (0, MachineInput.Right),
    pygame.K_w:      (0, MachineInput.Up),
    pygame.K_s:      (0, MachineInput.Down),
    pygame.K_a:      (0, MachineInput.Left),
    pygame.K_d:      (0, MachineInput.Right),

    # -- Player 0 fire / action --------------------------------------------
    pygame.K_z:      (0, MachineInput.Fire),
    pygame.K_SPACE:  (0, MachineInput.Fire),
    pygame.K_x:      (0, MachineInput.Fire2),

    # -- Keypad (for Atari Keypad controller) ------------------------------
    pygame.K_1:      (0, MachineInput.NumPad1),
    pygame.K_2:      (0, MachineInput.NumPad2),
    pygame.K_3:      (0, MachineInput.NumPad3),
    pygame.K_4:      (0, MachineInput.NumPad4),
    pygame.K_5:      (0, MachineInput.NumPad5),
    pygame.K_6:      (0, MachineInput.NumPad6),
    pygame.K_7:      (0, MachineInput.NumPad7),
    pygame.K_8:      (0, MachineInput.NumPad8),
    pygame.K_9:      (0, MachineInput.NumPad9),
    pygame.K_0:      (0, MachineInput.NumPad0),
    pygame.K_KP_MULTIPLY: (0, MachineInput.NumPadMult),
    pygame.K_ASTERISK:    (0, MachineInput.NumPadMult),
    pygame.K_HASH:        (0, MachineInput.NumPadHash),
    pygame.K_3:           (0, MachineInput.NumPad3),  # '#' is Shift+3; handled specially

    # -- Console switches --------------------------------------------------
    pygame.K_F1:     (0, MachineInput.Reset),
    pygame.K_F2:     (0, MachineInput.Select),
    pygame.K_F3:     (0, MachineInput.Color),
    pygame.K_F4:     (0, MachineInput.LeftDifficulty),
    pygame.K_F5:     (0, MachineInput.RightDifficulty),
    pygame.K_p:      (0, MachineInput.Pause),
}

# Console switches that toggle on key-down and auto-release on key-up.
_MOMENTARY_SWITCHES: frozenset[MachineInput] = frozenset({
    MachineInput.Reset,
    MachineInput.Select,
})

# Console switches that cycle through states on each key press
# (Color/BW, Left/Right difficulty).  They do not use the ``down`` flag in
# the same way; instead we send a press and an immediate release.
_TOGGLE_SWITCHES: frozenset[MachineInput] = frozenset({
    MachineInput.Color,
    MachineInput.LeftDifficulty,
    MachineInput.RightDifficulty,
    MachineInput.Pause,
})

# Joystick hat (D-pad) directions -> MachineInput.
# Hat values are (x, y) with x: -1=left, 1=right; y: -1=down, 1=up.
_HAT_MAP: dict[tuple[int, int], list[MachineInput]] = {
    (0,  1): [MachineInput.Up],
    (0, -1): [MachineInput.Down],
    (-1, 0): [MachineInput.Left],
    (1,  0): [MachineInput.Right],
    (-1,  1): [MachineInput.Left, MachineInput.Up],
    (1,   1): [MachineInput.Right, MachineInput.Up],
    (-1, -1): [MachineInput.Left, MachineInput.Down],
    (1,  -1): [MachineInput.Right, MachineInput.Down],
    (0,   0): [],  # centre -- release all
}

# Joystick axis thresholds.
_AXIS_THRESHOLD: float = 0.5

# Joystick button -> MachineInput.
_JOY_BUTTON_MAP: dict[int, MachineInput] = {
    0: MachineInput.Fire,
    1: MachineInput.Fire2,
    2: MachineInput.Fire2,
    3: MachineInput.Reset,
    6: MachineInput.Select,
    7: MachineInput.Reset,
}


class InputHandler:
    """Translates pygame keyboard and joystick events into emulator inputs.

    Parameters
    ----------
    machine:
        The emulated machine.  Expected interface:

        * ``input_state.raise_input(player: int, mi: MachineInput, down: bool)``

        If the machine exposes no ``input_state`` (or no ``raise_input``
        method) inputs are silently dropped.
    """

    def __init__(self, machine: object) -> None:
        self._machine = machine
        self._quit_requested: bool = False

        # Track pressed state for toggle switches so we fire exactly once
        # per key press.
        self._toggle_state: dict[MachineInput, bool] = {
            mi: False for mi in _TOGGLE_SWITCHES
        }

        # Joystick axis state tracking to generate press/release edges.
        self._axis_state: dict[int, dict[str, bool]] = {}
        # axis_id -> {"neg": bool, "pos": bool}

        # Previous hat state for edge detection.
        self._prev_hat_inputs: list[MachineInput] = []

        # Initialise joysticks.
        self._joysticks: list[pygame.joystick.Joystick] = []
        self._init_joysticks()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def quit_requested(self) -> bool:
        """``True`` if the user pressed Escape or closed the window."""
        return self._quit_requested

    def poll(self) -> None:
        """Pump the pygame event queue and process all pending events.

        This should be called once at the top of each frame.
        """
        for event in pygame.event.get():
            self.handle_event(event)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Process a single pygame event."""
        if event.type == pygame.QUIT:
            self._quit_requested = True
            return

        if event.type == pygame.KEYDOWN:
            self._on_key_down(event)
        elif event.type == pygame.KEYUP:
            self._on_key_up(event)
        elif event.type == pygame.JOYAXISMOTION:
            self._on_joy_axis(event)
        elif event.type == pygame.JOYHATMOTION:
            self._on_joy_hat(event)
        elif event.type == pygame.JOYBUTTONDOWN:
            self._on_joy_button(event, down=True)
        elif event.type == pygame.JOYBUTTONUP:
            self._on_joy_button(event, down=False)
        elif event.type == pygame.JOYDEVICEADDED:
            self._init_joysticks()
        elif event.type == pygame.JOYDEVICEREMOVED:
            self._init_joysticks()

    def clear_all(self) -> None:
        """Release all currently-held inputs."""
        for mi in MachineInput:
            self._send(0, mi, False)
        self._axis_state.clear()
        self._prev_hat_inputs.clear()

    # ------------------------------------------------------------------
    # Keyboard handlers
    # ------------------------------------------------------------------

    def _on_key_down(self, event: pygame.event.Event) -> None:
        key = event.key

        # Escape always requests quit.
        if key == pygame.K_ESCAPE:
            self._quit_requested = True
            return

        # Handle '#' which is typically Shift+3.
        if key == pygame.K_3 and (event.mod & pygame.KMOD_SHIFT):
            self._send(0, MachineInput.NumPadHash, True)
            return

        # Handle '*' via Shift+8.
        if key == pygame.K_8 and (event.mod & pygame.KMOD_SHIFT):
            self._send(0, MachineInput.NumPadMult, True)
            return

        mapping = _KEY_MAP.get(key)
        if mapping is None:
            return

        player, mi = mapping

        if mi in _TOGGLE_SWITCHES:
            # Toggle switches: send a press on key-down only (not repeats).
            if not self._toggle_state.get(mi, False):
                self._toggle_state[mi] = True
                self._send(player, mi, True)
            return

        # Regular / momentary inputs: send press.
        self._send(player, mi, True)

    def _on_key_up(self, event: pygame.event.Event) -> None:
        key = event.key

        if key == pygame.K_3 and not (event.mod & pygame.KMOD_SHIFT):
            self._send(0, MachineInput.NumPadHash, False)

        if key == pygame.K_8 and not (event.mod & pygame.KMOD_SHIFT):
            self._send(0, MachineInput.NumPadMult, False)

        mapping = _KEY_MAP.get(key)
        if mapping is None:
            return

        player, mi = mapping

        if mi in _TOGGLE_SWITCHES:
            self._toggle_state[mi] = False
            self._send(player, mi, False)
            return

        self._send(player, mi, False)

    # ------------------------------------------------------------------
    # Joystick handlers
    # ------------------------------------------------------------------

    def _on_joy_axis(self, event: pygame.event.Event) -> None:
        """Handle analogue stick / trigger axis motion."""
        axis = event.axis
        value = event.value

        if axis not in self._axis_state:
            self._axis_state[axis] = {"neg": False, "pos": False}
        state = self._axis_state[axis]

        # Horizontal axis (typically 0)
        if axis == 0:
            neg_now = value < -_AXIS_THRESHOLD
            pos_now = value > _AXIS_THRESHOLD
            if neg_now != state["neg"]:
                self._send(0, MachineInput.Left, neg_now)
                state["neg"] = neg_now
            if pos_now != state["pos"]:
                self._send(0, MachineInput.Right, pos_now)
                state["pos"] = pos_now

        # Vertical axis (typically 1; inverted: negative = up)
        elif axis == 1:
            neg_now = value < -_AXIS_THRESHOLD
            pos_now = value > _AXIS_THRESHOLD
            if neg_now != state["neg"]:
                self._send(0, MachineInput.Up, neg_now)
                state["neg"] = neg_now
            if pos_now != state["pos"]:
                self._send(0, MachineInput.Down, pos_now)
                state["pos"] = pos_now

    def _on_joy_hat(self, event: pygame.event.Event) -> None:
        """Handle D-pad hat switch events."""
        hat_value = event.value  # (x, y) tuple
        new_inputs = _HAT_MAP.get(hat_value, [])

        # Release any directions no longer active.
        for mi in self._prev_hat_inputs:
            if mi not in new_inputs:
                self._send(0, mi, False)

        # Press any newly-active directions.
        for mi in new_inputs:
            if mi not in self._prev_hat_inputs:
                self._send(0, mi, True)

        self._prev_hat_inputs = list(new_inputs)

    def _on_joy_button(self, event: pygame.event.Event, *, down: bool) -> None:
        """Handle joystick button press / release."""
        mi = _JOY_BUTTON_MAP.get(event.button)
        if mi is not None:
            self._send(0, mi, down)

    # ------------------------------------------------------------------
    # Joystick initialisation
    # ------------------------------------------------------------------

    def _init_joysticks(self) -> None:
        """(Re-)detect connected joysticks."""
        self._joysticks.clear()
        count = pygame.joystick.get_count()
        for i in range(count):
            try:
                js = pygame.joystick.Joystick(i)
                js.init()
                self._joysticks.append(js)
                logger.info(
                    "Joystick %d: %s (%d axes, %d buttons, %d hats)",
                    i,
                    js.get_name(),
                    js.get_numaxes(),
                    js.get_numbuttons(),
                    js.get_numhats(),
                )
            except pygame.error as exc:
                logger.warning("Failed to init joystick %d: %s", i, exc)

        if not self._joysticks:
            logger.info("No joysticks detected")

    # ------------------------------------------------------------------
    # Machine bridge
    # ------------------------------------------------------------------

    def _send(self, player: int, mi: MachineInput, down: bool) -> None:
        """Send an input event to the emulated machine.

        If the machine does not expose an ``input_state`` with a
        ``raise_input`` method the call is silently ignored.
        """
        input_state = getattr(self._machine, "input_state", None)
        if input_state is not None:
            raise_fn = getattr(input_state, "raise_input", None)
            if raise_fn is not None:
                raise_fn(player, mi, down)
                return

        # Fallback: try a top-level raise_input on the machine itself.
        raise_fn = getattr(self._machine, "raise_input", None)
        if raise_fn is not None:
            raise_fn(player, mi, down)
