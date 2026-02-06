"""
Main application window for EMU7800.
Uses pygame to create a display, drive the emulation main loop, and
coordinate audio, video, and input subsystems.

Typical usage::

    from emu7800.platform.window import Window

    machine = MachineFactory.create("game.a78")
    window = Window(machine, scale=3)
    window.run()
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pygame

from emu7800.core.types import MachineType
from emu7800.platform.audio import AudioDevice
from emu7800.platform.input_handler import InputHandler
from emu7800.shell.frame_renderer import FrameRenderer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_TITLE: str = "EMU7800 - Python"

# Minimum / maximum allowed display scale factors.
_MIN_SCALE: int = 1
_MAX_SCALE: int = 8

# Target frame rates (used as a fallback if the machine does not provide
# frame_hz).
_NTSC_HZ: int = 60
_PAL_HZ: int = 50


class Window:
    """Pygame window that owns the emulation main loop.

    Parameters
    ----------
    machine:
        A fully-wired machine instance.  Expected attributes:

        * ``frame_buffer`` -- :class:`~emu7800.core.frame_buffer.FrameBuffer`
        * ``first_scanline`` -- ``int``
        * ``frame_hz`` -- ``int`` (default 60)
        * ``machine_type`` -- :class:`MachineType`
        * ``compute_next_frame()`` -- advance the emulation by one frame
        * ``input_state`` / ``raise_input`` -- for controller input

    scale:
        Integer scale factor applied to the native resolution.
    enable_audio:
        Set to ``False`` to mute sound output entirely.
    vsync:
        When ``True``, request vertical sync from the display driver.
        Has no effect if the driver does not support it.
    """

    def __init__(
        self,
        machine: object,
        scale: int = 3,
        *,
        enable_audio: bool = True,
        vsync: bool = False,
    ) -> None:
        # ---- basic state -------------------------------------------------
        self._machine = machine
        self._scale: int = max(_MIN_SCALE, min(_MAX_SCALE, scale))
        self._running: bool = False
        self._paused: bool = False

        # ---- extract machine geometry ------------------------------------
        fb = machine.frame_buffer  # type: ignore[attr-defined]
        self._native_width: int = fb.visible_pitch
        first_scanline: int = getattr(machine, "first_scanline", 0)
        self._native_height: int = fb.scanlines - first_scanline
        self._frame_hz: int = getattr(machine, "frame_hz", _NTSC_HZ)

        # ---- init pygame display -----------------------------------------
        # Ensure pygame core is initialised (idempotent).
        if not pygame.get_init():
            pygame.init()
        # Ensure joystick subsystem is initialised for input handling.
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        self._display_width: int = self._native_width * self._scale
        self._display_height: int = self._native_height * self._scale

        flags = pygame.RESIZABLE
        if vsync:
            flags |= pygame.SCALED

        self._screen: pygame.Surface = pygame.display.set_mode(
            (self._display_width, self._display_height),
            flags,
        )

        title = self._build_title(machine)
        pygame.display.set_caption(title)

        self._clock: pygame.time.Clock = pygame.time.Clock()

        # ---- subsystems --------------------------------------------------
        self._frame_renderer: FrameRenderer = FrameRenderer(machine)
        self._audio: AudioDevice = AudioDevice(machine, enabled=enable_audio)
        self._input: InputHandler = InputHandler(machine)

        # ---- performance counters ----------------------------------------
        self._frame_count: int = 0
        self._fps_update_time: float = 0.0
        self._fps_display: float = 0.0

        logger.info(
            "Window: %dx%d native, %dx%d display (scale=%d, %d Hz)",
            self._native_width,
            self._native_height,
            self._display_width,
            self._display_height,
            self._scale,
            self._frame_hz,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = value

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def fps(self) -> float:
        """The measured frames-per-second (updated once per second)."""
        return self._fps_display

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Enter the main emulation loop.

        This method blocks until the user closes the window or presses
        Escape.  It:

        1. Polls input events and forwards them to the machine.
        2. Calls ``machine.compute_next_frame()`` to advance emulation.
        3. Submits audio samples to the audio device.
        4. Renders the frame buffer to the display.
        5. Throttles to the target frame rate.
        """
        self._running = True
        self._fps_update_time = time.monotonic()
        self._frame_count = 0

        logger.info("Entering main loop (target %d fps)", self._frame_hz)

        try:
            while self._running:
                self._tick()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Per-frame tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Execute one iteration of the main loop."""
        # ---- input -------------------------------------------------------
        self._input.poll()
        if self._input.quit_requested:
            self._running = False
            return

        # ---- handle window events that InputHandler does not consume -----
        # (InputHandler.poll already called pygame.event.get, so any
        #  remaining events were handled there.  We only check the quit
        #  flag it set.)

        # ---- emulation ---------------------------------------------------
        if not self._paused:
            compute = getattr(self._machine, "compute_next_frame", None)
            if compute is not None:
                compute()

            # ---- audio ---------------------------------------------------
            self._audio.submit_frame()

        # ---- video -------------------------------------------------------
        surface = self._frame_renderer.render()

        # Scale to display size.  If the window has been resized, adjust to
        # the new dimensions.
        current_size = self._screen.get_size()
        if surface.get_size() != current_size:
            scaled = pygame.transform.scale(surface, current_size)
        else:
            scaled = surface
        self._screen.blit(scaled, (0, 0))
        pygame.display.flip()

        # ---- timing ------------------------------------------------------
        self._clock.tick(self._frame_hz)
        self._update_fps()

    # ------------------------------------------------------------------
    # FPS tracking
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        """Update the displayed FPS counter roughly once per second."""
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_update_time
        if elapsed >= 1.0:
            self._fps_display = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_update_time = now

            # Optionally update the window title with FPS.
            title = self._build_title(self._machine)
            pygame.display.set_caption(
                f"{title}  [{self._fps_display:.1f} fps]"
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Clean up all subsystems."""
        logger.info("Shutting down")
        self._audio.shutdown()
        pygame.quit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_title(machine: object) -> str:
        """Build the window title string from machine metadata."""
        mt = getattr(machine, "machine_type", None)
        if mt is not None:
            return f"{_WINDOW_TITLE}  ({mt.name})"
        return _WINDOW_TITLE
