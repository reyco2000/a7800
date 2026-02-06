"""
Logging infrastructure for EMU7800.
Ported from ILogger.cs and Loggers.cs.
"""

from abc import ABC, abstractmethod


class ILogger(ABC):
    """Logging interface with level-based filtering."""

    @property
    @abstractmethod
    def level(self) -> int: ...

    @level.setter
    @abstractmethod
    def level(self, value: int): ...

    @abstractmethod
    def log(self, level: int, message: str): ...


class NullLogger(ILogger):
    """No-op logger implementation."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._level = 0
        return cls._instance

    @property
    def level(self) -> int:
        return self._level

    @level.setter
    def level(self, value: int):
        self._level = value

    def log(self, level: int, message: str):
        pass


class ConsoleLogger(ILogger):
    """Logger that prints to console."""

    def __init__(self, level: int = 0):
        self._level = level

    @property
    def level(self) -> int:
        return self._level

    @level.setter
    def level(self, value: int):
        self._level = value

    def log(self, level: int, message: str):
        if level <= self._level:
            print(f"[EMU7800:{level}] {message}")


# Default logger instance
DEFAULT_LOGGER: ILogger = NullLogger()
