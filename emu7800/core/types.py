"""
Core enumerations and type definitions for EMU7800.
Ported from C# enums: MachineType, Controller, ControllerAction, ConsoleSwitch, CartType, MachineInput.
"""

from enum import IntEnum, IntFlag


class MachineType(IntEnum):
    Unknown = 0
    A2600NTSC = 1
    A2600PAL = 2
    A7800NTSC = 3
    A7800NTSCbios = 4
    A7800NTSChsc = 5
    A7800NTSCxm = 6
    A7800PAL = 7
    A7800PALbios = 8
    A7800PALhsc = 9
    A7800PALxm = 10

    @staticmethod
    def is_2600(mt):
        return mt in (MachineType.A2600NTSC, MachineType.A2600PAL)

    @staticmethod
    def is_7800(mt):
        return mt in (
            MachineType.A7800NTSC, MachineType.A7800NTSCbios, MachineType.A7800NTSChsc, MachineType.A7800NTSCxm,
            MachineType.A7800PAL, MachineType.A7800PALbios, MachineType.A7800PALhsc, MachineType.A7800PALxm,
        )

    @staticmethod
    def is_ntsc(mt):
        return mt in (
            MachineType.A2600NTSC,
            MachineType.A7800NTSC, MachineType.A7800NTSCbios, MachineType.A7800NTSChsc, MachineType.A7800NTSCxm,
        )

    @staticmethod
    def is_pal(mt):
        return mt in (
            MachineType.A2600PAL,
            MachineType.A7800PAL, MachineType.A7800PALbios, MachineType.A7800PALhsc, MachineType.A7800PALxm,
        )

    @staticmethod
    def is_2600_ntsc(mt):
        return mt == MachineType.A2600NTSC

    @staticmethod
    def is_2600_pal(mt):
        return mt == MachineType.A2600PAL

    @staticmethod
    def is_7800_ntsc(mt):
        return mt in (MachineType.A7800NTSC, MachineType.A7800NTSCbios, MachineType.A7800NTSChsc, MachineType.A7800NTSCxm)

    @staticmethod
    def is_7800_pal(mt):
        return mt in (MachineType.A7800PAL, MachineType.A7800PALbios, MachineType.A7800PALhsc, MachineType.A7800PALxm)


class Controller(IntEnum):
    Non = 0
    Joystick = 1
    Paddles = 2
    Keypad = 3
    Driving = 4
    BoosterGrip = 5
    ProLineJoystick = 6
    Lightgun = 7


class ControllerAction(IntEnum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3
    Trigger = 4
    Trigger2 = 5
    Keypad1 = 6
    Keypad2 = 7
    Keypad3 = 8
    Keypad4 = 9
    Keypad5 = 10
    Keypad6 = 11
    Keypad7 = 12
    Keypad8 = 13
    Keypad9 = 14
    KeypadA = 15
    Keypad0 = 16
    KeypadP = 17
    Driving0 = 18
    Driving1 = 19
    Driving2 = 20
    Driving3 = 21


class ConsoleSwitch(IntEnum):
    GameReset = 0
    GameSelect = 1
    GameBW = 2
    LeftDifficultyA = 3
    RightDifficultyA = 4
    Pause = 2  # aliases GameBW on 7800


class MachineInput(IntEnum):
    End = 0
    Pause = 1
    Fire = 2
    Fire2 = 3
    Left = 4
    Right = 5
    Up = 6
    Down = 7
    NumPad7 = 8
    NumPad8 = 9
    NumPad9 = 10
    NumPad4 = 11
    NumPad5 = 12
    NumPad6 = 13
    NumPad1 = 14
    NumPad2 = 15
    NumPad3 = 16
    NumPadMult = 17
    NumPad0 = 18
    NumPadHash = 19
    Driving0 = 20
    Driving1 = 21
    Driving2 = 22
    Driving3 = 23
    Reset = 24
    Select = 25
    Color = 26
    LeftDifficulty = 27
    RightDifficulty = 28


class CartType(IntEnum):
    Unknown = 0
    # Atari 2600 carts
    A2K = 1
    A4K = 2
    A8K = 3
    A8KR = 4
    A16K = 5
    A16KR = 6
    DC8K = 7
    PB8K = 8
    TV8K = 9
    CBS12K = 10
    A32K = 11
    A32KR = 12
    MN16K = 13
    DPC = 14
    M32N12K = 15
    # Atari 7800 carts
    A7808 = 16
    A7816 = 17
    A7832 = 18
    A7832P = 19
    A7832PL = 20
    A7848 = 21
    A78SG = 22
    A78SGP = 23
    A78SGR = 24
    A78S9 = 25
    A78S9PL = 26
    A78S4 = 27
    A78S4R = 28
    A78AB = 29
    A78AC = 30
    A78BB32K = 31
    A78BB32KP = 32
    A78BB32KRPL = 33
    A78BB48K = 34
    A78BB48KP = 35
    A78BB52K = 36
    A78BB52KP = 37
    A78BB128K = 38
    A78BB128KR = 39
    A78BB128KP = 40
    A78BB128KRPL = 41


class TIACxFlags(IntFlag):
    PF = 1 << 0
    BL = 1 << 1
    M0 = 1 << 2
    M1 = 1 << 3
    P0 = 1 << 4
    P1 = 1 << 5


class TIACxPairFlags(IntFlag):
    M0P1 = 1 << 0
    M0P0 = 1 << 1
    M1P0 = 1 << 2
    M1P1 = 1 << 3
    P0PF = 1 << 4
    P0BL = 1 << 5
    P1PF = 1 << 6
    P1BL = 1 << 7
    M0PF = 1 << 8
    M0BL = 1 << 9
    M1PF = 1 << 10
    M1BL = 1 << 11
    BLPF = 1 << 12
    P0P1 = 1 << 13
    M0M1 = 1 << 14
