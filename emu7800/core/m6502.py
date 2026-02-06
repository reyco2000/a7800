"""
MOS 6502 CPU emulator for EMU7800.
Ported from the C# M6502 class.

Implements the full NMOS 6502 instruction set including decimal mode with
NMOS quirks, all legal opcodes, and a selection of common illegal opcodes.

Key NMOS-specific behaviours:

* Decimal-mode ADC sets N/V from the intermediate BCD result and Z from the
  binary result.  Decimal-mode SBC derives all flags from the binary result.
* JMP ($xxFF) wraps within the page (the famous indirect-jump bug).
* BRK pushes PC+2 and P with the B flag set, then vectors through $FFFE.
* JSR pushes the address of its *last* operand byte (PC-1), and RTS
  compensates by pulling and adding one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    pass


class M6502:
    """NMOS 6502 CPU emulator.

    Parameters
    ----------
    machine:
        Back-reference to the owning machine object.  Memory is accessed
        via ``machine.mem[addr]`` (an :class:`AddressSpace` instance).
    run_clocks_multiple:
        Multiplier applied to cycle counts when decrementing
        :attr:`run_clocks`.  Allows the caller to express *run_clocks* in
        a unit other than CPU cycles (e.g. colour clocks on the Atari).
    """

    # ------------------------------------------------------------------
    # Interrupt vectors
    # ------------------------------------------------------------------
    NMI_VEC: int = 0xFFFA
    RST_VEC: int = 0xFFFC
    IRQ_VEC: int = 0xFFFE

    # ------------------------------------------------------------------
    # Flag bit masks
    # ------------------------------------------------------------------
    FLAG_C: int = 0x01  # Carry
    FLAG_Z: int = 0x02  # Zero
    FLAG_I: int = 0x04  # Interrupt disable
    FLAG_D: int = 0x08  # Decimal
    FLAG_B: int = 0x10  # Break
    FLAG_V: int = 0x40  # Overflow
    FLAG_N: int = 0x80  # Negative

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, machine: object, run_clocks_multiple: int) -> None:
        self.m = machine
        self.run_clocks_multiple: int = run_clocks_multiple

        # Registers
        self.pc: int = 0x0000   # 16-bit program counter
        self.a: int = 0x00      # 8-bit accumulator
        self.x: int = 0x00      # 8-bit index register X
        self.y: int = 0x00      # 8-bit index register Y
        self.s: int = 0xFF      # 8-bit stack pointer
        self.p: int = 0x20      # 8-bit processor status (bit 5 always set)

        # Timing
        self.clock: int = 0
        self.run_clocks: int = 0

        # Control flags
        self.emulator_preempt_request: bool = False
        self.jammed: bool = False
        self.irq_interrupt_request: bool = False
        self.nmi_interrupt_request: bool = False

        # Build the opcode dispatch table (list of 256 callables)
        self._opcode_table: List[Callable[[], None]] = self._build_opcode_table()

    # ------------------------------------------------------------------
    # Flag properties
    # ------------------------------------------------------------------

    @property
    def fC(self) -> bool:
        return bool(self.p & self.FLAG_C)

    @fC.setter
    def fC(self, value: bool) -> None:
        if value:
            self.p |= self.FLAG_C
        else:
            self.p &= ~self.FLAG_C & 0xFF

    @property
    def fZ(self) -> bool:
        return bool(self.p & self.FLAG_Z)

    @fZ.setter
    def fZ(self, value: bool) -> None:
        if value:
            self.p |= self.FLAG_Z
        else:
            self.p &= ~self.FLAG_Z & 0xFF

    @property
    def fI(self) -> bool:
        return bool(self.p & self.FLAG_I)

    @fI.setter
    def fI(self, value: bool) -> None:
        if value:
            self.p |= self.FLAG_I
        else:
            self.p &= ~self.FLAG_I & 0xFF

    @property
    def fD(self) -> bool:
        return bool(self.p & self.FLAG_D)

    @fD.setter
    def fD(self, value: bool) -> None:
        if value:
            self.p |= self.FLAG_D
        else:
            self.p &= ~self.FLAG_D & 0xFF

    @property
    def fB(self) -> bool:
        return bool(self.p & self.FLAG_B)

    @fB.setter
    def fB(self, value: bool) -> None:
        if value:
            self.p |= self.FLAG_B
        else:
            self.p &= ~self.FLAG_B & 0xFF

    @property
    def fV(self) -> bool:
        return bool(self.p & self.FLAG_V)

    @fV.setter
    def fV(self, value: bool) -> None:
        if value:
            self.p |= self.FLAG_V
        else:
            self.p &= ~self.FLAG_V & 0xFF

    @property
    def fN(self) -> bool:
        return bool(self.p & self.FLAG_N)

    @fN.setter
    def fN(self, value: bool) -> None:
        if value:
            self.p |= self.FLAG_N
        else:
            self.p &= ~self.FLAG_N & 0xFF

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_fnz(self, val: int) -> None:
        """Set the N and Z flags from an 8-bit value."""
        self.fN = bool(val & 0x80)
        self.fZ = (val & 0xFF) == 0

    def clk(self, ticks: int) -> None:
        """Advance the clock and consume run clocks."""
        self.clock += ticks
        self.run_clocks -= ticks * self.run_clocks_multiple

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------

    def push(self, data: int) -> None:
        """Push a byte onto the stack."""
        self.m.mem[0x100 + self.s] = data & 0xFF
        self.s = (self.s - 1) & 0xFF

    def pull(self) -> int:
        """Pull a byte from the stack."""
        self.s = (self.s + 1) & 0xFF
        return self.m.mem[0x100 + self.s]

    # ------------------------------------------------------------------
    # Addressing modes  (all return a 16-bit effective address)
    # ------------------------------------------------------------------

    def a_rel(self) -> int:
        """Relative addressing -- returns the branch target address."""
        bo = self.m.mem[self.pc]
        self.pc = (self.pc + 1) & 0xFFFF
        if bo & 0x80:
            bo -= 256  # sign-extend
        return (self.pc + bo) & 0xFFFF

    def a_zpg(self) -> int:
        """Zero-page addressing."""
        ea = self.m.mem[self.pc]
        self.pc = (self.pc + 1) & 0xFFFF
        return ea

    def a_zpx(self) -> int:
        """Zero-page, X-indexed addressing."""
        ea = (self.m.mem[self.pc] + self.x) & 0xFF
        self.pc = (self.pc + 1) & 0xFFFF
        return ea

    def a_zpy(self) -> int:
        """Zero-page, Y-indexed addressing."""
        ea = (self.m.mem[self.pc] + self.y) & 0xFF
        self.pc = (self.pc + 1) & 0xFFFF
        return ea

    def a_abs(self) -> int:
        """Absolute addressing."""
        lsb = self.m.mem[self.pc]
        self.pc = (self.pc + 1) & 0xFFFF
        msb = self.m.mem[self.pc]
        self.pc = (self.pc + 1) & 0xFFFF
        return lsb | (msb << 8)

    def a_abx(self, eclk: int) -> int:
        """Absolute, X-indexed addressing with optional page-crossing penalty."""
        ea = self.a_abs()
        if (ea & 0xFF) + self.x > 0xFF:
            self.clk(eclk)
        return (ea + self.x) & 0xFFFF

    def a_aby(self, eclk: int) -> int:
        """Absolute, Y-indexed addressing with optional page-crossing penalty."""
        ea = self.a_abs()
        if (ea & 0xFF) + self.y > 0xFF:
            self.clk(eclk)
        return (ea + self.y) & 0xFFFF

    def a_idx(self) -> int:
        """Indexed indirect (pre-indexed) addressing -- (zp,X)."""
        zpa = (self.m.mem[self.pc] + self.x) & 0xFF
        self.pc = (self.pc + 1) & 0xFFFF
        lsb = self.m.mem[zpa]
        msb = self.m.mem[(zpa + 1) & 0xFF]
        return lsb | (msb << 8)

    def a_idy(self, eclk: int) -> int:
        """Indirect indexed (post-indexed) addressing -- (zp),Y."""
        zpa = self.m.mem[self.pc]
        self.pc = (self.pc + 1) & 0xFFFF
        lsb = self.m.mem[zpa]
        msb = self.m.mem[(zpa + 1) & 0xFF]
        if lsb + self.y > 0xFF:
            self.clk(eclk)
        return ((lsb | (msb << 8)) + self.y) & 0xFFFF

    def a_ind(self) -> int:
        """Indirect addressing -- used only by JMP (ind).

        Reproduces the NMOS page-boundary wrap bug: if the low byte of
        the pointer is 0xFF the high byte is fetched from xx00 instead
        of (xx+1)00.
        """
        ea = self.a_abs()
        lsb = self.m.mem[ea]
        ea2 = (((ea & 0xFF) + 1) & 0xFF) | (ea & 0xFF00)
        msb = self.m.mem[ea2]
        return lsb | (msb << 8)

    # ------------------------------------------------------------------
    # Instruction implementations -- Arithmetic
    # ------------------------------------------------------------------

    def i_adc(self, val: int) -> None:
        """Add with carry.  Handles NMOS decimal-mode quirks."""
        c = 1 if self.fC else 0
        if self.fD:
            # BCD addition (NMOS 6502 behaviour)
            al = (self.a & 0x0F) + (val & 0x0F) + c
            if al >= 0x0A:
                al = ((al + 0x06) & 0x0F) + 0x10
            s = (self.a & 0xF0) + (val & 0xF0) + al
            # N and V from intermediate BCD result
            self.fN = bool(s & 0x80)
            self.fV = bool(~(self.a ^ val) & (self.a ^ s) & 0x80)
            if s >= 0xA0:
                s += 0x60
            self.fC = s >= 0x100
            # Z from binary result (NMOS quirk)
            self.fZ = ((self.a + val + c) & 0xFF) == 0
            self.a = s & 0xFF
        else:
            # Binary addition
            s = self.a + val + c
            self.fV = bool(~(self.a ^ val) & (self.a ^ s) & 0x80)
            self.fC = s > 0xFF
            self.a = s & 0xFF
            self.set_fnz(self.a)

    def i_sbc(self, val: int) -> None:
        """Subtract with carry (borrow).  Handles NMOS decimal-mode quirks."""
        c = 1 if self.fC else 0
        borrow = 1 - c
        diff = self.a - val - borrow
        if self.fD:
            # BCD subtraction (NMOS 6502 behaviour)
            # All flags come from the binary result on NMOS.
            self.set_fnz(diff & 0xFF)
            self.fV = bool(((self.a ^ val) & (self.a ^ diff)) & 0x80)
            self.fC = diff >= 0
            # BCD correction for the accumulator
            al = (self.a & 0x0F) - (val & 0x0F) - borrow
            if al < 0:
                al = ((al - 0x06) & 0x0F) - 0x10
            s = (self.a & 0xF0) - (val & 0xF0) + al
            if s < 0:
                s -= 0x60
            self.a = s & 0xFF
        else:
            # Binary subtraction
            self.fV = bool(((self.a ^ val) & (self.a ^ diff)) & 0x80)
            self.fC = diff >= 0
            self.a = diff & 0xFF
            self.set_fnz(self.a)

    # ------------------------------------------------------------------
    # Instruction implementations -- Logic
    # ------------------------------------------------------------------

    def i_and(self, val: int) -> None:
        self.a &= val
        self.set_fnz(self.a)

    def i_ora(self, val: int) -> None:
        self.a |= val
        self.set_fnz(self.a)

    def i_eor(self, val: int) -> None:
        self.a ^= val
        self.set_fnz(self.a)

    # ------------------------------------------------------------------
    # Instruction implementations -- Shifts / Rotates
    # (operate on a byte, return the result)
    # ------------------------------------------------------------------

    def i_asl(self, val: int) -> int:
        self.fC = bool(val & 0x80)
        result = (val << 1) & 0xFF
        self.set_fnz(result)
        return result

    def i_lsr(self, val: int) -> int:
        self.fC = bool(val & 0x01)
        result = (val >> 1) & 0x7F
        self.set_fnz(result)
        return result

    def i_rol(self, val: int) -> int:
        c = 1 if self.fC else 0
        self.fC = bool(val & 0x80)
        result = ((val << 1) | c) & 0xFF
        self.set_fnz(result)
        return result

    def i_ror(self, val: int) -> int:
        c = 0x80 if self.fC else 0
        self.fC = bool(val & 0x01)
        result = ((val >> 1) | c) & 0xFF
        self.set_fnz(result)
        return result

    # ------------------------------------------------------------------
    # Instruction implementations -- Test / Compare
    # ------------------------------------------------------------------

    def i_bit(self, val: int) -> None:
        self.fN = bool(val & 0x80)
        self.fV = bool(val & 0x40)
        self.fZ = (self.a & val) == 0

    def i_cmp(self, val: int) -> None:
        result = self.a - val
        self.fC = result >= 0
        self.set_fnz(result & 0xFF)

    def i_cpx(self, val: int) -> None:
        result = self.x - val
        self.fC = result >= 0
        self.set_fnz(result & 0xFF)

    def i_cpy(self, val: int) -> None:
        result = self.y - val
        self.fC = result >= 0
        self.set_fnz(result & 0xFF)

    # ------------------------------------------------------------------
    # Instruction implementations -- Increment / Decrement
    # (operate on a byte, return result)
    # ------------------------------------------------------------------

    def i_inc(self, val: int) -> int:
        result = (val + 1) & 0xFF
        self.set_fnz(result)
        return result

    def i_dec(self, val: int) -> int:
        result = (val - 1) & 0xFF
        self.set_fnz(result)
        return result

    def i_inx(self) -> None:
        self.x = (self.x + 1) & 0xFF
        self.set_fnz(self.x)

    def i_iny(self) -> None:
        self.y = (self.y + 1) & 0xFF
        self.set_fnz(self.y)

    def i_dex(self) -> None:
        self.x = (self.x - 1) & 0xFF
        self.set_fnz(self.x)

    def i_dey(self) -> None:
        self.y = (self.y - 1) & 0xFF
        self.set_fnz(self.y)

    # ------------------------------------------------------------------
    # Instruction implementations -- Load / Store
    # ------------------------------------------------------------------

    def i_lda(self, val: int) -> None:
        self.a = val & 0xFF
        self.set_fnz(self.a)

    def i_ldx(self, val: int) -> None:
        self.x = val & 0xFF
        self.set_fnz(self.x)

    def i_ldy(self, val: int) -> None:
        self.y = val & 0xFF
        self.set_fnz(self.y)

    def i_sta(self, ea: int) -> None:
        self.m.mem[ea] = self.a

    def i_stx(self, ea: int) -> None:
        self.m.mem[ea] = self.x

    def i_sty(self, ea: int) -> None:
        self.m.mem[ea] = self.y

    # ------------------------------------------------------------------
    # Instruction implementations -- Transfer
    # ------------------------------------------------------------------

    def i_tax(self) -> None:
        self.x = self.a
        self.set_fnz(self.x)

    def i_tay(self) -> None:
        self.y = self.a
        self.set_fnz(self.y)

    def i_tsx(self) -> None:
        self.x = self.s
        self.set_fnz(self.x)

    def i_txa(self) -> None:
        self.a = self.x
        self.set_fnz(self.a)

    def i_txs(self) -> None:
        self.s = self.x  # No flags affected

    def i_tya(self) -> None:
        self.a = self.y
        self.set_fnz(self.a)

    # ------------------------------------------------------------------
    # Instruction implementations -- Stack
    # ------------------------------------------------------------------

    def i_pha(self) -> None:
        self.push(self.a)

    def i_php(self) -> None:
        self.push(self.p | 0x30)  # B and bit-5 always set when pushed

    def i_pla(self) -> None:
        self.a = self.pull()
        self.set_fnz(self.a)

    def i_plp(self) -> None:
        self.p = (self.pull() | 0x20) & 0xFF  # bit-5 always set

    # ------------------------------------------------------------------
    # Instruction implementations -- Flag set / clear
    # ------------------------------------------------------------------

    def i_clc(self) -> None:
        self.fC = False

    def i_sec(self) -> None:
        self.fC = True

    def i_cld(self) -> None:
        self.fD = False

    def i_sed(self) -> None:
        self.fD = True

    def i_cli(self) -> None:
        self.fI = False

    def i_sei(self) -> None:
        self.fI = True

    def i_clv(self) -> None:
        self.fV = False

    # ------------------------------------------------------------------
    # Branch helper
    # ------------------------------------------------------------------

    def br(self, cond: bool, ea: int) -> None:
        """Conditional branch.  Adds 1 cycle if taken (same page) or
        2 cycles if taken across a page boundary."""
        if cond:
            if (self.pc & 0xFF00) != (ea & 0xFF00):
                self.clk(2)
            else:
                self.clk(1)
            self.pc = ea

    # ------------------------------------------------------------------
    # Instruction implementations -- Jump / Subroutine / Return
    # ------------------------------------------------------------------

    def i_brk(self) -> None:
        """BRK -- software interrupt."""
        self.pc = (self.pc + 1) & 0xFFFF  # skip padding byte
        self.push((self.pc >> 8) & 0xFF)
        self.push(self.pc & 0xFF)
        self.push(self.p | 0x30)  # B and bit-5 set when pushed by BRK
        self.fI = True
        self.pc = (
            self.m.mem[self.IRQ_VEC]
            | (self.m.mem[self.IRQ_VEC + 1] << 8)
        )

    def i_jmp(self, ea: int) -> None:
        self.pc = ea

    def i_jsr(self, ea: int) -> None:
        ret = (self.pc - 1) & 0xFFFF
        self.push((ret >> 8) & 0xFF)
        self.push(ret & 0xFF)
        self.pc = ea

    def i_rts(self) -> None:
        lo = self.pull()
        hi = self.pull()
        self.pc = ((lo | (hi << 8)) + 1) & 0xFFFF

    def i_rti(self) -> None:
        self.p = self.pull()
        lo = self.pull()
        hi = self.pull()
        self.pc = lo | (hi << 8)
        self.fB = True

    # ------------------------------------------------------------------
    # Illegal opcode implementations
    # ------------------------------------------------------------------

    def i_kil(self) -> None:
        """KIL -- jam the processor."""
        self.jammed = True

    def i_lax(self, val: int) -> None:
        """LAX -- load both A and X from memory."""
        self.a = val & 0xFF
        self.x = val & 0xFF
        self.set_fnz(self.a)

    def i_isb(self, ea: int) -> None:
        """ISB (ISC) -- increment memory then SBC."""
        val = (self.m.mem[ea] + 1) & 0xFF
        self.m.mem[ea] = val
        self.i_sbc(val)

    def i_rla(self, ea: int) -> None:
        """RLA -- rotate left memory then AND with accumulator."""
        val = self.m.mem[ea]
        result = self.i_rol(val)
        self.m.mem[ea] = result
        self.a &= result
        self.set_fnz(self.a)

    def i_sax(self, ea: int) -> None:
        """SAX -- store A AND X to memory."""
        self.m.mem[ea] = self.a & self.x

    def i_alr(self, val: int) -> None:
        """ALR -- AND then LSR accumulator."""
        self.a &= val
        self.fC = bool(self.a & 0x01)
        self.a = (self.a >> 1) & 0x7F
        self.set_fnz(self.a)

    def i_anc(self, val: int) -> None:
        """ANC -- AND then copy bit 7 to carry."""
        self.a &= val
        self.set_fnz(self.a)
        self.fC = bool(self.a & 0x80)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the CPU to its power-on state."""
        self.s = 0xFF
        self.fI = True
        self.fZ = True
        self.pc = (
            self.m.mem[self.RST_VEC]
            | (self.m.mem[self.RST_VEC + 1] << 8)
        )
        self.clk(6)

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self) -> None:
        """Run instructions until *run_clocks* is exhausted, the emulator
        requests a preempt, or the CPU jams."""
        while (
            self.run_clocks > 0
            and not self.emulator_preempt_request
            and not self.jammed
        ):
            # -- NMI (edge-triggered, highest priority) --
            if self.nmi_interrupt_request:
                self.nmi_interrupt_request = False
                self.push((self.pc >> 8) & 0xFF)
                self.push(self.pc & 0xFF)
                self.fB = False
                self.push(self.p | 0x20)  # bit-5 always set
                self.fI = True
                self.pc = (
                    self.m.mem[self.NMI_VEC]
                    | (self.m.mem[self.NMI_VEC + 1] << 8)
                )
                self.clk(7)
                continue

            # -- IRQ (level-triggered, masked by I flag) --
            if self.irq_interrupt_request and not self.fI:
                self.irq_interrupt_request = False
                self.push((self.pc >> 8) & 0xFF)
                self.push(self.pc & 0xFF)
                self.fB = False
                self.push(self.p | 0x20)  # bit-5 always set
                self.fI = True
                self.pc = (
                    self.m.mem[self.IRQ_VEC]
                    | (self.m.mem[self.IRQ_VEC + 1] << 8)
                )
                self.clk(7)
                continue

            # -- Fetch and dispatch opcode --
            opcode = self.m.mem[self.pc]
            self.pc = (self.pc + 1) & 0xFFFF
            self._opcode_table[opcode]()

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return a serialisable snapshot of CPU state."""
        return {
            "pc": self.pc, "a": self.a, "x": self.x, "y": self.y,
            "s": self.s, "p": self.p, "clock": self.clock,
            "run_clocks": self.run_clocks,
            "run_clocks_multiple": self.run_clocks_multiple,
            "jammed": self.jammed,
            "irq_interrupt_request": self.irq_interrupt_request,
            "nmi_interrupt_request": self.nmi_interrupt_request,
            "emulator_preempt_request": self.emulator_preempt_request,
        }

    def restore_snapshot(self, snap: dict) -> None:
        """Restore CPU state from a previous snapshot."""
        self.pc = snap["pc"]
        self.a = snap["a"]
        self.x = snap["x"]
        self.y = snap["y"]
        self.s = snap["s"]
        self.p = snap["p"]
        self.clock = snap["clock"]
        self.run_clocks = snap["run_clocks"]
        self.run_clocks_multiple = snap["run_clocks_multiple"]
        self.jammed = snap["jammed"]
        self.irq_interrupt_request = snap["irq_interrupt_request"]
        self.nmi_interrupt_request = snap["nmi_interrupt_request"]
        self.emulator_preempt_request = snap["emulator_preempt_request"]

    # ==================================================================
    # Opcode dispatch table -- 256 entries
    # ==================================================================

    def _build_opcode_table(self) -> List[Callable[[], None]]:
        """Construct the 256-entry opcode dispatch table.

        Each entry is a zero-argument callable that executes one instruction.
        The opcode byte has already been fetched (and PC advanced) before the
        callable is invoked.
        """
        mem = self.m.mem

        # -- Default handler for undefined opcodes -----------------------
        def _make_undefined(op: int):
            def _handler():
                if (
                    hasattr(self.m, "nop_register_dumping")
                    and self.m.nop_register_dumping
                ):
                    msg = (
                        f"Unknown opcode ${op:02X} at "
                        f"${(self.pc - 1) & 0xFFFF:04X}  "
                        f"A={self.a:02X} X={self.x:02X} "
                        f"Y={self.y:02X} S={self.s:02X} P={self.p:02X}"
                    )
                    if hasattr(self.m, "logger"):
                        self.m.logger.log(1, msg)
                self.clk(2)
            return _handler

        t: List[Callable[[], None]] = [_make_undefined(i) for i in range(256)]

        # ================================================================
        # 0x00 -- BRK
        # ================================================================
        def op_00():
            self.i_brk(); self.clk(7)
        t[0x00] = op_00

        # ================================================================
        # 0x01 -- ORA (idx)
        # ================================================================
        def op_01():
            self.i_ora(mem[self.a_idx()]); self.clk(6)
        t[0x01] = op_01

        # ================================================================
        # 0x02 -- *KIL
        # ================================================================
        def op_02():
            self.i_kil(); self.clk(2)
        t[0x02] = op_02

        # ================================================================
        # 0x04 -- *DOP zpg (double-byte NOP)
        # ================================================================
        def op_04():
            self.a_zpg(); self.clk(3)
        t[0x04] = op_04

        # ================================================================
        # 0x05 -- ORA zpg
        # ================================================================
        def op_05():
            self.i_ora(mem[self.a_zpg()]); self.clk(3)
        t[0x05] = op_05

        # ================================================================
        # 0x06 -- ASL zpg
        # ================================================================
        def op_06():
            ea = self.a_zpg(); mem[ea] = self.i_asl(mem[ea]); self.clk(5)
        t[0x06] = op_06

        # ================================================================
        # 0x08 -- PHP
        # ================================================================
        def op_08():
            self.i_php(); self.clk(3)
        t[0x08] = op_08

        # ================================================================
        # 0x09 -- ORA #imm
        # ================================================================
        def op_09():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_ora(v); self.clk(2)
        t[0x09] = op_09

        # ================================================================
        # 0x0A -- ASL A
        # ================================================================
        def op_0a():
            self.a = self.i_asl(self.a); self.clk(2)
        t[0x0A] = op_0a

        # ================================================================
        # 0x0B -- *ANC #imm
        # ================================================================
        def op_0b():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_anc(v); self.clk(2)
        t[0x0B] = op_0b

        # ================================================================
        # 0x0C -- *TOP abs (triple-byte NOP)
        # ================================================================
        def op_0c():
            self.a_abs(); self.clk(2)
        t[0x0C] = op_0c

        # ================================================================
        # 0x0D -- ORA abs
        # ================================================================
        def op_0d():
            self.i_ora(mem[self.a_abs()]); self.clk(4)
        t[0x0D] = op_0d

        # ================================================================
        # 0x0E -- ASL abs
        # ================================================================
        def op_0e():
            ea = self.a_abs(); mem[ea] = self.i_asl(mem[ea]); self.clk(6)
        t[0x0E] = op_0e

        # ================================================================
        # 0x10 -- BPL rel
        # ================================================================
        def op_10():
            ea = self.a_rel(); self.br(not self.fN, ea); self.clk(2)
        t[0x10] = op_10

        # ================================================================
        # 0x11 -- ORA (idy)
        # ================================================================
        def op_11():
            self.i_ora(mem[self.a_idy(1)]); self.clk(5)
        t[0x11] = op_11

        # ================================================================
        # 0x12 -- *KIL
        # ================================================================
        def op_12():
            self.i_kil(); self.clk(2)
        t[0x12] = op_12

        # ================================================================
        # 0x15 -- ORA zpx
        # ================================================================
        def op_15():
            self.i_ora(mem[self.a_zpx()]); self.clk(4)
        t[0x15] = op_15

        # ================================================================
        # 0x16 -- ASL zpx
        # ================================================================
        def op_16():
            ea = self.a_zpx(); mem[ea] = self.i_asl(mem[ea]); self.clk(6)
        t[0x16] = op_16

        # ================================================================
        # 0x18 -- CLC
        # ================================================================
        def op_18():
            self.i_clc(); self.clk(2)
        t[0x18] = op_18

        # ================================================================
        # 0x19 -- ORA aby
        # ================================================================
        def op_19():
            self.i_ora(mem[self.a_aby(1)]); self.clk(4)
        t[0x19] = op_19

        # ================================================================
        # 0x1C -- *TOP abx (triple-byte NOP)
        # ================================================================
        def op_1c():
            self.a_abx(0); self.clk(2)
        t[0x1C] = op_1c

        # ================================================================
        # 0x1D -- ORA abx
        # ================================================================
        def op_1d():
            self.i_ora(mem[self.a_abx(1)]); self.clk(4)
        t[0x1D] = op_1d

        # ================================================================
        # 0x1E -- ASL abx
        # ================================================================
        def op_1e():
            ea = self.a_abx(0); mem[ea] = self.i_asl(mem[ea]); self.clk(7)
        t[0x1E] = op_1e

        # ================================================================
        # 0x20 -- JSR abs
        # ================================================================
        def op_20():
            self.i_jsr(self.a_abs()); self.clk(6)
        t[0x20] = op_20

        # ================================================================
        # 0x21 -- AND (idx)
        # ================================================================
        def op_21():
            self.i_and(mem[self.a_idx()]); self.clk(6)
        t[0x21] = op_21

        # ================================================================
        # 0x22 -- *KIL
        # ================================================================
        def op_22():
            self.i_kil(); self.clk(2)
        t[0x22] = op_22

        # ================================================================
        # 0x24 -- BIT zpg
        # ================================================================
        def op_24():
            self.i_bit(mem[self.a_zpg()]); self.clk(3)
        t[0x24] = op_24

        # ================================================================
        # 0x25 -- AND zpg
        # ================================================================
        def op_25():
            self.i_and(mem[self.a_zpg()]); self.clk(3)
        t[0x25] = op_25

        # ================================================================
        # 0x26 -- ROL zpg
        # ================================================================
        def op_26():
            ea = self.a_zpg(); mem[ea] = self.i_rol(mem[ea]); self.clk(5)
        t[0x26] = op_26

        # ================================================================
        # 0x28 -- PLP
        # ================================================================
        def op_28():
            self.i_plp(); self.clk(4)
        t[0x28] = op_28

        # ================================================================
        # 0x29 -- AND #imm
        # ================================================================
        def op_29():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_and(v); self.clk(2)
        t[0x29] = op_29

        # ================================================================
        # 0x2A -- ROL A
        # ================================================================
        def op_2a():
            self.a = self.i_rol(self.a); self.clk(2)
        t[0x2A] = op_2a

        # ================================================================
        # 0x2B -- *ANC #imm (second encoding)
        # ================================================================
        def op_2b():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_anc(v); self.clk(2)
        t[0x2B] = op_2b

        # ================================================================
        # 0x2C -- BIT abs
        # ================================================================
        def op_2c():
            self.i_bit(mem[self.a_abs()]); self.clk(4)
        t[0x2C] = op_2c

        # ================================================================
        # 0x2D -- AND abs
        # ================================================================
        def op_2d():
            self.i_and(mem[self.a_abs()]); self.clk(4)
        t[0x2D] = op_2d

        # ================================================================
        # 0x2E -- ROL abs
        # ================================================================
        def op_2e():
            ea = self.a_abs(); mem[ea] = self.i_rol(mem[ea]); self.clk(6)
        t[0x2E] = op_2e

        # ================================================================
        # 0x30 -- BMI rel
        # ================================================================
        def op_30():
            ea = self.a_rel(); self.br(self.fN, ea); self.clk(2)
        t[0x30] = op_30

        # ================================================================
        # 0x31 -- AND (idy)
        # ================================================================
        def op_31():
            self.i_and(mem[self.a_idy(1)]); self.clk(5)
        t[0x31] = op_31

        # ================================================================
        # 0x32 -- *KIL
        # ================================================================
        def op_32():
            self.i_kil(); self.clk(2)
        t[0x32] = op_32

        # ================================================================
        # 0x35 -- AND zpx
        # ================================================================
        def op_35():
            self.i_and(mem[self.a_zpx()]); self.clk(4)
        t[0x35] = op_35

        # ================================================================
        # 0x36 -- ROL zpx
        # ================================================================
        def op_36():
            ea = self.a_zpx(); mem[ea] = self.i_rol(mem[ea]); self.clk(6)
        t[0x36] = op_36

        # ================================================================
        # 0x38 -- SEC
        # ================================================================
        def op_38():
            self.i_sec(); self.clk(2)
        t[0x38] = op_38

        # ================================================================
        # 0x39 -- AND aby
        # ================================================================
        def op_39():
            self.i_and(mem[self.a_aby(1)]); self.clk(4)
        t[0x39] = op_39

        # ================================================================
        # 0x3C -- *TOP abx
        # ================================================================
        def op_3c():
            self.a_abx(0); self.clk(2)
        t[0x3C] = op_3c

        # ================================================================
        # 0x3D -- AND abx
        # ================================================================
        def op_3d():
            self.i_and(mem[self.a_abx(1)]); self.clk(4)
        t[0x3D] = op_3d

        # ================================================================
        # 0x3E -- ROL abx
        # ================================================================
        def op_3e():
            ea = self.a_abx(0); mem[ea] = self.i_rol(mem[ea]); self.clk(7)
        t[0x3E] = op_3e

        # ================================================================
        # 0x3F -- *RLA abx
        # ================================================================
        def op_3f():
            ea = self.a_abx(0); self.i_rla(ea); self.clk(4)
        t[0x3F] = op_3f

        # ================================================================
        # 0x40 -- RTI
        # ================================================================
        def op_40():
            self.i_rti(); self.clk(6)
        t[0x40] = op_40

        # ================================================================
        # 0x41 -- EOR (idx)
        # ================================================================
        def op_41():
            self.i_eor(mem[self.a_idx()]); self.clk(6)
        t[0x41] = op_41

        # ================================================================
        # 0x42 -- *KIL
        # ================================================================
        def op_42():
            self.i_kil(); self.clk(2)
        t[0x42] = op_42

        # ================================================================
        # 0x45 -- EOR zpg
        # ================================================================
        def op_45():
            self.i_eor(mem[self.a_zpg()]); self.clk(3)
        t[0x45] = op_45

        # ================================================================
        # 0x46 -- LSR zpg
        # ================================================================
        def op_46():
            ea = self.a_zpg(); mem[ea] = self.i_lsr(mem[ea]); self.clk(5)
        t[0x46] = op_46

        # ================================================================
        # 0x48 -- PHA
        # ================================================================
        def op_48():
            self.i_pha(); self.clk(3)
        t[0x48] = op_48

        # ================================================================
        # 0x49 -- EOR #imm
        # ================================================================
        def op_49():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_eor(v); self.clk(2)
        t[0x49] = op_49

        # ================================================================
        # 0x4A -- LSR A
        # ================================================================
        def op_4a():
            self.a = self.i_lsr(self.a); self.clk(2)
        t[0x4A] = op_4a

        # ================================================================
        # 0x4B -- *ALR #imm
        # ================================================================
        def op_4b():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_alr(v); self.clk(2)
        t[0x4B] = op_4b

        # ================================================================
        # 0x4C -- JMP abs
        # ================================================================
        def op_4c():
            self.i_jmp(self.a_abs()); self.clk(3)
        t[0x4C] = op_4c

        # ================================================================
        # 0x4D -- EOR abs
        # ================================================================
        def op_4d():
            self.i_eor(mem[self.a_abs()]); self.clk(4)
        t[0x4D] = op_4d

        # ================================================================
        # 0x4E -- LSR abs
        # ================================================================
        def op_4e():
            ea = self.a_abs(); mem[ea] = self.i_lsr(mem[ea]); self.clk(6)
        t[0x4E] = op_4e

        # ================================================================
        # 0x50 -- BVC rel
        # ================================================================
        def op_50():
            ea = self.a_rel(); self.br(not self.fV, ea); self.clk(2)
        t[0x50] = op_50

        # ================================================================
        # 0x51 -- EOR (idy)
        # ================================================================
        def op_51():
            self.i_eor(mem[self.a_idy(1)]); self.clk(5)
        t[0x51] = op_51

        # ================================================================
        # 0x52 -- *KIL
        # ================================================================
        def op_52():
            self.i_kil(); self.clk(2)
        t[0x52] = op_52

        # ================================================================
        # 0x55 -- EOR zpx
        # ================================================================
        def op_55():
            self.i_eor(mem[self.a_zpx()]); self.clk(4)
        t[0x55] = op_55

        # ================================================================
        # 0x56 -- LSR zpx
        # ================================================================
        def op_56():
            ea = self.a_zpx(); mem[ea] = self.i_lsr(mem[ea]); self.clk(6)
        t[0x56] = op_56

        # ================================================================
        # 0x58 -- CLI
        # ================================================================
        def op_58():
            self.i_cli(); self.clk(2)
        t[0x58] = op_58

        # ================================================================
        # 0x59 -- EOR aby
        # ================================================================
        def op_59():
            self.i_eor(mem[self.a_aby(1)]); self.clk(4)
        t[0x59] = op_59

        # ================================================================
        # 0x5C -- *TOP abx
        # ================================================================
        def op_5c():
            self.a_abx(0); self.clk(2)
        t[0x5C] = op_5c

        # ================================================================
        # 0x5D -- EOR abx
        # ================================================================
        def op_5d():
            self.i_eor(mem[self.a_abx(1)]); self.clk(4)
        t[0x5D] = op_5d

        # ================================================================
        # 0x5E -- LSR abx
        # ================================================================
        def op_5e():
            ea = self.a_abx(0); mem[ea] = self.i_lsr(mem[ea]); self.clk(7)
        t[0x5E] = op_5e

        # ================================================================
        # 0x60 -- RTS
        # ================================================================
        def op_60():
            self.i_rts(); self.clk(6)
        t[0x60] = op_60

        # ================================================================
        # 0x61 -- ADC (idx)
        # ================================================================
        def op_61():
            self.i_adc(mem[self.a_idx()]); self.clk(6)
        t[0x61] = op_61

        # ================================================================
        # 0x62 -- *KIL
        # ================================================================
        def op_62():
            self.i_kil(); self.clk(2)
        t[0x62] = op_62

        # ================================================================
        # 0x65 -- ADC zpg
        # ================================================================
        def op_65():
            self.i_adc(mem[self.a_zpg()]); self.clk(3)
        t[0x65] = op_65

        # ================================================================
        # 0x66 -- ROR zpg
        # ================================================================
        def op_66():
            ea = self.a_zpg(); mem[ea] = self.i_ror(mem[ea]); self.clk(5)
        t[0x66] = op_66

        # ================================================================
        # 0x68 -- PLA
        # ================================================================
        def op_68():
            self.i_pla(); self.clk(4)
        t[0x68] = op_68

        # ================================================================
        # 0x69 -- ADC #imm
        # ================================================================
        def op_69():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_adc(v); self.clk(2)
        t[0x69] = op_69

        # ================================================================
        # 0x6A -- ROR A
        # ================================================================
        def op_6a():
            self.a = self.i_ror(self.a); self.clk(2)
        t[0x6A] = op_6a

        # ================================================================
        # 0x6C -- JMP (ind)
        # ================================================================
        def op_6c():
            self.i_jmp(self.a_ind()); self.clk(5)
        t[0x6C] = op_6c

        # ================================================================
        # 0x6D -- ADC abs
        # ================================================================
        def op_6d():
            self.i_adc(mem[self.a_abs()]); self.clk(4)
        t[0x6D] = op_6d

        # ================================================================
        # 0x6E -- ROR abs
        # ================================================================
        def op_6e():
            ea = self.a_abs(); mem[ea] = self.i_ror(mem[ea]); self.clk(6)
        t[0x6E] = op_6e

        # ================================================================
        # 0x70 -- BVS rel
        # ================================================================
        def op_70():
            ea = self.a_rel(); self.br(self.fV, ea); self.clk(2)
        t[0x70] = op_70

        # ================================================================
        # 0x71 -- ADC (idy)
        # ================================================================
        def op_71():
            self.i_adc(mem[self.a_idy(1)]); self.clk(5)
        t[0x71] = op_71

        # ================================================================
        # 0x72 -- *KIL
        # ================================================================
        def op_72():
            self.i_kil(); self.clk(2)
        t[0x72] = op_72

        # ================================================================
        # 0x75 -- ADC zpx
        # ================================================================
        def op_75():
            self.i_adc(mem[self.a_zpx()]); self.clk(4)
        t[0x75] = op_75

        # ================================================================
        # 0x76 -- ROR zpx
        # ================================================================
        def op_76():
            ea = self.a_zpx(); mem[ea] = self.i_ror(mem[ea]); self.clk(6)
        t[0x76] = op_76

        # ================================================================
        # 0x78 -- SEI
        # ================================================================
        def op_78():
            self.i_sei(); self.clk(2)
        t[0x78] = op_78

        # ================================================================
        # 0x79 -- ADC aby
        # ================================================================
        def op_79():
            self.i_adc(mem[self.a_aby(1)]); self.clk(4)
        t[0x79] = op_79

        # ================================================================
        # 0x7C -- *TOP abx
        # ================================================================
        def op_7c():
            self.a_abx(0); self.clk(2)
        t[0x7C] = op_7c

        # ================================================================
        # 0x7D -- ADC abx
        # ================================================================
        def op_7d():
            self.i_adc(mem[self.a_abx(1)]); self.clk(4)
        t[0x7D] = op_7d

        # ================================================================
        # 0x7E -- ROR abx
        # ================================================================
        def op_7e():
            ea = self.a_abx(0); mem[ea] = self.i_ror(mem[ea]); self.clk(7)
        t[0x7E] = op_7e

        # ================================================================
        # 0x81 -- STA (idx)
        # ================================================================
        def op_81():
            self.i_sta(self.a_idx()); self.clk(6)
        t[0x81] = op_81

        # ================================================================
        # 0x83 -- *SAX (idx)
        # ================================================================
        def op_83():
            self.i_sax(self.a_idx()); self.clk(6)
        t[0x83] = op_83

        # ================================================================
        # 0x84 -- STY zpg
        # ================================================================
        def op_84():
            self.i_sty(self.a_zpg()); self.clk(3)
        t[0x84] = op_84

        # ================================================================
        # 0x85 -- STA zpg
        # ================================================================
        def op_85():
            self.i_sta(self.a_zpg()); self.clk(3)
        t[0x85] = op_85

        # ================================================================
        # 0x86 -- STX zpg
        # ================================================================
        def op_86():
            self.i_stx(self.a_zpg()); self.clk(3)
        t[0x86] = op_86

        # ================================================================
        # 0x87 -- *SAX zpg
        # ================================================================
        def op_87():
            self.i_sax(self.a_zpg()); self.clk(3)
        t[0x87] = op_87

        # ================================================================
        # 0x88 -- DEY
        # ================================================================
        def op_88():
            self.i_dey(); self.clk(2)
        t[0x88] = op_88

        # ================================================================
        # 0x8A -- TXA
        # ================================================================
        def op_8a():
            self.i_txa(); self.clk(2)
        t[0x8A] = op_8a

        # ================================================================
        # 0x8C -- STY abs
        # ================================================================
        def op_8c():
            self.i_sty(self.a_abs()); self.clk(4)
        t[0x8C] = op_8c

        # ================================================================
        # 0x8D -- STA abs
        # ================================================================
        def op_8d():
            self.i_sta(self.a_abs()); self.clk(4)
        t[0x8D] = op_8d

        # ================================================================
        # 0x8E -- STX abs
        # ================================================================
        def op_8e():
            self.i_stx(self.a_abs()); self.clk(4)
        t[0x8E] = op_8e

        # ================================================================
        # 0x8F -- *SAX abs
        # ================================================================
        def op_8f():
            self.i_sax(self.a_abs()); self.clk(4)
        t[0x8F] = op_8f

        # ================================================================
        # 0x90 -- BCC rel
        # ================================================================
        def op_90():
            ea = self.a_rel(); self.br(not self.fC, ea); self.clk(2)
        t[0x90] = op_90

        # ================================================================
        # 0x91 -- STA (idy) -- always 6 cycles (write)
        # ================================================================
        def op_91():
            self.i_sta(self.a_idy(0)); self.clk(6)
        t[0x91] = op_91

        # ================================================================
        # 0x92 -- *KIL
        # ================================================================
        def op_92():
            self.i_kil(); self.clk(2)
        t[0x92] = op_92

        # ================================================================
        # 0x94 -- STY zpx
        # ================================================================
        def op_94():
            self.i_sty(self.a_zpx()); self.clk(4)
        t[0x94] = op_94

        # ================================================================
        # 0x95 -- STA zpx
        # ================================================================
        def op_95():
            self.i_sta(self.a_zpx()); self.clk(4)
        t[0x95] = op_95

        # ================================================================
        # 0x96 -- STX zpy
        # ================================================================
        def op_96():
            self.i_stx(self.a_zpy()); self.clk(4)
        t[0x96] = op_96

        # ================================================================
        # 0x97 -- *SAX zpy
        # ================================================================
        def op_97():
            self.i_sax(self.a_zpy()); self.clk(4)
        t[0x97] = op_97

        # ================================================================
        # 0x98 -- TYA
        # ================================================================
        def op_98():
            self.i_tya(); self.clk(2)
        t[0x98] = op_98

        # ================================================================
        # 0x99 -- STA aby -- always 5 cycles (write)
        # ================================================================
        def op_99():
            self.i_sta(self.a_aby(0)); self.clk(5)
        t[0x99] = op_99

        # ================================================================
        # 0x9A -- TXS
        # ================================================================
        def op_9a():
            self.i_txs(); self.clk(2)
        t[0x9A] = op_9a

        # ================================================================
        # 0x9C -- *TOP abx (SHY / SYA in some references)
        # ================================================================
        def op_9c():
            self.a_abx(0); self.clk(2)
        t[0x9C] = op_9c

        # ================================================================
        # 0x9D -- STA abx -- always 5 cycles (write)
        # ================================================================
        def op_9d():
            self.i_sta(self.a_abx(0)); self.clk(5)
        t[0x9D] = op_9d

        # ================================================================
        # 0xA0 -- LDY #imm
        # ================================================================
        def op_a0():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_ldy(v); self.clk(2)
        t[0xA0] = op_a0

        # ================================================================
        # 0xA1 -- LDA (idx)
        # ================================================================
        def op_a1():
            self.i_lda(mem[self.a_idx()]); self.clk(6)
        t[0xA1] = op_a1

        # ================================================================
        # 0xA2 -- LDX #imm
        # ================================================================
        def op_a2():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_ldx(v); self.clk(2)
        t[0xA2] = op_a2

        # ================================================================
        # 0xA3 -- *LAX (idx)
        # ================================================================
        def op_a3():
            self.i_lax(mem[self.a_idx()]); self.clk(6)
        t[0xA3] = op_a3

        # ================================================================
        # 0xA4 -- LDY zpg
        # ================================================================
        def op_a4():
            self.i_ldy(mem[self.a_zpg()]); self.clk(3)
        t[0xA4] = op_a4

        # ================================================================
        # 0xA5 -- LDA zpg
        # ================================================================
        def op_a5():
            self.i_lda(mem[self.a_zpg()]); self.clk(3)
        t[0xA5] = op_a5

        # ================================================================
        # 0xA6 -- LDX zpg
        # ================================================================
        def op_a6():
            self.i_ldx(mem[self.a_zpg()]); self.clk(3)
        t[0xA6] = op_a6

        # ================================================================
        # 0xA7 -- *LAX zpg
        # ================================================================
        def op_a7():
            self.i_lax(mem[self.a_zpg()]); self.clk(3)
        t[0xA7] = op_a7

        # ================================================================
        # 0xA8 -- TAY
        # ================================================================
        def op_a8():
            self.i_tay(); self.clk(2)
        t[0xA8] = op_a8

        # ================================================================
        # 0xA9 -- LDA #imm
        # ================================================================
        def op_a9():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_lda(v); self.clk(2)
        t[0xA9] = op_a9

        # ================================================================
        # 0xAA -- TAX
        # ================================================================
        def op_aa():
            self.i_tax(); self.clk(2)
        t[0xAA] = op_aa

        # ================================================================
        # 0xAC -- LDY abs
        # ================================================================
        def op_ac():
            self.i_ldy(mem[self.a_abs()]); self.clk(4)
        t[0xAC] = op_ac

        # ================================================================
        # 0xAD -- LDA abs
        # ================================================================
        def op_ad():
            self.i_lda(mem[self.a_abs()]); self.clk(4)
        t[0xAD] = op_ad

        # ================================================================
        # 0xAE -- LDX abs
        # ================================================================
        def op_ae():
            self.i_ldx(mem[self.a_abs()]); self.clk(4)
        t[0xAE] = op_ae

        # ================================================================
        # 0xAF -- *LAX abs
        # ================================================================
        def op_af():
            self.i_lax(mem[self.a_abs()]); self.clk(5)
        t[0xAF] = op_af

        # ================================================================
        # 0xB0 -- BCS rel
        # ================================================================
        def op_b0():
            ea = self.a_rel(); self.br(self.fC, ea); self.clk(2)
        t[0xB0] = op_b0

        # ================================================================
        # 0xB1 -- LDA (idy)
        # ================================================================
        def op_b1():
            self.i_lda(mem[self.a_idy(1)]); self.clk(5)
        t[0xB1] = op_b1

        # ================================================================
        # 0xB2 -- *KIL
        # ================================================================
        def op_b2():
            self.i_kil(); self.clk(2)
        t[0xB2] = op_b2

        # ================================================================
        # 0xB3 -- *LAX (idy)
        # ================================================================
        def op_b3():
            self.i_lax(mem[self.a_idy(0)]); self.clk(6)
        t[0xB3] = op_b3

        # ================================================================
        # 0xB4 -- LDY zpx
        # ================================================================
        def op_b4():
            self.i_ldy(mem[self.a_zpx()]); self.clk(4)
        t[0xB4] = op_b4

        # ================================================================
        # 0xB5 -- LDA zpx
        # ================================================================
        def op_b5():
            self.i_lda(mem[self.a_zpx()]); self.clk(4)
        t[0xB5] = op_b5

        # ================================================================
        # 0xB6 -- LDX zpy
        # ================================================================
        def op_b6():
            self.i_ldx(mem[self.a_zpy()]); self.clk(4)
        t[0xB6] = op_b6

        # ================================================================
        # 0xB7 -- *LAX zpy
        # ================================================================
        def op_b7():
            self.i_lax(mem[self.a_zpy()]); self.clk(4)
        t[0xB7] = op_b7

        # ================================================================
        # 0xB8 -- CLV
        # ================================================================
        def op_b8():
            self.i_clv(); self.clk(2)
        t[0xB8] = op_b8

        # ================================================================
        # 0xB9 -- LDA aby
        # ================================================================
        def op_b9():
            self.i_lda(mem[self.a_aby(1)]); self.clk(4)
        t[0xB9] = op_b9

        # ================================================================
        # 0xBA -- TSX
        # ================================================================
        def op_ba():
            self.i_tsx(); self.clk(2)
        t[0xBA] = op_ba

        # ================================================================
        # 0xBC -- LDY abx
        # ================================================================
        def op_bc():
            self.i_ldy(mem[self.a_abx(1)]); self.clk(4)
        t[0xBC] = op_bc

        # ================================================================
        # 0xBD -- LDA abx
        # ================================================================
        def op_bd():
            self.i_lda(mem[self.a_abx(1)]); self.clk(4)
        t[0xBD] = op_bd

        # ================================================================
        # 0xBE -- LDX aby
        # ================================================================
        def op_be():
            self.i_ldx(mem[self.a_aby(1)]); self.clk(4)
        t[0xBE] = op_be

        # ================================================================
        # 0xBF -- *LAX aby
        # ================================================================
        def op_bf():
            self.i_lax(mem[self.a_aby(0)]); self.clk(6)
        t[0xBF] = op_bf

        # ================================================================
        # 0xC0 -- CPY #imm
        # ================================================================
        def op_c0():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_cpy(v); self.clk(2)
        t[0xC0] = op_c0

        # ================================================================
        # 0xC1 -- CMP (idx)
        # ================================================================
        def op_c1():
            self.i_cmp(mem[self.a_idx()]); self.clk(6)
        t[0xC1] = op_c1

        # ================================================================
        # 0xC4 -- CPY zpg
        # ================================================================
        def op_c4():
            self.i_cpy(mem[self.a_zpg()]); self.clk(3)
        t[0xC4] = op_c4

        # ================================================================
        # 0xC5 -- CMP zpg
        # ================================================================
        def op_c5():
            self.i_cmp(mem[self.a_zpg()]); self.clk(3)
        t[0xC5] = op_c5

        # ================================================================
        # 0xC6 -- DEC zpg
        # ================================================================
        def op_c6():
            ea = self.a_zpg(); mem[ea] = self.i_dec(mem[ea]); self.clk(5)
        t[0xC6] = op_c6

        # ================================================================
        # 0xC8 -- INY
        # ================================================================
        def op_c8():
            self.i_iny(); self.clk(2)
        t[0xC8] = op_c8

        # ================================================================
        # 0xC9 -- CMP #imm
        # ================================================================
        def op_c9():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_cmp(v); self.clk(2)
        t[0xC9] = op_c9

        # ================================================================
        # 0xCA -- DEX
        # ================================================================
        def op_ca():
            self.i_dex(); self.clk(2)
        t[0xCA] = op_ca

        # ================================================================
        # 0xCC -- CPY abs
        # ================================================================
        def op_cc():
            self.i_cpy(mem[self.a_abs()]); self.clk(4)
        t[0xCC] = op_cc

        # ================================================================
        # 0xCD -- CMP abs
        # ================================================================
        def op_cd():
            self.i_cmp(mem[self.a_abs()]); self.clk(4)
        t[0xCD] = op_cd

        # ================================================================
        # 0xCE -- DEC abs
        # ================================================================
        def op_ce():
            ea = self.a_abs(); mem[ea] = self.i_dec(mem[ea]); self.clk(6)
        t[0xCE] = op_ce

        # ================================================================
        # 0xD0 -- BNE rel
        # ================================================================
        def op_d0():
            ea = self.a_rel(); self.br(not self.fZ, ea); self.clk(2)
        t[0xD0] = op_d0

        # ================================================================
        # 0xD1 -- CMP (idy)
        # ================================================================
        def op_d1():
            self.i_cmp(mem[self.a_idy(1)]); self.clk(5)
        t[0xD1] = op_d1

        # ================================================================
        # 0xD2 -- *KIL
        # ================================================================
        def op_d2():
            self.i_kil(); self.clk(2)
        t[0xD2] = op_d2

        # ================================================================
        # 0xD5 -- CMP zpx
        # ================================================================
        def op_d5():
            self.i_cmp(mem[self.a_zpx()]); self.clk(4)
        t[0xD5] = op_d5

        # ================================================================
        # 0xD6 -- DEC zpx
        # ================================================================
        def op_d6():
            ea = self.a_zpx(); mem[ea] = self.i_dec(mem[ea]); self.clk(6)
        t[0xD6] = op_d6

        # ================================================================
        # 0xD8 -- CLD
        # ================================================================
        def op_d8():
            self.i_cld(); self.clk(2)
        t[0xD8] = op_d8

        # ================================================================
        # 0xD9 -- CMP aby
        # ================================================================
        def op_d9():
            self.i_cmp(mem[self.a_aby(1)]); self.clk(4)
        t[0xD9] = op_d9

        # ================================================================
        # 0xDC -- *TOP abx
        # ================================================================
        def op_dc():
            self.a_abx(0); self.clk(2)
        t[0xDC] = op_dc

        # ================================================================
        # 0xDD -- CMP abx
        # ================================================================
        def op_dd():
            self.i_cmp(mem[self.a_abx(1)]); self.clk(4)
        t[0xDD] = op_dd

        # ================================================================
        # 0xDE -- DEC abx
        # ================================================================
        def op_de():
            ea = self.a_abx(0); mem[ea] = self.i_dec(mem[ea]); self.clk(7)
        t[0xDE] = op_de

        # ================================================================
        # 0xE0 -- CPX #imm
        # ================================================================
        def op_e0():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_cpx(v); self.clk(2)
        t[0xE0] = op_e0

        # ================================================================
        # 0xE1 -- SBC (idx)
        # ================================================================
        def op_e1():
            self.i_sbc(mem[self.a_idx()]); self.clk(6)
        t[0xE1] = op_e1

        # ================================================================
        # 0xE4 -- CPX zpg
        # ================================================================
        def op_e4():
            self.i_cpx(mem[self.a_zpg()]); self.clk(3)
        t[0xE4] = op_e4

        # ================================================================
        # 0xE5 -- SBC zpg
        # ================================================================
        def op_e5():
            self.i_sbc(mem[self.a_zpg()]); self.clk(3)
        t[0xE5] = op_e5

        # ================================================================
        # 0xE6 -- INC zpg
        # ================================================================
        def op_e6():
            ea = self.a_zpg(); mem[ea] = self.i_inc(mem[ea]); self.clk(5)
        t[0xE6] = op_e6

        # ================================================================
        # 0xE8 -- INX
        # ================================================================
        def op_e8():
            self.i_inx(); self.clk(2)
        t[0xE8] = op_e8

        # ================================================================
        # 0xE9 -- SBC #imm
        # ================================================================
        def op_e9():
            v = mem[self.pc]; self.pc = (self.pc + 1) & 0xFFFF
            self.i_sbc(v); self.clk(2)
        t[0xE9] = op_e9

        # ================================================================
        # 0xEA -- NOP
        # ================================================================
        def op_ea():
            self.clk(2)
        t[0xEA] = op_ea

        # ================================================================
        # 0xEC -- CPX abs
        # ================================================================
        def op_ec():
            self.i_cpx(mem[self.a_abs()]); self.clk(4)
        t[0xEC] = op_ec

        # ================================================================
        # 0xED -- SBC abs
        # ================================================================
        def op_ed():
            self.i_sbc(mem[self.a_abs()]); self.clk(4)
        t[0xED] = op_ed

        # ================================================================
        # 0xEE -- INC abs
        # ================================================================
        def op_ee():
            ea = self.a_abs(); mem[ea] = self.i_inc(mem[ea]); self.clk(6)
        t[0xEE] = op_ee

        # ================================================================
        # 0xEF -- *ISB abs
        # ================================================================
        def op_ef():
            ea = self.a_abs(); self.i_isb(ea); self.clk(6)
        t[0xEF] = op_ef

        # ================================================================
        # 0xF0 -- BEQ rel
        # ================================================================
        def op_f0():
            ea = self.a_rel(); self.br(self.fZ, ea); self.clk(2)
        t[0xF0] = op_f0

        # ================================================================
        # 0xF1 -- SBC (idy)
        # ================================================================
        def op_f1():
            self.i_sbc(mem[self.a_idy(1)]); self.clk(5)
        t[0xF1] = op_f1

        # ================================================================
        # 0xF2 -- *KIL
        # ================================================================
        def op_f2():
            self.i_kil(); self.clk(2)
        t[0xF2] = op_f2

        # ================================================================
        # 0xF5 -- SBC zpx
        # ================================================================
        def op_f5():
            self.i_sbc(mem[self.a_zpx()]); self.clk(4)
        t[0xF5] = op_f5

        # ================================================================
        # 0xF6 -- INC zpx
        # ================================================================
        def op_f6():
            ea = self.a_zpx(); mem[ea] = self.i_inc(mem[ea]); self.clk(6)
        t[0xF6] = op_f6

        # ================================================================
        # 0xF8 -- SED
        # ================================================================
        def op_f8():
            self.i_sed(); self.clk(2)
        t[0xF8] = op_f8

        # ================================================================
        # 0xF9 -- SBC aby
        # ================================================================
        def op_f9():
            self.i_sbc(mem[self.a_aby(1)]); self.clk(4)
        t[0xF9] = op_f9

        # ================================================================
        # 0xFC -- *TOP abx
        # ================================================================
        def op_fc():
            self.a_abx(0); self.clk(2)
        t[0xFC] = op_fc

        # ================================================================
        # 0xFD -- SBC abx
        # ================================================================
        def op_fd():
            self.i_sbc(mem[self.a_abx(1)]); self.clk(4)
        t[0xFD] = op_fd

        # ================================================================
        # 0xFE -- INC abx
        # ================================================================
        def op_fe():
            ea = self.a_abx(0); mem[ea] = self.i_inc(mem[ea]); self.clk(7)
        t[0xFE] = op_fe

        # ================================================================
        # 0xFF -- *ISB abx
        # ================================================================
        def op_ff():
            ea = self.a_abx(0); self.i_isb(ea); self.clk(7)
        t[0xFF] = op_ff

        return t

    # ------------------------------------------------------------------
    # Debug / repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"M6502(PC=${self.pc:04X} A=${self.a:02X} "
            f"X=${self.x:02X} Y=${self.y:02X} "
            f"S=${self.s:02X} P=${self.p:02X} "
            f"clk={self.clock})"
        )
