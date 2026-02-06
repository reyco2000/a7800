# EMU7800 Cart mappers
"""
Cartridge mappers for Atari 2600 and 7800 systems.

Use :meth:`Cart.create(rom_bytes, cart_type) <cart.Cart.create>` to
instantiate the correct mapper for a given :class:`~emu7800.core.types.CartType`.
"""

from emu7800.core.carts.cart import Cart

from emu7800.core.carts.cart_2600 import (
    CartA2K,
    CartA4K,
    CartA8K,
    CartA8KR,
    CartA16K,
    CartA16KR,
    CartA32K,
    CartA32KR,
    CartCBS12K,
    CartDC8K,
    CartDPC,
    CartMN16K,
    CartPB8K,
    CartTV8K,
)

from emu7800.core.carts.cart_7800 import (
    Cart7808,
    Cart7816,
    Cart7832,
    Cart7832P,
    Cart7832PL,
    Cart7848,
    Cart78AB,
    Cart78AC,
    Cart78BB,
    Cart78BB32K,
    Cart78BB32KP,
    Cart78BB32KRPL,
    Cart78BB48K,
    Cart78BB48KP,
    Cart78BB52K,
    Cart78BB52KP,
    Cart78BB128K,
    Cart78BB128KP,
    Cart78BB128KR,
    Cart78BB128KRPL,
    Cart78S4,
    Cart78S4R,
    Cart78S9,
    Cart78S9PL,
    Cart78SG,
    Cart78SGP,
    Cart78SGR,
)

__all__ = [
    "Cart",
    # 2600
    "CartA2K",
    "CartA4K",
    "CartA8K",
    "CartA8KR",
    "CartA16K",
    "CartA16KR",
    "CartA32K",
    "CartA32KR",
    "CartCBS12K",
    "CartDC8K",
    "CartDPC",
    "CartMN16K",
    "CartPB8K",
    "CartTV8K",
    # 7800
    "Cart7808",
    "Cart7816",
    "Cart7832",
    "Cart7832P",
    "Cart7832PL",
    "Cart7848",
    "Cart78AB",
    "Cart78AC",
    "Cart78BB",
    "Cart78BB32K",
    "Cart78BB32KP",
    "Cart78BB32KRPL",
    "Cart78BB48K",
    "Cart78BB48KP",
    "Cart78BB52K",
    "Cart78BB52KP",
    "Cart78BB128K",
    "Cart78BB128KP",
    "Cart78BB128KR",
    "Cart78BB128KRPL",
    "Cart78S4",
    "Cart78S4R",
    "Cart78S9",
    "Cart78S9PL",
    "Cart78SG",
    "Cart78SGP",
    "Cart78SGR",
]
