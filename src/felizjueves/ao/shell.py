"""The contracted Cartesian shell, the basis, and the one home for normalisation.

A `Shell` is input data -- an angular momentum, a centre, and a contraction
(exponents and coefficients) -- not something the generator produces. It
therefore has no `Spec` field and no `key()`: `Spec` names a kernel, a shell is
what a kernel is eventually run on (`test_layering.py`, rule "ao").

THE NORMALISATION CONVENTION, STATED ONCE AND NOWHERE ELSE IN THIS PACKAGE.

    Our coefficients multiply UNNORMALISED Cartesian primitives

        x^i y^j z^k exp(-alpha r^2),     i + j + k = l,

    with r measured from the shell's centre. A shell's contracted function for
    the component (i, j, k) is  sum_p coefficients[p] * x^i y^j z^k
    exp(-exponents[p] r^2), and nothing is folded in behind the caller's back.

`primitive_norm(l, alpha)` is the factor that makes ONE such primitive have unit
self-overlap -- the AXIS component x^l (equivalently y^l, z^l):

    N(l, alpha) = (2 alpha / pi)^(3/4) (4 alpha)^(l/2) / sqrt((2l - 1)!!)

It is a function of l alone, so it does not normalise every component: with it,
the component (i, j, k) has self-overlap (2i-1)!!(2j-1)!!(2k-1)!! / (2l-1)!!
(1 for every axis component and at l <= 1; 1/3 for xy at l = 2). That is the
common Cartesian convention, and it is a property of N, not a defect.

Whoever wants a normalised contraction multiplies by `primitive_norm`
explicitly. Every mapping onto somebody else's convention (a basis-set file, a
reference code's normalised contractions) happens in the test oracle
(`test/oracle/basis.py`, #84), never here.

`cartesians(l)` enumerates the (l+1)(l+2)/2 components in THE order
`plan.targets.hermite_targets` uses for one |r| -- rx descending, then ry
descending: xx, xy, xz, yy, yz, zz at l = 2. It is written out here rather than
imported (this layer must not reach `plan`); `test/test_ao_shell.py` asserts
the two agree. The order is golden-stable: every fixture from #84 on depends on
it. NOTE the count is (l+1)(l+2)/2 per shell; the Hermite count of a pair,
(L+1)(L+2)(L+3)/6, is a different number.

Plain tuples and floats, like `reference/`: no numpy in the types. `l` is the
angular momentum's conventional name, hence the `noqa: E741`s.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

Cart = tuple[int, int, int]
Vec = tuple[float, float, float]


def _check_l(l: object) -> int:  # noqa: E741
    if type(l) is not int or l < 0:
        raise ValueError(f"angular momentum {l!r} is not an int >= 0")
    return l


def n_cartesians(l: int) -> int:  # noqa: E741
    """(l+1)(l+2)/2 -- the Cartesian components of ONE shell (not a pair's
    Hermite count)."""
    _check_l(l)
    return (l + 1) * (l + 2) // 2


def cartesians(l: int) -> tuple[Cart, ...]:  # noqa: E741
    """Every (i, j, k) with i + j + k = l: i descending, then j descending."""
    _check_l(l)
    return tuple(
        (i, j, l - i - j) for i in range(l, -1, -1) for j in range(l - i, -1, -1)
    )


def _double_factorial_odd(l: int) -> int:  # noqa: E741
    """(2l - 1)!!, with (-1)!! = 1."""
    out = 1
    for k in range(1, 2 * l, 2):
        out *= k
    return out


def primitive_norm(l: int, alpha: float) -> float:  # noqa: E741
    """N with N^2 <x^l e^{-alpha r^2} | x^l e^{-alpha r^2}> = 1.

    The convention is the module docstring's, and lives only here.
    """
    _check_l(l)
    if not (isinstance(alpha, (int, float)) and math.isfinite(alpha) and alpha > 0):
        raise ValueError(f"exponent {alpha!r} is not a finite float > 0")
    alpha = float(alpha)
    return ((2.0 * alpha / math.pi) ** 0.75 * (4.0 * alpha) ** (l / 2.0)
            / math.sqrt(_double_factorial_odd(l)))


def _floats(name: str, xs: object) -> tuple[float, ...]:
    if isinstance(xs, (str, bytes)) or not isinstance(xs, Sequence):
        raise ValueError(f"{name} {xs!r} is not a sequence of floats")
    try:
        out = tuple(float(x) for x in xs)
    except (TypeError, ValueError):
        raise ValueError(f"{name} {xs!r} is not a sequence of floats") from None
    if not all(math.isfinite(x) for x in out):
        raise ValueError(f"{name} {xs!r} has a non-finite entry")
    return out


@dataclass(frozen=True)
class Shell:
    """One contracted Cartesian shell; see the module docstring's convention.

    Sequences are accepted and stored as tuples of floats.
    """

    l: int  # noqa: E741 -- the name the issue and the literature use
    centre: Vec
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        _check_l(self.l)
        centre = _floats("centre", self.centre)
        if len(centre) != 3:
            raise ValueError(f"centre {self.centre!r} is not three floats")
        exponents = _floats("exponents", self.exponents)
        coefficients = _floats("coefficients", self.coefficients)
        if not exponents:
            raise ValueError("a shell needs at least one primitive")
        if len(exponents) != len(coefficients):
            raise ValueError(
                f"{len(exponents)} exponents but {len(coefficients)} "
                "coefficients"
            )
        if not all(a > 0 for a in exponents):
            raise ValueError(f"exponents {self.exponents!r} must all be > 0")
        object.__setattr__(self, "centre", centre)
        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def n_cart(self) -> int:
        """(l+1)(l+2)/2, this shell's AO count."""
        return n_cartesians(self.l)


@dataclass(frozen=True)
class Basis:
    """Shells in AO order, with each shell's first AO index and the total."""

    shells: tuple[Shell, ...]
    offsets: tuple[int, ...] = field(init=False)
    n_ao: int = field(init=False)

    def __post_init__(self) -> None:
        shells = tuple(self.shells)
        if not all(isinstance(s, Shell) for s in shells):
            raise ValueError("a Basis holds Shells only")
        offsets, n = [], 0
        for s in shells:
            offsets.append(n)
            n += s.n_cart
        object.__setattr__(self, "shells", shells)
        object.__setattr__(self, "offsets", tuple(offsets))
        object.__setattr__(self, "n_ao", n)
