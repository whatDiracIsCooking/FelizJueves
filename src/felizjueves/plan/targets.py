"""The Hermite targets of a class: every [r](0) with |r| <= L.

A target is what a kernel for a class of total angular momentum L must
produce (docs/implementation/j_engine.md, "Notation"). `hermite_targets` is
the enumeration; `n_targets` is the closed form, kept separate so a test can
assert one against the other rather than one being defined as the other.
"""

from __future__ import annotations

R = tuple[int, int, int]


def _check(total_am: int) -> None:
    if type(total_am) is not int or total_am < 0:
        raise ValueError(f"total angular momentum {total_am!r} is not an int >= 0")


def hermite_targets(total_am: int) -> tuple[R, ...]:
    """Every [r](0) with |r| <= L, in a FIXED order (golden-file stability).

    The order is by |r| ascending, then within one |r| by rx descending, then
    ry descending -- the conventional Cartesian order (xx, xy, xz, yy, yz, zz
    at |r| = 2). Emitted text will follow this order, so changing it is a
    golden-file diff, not a refactor.
    """
    _check(total_am)
    return tuple(
        (rx, ry, n - rx - ry)
        for n in range(total_am + 1)
        for rx in range(n, -1, -1)
        for ry in range(n - rx, -1, -1)
    )


def n_targets(total_am: int) -> int:
    """(L+1)(L+2)(L+3)/6 -- asserted against len(hermite_targets), never used
    as its implementation."""
    _check(total_am)
    L = total_am
    return (L + 1) * (L + 2) * (L + 3) // 6
