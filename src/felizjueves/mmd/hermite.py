"""The McMurchie-Davidson Hermite expansion coefficients `E`, undressed.

Per axis, for a primitive pair with exponents `alpha` on `A` and `beta` on `B`,
`zeta = alpha + beta`, `P = (alpha A + beta B) / zeta`, `mu = alpha beta / zeta`:

    E^{00}_0    = exp(-mu X_AB^2)
    E^{i+1,j}_t = (1/2zeta) E^{ij}_{t-1} + X_PA E^{ij}_t + (t+1) E^{ij}_{t+1}
    E^{i,j+1}_t = (1/2zeta) E^{ij}_{t-1} + X_PB E^{ij}_t + (t+1) E^{ij}_{t+1}

with `E^{ij}_t = 0` for `t < 0` or `t > i+j`. These are McMurchie & Davidson's
`d^{nn'}_N` (report eqs. 2.20-2.22, entry 46 of docs/references.md) with their
per-axis share of `E_IJ` (eq. 2.10) folded into `E^{00}_0`, so that

    x_A^i x_B^j exp(-alpha x_A^2) exp(-beta x_B^2) = sum_t E^{ij}_t Lambda_t(x)
    Lambda_t(x) = zeta^{t/2} H_t(sqrt(zeta) (x - P)) exp(-zeta (x - P)^2)

holds pointwise (eqs. 2.7, 2.16, 2.17). That identity is the only independent
check this quantity has (test/test_ao_hermite.py), so this module produces the
PURE coefficients: no contraction coefficient, no primitive normalisation, no
`(pi/zeta)^{3/2}`. All three dressings are `ao/pairs.py`'s, in one function
(docs/implementation/j_engine.md, "The `E` transform").

`mu X_AB^2` is formed as `-zeta X_PA X_PB`: with `X_PA = -(beta/zeta) X_AB` and
`X_PB = (alpha/zeta) X_AB` the two are equal, so the pair `(X_PA, X_PB)` already
determines the prefactor and no separate `X_AB` (or `mu`) argument is taken --
one would be either redundant or inconsistent with them.

Pure Python floats: `src/felizjueves` has no runtime dependencies.
"""

from __future__ import annotations

import math
from functools import cache

Vec = tuple[float, float, float]
Ang = tuple[int, int, int]


def _check_am(name: str, n: int) -> None:
    if type(n) is not int or n < 0:
        raise ValueError(f"{name} = {n!r} is not an int >= 0")


def _raise(prev: list[float], x: float, half_inv_zeta: float) -> list[float]:
    """One step of eq. 2.20 (x = X_PA) or 2.21 (x = X_PB): length n -> n+1."""
    n = len(prev)
    out = []
    for t in range(n + 1):
        v = 0.0
        if t >= 1:
            v += half_inv_zeta * prev[t - 1]
        if t < n:
            v += x * prev[t]
        if t + 1 < n:
            v += (t + 1) * prev[t + 1]
        out.append(v)
    return out


def e_axis(i: int, j: int, zeta: float, pa: float, pb: float) -> tuple[float, ...]:
    """`E^{ij}_t` for `t = 0..i+j` on one axis -- exactly `i+j+1` entries.

    `pa = P - A` and `pb = P - B` along the axis (MD's `PA_x`, eq. 2.14).
    Not symmetric under `i <-> j` unless `pa` and `pb` swap with it.
    """
    _check_am("i", i)
    _check_am("j", j)
    if not zeta > 0.0:
        raise ValueError(f"zeta = {zeta!r} is not > 0")
    e = [math.exp(zeta * pa * pb)]  # = exp(-mu X_AB^2); see the module doc
    h = 0.5 / zeta
    for _ in range(i):
        e = _raise(e, pa, h)
    for _ in range(j):
        e = _raise(e, pb, h)
    return tuple(e)


def hermite_order(total_am: int) -> tuple[Ang, ...]:
    """Every `t` with `|t| <= L`: `|t|` ascending, then tx, then ty descending.

    The order of `plan.targets.hermite_targets`, enumerated here because `ao`
    may not import `plan`; the test asserts the two are equal.
    """
    _check_am("total_am", total_am)
    return _hermite_order(total_am)


@cache
def _hermite_order(total_am: int) -> tuple[Ang, ...]:
    # Behind `hermite_order`'s check, so a 1.0 or a True is refused, never
    # served the int's entry; the tuple is immutable, so sharing it is safe.
    return tuple(
        (tx, ty, n - tx - ty)
        for n in range(total_am + 1)
        for tx in range(n, -1, -1)
        for ty in range(n - tx, -1, -1)
    )


def e_block(a: Ang, b: Ang, zeta: float, p: Vec, A: Vec, B: Vec) -> tuple[float, ...]:
    """The 3D `E^{ab}_t = E_x E_y E_z` over `hermite_order(|a|+|b|)`.

    An index with `t_d > a_d + b_d` on some axis is exactly 0.0, so the block
    has `(L+1)(L+2)(L+3)/6` entries whatever the split of `L` over axes.
    """
    axes = [
        e_axis(a[d], b[d], zeta, p[d] - A[d], p[d] - B[d]) for d in range(3)
    ]
    out = []
    for t in hermite_order(sum(a) + sum(b)):
        v = 1.0
        for d in range(3):
            v = v * axes[d][t[d]] if t[d] < len(axes[d]) else 0.0
        out.append(v)
    return tuple(out)
