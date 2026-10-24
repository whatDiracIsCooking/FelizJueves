"""The Cartesian multipole matrices `<a| q (r - C)^e |b>`, the dipole at `e = 1`.

    hermite_moments(e, x_pc, zeta)  -> M^e_t, t = 0..e, on one axis
    multipole_axis(ij_e, moments)   -> one axis's SUM_t E_t M^e_t
    multipole_block(sa, sb, C, e=1, q=1.0) -> [n_comp][n_cart(sa)][n_cart(sb)]
    multipole_matrix(basis, C, e=1, q=1.0) -> [n_comp][n_ao][n_ao]

THE HERMITE ROUTE (milestone 19, #520; Helgaker, Jorgensen & Olsen, sec.
9.5.4). A primitive product is `SUM_t E^{ab}_t Lambda_t` (`mmd/hermite.py`),
and on one axis the moment of a Hermite Gaussian about `C`,
`INT x_C^e Lambda_t dx = sqrt(pi/zeta) M^e_t`, obeys

    M^0_t     = delta_{t0}
    M^{e+1}_t = t M^e_{t-1} + X_PC M^e_t + M^e_{t+1} / (2 zeta)

(`X_PC = P_x - C_x`), so `M^e_t = 0` for `t > e` and one axis of the
integral is `SUM_{t <= min(e, i+j)} E^{ij}_t M^e_t` -- at `e = 1`,
`E^{ij}_0 X_PC + E^{ij}_1`. The `sqrt(pi/zeta)` each axis carries is NOT in
`M`: the three of them are the pair factor, in the record weight `s` below,
as `one_electron.kinetic` keeps it out of its axis tables. A structurally
absent `E` entry (`t > i + j`, e.g. `E^{00}_1`) is DROPPED from the sum, not
read as zero: the sum stops at `min(e, i + j)`.

THE COMPONENTS. One per `(ex, ey, ez)` in `ao.shell.cartesians(e)` order --
at `e = 1`, x, y, z, libcint's `int1e_r` order -- its value the product of
the three axis factors at orders `ex`, `ey`, `ez`, formed as `mmd.hermite.e_block`
forms a Hermite entry: `v = 1.0`, then `v = v * f_x`, `v * f_y`, `v * f_z`.

THE ASSOCIATION, the one the kernel (#521) will use, so it can be held to
this bitwise: per primitive pair, `s = c_a c_b (pi/zeta)^{3/2}` (`ao.pairs.dress`'s
scale, the record weight `s_ab`), and each element adds `(q * s) * value`
into an accumulator that starts at `0.0`, the pairs in
`ao.pairs.primitive_pairs` order (`i` outer). At `e = 0, q = 1` the one
component is `one_electron.overlap.overlap_block` bitwise (`1.0 * s` is `s`, and
`1.0 * E_x * E_y * E_z` is `e_block`'s `t = 0` entry). The charge `q` is a
plain factor of `s`, so `q = -1` is the exact negation of `q = 1`.

THE MATRIX. `one_electron.one_electron.matrix` places each component (the
operator is real and symmetric in `a, b`), so every component is exactly
symmetric; the block of a shell pair is computed once for all components.

NO `K_AB`, NO NORMALISATION: `one_electron.overlap`'s two prohibitions hold
here unchanged (the `e_axis` seed already carries `K_AB`, and a `Shell`'s
coefficients already carry the normalisation).

Plain lists and floats: `src/felizjueves` has no runtime dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from felizjueves.ao.pairs import primitive_pairs
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.mmd.hermite import e_axis
from felizjueves.one_electron.one_electron import matrix

Vec = tuple[float, float, float]


def _check_e(e: int) -> None:
    if type(e) is not int or e < 0:
        raise ValueError(f"e = {e!r} is not an int >= 0")


def _origin(C: Sequence[float]) -> Vec:
    c = tuple(float(x) for x in C)
    if len(c) != 3 or not all(math.isfinite(x) for x in c):
        raise ValueError(f"origin C = {C!r} is not three finite floats")
    return c  # type: ignore[return-value]


def hermite_moments(e: int, x_pc: float, zeta: float) -> tuple[float, ...]:
    """`M^e_t` for `t = 0..e` on one axis (module docstring), `M^0_0 = 1`."""
    _check_e(e)
    if not zeta > 0.0:
        raise ValueError(f"zeta = {zeta!r} is not > 0")
    m = [1.0]
    h = 0.5 / zeta
    for _ in range(e):
        n = len(m)
        nxt = []
        for t in range(n + 1):
            v = 0.0
            if t >= 1:
                v += t * m[t - 1]
            if t < n:
                v += x_pc * m[t]
            if t + 1 < n:
                v += h * m[t + 1]
            nxt.append(v)
        m = nxt
    return tuple(m)


def multipole_axis(ij_e: Sequence[float], moments: Sequence[float]) -> float:
    """`SUM_{t <= min(e, i+j)} E^{ij}_t M^e_t`, `t` ascending from `0.0`:
    `ij_e` is `e_axis(i, j, ...)` (`i + j + 1` entries) and `moments`
    `hermite_moments(e, ...)` (`e + 1`). Absent `E` entries are dropped."""
    v = 0.0
    for t in range(min(len(ij_e), len(moments))):
        v += ij_e[t] * moments[t]
    return v


def multipole_block(sa: Shell, sb: Shell, C: Sequence[float], e: int = 1,
                    q: float = 1.0) -> list[list[list[float]]]:
    """`<a| q (r - C)^(ex, ey, ez) |b>` for every component in
    `cartesians(e)` order, every `a` in `cartesians(sa.l)` (rows) and `b`
    in `cartesians(sb.l)` (columns), summed over `primitive_pairs(sa, sb)`
    in the module docstring's association."""
    _check_e(e)
    c = _origin(C)
    q = float(q)
    A, B = sa.centre, sb.centre
    comps = cartesians(e)
    carts = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
    nb = sb.n_cart
    acc = [[0.0] * len(carts) for _ in comps]
    # The records' order, P and zeta are `primitive_pairs`'; their dressed
    # block is not read, because it folds `s` into every E, and this
    # association applies `q * s` to the finished value instead. The weight
    # is `ao.pairs.dress`'s scale, spelled the same way.
    for pp in primitive_pairs(sa, sb):
        zeta, P = pp.zeta, pp.P
        s = sa.coefficients[pp.i] * sb.coefficients[pp.j] * (
            math.pi / zeta) ** 1.5
        qs = q * s
        mom = [[hermite_moments(k, P[d] - c[d], zeta) for k in range(e + 1)]
               for d in range(3)]
        for n, (a, b) in enumerate(carts):
            ax = [e_axis(a[d], b[d], zeta, P[d] - A[d], P[d] - B[d])
                  for d in range(3)]
            for k, ek in enumerate(comps):
                v = 1.0
                for d in range(3):
                    v = v * multipole_axis(ax[d], mom[d][ek[d]])
                acc[k][n] += qs * v
    return [[acc[k][r * nb:(r + 1) * nb] for r in range(sa.n_cart)]
            for k in range(len(comps))]


def multipole_matrix(basis: Basis, C: Sequence[float], e: int = 1,
                     q: float = 1.0, *,
                     pairs: Iterable[tuple[int, int]] | None = None
                     ) -> list[list[list[float]]]:
    """`MU[k][m][n]`, one exactly symmetric `basis.n_ao x basis.n_ao`
    matrix per component `k` of `cartesians(e)`, each placed by
    `one_electron.one_electron.matrix`; `pairs` screens them as it does
    `S`. Each shell pair's block is computed once for every component."""
    _check_e(e)
    c = _origin(C)
    pairs = None if pairs is None else tuple(pairs)
    cache: dict[tuple[int, int], list[list[list[float]]]] = {}

    def component(k: int):
        def block(sa: Shell, sb: Shell) -> list[list[float]]:
            key = (id(sa), id(sb))
            if key not in cache:
                cache[key] = multipole_block(sa, sb, c, e, q)
            return cache[key][k]
        return block

    return [matrix(basis, component(k), pairs=pairs)
            for k in range(len(cartesians(e)))]
