"""The spherical-aux packed three-center kernel (FelizJueves#743),
interpreted: `build_eri3_kernel(cls, packed=True, spherical_aux=True)`,
`speri3_{la}{lb}{lP}`, and its coefficient table `aux_harmonics(l)`.

THE CONTRACT (`kernel_eri3`'s THE SPHERICAL-AUX OUTPUT): the packed
kernel's cells and records, element `(a, b, m)` of a cell at `dst + n_aux
(a nB + b) + m`, `a`, `b` Cartesian and `m` the solid harmonic of order
`m - lP`, with `n_aux` and `dst` counting `2 lP + 1` columns per aux shell
(`eri.density_fit.eri3.spherical_offsets`).

THE WITNESS is the host's Cartesian staged block (`class_block`, itself
held to libcint by test_kernel_eri3.py) contracted by `aux_harmonics(lP)`
on the aux index. NOT BITWISE: the kernel forms only the top-degree
Hermite terms (Hobson), where the contraction forms every Cartesian row,
lower-degree terms included, and cancels them. THE BOUND, per element,
`(2 TOL_ULPS + nq + lP) eps` times the contracted scale `|c| . scale`,
`scale` the host's |.| twin of the block: the Cartesian kernel's own
`TOL_ULPS` bound on each side (the spherical kernel's terms are a subset
of `|c|`-weighted Cartesian terms, so its |.| sum is under the contracted
scale), `nq` roundings of the contraction's sum over the `nq` Cartesian
rows, and `lP` of the `(2 alpha)^{-lP}` factors folded into `w`.

SCOPE, by decision (#743): the classes (ll|l), l = 0..3, on a small
basis with diagonal and off-diagonal pairs; s and p fast, d and f `slow`
(above `FAST_L_MAX`). Every other class is built by the same code; the
consumer's full ladder (FreeBreakfast#1736) runs the rest. (ss|s) is
also bitwise the Cartesian packed kernel: `c` is 1 and every product of
the two forms is the same. The table is checked through l = 6, no
kernel. The card's run is test_eri3_spherical_device.py.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
import test_kernel_eri3 as te
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.eri.density_fit.eri3 import (
    class_block,
    eri3_classes,
    packed_cells,
    packed_columns,
    spherical_offsets,
)
from felizjueves.eri.density_fit.eri3_drive import launch_packed
from felizjueves.eri.density_fit.kernel_eri3 import (
    AUX_TABLES,
    OUT,
    PACKED_CELL_ARRAYS,
    aux_harmonics,
    build_eri3_kernel,
    eri3_name,
)
from felizjueves.ir import AtomicAdd, Decl, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

EPS = 2.0 ** -52
A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
# Two AO shells of each l <= 3 on two centres, so every (ll|.) class has two
# diagonal pairs and an off-diagonal one; contracted and single primitives.
AO = Basis((
    Shell(3, A, (0.8,), (1.0,)),
    Shell(3, B, (1.1, 0.35), (0.6, 0.5)),
    Shell(2, A, (0.9,), (1.0,)),
    Shell(2, B, (1.4, 0.4), (0.5, 0.6)),
    Shell(1, A, (1.9, 0.45), (0.38, 0.71)),
    Shell(1, B, (0.9,), (1.0,)),
    Shell(0, A, (3.4, 0.62), (0.15, 0.53)),
    Shell(0, B, (1.3,), (1.0,)),
))
# One aux shell of each l <= 3, on three centres, contracted and not.
AUX = Basis((
    Shell(0, C, (2.0, 0.4), (0.7, 0.5)),
    Shell(1, A, (0.85,), (1.0,)),
    Shell(2, B, (1.4, 0.35), (0.6, 0.45)),
    Shell(3, C, (0.7, 0.3), (0.8, 0.4)),
))
CLASSES = [(l, l, l) for l in range(4)]  # noqa: E741


def _param(c):
    p = by_l(c, sum(c), id=te._id(c))
    if max(c) <= 1:
        return p
    return pytest.param(*p.values, id=p.id, marks=(*p.marks, pytest.mark.slow))


@cache
def _classes():
    return eri3_classes(AO, AUX)


def _cls(am):
    return next(c for c in _classes() if (c.la, c.lb, c.lP) == am)


def _layout(cls):
    """The class's pairs from column 0, its cells' arrays over spherical
    aux columns, the column count and `n_aux`."""
    cols, n_col = packed_columns([cls])
    offs, n_aux = spherical_offsets(AUX)
    return packed_cells(cls, cols, n_aux, offs), cols, n_col, n_aux, offs


def _run(cls, cells, n_col, n_aux):
    return launch_packed(cls, cells, np.full((n_col, n_aux), np.nan),
                         boys=boys, threads=5, spherical_aux=True)


def _host(cls, k, cols, n_aux, offs):
    """Cell `k`'s expected block and its bound's scale, contracted onto the
    solid harmonics, `[a nB + b][m]`, and the flat output indices."""
    a_sh, b_sh, p_sh = cls.cells[k]
    blk = np.array(class_block(cls, k, pq=pq))  # [a][b][q]
    scale = np.array(te._scale(AO.shells[a_sh], AO.shells[b_sh],
                               AUX.shells[p_sh]))  # the staged block's
    c = np.array(aux_harmonics(cls.lP))  # [m][q]
    nab = blk.shape[0] * blk.shape[1]
    want = blk.reshape(nab, -1) @ c.T
    bound = scale.reshape(nab, -1) @ np.abs(c).T
    at = (cols[(a_sh, b_sh)] * n_aux + offs[p_sh]
          + n_aux * np.arange(nab)[:, None] + np.arange(2 * cls.lP + 1))
    return want, bound, at


@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_the_spherical_block_is_the_cartesian_block_contracted(am):
    """Every cell of (ll|l): the kernel's spherical block is the host's
    Cartesian block contracted by `aux_harmonics(lP)` within the bound
    (module docstring), every element of the class's cells is stored, and
    no other is."""
    cls = _cls(am)
    cells, cols, n_col, n_aux, offs = _layout(cls)
    got = _run(cls, cells, n_col, n_aux).reshape(-1)
    nq, lP = len(cartesians(cls.lP)), cls.lP
    tol = (2 * te.TOL_ULPS + nq + lP) * EPS
    stored = np.zeros(got.shape, dtype=bool)
    worst = 0.0
    for k in range(cls.n_cell):
        want, bound, at = _host(cls, k, cols, n_aux, offs)
        err = np.abs(got[at] - want)
        pos = bound > 0  # a zero scale is a structural zero: err must be 0
        worst = max(worst, float(np.max(err[pos] / (EPS * bound[pos]),
                                        initial=0.0)))
        assert np.all(err <= tol * bound), (am, cls.cells[k])
        stored[at] = True
    print(f"\n{am}: worst {worst:.2f} eps of the contracted scale, bound "
          f"{tol / EPS:.0f}")
    assert not np.isnan(got[stored]).any()
    assert np.isnan(got[~stored]).all()
    assert stored.sum() == cls.n_cell * len(cartesians(cls.la)) * len(
        cartesians(cls.lb)) * (2 * lP + 1)


def test_ss_s_is_the_cartesian_packed_kernel_bitwise():
    """At (ss|s) `c` is 1 and `w` has no `(2 alpha)` factor, so the
    spherical kernel does the Cartesian low-stack packed kernel's
    arithmetic: the same output, bitwise."""
    cls = _cls((0, 0, 0))
    cells, _, n_col, n_aux, _ = _layout(cls)  # one column per shell either way
    cart = launch_packed(cls, cells, np.full((n_col, n_aux), np.nan),
                         boys=boys, threads=5, lowstack=True)
    assert np.array_equal(_run(cls, cells, n_col, n_aux), cart,
                          equal_nan=True)


def test_the_aux_columns_are_2l_plus_1_per_shell():
    offs, n = spherical_offsets(AUX)
    assert offs == (0, 1, 4, 9) and n == 16
    cls = _cls((1, 1, 1))
    cells, cols, _, n_aux, _ = _layout(cls)
    assert tuple(cells) == PACKED_CELL_ARRAYS
    assert cells["dst"] == tuple(cols[(a, b)] * n_aux + offs[p]
                                 for a, b, p in cls.cells)


# --- the kernel ----------------------------------------------------------------


def test_the_spherical_kernel_is_named_apart_with_the_packed_signature():
    for am in ((0, 0, 0), (1, 1, 1), (0, 2, 1), (3, 3, 6), (0, 0, 6)):
        k = build_eri3_kernel(am, packed=True, spherical_aux=True)
        pk = build_eri3_kernel(am, packed=True, lowstack=True)
        assert k.name == eri3_name(am, spherical_aux=True) == (
            "speri3_" + "".join(map(str, am)))
        assert [p.name for p in k.params] == [p.name for p in pk.params]
        assert k.params[-1].name == OUT
        assert not any(isinstance(s, AtomicAdd) for s in walk(k.body))
        # no single-center tables: the aux record is the literal table
        assert not {s.name for s in walk(k.body) if isinstance(s, Decl)} & set(
            AUX_TABLES)


def test_the_mode_is_packed_and_low_stack_only():
    with pytest.raises(ValueError, match="packed"):
        build_eri3_kernel((1, 1, 1), spherical_aux=True)
    with pytest.raises(ValueError, match="low-stack"):
        build_eri3_kernel((1, 1, 1), packed=True, lowstack=False,
                          spherical_aux=True)
    # lowstack None and True are the same kernel; the default is unchanged
    assert build_eri3_kernel((1, 1, 1), packed=True, spherical_aux=True) == (
        build_eri3_kernel((1, 1, 1), packed=True, lowstack=True,
                          spherical_aux=True))
    assert build_eri3_kernel((1, 1, 1), packed=True) == build_eri3_kernel(
        (1, 1, 1), packed=True, spherical_aux=False)


# --- the table, through l = 6 ---------------------------------------------------


def _gram(p, q):
    s = [x + y for x, y in zip(p, q, strict=True)]
    if any(x % 2 for x in s):
        return 0
    return math.prod(math.prod(range(x - 1, 0, -2)) for x in s)


def _laplacian(col, l):  # noqa: E741
    """The coefficients of `nabla^2 SUM_tau c_tau x^tau`, by monomial."""
    out: dict[tuple[int, int, int], float] = {}
    for tau, c in zip(cartesians(l), col, strict=True):
        for d in range(3):
            if tau[d] >= 2:
                r = tuple(tau[e] - 2 * (e == d) for e in range(3))
                out[r] = out.get(r, 0.0) + c * tau[d] * (tau[d] - 1)
    return out


@pytest.mark.parametrize("l", range(7))
def test_the_table_is_the_harmonics_in_the_consumers_normalization(l):  # noqa: E741
    """`2l + 1` columns over `cartesians(l)`, each a harmonic polynomial
    (`nabla^2 = 0`), mutually orthogonal and of norm `(2l-1)!!` under the
    Gaussian-moment product: `c G c^T = (2l-1)!! I` -- FreeBreakfast's
    block `B sqrt((2l-1)!!)` with `B^T G B = I`."""
    carts = cartesians(l)
    c = np.array(aux_harmonics(l))
    assert c.shape == (2 * l + 1, len(carts))
    for col in c:
        lap = _laplacian(col, l)
        assert all(abs(v) <= 1e-13 * np.max(np.abs(col)) for v in lap.values())
    G = np.array([[_gram(p, q) for q in carts] for p in carts], dtype=float)
    norm = math.prod(range(2 * l - 1, 0, -2))
    assert np.allclose(c @ G @ c.T, norm * np.eye(2 * l + 1), rtol=0,
                       atol=1e-12 * norm)


@pytest.mark.parametrize("l", range(7))
def test_the_columns_run_m_minus_l_to_l_with_the_real_harmonic_signs(l):  # noqa: E741
    """Column `m + l`'s defining monomial and its sign: `x^m z^{l-m}`,
    positive, for `m >= 0` (the cosine branch); `x^{|m|-1} y z^{l-|m|}`,
    positive, for `m < 0` (the sine branch); and the other branch's
    monomial absent. s and p are the identity (`p` as x, y, z)."""
    carts = cartesians(l)
    c = aux_harmonics(l)
    if l <= 1:
        assert c == tuple(tuple(float(i == j) for i in range(len(carts)))
                          for j in range(len(carts)))
        return
    for m in range(-l, l + 1):
        col = c[m + l]
        ma = abs(m)
        cos, sin = (ma, 0, l - ma), (ma - 1, 1, l - ma) if ma else None
        if m >= 0:
            assert col[carts.index(cos)] > 0, m
            assert sin is None or col[carts.index(sin)] == 0.0, m
        else:
            assert col[carts.index(sin)] > 0, m
            assert col[carts.index(cos)] == 0.0, m


def test_the_d_table_is_pinned():
    """The d columns in closed form: sqrt(3) xy, sqrt(3) yz, z^2 - (x^2 +
    y^2)/2, sqrt(3) xz, sqrt(3)/2 (x^2 - y^2), over xx xy xz yy yz zz."""
    r3 = math.sqrt(3.0)
    want = ((0, r3, 0, 0, 0, 0), (0, 0, 0, 0, r3, 0), (-0.5, 0, 0, -0.5, 0, 1),
            (0, 0, r3, 0, 0, 0), (r3 / 2, 0, 0, -r3 / 2, 0, 0))
    assert np.allclose(aux_harmonics(2), want, rtol=0, atol=1e-15)
