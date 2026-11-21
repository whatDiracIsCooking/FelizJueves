"""The fully spherical packed three-center kernel (FelizJueves#745),
interpreted: `build_eri3_kernel(cls, packed=True, spherical_aux=True,
spherical_bra=True)`, `ssperi3_{la}{lb}{lP}`.

THE CONTRACT (`kernel_eri3`'s THE FULLY SPHERICAL OUTPUT): the packed
kernel's cells and records, element `(mu, nu, m)` of a cell at `dst +
n_aux (mu (2 lb + 1) + nu) + m`, all three real solid harmonics in
`aux_harmonics`' convention, `n_col` counting `(2 la + 1)(2 lb + 1)`
columns per pair (`packed_columns(..., spherical_bra=True)`) and `n_aux`
and `dst` `2 lP + 1` per aux shell (`spherical_offsets`).

THE WITNESS is the host's Cartesian staged block (`class_block`, held to
libcint by test_kernel_eri3.py) contracted by `aux_harmonics` on all three
indices. NOT BITWISE. THE BOUND, per element, by analysis:

    (2 TOL_ULPS + nq + lP + 2 (nA + nB) + 1) eps  x  (|c_a| |c_b| |c_P|) . scale

`scale` the host's |.| twin of the Cartesian block. Of it, `TOL_ULPS + nq +
lP` is test_kernel_eri3_spherical.py's bound (the Cartesian kernel's own
`TOL_ULPS` on each side, the host's `nq`-term aux contraction, the `lP`
factors of `(2 alpha)^{-lP}` in `w`); the bra adds, per primitive triple,
the kernel's two transform sums (`nA` terms into `T`, `nB` into the
element) and its `w S` product, and the host's own two bra contractions
(`nA`, `nB` terms). Each is a sum of `|c|`-weighted terms whose |.| sum is
under the contracted scale.

SCOPE, by decision (#745, on a GH200 VM with no AMD card): the classes
(aa|p), a = 0..3, p = 0..6, on a two-shell AO basis per l (two diagonal
pairs and an off-diagonal one) and two aux shells; L <= 4 fast, the rest
`slow`. At a <= 1 the bra transform is the identity and the kernel IS
`speri3_` (its body unchanged, one note added), so those classes are
bitwise `speri3_`'s output too. The card's run is
test_eri3_fully_spherical_device.py.
"""

from __future__ import annotations

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
    OUT,
    T_ARRAY,
    V_ARRAY,
    aux_harmonics,
    build_eri3_kernel,
    eri3_name,
)
from felizjueves.ir import AtomicAdd, Decl, Lit, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

EPS = 2.0 ** -52
A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
CLASSES = [(a, a, p) for a in range(4) for p in range(7)]


def _id(c: tuple[int, int, int]) -> str:
    return f"{'spdfghi'[c[0]]}{'spdfghi'[c[1]]}|{'spdfghi'[c[2]]}"


def _param(c):
    """`by_l`'s marks, and `slow` for any d or f bra (the d and f classes
    of test_kernel_eri3_spherical.py's rule)."""
    p = by_l(c, sum(c), id=_id(c))
    if c[0] <= 1:
        return p
    return pytest.param(*p.values, id=p.id, marks=(*p.marks, pytest.mark.slow))


@cache
def _bases(la: int, lP: int) -> tuple[Basis, Basis]:
    """Two AO shells of `la` on two centres (one contracted), so the class
    has two diagonal pairs and an off-diagonal one; two aux shells of `lP`,
    contracted and not, on two centres."""
    ao = Basis((Shell(la, A, (0.8,), (1.0,)),
                Shell(la, B, (1.1, 0.35), (0.6, 0.5))))
    aux = Basis((Shell(lP, C, (0.7, 0.3), (0.8, 0.4)),
                 Shell(lP, A, (1.2,), (1.0,))))
    return ao, aux


@cache
def _cls(am):
    ao, aux = _bases(am[0], am[2])
    (cls,) = eri3_classes(ao, aux)
    assert (cls.la, cls.lb, cls.lP) == am
    return cls


def _layout(am):
    """The class's pairs from column 0 in spherical bra functions, its
    cells' arrays over spherical aux columns, the column count, `n_aux`
    and the aux offsets."""
    cls = _cls(am)
    _, aux = _bases(am[0], am[2])
    cols, n_col = packed_columns([cls], spherical_bra=True)
    offs, n_aux = spherical_offsets(aux)
    return packed_cells(cls, cols, n_aux, offs), cols, n_col, n_aux, offs


def _run(am, spherical_bra=True):
    cls = _cls(am)
    cells, _, n_col, n_aux, _ = _layout(am)
    return launch_packed(cls, cells, np.full((n_col, n_aux), np.nan),
                         boys=boys, threads=5, spherical_aux=True,
                         spherical_bra=spherical_bra)


def tolerance(am) -> float:
    """The module docstring's bound, in units of the contracted scale."""
    la, lb, lP = am
    nA, nB, nq = (len(cartesians(x)) for x in am)
    return (2 * te.TOL_ULPS + nq + lP + 2 * (nA + nB) + 1) * EPS


def host(am, k, cols, n_aux, offs):
    """Cell `k`'s expected block, `[mu][nu][m]`, its bound's scale, and the
    flat output indices of its elements."""
    cls = _cls(am)
    ao, aux = _bases(am[0], am[2])
    a_sh, b_sh, p_sh = cls.cells[k]
    blk = np.array(class_block(cls, k, pq=pq))  # [a][b][q]
    scale = np.array(te._scale(ao.shells[a_sh], ao.shells[b_sh],
                               aux.shells[p_sh]))
    ca, cb, cp = (np.array(aux_harmonics(x)) for x in am)
    want = np.einsum("ia,jb,mq,abq->ijm", ca, cb, cp, blk)
    bound = np.einsum("ia,jb,mq,abq->ijm", np.abs(ca), np.abs(cb),
                      np.abs(cp), scale)
    sA, sB, sP = (2 * x + 1 for x in am)
    at = (cols[(a_sh, b_sh)] * n_aux + offs[p_sh]
          + n_aux * np.arange(sA * sB).reshape(sA, sB)[:, :, None]
          + np.arange(sP))
    return want, bound, at


def check(am, got) -> float:
    """Every cell within the bound, a diagonal pair's block symmetric in
    `(mu, nu)` exactly, every element of the class's cells stored and no
    other: the worst error in eps of the contracted scale."""
    cls = _cls(am)
    _, cols, _, n_aux, offs = _layout(am)
    got = got.reshape(-1)
    tol = tolerance(am)
    stored = np.zeros(got.shape, dtype=bool)
    worst = 0.0
    for k in range(cls.n_cell):
        want, bound, at = host(am, k, cols, n_aux, offs)
        g = got[at]
        err = np.abs(g - want)
        pos = bound > 0  # a zero scale is a structural zero: err must be 0
        worst = max(worst, float(np.max(err[pos] / (EPS * bound[pos]),
                                        initial=0.0)))
        assert np.all(err <= tol * bound), (am, cls.cells[k])
        a_sh, b_sh, _ = cls.cells[k]
        if a_sh == b_sh:
            assert np.array_equal(g, g.transpose(1, 0, 2)), (am, k)
        assert not stored[at].any()  # no two cells share an element
        stored[at] = True
    assert not np.isnan(got[stored]).any()
    assert np.isnan(got[~stored]).all()
    la, lb, lP = am
    assert stored.sum() == cls.n_cell * (2 * la + 1) * (2 * lb + 1) * (
        2 * lP + 1)
    return worst


@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_the_block_is_the_cartesian_block_contracted_on_all_three(am):
    """Every cell of (aa|p): within the bound, a diagonal pair symmetric,
    exactly the class's `(2la+1)(2lb+1)(2lP+1)` elements per cell stored
    (the output starts NaN)."""
    worst = check(am, _run(am))
    print(f"\n{am}: worst {worst:.2f} eps of the contracted scale, bound "
          f"{tolerance(am) / EPS:.0f}")


@pytest.mark.parametrize("am", [_param(c) for c in CLASSES if c[0] <= 1])
def test_s_and_p_bras_are_the_spherical_aux_kernel_bitwise(am):
    """At a <= 1 the bra transform is the identity and `2l + 1` is the
    Cartesian count, so the layout and every operation are `speri3_`'s:
    the same output, bitwise."""
    assert np.array_equal(_run(am), _run(am, spherical_bra=False),
                          equal_nan=True)


# --- the kernel ---------------------------------------------------------------


def _decls(k):
    return {s.name: s for s in walk(k.body) if isinstance(s, Decl)}


def test_the_kernel_is_named_apart_with_the_packed_signature():
    for am in ((0, 0, 0), (1, 1, 1), (0, 2, 1), (2, 3, 4), (3, 3, 6)):
        k = build_eri3_kernel(am, packed=True, spherical_aux=True,
                              spherical_bra=True)
        sk = build_eri3_kernel(am, packed=True, spherical_aux=True)
        assert k.name == eri3_name(am, spherical_bra=True) == (
            "ssperi3_" + "".join(map(str, am)))
        assert k.params == sk.params and k.params[-1].name == OUT
        assert not any(isinstance(s, AtomicAdd) for s in walk(k.body))


@pytest.mark.parametrize("am", [(0, 0, 3), (0, 1, 2), (1, 1, 6)])
def test_an_s_or_p_bra_is_the_spherical_aux_body_unchanged(am):
    """No local block and no transform below d: the body is `speri3_`'s,
    after the one leading note."""
    k = build_eri3_kernel(am, packed=True, spherical_aux=True,
                          spherical_bra=True)
    sk = build_eri3_kernel(am, packed=True, spherical_aux=True)
    assert k.body[1:] == sk.body
    assert not {V_ARRAY, T_ARRAY} & set(_decls(k))


@pytest.mark.parametrize("am", [(0, 2, 0), (1, 2, 3), (2, 2, 2), (3, 3, 6)])
def test_the_added_stack_is_one_cartesian_block_and_one_row(am):
    """`Vab[nA nB]` and `Tmu[nB]`, independent of `lP`: the issue's
    `nA nB + nB` doubles (880 B at (ff|.))."""
    la, lb, _ = am
    nA, nB = len(cartesians(la)), len(cartesians(lb))
    d = _decls(build_eri3_kernel(am, packed=True, spherical_aux=True,
                                 spherical_bra=True))
    assert d[V_ARRAY].shape == (Lit(nA * nB),)
    assert d[T_ARRAY].shape == (Lit(nB),)


def test_the_mode_needs_the_spherical_aux_output():
    with pytest.raises(ValueError, match="spherical_aux"):
        build_eri3_kernel((2, 2, 2), packed=True, spherical_bra=True)
    with pytest.raises(ValueError, match="packed"):
        build_eri3_kernel((2, 2, 2), spherical_aux=True, spherical_bra=True)
    with pytest.raises(ValueError, match="low-stack"):
        build_eri3_kernel((2, 2, 2), packed=True, lowstack=False,
                          spherical_aux=True, spherical_bra=True)


def test_the_columns_count_spherical_bra_functions():
    """`(2la+1)(2lb+1)` columns a pair under `spherical_bra`; the Cartesian
    count is unchanged without it."""
    cls = _cls((2, 2, 1))
    cols, n = packed_columns([cls], spherical_bra=True)
    assert n == 3 * 25 and sorted(cols.values()) == [0, 25, 50]
    _, n_cart = packed_columns([cls])
    assert n_cart == 3 * 36
