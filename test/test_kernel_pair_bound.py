"""kernel_pair_bound.py: the record bound `Ip` / `Iq`, interpreted.

THE WITNESS is `ao.pairs.pair_bound(zeta, L, one_centre=reference.pq.
one_centre)`: the host runs the Hermite recurrence at `PQ = 0` and sums the
`I_p` with `fsum`; the kernel evaluates the closed form the module docstring
derives. They share no code, so agreement is a check of the closed form, not
of a copy. They agree to rounding, not bitwise: the bar is `16 eps` of the
bound, per record, over exponents from 1e-3 to 1e5 (the module docstring,
"ROUND-OFF, NOT BITWISE"). The bound itself bounds `[p|q]` by
ao/pairs.py's proof, which test_ao_pairs.py holds; it is not re-proved here.

Every class `la <= lb <= 4` is built and interpreted, `L = la + lb` above
`tiers.FAST_L_MAX` in the slow tier.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from tiers import split_by_l

from felizjueves.ao.pairs import pair_bound
from felizjueves.eri import kernel_pair_bound as kpb
from felizjueves.eri.kernel_pair_bound import (
    build_pair_bound_kernel,
    pair_bound_coefficients,
    pair_bound_name,
    pair_bound_spec,
)
from felizjueves.ir import Call, Fn, walk
from felizjueves.ir.interp import Launch, interpret
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Operator, Spec, Target

LMAX = 4
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(la, LMAX + 1)]
CLASS_GRID = split_by_l(CLASSES, sum, ids=lambda c: f"{c[0]}{c[1]}")

# Tight to diffuse, spanning what a basis reaches, and a few in between.
ZETAS = np.array([1e-3, 0.013, 0.07, 0.25, 0.6, 1.0, 1.9, 3.0, 7.5, 21.0,
                  47.0, 130.0, 1e3, 6.4e3, 1e5])
REL_TOL = 16 * np.finfo(np.float64).eps


def _interpret(cls: tuple[int, int], zetas: np.ndarray,
               threads: int = 8) -> np.ndarray:
    n = len(zetas)
    grid, block = kpb.launch_shape(n, threads)
    out = interpret(build_pair_bound_kernel(cls),
                    launch=Launch(grid, block, threads), n_rec=n,
                    zeta=zetas.copy(), bound=np.full(n, np.nan))
    return out["bound"]


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_the_key_is_a_pair_class() -> None:
    assert pair_bound_spec(2, 1) == Spec(Operator.COULOMB, 0, (2, 1),
                                         Target.PAIR_BOUND)
    assert pair_bound_spec(2, 1).key() == "coulomb/d0/(2,1)/b1/pair_bound/literal"
    assert build_pair_bound_kernel(pair_bound_spec(2, 1)) == \
        build_pair_bound_kernel((2, 1))
    assert build_pair_bound_kernel((2, 1)).name == "pair_bound_21" == \
        pair_bound_name((2, 1))
    assert [p.name for p in build_pair_bound_kernel((0, 0)).params] == [
        "n_rec", "zeta", "bound"]


def test_what_the_key_refuses() -> None:
    with pytest.raises(ValueError, match="not a Target.PAIR_BOUND"):
        build_pair_bound_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.DQ))
    with pytest.raises(ValueError, match="never batched"):
        Spec(Operator.COULOMB, 0, (1, 1), Target.PAIR_BOUND, (), 3)
    with pytest.raises(ValueError):
        Spec(Operator.OVERLAP, 0, (1, 1), Target.PAIR_BOUND)
    with pytest.raises(TypeError):
        build_pair_bound_kernel(4)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------


def test_the_s_coefficient_is_the_bare_constant() -> None:
    # p = 0 only: c = 1, so D_0 = (2/sqrt(pi))^{1/2} 2^{-1/4} and the (ss)
    # bound is D_0 zeta^{1/4}.
    (d0,) = pair_bound_coefficients(0)
    assert d0 == math.sqrt(2.0 / math.sqrt(math.pi)) * 2.0 ** -0.25
    for z in (0.5, 2.0, 40.0):
        host = pair_bound(z, 0, one_centre=one_centre)
        assert abs(d0 * z ** 0.25 - host) <= REL_TOL * host


def test_every_coefficient_is_positive_and_one_per_order() -> None:
    for L in range(17):
        coeffs = pair_bound_coefficients(L)
        assert len(coeffs) == L + 1
        assert all(c > 0.0 for c in coeffs)
    with pytest.raises(ValueError):
        pair_bound_coefficients(-1)


def test_the_kernel_is_two_roots_and_a_fold() -> None:
    k = build_pair_bound_kernel((2, 2))
    calls = [e.fn for s in walk(k.body) for e in _calls(s)]
    assert calls.count(Fn.SQRT) == 2
    assert Fn.EXP not in calls  # no Boys, no E, no prefactor


def _calls(stmt):
    from felizjueves.ir import _exprs_of, _subexprs
    for root in _exprs_of(stmt):
        if root is None:
            continue
        for e in _subexprs(root):
            if isinstance(e, Call):
                yield e


def test_the_text_depends_on_the_hermite_order_only() -> None:
    assert build_pair_bound_kernel((0, 2), name="b") == \
        build_pair_bound_kernel((1, 1), name="b")
    assert build_pair_bound_kernel((0, 2), name="b") != \
        build_pair_bound_kernel((1, 2), name="b")


# ---------------------------------------------------------------------------
# Interpreted against the host: every class, every exponent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASS_GRID)
def test_the_kernel_is_the_hosts_pair_bound(cls: tuple[int, int]) -> None:
    L = sum(cls)
    got = _interpret(cls, ZETAS)
    want = np.array([pair_bound(float(z), L, one_centre=one_centre)
                     for z in ZETAS])
    assert np.all(want > 0.0)
    rel = np.abs(got - want) / want
    assert float(rel.max()) <= REL_TOL, (cls, float(rel.max()))


def test_the_lane_mapping_is_not_numerical() -> None:
    zetas = np.geomspace(1e-2, 1e4, 37)
    one = _interpret((1, 2), zetas, threads=1)
    for threads in (4, 32):
        assert _interpret((1, 2), zetas, threads=threads).tolist() == one.tolist()


def test_the_guard_leaves_spare_lanes_idle() -> None:
    # 3 records on a block of 32: the 29 spare threads write nothing past
    # n_rec (the interpreter bounds-checks every write).
    got = _interpret((0, 1), np.array([0.3, 1.0, 9.0]), threads=32)
    assert not np.isnan(got).any()
