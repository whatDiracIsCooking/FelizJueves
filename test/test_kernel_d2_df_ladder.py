"""The DF Hessian kernels through (ff|i) and (i|i) (#694), interpreted.

THE LADDER is the d1 DF ladder's (#613), one derivative order up:
`d2eri3` `la <= lb <= 3`, `lP <= 6` (70 classes) and `d2metric` `lP <= lQ
<= 6` (28). test_kernel_d2eri3.py and test_kernel_d2metric.py hold every
class at `L <= 4` on their own bases; this file holds every class of the
ladder.

THE GATE. Both builders have a LOW-STACK form (`kernel_d2eri3` /
`kernel_d2metric`, THE LOW-STACK FORM) picked from an `L` floor set from
measured rows (data/measured/d2_df_lowstack.json, `devtools/measure.sh
d2-df-lowstack-compile`, its rule declared before any row; `sm_86` rows
only so far): `d2eri3` from `kernel_d2eri3.LOWSTACK_MIN_L` = 3 (the
unrolled (sp|d) is 14,768 B of stack), `d2metric` from
`kernel_d2metric.LOWSTACK_MIN_L` = 9 (the unrolled (d|i) is 19,216 B).

THE WITNESS is the host, `hessian.host_eri3.eri3_hessian` and
`hessian.host_df.metric_hessian` (another route on every centre, each held
to finite differences of the d1 host), on a small basis that reaches the
class with a swapped pair, and for `la == lb` (`lP == lQ`) a diagonal and an
off-diagonal one: `|H - H_host| <= 16 eps` of the host's `absolute=True`
twin, test_kernel_d2eri3.py's bound. `max|H|` is not a scale here: the
Hermite tower cancels by orders of magnitude at high `L` (the (i|i)
kernel and host agree to 0.05 eps of the twin and differ by ~7,000 eps of
`max|H|`; the forced unrolled and low-stack forms differ from the host by
the same amount), and both routes round that cancellation. Every class of
the ladder in its default form, every class below the floor in the
low-stack form FORCED and two above it in the unrolled form forced, so
each form is held to the host where it builds. `L` for the tier cut is
`sum(class) + 2`, the Hermite order the Hessian reaches: above `FAST_L_MAX`
everything is `slow` (an interpreted low-stack (ff|i) class is minutes).
"""

from __future__ import annotations

import numpy as np
import pytest
from df_grad_launch import run_d2eri3, run_d2metric
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.hessian import host_df, host_eri3
from felizjueves.hessian import kernel_d2eri3 as ke
from felizjueves.hessian import kernel_d2metric as km
from felizjueves.ir import Decl, walk
from felizjueves.mmd import e_fragment as ef
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(twin): the module docstring
ERI3_LADDER = [(a, b, p) for a in range(4) for b in range(a, 4)
               for p in range(7)]
METRIC_LADDER = [(p, q) for p in range(7) for q in range(p, 7)]

A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
NUCLEI = ((8.0, A), (1.0, B), (1.0, C))


def _eid(c) -> str:
    return ke.d2eri3_name(c)


def _mid(c) -> str:
    return km.d2metric_name(c)


# --- the gate ------------------------------------------------------------------


def test_the_gate_is_an_l_floor():
    assert ke.LOWSTACK_MIN_L == 3 and km.LOWSTACK_MIN_L == 9
    for c in ERI3_LADDER:
        assert ke.d2eri3_lowstack(c) == (sum(c) >= ke.LOWSTACK_MIN_L), c
    for c in METRIC_LADDER:
        assert km.d2metric_lowstack(c) == (sum(c) >= km.LOWSTACK_MIN_L), c


def test_the_forms_share_name_parameters_and_launch():
    for cls in ((0, 1, 1), (1, 2, 3)):
        a = ke.build_d2eri3_kernel(cls, lowstack=True)
        b = ke.build_d2eri3_kernel(cls, lowstack=False)
        assert a.name == b.name and a.params == b.params
    for cls in ((1, 2), (3, 3)):
        a = km.build_d2metric_kernel(cls, lowstack=True)
        b = km.build_d2metric_kernel(cls, lowstack=False)
        assert a.name == b.name and a.params == b.params


def _array_doubles(k) -> int:
    n = 0
    for s in walk(k.body):
        if isinstance(s, Decl) and s.shape:
            size = 1
            for e in s.shape:
                size *= e.value
            n += size
    return n


def test_the_low_stack_arrays_at_the_top_of_each_ladder():
    # d2eri3 (ff|i): R n_t(14) = 680, U n_t(8) = 165, six operated factors
    # 3 x 9 each, the bra tables (5, 5) 396 a side, the aux tables 3 x 49,
    # the Boys array 15: 2,357 doubles, 18.9 KB
    i6 = ef.center_table_extent(6)
    assert i6 == 49 and ef.axis_table_extent(5, 5) == 396
    assert _array_doubles(ke.build_d2eri3_kernel((3, 3, 6))) == (
        680 + 165 + 6 * 27 + 3 * 396 + 3 * i6 + 15) == 2357
    # d2metric (i|i): R, U, both records' tables 3 x 49, the Boys array
    assert _array_doubles(km.build_d2metric_kernel((6, 6))) == (
        680 + 165 + 6 * i6 + 15) == 1154


# --- the host, every class of the ladder ----------------------------------------


def _eri3_bases(cls):
    """`la` at A, `lb` at B, the higher listed first (so `la < lb` is
    staged swapped); `la == lb`: two shells, so a diagonal and an
    off-diagonal pair; the aux shell at C. One primitive each."""
    la, lb, lP = cls
    sh = [Shell(lb, B, (1.1,), (1.0,)), Shell(la, A, (0.8,), (1.0,))]
    return Basis(tuple(sh)), Basis((Shell(lP, C, (1.6,), (1.0,)),))


def _metric_basis(cls):
    lP, lQ = cls
    return Basis((Shell(lQ, B, (1.3,), (1.0,)), Shell(lP, A, (0.9,), (1.0,))))


def _close(H, want, twin, what: str) -> None:
    scale = float(np.abs(twin).max())
    err = float(np.abs(H - want).max())
    print(f"\n{what}: {err / scale / EPS:.3f} eps of the twin")
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps"
    assert np.abs(H.sum(axis=0)).max() <= TOL * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= TOL * scale


def _check_eri3(cls, lowstack):
    basis, aux = _eri3_bases(cls)
    Gam = np.random.default_rng(5).standard_normal(
        (basis.n_ao, basis.n_ao, aux.n_ao))
    H, ran = run_d2eri3(basis, aux, NUCLEI, Gam, boys=boys, classes={cls},
                        lowstack=lowstack)
    assert ran == (cls,)
    sh = basis.shells
    cells = [(i, j, 0) for i in range(len(sh)) for j in range(i, len(sh))
             if tuple(sorted((sh[i].l, sh[j].l))) == cls[:2]]
    want, twin = (np.array(host_eri3.eri3_hessian(
        basis, aux, Gam, NUCLEI, pq=pq, cells=cells, absolute=a))
        for a in (False, True))
    _close(H, want, twin, f"{ke.d2eri3_name(cls)} lowstack={lowstack}")


def _check_metric(cls, lowstack):
    aux = _metric_basis(cls)
    W = np.random.default_rng(5).standard_normal((aux.n_ao, aux.n_ao))
    H, ran = run_d2metric(aux, NUCLEI, W, boys=boys, classes={cls},
                          lowstack=lowstack)
    assert ran == (cls,)
    pairs = [(0, 1)] if cls[0] != cls[1] else [(0, 0), (0, 1), (1, 1)]
    want, twin = (np.array(host_df.metric_hessian(
        aux, W, NUCLEI, pq=pq, pairs=pairs, absolute=a))
        for a in (False, True))
    _close(H, want, twin, f"{km.d2metric_name(cls)} lowstack={lowstack}")


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=_eid(c))
                                 for c in ERI3_LADDER])
def test_every_d2eri3_class_is_the_host_hessian(cls):
    _check_eri3(cls, None)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=_mid(c))
                                 for c in METRIC_LADDER])
def test_every_d2metric_class_is_the_host_hessian(cls):
    _check_metric(cls, None)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=_eid(c))
                                 for c in ERI3_LADDER
                                 if sum(c) < ke.LOWSTACK_MIN_L])
def test_the_forced_low_stack_d2eri3_is_the_host_hessian(cls):
    _check_eri3(cls, True)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=_mid(c))
                                 for c in METRIC_LADDER
                                 if sum(c) < km.LOWSTACK_MIN_L])
def test_the_forced_low_stack_d2metric_is_the_host_hessian(cls):
    _check_metric(cls, True)


@pytest.mark.slow
@pytest.mark.parametrize("cls", [(0, 1, 2), (1, 1, 1)], ids=_eid)
def test_the_unrolled_d2eri3_above_the_floor_is_the_host_hessian(cls):
    """Forced unrolled above the floor (where it is not kept, for its
    stack), so both forms are held to the host where both build."""
    _check_eri3(cls, False)


@pytest.mark.slow
@pytest.mark.parametrize("cls", [(3, 6), (4, 5)], ids=_mid)
def test_the_unrolled_d2metric_above_the_floor_is_the_host_hessian(cls):
    _check_metric(cls, False)
