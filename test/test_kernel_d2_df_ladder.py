"""The DF Hessian kernels through (ff|i) and (i|i) (#694), interpreted.

THE LADDER is the d1 DF ladder's (#613), one derivative order up:
`d2eri3` `la <= lb <= 3`, `lP <= 6` (70 classes) and `d2metric` `lP <= lQ
<= 6` (28). test_kernel_d2eri3.py and test_kernel_d2metric.py hold every
class at `L <= 4` on their own bases; this file holds every class of the
ladder.

THE GATE. Both builders have a LOW-STACK form (`kernel_d2eri3` /
`kernel_d2metric`, THE LOW-STACK FORM) picked from an `L` floor set from
measured rows (data/measured/d2_df_lowstack.json, `devtools/measure.sh
d2-df-lowstack-compile`, its rule declared before any row, `sm_86` and
`gfx1200` rows): `d2eri3` from `kernel_d2eri3.LOWSTACK_MIN_L` = 3 (the
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
from felizjueves.ir import Decl, Space, walk
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


def test_the_compact_tables_and_their_refusal():
    """`compact=True` (#1263) stores the low-stack form's bra tables
    compact, `t <= i + j` only: 216 doubles a side at (ff|.) against
    396; the name, parameters and every other array unchanged. The
    unrolled form has no tables to pack and refuses it."""
    from felizjueves.gradient.kernel_gk import compact_extent

    assert compact_extent(5, 5) == 216
    dense = ke.build_d2eri3_kernel((3, 3, 6))
    packed = ke.build_d2eri3_kernel((3, 3, 6), compact=True)
    assert packed.name == dense.name and packed.params == dense.params
    assert _array_doubles(dense) - _array_doubles(packed) == 3 * (396 - 216)
    with pytest.raises(ValueError, match="low-stack form's alone"):
        ke.build_d2eri3_kernel((0, 1, 1), lowstack=False, compact=True)


def _bitwise_compact(cls):
    basis, aux = _eri3_bases(cls)
    Gam = np.random.default_rng(5).standard_normal(
        (basis.n_ao, basis.n_ao, aux.n_ao))
    dense, _ = run_d2eri3(basis, aux, NUCLEI, Gam, boys=boys, classes={cls},
                          lowstack=True)
    packed, ran = run_d2eri3(basis, aux, NUCLEI, Gam, boys=boys,
                             classes={cls}, lowstack=True, compact=True)
    assert ran == (cls,)
    assert np.array_equal(dense, packed), ke.d2eri3_name(cls)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=_eid(c))
                                 for c in ERI3_LADDER])
def test_the_compact_low_stack_d2eri3_is_the_dense_one_bitwise(cls):
    """Compact tables move only where each entry lives, never a value or
    the order of a sum, so the two low-stack forms agree BITWISE (#1263)."""
    _bitwise_compact(cls)


def _local_doubles(k) -> int:
    n = 0
    for s in walk(k.body):
        if isinstance(s, Decl) and s.shape and s.space is Space.LOCAL:
            size = 1
            for e in s.shape:
                size *= e.value
            n += size
    return n


def test_the_warp_shared_tower_and_its_refusal():
    """`shared=True` (#1265) moves the bra tables and the tower `R` into
    shared memory, formed once per record pair: at (ff|i) a lane keeps
    `U`, the six operated factors, the aux tables and the Boys array,
    1,320 + 1,296 + 1,176 + 120 B; the name and parameters unchanged. The
    unrolled form refuses it."""
    i6 = ef.center_table_extent(6)
    packed = ke.build_d2eri3_kernel((3, 3, 6), compact=True)
    warp = ke.build_d2eri3_kernel((3, 3, 6), compact=True, shared=True)
    assert warp.name == packed.name and warp.params == packed.params
    assert _array_doubles(warp) == _array_doubles(packed)
    assert _local_doubles(warp) == 165 + 6 * 27 + 3 * i6 + 15 == 489
    with pytest.raises(ValueError, match="low-stack form's alone"):
        ke.build_d2eri3_kernel((0, 1, 1), lowstack=False, shared=True)


def _contracted_bases(cls):
    """`_eri3_bases` with two- and three-primitive shells: several bra and
    aux records a cell, so the lanes' places in a pair's items move with
    the pair (`kernel_d2eri3._shared_pairs`)."""
    la, lb, lP = cls
    sh = [Shell(lb, B, (1.1, 0.4), (0.6, 0.5)),
          Shell(la, A, (0.8, 0.3, 2.0), (1.0, 0.4, 0.2))]
    return Basis(tuple(sh)), Basis((Shell(lP, C, (1.6, 0.5), (1.0, 0.7)),))


def _long_aux_bases(cls):
    """`_contracted_bases` with a five-primitive aux shell: at (pp|d) a
    phase holds four aux records (`kernel_d2eri3.shared_slots`), so a
    bra record's last phase is partial."""
    basis, _ = _contracted_bases(cls)
    exps = (2.4, 1.6, 0.9, 0.5, 0.25)
    return basis, Basis((Shell(cls[2], C, exps, (0.3, 1.0, 0.7, 0.5, 0.2)),))


def _bitwise_shared(cls, bases, lanes: int = 4):
    basis, aux = bases(cls)
    Gam = np.random.default_rng(5).standard_normal(
        (basis.n_ao, basis.n_ao, aux.n_ao))
    packed, _ = run_d2eri3(basis, aux, NUCLEI, Gam, boys=boys, classes={cls},
                           lowstack=True, compact=True, lanes=lanes)
    warp, ran = run_d2eri3(basis, aux, NUCLEI, Gam, boys=boys, classes={cls},
                           lowstack=True, compact=True, shared=True,
                           lanes=lanes)
    assert ran == (cls,)
    assert np.array_equal(packed, warp), ke.d2eri3_name(cls)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=_eid(c))
                                 for c in ERI3_LADDER])
def test_the_warp_shared_d2eri3_is_the_compact_one_bitwise(cls):
    """The warp-shared tower forms the same tables, tower and `U` by the
    same statements, and each lane adds the same items in the same order
    as #632's per-item mapping, so the two agree BITWISE (#1265)."""
    _bitwise_shared(cls, _eri3_bases)


@pytest.mark.parametrize("case", [
    by_l(((0, 1, 1), n), 4, id=f"d2eri3_011-contracted-{n}lanes")
    for n in (2, 4, 32)] + [
    by_l(((1, 1, 0), 2), 4, id="d2eri3_110-contracted-2lanes"),
    by_l(((0, 1, 2), 4), 5, id="d2eri3_012-contracted-4lanes"),
    by_l(((1, 2, 0), 8), 5, id="d2eri3_120-contracted-8lanes"),
    *(by_l(((1, 1, 2), n, True), 6, id=f"d2eri3_112-partial-phase-{n}lanes")
      for n in (4, 32))])
def test_the_warp_shared_d2eri3_is_bitwise_over_many_records(case):
    """Several bra and aux records a cell, at warp widths that do and do
    not divide a phase's items, and a partial last phase: still BITWISE
    the compact form (#1265)."""
    cls, lanes, *long = case
    _bitwise_shared(cls, _long_aux_bases if long else _contracted_bases,
                    lanes)


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
