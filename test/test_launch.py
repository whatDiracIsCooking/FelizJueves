"""`launch`: the whole-molecule launch plan as a value (#138).

Fast tier, no markers: pure Python, no compiler, no card, no PySCF -- the
basis comes from a tracked fixture's `converted` record, and the plan
evaluates no `[p|q]` (it takes none). `reference/` is imported HERE, by the
test, to build the real bound the plan is injected with; the module itself
never reaches it (test_layering.py, rule "launch").

THE CASES. H2O/STO-3G (s and p: classes lp, lq in 0..2) from the tracked
fixture, with its stored density; and a symmetric H2 dimer built here, whose
equal keys are what the stability test needs.
"""

from __future__ import annotations

import itertools
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao import drive
from felizjueves.ao import j as aoj
from felizjueves.ao.pairs import pair_bound, primitive_pairs
from felizjueves.ao.shell import Basis, Shell
from felizjueves.kernel import build_pq_kernels
from felizjueves.launch import (
    ClassLaunch,
    PairRec,
    _admits,
    class_spec,
    launches,
)
from felizjueves.reference.pq import one_centre, pq

BOUND = partial(pair_bound, one_centre=one_centre)


@cache
def _h2o():
    fx = manifest.load(manifest.FIXTURES / "h2o_sto3g_j.json")
    return ob.from_fixture(fx), np.array(fx.density)


def _dimer():
    """Two identical H2 molecules side by side: s shells only, mirror
    symmetric, so distinct records share keys exactly."""
    h = _h2o()[0].shells[-1]  # an H s shell from the fixture
    shells = tuple(Shell(0, c, h.exponents, h.coefficients)
                   for c in ((0.0, 0.0, 0.0), (0.0, 0.0, 1.4),
                             (6.0, 0.0, 0.0), (6.0, 0.0, 1.4)))
    n = len(shells)
    D = np.array([[1.0 / (1 + abs(a - b)) for b in range(n)]
                  for a in range(n)])
    return Basis(shells), D


def _plan(**kw):
    basis, D = _h2o()
    return launches(basis, D, **{"bound": BOUND, "n_batches": 1,
                                 "tau": 0.0, **kw})


def _unique_records(basis):
    """(i, j, prim, L) of every record of the unique sweep, in pair order."""
    sh = basis.shells
    return [(a, b, k, sh[a].l + sh[b].l)
            for a in range(len(sh)) for b in range(a + 1)
            for k in range(len(primitive_pairs(sh[a], sh[b])))]


def _ids(recs):
    return [(r.i, r.j, r.prim, r.L) for r in recs]


# --- coverage ---------------------------------------------------------------


def test_every_record_appears_exactly_once_in_its_class():
    basis, _ = _h2o()
    lp_set = {L for *_, L in _unique_records(basis)}
    P = _plan()
    assert [(c.lp, c.lq) for c in P] == sorted(itertools.product(lp_set,
                                                                 lp_set))
    want = sorted(_unique_records(basis))
    for l_other in lp_set:
        bras = [r for c in P if c.lq == l_other for r in c.bras]
        kets = [r for c in P if c.lp == l_other for r in c.kets]
        assert sorted(_ids(bras)) == want  # exactly once: sorted lists equal
        assert sorted(_ids(kets)) == want
    for c in P:
        assert {r.L for r in c.bras} == {c.lp}
        assert {r.L for r in c.kets} == {c.lq}
        assert c.L == c.lp + c.lq


def test_bras_keep_pair_order_and_the_shell_pair_map_round_trips():
    basis, _ = _h2o()
    for c in _plan():
        want = [x for x in _unique_records(basis) if x[3] == c.lp]
        assert _ids(c.bras) == want  # pair order, i outer, prim ascending
        assert all(r.key == r.G for r in c.bras)
        # bra_blocks tiles the bra list, in order, one block per shell pair
        stops = [0] + [stop for *_, stop in c.bra_blocks]
        assert [start for _, _, start, _ in c.bra_blocks] == stops[:-1]
        assert stops[-1] == len(c.bras)
        for a, b, start, stop in c.bra_blocks:
            block = c.bras[start:stop]
            assert {r.shell_pair for r in block} == {(a, b)}
            assert [r.prim for r in block] == list(
                range(len(primitive_pairs(basis.shells[a],
                                          basis.shells[b]))))
        # record -> shell pair -> its block contains the record
        where = {(a, b): (s, e) for a, b, s, e in c.bra_blocks}
        for n, r in enumerate(c.bras):
            s, e = where[r.shell_pair]
            assert s <= n < e and c.bras[s + r.prim] is r


# --- the keys and the sort ----------------------------------------------------


def test_the_ket_key_is_the_host_sweeps_bound_bit_for_bit():
    """`key = G * Dmax` with G from `pair_bound_pq` is exactly `ao.j`'s
    `_Ket.bound` -- the factor `coulomb_matrix_screened` multiplies `I_bra`
    by -- for every ket, so #142's equality to 0 has the same screen."""
    basis, D = _h2o()
    P = launches(basis, D, bound=partial(aoj.pair_bound_pq, pq=pq),
                 n_batches=1, tau=0.0)
    n_sh = len(basis.shells)
    pairs = [(a, b, 1.0 if a == b else 2.0)
             for a in range(n_sh) for b in range(a + 1)]
    host = aoj._kets(basis, aoj._density(basis, D), pairs, pq=pq, tau=1.0)
    rec = _unique_records(basis)
    want = {r: k.bound for r, k in zip(rec, host, strict=True)}
    got = {(r.i, r.j, r.prim, r.L): r.key for c in P for r in c.kets}
    assert got == want
    assert any(r.G != 1.0 for c in P for r in c.bras)  # the real bound


def test_kets_are_sorted_descending_and_stably():
    basis, D = _dimer()
    P = launches(basis, D, bound=BOUND, n_batches=1, tau=0.0)
    (c,) = P.classes
    keys = [r.key for r in c.kets]
    assert keys == sorted(keys, reverse=True)
    ties = [(x, y) for x, y in itertools.pairwise(c.kets) if x.key == y.key]
    assert ties, "the dimer's symmetry gives equal keys; stability is tested"
    # pair order is lexicographic in (i, j, prim): ties keep it
    assert all((x.i, x.j, x.prim) < (y.i, y.j, y.prim) for x, y in ties)
    # the ties are real: reversing an ascending sort reverses them
    rev = sorted(sorted(c.kets, key=lambda r: (r.i, r.j, r.prim)),
                 key=lambda r: r.key)[::-1]
    assert rev != list(c.kets)


def test_density_sort_false_changes_the_order_not_the_set():
    on, off = _plan(), _plan(density_sort=False)
    assert on.density_sort
    assert not off.density_sort
    changed = False
    dglobal = max(r.key / r.G for c in on for r in c.kets if r.G)
    for a, b in zip(on, off, strict=True):
        assert (a.lp, a.lq) == (b.lp, b.lq)
        assert sorted(_ids(a.kets)) == sorted(_ids(b.kets))
        assert a.bras == b.bras
        for r in b.kets:
            assert r.key == r.G * dglobal  # one global Dmax, the largest
        keys = [r.key for r in b.kets]
        assert keys == sorted(keys, reverse=True)
        changed |= _ids(a.kets) != _ids(b.kets)
    assert changed, "the global fallback must be a different order here"
    # sound: the global key bounds every per-record key
    per = {tuple(_ids([r])[0]): r.key for c in on for r in c.kets}
    assert all(r.key >= per[_ids([r])[0]] for c in off for r in c.kets)


def test_the_bound_is_injected():
    """A stub bound's values are what the records carry, on both sides."""
    basis, D = _h2o()

    def stub(zeta: float, L: int) -> float:
        return 10.0 * L + zeta

    P = launches(basis, D, bound=stub, n_batches=1, tau=0.0)
    sh = basis.shells
    for c in P:
        for r in (*c.bras, *c.kets):
            zeta = primitive_pairs(sh[r.i], sh[r.j])[r.prim].zeta
            assert r.G == stub(zeta, r.L)
    with pytest.raises(ValueError, match="finite float >= 0"):
        launches(basis, D, bound=lambda z, L: -1.0, n_batches=1, tau=0.0)


# --- kernels and batch counts ---------------------------------------------------


def test_one_kernel_build_serves_every_shell_class_of_a_class():
    """The kernel reads only (lp, lq), so the canonical class's kernels are
    every shell class's -- why one launch carries a class's kernels."""
    for lp, lq, n in [(1, 1, 1), (2, 1, 3), (2, 2, 3)]:
        mine = build_pq_kernels(class_spec(lp, lq, n))
        for la, lc in itertools.product(range(lp + 1), range(lq + 1)):
            am = (la, lp - la, lc, lq - lc)
            assert build_pq_kernels(drive.class_spec(am, n)) == mine
            assert drive.admitted_batches(am, (1, 3, 6)) == tuple(
                m for m in (1, 3, 6)
                if _admits(lp, lq, m))
    for c in _plan(n_batches=3, strict=False):
        assert c.kernels == build_pq_kernels(class_spec(c.lp, c.lq,
                                                        c.n_batches))


def test_a_refused_batch_count_raises_or_is_recorded():
    with pytest.raises(ValueError, match=r"n_batches = 3 .*\(0, 0\)"):
        _plan(n_batches=3)
    P = _plan(n_batches=3, strict=False)
    assert P.refused == ((0, 0, 3),)
    assert P.n_batches == 3
    assert {(c.lp, c.lq): c.n_batches for c in P} == {
        (lp, lq): 1 if (lp, lq) == (0, 0) else 3
        for lp in range(3) for lq in range(3)}
    assert _plan().refused == ()


def test_a_one_sided_class_is_refused_not_empty():
    r = PairRec(0, 0, 0, 0, 1.0, 1.0)
    k = build_pq_kernels(class_spec(0, 0))
    with pytest.raises(ValueError, match=r"one side only"):
        ClassLaunch(0, 0, 1, (r,), (), k, ((0, 0, 0, 1),))
    with pytest.raises(ValueError, match=r"one side only"):
        ClassLaunch(0, 0, 1, (), (r,), k, ())


# --- the inputs' contracts ------------------------------------------------------


def test_the_plan_refuses_what_the_host_sweep_refuses():
    basis, D = _h2o()
    D = D.copy()
    D[0, 1] += 1e-12
    with pytest.raises(ValueError, match="not symmetric"):
        launches(basis, D, bound=BOUND, n_batches=1, tau=0.0)
    with pytest.raises(ValueError, match="tau"):
        _plan(tau=-1.0)
    assert _plan(tau=1e-10).tau == 1e-10
