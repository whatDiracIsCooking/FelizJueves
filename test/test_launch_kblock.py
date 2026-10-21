"""`launch_kblock`: the block kernel's launch plan, and its walk (#481).

Fast tier unless `slow`: pure Python, no compiler, no card, no PySCF. The
cases are test_launch_k.py's (tracked fixtures' `converted` records, a
synthetic pair of H2), with H2O/STO-3G whole -- its (pp|pp) at L = 4, inside
the fast cut (`tiers.FAST_L_MAX`) -- and the synthetic s, s, p basis.

THE CLAIMS.

  regrouped  the plan is `launch_k.k_launches`' plan, regrouped: per class
             the same bra and ket records (as multisets), bras grouped by
             first shell and sorted by `G` within it, kets grouped by first
             shell in `k_launches`' sorted order; the blocks are every
             `(a, b)` once and tile the class's quartets.
  the walk   at tau = 0 it is `walk_k` over the same records to 0 (the same
             terms, fsummed) and `ao.k.exchange_matrix` to rounding.
  screened   on the tau ladder it skips EXACTLY the ordered record pairs
             `walk_k` skips (so its K is `walk_k`'s to 0), its bra `break`
             fires somewhere, and it examines at most two bounds per
             (bra, block) beyond the pairs it contracts.
"""

from __future__ import annotations

from functools import cache, partial

import numpy as np
import pytest
from test_k_ondevice import _sp_basis
from test_launch_k import SWEEP_TOL, TAUS, _stub, schwarz
from test_launch_k import _case as _launch_case

from felizjueves.ao.k import exchange_matrix, record_bounds
from felizjueves.launch_k import k_launches, walk_k
from felizjueves.launch_kblock import KBlock, kb_launches, walk_kb
from felizjueves.reference.pq import pq

FAST = ("h2_sto3g_j", "h2o_sto3g_s", "two_h2", "sp", "h2o_sto3g_j")
# The walks contract every quartet in Python: H2O/STO-3G whole is `slow`
# (test_launch_k.py's choice for the same walk), the plan's shape is not.
WALKED = [pytest.param(n, id=n, marks=pytest.mark.slow)
          if n == "h2o_sto3g_j" else n for n in FAST]


@cache
def _case(name: str):
    if name == "sp":
        return _sp_basis()
    return _launch_case(name)


def _ids(recs):
    return sorted((r.i, r.j, r.prim) for r in recs)


@pytest.mark.parametrize("name", FAST)
def test_the_plan_is_k_launches_regrouped(name):
    basis, D = _case(name)
    K = k_launches(basis, D, bound=_stub, key=schwarz, tau=0.0)
    B = kb_launches(basis, D, bound=_stub, tau=0.0)
    assert [c.am for c in B] == [c.am for c in K]
    assert B.n_quartets == K.n_quartets and B.dm == K.dm
    for b, k in zip(B, K, strict=True):
        assert _ids(b.bras) == _ids(k.bras) and _ids(b.kets) == _ids(k.kets)
        assert [r.i for r in b.bras] == sorted(r.i for r in b.bras)
        assert [r.i for r in b.kets] == sorted(r.i for r in b.kets)
        # kets: each group a subsequence of k_launches' sorted list
        for i in {r.i for r in b.kets}:
            assert [r for r in b.kets if r.i == i] == [
                r for r in k.kets if r.i == i]
        # bras: each group sorted by G, ties in pair order
        for a in {r.i for r in b.bras}:
            grp = [r for r in b.bras if r.i == a]
            assert grp == sorted([r for r in k.bras if r.i == a],
                                 key=lambda r: -r.G)
        assert len({(x.a, x.b) for x in b.blocks}) == len(b.blocks) == (
            len({r.i for r in b.bras}) * len({r.i for r in b.kets}))
        assert sum((x.bra1 - x.bra0) * (x.ket1 - x.ket0)
                   for x in b.blocks) == len(b.bras) * len(b.kets)
        assert b.spec == k.spec and b.kernel is None


def test_a_block_refuses_an_unsorted_or_mixed_range():
    basis, D = _case("h2o_sto3g_j")
    c = next(c for c in kb_launches(basis, D, bound=_stub, tau=0.0)
             if len({r.i for r in c.bras}) > 1)
    from dataclasses import replace
    with pytest.raises(ValueError, match="mixes"):
        replace(c, blocks=(KBlock(c.blocks[0].a, c.blocks[0].b, 0,
                                  len(c.bras), 0, 1),))
    grp = c.blocks[0]
    if grp.bra1 - grp.bra0 > 1:
        bras = list(c.bras)
        bras[grp.bra0], bras[grp.bra1 - 1] = bras[grp.bra1 - 1], bras[grp.bra0]
        with pytest.raises(ValueError, match="not sorted"):
            replace(c, bras=tuple(bras))
    with pytest.raises(ValueError, match="empty"):
        KBlock(0, 0, 1, 1, 0, 1)


def test_the_builder_is_injected_once_per_spec():
    basis, D = _case("h2o_sto3g_j")
    calls = []

    def build(spec):
        calls.append(spec)
        return f"kernel {spec.key()}"

    B = kb_launches(basis, D, bound=_stub, tau=0.0, build=build)
    assert len(calls) == len(set(calls)) == len(B)
    assert all(c.kernel == f"kernel {c.spec.key()}" for c in B)


@pytest.mark.parametrize("name", WALKED)
def test_the_walk_is_walk_k_and_exchange_matrix(name):
    basis, D = _case(name)
    bound = partial(record_bounds, pq=pq)
    w = walk_kb(kb_launches(basis, D, bound=bound, tau=0.0), basis, D, pq=pq)
    v = walk_k(k_launches(basis, D, bound=bound, key=schwarz, tau=0.0),
               basis, D, pq=pq)
    assert w.K == v.K and w.scale == v.scale
    assert (w.n_pairs, w.n_skipped, w.n_examined) == (v.n_pairs, 0, 0)
    K0 = np.array(exchange_matrix(basis, D, pq=pq))
    big = float(np.max(np.abs(K0)))
    assert big > 0.0
    assert float(np.max(np.abs(np.array(w.K) - K0))) <= SWEEP_TOL * big


@pytest.mark.parametrize("name", WALKED)
def test_the_screened_walk_skips_exactly_what_walk_k_skips(name):
    basis, D = _case(name)
    bound = partial(record_bounds, pq=pq)
    bra_break = False
    for tau in TAUS:
        P = kb_launches(basis, D, bound=bound, tau=tau)
        w = walk_kb(P, basis, D, pq=pq)
        v = walk_k(k_launches(basis, D, bound=bound, key=schwarz, tau=tau),
                   basis, D, pq=pq)
        assert w.K == v.K, (name, tau)
        assert (w.n_pairs, w.n_skipped) == (v.n_pairs, v.n_skipped)
        assert (w.max_skipped, w.dropped) == (v.max_skipped, v.dropped)
        # per (bra, block): one bra test, the survivors, one failing ket
        assert w.n_examined <= (w.n_pairs - w.n_skipped) + 2 * sum(
            b.bra1 - b.bra0 for c in P for b in c.blocks)
        # the bra break fired when some bra was skipped whole
        for c in P:
            for b in c.blocks:
                g0 = c.kets[b.ket0].G
                bra_break |= any((r.G * g0) * P.dm < tau
                                 for r in c.bras[b.bra0:b.bra1])
    if name in ("two_h2", "h2o_sto3g_j"):
        assert bra_break, "the ladder never reached the bra break"
