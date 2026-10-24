"""`launch_k`: the K launch plan, and whether `ClassLaunch` is J-shaped (#204).

Fast tier, no device markers: pure Python, no compiler, no card, no PySCF.
The bases come from tracked fixtures' `converted` records; `reference/` is
imported HERE, by the test, for `pq` and the real bound the plan is injected
with -- `launch_k` itself never reaches it (test_layering.py, rule
"k_launch").

THE ANSWER IS OUTCOME 2 (launch_k's module docstring): a `KClassLaunch`
beside `ClassLaunch`, sharing `PairRec`, the stable sort, the bound check
and the batch admission from `launch.py` -- and the J plans unchanged, which
test/test_launch.py pins by a fingerprint taken before #204.

THE CLAIMS.

  coverage   every ORDERED pair's records once per side per partner class;
             bras in pair order, `bra_blocks` tiling them; each ket class
             sorted ONCE, descending by the injected key, stably, the same
             list in every class that has it.
  injected   the bound's values are the records' `G`, the key's their ket
             `key`; the builder is called once per `Spec`; none is imported.
  admission  the plan admits a K `Spec` exactly when it admits the J class
             of the same split -- why `batches_for` is shared.
  walk       the plan walked in Python reproduces `ao.k.exchange_matrix`,
             `exchange_matrix_reference` and `k_from_eris` to rounding on
             THREE molecules (below); its `scale` is ao.k's.
  screen     on the tau ladder the walk's `break` == its `continue` TO 0
             (fsum), skips exactly the ordered record pairs whose bound
             `(I_ac I_bd) Dm` is under tau (counted here by brute force),
             agrees with ao.k's screened sweeps to rounding, and stays
             within `tau * scale` of K(0).
  the J key  injected as the sort key (`I_bd * max|D_bd|`, J's `Dmax`
             shape), it is REFUSED by the walk; walked anyway
             (`check_sorted=False`) its `break` moves K at some rung.

THE L <= 2 RULE (milestone 10: no K quartet class with `L = la + lc + lb +
ld > 2` is computed before #231, host-side included). Every `pq` here is
`_guarded`, which RAISES on `|p| + |q| > 2` -- so the rule is enforced, not
promised. The walked cases are all-s: H2/STO-3G, H2O/STO-3G's four s shells
(with the matching block of its density) and two H2 6 bohr apart with the
hilbert density (#200's cases). The p shells enter through H2O/STO-3G whole,
planned with a STUB bound (no `pq` at all) and walked over its classes with
L <= 2 only, against `exchange_matrix` with the same truncation as a `k=`
hook (test_ao_k.py's). The whole-basis walk reaches (pp|pp), L = 4: written
below, marked `slow`, deferred to #231.

THE TOLERANCE, stated and not tuned: SWEEP_TOL = 1e-14 relative to
max|K(0)| (naive accumulation against fsum, test_ao_k.py's figure); the
break, the skip counts and the scale are exact.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.density import epair
from felizjueves.ao.shell import Basis, Shell, n_cartesians
from felizjueves.eri.j_coulomb.staged import coulomb_matrix_staged
from felizjueves.eri.k_exchange.k import (
    exchange_matrix,
    exchange_matrix_reference,
    exchange_matrix_screened,
    k_from_eris,
    k_quartet,
    record_bounds,
)
from felizjueves.eri.k_exchange.launch_k import (
    KClassLaunch,
    KLaunchPlan,
    exchange_launched,
    k_class_spec,
    k_launches,
    walk_k,
)
from felizjueves.eri.launch_common import PairRec, admits
from felizjueves.ir import Kernel
from felizjueves.plan.pipeline import plan as plan_of
from felizjueves.reference.pq import pq as _pq

TAUS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
SWEEP_TOL = 1e-14
L_MAX_RUN = 2  # the milestone's rule: nothing above this is evaluated


def _guarded(p, q, PQ, theta):
    """`reference.pq.pq`, refusing any Hermite pair above the rule."""
    if sum(p) + sum(q) > L_MAX_RUN:
        raise AssertionError(f"[p|q] at |p| + |q| = {sum(p) + sum(q)} > "
                             f"{L_MAX_RUN}: the milestone's L <= 2 rule")
    return _pq(p, q, PQ, theta)


pq = _guarded
BOUND = partial(record_bounds, pq=pq)


def _pq_for(name: str):
    """The guarded `pq` for a fast case; the slow one (#231's) needs L = 4."""
    return _pq if name in SLOW else _guarded


def schwarz(b: int, d: int, i_bd: float) -> float:
    """K's ket key, `I_bd` alone -- injected, as the plan requires."""
    return i_bd


# --- the cases ---------------------------------------------------------------


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return ob.from_fixture(fx), np.array(fx.density)


@cache
def _case(name: str) -> tuple[Basis, np.ndarray]:
    if name in ("h2_sto3g_j", "h2o_sto3g_j"):
        return _fixture(name)
    if name == "h2o_sto3g_s":
        basis, D = _fixture("h2o_sto3g_j")
        keep = [i for i, s in enumerate(basis.shells) if s.l == 0]
        aos = [basis.offsets[i] for i in keep]
        return Basis(tuple(basis.shells[i] for i in keep)), D[np.ix_(aos, aos)]
    assert name == "two_h2"
    s = _fixture("h2_sto3g_j")[0].shells[0]
    basis = Basis(tuple(
        Shell(0, (0.0, 0.0, 6.0 * m + dz), s.exponents, s.coefficients)
        for m in range(2) for dz in (0.0, 1.4)))
    n = basis.n_ao
    return basis, np.array([[1.0 / (i + j + 1) for j in range(n)]
                            for i in range(n)])


FAST = ("h2_sto3g_j", "h2o_sto3g_s", "two_h2")
SLOW = ("h2o_sto3g_j",)  # (pp|pp), L = 4: deferred to #231
PARAMS = ([pytest.param(n, id=n) for n in FAST]
          + [pytest.param(n, id=n, marks=pytest.mark.slow) for n in SLOW])


def _stub(ep) -> tuple[float, ...]:
    """A bound that evaluates nothing: distinct, positive, per record."""
    return tuple(1.0 + 0.1 * ep.la + 0.01 * ep.lb + rec[1] / (1.0 + rec[1])
                 for rec in ep.recs)


def _ordered_records(basis: Basis):
    """(i, j, prim, L) of every ORDERED pair's records, in pair order."""
    sh = basis.shells
    return [(a, c, k, sh[a].l + sh[c].l)
            for a in range(len(sh)) for c in range(len(sh))
            for k in range(len(epair(sh[a], sh[c]).recs))]


def _ids(recs):
    return [(r.i, r.j, r.prim, r.L) for r in recs]


def test_the_fast_cases_keep_to_l_le_2():
    """The rule, checked: every walked case is all-s, the slow one is not,
    and the guard refuses a Hermite pair above it."""
    for name in FAST:
        basis, D = _case(name)
        assert all(s.l == 0 for s in basis.shells), name
        assert np.array_equal(D, D.T)
    for name in SLOW:
        assert 4 * max(s.l for s in _case(name)[0].shells) > L_MAX_RUN
    with pytest.raises(AssertionError, match="L <= 2 rule"):
        pq((1, 0, 0), (1, 1, 0), (0.0, 0.0, 0.0), 1.0)


# --- the plan's shape (H2O/STO-3G whole, a stub bound: nothing evaluated) ----


def _h2o_plan(**kw) -> KLaunchPlan:
    basis, D = _case("h2o_sto3g_j")
    return k_launches(basis, D, **{"bound": _stub, "key": schwarz,
                                   "tau": 0.0, **kw})


def test_every_ordered_record_appears_once_per_side():
    basis, _ = _case("h2o_sto3g_j")
    P = _h2o_plan()
    sh = basis.shells
    pair_classes = sorted({(sh[a].l, sh[c].l) for a in range(len(sh))
                           for c in range(len(sh))})
    assert pair_classes == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [c.am for c in P] == [(*x, *y) for x, y in
                                 itertools.product(pair_classes, repeat=2)]
    want = sorted(_ordered_records(basis))
    for other in pair_classes:
        bras = [r for c in P if c.am[2:] == other for r in c.bras]
        kets = [r for c in P if c.am[:2] == other for r in c.kets]
        assert sorted(_ids(bras)) == want  # exactly once: sorted lists equal
        assert sorted(_ids(kets)) == want
    for c in P:
        assert {(sh[r.i].l, sh[r.j].l) for r in c.bras} == {c.am[:2]}
        assert {(sh[r.i].l, sh[r.j].l) for r in c.kets} == {c.am[2:]}
        assert (c.lp, c.lq, c.L) == (c.am[0] + c.am[1], c.am[2] + c.am[3],
                                     sum(c.am))
        assert c.spec == k_class_spec(c.am, c.n_batches)
        assert c.spec.target.value == "k" and c.spec.am == c.am
    # the plan covers every ordered (bra record, ket record) pair once
    n_rec = len(_ordered_records(basis))
    assert P.n_quartets == n_rec * n_rec


def test_bras_keep_pair_order_and_the_blocks_tile_them():
    basis, _ = _case("h2o_sto3g_j")
    sh = basis.shells
    for c in _h2o_plan():
        want = [x for x in _ordered_records(basis)
                if (sh[x[0]].l, sh[x[1]].l) == c.am[:2]]
        assert _ids(c.bras) == want  # pair order: a outer, c inner, prim
        assert all(r.key == r.G for r in c.bras)
        stops = [0] + [stop for *_, stop in c.bra_blocks]
        assert [start for _, _, start, _ in c.bra_blocks] == stops[:-1]
        assert stops[-1] == len(c.bras)
        for a, cc, start, stop in c.bra_blocks:
            block = c.bras[start:stop]
            assert {r.shell_pair for r in block} == {(a, cc)}
            assert [r.prim for r in block] == list(
                range(len(epair(sh[a], sh[cc]).recs)))


def test_each_ket_class_is_sorted_once_descending_by_its_key():
    P = _h2o_plan()
    seen: dict[tuple[int, int], tuple] = {}
    for c in P:
        keys = [r.key for r in c.kets]
        assert keys == sorted(keys, reverse=True)
        assert seen.setdefault(c.am[2:], c.kets) == c.kets  # one list
    # the stub's records are distinct, so the order is not pair order
    assert any(_ids(c.kets) != sorted(_ids(c.kets)) for c in P)


def test_the_sort_is_stable():
    """Two H2 molecules, mirror symmetric: equal `I_bd` on distinct
    records, and ties keep pair order."""
    basis, D = _case("two_h2")
    (c,) = k_launches(basis, D, bound=BOUND, key=schwarz, tau=0.0)
    ties = [(x, y) for x, y in itertools.pairwise(c.kets) if x.key == y.key]
    assert ties, "the mirror symmetry gives equal keys; stability is tested"
    assert all((x.i, x.j, x.prim) < (y.i, y.j, y.prim) for x, y in ties)


def test_the_bound_and_the_key_are_injected():
    basis, D = _case("h2o_sto3g_j")

    def key(b, d, g):
        return 3.0 * g + b + 0.5 * d

    P = _h2o_plan(key=key)
    sh = basis.shells
    for c in P:
        for r in (*c.bras, *c.kets):
            assert r.G == _stub(epair(sh[r.i], sh[r.j]))[r.prim]
        for r in c.kets:
            assert r.key == key(r.i, r.j, r.G)
    with pytest.raises(ValueError, match="finite float >= 0"):
        _h2o_plan(bound=lambda ep: tuple(-1.0 for _ in ep.recs))
    with pytest.raises(ValueError, match="finite float >= 0"):
        _h2o_plan(key=lambda b, d, g: float("nan"))
    with pytest.raises(ValueError, match="values for the"):
        _h2o_plan(bound=lambda ep: (1.0,) * (len(ep.recs) + 1))


def test_the_builder_is_injected_and_called_once_per_spec():
    calls = []

    def build(spec):
        calls.append(spec)
        return (Kernel(f"k_{len(calls)}", (), ()),)

    P = _h2o_plan(build=build)
    assert len(calls) == len(set(calls)) == len(P)  # one per class Spec
    assert {c.spec for c in P} == set(calls)
    assert all(c.kernels == (Kernel(f"k_{calls.index(c.spec) + 1}", (), ()),)
               for c in P)
    assert all(c.kernels == () for c in _h2o_plan())  # none injected
    with pytest.raises(ValueError, match="returned no kernels"):
        _h2o_plan(build=lambda spec: ())


def test_the_batch_admission_is_the_j_class_admission():
    """Why `launch_common.batches_for` is shared: the plan admits a K class exactly when
    it admits the J class of the same split (classes with L <= 2: planning,
    not evaluating, but the rule is kept)."""
    for am in itertools.product(range(3), repeat=4):
        if sum(am) > L_MAX_RUN:
            continue
        lp, lq = am[0] + am[1], am[2] + am[3]
        for n in (1, 3, 6):
            try:
                plan_of(k_class_spec(am, n))
                ok = True
            except ValueError:
                ok = False
            assert ok == admits(lp, lq, n), (am, n)


def test_a_refused_batch_count_raises_or_is_recorded():
    with pytest.raises(ValueError, match=r"n_batches = 3 .*\(0, 0\)"):
        _h2o_plan(n_batches=3)
    P = _h2o_plan(n_batches=3, strict=False)
    assert P.refused == ((0, 0, 0, 0, 3),)
    assert {c.am: c.n_batches for c in P} == {
        c.am: 1 if c.am == (0, 0, 0, 0) else 3 for c in P}
    assert _h2o_plan().refused == ()


def test_a_one_sided_class_is_refused_not_empty():
    r = PairRec(0, 0, 0, 0, 1.0, 1.0)
    with pytest.raises(ValueError, match="one side only"):
        KClassLaunch((0, 0, 0, 0), 1, (r,), (), (), ((0, 0, 0, 1),))
    with pytest.raises(ValueError, match="one side only"):
        KClassLaunch((0, 0, 0, 0), 1, (), (r,), (), ())
    with pytest.raises(ValueError, match="outside K class"):
        KClassLaunch((0, 1, 0, 0), 1, (r,), (r,), (), ((0, 0, 0, 1),))


def test_the_inputs_contracts():
    basis, D = _case("h2_sto3g_j")
    bad = D.copy()
    bad[0, 1] += 1e-12
    with pytest.raises(ValueError, match="not symmetric"):
        k_launches(basis, bad, bound=BOUND, key=schwarz, tau=0.0)
    with pytest.raises(ValueError, match="tau"):
        k_launches(basis, D, bound=BOUND, key=schwarz, tau=-1.0)
    P = k_launches(basis, D, bound=BOUND, key=schwarz, tau=1e-6)
    assert P.tau == 1e-6 and P.dm == float(np.max(np.abs(D)))
    with pytest.raises(ValueError, match="another density"):
        walk_k(P, basis, 2.0 * D, pq=pq)
    with pytest.raises(ValueError, match="order"):
        walk_k(P, basis, D, pq=pq, order="random")


def test_the_staged_j_driver_refuses_a_k_plan():
    """Why `KClassLaunch` is its own type: ao/staged.py's J driver would
    otherwise read a K class as `(lp, lq)`, upload its kets with a `Dmax`
    and run eq. 20 over its bra blocks. It refuses, before any work."""
    basis, D = _case("h2_sto3g_j")
    P = k_launches(basis, D, bound=BOUND, key=schwarz, tau=0.0)

    def boys(m, T):
        raise AssertionError("nothing is evaluated")

    with pytest.raises(TypeError, match="not a J `LaunchPlan`"):
        coulomb_matrix_staged(P, basis, D, boys=boys)


# --- the walk ----------------------------------------------------------------


@pytest.mark.parametrize("name", PARAMS)
def test_the_walk_reproduces_exchange_matrix(name):
    """At tau = 0, on three molecules (one slow): the plan walked is K."""
    basis, D = _case(name)
    pq = _pq_for(name)
    w = exchange_launched(basis, D, pq=pq, tau=0.0)
    K = np.array(w.K)
    K0 = np.array(exchange_matrix(basis, D, pq=pq))
    big = np.max(np.abs(K0))
    for f in (exchange_matrix_reference, k_from_eris):
        assert np.max(np.abs(K - np.array(f(basis, D, pq=pq)))) <= (
            SWEEP_TOL * big), f.__name__
    assert np.max(np.abs(K - K0)) <= SWEEP_TOL * big
    host = exchange_matrix_screened(basis, D, pq=pq, tau=0.0)
    assert w.scale == host.scale  # counted from the plan; ao.k's R(a) R(b)
    assert (w.n_skipped, w.n_examined) == (0, 0)
    n_rec = len(_ordered_records(basis))
    assert w.n_pairs == n_rec * n_rec
    print(f"\n{name}: max|K_walk - K| / max|K| = "
          f"{float(np.max(np.abs(K - K0)) / big):.2e}")


def _brute_skips(basis, D, tau, pq) -> int:
    """Ordered record pairs with `(I_ac * I_bd) * Dm < tau`, by brute
    force over every ordered pair of ordered shell pairs."""
    sh = basis.shells
    dm = float(np.max(np.abs(D)))
    Is = [g for a in range(len(sh)) for c in range(len(sh))
          for g in record_bounds(epair(sh[a], sh[c]), pq=pq)]
    return sum((x * y) * dm < tau for x in Is for y in Is)


@cache
def _ladder(name: str):
    basis, D = _case(name)
    pq = _pq_for(name)
    out = []
    for tau in TAUS:
        P = k_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       key=schwarz, tau=tau)
        out.append((tau, walk_k(P, basis, D, pq=pq),
                    walk_k(P, basis, D, pq=pq, order="unsorted")))
    return out


@pytest.mark.parametrize("name", PARAMS)
def test_the_screened_walk_is_exact_and_bounded(name):
    basis, D = _case(name)
    pq = _pq_for(name)
    rungs = _ladder(name)
    K0 = np.array(rungs[0][1].K)
    big = np.max(np.abs(K0))
    prev = -1
    for tau, a, b in rungs:
        assert a.K == b.K, f"{name} tau={tau}: break != continue"
        assert (a.n_skipped, a.n_pairs) == (b.n_skipped, b.n_pairs)
        assert a.dropped == b.dropped and a.max_skipped == b.max_skipped
        assert a.n_examined <= b.n_examined
        assert b.n_examined == (b.n_pairs if tau > 0.0 else 0)
        assert a.n_skipped == _brute_skips(basis, D, tau, pq)
        assert a.max_skipped < tau or a.n_skipped == 0
        err = np.abs(np.array(a.K) - K0)
        assert np.all(err <= tau * np.array(a.scale)), (name, tau)
        assert a.n_skipped >= prev
        prev = a.n_skipped
        host = exchange_matrix_screened(basis, D, pq=pq, tau=tau)
        for got in (host.K, exchange_matrix_reference(basis, D, pq=pq,
                                                      tau=tau)):
            assert np.max(np.abs(np.array(a.K) - np.array(got))) <= (
                SWEEP_TOL * big), (name, tau)
        print(f"\n{name} tau={tau:g}: skip {a.skip_fraction:.3f}, examined "
              f"{a.n_examined}/{b.n_examined} (host unique sweep: skip "
              f"{host.skip_fraction:.3f})")
    assert rungs[-1][1].n_skipped > 0  # the ladder is not vacuous


def _j_key(basis: Basis, D: np.ndarray):
    """The J engine's key, misapplied: `I_bd * max|D_bd|` (ao/k.py)."""
    off, sh = basis.offsets, basis.shells

    def key(b: int, d: int, i_bd: float) -> float:
        blk = D[off[b]:off[b] + sh[b].n_cart, off[d]:off[d] + sh[d].n_cart]
        return i_bd * float(np.max(np.abs(blk)))

    return key


@pytest.mark.parametrize("name", PARAMS)
def test_the_j_key_is_refused_and_its_break_moves_k(name):
    """The key is injectable, so the J key can be injected -- and the walk
    refuses it (its kets are not non-increasing in `G`, the factor the
    screen reads). Walked anyway, its break skips a pair whose bound is >=
    tau at some rung, and K moves; the Schwarz key's never does (above)."""
    basis, D = _case(name)
    pq = _pq_for(name)
    unsafe = []
    refused = False
    for tau, _, cont in _ladder(name)[1:]:
        P = k_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       key=_j_key(basis, D), tau=tau)
        try:
            walk_k(P, basis, D, pq=pq)
        except ValueError as e:
            assert "non-increasing in G" in str(e)
            refused = True
        j = walk_k(P, basis, D, pq=pq, check_sorted=False)
        assert j.n_skipped >= cont.n_skipped
        if j.n_skipped > cont.n_skipped:
            assert j.max_skipped >= tau and j.K != cont.K
            unsafe.append(tau)
    print(f"\n{name}: the J key's break is unsafe at tau in {unsafe}")
    assert refused and unsafe


# --- p shells, at L <= 2 ------------------------------------------------------


def _truncated(calls):
    """`k_quartet` on L <= 2; zeros above, nothing evaluated (test_ao_k)."""

    def hook(bra, ket, dcd):
        L = bra.la + bra.lb + ket.la + ket.lb
        calls.append(L)
        if L > L_MAX_RUN:
            return [[0.0] * n_cartesians(ket.la)
                    for _ in range(n_cartesians(bra.la))]
        return k_quartet(bra, ket, dcd, pq=pq)

    return hook


def test_the_walk_with_p_shells_at_l_le_2():
    """H2O/STO-3G whole, planned with the stub bound; its classes with
    L <= 2 walked, against `exchange_matrix` truncated the same way by its
    `k=` hook. K's L is invariant under the bra-ket swap, so the unique
    sweep's truncation is the ordered sweep's."""
    basis, D = _case("h2o_sto3g_j")
    P = _h2o_plan()
    low = tuple(c for c in P if c.L <= L_MAX_RUN)
    assert len(low) < len(P) and {c.L for c in low} == {0, 1, 2}
    assert any(1 in c.am for c in low)  # p shells walked, on both sides
    w = walk_k(replace(P, classes=low), basis, D, pq=pq)
    calls: list[int] = []
    K0 = np.array(exchange_matrix(basis, D, pq=pq, k=_truncated(calls)))
    assert max(calls) > L_MAX_RUN  # the cut is real
    big = np.max(np.abs(K0))
    assert np.max(np.abs(np.array(w.K) - K0)) <= SWEEP_TOL * big
    assert big > 0.0


# --- #480: the bra-ket-symmetric sweep, at shell granularity ------------------
#
# `sweep="symmetric"` keeps, per orbit `{(a,c|b,d), (b,d|a,c)}`, the member
# with `a <= b` -- both when `a == b` -- by cutting each ket group `b`'s bras
# to the prefix `a <= b`, and mirrors the `a < b` blocks. The ordered sweep,
# the default, is pinned by the fingerprint below, taken before #480.

import hashlib  # noqa: E402

ORDERED_SHA256 = (
    "6d192cb5c83be78f95f94210a4ff6530a2412ed027bdce2f2c4fab7fc681f3ff")


def _plan_dump(P: KLaunchPlan) -> str:
    out = [repr((P.tau, P.dm, P.n_batches, P.refused))]
    for c in P:
        out.append(repr((c.am, c.n_batches, c.bra_blocks, c.kernels)))
        for side in (c.bras, c.kets):
            out.append(";".join(repr((r.i, r.j, r.prim, r.L, r.G, r.key))
                                for r in side))
    return "\n".join(out)


def _ordered_plans() -> list[KLaunchPlan]:
    basis, D = _case("h2o_sto3g_j")
    plans = [k_launches(basis, D, bound=_stub, key=schwarz, tau=0.0),
             k_launches(basis, D, bound=_stub, key=schwarz, tau=1e-6,
                        n_batches=3, strict=False)]
    for name in FAST:
        basis, D = _case(name)
        plans.append(k_launches(basis, D, bound=BOUND, key=schwarz,
                                tau=1e-8))
    return plans


def test_the_ordered_sweep_is_unchanged_by_the_symmetric_one():
    plans = _ordered_plans()
    for P in plans:
        assert P.sweep == "ordered" and not P.mirrors
        assert all(c.ket_groups == () for c in P)
    got = hashlib.sha256("\n#\n".join(map(_plan_dump, plans)).encode())
    assert got.hexdigest() == ORDERED_SHA256


def _pair_id(r: PairRec) -> tuple[int, int, int]:
    return (r.i, r.j, r.prim)


def test_the_symmetric_plan_is_the_shell_prefix():
    """Per class: the ordered bras unchanged; the kets regrouped per first
    shell `b`, each group the ordered list filtered (so sorted, stably);
    `n_bra` the prefix `a <= b`. Over the plan: every ordered record pair
    computed once or covered once by a mirror, never both, never twice."""
    basis, D = _case("h2o_sto3g_j")
    Op = _h2o_plan()
    Sp = _h2o_plan(sweep="symmetric")
    assert Sp.sweep == "symmetric" and Sp.mirrors
    ordered = {c.am: c for c in Op}
    assert {c.am for c in Sp} <= set(ordered)
    computed: list[tuple] = []
    for c in Sp:
        o = ordered[c.am]
        assert (c.bras, c.bra_blocks, c.n_batches) == (
            o.bras, o.bra_blocks, o.n_batches)
        assert [g[0] for g in c.ket_groups] == sorted(
            {g[0] for g in c.ket_groups})
        for b, start, stop, n_bra in c.ket_groups:
            assert c.kets[start:stop] == tuple(r for r in o.kets if r.i == b)
            assert all(r.i <= b for r in c.bras[:n_bra])
            assert all(r.i > b for r in c.bras[n_bra:])
        for bras, kets in c.launches():
            computed.extend((_pair_id(x), _pair_id(y))
                            for x in bras for y in kets)
    assert len(computed) == len(set(computed)) == Sp.n_quartets
    mirrored = [(y, x) for x, y in computed if x[0] != y[0]]
    assert not set(mirrored) & set(computed)
    every = [(_pair_id(x), _pair_id(y)) for c in Op
             for x in c.bras for y in c.kets]
    assert sorted(computed + mirrored) == sorted(every)
    n_diag = sum(x[0] == y[0] for x, y in every)
    assert 2 * Sp.n_quartets == Op.n_quartets + n_diag
    assert Sp.n_quartets < Op.n_quartets
    print(f"\nh2o_sto3g: {Sp.n_quartets} of {Op.n_quartets} ordered record "
          f"pairs computed ({n_diag} in a == b blocks)")


def test_the_symmetric_sweep_refusals():
    basis, D = _case("h2_sto3g_j")
    with pytest.raises(ValueError, match="sweep"):
        k_launches(basis, D, bound=BOUND, key=schwarz, tau=0.0,
                   sweep="unique")
    with pytest.raises(ValueError):  # the mirror needs D_cd = D_dc
        bad = D.copy()
        bad[0, 1] += 1.0
        k_launches(basis, bad, bound=BOUND, key=schwarz, tau=0.0,
                   sweep="symmetric")
    c = _h2o_plan(sweep="symmetric").classes[0]
    b, start, stop, n_bra = c.ket_groups[0]
    for groups in (((b, start, stop, 0),), ((b, start + 1, stop, n_bra),),
                   ((b + 1, start, stop, n_bra),)):
        with pytest.raises(ValueError, match="ket group"):
            replace(c, ket_groups=groups + c.ket_groups[1:])
    if len(c.ket_groups) > 1:
        with pytest.raises(ValueError, match="tile"):
            replace(c, ket_groups=c.ket_groups[:1])


@pytest.mark.parametrize("name", PARAMS)
def test_the_symmetric_walk_reproduces_exchange_matrix(name):
    """At tau = 0: the mirrored walk is K, the ordered walk's K to
    rounding, ao.k's scale exactly, over the halved record-pair set."""
    basis, D = _case(name)
    pq = _pq_for(name)
    P = k_launches(basis, D, bound=partial(record_bounds, pq=pq),
                   key=schwarz, tau=0.0, sweep="symmetric")
    w = walk_k(P, basis, D, pq=pq)
    K = np.array(w.K)
    K0 = np.array(exchange_matrix(basis, D, pq=pq))
    big = np.max(np.abs(K0))
    for got in (K0, exchange_matrix_reference(basis, D, pq=pq),
                exchange_launched(basis, D, pq=pq, tau=0.0).K):
        assert np.max(np.abs(K - np.array(got))) <= SWEEP_TOL * big
    assert w.scale == exchange_matrix_screened(basis, D, pq=pq,
                                               tau=0.0).scale
    assert w.n_pairs == P.n_quartets
    assert (w.n_skipped, w.n_examined) == (0, 0)
    print(f"\n{name}: symmetric sweep, {P.n_quartets} record pairs, "
          f"max|K - K_unique| / max|K| = "
          f"{float(np.max(np.abs(K - K0)) / big):.2e}")


def _brute_skips_symmetric(basis, D, tau, pq) -> int:
    """`_brute_skips` over the record pairs with bra shell `a <= b`."""
    sh = basis.shells
    dm = float(np.max(np.abs(D)))
    Is = [(a, g) for a in range(len(sh)) for c in range(len(sh))
          for g in record_bounds(epair(sh[a], sh[c]), pq=pq)]
    return sum((x * y) * dm < tau for a, x in Is for b, y in Is if a <= b)


@pytest.mark.parametrize("name", PARAMS)
def test_the_symmetric_screened_walk_is_the_ordered_one(name):
    """On the tau ladder: break == continue to 0, the skips are the halved
    set's brute-force count, and K is the ordered walk's to rounding (the
    screen is symmetric under the swap) and within `tau * scale` of K(0)."""
    basis, D = _case(name)
    pq = _pq_for(name)
    rungs = _ladder(name)
    big = np.max(np.abs(np.array(rungs[0][1].K)))
    K0 = None  # the symmetric walk's own K(0): the mirror rounds differently
    for tau, ordered, _ in rungs:
        P = k_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       key=schwarz, tau=tau, sweep="symmetric")
        a = walk_k(P, basis, D, pq=pq)
        b = walk_k(P, basis, D, pq=pq, order="unsorted")
        assert a.K == b.K and a.n_skipped == b.n_skipped, (name, tau)
        assert a.n_skipped == _brute_skips_symmetric(basis, D, tau, pq)
        if K0 is None:
            K0 = np.array(a.K)
        assert a.scale == ordered.scale
        assert np.max(np.abs(np.array(a.K) - np.array(ordered.K))) <= (
            SWEEP_TOL * big), (name, tau)
        assert np.all(np.abs(np.array(a.K) - K0)
                      <= tau * np.array(a.scale)), (name, tau)
    assert a.n_skipped > 0


def test_the_symmetric_walk_with_p_shells_at_l_le_2():
    """`test_the_walk_with_p_shells_at_l_le_2`, mirrored: a class and its
    partner share `L`, so the truncation is the same set of orbits."""
    basis, D = _case("h2o_sto3g_j")
    P = _h2o_plan(sweep="symmetric")
    low = tuple(c for c in P if c.L <= L_MAX_RUN)
    assert len(low) < len(P) and any(1 in c.am for c in low)
    w = walk_k(replace(P, classes=low), basis, D, pq=pq)
    calls: list[int] = []
    K0 = np.array(exchange_matrix(basis, D, pq=pq, k=_truncated(calls)))
    big = np.max(np.abs(K0))
    assert np.max(np.abs(np.array(w.K) - K0)) <= SWEEP_TOL * big
    assert big > 0.0
