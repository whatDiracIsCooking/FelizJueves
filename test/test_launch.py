"""`launch`: the whole-molecule launch plan as a value (#138).

Fast tier, no device markers (one test carries `claim`, #139): pure Python,
no compiler, no card, no PySCF -- the basis comes from a tracked fixture's
`converted` record, and the plan evaluates no `[p|q]` (it takes none).
`reference/` is imported HERE, by the test, to build the real bound the plan
is injected with; the module itself never reaches it (test_layering.py, rule "launch").

THE CASES. H2O/STO-3G (s and p: classes lp, lq in 0..2) from the tracked
fixture, with its stored density; and a symmetric H2 dimer built here, whose
equal keys are what the stability test needs.
"""

from __future__ import annotations

import dataclasses
import hashlib
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


# --- the sort key, as a claim (#139) --------------------------------------------

TAU_SORT = 1e-6  # H2O/STO-3G's screen skips 385 (bra, ket) index pairs here


def _exceeds(A, B, scale, tau) -> bool:
    """Some element of `A` is further from `B` than the screen's own stated
    error, `tau * scale` (`ao.j`, "THE SORTED `break`"), allows."""
    return any(abs(a - b) > tau * s
               for xa, xb, xs in zip(A, B, scale, strict=True)
               for a, b, s in zip(xa, xb, xs, strict=True))


@pytest.mark.claim("jengine/ket-sort-key-is-the-product")
def test_the_ket_sort_key_is_the_product_and_the_break_needs_it(monkeypatch):
    """[JEngine, v2, §3.3, Alg. 1]: kets sorted descending by `I_q * Dmax`,
    so the screen can `break`. Not "the code sorts": the claim is that the
    PRODUCT is the key the `break` is sound under, so the falsifier is a
    different key giving a different `J` through the same screened walk --
    `ao.j.coulomb_matrix_screened`'s sorted `break`, with only its ket order
    (`_sorted_kets`) replaced, test-locally, by monkeypatch.

    (a) The plan's ket order is the product order -- the product recomputed
        here from each record's `G` and the host's `max_q |D_q|` -- and it
        is NOT the `G`-alone order, so the claim has content on this case.
    (b) Under the product order the `break` is exact: the sorted walk equals
        the unsorted `continue` to 0, with the same skip count.
    (c) Under another order -- `G` alone, or the unsorted pair order -- the
        same `break` returns a different `J`, outside the screen's own error
        bound `tau * scale` from `J(tau = 0)`: unsound, not merely looser.
    (d) The global-`Dmax` fallback (`density_sort=False`'s key, made the
        screen's key too) is sound -- inside `tau * scale` -- and skips no
        more than the product. How much less it skips, and what that costs
        on a card, is the measured arm (#147), not this test.
    """
    basis, D = _h2o()
    tau = TAU_SORT
    P = launches(basis, D, bound=partial(aoj.pair_bound_pq, pq=pq),
                 n_batches=1, tau=tau)
    n_sh = len(basis.shells)
    pairs = [(a, b, 1.0 if a == b else 2.0)
             for a in range(n_sh) for b in range(a + 1)]
    host = aoj._kets(basis, aoj._density(basis, D), pairs, pq=pq, tau=1.0)
    rec = _unique_records(basis)  # the host's ket records, in its order
    dmax = {r: max(abs(v) for v in k.dq)
            for r, k in zip(rec, host, strict=True)}
    G = {(r.i, r.j, r.prim, r.L): r.G for c in P for r in c.kets}

    # (a) the plan's order is the product order, and not the G order
    differs_from_g = False
    for c in P:
        ids = _ids(c.kets)
        product = [G[x] * dmax[x] for x in ids]
        assert product == sorted(product, reverse=True), (c.lp, c.lq)
        differs_from_g |= sorted(ids, key=G.__getitem__, reverse=True) != ids
    assert differs_from_g, "H2O's density must reorder some class's kets"

    # The screened walk with its ket order replaced. `_kets` builds fresh
    # records per call, in pair order, so a replacement keys them by `rec`.
    product_sort = aoj._sorted_kets

    def walk(order=None):
        if order is not None:
            monkeypatch.setattr(aoj, "_sorted_kets", order)
        try:
            return aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=tau)
        finally:
            monkeypatch.undo()

    def by_g(kets):
        ks = list(kets)
        return [ks[i] for i in sorted(range(len(ks)),
                                      key=lambda i: G[rec[i]], reverse=True)]

    J0 = aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=0.0)
    ref = aoj.coulomb_matrix_screened(basis, D, pq=pq, tau=tau,
                                      order="unsorted")
    assert ref.n_skipped > 0  # the screen acts at this tau

    # (b) the product order: the break is exact
    prod = walk()
    assert prod.J == ref.J
    assert prod.n_skipped == ref.n_skipped
    assert not _exceeds(prod.J, J0.J, prod.scale, tau)

    # (c) another key: a different J, outside the stated bound
    for name, order in (("G alone", by_g), ("pair order", list)):
        wrong = walk(order)
        assert wrong.J != ref.J, name
        assert _exceeds(wrong.J, J0.J, wrong.scale, tau), name

    # (d) the global-Dmax fallback, as the screen's key: sound, looser
    dglobal = max(dmax.values())
    loose = walk(lambda kets: product_sort(
        [dataclasses.replace(k, bound=G[r] * dglobal)
         for r, k in zip(rec, kets, strict=True)]))
    assert not _exceeds(loose.J, J0.J, loose.scale, tau)
    assert loose.n_skipped <= prod.n_skipped


# --- the J plans, unchanged by the K plan (#204) ------------------------------

# sha256 of `_j_plans_dump()`, taken on main at af3b0dc, BEFORE #204 touched
# launch.py. #204's acceptance: "a diff of the planned J launches before and
# after is empty". Every record (shell pair, prim, L, G, key -- floats by
# repr, so bit for bit), every class's batch count and bra blocks, each
# class's kernels (by the sha256 of their repr), and the plan's tau, batch
# count, density_sort and refusals, over both bounds, both sort keys, two
# taus and two batch counts on H2O/STO-3G, and the dimer.
J_PLANS_SHA256 = (
    "27ec51f766a2e629a827ed2ef1d07dc5d2417cbece5fbb7755550816688a7ce5")


def _j_plans_dump() -> str:
    def dump(P):
        out = [repr((P.tau, P.n_batches, P.density_sort, P.refused))]
        for c in P:
            out.append(repr((c.lp, c.lq, c.n_batches, c.bras, c.kets,
                             c.bra_blocks)))
            out.append(hashlib.sha256(repr(c.kernels).encode()).hexdigest())
        return "\n".join(out)

    cases = []
    basis, D = _h2o()
    for bound in (BOUND, partial(aoj.pair_bound_pq, pq=pq)):
        for tau in (0.0, 1e-6):
            for ds in (True, False):
                for n in (1, 3):
                    cases.append(dump(launches(
                        basis, D, bound=bound, n_batches=n, tau=tau,
                        density_sort=ds, strict=False)))
    basis, D = _dimer()
    cases.append(dump(launches(basis, D, bound=BOUND, n_batches=1, tau=0.0)))
    return "\n#\n".join(cases)


def test_the_j_plans_are_unchanged_by_the_k_plan():
    got = hashlib.sha256(_j_plans_dump().encode()).hexdigest()
    assert got == J_PLANS_SHA256


# --- #241: the one-electron launch, BESIDE ClassLaunch ------------------------
#
# `one_electron_launch` is a sibling of `ClassLaunch`, not a widening: J's
# types keep their fields (pinned here) and their plans their fingerprint
# (`test_the_j_plans_are_unchanged_by_the_k_plan`, above, re-run unchanged).
# One record set serves S, T and V: its per-kernel arrays equal what each
# kernel's own test feeds that kernel (test_kernel_{s,t,v}.py's `_records`),
# under the kernel's parameter names and in its parameter block's order.

import math  # noqa: E402

from test_kernel_s import _records as _s_records  # noqa: E402
from test_kernel_t import _records as _t_records  # noqa: E402
from test_kernel_v import _records as _v_records  # noqa: E402

from felizjueves.ao.pairs import (  # noqa: E402
    kinetic_bound,
    overlap_bound,
    potential_bound,
)
from felizjueves.kernel_s import build_s_kernel, s_spec  # noqa: E402
from felizjueves.kernel_t import build_t_kernel, t_spec  # noqa: E402
from felizjueves.kernel_v import build_v_kernel, v_spec  # noqa: E402
from felizjueves.launch import (  # noqa: E402
    LaunchPlan,
    OneEClass,
    OneELaunch,
    OneERec,
    one_electron_launch,
    walk_one_electron,
)
from felizjueves.spec import Target  # noqa: E402

_ONE_E = {Target.S: (s_spec, build_s_kernel, _s_records),
          Target.T: (t_spec, build_t_kernel, _t_records),
          Target.V: (v_spec, build_v_kernel, _v_records)}
_NON_RECORD = {"n_rec", "n_ao", "n_nuc", "Z", "Cnuc", "S", "T", "V"}


@cache
def _spdf():
    fx = manifest.load(manifest.FIXTURES / "prim_spdf.json")
    nuclei = tuple((float(a.charge), tuple(float(x) for x in a.coord))
                   for a in fx.atoms)
    return ob.from_fixture(fx), nuclei


def _h2o_1e():
    fx = manifest.load(manifest.FIXTURES / "h2o_sto3g_j.json")
    return _h2o()[0], tuple((float(a.charge), tuple(a.coord))
                            for a in fx.atoms)


def _plan_1e(basis, nuclei, tau=0.0, bound=overlap_bound):
    return one_electron_launch(basis, nuclei, tau=tau, bound=bound)


def test_j_shaped_types_keep_their_fields():
    # The sibling type was the call: nothing of J's was widened for it.
    def names(cls):
        return [f.name for f in dataclasses.fields(cls)]

    assert names(PairRec) == ["i", "j", "prim", "L", "G", "key"]
    assert names(ClassLaunch) == ["lp", "lq", "n_batches", "bras", "kets",
                                  "kernels", "bra_blocks"]
    assert names(LaunchPlan) == ["classes", "tau", "n_batches",
                                 "density_sort", "refused"]


@pytest.mark.parametrize("case", ["h2o", "spdf"])
def test_every_record_of_every_kept_pair_once_in_pair_order(case):
    basis, nuclei = _h2o_1e() if case == "h2o" else _spdf()
    plan = _plan_1e(basis, nuclei)
    sh, off = basis.shells, basis.offsets
    assert [(c.la, c.lb) for c in plan] == sorted(
        {(sh[i].l, sh[j].l) for i, j in plan.kept})
    seen = []
    for c in plan:
        pairs = [(i, j) for i, j, _, _ in c.blocks]
        assert pairs == [p for p in plan.kept
                         if (sh[p[0]].l, sh[p[1]].l) == (c.la, c.lb)]
        for i, j, start, stop in c.blocks:
            recs = c.recs[start:stop]
            pps = primitive_pairs(sh[i], sh[j])
            assert [r.prim for r in recs] == list(range(len(pps)))
            for r, pp in zip(recs, pps, strict=True):
                assert r.shell_pair == (i, j) and i <= j
                assert (r.off_a, r.off_b) == (off[i], off[j])
                assert r.diagonal == (i == j)
                assert (r.P, r.zeta) == (pp.P, pp.zeta)
                assert r.eb == sh[j].exponents[pp.j]
                seen.append((i, j, r.prim))
    assert len(seen) == len(set(seen)) == sum(
        len(primitive_pairs(sh[i], sh[j])) for i, j in plan.kept)


@pytest.mark.parametrize("target", [Target.S, Target.T, Target.V])
def test_one_record_set_carries_exactly_what_each_kernel_reads(target):
    spec_of, build, records = _ONE_E[target]
    basis, nuclei = _spdf()
    plan = _plan_1e(basis, nuclei)
    sh = basis.shells
    for c in plan:
        assert c.spec(target) == spec_of(c.la, c.lb)
        got = c.arrays(target)
        params = [p.name for p in build(c.spec(target)).params
                  if p.name not in _NON_RECORD]
        assert list(got) == params, (target, c.la, c.lb)
        pairs = [p for p in plan.kept
                 if (sh[p[0]].l, sh[p[1]].l) == (c.la, c.lb)]
        want = records(basis, pairs)
        assert set(want) == set(got)
        for k in want:
            assert np.array_equal(np.asarray(got[k], dtype=want[k].dtype),
                                  want[k]), (target, c.la, c.lb, k)
    if target is Target.V:
        assert plan.Z == [z for z, _ in nuclei]
        assert plan.Cnuc == [list(cn) for _, cn in nuclei]


def test_the_records_do_not_depend_on_the_matrix():
    # S, T and V of one basis: the same records, whichever bound screened
    # them at tau = 0 (it drops nothing), and the same records as a bound
    # that is everywhere 0.
    basis, nuclei = _h2o_1e()
    plans = [_plan_1e(basis, nuclei, bound=b) for b in (
        overlap_bound, kinetic_bound, lambda sa, sb: 0.0,
        lambda sa, sb: potential_bound(sa, sb, nuclei,
                                       one_centre=one_centre))]
    assert all(p == plans[0] for p in plans)
    assert plans[0].dropped == () and plans[0].tau == 0.0


def test_a_dropped_pair_leaves_its_records_and_maybe_its_class():
    basis, nuclei = _h2o_1e()
    plan = _plan_1e(basis, nuclei, tau=1e-12)
    assert plan.dropped  # same-centre s-p pairs, zero blocks
    recs = {r.shell_pair for c in plan for r in c.recs}
    assert recs == set(plan.kept) and not recs & set(plan.dropped)
    every = _plan_1e(basis, nuclei, tau=1e300)
    assert len(every) == 0 and every.kept == () and every.skip_fraction == 1.0


def test_the_one_electron_plan_refuses():
    basis, nuclei = _h2o_1e()
    with pytest.raises(ValueError, match="tau"):
        _plan_1e(basis, nuclei, tau=-1.0)
    with pytest.raises(ValueError, match="bound"):
        _plan_1e(basis, nuclei, bound=lambda sa, sb: math.nan)
    with pytest.raises(ValueError, match="nucleus"):
        _plan_1e(basis, ((1.0,),))
    with pytest.raises(ValueError, match="absent"):
        OneEClass(0, 0, (), ())
    plan = _plan_1e(basis, nuclei)
    with pytest.raises(ValueError, match="one-electron"):
        plan.classes[0].spec(Target.J)
    with pytest.raises(ValueError, match="one-electron"):
        walk_one_electron(plan, basis, Target.J)
    with pytest.raises(ValueError, match="one_centre"):
        walk_one_electron(plan, basis, Target.V)
    assert isinstance(plan, OneELaunch)
    assert all(isinstance(r, OneERec) for c in plan for r in c.recs)
