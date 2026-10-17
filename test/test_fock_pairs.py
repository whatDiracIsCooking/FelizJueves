"""The fused Fock build's pair stream (#285): `launch_f` and the bound it reads.

Fast tier, no device markers: pure Python, no compiler, no card, no PySCF.
A plan of a fixture whose bound reaches L > FAST_L_MAX (tiers.py) -- the
Schwarz bound of its highest pair `(ab|ab)`, L = 2 (la + lb), so any d shell
-- is `slow`.
The bases and densities come from tracked fixtures' `converted` records;
`reference/` is imported HERE, by the test, for the evaluator the bound is
injected with and for the ERIs the screens are checked against -- `launch_f`
never reaches it (test_layering.py, rule "f_launch").

THE CLAIMS (the issue's acceptance, in order).

  bound      `ao.pairs.primitive_schwarz_bound` is the existing `I_p` read
             through the dressed `E` (an `s`-`s` pair pinned to
             `hermite_bound`), and `G_ab G_cd` bounds every element of every
             contracted block of H2O/STO-3G -- exact AND pruned, the pruned
             one by the KEPT sums too.
  degree     every pair of a batch shares class, magnitude and `K_AB =
             len(prims)`, and the packed record is rows of exactly `K_AB`
             real entries: no padding.
  eq. 29     the magnitudes are `tau^(i/n)`, and `m_i m_j <= tau` exactly
             when `i + j >= n`, read on the exponents as fractions for every
             `n <= 12`; the pairs are assigned the magnitude above them; the
             batch pairs removed before launch are exactly `i + j >= n`, are
             absent from the launch list, and every ERI they hold is `< tau`
             (computed, on fixtures where the screen fires). Fig. 1's launch
             matrix at `n = 2` is reproduced.
  printed    the two defect records of #271 (asserted by #291 in
             test_fock_claims.py, which carries them to the two-electron F;
             these are the ERI-level halves): the printed assignment
             (largest magnitude BELOW G) lets
             the eq. 29 screen remove a quartet holding an ERI `>= tau`, and
             the printed inequality `i + j <= n` removes one too -- each
             shown on a computed ERI.
  pruning    bound-ordered, the kept set a prefix, the deleted sum `< eps`,
             no ERI moved by `tau` or more, a deleted primitive absent from
             the record.
  sort       inside a batch, descending by `G * max|D_ab|`, stable.
  limits     the counts limits.json records on the compact fixtures are
             recomputed here, exactly.

No tolerance is chosen: the bounds are checked as plain `<=` / `<` on the
computed blocks, and the counts are exact.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from functools import cache, partial
from pathlib import Path

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l, is_fast

from felizjueves.ao.eri import _quartet
from felizjueves.ao.hermite import hermite_order
from felizjueves.ao.pairs import (
    hermite_bound,
    primitive_pairs,
    primitive_schwarz_bound,
    primitive_schwarz_bounds,
    prune,
)
from felizjueves.ao.shell import Basis, Shell
from felizjueves.launch_f import (
    FBatch,
    FPlan,
    exponent,
    f_launches,
    magnitude_index,
    magnitudes,
    prelaunch_screened,
    printed_magnitude_index,
    printed_prelaunch_screened,
)
from felizjueves.reference.pq import one_centre, pq
from felizjueves.spec import Spec, Target

ROOT = Path(__file__).resolve().parents[1]
LIMITS = ROOT / "data" / "literature" / "fock" / "limits.json"

BOUND = partial(primitive_schwarz_bounds, one_centre=one_centre)

# Milestone 12's ladder: every shell l <= 2, pair classes la <= lb.
PAIR_CLASSES = [(la, lb) for la in range(3) for lb in range(la, 3)]
LADDER = [(*p, *q) for i, p in enumerate(PAIR_CLASSES)
          for q in PAIR_CLASSES[i:]]


@cache
def _fixture(name: str) -> tuple[Basis, np.ndarray]:
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return ob.from_fixture(fx), np.array(fx.density)


@cache
def _plan(name: str, tau: float, n: int) -> FPlan:
    basis, D = _fixture(name)
    return f_launches(basis, D, bound=BOUND, tau=tau, n=n)


def _schwarz_l(name: str) -> int:
    """The L of the highest pair bound `name`'s plan computes: `(ll|ll)`."""
    return 4 * max(s.l for s in _fixture(name)[0].shells)


def _by_fixture(name: str, *rest):
    """`(name, *rest)` as a parameter, `slow` above the cut."""
    return pytest.param(name, *rest,
                        marks=by_l(None, _schwarz_l(name)).marks)


def _block(basis: Basis, bra: tuple[int, int], ket: tuple[int, int],
           keep_bra=None, keep_ket=None) -> list[list[float]]:
    """`(ab|cd)` from the given primitive pairs only (all when None), by
    ao/eri.py's own primitive quartet."""
    sh = basis.shells
    sa, sb = sh[bra[0]], sh[bra[1]]
    sc, sd = sh[ket[0]], sh[ket[1]]
    bras = primitive_pairs(sa, sb)
    kets = primitive_pairs(sc, sd)
    if keep_bra is not None:
        bras = [bras[k] for k in keep_bra]
    if keep_ket is not None:
        kets = [kets[k] for k in keep_ket]
    ps, qs = hermite_order(sa.l + sb.l), hermite_order(sc.l + sd.l)
    out = [[0.0] * (sc.n_cart * sd.n_cart) for _ in range(sa.n_cart * sb.n_cart)]
    for x in bras:
        for y in kets:
            blk, _ = _quartet(x, y, ps, qs, pq, False)
            for r, row in zip(out, blk, strict=True):
                for k, v in enumerate(row):
                    r[k] += v
    return out


def _max_abs(blk) -> float:
    return max(abs(v) for row in blk for v in row)


def _unique_pairs(basis: Basis) -> list[tuple[int, int]]:
    sh = basis.shells
    out = []
    for i in range(len(sh)):
        for j in range(i, len(sh)):
            out.append((i, j) if (sh[i].l, i) <= (sh[j].l, j) else (j, i))
    return out


# -- the bound ---------------------------------------------------------------


def test_an_s_s_primitive_bound_is_the_existing_hermite_bound() -> None:
    basis, _ = _fixture("h2o_sto3g_j")
    sh = basis.shells
    s = [k for k, x in enumerate(sh) if x.l == 0]
    for pp in primitive_pairs(sh[s[0]], sh[s[-1]]):
        want = abs(pp.block[0][0]) * hermite_bound(pp.zeta, (0, 0, 0),
                                                   one_centre=one_centre)
        assert primitive_schwarz_bound(pp, one_centre=one_centre) == want


def test_the_contracted_bound_bounds_every_block_exact_and_pruned() -> None:
    # H2O/STO-3G, every unique pair against every unique pair (both orders
    # are the same block transposed), at a tau where pruning fires.
    basis, _ = _fixture("h2o_sto3g_j")
    plan = _plan("h2o_sto3g_j", 1e-2, 2)
    assert plan.n_prim_pruned > 0
    byp = {p.shell_pair: p for p in plan.pairs}
    assert len(byp) == len(_unique_pairs(basis))
    pairs = sorted(byp)
    for x, bra in enumerate(pairs):
        for ket in pairs[x:]:
            P, Q = byp[bra], byp[ket]
            exact = _block(basis, bra, ket)
            pruned = _block(basis, bra, ket, P.prims, Q.prims)
            assert _max_abs(exact) <= P.G * Q.G
            kept = math.fsum(P.prim_G) * math.fsum(Q.prim_G)
            assert _max_abs(pruned) <= kept
            # pruning moves no ERI by tau or more (launch_f, "PRUNING")
            diff = max(abs(u - v) for ru, rv in zip(exact, pruned, strict=True)
                       for u, v in zip(ru, rv, strict=True))
            assert diff < plan.tau


# -- pruning -----------------------------------------------------------------


def test_prune_keeps_a_bound_ordered_prefix() -> None:
    g = [0.5, 1e-9, 0.25, 3e-9, 1e-9, 2.0]
    order = [5, 0, 2, 3, 1, 4]  # descending, stable on the tie (1 before 4)
    assert prune(g, eps=0.0) == tuple(order)
    # tail sums of the order: 1e-9, 2e-9, 5e-9, 0.25+5e-9, ...
    assert prune(g, eps=5e-9) == tuple(order[:4])
    assert prune(g, eps=6e-9) == tuple(order[:3])
    assert prune(g, eps=10.0) == ()
    for eps in (0.0, 1.5e-9, 5e-9, 6e-9, 0.3, 10.0):
        keep = prune(g, eps=eps)
        assert keep == tuple(order[:len(keep)])
        assert math.fsum(g[k] for k in order[len(keep):]) < eps or (
            len(keep) == len(g))


@pytest.mark.parametrize("bad", [
    pytest.param({"eps": -1.0}, id="negative-eps"),
    pytest.param({"eps": math.nan}, id="nan-eps"),
    pytest.param({"eps": 0.0, "g": [1.0, -1e-3]}, id="negative-bound"),
    pytest.param({"eps": 0.0, "g": [math.inf]}, id="infinite-bound"),
])
def test_prune_refuses(bad) -> None:
    with pytest.raises(ValueError):
        prune(bad.get("g", [1.0]), eps=bad["eps"])


@pytest.mark.parametrize("name", ["h2o_sto3g_j", "h2o_631g_j"])
def test_the_plan_prunes_by_bound_and_records_only_the_kept(name) -> None:
    basis, _ = _fixture(name)
    plan = _plan(name, 1e-3, 2)
    assert plan.n_prim_pruned > 0
    assert plan.eps == plan.tau / (2.0 * plan.g_max)
    for p in plan.pairs:
        g = BOUND(basis.shells[p.a], basis.shells[p.b])
        assert p.n_prim == len(g) and p.G == math.fsum(g)
        order = sorted(range(len(g)), key=lambda k: -g[k])
        assert p.prims == tuple(order[:p.degree])  # a prefix of the order
        assert p.prim_G == tuple(g[k] for k in p.prims)
        dropped = order[p.degree:]
        assert math.fsum(g[k] for k in dropped) < plan.eps
        # the next one would have pushed the deleted sum past eps
        if p.degree:
            assert math.fsum(g[k] for k in order[p.degree - 1:]) >= plan.eps
        # a deleted primitive pair is absent from the staged record
        rows = [r for b in plan.batches for r in b.packed
                if r and r[0][:2] == p.shell_pair]
        assert len(rows) == 1
        assert {k for _, _, k in rows[0]}.isdisjoint(dropped)


def test_at_tau_zero_nothing_is_pruned_screened_or_deleted() -> None:
    plan = _plan("h2o_631g_j", 0.0, 2)
    assert plan.eps == 0.0 and plan.magnitudes == (1.0, 0.0)
    assert plan.n_prim_pruned == 0 and not plan.deleted and not plan.screened
    assert all(p.mag == 0 and p.degree == p.n_prim for p in plan.pairs)


def test_an_insignificant_pair_is_deleted_whole() -> None:
    # Two H2 molecules far apart: the cross pairs decay as exp(-mu R^2), so
    # at a loose tau every primitive pair of a cross pair is pruned.
    basis, _ = _fixture("h2_sto3g_j")
    h2 = basis.shells
    far = tuple(Shell(s.l, (s.centre[0], s.centre[1], s.centre[2] + 12.0),
                      s.exponents, s.coefficients) for s in h2)
    two = Basis(h2 + far)
    D = np.eye(two.n_ao)
    plan = f_launches(two, D, bound=BOUND, tau=1e-6, n=2)
    assert plan.deleted
    kept = {p.shell_pair for p in plan.pairs}
    for a, b in plan.deleted:
        assert (a, b) not in kept
        G = math.fsum(BOUND(two.shells[a], two.shells[b]))
        assert G < plan.eps  # so G_ab G_cd < tau / 2 for every partner
    assert plan.n_pairs_total == len(_unique_pairs(two))


# -- degree-uniform batches --------------------------------------------------


@pytest.mark.parametrize("name,tau", [_by_fixture("h2o_sto3g_j", 1e-2),
                                      _by_fixture("h2o_631g_j", 1e-10),
                                      _by_fixture("h2o_ccpvdz_j", 1e-10)])
def test_every_batch_is_uniform_and_its_record_unpadded(name, tau) -> None:
    basis, _ = _fixture(name)
    plan = _plan(name, tau, 2)
    seen = []
    for b in plan.batches:
        for p in b.pairs:
            assert (p.la, p.lb) == b.cls and p.mag == b.mag
            assert p.degree == b.degree == len(p.prims)
            assert (basis.shells[p.a].l, basis.shells[p.b].l) == (p.la, p.lb)
            assert p.la <= p.lb
            seen.append(p.shell_pair)
        rec = b.packed
        assert len(rec) == len(b.pairs)
        assert all(len(row) == b.degree for row in rec)  # no padding
        assert all(isinstance(k, int) and 0 <= k < p.n_prim
                   for p, row in zip(b.pairs, rec, strict=True)
                   for _, _, k in row)
    # every unique shell pair once: in exactly one batch, or deleted
    assert sorted(seen + list(plan.deleted)) == sorted(_unique_pairs(basis))
    keys = [(b.la, b.lb, b.degree, b.mag) for b in plan.batches]
    assert keys == sorted(set(keys))


def test_a_batch_refuses_a_pair_of_another_degree() -> None:
    plan = _plan("h2o_631g_j", 1e-10, 2)
    ss = [x for x in plan.batches if x.cls == (0, 0)]
    a = ss[0]
    b = next(x for x in ss if x.degree != a.degree)
    with pytest.raises(ValueError):
        FBatch(a.la, a.lb, a.degree, a.mag, a.magnitude, a.pairs + b.pairs)


def test_pruning_does_split_degrees_on_the_fixtures() -> None:
    # A consequence worth seeing: the degree is the KEPT count, so pruning
    # adds bins (6-31G at 1e-10 already prunes).
    unpruned = _plan("h2o_631g_j", 0.0, 2)
    pruned = _plan("h2o_631g_j", 1e-10, 2)
    assert pruned.n_prim_pruned > 0
    assert {b.degree for b in pruned.batches} > {
        b.degree for b in unpruned.batches}


# -- the sort ----------------------------------------------------------------


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py): cc-pVDZ's d pairs
def test_pairs_sit_descending_by_g_times_their_density_block_max() -> None:
    basis, D = _fixture("h2o_ccpvdz_j")
    plan = _plan("h2o_ccpvdz_j", 1e-10, 2)
    off = basis.offsets
    pos = {pr: k for k, pr in enumerate(_unique_pairs(basis))}
    for b in plan.batches:
        for p in b.pairs:
            sa, sb = basis.shells[p.a], basis.shells[p.b]
            blk = D[off[p.a]:off[p.a] + sa.n_cart, off[p.b]:off[p.b] + sb.n_cart]
            assert p.dmax == float(np.max(np.abs(blk)))
            assert p.key == p.G * p.dmax
        for x, y in zip(b.pairs, b.pairs[1:], strict=False):
            assert x.key > y.key or (x.key == y.key
                                     and pos[x.shell_pair] < pos[y.shell_pair])


# -- eq. 29 ------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 13))
def test_eq29_magnitudes_symbolically(n) -> None:
    tau = 1e-10
    mags = magnitudes(tau, n)
    assert mags[0] == 1.0 and len(mags) == n
    for i in range(n):
        assert exponent(i, n) == Fraction(i, n)
        assert mags[i] == tau ** (i / n)
        for j in range(n):
            # m_i m_j = tau^((i+j)/n) <= tau  <=>  (i+j)/n >= 1  <=>  i+j >= n
            assert prelaunch_screened(i, j, n) == (i + j >= n)
            assert printed_prelaunch_screened(i, j, n) == (i + j <= n)


def test_the_assignment_is_the_upper_bound_one() -> None:
    mags = magnitudes(1e-6, 3)  # 1, ~1e-2, ~1e-4
    m1, m2 = mags[1], mags[2]
    def below(x: float) -> float:
        return math.nextafter(x, 0.0)

    cases = {5.0: 0, 1.0: 0, 0.5: 0, m1: 0, below(m1): 1, m2: 1,
             below(m2): 2, 1e-12: 2, 0.0: 2}
    for G, i in cases.items():
        assert magnitude_index(G, mags) == i, G
        if i:
            assert G < mags[i]  # the magnitude bounds the pair from above
    # the printed one: the largest magnitude BELOW G -- a lower bound
    printed = {5.0: 0, 0.5: 1, m1: 2, below(m1): 2, 1e-12: 2}
    for G, i in printed.items():
        assert printed_magnitude_index(G, mags) == i, G
    assert mags[printed_magnitude_index(0.5, mags)] < 0.5


@pytest.mark.claim("fock/prelaunch-screening-removes-launches")
@pytest.mark.parametrize("name,tau", [_by_fixture("h2o_sto3g_j", 1e-2),
                                      _by_fixture("h2o_631g_j", 1e-3),
                                      _by_fixture("h2o_631g_j", 1e-2),
                                      _by_fixture("h2o_ccpvdz_j", 1e-3)])
def test_the_prelaunch_screen_removes_exactly_i_plus_j_at_least_n(name,
                                                                   tau) -> None:
    # fock/prelaunch-screening-removes-launches, the mechanism only: on eq.
    # 29's magnitudes the batch pairs with i + j >= n are absent from the
    # launch list (and every other one is in it), and no quartet they hold
    # has G_ab G_cd >= tau. How many launches that saves is not scored: the
    # fixtures are compact (pin 9; limits.json's screen-efficacy record).
    n = 2
    plan = _plan(name, tau, n)
    mags = plan.magnitudes
    for p in plan.pairs:
        assert p.mag == magnitude_index(p.G, mags)
        if p.mag:
            assert p.G < mags[p.mag]
    nb = len(plan.batches)
    want = {(i, j) for i in range(nb) for j in range(i, nb)
            if plan.batches[i].mag + plan.batches[j].mag >= n}
    assert want and set(plan.screened) == want  # it fires on these cases
    launched = {(x.bra, x.ket) for x in plan}
    assert launched.isdisjoint(want)  # a screened batch pair is never launched
    assert launched | want == {(i, j) for i in range(nb) for j in range(i, nb)}
    for i, j in plan.screened:
        for P in plan.batches[i].pairs:
            for Q in plan.batches[j].pairs:
                assert P.G * Q.G < tau


@pytest.mark.parametrize("name,tau", [("h2o_sto3g_j", 1e-2),
                                      ("h2o_631g_j", 1e-2)])
def test_every_eri_of_a_screened_batch_pair_is_below_tau(name, tau) -> None:
    # The screen checked against the answer: each quartet the plan never
    # launches, computed exactly (every primitive pair), is below tau.
    basis, _ = _fixture(name)
    plan = _plan(name, tau, 2)
    assert plan.screened
    n_quartets = 0
    for i, j in plan.screened:
        for P in plan.batches[i].pairs:
            for Q in plan.batches[j].pairs:
                assert _max_abs(_block(basis, P.shell_pair,
                                       Q.shell_pair)) < tau
                n_quartets += 1
    assert n_quartets


def test_fig1_kernel_launches_at_n_2() -> None:
    # Fig. 1's panel: magnitudes 1 and sqrt(tau); every batch pair holding a
    # magnitude-1 batch is launched, and no (sqrt(tau), sqrt(tau)) one.
    tau = 1e-8
    assert magnitudes(tau, 2) == (1.0, tau ** 0.5)
    built = {(i, j): not prelaunch_screened(i, j, 2)
             for i in range(2) for j in range(2)}
    assert built == {(0, 0): True, (0, 1): True, (1, 0): True, (1, 1): False}
    # and the printed inequality, read as removal, launches nothing at n = 2
    assert all(printed_prelaunch_screened(i, j, 2)
               for i in range(2) for j in range(2))


@pytest.mark.parametrize("name,tau", [_by_fixture("h2o_ccpvdz_j", 1e-10),
                                      _by_fixture("h2o_631g_j", 1e-2)])
def test_launches_are_the_upper_triangle_and_name_the_f_class(name,
                                                              tau) -> None:
    plan = _plan(name, tau, 2)
    for x in plan:
        bra, ket = plan.batches[x.bra], plan.batches[x.ket]
        assert x.bra <= x.ket and bra.cls <= ket.cls
        assert x.spec == Spec(x.spec.operator, 0, (*bra.cls, *ket.cls),
                              Target.F)
        assert x.spec.am in LADDER and x.spec.n_batches == 1
    assert len(LADDER) == 21


# -- the printed text, against computed ERIs (#271's defect records) ---------


def test_the_printed_assignment_screens_a_quartet_above_tau() -> None:
    # fock/batch-magnitude-bound-direction: assign by the text (largest
    # magnitude below G), screen by eq. 29 (i + j >= n), and a removed
    # quartet holds an ERI >= tau. The built plan removes none of it.
    name, tau, n = "h2o_sto3g_j", 1e-3, 2
    basis, _ = _fixture(name)
    plan = _plan(name, tau, n)
    mags = plan.magnitudes
    low = [p for p in plan.pairs if printed_magnitude_index(p.G, mags) == 1]
    assert low and prelaunch_screened(1, 1, n)
    P = max(low, key=lambda p: p.G)
    assert P.G * P.G >= tau
    assert _max_abs(_block(basis, P.shell_pair, P.shell_pair)) >= tau
    # the built assignment puts it in magnitude 0, which is never screened
    assert P.mag == 0 and not plan.screened


def test_the_printed_inequality_screens_a_quartet_above_tau() -> None:
    # fock/eq29-screen-inequality-reversed: read as removal, i + j <= n
    # removes (0, 0) -- the largest pairs.
    name, tau, n = "h2o_sto3g_j", 1e-10, 2
    basis, _ = _fixture(name)
    plan = _plan(name, tau, n)
    assert printed_prelaunch_screened(0, 0, n)
    assert not prelaunch_screened(0, 0, n)
    P = max(plan.pairs, key=lambda p: p.G)
    assert P.mag == 0
    assert _max_abs(_block(basis, P.shell_pair, P.shell_pair)) >= tau


# -- refusals ----------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    pytest.param({"tau": -1e-10}, id="negative-tau"),
    pytest.param({"tau": math.nan}, id="nan-tau"),
    pytest.param({"n": 0}, id="n-zero"),
    pytest.param({"n": 2.0}, id="n-float"),
])
def test_the_plan_refuses_bad_parameters(kw) -> None:
    basis, D = _fixture("h2_sto3g_j")
    args = {"tau": 1e-10, "n": 2} | kw
    with pytest.raises(ValueError):
        f_launches(basis, D, bound=BOUND, **args)


@pytest.mark.parametrize("bound", [
    pytest.param(lambda sa, sb: (1.0,), id="wrong-length"),
    pytest.param(lambda sa, sb: (-1.0,) * (len(sa.exponents)
                                           * len(sb.exponents)),
                 id="negative"),
])
def test_the_plan_refuses_a_bad_bound(bound) -> None:
    basis, D = _fixture("h2_sto3g_j")
    with pytest.raises(ValueError):
        f_launches(basis, D, bound=bound, tau=1e-10, n=2)


def test_the_plan_refuses_an_asymmetric_density() -> None:
    basis, D = _fixture("h2_sto3g_j")
    D = D.copy()
    D[0, 1] += 1.0
    with pytest.raises(ValueError):
        f_launches(basis, D, bound=BOUND, tau=1e-10, n=2)


# -- the record ----------------------------------------------------------------


def _pair_stream_record() -> dict:
    recs = [r for r in json.loads(LIMITS.read_text())["limits"]
            if r["claim"].startswith("The pair stream (#285)")]
    assert len(recs) == 1
    return recs[0]


def test_limits_records_the_counts_that_fire_on_the_fixtures() -> None:
    _the_counts_that_fire(lambda name: is_fast(_schwarz_l(name)))


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py): every fixture, cc-pVDZ's too
def test_limits_records_the_counts_that_fire_on_every_fixture() -> None:
    _the_counts_that_fire(lambda name: True)


def _the_counts_that_fire(keep) -> None:
    rec = _pair_stream_record()
    value = rec["asserted_value"]
    assert value["measured"] is False and value["n"] == 2
    fixtures = {k: v for k, v in value["fixtures"].items() if keep(k)}
    got = {name: {f"{tau:g}": _plan(name, tau, value["n"]).counts()
                  for tau in value["taus"]}
           for name in fixtures}
    assert got == fixtures
    # what the record says, recomputed: at the SCF-like tau nothing is
    # screened before launch on any fixture, though pruning already fires
    at = f"{min(value['taus']):g}"
    assert all(c[at]["batch_pairs_screened"] == 0
               for c in got.values())
    assert any(c[at]["primitive_pairs_pruned"] for c in got.values())
    # "the smallest contracted G_ab is above 0.02 on every fixture"
    assert all(min(p.G for p in _plan(name, 0.0, 2).pairs) > 0.02
               for name in fixtures)
    assert "follow-up" in rec["mechanism"]
