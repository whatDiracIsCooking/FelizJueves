"""K's Schwarz screen (#200): the bound, proved; the break and the continue;
and the sort the J engine cannot use.

ao/k.py's module docstring, "THE SCREEN", states it: a (bra record `(a,c)`,
ket record `(b,d)`) pair is skipped when `(I_ac * I_bd) * Dm < tau`, with
`I_ac = max_i sum_k sum_p |E^{a_i c_k}_p| I_p` built on `ao.pairs`'
pair-factor-free `I_p` and `Dm = max|D|`, and the kets of the kernel's
sorted `break` ordered descending by `I_bd` ALONE.

THE CLAIMS.

  the bound   |X_ij| <= I_ac I_bd max|D_cd| for every element of every record
              pair's contribution, over randomised records (fixed seed,
              N_QUARTETS shell quartets, all 15 classes with L <= 2, 1..3
              primitives a shell, three quarters of the draws with
              a shared centre). `max|D_cd|` of the quartet's own block is the
              strongest form: `Dm >= max|D_cd|`. `I_p` is pinned equal to
              `ao.pairs.hermite_bound` (the proved one), and `I_ac` to the
              same sum over `ao.pairs.primitive_pairs`' dressed block.
  exact       sorted-`break` K == unsorted-`continue` K TO 0 at every rung,
              the same skip count and the same skipped bounds (`ao.k`
              accumulates each element with `math.fsum`, so this is a claim
              about WHICH terms were summed); the break examines no more.
  bounded     |K(tau) - K(0)| <= tau * scale element by element, and every
              skipped bound is < tau.
  zero        at tau = 0 nothing is skipped and NO bound is evaluated: no
              screen test is counted, and the `[p|q]` evaluator sees exactly
              the calls of the unscreened sweep -- every bound evaluation is
              a `pq` call, so one evaluated bound would show.
  continue    `exchange_matrix(tau)` and `exchange_matrix_reference(tau)`
              (the continue inside the quartet contraction) agree with the
              screened sweep to rounding at every rung.
  the J key   sorting the kets by `I_bd * max|D_bd|` (the J engine's key,
              misapplied: the ket's own density block, `Dmax`'s shape)
              makes the break skip a pair whose bound is >= tau, and the
              matrix moves -- on every fast case, at some rung. So the
              break's precondition is the Schwarz-only order, tested.

THE L <= 2 RULE (milestone 10: no quartet class with
`L = la + lc + lb + ld > 2` is evaluated before #231, host-side included).
The randomised bound test draws only L <= 2 classes. The fast sweeps are
all-s: H2/STO-3G, H2O/STO-3G's four s shells (with the matching block of
the fixture density), and two H2 6 bohr apart with the hilbert density
(the extended case of test_ao_screening.py, shortened). Whole H2O/STO-3G
reaches (pp|pp), L = 4: written below, marked `slow`, deferred to #231.

THE TOLERANCES, stated and not tuned: the bound and the break are exact (no
tolerance); `continue` is SWEEP_TOL = 1e-14 relative to max|K(0)| (naive
accumulation against fsum, test_ao_k.py's figure).
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao import k as aok
from felizjueves.ao.j import epair
from felizjueves.ao.k import (
    exchange_matrix,
    exchange_matrix_reference,
    exchange_matrix_screened,
    k_quartet,
    record_bounds,
)
from felizjueves.ao.pairs import hermite_bounds, primitive_pairs
from felizjueves.ao.shell import Basis, Shell
from felizjueves.reference.pq import one_centre, pq

TAUS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
SWEEP_TOL = 1e-14
L_MAX_RUN = 2  # the milestone's rule: nothing above this is evaluated
N_QUARTETS = 600  # the randomised bound test's sample size, in shell quartets
SEED = 200


# --- the cases ---------------------------------------------------------------


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return ob.from_fixture(fx), np.array(fx.density)


def _h2_chain(n: int) -> Basis:
    s = _fixture("h2_sto3g_j")[0].shells[0]
    return Basis(tuple(
        Shell(0, (0.0, 0.0, 6.0 * m + dz), s.exponents, s.coefficients)
        for m in range(n) for dz in (0.0, 1.4)))


@cache
def _case(name: str) -> tuple[Basis, np.ndarray]:
    if name == "h2_sto3g_j" or name == "h2o_sto3g_j":
        return _fixture(name)
    if name == "h2o_sto3g_s":
        basis, D = _fixture("h2o_sto3g_j")
        keep = [i for i, s in enumerate(basis.shells) if s.l == 0]
        aos = [basis.offsets[i] for i in keep]
        return Basis(tuple(basis.shells[i] for i in keep)), D[np.ix_(aos, aos)]
    assert name == "two_h2"
    basis = _h2_chain(2)
    n = basis.n_ao
    return basis, np.array([[1.0 / (i + j + 1) for j in range(n)]
                            for i in range(n)])


FAST = ("h2_sto3g_j", "h2o_sto3g_s", "two_h2")
SLOW = ("h2o_sto3g_j",)  # (pp|pp), L = 4: deferred to #231
PARAMS = ([pytest.param(n, id=n) for n in FAST]
          + [pytest.param(n, id=n, marks=pytest.mark.slow) for n in SLOW])


def test_the_fast_cases_keep_to_l_le_2():
    """The rule, checked: every fast case is all-s (a sweep reaches L = 0),
    and the slow one reaches L = 4."""
    for name in FAST:
        basis, D = _case(name)
        assert all(s.l == 0 for s in basis.shells), name
        assert np.array_equal(D, D.T)
    for name in SLOW:
        assert 4 * max(s.l for s in _case(name)[0].shells) > L_MAX_RUN


# --- the bound ---------------------------------------------------------------


def test_i_p_is_the_proved_hermite_bound():
    """`I_p` as ao.k evaluates it (through `pq`) is `ao.pairs.hermite_bound`
    (through `one_centre`) -- the bound #84/#85 proved -- for every Hermite
    index a pair with `L <= 2` carries."""
    rng = random.Random(SEED)
    for _ in range(50):
        zeta = math.exp(rng.uniform(math.log(0.05), math.log(100.0)))
        for L in range(L_MAX_RUN + 1):
            got = aok._hermite_bounds_pq(zeta, L, pq=pq)
            want = hermite_bounds(zeta, L, one_centre=one_centre)
            assert np.allclose(got, want, rtol=1e-15, atol=0.0), (zeta, L)


def _shell(rng: random.Random, l: int, centre) -> Shell:  # noqa: E741
    n = rng.randint(1, 3)
    return Shell(l, centre,
                 [math.exp(rng.uniform(math.log(0.05), math.log(50.0)))
                  for _ in range(n)],
                 [rng.uniform(-1.0, 1.0) for _ in range(n)])


def _centre(rng: random.Random):
    return tuple(rng.uniform(-2.0, 2.0) for _ in range(3))


CLASSES = tuple((la, lc, lb, ld)
                for la in range(3) for lc in range(3)
                for lb in range(3) for ld in range(3)
                if la + lc + lb + ld <= L_MAX_RUN)


def test_record_bounds_is_the_sum_over_the_dressed_block():
    """`record_bounds` against a second ancestry: the same max-row sum over
    `ao.pairs.primitive_pairs`' dressed block with `hermite_bounds`."""
    rng = random.Random(SEED + 1)
    for _ in range(40):
        la, lc = rng.choice([(x, y) for x in range(3) for y in range(3)
                             if x + y <= L_MAX_RUN])
        sa, sc = _shell(rng, la, _centre(rng)), _shell(rng, lc, _centre(rng))
        got = record_bounds(epair(sa, sc), pq=pq)
        nc = sc.n_cart
        for g, rec in zip(got, primitive_pairs(sa, sc), strict=True):
            ip = hermite_bounds(rec.zeta, la + lc, one_centre=one_centre)
            want = max(math.fsum(abs(e) * i for k in range(nc)
                                 for e, i in zip(rec.block[r * nc + k], ip,
                                                 strict=True))
                       for r in range(sa.n_cart))
            assert math.isclose(g, want, rel_tol=1e-15, abs_tol=0.0)


def test_the_bound_bounds_every_record_pair():
    """THE PROOF, numerically: over N_QUARTETS randomised shell quartets of
    every class with L <= 2, every (bra record, ket record) pair's
    contribution `X = k_quartet` on that pair alone satisfies
    `|X_ij| <= I_ac I_bd max|D_cd|` for every element -- with no tolerance.
    A quarter of the draws put all four shells on one centre, a quarter `a`
    and `c` on one, a quarter `b` and `d` (the compact pairs where
    Cauchy-Schwarz is tightest), and a quarter are four free centres."""
    rng = random.Random(SEED)
    n_pairs = n_elements = 0
    tightest = 0.0
    seen = set()
    for t in range(N_QUARTETS):
        cls = CLASSES[t % len(CLASSES)]
        seen.add(cls)
        la, lc, lb, ld = cls
        cs = [_centre(rng) for _ in range(4)]
        mode = t % 4
        if mode == 0:
            cs = [cs[0]] * 4
        elif mode == 1:
            cs[1] = cs[0]
        elif mode == 2:
            cs[3] = cs[2]
        sa, sc, sb, sd = (_shell(rng, lv, c)
                          for lv, c in zip((la, lc, lb, ld), cs, strict=True))
        bra, ket = epair(sa, sc), epair(sb, sd)
        dcd = [rng.uniform(-1.0, 1.0) for _ in range(sc.n_cart * sd.n_cart)]
        dm = max(abs(x) for x in dcd)
        ib, ik = record_bounds(bra, pq=pq), record_bounds(ket, pq=pq)
        for i, br in enumerate(bra.recs):
            for j, kr in enumerate(ket.recs):
                X = k_quartet(replace(bra, recs=(br,)),
                              replace(ket, recs=(kr,)), dcd, pq=pq)
                bound = ib[i] * ik[j] * dm
                assert bound > 0.0
                for row in X:
                    for v in row:
                        assert abs(v) <= bound, (cls, t, i, j, v, bound)
                        tightest = max(tightest, abs(v) / bound)
                        n_elements += 1
                n_pairs += 1
    assert seen == set(CLASSES) and len(CLASSES) == 15
    print(f"\n{N_QUARTETS} quartets, {n_pairs} record pairs, {n_elements} "
          f"elements; tightest |X|/bound = {tightest:.9f}")


# --- the sweeps --------------------------------------------------------------


def _counting():
    calls = [0]

    def f(p, q, PQ, theta):
        calls[0] += 1
        return pq(p, q, PQ, theta)

    return f, calls


@pytest.mark.parametrize("name", FAST)
def test_tau_zero_skips_nothing_and_evaluates_no_bound(name):
    basis, D = _case(name)
    f, calls = _counting()
    exchange_matrix(basis, D, pq=f, tau=0.0)
    unscreened = calls[0]
    K0 = np.array(exchange_matrix(basis, D, pq=pq))
    for order in aok.ORDERS:
        f, calls = _counting()
        r = exchange_matrix_screened(basis, D, pq=f, tau=0.0, order=order)
        assert calls[0] == unscreened, (order, calls[0], unscreened)
        assert (r.n_skipped, r.n_examined) == (0, 0)
        assert r.max_skipped == 0.0 and r.dropped == 0.0
        K = np.array(r.K)
        assert np.max(np.abs(K - K0)) <= SWEEP_TOL * np.max(np.abs(K0))
    # and a bound, once evaluated, costs `pq` calls: the witness can see one
    f, calls = _counting()
    exchange_matrix_screened(basis, D, pq=f, tau=1e-300)
    assert calls[0] > unscreened


@cache
def _ladder(name: str):
    basis, D = _case(name)
    out = []
    for tau in TAUS:
        out.append((tau,
                    exchange_matrix_screened(basis, D, pq=pq, tau=tau),
                    exchange_matrix_screened(basis, D, pq=pq, tau=tau,
                                             order="unsorted")))
    return out


@pytest.mark.parametrize("name", PARAMS)
def test_break_equals_continue_and_the_error_is_bounded(name):
    basis, D = _case(name)
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
        assert a.max_skipped < tau or a.n_skipped == 0
        err = np.abs(np.array(a.K) - K0)
        assert np.all(err <= tau * np.array(a.scale)), (name, tau)
        assert a.n_skipped >= prev  # nested skipped sets
        prev = a.n_skipped
        # the continue inside the quartet contraction skips the same set
        for f in (exchange_matrix, exchange_matrix_reference):
            K = np.array(f(basis, D, pq=pq, tau=tau))
            assert np.max(np.abs(K - np.array(a.K))) <= SWEEP_TOL * big, (
                name, tau, f.__name__)
        print(f"\n{name} tau={tau:g}: skip {a.skip_fraction:.3f}, examined "
              f"{a.n_examined}/{b.n_examined}, max err/scale "
              f"{float(np.max(err / np.array(a.scale))):.2e}")
    # the screen fires on every fast case: the ladder is not vacuous
    assert rungs[-1][1].n_skipped > 0


def _j_key(basis: Basis, D: np.ndarray):
    """The J engine's key, misapplied: `I_bd * max|D_bd|`, the ket pair's
    own density block -- `Dmax_ket`'s shape."""
    off, sh = basis.offsets, basis.shells

    def key(b: int, d: int, i_bd: float) -> float:
        blk = D[off[b]:off[b] + sh[b].n_cart, off[d]:off[d] + sh[d].n_cart]
        return i_bd * float(np.max(np.abs(blk)))

    return key


@pytest.mark.parametrize("name", PARAMS)
def test_the_j_key_does_not_make_the_break_safe(name):
    """Sorted by the J key, the break stops at a ket whose bound is under
    `tau` while a later ket's is not -- `D_cd` straddles the records, so the
    ket's own density orders nothing the screen tests (ao/k.py, "THE KEY
    THE J ENGINE USES IS UNAVAILABLE FOR K"). At some rung it skips a pair
    whose bound is >= tau, and K moves; the Schwarz-only order never does
    (the test above)."""
    basis, D = _case(name)
    unsafe = []
    for tau, _, cont in _ladder(name)[1:]:
        j = aok._screened(basis, D, pq=pq, tau=tau, order="sorted",
                          ket_key=_j_key(basis, D))
        assert j.n_skipped >= cont.n_skipped  # it can only skip more
        if j.n_skipped > cont.n_skipped:
            assert j.max_skipped >= tau  # a pair the screen would keep
            assert j.K != cont.K
            unsafe.append(tau)
    print(f"\n{name}: the J key's break is unsafe at tau in {unsafe}")
    assert unsafe, f"{name}: the J key never broke early"
