"""The split HGP build's ordered pair stream (#403): `launch_hjk`.

Fast tier, no device markers: pure Python, no compiler, no card, no PySCF.
Every fixture here is H2 or H2O/STO-3G, whose highest quartet is `(pp|pp)`,
L = 4 = FAST_L_MAX (tiers.py), so nothing is `slow`. The bases and
densities come from tracked fixtures' `converted` records; `reference/` is
imported HERE, by the test, for the evaluator the bound is injected with and
for the ERIs (`eri/eri.py`) the screens are checked against -- `launch_hjk`
never reaches it (test_layering.py, rule "hjk_launch").

THE CLAIMS (the issue's acceptance, in order).

  once       every ordered quartet of shell pairs appears exactly once per
             mode, and the canonical and ordered streams visit the same
             quartet multiset, through the same batch pairs.
  routing    ordered mode names each launch's own class; canonical mode its
             orbit representative's, and the carried permutation rebuilds
             the ordered block from the representative's (computed).
  tau = 0    nothing is skipped, by either screen.
  ladder     at every rung, every skipped quartet's exact contribution
             through its screen's one density block -- `|D_kl (ij|kl)|` for
             J, `|D_jl (ij|kl)|` for K, from eri/eri.py -- is `<=` its key and
             the key `< tau`: proved per skip, no tolerance.
  misapplied J's key applied to a K build skips a quartet whose K
             contribution is `>= tau`.

The routing check compares two evaluations of one block that associate the
sum differently, so it alone reads a tolerance (`1e-14 * max|block|`); every
screen check is a plain `<=` / `<`.
"""

from __future__ import annotations

import math
from collections import Counter
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.pairs import primitive_schwarz_bounds
from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.eri import eri_block
from felizjueves.eri.fock import kernel_f
from felizjueves.eri.hjk.launch_hjk import (
    MAX_PRIM_PAIRS,
    MODES,
    SCREENS,
    HPlan,
    hjk_launches,
)
from felizjueves.hgp.orbits import apply, representative
from felizjueves.reference.pq import one_centre, pq

BOUND = partial(primitive_schwarz_bounds, one_centre=one_centre)
FIXTURES = ("h2_sto3g_j", "h2o_sto3g_j")
# Loose rungs too: compact H2 holds no key below 0.1, and soundness is
# claimed at every tau.
LADDER = (0.5, 0.2, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6)


@cache
def _fixture(name: str) -> tuple[Basis, np.ndarray]:
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return ob.from_fixture(fx), np.array(fx.density)


@cache
def _plan(name: str, tau: float, mode: str, screen: str) -> HPlan:
    basis, D = _fixture(name)
    return hjk_launches(basis, D, bound=BOUND, tau=tau, mode=mode,
                        screen=screen)


@cache
def _block(name: str, q: tuple[int, ...]) -> np.ndarray:
    """`(ab|cd)` of the ordered quartet, `[i][j][k][l]`, by eri/eri.py."""
    sh = _fixture(name)[0].shells
    s = [sh[x] for x in q]
    blk = np.array(eri_block((s[0], s[1]), (s[2], s[3]), pq=pq))
    return blk.reshape([x.n_cart for x in s])


def _aos(basis: Basis, s: int) -> range:
    o = basis.offsets[s]
    return range(o, o + basis.shells[s].n_cart)


def test_the_staging_cap_is_kernel_fs() -> None:
    assert MAX_PRIM_PAIRS == kernel_f.MAX_PRIM_PAIRS


@pytest.mark.parametrize("name", FIXTURES)
def test_every_shell_pair_is_ordered_and_batched_by_degree(name) -> None:
    basis, _ = _fixture(name)
    plan = _plan(name, 0.0, "ordered", "j")
    ns = len(basis.shells)
    assert sorted(p.shell_pair for p in plan.pairs) == [
        (a, b) for a in range(ns) for b in range(ns)]
    keys = [(b.la, b.lb, b.degree) for b in plan.batches]
    assert keys == sorted(set(keys))
    for b in plan.batches:
        for p in b.pairs:
            sa, sb = basis.shells[p.a], basis.shells[p.b]
            assert (p.la, p.lb) == (sa.l, sb.l) == b.cls
            assert p.degree == len(sa.exponents) * len(sb.exponents) == b.degree
            assert p.G == math.fsum(BOUND(sa, sb))
        # descending by G, stably (pair order -- a outer, b inner -- on ties)
        order = sorted(b.pairs, key=lambda p: (-p.G, p.a, p.b))
        assert list(b.pairs) == order


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("mode", MODES)
def test_every_ordered_quartet_appears_exactly_once(name, mode) -> None:
    ns = len(_fixture(name)[0].shells)
    plan = _plan(name, 0.0, mode, "j")
    seen = Counter(v.quartet for v in plan.visits)
    assert len(seen) == ns ** 4
    assert set(seen.values()) == {1}
    nb = len(plan.batches)
    assert len(plan) == nb * nb  # every ORDERED batch pair, no triangle
    for v in plan.visits:
        x = plan.launches[v.launch]
        bra, ket = plan.batches[x.bra], plan.batches[x.ket]
        assert v.quartet[:2] in {p.shell_pair for p in bra.pairs}
        assert v.quartet[2:] in {p.shell_pair for p in ket.pairs}


@pytest.mark.parametrize("name", FIXTURES)
def test_both_modes_visit_the_same_multiset(name) -> None:
    c, o = (_plan(name, 0.0, m, "j") for m in ("canonical", "ordered"))
    assert Counter(v.quartet for v in c.visits) == Counter(
        v.quartet for v in o.visits)
    assert [(x.bra, x.ket, x.ordered) for x in c] == [
        (x.bra, x.ket, x.ordered) for x in o]
    assert len(c.classes) <= len(o.classes)


@pytest.mark.parametrize("name", FIXTURES)
def test_ordered_mode_names_its_own_class(name) -> None:
    plan = _plan(name, 0.0, "ordered", "j")
    for x in plan:
        assert x.cls == x.ordered
        assert x.orientation.shells == x.orientation.axes == (0, 1, 2, 3)
    assert all(v.kernel_quartet == v.quartet for v in plan.visits)


@pytest.mark.parametrize("name", FIXTURES)
def test_canonical_mode_routes_to_the_representative(name) -> None:
    basis, _ = _fixture(name)
    plan = _plan(name, 0.0, "canonical", "j")
    assert plan.classes == tuple(sorted({representative(x.ordered)
                                         for x in plan}))
    done = set()
    for v in plan.visits:
        x = plan.launches[v.launch]
        assert x.cls == representative(x.ordered)
        assert apply(x.orientation.shells, x.ordered) == x.cls
        ls = tuple(basis.shells[s].l for s in v.kernel_quartet)
        assert ls == x.cls
        if x.ordered in done:
            continue
        done.add(x.ordered)
        # the carried permutation rebuilds the ordered block (computed)
        want = _block(name, v.quartet)
        got = _block(name, v.kernel_quartet).transpose(x.orientation.axes)
        assert got.shape == want.shape
        assert np.max(np.abs(got - want)) <= 1e-14 * np.max(np.abs(want))
    assert len(done) == len({x.ordered for x in plan})


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("screen", SCREENS)
@pytest.mark.parametrize("mode", MODES)
def test_at_tau_zero_nothing_is_skipped(name, screen, mode) -> None:
    plan = _plan(name, 0.0, mode, screen)
    assert not plan.skipped
    assert all(v.key >= 0.0 for v in plan.visits)


def _contribution(basis: Basis, D: np.ndarray, name: str, q, screen) -> float:
    """The largest `|D_kl (ij|kl)|` (J) or `|D_jl (ij|kl)|` (K) of `q`."""
    a, b, c, d = q
    blk = _block(name, q)
    if screen == "j":
        Dblk = D[np.ix_(_aos(basis, c), _aos(basis, d))]
        return float(np.max(np.abs(blk * Dblk[None, None, :, :])))
    Dblk = D[np.ix_(_aos(basis, b), _aos(basis, d))]
    return float(np.max(np.abs(blk * Dblk[None, :, None, :])))


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("screen", SCREENS)
def test_every_skip_is_proved_on_the_ladder(name, screen) -> None:
    basis, D = _fixture(name)
    fired = 0
    for tau in LADDER:
        plan = _plan(name, tau, "ordered", screen)
        canon = _plan(name, tau, "canonical", screen)
        assert {v.quartet for v in plan.skipped} == {
            v.quartet for v in canon.skipped}
        G = {p.shell_pair: p.G for p in plan.pairs}
        for v in plan.skipped:
            a, b, c, d = v.quartet
            assert v.key < tau
            assert float(np.max(np.abs(_block(name, v.quartet)))) <= (
                G[a, b] * G[c, d])
            assert _contribution(basis, D, name, v.quartet, screen) <= v.key
        fired += len(plan.skipped)
    assert fired > 0, "the ladder never fired the screen: nothing was proved"


def test_js_key_misapplied_to_k_skips_a_quartet_k_needs() -> None:
    # H2 with a density whose diagonal blocks vanish: J's key reads D_cd = 0
    # on every `(..|00)` and `(..|11)` ket, while K reads D_bd, which is 1
    # across the molecule.
    name, tau = "h2_sto3g_j", 1e-6
    basis, _ = _fixture(name)
    assert [s.l for s in basis.shells] == [0, 0]
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    j = hjk_launches(basis, D, bound=BOUND, tau=tau, mode="ordered",
                     screen="j")
    k = hjk_launches(basis, D, bound=BOUND, tau=tau, mode="ordered",
                     screen="k")
    needed = [v.quartet for v in j.skipped
              if _contribution(basis, D, name, v.quartet, "k") >= tau]
    assert (0, 1, 0, 0) in needed
    k_skipped = {v.quartet for v in k.skipped}
    assert k_skipped.isdisjoint(needed)
    for v in k.skipped:  # and K's own key stays sound on this density
        assert _contribution(basis, D, name, v.quartet, "k") <= v.key < tau


@pytest.mark.parametrize("bad", ["mode", "screen", "tau", "asym", "short",
                                 "degree"])
def test_refusals(bad) -> None:
    basis, D = _fixture("h2_sto3g_j")
    kw = dict(bound=BOUND, tau=0.0, mode="ordered", screen="j")
    if bad == "mode":
        kw["mode"] = "unique"
    elif bad == "screen":
        kw["screen"] = "f"
    elif bad == "tau":
        kw["tau"] = -1.0
    elif bad == "asym":
        D = np.array([[1.0, 0.5], [0.25, 1.0]])
    elif bad == "short":
        kw["bound"] = lambda sa, sb: (1.0,)
    else:  # 12 x 12 = 144 primitive pairs > 128
        s = Shell(0, (0.0, 0.0, 0.0), tuple(0.1 * 2 ** i for i in range(12)),
                  (1.0,) * 12)
        basis, D = Basis((s,)), np.eye(1)
        kw["bound"] = lambda sa, sb: (1.0,) * (len(sa.exponents)
                                               * len(sb.exponents))
    with pytest.raises(ValueError):
        hjk_launches(basis, D, **kw)
