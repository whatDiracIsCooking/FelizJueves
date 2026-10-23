"""The whole-molecule `F`, interpreted (#286): `launch_f`'s plan (#285) WALKED
by `walk_f` with `kernel_f`'s kernel (#282, #283) interpreted, closed with
`F <- F + F^T`, against `ao.fock`'s every-ordered-quartet sweep and against
libcint's `vj - vk/2` (the `_j` / `_k` fixture pair, `manifest.fock_pair`,
which asserts the two share density, shells and provenance before use;
milestone 12, pin 10).

THE WITNESS IS THE OTHER ENGINE. The sweep (`ao.fock.coulomb_exchange_
reference`, pin 6: every ordered quartet at weight 1, no symmetry) is fed
ERIs from `ao.eri` -- MMD through `reference.pq`, one block per ORDERED
quartet, each evaluated, none obtained by permuting another -- so the
walked `F` (HGP, interpreted) shares no recurrence with it.

THE STATED TOLERANCES, not tuned:

  * at `tau = 0`, nothing is skipped or pruned, and `F` must equal the full
    sweep to `TOL = 1e-13` of `scale_G` element by element -- the bound
    test_kernel_f.py states for one ERI in two operation orders (it holds
    the fused kernel to `ao.eri` at `1e-13` of scale per ERI), carried
    through the digestion by the triangle inequality: `scale_G` is the same
    sweep on `|D|` with `ao.eri`'s absolute-value twin. Against libcint,
    `LIBCINT_TOL = 1e-12` of the same scale (test_ao_fock.py's).
  * at `tau > 0` the screens fire and the comparison is to a bound COMPUTED
    from the answer, never a loosened constant: element by element,

        |F - G_sweep(kept)| <= TOL scale_G + prune
        |F - G_sweep(all)|  <= TOL scale_G + prune + skipped

    where `G_sweep(kept)` is the sweep restricted to the members of the
    quartets the walk COMPUTED, `skipped` is the sweep on `|D|` with
    `|(ab|cd)|` over the members of every quartet it skipped (all three
    screens), and `prune = tau x` the sweep on `|D|` with every ERI 1 over
    the computed members touching a pruned pair -- `ao/pairs.py` proves
    pruning moves no ERI by `tau` or more (launch_f, "PRUNING").

THE SCREENS AGAINST THE ANSWER. Every quartet the plan skips is held to the
EXACT contracted ERIs (`ao.eri`, full contraction, no pruning), by what its
screen claims: a quartet of a pre-launch screened batch pair or touching a
deleted pair has every `|(ab|cd)| < tau` (eq. 29's claim reads no
density); a quartet the kernel's six-block screen skips has every term it
would deposit, `|w_b D_comp x|` over the six blocks and every ERI, below
`tau` (`kernel_f`'s claim). Each screen case asserts its screens fired, so
none of these checks is vacuous -- except deletion, which no fixture here
fires at any `tau` used (pin 9: the fixtures are compact); that is asserted
too, so the day one does, the check runs.

THE WALKER'S INDEPENDENCE is structural (it does not import `launch_f`;
test_layering.py) and behavioural: a plan with a launch dropped, a batch
pair both launched and screened, a pair missing or doubled, or a launch
naming the wrong class is refused.

Fast tier: H2/STO-3G, H2O/STO-3G and H2O/6-31G. H2O/cc-pVDZ is `slow`
(measured on this box, #286: the 20,736 ordered `ao.eri` blocks 142 s,
once per process; the `tau = 0` walk 29 s, the sweeps under 1 s; each
screened walk 9-21 s). For scale: H2O/6-31G's blocks take 16 s and its
walk 5 s.
Interpreter only: no compiler, no card, no emission.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves import kernel_f as KF
from felizjueves.ao import fock
from felizjueves.ao.eri import eri_block_and_scale
from felizjueves.ao.pairs import primitive_schwarz_bounds
from felizjueves.ir import Scalar
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.launch_f import f_launches
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, pq
from felizjueves.walk_f import FWalk, walk_f

BOUND = partial(primitive_schwarz_bounds, one_centre=one_centre)
TOL = 1e-13  # test_kernel_f.py's, per ERI, carried by the triangle inequality
LIBCINT_TOL = 1e-12  # test_ao_fock.py's, against the fixtures

SMALL = ("h2_sto3g", "h2o_sto3g", "h2o_631g")
SLOW = pytest.mark.slow


def run(kernel, grid, block, args):
    """The injected executor: the interpreter, Boys from reference/."""
    a = {}
    for p in kernel.params:
        v = args[p.name]
        if p.array:
            v = np.array(v, dtype=np.int32 if p.type is Scalar.I32 else float)
        a[p.name] = v
    return Interpreter(kernel, launch=Launch(grid, block, 32),
                       boys=boys).run(**a)


@cache
def molecule(name):
    pair = manifest.fock_pair(name)  # asserts vj and vk share their inputs
    basis = ob.from_fixture(pair.j)
    return basis, np.array(pair.j.density), pair.two_electron()


@cache
def eris(name):
    """`ao.eri`'s block and scale for EVERY ordered shell quartet, each one
    evaluated (no permutation of another)."""
    basis, _, _ = molecule(name)
    sh = basis.shells
    out = {}
    for q in itertools.product(range(len(sh)), repeat=4):
        blk, scale = eri_block_and_scale((sh[q[0]], sh[q[1]]),
                                         (sh[q[2]], sh[q[3]]), pq=pq)
        out[q] = (np.array(blk), np.array(scale))
    return out


def members(quartets) -> frozenset:
    """Every ORDERED shell quartet the unique quartets `{P, Q}` stand for,
    by brute force: bra an ordering of one pair, ket of the other."""
    out = set()
    for P, Q in quartets:
        for X, Y in ((P, Q), (Q, P)):
            for p, q in {X, X[::-1]}:
                for r, s in {Y, Y[::-1]}:
                    out.add((p, q, r, s))
    return frozenset(out)


def sweep(name, keep=None, *, block="eri"):
    """`ao.fock`'s sweep over every ordered quartet, its `eri` zero outside
    `keep` (ordered members; all when `None`). `block`: "eri" gives `G = J -
    K/2` on `D`; "scale", "abs" and "one" give `J + K/2` on `|D|` with
    `ao.eri`'s scale twin, `|(ab|cd)|`, or every ERI 1."""
    basis, D, _ = molecule(name)
    E = eris(name)
    idx = {id(s): i for i, s in enumerate(basis.shells)}

    def eri(bra, ket):
        q = (idx[id(bra[0])], idx[id(bra[1])], idx[id(ket[0])],
             idx[id(ket[1])])
        blk, scale = E[q]
        if keep is not None and q not in keep:
            return np.zeros_like(blk).tolist()
        return {"eri": blk, "scale": scale, "abs": np.abs(blk),
                "one": np.ones_like(blk)}[block].tolist()

    dens = D if block == "eri" else np.abs(D)
    J, K = fock.coulomb_exchange_reference(basis, dens.tolist(), eri=eri)
    f = fock.EXCHANGE_FRACTION
    return np.array(J) + (-f if block == "eri" else f) * np.array(K)


@cache
def walk(name, tau, n) -> tuple[FWalk, object]:
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=tau, n=n)
    return walk_f(plan, basis, D, run=run), plan


def _worst(got, want, scale) -> float:
    return float(np.max(np.abs(got - want) / np.where(scale > 0, scale, 1.0)))


# ---------------------------------------------------------------------------
# tau = 0: the whole plan, nothing skipped, against the sweep and libcint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [*SMALL[:-1],  # 6-31G: ~6 s, slow by cost
                                  pytest.param("h2o_631g", marks=SLOW),
                                  pytest.param("h2o_ccpvdz", marks=SLOW)])
def test_the_walked_plan_is_the_sweep_and_libcint(name) -> None:
    """`tau = 0`: every unique quartet launched exactly once and none
    skipped; `F` against the FULL sweep to `TOL` of scale and libcint's
    `vj - vk/2` to `LIBCINT_TOL`."""
    _, _, libcint = molecule(name)
    w, plan = walk(name, 0.0, 1)
    assert not w.skipped and w.n_prim_pruned == 0
    assert w.n_launches == len(plan.launches) > 0
    F = np.array(w.F)
    want, scale = sweep(name), sweep(name, block="scale")
    assert np.max(np.abs(want)) > 0
    assert _worst(F, want, scale) <= TOL
    assert _worst(F, libcint, scale) <= LIBCINT_TOL


# ---------------------------------------------------------------------------
# tau > 0: every skipped quartet against the answer, and F within the bound
# ---------------------------------------------------------------------------

SCREENED = [
    ("h2o_sto3g", 0.1, 4),
    pytest.param("h2o_631g", 1e-3, 4, marks=SLOW),  # 4-18 s each, by cost
    pytest.param("h2o_631g", 0.1, 4, marks=SLOW),
    pytest.param("h2o_ccpvdz", 1e-3, 4, marks=SLOW),
    pytest.param("h2o_ccpvdz", 0.1, 4, marks=SLOW),
]


def _six_terms(name, q) -> float:
    """The largest `|w_b D_comp x|` the canonical member of quartet `q`
    would deposit, over its six blocks and every ERI: the exact ERIs."""
    basis, D, _ = molecule(name)
    (a, b), (c, d) = q
    s = (a, b, c, d)
    off = basis.offsets
    n = [basis.shells[x].n_cart for x in s]
    x = eris(name)[s][0].reshape(n)
    worst = 0.0
    for blk, w in zip(KF.BLOCKS, (KF.WEIGHTS.of(b) for b in KF.BLOCKS),
                      strict=True):
        u, v = sorted({0, 1, 2, 3} - set(blk))
        Dc = D[off[s[u]]:off[s[u]] + n[u], off[s[v]]:off[s[v]] + n[v]]
        shape = [1, 1, 1, 1]
        shape[u], shape[v] = n[u], n[v]
        worst = max(worst, float(np.max(np.abs(w * Dc.reshape(shape) * x))))
    return worst


@pytest.mark.parametrize(("name", "tau", "n"), SCREENED)
def test_every_skipped_quartet_is_below_threshold_in_the_answer(
        name, tau, n) -> None:
    w, plan = walk(name, tau, n)
    # the case exercises the screens it is about (deletion: see docstring)
    assert w.prelaunch_screened and w.kernel_screened and w.n_prim_pruned
    assert not w.deleted and not plan.deleted
    assert w.n_prim_pruned == plan.counts()["primitive_pairs_pruned"]
    assert w.kernel_screened <= w.launched
    E = eris(name)
    for q in w.prelaunch_screened | w.deleted:
        (a, b), (c, d) = q
        assert float(np.max(np.abs(E[(a, b, c, d)][0]))) < tau, q
    for q in w.kernel_screened:
        assert _six_terms(name, q) < tau, q


@pytest.mark.parametrize(("name", "tau", "n"), SCREENED)
def test_a_screened_walk_is_the_kept_sweep_within_the_pruning_bound(
        name, tau, n) -> None:
    w, plan = walk(name, tau, n)
    F = np.array(w.F)
    pruned = {p.shell_pair for p in plan.pairs if p.degree < p.n_prim}
    computed = members(w.computed)
    touched = members(q for q in w.computed if q[0] in pruned
                      or q[1] in pruned)
    scale = sweep(name, block="scale")
    prune = tau * sweep(name, touched, block="one")
    kept = sweep(name, computed)
    bound = TOL * scale + prune
    assert np.all(np.abs(F - kept) <= bound)
    skipped = sweep(name, members(w.skipped), block="abs")
    full = sweep(name)
    assert np.all(np.abs(F - full) <= bound + skipped)
    assert np.max(skipped) > 0  # the screens moved F: the bound is not idle


# ---------------------------------------------------------------------------
# The walker refuses a plan it cannot account for
# ---------------------------------------------------------------------------


def _bad(plan, **kw):
    basis, D, _ = molecule("h2o_sto3g")
    with pytest.raises(ValueError, match="walk_f refuses"):
        walk_f(replace(plan, **kw), basis, D, run=run)


def test_the_walker_refuses_a_plan_it_cannot_account_for() -> None:
    basis, D, _ = molecule("h2o_sto3g")
    plan = f_launches(basis, D, bound=BOUND, tau=0.1, n=4)
    assert plan.screened and len(plan.batches) > 1
    L0 = plan.launches[0]
    _bad(plan, launches=plan.launches[1:])  # a batch pair never met
    _bad(plan, screened=(*plan.screened, (L0.bra, L0.ket)))  # met twice
    _bad(plan, deleted=(plan.batches[0].pairs[0].shell_pair,))  # pair twice
    b0 = plan.batches[0]
    _bad(plan, batches=(replace(b0, pairs=b0.pairs[1:]) if len(b0.pairs) > 1
                        else plan.batches[1], *plan.batches[1:]))
    other = next(L for L in plan.launches if L.spec != L0.spec)
    _bad(plan, launches=(replace(L0, spec=other.spec), *plan.launches[1:]))
