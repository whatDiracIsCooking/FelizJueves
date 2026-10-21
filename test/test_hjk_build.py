"""Whole `J` and `K` on the split HGP build, interpreted (#406): `launch_hjk`'s
ordered pair stream (#403) WALKED by `walk_hjk` with `kernel_hj` (#404) and
`kernel_hk` (#405) interpreted, in both class modes, against `ao/j.py`'s
and `ao/k.py`'s every-ordered-quartet sweeps and against libcint's `vj` /
`vk` (the `_j` / `_k` fixture pair: `manifest.fock_pair` asserts that the
two share density, shells and provenance before either is used, and
`test_the_fixture_pairs_share_their_inputs` runs that check first).

Every class here has `L <= 4`: H2/STO-3G and H2O/STO-3G hold s and p shells,
and so does H2O/6-31G, so the highest class is `(pp|pp)`. cc-pVDZ (d shells)
is out of the cap and is not run.

THE WITNESS IS THE OTHER ENGINE. `ao/j.py`'s `coulomb_matrix_reference` and
`ao/k.py`'s `exchange_matrix_reference` are MMD (`reference.pq`), every
ordered quartet at weight 1, so the walked matrix (HGP, interpreted) shares
no recurrence with them. The restricted sweeps the screened cases need are
`ao.fock.coulomb_exchange_reference` (the same every-ordered-quartet sweep,
J and K from one pass) fed `ao.eri`'s MMD blocks.

THE STATED TOLERANCES, none new and none widened:

  * `TOL = 1e-12` is test/test_hgp_eri.py's per-ERI cross-engine bound
    (HGP against MMD within `1e-12 (scale_mmd + scale_hgp)`, each scale its
    engine's absolute-value twin; that file justifies it as `gamma_n` for
    about 9000 roundings). `J` and `K` are linear in the ERIs, so by the
    triangle inequality the walked matrix is held element by element to

        |X_walk - X_mmd| <= TOL S,   S = SUM |D_..| (scale_mmd + scale_hgp)

    the sweep run on `|D|` with the two twins summed -- exactly
    test_kernel_hj.py's / test_kernel_hk.py's per-class derivation, carried
    over the whole molecule. `scale_hgp` is taken in the orientation the
    kernel RAN: in canonical mode the orbit representative's twin, permuted
    back (`hgp.orbits.Orientation.gather`); in ordered mode the class's own.
  * Against libcint, `LIBCINT_TOL = 1e-12` of the same `S` -- test_engine.py's
    libcint yardstick (the summed scale of every engine) and
    test_fock_build.py's / test_oracle_fock.py's constant.
  * The two modes agree within `TOL (S_canonical + S_ordered)`, the HGP
    twins of both orientations on `|D|` (test_kernel_hj.py's "THE MODES
    AGREE", carried the same way).
  * At `tau > 0` the comparison is to a bound COMPUTED from the answer:

        |X - X_mmd(kept)| <= TOL S
        |X - X_mmd(all)|  <= TOL S + skipped

    `X_mmd(kept)` the sweep restricted to the ordered quartets the walk
    COMPUTED, `skipped` the sweep on `|D|` with `|(ab|cd)|` over the ones
    it skipped.

THE SCREENS AGAINST THE ANSWER. Every quartet the kernel's one-block screen
skips (the walk's `kernel_screened`, predicted in the kernel's operation
order) has every term it would deposit -- `|D_cd (ab|cd)|` for J, `|D_bd
(ab|cd)|` for K, every element, exact ERIs from `ao.eri` -- below `tau`;
and it is exactly the set `launch_hjk` marked skipped. Each screened case
asserts that its screen fired.

THE WALKER'S INDEPENDENCE is structural (it does not import `launch_hjk`;
test_layering.py) and behavioural: a plan with a launch dropped, a pair
missing, a launch naming the wrong class, a pruned degree, or a routing the
kernels do not serve is refused.

TIERS. Fast: H2/STO-3G in both modes and both builds at `tau = 0` and
`tau = 0.2`, and H2O/STO-3G's canonical `J` and `K` at `tau = 1e-2` (the
skip checks on a molecule with p shells). `slow`, by cost (measured on this
box, #406): every H2O/STO-3G `tau = 0` case (a walk ~7 s, the ordered-mode
HGP twins ~16 s) and every H2O/6-31G case (its 6,561 ordered `ao.eri`
blocks ~16 s, its ordered-mode HGP twins ~3 min, a walk ~1 min).
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

from felizjueves import launch_hjk
from felizjueves import walk_hjk as W
from felizjueves.ao import fock
from felizjueves.ao import hgp_drive as HD
from felizjueves.ao.eri import eri_block_and_scale
from felizjueves.ao.j import coulomb_matrix_reference
from felizjueves.ao.k import exchange_matrix_reference
from felizjueves.ao.pairs import primitive_schwarz_bounds
from felizjueves.hgp import orbits
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import Scalar
from felizjueves.launch_hjk import hjk_launches
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, pq

BOUND = partial(primitive_schwarz_bounds, one_centre=one_centre)
TOL = 1e-12  # test_hgp_eri.py's per-ERI cross-engine bound (docstring)
LIBCINT_TOL = 1e-12  # test_engine.py's / test_fock_build.py's, vs fixtures

MOLECULES = ("h2_sto3g", "h2o_sto3g", "h2o_631g")
MODES, SCREENS = W.MODES, W.SCREENS
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


def never(*_):
    raise AssertionError("a refused plan reached the executor")


@cache
def molecule(name):
    pair = manifest.fock_pair(name)  # asserts vj and vk share their inputs
    basis = ob.from_fixture(pair.j)
    return basis, np.array(pair.j.density), pair.vj, pair.vk


@cache
def plan(name, tau, mode, screen):
    basis, D, _, _ = molecule(name)
    return hjk_launches(basis, D, bound=BOUND, tau=tau, mode=mode,
                        screen=screen)


@cache
def walk(name, tau, mode, screen) -> W.HJKWalk:
    basis, D, _, _ = molecule(name)
    return W.walk_hjk(plan(name, tau, mode, screen), basis, D, run=run)


def _ls(basis, q):
    return tuple(basis.shells[s].l for s in q)


@cache
def mmd(name):
    """`ao.eri`'s block and twin for EVERY ordered shell quartet, each one
    evaluated (none obtained by permuting another), flat `[a][b][c][d]`."""
    basis, _, _, _ = molecule(name)
    sh = basis.shells
    out = {}
    for q in itertools.product(range(len(sh)), repeat=4):
        blk, scale = eri_block_and_scale((sh[q[0]], sh[q[1]]),
                                         (sh[q[2]], sh[q[3]]), pq=pq)
        out[q] = (np.array(blk).ravel(), np.array(scale).ravel())
    return out


@cache
def _hgp_twin(name, x):
    """`ao.hgp_drive`'s twin on the shells `x`, in that order, flat."""
    basis, _, _, _ = molecule(name)
    sh = basis.shells
    _, scale = HD.eri_block_and_scale((sh[x[0]], sh[x[1]]),
                                      (sh[x[2]], sh[x[3]]), boys=boys)
    return np.array(scale).ravel()


@cache
def hgp_twin(name, mode):
    """Per ordered quartet, the HGP twin in the orientation `mode`'s kernel
    runs it (docstring, "THE STATED TOLERANCES"), flat in the ordered
    quartet's own `[a][b][c][d]` layout."""
    basis, _, _, _ = molecule(name)
    out = {}
    for q in itertools.product(range(len(basis.shells)), repeat=4):
        if mode == "ordered":
            out[q] = _hgp_twin(name, q)
            continue
        o = orbits.to_representative(_ls(basis, q))
        x = tuple(q[j] for j in o.shells)
        out[q] = _hgp_twin(name, x)[list(o.gather)]
    return out


def sweep(name, screen, *, block, keep=None, mode=None):
    """`ao.fock`'s every-ordered-quartet sweep, `J` (screen "j") or `K`
    ("k"), its `eri` zero outside `keep` (ordered quartets; all when
    `None`). `block`: "eri" is the MMD block on `D`; on `|D|`, "abs" is
    `|(ab|cd)|`, "scale" the MMD twin plus `mode`'s HGP twin, and "modes"
    the two modes' HGP twins summed."""
    basis, D, _, _ = molecule(name)
    E = mmd(name)
    idx = {id(s): i for i, s in enumerate(basis.shells)}

    def eri(bra, ket):
        q = (idx[id(bra[0])], idx[id(bra[1])], idx[id(ket[0])],
             idx[id(ket[1])])
        n = (bra[0].n_cart * bra[1].n_cart, ket[0].n_cart * ket[1].n_cart)
        if keep is not None and q not in keep:
            return np.zeros(n).tolist()
        blk, scale = E[q]
        if block == "eri":
            v = blk
        elif block == "abs":
            v = np.abs(blk)
        elif block == "scale":
            v = scale + hgp_twin(name, mode)[q]
        else:
            v = hgp_twin(name, "canonical")[q] + hgp_twin(name, "ordered")[q]
        return v.reshape(n).tolist()

    dens = D if block == "eri" else np.abs(D)
    J, K = fock.coulomb_exchange_reference(basis, dens.tolist(), eri=eri)
    return np.array(J if screen == "j" else K)


@cache
def reference(name, screen):
    """`ao/j.py`'s / `ao/k.py`'s every-ordered-quartet sweep (MMD)."""
    basis, D, _, _ = molecule(name)
    f = coulomb_matrix_reference if screen == "j" else exchange_matrix_reference
    return np.array(f(basis, D.tolist(), pq=pq))


def _worst(got, want, scale) -> float:
    err = np.abs(got - want)
    assert np.all(err[scale == 0.0] == 0.0)
    return float(np.max(np.divide(err, scale, out=np.zeros_like(err),
                                  where=scale > 0)))


def _case(name, *rest, slow=False):
    marks = (SLOW,) if slow else ()
    return pytest.param(name, *rest, marks=marks,
                        id="-".join(map(str, (name, *rest))))


# ---------------------------------------------------------------------------
# The shared inputs, first
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MOLECULES)
def test_the_fixture_pairs_share_their_inputs(name) -> None:
    """`manifest.fock_pair` refuses a `_j` / `_k` pair whose density,
    shells or provenance differ; the walk's every case reads that pair."""
    pair = manifest.fock_pair(name)
    assert np.array_equal(pair.j.density, pair.k.density)
    basis, _, vj, vk = molecule(name)
    assert vj.shape == vk.shape == (basis.n_ao, basis.n_ao)
    assert max(s.l for s in basis.shells) <= 1  # every class L <= 4


# ---------------------------------------------------------------------------
# tau = 0: the whole stream, nothing skipped, against the host and libcint
# ---------------------------------------------------------------------------

UNSCREENED = [
    _case(name, mode, screen, slow=name != "h2_sto3g")
    for name in MOLECULES for mode in MODES for screen in SCREENS
]


@pytest.mark.parametrize(("name", "mode", "screen"), UNSCREENED)
def test_the_walked_stream_is_the_host_sweep_and_libcint(
        name, mode, screen) -> None:
    """`tau = 0`: every ordered quartet visited once and none skipped; the
    whole `J` / `K` against `ao/j.py` / `ao/k.py` to `TOL S` and libcint's
    `vj` / `vk` to `LIBCINT_TOL S`, element by element."""
    basis, _, vj, vk = molecule(name)
    w = walk(name, 0.0, mode, screen)
    p = plan(name, 0.0, mode, screen)
    assert (w.screen, w.mode) == (screen, mode)
    assert len(w.visited) == len(basis.shells) ** 4
    assert not w.kernel_screened and not p.skipped
    assert w.n_launches == len(p.launches) > 0
    X = np.array(w.X)
    S = sweep(name, screen, block="scale", mode=mode)
    want = reference(name, screen)
    assert np.max(np.abs(want)) > 0.1
    host = _worst(X, want, S)
    lib = _worst(X, vj if screen == "j" else vk, S)
    print(f"{name} {screen} {mode}: host {host:.3g} S, libcint {lib:.3g} S")
    assert host <= TOL
    assert lib <= LIBCINT_TOL


@pytest.mark.parametrize(("name", "screen"), [
    _case(name, screen, slow=name != "h2_sto3g")
    for name in MOLECULES for screen in SCREENS])
def test_canonical_and_ordered_agree(name, screen) -> None:
    """The two class modes visit the same quartets through different
    kernels -- one per orbit against one per ordered class -- and agree
    within `TOL` of both modes' HGP twins."""
    basis, _, _, _ = molecule(name)
    c, o = (walk(name, 0.0, m, screen) for m in MODES)
    assert c.visited == o.visited
    classes = {_ls(basis, q) for q in c.visited}
    reps = {orbits.representative(x) for x in classes}
    mod = W._BUILDER[screen]
    assert len(c.kernels) == len(reps) <= len(o.kernels) == len(classes)
    assert all(k.startswith(f"h{screen}_o_") for k in o.kernels)
    assert not any(k.startswith(f"h{screen}_o_") for k in c.kernels)
    assert mod is (W.HJ if screen == "j" else W.HK)
    S = sweep(name, screen, block="modes")
    w = _worst(np.array(c.X), np.array(o.X), S)
    print(f"{name} {screen}: canonical vs ordered {w:.3g} S")
    assert w <= TOL


# ---------------------------------------------------------------------------
# tau > 0: every skipped quartet against the answer, and X within the bound
# ---------------------------------------------------------------------------

SCREENED = [
    *(_case("h2_sto3g", 0.2, mode, screen)
      for mode in MODES for screen in SCREENS),
    *(_case("h2o_sto3g", 1e-2, "canonical", screen) for screen in SCREENS),
    *(_case("h2o_sto3g", 1e-2, "ordered", screen, slow=True)
      for screen in SCREENS),
    *(_case("h2o_631g", tau, mode, screen, slow=True)
      for tau in (1e-3, 1e-2) for mode in MODES for screen in SCREENS),
]


def _terms(name, screen, q) -> float:
    """The largest term ordered quartet `q` would deposit through its
    screen's one density block, exact ERIs: `|D_cd (ab|cd)|` (J) or
    `|D_bd (ab|cd)|` (K), over every element."""
    basis, D, _, _ = molecule(name)
    off = basis.offsets
    n = [basis.shells[s].n_cart for s in q]
    x = mmd(name)[q][0].reshape(n)
    u, v = W.DENSITY[screen]
    Dc = D[off[q[u]]:off[q[u]] + n[u], off[q[v]]:off[q[v]] + n[v]]
    shape = [1, 1, 1, 1]
    shape[u], shape[v] = n[u], n[v]
    return float(np.max(np.abs(Dc.reshape(shape) * x)))


@pytest.mark.parametrize(("name", "tau", "mode", "screen"), SCREENED)
def test_every_skipped_quartet_is_below_threshold_in_the_answer(
        name, tau, mode, screen) -> None:
    w = walk(name, tau, mode, screen)
    p = plan(name, tau, mode, screen)
    assert w.kernel_screened and w.computed  # the screen fired, not all
    assert w.kernel_screened == {v.quartet for v in p.skipped}
    for q in w.kernel_screened:
        assert _terms(name, screen, q) < tau, q


@pytest.mark.parametrize(("name", "tau", "mode", "screen"), SCREENED)
def test_a_screened_walk_is_the_kept_sweep_within_the_bound(
        name, tau, mode, screen) -> None:
    w = walk(name, tau, mode, screen)
    X = np.array(w.X)
    S = sweep(name, screen, block="scale", mode=mode)
    kept = sweep(name, screen, block="eri", keep=w.computed)
    worst = _worst(X, kept, S)
    print(f"{name} tau={tau} {screen} {mode}: kept {worst:.3g} S, "
          f"{len(w.kernel_screened)}/{len(w.visited)} skipped")
    assert worst <= TOL
    skipped = sweep(name, screen, block="abs", keep=w.kernel_screened)
    full = sweep(name, screen, block="eri")
    assert np.all(np.abs(X - full) <= TOL * S + skipped)
    assert np.max(skipped) > 0  # the screen moved X: the bound is not idle


# ---------------------------------------------------------------------------
# The walker refuses a plan it cannot account for
# ---------------------------------------------------------------------------


def _bad(p, **kw):
    basis, D, _, _ = molecule("h2o_sto3g")
    with pytest.raises(ValueError, match="walk_hjk refuses"):
        W.walk_hjk(replace(p, **kw), basis, D, run=never)


@pytest.mark.parametrize("screen", SCREENS)
def test_the_walker_refuses_a_plan_it_cannot_account_for(screen) -> None:
    c = plan("h2o_sto3g", 0.0, "canonical", screen)
    o = plan("h2o_sto3g", 0.0, "ordered", screen)
    L0 = c.launches[0]
    _bad(c, launches=c.launches[1:])  # a batch pair never met
    _bad(c, launches=(*c.launches, L0))  # met twice
    b0 = c.batches[0]
    rest = (replace(b0, pairs=b0.pairs[1:]) if len(b0.pairs) > 1
            else c.batches[1])
    _bad(c, batches=(rest, *c.batches[1:]))  # a pair missing
    p0 = b0.pairs[0]
    _bad(c, batches=(replace(b0, degree=p0.degree - 1,
                             pairs=tuple(replace(p, degree=p0.degree - 1)
                                         for p in b0.pairs)),
                     *c.batches[1:]))  # pruned: the split stream is not
    swapped = next(L for L in c.launches if L.orientation.shells
                   != launch_hjk.IDENTITY)
    i = c.launches.index(swapped)
    ident = replace(swapped, orientation=replace(
        swapped.orientation, shells=launch_hjk.IDENTITY, axes=launch_hjk.IDENTITY,
        rep=swapped.ordered))
    _bad(c, launches=(*c.launches[:i], ident, *c.launches[i + 1:]))
    wrong = replace(L0, ordered=tuple(reversed(L0.ordered)))
    if wrong.ordered != L0.ordered:
        _bad(c, launches=(wrong, *c.launches[1:]))
    _bad(o, mode="canonical")  # identity routing is not canonical's
    _bad(c, mode="ordered")  # nor is a representative's ordered's
    _bad(c, screen="x")


def test_the_walker_restates_the_streams_vocabulary() -> None:
    assert W.MODES == launch_hjk.MODES and W.SCREENS == launch_hjk.SCREENS
    assert W.IDENTITY == launch_hjk.IDENTITY
    assert W.kernel_spec("k", "canonical", (0, 1, 0, 0)).am == (0, 1, 0, 0)
    with pytest.raises(ValueError):
        W.kernel_spec("f", "canonical", (0, 0, 0, 0))
    with pytest.raises(ValueError, match="splits a shell pair"):
        W.kernel_pairs((0, 2, 1, 3), [(0, 1)], [(2, 3)])
