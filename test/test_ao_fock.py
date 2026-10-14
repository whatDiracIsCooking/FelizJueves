"""`ao.fock`: the two-electron Fock matrix `G = J - K/2` by every ordered
shell quartet, weight 1, no symmetry -- the witness milestone 12's fused,
symmetric build is checked against (#275).

THE CROSS-CHECKS (no PySCF, no toolchain: tracked fixtures only).

  FJ's paths   the sweep's `J` and `K` against `ao.j.coulomb_matrix` and
               `ao.k.exchange_matrix` (and their ordered references), and
               `G` against `J - K/2` built from them -- the combination the
               module docstring DERIVES, applied to the matrices FJ already
               trusts.
  libcint      `G` against `vj - vk/2` from the `_j` / `_k` fixture pair,
               after asserting the pair shares its inputs (milestone 12,
               pin 10). PySCF's relation is the derivation's consequence,
               checked here, not its source.
  can fail     a deliberately wrong evaluator (physicists' notation for
               chemists', and one ordered quartet perturbed) and the wrong
               combinations (`J - K`, `2J - K`, `J - K/4`) each miss by far
               more than the tolerance.

THE L <= 2 RULE (milestone 10's, kept for K's sake: no K-bearing case above
L = 2 runs in the fast tier, #231). A whole sweep over a basis holding a `p`
shell visits `(pp|pp)`, L = 4. So:

  fast   H2/STO-3G whole (s only), and the s sub-basis of H2O/STO-3G (its
         four s shells and the matching block of the fixture's density); and,
         against libcint, the s-s BLOCK of `G` on H2O/STO-3G and H2O/6-31G
         through an `eri` hook that returns zeros, computing nothing, for a
         quartet with L > 2. That block is exact: a quartet `(ab|cd)` writes
         `J[a][b]` and `K[a][c]`, so it touches an s-s element of `J` only
         with L = lc + ld, and of `K` only with L = lb + ld -- at most 2 in
         a basis with no `d`. (The hook is also the injection, exercised.)
  slow   H2O/STO-3G, H2O/6-31G and H2O/cc-pVDZ whole.

THE TOLERANCES, the tree's, stated and not tuned.

  sweeps     1e-14 of max|ref|, element by element (`ao.j` / `ao.k`'s
             SWEEP_TOL): the same integrals summed in a different order.
  symmetry   |G - G^T| <= 1e-15 max|G| (the sweep computes both triangles).
  libcint    1e-12 of `scale_G = scale_J + scale_K / 2`, element by element,
             where `scale_J`, `scale_K` are the sweep itself run on `|D|`
             with the ERI's absolute-value twin (`eri_block_and_scale`), so
             #85's per-ERI 1e-12 carries over by the triangle inequality
             (`ao.j`'s and `ao.k`'s JK_TOL / KJ_TOL).

A failure is a defect to find, not a tolerance to widen.
"""

from __future__ import annotations

from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao import fock
from felizjueves.ao.eri import eri_block, eri_block_and_scale
from felizjueves.ao.fock import (
    coulomb_exchange_reference,
    fock_build_reference,
    fock_two_electron_reference,
)
from felizjueves.ao.j import coulomb_matrix, coulomb_matrix_reference
from felizjueves.ao.k import exchange_matrix, exchange_matrix_reference
from felizjueves.ao.shell import Basis
from felizjueves.reference.pq import pq

SWEEP_TOL = 1e-14
SYM_TOL = 1e-15
LIBCINT_TOL = 1e-12
L_MAX_RUN = 2

ERI = partial(eri_block, pq=pq)


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx)


@cache
def _case(name: str) -> tuple[Basis, np.ndarray]:
    """A basis and its density: a `_j` fixture, or H2O/STO-3G's s part."""
    if name != "h2o_sto3g_s":
        fx, basis = _fixture(name)
        return basis, np.array(fx.density)
    basis, D = _case("h2o_sto3g_j")
    keep = [i for i, s in enumerate(basis.shells) if s.l == 0]
    aos = [basis.offsets[i] for i in keep]  # one AO per s shell
    return Basis(tuple(basis.shells[i] for i in keep)), D[np.ix_(aos, aos)]


def _lmax_quartet(basis: Basis) -> int:
    return 4 * max(s.l for s in basis.shells)


def _evaluators(truncate: bool):
    """`(value, scale)` ERI hooks over one memo of `eri_block_and_scale`, so
    the value sweep and the scale sweep share each quartet's evaluation.
    With `truncate`, a quartet with L > 2 is zeros, nothing computed."""
    memo: dict[tuple[int, ...], tuple] = {}

    def both(bra, ket):
        key = tuple(id(s) for s in (*bra, *ket))
        if key not in memo:
            if truncate and sum(s.l for s in (*bra, *ket)) > L_MAX_RUN:
                z = tuple((0.0,) * (ket[0].n_cart * ket[1].n_cart)
                          for _ in range(bra[0].n_cart * bra[1].n_cart))
                memo[key] = (z, z)
            else:
                memo[key] = eri_block_and_scale(bra, ket, pq=pq)
        return memo[key]

    return (lambda b, k: both(b, k)[0]), (lambda b, k: both(b, k)[1])


@cache
def _libcint_case(m: str, truncate: bool):
    """`G`, `J`, `K`, `scale_G = scale_J + scale_K / 2` (the sweep on `|D|`
    with the ERI scale), and `vj - vk/2`, for molecule `m`."""
    basis, D, vj, vk = _pair(m)
    value, scale = _evaluators(truncate)
    G = np.array(fock_two_electron_reference(basis, D, eri=value))
    J, K = (np.array(x) for x in coulomb_exchange_reference(basis, D,
                                                            eri=value))
    sj, sk = (np.array(x) for x in coulomb_exchange_reference(
        basis, np.abs(D), eri=scale))
    return basis, G, J, K, sj + fock.EXCHANGE_FRACTION * sk, vj - 0.5 * vk


FAST = ("h2_sto3g_j", "h2o_sto3g_s")
SLOW = ("h2o_sto3g_j", "h2o_631g_j", "h2o_ccpvdz_j")
PARAMS = ([pytest.param(n, id=n) for n in FAST]
          + [pytest.param(n, id=n, marks=pytest.mark.slow) for n in SLOW])

# every tracked molecule with a density and both PySCF matrices
PAIRS = ("h2_sto3g", "h2o_sto3g", "h2o_631g", "h2o_ccpvdz")
LIBCINT_FAST_WHOLE = ("h2_sto3g",)
LIBCINT_FAST_SS = ("h2o_sto3g", "h2o_631g")
LIBCINT_SLOW_WHOLE = ("h2o_sto3g", "h2o_631g", "h2o_ccpvdz")


def test_the_fast_cases_keep_to_the_rule():
    """Checked rather than trusted: every fast whole case reaches only
    L <= 2, the s-s block cases have no `d`, and the slow ones do not."""
    for name in FAST:
        assert _lmax_quartet(_case(name)[0]) <= L_MAX_RUN, name
    for name in SLOW:
        assert _lmax_quartet(_case(name)[0]) > L_MAX_RUN, name
    for m in LIBCINT_FAST_WHOLE:
        assert _lmax_quartet(_fixture(f"{m}_j")[1]) <= L_MAX_RUN, m
    for m in LIBCINT_FAST_SS:
        basis = _fixture(f"{m}_j")[1]
        assert max(s.l for s in basis.shells) == 1, m
    assert set(LIBCINT_FAST_WHOLE) | set(LIBCINT_SLOW_WHOLE) == set(PAIRS)
    # every tracked fixture carrying both `vj` (in `_j`) and `vk` (in `_k`)
    with_both = {p.name[:-len("_j.json")] for p in manifest.tracked()
                 if p.name.endswith("_j.json")
                 and (p.parent / p.name.replace("_j.json", "_k.json")).exists()}
    assert with_both == set(PAIRS)


def test_the_combination_is_derived_not_typed():
    """The constants are the derivation's: Coulomb sees the whole density,
    exchange one spin's, `D / N_SPIN`, closed shell."""
    assert fock.N_SPIN == 2
    assert fock.COULOMB_FRACTION == 1.0
    assert fock.EXCHANGE_FRACTION == 1.0 / fock.N_SPIN == 0.5
    assert fock_build_reference is fock_two_electron_reference


def _rel(x, ref) -> float:
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))


@pytest.mark.parametrize("name", PARAMS)
def test_fock_matches_fj_j_and_k(name):
    """The sweep's `J` and `K` equal FJ's (unique and ordered sweeps), and
    `G` equals `J - K/2` combined from FJ's own matrices; `G` is symmetric
    though nothing in the sweep made it so."""
    basis, D = _case(name)
    Js, Ks = (np.array(x) for x in coulomb_exchange_reference(basis, D,
                                                              eri=ERI))
    G = np.array(fock_two_electron_reference(basis, D, eri=ERI))
    J = np.array(coulomb_matrix(basis, D, pq=pq))
    Jr = np.array(coulomb_matrix_reference(basis, D, pq=pq))
    K = np.array(exchange_matrix(basis, D, pq=pq))
    Kr = np.array(exchange_matrix_reference(basis, D, pq=pq))
    want = J - 0.5 * K
    errs = {"J": max(_rel(Js, J), _rel(Js, Jr)),
            "K": max(_rel(Ks, K), _rel(Ks, Kr)),
            "G": _rel(G, want)}
    sym = float(np.max(np.abs(G - G.T)) / np.max(np.abs(G)))
    print(f"\n{name}: " + ", ".join(f"{k} {v:.2e}" for k, v in errs.items())
          + f", symmetry {sym:.2e}, max|G| {np.max(np.abs(G)):.3g}")
    assert np.array_equal(G, Js - 0.5 * Ks)
    for k, v in errs.items():
        assert v <= SWEEP_TOL, f"{name}: {k} disagrees with FJ's, {v:.2e}"
    assert sym <= SYM_TOL, f"{name}: G asymmetric, {sym:.2e}"


def _pair(m: str):
    """The `_j` / `_k` pair of molecule `m`, asserted to share its inputs
    (pin 10) before `vj` and `vk` are combined."""
    fj, basis = _fixture(f"{m}_j")
    fk, _ = _fixture(f"{m}_k")
    assert fk.provenance.mol == fj.provenance.mol, m
    assert fk.provenance.basis == fj.provenance.basis, m
    assert fk.provenance.cart is fj.provenance.cart is True, m
    assert fk.density_recipe == fj.density_recipe, m
    assert (fk.atoms, fk.shells, fk.nao, fk.ao_labels, fk.ao_loc,
            fk.converted) == (fj.atoms, fj.shells, fj.nao, fj.ao_labels,
                              fj.ao_loc, fj.converted), m
    assert np.array_equal(fk.density, fj.density), m
    vj = fj.arrays[manifest.J_ARRAY]
    vk = fk.arrays[manifest.K_ARRAY]
    return basis, np.array(fj.density), vj, vk


def _check_libcint(m, G, ref, scale, where):
    err = np.abs(G - ref)
    assert np.all(scale > 0)
    worst = float(np.max(err / scale))
    print(f"\n{m} ({where}): max|G - (vj - vk/2)| = {err.max():.2e}, "
          f"max err/scale = {worst:.2e}")
    bad = err > LIBCINT_TOL * scale
    assert not bad.any(), (f"{m} ({where}): {int(bad.sum())} of {err.size} "
                           f"elements exceed 1e-12 * scale; worst {worst:.2e}")


@pytest.mark.parametrize(
    "m", [pytest.param(m, id=m) for m in LIBCINT_FAST_WHOLE]
    + [pytest.param(m, id=m, marks=pytest.mark.slow)
       for m in LIBCINT_SLOW_WHOLE])
def test_fock_matches_libcint_whole(m):
    """`G` against `vj - vk/2` from the tracked pair, whole matrix."""
    _, G, _, _, scale, ref = _libcint_case(m, False)
    _check_libcint(m, G, ref, scale, "whole")


def _ss(basis: Basis):
    return [basis.offsets[i] for i, s in enumerate(basis.shells) if s.l == 0]


@pytest.mark.parametrize("m", LIBCINT_FAST_SS)
def test_fock_matches_libcint_ss_block(m):
    """`G`'s s-s block against `vj - vk/2`'s, through the truncating hook
    (module docstring: the block is exact, L <= 2 throughout)."""
    basis, G, _, _, scale, ref = _libcint_case(m, True)
    s = np.ix_(_ss(basis), _ss(basis))
    _check_libcint(m, G[s], ref[s], scale[s], "s-s block")


@pytest.mark.parametrize("m", LIBCINT_FAST_WHOLE + LIBCINT_FAST_SS)
def test_the_wrong_combinations_miss(m):
    """The comparison can tell the derived `J - K/2` from the combinations a
    re-typing would ship: `J - K` (exchange over both spins), `2J - K`
    (everything doubled, the same physics at the wrong scale) and `J - K/4`.
    Each misses `vj - vk/2` by far more than the tolerance."""
    basis, _, J, K, scale, ref = _libcint_case(m, m in LIBCINT_FAST_SS)
    s = np.ix_(_ss(basis), _ss(basis))
    J, K, scale, ref = J[s], K[s], scale[s], ref[s]
    assert np.all(np.abs(J - 0.5 * K - ref) <= LIBCINT_TOL * scale)
    for name, wrong in (("J - K", J - K), ("2J - K", 2 * J - K),
                        ("J - K/4", J - 0.25 * K)):
        worst = float(np.max(np.abs(wrong - ref) / scale))
        assert worst > 1e6 * LIBCINT_TOL, f"{m}: {name} not discriminated"


def test_a_wrong_evaluator_fails():
    """The ERI is injected, so the sweep can be handed a wrong one, and the
    comparison with FJ's `J - K/2` must catch it: (i) physicists' notation
    `<ab|cd> = (ac|bd)` where chemists' `(ab|cd)` is meant, on an s basis
    (every block 1 x 1, so only the index convention is wrong); (ii) one
    ordered quartet out of 256 perturbed by 1e-3 -- which also breaks `G`'s
    symmetry, showing the sweep copies no triangle."""
    basis, D = _case("h2o_sto3g_s")
    assert all(s.l == 0 for s in basis.shells) and len(basis.shells) == 4
    want = (np.array(coulomb_matrix(basis, D, pq=pq))
            - 0.5 * np.array(exchange_matrix(basis, D, pq=pq)))
    good = np.array(fock_two_electron_reference(basis, D, eri=ERI))
    assert _rel(good, want) <= SWEEP_TOL

    def physicist(bra, ket):
        return eri_block((bra[0], ket[0]), (bra[1], ket[1]), pq=pq)

    G = np.array(fock_two_electron_reference(basis, D, eri=physicist))
    assert _rel(G, want) > 1e8 * SWEEP_TOL

    sh = basis.shells
    target = (sh[0], sh[1], sh[2], sh[3])

    def one_off(bra, ket):
        blk = eri_block(bra, ket, pq=pq)
        if all(x is y for x, y in zip((*bra, *ket), target, strict=True)):
            return [[v * (1 + 1e-3) for v in r] for r in blk]
        return blk

    G = np.array(fock_two_electron_reference(basis, D, eri=one_off))
    assert _rel(G, want) > 1e6 * SWEEP_TOL
    assert float(np.max(np.abs(G - G.T)) / np.max(np.abs(G))) > 1e6 * SYM_TOL


def test_every_ordered_quartet_once():
    """Weight 1, no symmetry: the sweep asks for each of the `n_sh^4` ordered
    quartets exactly once (the shells here are pairwise distinct objects)."""
    basis, D = _case("h2o_sto3g_s")
    seen: list[tuple[int, ...]] = []
    idx = {id(s): i for i, s in enumerate(basis.shells)}

    def counting(bra, ket):
        seen.append(tuple(idx[id(s)] for s in (*bra, *ket)))
        return eri_block(bra, ket, pq=pq)

    fock_two_electron_reference(basis, D, eri=counting)
    n = len(basis.shells)
    assert len(seen) == n ** 4 == len(set(seen))


def test_refusals():
    """A block of the wrong shape, and an asymmetric density, are refused."""
    basis, D = _case("h2_sto3g_j")
    with pytest.raises(ValueError, match="wrong shape"):
        fock_two_electron_reference(basis, D, eri=lambda b, k: [[0.0, 0.0]])
    with pytest.raises(ValueError, match="wrong shape"):
        fock_two_electron_reference(basis, D, eri=lambda b, k: 1.0)
    Da = D.copy()
    Da[0, 1] += 1e-3
    with pytest.raises(ValueError, match="not symmetric"):
        fock_two_electron_reference(basis, Da, eri=ERI)
