"""`eri.k_exchange.k.density_decay`: the per-block density factor of [Fock, §4.3.1]
(#484).

THE CLAIMS, on a molecule whose density decays -- `chain()`, six H2 4 bohr
apart, one s primitive per atom, and a synthetic density `D_mn = 0.6
exp(-|z_m - z_n|)`: symmetric, and decaying with `|MN|` as a localised
density does (a converged SCF density needs PySCF, which the fast tier has
not; the bound holds for ANY symmetric `D`, so the synthetic one tests it).

  the rule    `Lambda >= 0` fitted, `H_off` the exact reduction, `F_ab <=
              max|D|` and symmetric; no visible decay gives `Lambda = 0`
              and `F_ab = max|D|` exactly; H2O/STO-3G's factor IS `max|D|`
              on the whole tau ladder.
  the bound   with NO tolerance (#200's pattern): every `(c, d)` a block can
              visit has `|D_cd| <= F_ab`, every visited record passes the
              cutoff, and every skipped record pair's own bound `I_ac I_bd
              max|D_cd|` is under `tau`.
  it fires    the factor skips strictly more record pairs than `max|D|` does
              (the counts pinned).
  the answer  screened `K` against unscreened `K`, element by element, within
              the sum of the skipped pairs' bounds (`KScreened.bound`) -- the
              walk's and the host sweep's; the host's sorted and unsorted
              orders skip the same set and give the same `K` to 0.

Fast tier: s shells only, no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from functools import cache, partial

import numpy as np
import pytest
from test_launch_k import TAUS
from test_launch_k import _case as _launch_case

from felizjueves.ao.density import checked_density
from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.k_exchange.k import (
    DECAY_MARGIN,
    density_decay,
    exchange_matrix_screened,
    record_bounds,
)
from felizjueves.eri.k_exchange.launch_kblock import kb_launches, walk_kb
from felizjueves.reference.pq import pq

BOUND = partial(record_bounds, pq=pq)
TAU = 1e-6
# The pinned counts: record pairs the walk skips at TAU, max|D| for every
# block against the per-block factor, of all the plan's pairs (#484).
SKIPPED = {"dm": 17_980, "decay": 18_154, "of": 20_736}


@cache
def chain(n: int = 6, spacing: float = 4.0, bond: float = 1.4,
          kappa: float = 1.0) -> tuple[Basis, np.ndarray]:
    shells = tuple(Shell(0, (0.0, 0.0, spacing * m + dz), (1.0,), (1.0,))
                   for m in range(n) for dz in (0.0, bond))
    basis = Basis(shells)
    z = np.array([s.centre[2] for s in shells])
    return basis, 0.6 * np.exp(-kappa * np.abs(z[:, None] - z[None, :]))


@cache
def _decay(tau: float = TAU):
    basis, D = chain()
    return density_decay(basis, D, bound=BOUND, tau=tau)


@cache
def _plan(decay: bool, tau: float = TAU):
    basis, D = chain()
    return kb_launches(basis, D, bound=BOUND, tau=tau,
                       decay=_decay(tau) if decay else None)


@cache
def _walk(decay: bool, tau: float = TAU):
    basis, D = chain()
    return walk_kb(_plan(decay, tau), basis, D, pq=pq)


def _off_atom(basis: Basis, D: np.ndarray):
    c = [s.centre for s in basis.shells]  # s shells: one AO per shell
    return [(float(np.linalg.norm(np.subtract(c[m], c[n]))), abs(D[m, n]))
            for m in range(len(c)) for n in range(len(c)) if c[m] != c[n]]


def test_the_rule_on_the_chain():
    basis, D = chain()
    dec = _decay()
    assert dec.lam > 0.0 and dec.dm == float(np.max(np.abs(D)))
    off = _off_atom(basis, D)
    prods = [v * np.exp(dec.lam * r) for r, v in off]
    assert dec.h_off == max(prods)  # the exact reduction, attained
    n = len(basis.shells)
    F = {(a, b): dec.factor(a, b, 0, 0) for a in range(n) for b in range(n)}
    assert all(0.0 <= f <= dec.dm for f in F.values())
    assert all(F[a, b] == F[b, a] for a, b in F)
    assert any(f < dec.dm for f in F.values())
    # the plan carries exactly these factors, and dm without a decay
    (c,) = _plan(True)
    assert all(blk.dmax == F[blk.a, blk.b] for blk in c.blocks)
    (c,) = _plan(False)
    assert all(blk.dmax == dec.dm for blk in c.blocks)


def test_no_visible_decay_gives_lambda_zero_and_dm():
    basis, _ = chain()
    D = np.full((basis.n_ao, basis.n_ao), 0.3)
    dec = density_decay(basis, D, bound=BOUND, tau=TAU)
    assert dec.lam == 0.0 and dec.h_off == 0.3
    n = len(basis.shells)
    assert all(dec.factor(a, b, 0, 0) == dec.dm
               for a in range(n) for b in range(n))


@pytest.mark.parametrize("name", ["h2o_sto3g_j", "h2_sto3g_j"])
def test_the_factor_is_max_d_on_compact_molecules(name):
    basis, D = _launch_case(name)
    for tau in TAUS:
        dec = density_decay(basis, D, bound=BOUND, tau=tau)
        P = kb_launches(basis, D, bound=BOUND, tau=tau, decay=dec)
        assert all(blk.dmax == P.dm for c in P for blk in c.blocks), tau


def test_the_bound_holds_with_no_tolerance():
    basis, D = chain()
    rows = np.array(checked_density(basis, D))
    dec = _decay()
    (c,) = _plan(True)
    cut = TAU * (1.0 - DECAY_MARGIN)
    passes = lambda r: r.G * (dec.g_max * dec.dm) >= cut  # noqa: E731
    n_visited = n_skipped = 0
    for blk in c.blocks:
        bras = c.bras[blk.bra0:blk.bra1]
        kets = c.kets[blk.ket0:blk.ket1]
        g0 = kets[0].G
        for br in bras:
            for kr in kets:
                dcd = abs(rows[br.j, kr.j])  # s shells: one AO each
                if passes(br) and passes(kr):
                    assert dcd <= blk.dmax, (blk, br, kr)
                visited = ((br.G * g0) * blk.dmax >= TAU
                           and (br.G * kr.G) * blk.dmax >= TAU)
                if visited:
                    n_visited += 1
                    assert passes(br) and passes(kr), (blk, br, kr)
                else:
                    n_skipped += 1
                    assert (br.G * kr.G) * dcd < TAU, (blk, br, kr)
    assert n_visited > 0 and n_skipped > 0


def test_the_screen_fires():
    w0, w1 = _walk(False), _walk(True)
    assert w0.n_pairs == w1.n_pairs == SKIPPED["of"]
    assert (w0.n_skipped, w1.n_skipped) == (SKIPPED["dm"], SKIPPED["decay"])
    assert w1.n_skipped > w0.n_skipped


def _within(K, K0, bound) -> None:
    K, K0, B = np.array(K), np.array(K0), np.array(bound)
    room = 2.0 * np.spacing(np.maximum(np.abs(K), np.abs(K0)))  # two fsums
    assert np.all(np.abs(K - K0) <= B + room)


def test_the_walk_moves_within_the_bound():
    K0 = _walk(False, 0.0).K
    w = _walk(True)
    assert np.max(np.array(w.bound)) > 0.0
    _within(w.K, K0, w.bound)
    _within(_walk(False).K, K0, _walk(False).bound)


def test_the_host_sweep_takes_the_factor():
    basis, D = chain()
    dec = _decay()
    s = exchange_matrix_screened(basis, D, pq=pq, tau=TAU, decay=dec)
    u = exchange_matrix_screened(basis, D, pq=pq, tau=TAU, decay=dec,
                                 order="unsorted")
    assert s.K == u.K and s.n_skipped == u.n_skipped
    plain = exchange_matrix_screened(basis, D, pq=pq, tau=TAU)
    assert s.n_skipped > plain.n_skipped
    K0 = exchange_matrix_screened(basis, D, pq=pq, tau=0.0).K
    _within(s.K, K0, s.bound)
    _within(plain.K, K0, plain.bound)


def test_a_decay_from_another_cut_is_refused():
    basis, D = chain()
    with pytest.raises(ValueError, match="another density"):
        kb_launches(basis, D, bound=BOUND, tau=1e-8, decay=_decay())
    with pytest.raises(ValueError, match="another density"):
        exchange_matrix_screened(basis, 0.5 * D, pq=pq, tau=TAU,
                                 decay=_decay())
    with pytest.raises(ValueError, match="width"):
        density_decay(basis, D, bound=BOUND, tau=TAU, width=0.0)
    blk = _plan(True).classes[0].blocks[0]
    with pytest.raises(ValueError, match="density factor"):
        dataclasses.replace(_plan(True).classes[0],
                            blocks=(dataclasses.replace(blk, dmax=None),))
