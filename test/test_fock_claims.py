"""Two of `[Fock]` §3.2's printed defects, carried to the two-electron `F`
(#291): `fock/batch-magnitude-bound-direction` and
`fock/eq29-screen-inequality-reversed`.

test_fock_pairs.py shows each printed rule dropping a quartet that holds an
ERI `>= tau`. The records ask for more -- that the drop is checked against
the UNSCREENED two-electron `F` of `eri/fock/fock.py` (#275) -- so here each rule
is applied to `launch_f`'s pairs (#285) and the quartets it would remove are
zeroed in `eri.fock.fock`'s every-ordered-quartet sweep (test_fock_build.py's
`sweep`, fed by `eri.eri`, MMD through `reference.pq`; no HGP in it):

  * the PRINTED rule removes quartets with `G_ab G_cd >= tau`, and the `F`
    it leaves misses the unscreened `F` by more than `tau`;
  * the BUILT rule removes only quartets with `G_ab G_cd < tau` (so every
    ERI they hold is below `tau`, by Schwarz, computed), and the `F` it
    leaves differs from the unscreened one by exactly what it removed --
    bounded by the sweep of `|D| |(ab|cd)|` over those quartets alone, to
    rounding.

One case, H2O/STO-3G at `tau = 1e-2` and `n = 2`, chosen because both
rules fire on it (the built screen removes one batch pair; asserted, so the
built half is not vacuous). Fast tier: no marker, no compiler, no card.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import test_fock_build as FB

from felizjueves.eri.fock.launch_f import (
    f_launches,
    magnitude_index,
    prelaunch_screened,
    printed_magnitude_index,
    printed_prelaunch_screened,
)

NAME, TAU, N = "h2o_sto3g", 1e-2, 2


def _plan():
    basis, D, _ = FB.molecule(NAME)
    return f_launches(basis, D, bound=FB.BOUND, tau=TAU, n=N)


def _unique(plan) -> list[tuple]:
    """Every unique quartet `{P, Q}` of the plan's pairs (deleted ones
    included: none is deleted here, asserted), as `(P, Q)` shell pairs."""
    assert not plan.deleted
    ps = plan.pairs
    return [(ps[i], ps[j]) for i in range(len(ps)) for j in range(i, len(ps))]


def _all_ordered() -> frozenset:
    basis, _, _ = FB.molecule(NAME)
    return frozenset(itertools.product(range(len(basis.shells)), repeat=4))


def _f_without(dropped) -> tuple[np.ndarray, np.ndarray]:
    """(`G` with the dropped quartets' members zeroed, the sweep of `|D|
    |(ab|cd)|` over those members alone)."""
    gone = FB.members((P.shell_pair, Q.shell_pair) for P, Q in dropped)
    return (FB.sweep(NAME, _all_ordered() - gone),
            FB.sweep(NAME, gone, block="abs"))


def _max_eri(P, Q) -> float:
    (a, b), (c, d) = P.shell_pair, Q.shell_pair
    return float(np.max(np.abs(FB.eris(NAME)[(a, b, c, d)][0])))


def _rounding() -> np.ndarray:
    return FB.TOL * FB.sweep(NAME, block="scale")


def _built_half(plan, full) -> None:
    """The plan as built: what eq. 29 removes is below `tau` and moves `F`
    by no more than itself."""
    by_batch = {(i, j) for i in range(len(plan.batches))
                for j in range(i, len(plan.batches))
                if prelaunch_screened(plan.batches[i].mag,
                                      plan.batches[j].mag, N)}
    assert by_batch and set(plan.screened) == by_batch
    dropped = [(P, Q) for i, j in plan.screened
               for P in plan.batches[i].pairs for Q in plan.batches[j].pairs]
    assert dropped
    for P, Q in dropped:
        assert P.G * Q.G < TAU
        assert _max_eri(P, Q) < TAU
    G, removed = _f_without(dropped)
    assert np.all(np.abs(full - G) <= removed + _rounding())
    assert np.max(removed) > 0  # it did remove something


@pytest.mark.claim("fock/batch-magnitude-bound-direction")
def test_the_printed_assignment_drops_f_and_the_upper_bound_one_does_not() -> None:
    """`fock/batch-magnitude-bound-direction`: assign each pair the largest
    magnitude BELOW its `G` (§3.2 as printed) and screen by eq. 29 (`i + j
    >= n`), and the removed quartets include some with `G_ab G_cd >= tau`
    and an ERI `>= tau`, and `F` misses the unscreened `F` by more than
    `tau`. The upper-bound assignment the pair stream uses (`G < m_i` for
    `i >= 1`) removes none such: the built half."""
    plan = _plan()
    full = FB.sweep(NAME)
    mags = plan.magnitudes
    for p in plan.pairs:
        assert p.mag == magnitude_index(p.G, mags)
        if p.mag:  # m_0 = 1 takes every larger G, and is never screened
            assert p.G < mags[p.mag]  # the magnitude bounds G from above
    idx = {p.shell_pair: printed_magnitude_index(p.G, mags) for p in plan.pairs}
    dropped = [(P, Q) for P, Q in _unique(plan)
               if prelaunch_screened(idx[P.shell_pair], idx[Q.shell_pair], N)]
    above = [(P, Q) for P, Q in dropped if P.G * Q.G >= TAU]
    assert above
    assert max(_max_eri(P, Q) for P, Q in above) >= TAU
    G, _ = _f_without(dropped)
    assert float(np.max(np.abs(full - G))) > TAU
    _built_half(plan, full)


@pytest.mark.claim("fock/eq29-screen-inequality-reversed")
def test_eq29s_printed_inequality_drops_f_and_i_plus_j_at_least_n_does_not(
) -> None:
    """`fock/eq29-screen-inequality-reversed`: on eq. 29's magnitudes the
    batch pairs the pair stream never launches are exactly those with `i + j
    >= n` (the rest are all launched), and they move `F` by no more than
    themselves; the printed `i + j <= n`, read as removal, removes the
    largest batches -- quartets with `G_ab G_cd >= tau` and an ERI `>= tau`
    -- and `F` misses the unscreened `F` by more than `tau`. Fig. 1's
    launch matrix at `n = 2` is the built one, and the printed inequality
    would launch none of it."""
    plan = _plan()
    full = FB.sweep(NAME)
    nb = len(plan.batches)
    upper = {(i, j) for i in range(nb) for j in range(i, nb)}
    launched = {(x.bra, x.ket) for x in plan}
    assert launched.isdisjoint(plan.screened)
    assert launched | set(plan.screened) == upper
    _built_half(plan, full)

    printed = [(i, j) for i, j in sorted(upper)
               if printed_prelaunch_screened(plan.batches[i].mag,
                                             plan.batches[j].mag, N)]
    dropped = [(P, Q) for i, j in printed
               for P in plan.batches[i].pairs for Q in plan.batches[j].pairs]
    above = [(P, Q) for P, Q in dropped if P.G * Q.G >= TAU]
    assert above
    assert max(_max_eri(P, Q) for P, Q in above) >= TAU
    G, _ = _f_without(dropped)
    assert float(np.max(np.abs(full - G))) > TAU

    # Fig. 1's Kernel Launches panel, n = 2: every pair holding the
    # magnitude-1 batch launched, the (sqrt(tau), sqrt(tau)) one not
    assert {(i, j): not prelaunch_screened(i, j, 2)
            for i in range(2) for j in range(2)} == {
        (0, 0): True, (0, 1): True, (1, 0): True, (1, 1): False}
    assert all(printed_prelaunch_screened(i, j, 2)
               for i in range(2) for j in range(2))
