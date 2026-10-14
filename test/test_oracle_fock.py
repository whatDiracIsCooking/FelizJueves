"""`F` against libcint: `vj` and `vk` from one fixture pair (#280).

The two-electron Fock matrix `G = J - K/2` of `felizjueves.ao.fock` (#275,
every ordered quartet, weight 1, no symmetry) against PySCF's `vj - vk/2`,
combined from the tracked `<m>_j` / `<m>_k` fixtures through the manifest's
named oracle quantity, `manifest.fock_pair(m).two_electron()`. No new fixture,
no PySCF run, no `oracle` marker: the comparison reads tracked files only.

THE PAIRING (milestone 12, pin 10), the part that can silently rot: before
`vj` and `vk` are combined, `manifest.check_pair` asserts the two files share
`shells`, `density`, `ao_loc`, `common_factor`, `provenance.mol` (and the rest
of their inputs, exactly). Here that check runs on all four molecules, and
is shown to fail loudly on mismatched pairs built in memory -- one field at a
time, a pair of two different molecules, and a raw file whose geometry moved.

THE FAST / SLOW SPLIT (the tree's L <= 2 rule for K-bearing cases, #231):

  fast   the pairing check on all four molecules (it reads files, nothing
         more); `G` whole on H2/STO-3G (s only); `G`'s s-s block on
         H2O/STO-3G and H2O/6-31G through an `eri` hook that returns zeros
         for a quartet with L > 2 (exact for that block: a quartet `(ab|cd)`
         writes `J[a][b]` and `K[a][c]`, so an s-s element meets L = lc + ld
         or L = lb + ld only, at most 2 without a `d` shell).
  slow   `G` whole on H2O/STO-3G, H2O/6-31G and H2O/cc-pVDZ.

THE TOLERANCE is the existing J and K comparisons': 1e-12 of `scale_G =
scale_J + scale_K / 2`, element by element, the scales being the sweep run
on `|D|` with the ERI's absolute-value twin (`eri_block_and_scale`), so
#85's per-ERI 1e-12 carries over. Stated, not tuned.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from oracle.manifest import FixtureError

from felizjueves.ao import fock
from felizjueves.ao.eri import eri_block_and_scale
from felizjueves.ao.fock import coulomb_exchange_reference
from felizjueves.reference.pq import pq

LIBCINT_TOL = 1e-12
L_MAX_RUN = 2

MOLECULES = ("h2_sto3g", "h2o_sto3g", "h2o_631g", "h2o_ccpvdz")
FAST_WHOLE = ("h2_sto3g",)
FAST_SS = ("h2o_sto3g", "h2o_631g")
SLOW_WHOLE = ("h2o_sto3g", "h2o_631g", "h2o_ccpvdz")


@cache
def _pair(m: str) -> manifest.FockPair:
    return manifest.fock_pair(m)


# --- the pairing ------------------------------------------------------------


def test_the_pairs_are_discovered_not_listed():
    """Every tracked molecule with both halves is one of the four, and the
    split covers each: whole fast or whole slow, s-s fast only without `d`."""
    assert manifest.fock_pairs() == sorted(MOLECULES)
    assert set(FAST_WHOLE) | set(SLOW_WHOLE) == set(MOLECULES)
    for m in FAST_WHOLE:
        assert 4 * max(s.l for s in ob.from_fixture(_pair(m).j).shells) \
            <= L_MAX_RUN, m
    for m in SLOW_WHOLE:
        assert 4 * max(s.l for s in ob.from_fixture(_pair(m).j).shells) \
            > L_MAX_RUN, m
    for m in FAST_SS:
        assert max(s.l for s in ob.from_fixture(_pair(m).j).shells) == 1, m


@pytest.mark.parametrize("m", MOLECULES)
def test_each_pair_shares_its_inputs(m):
    """`check_pair` accepts the tracked pair, and the issue's five fields are
    equal by direct comparison too (not only through the checker)."""
    p = _pair(m)
    j, k = p.j, p.k
    assert j.name == f"{m}_j" and k.name == f"{m}_k"
    assert j.shells == k.shells
    assert np.array_equal(j.density, k.density)
    assert j.ao_loc == k.ao_loc
    assert j.converted.common_factor == k.converted.common_factor
    assert j.provenance.mol == k.provenance.mol
    assert manifest.J_ARRAY not in k.arrays and manifest.K_ARRAY not in j.arrays
    g = p.two_electron()
    assert g.shape == (j.nao, j.nao) and not g.flags.writeable
    assert np.array_equal(g, j.arrays["vj"] - 0.5 * k.arrays["vk"])


def test_the_combination_constant_is_the_derivations():
    """The manifest may not import `src`, so its 1/2 is checked here against
    `ao.fock`'s derived `EXCHANGE_FRACTION = 1 / N_SPIN`."""
    assert manifest.FOCK_EXCHANGE_FRACTION == fock.EXCHANGE_FRACTION == 0.5


def _bump(t: tuple, i: int = 0) -> tuple:
    return (*t[:i], t[i] + 1, *t[i + 1:])


MISMATCHES = {
    # field: (how to rot the `_k` half, the message the refusal must carry)
    "shells": (lambda k: dataclasses.replace(
        k, shells=(dataclasses.replace(
            k.shells[0], exponents=_bump(k.shells[0].exponents)),
            *k.shells[1:])), "shells differ"),
    "density": (lambda k: dataclasses.replace(
        k, arrays={**k.arrays, "density": k.density * (1 + 2 ** -40)}),
        "density differ"),
    "ao_loc": (lambda k: dataclasses.replace(
        k, ao_loc=(*k.ao_loc[:-1], k.ao_loc[-1] + 1)), "ao_loc differ"),
    "common_factor": (lambda k: dataclasses.replace(
        k, converted=dataclasses.replace(
            k.converted,
            common_factor=(k.converted.common_factor[0] * 2,
                           *k.converted.common_factor[1:]))),
        "common_factor differ"),
    "provenance.mol": (lambda k: dataclasses.replace(
        k, provenance=dataclasses.replace(
            k.provenance, mol={**k.provenance.mol, "charge": 1})),
        "provenance.mol differ"),
    "ovlp": (lambda k: dataclasses.replace(
        k, arrays={**k.arrays, "ovlp": k.arrays["ovlp"] * 2}),
        "arrays.ovlp differ"),
    "missing vk": (lambda k: dataclasses.replace(
        k, arrays={n: a for n, a in k.arrays.items() if n != "vk"}),
        "carries no vk"),
}


@pytest.mark.parametrize("field", list(MISMATCHES))
def test_a_mismatched_pair_is_refused(field):
    """One field of the `_k` half rotted in memory: `check_pair` refuses,
    naming both files and that field. The density is rotted by one part in
    2^40 -- far below any numerical tolerance -- and still refused."""
    p = _pair("h2o_sto3g")
    rot, msg = MISMATCHES[field]
    bad = rot(p.k)
    with pytest.raises(FixtureError, match=msg) as e:
        manifest.check_pair(p.j, bad)
    assert str(p.j.path) in str(e.value) and str(p.k.path) in str(e.value)


def test_two_molecules_are_not_a_pair():
    """The realistic rot: halves from different molecules (the same water,
    two bases; and H2 against water) are refused, whichever comes first."""
    sto, g31, h2 = (_pair(m) for m in ("h2o_sto3g", "h2o_631g", "h2_sto3g"))
    for fj, fk in ((sto.j, g31.k), (g31.j, sto.k), (h2.j, sto.k)):
        with pytest.raises(FixtureError, match="not a Fock pair"):
            manifest.check_pair(fj, fk)
    with pytest.raises(FixtureError, match="carries no vj"):
        manifest.check_pair(sto.k, sto.j)  # halves swapped


def test_a_moved_geometry_in_a_raw_file_is_refused(tmp_path):
    """A `_k` file re-written with the molecule moved (a regenerated half
    that no longer matches): it still loads as a valid fixture, and
    `fock_pair` refuses it at the first input that differs."""
    for half in ("_j", "_k"):
        src = manifest.FIXTURES / f"h2_sto3g{half}.json"
        raw = json.loads(src.read_text())
        if half == "_k":
            raw["provenance"]["mol"]["atom"] = "H 0 0 0; H 0.3 -0.4 1.4"
        (tmp_path / src.name).write_text(manifest.dumps(raw))
    manifest.load(tmp_path / "h2_sto3g_k.json")  # valid on its own
    with pytest.raises(FixtureError, match="provenance.mol differ"):
        manifest.fock_pair("h2_sto3g", tmp_path)


# --- F against libcint --------------------------------------------------------


def _evaluators(truncate: bool):
    """`(value, scale)` hooks over one memo of `eri_block_and_scale`; with
    `truncate`, a quartet with L > 2 is zeros and nothing is computed."""
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


def _compare(m: str, truncate: bool):
    """`G` from the sweep, its scale, and the pair's `vj - vk/2`, restricted
    to the s-s block when `truncate`; asserted within 1e-12 of scale."""
    p = _pair(m)
    basis = ob.from_fixture(p.j)
    D = np.array(p.density)
    value, scale = _evaluators(truncate)
    J, K = (np.array(x) for x in coulomb_exchange_reference(basis, D,
                                                            eri=value))
    sj, sk = (np.array(x) for x in coulomb_exchange_reference(
        basis, np.abs(D), eri=scale))
    G = J - fock.EXCHANGE_FRACTION * K
    S = sj + fock.EXCHANGE_FRACTION * sk
    ref = p.two_electron()
    if truncate:
        ss = [basis.offsets[i] for i, s in enumerate(basis.shells) if s.l == 0]
        ix = np.ix_(ss, ss)
        G, S, ref = G[ix], S[ix], ref[ix]
    assert np.all(S > 0)
    err = np.abs(G - ref)
    worst = float(np.max(err / S))
    print(f"\n{m} ({'s-s block' if truncate else 'whole'}): "
          f"max|G - (vj - vk/2)| = {err.max():.2e}, max err/scale = "
          f"{worst:.2e}")
    bad = err > LIBCINT_TOL * S
    assert not bad.any(), (f"{m}: {int(bad.sum())} of {err.size} elements "
                           f"exceed 1e-12 * scale; worst {worst:.2e}")
    return G, ref, S


@pytest.mark.parametrize(
    "m", [pytest.param(m, id=m) for m in FAST_WHOLE]
    + [pytest.param(m, id=m, marks=pytest.mark.slow) for m in SLOW_WHOLE])
def test_fock_matches_the_pair_whole(m):
    """`G` from the ordered sweep against the pair's `vj - vk/2`, whole."""
    _compare(m, truncate=False)


@pytest.mark.parametrize("m", FAST_SS)
def test_fock_matches_the_pair_ss_block(m):
    """`G`'s s-s block against the pair's, L <= 2 throughout (docstring)."""
    _compare(m, truncate=True)
