"""The screen as an ERROR claim (#88): a tau ladder, on a molecule where it fires.

`ao.j.coulomb_matrix_screened` builds `J` by the unique sweep with the kernel's
screen -- a (bra record, ket record) pair is skipped when
`I_bra * (I_ket * Dmax) < tau` -- either as the kernel's sorted `break` or as
the unsorted `continue`. Here it runs at

    tau in TAUS = (0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)

on two cases, and the result is data/ao/screening.json (regenerate with
`PYTHONPATH=src uv run python test/test_ao_screening.py --write`).

THE CLAIMS, per rung, per case.

  exact    sorted-`break` J == unsorted-`continue` J, element by element, TO 0,
           and the two skip the same number of pairs -- the claim
           test_kernel.py makes on synthetic data, here on real bounds. It is
           a claim about WHICH terms were summed: `ao.j` accumulates each `J_p`
           with `math.fsum`, so summation order cannot move a bit.
  bounded  |J(tau) - J(0)| <= tau * scale, element by element. `scale`
           (`Screened.scale`, stated in ao/j.py's module docstring) is
           `n_ket * sum_{bra records} sum_p |E^{ab}_p|`: every skipped pair
           moves each `J_p` by less than `tau` (ao/pairs.py's bound), and a
           bra record can skip at most `n_ket` kets. It is independent of
           `tau`, so the recorded `err_over_scale` must be <= tau.
  monotone max |J(tau) - J(0)| is non-decreasing in tau (and so is the skip
           count: the skipped sets are nested).

THE CASES -- and the hazard the issue names: every compact molecule screens
nothing, so a compact case alone proves the screen by never testing it.

  compact   H2O/STO-3G, the `h2o_sto3g_j` fixture's basis and density. Its
            skip fraction is REPORTED at every rung, 0.0 included, and the
            test says where it is 0 -- there the error claim is vacuous (the
            error is exactly 0 because nothing was dropped), which is what
            "a compact molecule cannot exercise a screen" means. `slow`: the
            p shells make it ~20 s.
  extended  four H2 molecules (bond 1.4 bohr) whose centres are 6 bohr apart,
            collinear on z, STO-3G hydrogen from the `h2_sto3g_j` fixture's
            converted basis, and the fixtures' hilbert density
            `D[i][j] = 1/(i+j+1)` (so every `D_cd` is non-zero and a skip
            comes from the decay of the ket's dressed `E`, not from a zero in
            `D`). FOUR molecules, not more: that is #122's measured extent.
            When the ladder was last run `reference.boys` refused
            `T > 2000` (#122 had extended it from 500 with a measured large-T
            asymptote), and a fifth H2 puts a bra and a ket 25.4 bohr apart:
            `T = theta^2 |PQ|^2` reaches ~2210 (four reach ~1289). #265 has
            since carried the evaluator's measurement to T_MAX = 1e6, so it
            now reaches much further; the chain is kept at #122's extent
            (EXTENT_122) so the ladder, and screening.json, are not rewritten.
            Every shell is s, so only F_0 is ever asked for.
            `test_the_chain_is_as_long_as_the_evaluator_allows` pins both.
            Fast tier: it is small (~8 s).

`test_screening_json_is_recomputed` (`slow`) rebuilds both ladders and requires
the data file to equal them exactly; the fast tier checks the extended ladder
live and the file's recorded claims for both.

No timings: this is an accuracy claim, and the work saved is a skip fraction.
"""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao import j as aoj
from felizjueves.ao.j import coulomb_matrix, coulomb_matrix_screened
from felizjueves.ao.pairs import primitive_pairs
from felizjueves.ao.shell import Basis, Shell
from felizjueves.reference.boys import T_MAX
from felizjueves.reference.pq import pq

TAUS = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)
LOOSE = 1e-8  # "the looser rungs": tau >= this must skip on the extended case
SWEEP_TOL = 1e-14  # fsum vs naive accumulation at tau = 0 (test_ao_j's)
DATA = Path(__file__).resolve().parent.parent / "data" / "ao" / "screening.json"

BOND = 1.4  # bohr, H-H
SPACING = 6.0  # bohr, centre to centre of neighbouring H2
N_H2 = 4
# reference.boys' T_MAX as #122 measured it: the chain was sized to it, and is
# kept at it after #265 widened the evaluator, so screening.json stands.
EXTENT_122 = 2000.0


def _hilbert(n: int) -> np.ndarray:
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])


def _h2_chain(n: int) -> Basis:
    h = ob.from_fixture(manifest.load(manifest.FIXTURES / "h2_sto3g_j.json"))
    s = h.shells[0]
    return Basis(tuple(
        Shell(0, (0.0, 0.0, SPACING * m + dz), s.exponents, s.coefficients)
        for m in range(n) for dz in (0.0, BOND)))


@cache
def _case(name: str):
    if name == "compact":
        fx = manifest.load(manifest.FIXTURES / "h2o_sto3g_j.json")
        return ob.from_fixture(fx), np.array(fx.density)
    assert name == "extended"
    basis = _h2_chain(N_H2)
    return basis, _hilbert(basis.n_ao)


DESCRIPTION = {
    "compact": "H2O/STO-3G, Cartesian: the h2o_sto3g_j fixture's converted "
               "basis and its hilbert density",
    "extended": f"{N_H2} H2 (bond {BOND} bohr) with centres {SPACING} bohr "
                "apart, collinear on z; STO-3G H from the h2_sto3g_j "
                "fixture's converted basis; hilbert density "
                "D[i][j] = 1/(i+j+1)",
}


@cache
def ladder(name: str) -> dict:
    """Every rung, both orders; the record data/ao/screening.json holds."""
    basis, D = _case(name)
    zero = coulomb_matrix_screened(basis, D, pq=pq, tau=0.0)
    J0, S = np.array(zero.J), np.array(zero.scale)
    assert np.all(S > 0)
    naive = np.array(coulomb_matrix(basis, D, pq=pq))
    rungs = []
    for tau in TAUS:
        a = coulomb_matrix_screened(basis, D, pq=pq, tau=tau, order="sorted")
        b = coulomb_matrix_screened(basis, D, pq=pq, tau=tau,
                                    order="unsorted")
        err = np.abs(np.array(a.J) - J0)
        rungs.append({
            "tau": tau,
            "max_err": float(err.max()),
            "err_over_scale": float(np.max(err / S)),
            "err_over_max_J": float(err.max() / np.abs(J0).max()),
            "bounded": bool(np.all(err <= tau * S)),
            "n_skipped": a.n_skipped,
            "skip_fraction": a.skip_fraction,
            "sorted_equals_unsorted": a.J == b.J,
            "unsorted_n_skipped": b.n_skipped,
        })
    return {
        "description": DESCRIPTION[name],
        "n_ao": basis.n_ao,
        "n_shells": len(basis.shells),
        "n_pairs": zero.n_pairs,
        "max_abs_J0": float(np.abs(J0).max()),
        "max_scale": float(S.max()),
        "min_scale": float(S.min()),
        "J0_vs_coulomb_matrix": float(np.max(np.abs(J0 - naive))
                                      / np.abs(naive).max()),
        "rungs": rungs,
    }


def document() -> dict:
    return {
        "schema": 1,
        "about": ("#88: the J screen as an error claim. J(tau) by "
                  "ao.j.coulomb_matrix_screened (unique sweep, sorted break), "
                  "against J(0) from the same path. max_err = max |J(tau) - "
                  "J(0)|; err_over_scale = max over elements of |J(tau) - "
                  "J(0)| / scale, scale = n_ket * sum_{bra records} sum_p "
                  "|E^{ab}_p| (ao/j.py), which the bound says is <= tau; "
                  "skip_fraction = skipped / all (bra record, ket record) "
                  "index pairs. The evaluator is reference.pq.pq. No timings."),
        "command": "PYTHONPATH=src uv run python test/test_ao_screening.py "
                   "--write",
        "taus": list(TAUS),
        "cases": {name: ladder(name) for name in ("compact", "extended")},
    }


def _check(name: str, rec: dict) -> None:
    rungs = rec["rungs"]
    assert [r["tau"] for r in rungs] == list(TAUS)
    assert rec["J0_vs_coulomb_matrix"] <= SWEEP_TOL, name
    for r in rungs:
        tau = r["tau"]
        assert r["sorted_equals_unsorted"], f"{name} tau={tau}: break != continue"
        assert r["unsorted_n_skipped"] == r["n_skipped"], (name, tau)
        assert r["bounded"], f"{name} tau={tau}: err/scale {r['err_over_scale']}"
        assert r["err_over_scale"] <= tau, (name, tau, r["err_over_scale"])
        assert 0.0 <= r["skip_fraction"] <= 1.0
        if r["n_skipped"] == 0:  # nothing dropped: identical sums
            assert r["max_err"] == 0.0, (name, tau)
    assert rungs[0]["n_skipped"] == 0 and rungs[0]["max_err"] == 0.0
    for lo, hi in zip(rungs, rungs[1:], strict=False):
        assert hi["max_err"] >= lo["max_err"], (name, lo["tau"], hi["tau"])
        assert hi["n_skipped"] >= lo["n_skipped"], (name, lo["tau"], hi["tau"])


def _table(name: str, rec: dict) -> str:
    out = [f"{name}: {rec['description']}",
           "   tau      max err     err/scale   skip fraction"]
    out += [f"  {r['tau']:7.0e}  {r['max_err']:10.3e}  {r['err_over_scale']:10.3e}"
            f"   {r['skip_fraction']:.4f}" for r in rec["rungs"]]
    return "\n".join(out)


# --- live --------------------------------------------------------------------


def test_extended_ladder_live():
    """Every claim, recomputed, on the case where the screen fires; and it
    does fire at the looser rungs (else: a finding for limits.json)."""
    rec = ladder("extended")
    print("\n" + _table("extended", rec))
    _check("extended", rec)
    for r in rec["rungs"]:
        if r["tau"] >= LOOSE:
            assert r["n_skipped"] > 0, (
                f"tau={r['tau']}: the extended case skipped nothing -- a "
                "finding about the bound, for limits.json")


def test_the_chain_is_as_long_as_the_evaluator_allows():
    """Four H2, because a fifth takes `T` past #122's measured extent
    (EXTENT_122 = 2000): the chain's length was that evaluator's limit, and is
    kept so the ladder is not rewritten. Since #265 the evaluator reaches
    further (T_MAX = 1e6), so the chain must still be inside it. M_MAX cannot
    bite, the chain being all s shells (fundamentals need m = 0 only)."""
    assert all(s.l == 0 for s in _h2_chain(N_H2 + 1).shells)
    def max_T(basis: Basis) -> float:
        recs = [k for a in basis.shells for b in basis.shells
                for k in primitive_pairs(a, b)]
        return max(
            x.zeta * y.zeta / (x.zeta + y.zeta)
            * sum((x.P[d] - y.P[d]) ** 2 for d in range(3))
            for x in recs for y in recs)
    assert max_T(_h2_chain(N_H2)) <= EXTENT_122 < max_T(_h2_chain(N_H2 + 1))
    assert T_MAX >= max_T(_h2_chain(N_H2))


def test_the_break_needs_the_sort():
    """Mirror of test_kernel.py's: the sort is stable and descending by
    exactly the screened factor, and a break on the unsorted kets drops
    work the continue keeps (so a wrong answer, not just a slower one)."""
    basis, D = _case("extended")
    n_sh = len(basis.shells)
    pairs = [(a, b, 1.0 if a == b else 2.0)
             for a in range(n_sh) for b in range(a + 1)]
    kets = aoj._kets(basis, aoj._density(basis, D), pairs, pq=pq, tau=1.0)
    srt = aoj._sorted_kets(kets)
    pos = {id(k): i for i, k in enumerate(kets)}
    assert all(x.bound >= y.bound for x, y in zip(srt, srt[1:], strict=False))
    ties = [(x, y) for x, y in zip(srt, srt[1:], strict=False)
            if x.bound == y.bound]
    assert ties, "the chain's symmetry gives equal bounds; stability is tested"
    assert all(pos[id(x)] < pos[id(y)] for x, y in ties)

    tau = 1e-8
    bras = primitive_pairs(basis.shells[0], basis.shells[0])
    brk = aoj._bra_block_screened(bras, kets, pq=pq, tau=tau, brk=True)
    cont = aoj._bra_block_screened(bras, kets, pq=pq, tau=tau, brk=False)
    assert brk[2] > cont[2] and brk[0] != cont[0]
    ok = aoj._bra_block_screened(bras, srt, pq=pq, tau=tau, brk=True)
    assert ok[0] == cont[0] and ok[2] == cont[2]


def test_a_bad_order_is_rejected():
    basis, D = _case("extended")
    with pytest.raises(ValueError, match="order"):
        coulomb_matrix_screened(basis, D, pq=pq, tau=0.0, order="random")


# --- the recorded ladder -------------------------------------------------------


def _recorded() -> dict:
    return json.loads(DATA.read_text())


@pytest.mark.parametrize("name", ["compact", "extended"])
def test_recorded_ladder_claims(name):
    """The file's claims, both cases. The compact case's skip fraction is
    reported at every rung, 0.0 included: where it is 0 the error claim is
    vacuous, and the test says so instead of passing quietly."""
    doc = _recorded()
    assert doc["taus"] == list(TAUS)
    rec = doc["cases"][name]
    print("\n" + _table(name, rec))
    _check(name, rec)
    for r in rec["rungs"]:
        assert isinstance(r["skip_fraction"], float)
    if name == "compact":
        ext = doc["cases"]["extended"]["rungs"]
        idle = [r["tau"] for r in rec["rungs"] if r["n_skipped"] == 0]
        print(f"compact: skips NOTHING at tau in {idle} -- a compact molecule "
              "cannot exercise the screen there; the error claim is vacuous "
              "at those rungs and carried by the extended case")
        assert len(idle) > 1, "compact skipped at tau <= 1e-14: news"
        for c, e in zip(rec["rungs"][1:], ext[1:], strict=True):
            assert c["skip_fraction"] < e["skip_fraction"], c["tau"]


@pytest.mark.slow
def test_screening_json_is_recomputed():
    """Both ladders rebuilt from scratch equal the tracked file exactly."""
    want = document()
    print("\n" + "\n\n".join(_table(n, r) for n, r in want["cases"].items()))
    for name, rec in want["cases"].items():
        _check(name, rec)
    assert _recorded() == want, (
        "data/ao/screening.json is stale: PYTHONPATH=src uv run python "
        "test/test_ao_screening.py --write")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_ao_screening.py --write")
    doc = document()
    for name, rec in doc["cases"].items():
        _check(name, rec)
        print(_table(name, rec), end="\n\n")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {DATA}")
