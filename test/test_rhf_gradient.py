"""`gradient.rhf`: the RHF nuclear gradient assembled, interpreted (#565).

THE WITNESS is PySCF (`oracle.manifest.rhf_gradient`, written by `python -m
oracle.generate --regenerate --rhf-gradient`): per molecule its converged
`D` and `W`, every term libcint's derivative integrals contracted with
them, and the total from `nuc_grad_method().kernel()` -- held to the
terms' sum to 1e-10 by the generator and again by the loader. The witness
`D` and `W` are this side's inputs too, so a term compares the kernels
against libcint at one density, and the total compares against PySCF's own
gradient route.

- THE ASSEMBLY, on stub runners: which matrix each runner gets, each term's
  sign and half, the total as their fixed-order sum, the refusals.
- PER TERM (the one-electron and nuclear ones alone, fast) AND IN TOTAL
  (every term, the RHF assembly's), every kernel interpreted (`gradient.interpreted`):
  `gs_`, `gt_`, `gv_` (Hellmann-Feynman included), the J chain (`[p|q]` one
  bra order up, then `gjab_`) and `gk_`, plus the host's `V_nn^x`, within
  `ABS_TOL = 1e-10` absolute (the issue's bound) of PySCF, on H2O/STO-3G,
  H2O/6-31G, C1 NH3/STO-3G and H2O/cc-pVDZ. Each total carries no net
  force to rounding.
- THE HOST ROUTE (`host_runners`): the same assembly over the references,
  on H2O/STO-3G.

TIERS, by `tiers.by_l` on each kernel's raised class: the one-electron
terms on s, p bases reach `(pp) + 1 = 3` and stay fast; the J chain's
`[p|q]` and `gk_` reach L = 5 at `(pp|pp) + 1` (the milestone's one class
above 4), and cc-pVDZ's d shells take every term there, so they are
`slow`. Interpreted `gk_` dominates: ~2 minutes on H2O/STO-3G alone.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l

from felizjueves.gradient import host, rhf
from felizjueves.gradient.interpreted import interpreted_runners
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre

ABS_TOL = 1e-10
NET_TOL = 1e-12
ONE_E = ("overlap", "kinetic", "potential")
WITNESSES = ("h2o_sto3g_rhf_grad", "h2o_631g_rhf_grad",
             "nh3_sto3g_rhf_grad", "h2o_ccpvdz_rhf_grad")


@cache
def _witness(name: str):
    w = manifest.rhf_gradient(name)
    return w, ob.from_fixture(w), w.nuclei


def _lmax(name: str) -> int:
    return max(s.l for s in _witness(name)[1].shells)


def _term_l(name: str, term: str) -> int:
    """The highest raised class `term` builds on `name` (module docstring)."""
    lm = _lmax(name)
    if term in ONE_E:
        return 2 * lm + 1
    if term == "nuclear":
        return 0
    return 4 * lm + 1  # gjab's [p|q] at (lp + 1, lq), gk_ at L + 1


def _params(terms):
    return [by_l((n, t), _term_l(n, t), id=f"{n}-{t}")
            for n in WITNESSES for t in terms]


def _run(name: str, term: str) -> np.ndarray:
    w, basis, nuclei = _witness(name)
    if term == "nuclear":
        return np.array(host.nuclear_repulsion_gradient(nuclei))
    M = w.weighted if term == "overlap" else w.density
    G = getattr(interpreted_runners(boys=boys), term)(basis, nuclei, M)
    return rhf.SIGN[term] * np.asarray(G)


def _close(got, want, what: str) -> None:
    err = float(np.abs(np.asarray(got) - want).max())
    assert err <= ABS_TOL, f"{what}: {err:.2e} from PySCF, over {ABS_TOL:g}"


# --- the witnesses ------------------------------------------------------------


@pytest.mark.parametrize("name", WITNESSES)
def test_every_witness_loads_and_carries_no_net_force(name: str) -> None:
    w, basis, nuclei = _witness(name)
    assert basis.n_ao == w.density.shape[0]
    assert host.shell_atoms(basis, nuclei)  # every shell on its atom
    for t, g in (*w.terms.items(), ("total", w.total)):
        assert np.abs(g.sum(axis=0)).max() <= NET_TOL * max(
            1.0, float(np.abs(g).max())), t


def test_the_c1_molecule_has_no_mirror() -> None:
    """C1: a symmetry operation fixes N (the one heavy atom) and permutes the
    H's keeping their N-H distances, which all differ, so it fixes every
    atom; four fixed non-coplanar points leave only the identity."""
    _, _, nuclei = _witness("nh3_sto3g_rhf_grad")
    R = np.array([c for _, c in nuclei])
    assert abs(np.linalg.det(R[1:] - R[0])) > 1.0
    d = sorted(np.linalg.norm(R[i] - R[0]) for i in (1, 2, 3))
    assert min(np.diff(d)) > 0.01


# --- the assembly, on stub runners ----------------------------------------------


def _stub(tag: float, seen: dict, name: str):
    def run(basis, nuclei, M):
        seen[name] = np.asarray(M)
        return [[tag * (a + 1) + k for k in range(3)] for a in range(len(nuclei))]
    return run


def test_the_assembly_signs_each_term_and_sums_them_in_order() -> None:
    w, basis, nuclei = _witness("h2o_sto3g_rhf_grad")
    seen: dict = {}
    stubs = rhf.Runners(**{t: _stub(10.0 ** i, seen, t)
                           for i, t in enumerate(rhf.TERMS[:-1])})
    r = rhf.rhf_gradient(basis, nuclei, w.density, w.weighted, stubs)
    assert np.array_equal(seen["overlap"], w.weighted)
    for t in rhf.TERMS[1:-1]:
        assert np.array_equal(seen[t], w.density), t
    assert rhf.SIGN == {"overlap": -1.0, "kinetic": 1.0, "potential": 1.0,
                        "coulomb": 1.0, "exchange": -0.5, "nuclear": 1.0}
    for i, t in enumerate(rhf.TERMS[:-1]):
        want = [[rhf.SIGN[t] * (10.0 ** i * (a + 1) + k) for k in range(3)]
                for a in range(len(nuclei))]
        assert np.array_equal(r.term(t), want), t
    assert np.array_equal(r.nuclear, host.nuclear_repulsion_gradient(nuclei))
    total = np.zeros((len(nuclei), 3))
    for t in rhf.TERMS:
        total += np.array(r.term(t))
    assert np.array_equal(r.total, total)


def test_the_assembly_refuses_a_bad_g_a_stray_shell_and_an_unknown_term(
        ) -> None:
    w, basis, nuclei = _witness("h2o_sto3g_rhf_grad")
    stubs = {t: _stub(1.0, {}, t) for t in rhf.TERMS[:-1]}
    bad = rhf.Runners(**stubs | {"kinetic": lambda b, n, M: [[0.0] * 3]})
    with pytest.raises(ValueError, match="kinetic: G is not 3 x 3"):
        rhf.rhf_gradient(basis, nuclei, w.density, w.weighted, bad)
    with pytest.raises(ValueError, match="sits on no nucleus"):
        rhf.rhf_gradient(basis, nuclei[:2], w.density, w.weighted,
                         rhf.Runners(**stubs))
    r = rhf.rhf_gradient(basis, nuclei, w.density, w.weighted,
                         rhf.Runners(**stubs))
    with pytest.raises(ValueError, match="no term 'vhf'"):
        r.term("vhf")


# --- per term and in total, interpreted, against PySCF --------------------------


@pytest.mark.parametrize("case", _params((*ONE_E, "nuclear")))
def test_each_one_electron_and_nuclear_term_is_pyscfs(case) -> None:
    name, term = case
    _close(_run(name, term), _witness(name)[0].terms[term], f"{name} {term}")


@pytest.mark.parametrize("name", [
    by_l(n, max(_term_l(n, t) for t in rhf.TERMS), id=n) for n in WITNESSES])
def test_the_interpreted_rhf_gradient_is_pyscfs(name: str) -> None:
    w, basis, nuclei = _witness(name)
    r = rhf.rhf_gradient(basis, nuclei, w.density, w.weighted,
                         interpreted_runners(boys=boys))
    for t in rhf.TERMS:
        _close(r.term(t), w.terms[t], f"{name} {t}")
    _close(r.total, w.total, f"{name} total")
    G = np.array(r.total)
    assert np.abs(G.sum(axis=0)).max() <= NET_TOL * float(np.abs(G).max())


@pytest.mark.slow
def test_the_host_route_is_pyscfs() -> None:
    w, basis, nuclei = _witness("h2o_sto3g_rhf_grad")
    r = rhf.rhf_gradient(basis, nuclei, w.density, w.weighted,
                         rhf.host_runners(one_centre=one_centre))
    for t in rhf.TERMS:
        _close(r.term(t), w.terms[t], f"host {t}")
    _close(r.total, w.total, "host total")
