"""#486: the H10 chain's recorded density is RHF's, and it decays.

`h10_chain_ccpvdz_{j,k}` record PySCF's converged RHF density (recipe "rhf",
`oracle.generate.rhf_density`) where every other fixture records the hilbert
matrix `1/(i+j+1)`, which does not decay with distance. That pair exists so a
density-decay screen (#484's per-block `F_ab`) can fire on a libcint-checked
input. Tracked files only -- no PySCF, no marker, fast tier.

What "RHF's" means here, checked from the recorded arrays alone: exactly
symmetric; `Tr(D S)` = 10 electrons; idempotent in the metric (`D S D = 2 D`,
the total closed-shell density); stationary (`F D S - S D F` ~ 0, `F = hcore
+ vj - vk / 2` from the pair); and its energy `Tr(D h) + Tr(D G) / 2 + E_nn`
is the one the SCF reported when the fixture was written.

MEASURED (pyscf 2.14.0, when the fixture was written), not tuned:
  E_RHF                          -5.637578626458 Eh
  max |F D S - S D F|            5.3e-12
  max |D S D - 2 D|              1.6e-15
  max |D|                        0.3278
  max |D| unit 0 to unit u,      u = 1: 6.2e-2, 2: 6.3e-3, 3: 9.3e-4,
    over max |D|                 4 (the end units, ~21.6 bohr): 9.9e-5
  the hilbert density's corner (first 5 x last 5 AOs of H2O/cc-pVDZ):
    4.8e-2 of its max -- the contrast the fixture exists for.
"""

from __future__ import annotations

import numpy as np
import pytest
from oracle import manifest

PAIR = "h10_chain_ccpvdz"
N_ELECTRONS = 10
N_UNITS = 5  # H2 molecules, two atoms each, in atom order along z
E_RHF = -5.637578626458  # measured; the SCF's energy at regeneration
E_TOL = 1e-9
TRACE_TOL = 1e-10
IDEMPOTENT_TOL = 1e-12  # measured 1.6e-15
STATIONARY_TOL = 1e-9  # measured 5.3e-12; conv_tol_grad is 1e-10
END_DECAY = 1e-3  # measured 9.9e-5 of max |D|: a 10x margin


@pytest.fixture(scope="module")
def pair() -> manifest.FockPair:
    return manifest.fock_pair(PAIR)


def _units(fx: manifest.Fixture) -> np.ndarray:
    """The H2 unit of each AO (AO label's atom index // 2)."""
    return np.array([int(lab.split()[0]) // 2 for lab in fx.ao_labels])


def test_only_the_chain_records_an_rhf_density():
    by_recipe: dict[str, set[str]] = {}
    for p in manifest.tracked():
        fx = manifest.load(p)
        if fx.gradient is not None:  # #550: each converges its own RHF
            assert fx.density_recipe.startswith("rhf:"), fx.name
            continue
        if fx.metric is None and fx.three_center is None:
            by_recipe.setdefault(fx.density_recipe.split(":")[0],
                                 set()).add(fx.name)
    assert by_recipe["rhf"] == {f"{PAIR}_j", f"{PAIR}_k"}
    # #520: the density-free dipole fixture (manifest: THE DENSITY-FREE
    # DIPOLE FIXTURE), and #570's two one-electron pieces
    assert by_recipe["none"] == {"hf_ccpvtz_spd_dipole", "co_ccpvtz_j_st",
                                 "co_ccpvtz_j_vh"}
    assert set(by_recipe) == {"hilbert", "rhf", "none"}


def test_the_density_is_a_converged_rhf_density(pair):
    D = np.array(pair.density)
    S = pair.j.arrays[manifest.OVLP_ARRAY]
    h = pair.j.arrays[manifest.HCORE_ARRAY]
    assert np.array_equal(D, D.T)
    assert abs(float(np.trace(D @ S)) - N_ELECTRONS) <= TRACE_TOL
    assert np.max(np.abs(D @ S @ D - 2 * D)) <= IDEMPOTENT_TOL
    F = h + pair.two_electron()
    assert np.max(np.abs(F @ D @ S - S @ D @ F)) <= STATIONARY_TOL
    xyz = np.array([a.coord for a in pair.j.atoms])
    z = [a.charge for a in pair.j.atoms]
    e_nn = sum(z[a] * z[b] / float(np.linalg.norm(xyz[a] - xyz[b]))
               for a in range(len(z)) for b in range(a))
    e = float(np.sum(D * h) + 0.5 * np.sum(D * pair.two_electron()) + e_nn)
    assert abs(e - E_RHF) <= E_TOL, e


def test_the_density_decays_along_the_chain(pair):
    """max |D| between unit 0 and unit u falls with u, and between the end
    units (~21.6 bohr apart) is under END_DECAY of max |D|."""
    D = np.abs(np.array(pair.density))
    unit = _units(pair.j)
    assert sorted(set(unit)) == list(range(N_UNITS))
    top = float(D.max())
    far = [float(D[np.ix_(unit == 0, unit == u)].max()) / top
           for u in range(1, N_UNITS)]
    print(f"\nmax|D| {top:.4g}; unit 0 to unit u, over max|D|: "
          + ", ".join(f"{u + 1}: {r:.2e}" for u, r in enumerate(far)))
    assert all(a > b for a, b in zip(far, far[1:], strict=False))
    assert far[-1] < END_DECAY


def test_the_hilbert_density_does_not_decay():
    """The contrast: the hilbert matrix's far corner is a few percent of its
    max, so a decay screen cannot fire on any other fixture."""
    H = np.array(manifest.load(manifest.FIXTURES / "h2o_ccpvdz_j.json")
                 .density)
    assert float(np.abs(H[:5, -5:]).max()) / float(np.abs(H).max()) > 1e-2
