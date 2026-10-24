"""The capstone (#243): the core Hamiltonian `h = T + V` built on the card,
and the one-electron energy `Tr(D h)`, on the RTX 3080 and the RX 9060 XT.

WHY A SEPARATE CHECK. `T` and `V` each pass their own libcint oracle in
test_one_electron_device.py (#242). A normalisation or sign slip SHARED by
the two -- the class of bug the predecessor's milestone #13 pinned -- can
hide from both if the oracles share it; `hcore` is PySCF's own `get_hcore`,
read independently, and `Tr(D h)` contracts the whole matrix against a
density, so a per-class constant cannot hide in a corner of it. It is also
the first quantity here compared to a physical observable (an energy term)
rather than to an integral array. It scores nothing published: no claim, no
table cell, no measured row.

THE BUILD. One launch per molecule (`launch.one_electron_launch` at
`tau = 0`, nothing dropped), and from it `T` and `V` on the card by
`one_electron_driver.build_on_card` (#242), unchanged -- the same records,
the same generated units (`tier_units`: under devtools/kernel-tier.sh, the
`--one-e-l 2` files the generate phase wrote, byte-identical to `emit`; the
`la > lb` classes the driver emits itself). `h_dev = T_dev + V_dev` is the
one host-side addition, for each of the driver's `RUNS` runs.

THE REFERENCE, said plainly. "PySCF's one-electron energy" is
`E1 = einsum('ij,ji', D, hcore)` of TWO TRACKED ARRAYS of the fixture: its
`hcore` (PySCF `scf.hf.get_hcore`, checked by `oracle/generate.py` to equal
`int1e_kin + int1e_nuc` to 1e-14 before it was written) and its `density`
(`density_recipe`: the Hilbert matrix `D_ij = 1/(i+j+1)`, symmetric
positive definite, no RNG). No scalar is stored and PySCF is not run here;
the contraction is this module's. The density is not an SCF density, so
`E1` is not H2O's one-electron energy at convergence -- it is the same
functional `Tr(D h)` an SCF reads, at a density that weights every element.

THE TOLERANCES, stated before any run.

  element:  |h_dev - hcore| <= ONE_E_TOL (s_T + s_V)
            -- test_ao_potential.py's `T + V - hcore` bound, unchanged:
            `ONE_E_TOL = 1e-12` times the absolute-value scales
            `one_e_scale(E_T)` and `one_e_scale(E_V)` that
            test_ao_{kinetic,potential}.py state (every product the host
            adds, before any of it cancels).
  trace:    |Tr(D h_dev) - Tr(D hcore)|
                <= ONE_E_TOL SUM_ij |D_ji| (s_T + s_V)_ij
                   + 2 gamma_{n^2} SUM_ij |D_ji| (|hcore| + ONE_E_TOL s_h)_ij
            -- the element bound contracted against |D| (triangle
            inequality), plus each side's own fp64 contraction of n^2
            terms, `gamma_k = k u / (1 - k u)`, `u = 2^-53`.

WHAT THE CHECK IS FOR, demonstrated numerically (fast tier, from the
tracked `int1e_kin` / `int1e_nuc`, and again from the device's own `T` and
`V` in the printed line): flipping the sign of `-Z_C` (h = T - V), dropping
`T` (h = V), dropping `V` (h = T), or a normalisation slip shared by `T`
and `V` (every d function scaled by 1.01 in both) each moves `Tr(D h)` by
at least `SENSITIVITY = 1e6` trace tolerances. The measured multiples are
printed; they are far above that floor (1e8 to 1e11 here).

THE FAST TIER also holds the HOST's `T + V` (`ao.kinetic` + `ao.potential`)
to the same two tolerances against the fixture: the same comparison with
no card, so a device failure here is the device's.

The device tests are `gpu` plus `cuda` or `hip`, behind the card lock and
the VRAM probe; without the card they skip, and kernel-tier.sh exits 2.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from one_electron_driver import build_on_card, kernel_of, tier_units
from oracle import basis as ob
from oracle import manifest
from test_ao_kinetic import kinetic_scale_block
from test_ao_overlap import ONE_E_TOL, one_e_scale
from test_ao_potential import potential_scale_block
from test_kernel_device import BACKEND, _toolchain

from felizjueves.ao.pairs import overlap_bound
from felizjueves.eri.j_coulomb.launch import OneELaunch, one_electron_launch
from felizjueves.one_electron.kinetic import kinetic_matrix
from felizjueves.one_electron.one_electron import matrix
from felizjueves.one_electron.potential import potential_matrix
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Target

U = 2.0 ** -53
# classes through (dd); H2O/cc-pVDZ is `slow`, off the PR gate
MOLECULES = ("h2o_sto3g_j",
             pytest.param("h2o_ccpvdz_j", marks=pytest.mark.slow))
LMAX = 2  # the tier's `--one-e-l 2`
SENSITIVITY = 1e6  # each slip moves Tr(D h) by at least this many tolerances
D_SLIP = 1.01  # the shared normalisation slip: every d function, T and V


def _gamma(n: int) -> float:
    return n * U / (1.0 - n * U)


def _tr(D: np.ndarray, h: np.ndarray) -> float:
    return float(np.einsum("ij,ji", D, h))


# ---------------------------------------------------------------------------
# Inputs, scales and tolerances
# ---------------------------------------------------------------------------


@cache
def _case(name: str):
    """(fixture, basis, nuclei); the nuclei are the fixture's atoms."""
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx), tuple((float(a.charge), a.coord)
                                          for a in fx.atoms)


@cache
def _launch(name: str) -> OneELaunch:
    _, basis, nuclei = _case(name)
    launch = one_electron_launch(basis, nuclei, tau=0.0, bound=overlap_bound)
    assert not launch.dropped
    assert max(max(c.la, c.lb) for c in launch) <= LMAX
    return launch


@cache
def _scale_h(name: str) -> np.ndarray:
    """`s_T + s_V`: test_ao_potential.py's hcore scale."""
    _, basis, nuclei = _case(name)
    s_t = one_e_scale(np.array(matrix(basis, kinetic_scale_block)))
    s_v = one_e_scale(np.array(matrix(
        basis, lambda sa, sb: potential_scale_block(sa, sb, nuclei))))
    return s_t + s_v


def _tol_trace(name: str) -> float:
    fx, _, _ = _case(name)
    D, H, s = fx.density, fx.arrays[manifest.HCORE_ARRAY], _scale_h(name)
    n2 = D.size
    return (ONE_E_TOL * _tr(np.abs(D), s)
            + 2.0 * _gamma(n2) * _tr(np.abs(D), np.abs(H) + ONE_E_TOL * s))


def _e1(name: str) -> float:
    """PySCF's one-electron energy for the fixture's density: Tr(D hcore)
    of the two tracked arrays (module docstring)."""
    fx, _, _ = _case(name)
    return _tr(fx.density, fx.arrays[manifest.HCORE_ARRAY])


def _d_functions(fx) -> np.ndarray:
    """A mask of the fixture's d functions, from its `ao_labels`."""
    return np.array([lab.split()[2][1] == "d" for lab in fx.ao_labels])


def _slips(name: str, T: np.ndarray, V: np.ndarray) -> dict[str, float]:
    """`|Tr(D h') - Tr(D (T + V))|` for each slip `h'` (module docstring)."""
    fx, _, _ = _case(name)
    D = fx.density
    ref = _tr(D, T + V)
    out = {"-Z_C flipped": _tr(D, T - V), "T dropped": _tr(D, V),
           "V dropped": _tr(D, T)}
    d = _d_functions(fx)
    if d.any():
        N = np.diag(np.where(d, D_SLIP, 1.0))
        out[f"d norm x{D_SLIP} in T and V"] = _tr(D, N @ (T + V) @ N)
    return {k: abs(v - ref) for k, v in out.items()}


def _check(label: str, name: str, h: np.ndarray) -> tuple[float, float, float]:
    """`h` against the fixture's `hcore` (element) and `Tr(D h)` against
    `E1` (trace); returns (element ratio, trace error, trace ratio)."""
    fx, _, _ = _case(name)
    H, s = fx.arrays[manifest.HCORE_ARRAY], _scale_h(name)
    assert h.shape == H.shape and np.all(s > 0)
    assert np.all(np.isfinite(h)), f"{label} {name}: a non-finite element"
    err = np.abs(h - H)
    el = float(np.max(err / s))
    assert np.all(err <= ONE_E_TOL * s), (
        f"{label} {name}: |h - hcore| worst {el:.2e} of scale over "
        f"{ONE_E_TOL:g}")
    d_tr = abs(_tr(fx.density, h) - _e1(name))
    tol = _tol_trace(name)
    assert d_tr <= tol, (
        f"{label} {name}: |Tr(D h) - Tr(D hcore)| = {d_tr:.2e} over {tol:.2e}")
    return el, d_tr, d_tr / tol


# ---------------------------------------------------------------------------
# Fast tier: the fixture, the host, and what the check is for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MOLECULES)
def test_hcore_fixture_carries_the_capstone_inputs(name: str) -> None:
    """The fixture's `hcore` is its `int1e_kin + int1e_nuc` (generate.py's
    1e-14 check, re-read), its density the recipe's Hilbert matrix, and
    every one-e class it reaches is within the tier's `--one-e-l 2`."""
    fx, basis, _ = _case(name)
    H = fx.arrays[manifest.HCORE_ARRAY]
    T, V = fx.arrays[manifest.KIN_ARRAY], fx.arrays[manifest.NUC_ARRAY]
    assert float(np.max(np.abs(H - (T + V)))) <= 1e-14
    assert "hilbert" in fx.density_recipe
    n = fx.nao
    i = np.arange(n)
    assert np.array_equal(fx.density, 1.0 / (i[:, None] + i[None, :] + 1))
    assert len(_launch(name)) > 0 and basis.n_ao == n


@pytest.mark.parametrize("name", MOLECULES)
def test_host_hcore_and_its_trace_against_the_fixture(name: str) -> None:
    """The host's `T + V` against `hcore` and `Tr(D h)` against `E1`, at
    the device's tolerances: the comparison without a card."""
    _, basis, nuclei = _case(name)
    h = (np.array(kinetic_matrix(basis))
         + np.array(potential_matrix(basis, nuclei, one_centre=one_centre)))
    el, d_tr, ratio = _check("host", name, h)
    print(f"\nhost {name}: E1 = Tr(D hcore) = {_e1(name):.15f}; worst "
          f"|h - hcore| / scale {el:.2e} (tol {ONE_E_TOL:g}); "
          f"|Tr(D h) - E1| = {d_tr:.2e} = {ratio:.2e} of {_tol_trace(name):.2e}")


@pytest.mark.parametrize("name", MOLECULES)
def test_hcore_trace_sees_a_flipped_sign_a_dropped_term_or_a_shared_slip(
        name: str) -> None:
    """From the tracked libcint `T` and `V`: each slip moves `Tr(D h)` by at
    least SENSITIVITY trace tolerances -- what the check is for."""
    fx, _, _ = _case(name)
    T, V = fx.arrays[manifest.KIN_ARRAY], fx.arrays[manifest.NUC_ARRAY]
    tol = _tol_trace(name)
    moved = _slips(name, T, V)
    print(f"\n{name}: trace tol {tol:.2e}; each slip moves Tr(D h) by "
          + "; ".join(f"{k} {v:.3e} ({v / tol:.1e} tol)"
                      for k, v in moved.items()))
    assert len(moved) == (4 if _d_functions(fx).any() else 3)
    for k, v in moved.items():
        assert v >= SENSITIVITY * tol, (name, k, v, tol)


# ---------------------------------------------------------------------------
# On the card
# ---------------------------------------------------------------------------


def _on_card(backend: str, name: str, request, run_device_binary,
             tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    _, basis, _ = _case(name)
    launch = _launch(name)
    got = {}
    for target in (Target.T, Target.V):
        units = tier_units(backend, [kernel_of(c, target) for c in launch])
        work = tmp_path / target.name
        work.mkdir()
        got[target] = (build_on_card(launch, target, basis.n_ao, backend, tc,
                                     work, run_device_binary, units=units),
                       len(units))
    Ts, Vs = got[Target.T][0], got[Target.V][0]
    assert Ts.shape == Vs.shape and Ts.shape[0] >= 1
    worst = (0.0, 0.0, 0.0)
    for run in range(Ts.shape[0]):
        r = _check(f"{backend} run {run}", name, Ts[run] + Vs[run])
        worst = tuple(max(a, b) for a, b in zip(worst, r, strict=True))
    tol = _tol_trace(name)
    moved = _slips(name, Ts[0], Vs[0])
    e_dev = _tr(_case(name)[0].density, Ts[0] + Vs[0])
    print(f"\n{backend} {BACKEND[backend][1].name} {name}: {len(launch)} "
          f"classes {[f'{c.la}{c.lb}' for c in launch]} (T units the tier's "
          f"{got[Target.T][1]}, V {got[Target.V][1]}); E1 = Tr(D hcore) = "
          f"{_e1(name):.15f}, Tr(D h_dev) = {e_dev:.15f}; "
          f"worst over {Ts.shape[0]} runs: |h - hcore| / scale {worst[0]:.2e} "
          f"(tol {ONE_E_TOL:g}), |Tr(D h) - E1| {worst[1]:.2e} = "
          f"{worst[2]:.2e} of {tol:.2e}; slips on the device's T, V: "
          + "; ".join(f"{k} {v / tol:.1e} tol" for k, v in moved.items()))
    for k, v in moved.items():
        assert v >= SENSITIVITY * tol, (backend, name, k, v, tol)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_cuda_hcore_and_its_trace_on_the_card(name, request,
                                              run_device_binary,
                                              tmp_path) -> None:
    """`h = T + V` and `Tr(D h)` on the RTX 3080."""
    _on_card("cuda", name, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_hip_hcore_and_its_trace_on_the_card(name, request,
                                             run_device_binary,
                                             tmp_path) -> None:
    """`h = T + V` and `Tr(D h)` on the RX 9060 XT."""
    _on_card("hip", name, request, run_device_binary, tmp_path)
