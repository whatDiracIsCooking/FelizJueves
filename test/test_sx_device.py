"""The cross-basis overlap `S_{B1,B2}` on the card: every ordered class
(ss) through (ff) against the host, and the three tracked cross cases whole
(both orientations) against the host and libcint, on the RTX 3080 and the
RX 9060 XT.

THE BUILD is `sx_driver.build_on_card`: one program per launch
(`launch.cross_overlap_launch`), every ordered class's kernel
(`kernel_sx`) launched once into one zeroed `n_a x n_b` matrix, `RUNS`
times, compiled with devtools/config.sh's flags and FAILING on a warning
located in generated text (#92).

PART 1, PER CLASS, THROUGH (ff). HF/cc-pVTZ against itself as two bases:
all sixteen ordered classes `(0..3, 0..3)`, every record, same-centre and
equal-offset pairs included (the kernel has no diagonal rule, so they are
ordinary pairs), against `ao.overlap.cross_overlap_matrix`. Every element
is owned by one ordered shell pair, so the whole matrix is checked at once.

PART 2, PER CASE. `h2o_sto3g_x_631g`, `_x_ccpvdz` and `_x_631g_moved`,
rows-by-columns and swapped, against the host under the bound below and
against libcint's `intor_cross` (the fixture, transposed when swapped) at
test_ao_cross_overlap.py's tolerance, unchanged.

THE DEVICE-VERSUS-HOST BOUND, per element, stated before any run: `S`'s,
from test_one_electron_device.py, with `u = 2^-53`, `gamma_n = n u /
(1 - n u)`, `L = la + lb`, `K` the records of the owning pair and
`N = N_E(L) + 1` (`E`'s `4L + 8` roundings, times `s`):

    |S_dev - S_host| <= 2 gamma_{N + K} Mabs,

`Mabs` the sum over the owning pair's records of `|s| Eabs_x Eabs_y Eabs_z`
(`_axis_abs`: `E`'s recurrence from the same seed over `|X_PA|`, `|X_PB|`).
No Boys, so no `delta`. THE RUN-TO-RUN BOUND: only the order of the `K`
atomic adds differs, so `|S(run 1) - S(run 2)| <= 2 gamma_K Mabs`, and a
pair with one record repeats bit for bit.

NOT A MEASUREMENT: nothing is written to `data/measured/`.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build every program both
parts use. The device tests are `gpu` plus `cuda` or `hip`, behind the card
lock and the VRAM probe: without the card they skip, and kernel-tier.sh
exits 2.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from plan_driver import compile_checked
from sx_driver import build_on_card, sx_program
from test_ao_cross_overlap import _scale as libcint_scale
from test_ao_overlap import ONE_E_TOL
from test_kernel_device import BACKEND, _toolchain
from test_one_electron_device import _axis_abs, _gamma, _n_e

from felizjueves.ao.overlap import cross_overlap_matrix
from felizjueves.ao.pairs import overlap_bound
from felizjueves.ao.shell import cartesians
from felizjueves.launch import XLaunch, cross_overlap_launch

CROSS = ("h2o_sto3g_x_631g", "h2o_sto3g_x_ccpvdz", "h2o_sto3g_x_631g_moved")
CASES = ["hf_self", *(f"{n}:{o}" for n in CROSS for o in ("AB", "BA"))]


@cache
def _case(case: str):
    """`(A, B, libcint S or None)` of a case id."""
    if case == "hf_self":
        b = ob.from_fixture(manifest.load(manifest.FIXTURES / "hf_ccpvtz.json"))
        return b, b, None
    name, orient = case.split(":")
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    rows, S = manifest.cross_overlap(fx)
    A, B = ob.from_fixture(rows), ob.from_fixture(fx)
    return (B, A, S.T) if orient == "BA" else (A, B, S)


@cache
def _launch(case: str) -> XLaunch:
    A, B, _ = _case(case)
    plan = cross_overlap_launch(A, B, tau=0.0, bound=overlap_bound)
    assert not plan.dropped
    return plan


@cache
def _host(case: str) -> np.ndarray:
    A, B, _ = _case(case)
    return np.array(cross_overlap_matrix(A, B))


@cache
def _bound(case: str) -> tuple[np.ndarray, np.ndarray]:
    """`(device-vs-host, run-to-run)` tolerance per element."""
    plan = _launch(case)
    Mabs = np.zeros((plan.n_a, plan.n_b))
    K = np.zeros_like(Mabs)
    N = np.zeros_like(Mabs)
    for cls in plan:
        ca, cb = cartesians(cls.la), cartesians(cls.lb)
        for r in cls.recs:
            for i, a in enumerate(ca):
                for j, b in enumerate(cb):
                    e = 1.0
                    for d in range(3):
                        e *= _axis_abs(a[d], b[d], r.zeta, r.XPA[d],
                                       r.XPB[d])[0]
                    Mabs[r.off_a + i, r.off_b + j] += abs(r.s) * e
                    K[r.off_a + i, r.off_b + j] += 1
                    N[r.off_a + i, r.off_b + j] = _n_e(cls.la + cls.lb) + 1
    assert np.all(K > 0)  # every element owned
    host = np.vectorize(lambda n, k: 2 * _gamma(int(n + k)))(N, K) * Mabs
    runs = np.vectorize(lambda k: 2 * _gamma(int(k)) if k > 1 else 0.0)(K)
    return host, runs * Mabs


def _run(backend: str, case: str, request, run_device_binary,
         tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    plan = _launch(case)
    Ss = build_on_card(plan, backend, tc, tmp_path, run_device_binary)
    want = _host(case)
    tol, rtol = _bound(case)
    err = np.abs(Ss[0] - want)
    worst = float(np.max(err / np.where(tol > 0, tol, np.inf)))
    print(f"\n{backend} {BACKEND[backend][1].name} {case}: classes "
          f"{[f'{c.la}{c.lb}' for c in plan]}; max |S - host| = "
          f"{err.max():.2e}, worst / bound {worst:.2e}")
    assert np.all(err <= tol), worst
    assert np.all(np.abs(Ss[0] - Ss[1]) <= rtol)
    A, B, ref = _case(case)
    if ref is not None:
        scale = libcint_scale(A, B)
        lc = np.abs(Ss[0] - ref)
        print(f"{backend} {case}: max |S - intor_cross| / scale = "
              f"{float(np.max(lc / scale)):.2e} (tol {ONE_E_TOL:g})")
        assert np.all(lc <= ONE_E_TOL * scale)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES)
def test_cuda_cross_overlap_matches_the_host_and_libcint(
        case, request, run_device_binary, tmp_path):
    _run("cuda", case, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES)
def test_hip_cross_overlap_matches_the_host_and_libcint(
        case, request, run_device_binary, tmp_path):
    _run("hip", case, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions and the fast-tier pieces
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("case", CASES)
def test_cuda_sx_programs_compile(case, nvcc, tmp_path):
    compile_checked(nvcc, sx_program(_launch(case), "cuda", tmp_path),
                    tmp_path / "bin")


@pytest.mark.hip
@pytest.mark.parametrize("case", CASES)
def test_hip_sx_programs_compile(case, hipcc, tmp_path):
    compile_checked(hipcc, sx_program(_launch(case), "hip", tmp_path),
                    tmp_path / "bin")


@pytest.mark.slow  # plans (ff), L > FAST_L_MAX (tiers.py)
def test_part1_reaches_every_ordered_class_through_f():
    assert [(c.la, c.lb) for c in _launch("hf_self")] == [
        (la, lb) for la in range(4) for lb in range(4)]


def _bound_is_not_vacuous(cases) -> None:
    for case in cases:
        tol, _ = _bound(case)
        S = np.abs(_host(case))
        big = S > 1e-6
        assert np.max(tol[big] / S[big]) < 1e-10, case  # Schwarz's bar


def test_the_bound_is_not_vacuous():
    _bound_is_not_vacuous([c for c in CASES if c != "hf_self"])


@pytest.mark.slow  # hf_self reaches (ff), L > FAST_L_MAX (tiers.py)
def test_the_bound_is_not_vacuous_above_l4():
    _bound_is_not_vacuous(["hf_self"])
