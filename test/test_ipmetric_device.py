"""The uncontracted DF metric derivative (`ipmetric_`, #693) ON A CARD:
every class of an aux set launched into one `M1[3, n_aux, n_aux]`, against
`gradient.host_df.metric_derivative` and libcint's `-int2c2e_ip1` (the
arrays are `df_grad_launch.ipmetric_arrays`, the interpreted tests'; the
runner is `harness_driver`'s chain, `M1` carried from a sentinel).

PART 0 (fast tier's L): cc-pVDZ-JKFIT's s and p fitting shells on H2O,
classes (s|s), (s|p), (p|p), held to the host and to the numerical golden
store's `-int2c2e_ip1_cart` (#655) per axis.

PART 1 (`slow`): all of cc-pVDZ-JKFIT on H2O, O to f, every class through
(f|f) in one program, held to the host and to the golden per axis.

PART 2 (`slow` + `high_l`), THE LADDER THROUGH (i|i): the aux-i witness's
aux set (`manifest.df_gradient`, l = 0..6 on both atoms), one program per
aux `l` -- the classes `lP <= lQ = l` -- each compile within the user's
120 s per-.cu budget (`COMPILE_BUDGET_S`, a hard timeout). Held to the
host per class and to libcint through #613's contracted witness
(`test_kernel_ipmetric._contract`, exact linear algebra on `M1`).

THE BOUND, stated before any run, is the interpreted test's
(test_kernel_ipmetric.py): `|M1_dev - host| <= 16 eps` of the class's
scale over every element it owns -- `max|host|` at `lP + lQ <= 4`, the
host on `test_kernel_metric.pq_tower` (the absolute-value tower) above
it, and the program's largest class scale for a class with no derivative
(all its pairs one-centre: cc-pVDZ-JKFIT's (f|f), O alone having f) -- and
libcint within `1e-12` (the golden store's fp64 tier, and the
aux-i witness's largest class). The interpreted kernel is within 3.6 eps
of that gate; the card's `exp` and Boys differ by ulps. The stores are
plain and each element is one thread's, so the `RUNS` runs must be
BIT-IDENTICAL, every element no launched pair owns must keep the
sentinel, and an off-diagonal pair's mirror must be its exact negation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from df_grad_launch import ipmetric_arrays
from harness_driver import Step, build_steps, chain_specs, run_chain
from oracle import numeric as golden
from plan_driver import RUNS
from test_kernel_device import _toolchain
from test_kernel_ipmetric import GOLDEN, _big, _contract, _owned, _pairs, _sp
from test_kernel_ipmetric import _witness as _auxi
from test_kernel_metric import pq_tower

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.gradient import host_df
from felizjueves.gradient.kernel_ipmetric import (
    OUT,
    build_ipmetric_kernel,
    launch_shape,
)
from felizjueves.reference.pq import pq

EPS = 2.0 ** -52
DEV_TOL = 16 * EPS  # of the class's scale (module docstring)
LIBCINT_TOL = 1e-12  # of the aux-i witness's largest class max|G|
COMPILE_BUDGET_S = 120.0  # the user's per-.cu compile budget
THREADS = 32  # lanes per block, one aux shell pair per thread
SENTINEL = 7.25
LOW_L = 4  # lP + lQ <= LOW_L: max|host| is the scale; above, the tower


def _steps(aux: Basis, lQ: int | None = None
           ) -> tuple[list[Step], list[tuple[int, int]]]:
    """One step per metric class of `aux` (`lQ` that ket `l` only)."""
    steps, ran = [], []
    for c in metric_classes(aux):
        if lQ is not None and c.lQ != lQ:
            continue
        grid, block = launch_shape(c.n_pair, THREADS)
        steps.append(Step(build_ipmetric_kernel((c.lP, c.lQ)), grid, block,
                          {**ipmetric_arrays(c), "n_pair": c.n_pair,
                           "n_rec": c.n_rec, "n_aux": aux.n_ao}))
        ran.append((c.lP, c.lQ))
    return steps, ran


def _on_card(backend: str, request, tmp_path: Path, aux: Basis,
             lQ: int | None = None, timeout: float = 1800.0
             ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """`M1` after every launch, from the sentinel, once per run; the runs
    bit-identical, every element no pair owns the sentinel still."""
    tc = _toolchain(backend, request)
    steps, ran = _steps(aux, lQ)
    bins = build_steps(tc, backend, steps, tmp_path, timeout=timeout)
    got = run_chain(bins, steps, OUT, np.full((3, aux.n_ao, aux.n_ao),
                                              SENTINEL), tmp_path / "runs",
                    runs=RUNS)
    own = np.zeros((3, aux.n_ao, aux.n_ao), bool)
    for cls in ran:
        own |= _owned(aux, _pairs(aux, cls))
    print(f"\n{backend} {[s.kernel.name for s in steps]}: bit-identical "
          f"{all(np.array_equal(got[0], x) for x in got)}")
    for x in got:
        assert np.array_equal(x, got[0])
        assert np.all(x[~own] == SENTINEL)
    return got[0], ran


def _held_to_the_host(M1: np.ndarray, aux: Basis, ran) -> None:
    """Every class's owned elements within `DEV_TOL` of its scale, and the
    off-diagonal mirrors exact negations."""
    sh, off = aux.shells, aux.offsets
    want, scale = {}, {}
    for cls in ran:
        pairs = _pairs(aux, cls)
        want[cls] = np.array(host_df.metric_derivative(aux, pq=pq,
                                                       pairs=pairs))
        scale[cls] = float(np.abs(np.array(host_df.metric_derivative(
            aux, pq=pq if sum(cls) <= LOW_L else pq_tower,
            pairs=pairs))).max())
    # A class whose pairs are all one-centre (cc-pVDZ-JKFIT's (f|f): only O
    # has f) has no derivative; it is judged against the program's largest
    # class scale, as the golden store judges a sub-block (`compare`).
    big = max(scale.values())
    worst = {}
    for cls in ran:
        pairs = _pairs(aux, cls)
        own = _owned(aux, pairs)
        worst[cls] = float(np.abs(M1 - want[cls])[own].max()) / (
            scale[cls] or big) / EPS
        for i, j in pairs:
            if i != j:
                a = slice(off[i], off[i] + sh[i].n_cart)
                b = slice(off[j], off[j] + sh[j].n_cart)
                assert np.array_equal(M1[:, a, b],
                                      -M1[:, b, a].transpose(0, 2, 1))
    print("|M1 - host|, eps of the class's scale: "
          + ", ".join(f"{c}: {e:.2f}" for c, e in worst.items()))
    assert max(worst.values()) <= DEV_TOL / EPS, worst


def _held_to_the_golden(M1: np.ndarray, rows=None) -> None:
    for k, name in enumerate(GOLDEN):
        err = golden.compare(name, M1[k], index=None if rows is None
                             else (rows, rows))
        print(f"{name}: {err:.2e} of max")


# --- Part 0: the s and p fitting shells -----------------------------------------


def _part0(backend: str, request, tmp_path: Path) -> None:
    sub, rows = _sp(GOLDEN[0])
    M1, ran = _on_card(backend, request, tmp_path, sub)
    assert ran == [(0, 0), (0, 1), (1, 1)]
    _held_to_the_host(M1, sub, ran)
    _held_to_the_golden(M1, rows)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_ipmetric_sp_matches_the_host_and_libcint(request,
                                                       tmp_path) -> None:
    _part0("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_ipmetric_sp_matches_the_host_and_libcint(request,
                                                      tmp_path) -> None:
    _part0("hip", request, tmp_path)


# --- Part 1: all of cc-pVDZ-JKFIT, through (f|f) ---------------------------------


def _part1(backend: str, request, tmp_path: Path) -> None:
    aux = golden.basis(GOLDEN[0])
    M1, ran = _on_card(backend, request, tmp_path, aux)
    assert max(map(sum, ran)) == 6 and len(ran) == 10
    _held_to_the_host(M1, aux, ran)
    _held_to_the_golden(M1)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
def test_cuda_ipmetric_matches_the_host_and_libcint(request,
                                                    tmp_path) -> None:
    _part1("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
def test_hip_ipmetric_matches_the_host_and_libcint(request,
                                                   tmp_path) -> None:
    _part1("hip", request, tmp_path)


# --- Part 2: the ladder through (i|i) --------------------------------------------


def _part2(backend: str, lQ: int, request, tmp_path: Path) -> None:
    wit, aux, nuclei = _auxi()
    M1, ran = _on_card(backend, request, tmp_path, aux, lQ,
                       timeout=COMPILE_BUDGET_S)
    assert sorted(ran) == [(p, lQ) for p in range(lQ + 1)]
    _held_to_the_host(M1, aux, ran)
    big = _big(wit.gmetric)
    for cls in ran:
        g = _contract(M1, aux, nuclei, wit.w(), cls)
        err = float(np.abs(g - wit.gmetric[cls]).max()) / big
        print(f"ipmetric_{cls[0]}{cls[1]} vs libcint: {err:.2e} of max")
        assert err <= LIBCINT_TOL, (cls, err)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.parametrize("lQ", range(7), ids=lambda x: f"ipmetric-{x}")
def test_cuda_ipmetric_ladder_through_i(lQ, request, tmp_path) -> None:
    _part2("cuda", lQ, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.parametrize("lQ", range(7), ids=lambda x: f"ipmetric-{x}")
def test_hip_ipmetric_ladder_through_i(lQ, request, tmp_path) -> None:
    _part2("hip", lQ, request, tmp_path)


@pytest.mark.slow  # builds every class through (i|i)
def test_the_programs_cover_every_class_once() -> None:
    """Host-only: Part 1's program is every class of cc-pVDZ-JKFIT, and the
    ladder's parts are the whole ladder, each class once; every step's
    arguments convert against its kernel's parameter list."""
    aux = golden.basis(GOLDEN[0])
    steps, ran = _steps(aux)
    assert sorted(ran) == [(p, q) for p in range(4) for q in range(p, 4)]
    specs = chain_specs(steps, OUT, np.zeros((3, aux.n_ao, aux.n_ao)))
    assert [s["kernel"] for s in specs] == [s.kernel.name for s in steps]
    aux = _auxi()[1]
    seen = [c for lQ in range(7) for c in _steps(aux, lQ)[1]]
    assert sorted(seen) == [(p, q) for p in range(7) for q in range(p, 7)]
