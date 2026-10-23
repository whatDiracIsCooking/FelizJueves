"""The Schwarz matrix on the card: every class at L <= 4 against the host, and
H2O/STO-3G and H2O/6-31G against the host and libcint, on the RTX 3080 and the
RX 9060 XT.

THE BUILD is `schwarz_driver.build_on_card`: one program per basis, every
pair class's kernel (`kernel_schwarz`) launched once into one zeroed matrix,
`RUNS` times, compiled with devtools/config.sh's flags and FAILING on a
warning located in generated text (#92).

SCOPE, by decision: correctness is tested through L = 2 (la + lb) <= 4 --
(ss), (sp), (pp) and (sd). PART 1 is `test_kernel_schwarz.SYNTHETIC`'s
classes at L <= 4 (swapped pairs, diagonal and off-diagonal pairs) against
`ao.schwarz`; every element no launched pair owns must come back exactly 0.0.
PART 2 is the two s/p molecules whole, against the host under the bound below
and against libcint's `int2e_diag` under test_ao_schwarz.py's tolerance,
`|Q^2 - libcint| <= 1e-12 scale`, unchanged.

THE DEVICE-VERSUS-HOST BOUND, on the square, per element, stated before any
run (u = 2^-53, gamma_n = n u / (1 - n u), L = la + lb, K the records of the
owning shell pair, n_t the Hermite indices at L):

    |Q_dev^2 - (mn|mn)_host| <= (2 gamma_N + delta) scale,
    N = 2 (4L + 8) + (6L + 12) + 2 n_t + 3 + K^2 + 2,

`scale` being `ao.eri`'s absolute-value twin of the sum. Each side forms each
term within `gamma` of the roundings it passes through, relative to the
term's absolute value: two `E` entries (`4L + 8` each, test_j_chain_ondevice's
`N_E`), one `[p+q]^(0)` at Hermite order 2L (`3 (2L) + 12`, the
fundamental's `T`, `theta`, power and prefactor and two roundings per
reduction level), the two folds over `p` and `q` (at most `n_t` adds each),
the products by `Ebra`, `w = s_i s_k` and `s_i s_k` itself, the `K^2`-term
record fold, and the root and its square (2). FMA contraction only removes
roundings. `delta = BOYS_REL` (1e-13, test_kernel_device.py): the
device `fj_boys` is held within it of `reference.boys`, and every term is a
combination of `F_m` with its absolute value in `scale`.

THE RUN-TO-RUN WITNESS: plain stores, one owner per element -- the runs are
asserted bit-identical.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from plan_driver import compile_checked
from schwarz_driver import build_on_card, schwarz_program
from test_kernel_device import BACKEND, BOYS_REL, _toolchain
from test_kernel_schwarz import L_MAX, SYNTHETIC

from felizjueves.ao.schwarz import diagonal_matrix, scale_matrix, schwarz_classes
from felizjueves.mmd.hermite import hermite_order
from felizjueves.reference.pq import pq

U = 2.0 ** -53
LIBCINT_TOL = 1e-12  # x scale: test_ao_schwarz.py's, unchanged
MOLECULES = ("h2o_sto3g_schwarz", "h2o_631g_schwarz")


def _gamma(n: int) -> float:
    return n * U / (1.0 - n * U)


def _classes(basis):
    return [c for c in schwarz_classes(basis) if 2 * (c.la + c.lb) <= L_MAX]


@cache
def _fixture(name: str):
    return manifest.load(manifest.FIXTURES / f"{name}.json")


@cache
def _basis(name: str):
    return SYNTHETIC if name == "synthetic" else ob.from_fixture(
        _fixture(name))


@cache
def _host(name: str):
    basis = _basis(name)
    return (np.array(diagonal_matrix(basis, pq=pq)),
            np.array(scale_matrix(basis, pq=pq)))


def _bound(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The per-element tolerance on the square, and the owned mask."""
    basis = _basis(name)
    sh, off = basis.shells, basis.offsets
    _, scale = _host(name)
    tol = np.zeros_like(scale)
    owned = np.zeros(scale.shape, bool)
    for c in _classes(basis):
        L = c.la + c.lb
        n_t = len(hermite_order(L))
        for k, (a, b) in enumerate(c.pairs):
            K = c.rec1[k] - c.rec0[k]
            N = 2 * (4 * L + 8) + (6 * L + 12) + 2 * n_t + 3 + K * K + 2
            blk = (slice(off[a], off[a] + sh[a].n_cart),
                   slice(off[b], off[b] + sh[b].n_cart))
            t = (2 * _gamma(N) + BOYS_REL) * scale[blk]
            tol[blk] = t
            tol[blk[::-1]] = t.T
            owned[blk] = True
            owned[blk[::-1]] = True
    return tol, owned


def _run(backend: str, name: str, request, run_device_binary,
         tmp_path: Path) -> np.ndarray:
    tc = _toolchain(backend, request)
    basis = _basis(name)
    classes = _classes(basis)
    Qs = build_on_card(classes, basis.n_ao, backend, tc, tmp_path,
                       run_device_binary)
    want, _ = _host(name)
    tol, owned = _bound(name)
    err = np.abs(Qs[0] ** 2 - want)
    worst = float(np.max(err[owned] / tol[owned]))
    same = bool(np.array_equal(Qs[0], Qs[1]))
    print(f"\n{backend} {BACKEND[backend][1].name} {name}: classes "
          f"{[f'{c.la}{c.lb}' for c in classes]}; max |Q^2 - host| = "
          f"{err[owned].max():.2e}, worst / bound {worst:.2e}; runs "
          f"bit-identical {same}")
    assert same
    assert np.all(err[owned] <= tol[owned]), worst
    assert np.all(Qs[0][~owned] == 0.0)
    assert np.array_equal(Qs[0], Qs[0].T)
    return Qs[0]


# ---------------------------------------------------------------------------
# Part 1: every class at L <= 4, against the host
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_every_class_through_l4_matches_the_host(request,
                                                      run_device_binary,
                                                      tmp_path):
    _run("cuda", "synthetic", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_every_class_through_l4_matches_the_host(request,
                                                     run_device_binary,
                                                     tmp_path):
    _run("hip", "synthetic", request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: the molecules, against the host and libcint
# ---------------------------------------------------------------------------


def _part2(backend: str, name: str, request, run_device_binary,
           tmp_path: Path) -> None:
    assert len(_classes(_basis(name))) == len(schwarz_classes(_basis(name)))
    Q = _run(backend, name, request, run_device_binary, tmp_path)
    ref = _fixture(name).arrays[manifest.SCHWARZ_ARRAY]
    _, scale = _host(name)
    err = np.abs(Q ** 2 - ref)
    lc = float(np.max(err / scale))
    print(f"{backend} {name}: max |Q^2 - int2e_diag| / scale = {lc:.2e} "
          f"(tol {LIBCINT_TOL:g}); max |Q - sqrt(libcint)| / sqrt(libcint) "
          f"= {float(np.max(np.abs(Q - np.sqrt(ref)) / np.sqrt(ref))):.2e}")
    assert np.all(err <= LIBCINT_TOL * scale), lc


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_cuda_schwarz_matrix_matches_libcint(name, request, run_device_binary,
                                             tmp_path):
    _part2("cuda", name, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_hip_schwarz_matrix_matches_libcint(name, request, run_device_binary,
                                            tmp_path):
    _part2("hip", name, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`) and the fast-tier pieces
# ---------------------------------------------------------------------------

@pytest.mark.cuda
@pytest.mark.parametrize("name", ("synthetic", *MOLECULES))
def test_cuda_schwarz_programs_compile(name, nvcc, tmp_path):
    basis = _basis(name)
    compile_checked(nvcc, schwarz_program(_classes(basis), basis.n_ao, "cuda",
                                          tmp_path), tmp_path / "bin")


@pytest.mark.hip
@pytest.mark.parametrize("name", ("synthetic", *MOLECULES))
def test_hip_schwarz_programs_compile(name, hipcc, tmp_path):
    basis = _basis(name)
    compile_checked(hipcc, schwarz_program(_classes(basis), basis.n_ao, "hip",
                                           tmp_path), tmp_path / "bin")


def test_part1_reaches_every_class_at_l4_and_no_more():
    got = {(c.la, c.lb) for c in _classes(SYNTHETIC)}
    assert got == {(0, 0), (0, 1), (1, 1), (0, 2)}
    assert all(2 * (c.la + c.lb) <= L_MAX for c in _classes(SYNTHETIC))


def test_the_bound_is_not_vacuous():
    for name in ("synthetic", *MOLECULES):
        tol, owned = _bound(name)
        want, _ = _host(name)
        assert np.max(tol[owned] / np.abs(want[owned])) < 1e-10, name
        assert math.isfinite(float(tol.max()))
