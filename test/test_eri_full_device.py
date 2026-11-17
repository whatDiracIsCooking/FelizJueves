"""The packed full ERI, `J` and `K` by GEMV on the card: every class at
L <= 4 against the host, and H2 and H2O/STO-3G's `J` and `K` against PySCF.

THE BUILD is `eri_full_driver.build_on_card`, on the shared runner (#649):
one runner per class's unit (`kernel_eri`), compiled with devtools/config.sh's
flags and FAILING on a warning located in generated text or a harness header
(#92); every class launched once into one packed matrix carried from class to
class from zero, each chain `RUNS` times. Then `coulomb_on_card`: the GEMV
(`kernel_gemv`) over that matrix and the packed density.

SCOPE, by decision: L = la + lb + lc + ld <= 4 (FAST_L_MAX). PART 1 is
`test_kernel_eri_full.SYNTHETIC`'s classes at L <= 4 (swapped pairs, diagonal
and off-diagonal pairs, a d shell) against `full_eri.packed`; every element no
launched class owns must come back exactly 0.0. PART 2 is H2 and H2O in
STO-3G whole: the ERI against the host and `J` against PySCF's `vj`.

THE DEVICE-VERSUS-HOST BOUND, per element, stated before any run
(u = 2^-53, gamma_n = n u / (1 - n u), L the class's total, L_p and L_q the
two pairs', n_p and n_q their Hermite indices, K_b and K_k the records of the
owning bra and ket shell pairs):

    |ERI_dev - ERI_host| <= (2 gamma_N + delta) scale,
    N = (4 L_p + 8) + (4 L_q + 8) + (3 L + 12) + n_p + n_q + 3 + K_b K_k,

`scale` being `eri.eri`'s absolute-value twin. Each side forms each term
within `gamma` of the roundings it passes through, relative to the term's
absolute value: the two `E` entries (test_schwarz_device's `4L + 8` each),
one `[p+q]^(0)` at Hermite order L (`3 L + 12`), the two folds, the products
by `E`, `w = s_i s_k` and `s_i s_k` itself, and the record fold. FMA
contraction only removes roundings. `delta = BOYS_REL` (1e-13).

`J` is then held to PySCF at test_kernel_eri_full's `J_TOL` (1e-12 of
max |vj|), the same bar as the interpreted chain.

PART 3, `K`: the card's packed ERI, then `kernel_kperm` on the card into
`X_pp` -- held `==` to the host's `exchange_packed` of that same card ERI
(two loads, one add, one exact halving) and bitwise symmetric -- then the GEMV
over `X_pp`, `K` held to PySCF's `vk` at `J_TOL`.

THE RUN-TO-RUN WITNESS: plain stores, one owner per element, and a GEMV of one
thread per row -- the runner fails unless every run is bit-identical.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from eri_full_driver import (
    build_on_card,
    coulomb_on_card,
    eri_steps,
    exchange_on_card,
    gemv_step,
    kperm_step,
)
from harness_driver import build_steps
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import BACKEND, BOYS_REL, _toolchain
from test_kernel_eri_full import (
    F_BASIS,
    F_CLASSES,
    J_TOL,
    L_MAX,
    SYNTHETIC,
    check_bound,
)
from tiers import split_by_l

from felizjueves.eri.full_eri.packed import (
    eri_classes,
    exchange_packed,
    n_packed,
    owned,
    packed_eri_and_scale,
    pair_index,
)
from felizjueves.mmd.hermite import hermite_order
from felizjueves.reference.pq import pq

U = 2.0 ** -53
MOLECULES = ("h2_sto3g_j", "h2o_sto3g_j")
K_MOLECULES = ("h2_sto3g_k", "h2o_sto3g_k")


def _gamma(n: int) -> float:
    return n * U / (1.0 - n * U)


def _classes(basis):
    return [c for c in eri_classes(basis) if sum(c.am) <= L_MAX]


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
    keep = {c.am for c in _classes(basis)}
    want, scale = packed_eri_and_scale(basis, pq=pq, keep=keep)
    return np.array(want), np.array(scale), np.array(owned(basis, keep))


def _bound(name: str) -> np.ndarray:
    """The per-element tolerance (module docstring), placed as the kernel
    places its elements."""
    basis = _basis(name)
    sh, off = basis.shells, basis.offsets
    _, scale, _ = _host(name)
    n = basis.n_ao
    pidx = [[max(m, k) * (max(m, k) + 1) // 2 + min(m, k) for k in range(n)]
            for m in range(n)]
    tol = np.zeros_like(scale)
    for c in _classes(basis):
        la, lb, lc, ld = c.am
        lp, lq = la + lb, lc + ld
        L = lp + lq
        for tb, tk in zip(c.task_bra, c.task_ket, strict=True):
            kb = c.bra.rec1[tb] - c.bra.rec0[tb]
            kk = c.ket.rec1[tk] - c.ket.rec0[tk]
            N = ((4 * lp + 8) + (4 * lq + 8) + (3 * L + 12)
                 + len(hermite_order(lp)) + len(hermite_order(lq)) + 3
                 + kb * kk)
            t = 2 * _gamma(N) + BOYS_REL
            a_, b_ = c.bra.pairs[tb]
            c_, d_ = c.ket.pairs[tk]
            rows = {pidx[off[a_] + a][off[b_] + b]
                    for a in range(sh[a_].n_cart) for b in range(sh[b_].n_cart)}
            cols = {pidx[off[c_] + x][off[d_] + y]
                    for x in range(sh[c_].n_cart) for y in range(sh[d_].n_cart)}
            for r in rows:
                for q in cols:
                    tol[r, q] = tol[q, r] = t * scale[r, q]
    return tol


def _run(backend: str, name: str, request, tmp_path: Path) -> np.ndarray:
    tc = _toolchain(backend, request)
    basis = _basis(name)
    classes = _classes(basis)
    eri = build_on_card(classes, basis.n_ao, backend, tc, tmp_path)
    want, _, mine = _host(name)
    tol = _bound(name)
    err = np.abs(eri - want)
    worst = float(np.max(err[mine] / np.maximum(tol[mine], 1e-300)))
    print(f"\n{backend} {BACKEND[backend][1].name} {name}: classes "
          f"{[''.join(map(str, c.am)) for c in classes]}; max |ERI - host| = "
          f"{err[mine].max():.2e}, worst / bound {worst:.2e}; runs "
          "bit-identical (the runner's witness)")
    assert np.all(err[mine] <= tol[mine]), worst
    assert np.all(eri[~mine] == 0.0)
    assert np.array_equal(eri, eri.T)
    return eri


# ---------------------------------------------------------------------------
# Part 1: every class at L <= 4, against the host
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_every_eri_class_through_l4_matches_the_host(request, tmp_path):
    _run("cuda", "synthetic", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_every_eri_class_through_l4_matches_the_host(request, tmp_path):
    _run("hip", "synthetic", request, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: the molecules, the ERI against the host and J against PySCF
# ---------------------------------------------------------------------------


def _part2(backend: str, name: str, request, tmp_path: Path) -> None:
    basis = _basis(name)
    assert len(_classes(basis)) == len(eri_classes(basis))
    eri = _run(backend, name, request, tmp_path)
    fx = _fixture(name)
    J = coulomb_on_card(eri, fx.density, backend,
                        _toolchain(backend, request), tmp_path)
    ref = fx.arrays[manifest.J_ARRAY]
    rel = float(np.max(np.abs(J - ref)) / np.max(np.abs(ref)))
    print(f"{backend} {name}: max |J_gemv - vj| / max |vj| = {rel:.2e} "
          f"(tol {J_TOL:g})")
    assert rel < J_TOL
    assert np.array_equal(J, J.T)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_cuda_j_by_gemv_matches_pyscf(name, request, tmp_path):
    _part2("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_hip_j_by_gemv_matches_pyscf(name, request, tmp_path):
    _part2("hip", name, request, tmp_path)


# ---------------------------------------------------------------------------
# Part 3: K by GEMV over the exchange-ordered matrix, against PySCF
# ---------------------------------------------------------------------------


def _part3(backend: str, name: str, request, tmp_path: Path) -> None:
    basis = _basis(name)
    assert len(_classes(basis)) == len(eri_classes(basis))
    eri = _run(backend, name, request, tmp_path)
    fx = _fixture(name)
    X, K = exchange_on_card(eri, fx.density, backend,
                            _toolchain(backend, request), tmp_path)
    assert np.array_equal(X, np.array(exchange_packed(eri.tolist())))
    assert np.array_equal(X, X.T)
    ref = fx.arrays[manifest.K_ARRAY]
    rel = float(np.max(np.abs(K - ref)) / np.max(np.abs(ref)))
    print(f"{backend} {name}: X_pp bitwise the host's; max |K_gemv - vk| / "
          f"max |vk| = {rel:.2e} (tol {J_TOL:g})")
    assert rel < J_TOL
    assert np.array_equal(K, K.T)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", K_MOLECULES)
def test_cuda_k_by_gemv_matches_pyscf(name, request, tmp_path):
    _part3("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", K_MOLECULES)
def test_hip_k_by_gemv_matches_pyscf(name, request, tmp_path):
    _part3("hip", name, request, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`) and the fast-tier pieces
# ---------------------------------------------------------------------------


def _programs(name: str):
    basis = _basis(name)
    n_pp = n_packed(basis.n_ao)
    return [*eri_steps(_classes(basis), basis.n_ao),
            gemv_step(n_pp, np.zeros((n_pp, n_pp)), np.zeros(n_pp)),
            kperm_step(np.zeros((n_pp, n_pp)))]


@pytest.mark.cuda
@pytest.mark.parametrize("name", ("synthetic", *MOLECULES))
def test_cuda_eri_programs_compile(name, nvcc, tmp_path):
    build_steps(nvcc, "cuda", _programs(name), tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("name", ("synthetic", *MOLECULES))
def test_hip_eri_programs_compile(name, hipcc, tmp_path):
    build_steps(hipcc, "hip", _programs(name), tmp_path)


def test_part1_reaches_every_class_at_l4_and_no_more():
    got = {c.am for c in _classes(SYNTHETIC)}
    assert (1, 1, 1, 1) in got and (0, 0, 0, 2) in got and (0, 2, 0, 2) in got
    assert all(sum(am) <= L_MAX for am in got)


def test_the_bound_is_not_vacuous():
    for name in ("synthetic", *MOLECULES):
        tol = _bound(name)
        want, _, mine = _host(name)
        nz = mine & (np.abs(want) > 1e-8)
        assert np.max(tol[nz] / np.abs(want[nz])) < 1e-9, name
        assert math.isfinite(float(tol.max()))
        assert np.all(tol[mine] >= 0.0)


# ---------------------------------------------------------------------------
# Part 4: the low-stack form -- every class through (ff|ff), the forced form
# at L <= 4, HF/cc-pVTZ against libcint, CO/cc-pVTZ's J and K against PySCF
# ---------------------------------------------------------------------------
#
# THE BOUNDS, stated before any run. Against the host: test_kernel_eri_full's
# `packed_bound`, Part 1's analysis per element (the low-stack form passes
# every term through the same roundings, the folds nested over the row's
# reach and the axes). Against libcint: that bound plus test_ao_eri's
# stated host-versus-libcint TOL (1e-12), both times `eri.eri`'s scale --
# a triangle inequality, nothing measured. `J` and `K` against PySCF:
# `J_TOL`, as Parts 2 and 3.

HF_CCPVTZ = ("hf_ccpvtz_eri", "hf_ccpvtz_eri_l9", "hf_ccpvtz_eri_l10",
             "hf_ccpvtz_eri_l11", "hf_ccpvtz_eri_l12")
LIBCINT_TOL = 1e-12  # test_ao_eri.TOL, eri.eri against libcint, x scale
CO = ("co_ccpvtz_j", "co_ccpvtz_k")


def _low(backend: str, basis, classes, label: str, request,
         tmp_path: Path) -> np.ndarray:
    tc = _toolchain(backend, request)
    eri = build_on_card(classes, basis.n_ao, backend, tc, tmp_path,
                        lowstack=True)
    assert np.array_equal(eri, eri.T)
    return eri


def _f_class(backend: str, cls, request, tmp_path: Path) -> None:
    classes = [c for c in eri_classes(F_BASIS) if c.am == cls]
    eri = _low(backend, F_BASIS, classes, "f", request, tmp_path)
    worst = check_bound(eri, "f", frozenset({cls}), zero_elsewhere=True)
    print(f"\n{backend} {BACKEND[backend][1].name} ({''.join(map(str, cls))})"
          f" low-stack, s-to-f basis: worst / bound {worst:.2e}; runs "
          "bit-identical")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", split_by_l(F_CLASSES, sum,
                                           lambda c: "".join(map(str, c))))
def test_cuda_every_class_through_ff_ff_low_stack_matches_the_host(
        cls, request, tmp_path):
    _f_class("cuda", cls, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", split_by_l(F_CLASSES, sum,
                                           lambda c: "".join(map(str, c))))
def test_hip_every_class_through_ff_ff_low_stack_matches_the_host(
        cls, request, tmp_path):
    _f_class("hip", cls, request, tmp_path)


def _forced(backend: str, request, tmp_path: Path) -> None:
    classes = _classes(SYNTHETIC)
    eri = _low(backend, SYNTHETIC, classes, "synthetic", request, tmp_path)
    worst = check_bound(eri, "synthetic",
                        frozenset(c.am for c in classes), zero_elsewhere=True)
    print(f"\n{backend} forced low-stack, every synthetic class at L <= 4: "
          f"worst / bound {worst:.2e}")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_the_forced_low_stack_form_through_l4_matches_the_host(
        request, tmp_path):
    _forced("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_the_forced_low_stack_form_through_l4_matches_the_host(
        request, tmp_path):
    _forced("hip", request, tmp_path)


def _n_of(shells) -> int:
    """`N` for a quartet of shells, its records bounded by the primitive
    products (a record is a primitive pair)."""
    la, lb, lc, ld = (s.l for s in shells)
    lp, lq = la + lb, lc + ld
    k = math.prod(len(s.exponents) for s in shells)
    return ((4 * lp + 8) + (4 * lq + 8) + (3 * (lp + lq) + 12)
            + len(hermite_order(lp)) + len(hermite_order(lq)) + 3 + k)


def _libcint(backend: str, request, tmp_path: Path) -> None:
    from felizjueves.eri.eri import eri_block_and_scale

    fxs = {n: _fixture(n) for n in HF_CCPVTZ}
    basis = ob.from_fixture(fxs["hf_ccpvtz_eri"])
    sh, off = basis.shells, basis.offsets
    classes = eri_classes(basis)
    assert len(classes) == 55
    eri = _low(backend, basis, classes, "hf", request, tmp_path)
    pidx = np.array(pair_index(basis.n_ao))
    n_blocks = 0
    for name, fx in fxs.items():
        for q, ref in manifest.eri_blocks(fx).items():
            i, j, k, l_ = q
            r = [slice(off[s], off[s] + sh[s].n_cart) for s in q]
            got = eri[pidx[r[0], r[1]][:, :, None, None],
                      pidx[r[2], r[3]][None, None, :, :]]
            _, scale = eri_block_and_scale((sh[i], sh[j]), (sh[k], sh[l_]),
                                           pq=pq)
            scale = np.array(scale).reshape(ref.shape)
            n_max = _n_of([sh[s] for s in q])
            tol = (2 * _gamma(n_max) + BOYS_REL + LIBCINT_TOL) * scale
            err = np.abs(got - ref)
            worst = float(np.max(err / np.maximum(tol, 1e-300)))
            L = sum(sh[s].l for s in q)
            print(f"{backend} HF/cc-pVTZ {q} L={L}: max |card - libcint| "
                  f"{err.max():.2e}, worst / bound {worst:.2e}")
            assert np.all(err <= tol), (name, q, worst)
            n_blocks += 1
    assert n_blocks == 55


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
def test_cuda_hf_ccpvtz_low_stack_matches_libcint(request, tmp_path):
    _libcint("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
def test_hip_hf_ccpvtz_low_stack_matches_libcint(request, tmp_path):
    _libcint("hip", request, tmp_path)


def _co(backend: str, request, tmp_path: Path) -> None:
    fj, fk = (_fixture(n) for n in CO)
    assert np.array_equal(fj.density, fk.density)
    basis = ob.from_fixture(fj)
    classes = eri_classes(basis)
    assert max(max(c.am) for c in classes) == 3
    tc = _toolchain(backend, request)
    eri = _low(backend, basis, classes, "co", request, tmp_path)
    J = coulomb_on_card(eri, fj.density, backend, tc, tmp_path)
    X, K = exchange_on_card(eri, fj.density, backend, tc, tmp_path)
    assert np.array_equal(X, np.array(exchange_packed(eri.tolist())))
    for what, got, ref in (("J", J, fj.arrays[manifest.J_ARRAY]),
                           ("K", K, fk.arrays[manifest.K_ARRAY])):
        rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
        print(f"{backend} CO/cc-pVTZ ({basis.n_ao} AOs, {len(classes)} "
              f"classes, s to f): max |{what} - v{what.lower()}| / max = "
              f"{rel:.2e} (tol {J_TOL:g})")
        assert rel < J_TOL, what
        assert np.array_equal(got, got.T)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
def test_cuda_co_ccpvtz_j_and_k_low_stack_match_pyscf(request, tmp_path):
    _co("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
def test_hip_co_ccpvtz_j_and_k_low_stack_match_pyscf(request, tmp_path):
    _co("hip", request, tmp_path)
