"""The emitted [p|q] kernel, compiled by the vendor's compiler and run on the card.

THE FIRST NUMERICAL CLAIM THIS PROJECT MAKES. Everything before this file
either interpreted the IR on the host (test/test_kernel.py) or compiled the
emitted text without running it (test_emit_{cuda,hip}.py). Here the text that
`python -m felizjueves.emit` writes is compiled with nvcc / hipcc, launched on
the RTX 3080 / RX 9060 XT, and its `J_p` -- and, separately, its `fj_boys` --
compared against `felizjueves.reference`, whose ancestry differs from the
kernel's (quadrature Boys, a direct memoised recursion with no MILD, DAG,
sieve or schedule).

THE ORACLE IS IN THE DEVICE'S PRECISION, and for this milestone that is
float64: every value the kernel declares is `double` (#22), and so is
`reference`. Nothing here computes a float32 kernel against a float64 oracle.

THE TOLERANCES ARE DERIVED, NOT TUNED TO GREEN.

- `J_p`: the interpreter pin of test/test_kernel.py, unchanged --
  `|J_p - want_p| <= 1e-12 * SUM_k SUM_q |[p|q] D_q|` -- which is the
  recurrence pin's rel = 1e-12 carried through the signed contraction. The
  device differs from the interpreter in two places, and neither loosens it:
  (1) FMA CONTRACTION. nvcc contracts `a*b + c` into one fused multiply-add
  by default (`-fmad=true`), and hipcc (clang) does too for HIP sources
  (`-ffp-contract=fast-honor-pragmas` is its HIP default). That rounds once
  where the interpreter rounds twice: a different rounding, not a
  reassociation, and every bound it enters gets tighter, not looser. Neither
  flag set is touched (devtools/config.sh). (2) libm: `sqrt` is correctly
  rounded on both; `exp` is within 1 ulp on both vendors' double libm. Both
  are far inside 1e-12.

  HOW CLOSE THE WIDE CASE RUNS, AND WHY THAT IS THE ORACLE'S FLOOR. At high L
  with T in the tens-to-hundreds, `[p+q]^(0)` is itself a cancelling sum
  (the Hermite recurrence's two terms have opposite signs), and `scale` does
  not see that cancellation. Measured on this box (2026-10-11), the worst
  `max |J - want| / scale` is ~5e-13 at L = 16 on both cards; recomputing the
  same recurrence in 60-digit `decimal` (with reference.boys's fundamentals)
  shows the float64 ORACLE is itself 2.4e-13 of scale from it there, the
  device 3.2e-13, and both under 7e-16 of the absolute-value recurrence
  `SUM |D_q| |[p+q]|_abs` (up to 7.6e3 x scale). So what is left between
  kernel and oracle at L = 16 is two float64 evaluations of an ill-conditioned
  sum, not a defect in either; the pin is not loosened for it. If a change
  pushes the wide case over, measure against the decimal recurrence before
  concluding either side is wrong.
- `fj_boys`: the routine emit/_c.py writes (the Kummer series, BOYS_DEVICE),
  evaluated by itself on the card over m = 0..16 and T across [0, 500] --
  `reference.boys`'s measured domain. A forward error bound, per point:
  each of the N series steps rounds one division and one product into
  `term` (<= 2u each step, compounding to <= 2N u on the last), the sum of
  N positive terms adds <= N u, `exp` <= 1 ulp and the final product 1 u, so
  `|fj - F| / F <= (3N + 2) u + 1e-14`, the last term being reference.boys's
  own asserted error (test/test_reference_boys.py), u = 2^-53 and N the
  series length (recomputed on the host by the same loop). No FMA can form
  inside it (no `a*b + c` appears), so the bound is the same on both cards.
  Measured worst: printed by the test (`-s`), and recorded in the PR.

SPLIT CELLS (#53), at the bottom: every `measured.SUBGRID` cell with
n_batches > 1 runs as its whole atomic launch sequence and is held to the
same J_REL pin -- the first time the atomic accumulation is checked on
hardware rather than interpreted -- and an OPT-IN wall-clock sweep
(FJ_WALLCLOCK_STORE) times every SUBGRID cell into the measured store. The
sweep's section comment is its protocol; it skips in the kernel tier, by
design, because a timing taken beside the tier's compiles is contended.

WHAT SKIPS, AND WHY: every test here is `gpu` plus `cuda` or `hip`, takes the
compiler fixture AND the device fixture, and so skips -- not fails -- on a box
without the card. A green run of this file on a device-less box is EMPTY;
devtools/kernel-tier.sh's report says `device ABSENT -- numbers unproven`.

THE TIER'S OWN SOURCES. Under devtools/kernel-tier.sh, KERNEL_TIER_OUT names
the directory the generate phase wrote; when it is set the test also asserts
that the file there is byte-identical to the text it compiles, so the number
checked is the number of the generated translation unit, not of a sibling.
"""

from __future__ import annotations

import datetime
import hashlib
import math
import os
import shlex
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from conftest import VRAM_PROBE, check_vram

from felizjueves import asym, gate, measured
from felizjueves.device import GFX1200, SM_86, Arch
from felizjueves.emit import cuda, hip
from felizjueves.emit._c import BOYS_DEVICE
from felizjueves.kernel import (
    bra_functions,
    build_pq_kernel,
    ket_functions,
    launch_shape,
    pq_spec,
    split,
)
from felizjueves.lower import RECYCLE
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, theta_of
from felizjueves.spec import Spec

SUBGRID = measured.SUBGRID

# Every class the generator writes by default: L = 0..16.
LS = tuple(range(0, 17))
J_REL = 1e-12  # the interpreter pin's, propagated (module docstring)
U = 2.0**-53
BOYS_REF_ERR = 1e-14  # reference.boys's asserted error

PORTABLE = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define devMalloc       hipMalloc
#define devMemcpy       hipMemcpy
#define devMemcpyH2D    hipMemcpyHostToDevice
#define devMemcpyD2H    hipMemcpyDeviceToHost
#define devSynchronize  hipDeviceSynchronize
#define devGetLastError hipGetLastError
#define devGetErrorString hipGetErrorString
#define devError_t      hipError_t
#define devSuccess      hipSuccess
#else
#define devMalloc       cudaMalloc
#define devMemcpy       cudaMemcpy
#define devMemcpyH2D    cudaMemcpyHostToDevice
#define devMemcpyD2H    cudaMemcpyDeviceToHost
#define devSynchronize  cudaDeviceSynchronize
#define devGetLastError cudaGetLastError
#define devGetErrorString cudaGetErrorString
#define devError_t      cudaError_t
#define devSuccess      cudaSuccess
#endif
#define CHECK(x) do { devError_t e_ = (x); if (e_ != devSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, devGetErrorString(e_)); return 1; } } while (0)
""" + VRAM_PROBE + r"""

static bool slurp(const char* path, std::vector<double>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(n / sizeof(double));
    bool ok = std::fread(out.data(), sizeof(double), out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

static bool spill(const char* path, const std::vector<double>& v) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    bool ok = std::fwrite(v.data(), sizeof(double), v.size(), f) == v.size();
    std::fclose(f);
    return ok;
}

static double* upload(const double* h, size_t n) {
    double* d = nullptr;
    if (devMalloc(&d, n * sizeof(double) + 8) != devSuccess) return nullptr;
    if (n && devMemcpy(d, h, n * sizeof(double), devMemcpyH2D) != devSuccess)
        return nullptr;
    return d;
}
"""

# The [p|q] harness: one flat float64 input file, laid out as
#   n_bra n_ket n_p n_q lanes tau | P zeta Ip | Q eta Iq Dmax | D
# (the five counts stored as doubles, exact), J_p written back raw.
PQ_HARNESS = r"""
#include "{kernel}"
{portable}
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in) || in.size() < 6) {{
        std::fprintf(stderr, "bad input\n");
        return 2;
    }}
    const int n_bra = (int)in[0], n_ket = (int)in[1], n_p = (int)in[2],
              n_q = (int)in[3], lanes = (int)in[4];
    const double tau = in[5];
    const double* h = in.data() + 6;
    const size_t sizes[] = {{(size_t)n_bra * 3, (size_t)n_bra, (size_t)n_bra,
                            (size_t)n_ket * 3, (size_t)n_ket, (size_t)n_ket,
                            (size_t)n_ket, (size_t)n_ket * n_q}};
    if (fj_vram_probe(in.size() * sizeof(double) + 64
                          + (size_t)n_bra * n_p * sizeof(double),
                      {{(const void*){kernel_name}}})) return 0;
    double* d[8];
    for (int i = 0; i < 8; ++i) {{
        d[i] = upload(h, sizes[i]);
        if (!d[i]) {{ std::fprintf(stderr, "upload %d failed\n", i); return 1; }}
        h += sizes[i];
    }}
    if ((size_t)(h - in.data()) != in.size()) {{
        std::fprintf(stderr, "size mismatch\n");
        return 2;
    }}
    double* dJ = nullptr;
    CHECK(devMalloc(&dJ, (size_t)n_bra * n_p * sizeof(double)));
    const size_t shared = (size_t)n_p * lanes * sizeof(double);
    {kernel_name}<<<dim3(n_bra), dim3(1, lanes), shared>>>(
        n_bra, n_ket, tau, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], dJ);
    CHECK(devGetLastError());
    CHECK(devSynchronize());
    std::vector<double> J((size_t)n_bra * n_p);
    CHECK(devMemcpy(J.data(), dJ, J.size() * sizeof(double), devMemcpyD2H));
    if (!spill(argv[2], J)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("launched %s: grid %d, block (1, %d), %zu B shared\n",
                "{kernel_name}", n_bra, lanes, shared);
    return 0;
}}
"""

# The Boys harness: fj_boys exactly as emit/_c.py writes it, one thread per
# (m, T) point. Input: n | m[n] | T[n]; output F[n].
BOYS_HARNESS = r"""
{prologue}
{portable}
{boys}
__global__ void fj_boys_grid(int n, const double* m, const double* T, double* F) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) F[i] = fj_boys((int)m[i], T[i]);
}}

int main(int argc, char** argv) {{
    if (argc != 3) return 2;
    std::vector<double> in;
    if (!slurp(argv[1], in) || in.empty()) return 2;
    const int n = (int)in[0];
    if (fj_vram_probe((size_t)3 * n * sizeof(double) + 16,
                      {{(const void*)fj_boys_grid}})) return 0;
    double* dm = upload(in.data() + 1, n);
    double* dT = upload(in.data() + 1 + n, n);
    double* dF = nullptr;
    if (!dm || !dT) return 1;
    CHECK(devMalloc(&dF, (size_t)n * sizeof(double)));
    fj_boys_grid<<<(n + 63) / 64, 64>>>(n, dm, dT, dF);
    CHECK(devGetLastError());
    CHECK(devSynchronize());
    std::vector<double> F(n);
    CHECK(devMemcpy(F.data(), dF, F.size() * sizeof(double), devMemcpyD2H));
    return spill(argv[2], F) ? 0 : 1;
}}
"""

BACKEND = {
    "cuda": (cuda.emit, SM_86, "cu"),
    "hip": (hip.emit, GFX1200, "hip"),
}


def _kernel_text(backend: str, L: int) -> tuple[str, str]:
    emit, arch, ext = BACKEND[backend]
    k = build_pq_kernel(L)
    text = emit(k, arch)
    out = os.environ.get("KERNEL_TIER_OUT")
    if out and os.environ.get("KERNEL_TIER_BACKEND") == backend:
        generated = Path(out) / f"{k.name}.{ext}"
        assert generated.read_text() == text, (
            f"{generated} differs from what emit() writes now -- a stale "
            "generate phase, or a nondeterministic emitter")
    return k.name, text


# ---------------------------------------------------------------------------
# Inputs and the oracle
# ---------------------------------------------------------------------------


def _inputs(L: int, *, wide: bool, tau_quantile: float | None, seed: int,
            n_bra: int, n_ket: int) -> dict:
    """Bra and ket indices, kets sorted descending by Iq * Dmax.

    `wide=False` is the interpreter pin's regime (T < ~2). `wide=True`
    spreads centres over a 12-bohr box and exponents over [0.2, 2], so T
    runs from 0 (one ket placed on bra 0) to ~430 -- inside reference.boys's
    measured [0, 500], and below where the Kummer series overflows (~700).
    `tau_quantile` sets tau at that quantile of bra 0's bounds, so part of
    every warp's strip breaks.
    """
    spec = pq_spec(L)
    rng = np.random.default_rng(seed)
    n_q = len(ket_functions(spec))
    box, (elo, ehi) = (6.0, (0.2, 2.0)) if wide else (0.5, (0.6, 1.4))
    Iq = rng.uniform(0.2, 1.0, n_ket)
    Dmax = rng.uniform(0.5, 1.0, n_ket)
    srt = np.argsort(-(Iq * Dmax), kind="stable")
    P = rng.uniform(-box, box, (n_bra, 3))
    Q = rng.uniform(-box, box, (n_ket, 3))
    if wide:
        Q[n_ket // 2] = P[0]  # T = 0 exactly, for one pair
    args = dict(
        P=P, zeta=rng.uniform(elo, ehi, n_bra), Ip=rng.uniform(0.5, 1.0, n_bra),
        Q=Q, eta=rng.uniform(elo, ehi, n_ket),
        Iq=Iq[srt].copy(), Dmax=Dmax[srt].copy(),
        D=rng.uniform(-1.0, 1.0, (n_ket, n_q)),
    )
    args["tau"] = 0.0
    if tau_quantile is not None:
        bounds = args["Ip"][0] * args["Iq"] * args["Dmax"]
        args["tau"] = float(np.quantile(bounds, tau_quantile))
    return args


def _reference(L: int, a: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """want, scale = SUM_k [bound >= tau] SUM_q |[p|q] D_q|, and the largest T."""
    spec = pq_spec(L)
    lp, lq = split(spec)
    ps, qs = bra_functions(spec), ket_functions(spec)
    n_bra, n_ket = a["P"].shape[0], a["Q"].shape[0]
    want = np.zeros((n_bra, len(ps)))
    scale = np.zeros_like(want)
    t_max = 0.0
    sign = np.array([(-1.0) ** sum(q) for q in qs])
    for b in range(n_bra):
        for k in range(n_ket):
            if a["Ip"][b] * (a["Iq"][k] * a["Dmax"][k]) < a["tau"]:
                continue  # a continue, so this side does not rely on the sort
            PQ = tuple(float(c) for c in a["P"][b] - a["Q"][k])
            th = theta_of(float(a["zeta"][b]), float(a["eta"][k]))
            t_max = max(t_max, th * th * sum(c * c for c in PQ))
            R = {r: one_centre(r, 0, PQ, th) for r in hermite_targets(lp + lq)}
            for i, p in enumerate(ps):
                row = np.array([R[(p[0] + q[0], p[1] + q[1], p[2] + q[2])]
                                for q in qs]) * sign * a["D"][k]
                want[b, i] += row.sum()
                scale[b, i] += np.abs(row).sum()
    return want, scale, t_max


def _pack(L: int, a: dict, lanes: int) -> np.ndarray:
    spec = pq_spec(L)
    head = [a["P"].shape[0], a["Q"].shape[0], len(bra_functions(spec)),
            len(ket_functions(spec)), lanes, a["tau"]]
    body = [a[n].ravel() for n in ("P", "zeta", "Ip", "Q", "eta", "Iq", "Dmax", "D")]
    return np.concatenate([np.array(head, dtype=np.float64), *body])


def _toolchain(backend: str, request: pytest.FixtureRequest):
    if backend == "cuda":
        tc = request.getfixturevalue("nvcc")
        request.getfixturevalue("cuda_device")
    else:
        tc = request.getfixturevalue("hipcc")
        request.getfixturevalue("hip_device")
    return tc


# ---------------------------------------------------------------------------
# J_p on the card
# ---------------------------------------------------------------------------

# (wide, tau_quantile, n_bra, n_ket): the interpreter's regime with no
# screening; a wide-T spread with part of every strip screened. n_ket > 32
# lanes, and not a multiple of it, so every lane strides and some lanes run
# one ket fewer.
CASES = {
    "near": (False, None, 5, 77),
    "wide-screened": (True, 0.4, 4, 45),
}


def _run_pq(backend: str, L: int, request, compile_device_source,
            run_device_binary, tmp_path: Path) -> list[tuple[str, float, float]]:
    tc = _toolchain(backend, request)
    arch: Arch = BACKEND[backend][1]
    name, text = _kernel_text(backend, L)
    kpath = tmp_path / f"{name}.{BACKEND[backend][2]}"
    kpath.write_text(text)
    binary = compile_device_source(tc, PQ_HARNESS.format(
        kernel=kpath, portable=PORTABLE, kernel_name=name))
    results = []
    for case, (wide, tq, n_bra, n_ket) in CASES.items():
        a = _inputs(L, wide=wide, tau_quantile=tq, seed=1000 * L + len(case),
                    n_bra=n_bra, n_ket=n_ket)
        grid, (bx, lanes) = launch_shape(n_bra, arch.wavefront)
        assert (grid, bx) == (n_bra, 1)
        src, dst = tmp_path / f"{case}.in", tmp_path / f"{case}.out"
        _pack(L, a, lanes).tofile(src)
        run_device_binary(binary, str(src), str(dst), vram=backend)
        got = np.fromfile(dst, dtype=np.float64).reshape(n_bra, -1)
        want, scale, t_max = _reference(L, a)
        assert got.shape == want.shape
        assert np.all(np.isfinite(got)), f"{case}: non-finite J_p"
        err = float(np.max(np.abs(got - want) / scale))
        assert err <= J_REL, (
            f"L={L} {case} on {arch.name}: max |J - want| / scale = {err:.3e} "
            f"> {J_REL:.0e} (T up to {t_max:.1f})")
        results.append((case, err, t_max))
    print(f"\n{backend} {arch.name} L={L}: " + ", ".join(
        f"{c} err/scale {e:.2e} (T<={t:.0f})" for c, e, t in results))
    return results


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("L", LS)
def test_cuda_j_p_matches_the_reference(L, request, compile_device_source,
                                        run_device_binary, tmp_path) -> None:
    """The emitted CUDA kernel's J_p on the RTX 3080 agrees with the oracle."""
    _run_pq("cuda", L, request, compile_device_source, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("L", LS)
def test_hip_j_p_matches_the_reference(L, request, compile_device_source,
                                       run_device_binary, tmp_path) -> None:
    """The emitted HIP kernel's J_p on the RX 9060 XT agrees with the oracle."""
    _run_pq("hip", L, request, compile_device_source, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# fj_boys on the card
# ---------------------------------------------------------------------------

BOYS_M = range(0, 17)  # the fundamentals of a class run m = 0..L, L <= 16
BOYS_T = sorted({0.0, 1e-12, 1e-6, 1e-3, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0,
                 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0,
                 *np.random.default_rng(28).uniform(0.0, 500.0, 40).tolist(),
                 *np.random.default_rng(29).uniform(0.0, 3.0, 20).tolist()})


def series_length(m: int, T: float) -> int:
    """How many steps BOYS_DEVICE's loop takes for (m, T): the same loop, here."""
    term = 1.0 / (2 * m + 1)
    s = term
    for k in range(1, 4096):
        term = term * ((2.0 * T) / (2 * m + 2 * k + 1))
        if s + term == s:
            return k
        s = s + term
    return 4096


def _run_boys(backend: str, request, compile_device_source, run_device_binary,
              tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    prologue = "#include <hip/hip_runtime.h>" if backend == "hip" else ""
    binary = compile_device_source(tc, BOYS_HARNESS.format(
        prologue=prologue, portable=PORTABLE, boys=BOYS_DEVICE))
    pts = [(m, T) for m in BOYS_M for T in BOYS_T]
    ms = np.array([m for m, _ in pts], dtype=np.float64)
    Ts = np.array([T for _, T in pts], dtype=np.float64)
    src, dst = tmp_path / "boys.in", tmp_path / "boys.out"
    np.concatenate([[float(len(pts))], ms, Ts]).tofile(src)
    run_device_binary(binary, str(src), str(dst), vram=backend)
    got = np.fromfile(dst, dtype=np.float64)
    assert got.shape == (len(pts),)
    worst = (0.0, 0.0, None)
    bad = []
    for (m, T), g in zip(pts, got, strict=True):
        ref = boys(m, T)
        rel = abs(g - ref) / ref
        bound = (3 * series_length(m, T) + 2) * U + BOYS_REF_ERR
        if not rel <= bound:
            bad.append((m, T, rel, bound))
        if rel / bound > worst[1]:
            worst = (rel, rel / bound, (m, T))
    worst_rel = max(abs(g - boys(m, T)) / boys(m, T)
                    for (m, T), g in zip(pts, got, strict=True))
    print(f"\n{backend} fj_boys: {len(pts)} points, worst rel err "
          f"{worst_rel:.2e}; worst against its bound {worst[1]:.3f} at "
          f"(m, T) = {worst[2]}")
    assert not bad, f"fj_boys outside its forward error bound at {bad[:5]}"
    assert math.isfinite(worst_rel)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_fj_boys_matches_reference_boys(request, compile_device_source,
                                             run_device_binary, tmp_path) -> None:
    """The emitted Boys routine on the RTX 3080, against quadrature."""
    _run_boys("cuda", request, compile_device_source, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_fj_boys_matches_reference_boys(request, compile_device_source,
                                            run_device_binary, tmp_path) -> None:
    """The emitted Boys routine on the RX 9060 XT, against quadrature."""
    _run_boys("hip", request, compile_device_source, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# #53: the batched launch sequence on the card -- J_p, then wall-clock
# ---------------------------------------------------------------------------
#
# A SPLIT CELL IS ONE LAUNCH SEQUENCE. `build_pq_kernels(spec)` is one kernel
# per batch, every one taking the same arguments and launch shape, run in plan
# order into ONE J the caller zeroes (kernel.py, build_pq_kernels). The harness
# below includes every batch's emitted text, each in its own namespace (each
# carries its own `fj_boys`; the kernels themselves are `extern "C"` and named
# apart), and launches them back to back on the default stream with the
# dynamic shared memory each kernel's own `Jsh` declaration asks for. One
# sequence is what a cell IS, for the comparison and for the clock alike.
#
# THE COMPARISON, AND WHY ITS TOLERANCE IS STILL J_REL. Batch b adds its
# partial sum into J[bra, p] with `atomicAdd`. fp64 atomics are UNORDERED in
# general, so J is not promised bit-reproducible, and the comparison must
# allow any order of the n_b partials -- it does, without loosening anything:
# however n_b partials s_1..s_nb are added, the rounding is at most
# (n_b - 1) u SUM_b |s_b| <= 8 u * scale (n_b <= 9, and SUM_b |s_b| <= scale,
# the partials partitioning the very terms scale sums), about 1e-15 * scale,
# three orders inside J_REL = 1e-12 * scale -- the pin the interpreter already
# holds every batched cell to (test/test_kernel_batched.py), whose regrouping
# of the contraction by batch is this one's too. So the batched check is
# `_run_pq`'s pin, unchanged. (Here the launches are stream-ordered and each
# (bra, p) gets one atomic per kernel, from lane 0, so the order happens to be
# fixed; the sweep records whether J came out bit-identical across repeats as
# `wallclock:J_bitwise_repeatable` -- an observation, not an assertion.)
#
# THE TIMING does not care about any of that: it times the sequence and never
# compares its J against anything but itself.

BATCHED_HARNESS = r"""
{prologue}
{portable}
#include <chrono>
{includes}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devEvent_t           hipEvent_t
#define devEventCreate       hipEventCreate
#define devEventRecord(e)    hipEventRecord(e, 0)
#define devEventElapsedTime  hipEventElapsedTime
#define devMemset            hipMemset
#else
#define devEvent_t           cudaEvent_t
#define devEventCreate       cudaEventCreate
#define devEventRecord(e)    cudaEventRecord(e, 0)
#define devEventElapsedTime  cudaEventElapsedTime
#define devMemset            cudaMemset
#endif

// usage: in out warmup reps. Writes J of the FIRST timed sequence, then J of
// the LAST, to `out`; prints one `rep` line per timed sequence:
//   rep <i> wall_ns <host steady_clock, launches + synchronize>
//       events_ns <GPU events, first launch to last> batch_ns <one per batch>
int main(int argc, char** argv) {{
    if (argc != 5) {{ std::fprintf(stderr, "usage: in out warmup reps\n"); return 2; }}
    const int warmup = std::atoi(argv[3]), reps = std::atoi(argv[4]);
    std::vector<double> in;
    if (!slurp(argv[1], in) || in.size() < 6 || reps < 1 || warmup < 0) {{
        std::fprintf(stderr, "bad input\n");
        return 2;
    }}
    const int n_bra = (int)in[0], n_ket = (int)in[1], n_p = (int)in[2],
              n_q = (int)in[3], lanes = (int)in[4];
    const double tau = in[5];
    const double* h = in.data() + 6;
    const size_t sizes[] = {{(size_t)n_bra * 3, (size_t)n_bra, (size_t)n_bra,
                            (size_t)n_ket * 3, (size_t)n_ket, (size_t)n_ket,
                            (size_t)n_ket, (size_t)n_ket * n_q}};
    if (fj_vram_probe(in.size() * sizeof(double) + 64
                          + (size_t)n_bra * n_p * sizeof(double),
                      {{{vram_kernels}}})) return 0;
    double* d[8];
    for (int i = 0; i < 8; ++i) {{
        d[i] = upload(h, sizes[i]);
        if (!d[i]) {{ std::fprintf(stderr, "upload %d failed\n", i); return 1; }}
        h += sizes[i];
    }}
    if ((size_t)(h - in.data()) != in.size()) {{
        std::fprintf(stderr, "size mismatch\n");
        return 2;
    }}
    const size_t jbytes = (size_t)n_bra * n_p * sizeof(double);
    double* dJ = nullptr;
    CHECK(devMalloc(&dJ, jbytes));
    const int NB = {n_batches};
    devEvent_t ev[NB + 1];
    for (int i = 0; i <= NB; ++i) CHECK(devEventCreate(&ev[i]));
    std::vector<double> first((size_t)n_bra * n_p), last(first.size());
    for (int r = 0; r < warmup + reps; ++r) {{
        CHECK(devMemset(dJ, 0, jbytes));  // the caller zeroes J: off the clock
        CHECK(devSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(devEventRecord(ev[0]));
{launches}
        CHECK(devGetLastError());
        CHECK(devSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        if (r < warmup) continue;
        float total_ms = 0.f;
        CHECK(devEventElapsedTime(&total_ms, ev[0], ev[NB]));
        std::printf("rep %d wall_ns %lld events_ns %.0f batch_ns", r - warmup,
                    (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t1 - t0).count(), (double)total_ms * 1e6);
        for (int i = 0; i < NB; ++i) {{
            float ms = 0.f;
            CHECK(devEventElapsedTime(&ms, ev[i], ev[i + 1]));
            std::printf(" %.0f", (double)ms * 1e6);
        }}
        std::printf("\n");
        std::vector<double>& dst = (r == warmup) ? first : last;
        CHECK(devMemcpy(dst.data(), dJ, jbytes, devMemcpyD2H));
    }}
    if (reps == 1) last = first;
    std::vector<double> both(first);
    both.insert(both.end(), last.begin(), last.end());
    if (!spill(argv[2], both)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    return 0;
}}
"""

LAUNCH = ("        {ns}::{name}<<<dim3(n_bra), dim3(1, lanes), "
          "(size_t){slots} * lanes * sizeof(double)>>>(\n"
          "            n_bra, n_ket, tau, d[0], d[1], d[2], d[3], d[4], d[5], "
          "d[6], d[7], dJ);\n"
          "        CHECK(devEventRecord(ev[{next}]));")


def _batched_source(backend: str, spec: Spec,
                    workdir: Path) -> tuple[str, list[str], str]:
    """The harness for one cell: its source, its batch kernels' names in
    launch order, and the cell's source hash (as `measured.cell_sha256`)."""
    emit, arch, ext = BACKEND[backend]
    kernels = measured.cell_kernels(spec)  # build_pq_kernels, levers applied
    includes, launches, texts = [], [], []
    for i, k in enumerate(kernels):
        text = emit(k, arch)
        texts.append(text)
        path = workdir / f"{k.name}.{ext}"
        path.write_text(text)
        includes.append(f'namespace fj_b{i} {{\n#include "{path}"\n}}')
        launches.append(LAUNCH.format(ns=f"fj_b{i}", name=k.name,
                                      slots=measured.jsh_slots(k), next=i + 1))
    prologue = "#include <hip/hip_runtime.h>" if backend == "hip" else ""
    src = BATCHED_HARNESS.format(
        prologue=prologue, portable=PORTABLE, includes="\n".join(includes),
        n_batches=len(kernels), launches="\n".join(launches),
        vram_kernels=", ".join(f"(const void*)fj_b{i}::{k.name}"
                               for i, k in enumerate(kernels)))
    sha = hashlib.sha256("".join(texts).encode()).hexdigest()
    return src, [k.name for k in kernels], sha


@cache
def _case(L: int, case: str):
    """A CASES input of class L and its oracle -- shared by every n_batches,
    since the inputs depend on the class, never on its split."""
    wide, tq, n_bra, n_ket = CASES[case]
    a = _inputs(L, wide=wide, tau_quantile=tq, seed=1000 * L + len(case),
                n_bra=n_bra, n_ket=n_ket)
    return a, _reference(L, a)


def _check_batched(backend: str, spec: Spec, binary: Path, run_device_binary,
                   tmp_path: Path) -> list[tuple[str, float, float]]:
    """Every CASES input through one cell's launch sequence, against the
    oracle at J_REL (the comparison: see the section comment)."""
    arch: Arch = BACKEND[backend][1]
    L = spec.total_am
    results = []
    for case in CASES:
        a, (want, scale, t_max) = _case(L, case)
        n_bra = a["P"].shape[0]
        _, (_, lanes) = launch_shape(n_bra, arch.wavefront)
        src, dst = tmp_path / f"{case}.in", tmp_path / f"{case}.out"
        _pack(L, a, lanes).tofile(src)
        run_device_binary(binary, str(src), str(dst), "0", "1", vram=backend)
        got = np.fromfile(dst, dtype=np.float64).reshape(2, n_bra, -1)[0]
        assert got.shape == want.shape
        assert np.all(np.isfinite(got)), f"{case}: non-finite J_p"
        err = float(np.max(np.abs(got - want) / scale))
        assert err <= J_REL, (
            f"{spec.key()} {case} on {arch.name}: max |J - want| / scale = "
            f"{err:.3e} > {J_REL:.0e} (T up to {t_max:.1f})")
        results.append((case, err, t_max))
    print(f"\n{backend} {arch.name} {spec.key()}: " + ", ".join(
        f"{c} err/scale {e:.2e} (T<={t:.0f})" for c, e, t in results))
    return results


SPLIT_CELLS = [pytest.param(s, id=f"L{s.total_am}-b{s.n_batches}")
               for s in SUBGRID if s.n_batches > 1]


def _run_batched(backend: str, spec: Spec, request, compile_device_source,
                 run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    src, _, _ = _batched_source(backend, spec, tmp_path)
    binary = compile_device_source(tc, src)
    _check_batched(backend, spec, binary, run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", SPLIT_CELLS)
def test_cuda_batched_j_p_matches_the_reference(spec, request, compile_device_source,
                                                run_device_binary, tmp_path) -> None:
    """A split cell's atomic launch sequence on the RTX 3080, against the oracle."""
    _run_batched("cuda", spec, request, compile_device_source, run_device_binary,
                 tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", SPLIT_CELLS)
def test_hip_batched_j_p_matches_the_reference(spec, request, compile_device_source,
                                               run_device_binary, tmp_path) -> None:
    """A split cell's atomic launch sequence on the RX 9060 XT, against the oracle."""
    _run_batched("hip", spec, request, compile_device_source, run_device_binary,
                 tmp_path)


# --- the recycle lever (#55): the same pin, the other arm -------------------
#
# `lower.recycle` turns every named fp64 value into an element of one local
# array, reusing a slot only between closed live intervals; the arithmetic and
# its order are unchanged, so the recycled kernel is held to the SAME J_REL
# pin through the same launch sequence -- not a looser one. Every lever cell
# (`measured.lever_cells`: the whole SUBGRID, one batch included), on both
# cards.

RECYCLE_CELLS = [pytest.param(s, id=f"L{s.total_am}-b{s.n_batches}-recycle")
                 for s in measured.lever_cells(RECYCLE)]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", RECYCLE_CELLS)
def test_cuda_recycled_j_p_matches_the_reference(spec, request,
                                                 compile_device_source,
                                                 run_device_binary,
                                                 tmp_path) -> None:
    """A recycle-lever cell's launch sequence on the RTX 3080, against the oracle."""
    _run_batched("cuda", spec, request, compile_device_source, run_device_binary,
                 tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", RECYCLE_CELLS)
def test_hip_recycled_j_p_matches_the_reference(spec, request,
                                                compile_device_source,
                                                run_device_binary,
                                                tmp_path) -> None:
    """A recycle-lever cell's launch sequence on the RX 9060 XT, against the oracle."""
    _run_batched("hip", spec, request, compile_device_source, run_device_binary,
                 tmp_path)


# --- the asymmetric split (#113): the same pin, a batch list the generator ----
# --- refuses -------------------------------------------------------------------
#
# `asym.AsymCell` builds an asymmetric batch list (felizjueves.asym's
# docstring) through the same `_pq_kernel` + `atomize` as a split SUBGRID
# cell, and its batches partition the class (checked by the study itself), so
# its launch sequence is held to the SAME J_REL pin -- the batched comparison's
# argument above holds for any partition with n_b <= 9. Every study cell, on
# both cards, before any of them is timed.

ASYM_CELLS = [pytest.param(c, id=f"L{c.L}-b{c.n_batches}-asym")
              for c in asym.ASYM_CELLS]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cell", ASYM_CELLS)
def test_cuda_asym_j_p_matches_the_reference(cell, request,
                                             compile_device_source,
                                             run_device_binary,
                                             tmp_path) -> None:
    """An asymmetric cell's launch sequence on the RTX 3080, against the oracle."""
    _run_batched("cuda", cell, request, compile_device_source, run_device_binary,
                 tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cell", ASYM_CELLS)
def test_hip_asym_j_p_matches_the_reference(cell, request,
                                            compile_device_source,
                                            run_device_binary,
                                            tmp_path) -> None:
    """An asymmetric cell's launch sequence on the RX 9060 XT, against the oracle."""
    _run_batched("hip", cell, request, compile_device_source, run_device_binary,
                 tmp_path)


# --- the wall-clock sweep -----------------------------------------------------
#
# OPT-IN, AND RUN ALONE. It skips unless FJ_WALLCLOCK_STORE names the store to
# write (normally data/measured/kernels.json), because inside the kernel tier
# it would share the box with xdist workers compiling other tests -- and a
# contended timing is garbage, not a noisy measurement. Run it by itself, with
# nothing else compile-bound on the box; the card's lock is the device
# fixture's (test/conftest.py, GPU_LOCK_PATH), so no `flock` wrapper is needed
# -- one is harmless, the fixture recognising an ancestor's hold as its own:
#
#   env FJ_WALLCLOCK_STORE=data/measured/kernels.json \
#       uv run pytest test/test_kernel_device.py -p no:xdist -s -rs \
#       -m "gpu and cuda" -k wallclock
#
# FJ_WALLCLOCK_BATCHES (optional, e.g. "9,12") restricts the sweep to the
# SUBGRID cells of those n_batches -- every other row is left as it is. #60
# used it to time 9 and 12 in ONE session: its verdict compares the two
# against their IQR, and a cross-session comparison would mix the box's
# drift into the effect.
#
# FJ_WALLCLOCK_LEVER (optional, e.g. "recycle") sweeps the lever's cells,
# `measured.lever_cells(lever)`, instead: each cell's literal arm is compiled,
# checked and timed BESIDE it as a same-session control (near-T control then
# lever, wide-T lever then control) and lands in the LEVER row's raw as
# `wallclock:control:*`; the literal rows are not rewritten. measured.py's
# "A LEVER A/B" is the protocol's rationale and `measured.lever_ab` the
# verdict (#55). The compile rows of the lever cells must exist first:
#   python -m felizjueves.measured --remeasure --subgrid --lever recycle
#
# FJ_WALLCLOCK_ASYM=1 (optional) sweeps #113's asymmetric study cells,
# `asym.ASYM_CELLS`, instead: each cell's two nearest legal neighbours
# (literal SUBGRID Specs) are compiled, checked and timed BESIDE it as
# same-session controls (near-T lo, asym, hi; wide-T hi, asym, lo) and land
# in the ASYMMETRIC row's raw as `wallclock:control:b<n>:*`; no literal row
# is re-timed or rewritten. `asym`'s docstring is the rule, declared before
# any asymmetric row existed, and `asym.asym_vs_neighbours` the verdict. A
# study phase the card was taken during is retaken (STUDY_RETAKES). The
# compile rows must exist first:
#   python -m felizjueves.measured --remeasure --asym
#
# THE PROTOCOL, in order, all inside one test so the phases cannot overlap:
#
# 1. CPUSET from devtools/config.sh, if set, pins this process (and so the
#    harness) with sched_setaffinity; CPUS is a container quota and cannot be
#    applied to a bare process -- a set CPUS is reported, not silently
#    honoured. Both are empty by default.
# 2. COMPILE every SUBGRID cell's harness, one after another, with config.sh's
#    NVCC_FLAGS / HIPCC_FLAGS (the flag set the measured rows' registers came
#    from). No compile runs after this point.
# 3. CHECK every compiled binary against the oracle on CASES (`_check_batched`)
#    -- the TIMED binary is the CHECKED binary.
# 4. TIME each cell: wait for the box to be quiet (CPU busy at most BUSY_MAX
#    over a 1 s sample, for up to QUIET_WAIT_S; otherwise the test FAILS --
#    the run is discarded, not averaged in) AND for the card to be (its own
#    SMI's utilization at most GPU_BUSY_MAX, for up to GPU_QUIET_WAIT_S --
#    added by #55, after a process outside the container held the RTX 3080;
#    #93 records it per input: raw `wallclock:gpu_util_before` is the settled
#    reading the gate let through, `wallclock:gpu_util_after` one taken
#    GPU_SETTLE_S after the last timed sequence, `wallclock:near:*` and
#    `wallclock:control:*` likewise -- `null` where the SMI said nothing
#    parseable, which means UNKNOWN and never idle),
#    then WARMUP untimed sequences and
#    REPEATS timed ones, on each of two inputs -- no screening (tau = 0) so
#    every cell does all its work, n_bra x n_ket = TIMING_N_BRA x
#    TIMING_N_KET, and the same input for every n_batches of one L:
#      WIDE (CASES' wide regime, T up to ~430): THE ROW's number. The median
#        and IQR of the host wall-clock are runtime_ns / spread_ns.
#      NEAR (T < ~2): `wallclock:near:*` in raw. Each batch recomputes its
#        fundamentals (kernel.py), so the Boys series length -- long at
#        large T -- weighs on the batch-count ranking; timing both regimes
#        shows whether the ranking depends on it, instead of hiding it.
# 5. WRITE one row per cell, `measured.combine`d into the existing compile
#    row (same text, one row: measured.py's docstring).

WALLCLOCK_STORE = "FJ_WALLCLOCK_STORE"
WALLCLOCK_BATCHES = "FJ_WALLCLOCK_BATCHES"
WALLCLOCK_LEVER = "FJ_WALLCLOCK_LEVER"
WALLCLOCK_ASYM = "FJ_WALLCLOCK_ASYM"
# #113 only: a phase whose after-reading is non-idle or unknown was shared
# with something outside this process (the first study session saw 100%
# after-readings with bimodal times), so it is DISCARDED and retaken, up to
# this many times, and the row counts its discards in raw
# `wallclock:gate_discarded_phases`. The literal and lever sweeps record the
# after-reading instead, as they always have.
STUDY_RETAKES = 5
WARMUP = 5
REPEATS = 31
TIMING_N_BRA = 2048  # one warp each: several waves on either card
TIMING_N_KET = 128   # four kets per lane at wavefront 32
# THE TIMING GATE, on top of the correctness gate the device fixtures already
# hold (test/conftest.py: the card's lock and the free-VRAM probe), is
# `felizjueves.gate` -- lifted there by #109 so the ceiling probes (and #112's
# counter sweep) read the same code; its docstring has the readings and why
# the after-reading is taken the way it is. Utilisation is a reason to WAIT
# here and nowhere else: #55's timings came out bimodal while a process
# outside the container held the RTX 3080 at 100%, and a timing on a busy
# card is garbage where a correct answer is still correct. The names below
# are the gate's, re-exported so this file's protocol reads as it did.
BUSY_MAX = gate.BUSY_MAX
QUIET_WAIT_S = gate.QUIET_WAIT_S
GPU_BUSY_MAX = gate.GPU_BUSY_MAX
GPU_QUIET_WAIT_S = gate.GPU_QUIET_WAIT_S
GPU_SETTLE_S = gate.GPU_SETTLE_S
_cpu_busy = gate.cpu_busy
_card_util = gate.card_util


def _quiet(backend: str | None = None, seen: list[int] | None = None) -> float:
    """Block until the box is quiet -- the CPU and, given `backend`, the card
    (`gate.wait_quiet`) -- and return the CPU busy fraction it settled at; the
    card's settled utilization is appended to `seen`. Past the gate's
    deadline the test FAILS: the timing is discarded, not averaged in."""
    try:
        busy, util = gate.wait_quiet(
            backend, say=lambda line: print(line, flush=True))
    except gate.Contended as e:
        pytest.fail(f"{e} -- timing DISCARDED, not averaged in")
    if backend is not None and seen is not None:
        seen.append(util)
    return busy


def _cpu_list(spec: str) -> set[int]:
    """`0-11,16` -> {0..11, 16}."""
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


@dataclass(frozen=True)
class _Configured:
    """A conftest Toolchain, driven with devtools/config.sh's flag set."""

    name: str
    path: str
    arch: str
    ext: str

    def compile_flags(self) -> list[str]:
        b = measured.BACKENDS["cuda" if self.name == "nvcc" else "hip"]
        flags = shlex.split(measured.config_value(b.flags_var))
        if self.name == "nvcc":
            return [*flags, f"-arch=sm_{self.arch}"]
        return [*flags, f"--offload-arch={self.arch}"]


def _parse_reps(out: str, n_batches: int):
    wall, events, batches = [], [], []
    for line in out.splitlines():
        f = line.split()
        if not f or f[0] != "rep":
            continue
        assert (f[2], f[4], f[6]) == ("wall_ns", "events_ns", "batch_ns"), line
        wall.append(int(f[3]))
        events.append(float(f[5]))
        batches.append([float(x) for x in f[7:]])
        assert len(batches[-1]) == n_batches, line
    return wall, events, batches


def _wallclock(backend: str, request, compile_device_source,
               run_device_binary, tmp_path: Path) -> None:
    store_path = os.environ.get(WALLCLOCK_STORE)
    if not store_path:
        pytest.skip(f"wall-clock is opt-in and must run alone, uncontended: set "
                    f"{WALLCLOCK_STORE} (see the section comment)")
    base = _toolchain(backend, request)
    tc = _Configured(base.name, base.path, base.arch, base.ext)
    arch: Arch = BACKEND[backend][1]
    b = measured.BACKENDS[backend]
    toolchain, gpu = b.toolchain(b.compiler()), b.gpu()
    store = measured.load(Path(store_path))

    # 1. CPU bounds
    cpuset, cpus = measured.config_value("CPUSET"), measured.config_value("CPUS")
    if cpuset:
        os.sched_setaffinity(0, _cpu_list(cpuset))
    print(f"\nCPUSET={cpuset or '(empty)'} CPUS={cpus or '(empty)'}"
          + (" -- CPUS is a container quota, not applied here" if cpus else ""))

    # 2. compile everything first
    only = os.environ.get(WALLCLOCK_BATCHES, "").strip()
    wanted = {int(b) for b in only.split(",")} if only else None
    lever = os.environ.get(WALLCLOCK_LEVER, "").strip()
    study = bool(os.environ.get(WALLCLOCK_ASYM, "").strip())
    assert not (lever and study), f"{WALLCLOCK_LEVER} or {WALLCLOCK_ASYM}"
    grid = (asym.ASYM_CELLS if study
            else measured.lever_cells(lever) if lever else SUBGRID)
    sweep = [s for s in grid if wanted is None or s.n_batches in wanted]
    assert sweep, f"{WALLCLOCK_BATCHES}={only!r} selects no cell"
    print(f"{WALLCLOCK_BATCHES}={only or '(all)'} {WALLCLOCK_LEVER}="
          f"{lever or '(none)'} {WALLCLOCK_ASYM}={int(study)}: "
          f"{len(sweep)} cells")
    cells = []
    for spec in sweep:
        work = tmp_path / f"L{spec.total_am}-b{spec.n_batches}"
        work.mkdir()
        src, names, sha = _batched_source(backend, spec, work)
        binary = compile_device_source(tc, src).rename(work / "cell")
        # Same-session controls, as (raw prefix, Spec, binary): the literal
        # arm of a lever cell (measured_store, A LEVER A/B), or an asymmetric
        # cell's two nearest legal neighbours, lo then hi (asym, #113).
        arms = ([("wallclock:control:", replace(spec, levers=()))] if lever
                else [(asym.control_prefix(n), n) for n in spec.neighbours]
                if study else [])
        controls = []
        for prefix, cspec in arms:
            cwork = work / f"control-b{cspec.n_batches}"
            cwork.mkdir()
            csrc, _, _ = _batched_source(backend, cspec, cwork)
            controls.append((prefix, cspec, compile_device_source(
                tc, csrc).rename(cwork / "cell")))
        cells.append((spec, binary, names, sha, controls))
        print(f"compiled {spec.key()}" + "".join(
            f" and control {c[1].key()}" for c in controls), flush=True)
    command = " ".join([Path(tc.path).name, *tc.compile_flags()])

    # 3. the timed binary is the checked binary -- every arm of an A/B
    for spec, binary, _, _, controls in cells:
        _check_batched(backend, spec, binary, run_device_binary, tmp_path)
        for _, cspec, cbinary in controls:
            _check_batched(backend, cspec, cbinary, run_device_binary, tmp_path)

    # 4. time, one cell at a time
    lanes = arch.wavefront
    discarded = [0]  # the study's gate-discarded phases, per cell (below)

    def timed(spec: Spec, binary: Path, wide: bool):
        L = spec.total_am
        a = _inputs(L, wide=wide, tau_quantile=None, seed=53_000 + L,
                    n_bra=TIMING_N_BRA, n_ket=TIMING_N_KET)
        src, dst = tmp_path / "timing.in", tmp_path / "timing.out"
        _pack(L, a, lanes).tofile(src)
        args = (str(src), str(dst), str(WARMUP), str(REPEATS))
        check_vram(binary, args, backend)  # before the wait, off the clock
        for take in range(1 + (STUDY_RETAKES if study else 0)):
            seen: list[int] = []
            busy = _quiet(backend, seen)
            out = run_device_binary(binary, *args)
            # the gate's reading, and one after (gate.after_reading: one SMI
            # read discarded, GPU_SETTLE_S, then the reading -- our own run
            # excluded)
            card = (seen[-1], gate.after_reading(backend))
            if not study or (card[1] is not None and card[1] <= GPU_BUSY_MAX):
                break
            # #113: the card was taken DURING the phase (after-reading
            # non-idle or unknown): the phase is discarded and retaken
            discarded[0] += 1
            print(f"  DISCARDED {spec.key()} {'wide' if wide else 'near'}: "
                  f"card util {card[0]} -> {card[1]}% (take {take + 1})",
                  flush=True)
        else:
            pytest.fail(f"{spec.key()}: the card stayed contended through "
                        f"{STUDY_RETAKES + 1} takes -- nothing recorded")
        wall, events, batches = _parse_reps(out, spec.n_batches)
        assert len(wall) == REPEATS, out
        J = np.fromfile(dst, dtype=np.float64).reshape(2, TIMING_N_BRA, -1)
        assert np.all(np.isfinite(J)), f"{spec.key()}: non-finite J (timing input)"
        q1, med, q3 = (float(x) for x in np.percentile(wall, [25, 50, 75]))
        return wall, events, batches, J, (q1, med, q3), busy, card

    def gpu_fields(prefix: str, card: tuple[int, int | None]) -> dict:
        # measured.GPU_UTIL_FIELDS: the gate's settled reading before the
        # input's warm-up, and one GPU_SETTLE_S after its last timed sequence
        return {f"{prefix}gpu_util_before": card[0],
                f"{prefix}gpu_util_after": card[1]}

    table = []
    def control_stats(prefix: str, got) -> dict:
        w, _, _, _, stats, b_, g = got
        return {f"{prefix}q1_ns": round(stats[0]),
                f"{prefix}median_ns": round(stats[1]),
                f"{prefix}q3_ns": round(stats[2]),
                f"{prefix}min_ns": int(min(w)),
                f"{prefix}max_ns": int(max(w)),
                f"{prefix}cpu_busy_permille_before": round(1000 * b_),
                **gpu_fields(prefix, g)}

    for spec, binary, names, sha, controls in cells:
        L = spec.total_am
        load1 = os.getloadavg()[0]
        discarded[0] = 0
        ctrl: dict[str, int | None] = {}
        # Arm order, so no arm always runs first: near-T the first control
        # before the cell and the rest after it; wide-T the reverse (a lever:
        # near control-lever, wide lever-control; asym: near lo-asym-hi, wide
        # hi-asym-lo).
        for prefix, cspec, cbinary in controls[:1]:
            ctrl |= control_stats(prefix + "near:", timed(cspec, cbinary, False))
        # the secondary input first: near-T (T < ~2, short Boys series)
        n_wall, _, _, _, (n_q1, n_med, n_q3), n_busy, n_gpu = timed(
            spec, binary, False)
        for prefix, cspec, cbinary in controls[1:]:
            ctrl |= control_stats(prefix + "near:", timed(cspec, cbinary, False))
        for prefix, cspec, cbinary in reversed(controls[1:]):
            ctrl |= control_stats(prefix, timed(cspec, cbinary, True))
        # the row's number: wide-T (T up to ~430, the Boys series long)
        wall, events, batches, J, (q1, med, q3), busy, card = timed(
            spec, binary, True)
        for prefix, cspec, cbinary in controls[:1]:
            ctrl |= control_stats(prefix, timed(cspec, cbinary, True))
        busy_after = _cpu_busy(0.5)
        raw = {
            "wallclock:near:median_ns": round(n_med),
            "wallclock:near:q1_ns": round(n_q1),
            "wallclock:near:q3_ns": round(n_q3),
            "wallclock:near:min_ns": int(min(n_wall)),
            "wallclock:near:max_ns": int(max(n_wall)),
            "wallclock:near:cpu_busy_permille_before": round(1000 * n_busy),
            **gpu_fields("wallclock:near:", n_gpu),
            "wallclock:warmup": WARMUP,
            "wallclock:n_bra": TIMING_N_BRA,
            "wallclock:n_ket": TIMING_N_KET,
            "wallclock:lanes": lanes,
            "wallclock:min_ns": int(min(wall)),
            "wallclock:q1_ns": round(q1),
            "wallclock:q3_ns": round(q3),
            "wallclock:max_ns": int(max(wall)),
            "wallclock:events_median_ns": round(float(np.median(events))),
            "wallclock:cpu_busy_permille_before": round(1000 * busy),
            "wallclock:cpu_busy_permille_after": round(1000 * busy_after),
            "wallclock:loadavg1_milli": round(1000 * load1),
            **gpu_fields("wallclock:", card),
            "wallclock:J_bitwise_repeatable": int(np.array_equal(J[0], J[1])),
            **ctrl,
        }
        if study:
            raw["wallclock:gate_discarded_phases"] = discarded[0]
        per_batch = np.median(np.array(batches), axis=0)
        for name, t in zip(names, per_batch, strict=True):
            raw[f"wallclock:{name}:events_median_ns"] = round(float(t))
        row = measured.Measured(
            runtime_ns=round(med), repeats=REPEATS, spread_ns=round(q3 - q1),
            toolchain=toolchain, arch=arch.name, host=measured.host(), gpu=gpu,
            date=datetime.date.today().isoformat(), kernel=",".join(names),
            source_sha256=sha, raw=raw)
        key = measured.measured_key(spec, arch, toolchain)
        old = store.get(key.key())
        assert old is None or old.source_sha256 == sha, (
            f"{key.key()}: the store's compile row is of other text -- "
            "remeasure the compile side first")
        store = measured.merge(store, key, measured.combine(old, row))
        table.append((L, spec.n_batches, med, q3 - q1, n_med, n_q3 - n_q1))
        print(f"{key.key()}: wide median {med / 1e6:.3f} ms, IQR "
              f"{(q3 - q1) / 1e6:.3f} ms, range [{min(wall) / 1e6:.3f}, "
              f"{max(wall) / 1e6:.3f}] ms; near median {n_med / 1e6:.3f} ms, "
              f"IQR {(n_q3 - n_q1) / 1e6:.3f} ms; {REPEATS} timed (+{WARMUP} "
              f"warm-up) each; CPU busy {n_busy:.1%}/{busy:.1%} -> "
              f"{busy_after:.1%}, load1 {load1:.2f}; card util {n_gpu[0]}->"
              f"{n_gpu[1]}% / {card[0]}->{card[1]}%; J bit-repeatable "
              f"{raw['wallclock:J_bitwise_repeatable']}", flush=True)
        for prefix, cspec, _ in controls if study else ():
            print(f"  control {cspec.key()} (same session): wide median "
                  f"{ctrl[prefix + 'median_ns'] / 1e6:.3f} ms, near median "
                  f"{ctrl[prefix + 'near:median_ns'] / 1e6:.3f} ms", flush=True)
        if lever:
            print(f"  control (literal, same session): wide median "
                  f"{ctrl['wallclock:control:median_ns'] / 1e6:.3f} ms, near "
                  f"median {ctrl['wallclock:control:near:median_ns'] / 1e6:.3f}"
                  " ms", flush=True)

    # 5. one write, after every cell timed
    measured.save(store, Path(store_path))
    if lever:
        print(f"\n#55 A/B, {lever} against its same-session literal control:")
        for spec, *_ in cells:
            c = measured.lever_ab(store, arch.name, spec)
            print(f"  L={c.L} b{c.n_batches}: wide {c.wide_sign}, near "
                  f"{c.near_sign} -> {c.verdict.upper()} (spill {c.spill[0]} "
                  f"-> {c.spill[1]})")
    if study:
        print("\n#113, the asymmetric split against its same-session "
              "neighbours:")
        judged = [spec for spec, *_ in cells]
        for spec in judged:
            c = asym.asym_vs_neighbours(store, arch.name, spec)
            print(f"  L={c.L} b{c.n_batches} {c.ks}: wide {c.wide_sign} vs "
                  f"b{c.wide_best}, near {c.near_sign} vs b{c.near_best} -> "
                  f"{c.verdict.upper()} (spill {c.spill}, regs {c.registers})")
        print(f"  {arch.name}: "
              f"{asym.arch_verdict(store, arch.name, judged).upper()}")
    print(f"\n{backend} {arch.name} ({gpu}), compiled `{command}`")
    for col, label in ((2, "wide (the row)"), (4, "near (raw)")):
        print(f" {label}: median+-IQR, ms")
        for L in sorted({t[0] for t in table}):
            at_L = [t for t in table if t[0] == L]
            best = min(at_L, key=lambda t: t[col])
            print(f"  L={L}: " + "  ".join(
                f"b{t[1]} {t[col] / 1e6:.3f}+-{t[col + 1] / 1e6:.3f}"
                for t in at_L) + f"   -> lowest median at n_batches={best[1]}")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_wallclock_over_the_subgrid(request, compile_device_source,
                                         run_device_binary, tmp_path) -> None:
    """Wall-clock of every SUBGRID cell's launch sequence on the RTX 3080."""
    _wallclock("cuda", request, compile_device_source, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_wallclock_over_the_subgrid(request, compile_device_source,
                                        run_device_binary, tmp_path) -> None:
    """Wall-clock of every SUBGRID cell's launch sequence on the RX 9060 XT."""
    _wallclock("hip", request, compile_device_source, run_device_binary, tmp_path)
