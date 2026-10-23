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
- `fj_boys<M>`: the one template emit/_c.py writes (#301, BOYS_DEVICE),
  every order `F[0..M]` of a top `M`, evaluated by itself on the card for
  every `M` in 0..16 (a class's fundamentals run m = 0..L, L <= 16) and T
  across [0, 1e6] -- `reference.boys`' whole domain, every fixture's reach
  (V's 1.25e5 on HF/cc-pVTZ, the four-H2 chain's 1289) and the old narrow
  routine's overflow near 709 -- with each switch `T = M + 8` sampled on both
  sides down to the adjacent doubles. A forward error bound per point
  (`boys_bound`), plus reference.boys's own asserted error 1e-14
  (test/test_reference_boys.py), u = 2^-53:
  * below the switch, the series for F_M takes N steps, each rounding a
    division and a product into `term` (<= 2u a step, <= 2N u on the last),
    the sum of N positive terms adds <= N u, `exp` <= 2u and the product 1u:
    (3N + 3) u at M; each downward step `(2T f + e) * (1/(2m+1))` rounds four
    times (the reciprocal, the products and the sum of two positives) and
    passes the error it is given on with weight <= 1 (all terms positive),
    so F_m is within (3N + 3 + 4 (M - m)) u;
  * from the switch, F_0 = 0.5 sqrt(pi/T) erf(sqrt(T)) is within 8u (a
    division, two square roots, the product, and erf's 2 ulp; erf's
    sensitivity to its argument is below 1e-3 there), and each upward step
    `((2m-1) f - e) * h` rounds four times (the product, the difference, h =
    0.5/T and the last product) and divides the error it is given by
    `1 - r`, r = exp(-T) / ((2m-1) F_(m-1)) the subtraction's cancellation,
    which the bound computes from reference.boys at that point.
  Every point is also held to one stated tolerance, BOYS_REL = 1e-13, which
  every bound sits under. Measured worst: printed by the test (`-s`).

SPLIT CELLS (#53), at the bottom: every `measured.SUBGRID` cell with
n_batches > 1 runs as its whole atomic launch sequence and is held to the
same J_REL pin -- the first time the atomic accumulation is checked on
hardware rather than interpreted. Nothing here is TIMED: the wall-clock
sweep over every SUBGRID cell, the roofline counter sweep and the K
wall-clock sweep live in measure/measure_kernel_device.py
(`devtools/measure.sh wallclock | roofline | wallclock-k`), which import
this file's harnesses, inputs and checks, so the timed binary is built and
checked by exactly the code below.

THE K CELLS (#205), at the very bottom: every `measured_store.K_SUBGRID`
cell's launch sequence against the host staging (`slow`: a K unit takes up
to an hour to compile), and a recorded refusal re-derived on the card
instead where the cell cannot be launched. Their section comment is the
protocol.

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

import hashlib
import math
import os
import shlex
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from conftest import VRAM_PROBE, check_vram  # noqa: F401 (tkd.check_vram, levers)
from tiers import by_l

from felizjueves.emit import cuda, hip
from felizjueves.emit._c import BOYS_DEVICE, BOYS_SWITCH_PAD
from felizjueves.ir.lower import RECYCLE
from felizjueves.mmd.kernel import (
    bra_functions,
    build_pq_kernel,
    k_spec,
    ket_functions,
    launch_shape,
    pq_spec,
    split,
)
from felizjueves.performance import asym, measured, measured_store
from felizjueves.performance.device import GFX1200, SM_86, Arch
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, theta_of
from felizjueves.spec import Spec

SUBGRID = measured.SUBGRID

# Every class the generator writes by default: L = 0..16.
LS = tuple(range(0, 17))
# The PR gate (kernel-tier.sh without --full) runs L <= FAST_L_MAX; every cell
# above it, and so every subgrid cell (L >= 7), is `slow` (tiers.py).
L_PARAMS = [by_l(L, L, id=str(L)) for L in LS]
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

# The Boys harness: fj_boys<M> exactly as emit/_c.py writes it, one thread
# per T, every order 0..M of every top M in BOYS_TOPS. Input: n | T[n];
# output, top by top, F[n][M + 1].
BOYS_HARNESS = r"""
{prologue}
{portable}
{boys}
template <int M>
__global__ void fj_boys_grid(int n, const double* T, double* F) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        double f[M + 1];
        fj_boys<M>(T[i], f);
        for (int m = 0; m <= M; ++m) F[(size_t)i * (M + 1) + m] = f[m];
    }}
}}

int main(int argc, char** argv) {{
    if (argc != 3) return 2;
    std::vector<double> in;
    if (!slurp(argv[1], in) || in.empty()) return 2;
    const int n = (int)in[0];
    if (fj_vram_probe((size_t)n * ({width} + 1) * sizeof(double) + 16,
                      {{(const void*)fj_boys_grid<0>}})) return 0;
    double* dT = upload(in.data() + 1, n);
    if (!dT) return 1;
    std::vector<double> out;
{launches}
    return spill(argv[2], out) ? 0 : 1;
}}
"""
BOYS_LAUNCH = """\
    {{
        double* dF = nullptr;
        CHECK(devMalloc(&dF, (size_t)n * {w} * sizeof(double)));
        fj_boys_grid<{M}><<<(n + 63) / 64, 64>>>(n, dT, dF);
        CHECK(devGetLastError());
        CHECK(devSynchronize());
        std::vector<double> F((size_t)n * {w});
        CHECK(devMemcpy(F.data(), dF, F.size() * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), F.begin(), F.end());
    }}"""

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
@pytest.mark.parametrize("L", L_PARAMS)
def test_cuda_j_p_matches_the_reference(L, request, compile_device_source,
                                        run_device_binary, tmp_path) -> None:
    """The emitted CUDA kernel's J_p on the RTX 3080 agrees with the oracle."""
    _run_pq("cuda", L, request, compile_device_source, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("L", L_PARAMS)
def test_hip_j_p_matches_the_reference(L, request, compile_device_source,
                                       run_device_binary, tmp_path) -> None:
    """The emitted HIP kernel's J_p on the RX 9060 XT agrees with the oracle."""
    _run_pq("hip", L, request, compile_device_source, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# fj_boys on the card
# ---------------------------------------------------------------------------

BOYS_TOPS = range(0, 17)  # a class's fundamentals run m = 0..L, L <= 16
BOYS_REL = 1e-13  # the stated tolerance; every per-point bound sits under it
_SWITCHES = [float(M + BOYS_SWITCH_PAD) for M in BOYS_TOPS]
BOYS_T = sorted({
    0.0, 1e-300, 1e-12, 1e-6, 1e-3, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0,
    50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 400.0, 500.0,
    # each switch's neighbourhood, down to the adjacent doubles
    *(t for sw in _SWITCHES for t in (
        sw - 0.5, math.nextafter(sw, 0.0), sw, math.nextafter(sw, math.inf),
        sw + 0.5)),
    # the old narrow routine's overflow, the four-H2 chain (#88), every V
    # fixture's reach (H2O/STO-3G 858, 6-31G 3.6e4, cc-pVDZ 7.7e4, cc-pVTZ
    # 1.25e5), and reference.boys's T_MAX
    700.0, 709.0, 710.0, 858.0, 1289.0, 2000.0, 3.6e4, 7.7e4, 1.25e5, 1e6,
    *np.geomspace(500.0, 1e6, 40).tolist(),
    *np.random.default_rng(28).uniform(0.0, 500.0, 40).tolist(),
    *np.random.default_rng(29).uniform(0.0, 3.0, 20).tolist(),
    *np.random.default_rng(301).uniform(0.0, 30.0, 40).tolist(),
    *np.random.default_rng(302).uniform(0.0, 1e6, 20).tolist(),
})


def series_length(M: int, T: float) -> int:
    """How many steps BOYS_DEVICE's series takes for F_M(T): the same loop."""
    term = 1.0 / (2 * M + 1)
    s = term
    for k in range(1, 4096):
        term = term * ((2.0 * T) / (2 * M + 2 * k + 1))
        if s + term == s:
            return k
        s = s + term
    return 4096


def boys_bound(M: int, T: float) -> list[float]:
    """The per-order forward error bound of `fj_boys<M>(T)`, F_0..F_M
    (module docstring), reference.boys's own error included."""
    if T < M + BOYS_SWITCH_PAD:
        top = 3 * series_length(M, T) + 3
        return [(top + 4 * (M - m)) * U + BOYS_REF_ERR for m in range(M + 1)]
    err = [8 * U]
    e = math.exp(-T)
    for m in range(1, M + 1):
        r = e / ((2 * m - 1) * boys(m - 1, T))
        assert r < 0.5, (M, T, m, r)  # the switch's whole point
        err.append((err[-1] + 2 * U) / (1 - r) + 2 * U)
    return [x + BOYS_REF_ERR for x in err]


def _run_boys(backend: str, request, compile_device_source, run_device_binary,
              tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    prologue = "#include <hip/hip_runtime.h>" if backend == "hip" else ""
    launches = "\n".join(BOYS_LAUNCH.format(M=M, w=M + 1) for M in BOYS_TOPS)
    binary = compile_device_source(tc, BOYS_HARNESS.format(
        prologue=prologue, portable=PORTABLE, boys=BOYS_DEVICE,
        launches=launches, width=sum(M + 1 for M in BOYS_TOPS)))
    src, dst = tmp_path / "boys.in", tmp_path / "boys.out"
    np.concatenate([[float(len(BOYS_T))], BOYS_T]).tofile(src)
    run_device_binary(binary, str(src), str(dst), vram=backend)
    got = np.fromfile(dst, dtype=np.float64)
    assert got.shape == (len(BOYS_T) * sum(M + 1 for M in BOYS_TOPS),)
    worst = {"series": (0.0, None), "upward": (0.0, None)}
    bad = []
    at = 0
    for M in BOYS_TOPS:
        for T in BOYS_T:
            bounds = boys_bound(M, T)
            branch = "series" if T < M + BOYS_SWITCH_PAD else "upward"
            for m in range(M + 1):
                g, ref = got[at + m], boys(m, T)
                rel = abs(g - ref) / ref
                assert bounds[m] < BOYS_REL
                if not (math.isfinite(g) and rel <= bounds[m]):
                    bad.append((M, m, T, g, rel, bounds[m]))
                if rel > worst[branch][0]:
                    worst[branch] = (rel, (M, m, T))
            at += M + 1
    print(f"\n{backend} fj_boys<M>: M in 0..16, {len(BOYS_T)} T in [0, 1e6]; "
          f"worst rel err {worst['series'][0]:.2e} at (M, m, T) = "
          f"{worst['series'][1]} below the switch, {worst['upward'][0]:.2e} "
          f"at {worst['upward'][1]} from it")
    assert not bad, f"fj_boys outside its forward error bound at {bad[:5]}"


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_fj_boys_matches_reference_boys(request, compile_device_source,
                                             run_device_binary, tmp_path) -> None:
    """The emitted Boys template on the RTX 3080, T across [0, 1e6] (#301)."""
    _run_boys("cuda", request, compile_device_source, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_fj_boys_matches_reference_boys(request, compile_device_source,
                                            run_device_binary, tmp_path) -> None:
    """The emitted Boys template on the RX 9060 XT, T across [0, 1e6] (#301)."""
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
# THE TIMING (measure/measure_kernel_device.py) does not care about any of
# that: it times the sequence and never compares its J against anything but
# itself.

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


SPLIT_CELLS = [by_l(s, s.total_am, id=f"L{s.total_am}-b{s.n_batches}")
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

RECYCLE_CELLS = [by_l(s, s.total_am, id=f"L{s.total_am}-b{s.n_batches}-recycle")
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


# --- the sieve arm (#162): the same pin, the unsieved plan ----------------------
#
# `no-sieve` turns Alg. 3's sieve off in the plan layer, so the builder's
# kernel itself differs (a PLAN arm, measured_store's THE ARMS): more
# intermediates, the same targets, the same Boys values and contraction.
# It is held to the SAME J_REL pin through the same launch sequence. Every
# cell of the A/B (`measured.ab_cells`: the whole SUBGRID plus #162's L = 11),
# on both cards, before any of them is timed.

NO_SIEVE_CELLS = [by_l(s, s.total_am, id=f"L{s.total_am}-b{s.n_batches}-no-sieve")
                  for s in measured_store.SIEVE_CELLS]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", NO_SIEVE_CELLS)
def test_cuda_unsieved_j_p_matches_the_reference(spec, request,
                                                 compile_device_source,
                                                 run_device_binary,
                                                 tmp_path) -> None:
    """A no-sieve cell's launch sequence on the RTX 3080, against the oracle."""
    _run_batched("cuda", spec, request, compile_device_source, run_device_binary,
                 tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", NO_SIEVE_CELLS)
def test_hip_unsieved_j_p_matches_the_reference(spec, request,
                                                compile_device_source,
                                                run_device_binary,
                                                tmp_path) -> None:
    """A no-sieve cell's launch sequence on the RX 9060 XT, against the oracle."""
    _run_batched("hip", spec, request, compile_device_source, run_device_binary,
                 tmp_path)


# --- the tuned arm (#163): the same pin, the searched plan ----------------------
#
# `tuned` builds the searched board's recorded cut and seed order (#161, a PLAN
# arm): a different DAG of the same class, the same targets, Boys values and
# contraction, so it is held to the SAME J_REL pin through the same launch
# sequence. Every measurable cell of the A/B (`measured_store.TUNED_CELLS`:
# the subgrid less its four 12-batch cells, which have no recorded choice and
# are `NOT_MEASURABLE`), on both cards, before any of them is timed.

TUNED_CELLS = [by_l(s, s.total_am, id=f"L{s.total_am}-b{s.n_batches}-tuned")
               for s in measured_store.TUNED_CELLS]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", TUNED_CELLS)
def test_cuda_tuned_j_p_matches_the_reference(spec, request,
                                              compile_device_source,
                                              run_device_binary,
                                              tmp_path) -> None:
    """A tuned cell's launch sequence on the RTX 3080, against the oracle."""
    _run_batched("cuda", spec, request, compile_device_source, run_device_binary,
                 tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", TUNED_CELLS)
def test_hip_tuned_j_p_matches_the_reference(spec, request,
                                             compile_device_source,
                                             run_device_binary,
                                             tmp_path) -> None:
    """A tuned cell's launch sequence on the RX 9060 XT, against the oracle."""
    _run_batched("hip", spec, request, compile_device_source, run_device_binary,
                 tmp_path)


# --- the asymmetric split (#113): the same pin, a batch list the generator ----
# --- refuses -------------------------------------------------------------------
#
# `asym.AsymCell` builds an asymmetric batch list (felizjueves.performance.asym's
# docstring) through the same `_pq_kernel` + `atomize` as a split SUBGRID
# cell, and its batches partition the class (checked by the study itself), so
# its launch sequence is held to the SAME J_REL pin -- the batched comparison's
# argument above holds for any partition with n_b <= 9. Every study cell, on
# both cards, before any of them is timed.

ASYM_CELLS = [by_l(c, c.L, id=f"L{c.L}-b{c.n_batches}-asym")
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


# config.sh's flag set (NVCC_FLAGS / HIPCC_FLAGS) rather than conftest's: the
# K cells compile with it, and so does every measure/ sweep that times or
# profiles a binary these tests check -- the flags its measured rows were
# recorded under.
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


# --- the K cells (#205) -------------------------------------------------------
#
# `measured_store.K_SUBGRID`, the J subgrid's exchange twins, on the card:
# every cell's launch sequence (`kernel_k.build_k_kernels`, the partial-sum
# form) checked against the host's staged contraction. (The wall-clock sweep
# that times each K cell BESIDE its J twin -- the J subgrid cell of the same
# L and n_batches -- in one session, as #55 times a lever beside its literal
# control, is measure/measure_kernel_device.py's, `devtools/measure.sh
# wallclock-k`; it builds and checks each cell with the helpers below.)
#
# HOW A K CELL IS BUILT. A K batch kernel is fully unrolled (kernel_k: every
# stage-2 term is a statement), so one translation unit runs to 0.8 MB of
# source at L = 7 and 187 MB at L = 16: the J harness's one-TU build would
# put a whole cell through one compiler process. Here each batch kernel is
# its own object (`devcache.compile_unit(mode="-c")`, config.sh's flag set,
# FJ_K_JOBS at a time, cached), and a small main (K_HARNESS) declares them
# `extern "C"` and launches them -- each kernel's host stub registers it, so
# no relocatable device code is needed on either vendor. The text compiled
# is `measured.cell_kernels(spec)` emitted, hash for hash the row's
# `source_sha256` (measured.cell_sha256).
#
# THE LAUNCH IS kernel_k's PINNED MAPPING, UNCHANGED: blockDim (1, lanes),
# one warp per bra record, `Ksh` sized at launch -- each batch kernel's own
# declaration (`measured.jsh_slots`) x lanes x 8 bytes -- and NO opt-in to a
# larger dynamic shared allocation (the J harness makes none either). A cell
# whose largest batch kernel asks more than the card's per-block shared
# memory (`K_HARNESS limits`, read off the card) cannot be launched as
# pinned: that is a REFUSAL, recorded with its mechanism
# (`measured_store.K_REFUSED`), never a launch error read as a number.
#
# THE INPUTS. A K launch here is one synthetic `b` shell (dd = 0) and one
# density row (n_dc = 1, dc = 0): the kernel's arithmetic does not depend on
# which shells the records came from. P, zeta, Ip, Q, eta and Iq are
# `_inputs`' -- the very values the J twin gets for that L and regime -- and
# each record's two centres are placed so that P lies between them (A = P +
# (c/zeta) u, C = P - (a/zeta) u with a + c = zeta), so the undressed `E`'s
# seed exp(zeta XPA XPC) is at most 1, as it is for any real pair. Kets are
# sorted descending by Iq alone (kernel_k.SORTED_BY_IQ).
#
# THE CHECK, K_REL = test_ao_k.KJ_TOL = 1e-12 of the quartet scale,
# inherited: per bra record, `ao.k._k_quartet` (the host staging) on
# `reference.pq` (the direct recursion: no MILD, DAG, sieve or schedule),
# summed over the ket records, against the kernel's row -- elementwise,
# |K - X| <= K_REL * S, S the host's |.| twin. The same pin the K build holds
# against libcint (test_k_ondevice.py), so it is not loosened for high L.

K_SUBGRID = measured_store.K_SUBGRID
K_REL = 1e-12  # test_ao_k.KJ_TOL (module section comment)
K_JOBS = "FJ_K_JOBS"  # parallel unit compiles (default: every core)
K_COMPILE_TIMEOUT_S = 6 * 3600  # per unit: an unrolled L = 13 unit is slow
# (wide, tau_quantile, n_bra, n_ket): as CASES, kets > 32 lanes and not a
# multiple of them; two bra records, because the host staging is pure Python
# and a high-L quartet costs it seconds.
K_CASES = {
    "near": (False, None, 2, 37),
    "wide-screened": (True, 0.4, 2, 37),
}
K_SIGNATURE = (
    "int n_bra, int n_ket, int n_dc, double tau, double maxD, const double* P, "
    "const double* zeta_ac, const double* XPA, const double* XPC, "
    "const double* s_ac, const double* Ip, const int* dc, const double* Q, "
    "const double* eta, const double* XQB, const double* XQD, "
    "const double* s_bd, const double* Iq, const int* dd, const double* Dcd, "
    "double* K")

K_HARNESS = r"""
{portable}
#include <chrono>
#include <cstring>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devEvent_t           hipEvent_t
#define devEventCreate       hipEventCreate
#define devEventRecord(e)    hipEventRecord(e, 0)
#define devEventElapsedTime  hipEventElapsedTime
#define devMemset            hipMemset
#define devAttrSmem          hipDeviceAttributeMaxSharedMemoryPerBlock
#define devAttrSmemOptin     hipDeviceAttributeSharedMemPerBlockOptin
#else
#define devEvent_t           cudaEvent_t
#define devEventCreate       cudaEventCreate
#define devEventRecord(e)    cudaEventRecord(e, 0)
#define devEventElapsedTime  cudaEventElapsedTime
#define devMemset            cudaMemset
#define devAttrSmem          cudaDevAttrMaxSharedMemoryPerBlock
#define devAttrSmemOptin     cudaDevAttrMaxSharedMemoryPerBlockOptin
#endif
{decls}

static int* upload_ints(const double* h, size_t n) {{
    std::vector<int> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (int)h[i];
    int* d = nullptr;
    if (devMalloc(&d, n * sizeof(int) + 8) != devSuccess) return nullptr;
    if (n && devMemcpy(d, v.data(), n * sizeof(int), devMemcpyH2D) != devSuccess)
        return nullptr;
    return d;
}}

// usage: `limits` prints the card's per-block shared memory (default and
// opt-in maximum); `in out warmup reps` runs the cell as BATCHED_HARNESS
// does, K in place of J: K of the first timed sequence, then of the last.
int main(int argc, char** argv) {{
    if (argc == 2 && std::strcmp(argv[1], "limits") == 0) {{
        int smem = 0, optin = 0;
        CHECK(devDeviceGetAttribute(&smem, devAttrSmem, 0));
        CHECK(devDeviceGetAttribute(&optin, devAttrSmemOptin, 0));
        std::printf("smem_per_block %d optin %d\n", smem, optin);
        return 0;
    }}
    if (argc != 5) {{ std::fprintf(stderr, "usage: in out warmup reps\n"); return 2; }}
    const int warmup = std::atoi(argv[3]), reps = std::atoi(argv[4]);
    std::vector<double> in;
    if (!slurp(argv[1], in) || in.size() < 6 || reps < 1 || warmup < 0) {{
        std::fprintf(stderr, "bad input\n");
        return 2;
    }}
    const int n_bra = (int)in[0], n_ket = (int)in[1], n_dc = (int)in[2],
              lanes = (int)in[5];
    const double tau = in[3], maxD = in[4];
    const size_t nb = (size_t)n_bra, nk = (size_t)n_ket;
    // bra: P XPA XPC (3n), zeta s_ac Ip (n), dc (n); ket alike; then Dcd
    const size_t sizes[] = {{nb * 3, nb, nb * 3, nb * 3, nb, nb, nb,
                            nk * 3, nk, nk * 3, nk * 3, nk, nk, nk,
                            (size_t)n_dc * {n_cd}}};
    const size_t jbytes = nb * {n_ab} * sizeof(double);
    if (fj_vram_probe(in.size() * sizeof(double) + 64 + jbytes,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data() + 6;
    double* d[15];
    int* di[15] = {{nullptr}};
    for (int i = 0; i < 15; ++i) {{
        if (i == 6 || i == 13) {{
            di[i] = upload_ints(h, sizes[i]);
            if (!di[i]) {{ std::fprintf(stderr, "upload %d failed\n", i); return 1; }}
            d[i] = nullptr;
        }} else {{
            d[i] = upload(h, sizes[i]);
            if (!d[i]) {{ std::fprintf(stderr, "upload %d failed\n", i); return 1; }}
        }}
        h += sizes[i];
    }}
    if ((size_t)(h - in.data()) != in.size()) {{
        std::fprintf(stderr, "size mismatch\n");
        return 2;
    }}
    double* dK = nullptr;
    CHECK(devMalloc(&dK, jbytes));
    const int NB = {n_batches};
    devEvent_t ev[NB + 1];
    for (int i = 0; i <= NB; ++i) CHECK(devEventCreate(&ev[i]));
    std::vector<double> first(nb * {n_ab}), last(first.size());
    for (int r = 0; r < warmup + reps; ++r) {{
        CHECK(devMemset(dK, 0, jbytes));  // the caller zeroes K: off the clock
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
        CHECK(devMemcpy(dst.data(), dK, jbytes, devMemcpyD2H));
    }}
    if (reps == 1) last = first;
    std::vector<double> both(first);
    both.insert(both.end(), last.begin(), last.end());
    if (!spill(argv[2], both)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    return 0;
}}
"""

K_LAUNCH = ("        {name}<<<dim3(n_bra), dim3(1, lanes), "
            "(size_t){slots} * lanes * sizeof(double)>>>(\n"
            "            n_bra, n_ket, n_dc, tau, maxD, d[0], d[1], d[2], d[3], "
            "d[4], d[5], di[6],\n"
            "            d[7], d[8], d[9], d[10], d[11], d[12], di[13], d[14], "
            "dK);\n"
            "        CHECK(devEventRecord(ev[{next}]));")


def _k_dims(spec: Spec) -> tuple[int, int]:
    """(n_a n_b, n_c n_d): K's row length and D_cd's, of a K class."""
    def n(l: int) -> int:  # noqa: E741
        return (l + 1) * (l + 2) // 2
    la, lc, lb, ld = spec.am
    return n(la) * n(lb), n(lc) * n(ld)


@dataclass(frozen=True)
class _KCell:
    """One built K cell: its harness binary, batch kernels' names and shared
    slots in launch order, and the cell's source hash."""

    spec: Spec
    binary: Path
    names: tuple[str, ...]
    slots: tuple[int, ...]
    sha: str
    compile_s: float


def _k_build(backend: str, spec: Spec, tc, workdir: Path) -> _KCell:
    """Each batch kernel its own object, FJ_K_JOBS at a time; one main."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    import devcache

    from felizjueves.kernel_k import PARAMS
    assert PARAMS == tuple(
        a.split()[-1].lstrip("*") for a in K_SIGNATURE.split(", ")), PARAMS
    emit, arch, ext = BACKEND[backend]
    kernels = measured.cell_kernels(spec)
    texts = [emit(k, arch) for k in kernels]
    flags = tc.compile_flags()
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def obj(i: int) -> Path:
        out = workdir / f"{kernels[i].name}.o"
        got = devcache.compile_unit(
            tc.path, flags, f"{kernels[i].name}.{ext}", texts[i], {}, out,
            mode="-c", timeout=K_COMPILE_TIMEOUT_S)
        assert got.returncode == 0, (
            f"{tc.name} failed on {kernels[i].name}:\n{got.diagnostics[-4000:]}")
        return out

    jobs = int(os.environ.get(K_JOBS, "0") or 0) or (os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        objs = list(pool.map(obj, range(len(kernels))))
    n_ab, n_cd = _k_dims(spec)
    slots = tuple(measured.jsh_slots(k) for k in kernels)
    src = K_HARNESS.format(
        portable=PORTABLE, n_cd=n_cd, n_ab=n_ab, n_batches=len(kernels),
        decls="\n".join(f'extern "C" __global__ void {k.name}({K_SIGNATURE});'
                        for k in kernels),
        launches="\n".join(K_LAUNCH.format(name=k.name, slots=s, next=i + 1)
                           for i, (k, s) in enumerate(zip(kernels, slots,
                                                          strict=True))),
        vram_kernels=", ".join(f"(const void*){k.name}" for k in kernels))
    main_o = workdir / "k_main.o"
    got = devcache.compile_unit(tc.path, flags, f"k_main.{ext}", src, {},
                                main_o, mode="-c", timeout=600)
    assert got.returncode == 0, f"{tc.name} failed on k_main:\n{got.diagnostics}"
    binary = workdir / "kcell"
    done = subprocess.run(
        [tc.path, *devcache._link_flags(flags), str(main_o),
         *(str(o) for o in objs), "-o", str(binary)],
        capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"link failed:\n{done.stdout}{done.stderr}"
    return _KCell(spec, binary, tuple(k.name for k in kernels), slots,
                  hashlib.sha256("".join(texts).encode()).hexdigest(),
                  time.perf_counter() - t0)


def _k_limits(binary: Path, args: tuple[str, ...] = ("limits",)
              ) -> tuple[int, int]:
    """The card's per-block shared memory: (default, opt-in maximum)."""
    done = subprocess.run([str(binary), *args], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    f = done.stdout.split()
    assert (f[0], f[2]) == ("smem_per_block", "optin"), done.stdout
    return int(f[1]), int(f[3])


def k_shared_need(cell: _KCell, lanes: int) -> int:
    """The largest dynamic shared allocation the cell's launches ask for."""
    return max(cell.slots) * lanes * 8


def _k_inputs(spec: Spec, a: dict, seed: int) -> dict:
    """K's arguments from a J input `a` (module section comment)."""
    rng = np.random.default_rng(seed)
    _, n_cd = _k_dims(spec)
    srt = np.argsort(-a["Iq"], kind="stable")

    def side(X, z):
        n = X.shape[0]
        f = rng.uniform(0.25, 0.75, n)
        u = rng.normal(size=(n, 3))
        u *= (rng.uniform(0.5, 2.0, n) / np.linalg.norm(u, axis=1))[:, None]
        e1, e2 = f * z, (1 - f) * z
        A = X + (e2 / z)[:, None] * u
        B = X - (e1 / z)[:, None] * u
        return A, B, X - A, X - B, rng.uniform(0.5, 1.0, n)

    A, C, XPA, XPC, s_ac = side(a["P"], a["zeta"])
    Q, eta, Iq = a["Q"][srt], a["eta"][srt], a["Iq"][srt]
    B, D, XQB, XQD, s_bd = side(Q, eta)
    return dict(n_bra=a["P"].shape[0], n_ket=Q.shape[0], tau=a["tau"],
                maxD=1.0, P=a["P"], zeta=a["zeta"], XPA=XPA, XPC=XPC,
                s_ac=s_ac, Ip=a["Ip"], A=A, C=C, Q=Q, eta=eta, XQB=XQB,
                XQD=XQD, s_bd=s_bd, Iq=Iq, B=B, D=D,
                Dcd=rng.uniform(-1.0, 1.0, (1, n_cd)))


def _k_pack(k: dict, lanes: int) -> np.ndarray:
    nb, nk = k["n_bra"], k["n_ket"]
    head = [nb, nk, 1, k["tau"], k["maxD"], lanes]
    body = [k["P"], k["zeta"], k["XPA"], k["XPC"], k["s_ac"], k["Ip"],
            np.zeros(nb), k["Q"], k["eta"], k["XQB"], k["XQD"], k["s_bd"],
            k["Iq"], np.zeros(nk), k["Dcd"]]
    return np.concatenate([np.array(head, dtype=np.float64),
                           *(np.asarray(x, dtype=np.float64).ravel()
                             for x in body)])


def _k_timing_tau(k: dict, quantile: float | None) -> float:
    if quantile is None:
        return 0.0
    return float(np.quantile(k["Ip"][0] * k["Iq"] * k["maxD"], quantile))


@cache
def _k_case(L: int, case: str):
    """A K_CASES input of class L and its host reference (want, scale, T
    max) -- shared by every n_batches of L, as `_case` is for J."""
    from felizjueves.ao.density import EPair
    from felizjueves.ao.k import _k_quartet
    from felizjueves.mmd.kernel import k_spec

    spec = k_spec(L)
    wide, tq, n_bra, n_ket = K_CASES[case]
    a = _inputs(L, wide=wide, tau_quantile=None, seed=1000 * L + len(case),
                n_bra=n_bra, n_ket=n_ket)
    k = _k_inputs(spec, a, seed=7000 + L + len(case))
    k["tau"] = _k_timing_tau(k, tq)
    la, lc, lb, ld = spec.am
    targets = hermite_targets(la + lc + lb + ld)
    memo: dict = {}

    def pq(p, q, PQv, theta):
        key = (PQv, theta)
        if key not in memo:
            memo.clear()
            memo[key] = {r: one_centre(r, 0, PQv, theta) for r in targets}
        r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
        return (-1.0 if sum(q) % 2 else 1.0) * memo[key][r]

    dcd = [float(x) for x in k["Dcd"][0]]
    n_ab, _ = _k_dims(spec)
    want, scale = np.zeros((n_bra, n_ab)), np.zeros((n_bra, n_ab))
    t_max = 0.0

    def vec(x) -> tuple[float, float, float]:
        return (float(x[0]), float(x[1]), float(x[2]))

    for i in range(n_bra):
        bra = EPair(la, lc, vec(k["A"][i]), vec(k["C"][i]),
                    ((vec(k["P"][i]), float(k["zeta"][i]), float(k["s_ac"][i])),))
        for j in range(n_ket):
            if k["Ip"][i] * k["Iq"][j] * k["maxD"] < k["tau"]:
                continue  # a continue: this side does not rely on the sort
            ket = EPair(lb, ld, vec(k["B"][j]), vec(k["D"][j]),
                        ((vec(k["Q"][j]), float(k["eta"][j]),
                          float(k["s_bd"][j])),))
            th = theta_of(float(k["zeta"][i]), float(k["eta"][j]))
            PQ = k["P"][i] - k["Q"][j]
            t_max = max(t_max, th * th * float(PQ @ PQ))
            X, S = _k_quartet(bra, ket, dcd, pq=pq, want_scale=True)
            want[i] += np.array(X).ravel()
            scale[i] += np.array(S).ravel()
    return k, want, scale, t_max


def _k_check(backend: str, cell: _KCell, run_device_binary,
             tmp_path: Path) -> list[tuple[str, float, float]]:
    """Every K_CASES input through one cell's launch sequence, against the
    host staging at K_REL of the quartet scale, elementwise."""
    arch: Arch = BACKEND[backend][1]
    L = cell.spec.total_am
    out = []
    for case in K_CASES:
        k, want, scale, t_max = _k_case(L, case)
        src, dst = tmp_path / f"k-{case}.in", tmp_path / f"k-{case}.out"
        _k_pack(k, arch.wavefront).tofile(src)
        run_device_binary(cell.binary, str(src), str(dst), "0", "1",
                          vram=backend)
        got = np.fromfile(dst, dtype=np.float64).reshape(2, k["n_bra"], -1)[0]
        assert got.shape == want.shape
        assert np.all(np.isfinite(got)), f"{case}: non-finite K"
        bad = np.abs(got - want) > K_REL * scale
        err = float(np.max(np.abs(got - want) / np.where(scale > 0, scale, 1)))
        assert not bad.any(), (
            f"{cell.spec.key()} {case} on {arch.name}: max |K - X| / S = "
            f"{err:.3e} > {K_REL:.0e} (T up to {t_max:.1f})")
        out.append((case, err, t_max))
    print(f"\n{backend} {arch.name} {cell.spec.key()}: " + ", ".join(
        f"{c} err/S {e:.2e} (T<={t:.0f})" for c, e, t in out), flush=True)
    return out


K_LIMITS_PROBE = r"""
{portable}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devAttrSmem          hipDeviceAttributeMaxSharedMemoryPerBlock
#define devAttrSmemOptin     hipDeviceAttributeSharedMemPerBlockOptin
#else
#define devAttrSmem          cudaDevAttrMaxSharedMemoryPerBlock
#define devAttrSmemOptin     cudaDevAttrMaxSharedMemoryPerBlockOptin
#endif
int main() {{
    int smem = 0, optin = 0;
    CHECK(devDeviceGetAttribute(&smem, devAttrSmem, 0));
    CHECK(devDeviceGetAttribute(&optin, devAttrSmemOptin, 0));
    std::printf("smem_per_block %d optin %d\n", smem, optin);
    return 0;
}}
"""


def _run_k(backend: str, spec: Spec, request, compile_device_source,
           run_device_binary, tmp_path: Path) -> None:
    arch: Arch = BACKEND[backend][1]
    base = _toolchain(backend, request)
    tc = _Configured(base.name, base.path, base.arch, base.ext)
    untimed = measured_store.K_UNTIMED.get(arch.name, {}).get(spec)
    if untimed is not None:
        # the recorded refusal, re-derived: the cell's own kernels' Ksh
        # (built, not compiled) against the card's limit, read off the card
        probe = compile_device_source(tc, K_LIMITS_PROBE.format(
            portable=PORTABLE))
        smem, optin = _k_limits(probe, args=())
        assert (smem, optin) == measured_store.K_SMEM_PER_BLOCK[arch.name]
        slots = max(measured.jsh_slots(k) for k in measured.cell_kernels(spec))
        assert slots == measured_store.K_MAX_SLOTS[
            (spec.total_am, spec.n_batches)]
        assert slots * arch.wavefront * 8 > smem
        pytest.skip(f"NOT LAUNCHABLE, recorded (measured_store.K_UNTIMED), "
                    f"re-derived on the card: {untimed}")
    refused = measured_store.K_REFUSED.get(arch.name, {}).get(spec)
    if refused is not None:
        pytest.skip(f"REFUSED, recorded (measured_store.K_REFUSED): {refused}")
    cell = _k_build(backend, spec, tc, tmp_path / "build")
    smem, optin = _k_limits(cell.binary)
    need = k_shared_need(cell, arch.wavefront)
    assert need <= smem, (
        f"{spec.key()} asks {need} B of shared per block, the card gives "
        f"{smem} (opt-in {optin}) and the cell is not in K_UNTIMED")
    _k_check(backend, cell, run_device_binary, tmp_path)


K_CELLS = [by_l(s, s.total_am, id=f"L{s.total_am}-b{s.n_batches}-k")
           for s in K_SUBGRID]  # every K_SUBGRID L is above FAST_L_MAX


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", K_CELLS)
def test_cuda_k_cell_matches_the_host(spec, request, compile_device_source,
                                      run_device_binary, tmp_path) -> None:
    """A K_SUBGRID cell's launch sequence on the RTX 3080, against the host
    -- or, for a recorded refusal, the refusal re-derived on the card."""
    _run_k("cuda", spec, request, compile_device_source, run_device_binary,
           tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", K_CELLS)
def test_hip_k_cell_matches_the_host(spec, request, compile_device_source,
                                     run_device_binary, tmp_path) -> None:
    """A K_SUBGRID cell's launch sequence on the RX 9060 XT, against the host
    -- or, for a recorded refusal, the refusal re-derived on the card."""
    _run_k("hip", spec, request, compile_device_source, run_device_binary,
           tmp_path)


# CORRECTNESS-ONLY K CELLS: classes outside K_SUBGRID, checked on the card by
# the same `_run_k` (same inputs, same K_REL pin) and recorded nowhere -- no
# row, no refusal, no timing. L = 8 is (dd|dd) in k_spec's Hermite order,
# the top class a cc-pVDZ molecule reaches (#231), and the first K check on a
# card above #206's L <= 2. Each batch count is its own cell.
K_CHECK_CELLS = [pytest.param(k_spec(8, n_batches=b),
                              id=f"L8-b{b}-k", marks=[pytest.mark.slow])
                 for b in (1, 3, 6, 9, 12)]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", K_CHECK_CELLS)
def test_cuda_k_check_cell_matches_the_host(spec, request,
                                            compile_device_source,
                                            run_device_binary,
                                            tmp_path) -> None:
    """A correctness-only K cell's launch sequence on the RTX 3080, against
    the host staging."""
    _run_k("cuda", spec, request, compile_device_source, run_device_binary,
           tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", K_CHECK_CELLS)
def test_hip_k_check_cell_matches_the_host(spec, request,
                                           compile_device_source,
                                           run_device_binary,
                                           tmp_path) -> None:
    """A correctness-only K cell's launch sequence on the RX 9060 XT, against
    the host staging."""
    _run_k("hip", spec, request, compile_device_source, run_device_binary,
           tmp_path)
