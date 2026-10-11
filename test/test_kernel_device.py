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

import math
import os
from pathlib import Path

import numpy as np
import pytest

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
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, theta_of

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
        run_device_binary(binary, str(src), str(dst))
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
    run_device_binary(binary, str(src), str(dst))
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
