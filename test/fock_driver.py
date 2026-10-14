"""The fused Fock kernel ON A CARD (#288) -- harness.

    record(plan, basis, D, *, threads=)            -> (list[FRec], FWalk)
    fock_program(kernels, backend, workdir)        -> Program
    fock_stage(recs, kernels)                      -> np.ndarray
    run_on_card(prog, binary, recs, n_ao, run, *, timing=) -> [run] F
    interpret(recs, n_ao)                          -> F (not closed)
    synthetic_launch(cls, *, side, seed)           -> (FRec, basis)

THE LAUNCHES ARE `walk_f`'s, RECORDED. `record` walks `launch_f`'s plan with
`walk_f` (#286) and an injected `run` that only RECORDS each call -- the
kernel, the grid, the block and every argument by name, exactly what the
interpreted build is handed -- and returns the running `F` unchanged. So
the card replays the very launches the interpreted walk runs (the same
packed records, the same `diag`, the same `Dsh` and `D`), and nothing about
the plan, the packing or the screen is decided here. `interpret` runs those
recorded launches through `interp` (Boys from `reference/`, injected), one
after another into one `F`: walk_f's own executor, restated, so a subset of
the launches (one class) has a host reference too.

THE PROGRAM is one `main()` around a set of emitted F units (`kernel_f.
build_f_kernel`, spelt by `emit/cuda.py` / `emit/hip.py` unchanged, each in
its own namespace as test/one_electron_driver.py does), driven at RUN time
by its input: any number of launches of any of its kernels, in order, into
ONE `F`. The per-class parity runs and the wall-clock sweep therefore run
the SAME binary on different inputs -- the timed binary is the checked
binary. The input, every value a double (ints exact):

    n_launch n_sh n_ao tau
    per launch: kernel-index n_bra n_ket k_ab k_cd diag threads nd ni
    Dsh[n_sh, n_sh]  D[n_ao, n_ao]                 uploaded once, shared
    per launch: nd doubles -- the kernel's double record arrays in its own
                parameter order (bra_prim, AB if it has one, G_ab, ket_prim,
                CD if it has one, G_cd) -- then ni ints (bra_sh, bra_ao,
                ket_sh, ket_ao), each row-major

and a launch is `grid = n_bra`, `block = (threads, 1)` (`kernel_f.
launch_shape`). Every argument of a call is read off the kernel's own
parameter list, so the call cannot drift from the signature.

TWO MODES. `prog in out`: `RUNS` runs, each from a zeroed `F`, every launch,
one synchronize, the read-back -- `F` NOT closed (the host's `kernel_f.
close` is the caller's, as in walk_f). `prog in out WARMUP REPEATS`: the
same sequence WARMUP times untimed and REPEATS times timed, each from a
zeroed `F` (the zeroing and a synchronize OUTSIDE the clock), one `rep i
wall_ns W events_ns E` line per timed sequence (the host `steady_clock`
across every launch and the closing synchronize, and the GPU events
recorded around the launches), and the last two `F`s written.

THE SYNTHETIC LAUNCH (`synthetic_launch`) is the wall-clock sweep's input:
one OFF-diagonal launch of the class (`diag = 0`), `TIMING_N_BRA` bra pairs
by `TIMING_N_KET` ket pairs -- test/test_kernel_device.py's sizes for the
J rows -- of a basis of 3-primitive shells (`K_AB = K_CD = 9`) at random
centres in a cube of side `side`, `tau = 0` so no quartet is screened, a
random symmetric density. The bra and ket pairs are disjoint sets of the
class's canonical pairs; nothing is closed or compared but its finiteness
(the binary was checked on real launches first).

Harness, in `test/`: not a second emitter, no measurement decided here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plan_driver import UPLOAD_INT, Program, _unit
from test_kernel_device import BACKEND, PORTABLE

from felizjueves import kernel_f as KF
from felizjueves.ao.shell import Basis, Shell
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.reference.boys import boys
from felizjueves.walk_f import FWalk, pair_record, walk_f

THREADS = 32  # block width: kets strided over 32 lanes; not numerical
RUNS = 2  # runs per check-mode process: the run-to-run witness
SCALARS = ("n_bra", "n_ket", "k_ab", "k_cd", "n_sh", "n_ao", "diag", "tau")
DOUBLES = ("bra_prim", "AB", "G_ab", "ket_prim", "CD", "G_cd")
INTS = ("bra_sh", "bra_ao", "ket_sh", "ket_ao")
SHARED = ("Dsh", "D")
HEADER = 4  # n_launch n_sh n_ao tau
PER_LAUNCH = 9  # kernel-index n_bra n_ket k_ab k_cd diag threads nd ni
TIMING_N_BRA = 2048  # test_kernel_device.py's TIMING_N_BRA
TIMING_N_KET = 128  # and TIMING_N_KET


@dataclass(frozen=True)
class FRec:
    """One recorded launch: the kernel, its grid and block width, and every
    argument by name (arrays flat, row-major), as walk_f hands `run`."""

    kernel: Kernel
    grid: int
    threads: int
    args: Mapping[str, object]

    @property
    def name(self) -> str:
        return self.kernel.name

    def pairs(self, side: str) -> list[tuple[int, int]]:
        sh = [int(x) for x in self.args[f"{side}_sh"]]
        return list(zip(sh[::2], sh[1::2], strict=True))


def record(plan, basis: Basis, D, *, threads: int = THREADS
           ) -> tuple[list[FRec], FWalk]:
    """walk_f over `plan` with a run that records every launch and returns
    `F` unchanged (module docstring, THE LAUNCHES)."""
    recs: list[FRec] = []

    def run(kernel, grid, block, args):
        recs.append(FRec(kernel, grid, block[0], dict(args)))
        return {KF.FOCK: args[KF.FOCK]}

    w = walk_f(plan, basis, D, run=run, threads=threads)
    return recs, w


def interpret(recs: Sequence[FRec], n_ao: int) -> np.ndarray:
    """The recorded launches through the interpreter, in order, into one
    zeroed `F` (`n_ao x n_ao`, NOT closed)."""
    F = np.zeros(n_ao * n_ao)
    for r in recs:
        a = {}
        for p in r.kernel.params:
            v = F if p.name == KF.FOCK else r.args[p.name]
            if p.array:
                v = np.array(v, dtype=np.int32 if p.type is Scalar.I32
                             else float)
            a[p.name] = v
        F = Interpreter(r.kernel, launch=Launch(r.grid, (r.threads, 1), 32),
                        boys=boys).run(**a)[KF.FOCK]
    return np.asarray(F, dtype=float).reshape(n_ao, n_ao)


def _check_signature(k: Kernel) -> None:
    names = [p.name for p in k.params]
    want = [n for n in (*SCALARS, "bra_prim", "AB", "G_ab", "bra_sh",
                        "bra_ao", "ket_prim", "CD", "G_cd", "ket_sh",
                        "ket_ao", *SHARED, KF.FOCK) if n in names]
    if names != want or [p.name for p in k.params
                         if p.intent is Intent.OUT] != [KF.FOCK]:
        raise ValueError(f"{k.name}: signature {names} is not the fused F "
                         "kernel's (kernel_f.params_of, no tap)")


def _case(i: int, k: Kernel, ns: str) -> str:
    """The `switch` arm launching kernel `i`: its pointers carved off the
    launch's two buffers in parameter order, its arguments read off its own
    parameters."""
    has = {p.name for p in k.params}
    lines = [f"        case {i}: {{  // {k.name}"]
    rec = len(KF.REC)
    size = {"bra_prim": f"(size_t)nb * kab * {rec}", "AB": "(size_t)nb * 3",
            "G_ab": "(size_t)nb", "ket_prim": f"(size_t)nk * kcd * {rec}",
            "CD": "(size_t)nk * 3", "G_cd": "(size_t)nk",
            "bra_sh": "(size_t)2 * nb", "bra_ao": "(size_t)2 * nb",
            "ket_sh": "(size_t)2 * nk", "ket_ao": "(size_t)2 * nk"}
    for buf, names, ty in (("pd", DOUBLES, "double"), ("pi", INTS, "int")):
        at = "0"
        for n in names:
            if n not in has:
                continue
            lines.append(f"            const {ty}* a_{n} = {buf}[l] + {at};")
            at = f"{at} + {size[n]}"
    arg = {"n_bra": "nb", "n_ket": "nk", "k_ab": "kab", "k_cd": "kcd",
           "n_sh": "NSH", "n_ao": "NAO", "diag": "dg", "tau": "TAU",
           "Dsh": "dDsh", "D": "dD", KF.FOCK: "dF"}
    args = [arg.get(p.name, f"a_{p.name}") for p in k.params]
    lines += [
        f"            {ns}::{k.name}<<<dim3(nb), dim3(th, 1)>>>(",
        f"                {', '.join(args)});",
        "            break;",
        "        }",
    ]
    return "\n".join(lines)


def fock_program(kernels: Sequence[Kernel], backend: str, workdir: Path, *,
                 units: Mapping[str, Path] | None = None,
                 name: str = "fock") -> Program:
    """The `main()` around `kernels` (module docstring, THE PROGRAM); a
    launch names its kernel by its index here."""
    if not kernels or len({k.name for k in kernels}) != len(kernels):
        raise ValueError("fock_program takes distinct kernels")
    _, _, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    includes, paths, cases, ptrs = [], [], [], []
    for i, k in enumerate(kernels):
        _check_signature(k)
        path = _unit(k, backend, workdir, units)
        paths.append(path)
        ns = f"fj_f{i}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        cases.append(_case(i, k, ns))
        ptrs.append(f"(const void*){ns}::{k.name}")
    src = FOCK_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes),
        upload_int=UPLOAD_INT, n_k=len(kernels), runs=RUNS,
        header=HEADER, per=PER_LAUNCH, vram_kernels=", ".join(ptrs),
        cases="\n".join(cases))
    if name in {k.name for k in kernels}:
        raise ValueError(f"the program {name!r} would overwrite its unit")
    main = workdir / f"{name}.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def _global(recs: Sequence[FRec], key: str):
    vals = {tuple(np.ravel(np.asarray(r.args[key], dtype=float)))
            for r in recs}
    if len(vals) != 1:
        raise ValueError(f"the launches disagree on {key}")
    return np.array(next(iter(vals)))


def fock_stage(recs: Sequence[FRec], kernels: Sequence[Kernel]) -> np.ndarray:
    """The program's input for `recs`, each launching one of `kernels`
    (module docstring, THE PROGRAM's layout)."""
    if not recs:
        raise ValueError("no launch to stage")
    index = {k.name: i for i, k in enumerate(kernels)}
    n_sh, n_ao = int(_global(recs, "n_sh")[0]), int(_global(recs, "n_ao")[0])
    tau = float(_global(recs, "tau")[0])
    Dsh, D = _global(recs, "Dsh"), _global(recs, "D")
    assert Dsh.size == n_sh * n_sh and D.size == n_ao * n_ao
    head = [len(recs), n_sh, n_ao, tau]
    body: list[np.ndarray] = []
    for r in recs:
        if r.name not in index or r.kernel != kernels[index[r.name]]:
            raise ValueError(f"{r.name} is not one of the program's kernels")
        has = {p.name for p in r.kernel.params}
        dd = [np.ravel(np.asarray(r.args[n], dtype=float))
              for n in DOUBLES if n in has]
        ii = [np.ravel(np.asarray(r.args[n], dtype=float)) for n in INTS]
        nd, ni = sum(x.size for x in dd), sum(x.size for x in ii)
        a = r.args
        head += [index[r.name], a["n_bra"], a["n_ket"], a["k_ab"], a["k_cd"],
                 a["diag"], r.threads, nd, ni]
        assert r.grid == a["n_bra"], (r.grid, a["n_bra"])
        body += [*dd, *ii]
    return np.concatenate([np.array(head, dtype=np.float64), Dsh, D,
                           *body]).astype(np.float64)


_REP = re.compile(r"^rep (\d+) wall_ns (\d+) events_ns (\d+)$",
                  re.MULTILINE)
_DONE = re.compile(r"^fj-fock launches (\d+) runs (\d+)$", re.MULTILINE)


def reps(stdout: str) -> tuple[list[int], list[int]]:
    """The timed sequences' host wall-clock and GPU-event times, in order."""
    got = [(int(m.group(2)), int(m.group(3))) for m in _REP.finditer(stdout)]
    return [w for w, _ in got], [e for _, e in got]


def run_on_card(binary: Path, data: np.ndarray, n_ao: int, workdir: Path,
                run: Callable[..., str], *, backend: str,
                timing: tuple[int, int] | None = None,
                tag: str = "fock") -> tuple[np.ndarray, str]:
    """Run the program on `data`: `[2 or RUNS][n_ao, n_ao]` (not closed) and
    its stdout. `run` is the caller's `run_device_binary` (the VRAM probe
    first, behind the caller's card lock); in timing mode the caller has
    already probed and gated, so it passes a plain runner."""
    src, dst = workdir / f"{tag}.in", workdir / f"{tag}.out"
    data.tofile(src)
    args = [str(src), str(dst)]
    if timing is None:
        out = run(binary, *args, vram=backend)
    else:
        out = run(binary, *args, *map(str, timing))
    m = _DONE.search(out)
    assert m, out
    n = RUNS if timing is None else 2
    assert int(m.group(1)) == int(data[0]) and int(m.group(2)) == n, out
    F = np.fromfile(dst, dtype=np.float64)
    assert F.size == n * n_ao * n_ao, (F.size, n, n_ao)
    return F.reshape(n, n_ao, n_ao), out


# --- the wall-clock sweep's input -------------------------------------------

_EXPS = (3.0, 0.8, 0.25)  # 3 primitives per shell: K_AB = K_CD = 9
_COEFS = (0.15, 0.55, 0.45)


def _canonical_pairs(idx: Mapping[int, list[int]], la: int, lb: int
                     ) -> list[tuple[int, int]]:
    """Every canonical shell pair of class `(la, lb)`, `la <= lb`; on a tie
    of `l` the lower index first."""
    if la == lb:
        s = idx[la]
        return [(s[i], s[j]) for i in range(len(s)) for j in range(i, len(s))]
    return [(a, b) for a in idx[la] for b in idx[lb]]


def synthetic_basis(*, side: float, seed: int, counts=(66, 34, 4)) -> Basis:
    """Shells of `l = 0, 1, 2` (`counts` of each, 3 primitives, exponents
    jittered by up to 10%) at uniform random centres in a cube of `side`."""
    rng = np.random.default_rng(seed)
    shells = []
    for l, n in enumerate(counts):  # noqa: E741
        for _ in range(n):
            c = tuple(float(x) for x in rng.uniform(0.0, side, 3))
            e = tuple(float(x * (1.0 + 0.1 * rng.uniform())) for x in _EXPS)
            shells.append(Shell(l, c, e, _COEFS))
    return Basis(tuple(shells))


def max_boys_T(basis: Basis, side: float) -> float:
    """An upper bound on the Boys argument `T = rho |PQ|^2` of any primitive
    quartet of `basis` (centres in a cube of `side`): `rho <= p_max / 2`."""
    p_max = 2.0 * max(e for s in basis.shells for e in s.exponents)
    return p_max / 2.0 * 3.0 * side * side


def synthetic_launch(cls: tuple[int, int, int, int], *, side: float,
                     seed: int) -> tuple[FRec, Basis]:
    """The timing launch of `cls` (module docstring, THE SYNTHETIC LAUNCH)."""
    la, lb, lc, ld = cls
    basis = synthetic_basis(side=side, seed=seed)
    idx: dict[int, list[int]] = {}
    for i, s in enumerate(basis.shells):
        idx.setdefault(s.l, []).append(i)
    bra_all = _canonical_pairs(idx, la, lb)
    ket_all = _canonical_pairs(idx, lc, ld)
    bras = bra_all[:TIMING_N_BRA]
    kets = ket_all[-TIMING_N_KET:]
    assert len(bras) == TIMING_N_BRA and len(kets) == TIMING_N_KET
    assert not set(bras) & set(kets), "the bra and ket pairs must differ"

    def rec(a: int, b: int) -> KF.PairRec:
        n = len(basis.shells[a].exponents) * len(basis.shells[b].exponents)
        return pair_record(basis, a, b, G=1.0, prims=range(n))

    pb, pk = KF.pack([rec(*p) for p in bras]), KF.pack([rec(*p) for p in kets])
    rng = np.random.default_rng(seed + 1)
    r = rng.uniform(-1.0, 1.0, (basis.n_ao, basis.n_ao))
    D = (r + r.T) / 2.0
    Dsh = KF.density_blocks(D, basis.offsets,
                            [s.n_cart for s in basis.shells])
    k = KF.build_f_kernel(cls)
    every = {
        "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree, "k_cd": pk.degree,
        "n_sh": len(basis.shells), "n_ao": basis.n_ao, "diag": 0,
        "tau": 0.0, "bra_prim": pb.prim, "AB": pb.sep, "G_ab": pb.G,
        "bra_sh": pb.sh, "bra_ao": pb.ao, "ket_prim": pk.prim, "CD": pk.sep,
        "G_cd": pk.G, "ket_sh": pk.sh, "ket_ao": pk.ao, "Dsh": Dsh,
        "D": tuple(D.ravel()), KF.FOCK: (0.0,) * (basis.n_ao ** 2)}
    args = {n: every[n] for n in KF.params_of(cls)}
    grid, (threads, _) = KF.launch_shape(pb.n, THREADS)
    return FRec(k, grid, threads, args), basis


# The program. `{...}` are filled per kernel set; everything else is C++.
FOCK_PROGRAM = r"""// generated by test/fock_driver.py (#288) -- harness: the fused F.
{prologue}
#include <chrono>
{portable}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devMemset hipMemset
#define devFree   hipFree
#define devEvent_t           hipEvent_t
#define devEventCreate       hipEventCreate
#define devEventDestroy      hipEventDestroy
#define devEventRecord(e)    hipEventRecord(e, 0)
#define devEventElapsedTime  hipEventElapsedTime
#else
#define devMemset cudaMemset
#define devFree   cudaFree
#define devEvent_t           cudaEvent_t
#define devEventCreate       cudaEventCreate
#define devEventDestroy      cudaEventDestroy
#define devEventRecord(e)    cudaEventRecord(e, 0)
#define devEventElapsedTime  cudaEventElapsedTime
#endif
{includes}
{upload_int}
static const int NK = {n_k}, RUNS = {runs}, HEAD = {header}, PER = {per};

// usage: in out [warmup repeats]. `in`: n_launch n_sh n_ao tau, then per
// launch kernel-index n_bra n_ket k_ab k_cd diag threads nd ni, then
// Dsh[n_sh, n_sh] D[n_ao, n_ao], then per launch nd doubles and ni ints.
// `out`: each run's F [n_ao, n_ao], NOT closed (RUNS runs; the last two in
// timing mode).
int main(int argc, char** argv) {{
    if (argc != 3 && argc != 5) {{
        std::fprintf(stderr, "usage: in out [warmup repeats]\n");
        return 2;
    }}
    std::vector<double> in;
    if (!slurp(argv[1], in) || in.size() < (size_t)HEAD) {{
        std::fprintf(stderr, "bad input\n");
        return 2;
    }}
    const int NL = (int)in[0], NSH = (int)in[1], NAO = (int)in[2];
    const double TAU = in[3];
    if (NL < 1 || in.size() < (size_t)HEAD + (size_t)PER * NL) {{
        std::fprintf(stderr, "bad header\n");
        return 2;
    }}
    std::vector<int> kidx(NL), nbs(NL), nks(NL), kabs(NL), kcds(NL), dgs(NL),
        ths(NL);
    std::vector<size_t> nds(NL), nis(NL);
    size_t want = (size_t)HEAD + (size_t)PER * NL + (size_t)NSH * NSH
                  + (size_t)NAO * NAO;
    for (int l = 0; l < NL; ++l) {{
        const double* h = in.data() + HEAD + (size_t)PER * l;
        kidx[l] = (int)h[0]; nbs[l] = (int)h[1]; nks[l] = (int)h[2];
        kabs[l] = (int)h[3]; kcds[l] = (int)h[4]; dgs[l] = (int)h[5];
        ths[l] = (int)h[6]; nds[l] = (size_t)h[7]; nis[l] = (size_t)h[8];
        if (kidx[l] < 0 || kidx[l] >= NK) {{
            std::fprintf(stderr, "launch %d: kernel index %d\n", l, kidx[l]);
            return 2;
        }}
        want += nds[l] + nis[l];
    }}
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the launches want %zu\n",
                     in.size(), want);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double)
                          + (size_t)NAO * NAO * sizeof(double) + (size_t)64 * NL,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data() + HEAD + (size_t)PER * NL;
    double* dDsh = upload(h, (size_t)NSH * NSH);
    h += (size_t)NSH * NSH;
    double* dD = upload(h, (size_t)NAO * NAO);
    h += (size_t)NAO * NAO;
    if (!dDsh || !dD) {{ std::fprintf(stderr, "density upload failed\n"); return 1; }}
    std::vector<double*> pd(NL);
    std::vector<int*> pi(NL);
    for (int l = 0; l < NL; ++l) {{
        pd[l] = upload(h, nds[l]);
        h += nds[l];
        pi[l] = upload_int(h, nis[l]);
        h += nis[l];
        if (!pd[l] || !pi[l]) {{
            std::fprintf(stderr, "upload %d failed\n", l);
            return 1;
        }}
    }}
    double* dF = nullptr;
    CHECK(devMalloc(&dF, (size_t)NAO * NAO * sizeof(double)));
    auto sequence = [&]() -> int {{
        for (int l = 0; l < NL; ++l) {{
            const int nb = nbs[l], nk = nks[l], kab = kabs[l], kcd = kcds[l],
                      dg = dgs[l], th = ths[l];
            switch (kidx[l]) {{
{cases}
            default:
                std::fprintf(stderr, "launch %d: no kernel %d\n", l, kidx[l]);
                return 1;
            }}
            CHECK(devGetLastError());
        }}
        return 0;
    }};
    std::vector<double> F((size_t)NAO * NAO), out;
    const bool timing = argc == 5;
    const int warmup = timing ? std::atoi(argv[3]) : 0;
    const int repeats = timing ? std::atoi(argv[4]) : RUNS;
    if (timing && (warmup < 0 || repeats < 2)) {{
        std::fprintf(stderr, "warmup >= 0 and repeats >= 2\n");
        return 2;
    }}
    devEvent_t ev0, ev1;
    CHECK(devEventCreate(&ev0));
    CHECK(devEventCreate(&ev1));
    for (int r = 0; r < warmup + repeats; ++r) {{
        CHECK(devMemset(dF, 0, F.size() * sizeof(double)));
        CHECK(devSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(devEventRecord(ev0));
        if (sequence()) return 1;
        CHECK(devEventRecord(ev1));
        CHECK(devSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        if (r < warmup) continue;
        if (timing) {{
            float ms = 0.0f;
            CHECK(devEventElapsedTime(&ms, ev0, ev1));
            std::printf("rep %d wall_ns %lld events_ns %.0f\n", r - warmup,
                        (long long)std::chrono::duration_cast<
                            std::chrono::nanoseconds>(t1 - t0).count(),
                        1.0e6 * (double)ms);
            if (r < warmup + repeats - 2) continue;
        }}
        CHECK(devMemcpy(F.data(), dF, F.size() * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), F.begin(), F.end());
    }}
    for (int l = 0; l < NL; ++l) {{
        CHECK(devFree(pd[l]));
        CHECK(devFree(pi[l]));
    }}
    CHECK(devFree(dDsh));
    CHECK(devFree(dD));
    CHECK(devFree(dF));
    CHECK(devEventDestroy(ev0));
    CHECK(devEventDestroy(ev1));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-fock launches %d runs %d\n", NL, timing ? 2 : RUNS);
    return 0;
}}
"""
