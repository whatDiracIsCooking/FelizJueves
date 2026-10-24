"""The fused Fock kernel ON A CARD (#288) -- harness.

    record(plan, basis, D, *, threads=)            -> (list[FRec], FWalk)
    fock_program(kernels, backend, workdir)        -> Program
    fock_stage(recs, kernels)                      -> np.ndarray
    run_on_card(prog, binary, recs, n_ao, run, *, timing=) -> [run] F
    interpret(recs, n_ao)                          -> F (not closed)
    synthetic_launch(cls, *, side, seed)           -> (FRec, basis)
    fock_scatter_program / fock_scatter_stage / interpret_scatter
                                     the #358 scatter path (end of file)

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
(the binary was checked on real launches first). The basis is
`SYNTHETIC_COUNTS` shells of `l = 0, 1, 2` when that fills the launch, and
`DEFERRED_COUNTS` otherwise (#290: twelve of the sixteen classes above L =
2); the launch shape, the exponents and the cubes are the same either way.

Harness, in `test/`: not a second emitter, no measurement decided here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from plan_driver import UPLOAD_INT, Program, _unit
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.ao.density import checked_density
from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock import kernel_f_scatter as FS
from felizjueves.eri.fock.levers import scatter as SC
from felizjueves.eri.fock.walk_f import KET_BREAK, FWalk, pair_record, walk_f
from felizjueves.hgp import vrr
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys

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


def record_plan(plan, basis: Basis, D, *, threads: int = THREADS,
                levers: tuple[str, ...] = ()) -> list[FRec]:
    """The launches `record` would return, packed straight from `plan`:
    walk_f's per-launch arguments (`pair_record`, `kernel_f.pack`, `Dsh`,
    `D`) with none of its walk -- no partition of the unique quartets and no
    six-block replay, which at polyglycine's size are ~1e9 Python set
    members. Every launch's kernel is built with `levers` added to its own;
    a plan whose launches name `ket-break` or `scatter` (launch arguments
    walk_f adds) is refused. `test_fock_record_plan.py` holds it to
    `record` launch by launch."""
    rows = checked_density(basis, D)
    sh, n_sh, n_ao = basis.shells, len(basis.shells), basis.n_ao
    Dsh = KF.density_blocks(rows, basis.offsets, [s.n_cart for s in sh])
    D_flat = tuple(x for r in rows for x in r)
    records = [[pair_record(basis, p.a, p.b, G=p.G, prims=p.prims)
                for p in b.pairs] for b in plan.batches]
    kernels: dict[object, Kernel] = {}
    out = []
    for L in plan.launches:
        own = tuple(L.spec.levers)
        if set(own) & {KET_BREAK, SC.NAME}:
            raise ValueError(f"record_plan: launch ({L.bra}, {L.ket}) names "
                             f"{own}; walk_f's `record` stages those")
        spec = replace(L.spec, levers=tuple(sorted({*own, *levers})))
        if spec not in kernels:
            kernels[spec] = KF.build_f_kernel(spec)
        k = kernels[spec]
        pb, pk = KF.pack(records[L.bra]), KF.pack(records[L.ket])
        args = {
            "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree,
            "k_cd": pk.degree, "n_sh": n_sh, "n_ao": n_ao,
            "diag": int(L.bra == L.ket), "tau": float(plan.tau),
            "bra_prim": pb.prim, "AB": pb.sep, "G_ab": pb.G,
            "bra_sh": pb.sh, "bra_ao": pb.ao, "ket_prim": pk.prim,
            "CD": pk.sep, "G_cd": pk.G, "ket_sh": pk.sh, "ket_ao": pk.ao,
            "Dsh": Dsh, "D": D_flat,
        }
        grid, block = KF.launch_shape(pb.n, threads)
        out.append(FRec(k, grid, block[0],
                        {p.name: args[p.name] for p in k.params
                         if p.name != KF.FOCK}))
    return out


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


# #288's basis: fills every class of total L <= 2, and (00|12), (01|02),
# (01|11) at L = 3 and (01|12) at L = 4. #290: a class it cannot fill
# (fewer than TIMING_N_BRA bra pairs, or TIMING_N_KET ket pairs disjoint from
# them -- (ss|dd) has 10 ket pairs from 4 d shells) draws the same launch
# shape from DEFERRED_COUNTS, which fills every class of the ladder
# ((dd|dd): 2211 canonical pairs).
SYNTHETIC_COUNTS = (66, 34, 4)
DEFERRED_COUNTS = (66, 66, 66)


def synthetic_basis(*, side: float, seed: int,
                    counts=SYNTHETIC_COUNTS) -> Basis:
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
                     seed: int, n_ket: int = TIMING_N_KET
                     ) -> tuple[FRec, Basis]:
    """The timing launch of `cls` (module docstring, THE SYNTHETIC LAUNCH);
    `n_ket` ket pairs, the last of the canonical list (#363's lever clock
    asks fewer than the F rows' TIMING_N_KET)."""
    la, lb, lc, ld = cls
    for counts in (SYNTHETIC_COUNTS, DEFERRED_COUNTS):
        basis = synthetic_basis(side=side, seed=seed, counts=counts)
        idx: dict[int, list[int]] = {}
        for i, s in enumerate(basis.shells):
            idx.setdefault(s.l, []).append(i)
        bra_all = _canonical_pairs(idx, la, lb)
        ket_all = _canonical_pairs(idx, lc, ld)
        bras = bra_all[:TIMING_N_BRA]
        kets = ket_all[-n_ket:]
        if len(bras) == TIMING_N_BRA and len(kets) == n_ket \
                and not set(bras) & set(kets):
            break
    assert len(bras) == TIMING_N_BRA and len(kets) == n_ket
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


# --- #358: the atomic-free write-back's launch path --------------------------
#
# ADDITIVE: everything above is the default path, unchanged. A `scatter`
# lever kernel (eri/fock/levers/scatter.py) writes no F: it stores into a slot
# buffer W, and the class-agnostic `f_scatter` kernel (kernel_f_scatter.py)
# sums W into F. So each launch of the scatter program is THREE device
# steps, in stream order: zero W (`devMemset`, inside the sequence and so
# inside the clock -- it is the lever's cost), the F kernel, then the
# scatter over that launch's map. The program is FOCK_PROGRAM itself with
# other `cases` and a longer per-launch header:
#
#     per launch: kernel-index n_bra n_ket k_ab k_cd diag threads nd ni
#                 n_el n_slot                       (SCATTER_PER = 11)
#     ints:       bra_sh bra_ao ket_sh ket_ao, then the map: el (2 n_el),
#                 ptr (n_el + 1), src (n_slot)
#
# W is one device buffer grown to the largest launch's n_slot. `f_only=True`
# builds the same program without the scatter launch (W written, F left
# zero): the tail measurement's arm, never checked against F.

SCATTER_THREADS = 128  # the scatter kernel's block width; not numerical
SCATTER_PER = PER_LAUNCH + 2


def _cls_of(k: Kernel) -> tuple[int, int, int, int]:
    """The F class of an F unit, from its name (`f_<la><lb><lc><ld>...`)."""
    digits = k.name.split("_")[1]
    if not (k.name.startswith("f_") and len(digits) == 4
            and digits.isdigit()):
        raise ValueError(f"{k.name} is not an F unit's name")
    return tuple(int(c) for c in digits)  # type: ignore[return-value]


def _check_scatter_signature(k: Kernel) -> None:
    names = [p.name for p in k.params]
    want = [n for n in (*SCALARS, "bra_prim", "AB", "G_ab", "bra_sh",
                        "bra_ao", "ket_prim", "CD", "G_cd", "ket_sh",
                        "ket_ao", *SHARED, FS.SLOTS) if n in names]
    if names != want or [p.name for p in k.params
                         if p.intent is Intent.OUT] != [FS.SLOTS]:
        raise ValueError(f"{k.name}: signature {names} is not the scatter "
                         "lever's F kernel (F replaced by W, no tap)")


def scatter_map_np(cls, n_ket: int, bra_ao, ket_ao) -> FS.ScatterMap:
    """`kernel_f_scatter.scatter_map`, vectorised for the timing launches
    (millions of slots); held equal to it on small launches by
    test_f_lever_scatter_device.py's fast tier."""
    n = [len(vrr.cartesians(x)) for x in cls]
    bao = np.asarray(bra_ao, dtype=np.int64).reshape(-1, 2)
    kao = np.asarray(ket_ao, dtype=np.int64).reshape(-1, 2)
    nb, nk = len(bao), len(kao)
    assert nk == n_ket
    n_slot = FS.n_slot(tuple(cls), nb, nk)
    row_w = SC.row_width(tuple(cls), nk)
    rows = np.empty(n_slot, dtype=np.int64)
    cols = np.empty(n_slot, dtype=np.int64)
    base = np.arange(nb, dtype=np.int64) * row_w
    ab = KF.BLOCKS[0]
    for i in range(n[ab[0]]):
        for j in range(n[ab[1]]):
            s = base + i * n[ab[1]] + j
            rows[s], cols[s] = bao[:, 0] + i, bao[:, 1] + j
    off = SC.ket_offsets(tuple(cls))
    o = [bao[:, 0][:, None], bao[:, 1][:, None], kao[:, 0][None, :],
         kao[:, 1][None, :]]
    k = np.arange(nk, dtype=np.int64)[None, :]
    for b in KF.BLOCKS[1:]:
        for i in range(n[b[0]]):
            for j in range(n[b[1]]):
                s = (base[:, None] + SC.n_ab(tuple(cls))
                     + (off[b] + i * n[b[1]] + j) * nk + k)
                rows[s] = np.broadcast_to(o[b[0]] + i, s.shape)
                cols[s] = np.broadcast_to(o[b[1]] + j, s.shape)
    width = int(max(rows.max(), cols.max())) + 1
    key = rows * width + cols
    src = np.argsort(key, kind="stable")  # ascending slots within an element
    uniq, counts = np.unique(key[src], return_counts=True)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    el = np.stack([uniq // width, uniq % width], axis=1).ravel()
    # numpy arrays in the map's tuple fields: what staging reads is the same
    return FS.ScatterMap(len(uniq), n_slot, el, ptr, src)


def interpret_scatter(recs: Sequence[FRec], n_ao: int, *,
                      threads: int = SCATTER_THREADS) -> np.ndarray:
    """The recorded `scatter` launches through the interpreter, each F
    launch then its scatter, in order, into one zeroed `F` (NOT closed)."""
    F = np.zeros(n_ao * n_ao)
    sk = FS.build_scatter_kernel()
    for r in recs:
        cls = _cls_of(r.kernel)
        n_slot = FS.n_slot(cls, r.args["n_bra"], r.args["n_ket"])
        a = {}
        for p in r.kernel.params:
            v = np.zeros(n_slot) if p.name == FS.SLOTS else r.args[p.name]
            if p.array:
                v = np.array(v, dtype=np.int32 if p.type is Scalar.I32
                             else float)
            a[p.name] = v
        W = Interpreter(r.kernel, launch=Launch(r.grid, (r.threads, 1), 32),
                        boys=boys).run(**a)[FS.SLOTS]
        m = FS.scatter_map(cls, r.args["n_ket"], r.args["bra_ao"],
                           r.args["ket_ao"])
        grid, block = FS.launch_shape(m.n_el, threads)
        sa = {n: (np.array(v, dtype=np.int32) if n in ("el", "ptr", "src")
                  else v) for n, v in m.args(W, F, n_ao).items()}
        F = Interpreter(sk, launch=Launch(grid, block, 32)).run(**sa)[KF.FOCK]
    return np.asarray(F, dtype=float).reshape(n_ao, n_ao)


def _scatter_case(i: int, k: Kernel, ns: str, f_only: bool) -> str:
    has = {p.name for p in k.params}
    rec = len(KF.REC)
    size = {"bra_prim": f"(size_t)nb * kab * {rec}", "AB": "(size_t)nb * 3",
            "G_ab": "(size_t)nb", "ket_prim": f"(size_t)nk * kcd * {rec}",
            "CD": "(size_t)nk * 3", "G_cd": "(size_t)nk",
            "bra_sh": "(size_t)2 * nb", "bra_ao": "(size_t)2 * nb",
            "ket_sh": "(size_t)2 * nk", "ket_ao": "(size_t)2 * nk"}
    lines = [f"        case {i}: {{  // {k.name}, then f_scatter"]
    at_i = "0"
    for buf, names, ty in (("pd", DOUBLES, "double"), ("pi", INTS, "int")):
        at = "0"
        for n in names:
            if n not in has:
                continue
            lines.append(f"            const {ty}* a_{n} = {buf}[l] + {at};")
            at = f"{at} + {size[n]}"
        at_i = at
    head = "in[HEAD + (size_t)PER * l"
    lines.append(f"            const size_t nslot = (size_t){head} + 10];")
    if not f_only:  # the map is read only by the scatter launch
        lines += [
            f"            const int nel = (int){head} + 9];",
            f"            const int* a_el = pi[l] + {at_i};",
            "            const int* a_ptr = a_el + (size_t)2 * nel;",
            "            const int* a_src = a_ptr + (size_t)nel + 1;",
        ]
    lines += [
        "            if (nslot > fj_wcap) {",
        "                if (fj_w) CHECK(devFree(fj_w));",
        "                fj_w = nullptr;",
        "                CHECK(devMalloc(&fj_w, nslot * sizeof(double)));",
        "                fj_wcap = nslot;",
        "            }",
        "            CHECK(devMemset(fj_w, 0, nslot * sizeof(double)));",
    ]
    arg = {"n_bra": "nb", "n_ket": "nk", "k_ab": "kab", "k_cd": "kcd",
           "n_sh": "NSH", "n_ao": "NAO", "diag": "dg", "tau": "TAU",
           "Dsh": "dDsh", "D": "dD", FS.SLOTS: "fj_w"}
    args = [arg.get(p.name, f"a_{p.name}") for p in k.params]
    lines += [
        f"            {ns}::{k.name}<<<dim3(nb), dim3(th, 1)>>>(",
        f"                {', '.join(args)});",
    ]
    if not f_only:
        lines += [
            "            CHECK(devGetLastError());",
            f"            fj_sc::{FS.NAME}<<<dim3((nel + {SCATTER_THREADS} - 1)"
            f" / {SCATTER_THREADS}), dim3({SCATTER_THREADS}, 1)>>>(",
            "                nel, (int)nslot, NAO, a_el, a_ptr, a_src, fj_w, "
            "dF);",
        ]
    lines += ["            break;", "        }"]
    return "\n".join(lines)


def fock_scatter_program(kernels: Sequence[Kernel], backend: str,
                         workdir: Path, *, f_only: bool = False,
                         units: Mapping[str, Path] | None = None,
                         name: str = "fock_scatter") -> Program:
    """The scatter path's `main()` around `kernels` (each a `scatter` lever
    F kernel) and the `f_scatter` unit (the block comment above)."""
    if not kernels or len({k.name for k in kernels}) != len(kernels):
        raise ValueError("fock_scatter_program takes distinct kernels")
    _, _, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    sk = FS.build_scatter_kernel()
    includes, paths, cases, ptrs = [], [], [], []
    for i, k in enumerate(kernels):
        _check_scatter_signature(k)
        path = _unit(k, backend, workdir, units)
        paths.append(path)
        ns = f"fj_f{i}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        cases.append(_scatter_case(i, k, ns, f_only))
        ptrs.append(f"(const void*){ns}::{k.name}")
    spath = _unit(sk, backend, workdir, units)
    paths.append(spath)
    includes += [f'namespace fj_sc {{\n#include "{spath}"\n}}',
                 "static double* fj_w = nullptr;  // W, grown per launch",
                 "static size_t fj_wcap = 0;"]
    ptrs.append(f"(const void*)fj_sc::{FS.NAME}")
    src = FOCK_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes),
        upload_int=UPLOAD_INT, n_k=len(kernels), runs=RUNS,
        header=HEADER, per=SCATTER_PER, vram_kernels=", ".join(ptrs),
        cases="\n".join(cases))
    if name in {k.name for k in kernels} | {FS.NAME}:
        raise ValueError(f"the program {name!r} would overwrite its unit")
    main = workdir / f"{name}.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def fock_scatter_stage(recs: Sequence[FRec], kernels: Sequence[Kernel], *,
                       mapper=None) -> np.ndarray:
    """The scatter program's input for `recs` (each a `scatter` lever F
    launch of one of `kernels`), every launch's map computed here by
    `mapper` (`kernel_f_scatter.scatter_map` unless given)."""
    if not recs:
        raise ValueError("no launch to stage")
    mapper = FS.scatter_map if mapper is None else mapper
    index = {k.name: i for i, k in enumerate(kernels)}
    n_sh, n_ao = int(_global(recs, "n_sh")[0]), int(_global(recs, "n_ao")[0])
    tau = float(_global(recs, "tau")[0])
    Dsh, D = _global(recs, "Dsh"), _global(recs, "D")
    head = [len(recs), n_sh, n_ao, tau]
    body: list[np.ndarray] = []
    for r in recs:
        if r.name not in index or r.kernel != kernels[index[r.name]]:
            raise ValueError(f"{r.name} is not one of the program's kernels")
        has = {p.name for p in r.kernel.params}
        a = r.args
        m = mapper(_cls_of(r.kernel), a["n_ket"], a["bra_ao"], a["ket_ao"])
        assert m.n_slot == FS.n_slot(_cls_of(r.kernel), a["n_bra"], a["n_ket"])
        dd = [np.ravel(np.asarray(a[n], dtype=float))
              for n in DOUBLES if n in has]
        ii = [np.ravel(np.asarray(a[n], dtype=float)) for n in INTS]
        ii += [np.asarray(m.el, dtype=float), np.asarray(m.ptr, dtype=float),
               np.asarray(m.src, dtype=float)]
        nd, ni = sum(x.size for x in dd), sum(x.size for x in ii)
        head += [index[r.name], a["n_bra"], a["n_ket"], a["k_ab"], a["k_cd"],
                 a["diag"], r.threads, nd, ni, m.n_el, m.n_slot]
        assert r.grid == a["n_bra"], (r.grid, a["n_bra"])
        body += [*dd, *ii]
    return np.concatenate([np.array(head, dtype=np.float64), Dsh, D,
                           *body]).astype(np.float64)
