"""The three-center DF integral `(mn|P)` built ON A CARD -- harness.

    eri3_program(classes, n_ao, n_aux, backend, workdir) -> Program
    eri3_stage(classes)                                  -> np.ndarray
    build_on_card(classes, n_ao, n_aux, backend, tc, workdir,
                  run_device_binary)                     -> [run] -> T

The records are `eri.density_fit.eri3.eri3_classes`' (every AO shell pair `i <= j`
staged `la <= lb`, swapped when the basis order has the higher `l` first,
crossed with every aux shell; per class one bra record table, one aux record
table, and two record ranges and three offsets per cell), the kernels
`kernel_eri3.build_eri3_kernel`'s, emitted by `emit/cuda.py` / `emit/hip.py`
unchanged. Only the `main()` around them is written here, as
`schwarz_driver.py` does for `Q` -- harness, in `test/`, not a second
emitter (#657 moves it to the shared runner, as #654 did
`metric_driver.py`). Every unit is emitted into `workdir` (the kernel tier's
`KERNEL_GEN_CMD` writes no eri3 unit), and the program is compiled with
devtools/config.sh's flags under #92's warning filter
(`plan_driver.compile_checked`).

WHAT THE GENERATED `main()` DOES, `RUNS` times: zero ONE
`n_ao x n_ao x n_aux` tensor (`[m][n][P]`, aux fastest), launch every
class's kernel into it -- one launch per class, one thread per cell
(`launch_shape`, `THREADS` lanes) -- then one synchronize and the read-back.
Each element is owned by one cell and written by a plain store (mirrored in
`(m, n)`), so every run must be bit-identical.

THE UPLOADS, once: per class one buffer of its double record arrays (the
bra table `zeta_ab | P | XPA | XPB | s_ab`, then the aux table
`A | alpha | s_aux`, vectors row-major `[n_rec, 3]`) and one of its seven
int cell arrays (`ab_rec0 | ab_rec1 | p_rec0 | p_rec1 | off_a | off_b |
off_p`). A kernel argument is a pointer into those buffers, read off the
kernel's own parameter list, so the call cannot drift from the signature.
The record array `s` is the kernel's `s_aux` (`kernel_eri3.PARAM_OF`).

THE FORM (#606): `lowstack=` reaches `build_eri3_kernel` unchanged (None,
the default, is the gate's choice), so a program can launch either form of
a class; `timeout=` bounds the program's compile.

TIMED (#606, measure/measure_df_lowstack.py): `compile_timed` builds the
same program with `timed=True` -- run `warmup + reps` times, one event pair
around each class's launch, a `rep R class C events_ns X` line per timed
launch (`kblock_driver.timings` reads them), and only the last run's tensor
written. Each element is a plain store, so every run writes the same
tensor and nothing is zeroed between runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from kblock_driver import timings
from plan_driver import RUNS, UPLOAD_INT, Program, _unit, compile_checked
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.eri.density_fit.eri3 import Eri3Class
from felizjueves.eri.density_fit.kernel_eri3 import (
    OUT,
    PARAM_OF,
    build_eri3_kernel,
    launch_shape,
)
from felizjueves.eri.density_fit.kernel_hgp import build_heri3_kernel
from felizjueves.ir import Intent, Kernel
from felizjueves.spec import Engine

THREADS = 32  # lanes per block, one cell per thread; not numerical
WIDTH = {"zeta_ab": 1, "P": 3, "XPA": 3, "XPB": 3, "s_ab": 1,  # per bra rec
         "A": 3, "alpha": 1, "s": 1}  # per aux rec; Eri3Class's names
NDB = sum(WIDTH[n] for n in Eri3Class.BRA_ARRAYS)  # doubles per bra record
NDA = sum(WIDTH[n] for n in Eri3Class.AUX_ARRAYS)  # doubles per aux record
NI = len(Eri3Class.CELL_ARRAYS)  # ints per cell
SIZES = ("n_cell", "n_bra_rec", "n_aux_rec", "n_ao", "n_aux")


def _param(name: str) -> str:
    return PARAM_OF.get(name, name)


def _check_signature(kern: Kernel) -> None:
    names = [p.name for p in kern.params]
    want = [*SIZES,
            *(_param(n) for n in (*Eri3Class.BRA_ARRAYS,
                                  *Eri3Class.AUX_ARRAYS)),
            *Eri3Class.CELL_ARRAYS, OUT]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the records "
                         f"want {want}")


def _call(kern: Kernel, c: int, ns: str, cls: Eri3Class, *,
          timed: bool = False) -> str:
    at, off = {}, 0
    for name, n in ([(b, cls.n_bra_rec) for b in Eri3Class.BRA_ARRAYS]
                    + [(a, cls.n_aux_rec) for a in Eri3Class.AUX_ARRAYS]):
        at[_param(name)] = f"dd[{c}] + {off}" if off else f"dd[{c}]"
        off += WIDTH[name] * n
    for k, name in enumerate(Eri3Class.CELL_ARRAYS):
        at[name] = f"di[{c}] + {k * cls.n_cell}" if k else f"di[{c}]"
    at.update({"n_cell": str(cls.n_cell), "n_bra_rec": str(cls.n_bra_rec),
               "n_aux_rec": str(cls.n_aux_rec), "n_ao": "NAO",
               "n_aux": "NAUX", OUT: "dT"})
    args = [at[p.name] for p in kern.params]
    grid, (lanes, _) = launch_shape(cls.n_cell, THREADS)
    head = (f"        // class ({cls.la}{cls.lb}|{cls.lP}), {cls.n_cell} "
            f"cells, {cls.n_bra_rec} bra / {cls.n_aux_rec} aux records\n")
    launch = (f"        {ns}::{kern.name}<<<dim3({grid}), dim3({lanes}, 1)>>>(\n"
              f"            {', '.join(args)});\n"
              f"        CHECK(devGetLastError());\n"
              f"        ++launches;")
    return head + (timed_launch(launch, c) if timed else launch)


def eri3_program(classes: Sequence[Eri3Class], n_ao: int, n_aux: int,
                 backend: str, workdir: Path, *,
                 engine: Engine = Engine.MMD, lowstack: bool | None = None,
                 timed: bool = False) -> Program:
    """The `main()` building `(mn|P)` from every class in `classes`
    (module docstring: THE FORM, TIMED)."""
    _, _, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    includes, paths, calls, kernels = [], [], [], []
    for c, cls in enumerate(classes):
        build = (build_heri3_kernel if engine is Engine.HGP
                 else build_eri3_kernel)
        kern = (build((cls.la, cls.lb, cls.lP)) if engine is Engine.HGP
                else build((cls.la, cls.lb, cls.lP), lowstack=lowstack))
        _check_signature(kern)
        path = _unit(kern, backend, workdir, None)
        paths.append(path)
        ns = f"fj_e{c}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        kernels.append(f"(const void*){ns}::{kern.name}")
        calls.append(_call(kern, c, ns, cls, timed=timed))
    src = ERI3_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes), upload_int=UPLOAD_INT,
        n_c=len(classes), n_ao=n_ao, n_aux=n_aux, runs=RUNS, ndb=NDB,
        nda=NDA, ni=NI,
        nb=", ".join(str(c.n_bra_rec) for c in classes),
        na=", ".join(str(c.n_aux_rec) for c in classes),
        nc=", ".join(str(c.n_cell) for c in classes),
        vram_kernels=", ".join(kernels), launches="\n".join(calls))
    if timed:
        src = timed_source(src, "T")
    main = workdir / f"eri3_main.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def eri3_stage(classes: Sequence[Eri3Class]) -> np.ndarray:
    """The program's input: per class its double bra then aux record arrays
    in parameter order (vectors row-major), then its int cell arrays."""
    parts: list = []
    for cls in classes:
        arr = cls.arrays()
        for name in (*Eri3Class.BRA_ARRAYS, *Eri3Class.AUX_ARRAYS):
            parts.append(np.ravel(np.asarray(arr[name], dtype=np.float64)))
        for name in Eri3Class.CELL_ARRAYS:
            parts.append(np.asarray(arr[name], dtype=np.float64))
    return np.concatenate(parts)


def build_on_card(classes: Sequence[Eri3Class], n_ao: int, n_aux: int,
                  backend: str, tc: Toolchain, workdir: Path,
                  run_device_binary, *,
                  engine: Engine = Engine.MMD, lowstack: bool | None = None,
                  timeout: float = 1800.0) -> np.ndarray:
    """`(mn|P)` of `classes` built on the card, `[run][n_ao, n_ao, n_aux]`
    (`RUNS` runs). The caller holds the card (its device fixture)."""
    prog = eri3_program(classes, n_ao, n_aux, backend, workdir, engine=engine,
                        lowstack=lowstack)
    binary = compile_checked(tc, prog, workdir / "eri3_bin", timeout=timeout)
    src, dst = workdir / "eri3.in", workdir / "eri3.out"
    eri3_stage(classes).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    want = (f"fj-eri3 classes {len(classes)} launches "
            f"{len(classes) * RUNS} runs {RUNS}")
    assert want in out, out
    raw = np.fromfile(dst, dtype=np.float64)
    assert raw.size == RUNS * n_ao * n_ao * n_aux, (raw.size, RUNS, n_ao,
                                                    n_aux)
    return raw.reshape(RUNS, n_ao, n_ao, n_aux)


def compile_timed(classes: Sequence[Eri3Class], n_ao: int, n_aux: int,
                  backend: str, tc: Toolchain, workdir: Path, *,
                  lowstack: bool | None, timeout: float) -> Path:
    """The TIMED program of `classes` in one form, compiled."""
    prog = eri3_program(classes, n_ao, n_aux, backend, workdir,
                        lowstack=lowstack, timed=True)
    return compile_checked(tc, prog, workdir / "eri3_timed", timeout=timeout)


def run_timed(binary: Path, classes: Sequence[Eri3Class], n_ao: int,
              n_aux: int, backend: str, workdir: Path, run_device_binary, *,
              warmup: int, reps: int) -> tuple[np.ndarray, dict]:
    """The last run's tensor and `timings` of one run of a timed binary."""
    src, dst = workdir / "eri3_timed.in", workdir / "eri3_timed.out"
    if not src.exists():
        eri3_stage(classes).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), str(warmup),
                            str(reps), vram=backend)
    raw = np.fromfile(dst, dtype=np.float64)
    assert raw.size == n_ao * n_ao * n_aux, (raw.size, n_ao, n_aux)
    return raw.reshape(n_ao, n_ao, n_aux), timings(out)


def timed_launch(launch: str, c: int) -> str:
    """One launch between an event pair, its time printed once timed."""
    return ("        CHECK(devEventRecord(ev0));\n" + launch + "\n"
            "        CHECK(devEventRecord(ev1));\n"
            "        CHECK(devEventSynchronize(ev1));\n"
            "        if (r >= warmup) {\n"
            "            float ms = 0.0f;\n"
            "            CHECK(devEventElapsedTime(&ms, ev0, ev1));\n"
            f"            std::printf(\"rep %d class {c} events_ns %.0f\\n\",\n"
            "                        r - warmup, 1e6 * (double)ms);\n"
            "        }")


EVENTS = r"""
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devEvent_t           hipEvent_t
#define devEventCreate       hipEventCreate
#define devEventRecord(e)    hipEventRecord(e, 0)
#define devEventSynchronize  hipEventSynchronize
#define devEventElapsedTime  hipEventElapsedTime
#else
#define devEvent_t           cudaEvent_t
#define devEventCreate       cudaEventCreate
#define devEventRecord(e)    cudaEventRecord(e, 0)
#define devEventSynchronize  cudaEventSynchronize
#define devEventElapsedTime  cudaEventElapsedTime
#endif
"""


def timed_source(src: str, out: str) -> str:
    """A RUNS-loop program (this module's) made TIMED:
    `usage: in out warmup reps`, `warmup + reps` runs, only the last run's
    `out` array written. Each edit must apply exactly once."""
    def once(s: str, a: str, b: str) -> str:
        assert s.count(a) == 1, (a, s.count(a))
        return s.replace(a, b)

    src = once(src, "int main(int argc, char** argv) {",
               EVENTS + "int main(int argc, char** argv) {")
    src = once(src, 'if (argc != 3) { std::fprintf(stderr, "usage: in out\\n"); '
               "return 2; }",
               'if (argc != 5) { std::fprintf(stderr, "usage: in out warmup '
               'reps\\n"); return 2; }\n'
               "    const int warmup = std::atoi(argv[3]), "
               "reps = std::atoi(argv[4]);\n"
               "    if (warmup < 0 || reps < 1) return 2;\n"
               "    devEvent_t ev0, ev1;\n"
               "    CHECK(devEventCreate(&ev0));\n"
               "    CHECK(devEventCreate(&ev1));")
    src = once(src, "for (int r = 0; r < RUNS; ++r) {",
               "for (int r = 0; r < warmup + reps; ++r) {")
    src = once(src, f"out.insert(out.end(), {out}.begin(), {out}.end());",
               f"if (r == warmup + reps - 1) "
               f"out.insert(out.end(), {out}.begin(), {out}.end());")
    return src


# Every class into one tensor, RUNS times.
ERI3_PROGRAM = r"""// generated by test/eri3_driver.py: harness
{prologue}
{portable}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devMemset hipMemset
#define devFree   hipFree
#else
#define devMemset cudaMemset
#define devFree   cudaFree
#endif
{includes}
{upload_int}
static const int NC = {n_c}, NAO = {n_ao}, NAUX = {n_aux}, RUNS = {runs};
static const int NDB = {ndb}, NDA = {nda}, NI = {ni};
static const int nb[NC] = {{{nb}}};  // bra records per class
static const int na[NC] = {{{na}}};  // aux records per class
static const int nc[NC] = {{{nc}}};  // cells per class

// usage: in out. `in` is, per class, its double bra record arrays (NDB per
// bra record), its aux record arrays (NDA per aux record), then its int
// cell arrays (NI per cell); `out` is each run's tensor [NAO, NAO, NAUX],
// each from a zeroed tensor.
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    size_t want = 0;
    for (int c = 0; c < NC; ++c)
        want += (size_t)NDB * nb[c] + (size_t)NDA * na[c] + (size_t)NI * nc[c];
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the classes want %zu\n",
                     in.size(), want);
        return 2;
    }}
    const size_t nT = (size_t)NAO * NAO * NAUX;
    if (fj_vram_probe(in.size() * sizeof(double) + nT * sizeof(double)
                          + (size_t)64 * NC,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data();
    double* dd[NC];
    int* di[NC];
    for (int c = 0; c < NC; ++c) {{
        const size_t nd = (size_t)NDB * nb[c] + (size_t)NDA * na[c];
        dd[c] = upload(h, nd);
        h += nd;
        di[c] = upload_int(h, (size_t)NI * nc[c]);
        h += (size_t)NI * nc[c];
        if (!dd[c] || !di[c]) {{
            std::fprintf(stderr, "upload %d failed\n", c);
            return 1;
        }}
    }}
    double* dT = nullptr;
    CHECK(devMalloc(&dT, nT * sizeof(double)));
    std::vector<double> T(nT), out;
    int launches = 0;
    for (int r = 0; r < RUNS; ++r) {{
        CHECK(devMemset(dT, 0, T.size() * sizeof(double)));
{launches}
        CHECK(devSynchronize());
        CHECK(devMemcpy(T.data(), dT, T.size() * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), T.begin(), T.end());
    }}
    for (int c = 0; c < NC; ++c) {{
        CHECK(devFree(dd[c]));
        CHECK(devFree(di[c]));
    }}
    CHECK(devFree(dT));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-eri3 classes %d launches %d runs %d\n", NC, launches, RUNS);
    return 0;
}}
"""
