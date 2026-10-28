"""The DF Coulomb metric `(P|Q)` of an aux basis built ON A CARD -- harness.

    metric_program(classes, n_aux, backend, workdir) -> Program
    metric_stage(classes)                            -> np.ndarray
    build_on_card(classes, n_aux, backend, tc, workdir,
                  run_device_binary)                 -> [run] -> M

The records are `eri.density_fit.metric.metric_classes`' (every aux shell pair `P <= Q`
staged with `lP <= lQ`, swapped when the basis order has the higher `l`
first; per class one record table and two record ranges per pair), the
kernels `kernel_metric.build_metric_kernel`'s, emitted by `emit/cuda.py` /
`emit/hip.py` unchanged. Only the `main()` around them is written here, as
`schwarz_driver.py` does for `Q` -- harness, in `test/`, not a second
emitter. Every unit is emitted into `workdir` (the kernel tier's
`KERNEL_GEN_CMD` writes no metric unit), and the program is compiled with
devtools/config.sh's flags under #92's warning filter
(`plan_driver.compile_checked`).

WHAT THE GENERATED `main()` DOES, `RUNS` times: zero ONE `n_aux x n_aux`
matrix, launch every class's kernel into it -- one launch per class, one
thread per aux shell pair (`launch_shape`, `THREADS` lanes) -- then one
synchronize and the read-back. Each element is owned by one aux shell pair
and written by a plain store, so every run must be bit-identical.

THE UPLOADS, once: per class one buffer of its double record arrays
(`A | alpha | s_aux`, `A` row-major `[n_rec, 3]`) and one of its six int
pair arrays (`p_rec0 | p_rec1 | q_rec0 | q_rec1 | off_p | off_q`). A kernel
argument is a pointer into those buffers, read off the kernel's own
parameter list, so the call cannot drift from the signature. The record
array `s` is the kernel's `s_aux` (`kernel_metric.PARAM_OF`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from eri3_driver import timed_launch, timed_source
from kblock_driver import timings
from plan_driver import RUNS, UPLOAD_INT, Program, _unit, compile_checked
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.eri.density_fit.kernel_hgp import build_hmetric_kernel
from felizjueves.eri.density_fit.kernel_metric import (
    OUT,
    PARAM_OF,
    build_metric_kernel,
    launch_shape,
)
from felizjueves.eri.density_fit.metric import MetricClass
from felizjueves.ir import Intent, Kernel
from felizjueves.spec import Engine

THREADS = 32  # lanes per block, one aux shell pair per thread; not numerical
WIDTH = {"A": 3, "alpha": 1, "s": 1}  # by MetricClass.RECORD_ARRAYS name
ND = sum(WIDTH.values())  # doubles per record
NI = len(MetricClass.PAIR_ARRAYS)  # ints per aux shell pair


def _param(name: str) -> str:
    return PARAM_OF.get(name, name)


def _check_signature(kern: Kernel) -> None:
    names = [p.name for p in kern.params]
    want = ["n_pair", "n_rec", "n_aux",
            *(_param(n) for n in MetricClass.RECORD_ARRAYS),
            *MetricClass.PAIR_ARRAYS, OUT]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the records "
                         f"want {want}")


def _call(kern: Kernel, c: int, ns: str, cls: MetricClass, *,
          timed: bool = False) -> str:
    at, off = {}, 0
    for name in MetricClass.RECORD_ARRAYS:
        at[_param(name)] = f"dd[{c}] + {off}" if off else f"dd[{c}]"
        off += WIDTH[name] * cls.n_rec
    for k, name in enumerate(MetricClass.PAIR_ARRAYS):
        at[name] = f"di[{c}] + {k * cls.n_pair}" if k else f"di[{c}]"
    at.update({"n_pair": str(cls.n_pair), "n_rec": str(cls.n_rec),
               "n_aux": "NAUX", OUT: "dM"})
    args = [at[p.name] for p in kern.params]
    grid, (lanes, _) = launch_shape(cls.n_pair, THREADS)
    head = (f"        // aux class ({cls.lP}|{cls.lQ}), {cls.n_pair} "
            f"pairs, {cls.n_rec} records\n")
    launch = (f"        {ns}::{kern.name}<<<dim3({grid}), dim3({lanes}, 1)>>>(\n"
              f"            {', '.join(args)});\n"
              f"        CHECK(devGetLastError());\n"
              f"        ++launches;")
    return head + (timed_launch(launch, c) if timed else launch)


def metric_program(classes: Sequence[MetricClass], n_aux: int, backend: str,
                   workdir: Path, *,
                   engine: Engine = Engine.MMD, lowstack: bool | None = None,
                   timed: bool = False) -> Program:
    """The `main()` building `(P|Q)` from every class in `classes`;
    `lowstack=` and `timed=` as `eri3_driver.eri3_program`'s (#606)."""
    _, _, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    includes, paths, calls, kernels = [], [], [], []
    for c, cls in enumerate(classes):
        build = (build_hmetric_kernel if engine is Engine.HGP
                 else build_metric_kernel)
        kern = (build((cls.lP, cls.lQ)) if engine is Engine.HGP
                else build((cls.lP, cls.lQ), lowstack=lowstack))
        _check_signature(kern)
        path = _unit(kern, backend, workdir, None)
        paths.append(path)
        ns = f"fj_m{c}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        kernels.append(f"(const void*){ns}::{kern.name}")
        calls.append(_call(kern, c, ns, cls, timed=timed))
    src = METRIC_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes), upload_int=UPLOAD_INT,
        n_c=len(classes), n_aux=n_aux, runs=RUNS, nd=ND, ni=NI,
        nr=", ".join(str(c.n_rec) for c in classes),
        np_=", ".join(str(c.n_pair) for c in classes),
        vram_kernels=", ".join(kernels), launches="\n".join(calls))
    if timed:
        src = timed_source(src, "M")
    main = workdir / f"metric_main.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def metric_stage(classes: Sequence[MetricClass]) -> np.ndarray:
    """The program's input: per class its double record arrays in parameter
    order (`A` row-major), then its int pair arrays."""
    parts: list = []
    for cls in classes:
        arr = cls.arrays()
        for name in MetricClass.RECORD_ARRAYS:
            parts.append(np.ravel(np.asarray(arr[name], dtype=np.float64)))
        for name in MetricClass.PAIR_ARRAYS:
            parts.append(np.asarray(arr[name], dtype=np.float64))
    return np.concatenate(parts)


def build_on_card(classes: Sequence[MetricClass], n_aux: int, backend: str,
                  tc: Toolchain, workdir: Path,
                  run_device_binary, *,
                  engine: Engine = Engine.MMD, lowstack: bool | None = None,
                  timeout: float = 1800.0) -> np.ndarray:
    """`(P|Q)` of `classes` built on the card, `[run][n_aux, n_aux]` (`RUNS`
    runs). The caller holds the card (its device fixture)."""
    prog = metric_program(classes, n_aux, backend, workdir, engine=engine,
                          lowstack=lowstack)
    binary = compile_checked(tc, prog, workdir / "metric_bin",
                             timeout=timeout)
    src, dst = workdir / "metric.in", workdir / "metric.out"
    metric_stage(classes).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    want = (f"fj-metric classes {len(classes)} launches "
            f"{len(classes) * RUNS} runs {RUNS}")
    assert want in out, out
    raw = np.fromfile(dst, dtype=np.float64)
    assert raw.size == RUNS * n_aux * n_aux, (raw.size, RUNS, n_aux)
    return raw.reshape(RUNS, n_aux, n_aux)


def compile_timed(classes: Sequence[MetricClass], n_aux: int, backend: str,
                  tc: Toolchain, workdir: Path, *, lowstack: bool | None,
                  timeout: float) -> Path:
    """The TIMED program of `classes` in one form, compiled."""
    prog = metric_program(classes, n_aux, backend, workdir,
                          lowstack=lowstack, timed=True)
    return compile_checked(tc, prog, workdir / "metric_timed",
                           timeout=timeout)


def run_timed(binary: Path, classes: Sequence[MetricClass], n_aux: int,
              backend: str, workdir: Path, run_device_binary, *,
              warmup: int, reps: int) -> tuple[np.ndarray, dict]:
    """The last run's matrix and `timings` of one run of a timed binary."""
    src, dst = workdir / "metric_timed.in", workdir / "metric_timed.out"
    if not src.exists():
        metric_stage(classes).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), str(warmup),
                            str(reps), vram=backend)
    raw = np.fromfile(dst, dtype=np.float64)
    assert raw.size == n_aux * n_aux, (raw.size, n_aux)
    return raw.reshape(n_aux, n_aux), timings(out)


# Every class into one matrix, RUNS times.
METRIC_PROGRAM = r"""// generated by test/metric_driver.py: harness
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
static const int NC = {n_c}, NAUX = {n_aux}, RUNS = {runs}, ND = {nd}, NI = {ni};
static const int nr[NC] = {{{nr}}};  // aux records per class
static const int np[NC] = {{{np_}}};  // aux shell pairs per class

// usage: in out. `in` is, per class, its double record arrays (ND per
// record) then its int pair arrays (NI per pair); `out` is each run's
// matrix [NAUX, NAUX], each from a zeroed matrix.
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    size_t want = 0;
    for (int c = 0; c < NC; ++c) want += (size_t)ND * nr[c] + (size_t)NI * np[c];
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the classes want %zu\n",
                     in.size(), want);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double)
                          + (size_t)NAUX * NAUX * sizeof(double) + (size_t)64 * NC,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data();
    double* dd[NC];
    int* di[NC];
    for (int c = 0; c < NC; ++c) {{
        dd[c] = upload(h, (size_t)ND * nr[c]);
        h += (size_t)ND * nr[c];
        di[c] = upload_int(h, (size_t)NI * np[c]);
        h += (size_t)NI * np[c];
        if (!dd[c] || !di[c]) {{
            std::fprintf(stderr, "upload %d failed\n", c);
            return 1;
        }}
    }}
    double* dM = nullptr;
    CHECK(devMalloc(&dM, (size_t)NAUX * NAUX * sizeof(double)));
    std::vector<double> M((size_t)NAUX * NAUX), out;
    int launches = 0;
    for (int r = 0; r < RUNS; ++r) {{
        CHECK(devMemset(dM, 0, M.size() * sizeof(double)));
{launches}
        CHECK(devSynchronize());
        CHECK(devMemcpy(M.data(), dM, M.size() * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), M.begin(), M.end());
    }}
    for (int c = 0; c < NC; ++c) {{
        CHECK(devFree(dd[c]));
        CHECK(devFree(di[c]));
    }}
    CHECK(devFree(dM));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-metric classes %d launches %d runs %d\n", NC, launches, RUNS);
    return 0;
}}
"""
