"""The Schwarz matrix of a basis built ON A CARD -- harness.

    schwarz_program(classes, n_ao, backend, workdir) -> Program
    schwarz_stage(classes)                           -> np.ndarray
    build_on_card(classes, n_ao, backend, tc, workdir,
                  run_device_binary)                 -> [run] -> Q

The records are `ao.schwarz.schwarz_classes`' (every shell pair staged with
`la <= lb`, one contiguous record range per pair), the kernels
`kernel_schwarz.build_schwarz_kernel`'s, emitted by `emit/cuda.py` /
`emit/hip.py` unchanged. Only the `main()` around them is written here, as
`one_electron_driver.py` does for `S`, `T` and `V` -- harness, in `test/`,
not a second emitter. Every unit is emitted into `workdir` (the kernel tier's
`KERNEL_GEN_CMD` writes no Schwarz unit), and the program is compiled with
devtools/config.sh's flags under #92's warning filter
(`plan_driver.compile_checked`).

WHAT THE GENERATED `main()` DOES, `RUNS` times: zero ONE `n_ao x n_ao`
matrix, launch every class's kernel into it -- one launch per pair class, one
thread per shell pair (`launch_shape`, `THREADS` lanes) -- then one
synchronize and the read-back. Each element is owned by one shell pair and
written by a plain store, so every run must be bit-identical.

THE UPLOADS, once: per class one buffer of its double record arrays
(`zeta_ab | P | XPA | XPB | s_ab`, each row-major) and one of its four int
pair arrays (`rec0 | rec1 | off_a | off_b`). A kernel argument is a pointer
into those buffers, read off the kernel's own parameter list, so the call
cannot drift from the signature.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from plan_driver import RUNS, UPLOAD_INT, Program, _unit, compile_checked
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.ao.schwarz import SchwarzClass
from felizjueves.ir import Intent, Kernel
from felizjueves.kernel_schwarz import OUT, build_schwarz_kernel, launch_shape

THREADS = 32  # lanes per block, one shell pair per thread; not numerical
WIDTH = {"zeta_ab": 1, "P": 3, "XPA": 3, "XPB": 3, "s_ab": 1}
ND = sum(WIDTH.values())  # doubles per record
NI = len(SchwarzClass.PAIR_ARRAYS)  # ints per shell pair


def _check_signature(kern: Kernel) -> None:
    names = [p.name for p in kern.params]
    want = ["n_pair", "n_rec", "n_ao", *SchwarzClass.RECORD_ARRAYS,
            *SchwarzClass.PAIR_ARRAYS, OUT]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the records "
                         f"want {want}")


def _call(kern: Kernel, c: int, ns: str, cls: SchwarzClass) -> str:
    at, off = {}, 0
    for name in SchwarzClass.RECORD_ARRAYS:
        at[name] = f"dd[{c}] + {off}" if off else f"dd[{c}]"
        off += WIDTH[name] * cls.n_rec
    for k, name in enumerate(SchwarzClass.PAIR_ARRAYS):
        at[name] = f"di[{c}] + {k * cls.n_pair}" if k else f"di[{c}]"
    at.update({"n_pair": str(cls.n_pair), "n_rec": str(cls.n_rec),
               "n_ao": "NAO", OUT: "dQ"})
    args = [at[p.name] for p in kern.params]
    grid, (lanes, _) = launch_shape(cls.n_pair, THREADS)
    return (f"        // pair class ({cls.la}, {cls.lb}), {cls.n_pair} "
            f"pairs, {cls.n_rec} records\n"
            f"        {ns}::{kern.name}<<<dim3({grid}), dim3({lanes}, 1)>>>(\n"
            f"            {', '.join(args)});\n"
            f"        CHECK(devGetLastError());\n"
            f"        ++launches;")


def schwarz_program(classes: Sequence[SchwarzClass], n_ao: int, backend: str,
                    workdir: Path) -> Program:
    """The `main()` building `Q` from every class in `classes`."""
    _, _, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    includes, paths, calls, kernels = [], [], [], []
    for c, cls in enumerate(classes):
        kern = build_schwarz_kernel((cls.la, cls.lb))
        _check_signature(kern)
        path = _unit(kern, backend, workdir, None)
        paths.append(path)
        ns = f"fj_q{c}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        kernels.append(f"(const void*){ns}::{kern.name}")
        calls.append(_call(kern, c, ns, cls))
    src = SCHWARZ_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes), upload_int=UPLOAD_INT,
        n_c=len(classes), n_ao=n_ao, runs=RUNS, nd=ND, ni=NI,
        nr=", ".join(str(c.n_rec) for c in classes),
        np_=", ".join(str(c.n_pair) for c in classes),
        vram_kernels=", ".join(kernels), launches="\n".join(calls))
    main = workdir / f"schwarz_main.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def schwarz_stage(classes: Sequence[SchwarzClass]) -> np.ndarray:
    """The program's input: per class its double record arrays in parameter
    order, then its int pair arrays."""
    parts: list = []
    for cls in classes:
        arr = cls.arrays()
        for name in SchwarzClass.RECORD_ARRAYS:
            parts.append(np.ravel(np.asarray(arr[name], dtype=np.float64)))
        for name in SchwarzClass.PAIR_ARRAYS:
            parts.append(np.asarray(arr[name], dtype=np.float64))
    return np.concatenate(parts)


def build_on_card(classes: Sequence[SchwarzClass], n_ao: int, backend: str,
                  tc: Toolchain, workdir: Path,
                  run_device_binary) -> np.ndarray:
    """`Q` of `classes` built on the card, `[run][n_ao, n_ao]` (`RUNS`
    runs). The caller holds the card (its device fixture)."""
    prog = schwarz_program(classes, n_ao, backend, workdir)
    binary = compile_checked(tc, prog, workdir / "schwarz_bin")
    src, dst = workdir / "schwarz.in", workdir / "schwarz.out"
    schwarz_stage(classes).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    want = (f"fj-schwarz classes {len(classes)} launches "
            f"{len(classes) * RUNS} runs {RUNS}")
    assert want in out, out
    raw = np.fromfile(dst, dtype=np.float64)
    assert raw.size == RUNS * n_ao * n_ao, (raw.size, RUNS, n_ao)
    return raw.reshape(RUNS, n_ao, n_ao)


# Every class into one matrix, RUNS times.
SCHWARZ_PROGRAM = r"""// generated by test/schwarz_driver.py: harness
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
static const int NC = {n_c}, NAO = {n_ao}, RUNS = {runs}, ND = {nd}, NI = {ni};
static const int nr[NC] = {{{nr}}};  // records per pair class
static const int np[NC] = {{{np_}}};  // shell pairs per pair class

// usage: in out. `in` is, per class, its double record arrays (ND per
// record) then its int pair arrays (NI per pair); `out` is each run's
// matrix [NAO, NAO], each from a zeroed matrix.
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
                          + (size_t)NAO * NAO * sizeof(double) + (size_t)64 * NC,
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
    double* dQ = nullptr;
    CHECK(devMalloc(&dQ, (size_t)NAO * NAO * sizeof(double)));
    std::vector<double> Q((size_t)NAO * NAO), out;
    int launches = 0;
    for (int r = 0; r < RUNS; ++r) {{
        CHECK(devMemset(dQ, 0, Q.size() * sizeof(double)));
{launches}
        CHECK(devSynchronize());
        CHECK(devMemcpy(Q.data(), dQ, Q.size() * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), Q.begin(), Q.end());
    }}
    for (int c = 0; c < NC; ++c) {{
        CHECK(devFree(dd[c]));
        CHECK(devFree(di[c]));
    }}
    CHECK(devFree(dQ));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-schwarz classes %d launches %d runs %d\n", NC, launches, RUNS);
    return 0;
}}
"""
