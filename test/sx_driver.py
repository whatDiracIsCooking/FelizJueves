"""The cross-basis overlap `S_{B1,B2}` built ON A CARD from its launch --
harness.

    sx_program(launch, backend, workdir, units=) -> Program
    sx_stage(launch)                             -> np.ndarray
    build_on_card(launch, backend, tc, workdir, run_device_binary,
                  units=)                        -> [run] -> S[n_a, n_b]

The launch is `launch.cross_overlap_launch`'s value, the records
`XClass.arrays()` in the kernel's own parameter order, and the kernels
`kernel_sx.build_sx_kernel`'s, emitted by `emit/cuda.py` / `emit/hip.py`
unchanged; only the `main()` around them is written here, as
`one_electron_driver.py` does for `S`, `T` and `V` -- harness, in `test/`,
not a second emitter. Compiled with devtools/config.sh's flags under #92's
warning filter (`plan_driver.compile_checked`).

WHAT THE GENERATED `main()` DOES, `RUNS` times: zero ONE `n_a x n_b`
matrix, launch every ordered class's kernel into it -- one launch per class,
one thread per record (`launch_shape`, `THREADS` lanes) -- then one
synchronize and the read-back. Every element is owned by exactly one
ordered shell pair, whose records add into it atomically.

THE UPLOADS, once: per class one buffer of its double record arrays
(`zeta_ab | XPA | XPB | s`, each row-major) and one of its two int offsets
(`off_a | off_b`). A kernel argument is a pointer into those buffers, read
off the kernel's own parameter list.

THE UNITS. `units` maps a kernel name to a file already written (the kernel
tier's generate phase, when `KERNEL_GEN_CMD` asks for `--sx-l`; it does not
today); any other class is emitted into `workdir` here, the same text.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from conftest import Toolchain
from plan_driver import RUNS, UPLOAD_INT, Program, _unit, compile_checked
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.kernel_sx import build_sx_kernel, launch_shape
from felizjueves.launch import XClass, XLaunch

THREADS = 32  # lanes per block, one record per thread; not numerical
WIDTH = {"zeta_ab": 1, "XPA": 3, "XPB": 3, "s": 1}
ND = sum(WIDTH.values())  # doubles per record
INTS = ("off_a", "off_b")


def _check_signature(kern: Kernel, cls: XClass) -> None:
    names = [p.name for p in kern.params]
    want = ["n_rec", "n_a", "n_b", *cls.arrays(), "S"]
    if (names != want or list(cls.arrays())[:len(WIDTH)] != list(WIDTH)
            or kern.params[-1].intent is not Intent.OUT):
        raise ValueError(f"{kern.name}: signature {names}, the launch's "
                         f"records want {want}")


def _call(kern: Kernel, c: int, ns: str, cls: XClass) -> str:
    n = len(cls.recs)
    at, off = {}, 0
    for name, width in WIDTH.items():
        at[name] = f"dd[{c}] + {off}" if off else f"dd[{c}]"
        off += width * n
    at.update({"off_a": f"di[{c}]", "off_b": f"di[{c}] + {n}",
               "n_rec": str(n), "n_a": "NA", "n_b": "NB"})
    args = []
    for p in kern.params:
        if p.intent is Intent.OUT:
            args.append("dM")
        elif p.array or p.type is Scalar.I32:
            args.append(at[p.name])
        else:
            raise ValueError(f"{kern.name}: no argument for {p.name}")
    grid, (lanes, _) = launch_shape(n, THREADS)
    return (f"        // ordered class ({cls.la}, {cls.lb}), {n} records\n"
            f"        {ns}::{kern.name}<<<dim3({grid}), dim3({lanes}, 1)>>>(\n"
            f"            {', '.join(args)});\n"
            f"        CHECK(devGetLastError());\n"
            f"        ++launches;")


def sx_program(launch: XLaunch, backend: str, workdir: Path, *,
               units: Mapping[str, Path] | None = None) -> Program:
    """The `main()` building `S_{B1,B2}` from every class of `launch`."""
    workdir.mkdir(parents=True, exist_ok=True)
    _, _, ext = BACKEND[backend]
    includes, paths, calls, kernels = [], [], [], []
    for c, cls in enumerate(launch):
        kern = build_sx_kernel(cls.spec())
        _check_signature(kern, cls)
        path = _unit(kern, backend, workdir, units)
        paths.append(path)
        ns = f"fj_sx{c}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        kernels.append(f"(const void*){ns}::{kern.name}")
        calls.append(_call(kern, c, ns, cls))
    src = SX_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes), upload_int=UPLOAD_INT,
        n_c=len(launch), n_a=launch.n_a, n_b=launch.n_b, runs=RUNS, nd=ND,
        nr=", ".join(str(len(cls.recs)) for cls in launch),
        vram_kernels=", ".join(kernels), launches="\n".join(calls))
    main = workdir / f"sx_main.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def sx_stage(launch: XLaunch) -> np.ndarray:
    """The program's input: per class its double record arrays in parameter
    order, then `off_a | off_b`."""
    parts: list = []
    for cls in launch:
        arr = cls.arrays()
        for name in WIDTH:
            parts.append(np.ravel(np.asarray(arr[name], dtype=np.float64)))
        parts += [arr["off_a"], arr["off_b"]]
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])


def build_on_card(launch: XLaunch, backend: str, tc: Toolchain, workdir: Path,
                  run_device_binary, *,
                  units: Mapping[str, Path] | None = None) -> np.ndarray:
    """`S_{B1,B2}` of `launch` built on the card, `[run][n_a, n_b]`. The
    caller holds the card (its device fixture); the binary runs in
    VRAM-probe mode first (`run_device_binary(vram=)`)."""
    prog = sx_program(launch, backend, workdir, units=units)
    binary = compile_checked(tc, prog, workdir / "sx_bin")
    src, dst = workdir / "sx.in", workdir / "sx.out"
    sx_stage(launch).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    want = (f"fj-sx classes {len(launch)} launches {len(launch) * RUNS} "
            f"runs {RUNS}")
    assert want in out, out
    raw = np.fromfile(dst, dtype=np.float64)
    assert raw.size == RUNS * launch.n_a * launch.n_b, raw.size
    return raw.reshape(RUNS, launch.n_a, launch.n_b)


SX_PROGRAM = r"""// generated by test/sx_driver.py: harness, the cross-basis overlap
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
static const int NC = {n_c}, NA = {n_a}, NB = {n_b}, RUNS = {runs}, ND = {nd};
static const int nr[NC] = {{{nr}}};  // records per ordered class

// usage: in out. `in` is, per class, its double record arrays (ND per
// record) and off_a off_b; `out` is each run's matrix [NA, NB], each from a
// zeroed matrix.
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    size_t want = 0;
    for (int c = 0; c < NC; ++c) want += (size_t)(ND + 2) * nr[c];
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the launch wants %zu\n",
                     in.size(), want);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double)
                          + (size_t)NA * NB * sizeof(double) + (size_t)64 * NC,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data();
    double* dd[NC];
    int* di[NC];
    for (int c = 0; c < NC; ++c) {{
        dd[c] = upload(h, (size_t)ND * nr[c]);
        h += (size_t)ND * nr[c];
        di[c] = upload_int(h, (size_t)2 * nr[c]);
        h += (size_t)2 * nr[c];
        if (!dd[c] || !di[c]) {{
            std::fprintf(stderr, "upload %d failed\n", c);
            return 1;
        }}
    }}
    double* dM = nullptr;
    CHECK(devMalloc(&dM, (size_t)NA * NB * sizeof(double)));
    std::vector<double> M((size_t)NA * NB), out;
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
    std::printf("fj-sx classes %d launches %d runs %d\n", NC, launches, RUNS);
    return 0;
}}
"""
