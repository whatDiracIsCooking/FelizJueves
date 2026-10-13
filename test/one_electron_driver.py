"""S, T or V of a basis built ON A CARD from its one-electron launch (#242)
-- harness.

    one_e_program(launch, target, n_ao, backend, workdir, units=) -> Program
    one_e_stage(launch, target)                                 -> np.ndarray
    one_e_read_back(raw, n_ao)                                  -> [run] -> M
    build_on_card(launch, target, n_ao, backend, tc, workdir,
                  run_device_binary, units=)                    -> [run] -> M
    tier_units(backend, kernels)                                -> {name: path}

THE ONE FUNCTION a later test reuses (#243's `h = T + V` capstone) is
`build_on_card`: it compiles the program for one target of one launch with
devtools/config.sh's flags under #92's warning filter
(`plan_driver.compile_checked`), stages the launch's records, runs the
binary (behind the caller's card lock and the VRAM probe) and returns each
run's matrix. Nothing else is decided here: the launch is
`launch.one_electron_launch`'s value, the records are `OneEClass.arrays
(target)` in the kernel's own parameter order, and the kernels are
`kernel_s` / `kernel_t` / `kernel_v`'s, emitted by `emit/cuda.py` /
`emit/hip.py` unchanged. Only the `main()` around them is written here, as
`plan_driver.py` does for `J` -- harness, in `test/`, not a second emitter.

WHAT THE GENERATED `main()` DOES, `RUNS` times (so the fp64 atomics'
run-to-run spread is witnessed, not assumed): zero ONE `n_ao x n_ao` matrix,
launch every class's kernel of `target` into it -- one launch per pair
class, one thread per record (`launch_shape`, `THREADS` lanes) -- then one
synchronize and the read-back. Every class ADDS (each kernel's scatter is
an `AtomicAdd`, kernel_jab's rule), so several classes into one matrix is
the kernels' own contract: an element is owned by exactly one shell pair.

THE UPLOADS, once before the first run: per class one buffer of its
double-valued record arrays (`arrays(target)` minus the offsets, each
row-major, concatenated in parameter order) and one of its two `int`
offsets; for `V`, the launch's nuclei `Z[n_nuc]` and `Cnuc[n_nuc, 3]`, once.
A kernel argument is a pointer into those buffers, read off the kernel's
own parameter list, so the call cannot drift from the signature.

THE UNITS. `units` maps a kernel name to a file already written -- under
devtools/kernel-tier.sh, the generate phase's own (`tier_units`, which
asserts each byte-identical to `emit(...)`); any class without one (a class
`la > lb` the ladder does not write, or a run outside the tier) is emitted
into `workdir` here, the same text.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from plan_driver import RUNS, UPLOAD_INT, Program, _unit, compile_checked
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.kernel_s import build_s_kernel
from felizjueves.kernel_s import launch_shape as s_launch_shape
from felizjueves.kernel_t import build_t_kernel
from felizjueves.kernel_t import launch_shape as t_launch_shape
from felizjueves.kernel_v import build_v_kernel
from felizjueves.kernel_v import launch_shape as v_launch_shape
from felizjueves.launch import OneEClass, OneELaunch
from felizjueves.spec import Target

THREADS = 32  # lanes per block, one record per thread; not numerical

BUILD: dict[Target, Callable[[tuple[int, int]], Kernel]] = {
    Target.S: build_s_kernel, Target.T: build_t_kernel,
    Target.V: build_v_kernel}
SHAPE = {Target.S: s_launch_shape, Target.T: t_launch_shape,
         Target.V: v_launch_shape}
INTS = ("off_a", "off_b")  # the int record arrays, last in every block
NUCLEI = ("Z", "Cnuc")  # V's per-nucleus arrays, the launch's


def kernel_of(cls: OneEClass, target: Target) -> Kernel:
    """The class's kernel of `target`: its `spec(target)` built."""
    return BUILD[target](cls.spec(target))


def _doubles(cls: OneEClass, target: Target) -> list[tuple[str, int]]:
    """`(name, width)` of each double-valued record array, parameter order."""
    return [(k, len(v[0]) if isinstance(v[0], list) else 1)
            for k, v in cls.arrays(target).items() if k not in INTS]


def _check_signature(kern: Kernel, cls: OneEClass, target: Target) -> None:
    """The kernel's parameters are the scalars, the record arrays in
    `arrays(target)`'s order, V's nuclei, and the one OUT matrix."""
    names = [p.name for p in kern.params]
    out = [p.name for p in kern.params if p.intent is Intent.OUT]
    scalars = ["n_rec", "n_ao"] + (["n_nuc"] if target is Target.V else [])
    want = [*scalars, *cls.arrays(target),
            *(NUCLEI if target is Target.V else ()), *out]
    if names != want or len(out) != 1:
        raise ValueError(f"{kern.name}: signature {names}, the launch's "
                         f"records want {want}")


def _call(kern: Kernel, c: int, ns: str, cls: OneEClass,
          target: Target) -> str:
    """The launch of class `c`'s kernel, its arguments read off the kernel's
    own parameters (module docstring, THE UPLOADS)."""
    n = len(cls.recs)
    at, off = {}, 0
    for name, width in _doubles(cls, target):
        at[name] = f"dd[{c}] + {off}" if off else f"dd[{c}]"
        off += width * n
    at.update({"off_a": f"di[{c}]", "off_b": f"di[{c}] + {n}",
               "Z": "dZ", "Cnuc": "dC", "n_rec": str(n), "n_ao": "NAO",
               "n_nuc": "NNUC"})
    args = []
    for p in kern.params:
        if p.intent is Intent.OUT:
            args.append("dM")
        elif p.array or p.type is Scalar.I32:
            args.append(at[p.name])
        else:
            raise ValueError(f"{kern.name}: no argument for {p.name}")
    grid, (lanes, _) = SHAPE[target](n, THREADS)
    return (f"        // pair class ({cls.la}, {cls.lb}), {n} records\n"
            f"        {ns}::{kern.name}<<<dim3({grid}), dim3({lanes}, 1)>>>(\n"
            f"            {', '.join(args)});\n"
            f"        CHECK(devGetLastError());\n"
            f"        ++launches;")


def one_e_program(launch: OneELaunch, target: Target, n_ao: int,
                  backend: str, workdir: Path, *,
                  units: Mapping[str, Path] | None = None) -> Program:
    """The `main()` building `target`'s matrix from every class of `launch`
    (module docstring)."""
    if target not in BUILD:
        raise ValueError(f"{target!r} is not a one-electron target")
    _, arch, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    includes, paths, calls, kernels, nd = [], [], [], [], []
    for c, cls in enumerate(launch):
        kern = kernel_of(cls, target)
        _check_signature(kern, cls, target)
        path = _unit(kern, backend, workdir, units)
        paths.append(path)
        ns = f"fj_oe{c}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        kernels.append(f"(const void*){ns}::{kern.name}")
        calls.append(_call(kern, c, ns, cls, target))
        nd.append(sum(w for _, w in _doubles(cls, target)))
    n_nuc = len(launch.nuclei) if target is Target.V else 0
    src = ONE_E_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes), upload_int=UPLOAD_INT,
        target=target.name, n_c=len(launch), n_ao=n_ao, n_nuc=n_nuc, runs=RUNS,
        nr=", ".join(str(len(cls.recs)) for cls in launch),
        nd=", ".join(map(str, nd)), vram_kernels=", ".join(kernels),
        launches="\n".join(calls))
    main = workdir / f"one_e_{target.name.lower()}.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def one_e_stage(launch: OneELaunch, target: Target) -> np.ndarray:
    """The program's input: V's `Z | Cnuc` (nothing for S and T), then per
    class its double record arrays in parameter order, then `off_a | off_b`."""
    parts: list = []
    if target is Target.V:
        parts += [launch.Z, np.ravel(launch.Cnuc)]
    for cls in launch:
        arr = cls.arrays(target)
        for name, _ in _doubles(cls, target):
            parts.append(np.ravel(np.asarray(arr[name], dtype=np.float64)))
        parts += [arr["off_a"], arr["off_b"]]
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])


def one_e_read_back(raw: np.ndarray, n_ao: int) -> np.ndarray:
    """`[run]` -> the `n_ao x n_ao` matrix that run built."""
    assert raw.size == RUNS * n_ao * n_ao, (raw.size, RUNS, n_ao)
    return raw.reshape(RUNS, n_ao, n_ao)


def tier_units(backend: str, kernels: Sequence[Kernel]) -> dict[str, Path]:
    """Under the kernel tier, the generate phase's own file of every kernel
    in `kernels` it WROTE, each asserted byte-identical to `emit(...)`;
    outside the tier (or for another backend), nothing."""
    out = os.environ.get("KERNEL_TIER_OUT")
    if not out or os.environ.get("KERNEL_TIER_BACKEND") != backend:
        return {}
    emit, arch, ext = BACKEND[backend]
    got = {}
    for k in kernels:
        path = Path(out) / f"{k.name}.{ext}"
        if not path.exists():
            continue
        assert path.read_text() == emit(k, arch), (
            f"{path} differs from what emit() writes now -- a stale generate "
            "phase, or a nondeterministic emitter")
        got[k.name] = path
    return got


def build_on_card(launch: OneELaunch, target: Target, n_ao: int, backend: str,
                  tc: Toolchain, workdir: Path, run_device_binary, *,
                  units: Mapping[str, Path] | None = None) -> np.ndarray:
    """`target`'s matrix of `launch` built on the card, `[run][n_ao, n_ao]`
    (`RUNS` runs). The caller holds the card (its device fixture); the
    binary runs in VRAM-probe mode first (`run_device_binary(vram=)`)."""
    prog = one_e_program(launch, target, n_ao, backend, workdir, units=units)
    binary = compile_checked(tc, prog, workdir / f"{prog.path.stem}_bin")
    src, dst = workdir / f"{prog.path.stem}.in", workdir / f"{prog.path.stem}.out"
    one_e_stage(launch, target).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    want = (f"fj-one-e target {target.name} classes {len(launch)} launches "
            f"{len(launch) * RUNS} runs {RUNS}")
    assert want in out, out
    return one_e_read_back(np.fromfile(dst, dtype=np.float64), n_ao)


# One target of one launch, every class into one matrix, RUNS times.
ONE_E_PROGRAM = r"""// generated by test/one_electron_driver.py (#242): harness {target}
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
static const int NC = {n_c}, NAO = {n_ao}, NNUC = {n_nuc}, RUNS = {runs};
static const int nr[NC] = {{{nr}}};  // records per pair class
static const int nd[NC] = {{{nd}}};  // doubles per record

// usage: in out. `in` is Z[NNUC] Cnuc[NNUC, 3] (V only), then per class its
// double record arrays (nd per record) and off_a off_b; `out` is each run's
// matrix [NAO, NAO], each from a zeroed matrix.
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    size_t want = (size_t)4 * NNUC;
    for (int c = 0; c < NC; ++c) want += (size_t)(nd[c] + 2) * nr[c];
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the launch wants %zu\n",
                     in.size(), want);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double)
                          + (size_t)NAO * NAO * sizeof(double) + (size_t)64 * NC,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data();
    double* dZ = nullptr;
    double* dC = nullptr;
    if (NNUC) {{
        dZ = upload(h, (size_t)NNUC);
        h += NNUC;
        dC = upload(h, (size_t)3 * NNUC);
        h += (size_t)3 * NNUC;
        if (!dZ || !dC) {{ std::fprintf(stderr, "nuclei upload failed\n"); return 1; }}
    }}
    double* dd[NC];
    int* di[NC];
    for (int c = 0; c < NC; ++c) {{
        dd[c] = upload(h, (size_t)nd[c] * nr[c]);
        h += (size_t)nd[c] * nr[c];
        di[c] = upload_int(h, (size_t)2 * nr[c]);
        h += (size_t)2 * nr[c];
        if (!dd[c] || !di[c]) {{
            std::fprintf(stderr, "upload %d failed\n", c);
            return 1;
        }}
    }}
    double* dM = nullptr;
    CHECK(devMalloc(&dM, (size_t)NAO * NAO * sizeof(double)));
    std::vector<double> M((size_t)NAO * NAO), out;
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
    if (NNUC) {{
        CHECK(devFree(dZ));
        CHECK(devFree(dC));
    }}
    CHECK(devFree(dM));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-one-e target {target} classes %d launches %d runs %d\n",
                NC, launches, RUNS);
    return 0;
}}
"""
