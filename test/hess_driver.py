"""The contracted one-electron nuclear hessians built ON A CARD by the d2
kernels -- harness.

    HessLaunch(kernel, n_rec, arrays, scalars)          one kernel launch
    one_electron_launches(basis, nuclei, target, M)     -> [HessLaunch]
    hess_program(launches, n_atom, backend, workdir, units=) -> Program
    compile_hess(launches, n_atom, backend, tc, workdir, units=) -> HessBinary
    HessBinary.run(run_device_binary)          -> [run][n_atom][3][n_atom][3]

`grad_driver.py`'s pattern with the one OUT `H[n_atom, 3, n_atom, 3]` in
place of `G[n_atom, 3]`: every array argument of every launch is staged
(int arrays as exact doubles, converted on upload,
`plan_driver.UPLOAD_INT`), scalar arguments are spelled into the call,
and `H` is shared by every launch. The kernels are
`hessian.kernel_h1e`'s, emitted by `emit/cuda.py` / `emit/hip.py`
unchanged; a unit the tier's generate phase wrote (`--hess-l`) is that
file. Only the `main()` is written here.

WHAT THE GENERATED `main()` DOES, `RUNS` times (the fp64 atomics'
run-to-run spread witnessed, not assumed): zero ONE `H` (the kernels'
precondition, `kernel_h1e.ZEROED_H`), launch every kernel into it -- one
thread per record, `THREADS` lanes -- then one synchronize and the
read-back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from conftest import Toolchain
from plan_driver import RUNS, UPLOAD_INT, Program, _unit, compile_checked
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.gradient import host
from felizjueves.hessian.kernel_h1e import build_h1e_kernel, launch_shape
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.spec import Target

__all__ = ["HessBinary", "HessLaunch", "compile_hess", "hess_program",
           "one_electron_launches"]

THREADS = 32  # lanes per block, one record per thread; not numerical
H = "H"


@dataclass(frozen=True)
class HessLaunch:
    """One kernel over `n_rec` records: its array arguments by parameter
    name (every array parameter but `H`) and its scalar ones."""

    kernel: Kernel
    n_rec: int
    arrays: Mapping[str, np.ndarray]
    scalars: Mapping[str, int] = field(default_factory=dict)
    out: str = H  # the one OUT, shared by every launch (`hdq_`'s is `D1`)

    def __post_init__(self) -> None:
        want = {p.name for p in self.kernel.params if p.array} - {self.out}
        if set(self.arrays) != want:
            raise ValueError(f"{self.kernel.name}: arrays {sorted(self.arrays)}"
                             f", the kernel takes {sorted(want)}")
        out = [p for p in self.kernel.params if p.intent is Intent.OUT]
        if [p.name for p in out] != [self.out]:
            raise ValueError(f"{self.kernel.name}: the one OUT is not "
                             f"{self.out}")


def one_electron_launches(basis, nuclei, target: Target,
                          M) -> list[HessLaunch]:
    """One `d2s_` / `d2t_` / `d2v_` launch per ordered-by-`i <= j` pair
    class of `basis`, every record of the class."""
    out = []
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        kern = build_h1e_kernel((target, la, lb))
        rec = host.records(basis, nuclei, la, lb)
        types = {p.name: p.type for p in kern.params if p.array}
        arrs = {k: np.asarray(v, dtype=np.int32 if types[k] is Scalar.I32
                              else np.float64)
                for k, v in rec.items() if k in types}
        arrs["M"] = np.asarray(M, dtype=np.float64)
        if target is Target.V:
            arrs["Z"] = np.array([z for z, _ in nuclei], dtype=np.float64)
            arrs["Cnuc"] = np.array([c for _, c in nuclei], dtype=np.float64)
        n = len(rec["s_ab"])
        out.append(HessLaunch(kern, n, arrs, {
            "n_rec": n, "n_ao": basis.n_ao, "n_atom": len(nuclei)}))
    return out


def _buffers(launches: Sequence[HessLaunch]):
    """Every staged array, launch by launch in parameter order:
    `(launch, name, flat doubles, is_int)`."""
    for c, x in enumerate(launches):
        for p in x.kernel.params:
            if p.array and p.name != x.out:
                a = np.asarray(x.arrays[p.name])
                yield c, p.name, a.astype(np.float64).ravel(), (
                    p.type is Scalar.I32)


def _call(x: HessLaunch, c: int, ns: str, slot: Mapping[tuple[int, str], int]
          ) -> str:
    args = []
    for p in x.kernel.params:
        if p.name == x.out:
            args.append("dH")
        elif p.array:
            t = "int" if p.type is Scalar.I32 else "double"
            args.append(f"(const {t}*)buf[{slot[c, p.name]}]")
        else:
            args.append(str(x.scalars[p.name]))
    grid, (bx, by) = launch_shape(x.n_rec, THREADS)
    return (f"        // {x.kernel.name}, {x.n_rec} records\n"
            f"        {ns}::{x.kernel.name}<<<dim3({grid}), dim3({bx}, {by})>>>(\n"
            f"            {', '.join(args)});\n"
            f"        CHECK(devGetLastError());\n"
            f"        ++launches;")


def hess_program(launches: Sequence[HessLaunch], n_atom: int, backend: str,
                 workdir: Path, *, units: Mapping[str, Path] | None = None,
                 n_out: int | None = None) -> Program:
    """The `main()` building `H` from every launch (module docstring);
    `n_out` doubles of OUT instead of `H`'s `9 n_atom^2` when given."""
    _, _, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    includes, paths, calls, kernels = [], [], [], []
    slot: dict[tuple[int, str], int] = {}
    sizes, ints = [], []
    for c, name, flat, is_int in _buffers(launches):
        slot[c, name] = len(sizes)
        sizes.append(str(flat.size))
        ints.append("1" if is_int else "0")
    seen: dict[str, str] = {}
    for c, x in enumerate(launches):
        ns = f"fj_h{c}"
        if x.kernel.name not in seen:
            path = _unit(x.kernel, backend, workdir, units)
            paths.append(path)
            includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
            kernels.append(f"(const void*){ns}::{x.kernel.name}")
            seen[x.kernel.name] = ns
        calls.append(_call(x, c, seen[x.kernel.name], slot))
    src = HESS_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes),
        upload_int=UPLOAD_INT, n_buf=len(sizes), runs=RUNS,
        n_out=9 * n_atom * n_atom if n_out is None else n_out,
        sizes=", ".join(sizes), ints=", ".join(ints),
        vram_kernels=", ".join(kernels), launches="\n".join(calls))
    main = workdir / f"hess.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def hess_stage(launches: Sequence[HessLaunch]) -> np.ndarray:
    """The program's input: every staged array, flat, in `_buffers` order."""
    return np.concatenate([f for _, _, f, _ in _buffers(launches)])


@dataclass(frozen=True)
class HessBinary:
    """One compiled hessian program; `run` returns
    `[run][n_atom][3][n_atom][3]`, or `[run][n_out]` given `n_out`."""

    launches: tuple[HessLaunch, ...]
    n_atom: int
    backend: str
    binary: Path
    workdir: Path
    n_out: int | None = None

    def run(self, run_device_binary) -> np.ndarray:
        src, dst = self.workdir / "hess.in", self.workdir / "hess.out"
        hess_stage(self.launches).tofile(src)
        out = run_device_binary(self.binary, str(src), str(dst),
                                vram=self.backend)
        want = f"fj-hess launches {len(self.launches) * RUNS} runs {RUNS}"
        assert want in out, out
        raw = np.fromfile(dst, dtype=np.float64)
        if self.n_out is not None:
            assert raw.size == RUNS * self.n_out, raw.size
            return raw.reshape(RUNS, self.n_out)
        n = self.n_atom
        assert raw.size == RUNS * 9 * n * n, raw.size
        return raw.reshape(RUNS, n, 3, n, 3)


def compile_hess(launches: Sequence[HessLaunch], n_atom: int, backend: str,
                 tc: Toolchain, workdir: Path, *,
                 units: Mapping[str, Path] | None = None,
                 n_out: int | None = None) -> HessBinary:
    """Compile the program with devtools/config.sh's flags, FAILING on a
    warning located in generated text (#92)."""
    prog = hess_program(launches, n_atom, backend, workdir, units=units,
                        n_out=n_out)
    binary = compile_checked(tc, prog, workdir / "hess_bin")
    return HessBinary(tuple(launches), n_atom, backend, binary, workdir,
                      n_out)


# Every launch into one H, RUNS times.
HESS_PROGRAM = r"""// generated by test/hess_driver.py: harness, the d2 hessian kernels
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
static const int NB = {n_buf}, RUNS = {runs};
static const size_t sz[NB] = {{{sizes}}};  // doubles per staged array
static const int is_int[NB] = {{{ints}}};  // 1: converted to int on upload

// usage: in out. `in` is every staged array, flat, in order; `out` is each
// run's OUT (H [n_atom, 3, n_atom, 3] unless sized otherwise), each zeroed.
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    size_t want = 0;
    for (int b = 0; b < NB; ++b) want += sz[b];
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the launches want %zu\n",
                     in.size(), want);
        return 2;
    }}
    const size_t nh = (size_t){n_out};  // doubles of the one OUT
    if (fj_vram_probe(in.size() * sizeof(double) + nh * sizeof(double)
                          + (size_t)16 * NB,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data();
    void* buf[NB];
    for (int b = 0; b < NB; ++b) {{
        buf[b] = is_int[b] ? (void*)upload_int(h, sz[b]) : (void*)upload(h, sz[b]);
        h += sz[b];
        if (!buf[b]) {{ std::fprintf(stderr, "upload %d failed\n", b); return 1; }}
    }}
    double* dH = nullptr;
    CHECK(devMalloc(&dH, nh * sizeof(double)));
    std::vector<double> Hh(nh), out;
    int launches = 0;
    for (int r = 0; r < RUNS; ++r) {{
        CHECK(devMemset(dH, 0, nh * sizeof(double)));
{launches}
        CHECK(devSynchronize());
        CHECK(devMemcpy(Hh.data(), dH, nh * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), Hh.begin(), Hh.end());
    }}
    for (int b = 0; b < NB; ++b) CHECK(devFree(buf[b]));
    CHECK(devFree(dH));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-hess launches %d runs %d\n", launches, RUNS);
    return 0;
}}
"""
