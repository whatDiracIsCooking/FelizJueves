"""The exchange-energy gradient `gk_` ON A CARD, one `G` per launch, every
unit compiled SEPARATELY under a per-unit budget -- harness (#618).

    GkLaunch(kernel, n_bra, n_ket, arrays)         one class launch
    gk_class_launches(basis, nuclei, D, classes, lowstack=) -> [GkLaunch]
    compile_gk(launches, n_atom, backend, tc, workdir, budget=, jobs=)
        -> GkBinary
    GkBinary.run(run_device_binary)                -> [run][launch][n_atom][3]

`grad_driver.py`'s program puts every unit into ONE translation unit,
which at CO/cc-pVTZ (256 K classes) no compiler finishes within the
user's per-compile budget. Here each kernel unit is its own translation
unit, `<name>.<ext>` with a two-function host wrapper appended by a
SEPARATE file that includes it (`fjl_<name>`: the launch, `fjk_<name>`:
the kernel's address for the VRAM probe), compiled `-c` through the
compiler cache (test/devcache.py) under `budget` seconds (default
`UNIT_BUDGET_S = 10`, the user's #618 budget: a unit over it FAILS, never
retries with more); the `main()` is another, and the objects are linked.
Every array argument of every launch is staged (int arrays as exact
doubles, converted on upload, `plan_driver.UPLOAD_INT`); launch `i`
writes its own `G_i[n_atom, 3]` slice (zeroed every run, `kernel_gk`'s
precondition), so a class's `G` is read back alone, and the whole `dE_K`
is the sum of the slices. One warp of `THREADS` lanes per bra record
(`kernel_gk.launch_shape`). `RUNS` runs witness the atomics' spread.

Two launches may run the same class in its two forms (`lowstack=`) on the
same staged arrays: the on-card differential.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import devcache
import numpy as np
from conftest import Toolchain
from plan_driver import FLAGS_VAR, RUNS, UPLOAD_INT, generated_warnings
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.gradient import host
from felizjueves.gradient.kernel_gk import build_gk_kernel, launch_shape
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.performance import measured

__all__ = ["GkBinary", "GkLaunch", "UNIT_BUDGET_S", "compile_gk",
           "gk_class_launches"]

THREADS = 32  # lanes per warp, one warp per bra record; not numerical
UNIT_BUDGET_S = 10.0  # the user's per-compile budget for #618
G = "G"


@dataclass(frozen=True)
class GkLaunch:
    """One `gk_` launch over a class's every bra and ket record."""

    kernel: Kernel
    n_bra: int
    n_ket: int
    n_ao: int
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        want = {p.name for p in self.kernel.params if p.array} - {G}
        if set(self.arrays) != want:
            raise ValueError(f"{self.kernel.name}: arrays {sorted(self.arrays)}"
                             f", the kernel takes {sorted(want)}")
        out = [p.name for p in self.kernel.params if p.intent is Intent.OUT]
        if out != [G]:
            raise ValueError(f"{self.kernel.name}: the one OUT is not G")


def gk_class_launches(basis, nuclei, D, classes, *,
                      lowstack: bool | None = None) -> list[GkLaunch]:
    """One launch per K class of `classes` (`(la, lc, lb, ld)`), each in
    `build_gk_kernel`'s form `lowstack` (None: the default form)."""
    out = []
    M = np.asarray(D, dtype=np.float64)
    for am in classes:
        kern = build_gk_kernel(am, lowstack=lowstack)
        bra, ket = host.exchange_records(basis, nuclei, am)
        types = {p.name: p.type for p in kern.params if p.array}
        arrs = {k: np.asarray(v, dtype=np.int32 if types[k] is Scalar.I32
                              else np.float64)
                for k, v in {**bra, **ket}.items()}
        arrs["M"] = M
        out.append(GkLaunch(kern, len(bra["s_ac"]), len(ket["s_bd"]),
                            basis.n_ao, arrs))
    return out


def _ctype(p) -> str:
    base = "int" if p.type is Scalar.I32 else "double"
    if not p.array:
        return base
    return f"{base}*" if p.intent is Intent.OUT else f"const {base}*"


def _wrapper(kern: Kernel, unit: str) -> str:
    """The unit's host wrapper TU: `fjl_<name>(grid, lanes, args...)` and
    `fjk_<name>()`, the kernel's address for the VRAM probe."""
    sig = ", ".join(f"{_ctype(p)} {p.name}" for p in kern.params)
    args = ", ".join(p.name for p in kern.params)
    return (f'#include "{unit}"\n'
            f'extern "C" void fjl_{kern.name}(int grid, int lanes, {sig}) {{\n'
            f"    {kern.name}<<<dim3(grid), dim3(1, lanes)>>>({args});\n"
            "}\n"
            f'extern "C" const void* fjk_{kern.name}() {{\n'
            f"    return (const void*){kern.name};\n"
            "}\n")


def _decl(kern: Kernel) -> str:
    sig = ", ".join(f"{_ctype(p)} {p.name}" for p in kern.params)
    return (f'extern "C" void fjl_{kern.name}(int grid, int lanes, {sig});\n'
            f'extern "C" const void* fjk_{kern.name}();')


def _buffers(launches: Sequence[GkLaunch]):
    for c, x in enumerate(launches):
        for p in x.kernel.params:
            if p.array and p.name != G:
                a = np.asarray(x.arrays[p.name])
                yield c, p.name, a.astype(np.float64).ravel(), (
                    p.type is Scalar.I32)


def _call(x: GkLaunch, c: int, slot: Mapping[tuple[int, str], int],
          n_atom: int) -> str:
    scalars = {"n_bra": x.n_bra, "n_ket": x.n_ket, "n_ao": x.n_ao,
               "n_atom": n_atom}
    args = []
    for p in x.kernel.params:
        if p.name == G:
            args.append(f"dG + {c * n_atom * 3}")
        elif p.array:
            t = "int" if p.type is Scalar.I32 else "double"
            args.append(f"(const {t}*)buf[{slot[c, p.name]}]")
        else:
            args.append(str(scalars[p.name]))
    grid, (_, lanes) = launch_shape(x.n_bra, THREADS)
    return (f"        // launch {c}: {x.kernel.name}, {x.n_bra} x {x.n_ket}\n"
            f"        fjl_{x.kernel.name}({grid}, {lanes}, {', '.join(args)});\n"
            f"        CHECK(devGetLastError());\n"
            f"        ++launches;")


def gk_stage(launches: Sequence[GkLaunch]) -> np.ndarray:
    return np.concatenate([f for _, _, f, _ in _buffers(launches)])


@dataclass(frozen=True)
class GkBinary:
    """One linked program; `run` returns `[run][launch][n_atom][3]`."""

    launches: tuple[GkLaunch, ...]
    n_atom: int
    backend: str
    binary: Path
    workdir: Path
    compile_s: Mapping[str, float]  # unit -> wall seconds of its `-c`

    def run(self, run_device_binary) -> np.ndarray:
        src, dst = self.workdir / "gk.in", self.workdir / "gk.out"
        gk_stage(self.launches).tofile(src)
        out = run_device_binary(self.binary, str(src), str(dst),
                                vram=self.backend)
        want = f"fj-gk launches {len(self.launches) * RUNS} runs {RUNS}"
        assert want in out, out
        raw = np.fromfile(dst, dtype=np.float64)
        n = len(self.launches)
        assert raw.size == RUNS * n * self.n_atom * 3, raw.size
        return raw.reshape(RUNS, n, self.n_atom, 3)


def _flags(tc: Toolchain, backend: str) -> list[str]:
    flags = shlex.split(measured.config_value(FLAGS_VAR[backend]))
    arch = (f"-arch=sm_{tc.arch}" if tc.name == "nvcc"
            else f"--offload-arch={tc.arch}")
    lang = ["-x", "hip"] if backend == "hip" else []
    return [*flags, arch, *lang]


def _compile_obj(tc: Toolchain, flags: list[str], main: Path,
                 extra: Sequence[Path], obj: Path, budget: float) -> float:
    """`-c` one TU through the cache, killed at `budget` s; FAIL on an
    error, a kill, or a warning located in generated text (#92)."""
    import time

    text, units = devcache.localize(main.read_text(), main.parent)
    t0 = time.perf_counter()
    try:
        got = devcache.compile_unit(tc.path, flags, main.name, text, units,
                                    obj, mode="-c", timeout=budget)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"{main.name}: OVER the {budget:g} s per-unit budget (#618) -- "
            "re-emit a smaller unit, never a longer budget") from e
    wall = time.perf_counter() - t0
    assert got.returncode == 0, f"{tc.name} failed on {main.name}:\n" + (
        "\n".join(shlex.join(c) for c in got.commands) + f"\n{got.diagnostics}")

    class _Drv:  # what generated_warnings reads
        path = main
        units = tuple(extra)

    warned = generated_warnings(_Drv, got.diagnostics)
    assert not warned, (f"{tc.name}: warning(s) in generated text (#92):\n  "
                        + "\n  ".join(warned))
    return wall


def compile_gk(launches: Sequence[GkLaunch], n_atom: int, backend: str,
               tc: Toolchain, workdir: Path, *,
               budget: float = UNIT_BUDGET_S, jobs: int = 4) -> GkBinary:
    """Emit each distinct unit, compile it and its wrapper alone, then the
    `main()`, and link (module docstring)."""
    emit, arch, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    flags = _flags(tc, backend)
    seen: dict[str, Kernel] = {}
    for x in launches:  # one unit per distinct kernel TEXT
        k = seen.get(x.kernel.name)
        if k is not None and k is not x.kernel and emit(k, arch) != emit(
                x.kernel, arch):
            raise ValueError(f"two different kernels named {k.name}: give "
                             "one form another name")
        seen.setdefault(x.kernel.name, x.kernel)
    jobs_list = []
    for name, k in seen.items():
        unit = workdir / f"{name}.{ext}"
        unit.write_text(emit(k, arch))
        wrap = workdir / f"fjl_{name}.{ext}"
        wrap.write_text(_wrapper(k, unit.name))
        jobs_list.append((name, wrap, unit))
    objs = {name: workdir / f"fjl_{name}.o" for name, _, _ in jobs_list}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        walls = dict(zip([n for n, _, _ in jobs_list], pool.map(
            lambda j: _compile_obj(tc, flags, j[1], (j[2],), objs[j[0]],
                                   budget), jobs_list), strict=True))
    slot: dict[tuple[int, str], int] = {}
    sizes, ints = [], []
    for c, name, flat, is_int in _buffers(launches):
        slot[c, name] = len(sizes)
        sizes.append(str(flat.size))
        ints.append("1" if is_int else "0")
    src = GK_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, decls="\n".join(_decl(k) for k in seen.values()),
        upload_int=UPLOAD_INT, n_buf=len(sizes), n_atom=n_atom, runs=RUNS,
        n_launch=len(launches), sizes=", ".join(sizes), ints=", ".join(ints),
        vram_kernels=", ".join(f"fjk_{n}()" for n in seen),
        launches="\n".join(_call(x, c, slot, n_atom)
                           for c, x in enumerate(launches)))
    main = workdir / f"gk_main.{ext}"
    main.write_text(src)
    main_obj = workdir / "gk_main.o"
    walls["gk_main"] = _compile_obj(tc, flags, main, (), main_obj, budget)
    binary = workdir / "gk_bin"
    link = [tc.path, *devcache._link_flags(flags), str(main_obj),
            *(str(o) for o in objs.values()), "-o", str(binary)]
    done = subprocess.run(link, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, shlex.join(link) + "\n" + done.stdout + \
        done.stderr
    return GkBinary(tuple(launches), n_atom, backend, binary, workdir, walls)


# Every launch into its own G slice, RUNS times.
GK_PROGRAM = r"""// generated by test/gk_driver.py: harness, the gk_ kernels (#618)
{prologue}
{portable}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devMemset hipMemset
#define devFree   hipFree
#else
#define devMemset cudaMemset
#define devFree   cudaFree
#endif
{decls}
{upload_int}
static const int NB = {n_buf}, NATOM = {n_atom}, RUNS = {runs};
static const int NL = {n_launch};
static const size_t sz[NB] = {{{sizes}}};  // doubles per staged array
static const int is_int[NB] = {{{ints}}};  // 1: converted to int on upload

// usage: in out. `in` is every staged array, flat, in order; `out` is each
// run's G [NL, NATOM, 3], each from a zeroed G.
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
    const size_t ng = (size_t)NL * NATOM * 3;
    if (fj_vram_probe(in.size() * sizeof(double) + ng * sizeof(double)
                          + (size_t)16 * NB,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data();
    void* buf[NB];
    for (int b = 0; b < NB; ++b) {{
        buf[b] = is_int[b] ? (void*)upload_int(h, sz[b]) : (void*)upload(h, sz[b]);
        h += sz[b];
        if (!buf[b]) {{ std::fprintf(stderr, "upload %d failed\n", b); return 1; }}
    }}
    double* dG = nullptr;
    CHECK(devMalloc(&dG, ng * sizeof(double)));
    std::vector<double> Gh(ng), out;
    int launches = 0;
    for (int r = 0; r < RUNS; ++r) {{
        CHECK(devMemset(dG, 0, ng * sizeof(double)));
{launches}
        CHECK(devSynchronize());
        CHECK(devMemcpy(Gh.data(), dG, ng * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), Gh.begin(), Gh.end());
    }}
    for (int b = 0; b < NB; ++b) CHECK(devFree(buf[b]));
    CHECK(devFree(dG));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-gk launches %d runs %d\n", launches, RUNS);
    return 0;
}}
"""
