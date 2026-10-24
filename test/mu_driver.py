"""The dipole `MU` of a basis built ON A CARD from its one-electron launch
(#523) -- harness.

    mu_program(launch, n_ao, backend, workdir, units=) -> Program
    mu_stage(launch, origin, q)                       -> np.ndarray
    mu_read_back(raw, n_ao)                           -> [run][3][n_ao][n_ao]
    compile_mu(launch, n_ao, backend, tc, workdir, units=) -> MuBinary
    MuBinary.run(origin, q, run_device_binary)        -> [run][3][n_ao][n_ao]
    tier_units(backend, kernels)                      -> {name: path}

`one_electron_driver.py`'s pattern for `Target.DIPOLE`, with one
difference: the origin `C` and the charge `q` are the kernel's RUNTIME
scalars (`kernel_mu`'s module docstring), so they are READ FROM THE INPUT
FILE, not spelled into the program. One compiled binary serves every origin
and every charge -- the origin-shift and `q = -1` checks run the SAME
machine code at two inputs, so they test the kernel's runtime arguments,
not two compilations.

Nothing else is decided here: the launch is `launch.one_electron_launch`'s
value (no nuclei: the dipole reads none), the records are
`OneEClass.arrays(Target.DIPOLE)` in the kernel's own parameter order, and
the kernels are `kernel_mu.build_mu_kernel`'s (`mu_{la}{lb}`, every
component in one launch), emitted by `emit/cuda.py` / `emit/hip.py`
unchanged. Only the `main()` around them is written here -- harness, in
`test/`, not a second emitter.

WHAT THE GENERATED `main()` DOES, `RUNS` times (so the fp64 atomics'
run-to-run spread is witnessed, not assumed): zero ONE `3 x n_ao x n_ao`
`MU` (the kernel's precondition, `kernel_mu.ZEROED_MU`), launch every
class's kernel into it -- one launch per pair class, one thread per record
(`kernel_mu.launch_shape`, `THREADS` lanes) -- then one synchronize and the
read-back. Every class ADDS (kernel_jab's scatter rule, per component), so
several classes into one `MU` is the kernels' own contract.

THE INPUT: `C_x C_y C_z q`, then per class its double-valued record arrays
(`arrays(Target.DIPOLE)` minus the offsets, row-major, in parameter order)
and its two `int` offsets. Uploaded once before the first run.

THE UNITS. As `one_electron_driver.py`: a kernel the tier's generate phase
wrote (`--mu-l`) is that file, asserted byte-identical to `emit(...)`; any
other class (`la > lb`, which the ladder never writes, or a tier run whose
generator did not ask for `--mu-l`) is emitted into `workdir` here, the same
text. Nothing above (dd) is ever emitted: `MU_LMAX`, refused (milestone
19's cap).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from conftest import Toolchain
from one_electron_driver import tier_units
from plan_driver import RUNS, UPLOAD_INT, Program, _unit, compile_checked
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.eri.j_coulomb.launch import OneEClass, OneELaunch
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.one_electron.kernel_mu import (
    MU,
    build_mu_kernel,
    launch_shape,
    n_components,
)
from felizjueves.spec import Target

__all__ = ["MU_LMAX", "MuBinary", "compile_mu", "kernel_of", "mu_program",
           "mu_read_back", "mu_stage", "tier_units", "units_of"]

THREADS = 32  # lanes per block, one record per thread; not numerical
MU_LMAX = 2  # milestone 19: every shell l <= 2, nothing above (dd)
NCOMP = n_components()  # x, y, z
INTS = ("off_a", "off_b")  # the int record arrays, last in every block
SCALARS = ("C_x", "C_y", "C_z", "q")  # the runtime scalars, the input's head


def kernel_of(cls: OneEClass) -> Kernel:
    """The class's dipole kernel, refusing anything above (dd)."""
    if max(cls.la, cls.lb) > MU_LMAX:
        raise ValueError(f"pair class ({cls.la}, {cls.lb}) is above (dd): "
                         "milestone 19 emits nothing there")
    return build_mu_kernel(cls.spec(Target.DIPOLE))


def _doubles(cls: OneEClass) -> list[str]:
    """The double-valued record arrays, parameter order (each one double
    per record: the dipole's records are flat, `kernel_s`'s layout)."""
    return [k for k in cls.arrays(Target.DIPOLE) if k not in INTS]


def _check_signature(kern: Kernel, cls: OneEClass) -> None:
    """`n_rec n_ao`, the record arrays in `arrays(DIPOLE)`'s order, the
    origin and charge, and the one OUT `MU`."""
    names = [p.name for p in kern.params]
    want = ["n_rec", "n_ao", *cls.arrays(Target.DIPOLE), *SCALARS, MU]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the launch's "
                         f"records want {want}")


def _call(kern: Kernel, c: int, ns: str, cls: OneEClass) -> str:
    """The launch of class `c`'s kernel, its arguments read off the kernel's
    own parameters."""
    n = len(cls.recs)
    at = {name: (f"dd[{c}] + {k * n}" if k else f"dd[{c}]")
          for k, name in enumerate(_doubles(cls))}
    at.update({"off_a": f"di[{c}]", "off_b": f"di[{c}] + {n}",
               "n_rec": str(n), "n_ao": "NAO", "C_x": "Cx", "C_y": "Cy",
               "C_z": "Cz", "q": "Q", MU: "dM"})
    args = []
    for p in kern.params:
        if p.name not in at:
            raise ValueError(f"{kern.name}: no argument for {p.name}")
        if not (p.array or p.type in (Scalar.I32, Scalar.F64)):
            raise ValueError(f"{kern.name}: {p.name} has no spelling here")
        args.append(at[p.name])
    grid, (lanes, _) = launch_shape(n, THREADS)
    return (f"        // pair class ({cls.la}, {cls.lb}), {n} records\n"
            f"        {ns}::{kern.name}<<<dim3({grid}), dim3({lanes}, 1)>>>(\n"
            f"            {', '.join(args)});\n"
            f"        CHECK(devGetLastError());\n"
            f"        ++launches;")


def mu_program(launch: OneELaunch, n_ao: int, backend: str, workdir: Path,
               *, units: Mapping[str, Path] | None = None) -> Program:
    """The `main()` building `MU` from every class of `launch` (module
    docstring)."""
    _, arch, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    includes, paths, calls, kernels, nd = [], [], [], [], []
    for c, cls in enumerate(launch):
        kern = kernel_of(cls)
        _check_signature(kern, cls)
        path = _unit(kern, backend, workdir, units)
        paths.append(path)
        ns = f"fj_mu{c}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        kernels.append(f"(const void*){ns}::{kern.name}")
        calls.append(_call(kern, c, ns, cls))
        nd.append(len(_doubles(cls)))
    src = MU_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes), upload_int=UPLOAD_INT,
        n_c=len(launch), n_ao=n_ao, ncomp=NCOMP, runs=RUNS,
        nr=", ".join(str(len(cls.recs)) for cls in launch),
        nd=", ".join(map(str, nd)), vram_kernels=", ".join(kernels),
        launches="\n".join(calls))
    main = workdir / f"mu.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def mu_stage(launch: OneELaunch, origin: Sequence[float],
             q: float) -> np.ndarray:
    """The program's input: `C_x C_y C_z q`, then per class its double
    record arrays in parameter order, then `off_a | off_b`."""
    head = [float(x) for x in origin]
    if len(head) != 3:
        raise ValueError(f"origin {origin!r} is not three floats")
    parts: list = [np.array([*head, float(q)])]
    for cls in launch:
        arr = cls.arrays(Target.DIPOLE)
        for name in _doubles(cls):
            parts.append(np.asarray(arr[name], dtype=np.float64))
        parts += [np.asarray(arr["off_a"], dtype=np.float64),
                  np.asarray(arr["off_b"], dtype=np.float64)]
    return np.concatenate(parts)


def mu_read_back(raw: np.ndarray, n_ao: int) -> np.ndarray:
    """`[run]` -> the `3 x n_ao x n_ao` `MU` that run built."""
    assert raw.size == RUNS * NCOMP * n_ao * n_ao, (raw.size, RUNS, n_ao)
    return raw.reshape(RUNS, NCOMP, n_ao, n_ao)


@dataclass(frozen=True)
class MuBinary:
    """One compiled `MU` program of one launch; `run` feeds it an origin
    and a charge."""

    launch: OneELaunch
    n_ao: int
    backend: str
    binary: Path
    workdir: Path

    def run(self, origin: Sequence[float], q: float,
            run_device_binary) -> np.ndarray:
        """`MU` at `origin` and charge `q`, `[run][3][n_ao][n_ao]`. The
        caller holds the card (its device fixture); the binary runs in
        VRAM-probe mode first."""
        tag = f"{len(list(self.workdir.glob('mu_*.in')))}"
        src = self.workdir / f"mu_{tag}.in"
        dst = self.workdir / f"mu_{tag}.out"
        mu_stage(self.launch, origin, q).tofile(src)
        out = run_device_binary(self.binary, str(src), str(dst),
                                vram=self.backend)
        want = (f"fj-mu classes {len(self.launch)} launches "
                f"{len(self.launch) * RUNS} runs {RUNS}")
        assert want in out, out
        return mu_read_back(np.fromfile(dst, dtype=np.float64), self.n_ao)


def compile_mu(launch: OneELaunch, n_ao: int, backend: str, tc: Toolchain,
               workdir: Path, *, units: Mapping[str, Path] | None = None
               ) -> MuBinary:
    """Compile the `MU` program of `launch` with devtools/config.sh's flags,
    FAILING on a warning located in generated text (#92)."""
    prog = mu_program(launch, n_ao, backend, workdir, units=units)
    binary = compile_checked(tc, prog, workdir / "mu_bin")
    return MuBinary(launch, n_ao, backend, binary, workdir)


def units_of(backend: str, launch: OneELaunch) -> dict[str, Path]:
    """The tier's own files of the launch's kernels it wrote (none unless
    its generator asked for `--mu-l`)."""
    return tier_units(backend, [kernel_of(c) for c in launch])


# Every class of one launch into one MU, RUNS times; C and q from the input.
MU_PROGRAM = r"""// generated by test/mu_driver.py (#523): harness DIPOLE
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
static const int NC = {n_c}, NAO = {n_ao}, NCOMP = {ncomp}, RUNS = {runs};
static const int nr[NC] = {{{nr}}};  // records per pair class
static const int nd[NC] = {{{nd}}};  // doubles per record

// usage: in out. `in` is C_x C_y C_z q, then per class its double record
// arrays (nd per record) and off_a off_b; `out` is each run's MU
// [NCOMP, NAO, NAO], each from a zeroed MU.
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    size_t want = 4;
    for (int c = 0; c < NC; ++c) want += (size_t)(nd[c] + 2) * nr[c];
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the launch wants %zu\n",
                     in.size(), want);
        return 2;
    }}
    const size_t nmu = (size_t)NCOMP * NAO * NAO;
    if (fj_vram_probe(in.size() * sizeof(double) + nmu * sizeof(double)
                          + (size_t)64 * NC,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data();
    const double Cx = h[0], Cy = h[1], Cz = h[2], Q = h[3];
    h += 4;
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
    CHECK(devMalloc(&dM, nmu * sizeof(double)));
    std::vector<double> M(nmu), out;
    int launches = 0;
    for (int r = 0; r < RUNS; ++r) {{
        CHECK(devMemset(dM, 0, nmu * sizeof(double)));
{launches}
        CHECK(devSynchronize());
        CHECK(devMemcpy(M.data(), dM, nmu * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), M.begin(), M.end());
    }}
    for (int c = 0; c < NC; ++c) {{
        CHECK(devFree(dd[c]));
        CHECK(devFree(di[c]));
    }}
    CHECK(devFree(dM));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-mu classes %d launches %d runs %d\n", NC, launches, RUNS);
    return 0;
}}
"""
