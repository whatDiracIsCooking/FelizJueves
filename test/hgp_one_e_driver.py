"""S, T, V or the dipole of a basis built ON A CARD by the HGP one-electron
kernels (`one_electron.kernel_hgp`, #598) -- harness.

    hgp_one_e_program(launch, target, basis, backend, workdir,
                      origin=, q=)                            -> Program
    hgp_one_e_stage(launch, target, basis)                    -> np.ndarray
    build_on_card(launch, target, basis, backend, tc, workdir,
                  run_device_binary, origin=, q=)             -> [run] -> M

`one_electron_driver.py`'s shape, for the other engine's staging: one
thread per CONTRACTED SHELL PAIR (`kernel_hgp.launch_shape(n_pair)`), the
arrays `OneEClass.hgp_arrays(target, basis)` gives, and PLAIN STORES (no
atomics: each element has one owner). Every argument is read off the
kernel's own parameter list, so the call cannot drift from the signature:
an `int` scalar is a launch extent (`n_pair`, `n_rec`, `n_ao`, `n_nuc`), a
`double` scalar the dipole's `C_x C_y C_z q` (baked into the `main()` as
literals, `repr` round-trips), an array a slice of the one input file
(`int` arrays through `upload_int`), and the one OUT array the matrix.

WHAT THE GENERATED `main()` DOES, `RUNS` times: zero the output, launch
every class's kernel into it, synchronize, read back. Plain stores into
disjoint elements, so the runs must be bit-identical.

Harness, in `test/`, not a second emitter: the units are `emit/cuda.py` /
`emit/hip.py`'s text of `build_hgp_kernel`, unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from plan_driver import RUNS, UPLOAD_INT, Program, _unit, compile_checked
from test_kernel_device import BACKEND, PORTABLE

from felizjueves.ao.shell import Basis
from felizjueves.eri.j_coulomb.launch import OneEClass, OneELaunch
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.one_electron import kernel_hgp as kh
from felizjueves.spec import Target

THREADS = 32  # lanes per block, one shell pair per thread; not numerical
DIPOLE_SCALARS = ("C_x", "C_y", "C_z", "q")


def kernel_of(cls: OneEClass, target: Target) -> Kernel:
    return kh.build_hgp_kernel(kh.hgp_spec(target, cls.la, cls.lb))


def _arrays(launch: OneELaunch, cls: OneEClass, target: Target,
            basis: Basis) -> dict[str, np.ndarray]:
    """Every input array of the class's kernel, by parameter name."""
    out = {k: np.ravel(np.asarray(v, dtype=np.float64))
           for k, v in cls.hgp_arrays(target, basis).items()}
    if target is Target.V:
        out["Z"] = np.asarray(launch.Z, dtype=np.float64)
        out["Cnuc"] = np.ravel(np.asarray(launch.Cnuc, dtype=np.float64))
    return out


def _inputs(kern: Kernel) -> list:
    return [p for p in kern.params if p.array and p.intent is not Intent.OUT]


def hgp_one_e_program(launch: OneELaunch, target: Target, basis: Basis,
                      backend: str, workdir: Path, *,
                      origin: Sequence[float] = (0.0, 0.0, 0.0),
                      q: float = 1.0) -> Program:
    """The `main()` building `target`'s matrix from every class of `launch`."""
    if target not in kh.TARGETS:
        raise ValueError(f"{target!r} is not an HGP one-electron target")
    _, _, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    n_ao = basis.n_ao
    comps = 3 if target is Target.DIPOLE else 1
    scalars = dict(zip(DIPOLE_SCALARS, (*map(float, origin), float(q)),
                       strict=True))
    includes, paths, kernels, body = [], [], [], []
    at = 0  # doubles consumed from the input file
    for c, cls in enumerate(launch):
        kern = kernel_of(cls, target)
        arrs = _arrays(launch, cls, target, basis)
        want = {p.name for p in _inputs(kern)}
        if want != set(arrs):
            raise ValueError(f"{kern.name}: arrays {sorted(want)}, staged "
                             f"{sorted(arrs)}")
        path = _unit(kern, backend, workdir, None)
        paths.append(path)
        ns = f"fj_h1e{c}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        kernels.append(f"(const void*){ns}::{kern.name}")
        ext_of = {"n_pair": len(cls.blocks), "n_rec": len(cls.recs),
                  "n_ao": n_ao, "n_nuc": len(launch.nuclei)}
        args = []
        for p in kern.params:
            if p.intent is Intent.OUT:
                args.append("dM")
            elif p.array:
                n = arrs[p.name].size
                var = f"a{c}_{p.name}"
                up = "upload_int" if p.type is Scalar.I32 else "upload"
                ty = "int" if p.type is Scalar.I32 else "double"
                body.append(f"    {ty}* {var} = {up}(h + {at}, {n});\n"
                            f"    if (!{var}) {{ std::fprintf(stderr, "
                            f"\"upload {var}\\n\"); return 1; }}")
                at += n
                args.append(var)
            elif p.type is Scalar.I32:
                args.append(str(ext_of[p.name]))
            else:
                args.append(repr(scalars[p.name]))
        grid, (lanes, _) = kh.launch_shape(len(cls.blocks), THREADS)
        body.append(f"#define LAUNCH_{c} {ns}::{kern.name}<<<dim3({grid}), "
                    f"dim3({lanes}, 1)>>>({', '.join(args)})")
    launches = "\n".join(
        f"        LAUNCH_{c};\n        CHECK(devGetLastError());"
        for c in range(len(launch)))
    src = PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes),
        upload_int=UPLOAD_INT, target=target.name, n_c=len(launch),
        n_out=comps * n_ao * n_ao, n_in=at, runs=RUNS,
        vram_kernels=", ".join(kernels), uploads="\n".join(body),
        launches=launches)
    main = workdir / f"hgp_one_e_{target.name.lower()}.{ext}"
    main.write_text(src)
    return Program(backend, src, main, tuple(paths))


def hgp_one_e_stage(launch: OneELaunch, target: Target,
                    basis: Basis) -> np.ndarray:
    """The input file: per class, its input arrays in parameter order."""
    parts = []
    for cls in launch:
        arrs = _arrays(launch, cls, target, basis)
        parts += [arrs[p.name] for p in _inputs(kernel_of(cls, target))]
    return np.concatenate(parts)


def build_on_card(launch: OneELaunch, target: Target, basis: Basis,
                  backend: str, tc: Toolchain, workdir: Path,
                  run_device_binary, *, origin=(0.0, 0.0, 0.0),
                  q: float = 1.0) -> np.ndarray:
    """`[run]` -> the matrix (`[3, n, n]` for the dipole) built on the card."""
    prog = hgp_one_e_program(launch, target, basis, backend, workdir,
                             origin=origin, q=q)
    binary = compile_checked(tc, prog, workdir / f"{prog.path.stem}_bin")
    src = workdir / f"{prog.path.stem}.in"
    dst = workdir / f"{prog.path.stem}.out"
    hgp_one_e_stage(launch, target, basis).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    assert (f"fj-hgp-one-e target {target.name} classes {len(launch)} "
            f"runs {RUNS}") in out, out
    n = basis.n_ao
    shape = (RUNS, 3, n, n) if target is Target.DIPOLE else (RUNS, n, n)
    return np.fromfile(dst, dtype=np.float64).reshape(shape)


PROGRAM = r"""// generated by test/hgp_one_e_driver.py: harness {target}
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
static const int NOUT = {n_out}, RUNS = {runs};
static const size_t NIN = {n_in};

int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    if (in.size() != NIN) {{
        std::fprintf(stderr, "input has %zu doubles, want %zu\n", in.size(), NIN);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double) + (size_t)NOUT * sizeof(double),
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data();
{uploads}
    double* dM = nullptr;
    CHECK(devMalloc(&dM, (size_t)NOUT * sizeof(double)));
    std::vector<double> M((size_t)NOUT), out;
    for (int r = 0; r < RUNS; ++r) {{
        CHECK(devMemset(dM, 0, M.size() * sizeof(double)));
{launches}
        CHECK(devSynchronize());
        CHECK(devMemcpy(M.data(), dM, M.size() * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), M.begin(), M.end());
    }}
    CHECK(devFree(dM));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-hgp-one-e target {target} classes %d runs %d\n", {n_c}, RUNS);
    return 0;
}}
"""
