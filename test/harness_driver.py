"""The shared C++ harness's Python half: one runner per unit, npy in and out (#649).

    HARNESS                                    -> test/harness/, the C++ side
    runner_program(backend, unit, kernels, workdir) -> Program (the runner's source)
    build_runner(tc, program, out)             -> Path, via compile_checked (#92)
    launch_spec(kernel, grid, block, args)     -> (spec dict, {file: array})
    write_case(workdir, spec, arrays)          -> the spec's path
    run_case(binary, kernel, workdir, ...)     -> Ran(outs, events_ns)
    compile_checked(tc, program, out)          -> Path, #92's warning filter
    Step / chain_runners / build_chain / run_chain -> several launches, one
                                                  carried output (#659)
    chain_specs(steps, carried, init)          -> the specs, checked host-side
    build_steps(tc, backend, steps, workdir)   -> chain_runners + build_chain (#658)
    carried(steps)                             -> the one OUT every step writes

WHY. Every device test before this wrote its own `main()` (test_f32_device.py's
HARNESS, plan_driver.py's DRIVER, ...): the uploads, the launch and the
read-back spelled again per kernel family, through raw `.bin` files whose
layout lived in two places. The runner is the one `main()`: it takes a
generated unit, a launch spec naming the kernel, its grid and block and
EVERY argument by the IR `Kernel`'s parameter name, and input `.npy` files,
and writes output `.npy` files. test/harness/runner.h has the C++ contract.

THE SPEC IS BUILT FROM THE IR, not typed by hand. `launch_spec` walks
`kernel.params` in order: a scalar takes a number, an IN array an ndarray of
exactly its dtype, an OUT array an ndarray (read and updated) or a shape
(zero-filled). A missing, extra or mistyped argument raises here, and the
runner checks the spec again against the compiled signature
(test/harness/kernel_table.h), so the two halves cannot drift silently.

COMPILED LIKE EVERY OTHER TEST-SIDE DRIVER: `compile_checked`, the one
compile entry point (folded in from plan_driver.py by #659, which still
re-exports it for the drivers not yet ported), so devtools/config.sh's
NVCC_FLAGS / HIPCC_FLAGS, the compiler cache (#186) and #92's warning
filter -- which here covers the unit AND every harness header, so "the
runner builds with no warnings" is a checked statement.

CHAINS (#659): several launches into ONE output -- a gradient's `G`, a
hessian's `H` -- are `Step`s: each launch its own process, the output of
one the read-and-update input of the next (#654's pattern), the chain
repeated `runs` times from the same initial value by `run_chain`. A chain
whose kernels add with fp64 atomics is not bit-reproducible launch to
launch, so its runs are separate processes and the test bounds their
spread, as the string-built programs' `RUNS` loop did.

REPEATS (#654): `runs=` / `warmup=`, runner.h's run-to-run witness and clock.

The harness is C++17 (the tier's `--std=c++17`) and C++20-clean: no modules,
no `import std`, nothing from FreeBreakfast.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import devcache
import numpy as np
from conftest import Toolchain
from test_kernel_device import BACKEND

from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.performance import measured

HARNESS = Path(__file__).resolve().parent / "harness"
HEADERS = tuple(HARNESS / n for n in ("fj_gpu.h", "device_buffer.h", "npy.h",
                                      "json.h", "kernel_table.h", "runner.h"))
EXT = {"cuda": "cu", "hip": "hip"}

# IR element type -> (the spec's spelling, numpy's dtype). runner.h also
# takes i64 / u32 / u64, which no IR type spells yet.
TYPE = {Scalar.F64: "f64", Scalar.F32: "f32", Scalar.I32: "i32"}
DTYPE = {Scalar.F64: np.dtype("<f8"), Scalar.F32: np.dtype("<f4"),
         Scalar.I32: np.dtype("<i4")}

@dataclass(frozen=True)
class Program:
    """A generated translation unit and the emitted units it includes:
    what `compile_checked` compiles and filters (it reads `backend`, `path`,
    `units`; `plan_driver.Driver` / `KDriver` carry the same three)."""

    backend: str
    source: str
    path: Path
    units: tuple[Path, ...]




# --- compiling it: the tier's flags, and #92's warning filter -------------

FLAGS_VAR = {"cuda": "NVCC_FLAGS", "hip": "HIPCC_FLAGS"}
# kernel-tier.sh's two spellings of a located warning: nvcc's front end
# `file(LINE): warning ...`, clang/gcc `file:LINE[:COL]: warning: ...`.
_WARNING = r"(?:\((\d+)\): warning|:(\d+)(?::\d+)?: warning:?)\s*(.*)"


def compile_command(tc: Toolchain, drv: Program, out: Path) -> list[str]:
    """The tier's compile command for the driver: devtools/config.sh's
    NVCC_FLAGS / HIPCC_FLAGS (the flag set the compile phase uses, warnings
    on) and the arch, linked to an executable."""
    flags = shlex.split(measured.config_value(FLAGS_VAR[drv.backend]))
    arch = (f"-arch=sm_{tc.arch}" if tc.name == "nvcc"
            else f"--offload-arch={tc.arch}")
    return [tc.path, *flags, arch, str(drv.path), "-o", str(out)]


def generated_warnings(drv: Program, diagnostics: str) -> list[str]:
    """Every warning whose LOCATION is generated text -- the driver or one of
    the kernel units it includes -- as `file, line N: message` (#92: a
    warning in a vendor header is not the generator's and is not counted)."""
    hits: list[str] = []
    for f in (drv.path, *drv.units):
        for name in {str(f), f.name}:
            pat = re.compile(r"(?:^|[\s\"'/])" + re.escape(name) + _WARNING,
                             re.MULTILINE)
            for m in pat.finditer(diagnostics):
                hit = f"{f.name}, line {m.group(1) or m.group(2)}: {m.group(3)}"
                if hit not in hits:
                    hits.append(hit)
    return hits


def compile_checked(tc: Toolchain, drv: Program, out: Path, *,
                    env: Mapping[str, str] | None = None,
                    timeout: float = 1800.0) -> Path:
    """Compile the driver with the tier's flags; FAIL on an error, and on any
    warning located in generated text (#92), naming file and line.

    `compile_command`'s flags, run through the compiler cache (#186,
    test/devcache.py): the driver and every unit it includes are carried, by
    name, into a content-addressed directory and compiled there -- `-c`
    through sccache, then linked -- so the same plan in a fresh `tmp_path`
    is a hit. The compiler always runs; on a hit sccache replays the stored
    diagnostics, and the filter below reads them the same way (the carried
    units keep their names, which is what `generated_warnings` matches)."""
    cmd = compile_command(tc, drv, out)
    text, units = devcache.localize(drv.path.read_text(), drv.path.parent)
    got = devcache.compile_unit(cmd[0], cmd[1:-3], drv.path.name, text, units,
                                out, env=env, timeout=timeout)
    diag = got.diagnostics
    assert got.returncode == 0, (
        f"{tc.name} failed in {got.workdir}:\n"
        + "\n".join(shlex.join(c) for c in got.commands) + f"\n{diag}")
    warned = generated_warnings(drv, diag)
    assert not warned, (
        f"{tc.name}: {len(warned)} warning(s) in generated text (#92) -- fix "
        "the emitter or the driver, never a flag:\n  " + "\n  ".join(warned))
    assert out.exists() and out.stat().st_size > 0, f"{out} is empty"
    return out


RUNNER = """\
// runner for {unit}: generated by test/harness_driver.py (#649) -- harness.
#include "{harness}"
#include "{unit_path}"

static const fj::KernelEntry fj_kernels[] = {{{entries}}};

int main(int argc, char** argv) {{ return fj::runner_main(argc, argv, fj_kernels); }}
"""


def runner_program(backend: str, unit: Path, kernels: Sequence[str],
                   workdir: Path) -> Program:
    """The runner's source for `unit` (an emitted `cuda` / `hip` file), able
    to launch each kernel named in `kernels`, written into `workdir`."""
    if backend not in EXT:
        raise ValueError(f"backend {backend!r}: the runner takes cuda or hip")
    if not kernels:
        raise ValueError("a runner needs at least one kernel")
    for k in kernels:
        if not k.isidentifier():
            raise ValueError(f"{k!r} is not a kernel name")
    unit = unit.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    src = RUNNER.format(unit=unit.name, harness=HARNESS / "runner.h",
                        unit_path=unit,
                        entries=", ".join(f"FJ_KERNEL({k})" for k in kernels))
    path = workdir / f"runner_{unit.stem}.{EXT[backend]}"
    path.write_text(src)
    return Program(backend, src, path, (unit, *HEADERS))


def build_runner(tc: Toolchain, program: Program, out: Path, *,
                 timeout: float = 1800.0) -> Path:
    """Compile the runner with the tier's flags; fail on an error or on any
    warning located in the unit or a harness header (#92)."""
    return compile_checked(tc, program, out, timeout=timeout)


def _spec_value(p, value) -> dict:
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        raise TypeError(f"{p.name}: a scalar parameter takes a number")
    if p.type is Scalar.I32:
        if isinstance(value, (bool, np.bool_)) or int(value) != value:
            raise TypeError(f"{p.name}: an i32 takes an integer, got {value!r}")
        return {"value": int(value)}
    if p.type is Scalar.F32:
        return {"value": float(np.float32(value))}
    return {"value": float(value)}


def launch_spec(kernel: Kernel, grid: Sequence[int] | int,
                block: Sequence[int] | int, args: Mapping[str, object], *,
                shared_bytes: int = 0, runs: int = 1,
                warmup: int = 0) -> tuple[dict, dict[str, np.ndarray]]:
    """The runner's spec for one launch of `kernel`, and the arrays to write
    beside it (`{file name: array}`).

    `args` names every parameter exactly once: a number for a scalar, an
    ndarray of exactly the parameter's dtype for an IN array, and for an OUT
    array either an ndarray (uploaded, updated, read back) or a shape tuple
    (zero-filled, read back). Each OUT array is written as `<name>.out.npy`.
    """
    names = [p.name for p in kernel.params]
    extra = sorted(set(args) - set(names))
    missing = [n for n in names if n not in args]
    if extra or missing:
        raise ValueError(f"{kernel.name}: args missing {missing}, unknown {extra}")
    spec_args: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for p in kernel.params:
        value = args[p.name]
        entry: dict = {"name": p.name, "type": TYPE[p.type]}
        if not p.array:
            entry.update(_spec_value(p, value))
            spec_args.append(entry)
            continue
        entry["array"] = True
        if isinstance(value, np.ndarray):
            if value.dtype != DTYPE[p.type]:
                raise TypeError(f"{p.name}: dtype {value.dtype}, the kernel takes "
                                f"{DTYPE[p.type]} -- convert it deliberately")
            fname = f"{p.name}.in.npy"
            arrays[fname] = np.ascontiguousarray(value)
            entry["in"] = fname
        elif p.intent is Intent.OUT:
            shape = (value,) if isinstance(value, int) else tuple(value)
            if not all(isinstance(d, (int, np.integer)) and d >= 0 for d in shape):
                raise TypeError(f"{p.name}: an OUT array takes an ndarray or a "
                                f"shape, got {value!r}")
            entry["shape"] = [int(d) for d in shape]
        else:
            raise TypeError(f"{p.name}: an IN array takes an ndarray, got "
                            f"{type(value).__name__}")
        if p.intent is Intent.OUT:
            entry["out"] = f"{p.name}.out.npy"
        spec_args.append(entry)

    def dims(d: Sequence[int] | int, what: str) -> list[int]:
        got = [d] if isinstance(d, int) else [int(x) for x in d]
        if not 1 <= len(got) <= 3 or min(got) < 1:
            raise ValueError(f"{what} {d!r}: 1 to 3 positive extents")
        return got

    def at_least(n: int, least: int, what: str) -> int:
        if isinstance(n, bool) or int(n) != n or n < least:
            raise ValueError(f"{what} {n!r}: an integer >= {least}")
        return int(n)

    spec = {"kernel": kernel.name, "grid": dims(grid, "grid"),
            "block": dims(block, "block"), "shared_bytes": int(shared_bytes),
            "args": spec_args}
    runs, warmup = at_least(runs, 1, "runs"), at_least(warmup, 0, "warmup")
    if runs != 1:
        spec["runs"] = runs
    if warmup:
        spec["warmup"] = warmup
    return spec, arrays


def write_case(workdir: Path, spec: dict, arrays: Mapping[str, np.ndarray]) -> Path:
    """`spec.json` and its input `.npy` files, in `workdir`."""
    workdir.mkdir(parents=True, exist_ok=True)
    for fname, a in arrays.items():
        np.save(workdir / fname, a)
    path = workdir / "spec.json"
    path.write_text(json.dumps(spec, indent=1) + "\n")
    return path


_RUN = re.compile(r"^fj-runner run \d+ events_ns ([0-9.]+)$", re.MULTILINE)


class Ran(NamedTuple):
    outs: dict[str, Path]  # each OUT array's `.npy`, by parameter name
    events_ns: list[float]  # per timed run, in order


def run_case(binary: Path, kernel: Kernel, workdir: Path,
             grid: Sequence[int] | int, block: Sequence[int] | int,
             args: Mapping[str, object], *, shared_bytes: int = 0,
             runs: int = 1, warmup: int = 0, timeout: float = 300) -> Ran:
    """Write the case, run the runner on it, and return each OUT array's
    `.npy` path and each run's event time. A nonzero exit -- a launch
    whose outputs differ from the first's among them -- fails with its
    stderr."""
    spec, arrays = launch_spec(kernel, grid, block, args,
                               shared_bytes=shared_bytes, runs=runs,
                               warmup=warmup)
    path = write_case(workdir, spec, arrays)
    done = subprocess.run([str(binary), str(path)], capture_output=True,
                          text=True, timeout=timeout)
    assert done.returncode == 0, (
        f"{binary.name} {path} exited {done.returncode}\n"
        f"stdout: {done.stdout}\nstderr: {done.stderr}")
    outs = {a["name"]: workdir / a["out"] for a in spec["args"] if "out" in a}
    for name, p in outs.items():
        assert f"fj-runner out {name} {p}" in done.stdout, done.stdout
    times = [float(m.group(1)) for m in _RUN.finditer(done.stdout)]
    assert len(times) == runs, done.stdout
    return Ran(outs, times)


# --- chains: several launches carrying one output (#659) ----------------------


def emit_unit(kern: Kernel, backend: str, workdir: Path,
              units: Mapping[str, Path] | None = None) -> Path:
    """The emitted text of `kern`: `units[kern.name]` when the caller has the
    file (the tier's own, asserted byte-identical by the caller), else
    written into `workdir`."""
    if units is not None and kern.name in units:
        return units[kern.name]
    emit, arch, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"{kern.name}.{ext}"
    path.write_text(emit(kern, arch))
    return path


def as_param(kernel: Kernel, name: str, value) -> np.ndarray:
    """`value` as parameter `name`'s dtype, converted deliberately: an integer
    parameter takes only values its dtype holds exactly (the int arrays the
    staging builds as float64, the old harnesses' exact doubles)."""
    (p,) = [q for q in kernel.params if q.name == name]
    if not p.array:
        raise TypeError(f"{kernel.name}.{name} is a scalar")
    a = np.asarray(value)
    want = DTYPE[p.type]
    if a.dtype == want:
        return a
    got = a.astype(want)
    if p.type is Scalar.I32 and not np.array_equal(got, a):
        raise ValueError(f"{kernel.name}.{name}: a value is not an exact "
                         f"{want}")
    return got


@dataclass(frozen=True)
class Step:
    """One launch of a chain: `kernel` over `grid` x `block`, with every
    argument but the carried one (`run_chain`), arrays converted to their
    parameter's dtype by `as_param`."""

    kernel: Kernel
    grid: Sequence[int] | int
    block: Sequence[int] | int
    args: Mapping[str, object]
    shared_bytes: int = 0


def _step_args(s: Step) -> dict:
    return {k: as_param(s.kernel, k, v) if isinstance(v, np.ndarray) else v
            for k, v in s.args.items()}


def chain_specs(steps: Sequence[Step], carried: str,
                init: np.ndarray) -> list[dict]:
    """Every step's launch spec, as `run_chain` would write it: each
    argument named, typed and converted against its kernel's parameter
    list on the host -- no compiler, no card."""
    return [launch_spec(s.kernel, s.grid, s.block,
                        {**_step_args(s), carried: np.ascontiguousarray(init)},
                        shared_bytes=s.shared_bytes)[0] for s in steps]


def chain_runners(kernels: Iterable[Kernel], backend: str, workdir: Path, *,
                  units: Mapping[str, Path] | None = None
                  ) -> dict[str, Program]:
    """One runner per distinct kernel name, its unit `emit_unit`'s."""
    out: dict[str, Program] = {}
    for k in kernels:
        if k.name not in out:
            unit = emit_unit(k, backend, workdir, units)
            out[k.name] = runner_program(backend, unit, [k.name], workdir)
    return out


def build_chain(tc: Toolchain, runners: Mapping[str, Program], workdir: Path,
                *, timeout: float = 1800.0) -> dict[str, Path]:
    """Every runner compiled (`build_runner`), by kernel name; `timeout`
    caps each compile."""
    return {name: build_runner(tc, prog, workdir / f"runner_{name}",
                               timeout=timeout)
            for name, prog in runners.items()}


def carried(steps: Sequence[Step]) -> str:
    """The one OUT array every step's kernel writes: the chain's carried
    output, refused unless there is exactly one (#658)."""
    names = {p.name for s in steps for p in s.kernel.params
             if p.intent is Intent.OUT}
    if len(names) != 1:
        raise ValueError(f"a chain carries one OUT array, the steps write "
                         f"{sorted(names)}")
    return names.pop()


def build_steps(tc: Toolchain, backend: str, steps: Sequence[Step],
                workdir: Path, *, units: Mapping[str, Path] | None = None,
                timeout: float = 1800.0) -> dict[str, Path]:
    """Every step's runner, compiled: `chain_runners` of the steps' kernels
    (`units` as `emit_unit`'s), then `build_chain`."""
    return build_chain(tc, chain_runners([s.kernel for s in steps], backend,
                                         workdir, units=units),
                       workdir, timeout=timeout)


def run_chain(binaries: Mapping[str, Path], steps: Sequence[Step],
              carried: str, init: np.ndarray, workdir: Path, *,
              runs: int) -> list[np.ndarray]:
    """`runs` times: `carried` from `init`, through every step in order --
    each launch reads and updates it, its output the next one's input --
    and that run's final value. Each launch is its own process (one
    `run_case`); the runs are independent, so a kernel whose atomics make
    one launch's sum order the hardware's shows its run-to-run spread here
    rather than failing the runner's bitwise witness."""
    init = np.ascontiguousarray(init)
    out = []
    for r in range(runs):
        x = init
        for c, s in enumerate(steps):
            ran = run_case(binaries[s.kernel.name], s.kernel,
                           workdir / f"run{r}" / f"step{c}", s.grid, s.block,
                           {**_step_args(s), carried: x},
                           shared_bytes=s.shared_bytes)
            x = np.load(ran.outs[carried])
            assert x.shape == init.shape, (s.kernel.name, x.shape, init.shape)
        out.append(x)
    return out
