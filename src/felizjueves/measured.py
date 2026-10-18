"""The `--remeasure` driver: compile each cell, read what the compiler says.

    python -m felizjueves.measured                      # print the store
    python -m felizjueves.measured --remeasure [--backend cuda|hip|both]
                                   [--L 0 4 8] [-j N]   # one-batch classes
    python -m felizjueves.measured --remeasure --subgrid [-j N]  # SUBGRID
                                   [--n-batches 12]     # only those cells
    python -m felizjueves.measured --nine-vs-twelve     # #60's comparison
    python -m felizjueves.measured --remeasure --subgrid --lever recycle
    python -m felizjueves.measured --lever-ab recycle   # #55's comparison
    python -m felizjueves.measured --remeasure --subgrid --lever no-sieve
    python -m felizjueves.measured --lever-ab no-sieve  # #162's, + verdict
    python -m felizjueves.measured --remeasure --subgrid --lever tuned
    python -m felizjueves.measured --lever-ab tuned     # #163's, + verdict
    python -m felizjueves.measured --remeasure --asym   # #113's study cells
    python -m felizjueves.measured --asym-verdict       # #113's comparison
    python -m felizjueves.measured --roofline           # #112's verdicts
    python -m felizjueves.measured --remeasure --subgrid --target k
                [--subgrid-ls 7 10] [--compile-timeout S]  # #205's K cells
    python -m felizjueves.measured --remeasure --target f --L 0 1 2
                                                        # #288's F cells
    python -m felizjueves.measured --remeasure --lever <kernel lever>
                [--L ... | --subgrid [--target k] | --target f --L ...]
                                                        # #339's lever rows
    python -m felizjueves.measured --remeasure --ledger [--backend ...]
                                                        # #362's lever ledger

(with `PYTHONPATH=src uv run` in front, as for every entry point here.)

THE STORE IS `felizjueves.measured_store` (#94): the `Measured` record, the
file, the read path, the subgrid and the verdicts (#60's NINE AGAINST TWELVE,
#55's A LEVER A/B -- their definitions are that module's docstring). This
module is the half that knows a toolchain: it emits each cell's translation
units, runs nvcc / hipcc on them with a report flag, parses the report, and
merges one row per cell into the store. It compiles; it launches nothing on
a card and times nothing -- the wall-clock side of a row is written by
measure/measure_kernel_device.py's sweep. The `--nine-vs-twelve` and `--lever-ab`
views print the store's verdicts.

WHERE THE NUMBERS COME FROM, per vendor -- they are not the same quantity:

- nvcc: `-Xptxas -v`, the block after `Compiling entry function '<kernel>'`:
  `Used R registers` and `S bytes spill stores, L bytes spill loads`. Units:
  32-bit registers per thread, and BYTES of spill stores / loads.
- hipcc: `-Rpass-analysis=kernel-resource-usage`, the block after
  `Function Name: <kernel>`: `VGPRs`, `VGPRs Spill`, `SGPRs Spill`. Units:
  VGPRs per lane, and spilled REGISTER COUNTS. The AMD backend reports no
  store/load split, so `spill_store` and `spill_load` both carry the
  VGPR + SGPR spill count, and `spill_unit` says so. Every number the
  remark printed (ScratchSize, Occupancy, ...) is kept in `raw`.

THE ASYMMETRIC STUDY (#113) is `felizjueves.asym`: its `AsymCell`s are
compiled here like any cell (`cell_kernels` builds them through the study,
`measured_key` keys them `.../asym@...`), and its rule is that module's
docstring. An `AsymCell` is not a `Spec`; nothing else here accepts one.

Neither measurement flag changes code generation: both are reports. The rest
of the flag set is `devtools/config.sh`'s NVCC_FLAGS / HIPCC_FLAGS, read by
sourcing that file -- the one place that knows them -- and recorded in the row.

A CELL IS N KERNELS, ONE ROW. A split class is `build_pq_kernels(spec)`: one
kernel, one translation unit, one compile per batch. The store is keyed on
`MeasuredKey(spec, arch, toolchain)`, which has no batch index, so a cell is
ONE row, aggregated over its batch kernels as follows -- each batch is its
own launch, run one after another into one J:

- `registers`, `shared_bytes`: the MAX over the batch kernels (a per-launch
  resource; the launches never coexist);
- `spill_store`, `spill_load`: the SUM (the cell's whole spill, every batch
  kernel being run once per class); the per-batch numbers, and so the max,
  are in `raw`;
- `compile_s`: the sum of the batch compiles' wall times;
- `kernel`: the batch kernels' names, comma-separated, in partition order;
- `source_sha256`: of the batch texts concatenated in that order;
- `raw`: every number each compiler printed, keyed `<kernel>:<field>`.

A one-batch cell (n_batches = 1) is one kernel, and its row reads exactly as
before: unprefixed `raw`, the one kernel's name and source hash. A cell's
kernels are `cell_kernels(spec)`: `build_pq_kernels`, then lowered as the
Spec's lowering levers say (#55) -- for a Spec with none, exactly the
builder's.

`shared_bytes` is what the compiler reports, which is the STATIC footprint:
the kernel's `Jsh` is `extern __shared__`, sized at launch, so nvcc prints no
smem and hipcc an `LDS Size` of 0. The dynamic footprint is the kernel's own
declaration, not a toolchain observation: `raw` carries it as
`jsh_dynamic_bytes` (`Jsh` doubles per lane x the arch's wavefront x 8, at
the launch `kernel.launch_shape` fixes, blockDim.y = wavefront).

`compile_s` is the wall time of one compile while up to `-j` others run
beside it: a cost indication, never a benchmark.

THE K CELLS (#205) are `measured_store.K_SUBGRID`, compiled the same way
(`cell_kernels` builds them with `kernel_k.build_k_kernels`). A K batch
kernel's stage-2 digest is fully unrolled, so its unit runs from 0.8 MB of
source (L = 7) to 187 MB (L = 16), and the compile with it: each unit has a
wall-time budget (`--compile-timeout`, COMPILE_TIMEOUT_S by default), a
compile past it is killed with its whole process group and its cell gets
NO ROW (`CompileTimeout`, never a partial cell), and each cell's largest
unit is compiled first, so a cell that cannot be compiled is known -- and
its other units cancelled -- after one budget, not after all of them.

THE F CELLS (#288) are milestone 12's fused Fock classes (`F_CLASSES`, the
21-class ladder, every shell `l <= 2`), selected by total L with `--target
f --L ...` (`f_cells`): each is `kernel_f.build_f_kernel(spec)`, ONE unit,
so its row reads as a one-batch J row does -- unprefixed `raw`, the unit's
name and source hash -- keyed `.../b1/f/literal@...` (measured_store's THE
F CELL). Its one shared array, `ab_sh`, has a literal extent, so nvcc's
`smem` and hipcc's `LDS Size` report it as `shared_bytes` and `raw`'s
`jsh_dynamic_bytes` is 0. #288 recorded `--L 0 1 2`, the five classes its
default tier runs on the cards; the clock beside each is written by
measure/measure_fock_ondevice.py's opt-in sweep, behind the gate.

THE LEVER ROWS (#339) go through the store's one recording path
(`measured_store.lever_arm`, then `merge`): `--lever NAME` with a
kernel-body lever (`levers.VOCABULARY`) adds the lever to whatever cell set
the rest of the command selects -- the one-batch J classes of `--L`,
`--subgrid`'s J cells or, with `--target k`, its K cells, or `--target f
--L ...`'s F classes -- and each row is keyed by the full lever-carrying
`Spec.key()`. `cell_kernels` passes a kernel-body lever to the builder (J,
K and F alike; `Spec` has already refused an illegal (target, lever)), and
names each unit `lever_ladder.unit_name` of its literal name, the name the
generator CLI's lever ladder writes, so a row's `source_sha256` is the
emitted file's. A plan or lowering arm (`ARMS`) is still `--subgrid`'s J
cells only (`ab_cells`), as #55, #162 and #163 recorded it.

THE LEVER LEDGER (#362) is `--remeasure --ledger`: every cell of
`measured_store.ledger_cells()` -- the literal classes at total L <= 4 of
J, K and F, and every ledger arm (each kernel-body lever, a lever that only
builds with another as its composition, and the declared compositions) on
its target's classes -- that has NO row under this toolchain, or whose row's
`source_sha256` is not the text the generator emits today, is compiled
through the same `remeasure` (compile-only: no card, no clock), and then
`felizjueves.leverledger` rewrites data/measured/lever_ledger.json from the
store. A row whose text is unchanged is never recompiled, so a new lever
(one `levers.VOCABULARY` entry, and a `measured_store.REQUIRES` entry if it
only builds with another) is ONE re-run of this command.

COUNTERS (#110) are read here too: `profile` runs a built binary under the
vendor's profiler and `parse_ncu_csv` / `parse_rocprof_csv` read what it
wrote, one `KernelCounters` per launch; `counter_rows` files one cell's
launches as the store's per-kernel counter rows (`measured_store`'s THE
COUNTER ROW SHAPE). What each counter means, per vendor, is that module's
COUNTERS section; the invocations are:

- NVIDIA: `ncu --csv --print-units base --clock-control none --metrics
  <NCU_METRICS> --kernel-name regex:<re> <binary>`. `--print-units base`
  so a byte is a byte (each metric's printed unit is checked against
  NCU_UNITS, and a different one refused -- a renamed or rescaled counter
  is news, not a number); `--clock-control none` so the card runs at the
  clock it chooses, as the ceiling probes (#109) and the timed runs do,
  rather than ncu's default lock to base clock.
- AMD: `rocprofv3 --pmc <pass> --kernel-include-regex <re> --output-format
  csv -d <dir>/pass<i> -o counters -- <binary>`, once per ROCPROF_PASSES
  entry (gfx1200's GL2C collects at most four of the five byte counters at
  once, #140), each pass's `*counter_collection.csv` read by column name (`Dispatch_Id`,
  `Kernel_Name`, `Counter_Name`, `Counter_Value`, `Start_Timestamp`,
  `End_Timestamp`).

COULD NOT MEASURE IS NOT ZERO. A profiler that is absent, cannot load (an
image whose rocprofv3 lacks `libdw.so.1` or `libhsa-amd-aqlprofile64.so.1`,
as milestone 5's did), is refused the counters (ncu's `ERR_NVGPUCTRPERM` on
a host that restricts them to admins), or profiles no kernel raises
`CouldNotMeasure` and yields no row. A counter it did not report is None in
the row, never 0.

WHAT A CARD CAN COUNT (#140) is read off the profiler, not assumed:
`parse_rocprof_avail` reads `rocprofv3 --list-avail`'s counter names per
agent and `fp64_counters` picks the ones that count fp64 operations. On
gfx1200 that set is empty (`measured_store.MISSING_COUNTERS`), which is why
the card is in `measured_store.UNMEASURED` for good.

Counter collection serialises and replays each kernel, so the profiled run is
a SEPARATE run of the same binary from any timed one, and its duration is
`counter_ns`, never `runtime_ns`.

Nothing under `plan/` or `literature/` may import this module
(test/test_layering.py): a measurement may never feed a literature floor.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from felizjueves import leverledger
from felizjueves.asym import (
    ASYM_CELLS,
    AsymCell,
    AsymKey,
    arch_verdict,
    asym_vs_neighbours,
)
from felizjueves.device import GFX1200, SM_86, Arch, MeasuredKey
from felizjueves.emit import cuda, hip
from felizjueves.ir import Decl, Kernel, Lit, Space
from felizjueves.kernel import build_pq_kernels, pq_spec
from felizjueves.kernel_f import build_f_kernel, f_class_spec
from felizjueves.kernel_k import build_k_kernels
from felizjueves.lever_ladder import unit_name
from felizjueves.lower import LEVERS, RECYCLE, lower
from felizjueves.measured_store import (
    GPU_UTIL_FIELDS as GPU_UTIL_FIELDS,  # re-exported: measure_kernel_device's sweep
)
from felizjueves.measured_store import (
    K_SUBGRID,
    KERNEL,
    MEASUREMENTS,
    NO_SIEVE,
    NOT_MEASURABLE,
    REPO,
    STORE,
    SUBGRID,
    SUBGRID_LS,
    TUNED,
    UNMEASURED,
    Measured,
    ab_cells,
    cell_roofline,
    combine,
    kernel_key,
    kernel_roofline,
    kind_of,
    ledger_cells,
    lever_ab,
    lever_arm,
    load,
    load_counters,
    merge,
    nine_vs_twelve,
    save,
    sieve_verdict,
    split_kernel_key,
    tuned_verdict,
)
from felizjueves.measured_store import ROOFLINE as ROOFLINE
from felizjueves.measured_store import (
    lever_cells as lever_cells,  # re-exported: test_kernel_device's cells
)
from felizjueves.measured_store import (
    save_counters as save_counters,  # re-exported: measure_kernel_device's sweep
)
from felizjueves.spec import Spec, Target

DEFAULT_LS = tuple(range(0, 17))

# THE F CELLS (#288; module docstring): milestone 12's 21-class ladder,
# restated as emit.__main__ restates it (a driver names the HGP engine by its
# entry point only, test/test_layering.py) and held equal to hgp.vrr.CLASSES
# by test/test_measured.py.
_F_PAIRS = tuple((a, b) for a in range(3) for b in range(a, 3))
F_CLASSES: tuple[tuple[int, int, int, int], ...] = tuple(
    bra + ket for i, bra in enumerate(_F_PAIRS) for ket in _F_PAIRS[i:])


def f_cells(Ls: Sequence[int] = DEFAULT_LS) -> list[Spec]:
    """The F cells whose total `L = la + lb + lc + ld` is in `Ls`, in ladder
    order: `kernel_f.f_class_spec` of each (AO order, one batch, no lever)."""
    return [f_class_spec(*c) for c in F_CLASSES if sum(c) in set(Ls)]


def cell_kernels(spec: Spec | AsymCell) -> tuple[Kernel, ...]:
    """A cell's kernels, BUILT then LOWERED: `build_pq_kernels` of `spec`
    without the lowering levers (`lower.LEVERS`, which neither the builder
    nor the plan layer accepts), then `lower` on each finished kernel as
    those levers say. No lowering lever: `build_pq_kernels(spec)`, the very
    same objects -- the literal arm is not touched. An `AsymCell` (#113) is
    the study's own batch list (`AsymCell.kernels`), never lowered. A
    `Target.K` cell (#205) is `kernel_k.build_k_kernels(spec)`: the
    partial-sum form, one kernel per batch, atomized when split. No lever is
    declared for K, so a K Spec carrying one is refused, never built as
    something else. A `Target.F` cell (#288) is `kernel_f.build_f_kernel
    (spec)`, one unit, and refuses a lever the same way.

    A KERNEL-BODY lever (`levers.VOCABULARY`, #339) is the builder's: it is
    passed to the builder on every target (only a lever outside the
    vocabulary is refused on K and F), and every unit of such a cell is
    named `lever_ladder.unit_name` of its literal name -- the generator
    CLI's lever-ladder name. A cell without one keeps its builder's names."""
    if isinstance(spec, AsymCell):
        return spec.kernels()
    body = spec.kernel_body_levers()
    if not body:
        return _built(spec)
    return tuple(replace(k, name=unit_name(k.name, body))
                 for k in _built(spec))


def _built(spec: Spec) -> tuple[Kernel, ...]:
    other = tuple(lv for lv in spec.levers
                  if lv not in spec.kernel_body_levers())
    if spec.target is Target.F:  # #288: one unit, the fused kernel
        if other:
            raise ValueError(f"{spec.key()}: no lever is declared for an F "
                             "cell (#278) but a kernel-body one (#339)")
        return (build_f_kernel(spec),)
    if spec.target is Target.K:
        if other:
            raise ValueError(f"{spec.key()}: no lever is declared for a K "
                             "cell (#205) but a kernel-body one (#339)")
        return build_k_kernels(spec)
    rest = tuple(lv for lv in spec.levers if lv not in LEVERS)
    if rest == spec.levers:
        return build_pq_kernels(spec)
    return tuple(lower(k, recycle_slots=RECYCLE in spec.levers)
                 for k in build_pq_kernels(replace(spec, levers=rest)))


def measured_key(spec: Spec | AsymCell, arch: Arch,
                 toolchain: str) -> MeasuredKey | AsymKey:
    """The row key of a cell: `MeasuredKey` for a Spec, `asym.AsymKey`
    (`.../asym@...`) for the asymmetric study's cell."""
    if isinstance(spec, AsymCell):
        return AsymKey(spec, arch, toolchain)
    return MeasuredKey(spec, arch, toolchain)


NV_SPILL_UNIT = "bytes (ptxas -v: spill stores / spill loads)"
AMD_SPILL_UNIT = ("registers (hipcc kernel-resource-usage: VGPRs Spill + "
                  "SGPRs Spill; no store/load split, both fields carry it)")



# ---------------------------------------------------------------------------
# Parsing what each compiler prints
# ---------------------------------------------------------------------------

_NV_ENTRY = re.compile(r"Compiling entry function '([^']+)'")
_NV_SPILL = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_NV_REGS = re.compile(r"Used (\d+) registers")
_NV_SMEM = re.compile(r"(\d+) bytes smem")


def parse_ptxas(text: str, kernel: str) -> dict[str, int]:
    """`ptxas -v`'s numbers for the entry function `kernel`."""
    blocks = _NV_ENTRY.split(text)
    for name, body in zip(blocks[1::2], blocks[2::2], strict=True):
        if name != kernel:
            continue
        regs, spill = _NV_REGS.search(body), _NV_SPILL.search(body)
        if not regs or not spill:
            break
        got = {"registers": int(regs[1]), "stack_frame_bytes": int(spill[1]),
               "spill_store_bytes": int(spill[2]),
               "spill_load_bytes": int(spill[3])}
        if smem := _NV_SMEM.search(body):
            got["smem_bytes"] = int(smem[1])
        return got
    raise ValueError(f"no ptxas -v resource report for entry {kernel!r}")


_AMD_FIELD = re.compile(r"remark:\s+([A-Za-z][A-Za-z ]*?)(?: \[[^\]]*\])?: (\S+)")


def parse_amd_remarks(text: str, kernel: str) -> dict[str, int]:
    """`-Rpass-analysis=kernel-resource-usage`'s numbers for `kernel`."""
    found: dict[str, int] | None = None
    for line in text.splitlines():
        m = _AMD_FIELD.search(line)
        if not m:
            continue
        key, value = m[1].strip(), m[2]
        if key == "Function Name":
            if found is not None:
                break
            if value == kernel:
                found = {}
            continue
        if found is not None and re.fullmatch(r"\d+", value):
            found[key] = int(value)
    if not found or "VGPRs" not in found or "VGPRs Spill" not in found:
        raise ValueError(f"no kernel-resource-usage remark for {kernel!r}")
    return found


# ---------------------------------------------------------------------------
# The toolchains
# ---------------------------------------------------------------------------


def config_value(name: str) -> str:
    """A devtools/config.sh setting, by sourcing it (the one place it lives)."""
    done = subprocess.run(
        ["bash", "-c", f'. devtools/config.sh && printf %s "${name}"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    return done.stdout


def _which(tool: str) -> str | None:
    found = shutil.which(tool)
    if found:
        return found
    for c in (f"/usr/local/cuda/bin/{tool}", f"/opt/rocm/bin/{tool}"):
        if os.access(c, os.X_OK):
            return c
    return None


def _out(cmd: list[str]) -> str:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout + done.stderr


@dataclass(frozen=True)
class Backend:
    name: str
    arch: Arch
    tool: str
    flags_var: str
    ext: str

    def compiler(self) -> str:
        path = _which(config_value(self.tool.upper()) or self.tool)
        if path is None:
            raise RuntimeError(f"{self.tool} not found")
        return path

    def toolchain(self, compiler: str) -> str:
        if self.name == "cuda":
            m = re.search(r"release (\d+\.\d+)", _out([compiler, "--version"]))
            return f"nvcc {m[1]}" if m else "nvcc (unknown version)"
        root = Path(compiler).resolve().parent.parent
        version = root / ".info" / "version"
        if version.is_file():
            return f"hipcc / ROCm {version.read_text().strip()}"
        m = re.search(r"HIP version: (\S+)", _out([compiler, "--version"]))
        return f"hipcc / HIP {m[1]}" if m else "hipcc (unknown version)"

    def gpu(self) -> str:
        if self.name == "cuda":
            names = _out(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
            return names.strip().splitlines()[0] if names.strip() else "ABSENT"
        text = _out(["rocm-smi", "--showproductname"])
        series = re.search(r"Card Series:\s*(.+)", text)
        model = re.search(r"Card Model:\s*(\S+)", text)
        gfx = re.search(r"GFX Version:\s*(\S+)", text)
        if not series:
            return "ABSENT"
        extra = ", ".join(x for x in (
            model and f"device {model[1]}", gfx and gfx[1]) if x)
        return f"{series[1].strip()} ({extra})" if extra else series[1].strip()

    def command(self, compiler: str, src: Path, obj: Path) -> list[str]:
        flags = shlex.split(config_value(self.flags_var))
        if self.name == "cuda":
            return [compiler, *flags, f"-arch={self.arch.name}", "-Xptxas", "-v",
                    "-c", str(src), "-o", str(obj)]
        return [compiler, *flags, f"--offload-arch={self.arch.name}", "-x", "hip",
                "-Rpass-analysis=kernel-resource-usage", "-c", str(src),
                "-o", str(obj)]

    def units(self, cls: int | Spec) -> tuple[Unit, ...]:
        """The cell's translation units: one per batch kernel, in plan order."""
        spec = pq_spec(cls) if isinstance(cls, int) else cls
        emit = cuda.emit if self.name == "cuda" else hip.emit
        # an F kernel's one shared array (`ab_sh`) has a literal extent, so
        # the compiler reports it as static smem / LDS: no dynamic bytes
        # (an AsymCell, #113, is not a Spec and has no target: an MMD J cell)
        f = getattr(spec, "target", Target.J) is Target.F
        lane = 0 if f else self.arch.wavefront * 8
        return tuple(Unit(k.name, emit(k, self.arch), jsh_slots(k) * lane)
                     for k in cell_kernels(spec))


@dataclass(frozen=True)
class Unit:
    """One emitted translation unit: a batch kernel's name and text, and the
    dynamic shared bytes its own declaration asks for at the standard launch."""

    name: str
    text: str
    jsh_dynamic_bytes: int


def jsh_slots(k: Kernel) -> int:
    """Doubles per lane of the kernel's SHARED declarations (the `Jsh` rows)."""
    return sum(d.shape[0].value for d in k.body
               if isinstance(d, Decl) and d.space is Space.SHARED
               and d.shape and isinstance(d.shape[0], Lit))


def cell_sha256(units: Sequence[Unit]) -> str:
    """The cell's source hash: its batch texts concatenated in plan order
    (for one batch, the hash of the one text)."""
    return hashlib.sha256("".join(u.text for u in units).encode()).hexdigest()


BACKENDS = {
    "cuda": Backend("cuda", SM_86, "nvcc", "NVCC_FLAGS", "cu"),
    "hip": Backend("hip", GFX1200, "hipcc", "HIPCC_FLAGS", "hip"),
}


def host() -> str:
    """The hostname, marked when it is a container's (its id, not the box's)."""
    name = socket.gethostname()
    return f"{name} (container)" if Path("/.dockerenv").exists() else name


COMPILE_TIMEOUT_S = 1800  # per unit; a K cell's unrolled digest asks more


class CompileTimeout(RuntimeError):
    """A unit's compile did not finish within its budget: no row, and the
    budget, the unit and its size are in the message (#205)."""


def compile_unit(backend: Backend, unit: Unit, *, compiler: str,
                 workdir: Path, timeout: float = COMPILE_TIMEOUT_S
                 ) -> tuple[dict[str, int], float, str]:
    """Compile one translation unit with the measurement flag.

    Returns the compiler's numbers for the unit's kernel, the compile's wall
    time, and the command as recorded (paths shown as `<file>`). A compile
    past `timeout` seconds is killed and raises `CompileTimeout`.
    """
    src = workdir / f"{unit.name}.{backend.ext}"
    obj = workdir / f"{unit.name}.o"
    src.write_text(unit.text)
    cmd = backend.command(compiler, src, obj)
    t0 = time.perf_counter()
    # its own session, so a timeout kills the driver's children too (nvcc's
    # cicc / ptxas, hipcc's clang -cc1), not the driver alone
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        done = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired as e:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise CompileTimeout(
            f"{backend.tool} did not finish compiling {unit.name} "
            f"({len(unit.text)} bytes of source) within {timeout:.0f} s"
        ) from e
    finally:  # a K cell's sources run to hundreds of MB: keep none
        src.unlink(missing_ok=True)
        obj.unlink(missing_ok=True)
    compile_s = time.perf_counter() - t0
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{done.stderr[-4000:]}")
    report = done.stdout + done.stderr
    if backend.name == "cuda":
        raw = parse_ptxas(report, unit.name)
    else:
        raw = parse_amd_remarks(report, unit.name)
    shown = [Path(c).name if i == 0 else c for i, c in enumerate(cmd)]
    shown = [f"<{Path(c).name}>" if c in (str(src), str(obj)) else c
             for c in shown]
    return raw, compile_s, " ".join(shown)


def _numbers(backend: Backend,
             raw: Mapping[str, int]) -> tuple[int, int, int, int | None]:
    """(registers, spill_store, spill_load, shared_bytes) of one unit's raw."""
    if backend.name == "cuda":
        # shared: None when ptxas printed no smem
        return (raw["registers"], raw["spill_store_bytes"],
                raw["spill_load_bytes"], raw.get("smem_bytes"))
    spill = raw["VGPRs Spill"] + raw.get("SGPRs Spill", 0)
    return raw["VGPRs"], spill, spill, raw.get("LDS Size")


def cell_row(backend: Backend, spec: Spec, units: Sequence[Unit],
             compiled: Sequence[tuple[dict[str, int], float, str]], *,
             toolchain: str, gpu: str) -> tuple[MeasuredKey, Measured]:
    """One cell's row, aggregated over its batch kernels (module docstring)."""
    if len(units) != len(compiled) or not units:
        raise ValueError(f"{spec.key()}: {len(units)} units, {len(compiled)} "
                         "compiles")
    nums = [_numbers(backend, raw) for raw, _, _ in compiled]
    shared = [n[3] for n in nums if n[3] is not None]
    if len(units) == 1:
        command = compiled[0][2]
        raw = {**compiled[0][0], "jsh_dynamic_bytes": units[0].jsh_dynamic_bytes}
    else:
        raw = {f"{u.name}:{k}": v
               for u, (r, _, _) in zip(units, compiled, strict=True)
               for k, v in {**r, "jsh_dynamic_bytes": u.jsh_dynamic_bytes}.items()}
        stem = units[0].name.rsplit("_", 2)[0]
        command = compiled[0][2].replace(units[0].name, f"{stem}_*")
    row = Measured(
        registers=max(n[0] for n in nums),
        spill_store=sum(n[1] for n in nums),
        spill_load=sum(n[2] for n in nums),
        shared_bytes=max(shared) if shared else None,
        compile_s=round(sum(c[1] for c in compiled), 3),
        toolchain=toolchain, arch=backend.arch.name,
        host=host(), gpu=gpu,
        date=datetime.date.today().isoformat(),
        kernel=",".join(u.name for u in units),
        spill_unit=NV_SPILL_UNIT if backend.name == "cuda" else AMD_SPILL_UNIT,
        command=command, source_sha256=cell_sha256(units), raw=raw)
    return measured_key(spec, backend.arch, toolchain), row


def measure(backend: Backend, cls: int | Spec, *, compiler: str,
            toolchain: str, gpu: str, workdir: Path,
            timeout: float = COMPILE_TIMEOUT_S,
            jobs: int = 1) -> tuple[MeasuredKey, Measured]:
    """Compile one cell's kernels with the measurement flag; one row.

    `cls` is a total L (`pq_spec(L)`, one batch) or a full `Spec` -- the
    `n_batches` axis (#52): a split cell compiles each of its
    `build_pq_kernels` units (`build_k_kernels` for K) -- `jobs` at a
    time -- and aggregates them into one row.
    """
    spec = pq_spec(cls) if isinstance(cls, int) else cls
    units = backend.units(spec)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        compiled = list(pool.map(
            lambda u: compile_unit(backend, u, compiler=compiler,
                                   workdir=workdir, timeout=timeout), units))
    return cell_row(backend, spec, units, compiled, toolchain=toolchain, gpu=gpu)


def remeasure(names: Sequence[str], cells: Sequence[int | Spec], jobs: int,
              path: Path = STORE, *,
              timeout: float = COMPILE_TIMEOUT_S) -> dict[str, Measured]:
    """Recompile every cell on each backend and merge its row into the store.

    The pool runs over translation units, not cells, so a 9-batch cell's
    kernels compile side by side; `jobs` bounds how many compilers run at
    once (#52's memory hazard: keep it modest for large classes).

    A cell one of whose units fails or runs past `timeout` gets NO row (a
    partial cell is not a cell); the others are still merged, the store is
    saved after every finished cell (a K sweep runs for hours, #205), and
    the failures are raised together at the end, each with its reason.
    """
    got = load(path) if path.exists() else {}
    specs = [pq_spec(c) if isinstance(c, int) else c for c in cells]
    failed: list[str] = []
    for bname in names:
        b = BACKENDS[bname]
        compiler = b.compiler()
        toolchain, gpu = b.toolchain(compiler), b.gpu()
        cell_units = [b.units(s) for s in specs]
        with tempfile.TemporaryDirectory(prefix=f"fj-measure-{bname}-") as tmp:
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                # Each cell's LARGEST unit is submitted first, every cell's
                # before any cell's second: a cell whose unit fails gets no
                # row, so its other units are cancelled rather than compiled
                # (#205: a K cell's units run to hours each).
                futures: list[list[Future | None]] = [
                    [None] * len(units) for units in cell_units]
                order = [sorted(range(len(units)),
                                key=lambda i, us=units: -len(us[i].text))
                         for units in cell_units]

                dead: set[int] = set()

                def cancel_rest(f: Future, c: int, fut=futures,
                                dead=dead) -> None:
                    if not f.cancelled() and f.exception() is not None:
                        dead.add(c)
                        for g in fut[c]:
                            if g is not None:
                                g.cancel()

                for rank in range(max((len(u) for u in cell_units), default=0)):
                    for c, units in enumerate(cell_units):
                        if rank >= len(units) or c in dead:
                            continue
                        i = order[c][rank]
                        f = pool.submit(compile_unit, b, units[i],
                                        compiler=compiler, workdir=Path(tmp),
                                        timeout=timeout)
                        futures[c][i] = f
                        f.add_done_callback(lambda f, c=c: cancel_rest(f, c))
                for spec, units, fs in zip(specs, cell_units, futures,
                                           strict=True):
                    try:
                        if any(f is None for f in fs):  # never submitted
                            raise CancelledError
                        compiled = [f.result() for f in fs]
                    except (RuntimeError, CancelledError) as e:
                        for f in fs:  # the failure, not a sibling's cancel
                            if f is not None and not f.cancelled():
                                f.exception()  # wait for it
                        first = next((f.exception() for f in fs
                                      if f is not None and not f.cancelled()
                                      and f.exception() is not None), e)
                        why = f"{bname} {spec.key()}: {first}".splitlines()[0]
                        failed.append(why)
                        print(f"measured: NO ROW -- {why}", file=sys.stderr,
                              flush=True)
                        continue
                    key, row = cell_row(b, spec, units, compiled,
                                        toolchain=toolchain, gpu=gpu)
                    got = merge(got, key, combine(got.get(key.key()), row))
                    save(got, path)
                    print(f"{key.key()}: {row.registers} regs, spill "
                          f"{row.spill_store}/{row.spill_load}, "
                          f"{row.compile_s}s", flush=True)
    save(got, path)
    if failed:
        raise RuntimeError(f"{len(failed)} cells got no row:\n"
                           + "\n".join(failed))
    return got


def ledger_todo(backend: Backend, toolchain: str,
                store: Mapping[str, Measured]) -> list[Spec]:
    """THE LEVER LEDGER's cells (#362; module docstring) that `backend` under
    `toolchain` must compile: no row at their key, or a row whose
    `source_sha256` is not today's emitted text."""
    out = []
    for spec in ledger_cells():
        row = store.get(MeasuredKey(spec, backend.arch, toolchain).key())
        if row is None or row.source_sha256 != cell_sha256(
                backend.units(spec)):
            out.append(spec)
    return out


def remeasure_ledger(names: Sequence[str], jobs: int, path: Path = STORE,
                     ledger: Path | None = None) -> dict[str, Measured]:
    """`--remeasure --ledger`: the missing or stale ledger rows on each
    backend through `remeasure`, then the ledger file rewritten from the
    store (`leverledger.write`). A failed cell gets no row and fails the
    run -- and the ledger, which refuses a missing row."""
    got = load(path) if path.exists() else {}
    for bname in names:
        b = BACKENDS[bname]
        todo = ledger_todo(b, b.toolchain(b.compiler()), got)
        print(f"measured: ledger {bname}: {len(todo)} cells to compile",
              flush=True)
        if todo:
            got = remeasure([bname], todo, jobs, path)
    leverledger.write(load(path),
                      leverledger.LEDGER if ledger is None else ledger)
    return got


# ---------------------------------------------------------------------------
# #110: reading the card's performance counters (module docstring, COUNTERS)
# ---------------------------------------------------------------------------

class CouldNotMeasure(RuntimeError):
    """The profiler could not report counters: no row, and never a zero."""


# The exact counter names asked for, and the base unit ncu must print for each.
NCU_DADD = "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"
NCU_DMUL = "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"
NCU_DFMA = "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"
NCU_READ = "dram__bytes_read.sum"
NCU_WRITE = "dram__bytes_write.sum"
NCU_TIME = "gpu__time_duration.sum"
NCU_UNITS = {NCU_DADD: "inst", NCU_DMUL: "inst", NCU_DFMA: "inst",
             NCU_READ: "byte", NCU_WRITE: "byte", NCU_TIME: "ns"}
NCU_METRICS = tuple(NCU_UNITS)

ROCPROF_RD32 = "GL2C_EA_RDREQ_32B_sum"
ROCPROF_RD64 = "GL2C_EA_RDREQ_64B_sum"
ROCPROF_RD128 = "GL2C_EA_RDREQ_128B_sum"
ROCPROF_WR = "GL2C_EA_WRREQ_sum"
ROCPROF_WR64 = "GL2C_EA_WRREQ_64B_sum"
ROCPROF_COUNTERS = (ROCPROF_RD32, ROCPROF_RD64, ROCPROF_RD128, ROCPROF_WR,
                    ROCPROF_WR64)
# #140, measured on the RX 9060 XT: gfx1200's GL2C block collects at most
# four of these in one pass -- all five abort rocprofv3 with "Request exceeds
# the capabilities of the hardware to collect" (error code 38), after which
# the profiled process HANGS rather than exits. So two passes, read and
# written, each a separate run of the same binary; `parse_rocprof_csv` joins
# them dispatch by dispatch.
ROCPROF_PASSES = ((ROCPROF_RD32, ROCPROF_RD64, ROCPROF_RD128),
                  (ROCPROF_WR, ROCPROF_WR64))
ROCPROF_TIMEOUT_S = 600  # per pass: a pass that aborted hangs, so bound it

NV_COUNTER_UNIT = (
    "ncu: fp64_flops = dadd + dmul + 2 x dfma thread instructions executed "
    "(predicated on); dram_bytes_read / _written = dram__bytes_read.sum / "
    "dram__bytes_write.sum, DRAM traffic at 32-byte-sector granularity; "
    "counter_ns = gpu__time_duration.sum under replay, --clock-control none")
AMD_COUNTER_UNIT = (
    "rocprofv3: fp64_flops NOT COUNTED (gfx1200 exposes no fp64 instruction "
    "counter); dram_bytes_read = 32 x GL2C_EA_RDREQ_32B_sum + 64 x "
    "GL2C_EA_RDREQ_64B_sum + 128 x GL2C_EA_RDREQ_128B_sum, dram_bytes_written "
    "= 64 x GL2C_EA_WRREQ_64B_sum + 32 x (GL2C_EA_WRREQ_sum - "
    "GL2C_EA_WRREQ_64B_sum): GL2C <-> memory-fabric bytes, i.e. L2 misses "
    "the Infinity Cache may serve, an UPPER bound on DRAM bytes; counter_ns "
    "= End_Timestamp - Start_Timestamp under counter collection, of the "
    "read pass (two passes: gfx1200's GL2C collects at most four of the "
    "five counters at once)")


@dataclass(frozen=True)
class KernelCounters:
    """One profiled launch: the kernel's name (no signature), every counter
    by its exact name, and the four counter fields read off them (None:
    not counted)."""

    kernel: str
    raw: Mapping[str, int]
    fp64_flops: int | None
    dram_bytes_read: int | None
    dram_bytes_written: int | None
    counter_ns: float | None


def _int(v: str) -> int:
    v = v.strip().replace(",", "")
    if re.fullmatch(r"\d+", v):
        return int(v)
    f = float(v)
    if f != int(f) or f < 0:
        raise ValueError(f"counter value {v!r} is not a count")
    return int(f)


def parse_ncu_csv(text: str) -> list[KernelCounters]:
    """`ncu --csv --print-units base`'s rows, one `KernelCounters` per
    profiled launch in launch order (ncu's ID). Refuses a metric printed in
    a unit other than NCU_UNITS', and a launch missing one of NCU_METRICS."""
    start = text.find('"ID"')
    if start < 0:
        raise ValueError("no ncu CSV header in the output")
    launches: dict[int, tuple[str, dict[str, int]]] = {}
    for rec in csv.DictReader(io.StringIO(text[start:])):
        if not rec.get("ID") or rec.get("Metric Name") is None:
            continue
        name, unit = rec["Metric Name"], rec.get("Metric Unit", "")
        if name in NCU_UNITS and unit != NCU_UNITS[name]:
            raise ValueError(f"ncu printed {name} in {unit!r}, not "
                             f"{NCU_UNITS[name]!r}")
        kernel = rec["Kernel Name"].split("(", 1)[0].strip()
        got = launches.setdefault(int(rec["ID"]), (kernel, {}))[1]
        got[name] = _int(rec["Metric Value"])
    out = []
    for i in sorted(launches):
        kernel, raw = launches[i]
        missing = [m for m in NCU_METRICS if m not in raw]
        if missing:
            raise ValueError(f"ncu launch {i} ({kernel}) has no {missing}")
        out.append(KernelCounters(
            kernel=kernel, raw=raw,
            fp64_flops=raw[NCU_DADD] + raw[NCU_DMUL] + 2 * raw[NCU_DFMA],
            dram_bytes_read=raw[NCU_READ], dram_bytes_written=raw[NCU_WRITE],
            counter_ns=float(raw[NCU_TIME])))
    return out


def _rocprof_dispatches(text: str) -> list[tuple[int, str, dict[str, int], int]]:
    """One counter_collection CSV's dispatches, in dispatch order: (id,
    kernel, every counter by name, End - Start ns)."""
    dispatches: dict[int, tuple[str, dict[str, int], int]] = {}
    for rec in csv.DictReader(io.StringIO(text)):
        if not rec.get("Dispatch_Id"):
            continue
        d = int(rec["Dispatch_Id"])
        kernel = rec["Kernel_Name"].split("(", 1)[0].strip()
        ns = _int(rec["End_Timestamp"]) - _int(rec["Start_Timestamp"])
        got = dispatches.setdefault(d, (kernel, {}, ns))[1]
        got[rec["Counter_Name"]] = _int(rec["Counter_Value"])
    return [(d, *dispatches[d]) for d in sorted(dispatches)]


def parse_rocprof_csv(*texts: str) -> list[KernelCounters]:
    """`rocprofv3 --pmc ... --output-format csv`'s counter_collection rows,
    one `KernelCounters` per dispatch in dispatch order. Several texts are
    the PASSES of one run (ROCPROF_PASSES), each a separate run of the same
    binary: joined dispatch by dispatch, which must name the same kernels in
    the same order, with counter_ns the FIRST pass's. Refuses a dispatch
    missing one of ROCPROF_COUNTERS, a counter reported by two passes, and
    passes that disagree on what was launched."""
    if not texts:
        raise ValueError("no rocprofv3 counter CSV")
    first, *rest = [_rocprof_dispatches(t) for t in texts]
    for other in rest:
        if [k for _, k, _, _ in other] != [k for _, k, _, _ in first]:
            raise ValueError(
                "rocprofv3 passes launched different kernels: "
                f"{[k for _, k, _, _ in first]} vs {[k for _, k, _, _ in other]}")
        for (d, _, raw, _), (_, _, more, _) in zip(first, other, strict=True):
            twice = sorted(set(raw) & set(more))
            if twice:
                raise ValueError(f"rocprofv3 dispatch {d}: {twice} reported "
                                 "by two passes")
            raw.update(more)
    out = []
    for d, kernel, raw, ns in first:
        missing = [c for c in ROCPROF_COUNTERS if c not in raw]
        if missing:
            raise ValueError(f"rocprofv3 dispatch {d} ({kernel}) has no "
                             f"{missing}")
        wr64 = raw[ROCPROF_WR64]
        if raw[ROCPROF_WR] < wr64:
            raise ValueError(f"rocprofv3 dispatch {d}: {ROCPROF_WR} < "
                             f"{ROCPROF_WR64}")
        out.append(KernelCounters(
            kernel=kernel, raw=raw, fp64_flops=None,
            dram_bytes_read=(32 * raw[ROCPROF_RD32] + 64 * raw[ROCPROF_RD64]
                             + 128 * raw[ROCPROF_RD128]),
            dram_bytes_written=64 * wr64 + 32 * (raw[ROCPROF_WR] - wr64),
            counter_ns=float(ns)))
    return out


_AVAIL_AGENT = re.compile(r"^\s*Name\s*:\s*(\S+)\s*$")
_AVAIL_COUNTER = re.compile(r"^\s*Counter_Name\s*:\s*(\S+)\s*$")
# A counter name that counts fp64 operations, in rocprofiler-sdk's spelling:
# SQ_INSTS_VALU_{ADD,MUL,FMA,TRANS,MFMA}_F64, SQ_INSTS_VALU_FLOPS_FP64,
# TOTAL_64_OPS, MfmaFlops*. A byte counter (GL2C_EA_RDREQ_64B) is not one.
_FP64_COUNTER = re.compile(r"F64|FP64|FLOP|64_OPS", re.I)


def parse_rocprof_avail(text: str) -> dict[str, frozenset[str]]:
    """`rocprofv3 --list-avail`'s counter names, per agent (`gfx1200`).
    Refuses output that names no counter: an empty list is not a card that
    counts nothing."""
    got: dict[str, set[str]] = {}
    agent = None
    for line in text.splitlines():
        if m := _AVAIL_AGENT.match(line):
            agent = m[1]
        elif (m := _AVAIL_COUNTER.match(line)) and agent is not None:
            got.setdefault(agent, set()).add(m[1])
    if not got:
        raise ValueError("rocprofv3 --list-avail named no counter")
    return {a: frozenset(n) for a, n in got.items()}


def fp64_counters(names: frozenset[str] | set[str]) -> list[str]:
    """The counters among `names` that count fp64 operations (or FLOPs)."""
    return sorted(n for n in names if _FP64_COUNTER.search(n))


PROFILERS = {"cuda": "ncu", "hip": "rocprofv3"}


def counter_tool(path: str, backend: str) -> str:
    """The profiler as it reports itself: `ncu 2025.3.1.0`, `rocprofv3 1.1.0`."""
    text = _out([path, "--version"])
    pat = r"^Version (\S+)" if backend == "cuda" else r"^\s*version: (\S+)"
    m = re.search(pat, text, re.M)
    name = PROFILERS[backend]
    return f"{name} {m[1]}" if m else f"{name} (unknown version)"


def profile_commands(backend: str, tool: str, argv: Sequence[str], *,
                     kernel_regex: str, outdir: Path) -> list[list[str]]:
    """The profiler invocations around `argv`, one per pass (module
    docstring, COUNTERS): ncu replays in one; rocprofv3 needs
    ROCPROF_PASSES, each writing under its own `outdir/pass<i>`."""
    if backend == "cuda":
        return [[tool, "--csv", "--print-units", "base", "--clock-control",
                 "none", "--metrics", ",".join(NCU_METRICS), "--kernel-name",
                 f"regex:{kernel_regex}", *argv]]
    return [[tool, "--pmc", *pmc, "--kernel-include-regex", kernel_regex,
             "--output-format", "csv", "-d", str(outdir / f"pass{i}"), "-o",
             "counters", "--", *argv] for i, pmc in enumerate(ROCPROF_PASSES)]


# rocprofv3's own report that a pass's counter set does not fit the card.
_ROCPROF_CANNOT = "exceeds the capabilities of the hardware to collect"
# #140, measured on the RX 9060 XT (rocprofv3 1.1.0, ROCm 7.2.4, amdgpu
# module 6.16.13, host Linux 6.8): every GL2C <-> EA byte counter reads 0
# for a launch that streams 384 MiB (as do TCP_*, TA_*, SQ_INSTS_*), while
# GRBM_GUI_ACTIVE and SQ_WAVES count. A launch whose byte counters ALL read
# 0 is refused: no J-engine kernel moves no bytes, so that 0 is the counter
# not counting, and an absence never becomes a zero.
ROCPROF_ZERO_BYTES = "rocprofv3's GL2C <-> EA byte counters all read 0"


_NOT_LOADED = re.compile(r"error while loading shared libraries: (\S+?):")


def profile(backend: str, argv: Sequence[str], *, kernel_regex: str,
            workdir: Path) -> tuple[list[KernelCounters], str, str]:
    """Run `argv` under `backend`'s profiler; its launches matching
    `kernel_regex`, the profiler as it reports itself, and the invocation as
    recorded (the binary and output directory shown as `<name>`).

    Raises CouldNotMeasure when the profiler is absent, cannot load, is
    refused the counters, or reports no launch -- never a row of zeros.
    """
    name = PROFILERS[backend]
    tool = _which(name)
    if tool is None:
        raise CouldNotMeasure(f"{name} not found: could not measure")
    outdir = workdir / f"{name}-out"
    cmds = profile_commands(backend, tool, argv, kernel_regex=kernel_regex,
                            outdir=outdir)
    texts = []
    workdir.mkdir(parents=True, exist_ok=True)
    for i, cmd in enumerate(cmds):
        try:
            # rocprofv3 drops a .rocprofv3/ directory in its cwd: keep it in
            # workdir, out of the checkout
            done = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=None if backend == "cuda" else workdir,
                timeout=3600 if backend == "cuda" else ROCPROF_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise CouldNotMeasure(f"{name}: {e}: could not measure") from e
        report = done.stdout + done.stderr
        if "ERR_NVGPUCTRPERM" in report:
            raise CouldNotMeasure(
                "ncu: ERR_NVGPUCTRPERM -- this host restricts GPU performance "
                "counters to admins (NVreg_RestrictProfilingToAdminUsers): "
                "could not measure")
        libs = sorted(set(_NOT_LOADED.findall(report)))
        if libs:
            raise CouldNotMeasure(f"{name} cannot load {', '.join(libs)}: "
                                  "could not measure")
        if backend == "cuda":
            if done.returncode != 0 or "==ERROR==" in report:
                raise CouldNotMeasure(f"ncu exited {done.returncode}: could "
                                      f"not measure\n{report[-2000:]}")
            if '"ID"' in report:
                texts.append(report)
            continue
        if _ROCPROF_CANNOT in report:
            raise CouldNotMeasure(
                f"rocprofv3 pass {i} ({', '.join(ROCPROF_PASSES[i])}): "
                f"{_ROCPROF_CANNOT}: could not measure")
        files = sorted((outdir / f"pass{i}").rglob("*counter_collection.csv"))
        if done.returncode != 0 or len(files) > 1:
            raise CouldNotMeasure(
                f"rocprofv3 pass {i} exited {done.returncode} with "
                f"{len(files)} counter files: could not measure\n"
                f"{report[-2000:]}")
        if not files:
            texts = []
            break
        texts.append(files[0].read_text())
    if backend == "cuda":
        got = parse_ncu_csv(texts[0]) if texts else []
    else:
        got = parse_rocprof_csv(*texts) if texts else []
    if not got:
        raise CouldNotMeasure(f"{name} profiled no launch matching "
                              f"{kernel_regex!r}: could not measure")
    for c in got:
        if backend == "hip" and not any(c.raw[n] for n in ROCPROF_COUNTERS):
            raise CouldNotMeasure(
                f"{ROCPROF_ZERO_BYTES} for {c.kernel} ({c.counter_ns:.0f} ns "
                "under collection): the counter not counting, not a "
                "measurement -- could not measure")
    # the binary as `<name>`, a pass's output directory as `<name-out/passN>`
    hidden = {str(argv[0]): f"<{Path(argv[0]).name}>"} if argv else {}
    hidden.update({str(outdir / f"pass{i}"): f"<{outdir.name}/pass{i}>"
                   for i in range(len(cmds))})
    shown = [" ".join([Path(cmd[0]).name, *(hidden.get(c, c) for c in cmd[1:])])
             for cmd in cmds]
    return got, counter_tool(tool, backend), " ; ".join(shown)


def counter_rows(backend: Backend, spec: Spec, units: Sequence[Unit],
                 launches: Sequence[KernelCounters], *, toolchain: str,
                 gpu: str, tool: str, command: str) -> dict[str, Measured]:
    """One cell's counter rows, keyed `kernel_key(MeasuredKey, batch)`: one
    per batch kernel, `units` in plan order, each matched by name to exactly
    one profiled launch (measured_store, THE COUNTER ROW SHAPE)."""
    if not units:
        raise ValueError(f"{spec.key()}: no units")
    key = MeasuredKey(spec, backend.arch, toolchain)
    out = {}
    for b, u in enumerate(units):
        mine = [c for c in launches if c.kernel == u.name]
        if len(mine) != 1:
            raise ValueError(f"{spec.key()}: kernel {u.name} was profiled "
                             f"{len(mine)} times, not once")
        (c,) = mine
        out[kernel_key(key, b)] = Measured(
            fp64_flops=c.fp64_flops, dram_bytes_read=c.dram_bytes_read,
            dram_bytes_written=c.dram_bytes_written, counter_ns=c.counter_ns,
            toolchain=toolchain, arch=backend.arch.name, host=host(), gpu=gpu,
            date=datetime.date.today().isoformat(), kernel=u.name,
            command=command,
            source_sha256=hashlib.sha256(u.text.encode()).hexdigest(),
            raw=dict(c.raw), counter_tool=tool,
            counter_unit=(NV_COUNTER_UNIT if backend.name == "cuda"
                          else AMD_COUNTER_UNIT))
    return out


def print_roofline(path: Path = ROOFLINE) -> int:
    """#112's view: each SUBGRID cell's verdict, and its kernels', against
    the card's measured ridge (`measured_store.roofline_point`), for every
    card and toolchain the counter store holds; a card in UNMEASURED is named
    with its reason. Exit 2 when the store cannot be read."""
    try:
        got = load_counters(path)
    except (OSError, ValueError) as e:
        print(f"measured: {e}", file=sys.stderr)
        return 2
    cells = {split_kernel_key(k)[0] for k in got}
    where = sorted({c.split("@", 1)[1] for c in cells})
    for at in where:
        for spec in SUBGRID:
            c = f"{spec.key()}@{at}"
            if c not in cells:
                print(f"{c}: no counter rows")
                continue
            p = cell_roofline(got, c)
            ks = kernel_roofline(got, c)
            print(f"{c}: I {p.intensity:.2f} FLOP/B vs ridge {p.ridge:.2f}, "
                  f"{p.achieved_flops_per_s / 1e9:.1f} GFLOP/s "
                  f"({p.of_peak:.0%} of peak) -> "
                  f"{'MEMORY' if p.memory_bound else 'COMPUTE'}-bound; kernels "
                  f"{sum(k.memory_bound for k in ks)}/{len(ks)} memory-bound, "
                  f"I in [{min(k.intensity for k in ks):.2f}, "
                  f"{max(k.intensity for k in ks):.2f}]")
    for arch, why in UNMEASURED.items():
        print(f"{arch}: no rows -- {why}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m felizjueves.measured")
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--backend", choices=("cuda", "hip", "both"), default="both")
    ap.add_argument("--L", dest="Ls", type=int, nargs="+", default=list(DEFAULT_LS))
    ap.add_argument("--subgrid", action="store_true",
                    help="remeasure SUBGRID (every n_batches) instead of --L")
    ap.add_argument("--n-batches", type=int, nargs="+", default=None,
                    help="with --subgrid: only the cells of these n_batches")
    ap.add_argument("--nine-vs-twelve", action="store_true",
                    help="print #60's 9-against-12 comparison per arch and L")
    ap.add_argument("--lever", default=None,
                    help="with --subgrid: the A/B cells of this lever "
                         "(#55; ab_cells: no-sieve adds L = 11, #162; "
                         "tuned drops its NOT_MEASURABLE cells, #163); a "
                         "kernel-body lever (#339) is added to whichever "
                         "cells --L / --subgrid / --target select")
    ap.add_argument("--ledger", action="store_true",
                    help="with --remeasure: every lever-ledger cell (#362) "
                         "with no row or a stale one, then rewrite "
                         "data/measured/lever_ledger.json")
    ap.add_argument("--lever-ab", default=None, metavar="LEVER",
                    help="print #55's A/B of LEVER against literal, per arch "
                         "and cell")
    ap.add_argument("--asym", action="store_true",
                    help="with --remeasure: #113's asymmetric study cells "
                         "(asym.ASYM_CELLS)")
    ap.add_argument("--asym-verdict", action="store_true",
                    help="print #113's asymmetric split against its "
                         "neighbours, per arch and cell")
    ap.add_argument("--roofline", action="store_true",
                    help="print #112's roofline verdict per cell and per "
                         "batch kernel, from the counter store")
    ap.add_argument("--subgrid-ls", type=int, nargs="+", default=None,
                    metavar="L", help="with --subgrid: only the cells of "
                                      "these L")
    ap.add_argument("--target", choices=("j", "k", "f"), default="j",
                    help="with --subgrid: the J subgrid (default) or "
                         "K_SUBGRID, its exchange twins (#205); f: the F "
                         "ladder's classes whose total L is in --L (#288)")
    ap.add_argument("--compile-timeout", type=float,
                    default=COMPILE_TIMEOUT_S, metavar="S",
                    help="per unit: a compile past S seconds is killed and "
                         "its cell gets no row")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--counters", type=Path, default=ROOFLINE)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if args.ledger:  # #362: its own cell set, nothing else selects
        if not args.remeasure or args.lever or args.subgrid or args.asym:
            print("measured: --ledger needs --remeasure, and takes no "
                  "--lever, --subgrid or --asym", file=sys.stderr)
            return 2
        names = ("cuda", "hip") if args.backend == "both" else (args.backend,)
        try:
            remeasure_ledger(names, args.j, args.store)
        except (RuntimeError, ValueError) as e:
            print(f"measured: {e}", file=sys.stderr)
            return 2
        return 0
    if args.remeasure:
        names = ("cuda", "hip") if args.backend == "both" else (args.backend,)
        try:
            # #339: a kernel-body lever rides on any cell set; a plan or
            # lowering arm is the J subgrid's A/B (ab_cells), as before
            body = bool(args.lever) and kind_of(args.lever) == KERNEL
            arm = args.lever if not body else None
            if arm and not args.subgrid:
                raise ValueError("--lever needs --subgrid")
            if args.asym and (args.subgrid or args.lever):
                raise ValueError("--asym is its own cell set")
            if args.target == "k" and (arm or args.asym or not args.subgrid):
                raise ValueError("--target k is K_SUBGRID: it needs "
                                 "--subgrid, and takes no plan or lowering "
                                 "lever")
            if args.target == "f" and (arm or args.asym or args.subgrid
                                       or args.n_batches or args.subgrid_ls):
                raise ValueError("--target f is the F ladder by --L: one "
                                 "batch, no plan or lowering lever, no "
                                 "subgrid")
            grid = (ab_cells(arm) if arm
                    else K_SUBGRID if args.target == "k" else SUBGRID)
            cells = (grid if args.subgrid else f_cells(args.Ls)
                     if args.target == "f" else args.Ls)
            if not cells:
                raise ValueError(f"--L {args.Ls} selects no cell")
            if body:  # the store's one recording path (lever_arm)
                grid = tuple(lever_arm(s, args.lever) for s in grid)
                cells = [lever_arm(pq_spec(c) if isinstance(c, int) else c,
                                   args.lever) for c in cells]
            if args.asym:
                grid = cells = ASYM_CELLS
            if (args.subgrid or args.asym) and args.n_batches:
                cells = [s for s in grid if s.n_batches in args.n_batches]
            if args.subgrid_ls:
                if not args.subgrid:
                    raise ValueError("--subgrid-ls needs --subgrid")
                cells = [s for s in cells if s.total_am in args.subgrid_ls]
            if arm:  # #163: said, never silently skipped
                for s, why in NOT_MEASURABLE.get(arm, {}).items():
                    print(f"measured: {s.key()} NOT MEASURABLE -- {why}",
                          file=sys.stderr)
            if args.compile_timeout != COMPILE_TIMEOUT_S:
                remeasure(names, cells, args.j, args.store,
                          timeout=args.compile_timeout)
            else:  # the call #163's CLI test pins, unchanged
                remeasure(names, cells, args.j, args.store)
        except (RuntimeError, ValueError) as e:
            print(f"measured: {e}", file=sys.stderr)
            return 2
        return 0
    if args.roofline:
        return print_roofline(args.counters)
    try:
        got = load(args.store)
    except (OSError, ValueError) as e:
        print(f"measured: {e}", file=sys.stderr)
        return 2
    if args.lever_ab:
        try:
            cells = ab_cells(args.lever_ab)
        except ValueError as e:
            print(f"measured: {e}", file=sys.stderr)
            return 2
        ms = 1e6
        for arch in (SM_86.name, GFX1200.name):
            for spec in cells:
                try:
                    c = lever_ab(got, arch, spec)
                except KeyError as e:
                    print(f"{arch} {e}")
                    continue
                print(f"{arch} L={c.L} b{c.n_batches}: spill {c.spill[0]} -> "
                      f"{c.spill[1]}; wide {c.wide[0] / ms:.3f}+-"
                      f"{c.wide[1] / ms:.3f} -> {c.wide[2] / ms:.3f}+-"
                      f"{c.wide[3] / ms:.3f} ms {c.wide_sign}; near "
                      f"{c.near[0] / ms:.3f}+-{c.near[1] / ms:.3f} -> "
                      f"{c.near[2] / ms:.3f}+-{c.near[3] / ms:.3f} ms "
                      f"{c.near_sign} -> {c.lever} {c.verdict.upper()}")
            if args.lever_ab == NO_SIEVE:  # #159's declared verdict, per arch
                try:
                    print(f"{arch}: the sieve "
                          f"{sieve_verdict(got, arch).upper()}")
                except KeyError as e:
                    print(f"{arch}: the sieve cannot be judged ({e})")
            if args.lever_ab == TUNED:  # #163's declared verdict, per arch
                for s, why in NOT_MEASURABLE[TUNED].items():
                    print(f"{arch} {s.key()}: NOT MEASURABLE -- {why}")
                try:
                    print(f"{arch}: the search "
                          f"{tuned_verdict(got, arch).upper()}")
                except KeyError as e:
                    print(f"{arch}: the search cannot be judged ({e})")
        return 0
    if args.asym_verdict:
        ms = 1e6
        for arch in (SM_86.name, GFX1200.name):
            for cell in ASYM_CELLS:
                try:
                    c = asym_vs_neighbours(got, arch, cell)
                except KeyError as e:
                    print(f"{arch} {e}")
                    continue
                lo, hi = c.neighbours

                def arms(t: tuple, lo: int = lo, hi: int = hi) -> str:
                    return "  ".join(f"{n} {m / ms:.3f}+-{q / ms:.3f}"
                                     for n, (m, q) in zip(
                                         (f"b{lo}", "asym", f"b{hi}"), t,
                                         strict=True))

                print(f"{arch} L={c.L} b{c.n_batches} {c.ks}: spill "
                      f"{c.spill}, regs {c.registers}; wide {arms(c.wide)} "
                      f"ms -> {c.wide_sign} vs b{c.wide_best}; near "
                      f"{arms(c.near)} ms -> {c.near_sign} vs "
                      f"b{c.near_best} => {c.verdict.upper()}")
            try:
                print(f"{arch}: {arch_verdict(got, arch).upper()}")
            except KeyError as e:
                print(f"{arch}: cannot judge ({e})")
        return 0
    if args.nine_vs_twelve:
        for arch in (SM_86.name, GFX1200.name):
            for L in SUBGRID_LS:
                try:
                    c = nine_vs_twelve(got, arch, L)
                except KeyError as e:
                    print(f"{arch} L={L}: {e}")
                    continue
                ms = 1e6
                print(f"{arch} L={L}: spill {c.spill[0]} -> {c.spill[1]}"
                      f"{' FEWER' if c.fewer_spills else ''}; wide "
                      f"{c.wide[0] / ms:.3f} -> {c.wide[1] / ms:.3f} ms "
                      f"(IQR {c.wide[2] / ms:.3f})"
                      f"{' FASTER' if c.faster_wide else ''}; near "
                      f"{c.near[0] / ms:.3f} -> {c.near[1] / ms:.3f} ms "
                      f"(IQR {c.near[2] / ms:.3f})"
                      f"{' FASTER' if c.faster_near else ''} -> 9 "
                      f"{'suffices' if c.nine_suffices else 'does NOT suffice'}")
        return 0
    for k, r in got.items():
        shown = ", ".join(f"{n}={getattr(r, n)}" for n in MEASUREMENTS
                          if getattr(r, n) is not None)
        print(f"{k}: {shown} on {r.gpu}, {r.date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
