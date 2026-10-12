"""The `--remeasure` driver: compile each cell, read what the compiler says.

    python -m felizjueves.measured                      # print the store
    python -m felizjueves.measured --remeasure [--backend cuda|hip|both]
                                   [--L 0 4 8] [-j N]   # one-batch classes
    python -m felizjueves.measured --remeasure --subgrid [-j N]  # SUBGRID
                                   [--n-batches 12]     # only those cells
    python -m felizjueves.measured --nine-vs-twelve     # #60's comparison
    python -m felizjueves.measured --remeasure --subgrid --lever recycle
    python -m felizjueves.measured --lever-ab recycle   # #55's comparison
    python -m felizjueves.measured --remeasure --asym   # #113's study cells
    python -m felizjueves.measured --asym-verdict       # #113's comparison
    python -m felizjueves.measured --roofline           # #112's verdicts

(with `PYTHONPATH=src uv run` in front, as for every entry point here.)

THE STORE IS `felizjueves.measured_store` (#94): the `Measured` record, the
file, the read path, the subgrid and the verdicts (#60's NINE AGAINST TWELVE,
#55's A LEVER A/B -- their definitions are that module's docstring). This
module is the half that knows a toolchain: it emits each cell's translation
units, runs nvcc / hipcc on them with a report flag, parses the report, and
merges one row per cell into the store. It compiles; it launches nothing on
a card and times nothing -- the wall-clock side of a row is written by
test/test_kernel_device.py's sweep. The `--nine-vs-twelve` and `--lever-ab`
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
- AMD: `rocprofv3 --pmc <ROCPROF_COUNTERS> --kernel-include-regex <re>
  --output-format csv -d <dir> -o counters -- <binary>`, one pass, and its
  `*counter_collection.csv` read by column name (`Dispatch_Id`,
  `Kernel_Name`, `Counter_Name`, `Counter_Value`, `Start_Timestamp`,
  `End_Timestamp`).

COULD NOT MEASURE IS NOT ZERO. A profiler that is absent, cannot load (this
container's rocprofv3 lacks `libdw.so.1` and `libhsa-amd-aqlprofile64.so.1`),
is refused the counters (ncu's `ERR_NVGPUCTRPERM` on a host that restricts
them to admins), or profiles no kernel raises `CouldNotMeasure` and yields
no row. A counter it did not report is None in the row, never 0.

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
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

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
from felizjueves.lower import LEVERS, RECYCLE, lower
from felizjueves.measured_store import (
    GPU_UTIL_FIELDS as GPU_UTIL_FIELDS,  # re-exported: test_kernel_device's sweep
)
from felizjueves.measured_store import (
    MEASUREMENTS,
    REPO,
    STORE,
    SUBGRID,
    SUBGRID_LS,
    UNMEASURED,
    Measured,
    cell_roofline,
    combine,
    kernel_key,
    kernel_roofline,
    lever_ab,
    lever_cells,
    load,
    load_counters,
    merge,
    nine_vs_twelve,
    save,
    split_kernel_key,
)
from felizjueves.measured_store import ROOFLINE as ROOFLINE
from felizjueves.measured_store import (
    save_counters as save_counters,  # re-exported: test_kernel_device's sweep
)
from felizjueves.spec import Spec

DEFAULT_LS = tuple(range(0, 17))


def cell_kernels(spec: Spec | AsymCell) -> tuple[Kernel, ...]:
    """A cell's kernels, BUILT then LOWERED: `build_pq_kernels` of `spec`
    without the lowering levers (`lower.LEVERS`, which neither the builder
    nor the plan layer accepts), then `lower` on each finished kernel as
    those levers say. No lowering lever: `build_pq_kernels(spec)`, the very
    same objects -- the literal arm is not touched. An `AsymCell` (#113) is
    the study's own batch list (`AsymCell.kernels`), never lowered."""
    if isinstance(spec, AsymCell):
        return spec.kernels()
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
        return tuple(Unit(k.name, emit(k, self.arch),
                          jsh_slots(k) * self.arch.wavefront * 8)
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


def compile_unit(backend: Backend, unit: Unit, *, compiler: str,
                 workdir: Path) -> tuple[dict[str, int], float, str]:
    """Compile one translation unit with the measurement flag.

    Returns the compiler's numbers for the unit's kernel, the compile's wall
    time, and the command as recorded (paths shown as `<file>`).
    """
    src = workdir / f"{unit.name}.{backend.ext}"
    obj = workdir / f"{unit.name}.o"
    src.write_text(unit.text)
    cmd = backend.command(compiler, src, obj)
    t0 = time.perf_counter()
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
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
            toolchain: str, gpu: str,
            workdir: Path) -> tuple[MeasuredKey, Measured]:
    """Compile one cell's kernels with the measurement flag; one row.

    `cls` is a total L (`pq_spec(L)`, one batch) or a full `Spec` -- the
    `n_batches` axis (#52): a split cell compiles each of its
    `build_pq_kernels` units in turn and aggregates them into one row.
    """
    spec = pq_spec(cls) if isinstance(cls, int) else cls
    units = backend.units(spec)
    compiled = [compile_unit(backend, u, compiler=compiler, workdir=workdir)
                for u in units]
    return cell_row(backend, spec, units, compiled, toolchain=toolchain, gpu=gpu)


def remeasure(names: Sequence[str], cells: Sequence[int | Spec], jobs: int,
              path: Path = STORE) -> dict[str, Measured]:
    """Recompile every cell on each backend and merge its row into the store.

    The pool runs over translation units, not cells, so a 9-batch cell's
    kernels compile side by side; `jobs` bounds how many compilers run at
    once (#52's memory hazard: keep it modest for large classes).
    """
    got = load(path) if path.exists() else {}
    specs = [pq_spec(c) if isinstance(c, int) else c for c in cells]
    for bname in names:
        b = BACKENDS[bname]
        compiler = b.compiler()
        toolchain, gpu = b.toolchain(compiler), b.gpu()
        cell_units = [b.units(s) for s in specs]
        with tempfile.TemporaryDirectory(prefix=f"fj-measure-{bname}-") as tmp:
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                futures = [[pool.submit(compile_unit, b, u, compiler=compiler,
                                        workdir=Path(tmp)) for u in units]
                           for units in cell_units]
                for spec, units, fs in zip(specs, cell_units, futures,
                                           strict=True):
                    key, row = cell_row(b, spec, units,
                                        [f.result() for f in fs],
                                        toolchain=toolchain, gpu=gpu)
                    got = merge(got, key, combine(got.get(key.key()), row))
                    print(f"{key.key()}: {row.registers} regs, spill "
                          f"{row.spill_store}/{row.spill_load}, "
                          f"{row.compile_s}s", flush=True)
    save(got, path)
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
    "= End_Timestamp - Start_Timestamp under counter collection")


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


def parse_rocprof_csv(text: str) -> list[KernelCounters]:
    """`rocprofv3 --pmc ... --output-format csv`'s counter_collection rows,
    one `KernelCounters` per dispatch in dispatch order. Refuses a dispatch
    missing one of ROCPROF_COUNTERS."""
    dispatches: dict[int, tuple[str, dict[str, int], int]] = {}
    for rec in csv.DictReader(io.StringIO(text)):
        if not rec.get("Dispatch_Id"):
            continue
        d = int(rec["Dispatch_Id"])
        kernel = rec["Kernel_Name"].split("(", 1)[0].strip()
        ns = _int(rec["End_Timestamp"]) - _int(rec["Start_Timestamp"])
        got = dispatches.setdefault(d, (kernel, {}, ns))[1]
        got[rec["Counter_Name"]] = _int(rec["Counter_Value"])
    out = []
    for d in sorted(dispatches):
        kernel, raw, ns = dispatches[d]
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


PROFILERS = {"cuda": "ncu", "hip": "rocprofv3"}


def counter_tool(path: str, backend: str) -> str:
    """The profiler as it reports itself: `ncu 2025.3.1.0`, `rocprofv3 1.1.0`."""
    text = _out([path, "--version"])
    pat = r"^Version (\S+)" if backend == "cuda" else r"^\s*version: (\S+)"
    m = re.search(pat, text, re.M)
    name = PROFILERS[backend]
    return f"{name} {m[1]}" if m else f"{name} (unknown version)"


def profile_command(backend: str, tool: str, argv: Sequence[str], *,
                    kernel_regex: str, outdir: Path) -> list[str]:
    """The profiler invocation around `argv` (module docstring, COUNTERS)."""
    if backend == "cuda":
        return [tool, "--csv", "--print-units", "base", "--clock-control",
                "none", "--metrics", ",".join(NCU_METRICS), "--kernel-name",
                f"regex:{kernel_regex}", *argv]
    return [tool, "--pmc", *ROCPROF_COUNTERS, "--kernel-include-regex",
            kernel_regex, "--output-format", "csv", "-d", str(outdir), "-o",
            "counters", "--", *argv]


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
    cmd = profile_command(backend, tool, argv, kernel_regex=kernel_regex,
                          outdir=outdir)
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise CouldNotMeasure(f"{name}: {e}: could not measure") from e
    report = done.stdout + done.stderr
    if "ERR_NVGPUCTRPERM" in report:
        raise CouldNotMeasure(
            "ncu: ERR_NVGPUCTRPERM -- this host restricts GPU performance "
            "counters to admins (NVreg_RestrictProfilingToAdminUsers): could "
            "not measure")
    libs = sorted(set(_NOT_LOADED.findall(report)))
    if libs:
        raise CouldNotMeasure(f"{name} cannot load {', '.join(libs)}: could "
                              "not measure")
    if backend == "cuda":
        if done.returncode != 0 or "==ERROR==" in report:
            raise CouldNotMeasure(f"ncu exited {done.returncode}: could not "
                                  f"measure\n{report[-2000:]}")
        got = parse_ncu_csv(report) if '"ID"' in report else []
    else:
        files = sorted(outdir.rglob("*counter_collection.csv"))
        if done.returncode != 0 or len(files) > 1:
            raise CouldNotMeasure(
                f"rocprofv3 exited {done.returncode} with {len(files)} counter "
                f"files: could not measure\n{report[-2000:]}")
        got = parse_rocprof_csv(files[0].read_text()) if files else []
    if not got:
        raise CouldNotMeasure(f"{name} profiled no launch matching "
                              f"{kernel_regex!r}: could not measure")
    shown = [Path(cmd[0]).name, *cmd[1:]]
    shown = [f"<{Path(c).name}>" if c in (str(argv[0]) if argv else "",
                                          str(outdir)) else c for c in shown]
    return got, counter_tool(tool, backend), " ".join(shown)


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
                    help="with --subgrid: the lever_cells of this lever (#55)")
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
    ap.add_argument("-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--counters", type=Path, default=ROOFLINE)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if args.remeasure:
        names = ("cuda", "hip") if args.backend == "both" else (args.backend,)
        try:
            if args.lever and not args.subgrid:
                raise ValueError("--lever needs --subgrid")
            if args.asym and (args.subgrid or args.lever):
                raise ValueError("--asym is its own cell set")
            grid = lever_cells(args.lever) if args.lever else SUBGRID
            cells = grid if args.subgrid else args.Ls
            if args.asym:
                grid = cells = ASYM_CELLS
            if (args.subgrid or args.asym) and args.n_batches:
                cells = [s for s in grid if s.n_batches in args.n_batches]
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
            cells = lever_cells(args.lever_ab)
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
