"""What a device toolchain reports about a generated kernel: `Measured` rows.

    python -m felizjueves.measured                      # print the store
    python -m felizjueves.measured --remeasure [--backend cuda|hip|both]
                                   [--L 0 4 8] [-j N]   # one-batch classes
    python -m felizjueves.measured --remeasure --subgrid [-j N]  # SUBGRID
                                   [--n-batches 12]     # only those cells
    python -m felizjueves.measured --nine-vs-twelve     # #60's comparison

(with `PYTHONPATH=src uv run` in front, as for every entry point here.)

OBSERVATIONS OF A TOOLCHAIN, NOT CLAIMS ABOUT THIS CODE. A row is what one
compiler, driven at one arch with one flag set, said about one emitted
translation unit: registers, spill traffic, and where and when it was asked.
A compiler bump moving a number is news about the compiler. Nothing here is
compared against `[JEngine, v2, Table 4]` -- their kernels, their compiler,
their A100; data/literature/jengine/README.md is why that table is not even
transcribed.

KEYED ON `device.MeasuredKey(spec, arch, toolchain).key()`, as #37 decided:
`Spec` is the generation key and has no arch, so an `sm_86` row and a
`gfx1200` row for the same class must be told apart by the arch, and two
compiler versions on one card by the toolchain.

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

Neither measurement flag changes code generation: both are reports. The rest
of the flag set is `devtools/config.sh`'s NVCC_FLAGS / HIPCC_FLAGS, read by
sourcing that file -- the one place that knows them -- and recorded in the row.

THE RECORD (#50). Every MEASUREMENT field is optional -- a compile-only row has
no runtime, a timing row has no register count -- but a row with all of them
None records nothing and is refused. The PROVENANCE quintet (`toolchain`,
`arch`, `host`, `gpu`, `date`) is required and non-empty, always.

THE STORE is one JSON file, `data/measured/kernels.json`, in the idiom of
`literature/tables.py`: a `"schema"` version and an `"about"` string up top,
and a refusal -- naming the path -- of a file that is malformed, of another
schema, or whose row disagrees with its own key. A missing store is an error,
not an empty one: `load` raises, so "nothing measured yet" (a valid file with
no rows) and "no store here" cannot be confused. One file for both arches,
not one per arch: rows are keyed on the arch already, `remeasure` merges
into what is there (a one-card remeasure diffs only that card's rows), and
the file's path is what the docs map points at.

THE SUBGRID (#52) is `SUBGRID`, the one declared set of `(L, n_batches)`
cells the compile-side measurement sweeps: L in {7, 10, 13, 16} x n_batches
in {1, 3, 6, 9}, `[JEngine, v2, §7]`'s own reporting range. Every cell is a
default-split `pq_spec(L, n_batches=b)` that the plan layer admits (its
partition refuses an empty batch, #64; test/test_measured.py checks every
cell). Tests parametrize over it, #53's wall-clock rows join on it, and a
cell is added here and nowhere else. Widen per experiment, never by default.

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
before: unprefixed `raw`, the one kernel's name and source hash.

`shared_bytes` is what the compiler reports, which is the STATIC footprint:
the kernel's `Jsh` is `extern __shared__`, sized at launch, so nvcc prints no
smem and hipcc an `LDS Size` of 0. The dynamic footprint is the kernel's own
declaration, not a toolchain observation: `raw` carries it as
`jsh_dynamic_bytes` (`Jsh` doubles per lane x the arch's wavefront x 8, at
the launch `kernel.launch_shape` fixes, blockDim.y = wavefront).

`compile_s` is the wall time of one compile while up to `-j` others run
beside it: a cost indication, never a benchmark.

WALL-CLOCK (#53) is `runtime_ns`, `repeats` and `spread_ns`, written by the
opt-in, `gpu`-marked sweep in test/test_kernel_device.py
(`test_<backend>_wallclock_over_the_subgrid`, run alone and uncontended; its
docstring is the protocol). What one cell's number IS: the host
`steady_clock` time of the cell's WHOLE launch sequence -- every batch kernel
of `build_pq_kernels(spec)`, in plan order, into one J zeroed beforehand
(outside the clock), then one device synchronize -- producing one `J_p`.
`runtime_ns` is the MEDIAN over `repeats` timed sequences after a warm-up,
and `spread_ns` the interquartile range of those same sequences, on the
sweep's wide-T input; min, max, the quartiles, the per-batch GPU-event
medians, the problem size, the box's load as sampled before the cell, and
the same statistics on a second, near-T input (`wallclock:near:*`) are in
`raw`, keyed `wallclock:...`.
The binary is built from the same emitted text with config.sh's flag set
(the report flag aside, which changes no code). Absolute times are NOT
comparable across vendors (gfx1200 rate-limits fp64): only the ranking over
`n_batches` within one arch is a claim.

ONE ROW CARRIES BOTH SIDES. A timing and a compile observation of one cell
share a `MeasuredKey`, so they are one row: `combine` keeps the other side's
measurements (and `raw`) when, and only when, both observed the same text
(`source_sha256`). A changed kernel drops the stale side rather than pairing
a register count with a time it did not produce. The row's `date`, `host`,
`gpu` and `command` are the latest observation's (a timing row records no
`command`, so the compile's is kept).

NINE AGAINST TWELVE (#60) is `nine_vs_twelve`: whether `[JEngine, v2, §7]`'s
"for L <= 16, at most 9 batches are needed" holds on OUR cards. DECLARED
BEFORE ANY 12-BATCH ROW EXISTED, and not to be tuned after reading one. For
one arch and one subgrid L, with r9 and r12 the rows of pq_spec(L, 9) and
pq_spec(L, 12) on that arch:

- SPILL: 12 reduces spill traffic iff  spill(r12) < spill(r9), spill being
  the row's spill_store + spill_load (the cell's whole spill, summed over
  its batch kernels; bytes on sm_86, register counts on gfx1200 -- only the
  within-arch comparison is made). Compiler output, so no tolerance.
- WALL-CLOCK, judged on BOTH of #53's inputs, separately: 12 improves on an
  input iff  median(r9) - median(r12) > max(IQR(r9), IQR(r12)), i.e. it is
  faster by more than the larger recorded spread. WIDE-T is runtime_ns /
  spread_ns; NEAR-T is raw `wallclock:near:median_ns` and q3 - q1. Near-T
  is the informative one (#53: there splitting wins from L = 13 and b9 is
  best at L = 16), but a win on EITHER input counts against 9.
- 9 SUFFICES at (arch, L) iff 12 does none of the three (SPILL, WIDE, NEAR);
  9 SUFFICES ON AN ARCH iff it suffices at every subgrid L. The two arches
  are judged apart and may disagree; sm_86 and gfx1200 are our cards, not
  the paper's A100, and that is the strongest form the claim can take here.

Nothing under `plan/` or `literature/` may import this module
(test/test_layering.py): a measurement may never feed a literature floor.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
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
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from felizjueves.device import GFX1200, SM_86, Arch, MeasuredKey
from felizjueves.emit import cuda, hip
from felizjueves.ir import Decl, Kernel, Lit, Space
from felizjueves.kernel import build_pq_kernels, pq_spec
from felizjueves.spec import Spec

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "data" / "measured" / "kernels.json"
DEFAULT_LS = tuple(range(0, 17))

# THE declared subgrid (#52): see the module docstring. One edit adds a cell.
SUBGRID_LS = (7, 10, 13, 16)
SUBGRID: tuple[Spec, ...] = tuple(
    pq_spec(L, n_batches=b) for L in SUBGRID_LS for b in (1, 3, 6, 9))

SCHEMA = 1

ABOUT = ("Measured rows (felizjueves.measured): what one device toolchain "
         "reported about one emitted translation unit on one card, keyed on "
         "device.MeasuredKey(spec, arch, toolchain).key(). Observations of a "
         "toolchain, not claims about this code. NOT comparable to [JEngine, "
         "v2, Table 4]: their kernels, their compiler, their A100.")

HEADER = {
    "what": "Measured rows: what a device toolchain reported about one "
            "emitted translation unit (felizjueves.measured).",
    "key": "Rows are keyed on felizjueves.device.MeasuredKey(spec, arch, "
           "toolchain).key() -- the generation Spec, the card's Arch, the "
           "compiler as it reports itself (#37).",
    "nature": "Observations of a toolchain, not claims about this code: a "
              "compiler bump moving a number is news about the compiler.",
    "not_compared": "Nothing here is compared against [JEngine, v2, Table 4] "
                    "(their kernels, compiler and A100); see "
                    "data/literature/jengine/README.md.",
    "units": "nvcc: registers per thread, spill BYTES (ptxas -v). hipcc: "
             "VGPRs per lane, spilled REGISTER COUNTS (VGPR + SGPR; no "
             "store/load split, so both spill fields carry it). See each "
             "row's spill_unit and raw.",
    "remeasure": "PYTHONPATH=src uv run python -m felizjueves.measured "
                 "--remeasure --backend both",
    "wallclock": "runtime_ns is the MEDIAN host wall-clock of a cell's whole "
                 "batched launch sequence (every build_pq_kernels batch into "
                 "one zeroed J, then a synchronize), over `repeats` timed "
                 "sequences after a warm-up, on a wide-T input; spread_ns is "
                 "their interquartile range; raw 'wallclock:*' has the rest, "
                 "and 'wallclock:near:*' the same on a near-T input. Taken "
                 "alone on an "
                 "uncontended box by test/test_kernel_device.py's opt-in "
                 "wall-clock sweep (FJ_WALLCLOCK_STORE). Compare only within "
                 "one arch, across n_batches -- never across vendors.",
}

NV_SPILL_UNIT = "bytes (ptxas -v: spill stores / spill loads)"
AMD_SPILL_UNIT = ("registers (hipcc kernel-resource-usage: VGPRs Spill + "
                  "SGPRs Spill; no store/load split, both fields carry it)")


MEASUREMENTS = ("registers", "spill_store", "spill_load", "shared_bytes",
                "compile_s", "runtime_ns", "repeats", "spread_ns")
PROVENANCE = ("toolchain", "arch", "host", "gpu", "date")
_INTS = ("registers", "spill_store", "spill_load", "shared_bytes", "repeats")


@dataclass(frozen=True, kw_only=True)
class Measured:
    """One observation of one toolchain on one card. NOT a claim about this code.

    Every measurement is optional (None: not observed by this row), but at
    least one must be present. The provenance is required.
    """

    # compile-side, from the device compiler
    registers: int | None = None
    spill_store: int | None = None
    spill_load: int | None = None
    shared_bytes: int | None = None
    compile_s: float | None = None
    # device-side (C1)
    runtime_ns: float | None = None
    repeats: int | None = None
    spread_ns: float | None = None
    # provenance, all REQUIRED
    toolchain: str  # "nvcc 13.0", "hipcc / ROCm 7.2.4"
    arch: str  # "sm_86", "gfx1200"
    host: str
    gpu: str
    date: str
    # how the numbers were asked for
    kernel: str = ""
    spill_unit: str = ""
    command: str = ""  # the compiler invocation, measurement flag included
    source_sha256: str = ""  # of the emitted translation unit
    raw: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for n in MEASUREMENTS:
            v = getattr(self, n)
            if v is None:
                continue
            if n in _INTS:
                if type(v) is not int or v < 0:
                    raise ValueError(f"{n} = {v!r} must be an int >= 0 or None")
            elif type(v) not in (int, float) or v < 0:
                raise ValueError(f"{n} = {v!r} must be a number >= 0 or None")
        if self.repeats is not None and self.repeats < 1:
            raise ValueError(f"repeats = {self.repeats!r} must be >= 1")
        if all(getattr(self, n) is None for n in MEASUREMENTS):
            raise ValueError("every measurement is None: the row records nothing")
        for n in PROVENANCE:
            if not isinstance(getattr(self, n), str) or not getattr(self, n):
                raise ValueError(f"{n} must be a non-empty string")


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
                     for k in build_pq_kernels(spec))


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
    return MeasuredKey(spec, backend.arch, toolchain), row


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


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


_FIELDS = frozenset(f.name for f in fields(Measured))


def split_key(key: str) -> tuple[str, str, str]:
    """`MeasuredKey.key()` back into (spec key, arch name, toolchain)."""
    spec_key, sep, rest = key.partition("@")
    arch, sep2, toolchain = rest.partition("/")
    if not (sep and sep2 and spec_key and arch and toolchain):
        raise ValueError(f"{key!r} is not a MeasuredKey.key()")
    return spec_key, arch, toolchain


def _row(path: Path, key: str, v: object) -> Measured:
    if not isinstance(v, dict):
        raise ValueError(f"{path}: row {key!r} is not an object")
    unknown = sorted(set(v) - _FIELDS)
    if unknown:
        raise ValueError(f"{path}: row {key!r} has unknown fields {unknown}")
    try:
        row = Measured(**v)
        _, arch, toolchain = split_key(key)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: row {key!r}: {e}") from e
    if (arch, toolchain) != (row.arch, row.toolchain):
        raise ValueError(f"{path}: row {key!r} records arch/toolchain "
                         f"{row.arch!r}/{row.toolchain!r}, not its key's")
    return row


def load(path: Path = STORE) -> dict[str, Measured]:
    """Every row in the store, by key. Refuses a malformed or foreign file.

    A missing file raises FileNotFoundError: an empty store is a valid file
    with no rows, and is a different thing from no store at all.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON ({e})") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top level is not an object")
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA}")
    if not isinstance(doc.get("about"), str) or not doc["about"]:
        raise ValueError(f"{path}: no \"about\" string")
    got = doc.get("rows")
    if not isinstance(got, dict):
        raise ValueError(f"{path}: \"rows\" is not an object")
    return {k: _row(path, k, v) for k, v in got.items()}


def merge(store: Mapping[str, Measured], key: MeasuredKey,
          row: Measured) -> dict[str, Measured]:
    """`store` with `row` at `key`, REPLACING any row already there."""
    if (key.arch.name, key.toolchain) != (row.arch, row.toolchain):
        raise ValueError(f"row is {row.arch}/{row.toolchain}, key {key.key()!r}")
    return {**store, key.key(): row}


def combine(old: Measured | None, new: Measured) -> Measured:
    """`new`, with `old`'s other-side measurements kept when both saw one text.

    A field `new` leaves None is taken from `old`, and `old`'s `raw` entries
    survive under `new`'s, only when the two rows record the same arch,
    toolchain and non-empty `source_sha256`; otherwise `new` alone (module
    docstring, ONE ROW CARRIES BOTH SIDES).
    """
    if (old is None or not new.source_sha256
            or (old.arch, old.toolchain, old.source_sha256)
            != (new.arch, new.toolchain, new.source_sha256)):
        return new
    kept = {n: getattr(old, n) for n in MEASUREMENTS
            if getattr(new, n) is None and getattr(old, n) is not None}
    return replace(new, **kept, raw={**old.raw, **new.raw},
                   kernel=new.kernel or old.kernel,
                   spill_unit=new.spill_unit or old.spill_unit,
                   command=new.command or old.command)


def rows(spec: Spec | None = None, arch: Arch | str | None = None,
         toolchain: str | None = None, *,
         store: Mapping[str, Measured] | Path = STORE) -> dict[str, Measured]:
    """The read path: the rows matching every filter given, by key.

    `store` is an already-loaded mapping, or a path to load (by default the
    tracked store).
    """
    got = load(store) if isinstance(store, Path) else store
    arch_name = arch.name if isinstance(arch, Arch) else arch
    out = {}
    for k, r in got.items():
        s, a, t = split_key(k)
        if spec is not None and s != spec.key():
            continue
        if arch_name is not None and a != arch_name:
            continue
        if toolchain is not None and t != toolchain:
            continue
        out[k] = r
    return out


def save(store: Mapping[str, Measured], path: Path = STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": SCHEMA, "about": ABOUT, "header": HEADER,
           "rows": {k: asdict(store[k]) for k in sorted(store, key=_order)}}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _order(key: str) -> tuple:
    m = re.search(r"\((\d+),(\d+)\|(\d+),(\d+)\)", key)
    am = tuple(int(x) for x in m.groups()) if m else ()
    return (key.split("@", 1)[-1], sum(am), am, key)


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
# #60: do 9 batches suffice? (the definition is in the module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwelveVsNine:
    """What the 12-batch cell does against the 9-batch one, at one (arch, L)."""

    arch: str
    L: int
    spill: tuple[int, int]  # (spill(r9), spill(r12))
    wide: tuple[float, float, float]  # (median r9, median r12, max IQR)
    near: tuple[float, float, float]

    @staticmethod
    def _faster(t: tuple[float, float, float]) -> bool:
        return t[0] - t[1] > t[2]

    @property
    def fewer_spills(self) -> bool:
        return self.spill[1] < self.spill[0]

    @property
    def faster_wide(self) -> bool:
        return self._faster(self.wide)

    @property
    def faster_near(self) -> bool:
        return self._faster(self.near)

    @property
    def nine_suffices(self) -> bool:
        return not (self.fewer_spills or self.faster_wide or self.faster_near)


def nine_vs_twelve(store: Mapping[str, Measured], arch: str,
                   L: int) -> TwelveVsNine:
    """#60's comparison at one (arch, L), from the store's rows.

    Raises KeyError when either cell is unrecorded, uncompiled or untimed
    on `arch`.
    """
    def one(b: int) -> Measured:
        got = rows(spec=pq_spec(L, n_batches=b), arch=arch, store=store)
        if len(got) != 1:
            raise KeyError(f"L={L}, n_batches={b} @ {arch}: {len(got)} rows")
        (r,) = got.values()
        if r.runtime_ns is None or r.spread_ns is None or r.spill_store is None \
                or r.spill_load is None:
            raise KeyError(f"L={L}, n_batches={b} @ {arch}: not both compiled "
                           "and timed")
        return r

    def near(r: Measured) -> tuple[float, float]:
        return (r.raw["wallclock:near:median_ns"],
                r.raw["wallclock:near:q3_ns"] - r.raw["wallclock:near:q1_ns"])

    r9, r12 = one(9), one(12)
    n9, n12 = near(r9), near(r12)
    return TwelveVsNine(
        arch=arch, L=L,
        spill=(r9.spill_store + r9.spill_load, r12.spill_store + r12.spill_load),
        wide=(r9.runtime_ns, r12.runtime_ns, max(r9.spread_ns, r12.spread_ns)),
        near=(n9[0], n12[0], max(n9[1], n12[1])))


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
    ap.add_argument("-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--store", type=Path, default=STORE)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if args.remeasure:
        names = ("cuda", "hip") if args.backend == "both" else (args.backend,)
        try:
            cells = SUBGRID if args.subgrid else args.Ls
            if args.subgrid and args.n_batches:
                cells = [s for s in SUBGRID if s.n_batches in args.n_batches]
            remeasure(names, cells, args.j, args.store)
        except (RuntimeError, ValueError) as e:
            print(f"measured: {e}", file=sys.stderr)
            return 2
        return 0
    try:
        got = load(args.store)
    except (OSError, ValueError) as e:
        print(f"measured: {e}", file=sys.stderr)
        return 2
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
