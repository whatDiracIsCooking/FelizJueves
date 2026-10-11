"""What a device toolchain reports about a generated kernel: `Measured` rows.

    python -m felizjueves.measured                      # print the store
    python -m felizjueves.measured --remeasure [--backend cuda|hip|both]
                                   [--L 0 4 8] [-j N]   # recompile, rewrite rows

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
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from felizjueves.device import GFX1200, SM_86, Arch, MeasuredKey
from felizjueves.emit import cuda, hip
from felizjueves.kernel import build_pq_kernel, pq_spec
from felizjueves.spec import Spec

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "data" / "measured" / "kernels.json"
DEFAULT_LS = tuple(range(0, 17))

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

    def emit(self, cls: int | Spec) -> tuple[str, str]:
        k = build_pq_kernel(cls)
        emit = cuda.emit if self.name == "cuda" else hip.emit
        return k.name, emit(k, self.arch)


BACKENDS = {
    "cuda": Backend("cuda", SM_86, "nvcc", "NVCC_FLAGS", "cu"),
    "hip": Backend("hip", GFX1200, "hipcc", "HIPCC_FLAGS", "hip"),
}


def host() -> str:
    """The hostname, marked when it is a container's (its id, not the box's)."""
    name = socket.gethostname()
    return f"{name} (container)" if Path("/.dockerenv").exists() else name


def measure(backend: Backend, cls: int | Spec, *, compiler: str,
            toolchain: str, gpu: str,
            workdir: Path) -> tuple[MeasuredKey, Measured]:
    """Compile one class's kernel with the measurement flag; parse the report.

    `cls` is a total L (`pq_spec(L)`, one batch) or a full `Spec` -- the
    latter is the `n_batches` axis (#52). The builder decides which Specs it
    can build; today it refuses `n_batches != 1`, and that refusal is raised
    here before anything is compiled.
    """
    spec = pq_spec(cls) if isinstance(cls, int) else cls
    name, text = backend.emit(spec)
    src = workdir / f"{name}.{backend.ext}"
    src.write_text(text)
    cmd = backend.command(compiler, src, workdir / f"{name}.o")
    t0 = time.perf_counter()
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    compile_s = round(time.perf_counter() - t0, 3)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{done.stderr[-4000:]}")
    report = done.stdout + done.stderr
    if backend.name == "cuda":
        raw = parse_ptxas(report, name)
        regs, store, load = (raw["registers"], raw["spill_store_bytes"],
                             raw["spill_load_bytes"])
        shared = raw.get("smem_bytes")  # None when ptxas printed no smem
        unit = NV_SPILL_UNIT
    else:
        raw = parse_amd_remarks(report, name)
        regs = raw["VGPRs"]
        store = load = raw["VGPRs Spill"] + raw.get("SGPRs Spill", 0)
        shared = raw.get("LDS Size")
        unit = AMD_SPILL_UNIT
    shown = [Path(c).name if i == 0 else c for i, c in enumerate(cmd)]
    shown = [f"<{Path(c).name}>" if c in (str(src), str(workdir / f"{name}.o"))
             else c for c in shown]
    row = Measured(
        registers=regs, spill_store=store, spill_load=load,
        shared_bytes=shared, compile_s=compile_s,
        toolchain=toolchain, arch=backend.arch.name,
        host=host(), gpu=gpu,
        date=datetime.date.today().isoformat(),
        kernel=name, spill_unit=unit, command=" ".join(shown),
        source_sha256=hashlib.sha256(text.encode()).hexdigest(), raw=raw)
    return MeasuredKey(spec, backend.arch, toolchain), row


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


def remeasure(names: Sequence[str], Ls: Sequence[int], jobs: int,
              path: Path = STORE) -> dict[str, Measured]:
    got = load(path) if path.exists() else {}
    for bname in names:
        b = BACKENDS[bname]
        compiler = b.compiler()
        toolchain, gpu = b.toolchain(compiler), b.gpu()
        with tempfile.TemporaryDirectory(prefix=f"fj-measure-{bname}-") as tmp:
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                futures = [pool.submit(measure, b, L, compiler=compiler,
                                       toolchain=toolchain, gpu=gpu,
                                       workdir=Path(tmp)) for L in Ls]
                for f in futures:
                    key, row = f.result()
                    got = merge(got, key, row)
                    print(f"{key.key()}: {row.registers} regs, spill "
                          f"{row.spill_store}/{row.spill_load}")
    save(got, path)
    return got


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m felizjueves.measured")
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--backend", choices=("cuda", "hip", "both"), default="both")
    ap.add_argument("--L", dest="Ls", type=int, nargs="+", default=list(DEFAULT_LS))
    ap.add_argument("-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--store", type=Path, default=STORE)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if args.remeasure:
        names = ("cuda", "hip") if args.backend == "both" else (args.backend,)
        try:
            remeasure(names, args.Ls, args.j, args.store)
        except (RuntimeError, ValueError) as e:
            print(f"measured: {e}", file=sys.stderr)
            return 2
        return 0
    try:
        got = load(args.store)
    except (OSError, ValueError) as e:
        print(f"measured: {e}", file=sys.stderr)
        return 2
    for k, r in got.items():
        shown = ", ".join(f"{n}={getattr(r, n)}" for n in MEASUREMENTS
                          if getattr(r, n) is not None)
        print(f"{k}: {shown} on {r.gpu}, {r.date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
