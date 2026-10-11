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
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from felizjueves.device import GFX1200, SM_86, Arch, MeasuredKey
from felizjueves.emit import cuda, hip
from felizjueves.kernel import build_pq_kernel, pq_spec

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "data" / "measured" / "kernels.json"
DEFAULT_LS = tuple(range(0, 17))

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


@dataclass(frozen=True)
class Measured:
    """One toolchain's report on one kernel, with where and when it was asked."""

    registers: int
    spill_store: int
    spill_load: int
    toolchain: str  # "nvcc 13.0", "hipcc / ROCm 7.2.4"
    arch: str  # "sm_86", "gfx1200"
    host: str
    gpu: str
    date: str
    # provenance beyond the four numbers: how they were asked for
    kernel: str = ""
    spill_unit: str = ""
    command: str = ""  # the compiler invocation, measurement flag included
    source_sha256: str = ""  # of the emitted translation unit
    raw: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for n in ("registers", "spill_store", "spill_load"):
            v = getattr(self, n)
            if type(v) is not int or v < 0:
                raise ValueError(f"{n} = {v!r} must be an int >= 0")
        for n in ("toolchain", "arch", "host", "gpu", "date"):
            if not isinstance(getattr(self, n), str) or not getattr(self, n):
                raise ValueError(f"{n} must be a non-empty string")


# ---------------------------------------------------------------------------
# Parsing what each compiler prints
# ---------------------------------------------------------------------------

_NV_ENTRY = re.compile(r"Compiling entry function '([^']+)'")
_NV_SPILL = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_NV_REGS = re.compile(r"Used (\d+) registers")


def parse_ptxas(text: str, kernel: str) -> dict[str, int]:
    """`ptxas -v`'s numbers for the entry function `kernel`."""
    blocks = _NV_ENTRY.split(text)
    for name, body in zip(blocks[1::2], blocks[2::2], strict=True):
        if name != kernel:
            continue
        regs, spill = _NV_REGS.search(body), _NV_SPILL.search(body)
        if not regs or not spill:
            break
        return {"registers": int(regs[1]), "stack_frame_bytes": int(spill[1]),
                "spill_store_bytes": int(spill[2]),
                "spill_load_bytes": int(spill[3])}
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

    def emit(self, L: int) -> tuple[str, str]:
        k = build_pq_kernel(L)
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


def measure(backend: Backend, L: int, *, compiler: str, toolchain: str,
            gpu: str, workdir: Path) -> tuple[MeasuredKey, Measured]:
    """Compile `build_pq_kernel(L)` with the measurement flag; parse the report."""
    name, text = backend.emit(L)
    src = workdir / f"{name}.{backend.ext}"
    src.write_text(text)
    cmd = backend.command(compiler, src, workdir / f"{name}.o")
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{done.stderr[-4000:]}")
    report = done.stdout + done.stderr
    if backend.name == "cuda":
        raw = parse_ptxas(report, name)
        regs, store, load = (raw["registers"], raw["spill_store_bytes"],
                             raw["spill_load_bytes"])
        unit = NV_SPILL_UNIT
    else:
        raw = parse_amd_remarks(report, name)
        regs = raw["VGPRs"]
        store = load = raw["VGPRs Spill"] + raw.get("SGPRs Spill", 0)
        unit = AMD_SPILL_UNIT
    shown = [Path(c).name if i == 0 else c for i, c in enumerate(cmd)]
    shown = [f"<{Path(c).name}>" if c in (str(src), str(workdir / f"{name}.o"))
             else c for c in shown]
    row = Measured(
        registers=regs, spill_store=store, spill_load=load,
        toolchain=toolchain, arch=backend.arch.name,
        host=host(), gpu=gpu,
        date=datetime.date.today().isoformat(),
        kernel=name, spill_unit=unit, command=" ".join(shown),
        source_sha256=hashlib.sha256(text.encode()).hexdigest(), raw=raw)
    return MeasuredKey(pq_spec(L), backend.arch, toolchain), row


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def load(path: Path = STORE) -> dict[str, Measured]:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    return {k: Measured(**v) for k, v in doc["rows"].items()}


def save(rows: Mapping[str, Measured], path: Path = STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"header": HEADER,
           "rows": {k: asdict(rows[k]) for k in sorted(rows, key=_order)}}
    path.write_text(json.dumps(doc, indent=2) + "\n")


def _order(key: str) -> tuple:
    m = re.search(r"\((\d+),(\d+)\|(\d+),(\d+)\)", key)
    am = tuple(int(x) for x in m.groups()) if m else ()
    return (key.split("@", 1)[-1], sum(am), am, key)


def remeasure(names: Sequence[str], Ls: Sequence[int], jobs: int,
              path: Path = STORE) -> dict[str, Measured]:
    rows = load(path)
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
                    rows[key.key()] = row
                    print(f"{key.key()}: {row.registers} regs, spill "
                          f"{row.spill_store}/{row.spill_load}")
    save(rows, path)
    return rows


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
    for k, r in load(args.store).items():
        print(f"{k}: {r.registers} regs, spill {r.spill_store}/{r.spill_load} "
              f"[{r.spill_unit.split(' ')[0]}] on {r.gpu}, {r.date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
