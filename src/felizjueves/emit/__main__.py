"""`python -m felizjueves.emit`: write the generated kernels to a directory.

THE GENERATOR'S ENTRY POINT, and the command `devtools/config.sh` names as
`KERNEL_GEN_CMD`. The project is a virtual uv project -- nothing is installed
-- so `src/` goes on the path by hand:

    PYTHONPATH=src uv run python -m felizjueves.emit --backend cuda --out DIR
    PYTHONPATH=src uv run python -m felizjueves.emit --backend hip --L 4 8

One translation unit per class: `build_pq_kernel(L)` -- the default class of
total L, `kernel.pq_spec(L)`, one batch -- emitted by the backend for the
arch, to `<out>/<kernel name>.<ext>` (`.cu` for CUDA, `.hip` for HIP). The two
vendors' texts differ only in the backend hooks (test/test_emit_hip.py), and
HIP's shuffle needs a 64-bit mask that the CUDA text does not carry, so each
backend writes its own spelling; nothing here compiles anything.

DEFAULTS FROM THE KERNEL TIER. `devtools/kernel-tier.sh` exports, for the
command it runs, `KERNEL_TIER_BACKEND` (cuda | hip), `KERNEL_TIER_ARCH` (the
arch the compiler is driven at, a `device.Arch` name) and `KERNEL_TIER_OUT`
(where it will look for sources). Each flag defaults to its variable, so the
config line needs no arguments; a flag given on the command line wins. An arch
this project does not declare (`felizjueves.device.ARCHS`) is an error, not a
pass-through.

THE CLASSES are L = 0..16 by default: every class the plan layer schedules at
n_batches = 1 under the default split ([JEngine] Tables 1-3 run L <= 16).

Exit status: 0 written, 2 could not run (bad arguments) -- the tier's rule.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from felizjueves.device import Arch, Vendor, by_name, from_env
from felizjueves.emit import cuda, hip
from felizjueves.kernel import build_pq_kernel

BACKENDS = {
    "cuda": (Vendor.NVIDIA, cuda.emit, "cu"),
    "hip": (Vendor.AMD, hip.emit, "hip"),
}
DEFAULT_LS = tuple(range(0, 17))


def source_name(L: int, backend: str) -> str:
    """The file `build_pq_kernel(L)` is written to for `backend`."""
    return f"{build_pq_kernel(L).name}.{BACKENDS[backend][2]}"


def resolve_arch(backend: str, arch: str | None) -> Arch:
    vendor = BACKENDS[backend][0]
    a = by_name(arch) if arch else from_env(vendor)
    if a.vendor is not vendor:
        raise ValueError(f"arch {a.name} is {a.vendor}, not the {backend} backend's")
    return a


def generate(backend: str, arch: Arch, out: Path, Ls: Sequence[int]) -> list[Path]:
    """Emit one translation unit per class into `out`; return the paths."""
    _, emit, ext = BACKENDS[backend]
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for L in Ls:
        k = build_pq_kernel(L)
        path = out / f"{k.name}.{ext}"
        path.write_text(emit(k, arch))
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    env = os.environ
    ap = argparse.ArgumentParser(prog="python -m felizjueves.emit",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=sorted(BACKENDS),
                    default=env.get("KERNEL_TIER_BACKEND") or None)
    ap.add_argument("--arch", default=env.get("KERNEL_TIER_ARCH") or None)
    ap.add_argument("--out", type=Path, default=env.get("KERNEL_TIER_OUT") or None)
    ap.add_argument("--L", dest="Ls", type=int, nargs="+", default=list(DEFAULT_LS))
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if args.backend is None or args.out is None:
        print("emit: --backend and --out are required (or KERNEL_TIER_BACKEND / "
              "KERNEL_TIER_OUT)", file=sys.stderr)
        return 2
    try:
        arch = resolve_arch(args.backend, args.arch)
    except ValueError as e:
        print(f"emit: {e}", file=sys.stderr)
        return 2
    written = generate(args.backend, arch, args.out, args.Ls)
    print(f"emit: {len(written)} {args.backend} translation unit(s) for "
          f"{arch.name} -> {args.out}")
    for p in written:
        print(f"   {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
