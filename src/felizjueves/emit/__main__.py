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
That default is KERNEL_GEN_CMD's contract, and the selector below does not
touch it: `data/measured/kernels.json` pins each row's `source_sha256` to the
text this path writes.

THE CLASS SELECTOR (#89): a SEPARATE path, for the classes a real basis
reaches rather than the balanced ladder `pq_spec(L)` produces.

    ... -m felizjueves.emit --backend cuda --out DIR --class 1 0 1 1 [--class ...]
    ... -m felizjueves.emit --backend hip --out DIR --shell-ls 0 0 1 0 0 \\
            --n-batches 1 3 --admitted-only

`--class LA LB LC LD` names a class `(la, lb | lc, ld)` by its REAL shell
angular momenta -- the `Spec` `ao.drive.class_spec` builds, so `lp = la + lb`,
`lq = lc + ld`, and `(p,p|s,s)` is not `(s,s|p,p)`. `--shell-ls` derives the
set from a basis's shell angular momenta, in basis order, as `ao.j`'s unique
sweep reaches it (`classes_reached`). For every class and every `--n-batches`
(default 1) the cell's kernels -- `build_pq_kernels`, one per batch, atomized
when split -- are written to `<out>/<cell_dir>/`, with an `order` file naming
them in launch (`Plan.dags`) order: they run, in that order, into ONE zeroed
`J`. A batch count the plan refuses is an error unless `--admitted-only`,
which skips it and says so; a class with NO admitted count is still an error.
`--L` and the selector are separate paths and are not combined.

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
from felizjueves.kernel import build_pq_kernel, build_pq_kernels
from felizjueves.spec import Operator, Spec, Target

BACKENDS = {
    "cuda": (Vendor.NVIDIA, cuda.emit, "cu"),
    "hip": (Vendor.AMD, hip.emit, "hip"),
}
DEFAULT_LS = tuple(range(0, 17))
Am = tuple[int, int, int, int]


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


# ---------------------------------------------------------------------------
# The class selector (#89)
# ---------------------------------------------------------------------------


def class_spec(am: Am, n_batches: int = 1) -> Spec:
    """The real class `(la, lb | lc, ld)` as a generation key -- the key
    `ao.drive.class_spec` builds (test_ao_j_ondevice.py pins them equal)."""
    return Spec(Operator.COULOMB, 0, tuple(am), Target.J, (), n_batches)


def classes_reached(shell_ls: Sequence[int]) -> list[Am]:
    """Every `(la, lb | lc, ld)` `ao.j`'s unique sweep launches for a basis
    whose shells, in basis order, have angular momenta `shell_ls`: bra and
    ket shell pairs are `(a, b)` with `b <= a`, taken as `(l_a, l_b)`."""
    ams = sorted({(shell_ls[a], shell_ls[b]) for a in range(len(shell_ls))
                  for b in range(a + 1)})
    return [(*bra, *ket) for bra in ams for ket in ams]


def cell_dir(am: Am, n_batches: int) -> str:
    """The subdirectory one `(class, n_batches)` cell is written to."""
    return "c" + "_".join(map(str, am)) + f"_b{n_batches}"


def generate_classes(backend: str, arch: Arch, out: Path, classes: Sequence[Am],
                     batch_counts: Sequence[int], *, admitted_only: bool = False
                     ) -> tuple[dict[tuple[Am, int], list[Path]],
                                list[tuple[Am, int]]]:
    """The selector's writer: `{(am, n): [paths, in launch order]}`, and the
    `(am, n)` cells the plan refused (skipped only with `admitted_only`;
    otherwise a refusal raises `ValueError`)."""
    _, emit, ext = BACKENDS[backend]
    written: dict[tuple[Am, int], list[Path]] = {}
    refused: list[tuple[Am, int]] = []
    for am in classes:
        any_cell = False
        for n in batch_counts:
            try:
                kernels = build_pq_kernels(class_spec(am, n))
            except ValueError as e:
                if not admitted_only:
                    raise ValueError(f"class {am} at n_batches = {n}: {e}") from e
                refused.append((am, n))
                continue
            d = out / cell_dir(am, n)
            d.mkdir(parents=True, exist_ok=True)
            paths = []
            for k in kernels:
                path = d / f"{k.name}.{ext}"
                path.write_text(emit(k, arch))
                paths.append(path)
            (d / "order").write_text("".join(f"{p.name}\n" for p in paths))
            written[(am, n)] = paths
            any_cell = True
        if not any_cell:
            raise ValueError(f"class {am}: the plan admits none of n_batches "
                             f"{tuple(batch_counts)}")
    return written, refused


def _main_classes(args: argparse.Namespace, arch: Arch) -> int:
    classes: list[Am] = [tuple(c) for c in args.classes or []]
    if args.shell_ls:
        classes += classes_reached(args.shell_ls)
    classes = list(dict.fromkeys(classes))
    try:
        written, refused = generate_classes(
            args.backend, arch, args.out, classes, args.n_batches,
            admitted_only=args.admitted_only)
    except ValueError as e:
        print(f"emit: {e}", file=sys.stderr)
        return 2
    units = sum(len(v) for v in written.values())
    print(f"emit: {len(written)} (class, n_batches) cell(s), {units} "
          f"{args.backend} translation unit(s) for {arch.name} -> {args.out}")
    for (am, n), paths in written.items():
        print(f"   {cell_dir(am, n)}: " + " ".join(p.name for p in paths))
    for am, n in refused:
        print(f"   {cell_dir(am, n)}: refused by the plan, not written")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    env = os.environ
    ap = argparse.ArgumentParser(prog="python -m felizjueves.emit",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=sorted(BACKENDS),
                    default=env.get("KERNEL_TIER_BACKEND") or None)
    ap.add_argument("--arch", default=env.get("KERNEL_TIER_ARCH") or None)
    ap.add_argument("--out", type=Path, default=env.get("KERNEL_TIER_OUT") or None)
    ap.add_argument("--L", dest="Ls", type=int, nargs="+", default=None)
    ap.add_argument("--class", dest="classes", type=int, nargs=4, action="append",
                    metavar=("LA", "LB", "LC", "LD"))
    ap.add_argument("--shell-ls", type=int, nargs="+", default=None)
    ap.add_argument("--n-batches", type=int, nargs="+", default=[1])
    ap.add_argument("--admitted-only", action="store_true")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if args.backend is None or args.out is None:
        print("emit: --backend and --out are required (or KERNEL_TIER_BACKEND / "
              "KERNEL_TIER_OUT)", file=sys.stderr)
        return 2
    selector = bool(args.classes or args.shell_ls)
    if selector and args.Ls is not None:
        print("emit: --L and --class / --shell-ls are separate paths; give one",
              file=sys.stderr)
        return 2
    if not selector and (args.n_batches != [1] or args.admitted_only):
        print("emit: --n-batches / --admitted-only need --class or --shell-ls",
              file=sys.stderr)
        return 2
    try:
        arch = resolve_arch(args.backend, args.arch)
    except ValueError as e:
        print(f"emit: {e}", file=sys.stderr)
        return 2
    if selector:
        return _main_classes(args, arch)
    Ls = DEFAULT_LS if args.Ls is None else args.Ls
    written = generate(args.backend, arch, args.out, Ls)
    print(f"emit: {len(written)} {args.backend} translation unit(s) for "
          f"{arch.name} -> {args.out}")
    for p in written:
        print(f"   {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
