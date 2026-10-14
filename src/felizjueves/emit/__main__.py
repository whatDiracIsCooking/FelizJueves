"""`python -m felizjueves.emit`: write the generated kernels to a directory.

THE GENERATOR'S ENTRY POINT, and the command `devtools/config.sh` names as
`KERNEL_GEN_CMD`. The project is a virtual uv project -- nothing is installed
-- so `src/` goes on the path by hand:

    PYTHONPATH=src uv run python -m felizjueves.emit --backend cuda --out DIR
    PYTHONPATH=src uv run python -m felizjueves.emit --backend hip --L 4 8
    PYTHONPATH=src uv run python -m felizjueves.emit --backend hip --pair-l 2
    PYTHONPATH=src uv run python -m felizjueves.emit --backend cuda --one-e-l 2
    PYTHONPATH=src uv run python -m felizjueves.emit --backend hip --fock-l 1

ONE LADDER PER KERNEL FAMILY (#156), one translation unit per class, each
emitted by the backend for the arch to `<out>/<kernel name>.<ext>` (`.cu`
for CUDA, `.hip` for HIP):

- the `[p|q]` ladder, `--L`: `build_pq_kernel(L)` -- the default class of
  total L, `kernel.pq_spec(L)`, one batch -- named `pq_L{L}_p{lp}_q{lq}`;
- the pair ladder, `--pair-l LMAX`: for every pair class `(la, lb)`,
  `la <= lb <= LMAX`, the pre-contraction kernel `kernel_dq.build_dq_kernel`
  (`dq_{la}_{lb}`, `Target.DQ`, #153) and the bra-transform-and-scatter
  kernel `kernel_jab.build_jab_kernel` (`jab_{la}{lb}`, `Target.JAB`, #154).
  Small units: an inlined `E` block, no Boys, no `[p|q]`.

- the one-electron ladder, `--one-e-l [LMAX]` (#242): for every pair class
  `(la, lb)`, `la <= lb <= LMAX` (3 when the flag is given bare), the
  overlap, kinetic and nuclear-attraction kernels `kernel_s.build_s_kernel`
  (`s_{la}{lb}`, `Target.S`), `kernel_t.build_t_kernel` (`t_{la}{lb}`,
  `Target.T`) and `kernel_v.build_v_kernel` (`v_{la}{lb}`, `Target.V`, the
  one that carries the wide `fj_boys`). NEVER in the no-flag default.

With NEITHER `--L` nor `--pair-l` both J ladders are written -- `[p|q]`
L = 0..16 and the pair classes (ss) through (ff), 17 + 20 units: the J chain
is what the bare command writes, and `data/measured/kernels.json` pins those
37 units' `source_sha256`. `--L` alone writes only `[p|q]` (so `--L 1`
writes exactly `pq_L1_p1_q0.<ext>`), `--pair-l` alone only the pair ladder,
the two together both. `--one-e-l` ADDS the one-electron ladder to whichever
J ladders those two select (so `--one-e-l 2` alone writes the 37 J units
plus 18): it is a second family beside the J chain, not a replacement for
it, and the kernel tier asks for it explicitly (devtools/config.sh's
KERNEL_GEN_CMD). Its units are named `s_`, `t_`, `v_` plus two digits, so
none collides with `pq_`, `dq_` or `jab_`.

WHY ONLY `la <= lb` ON THE ONE-ELECTRON LADDER. It is the pair ladder's set
of classes, as #242 pins it. A real basis's one-electron launch
(`launch.one_electron_launch`) keys a class by the shell pair `i <= j` in
BASIS order, so a basis whose p or d shells precede its s shells (H2O: O's
p before H's s) also reaches `(1, 0)`, `(2, 0)`, `(2, 1)`; a driver builds
those from `build_*_kernel((la, lb))` itself (test/one_electron_driver.py),
the same text this module would write for them.

THE FOCK PATH, `--fock-class LA LB LC LD` / `--fock-l LMAX` (#287): the
fused Fock kernel of `[Fock]` §3.2, `kernel_f.build_f_kernel` (`Target.F`,
engine #2, milestone 12), one unit per class, `f_{la}{lb}{lc}{ld}`.
`--fock-l LMAX` (0..`FOCK_L_MAX` = 2) writes every class of milestone 12's
ladder (`FOCK_CLASSES`, the 21 canonical classes, each shell `l <= 2`)
whose shells are all `l <= LMAX` -- 1, 6 or 21 units, in ladder order;
`--fock-class` (repeatable) names classes one by one, and each must be ON
the ladder: AO order, canonical `la <= lb`, `lc <= ld`, bra pair `<=` ket
pair (the only classes `launch_f` launches), so `(10|00)` is a 2 and not a
silent re-ordering. The two flags together write the union, in ladder
order. It is a SEPARATE path, like the selector: never combined with `--L`,
`--pair-l`, `--one-e-l`, `--class` or `--shell-ls`, and NEVER in the no-flag
default -- `data/measured/kernels.json` pins the 37 J-chain units'
`source_sha256` (m11's lesson), and KERNEL_GEN_CMD does not ask for it.
Every F unit is named `f_` plus four digits, so none collides with `pq_`,
`dq_`, `jab_`, `s_`, `t_` or `v_`.

THE LADDER IS SPELLED HERE (`FOCK_CLASSES`), not imported from `hgp.vrr`,
and test/test_emit_fock.py holds the two equal. This driver names each
engine by its ENTRY POINT only (`kernel` for MMD, `kernel_f` for HGP; #292's
rule, test/test_layering.py): `hgp.vrr` is the HGP engine's private
machinery, behind that entry point.

THE SHARED STAGING FITS, so the kernel is emitted as `kernel_f` builds it.
Its one shared array, `ab_sh`, is ONE per block of literal extent
`MAX_PRIM_PAIRS x REC` = 128 x 8 doubles = 8,192 bytes, independent of
`blockDim` (every thread writes the same values into the same array; it is
not a copy per thread). The compilers report exactly that for all 21
classes: ptxas `8192 bytes smem` on the NVIDIA arch (48 KiB static per
block) and `LDS Size [bytes/block]: 8192` on the AMD one (64 KiB per
workgroup), pinned by test/test_emit_fock.py's compile cases.

The two vendors' texts differ only in the backend hooks
(test/test_emit_hip.py), and HIP's shuffle needs a 64-bit mask that the CUDA
text does not carry, so each backend writes its own spelling; nothing here
compiles anything.

DEFAULTS FROM THE KERNEL TIER. `devtools/kernel-tier.sh` exports, for the
command it runs, `KERNEL_TIER_BACKEND` (cuda | hip), `KERNEL_TIER_ARCH` (the
arch the compiler is driven at, a `device.Arch` name) and `KERNEL_TIER_OUT`
(where it will look for sources). Each flag defaults to its variable, so the
config line needs no arguments; a flag given on the command line wins. An arch
this project does not declare (`felizjueves.device.ARCHS`) is an error, not a
pass-through.

THE `[p|q]` CLASSES are L = 0..16 by default: every class the plan layer
schedules at n_batches = 1 under the default split ([JEngine] Tables 1-3 run
L <= 16). That default is KERNEL_GEN_CMD's contract, and neither the pair
ladder nor the selector touches it: `data/measured/kernels.json` pins each
row's `source_sha256` to the text this path writes (test/test_emit_cli.py
asserts it).

THE PAIR CLASSES are through f by default (`DEFAULT_PAIR_L = 3`): the
tracked bases reach f, and nothing checks above it. `--pair-l` accepts
0..`PAIR_L_MAX` = 8 -- the pair class whose (bb) side has L = 16, the
`[p|q]` ladder's top; outside that range it is a 2. Every pair kernel name
starts `dq_` or `jab_`, so no unit collides with the `[p|q]` naming that
test_kernel_device.py and the measured driver find their units by.

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
(default 1) the cell's kernels are written to `<out>/<cell_dir>/` with an
`order` file naming them in the CHAIN's launch order (#156):

    dq_{lc}_{ld}       the ket pair class's pre-contraction (D_q, Dmax)
    <batch kernels>    `build_pq_kernels`, one per batch, atomized when
                       split, in `Plan.dags` order, into ONE zeroed J_p
    jab_{la}{lb}       the bra pair class's transform and scatter into J

Each cell is self-contained: every name in its `order` is a file in the
cell's own directory, so a pair kernel two cells share is written in both.
The order is ONE class's dependency order; a whole-molecule driver launches
each `DQ` once per ket pair class, every class's `[p|q]` batches, then each
`JAB` once per bra pair class (j_engine.md, "The J chain as device code").
A batch count the plan refuses is an error unless `--admitted-only`, which
skips it and says so; a class with NO admitted count is still an error.
The ladders (`--L`, `--pair-l`) and the selector are separate paths and are
not combined.

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
from felizjueves.kernel_dq import build_dq_kernel
from felizjueves.kernel_f import build_f_kernel
from felizjueves.kernel_jab import build_jab_kernel
from felizjueves.kernel_s import build_s_kernel
from felizjueves.kernel_t import build_t_kernel
from felizjueves.kernel_v import build_v_kernel
from felizjueves.spec import Operator, Spec, Target

BACKENDS = {
    "cuda": (Vendor.NVIDIA, cuda.emit, "cu"),
    "hip": (Vendor.AMD, hip.emit, "hip"),
}
DEFAULT_LS = tuple(range(0, 17))
DEFAULT_PAIR_L = 3  # (ss) through (ff): 10 pair classes, 20 units
PAIR_L_MAX = 8  # (bb) at L = 16, the [p|q] ladder's top
DEFAULT_ONE_E_L = 3  # `--one-e-l` given bare: (ss) through (ff), 30 units
ONE_E_L_MAX = PAIR_L_MAX  # the range the pair ladder accepts
FOCK_L_MAX = 2  # milestone 12's ladder tops out at (dd|dd) (pin 3)
_FOCK_PAIRS = tuple((a, b) for a in range(FOCK_L_MAX + 1)
                    for b in range(a, FOCK_L_MAX + 1))
# the 21 classes: unordered pairs of canonical pairs, bra <= ket (module
# docstring; == hgp.vrr.CLASSES, held by test/test_emit_fock.py)
FOCK_CLASSES: tuple[Am, ...] = tuple(
    bra + ket for i, bra in enumerate(_FOCK_PAIRS) for ket in _FOCK_PAIRS[i:])
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


def _write(backend: str, arch: Arch, d: Path, k) -> Path:
    _, emit, ext = BACKENDS[backend]
    path = d / f"{k.name}.{ext}"
    path.write_text(emit(k, arch))
    return path


def generate(backend: str, arch: Arch, out: Path, Ls: Sequence[int]) -> list[Path]:
    """Emit one `[p|q]` translation unit per class into `out`; return the paths."""
    out.mkdir(parents=True, exist_ok=True)
    return [_write(backend, arch, out, build_pq_kernel(L)) for L in Ls]


# ---------------------------------------------------------------------------
# The pair ladder (#156)
# ---------------------------------------------------------------------------


def pair_classes(lmax: int) -> list[tuple[int, int]]:
    """Every pair class `(la, lb)`, `la <= lb <= lmax`, `la` outer."""
    if type(lmax) is not int or not 0 <= lmax <= PAIR_L_MAX:
        raise ValueError(f"--pair-l {lmax!r} is outside 0..{PAIR_L_MAX}")
    return [(la, lb) for la in range(lmax + 1) for lb in range(la, lmax + 1)]


def generate_pairs(backend: str, arch: Arch, out: Path, lmax: int) -> list[Path]:
    """Emit the `DQ` and the `JAB` unit of every pair class through `lmax`
    into `out`; return the paths."""
    classes = pair_classes(lmax)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for c in classes:
        written.append(_write(backend, arch, out, build_dq_kernel(c)))
        written.append(_write(backend, arch, out, build_jab_kernel(c)))
    return written


# ---------------------------------------------------------------------------
# The one-electron ladder (#242)
# ---------------------------------------------------------------------------

ONE_E_BUILDERS = (build_s_kernel, build_t_kernel, build_v_kernel)


def one_e_classes(lmax: int) -> list[tuple[int, int]]:
    """Every pair class `(la, lb)`, `la <= lb <= lmax`: `pair_classes`'
    set, over `0..ONE_E_L_MAX`."""
    if type(lmax) is not int or not 0 <= lmax <= ONE_E_L_MAX:
        raise ValueError(f"--one-e-l {lmax!r} is outside 0..{ONE_E_L_MAX}")
    return pair_classes(lmax)


def one_e_kernels(lmax: int) -> list:
    """`s_{la}{lb}`, `t_{la}{lb}`, `v_{la}{lb}` for every class through
    `lmax`, class-major."""
    return [build(c) for c in one_e_classes(lmax) for build in ONE_E_BUILDERS]


def generate_one_e(backend: str, arch: Arch, out: Path, lmax: int) -> list[Path]:
    """Emit the `S`, `T` and `V` unit of every pair class through `lmax`
    into `out`; return the paths."""
    kernels = one_e_kernels(lmax)
    out.mkdir(parents=True, exist_ok=True)
    return [_write(backend, arch, out, k) for k in kernels]


# ---------------------------------------------------------------------------
# The Fock path (#287)
# ---------------------------------------------------------------------------


def fock_classes(lmax: int) -> list[Am]:
    """Every ladder class (`FOCK_CLASSES`) whose shells are all `l <=
    lmax`, in ladder order."""
    if type(lmax) is not int or not 0 <= lmax <= FOCK_L_MAX:
        raise ValueError(f"--fock-l {lmax!r} is outside 0..{FOCK_L_MAX}")
    return [c for c in FOCK_CLASSES if max(c) <= lmax]


def fock_selection(lmax: int | None, classes: Sequence[Sequence[int]] = ()
                   ) -> list[Am]:
    """The union of `--fock-l lmax` and the `--fock-class` list, in ladder
    order; a named class off the ladder raises `ValueError`."""
    chosen = set(fock_classes(lmax)) if lmax is not None else set()
    for c in classes:
        c = tuple(c)
        if c not in FOCK_CLASSES:
            raise ValueError(
                f"--fock-class {' '.join(map(str, c))} is not on milestone "
                "12's ladder: AO order, each shell l <= 2, canonical la <= lb, "
                "lc <= ld and bra pair <= ket pair")
        chosen.add(c)
    return [c for c in FOCK_CLASSES if c in chosen]


def generate_fock(backend: str, arch: Arch, out: Path,
                  classes: Sequence[Am]) -> list[Path]:
    """Emit the fused Fock unit `f_{la}{lb}{lc}{ld}` of every class into
    `out`; return the paths."""
    out.mkdir(parents=True, exist_ok=True)
    return [_write(backend, arch, out, build_f_kernel(tuple(c)))
            for c in classes]


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


def chain_kernels(am: Am, n_batches: int) -> list:
    """One class's chain in launch order: the ket pair class's `DQ`, the
    class's `[p|q]` batch kernels, the bra pair class's `JAB`. Raises
    `ValueError` when the plan refuses `n_batches`."""
    la, lb, lc, ld = am
    return [build_dq_kernel((lc, ld)), *build_pq_kernels(class_spec(am, n_batches)),
            build_jab_kernel((la, lb))]


def generate_classes(backend: str, arch: Arch, out: Path, classes: Sequence[Am],
                     batch_counts: Sequence[int], *, admitted_only: bool = False
                     ) -> tuple[dict[tuple[Am, int], list[Path]],
                                list[tuple[Am, int]]]:
    """The selector's writer: `{(am, n): [paths, in chain launch order]}`, and
    the `(am, n)` cells the plan refused (skipped only with `admitted_only`;
    otherwise a refusal raises `ValueError`)."""
    written: dict[tuple[Am, int], list[Path]] = {}
    refused: list[tuple[Am, int]] = []
    for am in classes:
        any_cell = False
        for n in batch_counts:
            try:
                kernels = chain_kernels(am, n)
            except ValueError as e:
                if not admitted_only:
                    raise ValueError(f"class {am} at n_batches = {n}: {e}") from e
                refused.append((am, n))
                continue
            d = out / cell_dir(am, n)
            d.mkdir(parents=True, exist_ok=True)
            paths = [_write(backend, arch, d, k) for k in kernels]
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
    ap.add_argument("--pair-l", dest="pair_l", type=int, default=None,
                    metavar="LMAX")
    ap.add_argument("--one-e-l", dest="one_e_l", type=int, nargs="?",
                    const=DEFAULT_ONE_E_L, default=None, metavar="LMAX")
    ap.add_argument("--fock-l", dest="fock_l", type=int, default=None,
                    metavar="LMAX")
    ap.add_argument("--fock-class", dest="fock_classes", type=int, nargs=4,
                    action="append", metavar=("LA", "LB", "LC", "LD"))
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
    fock = args.fock_l is not None or bool(args.fock_classes)
    if fock and (selector or args.Ls is not None or args.pair_l is not None
                 or args.one_e_l is not None):
        print("emit: --fock-l / --fock-class is a separate path from --L / "
              "--pair-l / --one-e-l / --class / --shell-ls; give one",
              file=sys.stderr)
        return 2
    if fock and (args.n_batches != [1] or args.admitted_only):
        print("emit: an F class is one batch (spec.py, #272); --n-batches / "
              "--admitted-only belong to --class or --shell-ls",
              file=sys.stderr)
        return 2
    fock_cls: list[Am] = []
    if fock:
        try:
            fock_cls = fock_selection(args.fock_l, args.fock_classes or ())
        except ValueError as e:
            print(f"emit: {e}", file=sys.stderr)
            return 2
    if selector and (args.Ls is not None or args.pair_l is not None
                     or args.one_e_l is not None):
        print("emit: --L / --pair-l / --one-e-l and --class / --shell-ls are "
              "separate paths; give one", file=sys.stderr)
        return 2
    if not selector and (args.n_batches != [1] or args.admitted_only):
        print("emit: --n-batches / --admitted-only need --class or --shell-ls",
              file=sys.stderr)
        return 2
    if args.pair_l is not None and not 0 <= args.pair_l <= PAIR_L_MAX:
        print(f"emit: --pair-l {args.pair_l} is outside 0..{PAIR_L_MAX}",
              file=sys.stderr)
        return 2
    if args.one_e_l is not None and not 0 <= args.one_e_l <= ONE_E_L_MAX:
        print(f"emit: --one-e-l {args.one_e_l} is outside 0..{ONE_E_L_MAX}",
              file=sys.stderr)
        return 2
    try:
        arch = resolve_arch(args.backend, args.arch)
    except ValueError as e:
        print(f"emit: {e}", file=sys.stderr)
        return 2
    if selector:
        return _main_classes(args, arch)
    if fock:
        written = generate_fock(args.backend, arch, args.out, fock_cls)
        print(f"emit: {len(written)} {args.backend} Fock translation unit(s) "
              f"for {arch.name} -> {args.out}")
        for p in written:
            print(f"   {p.name}")
        return 0
    if args.Ls is None and args.pair_l is None:  # the default: both J ladders
        Ls, pair_l = DEFAULT_LS, DEFAULT_PAIR_L
    else:
        Ls, pair_l = args.Ls or (), args.pair_l
    written = generate(args.backend, arch, args.out, Ls)
    if pair_l is not None:
        written += generate_pairs(args.backend, arch, args.out, pair_l)
    if args.one_e_l is not None:  # beside the J chain, never by default
        written += generate_one_e(args.backend, arch, args.out, args.one_e_l)
    print(f"emit: {len(written)} {args.backend} translation unit(s) for "
          f"{arch.name} -> {args.out}")
    for p in written:
        print(f"   {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
