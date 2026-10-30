"""The split HGP `J` / `K` kernels ON A CARD (#408) -- harness.

    record(plan, basis, D, *, threads=)                -> (list[HRec], HJKWalk)
    record_launches(plan, basis, D, *, keep, threads=) -> list[HRec]
    hjk_program(kernels, backend, workdir, *, name=)   -> Program
    hjk_stage(recs, kernels)                           -> fock_driver.Staged
    run_on_card(...)                                   fock_driver's, reused
    interpret(recs, n_ao)                              -> X (J or K)
    quartets(rec)                                      -> its ordered quartets

test/fock_driver.py's harness (#288) for the SPLIT build (milestone 14):
`launch_hjk`'s ordered pair stream (#403), `kernel_hj` (#404) / `kernel_hk`
(#405) emitted by `emit/` unchanged (#407), replayed on the card.

THE LAUNCHES ARE `walk_hjk`'s, RECORDED. `record` walks the plan with
`walk_hjk` (#406) and an injected `run` that only RECORDS each call -- the
kernel, its grid, its block width and every argument by name -- and returns
the output matrix unchanged, so the card replays exactly the launches the
interpreted build runs, and nothing about the plan, the routing or the
packing is decided here. `walk_hjk` refuses any plan that is not the whole
stream, and builds the kernel of every class the plan reaches; a basis
with a d shell reaches `(dd|dd)` (L = 8), which nothing here may build.
So `record_launches` RESTATES the walk's launch loop for the launches whose
kernel `keep` admits -- the same `kernel_spec`, `kernel_pairs`,
`pair_record`, `kernel_f.pack` and argument names, every one of them
`walk_hjk`'s own function -- and builds only those kernels.
test/test_hjk_ondevice.py holds the two equal, launch for launch, on every
full plan of H2/STO-3G and H2O/STO-3G (`keep` admitting everything), so the
restatement cannot drift from the walk unseen.

THE PROGRAM IS fock_driver's: the shared runner (#649; ported by #657) over
a set of emitted units, each in its own namespace, driven at run time by a
sequence (`fock_driver.stage`) -- any number of launches of any of its
kernels, in order, into ONE output matrix, `RUNS` runs each from a zeroed
matrix, every run kept (the split kernels add by atomics). The split
kernels' signature is the fused one's with `perm` where `diag` was (absent
in the `ordered` mode) and `J` / `K` where `F` was, and every argument is
read off the kernel's own parameter list, so a call cannot drift from the
signature. A launch is `grid = n_bra`, `block = (threads, 1)` (`kernel_hj.
launch_shape`). Nothing is closed or mirrored after the last launch: a
split kernel writes each element from its own blocks (kernel_hj, "THE
DEPOSIT").

No timing mode is used here (#331 owns any clock). Harness, in `test/`: not
a second emitter, no measurement decided here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fock_driver import Staged, runner_of, stage
from fock_driver import run_on_card as run_on_card  # re-exported: one runner
from plan_driver import Program

from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.hjk import kernel_hj as HJ
from felizjueves.eri.hjk import kernel_hk as HK
from felizjueves.eri.hjk import walk_hjk as W
from felizjueves.hgp.orbits import GROUP
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys

THREADS = 32  # block width: kets strided over 32 lanes; not numerical
OUTPUTS = (HJ.J_OUT, HK.K_OUT)
SCALARS = ("n_bra", "n_ket", "k_ab", "k_cd", "n_sh", "n_ao", HJ.PERM, "tau")
SHARED = ("Dsh", "D")


@dataclass(frozen=True)
class HRec:
    """One recorded launch: the kernel, its grid and block width, and every
    argument by name (arrays flat, row-major), as walk_hjk hands `run`."""

    kernel: Kernel
    grid: int
    threads: int
    args: Mapping[str, object]

    @property
    def name(self) -> str:
        return self.kernel.name

    @property
    def out(self) -> str:
        """The output matrix's parameter name: `J` or `K`."""
        return output_of(self.kernel)

    @property
    def perm(self) -> int:
        """The launch's `GROUP` index; 0 (the identity) without `perm`."""
        return int(self.args.get(HJ.PERM, 0))


def output_of(k: Kernel) -> str:
    outs = [p.name for p in k.params if p.intent is Intent.OUT]
    if len(outs) != 1 or outs[0] not in OUTPUTS:
        raise ValueError(f"{k.name}: outputs {outs} are not one of {OUTPUTS}")
    return outs[0]


def quartets(r: HRec) -> list[tuple[int, int, int, int]]:
    """The ordered quartets launch `r` meets: the kernel's built-order shells
    `x = (bra pair, ket pair)` carried back by `GROUP[perm]` (`q[i] =
    x[axes[i]]`, walk_hjk's own relabelling)."""
    shells = GROUP[r.perm]
    axes = tuple(shells.index(i) for i in range(4))
    bra = [int(v) for v in r.args["bra_sh"]]
    ket = [int(v) for v in r.args["ket_sh"]]
    out = []
    for i in range(0, len(bra), 2):
        for j in range(0, len(ket), 2):
            x = (bra[i], bra[i + 1], ket[j], ket[j + 1])
            out.append(tuple(x[axes[k]] for k in range(4)))
    return out  # type: ignore[return-value]


def record(plan, basis, D, *, threads: int = THREADS
           ) -> tuple[list[HRec], W.HJKWalk]:
    """walk_hjk over the whole `plan` with a run that records every launch
    and returns the output unchanged (module docstring)."""
    recs: list[HRec] = []

    def run(kernel, grid, block, args):
        recs.append(HRec(kernel, grid, block[0], dict(args)))
        out = output_of(kernel)
        return {out: args[out]}

    w = W.walk_hjk(plan, basis, D, run=run, threads=threads)
    return recs, w


def kernel_name(screen: str, mode: str, ordered) -> str:
    """The name of the kernel that serves the ordered class `ordered`."""
    spec = W.kernel_spec(screen, mode, tuple(ordered))
    return HJ.hj_name(spec) if screen == "j" else HK.hk_name(spec)


def record_launches(plan, basis, D, *, keep: Callable[[str], bool],
                    threads: int = THREADS) -> list[HRec]:
    """walk_hjk's launch loop (`walk_hjk.walk_hjk`, restated with its own
    functions) for the launches whose kernel name `keep` admits, in plan
    order, every output argument zero; builds only those kernels. The
    plan is not checked here: `record` on the same plan is (and the test
    holds the two equal)."""
    screen, mode = plan.screen, plan.mode
    rows = np.asarray(D, dtype=float)
    sh = basis.shells
    n_sh, n_ao = len(sh), basis.n_ao
    out_name = W.OUTPUT[screen]
    mod = W._BUILDER[screen]
    Dsh = KF.density_blocks(rows.tolist(), basis.offsets,
                            [s.n_cart for s in sh])
    D_flat = tuple(float(x) for x in rows.ravel())
    side = [[(p.a, p.b) for p in b.pairs] for b in plan.batches]
    G = {(p.a, p.b): float(p.G) for b in plan.batches for p in b.pairs}
    zero = (0.0,) * (n_ao * n_ao)
    kernels: dict[str, Kernel] = {}
    recs: list[HRec] = []
    for L in plan.launches:
        b, k = plan.batches[L.bra], plan.batches[L.ket]
        spec = W.kernel_spec(screen, mode, (b.la, b.lb, k.la, k.lb))
        name = HJ.hj_name(spec) if screen == "j" else HK.hk_name(spec)
        if not keep(name):
            continue
        if name not in kernels:
            kernels[name] = (HJ.build_hj_kernel(spec) if screen == "j"
                             else HK.build_hk_kernel(spec))
        kernel = kernels[name]
        shells = tuple(L.orientation.shells)
        kb, kk = W.kernel_pairs(shells, side[L.bra], side[L.ket])
        pb = KF.pack([W.pair_record(basis, *P, G=G[P]) for P in kb])
        pk = KF.pack([W.pair_record(basis, *P, G=G[P]) for P in kk])
        args = {
            "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree,
            "k_cd": pk.degree, "n_sh": n_sh, "n_ao": n_ao,
            HJ.PERM: GROUP.index(shells), "tau": float(plan.tau),
            "bra_prim": pb.prim, "AB": pb.sep, "G_ab": pb.G,
            "bra_sh": pb.sh, "bra_ao": pb.ao, "ket_prim": pk.prim,
            "CD": pk.sep, "G_cd": pk.G, "ket_sh": pk.sh, "ket_ao": pk.ao,
            "Dsh": Dsh, "D": D_flat, out_name: zero,
        }
        grid, block = mod.launch_shape(pb.n, threads)
        recs.append(HRec(kernel, grid, block[0],
                         {p.name: args[p.name] for p in kernel.params}))
    return recs


def interpret(recs: Sequence[HRec], n_ao: int) -> np.ndarray:
    """The recorded launches through the interpreter (Boys from
    `reference/`), in order, into one zeroed `n_ao x n_ao` matrix."""
    if len({r.out for r in recs}) != 1:
        raise ValueError("the launches do not share one output matrix")
    X = np.zeros(n_ao * n_ao)
    for r in recs:
        a = {}
        for p in r.kernel.params:
            v = X if p.name == r.out else r.args[p.name]
            if p.array:
                v = np.array(v, dtype=np.int32 if p.type is Scalar.I32
                             else float)
            a[p.name] = v
        X = Interpreter(r.kernel, launch=Launch(r.grid, (r.threads, 1), 32),
                        boys=boys).run(**a)[r.out]
    return np.asarray(X, dtype=float).reshape(n_ao, n_ao)


def _check_signature(k: Kernel) -> str:
    names = [p.name for p in k.params]
    out = output_of(k)
    want = [n for n in (*SCALARS, "bra_prim", "AB", "G_ab", "bra_sh",
                        "bra_ao", "ket_prim", "CD", "G_cd", "ket_sh",
                        "ket_ao", *SHARED, out) if n in names]
    if names != want:
        raise ValueError(f"{k.name}: signature {names} is not the split HGP "
                         "kernel's (kernel_hj / kernel_hk params_of, no tap)")
    return out


def hjk_program(kernels: Sequence[Kernel], backend: str, workdir: Path, *,
                units: Mapping[str, Path] | None = None,
                name: str = "hjk") -> Program:
    """The runner over `kernels`' units (module docstring, THE PROGRAM).
    One output matrix: every kernel writes `J`, or every one `K`."""
    if not kernels or len({k.name for k in kernels}) != len(kernels):
        raise ValueError("hjk_program takes distinct kernels")
    if name in {k.name for k in kernels}:
        raise ValueError(f"the program {name!r} would overwrite its unit")
    outs = {_check_signature(k) for k in kernels}
    if len(outs) != 1:
        raise ValueError(f"the kernels write {sorted(outs)}, not one matrix")
    return runner_of(kernels, backend, workdir, units=units, name=name)


def hjk_stage(recs: Sequence[HRec], kernels: Sequence[Kernel]) -> Staged:
    """The sequence of `recs`, each launching one of `kernels`, into their
    one output matrix (fock_driver's `stage`)."""
    outs = {r.out for r in recs}
    if len(outs) != 1:
        raise ValueError(f"the launches write {sorted(outs)}, not one matrix")
    (out,) = outs
    return stage(recs, kernels, out=out, shared=SHARED)
