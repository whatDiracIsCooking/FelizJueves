"""The contracted one-electron nuclear hessians built ON A CARD by the d2
kernels -- harness.

    HessLaunch(kernel, n_rec, arrays, scalars)          one kernel launch
    one_electron_launches(basis, nuclei, target, M)     -> [HessLaunch]
    d2metric_launches(aux, nuclei, W, *, lQ=)          -> [HessLaunch]  (#694)
    d2eri3_launches(basis, aux, nuclei, Gam, *, lP=)   -> [HessLaunch]  (#694)
    hkx_launches(basis, nuclei, D, classes, *, pairs=) -> [HessLaunch]  (#691)
    kx_launches(basis, nuclei, D, classes, *, form=, atom=, pairs=)
                                                       -> [HessLaunch]  (#689)
    hdq_on_card(basis, nuclei, D, backend, tc, workdir) -> {(lc, ld): (rec, D1)}
    gather_dkets(d1s)                                  -> [DKets]  (#690)
    hjab_tail(plan2, basis, nuclei, D, dkets, backend) -> plan_driver.Tail
    hess_steps(launches)                               -> [harness_driver.Step]
    hess_runners(launches, backend, workdir, units=)    -> {kernel: Program}
    compile_hess(launches, n_atom, backend, tc, workdir, units=) -> HessBinary
    HessBinary.run()                           -> [run][n_atom][3][n_atom][3]

`grad_driver.py`'s pattern with the one OUT `H[n_atom, 3, n_atom, 3]` in
place of `G[n_atom, 3]` (`hdq_`'s is `D1`, sized by `n_out`), on the shared
runner (#649, ported by #659): one runner per distinct kernel
(`harness_driver.chain_runners`, compiled under #92's warning filter), one
launch per `HessLaunch` -- one thread per record, `THREADS` lanes, or for
`hkx_` (`kernel_hkx`, the exchange Hessian) one warp of `THREADS` lanes per
bra record over its kets grouped by atom pair -- every
argument named from the kernel's parameter list, and the OUT carried from
zero (the kernels' precondition, `kernel_h1e.ZEROED_H`) through every
launch. The kernels are `hessian.kernel_h1e`'s (and `kernel_hjab` /
`kernel_hdq`'s, `kernel_kx`'s -- the exchange derivative matrix, its one
OUT `X[3, n_ao, n_ao]`, one warp per bra record as `hkx_` -- and the DF
`kernel_d2metric` / `kernel_d2eri3`'s, one thread per aux shell pair or
cell, on `df_grad_launch`'s arrays), emitted by `emit/cuda.py` /
`emit/hip.py` unchanged; a unit the tier's generate phase wrote
(`--hess-l`) is that file.

THE RUNS: the whole chain `RUNS` times, each from a zeroed OUT in its own
processes (`harness_driver.run_chain`), so the fp64 atomics' run-to-run
spread is witnessed by the test, not assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from conftest import Toolchain
from df_grad_launch import eri3_arrays, metric_arrays
from harness_driver import (
    Program,
    Ref,
    Step,
    Zeros,
    build_chain,
    chain_runners,
    run_chain,
)
from plan_driver import RUNS, Tail, as_exact_int, bra_rows, jp_buffer
from test_kernel_device import BACKEND

from felizjueves.ao.density import epair
from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.eri.j_coulomb.launch import class_kernels as plan_kernels
from felizjueves.eri.j_coulomb.launch import class_spec as plan_class_spec
from felizjueves.eri.j_coulomb.staged import _accumulating
from felizjueves.gradient import host
from felizjueves.hessian import host as hessian_host
from felizjueves.hessian import host_kx
from felizjueves.hessian.kernel_d2eri3 import build_d2eri3_kernel
from felizjueves.hessian.kernel_d2metric import build_d2metric_kernel
from felizjueves.hessian.kernel_h1e import build_h1e_kernel, launch_shape
from felizjueves.hessian.kernel_hdq import build_hdq_kernel, n_q1
from felizjueves.hessian.kernel_hjab import build_hjab_kernel, n_p1
from felizjueves.hessian.kernel_hkx import build_hkx_kernel
from felizjueves.hessian.kernel_hkx import launch_shape as hkx_launch_shape
from felizjueves.hessian.kernel_kx import build_kx_kernel
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.mmd.kernel import bra_functions
from felizjueves.performance import measured
from felizjueves.spec import Target

__all__ = ["DKets", "HessBinary", "HessLaunch", "HjabChainLaunch",
           "compile_hess", "d2eri3_launches", "d2metric_launches",
           "gather_dkets", "hdq_launch", "hdq_on_card", "hess_runners",
           "hess_steps", "hjab_chain_launches", "hjab_tail", "hkx_launches",
           "jp1_buffer", "kx_launches", "ket_pair_classes", "one_electron_launches",
           "record_arrays"]

THREADS = 32  # lanes per block, one record per thread; not numerical
H = "H"


@dataclass(frozen=True)
class HessLaunch:
    """One kernel over `n_rec` records: its array arguments by parameter
    name (every array parameter but `H`) and its scalar ones."""

    kernel: Kernel
    n_rec: int
    arrays: Mapping[str, np.ndarray]
    scalars: Mapping[str, int] = field(default_factory=dict)
    out: str = H  # the one OUT, shared by every launch (`hdq_`'s is `D1`)
    warp_per_record: bool = False  # `hkx_`: one warp per bra record

    def __post_init__(self) -> None:
        want = {p.name for p in self.kernel.params if p.array} - {self.out}
        if set(self.arrays) != want:
            raise ValueError(f"{self.kernel.name}: arrays {sorted(self.arrays)}"
                             f", the kernel takes {sorted(want)}")
        out = [p for p in self.kernel.params if p.intent is Intent.OUT]
        if [p.name for p in out] != [self.out]:
            raise ValueError(f"{self.kernel.name}: the one OUT is not "
                             f"{self.out}")


def one_electron_launches(basis, nuclei, target: Target,
                          M) -> list[HessLaunch]:
    """One `d2s_` / `d2t_` / `d2v_` launch per ordered-by-`i <= j` pair
    class of `basis`, every record of the class."""
    out = []
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        kern = build_h1e_kernel((target, la, lb))
        rec = host.records(basis, nuclei, la, lb)
        types = {p.name: p.type for p in kern.params if p.array}
        arrs = {k: np.asarray(v, dtype=np.int32 if types[k] is Scalar.I32
                              else np.float64)
                for k, v in rec.items() if k in types}
        arrs["M"] = np.asarray(M, dtype=np.float64)
        if target is Target.V:
            arrs["Z"] = np.array([z for z, _ in nuclei], dtype=np.float64)
            arrs["Cnuc"] = np.array([c for _, c in nuclei], dtype=np.float64)
        n = len(rec["s_ab"])
        out.append(HessLaunch(kern, n, arrs, {
            "n_rec": n, "n_ao": basis.n_ao, "n_atom": len(nuclei)}))
    return out


def d2metric_launches(aux, nuclei, W, *, lQ: int | None = None
                      ) -> list[HessLaunch]:
    """One `d2metric_` launch per metric class of `aux` (`metric_classes`'
    staging, the interpreted tests'), each in its default form; `lQ` keeps
    the classes with that ket `l` only."""
    out = []
    for c in metric_classes(aux):
        if lQ is not None and c.lQ != lQ:
            continue
        arrs = metric_arrays(c, aux, nuclei) | {
            "M": np.asarray(W, dtype=np.float64)}
        out.append(HessLaunch(build_d2metric_kernel((c.lP, c.lQ)), c.n_pair,
                              arrs, {"n_pair": c.n_pair, "n_rec": c.n_rec,
                                     "n_aux": aux.n_ao,
                                     "n_atom": len(nuclei)}))
    return out


def d2eri3_launches(basis, aux, nuclei, Gam, *, lP: int | None = None
                    ) -> list[HessLaunch]:
    """One `d2eri3_` launch per three-center class of `basis` x `aux`
    (`eri3_classes`' staging), each in its default form; `lP` keeps the
    classes with that aux `l` only."""
    out = []
    for c in eri3_classes(basis, aux):
        if lP is not None and c.lP != lP:
            continue
        arrs = eri3_arrays(c, basis, aux, nuclei) | {
            "M": np.asarray(Gam, dtype=np.float64)}
        out.append(HessLaunch(build_d2eri3_kernel((c.la, c.lb, c.lP)),
                              c.n_cell, arrs, {
            "n_cell": c.n_cell, "n_bra_rec": c.n_bra_rec,
            "n_aux_rec": c.n_aux_rec, "n_ao": basis.n_ao,
            "n_aux": aux.n_ao, "n_atom": len(nuclei)}))
    return out


def hkx_launches(basis, nuclei, D, classes=None, *, pairs=None,
                 lowstack: bool | None = None) -> list[HessLaunch]:
    """One `hkx_` launch per K class `(la, lc, lb, ld)` (#691), every
    ordered bra record of `(la, lc)` against every ordered ket record of
    `(lb, ld)`, the kets grouped by atom pair
    (`hessian.host.exchange_hessian_records`); one warp of `THREADS` lanes
    per bra record (`kernel_hkx.launch_shape`). `classes` defaults to every
    class the basis reaches; `pairs` maps a class to `(bra_pairs,
    ket_pairs)`, the shell pairs a witness launches (all when absent);
    `lowstack` forces a form (the default form otherwise)."""
    if classes is None:
        ls = sorted({s.l for s in basis.shells})
        classes = [(a, c, b, d) for a in ls for c in ls for b in ls
                   for d in ls]
    out = []
    for am in classes:
        kern = build_hkx_kernel(am, lowstack=lowstack)
        bp, kp = (pairs or {}).get(am, (None, None))
        bra, ket, seg = hessian_host.exchange_hessian_records(
            basis, nuclei, am, bra_pairs=bp, ket_pairs=kp)
        types = {p.name: p.type for p in kern.params if p.array}
        arrs = {k: np.asarray(v, dtype=np.int32 if types[k] is Scalar.I32
                              else np.float64)
                for k, v in {**bra, **ket, **seg}.items() if k in types}
        arrs["M"] = np.asarray(D, dtype=np.float64)
        n_bra, n_ket = len(bra["s_ac"]), len(ket["s_bd"])
        if not n_bra or not n_ket:
            continue
        out.append(HessLaunch(kern, n_bra, arrs, {
            "n_bra": n_bra, "n_ket": n_ket, "n_seg": len(seg["seg_b"]),
            "n_ao": basis.n_ao, "n_atom": len(nuclei)},
            warp_per_record=True))
    return out


def _only(rec: Mapping, keys: tuple[str, str], offs) -> dict:
    keep = [i for i, o in enumerate(zip(rec[keys[0]], rec[keys[1]],
                                        strict=True)) if o in offs]
    return {k: [v[i] for i in keep] for k, v in rec.items()}


def kx_launches(basis, nuclei, D, classes=None, *, form: Target,
                atom: int | None = None, pairs=None) -> list[HessLaunch]:
    """One `kx1_` (`form=Target.KX1`) or `kxc_` (`Target.KXC`, the bra
    records whose `c` sits on `atom`) launch per K class `(la, lc, lb,
    ld)` (#689), every bra record of `(la, lc)` against every ket record
    of `(lb, ld)` (`hessian.host_kx.kx_records`), one warp of `THREADS`
    lanes per bra record (`kernel_kx.launch_shape`, `hkx_`'s), into the
    one OUT `X[3, n_ao, n_ao]`. `classes` defaults to every class the
    basis reaches; `pairs` maps a class to `(bra_pairs, ket_pairs)`, the
    shell pairs a witness launches (all when absent). A class with no
    record (`kxc_` on an atom no `c` shell of it sits on) launches
    nothing."""
    if (form is Target.KXC) != (atom is not None):
        raise ValueError("kxc_ launches per atom, kx1_ for every atom")
    if classes is None:
        ls = sorted({s.l for s in basis.shells})
        classes = [(a, c, b, d) for a in ls for c in ls for b in ls
                   for d in ls]
    off = basis.offsets
    out = []
    for am in classes:
        kern = build_kx_kernel(am, form=form)
        bra, ket = host_kx.kx_records(basis, nuclei, am, atom=atom)
        bp, kp = (pairs or {}).get(am, (None, None))
        if bp is not None:
            bra = _only(bra, ("off_a", "off_c"),
                        {(off[i], off[k]) for i, k in bp})
        if kp is not None:
            ket = _only(ket, ("off_b", "off_d"),
                        {(off[j], off[m]) for j, m in kp})
        n_bra, n_ket = len(bra["s_ac"]), len(ket["s_bd"])
        if not n_bra or not n_ket:
            continue
        types = {p.name: p.type for p in kern.params if p.array}
        arrs = {k: np.asarray(v, dtype=np.int32 if types[k] is Scalar.I32
                              else np.float64)
                for k, v in {**bra, **ket}.items() if k in types}
        arrs["M"] = np.asarray(D, dtype=np.float64)
        out.append(HessLaunch(kern, n_bra, arrs, {
            "n_bra": n_bra, "n_ket": n_ket, "n_ao": basis.n_ao},
            out="X", warp_per_record=True))
    return out


def hess_steps(launches: Sequence[HessLaunch]) -> list[Step]:
    """Each launch as a `harness_driver.Step`, one thread per record,
    `THREADS` lanes -- or for `hkx_` one warp of `THREADS` lanes per bra
    record; its OUT carried."""
    out = []
    for x in launches:
        shape = hkx_launch_shape if x.warp_per_record else launch_shape
        grid, block = shape(x.n_rec, THREADS)
        out.append(Step(x.kernel, grid, block, {**x.arrays, **x.scalars}))
    return out


def hess_runners(launches: Sequence[HessLaunch], backend: str, workdir: Path,
                 *, units: Mapping[str, Path] | None = None
                 ) -> dict[str, Program]:
    """One runner per distinct kernel of `launches` (module docstring)."""
    return chain_runners((x.kernel for x in launches), backend, workdir,
                         units=units)


@dataclass(frozen=True)
class HessBinary:
    """Every runner of a hessian's launches, compiled; `run` returns
    `[run][n_atom][3][n_atom][3]`, or `[run][n_out]` given `n_out`."""

    launches: tuple[HessLaunch, ...]
    n_atom: int
    backend: str
    binaries: Mapping[str, Path]  # by kernel name
    workdir: Path
    n_out: int | None = None

    def run(self) -> np.ndarray:
        """`RUNS` runs, each from a zeroed OUT through every launch. The
        caller holds the card (its device fixture)."""
        outs = {x.out for x in self.launches}
        if len(outs) != 1:
            raise ValueError(f"the launches carry {sorted(outs)}, not one OUT")
        n = self.n_atom
        init = (np.zeros((n, 3, n, 3)) if self.n_out is None
                else np.zeros(self.n_out))
        return np.array(run_chain(self.binaries, hess_steps(self.launches),
                                  outs.pop(), init, self.workdir / "runs",
                                  runs=RUNS))


def compile_hess(launches: Sequence[HessLaunch], n_atom: int, backend: str,
                 tc: Toolchain, workdir: Path, *,
                 units: Mapping[str, Path] | None = None,
                 n_out: int | None = None,
                 timeout: float = 1800.0) -> HessBinary:
    """Compile every runner with devtools/config.sh's flags, FAILING on a
    warning located in generated text (#92)."""
    runners = hess_runners(launches, backend, workdir, units=units)
    return HessBinary(tuple(launches), n_atom, backend,
                      build_chain(tc, runners, workdir, timeout=timeout),
                      workdir, n_out)


# --- the Coulomb Hessian's chain on the card (#690) ----------------------------
#
# `hdq_` on the card per ket class, `D1` read back; the HOST gathers the
# derivative kets per atom, axis and ket Hermite class (the design's step 3,
# `hessian.host.coulomb_dkets` its reference); then ONE runner process: the
# J chain's plan with every bra class raised by TWO (`grad_driver.
# raised_plan(plan, 2)`, `kernel_hjab.hjab_pq2_spec`'s classes) into
# `J<lp>` -- `Jp2` -- and, as the tail (`plan_driver.Tail`), the `[p|q]`
# launches of `hjab_pq1_spec`'s classes over the derivative kets into one
# `jp1_<L>` buffer per bra Hermite class `L` -- `Jp1[n_atom, 3, n_rows,
# n_p1]`, each `(C, l)` slab a `Ref` -- then one `hjab_` launch per bra
# pair class reading both where they lie, into one zeroed `H`.

HJAB_PARAMS = ("n_rec", "n_ao", "n_atom", "n_rows", "zeta_ab", "XPA", "XPB",
               "s", "ea_bra", "eb_ket", "off_a", "off_b", "row", "atom_a",
               "atom_b", "Jp2", "Jp1", "M", "H")
UNIT_KET = 1.0  # a derivative ket's Iq and Dmax: the plan's tau is 0


def record_arrays(rec, *, atoms: bool = True) -> dict[str, np.ndarray]:
    """`gradient.host.records`' lists as the J chain's spelling of a record
    head (`XPA` / `XPB` interleaved, `s`), int32 offsets and atoms."""
    out = dict(
        zeta_ab=np.array(rec["zeta_ab"], dtype=np.float64),
        XPA=np.array([rec[f"pa_{a}"] for a in "xyz"]).T.copy(),
        XPB=np.array([rec[f"pb_{a}"] for a in "xyz"]).T.copy(),
        s=np.array(rec["s_ab"], dtype=np.float64),
        ea_bra=np.array(rec["ea_bra"], dtype=np.float64),
        eb_ket=np.array(rec["eb_ket"], dtype=np.float64))
    for k in ("off_a", "off_b") + (("atom_a", "atom_b") if atoms else ()):
        out[k] = np.array(rec[k], dtype=np.int32)
    return out


def ket_pair_classes(basis) -> list[tuple[int, int]]:
    """Every ket pair class `(l_c, l_d)` of the unique pairs `c <= d`
    (`hessian.host.coulomb_dkets`' pairs), sorted."""
    sh = basis.shells
    return sorted({(sh[i].l, sh[j].l) for i in range(len(sh))
                   for j in range(i, len(sh))})


def hdq_launch(basis, nuclei, D, lc: int, ld: int, *,
               lowstack: bool | None = None) -> tuple[HessLaunch, dict]:
    """One `hdq_` launch over every record of ket class `(lc, ld)`, `M =
    D`: the launch (its OUT `D1`) and the records."""
    rec = host.records(basis, nuclei, lc, ld)
    n = len(rec["s_ab"])
    arrs = record_arrays(rec, atoms=False) | {
        "M": np.asarray(D, dtype=np.float64)}
    return HessLaunch(build_hdq_kernel((lc, ld), lowstack=lowstack), n,
                      arrs, {"n_rec": n, "n_ao": basis.n_ao}, out="D1"), rec


def hdq_on_card(basis, nuclei, D, backend: str, tc: Toolchain,
                workdir: Path, *, timeout: float = 1800.0
                ) -> dict[tuple[int, int], tuple[dict, np.ndarray]]:
    """Per ket pair class, its records and `D1[6, n_rec, n_q1]` from the
    card (`RUNS` runs, required bit-identical: plain stores, one owner per
    element)."""
    out = {}
    for lc, ld in ket_pair_classes(basis):
        x, rec = hdq_launch(basis, nuclei, D, lc, ld)
        shape = (6, x.n_rec, n_q1(lc, ld))
        got = compile_hess([x], len(nuclei), backend, tc,
                           workdir / f"hdq_{lc}{ld}", n_out=int(np.prod(shape)),
                           timeout=timeout).run()
        assert all(np.array_equal(g, got[0]) for g in got), (lc, ld)
        out[lc, ld] = (rec, got[0].reshape(shape))
    return out


@dataclass(frozen=True)
class DKets:
    """The derivative kets of one atom `C`, axis `l` and ket Hermite class
    `lq` (`lc + ld + 1`): `Q[n, 3]`, `eta[n]`, `D[n, n_q]`."""

    C: int
    l: int  # noqa: E741
    lq: int
    Q: np.ndarray
    eta: np.ndarray
    D: np.ndarray


def gather_dkets(d1s: Mapping[tuple[int, int], tuple[dict, np.ndarray]]
                 ) -> list[DKets]:
    """THE GATHER (the design's step 3, the host's): every ket record whose
    first shell sits on `C` carries slab `l` of its `D1`, one whose second
    does slab `3 + l` (a ket with both on `C` appears twice), grouped by
    `(C, l, lq)`, `lq` its ket Hermite class one up. Records in class, then
    record, then side order (`hessian.host.coulomb_dkets`')."""
    groups: dict[tuple[int, int, int], list] = {}
    for (lc, ld), (rec, D1) in sorted(d1s.items()):
        lq = lc + ld + 1
        for r in range(len(rec["s_ab"])):
            Q = (rec["p_x"][r], rec["p_y"][r], rec["p_z"][r])
            for side, at in ((0, "atom_a"), (1, "atom_b")):
                for l in range(3):  # noqa: E741
                    groups.setdefault((rec[at][r], l, lq), []).append(
                        (Q, rec["zeta_ab"][r], D1[3 * side + l, r]))
    return [DKets(C, l, lq, np.array([k[0] for k in ks], dtype=np.float64),
                  np.array([k[1] for k in ks], dtype=np.float64),
                  np.array([k[2] for k in ks], dtype=np.float64))
            for (C, l, lq), ks in sorted(groups.items())]  # noqa: E741


@dataclass(frozen=True)
class HjabChainLaunch:
    """One bra pair class's `hjab_` launch, reading `J<lp>` (`lp = la + lb
    + 2`, the twice-raised plan's) and `jp1_<la + lb>` on the card: `data`
    per record `zeta_ab XPA XPB s ea_bra eb_ket`, `ints` `off_a off_b row
    atom_a atom_b` (exact as doubles)."""

    la: int
    lb: int
    lp: int
    kernel: Kernel
    data: np.ndarray  # (n_rec, 10)
    ints: np.ndarray  # (n_rec, 5)

    @property
    def n_rec(self) -> int:
        return int(self.data.shape[0])


def hjab_chain_launches(plan2, basis, nuclei, *,
                        lowstack: bool | None = None
                        ) -> list[HjabChainLaunch]:
    """Per bra pair class `(la, lb)`, sorted: every record of every bra
    shell pair of the class in `plan2` (`raised_plan(plan, 2)`'s value),
    `gradient.host.records`' values over those pairs, each record's `row`
    its row of `J<la + lb + 2>`."""
    shells = basis.shells
    blocks: dict[int, tuple] = {}
    for cl in plan2:
        blocks.setdefault(cl.lp, cl.bra_blocks)
    groups: dict[tuple[int, int], list] = {}
    for lp, blks in blocks.items():
        for a, b, start, stop in blks:
            la, lb = shells[a].l, shells[b].l
            if la + lb + 2 != lp:
                raise ValueError(f"bra pair {(a, b)} filed under lp = {lp}")
            if stop - start != len(epair(shells[a], shells[b]).recs):
                raise ValueError(f"bra block {(a, b)}: {stop - start} rows")
            groups.setdefault((la, lb), []).append((a, b, start))
    out = []
    for la, lb in sorted(groups):
        members = groups[(la, lb)]
        kern = build_hjab_kernel((la, lb), lowstack=lowstack)
        names = tuple(p.name for p in kern.params)
        if names != HJAB_PARAMS:
            raise ValueError(f"{kern.name}: signature {names}, the driver "
                             f"passes {HJAB_PARAMS}")
        rec = host.records(basis, nuclei, la, lb,
                           pairs=[(a, b) for a, b, _ in members])
        eps = [epair(shells[a], shells[b]).recs for a, b, _ in members]
        rows = [start + k for (_, _, start), r in zip(members, eps, strict=True)
                for k in range(len(r))]
        if rec["zeta_ab"] != [z for r in eps for _, z, _ in r]:
            raise ValueError(f"({la}, {lb}): host.records and the plan's "
                             "bra records disagree")
        data = np.column_stack([
            rec["zeta_ab"], *(rec[f"pa_{a}"] for a in "xyz"),
            *(rec[f"pb_{a}"] for a in "xyz"), rec["s_ab"], rec["ea_bra"],
            rec["eb_ket"]]).astype(np.float64)
        ints = np.column_stack([
            rec["off_a"], rec["off_b"], rows, rec["atom_a"],
            rec["atom_b"]]).astype(np.float64)
        out.append(HjabChainLaunch(la, lb, la + lb + 2, kern, data, ints))
    return out


def jp1_buffer(L: int) -> str:  # noqa: N803
    """Bra Hermite class `L`'s `Jp1[n_atom, 3, n_rows, n_p1]` buffer."""
    return f"jp1_{L}"


def hjab_tail(plan2, basis, nuclei, D, dkets: Sequence[DKets], backend: str,
              *, lowstack: bool | None = None) -> Tail:
    """The `plan_driver.Tail` of the Coulomb Hessian's chain (the section
    note above): the `Jp1` `[p|q]` launches over `dkets`, every bra
    Hermite class against every `(C, l, lq)` group, each kernel atomized
    (several ket classes add into one slab), then every bra pair class's
    `hjab_`, `M = D`, into `H[n_atom, 3, n_atom, 3]`. `H` and every `jp1_`
    buffer are zeroed by the tail's first launch."""
    if plan2.tau != 0.0:
        raise ValueError("the derivative kets are unsorted: the plan's tau "
                         "must be 0, so the [p|q] kernel's break never fires")
    _, arch, _ = BACKEND[backend]
    xs = hjab_chain_launches(plan2, basis, nuclei, lowstack=lowstack)
    rows = bra_rows(plan2)
    n_ao, n_atom = basis.n_ao, len(nuclei)
    Dm = np.ascontiguousarray(D, dtype=np.float64)
    assert Dm.shape == (n_ao, n_ao), Dm.shape
    buffers: dict = {"M": Dm}
    kernels: list[Kernel] = []
    steps: list[Step] = []
    for g in dkets:
        m = len(g.eta)
        buffers[f"dk_{g.C}_{g.l}_{g.lq}"] = np.concatenate([
            g.Q.ravel(), g.eta, np.full(m, UNIT_KET), np.full(m, UNIT_KET),
            g.D.ravel()])
    for L in sorted({x.la + x.lb for x in xs}):
        lp2, n = L + 2, rows[L + 2]
        n1 = n_p1(*_any_class(xs, L))
        buffers[jp1_buffer(L)] = Zeros((n_atom, 3, n, n1))
        b = f"bra{lp2}"
        for g in dkets:
            k, m = f"dk_{g.C}_{g.l}_{g.lq}", len(g.eta)
            for kern in map(_accumulating,
                            plan_kernels(plan_class_spec(L + 1, g.lq, 1))):
                assert len(bra_functions(plan_class_spec(L + 1, g.lq))) == n1
                kernels.append(kern)
                steps.append(Step(kern, n, (1, arch.wavefront), {
                    "n_bra": n, "n_ket": m, "tau": 0.0,
                    "P": Ref(b), "zeta": Ref(b, 3 * n), "Ip": Ref(b, 4 * n),
                    "Q": Ref(k), "eta": Ref(k, 3 * m), "Iq": Ref(k, 4 * m),
                    "Dmax": Ref(k, 5 * m), "D": Ref(k, 6 * m),
                    "J": Ref(jp1_buffer(L), (3 * g.C + g.l) * n * n1)},
                    shared_bytes=measured.jsh_slots(kern) * arch.wavefront
                    * 8))
    for c, x in enumerate(xs):
        n, d = x.n_rec, x.data
        f, i = f"hd{c}", f"hi{c}"
        buffers[f] = np.concatenate([d[:, 0], d[:, 1:4].ravel(),
                                     d[:, 4:7].ravel(), d[:, 7], d[:, 8],
                                     d[:, 9]])
        buffers[i] = as_exact_int(np.concatenate(
            [x.ints[:, k] for k in range(5)]))
        grid, block = launch_shape(n, THREADS)
        kernels.append(x.kernel)
        steps.append(Step(x.kernel, grid, block, {
            "n_rec": n, "n_ao": n_ao, "n_atom": n_atom, "n_rows": rows[x.lp],
            "zeta_ab": Ref(f), "XPA": Ref(f, n), "XPB": Ref(f, 4 * n),
            "s": Ref(f, 7 * n), "ea_bra": Ref(f, 8 * n),
            "eb_ket": Ref(f, 9 * n), "off_a": Ref(i), "off_b": Ref(i, n),
            "row": Ref(i, 2 * n), "atom_a": Ref(i, 3 * n),
            "atom_b": Ref(i, 4 * n), "Jp2": jp_buffer(x.lp),
            "Jp1": jp1_buffer(x.la + x.lb), "M": "M", H: H}))
    zero = (H, *sorted(k for k in buffers if k.startswith("jp1_")))
    steps[0] = replace(steps[0], zero=zero)
    return Tail(tuple(kernels), buffers, tuple(steps), H,
                (n_atom, 3, n_atom, 3))


def _any_class(xs: Sequence[HjabChainLaunch], L: int) -> tuple[int, int]:  # noqa: N803
    return next((x.la, x.lb) for x in xs if x.la + x.lb == L)
