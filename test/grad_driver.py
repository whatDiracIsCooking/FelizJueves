"""The contracted nuclear gradients built ON A CARD by the d1 kernels --
harness.

    GradLaunch(kernel, n_rec, arrays, scalars)      one kernel launch
    one_electron_launches(basis, nuclei, target, M) -> [GradLaunch]
    hgs_launches(basis, nuclei, M)                  -> [GradLaunch]
    gjab_launches(basis, nuclei, D, one_centre=)            -> [GradLaunch]
    gk_launches(basis, nuclei, D)                   -> [GradLaunch]
    grad_steps(launches)                            -> [harness_driver.Step]
    grad_runners(launches, backend, workdir, units=) -> {kernel: Program}
    compile_grad(launches, n_atom, backend, tc, workdir, units=) -> GradBinary
    GradBinary.run()                                -> [run][n_atom][3]
    raised_plan(plan)                               -> LaunchPlan, bras one up
    gjab_tail(plan, basis, nuclei, D, lowstack=)    -> plan_driver.ChainTail
    gjab_read_back(drv, raw, n_atom)                -> ([run][lp] J_p, G[run])

ON THE SHARED RUNNER (#649, ported by #659): one runner per distinct
kernel (`harness_driver.chain_runners`, compiled by `compile_checked` under
#92's warning filter), and one launch per `GradLaunch` -- one thread per
record, `THREADS` lanes, or for `gk_` one warp of `THREADS` lanes per bra
record (`kernel_gk.launch_shape`) -- every argument named from the kernel's
own parameter list. `G` is the one OUT, carried: zeroed (the kernels'
precondition, `kernel_g1e.ZEROED_G`) into the first launch, each launch's
output the next one's input. The kernels are `gradient.kernel_g1e` /
`kernel_gjab` / `kernel_gk`'s, emitted by `emit/cuda.py` / `emit/hip.py`
unchanged; a unit the tier's generate phase wrote (`--grad-l`) is that
file.

THE RUNS: the whole chain `RUNS` times, each run its own processes from a
zeroed `G` (`harness_driver.run_chain`): the kernels add an atom's records
with fp64 atomics in an order the hardware chooses, so the run-to-run
spread is witnessed by the test, not assumed, and not the runner's bitwise
check.

`gjab`'s `Jp` here is the HOST's (`gradient.host.coulomb_jp` at order
`L + 1`): the kernel is tested, not the `[p|q]` launch that would feed it
on the card (`kernel_gjab.gjab_pq_spec`'s class, an ordinary d0 `[p|q]`
kernel, witnessed by the J chain's own device tests at d0).

THE CHAIN ON THE CARD (#619) is not this `main()` but the J chain's own
driver (`plan_driver.driver`, #157), its JAB step replaced by `gjab`
(`tail=`, `plan_driver.ChainTail`):

  1. DQ on the card, `D_q` / `Dmax` read back (the J chain's one
     round-trip), and the J chain's plan keyed and sorted on them;
  2. `raised_plan`: the same plan with every bra class `lp` raised to
     `lp + 1` -- the same bra records (`P`, `zeta`), the same sorted kets,
     the kernels of `class_spec(lp + 1, lq)`, which `split` reads as
     `gjab_pq_spec`'s class -- launched into one `J_p` buffer per raised
     bra class;
  3. `gjab_tail`: in the same process, after every `[p|q]` launch on the
     same stream, one `gjab_` launch per bra pair class, each record's `row`
     its row of its raised buffer, reading `J_p` where it lies on the card
     and adding into one zeroed `G`, read back per run (`gjab_read_back`).

No `J_p` comes from the host, or back from the card to be re-uploaded; it
is read back only as a witness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Program, Step, build_chain, chain_runners, run_chain
from plan_driver import RUNS, UPLOAD_INT, ChainTail, Driver, read_back

from felizjueves.ao.density import epair
from felizjueves.eri.j_coulomb.launch import ClassLaunch, LaunchPlan
from felizjueves.eri.j_coulomb.launch import class_kernels as plan_kernels
from felizjueves.eri.j_coulomb.launch import class_spec as plan_class_spec
from felizjueves.gradient import host
from felizjueves.gradient.kernel_g1e import build_g1e_kernel, launch_shape
from felizjueves.gradient.kernel_gjab import build_gjab_kernel, gjab_pq_spec, n_p1
from felizjueves.gradient.kernel_gk import build_gk_kernel
from felizjueves.gradient.kernel_gk import launch_shape as gk_launch_shape
from felizjueves.gradient.kernel_hgs import build_hgs_kernel
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.mmd.kernel import bra_functions, split
from felizjueves.spec import Target

__all__ = ["GjabChainLaunch", "GradBinary", "GradLaunch", "compile_grad",
           "gjab_chain_launches", "gjab_launches", "gjab_read_back",
           "gjab_tail", "gk_launches", "grad_runners", "grad_steps",
           "hgs_launches", "one_electron_launches", "raised_plan"]

THREADS = 32  # lanes per block, one record per thread; not numerical
G = "G"


@dataclass(frozen=True)
class GradLaunch:
    """One kernel over `n_rec` records: its array arguments by parameter
    name (every array parameter but `G`) and its scalar ones."""

    kernel: Kernel
    n_rec: int
    arrays: Mapping[str, np.ndarray]
    scalars: Mapping[str, int] = field(default_factory=dict)
    warp_per_record: bool = False  # `gk_`: one warp per record, (1, THREADS)

    def __post_init__(self) -> None:
        want = {p.name for p in self.kernel.params if p.array} - {G}
        if set(self.arrays) != want:
            raise ValueError(f"{self.kernel.name}: arrays {sorted(self.arrays)}"
                             f", the kernel takes {sorted(want)}")
        out = [p for p in self.kernel.params if p.intent is Intent.OUT]
        if [p.name for p in out] != [G]:
            raise ValueError(f"{self.kernel.name}: the one OUT is not G")


def _rec_arrays(rec: Mapping[str, list], kern: Kernel) -> dict:
    names = {p.name: p for p in kern.params if p.array}
    out = {}
    for k, v in rec.items():
        if k in names:
            out[k] = np.asarray(v, dtype=np.int32 if names[k].type
                                is Scalar.I32 else np.float64)
    return out


def one_electron_launches(basis, nuclei, target: Target,
                          M) -> list[GradLaunch]:
    """One `gs_` / `gt_` / `gv_` launch per ordered-by-`i <= j` pair class
    of `basis`, every record of the class."""
    out = []
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        kern = build_g1e_kernel((target, la, lb))
        rec = host.records(basis, nuclei, la, lb)
        arrs = _rec_arrays(rec, kern)
        arrs["M"] = np.asarray(M, dtype=np.float64)
        if target is Target.V:
            arrs["Z"] = np.array([z for z, _ in nuclei], dtype=np.float64)
            arrs["Cnuc"] = np.array([c for _, c in nuclei], dtype=np.float64)
        n = len(rec["s_ab"])
        out.append(GradLaunch(kern, n, arrs, {
            "n_rec": n, "n_ao": basis.n_ao, "n_atom": len(nuclei)}))
    return out


def hgs_launches(basis, nuclei, M) -> list[GradLaunch]:
    """One `hgs_` launch (the overlap gradient on HGP, `kernel_hgs`) per
    ordered-by-`i <= j` pair class of `basis`: one thread per CONTRACTED
    SHELL PAIR, so the launch's `n_rec` is its shell-pair count
    (`kernel_hgs.launch_shape` is `kernel_g1e`'s over that count)."""
    out = []
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        kern = build_hgs_kernel((la, lb))
        rec = host.pair_records(basis, nuclei, la, lb)
        arrs = _rec_arrays(rec, kern)
        arrs["M"] = np.asarray(M, dtype=np.float64)
        n_pair = len(rec["rec0"])
        out.append(GradLaunch(kern, n_pair, arrs, {
            "n_pair": n_pair, "n_rec": len(rec["s_ab"]), "n_ao": basis.n_ao,
            "n_atom": len(nuclei)}))
    return out


def gjab_launches(basis, nuclei, D, *, one_centre) -> list[GradLaunch]:
    """One `gjab_` launch per bra pair class, `Jp` the host's at order
    `L + 1` (module docstring), one row per record."""
    kets = host.coulomb_kets(basis, D)
    out = []
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        kern = build_gjab_kernel((la, lb))
        rec = host.records(basis, nuclei, la, lb)
        n = len(rec["s_ab"])
        jp = np.array([host.coulomb_jp(
            (rec["p_x"][r], rec["p_y"][r], rec["p_z"][r]), rec["zeta_ab"][r],
            la + lb + 1, kets, one_centre=one_centre) for r in range(n)])
        arrs = _rec_arrays(rec, kern)
        arrs |= {
            "XPA": np.array([rec[f"pa_{a}"] for a in "xyz"]).T.copy(),
            "XPB": np.array([rec[f"pb_{a}"] for a in "xyz"]).T.copy(),
            "s": np.asarray(rec["s_ab"], dtype=np.float64),
            "row": np.arange(n, dtype=np.int32), "Jp": jp,
            "M": np.asarray(D, dtype=np.float64)}
        out.append(GradLaunch(kern, n, arrs, {
            "n_rec": n, "n_ao": basis.n_ao, "n_atom": len(nuclei),
            "n_rows": n}))
    return out


def gk_launches(basis, nuclei, D) -> list[GradLaunch]:
    """One `gk_` launch per K class `(la, lc, lb, ld)` the basis reaches,
    every ordered bra record of `(la, lc)` against every ordered ket
    record of `(lb, ld)` (`gradient.host.exchange_records`)."""
    ls = sorted({s.l for s in basis.shells})
    out = []
    for am in [(a, c, b, d) for a in ls for c in ls for b in ls for d in ls]:
        kern = build_gk_kernel(am)
        bra, ket = host.exchange_records(basis, nuclei, am)
        arrs = _rec_arrays({**bra, **ket}, kern)
        arrs["M"] = np.asarray(D, dtype=np.float64)
        n_bra, n_ket = len(bra["s_ac"]), len(ket["s_bd"])
        out.append(GradLaunch(kern, n_bra, arrs, {
            "n_bra": n_bra, "n_ket": n_ket, "n_ao": basis.n_ao,
            "n_atom": len(nuclei)}, warp_per_record=True))
    return out


def grad_steps(launches: Sequence[GradLaunch]) -> list[Step]:
    """Each launch as a `harness_driver.Step`: its arrays and scalars by
    parameter name, `G` carried; one thread per record, `THREADS` lanes, or
    for `gk_` one warp of `THREADS` lanes per bra record."""
    out = []
    for x in launches:
        shape = gk_launch_shape if x.warp_per_record else launch_shape
        grid, block = shape(x.n_rec, THREADS)
        out.append(Step(x.kernel, grid, block, {**x.arrays, **x.scalars}))
    return out


def grad_runners(launches: Sequence[GradLaunch], backend: str, workdir: Path,
                 *, units: Mapping[str, Path] | None = None
                 ) -> dict[str, Program]:
    """One runner per distinct kernel of `launches` (module docstring)."""
    return chain_runners((x.kernel for x in launches), backend, workdir,
                         units=units)


@dataclass(frozen=True)
class GradBinary:
    """Every runner of a gradient's launches, compiled; `run` returns
    `[run][n_atom][3]`."""

    launches: tuple[GradLaunch, ...]
    n_atom: int
    backend: str
    binaries: Mapping[str, Path]  # by kernel name
    workdir: Path

    def run(self) -> np.ndarray:
        """`RUNS` runs, each from a zeroed `G` through every launch. The
        caller holds the card (its device fixture)."""
        Gs = run_chain(self.binaries, grad_steps(self.launches), G,
                       np.zeros((self.n_atom, 3)), self.workdir / "runs",
                       runs=RUNS)
        return np.array(Gs)


def compile_grad(launches: Sequence[GradLaunch], n_atom: int, backend: str,
                 tc: Toolchain, workdir: Path, *,
                 units: Mapping[str, Path] | None = None,
                 timeout: float = 1800.0) -> GradBinary:
    """Compile every runner with devtools/config.sh's flags, FAILING on a
    warning located in generated text (#92); `timeout` (s) caps each
    compile (#613's ladder parts pass the user's 300 s budget)."""
    runners = grad_runners(launches, backend, workdir, units=units)
    return GradBinary(tuple(launches), n_atom, backend,
                      build_chain(tc, runners, workdir, timeout=timeout),
                      workdir)


# --- the gjab chain on the card (#619): the J chain's driver, gjab for JAB ---

GJAB_PARAMS = ("n_rec", "n_ao", "n_atom", "n_rows", "zeta_ab", "XPA", "XPB",
               "s", "ea_bra", "eb_ket", "off_a", "off_b", "row", "atom_a",
               "atom_b", "Jp", "M", "G")
GJAB_DOUBLES = 10  # per record: zeta_ab XPA(3) XPB(3) s ea_bra eb_ket
GJAB_INTS = 5  # per record: off_a off_b row atom_a atom_b


def raised_plan(plan: LaunchPlan) -> LaunchPlan:
    """`plan` with every bra class `lp` raised to `lp + 1` (module
    docstring, "THE CHAIN ON THE CARD"): the same bra records and blocks,
    the same sorted kets and batch count, `class_spec(lp + 1, lq)`'s
    kernels. Raising is injective in `lp`, so no two bra classes merge."""
    out = []
    for cl in plan:
        lp = cl.lp + 1
        out.append(ClassLaunch(
            lp, cl.lq, cl.n_batches,
            tuple(replace(r, L=lp) for r in cl.bras), cl.kets,
            plan_kernels(plan_class_spec(lp, cl.lq, cl.n_batches)),
            cl.bra_blocks))
    return replace(plan, classes=tuple(out))


@dataclass(frozen=True)
class GjabChainLaunch:
    """One bra pair class's `gjab_` launch, reading `J_p[lp]` of the raised
    plan on the card: `data` per record `zeta_ab XPA XPB s ea_bra eb_ket`,
    `ints` `off_a off_b row atom_a atom_b` (exact as doubles)."""

    la: int
    lb: int
    lp: int
    ns: str
    kernel: Kernel
    data: np.ndarray  # (n_rec, GJAB_DOUBLES)
    ints: np.ndarray  # (n_rec, GJAB_INTS)

    @property
    def n_rec(self) -> int:
        return int(self.data.shape[0])


def gjab_chain_launches(plan: LaunchPlan, basis, nuclei, *,
                        lowstack: bool | None = None
                        ) -> list[GjabChainLaunch]:
    """Per bra pair class `(la, lb)`, sorted: every record of every bra
    shell pair of the class in `plan` (`raised_plan`'s value),
    `host.records`' values over those pairs, each record's `row` its row of
    `J_p[la + lb + 1]`."""
    shells = basis.shells
    blocks: dict[int, tuple] = {}
    for cl in plan:
        blocks.setdefault(cl.lp, cl.bra_blocks)
    groups: dict[tuple[int, int], list] = {}
    for lp, blks in blocks.items():
        for a, b, start, stop in blks:
            la, lb = shells[a].l, shells[b].l
            if la + lb + 1 != lp:
                raise ValueError(f"bra pair {(a, b)} filed under lp = {lp}")
            if stop - start != len(epair(shells[a], shells[b]).recs):
                raise ValueError(f"bra block {(a, b)}: {stop - start} rows")
            groups.setdefault((la, lb), []).append((a, b, start))
    out = []
    for c, (la, lb) in enumerate(sorted(groups)):
        members = groups[(la, lb)]
        kern = build_gjab_kernel((la, lb), lowstack=lowstack)
        names = tuple(p.name for p in kern.params)
        if names != GJAB_PARAMS:
            raise ValueError(f"{kern.name}: signature {names}, the driver "
                             f"passes {GJAB_PARAMS}")
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
        out.append(GjabChainLaunch(la, lb, la + lb + 1, f"fj_gjab{c}", kern,
                                   data, ints))
    return out


def _gjab_parts(x: GjabChainLaunch) -> list[np.ndarray]:
    """One gjab class's input, in the order the upload slot reads it."""
    d = x.data
    return [d[:, 0], d[:, 1:4].ravel(), d[:, 4:7].ravel(), d[:, 7],
            d[:, 8], d[:, 9], *(x.ints[:, k] for k in range(GJAB_INTS))]


def _gjab_call(x: GjabChainLaunch, c: int, jp: str, n_rows: str) -> str:
    n = x.n_rec
    grid, (bx, by) = launch_shape(n, THREADS)
    f, i = f"gd[{c}]", f"gi[{c}]"
    return (f"        // bra pair class ({x.la}, {x.lb}): J_p[{x.lp}] -> G\n"
            f"        {x.ns}::{x.kernel.name}<<<dim3({grid}), dim3({bx}, {by})>>>(\n"
            f"            {n}, NAO, NATOM, {n_rows}, {f}, {f} + {n}, "
            f"{f} + {4 * n}, {f} + {7 * n}, {f} + {8 * n}, {f} + {9 * n},\n"
            f"            {i}, {i} + {n}, {i} + {2 * n}, {i} + {3 * n}, "
            f"{i} + {4 * n}, {jp}, dM, dG);\n"
            f"        CHECK(devGetLastError());\n"
            f"        ++gjab_launches;\n")


def gjab_tail(plan: LaunchPlan, basis, nuclei, D, *,
              lowstack: bool | None = None) -> ChainTail:
    """The `plan_driver.ChainTail` launching every bra pair class's `gjab_`
    after the `[p|q]` launches of `plan` (`raised_plan`'s value), `M = D`:
    `G[n_atom, 3]` per run, after the `J_p` runs."""
    xs = gjab_chain_launches(plan, basis, nuclei, lowstack=lowstack)
    for x in xs:
        assert n_p1(x.la, x.lb) == len(bra_functions(
            plan_class_spec(x.lp, 0))), (x.la, x.lb)
        assert split(gjab_pq_spec(x.la, x.lb, 0, 0)) == split(
            plan_class_spec(x.lp, 0)), (x.la, x.lb)
    n_ao, n_atom = basis.n_ao, len(nuclei)
    Dm = np.asarray(D, dtype=np.float64)
    assert Dm.shape == (n_ao, n_ao), Dm.shape
    parts = [p for x in xs for p in _gjab_parts(x)] + [Dm.ravel()]
    per = GJAB_DOUBLES + GJAB_INTS

    def text(bslot: dict[int, int]) -> dict[str, str]:
        calls = "".join(_gjab_call(x, c, f"dJ[{bslot[x.lp]}]",
                                   f"nb[{bslot[x.lp]}]")
                        for c, x in enumerate(xs))
        return dict(
            jab_helpers=(
                "\n// the gjab chain (#619): one gjab launch per bra pair "
                f"class\nstatic const int NG = {len(xs)}, NAO = {n_ao}, "
                f"NATOM = {n_atom};\nstatic const int ng_[NG] = {{"
                + ", ".join(str(x.n_rec) for x in xs) + "};" + UPLOAD_INT),
            jab_count=(f"    for (int c = 0; c < NG; ++c) want += (size_t)"
                       f"{per} * ng_[c];\n"
                       "    want += (size_t)NAO * NAO;\n"),
            jab_need=("\n                          + (size_t)NATOM * 3 * "
                      "sizeof(double)"),
            jab_upload=(
                "    int uploads_gjab = 0, gjab_launches = 0, g_memsets = 0, "
                "g_readbacks = 0;\n"
                "    double* gd[NG];\n"
                "    int* gi[NG];\n"
                "    for (int c = 0; c < NG; ++c) {\n"
                f"        gd[c] = upload(h, (size_t){GJAB_DOUBLES} * ng_[c]);\n"
                f"        h += (size_t){GJAB_DOUBLES} * ng_[c];\n"
                f"        gi[c] = upload_int(h, (size_t){GJAB_INTS} * ng_[c]);\n"
                f"        h += (size_t){GJAB_INTS} * ng_[c];\n"
                "        if (!gd[c] || !gi[c]) { std::fprintf(stderr, \"gjab "
                "upload %d failed\\n\", c); return 1; }\n"
                "        ++uploads_gjab;\n"
                "    }\n"
                "    double* dM = upload(h, (size_t)NAO * NAO);\n"
                "    h += (size_t)NAO * NAO;\n"
                "    if (!dM) { std::fprintf(stderr, \"M upload failed\\n\"); "
                "return 1; }\n"
                "    double* dG = nullptr;\n"
                "    CHECK(devMalloc(&dG, (size_t)NATOM * 3 * sizeof(double)));\n"
                "    std::vector<double> Gh((size_t)NATOM * 3), "
                "gfirst(Gh.size()), glast(Gh.size());\n"),
            jab_launch=(
                "        // the gjab chain (#619): every [p|q] launch above is "
                "on this stream,\n"
                "        // so each raised J_p row is complete; gjab reads it "
                "on the card\n"
                "        CHECK(devMemset(dG, 0, (size_t)NATOM * 3 * "
                "sizeof(double)));\n"
                "        ++g_memsets;\n" + calls),
            jab_readback=(
                "        CHECK(devMemcpy(Gh.data(), dG, Gh.size() * "
                "sizeof(double),\n                        devMemcpyD2H));\n"
                "        ++g_readbacks;\n"),
            jab_keep=("        if (r == warmup) gfirst = Gh;\n"
                      "        glast = Gh;\n"),
            jab_free=(
                "    out.insert(out.end(), gfirst.begin(), gfirst.end());\n"
                "    out.insert(out.end(), glast.begin(), glast.end());\n"
                "    for (int c = 0; c < NG; ++c) {\n"
                "        CHECK(devFree(gd[c]));\n"
                "        CHECK(devFree(gi[c]));\n"
                "    }\n"
                "    CHECK(devFree(dM));\n"
                "    CHECK(devFree(dG));\n"),
            jab_counts=(
                "    std::printf(\"fj-gjab classes %d uploads_gjab %d "
                "gjab_launches %d \"\n                \"g_memsets %d "
                "g_readbacks %d\\n\", NG, uploads_gjab,\n                "
                "gjab_launches, g_memsets, g_readbacks);\n"),
        )

    return ChainTail(tuple((x.ns, x.kernel) for x in xs), text, tuple(parts),
                     n_atom * 3)


def gjab_read_back(drv: Driver, raw: np.ndarray, n_atom: int
                   ) -> tuple[list[dict[int, np.ndarray]], np.ndarray]:
    """A gjab chain driver's output: `([run][lp] -> J_p, G[run][n_atom][3])`."""
    assert drv.tail is not None and drv.tail.n_output == n_atom * 3
    tail = raw[raw.size - RUNS * n_atom * 3:]
    return read_back(drv, raw), tail.reshape(RUNS, n_atom, 3)
