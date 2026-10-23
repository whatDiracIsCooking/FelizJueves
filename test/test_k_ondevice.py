"""The capstone (#206): the whole molecule's `K_mn` from the K LAUNCH PLAN, on
the RTX 3080 and the RX 9060 XT.

WHAT RUNS WHERE. `launch_k.k_launches` plans the molecule (#204), with
`kernel_k.build_k_kernels` (#198, #203) injected as the plan's `build=`; a
driver `main()` generated FROM THAT PLAN (`plan_driver.k_driver`, harness in
`test/`, never in `src/`) uploads each bra pair class, each ket pair class
and each density class ONCE, zeroes ONE K buffer once per run, launches every
batch kernel of every K class once per `b` shell of its kets (the K kernel
writes one `K[a, b]` block per bra record, `b` from the ket:
`plan_driver`'s "K on the card"), every launch ADDING into its
(bra pair class, b) block, and reads the buffer back. The scatter of the
per-record blocks into `K_mn` is host-side (`plan_driver.k_scatter`), as eq.
20 is for J. The screen is REAL on the screened cells: `Ip`, `Iq` are the
plan's `I_ac`, `I_bd` (`ao.k.record_bounds`), `maxD` the plan's `max|D|`,
`tau` the plan's, and the kets arrive in the plan's order, sorted by `I_bd`
alone.

THREE ANCESTRIES IN THE ASSERTION, and libcint where a fixture exists:

  host blocks  every (bra record, ket record) of every launch through
               `ao.k._k_quartet` (the host's staged arithmetic), the screen
               applied with the kernel's own association: the device BUFFER,
               record by record, `|dev - host| <= ORACLE_TOL max|host|`.
  launch walk  `launch_k.walk_k` over the SAME plan (#204): `K_mn`,
               `|K - K_walk| <= ORACLE_TOL max|K_walk|`.
  host sweep   `ao.k.exchange_matrix`, the UNIQUE sweep (#197) -- a
               different quadruple set and weights from the plan's ordered
               one -- at tau = 0: `|K - K0| <= ORACLE_TOL max|K0| + tau
               scale` (scale the walk's, #200's screening claim). On the
               truncated cells through test_launch_k's `_truncated` hook.
  libcint      the tracked `get_jk` fixture (#199), `|K - vk| <= 1e-12
               scale_K` (+ `tau scale` when screened): the whole matrix on
               H2, the s-s block on the truncated H2O cells (#199's
               argument: an s-s element sums only quartets with L <= 2).

ORACLE_TOL is test_ao_j_ondevice.py's INTERP_TOL (1e-13 relative to the
largest element) and the libcint tolerance test_ao_k.py's KJ_TOL (1e-12 of
scale_K): both INHERITED from the J build, not re-derived and not loosened.
The card differs from the host by FMA contraction and libm `exp`/`sqrt`: a
different rounding, far inside them.

THE fp64-ATOMIC RUN-TO-RUN TOLERANCE (the J build's argument, carried to K).
The driver runs the whole plan RUNS = 2 times in one process. Each element of
a block receives W atomic adds, one per launch writing that block (W =
`KDriver.writers`), each adding a partial the kernel's own deterministic
arithmetic produced, every partial a sub-sum of the record pairs' terms. So

    |buf(run 1) - buf(run 2)| <= 2 gamma_{W-1} A,   element by element,

`A` the absolute-value twin (`_k_quartet(want_scale=True)`) summed over
EVERY ket of the launch, which bounds a screened run's partials too. Whether
the runs came out bit-identical is printed, not asserted.

THE L <= 2 RULE (milestone 10's standing instruction: no class with
`L = la + lc + lb + ld > 2` is compiled or run before #231). The fast cells
are all-s molecules (H2/STO-3G, H2O/STO-3G's s sub-basis, two H2 6 bohr
apart, test_launch_k.py's cases) and H2O/STO-3G planned WHOLE and TRUNCATED
to its classes with L <= 2 -- every L <= 2 placement of its s and p shells
on the card, at 1 and 3 batches. Every host `[p|q]` of a fast cell goes
through test_launch_k's `_guarded`, which RAISES above the rule, and
`_build` refuses to build a kernel above it; the truncated cells plan with
a STUB bound (no `pq`; tau = 0), since a p-pair's bound needs [p|p] at
|p| = 2. The issue's acceptance case, H2O/STO-3G WHOLE (`(pp|pp)`, L = 4)
at 1 and 3 batches against the whole `vk`, is written below and marked
`slow`: deferred to #231, which runs it on both cards.

COMPILE-ONLY COMPANION. `test_<backend>_k_driver_compiles` (`cuda` / `hip`,
no `gpu`) generates and compiles every fast cell's driver where no card is,
with devtools/config.sh's flags; a warning located in generated text fails
it (#92), as it fails the device tests' own compiles.

FAST TIER (no compiler, no card, no PySCF): the driver's shape against the
plan, and the driver INTERPRETED -- the staged input sliced exactly as the
generated `main()` slices it, each launch run through `interp`, scattered --
against the walk, on a synthetic basis with two s shells and a p shell
(several `b` shells per ket class, several `d` shells per launch) and on H2.

WHAT SKIPS: the device tests are `gpu` plus `cuda` or `hip` and take the
compiler and device fixtures, so on a box without the card they SKIP -- the
run is green and empty, and devtools/kernel-tier.sh exits 2. The VRAM probe
(`run_device_binary(..., vram=)`) runs first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache, partial
from pathlib import Path

import numpy as np
import pytest
import test_ao_j_ondevice
import test_ao_k
from oracle import manifest
from plan_driver import (
    RUNS,
    KDriver,
    compile_checked,
    k_block,
    k_counts,
    k_driver,
    k_read_back,
    k_scatter,
    k_stage,
)
from test_kernel_device import BACKEND, _toolchain
from test_launch_k import _case as _launch_case
from test_launch_k import _guarded, _stub, _truncated, schwarz

from felizjueves.ao.density import checked_density, epair
from felizjueves.ao.eri import eri_block_and_scale
from felizjueves.ao.k import (
    _dcd,
    _k_quartet,
    exchange_matrix,
    exchange_matrix_and_scale,
    record_bounds,
)
from felizjueves.ao.shell import Basis, Shell
from felizjueves.ir import Assign, AtomicAdd, Index, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.kernel_k import build_k_kernels
from felizjueves.kernel_k import launch_shape as k_launch_shape
from felizjueves.launch_k import KLaunchPlan, k_launches, walk_k
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq as ref_pq
from felizjueves.spec import Spec

L_MAX_RUN = 2  # the milestone's rule: nothing above this is compiled or run
JK_TOL = test_ao_k.KJ_TOL  # 1e-12 scale_K vs libcint, inherited
ORACLE_TOL = test_ao_j_ondevice.INTERP_TOL  # 1e-13 max|K|, inherited
U = 2.0**-53


@dataclass(frozen=True)
class Cell:
    """One device build: a molecule, `tau`, `n_batches`, and whether the
    plan is truncated to its classes with L <= 2."""

    case: str
    tau: float
    n: int
    truncated: bool = False
    slow: bool = False

    @property
    def id(self) -> str:
        return (f"{self.case}-tau{self.tau:g}-b{self.n}"
                + ("-Lle2" if self.truncated else ""))

    @property
    def pq(self):
        """`_guarded` for a fast cell (raises above L = 2); a slow cell's
        whole H2O reaches L = 4 (#231's)."""
        return ref_pq if self.slow else _guarded


FAST = (
    Cell("h2_sto3g_k", 0.0, 1),
    Cell("h2_sto3g_k", 1e-3, 1),
    Cell("h2o_sto3g_s", 0.0, 1),
    Cell("h2o_sto3g_s", 1e-4, 1),
    Cell("two_h2", 0.0, 1),
    Cell("two_h2", 1e-6, 1),
    Cell("h2o_sto3g_k", 0.0, 1, truncated=True),
    Cell("h2o_sto3g_k", 0.0, 3, truncated=True),
)
# The issue's acceptance case: H2O/STO-3G whole, (pp|pp) at L = 4. Deferred
# to #231 under the L <= 2 rule.
SLOW = (
    Cell("h2o_sto3g_k", 0.0, 1, slow=True),
    Cell("h2o_sto3g_k", 0.0, 3, slow=True),
)
PARAMS = ([pytest.param(c, id=c.id) for c in FAST]
          + [pytest.param(c, id=c.id, marks=pytest.mark.slow) for c in SLOW])
FAST_PARAMS = [pytest.param(c, id=c.id) for c in FAST]


@cache
def _case(name: str):
    """(libcint fixture or None, basis, D)."""
    if name.endswith("_k"):
        fx, basis = test_ao_k._kfx(name)
        return fx, basis, np.array(fx.density)
    basis, D = _launch_case(name)
    return None, basis, D


def _build(spec: Spec, *, allow_above: bool = False):
    """`kernel_k.build_k_kernels`, refusing a class above the rule."""
    if sum(spec.am) > L_MAX_RUN and not allow_above:
        raise AssertionError(f"K class {spec.am} at L = {sum(spec.am)} > "
                             f"{L_MAX_RUN}: the milestone's L <= 2 rule")
    return build_k_kernels(spec)


def _bare_plan(cell: Cell) -> KLaunchPlan:
    """The plan, no kernels built; truncated to L <= 2 when the cell is."""
    _, basis, D = _case(cell.case)
    stub = cell.truncated or cell.slow
    assert not (stub and cell.tau), "a stub bound screens nothing"
    bound = _stub if stub else partial(record_bounds, pq=cell.pq)
    P = k_launches(basis, D, bound=bound, key=schwarz, tau=cell.tau,
                   n_batches=cell.n, strict=False)
    if cell.truncated:
        P = replace(P, classes=tuple(c for c in P if c.L <= L_MAX_RUN))
    return P


@cache
def _plan(cell: Cell) -> KLaunchPlan:
    """The plan with every class's kernels built (`build=`'s builder, called
    after the truncation so nothing above the rule is ever built)."""
    P = _bare_plan(cell)
    if not cell.slow:
        assert all(c.L <= L_MAX_RUN for c in P), [c.am for c in P]
    built: dict[Spec, tuple] = {}
    out = []
    for c in P:
        if c.spec not in built:
            built[c.spec] = _build(c.spec, allow_above=cell.slow)
        out.append(replace(c, kernels=built[c.spec]))
    return replace(P, classes=tuple(out))


def _gamma(n: int) -> float:
    return n * U / (1.0 - n * U)


# ---------------------------------------------------------------------------
# The host twins of the device buffer
# ---------------------------------------------------------------------------


def _host_blocks(drv: KDriver, basis: Basis, D, pq):
    """`(X, A)`, each a K buffer: per launch segment (once, not per batch),
    per bra record, `ao.k._k_quartet` over the segment's kets in launch
    order -- `X` screened with the KERNEL's association `Ib * (Iq * maxD)
    < tau`, a `break`; `A` the absolute-value twin over EVERY ket."""
    rows = checked_density(basis, D)
    shells = basis.shells
    bras, kets = dict(drv.bra), dict(drv.ket)
    one: dict = {}

    def rec(r):
        key = (r.i, r.j, r.prim)
        if key not in one:
            ep = epair(shells[r.i], shells[r.j])
            one[key] = replace(ep, recs=(ep.recs[r.prim],))
        return one[key]

    X, A = np.zeros(drv.n_k), np.zeros(drv.n_k)
    seen = set()
    for x in drv.launches:
        if (x.am, x.b) in seen:
            continue
        seen.add((x.am, x.b))
        blk = drv.blocks[x.block]
        seg = kets[x.am[2:]][x.start:x.start + x.n_ket]
        for row, br in enumerate(bras[x.am[:2]]):
            at = blk.offset + row * blk.n_ab
            live = True
            for kr in seg:
                if drv.tau > 0.0 and br.G * (kr.G * drv.dm) < drv.tau:
                    live = False
                Xq, Sq = _k_quartet(rec(br), rec(kr),
                                    _dcd(basis, rows, br.j, kr.j), pq=pq,
                                    want_scale=True)
                A[at:at + blk.n_ab] += np.ravel(Sq)
                if live:
                    X[at:at + blk.n_ab] += np.ravel(Xq)
    return X, A


def _ss_scale(basis: Basis, D: np.ndarray, pq) -> tuple[list[int], np.ndarray]:
    """The s AOs and their block of `scale_K = sum_cd |D_cd| scale_eri
    (ac|bd)` -- test_ao_k.py's s-s block scale, every quartet at L <= 2."""
    sh, off = basis.shells, basis.offsets
    s_idx = [i for i, s in enumerate(sh) if s.l == 0]
    aos = [off[i] for i in s_idx]
    scale = np.zeros((len(aos), len(aos)))
    for ia, a in enumerate(s_idx):
        for ib, b in enumerate(s_idx):
            for c, sc in enumerate(sh):
                for d, sd in enumerate(sh):
                    assert sh[a].l + sc.l + sh[b].l + sd.l <= L_MAX_RUN
                    _, s = eri_block_and_scale((sh[a], sc), (sh[b], sd), pq=pq)
                    s = np.array(s).reshape(sc.n_cart, sd.n_cart)
                    dcd = np.abs(D[off[c]:off[c] + sc.n_cart,
                                   off[d]:off[d] + sd.n_cart])
                    scale[ia, ib] += float(np.sum(s * dcd))
    return aos, scale


# ---------------------------------------------------------------------------
# The device run
# ---------------------------------------------------------------------------


def _check_counts(c: dict[str, int], drv: KDriver) -> None:
    nb, nk, ne, nl = len(drv.bra), len(drv.ket), len(drv.dcd), len(drv.launches)
    assert c == dict(runs=RUNS, bra_classes=nb, ket_classes=nk,
                     dcd_classes=ne, uploads_bra=nb, uploads_ket=nk,
                     uploads_dcd=ne, k_buffers=1, memsets=RUNS,
                     launches=nl * RUNS, readbacks=RUNS, plan_launches=nl), c


def _check(backend: str, cell: Cell, request, run_device_binary,
           workdir: Path) -> None:
    tc = _toolchain(backend, request)
    fx, basis, D = _case(cell.case)
    plan = _plan(cell)
    drv = k_driver(plan, basis, backend, workdir / "gen")
    binary = compile_checked(tc, drv, workdir / "k_driver")
    src, dst = workdir / "in.bin", workdir / "out.bin"
    k_stage(drv, basis, D).tofile(src)
    out = run_device_binary(binary, str(src), str(dst), vram=backend)
    _check_counts(k_counts(out), drv)
    runs = k_read_back(drv, np.fromfile(dst, dtype=np.float64))
    for r in runs:
        assert np.all(np.isfinite(r)), "non-finite K buffer"

    # the host twins of the buffer: record level, and the atomic bound
    Xh, A = _host_blocks(drv, basis, D, cell.pq)
    bound = np.zeros(drv.n_k)
    for q, W in drv.writers.items():
        b = drv.blocks[q]
        bound[b.offset:b.offset + b.size] = 2.0 * _gamma(W - 1)
    bound *= A
    diff = np.abs(runs[0] - runs[1])
    bitwise = bool(np.all(diff == 0.0))
    assert np.all(diff <= bound), (
        f"K buffer run to run: worst {float(np.max(diff)):.3e} over the "
        "fp64-atomic bound")
    rec = float(np.max(np.abs(runs[0] - Xh)) / np.max(np.abs(Xh)))

    # K_mn: the walk over the same plan, the host sweep, libcint
    K = k_scatter(drv, basis, runs[0])
    w = walk_k(plan, basis, D, pq=cell.pq)
    Kw, scale_tau = np.array(w.K), cell.tau * np.array(w.scale)
    rel_w = float(np.max(np.abs(K - Kw)) / np.max(np.abs(Kw)))
    if cell.truncated:
        calls: list[int] = []
        K0 = np.array(exchange_matrix(basis, D, pq=cell.pq,
                                      k=_truncated(calls)))
        assert max(calls) > L_MAX_RUN  # the cut is real
    else:
        K0 = np.array(exchange_matrix(basis, D, pq=cell.pq))
    big0 = float(np.max(np.abs(K0)))
    rel_0 = float(np.max(np.abs(K - K0)) / big0)
    use_0 = float(np.max(np.abs(K - K0) / (ORACLE_TOL * big0 + scale_tau)))
    lib = "none"
    if fx is not None:
        vk = np.array(fx.arrays[manifest.K_ARRAY])
        if cell.truncated:
            aos, sc = _ss_scale(basis, D, cell.pq)
            ix = np.ix_(aos, aos)
            got, ref, allowed = K[ix], vk[ix], JK_TOL * sc + scale_tau[ix]
            where = f"s-s block {len(aos)}^2 of {basis.n_ao}^2"
        else:
            sc = np.array(exchange_matrix_and_scale(basis, D, pq=cell.pq)[1])
            got, ref, allowed = K, vk, JK_TOL * sc + scale_tau
            where = "whole"
        lib = (f"{where}: max|K - vk|/scale = "
               f"{float(np.max(np.abs(got - ref) / sc)):.2e} (tol {JK_TOL:g}"
               f"{' + tau scale' if cell.tau else ''}; worst "
               f"{float(np.max(np.abs(got - ref) / allowed)):.2e} of it)")

    arch = BACKEND[backend][1]
    ran = " ".join("({}{}|{}{})".format(*c.am) + f"b{c.n_batches}"
                   for c in plan)
    print(f"\n{backend} {arch.name} {cell.id}: {len(plan)} classes [{ran}], "
          f"refused {list(plan.refused)}; {len(drv.launches)} launches per "
          f"run into {len(drv.blocks)} blocks of one K buffer; walk skipped "
          f"{w.n_skipped} of {w.n_pairs}; buffer vs host blocks {rec:.2e}, "
          f"K vs walk {rel_w:.2e}, K vs exchange_matrix {rel_0:.2e} (tol "
          f"{ORACLE_TOL:g}{' + tau scale' if cell.tau else ''}; worst "
          f"{use_0:.2e} of it); libcint "
          f"{lib}; runs bit-identical: {bitwise}")

    if cell.tau > 0.0:
        assert w.n_skipped > 0, "the screen never fired"
    else:
        assert w.n_skipped == 0
    assert rec <= ORACLE_TOL, f"buffer vs host blocks: {rec:.2e}"
    assert rel_w <= ORACLE_TOL, f"K vs the launch walk: {rel_w:.2e}"
    assert np.all(np.abs(K - K0) <= ORACLE_TOL * big0 + scale_tau), (
        f"K vs exchange_matrix: {rel_0:.2e}")
    if fx is not None:
        assert np.all(np.abs(got - ref) <= allowed), f"K vs libcint, {lib}"


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cell", PARAMS)
def test_cuda_k_from_the_launch_plan(cell, request, run_device_binary,
                                     tmp_path) -> None:
    """The whole molecule's `K` from the K plan on the RTX 3080."""
    _check("cuda", cell, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cell", PARAMS)
def test_hip_k_from_the_launch_plan(cell, request, run_device_binary,
                                    tmp_path) -> None:
    """The whole molecule's `K` from the K plan on the RX 9060 XT."""
    _check("hip", cell, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions: no card, the tier's flags, #92's warning filter
# ---------------------------------------------------------------------------


def _compile_only(backend: str, tc, cell: Cell, tmp_path: Path) -> None:
    _, basis, _ = _case(cell.case)
    drv = k_driver(_plan(cell), basis, backend, tmp_path / "gen")
    compile_checked(tc, drv, tmp_path / "k_driver")


@pytest.mark.cuda
@pytest.mark.parametrize("cell", FAST_PARAMS)
def test_cuda_k_driver_compiles(cell, nvcc, tmp_path) -> None:
    _compile_only("cuda", nvcc, cell, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cell", FAST_PARAMS)
def test_hip_k_driver_compiles(cell, hipcc, tmp_path) -> None:
    _compile_only("hip", hipcc, cell, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the rule, the driver's shape, and the driver interpreted
# ---------------------------------------------------------------------------


def test_the_fast_cells_keep_to_l_le_2():
    """Every fast cell's plan is L <= 2 (the slow ones are not), the builder
    refuses a class above it, and the guard refuses a Hermite pair above it."""
    for cell in FAST:
        P = _bare_plan(cell)
        assert P.classes and all(c.L <= L_MAX_RUN for c in P), cell.id
        if not cell.truncated:
            _, basis, _ = _case(cell.case)
            assert all(s.l == 0 for s in basis.shells), cell.id
    for cell in SLOW:
        assert max(c.L for c in _bare_plan(cell)) > L_MAX_RUN, cell.id
    with pytest.raises(AssertionError, match="L <= 2 rule"):
        _build(_bare_plan(SLOW[0]).classes[-1].spec)
    with pytest.raises(AssertionError, match="L <= 2 rule"):
        _guarded((1, 0, 0), (1, 1, 0), (0.0, 0.0, 0.0), 1.0)
    # p shells on every slot, and split classes, reach the card
    trunc = _bare_plan(FAST[-1])
    for slot in range(4):
        assert any(c.am[slot] == 1 for c in trunc)
    assert any(c.n_batches == 3 for c in trunc)


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cell", FAST_PARAMS)
def test_the_k_driver_is_the_plans_shape(backend, cell, tmp_path):
    """Emitted, not compiled: one launch per (class, batch kernel, b shell),
    each over a contiguous run of one b shell's kets still sorted by I_bd;
    one block per (bra class, b); every kernel adds into K; the staged input
    is the plan's size."""
    _, basis, D = _case(cell.case)
    plan = _plan(cell)
    drv = k_driver(plan, basis, backend, tmp_path)
    kets = dict(drv.ket)
    for kls, recs in drv.ket:
        assert sorted(recs, key=lambda r: (r.i, r.j, r.prim)) == sorted(
            next(c.kets for c in plan if c.am[2:] == kls),
            key=lambda r: (r.i, r.j, r.prim))
    want = sum(len(c.kernels) * len({r.i for r in c.kets}) for c in plan)
    assert len(drv.launches) == want
    for x in drv.launches:
        seg = kets[x.am[2:]][x.start:x.start + x.n_ket]
        assert seg and all(r.i == x.b for r in seg)
        assert all(a.G >= b.G for a, b in zip(seg, seg[1:], strict=False))
        assert drv.blocks[x.block].b == x.b
        assert drv.blocks[x.block].bra == x.am[:2]
        stmts = list(walk(x.kernel.body))
        assert any(isinstance(s, AtomicAdd) for s in stmts)
        assert not any(isinstance(s, Assign) and isinstance(s.dst, Index)
                       and s.dst.array == "K" for s in stmts)
        assert x.kernel.name in drv.source
    assert len({(b.bra, b.b) for b in drv.blocks}) == len(drv.blocks)
    assert k_stage(drv, basis, D).size == drv.n_input
    assert sum(drv.writers.values()) == len(drv.launches)


def _interp_buffer(drv: KDriver, staged: np.ndarray, *, lanes: int
                   ) -> np.ndarray:
    """The generated `main()`, in Python: the staged input sliced exactly as
    `K_DRIVER` slices it (pointer offsets and all), each launch run through
    `interp` with the reference Boys, adding into one zeroed K buffer."""
    tau, maxD = float(staged[0]), float(staged[1])
    h = 2
    bp, bi, kp, ki, dp = [], [], [], [], []
    for _, recs in drv.bra:
        n = len(recs)
        bp.append(staged[h:h + 12 * n])
        bi.append(staged[h + 12 * n:h + 13 * n].astype(np.int32))
        h += 13 * n
    for _, recs in drv.ket:
        n = len(recs)
        kp.append(staged[h:h + 12 * n])
        ki.append(staged[h + 12 * n:h + 13 * n].astype(np.int32))
        h += 13 * n
    for lc, ld in drv.dcd:
        m = drv.dcd_doubles(lc, ld)
        dp.append(staged[h:h + m])
        h += m
    assert h == staged.size
    bslot = {x: i for i, (x, _) in enumerate(drv.bra)}
    kslot = {x: i for i, (x, _) in enumerate(drv.ket)}
    eslot = {x: i for i, x in enumerate(drv.dcd)}
    n_dc = drv.n_sh * drv.n_sh
    buf = np.zeros(drv.n_k)
    for x in drv.launches:
        i, j = bslot[x.am[:2]], kslot[x.am[2:]]
        e = eslot[(x.am[1], x.am[3])]
        n, nk, s, m = len(drv.bra[i][1]), len(drv.ket[j][1]), x.start, x.n_ket
        b, k = bp[i], kp[j]
        blk = drv.blocks[x.block]
        args = dict(
            n_bra=n, n_ket=m, n_dc=n_dc, tau=tau, maxD=maxD,
            P=b[0:3 * n].reshape(n, 3), zeta_ac=b[3 * n:4 * n],
            XPA=b[4 * n:7 * n].reshape(n, 3), XPC=b[7 * n:10 * n].reshape(n, 3),
            s_ac=b[10 * n:11 * n], Ip=b[11 * n:12 * n], dc=bi[i],
            Q=k[3 * s:3 * s + 3 * m].reshape(m, 3),
            eta=k[3 * nk + s:3 * nk + s + m],
            XQB=k[4 * nk + 3 * s:4 * nk + 3 * s + 3 * m].reshape(m, 3),
            XQD=k[7 * nk + 3 * s:7 * nk + 3 * s + 3 * m].reshape(m, 3),
            s_bd=k[10 * nk + s:10 * nk + s + m],
            Iq=k[11 * nk + s:11 * nk + s + m], dd=ki[j][s:s + m],
            Dcd=dp[e].reshape(n_dc, -1),
            K=k_block(drv, buf, x.block).copy())
        grid, block = k_launch_shape(n, lanes)
        got = Interpreter(x.kernel, launch=Launch(grid, block, lanes),
                          boys=boys).run(**args)["K"]
        buf[blk.offset:blk.offset + blk.size] = np.ravel(got)
    return buf


def _sp_basis() -> tuple[Basis, np.ndarray]:
    """Two s shells and a p shell, uncontracted, three centres: ket classes
    with two b shells, launches with two d shells."""
    basis = Basis((Shell(0, (0.0, 0.0, 0.0), (1.1,), (1.0,)),
                   Shell(0, (0.0, 0.9, 0.4), (0.7,), (1.0,)),
                   Shell(1, (0.6, -0.3, 0.8), (0.9,), (1.0,))))
    n = basis.n_ao
    return basis, np.array([[1.0 / (i + j + 1) for j in range(n)]
                            for i in range(n)])


@pytest.mark.parametrize("n_batches", [1, 3])
def test_the_interpreted_driver_is_the_walk_with_p_shells(n_batches, tmp_path):
    """The synthetic s, s, p basis, planned whole and truncated to L <= 2:
    the driver's layout, interpreted, scattered, against the walk."""
    basis, D = _sp_basis()
    P = k_launches(basis, D, bound=_stub, key=schwarz, tau=0.0,
                   n_batches=n_batches, strict=False)
    P = replace(P, classes=tuple(
        replace(c, kernels=_build(c.spec)) for c in P if c.L <= L_MAX_RUN))
    assert {c.L for c in P} == {0, 1, 2}
    drv = k_driver(P, basis, "cuda", tmp_path)
    assert any(len({r.i for r in recs}) > 1 for _, recs in drv.ket)
    assert any(len({r.j for r in drv.ket[[k for k, _ in drv.ket].index(
        x.am[2:])][1][x.start:x.start + x.n_ket]}) > 1 for x in drv.launches)
    staged = k_stage(drv, basis, D)
    for lanes in (1, 4):
        K = k_scatter(drv, basis, _interp_buffer(drv, staged, lanes=lanes))
        Kw = np.array(walk_k(P, basis, D, pq=_guarded).K)
        big = float(np.max(np.abs(Kw)))
        assert big > 0.0
        assert float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * big, lanes


def test_the_interpreted_driver_is_the_walk_on_h2_screened(tmp_path):
    """H2/STO-3G at tau = 1e-3 (the screen fires): the driver interpreted
    against the walk over the same plan."""
    cell = FAST[1]
    assert cell.tau > 0.0
    _, basis, D = _case(cell.case)
    plan = _plan(cell)
    drv = k_driver(plan, basis, "hip", tmp_path)
    K = k_scatter(drv, basis, _interp_buffer(drv, k_stage(drv, basis, D),
                                             lanes=1))
    w = walk_k(plan, basis, D, pq=_guarded)
    assert w.n_skipped > 0
    Kw = np.array(w.K)
    assert float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * float(
        np.max(np.abs(Kw)))


def test_the_host_blocks_are_the_walk_and_bound_the_partials(tmp_path):
    """`A` bounds every partial (|X| <= A record by record, A > 0), and the
    record-level host twin scattered is the walk over the same plan -- so the
    device buffer's record-level check and its `K_mn` check read the buffer
    the same way."""
    cell = FAST[5]
    _, basis, D = _case(cell.case)
    plan = _plan(cell)
    drv = k_driver(plan, basis, "cuda", tmp_path)
    X, A = _host_blocks(drv, basis, D, cell.pq)
    assert np.all(np.abs(X) <= A) and np.all(A > 0.0)
    K = k_scatter(drv, basis, X)
    w = walk_k(plan, basis, D, pq=cell.pq)
    Kw = np.array(w.K)
    assert w.n_skipped > 0
    assert float(np.max(np.abs(K - Kw))) <= 1e-14 * float(np.max(np.abs(Kw)))
    assert math.isfinite(float(np.sum(A)))
