"""The atomic-free F write-back (#358): slot buffers, then a scatter kernel.

Held here, the issue's acceptance in order (the tail fraction is a card's,
and a clock: measure/measure_f_scatter_tail.py, `devtools/measure.sh
scatter-tail`, not a test):

1. THE LEVER IS A MODULE ON THE SEAMS: one `levers.VOCABULARY` entry legal
   on `"f"` alone, one `f_levers.REGISTRY` line; it contains block-fab's
   reduction (#351) and moves the two flushes and the params, and the pair
   `block-fab` + `scatter` (the same kernel under a second key) and a lever
   that moved either flush are REFUSED. The F kernel holds NO atomic, and
   the scatter kernel (`kernel_f_scatter`, one unit for every class) none.
2. THE FOOTPRINT, DERIVED AND ASSERTED. `launch_f.launch_footprint`
   declares each launch's slot count and the AO footprint the scatter
   writes, from the plan and the basis; `f_levers.scatter`'s layout and
   `kernel_f_scatter.scatter_map` derive them again from the class and the
   packed AO offsets; the two are held equal on every launch of H2O's
   three plans (STO-3G's in the fast tier; 6-31G's and cc-pVDZ's, 1,200
   launches of host arithmetic, `slow`), the map's elements under the
   bound and AT it for an off-diagonal launch of distinct shells. A buffer
   TOO SMALL is refused -- by the interpreter at launch (`W`'s declared
   extent), by the map -- and the reason it is slots and not an AO tile is
   shown: two kets of one launch share a shell, so an AO-indexed tile would
   have two writers.
3. PARITY, interpreted, on EVERY F class with L <= 4 (milestone 13's cap;
   the issue asks L <= 2), off the diagonal and on it, at one warp and two:
   against block-fab's atomic path (H7) by the DERIVED re-association bound
   -- each slot holds the value block-fab adds atomically, so the two `F`s
   are sum trees over the same addends and `|F_scatter - F_H7| <= 2
   gamma_{n-1} SUM |a_i|` per element over H7's `n` addends -- and against
   `ao.fock`'s sweep at the literal kernel's own `TOL`. One bra by one ket
   of distinct shells is BITWISE the literal kernel.
4. THE WALK (`walk_f`, #286): H2O/STO-3G and H2O/6-31G (every class L <= 4)
   built through `scatter` streams -- the scatter after each F launch --
   against the sweep and libcint at `tau = 0` (both `slow`: 6 s and 26 s
   interpreted), and against the literal walk at `2 TOL` with the same fates
   when the screens fire (H2O/STO-3G in the fast tier, the lever's whole-
   molecule check there; 6-31G `slow`); with `ket-break` too, the two
   levers composed.

The goldens (`test/golden/{cuda,hip}/f_scatter/`) pin `(ps|ss)` and
`(pp|pp)` with the lever and the scatter unit. Fast tier: no compiler, no
card, no PySCF; any test over about 3 s is `slow`, so the fast tier keeps
the per-class parity (part 3) and STO-3G's screened walk.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_fock_build as tfb
import test_kernel_f as tk
from test_f_lever_block_fab import _gamma, _Recorder

from felizjueves import f_levers, launch_f, levers, walk_f
from felizjueves import kernel_f as KF
from felizjueves import kernel_f_scatter as FS
from felizjueves.ao.shell import Basis
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import block_fab, scatter
from felizjueves.f_levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.hgp import vrr
from felizjueves.ir import AtomicAdd, Barrier, Index, Kernel, Lit, walk
from felizjueves.ir.interp import InterpError
from felizjueves.launch_f import f_launches, launch_footprint
from felizjueves.spec import Operator, Spec, Target

NAME = scatter.NAME
L4 = [c for c in vrr.CLASSES if sum(c) <= 4]
THREADS = (32, 64)  # one warp, two (32 lanes, as tk.fused)
H2O = ("h2o_sto3g", "h2o_631g")  # every class L <= 4
SLOW = pytest.mark.slow


def lever(cls, extra: tuple[str, ...] = ()) -> Kernel:
    return KF.build_f_kernel(KF.f_class_spec(*cls, tuple(sorted(
        (NAME, *extra)))))


# ---------------------------------------------------------------------------
# 1. The lever is a module on the seams; no atomics anywhere
# ---------------------------------------------------------------------------


def test_the_lever_is_one_vocabulary_entry_and_one_registry_line() -> None:
    lv = levers.VOCABULARY[NAME]
    assert lv.issue == 358 and lv.legal == {"f"}
    for slot in ("j", "k"):
        assert lv.refusals[slot] != levers.DEFAULT_REFUSALS[slot]
    assert f_levers.REGISTRY[NAME] is scatter.apply
    hooks = f_levers.hooks_for(KF.f_class_spec(1, 1, 0, 0, (NAME,)))
    moved = {f.name for f in dataclasses.fields(hooks)
             if not hooks.is_default(f.name)}
    assert moved == {"shared_decl", "flush_bra", "flush_ket", "params"}
    # the two restated names are the lever's
    assert launch_f.SCATTER == walk_f.SCATTER == NAME
    with pytest.raises(ValueError):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, (NAME,))


def test_block_fab_beside_scatter_is_refused_as_redundant() -> None:
    with pytest.raises(ValueError, match="second key"):
        KF.build_f_kernel(KF.f_class_spec(0, 0, 0, 1, (block_fab.NAME, NAME)))


def test_a_lever_that_moved_a_flush_is_refused() -> None:
    spec = KF.f_class_spec(0, 0, 0, 1, (NAME,))
    for hook in ("flush_ket", "flush_bra"):
        other = dataclasses.replace(DEFAULT_HOOKS, **{hook: lambda ctx: ()})
        with pytest.raises(ValueError, match="another lever moved"):
            scatter.apply(other, spec)
    assert isinstance(scatter.apply(DEFAULT_HOOKS, spec), FHooks)


@pytest.mark.parametrize("cls", L4, ids=str)
def test_the_f_kernel_holds_no_atomic_and_writes_only_w(cls) -> None:
    lit, lev = KF.build_f_kernel(cls), lever(cls)
    stmts = list(walk(lev.body))
    assert not any(isinstance(s, AtomicAdd) for s in stmts)
    assert sum(isinstance(s, Barrier) for s in stmts) == 1  # block-fab's
    names = [p.name for p in lev.params]
    assert names == [scatter.SLOTS if p.name == KF.FOCK else p.name
                     for p in lit.params]
    assert not any(isinstance(e, Index) and e.array == KF.FOCK
                   for s in stmts for e in _indices(s))
    w = next(p for p in lev.params if p.name == scatter.SLOTS)
    assert len(w.shape) == 1  # the declared extent the interpreter checks
    # the stores: per thread, n_ab f_ab columns (one loop) and S5 ket slots
    stores = [s for s in stmts if getattr(getattr(s, "dst", None), "array",
                                          None) == scatter.SLOTS]
    assert len(stores) == 1 + scatter.ket_width(cls)


def _indices(s):
    """Every `Index` an expression of `s` reads (shallow over the IR)."""
    from felizjueves.ir import _exprs_of, _subexprs
    for e in _exprs_of(s):
        if e is not None:
            yield from (x for x in _subexprs(e) if isinstance(x, Index))


def test_the_scatter_kernel_is_one_class_agnostic_unit_with_no_atomic() -> None:
    k = FS.build_scatter_kernel()
    assert k.name == FS.NAME == "f_scatter"
    assert tuple(p.name for p in k.params) == FS.PARAMS
    stmts = list(walk(k.body))
    assert not any(isinstance(s, (AtomicAdd, Barrier)) for s in stmts)
    lits = {e.value for s in stmts for e in _lits(s)}
    assert lits <= {0, 1, 2}  # no class, AO count or block in it
    assert FS.build_scatter_kernel() == k


def _lits(s):
    from felizjueves.ir import _exprs_of, _subexprs
    for e in _exprs_of(s):
        if e is not None:
            yield from (x for x in _subexprs(e) if isinstance(x, Lit))


# ---------------------------------------------------------------------------
# 2. The footprint: derived twice, asserted; a buffer too small refused
# ---------------------------------------------------------------------------


def _plans(names=(*H2O, "h2o_ccpvdz")):
    for name in names:
        basis, D, _ = tfb.molecule(name)
        for tau, n in ((0.0, 1), (0.1, 4)):
            yield name, basis, f_launches(basis, D, bound=tfb.BOUND, tau=tau,
                                          n=n, scatter=True)


# STO-3G's 39 launches in the fast tier; 6-31G's 354 and cc-pVDZ's 845 (its
# d classes too: host arithmetic, nothing runs) are 7 s, so `slow`
FOOTPRINT_MOLECULES = ["h2o_sto3g", pytest.param("h2o_631g", marks=SLOW),
                       pytest.param("h2o_ccpvdz", marks=SLOW)]


@pytest.mark.parametrize("name", FOOTPRINT_MOLECULES)
def test_the_declared_footprint_is_the_kernels_on_every_launch(name) -> None:
    """Every launch of one of H2O's three plans: `launch_f`'s declaration,
    derived from the plan and the basis, against the lever's layout and the
    map."""
    n = 0
    for _, basis, plan in _plans((name,)):
        for L in plan.launches:
            fp = launch_footprint(plan, L, basis)
            cls = tuple(L.spec.am)
            bras = [walk_f.pair_record(basis, p.a, p.b, G=p.G, prims=p.prims)
                    for p in plan.batches[L.bra].pairs]
            kets = [walk_f.pair_record(basis, p.a, p.b, G=p.G, prims=p.prims)
                    for p in plan.batches[L.ket].pairs]
            pb, pk = KF.pack(bras), KF.pack(kets)
            assert (fp.n_bra, fp.n_ket) == (pb.n, pk.n)
            assert fp.n_slot == scatter.n_slot(cls, pb.n, pk.n)
            assert fp.n_elem_bound == FS.footprint_bound(cls, pb.ao, pk.ao)
            m = FS.scatter_map(cls, pk.n, pb.ao, pk.ao)
            assert m.n_slot == fp.n_slot
            assert sorted(m.src) == list(range(m.n_slot))  # each slot once
            el = list(zip(m.el[::2], m.el[1::2], strict=True))
            assert el == sorted(set(el)) and m.n_el == len(el) <= fp.n_elem_bound
            assert all(r in fp.bra_aos + fp.ket_aos for r, _ in el)
            assert m.ptr[0] == 0 and m.ptr[-1] == m.n_slot
            assert all(a < b for a, b in zip(m.ptr, m.ptr[1:], strict=False))
            n += 1
    assert n > 30


@pytest.mark.parametrize("cls", L4, ids=str)
def test_the_bound_is_tight_off_the_diagonal_of_distinct_shells(cls) -> None:
    basis, bras, kets = _launch(cls, diag=False)
    pb, pk = KF.pack(bras), KF.pack(kets)
    m = FS.scatter_map(cls, pk.n, pb.ao, pk.ao)
    assert m.n_el == FS.footprint_bound(cls, pb.ao, pk.ao)


def test_a_buffer_too_small_is_refused_never_a_dropped_term() -> None:
    cls = (0, 1, 0, 1)
    basis, bras, kets = _launch(cls, diag=False)
    args, ns = _args(cls, basis, tk._density(basis.n_ao), bras, kets, 0)
    k = lever(cls)
    for short in (ns - 1, ns - scatter.ket_width(cls)):
        bad = dict(args, **{scatter.SLOTS: np.zeros(short)})
        with pytest.raises(InterpError, match="declared extents"):
            _interp(k, len(bras), 32, bad)
    pb, pk = KF.pack(bras), KF.pack(kets)
    with pytest.raises(ValueError, match="n_ket"):
        FS.scatter_map(cls, pk.n - 1, pb.ao, pk.ao)
    m = FS.scatter_map(cls, pk.n, pb.ao, pk.ao)
    sargs = m.args(np.zeros(ns - 1), np.zeros(basis.n_ao ** 2), basis.n_ao)
    with pytest.raises(InterpError, match="declared extents"):
        _interp(FS.build_scatter_kernel(), 1, 32, sargs)
    with pytest.raises(ValueError, match="I32"):
        scatter.n_slot((2, 2, 2, 2), 2 ** 16, 2 ** 16)


def test_an_ao_tile_would_have_two_writers_so_the_buffer_is_slots() -> None:
    """The reason the buffer is per quartet: in H2O/cc-pVDZ's plan a ket
    batch holds two pairs that share a shell, so two threads of one block
    deposit `f_ac` (or `f_ad`, `f_bc`, `f_bd`) into the same AO element."""
    basis, D, _ = tfb.molecule("h2o_ccpvdz")
    plan = f_launches(basis, D, bound=tfb.BOUND, tau=0.0, n=1)
    shared = 0
    for L in plan.launches:
        kets = [(p.a, p.b) for p in plan.batches[L.ket].pairs]
        for x in range(len(kets)):
            for y in range(x + 1, len(kets)):
                shared += bool(set(kets[x]) & set(kets[y]))
    assert shared > 0


# ---------------------------------------------------------------------------
# 3. Parity, interpreted
# ---------------------------------------------------------------------------


def _launch(cls, diag: bool):
    """test_f_lever_block_fab.py's launches: two bras by two kets off the
    diagonal, or a diagonal batch of three same-`l` pairs."""
    if diag:
        l = cls[0]  # noqa: E741
        basis = Basis(tuple(tk._same_l_shells(l, 2)))
        recs = [tk.records(basis, *p) for p in [(0, 0), (0, 1), (1, 1)]]
        return basis, recs, recs
    geoms = [tk.CENTRES, tk._moved((0.3, -0.2, 0.1))]
    basis = Basis(tuple(s for g in geoms for s in tk._shells(cls, g)))
    return (basis, [tk.records(basis, 0, 1), tk.records(basis, 4, 5)],
            [tk.records(basis, 2, 3), tk.records(basis, 6, 7)])


def _args(cls, basis, D, bras, kets, diag):
    pb, pk = KF.pack(bras), KF.pack(kets)
    n_ao = basis.n_ao
    Dsh = KF.density_blocks(D, basis.offsets, [s.n_cart for s in basis.shells])
    ns = scatter.n_slot(cls, pb.n, pk.n)
    return {
        "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree, "k_cd": pk.degree,
        "n_sh": len(basis.shells), "n_ao": n_ao, "diag": int(diag),
        "tau": 0.0, "bra_prim": pb.prim, "AB": pb.sep, "G_ab": pb.G,
        "bra_sh": pb.sh, "bra_ao": pb.ao, "ket_prim": pk.prim, "CD": pk.sep,
        "G_cd": pk.G, "ket_sh": pk.sh, "ket_ao": pk.ao, "Dsh": Dsh,
        "D": tuple(np.asarray(D, dtype=float).ravel()),
        KF.FOCK: (0.0,) * (n_ao * n_ao), scatter.SLOTS: (0.0,) * ns}, ns


def _interp(kernel, grid, threads, args):
    return tfb.run(kernel, grid, (threads, 1),
                   {p.name: args[p.name] for p in kernel.params})


def run_scatter(cls, basis, D, bras, kets, *, diag=0, threads=32):
    """The lever's F launch, then the scatter over its map: `F` (not
    closed), as the host driver runs the two."""
    args, _ = _args(cls, basis, D, bras, kets, diag)
    W = _interp(lever(cls), len(bras), threads, args)[scatter.SLOTS]
    pb, pk = KF.pack(bras), KF.pack(kets)
    m = FS.scatter_map(cls, pk.n, pb.ao, pk.ao)
    grid, (t, _) = FS.launch_shape(m.n_el, threads)
    F = _interp(FS.build_scatter_kernel(), grid, t,
                m.args(tuple(W), args[KF.FOCK], basis.n_ao))[KF.FOCK]
    return np.asarray(F, dtype=float).reshape(basis.n_ao, basis.n_ao)


def run_block_fab(cls, basis, D, bras, kets, *, diag=0, threads=32):
    args, _ = _args(cls, basis, D, bras, kets, diag)
    k = KF.build_f_kernel(KF.f_class_spec(*cls, (block_fab.NAME,)))
    F = _interp(k, len(bras), threads, args)[KF.FOCK]
    return np.asarray(F, dtype=float).reshape(basis.n_ao, basis.n_ao)


PARITY = [pytest.param(c, t, d, id=f"{''.join(map(str, c))}-t{t}"
                       + ("-diag" if d else ""))
          for c in L4 for t in THREADS for d in (False, True)
          if not d or len(set(c)) == 1]


@pytest.mark.parametrize(("cls", "threads", "diag"), PARITY)
def test_parity_with_block_fabs_atomic_path_and_the_sweep(
        cls, threads, diag, monkeypatch) -> None:
    basis, bras, kets = _launch(cls, diag)
    D = tk._density(basis.n_ao)
    rec = _Recorder(monkeypatch, basis.n_ao)
    F0 = run_block_fab(cls, basis, D, bras, kets, diag=int(diag),
                       threads=threads)
    monkeypatch.undo()
    rec1 = _Recorder(monkeypatch, basis.n_ao)
    F1 = run_scatter(cls, basis, D, bras, kets, diag=int(diag),
                     threads=threads)
    monkeypatch.undo()
    assert int(rec1.count.sum()) == 0  # not one atomic on F
    assert np.max(np.abs(F0)) > 0
    # against H7's atomic path, by the derived re-association bound
    bound = 2 * np.vectorize(_gamma)(np.maximum(rec.count - 1, 0)) * rec.absum
    assert np.all(np.abs(F1 - F0) <= bound), float(np.max(np.abs(F1 - F0)
                                                          - bound))
    # against the sweep, at the literal kernel's own TOL
    want, scale = tk.sweep(basis, D, tk._members([r.sh for r in bras],
                                                  [r.sh for r in kets]))
    assert tk._worst(np.array(KF.close(F1)), want, scale) <= tk.TOL


@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (1, 1, 1, 1)], ids=str)
def test_one_bra_one_ket_is_bitwise_the_literal_kernel(cls) -> None:
    basis = Basis(tuple(tk._shells(cls)))
    bras, kets = [tk.records(basis, 0, 1)], [tk.records(basis, 2, 3)]
    D = tk._density(basis.n_ao)
    F0, _, _ = tk.fused(cls, basis, D, bras, kets, threads=32)
    F1 = run_scatter(cls, basis, D, bras, kets, threads=64)
    assert np.max(np.abs(F0)) > 0
    assert np.array_equal(F0, F1)


# ---------------------------------------------------------------------------
# 4. The walk: H2O through scatter streams
# ---------------------------------------------------------------------------


def test_the_default_stream_is_unchanged_and_scatter_only_names_its_lever(
        ) -> None:
    basis, D, _ = tfb.molecule("h2o_631g")
    for kb in (False, True):
        base = f_launches(basis, D, bound=tfb.BOUND, tau=0.1, n=4,
                          ket_break=kb)
        assert base == f_launches(basis, D, bound=tfb.BOUND, tau=0.1, n=4,
                                  ket_break=kb, scatter=False)
        sc = f_launches(basis, D, bound=tfb.BOUND, tau=0.1, n=4,
                        ket_break=kb, scatter=True)
        assert sc.batches == base.batches and sc.screened == base.screened
        for a, b in zip(base.launches, sc.launches, strict=True):
            assert (a.bra, a.ket, a.spec.am) == (b.bra, b.ket, b.spec.am)
            assert b.spec.levers == tuple(sorted((*a.spec.levers, NAME)))
    with pytest.raises(ValueError, match="scatter"):
        f_launches(basis, D, bound=tfb.BOUND, tau=0.0, n=1, scatter=1)


def _walk(name, tau, n, *, scatter_on, ket_break=False):
    basis, D, _ = tfb.molecule(name)
    plan = f_launches(basis, D, bound=tfb.BOUND, tau=tau, n=n,
                      scatter=scatter_on, ket_break=ket_break)
    return walk_f.walk_f(plan, basis, D, run=tfb.run, threads=32), plan


@pytest.mark.slow  # 6 s (STO-3G) and 26 s (6-31G) interpreted
@pytest.mark.parametrize("name", H2O)
def test_the_scatter_walk_is_the_sweep_and_libcint(name) -> None:
    _, _, libcint = tfb.molecule(name)
    w, plan = _walk(name, 0.0, 1, scatter_on=True)
    assert all(NAME in L.spec.levers for L in plan.launches)
    assert max(sum(L.spec.am) for L in plan.launches) <= 4
    assert not w.skipped
    F = np.array(w.F)
    want, scale = tfb.sweep(name), tfb.sweep(name, block="scale")
    assert tfb._worst(F, want, scale) <= tfb.TOL
    assert tfb._worst(F, libcint, scale) <= tfb.LIBCINT_TOL


@pytest.mark.parametrize("ket_break", [False, True], ids=["", "ket-break"])
@pytest.mark.parametrize(("name", "tau", "n"),
                         [("h2o_sto3g", 0.1, 4),
                          pytest.param("h2o_631g", 0.1, 4,
                                       marks=pytest.mark.slow)])
def test_a_screened_scatter_walk_is_the_literal_walk(name, tau, n,
                                                     ket_break) -> None:
    w1, _ = _walk(name, tau, n, scatter_on=True, ket_break=ket_break)
    w0, _ = _walk(name, tau, n, scatter_on=False, ket_break=ket_break)
    for fate in ("launched", "kernel_screened", "prelaunch_screened",
                 "deleted", "kernel_broken"):
        assert getattr(w1, fate) == getattr(w0, fate), fate
    assert w0.kernel_screened  # the screens fire
    scale = tfb.sweep(name, block="scale")
    assert tfb._worst(np.array(w1.F), np.array(w0.F), scale) <= 2 * tfb.TOL


# ---------------------------------------------------------------------------
# The goldens: (ps|ss) and (pp|pp) with the lever, and the scatter unit
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.kernel_f_scatter import build_scatter_kernel; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   ks = [build_f_kernel(f_class_spec(*c, ('scatter',))) for c in \
#   ((0,0,0,1), (1,1,1,1))] + [build_scatter_kernel()]; \
#   [(Path(f'test/golden/cuda/f_scatter/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)), Path(f'test/golden/hip/f_scatter/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))) for k in ks]"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLASSES = ((0, 0, 0, 1), (1, 1, 1, 1))
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_scatter_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    where = GOLDEN / backend / "f_scatter"
    kernels = [lever(c) for c in GOLDEN_CLASSES] + [FS.build_scatter_kernel()]
    assert sorted(p.name for p in where.iterdir()) == sorted(
        f"{k.name}.{ext}" for k in kernels)
    for kern in kernels:
        text = emit(kern, arch)
        assert text == (where / f"{kern.name}.{ext}").read_text()
        assert "atomicAdd" not in text
