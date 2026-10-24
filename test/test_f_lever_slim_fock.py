"""The per-bra-element Fock-buffer split, `eri/fock/levers/slim_fock.py` (#360),
interpreted.

The lever gives each of #356's slices `(a, b)` a reduced buffer set -- `a`'s
rows of `f_ac` / `f_ad`, `b`'s of `f_bc` / `f_bd`, zeroed and flushed per
slice -- streams `f_cd` by one `AtomicAdd` per ERI, and leaves `f_ab` as the
literal kernel carries it. What is asserted, the issue's acceptance in order:

1. THE DECLARATION: `slim-fock` is legal on `Target.F` only, refused on the
   other targets with its own reasons, registered in `f_levers.REGISTRY`;
   it is a COMPOSITION on `batched-vrr+sliced-hrr`: refused without
   `sliced-hrr` (with its reason, and by the CLI with exit 2), refused when
   applied before the slices, after a lever that moved `contracted`, or
   after one that moved a hook it replaces; it composes with #351's
   `block-fab`, #352's `ket-break` and #353's `struct-reads`, which move
   other hooks. The literal kernel and #356's composition are untouched
   (their digests and goldens are other files' pins).
2. THE DECLARED FUNCTIONS OF THE CLASS, pinned for all 21 classes (no
   build): the per-slice set `1 + 2 n_c + 2 n_d`, the live Fock buffers
   `n_a n_b + 2 n_c + 2 n_d` against `kernel_f.six_block_size`, and the
   ket-dependent atomics per worked quartet, literal and lever.
3. THE KERNEL'S SHAPE: each slice declares exactly its four rows and flushes
   exactly them; no ket-dependent buffer is declared outside a slice and
   no `f_cd` buffer at all; the `AtomicAdd` statements are the streamed
   `n_eri`, the rows per slice and the bra's `n_a n_b`.
4. PARITY, interpreted, on EVERY F class with L <= 4 (milestone 13's cap;
   the issue asks L <= 2; above it `slow`, not run under the cap): the tap
   `g` is the literal kernel's BIT FOR BIT; `F` against the literal kernel
   (and so #356's, which equals it bitwise) by the DERIVED re-association
   bound `|F_lever - F_literal| <= 2 gamma_{n-1} SUM |p_i|` per element
   over its `n` products, recorded by a witness kernel that atomically adds
   every product as it is formed; `F` against `eri.fock.fock`'s sweep at the
   literal kernel's own `TOL`, and every degeneracy case within the cap.
5. THE ATOMIC TRAFFIC AS A COUNT: the interpreter's atomics on `F`, per
   worked quartet, are `ket_atomics(cls, lever=)` in both kernels.
6. THE GOLDENS: `(sp|sp)` with the three levers, one unit per backend.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_kernel_f as tk
from test_f_lever_block_fab import _gamma, _Recorder
from tiers import split_by_l

from felizjueves import levers
from felizjueves.ao.shell import Basis
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock import levers as f_levers
from felizjueves.eri.fock.kernel_f import build_f_kernel, f_class_spec
from felizjueves.eri.fock.levers import batched_vrr as bv
from felizjueves.eri.fock.levers import block_fab, ket_break, seams, struct_reads
from felizjueves.eri.fock.levers import sliced_hrr as sh
from felizjueves.eri.fock.levers import slim_fock as sf
from felizjueves.eri.fock.levers.seams import BLOCKS, DEFAULT_HOOKS, FContext, FHooks
from felizjueves.hgp import hrr, vrr
from felizjueves.ir import AtomicAdd, Call, Decl, Fn, For, Index, Var, walk
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.spec import Operator, Spec, Target

NAME = sf.NAME
STACK = (bv.NAME, sh.NAME, NAME)
CAP = 4  # milestone 13: every test at L <= 4

LADDER = split_by_l(vrr.CLASSES, sum, ids=lambda c: "".join(map(str, c)))
CAPPED = [c for c in vrr.CLASSES if sum(c) <= CAP]


def lever_spec(cls, extra=()) -> Spec:
    return f_class_spec(*cls, tuple(sorted((*STACK, *extra))))


def lever(cls, eris: bool = False, extra=()) -> KF.Kernel:
    return build_f_kernel(lever_spec(cls, extra), eris=eris)


def _moved(hooks: FHooks) -> set[str]:
    return {f.name for f in dataclasses.fields(FHooks)
            if not hooks.is_default(f.name)}


# ---------------------------------------------------------------------------
# 1. The declaration: a composition on #356
# ---------------------------------------------------------------------------


def test_slim_fock_is_declared_legal_on_f_only_with_its_own_reasons() -> None:
    lv = levers.VOCABULARY[NAME]
    assert (lv.name, lv.issue, lv.legal) == (NAME, 360, frozenset({"f"}))
    assert "Fock digestion" in lv.refusals["j"]
    assert "Fock digestion" in lv.refusals["k"]
    assert "sliced-hrr" in lv.what
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lv.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    for target in (Target.J, Target.K):
        with pytest.raises(ValueError, match="Fock digestion"):
            Spec(Operator.COULOMB, 0, (1, 0, 1, 0), target, (NAME,))


def test_slim_fock_is_registered_keyed_and_composes_on_the_slices() -> None:
    assert f_levers.REGISTRY[NAME] is sf.apply
    spec = lever_spec((0, 1, 0, 1))
    assert spec.key() == ("coulomb/d0/(0,1|0,1)/b1/f/"
                          "batched-vrr+sliced-hrr+slim-fock")
    assert spec.kernel_body_levers() == STACK  # key order: #356 before it
    hooks = f_levers.hooks_for(spec)
    assert _moved(hooks) == {"boundary_init", "nest", "accumulate",
                             "boundary_read", "contracted", "zero_buffers",
                             "predigest_eri", "flush_ket"}
    kern = build_f_kernel(spec)
    assert kern.name == "f_0101"
    notes = [s.prov for s in walk(kern.body) if hasattr(s, "prov")]
    assert all(levers.provenance(n) in notes for n in STACK)


@pytest.mark.parametrize("alone", [(NAME,), (bv.NAME, NAME)],
                         ids=["alone", "without-sliced-hrr"])
def test_slim_fock_without_the_slices_is_refused_with_its_reason(
        alone, tmp_path) -> None:
    spec = f_class_spec(0, 1, 0, 1, alone)  # legal on F: the key admits it
    with pytest.raises(ValueError, match="requires sliced-hrr") as err:
        build_f_kernel(spec)
    assert sf.REFUSAL in str(err.value)
    assert emit_cli.main(["--backend", "cuda", "--out", str(tmp_path),
                          "--klever-slim-fock-fock-l", "1"]) == 2
    assert not list(tmp_path.iterdir())


def test_an_incoherent_order_or_a_moved_hook_is_refused() -> None:
    spec = lever_spec((0, 1, 0, 1))
    staged = bv.apply(DEFAULT_HOOKS, spec)
    with pytest.raises(ValueError, match="applied before sliced-hrr"):
        sf.apply(staged, spec)
    sliced = sh.apply(staged, spec)
    other = dataclasses.replace(sliced, contracted=sh.contracted_with(
        sh.slice_body))  # another lever's contracted, even an equal one
    with pytest.raises(ValueError, match="replace silently"):
        sf.apply(other, spec)
    for hook in ("zero_buffers", "predigest_eri", "flush_ket"):
        moved = dataclasses.replace(sliced, **{hook: lambda *a: ()})
        with pytest.raises(ValueError, match="dropped silently"):
            sf.apply(moved, spec)
    assert _moved(sf.apply(sliced, spec)) == _moved(
        f_levers.hooks_for(spec))


@pytest.mark.parametrize("other", [block_fab.NAME, ket_break.NAME,
                                   struct_reads.NAME])
def test_it_composes_with_the_levers_that_move_other_hooks(other) -> None:
    spec = lever_spec((1, 1, 1, 1), (other,))
    alone = f_levers.hooks_for(f_class_spec(1, 1, 1, 1, (other,)))
    hooks = f_levers.hooks_for(spec)
    assert _moved(hooks) == _moved(f_levers.hooks_for(lever_spec(
        (1, 1, 1, 1)))) | _moved(alone)
    for h in _moved(alone):
        assert getattr(hooks, h) is not getattr(DEFAULT_HOOKS, h)
    build_f_kernel(spec)  # builds


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_literal_and_the_sliced_kernels_are_untouched(cls) -> None:
    for spec in (f_class_spec(*cls), f_class_spec(*cls, (bv.NAME, sh.NAME))):
        hooks = f_levers.hooks_for(spec)
        for h in ("zero_buffers", "predigest_eri", "flush_ket"):
            assert hooks.is_default(h)


# ---------------------------------------------------------------------------
# 2. The declared functions of the class
# ---------------------------------------------------------------------------

# class -> (slice_set, live_buffers, six_block_size, ket atomics per worked
# quartet literal, lever). Pinned so a change to the cut is a visible diff.
BUFFER_TABLE = {
    (0, 0, 0, 0): (5, 5, 6, 5, 5),
    (0, 0, 0, 1): (9, 9, 12, 11, 11),
    (0, 0, 0, 2): (15, 15, 21, 20, 20),
    (0, 0, 1, 1): (13, 13, 22, 21, 21),
    (0, 0, 1, 2): (19, 19, 37, 36, 36),
    (0, 0, 2, 2): (25, 25, 61, 60, 60),
    (0, 1, 0, 1): (9, 11, 22, 19, 33),
    (0, 1, 0, 2): (15, 17, 37, 34, 60),
    (0, 1, 1, 1): (13, 15, 36, 33, 63),
    (0, 1, 1, 2): (19, 21, 57, 54, 108),
    (0, 1, 2, 2): (25, 27, 87, 84, 180),
    (0, 2, 0, 2): (15, 20, 61, 55, 120),
    (0, 2, 1, 1): (13, 18, 57, 51, 126),
    (0, 2, 1, 2): (19, 24, 87, 81, 216),
    (0, 2, 2, 2): (25, 30, 126, 120, 360),
    (1, 1, 1, 1): (13, 21, 54, 45, 189),
    (1, 1, 1, 2): (19, 27, 81, 72, 324),
    (1, 1, 2, 2): (25, 33, 117, 108, 540),
    (1, 2, 1, 2): (19, 36, 117, 99, 648),
    (1, 2, 2, 2): (25, 42, 162, 144, 1080),
    (2, 2, 2, 2): (25, 60, 216, 180, 2160),
}


def test_the_buffer_table_covers_the_ladder_and_is_the_definition() -> None:
    assert tuple(BUFFER_TABLE) == vrr.CLASSES
    for cls, (sset, live, six, lit, lev) in BUFFER_TABLE.items():
        na, nb, nc, nd = (len(vrr.cartesians(x)) for x in cls)
        assert sf.slice_set(cls) == sset == 1 + 2 * nc + 2 * nd, cls
        assert sf.reduced_rows(cls) == sset - 1, cls
        assert sf.live_buffers(cls) == live == na * nb + 2 * nc + 2 * nd, cls
        assert sf.six_block(cls) == six == KF.six_block_size(cls), cls
        assert live <= six, cls  # never more buffers than the six blocks
        assert sf.ket_atomics(cls, lever=False) == lit == six - na * nb, cls
        assert sf.ket_atomics(cls, lever=True) == lev == na * nb * (
            nc * nd + 2 * nc + 2 * nd), cls
    # (dd|dd), the predecessor's cell: 216 buffers -> 60, 3.6x; atomics 12x
    assert BUFFER_TABLE[(2, 2, 2, 2)][2] / BUFFER_TABLE[(2, 2, 2, 2)][1] == 3.6


# ---------------------------------------------------------------------------
# 3. The kernel's shape
# ---------------------------------------------------------------------------


def _ket_buffer(name: str) -> bool:
    return name.startswith("f_") and not name.startswith("f_ab_")


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_each_slice_declares_and_flushes_exactly_its_rows(cls) -> None:
    kern = lever(cls)
    na, nb, nc, nd = (len(vrr.cartesians(x)) for x in cls)
    loops = [s for s in walk(kern.body) if isinstance(s, For)
             and s.var == sh.SLICE]
    assert len(loops) == sh.n_slices(cls) == na * nb
    inside = 0
    for i, loop in enumerate(loops):
        a, b = divmod(i, nb)
        body = list(walk(loop.body))
        decls = [s.name for s in body if isinstance(s, Decl)
                 and _ket_buffer(s.name)]
        want = ([f"f_ac_{a}_{j}" for j in range(nc)]
                + [f"f_ad_{a}_{j}" for j in range(nd)]
                + [f"f_bc_{b}_{j}" for j in range(nc)]
                + [f"f_bd_{b}_{j}" for j in range(nd)])
        assert decls == want and len(decls) == sf.reduced_rows(cls)
        flushed = [s.value.name for s in body if isinstance(s, AtomicAdd)
                   and isinstance(s.value, Var)]
        assert flushed == want
        streamed = [s for s in body if isinstance(s, AtomicAdd)
                    and isinstance(s.value, Call) and s.value.fn is Fn.MUL]
        assert len(streamed) == hrr.slices(cls)[i].n_eri == nc * nd
        inside += len(decls)
    every = [s for s in walk(kern.body) if isinstance(s, Decl)
             and _ket_buffer(s.name)]
    assert len(every) == inside  # no ket buffer outside a slice
    assert not any(s.name.startswith("f_cd_") for s in every)
    atomics = sum(isinstance(s, AtomicAdd) for s in walk(kern.body))
    assert atomics == (hrr.build(cls).n_eri + na * nb * sf.reduced_rows(cls)
                       + na * nb)


# ---------------------------------------------------------------------------
# 4. Parity; 5. the atomic count
# ---------------------------------------------------------------------------


def _stream_all(ctx: FContext, t, idx) -> tuple:
    """The witness's predigest: EVERY block's product `AtomicAdd`-ed into
    `F` as it is formed, so the recorder sees each leaf of both sum trees."""
    x = ctx.hooks.eri(ctx, t)
    out = []
    for b in BLOCKS:
        c = ctx.comp(b)
        out.append(AtomicAdd(Index(seams.FOCK, ctx.ao(b, idx[b[0]],
                                                       idx[b[1]])),
                             Call(Fn.MUL, (Var(ctx.dens(c, idx[c[0]],
                                                        idx[c[1]])), x))))
    return tuple(out)


def witness(cls) -> KF.Kernel:
    """The lever's kernel with every product streamed and no buffer flushed:
    the leaves of `F`, one atomic each."""
    spec = lever_spec(cls)
    hooks = dataclasses.replace(
        f_levers.hooks_for(spec), predigest_eri=_stream_all,
        zero_buffers=lambda ctx: (), flush_ket=lambda ctx: (),
        flush_bra=lambda ctx: ())
    return build_f_kernel(spec, hooks=hooks)


def _two_by_two(cls):
    geoms = [tk.CENTRES, tk._moved((0.3, -0.2, 0.1))]
    basis = Basis(tuple(s for g in geoms for s in tk._shells(cls, g)))
    return (basis, [tk.records(basis, 0, 1), tk.records(basis, 4, 5)],
            [tk.records(basis, 2, 3), tk.records(basis, 6, 7)])


def _run(kern, cls, monkeypatch, *, eris=False, record=False, threads=2,
         launch=None):
    basis, bras, kets = launch or _two_by_two(cls)
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    rec = _Recorder(monkeypatch, basis.n_ao) if record else None
    F, g, _ = tk.fused(cls, basis, tk._density(basis.n_ao), bras, kets,
                       threads=threads, eris=eris)
    monkeypatch.undo()
    return F, g, rec


def _bound(leaves: _Recorder) -> np.ndarray:
    return 2 * np.vectorize(_gamma)(np.maximum(leaves.count - 1, 0)) \
        * leaves.absum


@pytest.mark.parametrize("cls", LADDER)
def test_parity_with_the_literal_kernel_and_the_sweep(cls,
                                                      monkeypatch) -> None:
    F0, g0, rec0 = _run(build_f_kernel(cls, eris=True), cls, monkeypatch,
                        eris=True, record=True)
    F1, g1, rec1 = _run(lever(cls, eris=True), cls, monkeypatch, eris=True,
                        record=True)
    _, _, leaves = _run(witness(cls), cls, monkeypatch, record=True)
    assert np.any(g0 != 0) and np.max(np.abs(F0)) > 0
    # 4a. the ERIs are #356's, hence the literal's, bit for bit
    assert np.array_equal(g1, g0)
    # 4b. F against the literal kernel by the derived bound over the leaves
    bound = _bound(leaves)
    assert np.all(np.abs(F1 - F0) <= bound), float(np.max(
        np.abs(F1 - F0) - bound))
    # ... and #356's composition, which is the literal kernel bit for bit
    Fs, gs, _ = _run(build_f_kernel(f_class_spec(*cls, (bv.NAME, sh.NAME)),
                                    eris=True), cls, monkeypatch, eris=True)
    assert np.array_equal(Fs, F0) and np.array_equal(gs, g0)
    # 4c. against the sweep, at the literal kernel's own TOL
    basis, bras, kets = _two_by_two(cls)
    want, scale = tk.sweep(basis, tk._density(basis.n_ao), tk._members(
        [r.sh for r in bras], [r.sh for r in kets]))
    assert tk._worst(np.array(KF.close(F1)), want, scale) <= tk.TOL
    # 5. the atomic traffic: 2 bras x 2 kets x 2 threads, every quartet
    # worked (tau = 0, off the diagonal), each thread flushing f_ab once
    na, nb = (len(vrr.cartesians(x)) for x in cls[:2])
    bra_flush = len(bras) * 2 * na * nb
    n_work = len(bras) * len(kets)
    assert int(rec0.count.sum()) - bra_flush == n_work * sf.ket_atomics(
        cls, lever=False)
    assert int(rec1.count.sum()) - bra_flush == n_work * sf.ket_atomics(
        cls, lever=True)


@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (1, 1, 1, 1)], ids=str)
def test_one_bra_one_ket_is_bitwise(cls, monkeypatch) -> None:
    """One bra by one ket of four distinct shells, one thread working: each
    row element of F gets its one slice's partial, `f_cd`'s each streamed
    product -- for `(ss|ss)` one leaf per element, so the two agree bit for
    bit; above it the bound holds and is the check."""
    basis = Basis(tuple(tk._shells(cls)))
    launch = (basis, [tk.records(basis, 0, 1)], [tk.records(basis, 2, 3)])
    F0, _, _ = _run(build_f_kernel(cls), cls, monkeypatch, launch=launch)
    F1, _, _ = _run(lever(cls), cls, monkeypatch, launch=launch)
    _, _, leaves = _run(witness(cls), cls, monkeypatch, launch=launch,
                        record=True)
    assert np.max(np.abs(F0)) > 0
    if cls == (0, 0, 0, 0):
        assert np.array_equal(F0, F1)
    assert np.all(np.abs(F1 - F0) <= _bound(leaves))


@pytest.mark.parametrize("cls", [(0, 1, 0, 1), (1, 1, 1, 1)], ids=str)
def test_with_block_fab_the_bound_still_holds(cls, monkeypatch) -> None:
    """#351's block reduction of `f_ab` composed on top (one warp of 32
    threads, its launch contract): the same leaves, another sum tree."""
    F0, _, _ = _run(build_f_kernel(cls), cls, monkeypatch, threads=32)
    F1, _, _ = _run(lever(cls, extra=(block_fab.NAME,)), cls, monkeypatch,
                    threads=32)
    _, _, leaves = _run(witness(cls), cls, monkeypatch, threads=32,
                        record=True)
    assert np.max(np.abs(F0)) > 0
    assert np.all(np.abs(F1 - F0) <= _bound(leaves))


def _against_sweep(cls, basis, bra_ix, ket_ix, diag, monkeypatch):
    kern = lever(cls)
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    worst, *_ = tk._check_launch(cls, basis, tk._density(basis.n_ao), bra_ix,
                                 ket_ix, diag=diag)
    return worst


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_one_orbit_equals_its_ordered_members_in_the_sweep(
        cls, monkeypatch) -> None:
    basis = Basis(tuple(tk._shells(cls)))
    worst = _against_sweep(cls, basis, [(0, 1)], [(2, 3)], 0, monkeypatch)
    assert worst <= tk.TOL, worst


@pytest.mark.parametrize("case", [c for c in tk.COINCIDENCE
                                  if sum(tk.COINCIDENCE[c][0]) <= CAP])
def test_every_degeneracy_case_within_the_cap_equals_the_sweep(
        case, monkeypatch) -> None:
    cls, n, bra_ix, ket_ix, diag = tk.COINCIDENCE[case]
    basis = Basis(tuple(tk._same_l_shells(cls[0], n)))
    worst = _against_sweep(cls, basis, bra_ix, ket_ix, diag, monkeypatch)
    assert worst <= tk.TOL, (case, worst)


# ---------------------------------------------------------------------------
# 6. The goldens: (sp|sp) with the three levers, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.eri.fock.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_f_kernel(f_class_spec(0, 1, 0, 1, \
#   ('batched-vrr', 'sliced-hrr', 'slim-fock'))); \
#   Path(f'test/golden/cuda/f_slim_fock/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); Path(f'test/golden/hip/f_slim_fock/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLS = (0, 1, 0, 1)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_slim_fock_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = lever(GOLDEN_CLS)
    where = GOLDEN / backend / "f_slim_fock"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    assert text.count(f"for (int {sh.SLICE} = ") == sh.n_slices(GOLDEN_CLS)
    assert "f_cd_" not in text
