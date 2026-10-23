"""The block-level `f_ab` reduction (#351): `ir.Barrier`, and the lever on it.

Held here, the issue's acceptance in order, plus the IR node it needed:

0. `ir.Barrier` (#351 adds it ADDITIVELY; #333 named it missing): the
   interpreter makes a SHARED write before it visible to every warp after
   it, refuses a barrier some thread of the block does not reach (a
   divergent one, a finished warp), refuses one in a fragment run alone,
   and runs a barrier-free kernel in the order it always did (the goldens
   and test_f_seams.py's digests are the byte-level check of that, and of
   every emitted text being unchanged); both backends spell it
   `__syncthreads();`.
1. THE LEVER IS A MODULE ON THE SEAMS: one `levers.VOCABULARY` entry legal
   on `"f"` alone, one `f_levers.REGISTRY` line, moving `shared_decl` and
   `flush_bra` only; the literal build is untouched.
2. PARITY with the literal kernel, interpreted, on EVERY F class with
   L <= 4 (milestone 13's cap; the issue asks L <= 2), at one warp, two
   and four: against `ao.fock`'s every-ordered-quartet sweep at the literal
   kernel's own `TOL`, and against the literal kernel by a DERIVED bound --
   the two `F`s are sum trees over the same addends (every thread's `f_ab`
   partial is the literal value; only the order of the sums into `F`
   moves), so per element `|F_lever - F_literal| <= 2 gamma_{n-1} SUM |a_i|`
   over the literal run's `n` addends `a_i`, recorded from the interpreter.
3. THE SHARED FOOTPRINT, declared per class, is the declared `fab_sh`.
4. THE ATOMIC TRAFFIC AS A COUNT: the interpreter's atomics on `F`, literal
   minus lever, are exactly `n_bra (threads - 1) n_a n_b` per launch.

The emitted goldens (`test/golden/{cuda,hip}/f_block_fab/`) pin the lever's
text at `(ps|ss)` and `(pp|pp)`. Fast tier: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import test_kernel_f as tk

from felizjueves import f_levers, levers
from felizjueves import kernel_f as KF
from felizjueves.ao.shell import Basis
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import block_fab
from felizjueves.f_levers.seams import DEFAULT_HOOKS
from felizjueves.hgp import vrr
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Barrier,
    Call,
    Decl,
    Fn,
    Fragment,
    If,
    Index,
    Intent,
    Kernel,
    Lit,
    Param,
    Scalar,
    Space,
    Var,
    WarpReduce,
    interp,
    walk,
)
from felizjueves.ir.interp import InterpError, Interpreter, Launch, interpret_fragment
from felizjueves.spec import Target

NAME = block_fab.NAME
L4 = [c for c in vrr.CLASSES if sum(c) <= 4]
THREADS = (32, 64, 128)  # one warp, two, four (32 lanes, as tk.fused)


def lever(cls, eris: bool = False) -> Kernel:
    return KF.build_f_kernel(KF.f_class_spec(*cls, (NAME,)), eris=eris)


# ---------------------------------------------------------------------------
# 0. ir.Barrier
# ---------------------------------------------------------------------------

N = 64


def _mirror(barrier: bool) -> Kernel:
    """Each thread writes `s[t] = t`, then reads `s[N-1-t]`: across warps,
    correct only with the barrier between."""
    t = Call(Fn.THREAD_X)
    return Kernel("mirror", (Param("out", Scalar.F64, array=True,
                                   intent=Intent.OUT),), (
        Decl("s", Scalar.F64, (Lit(N),), Space.SHARED),
        Assign(Index("s", (t,)), t),
        *((Barrier(),) if barrier else ()),
        Assign(Index("out", (t,)), Index("s", (Call(Fn.SUB, (Lit(N - 1), t)),))),
    ))


def _run(k: Kernel, block: int = N, lanes: int = 32, **kw):
    return Interpreter(k, launch=Launch(1, (block, 1), lanes)).run(**kw)


def test_a_barrier_publishes_shared_writes_across_warps() -> None:
    out = _run(_mirror(True), out=np.zeros(N))["out"]
    assert out.tolist() == [float(N - 1 - t) for t in range(N)]
    # without it, warp 0 reads rows warp 1 has not written: refused
    with pytest.raises(InterpError, match="never written|not been written|"
                       "before"):
        _run(_mirror(False), out=np.zeros(N))


def test_a_barrier_some_thread_does_not_reach_is_refused() -> None:
    t = Call(Fn.THREAD_X)
    div = Kernel("div", (Param("out", Scalar.F64, array=True,
                                intent=Intent.OUT),), (
        If(Call(Fn.LT, (t, Lit(40))), (Barrier(),)),
        Assign(Index("out", (t,)), Lit(1.0)),
    ))
    with pytest.raises(InterpError, match="Barrier not reached"):
        _run(div, out=np.zeros(N))  # warp 1 splits at the If
    early = Kernel("early", div.params, (
        If(Call(Fn.LT, (t, Lit(32))), (Barrier(),)),
        Assign(Index("out", (t,)), Lit(1.0)),
    ))
    with pytest.raises(InterpError, match="Barrier not reached"):
        _run(early, out=np.zeros(N))  # warp 1 finishes, warp 0 waits
    # a block-uniform If around it is legal
    ok = Kernel("ok", div.params, (
        If(Call(Fn.LT, (Call(Fn.BLOCK_X), Lit(1))), (Barrier(),)),
        Assign(Index("out", (t,)), Lit(1.0)),
    ))
    assert _run(ok, out=np.zeros(N))["out"].sum() == N


def test_a_barrier_in_a_fragment_alone_is_refused() -> None:
    with pytest.raises(InterpError, match="Barrier in a fragment"):
        interpret_fragment(Fragment("f", (), (Barrier(),)))


def test_a_barrier_and_a_warp_reduce_compose() -> None:
    """Per warp a WarpReduce, lane 0 into shared, the barrier, thread 0 sums:
    the block reduction's shape, on 3 warps of 4 lanes."""
    t = Call(Fn.THREAD_X)
    k = Kernel("red", (Param("x", Scalar.F64, array=True),
                       Param("out", Scalar.F64, array=True,
                             intent=Intent.OUT)), (
        Decl("s", Scalar.F64, (Lit(3),), Space.SHARED),
        Decl("r", Scalar.F64),
        WarpReduce(Var("r"), Index("x", (t,))),
        If(Call(Fn.LT, (Call(Fn.SUB, (t, Call(Fn.MUL, (Call(
            Fn.DIV, (t, Lit(4))), Lit(4))))), Lit(1))),
           (Assign(Index("s", (Call(Fn.DIV, (t, Lit(4))),)), Var("r")),)),
        Barrier(),
        If(Call(Fn.LT, (t, Lit(1))), (Assign(Index("out", (Lit(0),)), Call(
            Fn.ADD, (Call(Fn.ADD, (Index("s", (Lit(0),)), Index(
                "s", (Lit(1),)))), Index("s", (Lit(2),))))),)),
    ))
    x = np.arange(12, dtype=float)
    out = _run(k, block=12, lanes=4, x=x, out=np.zeros(1))["out"]
    assert out[0] == x.sum()


@pytest.mark.parametrize("emit,arch", [(cuda.emit, SM_86),
                                       (hip.emit, GFX1200)], ids=["cuda", "hip"])
def test_both_backends_spell_a_barrier_syncthreads(emit, arch) -> None:
    text = emit(_mirror(True), arch)
    assert text.count("__syncthreads();") == 1
    assert "__syncthreads" not in emit(_mirror(False), arch)


def test_no_literal_kernel_holds_a_barrier() -> None:
    for cls in L4:
        assert not any(isinstance(s, Barrier)
                       for s in walk(KF.build_f_kernel(cls).body))


# ---------------------------------------------------------------------------
# 1. The lever is a module on the seams
# ---------------------------------------------------------------------------


def test_the_lever_is_one_vocabulary_entry_and_one_registry_line() -> None:
    lv = levers.VOCABULARY[NAME]
    assert lv.issue == 351 and lv.legal == {"f"}
    for slot in ("j", "k"):
        assert "already" in lv.refusals[slot]  # its own reason, not a default
    assert f_levers.REGISTRY[NAME] is block_fab.apply
    hooks = f_levers.hooks_for(KF.f_class_spec(1, 1, 0, 0, (NAME,)))
    moved = {f.name for f in dataclasses.fields(hooks)
             if not hooks.is_default(f.name)}
    assert moved == {"shared_decl", "flush_bra"}
    assert f_levers.hooks_for(KF.f_class_spec(1, 1, 0, 0)) is DEFAULT_HOOKS


def test_the_lever_is_refused_off_f() -> None:
    from felizjueves.spec import Operator, Spec
    with pytest.raises(ValueError, match="already"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, (NAME,))


@pytest.mark.parametrize("cls", L4, ids=str)
def test_one_flush_atomic_statement_and_one_barrier(cls) -> None:
    lit, lev = KF.build_f_kernel(cls), lever(cls)
    n = block_fab.n_elems(cls)
    f = [s for s in walk(lit.body) if isinstance(s, AtomicAdd)]
    g = [s for s in walk(lev.body) if isinstance(s, AtomicAdd)]
    assert len(f) - len(g) == n - 1  # n_ab bra-flush statements -> one
    assert sum(isinstance(s, Barrier) for s in walk(lev.body)) == 1
    assert sum(isinstance(s, WarpReduce) for s in walk(lev.body)) == n + 1
    # nothing outside the two hooks moved: strip the lever's own statements
    # and the per_bra tail is the only difference
    assert lit.params == lev.params and lit.name == lev.name


# ---------------------------------------------------------------------------
# 3. The shared footprint, declared per class
# ---------------------------------------------------------------------------

FOOTPRINT = {(0, 0): 256, (0, 1): 768, (1, 1): 2304, (0, 2): 1536,
             (1, 2): 4608, (2, 2): 9216}


@pytest.mark.parametrize("cls", L4, ids=str)
def test_the_shared_footprint_is_declared_per_class(cls) -> None:
    shared = {s.name: s for s in walk(lever(cls).body)
              if isinstance(s, Decl) and s.space is Space.SHARED}
    assert set(shared) == {KF.AB_SH, block_fab.RED_SH}
    red = shared[block_fab.RED_SH]
    assert red.shape == (Lit(block_fab.MAX_WARPS), Lit(block_fab.n_elems(cls)))
    assert block_fab.reduce_bytes(cls) == FOOTPRINT[tuple(cls[:2])]
    total = 8 * math.prod(d.value for d in red.shape) + 8 * math.prod(
        d.value for d in shared[KF.AB_SH].shape)
    assert block_fab.footprint_bytes(cls) == total
    assert block_fab.footprint_bytes((2, 2, 0, 0)) == 17408


def test_the_launch_contract_is_refused_outside_it() -> None:
    assert block_fab.launch_shape(5, 128, 32) == (5, (128, 1))
    for threads, wf in ((48, 32), (2048, 32), (32, 16), (96, 64)):
        with pytest.raises(ValueError):
            block_fab.launch_shape(5, threads, wf)


def test_a_partial_warp_is_refused_by_the_interpreter(monkeypatch) -> None:
    with pytest.raises(InterpError, match="partial warp"):
        _parity_run((0, 0, 0, 0), 48, monkeypatch, lever((0, 0, 0, 0)))


# ---------------------------------------------------------------------------
# 2. Parity, interpreted; 4. the atomic count
# ---------------------------------------------------------------------------

U = 2.0 ** -53


def _gamma(n: int) -> float:
    return n * U / (1 - n * U)


class _Recorder:
    """Every `AtomicAdd` into `F` an interpreted run makes: per element, the
    addends' count and their absolute sum."""

    def __init__(self, monkeypatch, n_ao: int) -> None:
        self.count = np.zeros((n_ao, n_ao), dtype=int)
        self.absum = np.zeros((n_ao, n_ao))
        stmt = interp._Thread.stmt
        rec = self

        def recording(thread, s):
            if isinstance(s, AtomicAdd) and s.dst.array == KF.FOCK:
                at = tuple(thread.eval(i) for i in s.dst.at)
                rec.count[at] += 1
                rec.absum[at] += abs(float(thread.eval(s.value)))
            return stmt(thread, s)

        monkeypatch.setattr(interp._Thread, "stmt", recording)


def _launch(cls, diag: bool):
    """(basis, bras, kets): two bras by two kets off the diagonal, or a
    diagonal batch of three same-`l` pairs (where the ket flushes land on
    the bra block's elements too)."""
    if diag:
        l = cls[0]  # noqa: E741
        basis = Basis(tuple(tk._same_l_shells(l, 2)))
        pairs = [(0, 0), (0, 1), (1, 1)]
        recs = [tk.records(basis, *p) for p in pairs]
        return basis, recs, recs
    geoms = [tk.CENTRES, tk._moved((0.3, -0.2, 0.1))]
    basis = Basis(tuple(s for g in geoms for s in tk._shells(cls, g)))
    return (basis, [tk.records(basis, 0, 1), tk.records(basis, 4, 5)],
            [tk.records(basis, 2, 3), tk.records(basis, 6, 7)])


def _parity_run(cls, threads, monkeypatch, kern, diag=False, recorder=False):
    basis, bras, kets = _launch(cls, diag)
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    rec = _Recorder(monkeypatch, basis.n_ao) if recorder else None
    F, _, _ = tk.fused(cls, basis, tk._density(basis.n_ao), bras, kets,
                       diag=int(diag), threads=threads)
    monkeypatch.undo()
    return F, rec, basis, bras, kets


def _same_l(cls) -> bool:
    return len(set(cls)) == 1


PARITY = [pytest.param(c, t, d, id=f"{''.join(map(str, c))}-t{t}"
                       + ("-diag" if d else ""))
          for c in L4 for t in THREADS for d in (False, True)
          if not d or _same_l(c)]


@pytest.mark.parametrize(("cls", "threads", "diag"), PARITY)
def test_parity_with_the_literal_kernel_and_the_sweep(cls, threads, diag,
                                                      monkeypatch) -> None:
    F0, rec, basis, bras, kets = _parity_run(
        cls, threads, monkeypatch, KF.build_f_kernel(cls), diag, True)
    lit_atomics = int(rec.count.sum())
    F1, rec1, *_ = _parity_run(cls, threads, monkeypatch, lever(cls), diag,
                               True)
    # 2a. against the literal kernel, by the derived re-association bound
    bound = 2 * np.vectorize(_gamma)(np.maximum(rec.count - 1, 0)) * rec.absum
    assert np.all(np.abs(F1 - F0) <= bound), float(np.max(np.abs(F1 - F0)
                                                          - bound))
    assert np.max(np.abs(F0)) > 0
    # 2b. against the sweep, at the literal kernel's own TOL
    bra_ix = [r.sh for r in bras]
    ket_ix = [r.sh for r in kets]
    want, scale = tk.sweep(basis, tk._density(basis.n_ao),
                           tk._members(bra_ix, ket_ix))
    assert tk._worst(np.array(KF.close(F1)), want, scale) <= tk.TOL
    # 4. the atomic traffic, counted
    n_bra = len(bras)
    removed = n_bra * (threads - 1) * block_fab.n_elems(cls)
    assert lit_atomics - int(rec1.count.sum()) == removed
    assert removed == n_bra * (
        block_fab.bra_flush_atomics(cls, threads, lever=False)
        - block_fab.bra_flush_atomics(cls, threads, lever=True))


@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (1, 1, 1, 1)], ids=str)
def test_one_bra_one_ket_is_bitwise(cls, monkeypatch) -> None:
    """One bra by one ket of four distinct shells: each element of `F` gets
    one nonzero addend, every other partial an exact zero, so the
    re-association is exact and the two kernels agree bit for bit."""
    basis = Basis(tuple(tk._shells(cls)))
    bras, kets = [tk.records(basis, 0, 1)], [tk.records(basis, 2, 3)]
    D = tk._density(basis.n_ao)
    out = []
    for kern in (KF.build_f_kernel(cls), lever(cls)):
        monkeypatch.setattr(tk, "_kernel", lambda *_, k=kern: k)
        out.append(tk.fused(cls, basis, D, bras, kets, threads=64)[0])
    assert np.max(np.abs(out[0])) > 0
    assert np.array_equal(out[0], out[1])


# ---------------------------------------------------------------------------
# The goldens: (ps|ss) and (pp|pp) with the lever, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   [(Path(f'test/golden/cuda/f_block_fab/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)), Path(f'test/golden/hip/f_block_fab/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))) for k in (build_f_kernel( \
#   f_class_spec(*c, ('block-fab',))) for c in ((0,0,0,1), (1,1,1,1)))]"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLASSES = ((0, 0, 0, 1), (1, 1, 1, 1))
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_block_fab_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    where = GOLDEN / backend / "f_block_fab"
    kernels = [lever(c) for c in GOLDEN_CLASSES]
    assert sorted(p.name for p in where.iterdir()) == sorted(
        f"{k.name}.{ext}" for k in kernels)
    for kern, c in zip(kernels, GOLDEN_CLASSES, strict=True):
        text = emit(kern, arch)
        assert text == (where / f"{kern.name}.{ext}").read_text()
        assert text.count("__syncthreads();") == 1
        assert (f"__shared__ double {block_fab.RED_SH}"
                f"[{block_fab.MAX_WARPS}][{block_fab.n_elems(c)}];") in text
