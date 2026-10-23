"""`kernel_s`: the overlap kernel, `Target.S`, interpreted (#237).

THE WITNESS is `ao.overlap.overlap_block` (#233), the host's `S` block: the
summed `t = 0` column of the blocks `ao.pairs.primitive_pairs` dresses. The
kernel's records are those pairs' `(zeta, P - A, P - B)` and the host's one
scalar `s` per record (`ao.density.epair`, `pairs.dress`'s own `scale`), and its
scatter is `kernel_jab`'s rule. Against it:

- PER CLASS, every ordered pair class (ss) through (ff) on the tracked
  HF/cc-pVTZ basis (the one fixture whose shells reach f), in ONE launch
  over every record of every shell pair `i <= j` of the class -- the
  orientation `ao.one_electron.matrix` calls `block(sa, sb)` in --
  diagonal and off-diagonal pairs mixed.
- THE WHOLE MATRIX, one launch per class into one zeroed `S`, on H2O/STO-3G
  and HF/cc-pVTZ.

`S` has no Boys and no `[r]` tower, and the whole file, every class through
(ff) and HF/cc-pVTZ's whole matrix included, interprets in under a second;
still, every case that builds a class above `la + lb = 4` (the d-f and f-f
classes, and HF/cc-pVTZ's whole matrix) is `slow`, by test/tiers.py's cut.

EXACT, NOT A TOLERANCE, against the block. Interpreted with the records in
order, the kernel performs `overlap_block`'s multiplications in its
association and its additions in its order (kernel_s's docstring names the
two admitted departures and why neither changes a bit), so the reference is
`overlap_block` placed by `kernel_jab`'s rule and the assertion is
`np.array_equal`.

AGAINST `overlap_matrix`, A STATED BOUND. `ao.one_electron.matrix` reads a
diagonal shell pair's UPPER triangle; `kernel_jab`'s rule keeps the LOWER.
`<a_i|b_j>` and `<a_j|b_i>` are the same integral reached along different
`E` recurrence paths, so they agree to rounding, not by construction. The
bound is `8 eps |S|` per element (`eps = 2^-52`): a few roundings of the
three-factor product and of the record sum, never a fitted number. The
off-diagonal elements, which both read from the same block entry, are held
`==`.

The libcint comparison of `S` is milestone 11's `O1`, not this issue's.
Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l, split_by_l

from felizjueves.ao.density import epair
from felizjueves.ao.overlap import overlap_block, overlap_matrix
from felizjueves.ao.pairs import primitive_pairs
from felizjueves.emit import cuda, hip
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Fn,
    If,
    Index,
    Inline,
    Lit,
    Note,
    Param,
    Var,
    _subexprs,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.kernel_s import (
    ATOMIC_REASON,
    ZEROED_S,
    build_s_kernel,
    launch_shape,
    s_spec,
)
from felizjueves.launch import per_axis
from felizjueves.mmd import e_fragment as ef
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.spec import Operator, Spec, Target

LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
EPS = 2.0 ** -52
DIAG_BOUND = 8  # x eps |S|: see the module docstring


def _cls_param(cls):
    return by_l(cls, sum(cls), id=f"{cls[0]}{cls[1]}")


@cache
def _basis(name: str):
    return ob.from_fixture(manifest.load(manifest.FIXTURES / f"{name}.json"))


def _pairs_of(basis, cls):
    """Shell pairs `i <= j` of class `cls`, in `one_electron.matrix` order."""
    sh = basis.shells
    return [(i, j) for i in range(len(sh)) for j in range(i, len(sh))
            if (sh[i].l, sh[j].l) == cls]


def _axes(name, vecs):
    """A record vector as the kernel takes it, one flat array per axis."""
    return {k: np.array(v) for k, v in per_axis(name, vecs).items()}


def _records(basis, pairs):
    """The kernel's per-record arrays for `pairs`, records in pair order and,
    within a pair, in `primitive_pairs` order."""
    sh, off = basis.shells, basis.offsets
    zeta, xpa, xpb, s, oa, obb = [], [], [], [], [], []
    for i, j in pairs:
        ep = epair(sh[i], sh[j])
        for P, z, sk in ep.recs:
            zeta.append(z)
            xpa.append([P[d] - ep.A[d] for d in range(3)])
            xpb.append([P[d] - ep.B[d] for d in range(3)])
            s.append(sk)
            oa.append(off[i])
            obb.append(off[j])
    return dict(zeta_ab=np.array(zeta), **_axes("pa", xpa), **_axes("pb", xpb),
                s_ab=np.array(s), off_a=np.array(oa, dtype=np.int32),
                off_b=np.array(obb, dtype=np.int32))


def _launch(cls, basis, pairs, S, threads=5):
    """One launch of class `cls` over every record of `pairs`, into `S`."""
    rec = _records(basis, pairs)
    n_rec = len(rec["s_ab"])
    grid, block = launch_shape(n_rec, threads)
    return Interpreter(build_s_kernel(s_spec(*cls)),
                       launch=Launch(grid, block, 1)).run(
        n_rec=n_rec, n_ao=basis.n_ao, S=S, **rec)["S"]


def _host(basis, pairs, M):
    """`overlap_block` placed into `M` by `kernel_jab`'s scatter rule."""
    sh, off = basis.shells, basis.offsets
    for i, j in pairs:
        blk = overlap_block(sh[i], sh[j])
        for r in range(sh[i].n_cart):
            for c in range(sh[j].n_cart):
                if i == j and c > r:
                    continue  # diagonal: the lower triangle, mirrored
                M[off[i] + r, off[j] + c] = blk[r][c]
                M[off[j] + c, off[i] + r] = blk[r][c]
    return M


def _owned(basis, pairs) -> np.ndarray:
    sh, off = basis.shells, basis.offsets
    m = np.zeros((basis.n_ao, basis.n_ao), bool)
    for i, j in pairs:
        m[off[i]:off[i] + sh[i].n_cart, off[j]:off[j] + sh[j].n_cart] = True
    return m | m.T


def _whole(basis) -> np.ndarray:
    S = np.zeros((basis.n_ao, basis.n_ao))
    for cls in sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                       for b in basis.shells[k:]}):
        S = _launch(cls, basis, _pairs_of(basis, cls), S)
    return S


def _diag_mask(basis) -> np.ndarray:
    """True on the elements a DIAGONAL shell pair owns."""
    sh, off = basis.shells, basis.offsets
    m = np.zeros((basis.n_ao, basis.n_ao), bool)
    for i, s in enumerate(sh):
        m[off[i]:off[i] + s.n_cart, off[i]:off[i] + s.n_cart] = True
    return m


# ---------------------------------------------------------------------------
# Per class: one launch over every record of the class
# ---------------------------------------------------------------------------


def test_hf_ccpvtz_reaches_every_class_and_a_diagonal_in_each_square_one():
    basis = _basis("hf_ccpvtz")
    for cls in CLASSES:
        pairs = _pairs_of(basis, cls)
        assert pairs, cls  # both orientations occur with i <= j
        if cls[0] == cls[1]:
            assert any(i == j for i, j in pairs), cls
    # Accumulation over records is exercised: some pairs have many.
    sh = basis.shells
    assert max(len(sh[i].exponents) * len(sh[j].exponents)
               for i, j in _pairs_of(basis, (0, 0))) > 32


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_one_launch_over_the_class_equals_the_host_block(cls):
    basis = _basis("hf_ccpvtz")
    pairs = _pairs_of(basis, cls)
    n = basis.n_ao
    got = _launch(cls, basis, pairs, np.zeros((n, n)))
    want = _host(basis, pairs, np.zeros((n, n)))
    assert np.array_equal(got, want)
    owned = _owned(basis, pairs)
    assert np.all(got[~owned] == 0.0)  # exactly the pairs' blocks and mirrors
    assert np.array_equal(got, got.T)


# ---------------------------------------------------------------------------
# The whole matrix
# ---------------------------------------------------------------------------

WHOLE = [pytest.param("h2o_sto3g_j", id="h2o_sto3g"),
         by_l("hf_ccpvtz", 6, id="hf_ccpvtz")]  # f shells: (ff) at L = 6


@pytest.mark.parametrize("name", WHOLE)
def test_the_whole_matrix(name):
    basis = _basis(name)
    S = _whole(basis)
    n = basis.n_ao
    sh = basis.shells
    every = [(i, j) for i in range(len(sh)) for j in range(i, len(sh))]
    assert np.array_equal(S, _host(basis, every, np.zeros((n, n))))  # bitwise
    assert np.array_equal(S, S.T)
    R = np.array(overlap_matrix(basis))
    diag = _diag_mask(basis)
    assert np.array_equal(S[~diag], R[~diag])  # the same block entry, read
    err = np.abs(S - R)[diag]
    bound = DIAG_BOUND * EPS * np.abs(R[diag])
    print(f"\n{name}: max|S - overlap_matrix| on diagonal shell pairs "
          f"= {err.max():.2e}")
    assert np.all(err <= bound)


# ---------------------------------------------------------------------------
# The diagonal rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l", split_by_l(range(LMAX + 1), lambda n: 2 * n,
                                         ids=str))
def test_a_diagonal_pair_adds_into_every_element_once_per_record(l):  # noqa: E741
    """Structurally: the diagonal branch's destinations are the block's
    `n^2` elements, each exactly once. Numerically: ONE record of a
    diagonal shell pair leaves `S` equal to that record's dressed `t = 0`
    column, lower triangle mirrored, and a naive double count differs."""
    k = build_s_kernel((l, l))
    (scatter,) = [s for s in walk(k.body) if isinstance(s, If)
                  and any(isinstance(t, AtomicAdd) for t in s.then)]
    n = len(ef.cartesians(l))
    dsts = []
    for s in scatter.orelse:
        if isinstance(s, AtomicAdd):
            rows, cols = s.dst.at  # (oa + i, oa + j)
            dsts.append((rows.args[1].value, cols.args[1].value))
    assert sorted(dsts) == [(i, j) for i in range(n) for j in range(n)]

    basis = _basis("hf_ccpvtz")
    sh, off = basis.shells, basis.offsets
    i = next(m for m, s in enumerate(sh) if s.l == l)
    one = {key: v[:1] for key, v in _records(basis, [(i, i)]).items()}
    N = basis.n_ao
    got = Interpreter(k, launch=Launch(1, (1, 1), 1)).run(
        n_rec=1, n_ao=N, S=np.zeros((N, N)), **one)["S"]
    ep = epair(sh[i], sh[i])
    pp = primitive_pairs(sh[i], sh[i])[0]
    assert pp.zeta == ep.recs[0][1]
    want = np.zeros((N, N))
    o = off[i]
    for r in range(n):
        for c in range(r + 1):
            want[o + r, o + c] = want[o + c, o + r] = pp.block[r * n + c][0]
    assert np.array_equal(got, want)
    d = np.diag(got)[o:o + n]
    assert np.all(d != 0.0)
    naive = got.copy()
    naive[o:o + n, o:o + n] += np.diag(d)
    assert not np.array_equal(naive, want)


# ---------------------------------------------------------------------------
# The contract, structurally
# ---------------------------------------------------------------------------


def test_the_key_is_a_pair_class():
    assert s_spec(2, 1) == Spec(Operator.OVERLAP, 0, (2, 1), Target.S)
    assert build_s_kernel(s_spec(2, 1)) == build_s_kernel((2, 1))
    assert build_s_kernel((2, 1)).name == "s_21"
    with pytest.raises(ValueError, match="Target.S"):
        build_s_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.JAB))
    with pytest.raises(ValueError):
        build_s_kernel(Spec(Operator.OVERLAP, 0, (1, 1), Target.S, ("recycle",)))


def test_the_parameter_block():
    k = build_s_kernel((1, 2))
    assert [p.name for p in k.params] == [
        "n_rec", "n_ao", "zeta_ab", "pa_x", "pa_y", "pa_z",
        "pb_x", "pb_y", "pb_z", "s_ab", "off_a", "off_b", "S"]
    assert all(p.shape == (Var("n_rec"),) for p in k.params
               if p.name.startswith(("pa_", "pb_")))
    (S,) = [p for p in k.params if p.name == "S"]
    assert S.intent.name == "OUT"


def test_launch_shape():
    assert launch_shape(1) == (1, (32, 1))
    assert launch_shape(33, 32) == (2, (32, 1))
    for bad in ((0, 32), (5, 0), (2.0, 32)):
        with pytest.raises(ValueError):
            launch_shape(*bad)


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_no_prefactor_is_formed_in_the_kernel(cls):
    """`s` is the host's one scalar per record. Read off the `Decl` set
    (fragment included): no power, no square root, no Boys, no `pi` -- the
    only float literal is `0.5` (`1/2zeta`), and the only transcendental
    is the three `E` seeds' `exp` (which carry `K_AB`, never reinstated)."""
    k = build_s_kernel(cls)
    stmts = list(walk(k.body))
    exprs = []
    for s in stmts:
        if isinstance(s, Decl) and s.init is not None:
            exprs.append(s.init)
        elif isinstance(s, Inline):
            exprs += [e for _, e in s.bindings]
    subs = [e for x in exprs for e in _subexprs(x)]
    fns = {e.fn for e in subs if isinstance(e, Call)}
    assert fns <= {Fn.ADD, Fn.MUL, Fn.DIV, Fn.LT, Fn.EXP, Fn.THREAD_X,
                   Fn.BLOCK_X, Fn.BLOCK_DIM_X}
    assert sum(isinstance(e, Call) and e.fn is Fn.EXP for e in subs) == 3
    floats = {e.value for e in subs
              if isinstance(e, Lit) and isinstance(e.value, float)}
    assert floats <= {0.5}
    # ...and S_ab is `sk * E0[r]`, nothing else multiplied in.
    vs = [s for s in stmts if isinstance(s, Decl) and s.name.startswith("v")]
    assert len(vs) == len(ef.block_rows(*cls))
    for r, v in enumerate(vs):
        assert v.init.fn is Fn.MUL
        assert v.init.args[1] == Index(ef.E0_ARRAY, (Lit(r),))
    assert not [p for p in k.params if isinstance(p, Param)
                and "pi" in p.name.lower()]


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_every_write_to_S_is_an_atomic_add_with_its_reason(cls):
    k = build_s_kernel(cls)
    stmts = list(walk(k.body))
    notes = {s.prov for s in stmts if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_S in notes
    assert not [s for s in stmts if isinstance(s, Assign)
                and isinstance(s.dst, Index) and s.dst.array == "S"]
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert {s.dst.array for s in adds} == {"S"}
    na, nb = len(ef.cartesians(cls[0])), len(ef.cartesians(cls[1]))
    (scatter,) = [s for s in stmts if isinstance(s, If)
                  and any(isinstance(t, AtomicAdd) for t in s.then)]
    off = [s for s in scatter.then if isinstance(s, AtomicAdd)]
    diag = [s for s in scatter.orelse if isinstance(s, AtomicAdd)]
    assert len(off) == 2 * na * nb
    assert len(diag) == (na * na if cls[0] == cls[1] else 0)


def test_it_inlines_the_overlap_fragment_not_the_block():
    k = build_s_kernel((2, 2))
    (inl,) = [s for s in walk(k.body, into_inline=False)
              if isinstance(s, Inline)]
    assert inl.fragment == ef.overlap_fragment(2, 2)


# ---------------------------------------------------------------------------
# Goldens: the emitted text, both backends (compile and run are later issues)
# ---------------------------------------------------------------------------

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLASS = (1, 1)


def test_the_cuda_golden_is_unchanged() -> None:
    # Regenerate (never hand-edit):
    #   PYTHONPATH=src uv run python -c "from felizjueves.kernel_s import \
    #   build_s_kernel as b; from felizjueves.performance.device import SM_86; \
    #   from felizjueves.emit.cuda import emit; print(emit(b((1, 1)), SM_86), \
    #   end='')" > test/golden/cuda/s_11.cu
    text = cuda.emit(build_s_kernel(GOLDEN_CLASS), SM_86)
    assert text == (GOLDEN / "cuda" / "s_11.cu").read_text()


def test_the_hip_golden_is_unchanged() -> None:
    #   ... the same with GFX1200 and felizjueves.emit.hip, into
    #   test/golden/hip/s_11.hip
    text = hip.emit(build_s_kernel(GOLDEN_CLASS), GFX1200)
    assert text == (GOLDEN / "hip" / "s_11.hip").read_text()
