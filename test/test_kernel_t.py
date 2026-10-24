"""`kernel_t`: the kinetic-energy kernel, `Target.T`, interpreted (#238).

THE WITNESS is `ao.kinetic.kinetic_block` (#234), the host's `T` block: the
fixed ket shift on the axis tables at `lb + 2`, dressed by `ao.pairs.dress`
and summed over `primitive_pairs` order. The kernel's records are those
pairs' `(zeta, P - A, P - B)`, the host's one scalar `s` per record
(`ao.density.epair`, `pairs.dress`'s own `scale`) and the ket exponent `beta`
(`eb_ket`), and its scatter is `kernel_jab`'s rule. Against it:

- PER CLASS, every ordered pair class (ss) through (ff) on the tracked
  HF/cc-pVTZ basis (the one fixture whose shells reach f), in ONE launch
  over every record of every shell pair `i <= j` of the class -- the
  orientation `ao.one_electron.matrix` calls `block(sa, sb)` in --
  diagonal and off-diagonal pairs mixed.
- THE WHOLE MATRIX, one launch per class into one zeroed `T`, on H2O/STO-3G
  and HF/cc-pVTZ.

The widest table is (ff)'s, read at `lb + 2 = 5`: Hermite order 8 on an
axis, within the run's `L <= 8` cap. No Boys, no `[r]` tower; still, every
case that builds a class above `la + lb = 4` (the d-f and f-f classes, and
HF/cc-pVTZ's whole matrix) is `slow`, by test/tiers.py's cut.

EXACT, NOT A TOLERANCE, against the block. Interpreted with the records in
order, the kernel performs `kinetic_block`'s multiplications in its
association and its additions in its order (kernel_t's docstring), so the
reference is `kinetic_block` placed by `kernel_jab`'s rule and the
assertion is `np.array_equal`.

AGAINST `kinetic_matrix`, A STATED BOUND. `ao.one_electron.matrix` reads a
diagonal shell pair's UPPER triangle; `kernel_jab`'s rule keeps the LOWER.
`<a_i|T|a_j>` and `<a_j|T|a_i>` are the same integral reached along
different `E` paths and different shifts, so they agree to rounding, not by
construction. `T`'s bracket is a signed sum (the `-2 beta^2` term against
the rest), so the scale a rounding is relative to is the diagonal block's
largest element, not the element itself: the bound is `8 eps max|T_blk|`
per element of a diagonal shell pair's block (`eps = 2^-52`), `kernel_s`'s
count of roundings on that scale -- never a fitted number (observed: 0 on
H2O/STO-3G, 0.31 on HF/cc-pVTZ). The off-diagonal elements, which both read from the
same block entry, are held `==`.

The libcint comparison of `T` is milestone 11's `O1`, not this issue's.
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
from felizjueves.ao.pairs import primitive_pairs
from felizjueves.emit import cuda, hip
from felizjueves.eri.j_coulomb.launch import per_axis
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
    Param,
    Var,
    _subexprs,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd import e_fragment as ef
from felizjueves.one_electron.kernel_t import (
    ATOMIC_REASON,
    ZEROED_T,
    build_t_kernel,
    launch_shape,
    t_spec,
)
from felizjueves.one_electron.kinetic import kinetic_block, kinetic_matrix
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.spec import Operator, Spec, Target

LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
EPS = 2.0 ** -52
DIAG_BOUND = 8 # x eps max|T_blk|: see the module docstring


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
    zeta, xpa, xpb, s, eb, oa, obb = [], [], [], [], [], [], []
    for i, j in pairs:
        pps = primitive_pairs(sh[i], sh[j])
        ep = epair(sh[i], sh[j], pps)
        for pp, (P, z, sk) in zip(pps, ep.recs, strict=True):
            assert z == sh[i].exponents[pp.i] + sh[j].exponents[pp.j]
            zeta.append(z)
            xpa.append([P[d] - ep.A[d] for d in range(3)])
            xpb.append([P[d] - ep.B[d] for d in range(3)])
            s.append(sk)
            eb.append(sh[j].exponents[pp.j])  # the KET primitive's exponent
            oa.append(off[i])
            obb.append(off[j])
    return dict(zeta_ab=np.array(zeta), **_axes("pa", xpa), **_axes("pb", xpb),
                s_ab=np.array(s), eb_ket=np.array(eb),
                off_a=np.array(oa, dtype=np.int32),
                off_b=np.array(obb, dtype=np.int32))


def _launch(cls, basis, pairs, T, threads=5):
    """One launch of class `cls` over every record of `pairs`, into `T`."""
    rec = _records(basis, pairs)
    n_rec = len(rec["s_ab"])
    grid, block = launch_shape(n_rec, threads)
    return Interpreter(build_t_kernel(t_spec(*cls)),
                       launch=Launch(grid, block, 1)).run(
        n_rec=n_rec, n_ao=basis.n_ao, T=T, **rec)["T"]


def _host(basis, pairs, M):
    """`kinetic_block` placed into `M` by `kernel_jab`'s scatter rule."""
    sh, off = basis.shells, basis.offsets
    for i, j in pairs:
        blk = kinetic_block(sh[i], sh[j])
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
    T = np.zeros((basis.n_ao, basis.n_ao))
    for cls in sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                       for b in basis.shells[k:]}):
        T = _launch(cls, basis, _pairs_of(basis, cls), T)
    return T


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
    assert np.array_equal(got, want)  # bitwise
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
    T = _whole(basis)
    n = basis.n_ao
    sh, off = basis.shells, basis.offsets
    every = [(i, j) for i in range(len(sh)) for j in range(i, len(sh))]
    assert np.array_equal(T, _host(basis, every, np.zeros((n, n))))  # bitwise
    assert np.array_equal(T, T.T)
    R = np.array(kinetic_matrix(basis))
    diag = np.zeros((n, n), bool)
    worst = 0.0
    for i, s in enumerate(sh):
        blk = np.s_[off[i]:off[i] + s.n_cart, off[i]:off[i] + s.n_cart]
        diag[blk] = True
        err = np.abs(T[blk] - R[blk]).max()
        scale = np.abs(R[blk]).max()
        worst = max(worst, err / (EPS * scale))
        assert err <= DIAG_BOUND * EPS * scale, (i, err, scale)
    assert np.array_equal(T[~diag], R[~diag])  # the same block entry, read
    print(f"\n{name}: max|T - kinetic_matrix| on diagonal shell pairs "
          f"= {worst:.2f} eps max|T_blk|")


# ---------------------------------------------------------------------------
# The diagonal rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l", split_by_l(range(LMAX + 1), lambda n: 2 * n,
                                         ids=str))
def test_a_diagonal_pair_adds_into_every_element_once_per_record(l):  # noqa: E741
    """ONE record of a diagonal shell pair leaves `T` equal to that
    record's dressed bracket, lower triangle mirrored."""
    k = build_t_kernel((l, l))
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
    i = next(m for m, s in enumerate(sh) if s.l == l and len(s.exponents) == 1)
    one = _records(basis, [(i, i)])
    assert len(one["s_ab"]) == 1
    N = basis.n_ao
    got = Interpreter(k, launch=Launch(1, (1, 1), 1)).run(
        n_rec=1, n_ao=N, T=np.zeros((N, N)), **one)["T"]
    blk = kinetic_block(sh[i], sh[i])  # one primitive: one record
    want = np.zeros((N, N))
    o = off[i]
    for r in range(n):
        for c in range(r + 1):
            want[o + r, o + c] = want[o + c, o + r] = blk[r][c]
    assert np.array_equal(got, want)
    assert np.all(np.diag(got)[o:o + n] > 0.0)  # kinetic energy is positive


# ---------------------------------------------------------------------------
# The contract, structurally
# ---------------------------------------------------------------------------


def test_the_key_is_a_pair_class():
    assert t_spec(2, 1) == Spec(Operator.KINETIC, 0, (2, 1), Target.T)
    assert build_t_kernel(t_spec(2, 1)) == build_t_kernel((2, 1))
    assert build_t_kernel((2, 1)).name == "t_21"
    with pytest.raises(ValueError, match="Target.T"):
        build_t_kernel(Spec(Operator.OVERLAP, 0, (1, 1), Target.S))
    with pytest.raises(ValueError):
        build_t_kernel(Spec(Operator.KINETIC, 0, (1, 1), Target.T, ("recycle",)))


def test_the_parameter_block_is_kernel_s_plus_the_ket_exponent():
    from felizjueves.one_electron.kernel_s import build_s_kernel

    k = build_t_kernel((1, 2))
    s_names = [p.name for p in build_s_kernel((1, 2)).params]
    names = [p.name for p in k.params]
    at = s_names.index("s_ab") + 1
    assert names == [*s_names[:at], "eb_ket", *s_names[at:-1], "T"]
    (T,) = [p for p in k.params if p.name == "T"]
    assert T.intent.name == "OUT"


def test_launch_shape():
    assert launch_shape(1) == (1, (32, 1))
    assert launch_shape(33, 32) == (2, (32, 1))
    for bad in ((0, 32), (5, 0), (2.0, 32)):
        with pytest.raises(ValueError):
            launch_shape(*bad)


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_no_prefactor_is_formed_in_the_kernel(cls):
    """`s` is the host's one scalar per record. No power, no square root, no
    Boys, no `pi`; the float literals are `0.5` (`1/2zeta`) and the shift's
    own `-2`, `2j+1` and `j(j-1)/2`; the only transcendental is the three
    `E` seeds' `exp` (which carry `K_AB`, never reinstated)."""
    k = build_t_kernel(cls)
    stmts = list(walk(k.body))
    exprs = []
    for s in stmts:
        if isinstance(s, Decl) and s.init is not None:
            exprs.append(s.init)
        elif isinstance(s, Assign):
            exprs.append(s.value)
        elif isinstance(s, Inline):
            exprs += [e for _, e in s.bindings]
    subs = [e for x in exprs for e in _subexprs(x)]
    fns = {e.fn for e in subs if isinstance(e, Call)}
    assert fns <= {Fn.ADD, Fn.SUB, Fn.MUL, Fn.DIV, Fn.LT, Fn.EXP, Fn.THREAD_X,
                   Fn.BLOCK_X, Fn.BLOCK_DIM_X}
    assert sum(isinstance(e, Call) and e.fn is Fn.EXP for e in subs) == 3
    floats = {e.value for e in subs
              if isinstance(e, Lit) and isinstance(e.value, float)}
    lb = cls[1]
    shift = ({-2.0} | {float(2 * j + 1) for j in range(lb + 1)}
             | {0.5 * j * (j - 1) for j in range(2, lb + 1)})
    assert floats <= {0.5} | shift
    vs = [s for s in stmts if isinstance(s, Decl) and s.name.startswith("v")]
    assert len(vs) == len(ef.block_rows(*cls))
    for r, v in enumerate(vs):
        assert v.init == Call(Fn.MUL, (Var("sk"),
                                       Index(ef.KE_ARRAY, (Lit(r),))))
    assert not [p for p in k.params if isinstance(p, Param)
                and "pi" in p.name.lower()]


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_every_write_to_T_is_an_atomic_add_with_its_reason(cls):
    k = build_t_kernel(cls)
    stmts = list(walk(k.body))
    notes = {s.prov for s in stmts if hasattr(s, "prov")}
    assert ATOMIC_REASON in notes and ZEROED_T in notes
    assert not [s for s in stmts if isinstance(s, Assign)
                and isinstance(s.dst, Index) and s.dst.array == "T"]
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert {s.dst.array for s in adds} == {"T"}
    na, nb = len(ef.cartesians(cls[0])), len(ef.cartesians(cls[1]))
    (scatter,) = [s for s in stmts if isinstance(s, If)
                  and any(isinstance(t, AtomicAdd) for t in s.then)]
    off = [s for s in scatter.then if isinstance(s, AtomicAdd)]
    diag = [s for s in scatter.orelse if isinstance(s, AtomicAdd)]
    assert len(off) == 2 * na * nb
    assert len(diag) == (na * na if cls[0] == cls[1] else 0)


def test_it_inlines_the_kinetic_fragment_and_binds_eb_from_eb_ket():
    k = build_t_kernel((2, 1))
    (inl,) = [s for s in walk(k.body, into_inline=False)
              if isinstance(s, Inline)]
    assert inl.fragment == ef.kinetic_fragment(2, 1)
    assert dict(inl.bindings)[ef.EB] == Var(ef.EB)
    (eb,) = [s for s in walk(k.body, into_inline=False)
             if isinstance(s, Decl) and s.name == ef.EB]
    assert eb.init == Index("eb_ket", (Var("rec"),))


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_the_widened_table_is_read_only_through_the_shift(cls):
    """The fragment declares `E{d}_i_j_0` up to `j = lb + 2`; an entry above
    `lb` is read by a `K_ij` and by nothing the kernel adds into `T`."""
    la, lb = cls
    frag = build_t_kernel(cls)
    (inl,) = [s for s in walk(frag.body, into_inline=False)
              if isinstance(s, Inline)]
    body = inl.fragment.body
    decls = {s.name: s for s in body if isinstance(s, Decl)}
    for d in range(3):
        assert ef.e_name(d, la, lb + 2, 0) in decls
        assert ef.e_name(d, la + 1, 0, 0) not in decls  # the bra never widens
    wide = {n for n in decls if n.startswith("E")
            and int(n.split("_")[2]) > lb}
    for s in body:
        reads = {v.name for v in _subexprs(
            s.init if isinstance(s, Decl) else s.value) if isinstance(v, Var)} \
            if isinstance(s, (Decl, Assign)) else set()
        if isinstance(s, Assign):
            assert not reads & wide, s
        elif isinstance(s, Decl) and s.name.startswith("K"):
            j = int(s.name.split("_")[2])
            got = {int(n.split("_")[2]) for n in reads & wide}
            assert got <= {j + 2} and (j < lb - 1 or got == {j + 2}), s


# ---------------------------------------------------------------------------
# Goldens: the emitted text, both backends (compile and run are later issues)
# ---------------------------------------------------------------------------

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLASS = (1, 2)  # lb = 2: the j(j-1) term is present


def test_the_cuda_golden_is_unchanged() -> None:
    # Regenerate (never hand-edit):
    #   PYTHONPATH=src uv run python -c "from felizjueves.one_electron.kernel_t import \
    #   build_t_kernel as b; from felizjueves.performance.device import SM_86; \
    #   from felizjueves.emit.cuda import emit; print(emit(b((1, 2)), SM_86), \
    #   end='')" > test/golden/cuda/t_12.cu
    text = cuda.emit(build_t_kernel(GOLDEN_CLASS), SM_86)
    assert text == (GOLDEN / "cuda" / "t_12.cu").read_text()


def test_the_hip_golden_is_unchanged() -> None:
    #   ... the same with GFX1200 and felizjueves.emit.hip, into
    #   test/golden/hip/t_12.hip
    text = hip.emit(build_t_kernel(GOLDEN_CLASS), GFX1200)
    assert text == (GOLDEN / "hip" / "t_12.hip").read_text()
