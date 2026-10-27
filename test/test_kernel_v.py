"""`kernel_v`: the nuclear-attraction kernel, `Target.V`, interpreted (#239).

THE WITNESS is `one_electron.potential.potential_block` (#235) with
`reference.pq.one_centre` injected, placed into the matrix by `kernel_jab`'s
scatter rule (the diagonal shell pair's lower triangle, mirrored). The
kernel's records are `ao.density.epair`'s `(P, zeta, s)` per primitive pair, plus
`P - A` and `P - B`; its Boys function is `reference.boys.boys`, injected
into the interpreter.

ROUND-OFF, NOT BITWISE -- the contrast with `kernel_s` / `kernel_jab`,
which are `==`. The host recurses each `[t]^(0)` along
`one_centre`'s fixed rule, memoised; the kernel runs the plan's DAG (MILD's
directions, the sieve's re-parenting) in `schedule(dag)`'s order. The two
reach the same `[t]^(0)` along different paths, so they agree to rounding
and not by construction -- which is the point of having both. The kernel
also multiplies `s` in once per row after the nucleus sum, where the host
dresses every `E` entry first.

THE BOUND is per element, `TOL_ULPS * eps * M`, with `M` the magnitude the
element is summed from:

    M[a, b] = SUM_records SUM_C |Z_C| SUM_t |s E^{ab}_t| |[t]^(0)|

computed host-side from the dressed blocks and `one_centre` (`_magnitude`).
That is the scale of a running-error bound: every term of the sum carries a
relative error of a few roundings (its `[t]` along at most `L + 1`
reduction levels of at most two terms, the three-factor `E` product, the
products by `Z` and `s`), and cancellation between terms is what makes a
bound relative to `|V|` meaningless. `TOL_ULPS = 16` is that "few", stated
before the measurement and not fitted to it; the worst observed while
building this file was about 1.6 eps M, through (ff).

THE BASIS. `reference.boys` refuses `T > 2000`, and a real core exponent
against a distant nucleus exceeds it (HF/cc-pVTZ's F 1s against H reaches
T ~ 1e5), so the per-class witness is a synthetic set of contracted shells,
s to f on three centres with moderate exponents, under four nuclei; the
whole-matrix witness is H2O/STO-3G from its tracked fixture, nuclei from
its atoms (worst T ~ 900). The libcint comparison of `V` is milestone 11's
`O1`, not this issue's.

TIERS. Classes with `la + lb <= 4` (through (dd)) are fast tier; (df), (fd)
and (ff) (`L = 5, 6`) are `slow`, per the issue. Nothing compiles and
nothing launches on a card.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l

from felizjueves.ao.density import epair
from felizjueves.ao.pairs import primitive_pairs
from felizjueves.ao.shell import Basis, Shell
from felizjueves.emit import cuda, hip
from felizjueves.emit._c import BOYS_DEVICE
from felizjueves.eri.j_coulomb.launch import per_axis
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Boys,
    Call,
    Decl,
    Fn,
    For,
    If,
    Index,
    Inline,
    Lit,
    Var,
    _subexprs,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.hermite import hermite_order
from felizjueves.mmd.recurrence import fundamental_prelude, recurrence_body
from felizjueves.one_electron.kernel_v import (
    ATOMIC_REASON,
    LOWSTACK_MIN_LSUM,
    NUC,
    ZEROED_V,
    build_v_kernel,
    launch_shape,
    v_lowstack,
    v_plan,
    v_spec,
)
from felizjueves.one_electron.potential import potential_block
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import schedule
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL_ULPS = 16  # x eps M: see the module docstring

LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]


def _cls_param(cls):
    return by_l(cls, sum(cls), id=f"{cls[0]}{cls[1]}")


A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
SYNTHETIC = Basis((
    Shell(0, A, (3.4, 0.62, 0.17), (0.15, 0.53, 0.44)),
    Shell(1, A, (1.9, 0.45), (0.38, 0.71)),
    Shell(2, A, (1.1, 0.3), (0.5, 0.6)),
    Shell(3, A, (0.8,), (1.0,)),
    Shell(0, B, (1.3, 0.21), (0.6, 0.47)),
    Shell(1, B, (0.9,), (1.0,)),
    Shell(2, B, (0.8,), (1.0,)),
    Shell(3, C, (0.55, 0.12), (0.8, 0.3)),
    Shell(2, C, (0.7,), (1.0,)),
    Shell(1, C, (0.9,), (1.0,)),
    Shell(0, C, (2.0,), (1.0,)),
))
NUCLEI = ((8.0, A), (1.0, B), (1.0, (0.5, 0.5, -0.4)), (3.0, C))


@cache
def _fixture(name: str):
    return manifest.load(manifest.FIXTURES / f"{name}.json")


def _pairs_of(basis, cls):
    """Shell pairs `i <= j` of class `cls`, in `one_electron.matrix` order."""
    sh = basis.shells
    return [(i, j) for i in range(len(sh)) for j in range(i, len(sh))
            if (sh[i].l, sh[j].l) == cls]


def _records(basis, pairs):
    """The kernel's per-record arrays for `pairs`, records in pair order and,
    within a pair, in `primitive_pairs` order."""
    sh, off = basis.shells, basis.offsets
    cols: dict[str, list] = {k: [] for k in (
        "zeta_ab", "P", "PA", "PB", "s_ab", "off_a", "off_b")}
    for i, j in pairs:
        ep = epair(sh[i], sh[j])
        for P, z, sk in ep.recs:
            cols["zeta_ab"].append(z)
            cols["P"].append(list(P))
            cols["PA"].append([P[d] - ep.A[d] for d in range(3)])
            cols["PB"].append([P[d] - ep.B[d] for d in range(3)])
            cols["s_ab"].append(sk)
            cols["off_a"].append(off[i])
            cols["off_b"].append(off[j])
    for vec, name in (("P", "p"), ("PA", "pa"), ("PB", "pb")):
        cols |= per_axis(name, cols.pop(vec))
    return {k: np.array(v, dtype=np.int32 if k.startswith("off") else float)
            for k, v in cols.items()}


def _nuc_arrays(nuclei):
    return dict(n_nuc=len(nuclei), Z=np.array([z for z, _ in nuclei]),
                Cnuc=np.array([list(c) for _, c in nuclei]))


def _launch(cls, basis, pairs, nuclei, V, threads=5, lowstack=None):
    """One launch of class `cls` over every record of `pairs`, into `V`;
    `lowstack` is `build_v_kernel`'s (None: the class's default form)."""
    rec = _records(basis, pairs)
    n_rec = len(rec["s_ab"])
    grid, block = launch_shape(n_rec, threads)
    return Interpreter(build_v_kernel(v_spec(*cls), lowstack=lowstack),
                       launch=Launch(grid, block, 1), boys=boys).run(
        n_rec=n_rec, n_ao=basis.n_ao, V=V, **_nuc_arrays(nuclei), **rec)["V"]


def _place(basis, pairs, M, block):
    """`block(sa, sb)` placed into `M` by `kernel_jab`'s scatter rule."""
    sh, off = basis.shells, basis.offsets
    for i, j in pairs:
        blk = block(sh[i], sh[j])
        for r in range(sh[i].n_cart):
            for c in range(sh[j].n_cart):
                if i == j and c > r:
                    continue  # diagonal: the lower triangle, mirrored
                M[off[i] + r, off[j] + c] = blk[r][c]
                M[off[j] + c, off[i] + r] = blk[r][c]
    return M


def _host(basis, pairs, nuclei):
    n = basis.n_ao
    return _place(basis, pairs, np.zeros((n, n)), lambda sa, sb:
                  potential_block(sa, sb, nuclei, one_centre=one_centre))


def _magnitude_block(sa, sb, nuclei):
    """`SUM_pp SUM_C |Z| SUM_t |block_pp[ab][t]| |[t]^(0)|`, per (a, b)."""
    ts = hermite_order(sa.l + sb.l)
    nb = sb.n_cart
    mag = [0.0] * (sa.n_cart * nb)
    for pp in primitive_pairs(sa, sb):
        theta = math.sqrt(pp.zeta)
        for Z, Cc in nuclei:
            pc = tuple(p - c for p, c in zip(pp.P, Cc, strict=True))
            R = [abs(one_centre(t, 0, pc, theta)) for t in ts]
            for k, row in enumerate(pp.block):
                mag[k] += abs(Z) * sum(abs(e) * r
                                       for e, r in zip(row, R, strict=True))
    return [mag[r * nb:(r + 1) * nb] for r in range(sa.n_cart)]


def _magnitude(basis, pairs, nuclei):
    n = basis.n_ao
    return _place(basis, pairs, np.zeros((n, n)), lambda sa, sb:
                  _magnitude_block(sa, sb, nuclei))


def _owned(basis, pairs) -> np.ndarray:
    sh, off = basis.shells, basis.offsets
    m = np.zeros((basis.n_ao, basis.n_ao), bool)
    for i, j in pairs:
        m[off[i]:off[i] + sh[i].n_cart, off[j]:off[j] + sh[j].n_cart] = True
    return m | m.T


def _check(got, want, mag, label):
    err = np.abs(got - want)
    worst = float(np.max(err / np.where(mag > 0, EPS * mag, 1.0)))
    print(f"\n{label}: max|V - host| = {err.max():.2e}, worst {worst:.2f} eps M")
    assert np.all(err <= TOL_ULPS * EPS * mag), label
    # the bound is not vacuous: it is far below the values it bounds
    big = np.abs(want) > 0
    assert np.max(TOL_ULPS * EPS * mag[big] / np.abs(want[big])) < 1e-10


# ---------------------------------------------------------------------------
# Per class: one launch over every record of the class
# ---------------------------------------------------------------------------


def test_the_synthetic_basis_reaches_every_class_and_a_diagonal_pair():
    for cls in CLASSES:
        pairs = _pairs_of(SYNTHETIC, cls)
        assert pairs, cls
        if cls[0] == cls[1]:
            assert any(i == j for i, j in pairs), cls
            assert any(i != j for i, j in pairs), cls


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_one_launch_over_the_class_agrees_with_the_host_to_round_off(cls):
    pairs = _pairs_of(SYNTHETIC, cls)
    n = SYNTHETIC.n_ao
    got = _launch(cls, SYNTHETIC, pairs, NUCLEI, np.zeros((n, n)))
    want = _host(SYNTHETIC, pairs, NUCLEI)
    mag = _magnitude(SYNTHETIC, pairs, NUCLEI)
    _check(got, want, mag, f"({cls[0]}{cls[1]})")
    owned = _owned(SYNTHETIC, pairs)
    assert np.all(got[~owned] == 0.0)  # exactly the pairs' blocks and mirrors
    assert np.array_equal(got, got.T)  # the scatter mirrors the same value


def test_dd_builds_and_interprets_the_full_block_and_the_L4_tower():
    """(dd): every column of the 6 x 6 block's 35 Hermite indices meets the
    `L = 4` tower. One record, one nucleus, against the host."""
    k = build_v_kernel((2, 2))
    (inl,) = [s for s in walk(k.body, into_inline=False)
              if isinstance(s, Inline)]
    assert inl.fragment == ef.block_fragment(2, 2)
    dag, _ = v_plan((2, 2))
    assert {r for r, m in dag.targets if m == 0} == set(hermite_order(4))
    assert len(hermite_order(4)) == 35
    sa = Shell(2, A, (0.9,), (1.0,))
    sb = Shell(2, B, (0.6,), (1.0,))
    basis = Basis((sa, sb))
    nuc = ((2.0, C),)
    got = _launch((2, 2), basis, [(0, 1)], nuc, np.zeros((12, 12)), threads=1)
    _check(got, _host(basis, [(0, 1)], nuc),
           _magnitude(basis, [(0, 1)], nuc), "(dd), one record")
    assert np.count_nonzero(got) == 2 * 36


def test_the_whole_matrix_on_h2o_sto3g():
    fx = _fixture("h2o_sto3g_j")
    basis = ob.from_fixture(fx)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    n = basis.n_ao
    sh = basis.shells
    V = np.zeros((n, n))
    for cls in sorted({(a.l, b.l) for k, a in enumerate(sh) for b in sh[k:]}):
        V = _launch(cls, basis, _pairs_of(basis, cls), nuclei, V)
    every = [(i, j) for i in range(len(sh)) for j in range(i, len(sh))]
    _check(V, _host(basis, every, nuclei), _magnitude(basis, every, nuclei),
           "H2O/STO-3G")
    assert np.array_equal(V, V.T)
    assert np.all(np.diag(V) < 0.0)  # positive nuclei attract


def test_nuclei_add_and_the_sign_is_attractive():
    """Two launches with one nucleus each add (to rounding) to one launch
    with both, and a single positive charge gives a negative diagonal."""
    cls = (1, 1)
    pairs = _pairs_of(SYNTHETIC, cls)
    n = SYNTHETIC.n_ao
    one = _launch(cls, SYNTHETIC, pairs, NUCLEI[:1], np.zeros((n, n)))
    two = _launch(cls, SYNTHETIC, pairs, NUCLEI[1:2], one.copy())
    both = _launch(cls, SYNTHETIC, pairs, NUCLEI[:2], np.zeros((n, n)))
    mag = _magnitude(SYNTHETIC, pairs, NUCLEI[:2])
    assert np.all(np.abs(two - both) <= TOL_ULPS * EPS * mag)
    owned = _owned(SYNTHETIC, pairs)
    d = np.diag(one)[np.diag(owned)]
    assert d.size and np.all(d < 0.0)


# ---------------------------------------------------------------------------
# The plan layer, reused byte for byte
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [_cls_param(c) for c in
                                 [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2),
                                  (3, 1), (3, 3)]])
def test_the_recurrence_is_the_plan_layers_statements_unchanged(cls):
    """The nucleus loop's body contains, contiguously, exactly
    `fundamental_prelude(ms) + recurrence_body(dag, schedule(dag), f)` for
    the one DAG `pipeline.plan` returns -- built HERE, from the plan layer,
    so a local edit to the recurrence inside `kernel_v.py` fails."""
    spec = Spec(Operator.NUCLEAR, 0, cls, Target.V)
    (dag,) = plan(spec).dags
    order = schedule(dag)
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    want = (*prelude, *recurrence_body(dag, order, fundamental))

    (loop,) = [s for s in walk(build_v_kernel(cls).body)
               if isinstance(s, For) and s.var == NUC]
    body = loop.body
    starts = [i for i in range(len(body) - len(want) + 1)
              if body[i:i + len(want)] == want]
    assert len(starts) == 1
    # ...and nothing else in the kernel declares a recurrence node
    names = {s.name for s in want if isinstance(s, Decl)}
    decls = [s for s in walk(build_v_kernel(cls).body)
             if isinstance(s, Decl) and s.name in names]
    assert len(decls) == len(names)
    vdag, vorder = v_plan(cls)  # BatchDag is eq=False: compare its fields
    assert (vdag.batch, vdag.targets, dict(vdag.parents), vorder) == (
        dag.batch, dag.targets, dict(dag.parents), order)


def test_n_batches_above_one_is_refused_at_the_consumer():
    with pytest.raises(ValueError, match="unbatched by milestone 11's "
                                         "decision"):
        build_v_kernel(Spec(Operator.NUCLEAR, 0, (1, 1), Target.V, (), 3))


def test_the_key_is_a_pair_class():
    assert v_spec(2, 1) == Spec(Operator.NUCLEAR, 0, (2, 1), Target.V)
    assert v_spec(2, 1).n_batches == 1
    assert build_v_kernel(v_spec(2, 1)) == build_v_kernel((2, 1))
    assert build_v_kernel((2, 1)).name == "v_21"
    with pytest.raises(ValueError, match="Target.V"):
        build_v_kernel(Spec(Operator.OVERLAP, 0, (1, 1), Target.S))
    with pytest.raises(ValueError):
        build_v_kernel(Spec(Operator.NUCLEAR, 0, (1, 1), Target.V,
                            ("recycle",)))


# ---------------------------------------------------------------------------
# The contract, structurally
# ---------------------------------------------------------------------------


def test_the_parameter_block():
    k = build_v_kernel((1, 2))
    assert [p.name for p in k.params] == [
        "n_rec", "n_ao", "n_nuc", "zeta_ab", "p_x", "p_y", "p_z",
        "pa_x", "pa_y", "pa_z", "pb_x", "pb_y", "pb_z", "s_ab",
        "off_a", "off_b", "Z", "Cnuc", "V"]
    (V,) = [p for p in k.params if p.name == "V"]
    assert V.intent.name == "OUT"


def test_launch_shape():
    assert launch_shape(1) == (1, (32, 1))
    assert launch_shape(33, 32) == (2, (32, 1))
    for bad in ((0, 32), (5, 0), (2.0, 32)):
        with pytest.raises(ValueError):
            launch_shape(*bad)


def test_the_nuclei_are_one_serial_loop_inside_the_record():
    k = build_v_kernel((1, 1))
    (loop,) = [s for s in walk(k.body) if isinstance(s, For)]
    assert loop.var == NUC
    assert (loop.start, loop.stop, loop.step) == (Lit(0), Var("n_nuc"), Lit(1))
    # the recurrence lives inside it, the E block and the scatter outside
    inside = list(walk(loop.body))
    assert any(isinstance(s, Decl) and s.name == "T" for s in inside)
    assert not [s for s in inside if isinstance(s, (Inline, AtomicAdd))]


@pytest.mark.parametrize("cls", [_cls_param(c) for c in
                                 [(0, 0), (1, 1), (2, 1), (2, 2), (3, 2)]])
def test_no_prefactor_is_formed_in_the_kernel(cls):
    """`s_ab` is the host's one scalar per record. Read off every `Decl`,
    `Assign` and bind: one `sqrt` (theta), the three `E` seeds' `exp`, no
    `pi` but the prelude's `2/sqrt(pi)`, and `V_ab` is `sk * acc_r`."""
    k = build_v_kernel(cls)
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
    calls = [e for e in subs if isinstance(e, Call)]
    assert sum(e.fn is Fn.SQRT for e in calls) == 1
    assert sum(e.fn is Fn.EXP for e in calls) == 3
    assert {e.fn for e in calls} <= {
        Fn.ADD, Fn.SUB, Fn.MUL, Fn.DIV, Fn.LT, Fn.SQRT, Fn.EXP,
        Fn.THREAD_X, Fn.BLOCK_X, Fn.BLOCK_DIM_X}
    floats = {e.value for e in subs
              if isinstance(e, Lit) and isinstance(e.value, float)}
    assert floats <= {0.0, 0.5, -2.0, 2.0 / math.sqrt(math.pi)}
    if v_lowstack(cls):  # one rolled value, sk * acc[i nb + j] (#586)
        (v,) = {s.init for s in stmts if isinstance(s, Decl) and s.name == "v"}
        assert v.fn is Fn.MUL and v.args[0] == Var("sk")
        assert isinstance(v.args[1], Index) and v.args[1].array == "acc"
        return
    vs = [s for s in stmts if isinstance(s, Decl) and s.name.startswith("v")
          and s.name[1:].isdigit()]
    assert len(vs) == len(ef.block_rows(*cls))
    for r, v in enumerate(vs):
        assert v.init == Call(Fn.MUL, (Var("sk"), Var(f"acc{r}")))


@pytest.mark.parametrize("cls", [(1, 0), (1, 1), (2, 2)],
                         ids=lambda c: f"{c[0]}{c[1]}")
def test_E_is_read_only_where_the_fragment_writes_it(cls):
    k = build_v_kernel(cls)
    (inl,) = [s for s in walk(k.body, into_inline=False)
              if isinstance(s, Inline)]
    written = {tuple(a.value for a in s.dst.at) for s in inl.fragment.body
               if isinstance(s, Assign)}
    read = {tuple(a.value for a in e.at)
            for s in walk(k.body, into_inline=False) if isinstance(s, Decl)
            and s.init is not None for e in _subexprs(s.init)
            if isinstance(e, Index) and e.array == ef.E_ARRAY}
    assert read == written  # every column the block writes, nothing else


@pytest.mark.parametrize("cls", [(0, 0), (1, 2), (2, 2)],
                         ids=lambda c: f"{c[0]}{c[1]}")
def test_every_write_to_V_is_an_atomic_add_with_its_reason(cls):
    k = build_v_kernel(cls)
    stmts = list(walk(k.body))
    notes = {s.prov for s in stmts if hasattr(s, "prov")}
    assert ATOMIC_REASON in notes and ZEROED_V in notes
    assert not [s for s in stmts if isinstance(s, Assign)
                and isinstance(s.dst, Index) and s.dst.array == "V"]
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert {s.dst.array for s in adds} == {"V"}
    na, nb = len(ef.cartesians(cls[0])), len(ef.cartesians(cls[1]))
    (scatter,) = [s for s in stmts if isinstance(s, If)
                  and any(isinstance(t, AtomicAdd) for t in s.then)]
    off = [s for s in scatter.then if isinstance(s, AtomicAdd)]
    diag = [s for s in scatter.orelse if isinstance(s, AtomicAdd)]
    assert len(off) == 2 * na * nb
    assert len(diag) == (na * na if cls[0] == cls[1] else 0)


# ---------------------------------------------------------------------------
# Emission: `fj_boys` exactly once, and the goldens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [(0, 0), (2, 1)], ids=lambda c: f"{c[0]}{c[1]}")
def test_the_emitted_text_carries_fj_boys_exactly_once(cls):
    k = build_v_kernel(cls)
    (fill,) = [s for s in walk(k.body) if isinstance(s, Boys)]
    assert fill.top == sum(cls)  # one fill, every order of the class
    for text in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)):
        # the one template every kernel carries (#301): T = zeta |P - C|^2
        # is undamped, and it holds on all of reference.boys's domain
        assert text.count(BOYS_DEVICE) == 1
        assert text.count(f"fj_boys<{sum(cls)}>(T, boys);") == 1


GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLASS = (1, 1)


def test_the_cuda_golden_is_unchanged() -> None:
    # Regenerate (never hand-edit):
    #   PYTHONPATH=src uv run python -c "from felizjueves.one_electron.kernel_v import \
    #   build_v_kernel as b; from felizjueves.performance.device import SM_86; \
    #   from felizjueves.emit.cuda import emit; print(emit(b((1, 1)), SM_86), \
    #   end='')" > test/golden/cuda/v_11.cu
    text = cuda.emit(build_v_kernel(GOLDEN_CLASS), SM_86)
    assert text == (GOLDEN / "cuda" / "v_11.cu").read_text()


def test_the_hip_golden_is_unchanged() -> None:
    #   ... the same with GFX1200 and felizjueves.emit.hip, into
    #   test/golden/hip/v_11.hip
    text = hip.emit(build_v_kernel(GOLDEN_CLASS), GFX1200)
    assert text == (GOLDEN / "hip" / "v_11.hip").read_text()


# ---------------------------------------------------------------------------
# The low-stack form (#586): at (ff) by default, every other text unchanged
# ---------------------------------------------------------------------------

# sha256 of the unrolled V text of every class below la + lb = 6 -- through
# (dd), and the mixed f classes -- cuda and hip, as emitted at cc5d16f
# (before #586): the low-stack form must not move a byte of it.
UNROLLED_DIGESTS = {
    (0, 0): ("b7180b64ee2c74943cdf75e4e73e7fe846af5eba3835f9d2140dd65969f7124e",
             "842b0bd01c614451d087494c57c48b67387cee6f18da882485a350163c138403"),
    (0, 1): ("a3bfb3693d6e36f1b3bafc935b0a3403db8ab7ad5ad004337c3d9d6adf2b5fc7",
             "193c25a62625b7f958e11c817d0aef35ffc5398cfc7fe36564280f97c9a4b429"),
    (0, 2): ("b64c919ec89930e99ac69d1f96e67b1c78ea54d0bf7cdb2521d04f7367996244",
             "2bcbf1d1210e8b5fc2cea50238b66de2511e3d751c6a6d9fb72678b41ae79160"),
    (1, 0): ("090760a4286d34feb149463a700aacc8ac54b32b8f809cab81735a4b501501ed",
             "f51a898e8a7f1c8762a933fc5994afad2fd076313dab8ae3bc05c41bc57eb14b"),
    (1, 1): ("18984ed53b0da485d44fc59938d8658b7364106fa6fb9e34a6fcc57cf06c484f",
             "fabecff37d7d647b86c42785f8164389c456f61b789f1356c1c3cfa497acb4a8"),
    (1, 2): ("28459ee7350cd0dbcc134c3c1a2162a179ed4da3189767f63d55765de926ba1d",
             "f65855fd351c47db944315a87f87aac5455b7701b7262599f238c67f22da56d5"),
    (2, 0): ("bc73f12f5353e890c19d6a611e88addfef92570bc44580f799461b31ffb7edd8",
             "bcc163887e7c38468949e94379e7021c51889f678728fd2b72ee0e2dccd69ea9"),
    (2, 1): ("ff7f2c23cb0f412c3883384dd7a23076dbea871e2c2bce941335878bb5611693",
             "a99288902121639ae99a6a752d9f6711b108233d33e40e8706245cb8f46c4670"),
    (2, 2): ("0cf5a7dccc8790ac98080ee2d2fd28b65d8cb624c716e1c0487200e080028184",
             "7a935c294474cc0f112cbc5adb62e604868da3ce02e990d4136c4affed18fc0a"),
    (0, 3): ("ff16f783741ebd92d2908c6bc2ef37f94bfb41c01f9e3f09bfba6ac9ded73ae2",
             "d97afe8be5fc60b20256c823c2c021acc4b8f94bb6bf49fd67cc4a1d34782f71"),
    (1, 3): ("85fd9b1366b4e9895875a5e0a6d330fd439b489d0df79927c6662a1f6c937062",
             "13ae0d8d557e0e7c6a96ddaf2c952c11e0106fb38f6888f8461f9d9e09e23cde"),
    (2, 3): ("54275f35c933f3a4fb0c1045065a5feb00810c3d3eae869ba4881a38e78272ad",
             "f446e3ef3e395e1aea3f1c4c1e8b91627edcce8dc1b959931697ff5dacfea99f"),
    (3, 0): ("280171a1c26dd9873c4e630b2021500aa7667864dbe14a1d9a8e830d0169cfd3",
             "06dc7a12d084ac99c5f990640be0f21f4ec2be33b380ff40c29c5263714ca460"),
    (3, 1): ("cee007584c2a3f841c60a8c447b3a88e16b1fe22c91bcae7a677b20c8ab2f327",
             "30ab756b5e14a97fb6909644eecec1351e9ac59f760b1b429b5b588694e810c1"),
    (3, 2): ("f022e68f8bd1d7f941feb88464f6a66aa507dd531f75a71b3a0f982cd7210442",
             "392869757dedfa188228ed6d6dbea838df603ac966e7bf312c50d1c76b0669ae"),
}

# #586's budget: at most 8 KB of stack per thread at (ff) on sm_86. The
# kernel's local arrays are the whole of what it can spill by design, so
# they are held to it here, without a compiler (the measured frame is in
# docs/implementation/one_electron.md).
STACK_BUDGET = 8192


def test_the_selection_rule_is_the_hermite_order_six():
    # (ff) alone through f; past f by la + lb, so (gs) and (gp) unrolled
    assert LOWSTACK_MIN_LSUM == 6
    for la in range(5):
        for lb in range(5):
            assert v_lowstack((la, lb)) is (la + lb >= 6)
            assert v_lowstack(v_spec(la, lb)) is v_lowstack((la, lb))
    assert [c for c in CLASSES if v_lowstack(c)] == [(3, 3)]


def test_every_unrolled_default_through_f_is_pinned():
    assert sorted(UNROLLED_DIGESTS) == [c for c in CLASSES if not v_lowstack(c)]


@pytest.mark.parametrize("cls", sorted(UNROLLED_DIGESTS),
                         ids=lambda c: f"{c[0]}{c[1]}")
def test_the_unrolled_text_is_byte_identical(cls):
    import hashlib
    k = build_v_kernel(cls)
    assert k == build_v_kernel(cls, lowstack=False)
    got = tuple(hashlib.sha256(t.encode()).hexdigest()
                for t in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)))
    assert got == UNROLLED_DIGESTS[cls]


@pytest.mark.parametrize("cls", [(1, 1), (2, 3), (3, 3)],
                         ids=lambda c: f"{c[0]}{c[1]}")
def test_both_forms_share_the_name_parameters_and_launch(cls):
    lo = build_v_kernel(cls, lowstack=True)
    hi = build_v_kernel(cls, lowstack=False)
    assert lo.name == hi.name == f"v_{cls[0]}{cls[1]}"
    assert lo.params == hi.params
    assert build_v_kernel(cls) == (lo if v_lowstack(cls) else hi)
    rec = [s for s in lo.body if isinstance(s, Decl) and s.name == "rec"]
    assert rec == [s for s in hi.body if isinstance(s, Decl) and s.name == "rec"]


def _local_doubles(k) -> dict[str, int]:
    return {s.name: math.prod(d.value for d in s.shape)
            for s in walk(k.body) if isinstance(s, Decl) and s.shape}


@pytest.mark.parametrize("cls", [(0, 3), (1, 3), (2, 3), (3, 3), (3, 0)],
                         ids=lambda c: f"{c[0]}{c[1]}")
def test_the_lowstack_form_holds_no_e_block(cls):
    la, lb = cls
    k = build_v_kernel(cls, lowstack=True)
    rows, cols = ef.block_shape(la, lb)
    arrays = _local_doubles(k)
    ext = ef.axis_table_extent(la, lb)
    assert arrays == {"Ex": ext, "Ey": ext, "Ez": ext, "acc": rows,
                      "R": cols, "boys": la + lb + 1}
    assert 8 * sum(arrays.values()) <= STACK_BUDGET
    assert ef.E_ARRAY not in arrays
    (inl,) = [s for s in walk(k.body, into_inline=False)
              if isinstance(s, Inline)]
    assert inl.fragment == ef.axis_tables_fragment(la, lb)
    # the recurrence is still the plan's, inside the one nucleus loop
    (loop,) = [s for s in walk(k.body) if isinstance(s, For) and s.var == NUC]
    assert not [s for s in walk(loop.body) if isinstance(s, (Inline, AtomicAdd))]


def test_the_ff_lowstack_arrays_are_527_doubles():
    # 3 x 112 axis-table entries, 100 accumulators, 84 [t]^(0), boys[7]:
    # 4,216 B, of which sm_86's frame holds all but boys (4,160 B measured).
    assert sum(_local_doubles(build_v_kernel((3, 3))).values()) == 527


def _other_form(cls):
    return not v_lowstack(cls)


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_the_other_form_agrees_with_the_host_to_round_off(cls):
    """Every class through (ff) in the form `build_v_kernel` does NOT
    choose by default -- the low-stack form below (ff), the unrolled one at
    (ff) -- against the host, at the same `TOL_ULPS`: with the default-form
    test above, both forms are held at every class."""
    pairs = _pairs_of(SYNTHETIC, cls)
    n = SYNTHETIC.n_ao
    got = _launch(cls, SYNTHETIC, pairs, NUCLEI, np.zeros((n, n)),
                  lowstack=_other_form(cls))
    want = _host(SYNTHETIC, pairs, NUCLEI)
    mag = _magnitude(SYNTHETIC, pairs, NUCLEI)
    _check(got, want, mag, f"({cls[0]}{cls[1]}) other form")
    assert np.all(got[~_owned(SYNTHETIC, pairs)] == 0.0)
    assert np.array_equal(got, got.T)


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_the_two_forms_agree_with_each_other(cls):
    pairs = _pairs_of(SYNTHETIC, cls)
    n = SYNTHETIC.n_ao
    lo = _launch(cls, SYNTHETIC, pairs, NUCLEI, np.zeros((n, n)),
                 lowstack=True)
    hi = _launch(cls, SYNTHETIC, pairs, NUCLEI, np.zeros((n, n)),
                 lowstack=False)
    mag = _magnitude(SYNTHETIC, pairs, NUCLEI)
    assert np.all(np.abs(lo - hi) <= TOL_ULPS * EPS * mag)


LOWSTACK_GOLDEN_CLASS = (3, 3)


def test_the_lowstack_cuda_golden_is_unchanged() -> None:
    # Regenerate (never hand-edit), the default form at (ff):
    #   PYTHONPATH=src uv run python -c "from felizjueves.one_electron.kernel_v import \
    #   build_v_kernel as b; from felizjueves.performance.device import SM_86; \
    #   from felizjueves.emit.cuda import emit; print(emit(b((3, 3)), SM_86), \
    #   end='')" > test/golden/cuda/v_33.cu
    text = cuda.emit(build_v_kernel(LOWSTACK_GOLDEN_CLASS), SM_86)
    assert text == (GOLDEN / "cuda" / "v_33.cu").read_text()


def test_the_lowstack_hip_golden_is_unchanged() -> None:
    #   ... the same with GFX1200 and felizjueves.emit.hip, into
    #   test/golden/hip/v_33.hip
    text = hip.emit(build_v_kernel(LOWSTACK_GOLDEN_CLASS), GFX1200)
    assert text == (GOLDEN / "hip" / "v_33.hip").read_text()
