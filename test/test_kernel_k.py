"""`kernel_k`: the exchange kernel, the staged contraction as IR (#198).

The kernel is checked against the HOST staged contraction it implements --
`eri.k_exchange.k.k_quartet` (#197), the `k=` seam's reference -- interpreted, on
synthetic contracted shells (two primitives each, one with a negative
coefficient, four distinct centres), no fixture and no PySCF.

EXACT EQUALITY, NOT A TOLERANCE. The kernel performs the host's operations in
the host's order (`kernel_k`'s module docstring: stage 1 a left fold over `q`
from `+0.0`, stage 2 a left fold over `p`, stage 3 `K += D * V`, the absent
structural zeros an identity), and its `E` is the fragment that is bitwise
`mmd.hermite.e_block` (test/test_e_fragment.py). Two things make the
comparison bitwise and are stated rather than hidden:

- `pq=` is the RECURRENCE'S OWN `[p+q]^(0)`: `recurrence.build_recurrence_kernel`
  over the same plan DAG, interpreted with the same injected Boys, with the
  sign `(-1)^|q|` applied exactly. So this checks the digestion, the
  dressing, the layout and the screen given the recurrence; the recurrence
  itself against `reference/` is test_kernel.py's and test_recurrence's
  business, and the plan-untouched test below says it is the same program.
- ONE LANE, and one bra record per comparison: the kernel writes one
  `K[a, b]` block per bra record (J's mapping), so each row is compared with
  `k_quartet` on a bra pair of that one record; with an uncontracted bra pair
  that IS the whole quartet. With more lanes the warp butterfly reassociates
  the per-lane partial sums (`test_more_lanes_reassociate_only`).

THE L <= 2 RULE (milestone 10's standing instruction: no class with
`L = la + lc + lb + ld > 2` is built, interpreted or run before #231). Every
K class with L <= 2 runs here, in both bra/ket placements; the classes above
it through `(dd|dd)` are written and marked `slow`, deferred to #231. The
footprint table through `(dd|dd)` and the class where 48 KiB stops fitting
are ARITHMETIC of the class (`kernel_k.shared_doubles`), built from nothing.

Fast tier: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from functools import cache

import numpy as np
import pytest

from felizjueves.ao.density import EPair, epair
from felizjueves.ao.shell import Shell
from felizjueves.eri.k_exchange import kernel_k
from felizjueves.eri.k_exchange.k import k_quartet
from felizjueves.eri.k_exchange.kernel_k import (
    KET,
    KSH,
    NO_DENSITY_KEY,
    SORTED_BY_IQ,
    build_k_kernel,
    k_class_spec,
    launch_shape,
    shared_bytes,
    shared_doubles,
)
from felizjueves.ir import (
    AtomicAdd,
    Call,
    Decl,
    Fn,
    For,
    Inline,
    Lit,
    Note,
    Raw,
    Space,
    WarpReduce,
    _exprs_of,
    _subexprs,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.kernel import build_pq_kernel, k_spec
from felizjueves.mmd.kernel import launch_shape as j_launch_shape
from felizjueves.mmd.recurrence import build_recurrence_kernel, target_order
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

# Every K class with L <= 2 -- the fast tier, both placements.
FAST: list[Am] = [am for am in itertools.product(range(3), repeat=4)
                  if sum(am) <= 2]
# Above L = 2, through (dd|dd): written, `slow`, deferred to #231.
DEFERRED: list[Am] = [(1, 1, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1),
                      (2, 0, 0, 2), (2, 2, 0, 0), (0, 0, 2, 2), (2, 1, 1, 0),
                      (2, 1, 2, 1), (2, 2, 1, 1), (2, 2, 2, 2)]


def _id(am: Am) -> str:
    sym = "spdfg"
    return "({}{}|{}{})".format(*(sym[x] for x in am))


CLASSES = ([pytest.param(am, id=_id(am)) for am in FAST]
           + [pytest.param(am, id=_id(am), marks=pytest.mark.slow)
              for am in DEFERRED])


def test_the_fast_set_is_every_class_with_l_at_most_2() -> None:
    assert len(FAST) == 15 and all(sum(am) <= 2 for am in FAST)
    assert all(sum(am) > 2 and max(am) <= 2 for am in DEFERRED)
    # both placements: a d (or p) on each of the four slots
    for slot in range(4):
        assert any(am[slot] == 2 for am in FAST)


# ---------------------------------------------------------------------------
# Synthetic shells: contracted, a negative coefficient, four centres
# ---------------------------------------------------------------------------

CENTRES = {"A": (0.0, 0.0, 0.0), "C": (0.3, -0.2, 0.9),
           "B": (1.1, 0.4, -0.3), "D": (-0.5, 0.8, 0.2),
           "E": (0.7, -0.6, -0.4)}
PRIMS = {0: ((1.3, 0.4), (0.6, 0.5)),
         1: ((0.9, 0.3), (0.7, -0.4)),
         2: ((0.8, 0.25), (0.5, 0.6))}


def _shell(l: int, at: str, *, contracted: bool = True) -> Shell:  # noqa: E741
    ex, co = PRIMS[l]
    n = 2 if contracted else 1
    return Shell(l, CENTRES[at], ex[:n], co[:n])


def _pairs(am: Am, *, contracted: bool = True) -> tuple[EPair, EPair]:
    la, lc, lb, ld = am
    bra = epair(_shell(la, "A", contracted=contracted),
                _shell(lc, "C", contracted=contracted))
    ket = epair(_shell(lb, "B"), _shell(ld, "D"))
    return bra, ket


def _dcd(lc: int, ld: int, shift: float = 0.0) -> list[float]:
    n = (lc + 1) * (lc + 2) // 2 * ((ld + 1) * (ld + 2) // 2)
    return [0.31 - 0.17 * i + 0.045 * i * i + shift for i in range(n)]


def _one(bra: EPair, i: int) -> EPair:
    return dataclasses.replace(bra, recs=(bra.recs[i],))


# ---------------------------------------------------------------------------
# The pq the kernel computes: the recurrence over the same plan, interpreted
# ---------------------------------------------------------------------------


@cache
def _recurrence(am: Am):
    (dag,) = plan(k_class_spec(*am)).dags
    rk = build_recurrence_kernel(dag, schedule(dag))
    return rk, {r: i for i, r in enumerate(target_order(dag))}


def kernel_pq(am: Am):
    """`[p|q] = (-1)^|q| [p+q]^(0)`, the `[r]^(0)` the kernel's own
    recurrence computes (same DAG, same schedule, same interpreter, same
    injected Boys)."""
    rk, where = _recurrence(am)
    memo: dict = {}

    def pq(p, q, PQv, theta) -> float:
        key = (tuple(PQv), theta)
        if key not in memo:
            out = Interpreter(rk, boys=boys).run(
                PQ_x=PQv[0], PQ_y=PQv[1], PQ_z=PQv[2], theta=theta,
                out=np.zeros(len(where)))["out"]
            memo[key] = out
        r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
        v = float(memo[key][where[r]])
        return -v if sum(q) % 2 else v

    return pq


# ---------------------------------------------------------------------------
# A launch: one bra pair's records, kets of one or more ket pairs
# ---------------------------------------------------------------------------


def run_k(am: Am, bra: EPair, kets: list[tuple[EPair, int]],
          dcd_rows: list[list[float]], *, lanes: int = 1, dc: int = 0,
          tau: float = 0.0, maxD: float = 1.0, Ip: list[float] | None = None,
          Iq: list[float] | None = None) -> tuple[np.ndarray, Interpreter]:
    """`K[n_bra, n_a n_b]` from the interpreted kernel, and its interpreter.
    `kets` is `(ket pair, dd)` per ket shell pair, in launch order."""
    args = k_args(am, bra, kets, dcd_rows, dc=dc, tau=tau, maxD=maxD, Ip=Ip,
                  Iq=Iq)
    grid, block = launch_shape(args["n_bra"], lanes)
    it = Interpreter(build_k_kernel(am), launch=Launch(grid, block, lanes),
                     boys=boys)
    return it.run(**args)["K"], it


def k_args(am: Am, bra: EPair, kets: list[tuple[EPair, int]],
           dcd_rows: list[list[float]], *, dc: int = 0, tau: float = 0.0,
           maxD: float = 1.0, Ip: list[float] | None = None,
           Iq: list[float] | None = None) -> dict:
    """The kernel's arguments for one launch (`K` NaN-filled: the one-batch
    kernel writes every entry; a split class's caller zeroes it)."""
    P = [rec[0] for rec in bra.recs]
    krecs = [(rec, ket, dd) for ket, dd in kets for rec in ket.recs]
    n_bra, n_ket = len(P), len(krecs)

    def diff(xs, c):
        return np.array([[x[d] - c[d] for d in range(3)] for x in xs])

    Q = [rec[0] for rec, _, _ in krecs]
    args = dict(
        n_bra=n_bra, n_ket=n_ket, n_dc=len(dcd_rows), tau=tau, maxD=maxD,
        P=np.array(P), zeta_ac=np.array([z for _, z, _ in bra.recs]),
        XPA=diff(P, bra.A), XPC=diff(P, bra.B),
        s_ac=np.array([s for _, _, s in bra.recs]),
        Ip=np.array(Ip if Ip is not None else [1.0] * n_bra),
        dc=np.full(n_bra, dc, np.int32),
        Q=np.array(Q), eta=np.array([rec[1] for rec, _, _ in krecs]),
        XQB=np.array([[rec[0][d] - ket.A[d] for d in range(3)]
                      for rec, ket, _ in krecs]),
        XQD=np.array([[rec[0][d] - ket.B[d] for d in range(3)]
                      for rec, ket, _ in krecs]),
        s_bd=np.array([rec[2] for rec, _, _ in krecs]),
        Iq=np.array(Iq if Iq is not None else [1.0] * n_ket),
        dd=np.array([dd for _, _, dd in krecs], np.int32),
        Dcd=np.array(dcd_rows, dtype=np.float64),
        K=np.full((n_bra, len(_flat(am))), np.nan),
    )
    return args


def _flat(am: Am) -> list[tuple[int, int]]:
    la, _, lb, _ = am
    return [(i, j) for i in range(len(ef.cartesians(la)))
            for j in range(len(ef.cartesians(lb)))]


def _host(bra: EPair, ket: EPair, dcd, am: Am) -> list[float]:
    X = k_quartet(bra, ket, dcd, pq=kernel_pq(am))
    return [v for row in X for v in row]


# ---------------------------------------------------------------------------
# Interpreted against eri.k_exchange.k.k_quartet, bitwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("am", CLASSES)
def test_each_bra_record_is_the_host_staged_quartet_bitwise(am: Am) -> None:
    """Contracted shells on both sides (4 bra records, 4 ket records): row
    `i` of `K` is `k_quartet` on bra record `i` and every ket record,
    bit for bit."""
    bra, ket = _pairs(am)
    dcd = _dcd(am[1], am[3])
    K, _ = run_k(am, bra, [(ket, 0)], [dcd])
    assert K.shape == (len(bra.recs), len(_flat(am)))
    for i in range(len(bra.recs)):
        assert K[i].tolist() == _host(_one(bra, i), ket, dcd, am), i
    assert np.any(K != 0.0)


@pytest.mark.parametrize("am", CLASSES)
def test_an_uncontracted_bra_is_the_whole_quartet_bitwise(am: Am) -> None:
    bra, ket = _pairs(am, contracted=False)
    assert len(bra.recs) == 1 and len(ket.recs) == 4
    dcd = _dcd(am[1], am[3], shift=-0.2)
    K, _ = run_k(am, bra, [(ket, 0)], [dcd])
    assert K[0].tolist() == _host(bra, ket, dcd, am)


@pytest.mark.parametrize("am", CLASSES)
def test_against_the_reference_pq_to_the_quartet_scale(am: Am) -> None:
    """The second ancestry: `k_quartet` on `reference.pq` (the direct
    recursion, no MILD, no DAG) agrees with the kernel to `1e-14` of the
    per-element scale `eri.k_exchange.k` defines for a quartet (test_ao_k.py's
    tolerance, inherited) -- so the recurrence-derived `pq=` above hides
    no sign or index error."""
    from felizjueves.eri.k_exchange.k import _k_quartet
    from felizjueves.reference.pq import pq

    bra, ket = _pairs(am, contracted=False)
    dcd = _dcd(am[1], am[3])
    K, _ = run_k(am, bra, [(ket, 0)], [dcd])
    X, S = _k_quartet(bra, ket, dcd, pq=pq, want_scale=True)
    X, S = np.array(X).ravel(), np.array(S).ravel()
    assert np.all(np.abs(K[0] - X) <= 1e-14 * S)


def test_the_density_row_is_dc_plus_dd() -> None:
    """D_cd straddles the records: its row is `dc[bra] + dd[k]`. Two ket
    shell pairs `(b, d1)`, `(b, d2)` in one launch; with one of their rows of
    `Dcd` zero the launch is exactly the other's quartet (adding `+-0.0` is
    the identity), and an offset `dc` skips a junk row."""
    am = (1, 0, 0, 1)
    bra, ket1 = _pairs(am)
    ket2 = epair(_shell(0, "B"), _shell(1, "E"))
    d1, d2 = _dcd(0, 1), _dcd(0, 1, shift=0.4)
    zero, junk = [0.0] * len(d1), [1e3] * len(d1)
    kets = [(ket1, 0), (ket2, 1)]
    K1, _ = run_k(am, bra, kets, [junk, d1, zero], dc=1)
    K2, _ = run_k(am, bra, kets, [zero, d2])
    for i in range(len(bra.recs)):
        assert K1[i].tolist() == _host(_one(bra, i), ket1, d1, am)
        assert K2[i].tolist() == _host(_one(bra, i), ket2, d2, am)


@pytest.mark.parametrize("lanes", [4, 32])
def test_more_lanes_reassociate_only(lanes: int) -> None:
    """One ket record: the other lanes hold `+0.0` and the butterfly adds it
    exactly, so the result is bitwise the one-lane one. Four: the per-lane
    partial sums are reassociated, so rounding, nothing more."""
    am = (1, 0, 1, 0)
    bra, ket = _pairs(am)
    dcd = _dcd(0, 0)
    one = dataclasses.replace(ket, recs=ket.recs[:1])
    a, _ = run_k(am, bra, [(one, 0)], [dcd], lanes=lanes)
    b, _ = run_k(am, bra, [(one, 0)], [dcd], lanes=1)
    assert a.tolist() == b.tolist()
    a, _ = run_k(am, bra, [(ket, 0)], [dcd], lanes=lanes)
    b, _ = run_k(am, bra, [(ket, 0)], [dcd], lanes=1)
    assert np.max(np.abs(a - b)) <= 1e-15 * np.max(np.abs(b))


def test_the_screen_is_a_break_on_kets_sorted_by_iq_alone() -> None:
    """`Ip * Iq * maxD < tau` is a break: with kets sorted descending by
    `Iq`, the launch is exactly the launch over the kets above the cut, and
    the loop is entered once more per bra (the iteration that breaks)."""
    am = (1, 0, 1, 0)
    bra, ket = _pairs(am)
    dcd = _dcd(0, 0)
    Iq = [4.0, 3.0, 2.0, 1.0]
    K, it = run_k(am, bra, [(ket, 0)], [dcd], tau=2.5 * 0.5 * 2.0, maxD=2.0,
                  Ip=[0.5] * 4, Iq=Iq)
    head = dataclasses.replace(ket, recs=ket.recs[:2])
    want, it0 = run_k(am, bra, [(head, 0)], [dcd])
    assert K.tolist() == want.tolist()
    assert it.iterations[KET] == 4 * 3 and it0.iterations[KET] == 4 * 2
    # at tau = 0 nothing is skipped
    _, full = run_k(am, bra, [(ket, 0)], [dcd], maxD=2.0, Ip=[0.5] * 4, Iq=Iq)
    assert full.iterations[KET] == 4 * 4


# ---------------------------------------------------------------------------
# The plan is untouched: the [p|q] program is J's
# ---------------------------------------------------------------------------


def _ket_loop(kern) -> tuple:
    (loop,) = [s for s in walk(kern.body) if isinstance(s, For)
               and s.var == KET]
    return loop.body


@pytest.mark.parametrize("am", CLASSES)
def test_the_plan_is_untouched(am: Am) -> None:
    """The plan of `Target.K` with `am` is the plan of `Target.J` with the
    same `am` -- DAG, schedule, `PlanCost` -- and the lowered `[p|q]` program
    in the kernel (PQ, theta, the fundamentals' prelude, every recurrence
    node, in order) is `==` the one `build_pq_kernel` emits for it."""
    ks = k_class_spec(*am)
    js = Spec(Operator.COULOMB, 0, am, Target.J)
    kp, jp = plan(ks), plan(js)
    assert kp.cost == jp.cost
    assert len(kp.dags) == len(jp.dags) == 1
    kd, jd = kp.dags[0], jp.dags[0]
    assert (kd.batch, kd.targets, dict(kd.parents)) == (
        jd.batch, jd.targets, dict(jd.parents))
    assert schedule(kd) == schedule(jd)
    jl, kl = _ket_loop(build_pq_kernel(js)), _ket_loop(build_k_kernel(am))
    # J's ket body: bound, Note, If(break), then the [p|q] program up to its
    # contraction Note. K's carries the same program at the same place.
    end = next(i for i, s in enumerate(jl)
               if isinstance(s, Note) and s.prov.startswith("J_p +="))
    program = jl[3:end]
    assert any(isinstance(s, Decl) and s.name.startswith("R") for s in program)
    assert kl[3:end] == program


# ---------------------------------------------------------------------------
# Structure: the Decl set, two fragments, no dressing, nothing vendor-specific
# ---------------------------------------------------------------------------


def _counts(am: Am) -> dict[str, int]:
    la, lc, lb, ld = am
    nx = [len(ef.cartesians(x)) for x in am]
    return dict(n_ac=nx[0] * nx[1], n_bd=nx[2] * nx[3],
                n_p=len(hermite_targets(la + lc)),
                n_q=len(hermite_targets(lb + ld)), n_ab=nx[0] * nx[2])


@pytest.mark.parametrize("am", [pytest.param(a, id=_id(a)) for a in FAST])
def test_the_decl_set_holds_no_materialised_v(am: Am) -> None:
    """The arrays are exactly `E^ac` and `E^bd` (local, one bra / one ket
    record), the prelude's `boys` (local, F_0..F_L, #301) and the shared
    `I` + `K` column; there is no array whose extent
    is a product of the two Cartesian pair-block sizes (`V`, `n_ac x n_bd`)
    nor an `I` over the `(b,d)` rows (`n_p x n_bd`), and no `V` at all."""
    c = _counts(am)
    kern = build_k_kernel(am)
    decls = [s for s in walk(kern.body) if isinstance(s, Decl)]
    arrays = {(s.name, s.shape, s.space) for s in decls if s.shape}
    assert arrays == {
        ("Eac", (Lit(c["n_ac"]), Lit(c["n_p"])), Space.LOCAL),
        ("Ebd", (Lit(c["n_bd"]), Lit(c["n_q"])), Space.LOCAL),
        (KSH, (Lit(c["n_p"] + c["n_ab"]), Call(Fn.BLOCK_DIM_Y, ())),
         Space.SHARED),
        ("boys", (Lit(sum(am) + 1),), Space.LOCAL),
    }
    assert c["n_p"] + c["n_ab"] == shared_doubles(am)
    assert not any(s.name.startswith("V") for s in decls)


@pytest.mark.parametrize("am", [pytest.param(a, id=_id(a)) for a in FAST])
def test_two_undressed_fragments_and_nothing_vendor_specific(am: Am) -> None:
    """Exactly two `Inline`s, the undressed blocks `(la, lc)` and `(lb, ld)`;
    no float literal that could be a dressing, a weight or a coefficient
    (only the accumulators' `0.0`, `1/2zeta`'s `0.5` and the recurrence's
    `-2` and `2/sqrt(pi)`, which `Target.J` has too); no `Raw`, no atomic;
    one `WarpReduce` per `(a, b)`."""
    la, lc, lb, ld = am
    kern = build_k_kernel(am)
    stmts = list(walk(kern.body))
    inl = [s.fragment for s in stmts if isinstance(s, Inline)]
    assert inl == [ef.block_fragment(la, lc), ef.block_fragment(lb, ld)]
    lits = {e.value for s in stmts for root in _exprs_of(s)
            if root is not None for e in _subexprs(root)
            if isinstance(e, Lit) and isinstance(e.value, float)}
    assert lits <= {0.0, 0.5, -2.0, 2.0 / math.sqrt(math.pi)}
    assert not any(isinstance(s, (Raw, AtomicAdd)) for s in stmts)
    assert sum(isinstance(s, WarpReduce) for s in stmts) == _counts(am)["n_ab"]


def test_the_notes_carry_the_sort_and_the_missing_density_key() -> None:
    texts = [s.prov for s in walk(build_k_kernel((1, 0, 0, 1)).body)
             if isinstance(s, Note)]
    assert SORTED_BY_IQ in texts and NO_DENSITY_KEY in texts
    assert "Iq ALONE" in SORTED_BY_IQ
    assert "I_ket * Dmax_ket" in NO_DENSITY_KEY and "straddles" in NO_DENSITY_KEY
    assert "[Fock, §4.3.1]" in NO_DENSITY_KEY


# ---------------------------------------------------------------------------
# The key, the launch, the refusals
# ---------------------------------------------------------------------------


def test_the_key_is_the_hermite_quartet() -> None:
    am = (1, 0, 0, 1)
    assert k_class_spec(*am) == Spec(Operator.COULOMB, 0, am, Target.K)
    assert build_k_kernel(k_class_spec(*am)) == build_k_kernel(am)
    assert build_k_kernel(am).name == "k_10_01"
    assert tuple(p.name for p in build_k_kernel(am).params) == kernel_k.PARAMS
    assert build_k_kernel(k_spec(2)) == build_k_kernel(k_spec(2).am)
    # the AO order (la, lb, lc, ld) is a different class: a different split
    assert build_k_kernel((1, 0, 0, 1)) != build_k_kernel((1, 0, 1, 0))
    with pytest.raises(ValueError, match="not a Target.K"):
        build_k_kernel(Spec(Operator.COULOMB, 0, am, Target.J))
    with pytest.raises(ValueError, match="n_batches = 3"):
        build_k_kernel(Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K,
                            n_batches=3))
    with pytest.raises(TypeError):
        build_k_kernel(2)  # type: ignore[arg-type]


def test_the_builder_reaches_neither_its_witness_nor_a_machine() -> None:
    """TRANSITIVELY (test_layering.py's graph, its `k_builder` rule): no
    chain of imports from `kernel_k` reaches the AO layer it is checked
    against (`eri.k_exchange.k.k_quartet`), the interpreter, reference/, a backend or a
    card."""
    import test_layering as tl

    graph = tl._graph()
    seen: set[str] = set()
    todo = ["eri.k_exchange.kernel_k"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    for banned in ("reference", "ao", "ir.interp", "emit", "performance.device"):
        hit = sorted(e for e in seen if tl._under(e, banned))
        assert not hit, f"kernel_k reaches {hit}"
    assert {"mmd.e_fragment", "mmd.recurrence", "mmd.kernel"} <= seen


def test_one_batch_delegates_and_a_split_is_build_k_kernels() -> None:
    """#203: `build_k_kernels` at one batch is `(build_k_kernel,)`, a plain
    write; `build_k_kernel` refuses a split and names the builder that takes
    it. The split class itself is test_kernel_k_batched.py's."""
    from felizjueves.eri.k_exchange.kernel_k import build_k_kernels

    for am in [(0, 0, 0, 0), (1, 0, 1, 0), (0, 2, 0, 0)]:
        spec = k_class_spec(*am)
        assert build_k_kernels(spec) == (build_k_kernel(am),)
    with pytest.raises(ValueError, match="build_k_kernels builds it"):
        build_k_kernel(Spec(Operator.COULOMB, 0, (1, 0, 0, 0), Target.K,
                            n_batches=3))
    with pytest.raises(ValueError, match="not a Target.K"):
        build_k_kernels(Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J))
    with pytest.raises(TypeError):
        build_k_kernels((1, 0, 1, 0))  # type: ignore[arg-type]


def test_the_launch_is_js() -> None:
    for n_bra, lanes in [(1, 32), (7, 32), (5, 1)]:
        assert launch_shape(n_bra, lanes) == j_launch_shape(n_bra, lanes)
        assert launch_shape(n_bra, lanes) == (n_bra, (1, lanes))


# ---------------------------------------------------------------------------
# The shared footprint: arithmetic of the class, no kernel built
# ---------------------------------------------------------------------------

LIMIT = 48 * 1024  # bytes a block gets on both of this box's cards
LANES = 32  # sm_86's warp and gfx1200's wave32 (device.Arch.wavefront)


def test_the_footprint_through_dd_dd_fits_and_where_it_stops() -> None:
    """`(n_p + n_a n_b) * lanes * 8`: through `(dd|dd)` at most 71 doubles
    per lane, 18,176 bytes; every class with shells through `f` fits (the
    largest, `(ff|ff)`, 184 doubles, 47,104 bytes); the first that does not
    is `(gs|gs)` at L = 8 (35 + 225 = 260 doubles, 66,560 bytes), and on
    `k_spec`'s ladder L = 13, `(gf|ff)`. Arithmetic only: nothing built."""
    def nx(lmax):
        return itertools.product(range(lmax + 1), repeat=4)

    assert max(shared_doubles(am) for am in nx(2)) == 71
    assert shared_bytes((2, 2, 2, 2), LANES) == 18_176
    assert max(shared_bytes(am, LANES) for am in nx(3)) == 47_104 <= LIMIT
    over = sorted((am for am in nx(4) if shared_bytes(am, LANES) > LIMIT),
                  key=lambda a: (sum(a), a))
    assert over[0] == (4, 0, 4, 0) and shared_doubles(over[0]) == 260
    assert shared_bytes(over[0], LANES) == 66_560
    ladder = [L for L in range(17) if shared_bytes(k_spec(L).am, LANES) > LIMIT]
    assert ladder[0] == 13 and k_spec(13).am == (4, 3, 3, 3)
    # the formula, by hand, at (dd|dd): n_p(4) = 35, n_d^2 = 36 -- against
    # the milestone's materialised I (n_p x n_bd) and V (n_ac x n_bd)
    assert shared_doubles((2, 2, 2, 2)) == 35 + 36
    assert 35 * 36 + 36 * 36 == 2_556 and 2_556 * LANES * 8 == 654_336


@pytest.mark.parametrize("am", [pytest.param(a, id=_id(a)) for a in FAST])
def test_the_formula_is_the_built_extent(am: Am) -> None:
    (sh,) = [s for s in walk(build_k_kernel(am).body)
             if isinstance(s, Decl) and s.name == KSH]
    assert sh.shape[0] == Lit(shared_doubles(am))
