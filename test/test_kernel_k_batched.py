"""`kernel_k.build_k_kernels`: a split K class, the partial-sum form (#203).

The batched analogue of test_kernel_batched.py, for `Target.K`. Each batch
kernel runs its batch's DAG and adds its partial `K[a, b]` atomically; the
caller zeroes `K` and runs them in `Plan.dags` order (`kernel_k`'s module
docstring, "THE SPLIT CLASS").

WHAT IS EXACT AND WHAT IS ROUNDING, stated before the tests:

- BITWISE: interpreted with one lane, each batch kernel is `==`
  `ao.k.k_quartet` with `pq=` restricted to the batch's targets -- the
  `[r]^(0)` of that batch's OWN recurrence (`build_recurrence_kernel` over
  the batch's DAG, the same injected Boys), and `0.0` for any `p + q` the
  batch does not hold -- per bra record. Every term the kernel omits is a
  host `+-0.0`, the identity on the accumulator. And the batches threaded
  through one zeroed `K` are `==` the left fold of those host partials in
  `Plan.dags` order: an atomic add onto `0.0` is exact, and the interpreter
  runs the kernels in order.
- ROUNDING ONLY: the batches' sum against the ONE-BATCH kernel. The split
  reassociates every sum it cuts (`I[p]` over `q`, `V` over `p`, and
  `D (V_1 + V_2)` into `D V_1 + D V_2`), and each batch's recurrence may
  reach a target by another path; so the tolerance is test_kernel_k.py's,
  inherited, not loosened: `1e-14` of the per-element quartet scale.

E1'S PREDICTION, CROSS-CHECKED, NOT FED: the per-quartet stage-1, stage-2
and stage-3 multiply-adds are counted OFF THE BUILT KERNELS (the `Ebd`,
`Eac` and `Dcd` reads in each batch kernel's ket loop, which runs once per
primitive quartet) and asserted `==` `kdup.count` and the tracked
`k_duplication.json` row, at every cell E1 records. Neither side is adjusted
to the other.

THE L <= 2 RULE (milestone 10): no class with `L = la + lc + lb + ld > 2` is
built or interpreted before #231. The plan admits `n_batches = 3` at L = 1
and 2 (6 and 9 are refused there), so every K class with 1 <= L <= 2 runs
at 3 batches here, in the fast tier; the classes above it through `(dd|dd)`
and E1's `SUBGRID_LS` cells are written below, marked `slow`, NOT run --
#231 runs them. Their batch counts are hardcoded, not asked of the plan at
collection.

Fast tier: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cache

import numpy as np
import pytest
import test_kernel_k as tk

from felizjueves import kdup
from felizjueves.ao.k import k_quartet
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Decl,
    For,
    If,
    Index,
    Kernel,
    Note,
    Var,
    WarpReduce,
    _subexprs,
    walk,
)
from felizjueves.kernel import build_pq_kernels, k_spec
from felizjueves.kernel_k import (
    KET,
    KSH,
    SORTED_BY_IQ,
    SPLIT_WRITE,
    build_k_kernel,
    build_k_kernels,
    k_class_spec,
    launch_shape,
    shared_doubles,
)
from felizjueves.passes import LANE0
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.recurrence import build_recurrence_kernel, node_name, target_order
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

# Every K class with 1 <= L <= 2, at the only split the plan admits there.
FAST_SPLIT: list[tuple[Am, int]] = [(am, 3) for am in tk.FAST if sum(am) > 0]
# Above L = 2, through (dd|dd): written, `slow`, deferred to #231.
DEFERRED_SPLIT: list[tuple[Am, int]] = [(am, n) for am in tk.DEFERRED
                                        for n in (3, 6)]


def _cid(cell: tuple[Am, int]) -> str:
    return f"{tk._id(cell[0])}-n{cell[1]}"


CELLS = ([pytest.param(c, id=_cid(c)) for c in FAST_SPLIT]
         + [pytest.param(c, id=_cid(c), marks=pytest.mark.slow)
            for c in DEFERRED_SPLIT])
FAST_CELLS = [pytest.param(c, id=_cid(c)) for c in FAST_SPLIT]


def _spec(am: Am, n: int) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, (), n)


@cache
def _kernels(spec: Spec) -> tuple[Kernel, ...]:
    return build_k_kernels(spec)


def test_the_fast_grid_is_every_split_the_plan_admits_at_l_le_2() -> None:
    """14 classes at 3 batches; (ss|ss) cannot split, and 6 and 9 batches are
    refused at L <= 2 -- by the plan, before any kernel is built."""
    assert len(FAST_SPLIT) == 14
    assert {sum(am) for am, _ in FAST_SPLIT} == {1, 2}
    for am, n in [((0, 0, 0, 0), 3), ((1, 0, 0, 0), 6), ((1, 0, 1, 0), 6),
                  ((0, 0, 1, 1), 9)]:
        with pytest.raises(ValueError, match="empty"):
            build_k_kernels(_spec(am, n))


@pytest.mark.parametrize("n", [0, 2, 4, 5, 7, 8, -3])
def test_the_batch_count_rule_is_spec_s(n: int) -> None:
    """1, or a multiple of 3: any other count raises when the `Spec` is made,
    so no K builder is ever handed one."""
    with pytest.raises(ValueError, match="neither 1 nor a positive multiple"):
        _spec((1, 0, 1, 0), n)
    with pytest.raises(ValueError, match="neither 1 nor a positive multiple"):
        k_spec(2, n_batches=n)


# ---------------------------------------------------------------------------
# The batch's own [p|q]: its DAG's recurrence, interpreted, zero elsewhere
# ---------------------------------------------------------------------------


@cache
def _batch_recurrence(am: Am, n: int, b: int):
    dag = plan(_spec(am, n)).dags[b]
    rk = build_recurrence_kernel(dag, schedule(dag))
    return rk, {r: i for i, r in enumerate(target_order(dag))}


def batch_pq(am: Am, n: int, b: int):
    """`[p|q] = (-1)^|q| [p+q]^(0)` from batch `b`'s own recurrence, and
    `0.0` for a `p + q` the batch does not hold."""
    rk, where = _batch_recurrence(am, n, b)
    memo: dict = {}

    def pq(p, q, PQv, theta) -> float:
        r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
        if r not in where:
            return 0.0
        key = (tuple(PQv), theta)
        if key not in memo:
            memo[key] = Interpreter(rk, boys=boys).run(
                PQ_x=PQv[0], PQ_y=PQv[1], PQ_z=PQv[2], theta=theta,
                out=np.zeros(len(where)))["out"]
        v = float(memo[key][where[r]])
        return -v if sum(q) % 2 else v

    return pq


def run_batches(am: Am, n: int, args: dict, *, lanes: int = 1,
                only: int | None = None) -> tuple[np.ndarray, list]:
    """Every batch kernel (or batch `only`) in `Plan.dags` order, `K`
    threaded through from zero."""
    grid, block = launch_shape(args["n_bra"], lanes)
    K = np.zeros_like(args["K"])
    its = []
    for b, kern in enumerate(_kernels(_spec(am, n))):
        if only is not None and b != only:
            continue
        it = Interpreter(kern, launch=Launch(grid, block, lanes), boys=boys)
        K = it.run(**{**args, "K": K})["K"]
        its.append(it)
    return K, its


def _setup(am: Am, *, contracted: bool = True):
    bra, ket = tk._pairs(am, contracted=contracted)
    dcd = tk._dcd(am[1], am[3])
    return bra, ket, dcd, tk.k_args(am, bra, [(ket, 0)], [dcd])


# ---------------------------------------------------------------------------
# 1. Bitwise: each batch is the host on its targets; the sum is their fold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", CELLS)
def test_each_batch_is_the_host_quartet_on_its_targets_bitwise(cell) -> None:
    am, n = cell
    bra, ket, dcd, args = _setup(am)
    folds = [[0.0] * len(tk._flat(am)) for _ in bra.recs]
    for b in range(n):
        got, _ = run_batches(am, n, args, only=b)
        for i in range(len(bra.recs)):
            X = k_quartet(tk._one(bra, i), ket, dcd, pq=batch_pq(am, n, b))
            want = [v for row in X for v in row]
            assert got[i].tolist() == want, (b, i)
            folds[i] = [f + w for f, w in zip(folds[i], want, strict=True)]
    K, _ = run_batches(am, n, args)
    assert K.tolist() == folds
    assert np.any(K != 0.0)


# ---------------------------------------------------------------------------
# 2. Rounding: the batches' sum against the one-batch kernel
# ---------------------------------------------------------------------------


def _scale(am: Am, bra, ket, dcd) -> np.ndarray:
    from felizjueves.ao.k import _k_quartet
    return np.array([np.array(_k_quartet(tk._one(bra, i), ket, dcd,
                                         pq=tk.kernel_pq(am),
                                         want_scale=True)[1]).ravel()
                     for i in range(len(bra.recs))])


@pytest.mark.parametrize("lanes", [1, 4])
@pytest.mark.parametrize("cell", CELLS)
def test_the_batches_sum_to_the_one_batch_kernel_to_rounding(cell,
                                                             lanes) -> None:
    am, n = cell
    bra, ket, dcd, args = _setup(am)
    K, _ = run_batches(am, n, args, lanes=lanes)
    one, _ = tk.run_k(am, bra, [(ket, 0)], [dcd], lanes=lanes)
    S = _scale(am, bra, ket, dcd)
    assert np.all(np.abs(K - one) <= 1e-14 * S), np.max(np.abs(K - one) / S)


def test_a_split_class_on_a_wide_warp() -> None:
    am, n = (1, 0, 1, 0), 3
    bra, ket, dcd, args = _setup(am)
    K, _ = run_batches(am, n, args, lanes=32)
    one, _ = tk.run_k(am, bra, [(ket, 0)], [dcd], lanes=1)
    assert np.all(np.abs(K - one) <= 1e-14 * _scale(am, bra, ket, dcd))


# ---------------------------------------------------------------------------
# 3. The partition partitioned: every target in exactly one batch's stage 1
# ---------------------------------------------------------------------------


def _ket_loop(k: Kernel) -> tuple:
    (loop,) = [s for s in walk(k.body) if isinstance(s, For) and s.var == KET]
    return loop.body


def _stage1_targets(k: Kernel) -> set:
    names = {node_name((r, 0)): r for r in hermite_targets(8)}
    out = set()
    for s in walk(_ket_loop(k)):
        if isinstance(s, Assign) and isinstance(s.dst, Index) \
                and s.dst.array == KSH:
            for e in _subexprs(s.value):
                if isinstance(e, Var) and e.name in names:
                    out.add(names[e.name])
    return out


@pytest.mark.parametrize("cell", CELLS)
def test_every_target_is_contracted_by_exactly_one_batch(cell) -> None:
    am, n = cell
    spec = _spec(am, n)
    seen: dict = {}
    for dag, k in zip(plan(spec).dags, _kernels(spec), strict=True):
        got = _stage1_targets(k)
        assert got == {r for r, _ in dag.targets}
        for r in got:
            assert r not in seen, (r, seen[r], k.name)
            seen[r] = k.name
    assert set(seen) == set(hermite_targets(sum(am)))


# ---------------------------------------------------------------------------
# 4. Atomic throughout when split; plain when not; untouched is untouched
# ---------------------------------------------------------------------------


def _writes(k: Kernel, kind: type) -> list:
    return [s for s in walk(k.body) if isinstance(s, kind)
            and isinstance(s.dst, Index) and s.dst.array == "K"]


@pytest.mark.parametrize("cell", CELLS)
def test_no_plain_assign_to_k_survives_a_split(cell) -> None:
    am, n = cell
    for k in _kernels(_spec(am, n)):
        stmts = list(walk(k.body))
        assert not _writes(k, Assign), k.name
        atomics = _writes(k, AtomicAdd)
        reduces = [s for s in stmts if isinstance(s, WarpReduce)]
        assert len(atomics) == len(reduces) >= 1
        assert [a.value for a in atomics] == [r.dst for r in reduces]
        (lane0,) = [s for s in stmts if isinstance(s, If) and s.cond == LANE0]
        assert list(lane0.then) == atomics
        notes = [s.prov for s in stmts if isinstance(s, Note)]
        assert SPLIT_WRITE in notes and SORTED_BY_IQ in notes
        assert not any("NOT atomic" in t or "no atomics" in t for t in notes)


def test_one_batch_is_the_plain_write_kernel() -> None:
    for am in tk.FAST:
        (k,) = build_k_kernels(k_class_spec(*am))
        assert k == build_k_kernel(am)
        assert not _writes(k, AtomicAdd) and _writes(k, Assign)


def _slots(k: Kernel) -> int:
    (sh,) = [s for s in walk(k.body) if isinstance(s, Decl) and s.name == KSH]
    return sh.shape[0].value


@pytest.mark.parametrize("cell", FAST_CELLS)
def test_a_batch_holds_slots_only_for_what_it_touches(cell) -> None:
    """`Ksh` per batch is the `p` it holds plus the `K[a,b]` it writes, never
    more than the one-batch footprint; an untouched `K[a,b]` gets no reduce
    and no atomic, and interpreted it stays exactly `0.0`."""
    am, n = cell
    bra, ket, dcd, args = _setup(am)
    for b, k in enumerate(_kernels(_spec(am, n))):
        written = [a.dst.at[1].value for a in _writes(k, AtomicAdd)]
        i_rows = _slots(k) - len(written)
        assert 1 <= i_rows <= len(hermite_targets(am[0] + am[1]))
        assert _slots(k) <= shared_doubles(am)
        got, _ = run_batches(am, n, args, only=b)
        for s in range(len(tk._flat(am))):
            if s not in written:
                assert np.all(got[:, s] == 0.0)


def test_some_batch_leaves_a_k_entry_untouched() -> None:
    """Not inferred: at (ps|ss), 3 batches, the y and z batches write one
    `K[a, s]` each, the x batch all three (it holds p = 0)."""
    ks = _kernels(_spec((1, 0, 0, 0), 3))
    assert [len(_writes(k, AtomicAdd)) for k in ks] == [3, 1, 1]


def test_kernel_names_follow_the_batch_convention() -> None:
    spec = _spec((1, 0, 1, 0), 3)
    names = [k.name for k in _kernels(spec)]
    assert names == [f"k_10_10_{d}{a}_{b}" for d, a, b in
                     (dag.batch for dag in plan(spec).dags)]
    assert names == ["k_10_10_x0_2", "k_10_10_y0_2", "k_10_10_z0_2"]
    assert build_k_kernels(spec) == _kernels(spec)  # deterministic


# ---------------------------------------------------------------------------
# 5. The plan is untouched, per batch: each [p|q] program is J's batch's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", CELLS)
def test_each_batch_runs_the_j_batch_s_recurrence(cell) -> None:
    am, n = cell
    js = Spec(Operator.COULOMB, 0, am, Target.J, (), n)
    for jk, kk in zip(build_pq_kernels(js), _kernels(_spec(am, n)),
                      strict=True):
        jl, kl = _ket_loop(jk), _ket_loop(kk)
        end = next(i for i, s in enumerate(jl)
                   if isinstance(s, Note) and s.prov.startswith("J_p +="))
        assert tk.same_program(jl, kl, end)


# ---------------------------------------------------------------------------
# 6. The screen still breaks, per batch
# ---------------------------------------------------------------------------


def test_the_screen_is_a_break_in_every_batch() -> None:
    am, n = (1, 0, 1, 0), 3
    bra, ket, dcd, _ = _setup(am)
    Iq = [4.0, 3.0, 2.0, 1.0]
    args = tk.k_args(am, bra, [(ket, 0)], [dcd], tau=2.5 * 0.5 * 2.0,
                     maxD=2.0, Ip=[0.5] * 4, Iq=Iq)
    K, its = run_batches(am, n, args)
    head = dataclasses.replace(ket, recs=ket.recs[:2])
    want, _ = run_batches(am, n, tk.k_args(am, bra, [(head, 0)], [dcd]))
    assert K.tolist() == want.tolist()
    assert [it.iterations[KET] for it in its] == [4 * 3] * n


# ---------------------------------------------------------------------------
# 7. E1's prediction, against the kernels actually built
# ---------------------------------------------------------------------------


def built_counts(spec: Spec) -> dict[str, int]:
    """Per primitive quartet, summed over the batch kernels: the stage-1
    (`Ebd` reads), stage-2 (`Eac` reads) and stage-3 (`Dcd` reads)
    multiply-adds in each kernel's ket loop, which runs once per quartet."""
    c = {"Ebd": 0, "Eac": 0, "Dcd": 0}
    for k in build_k_kernels(spec):
        for s in walk(_ket_loop(k)):
            if isinstance(s, Assign):
                for e in _subexprs(s.value):
                    if isinstance(e, Index) and e.array in c:
                        c[e.array] += 1
    return {"stage1_fma": c["Ebd"], "stage2_fma": c["Eac"],
            "stage3_fma": c["Dcd"]}


def _check_e1(L: int, n: int) -> None:
    one = built_counts(k_spec(L))
    got = built_counts(k_spec(L, n_batches=n))
    want = kdup.count(L, n)["k"]
    rows = json.loads(kdup.DATA.read_text())["rows"]
    (stored,) = [r for r in rows if (r["L"], r["n_batches"]) == (L, n)]
    for stage in ("stage1", "stage2", "stage3"):
        assert got[f"{stage}_fma"] == want[f"{stage}_fma"], (L, n, stage)
        dup = got[f"{stage}_fma"] - one[f"{stage}_fma"]
        assert dup == want[f"dup_{stage}_fma"], (L, n, stage)
        assert stored["k"][f"dup_{stage}_fma"] == dup
    assert stored["k"] == want


E1_CELLS = ([pytest.param(L, n, id=f"L{L}-n{n}")
             for L, n in [(0, 1), (1, 1), (1, 3), (2, 1), (2, 3)]]
            + [pytest.param(L, n, id=f"L{L}-n{n}", marks=pytest.mark.slow)
               for L in kdup.SUBGRID_LS for n in (1, 3, 6, 9, 12)])


@pytest.mark.parametrize(("L", "n"), E1_CELLS)
def test_duplicated_stage2_work_agrees_with_e1(L: int, n: int) -> None:
    """Counted off the built kernels, against `kdup.count` and the tracked
    record: per stage and per duplicate. At L = 2, 3 batches: stage 2 is
    18 multiply-adds at one batch and 27 split, 9 duplicated."""
    _check_e1(L, n)


def test_the_l2_split_duplicates_stage2_and_j_does_not() -> None:
    """The finding, read off the built kernels at the one L = 2 split: K's
    stage 2 is duplicated (18 -> 27), stage 1 is not (24 = 24) -- the
    partition is over r -- and stage 3 grows 9 -> 19."""
    one, got = built_counts(k_spec(2)), built_counts(k_spec(2, n_batches=3))
    assert (one["stage2_fma"], got["stage2_fma"]) == (18, 27)
    assert one["stage1_fma"] == got["stage1_fma"] == 24
    assert (one["stage3_fma"], got["stage3_fma"]) == (9, 19)
