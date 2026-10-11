"""kernel.py: a split class as a set of kernels, `build_pq_kernels`, interpreted.

THE PIN, ACROSS THE BATCH AXIS. For every (L, n_batches) in L = 0..16 x
{1, 3, 6, 9} that the plan layer ADMITS -- derived here by asking `plan`,
never listed (#64 made `partition` refuse an empty batch, so 59 of the 68
pairs plan) -- each batch kernel is interpreted in `Plan.dags` order, `J`
threaded through from zero, and the result compared against
`reference.pq.one_centre` contracted with the same `D_q`: a path with no MILD,
no DAG, no sieve, no schedule, no batching and no kernel.

THE TOLERANCE IS test_kernel.py's, PROPAGATED, NOT LOOSENED:

    |J_p - want_p| <= 1e-12 * SUM_k SUM_q |[p|q] D_q|

The whole grid interprets in seconds on this host, so it is in the fast tier
unshrunk; no cell is marked `slow`.

Nothing compiles and nothing launches on a card: interpreting is not emitting.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Decl,
    For,
    If,
    Index,
    Kernel,
    Lit,
    Note,
    Raw,
    Space,
    Var,
    WarpReduce,
    walk,
)
from felizjueves.kernel import (
    JSH,
    KET,
    SORTED_KETS,
    batch_contraction,
    bra_functions,
    build_pq_kernel,
    build_pq_kernels,
    ket_functions,
    launch_shape,
    pq_spec,
    split,
)
from felizjueves.passes import ATOMIC_NOTE
from felizjueves.plan.pipeline import plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.recurrence import node_name
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, theta_of
from felizjueves.spec import Spec

BATCH_COUNTS = (1, 3, 6, 9)


def _admitted(L: int, n_batches: int) -> bool:
    try:
        plan(pq_spec(L, n_batches=n_batches))
    except ValueError:
        return False
    return True


GRID = tuple((L, n) for L in range(17) for n in BATCH_COUNTS if _admitted(L, n))


def test_the_grid_is_what_the_plan_layer_admits() -> None:
    """59 of 68 since #64; the nine refused pairs are refused, not dropped."""
    assert len(GRID) == 59
    refused = {(L, n) for L in range(17) for n in BATCH_COUNTS} - set(GRID)
    assert refused == {(0, 3), (1, 6), (2, 6), (2, 9), (3, 9), (5, 9),
                       (0, 6), (0, 9), (1, 9)}
    for L, n in refused:
        with pytest.raises(ValueError):
            build_pq_kernels(pq_spec(L, n_batches=n))


@cache
def _kernels(spec: Spec) -> tuple[Kernel, ...]:
    return build_pq_kernels(spec)


def _inputs(spec: Spec, n_bra: int, n_ket: int, seed: int) -> dict:
    """As test_kernel.py's: kets sorted descending by Iq * Dmax, J zeroed."""
    rng = np.random.default_rng(seed)
    n_p, n_q = len(bra_functions(spec)), len(ket_functions(spec))
    Iq = rng.uniform(0.2, 1.0, n_ket)
    Dmax = rng.uniform(0.5, 1.0, n_ket)
    srt = np.argsort(-(Iq * Dmax), kind="stable")
    return dict(
        n_bra=n_bra, n_ket=n_ket, tau=0.0,
        P=rng.uniform(-0.5, 0.5, (n_bra, 3)), zeta=rng.uniform(0.6, 1.4, n_bra),
        Ip=rng.uniform(0.5, 1.0, n_bra),
        Q=rng.uniform(-0.5, 0.5, (n_ket, 3)), eta=rng.uniform(0.6, 1.4, n_ket),
        Iq=Iq[srt].copy(), Dmax=Dmax[srt].copy(),
        D=rng.uniform(-1.0, 1.0, (n_ket, n_q)),
        J=np.zeros((n_bra, n_p)),
    )


def _run(spec: Spec, args: dict, lanes: int) -> tuple[np.ndarray, list[Interpreter]]:
    """Every batch kernel in Plan.dags order, J threaded through from zero."""
    grid, block = launch_shape(args["n_bra"], lanes)
    J = np.zeros_like(args["J"])
    its = []
    for k in _kernels(spec):
        it = Interpreter(k, launch=Launch(grid, block, lanes), boys=boys)
        J = it.run(**{**args, "J": J})["J"]
        its.append(it)
    return J, its


def _reference(spec: Spec, args: dict):
    """J_p = SUM_k [bound_k >= tau] SUM_q (-1)^|q| [p+q]^(0) D_q, and the
    scale SUM_k SUM_q |[p|q] D_q|; no batching anywhere."""
    lp, lq = split(spec)
    ps, qs = bra_functions(spec), ket_functions(spec)
    want = np.zeros((args["n_bra"], len(ps)))
    scale = np.zeros_like(want)
    for b in range(args["n_bra"]):
        for k in range(args["n_ket"]):
            if args["Ip"][b] * (args["Iq"][k] * args["Dmax"][k]) < args["tau"]:
                continue
            PQ = tuple(float(c) for c in args["P"][b] - args["Q"][k])
            th = theta_of(float(args["zeta"][b]), float(args["eta"][k]))
            R = {r: one_centre(r, 0, PQ, th) for r in hermite_targets(lp + lq)}
            for i, p in enumerate(ps):
                for j, q in enumerate(qs):
                    r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
                    t = (-1.0) ** sum(q) * R[r] * args["D"][k, j]
                    want[b, i] += t
                    scale[b, i] += abs(t)
    return want, scale


def _assert_matches(got: np.ndarray, want: np.ndarray, scale: np.ndarray) -> None:
    assert got.shape == want.shape
    assert np.all(np.abs(got - want) <= 1e-12 * scale), (
        np.max(np.abs(got - want) / scale))


# ---------------------------------------------------------------------------
# 1. The pin, across the batch axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("L", "n_batches"), GRID)
def test_j_p_matches_the_reference_across_the_batch_axis(L: int, n_batches: int):
    spec = pq_spec(L, n_batches=n_batches)
    assert len(_kernels(spec)) == len(plan(spec).dags) == n_batches
    args = _inputs(spec, n_bra=2, n_ket=3, seed=1000 * n_batches + L)
    got, _ = _run(spec, args, lanes=2)
    _assert_matches(got, *_reference(spec, args))


def test_a_split_class_on_a_wide_warp() -> None:
    spec = pq_spec(7, n_batches=6)
    args = _inputs(spec, n_bra=3, n_ket=40, seed=7)
    got, _ = _run(spec, args, lanes=32)
    _assert_matches(got, *_reference(spec, args))


# ---------------------------------------------------------------------------
# 2. The partition actually partitioned
# ---------------------------------------------------------------------------


def _contracted(k: Kernel) -> set[tuple[int, int, int]]:
    """The targets [r]^(0) the kernel's J_p contraction reads, off the IR."""
    out: set[tuple[int, int, int]] = set()
    target = {node_name((r, 0)): r for r in hermite_targets(16)}

    def names(e) -> None:
        if isinstance(e, Var) and e.name in target:
            out.add(target[e.name])
        for f in getattr(e, "args", ()) or ():
            names(f)

    for s in walk(k.body):
        if isinstance(s, Assign) and isinstance(s.dst, Index) and s.dst.array == JSH:
            names(s.value)
    return out


@pytest.mark.parametrize(("L", "n_batches"), GRID)
def test_every_target_is_contracted_by_exactly_one_batch(L: int, n_batches: int):
    spec = pq_spec(L, n_batches=n_batches)
    dags = plan(spec).dags
    kernels = _kernels(spec)
    seen: dict[tuple[int, int, int], int] = {}
    for dag, k in zip(dags, kernels, strict=True):
        got = _contracted(k)
        assert got == {r for r, _ in dag.targets}  # each batch, all its own
        for r in got:
            assert r not in seen, f"{r} contracted by batches {seen[r]} and {k.name}"
            seen[r] = k.name
    assert set(seen) == set(hermite_targets(L))


def test_the_split_happened() -> None:
    """Not inferred from n_batches: nine kernels, nine disjoint proper subsets."""
    spec = pq_spec(6, n_batches=9)
    everything = set(hermite_targets(6))
    parts = [_contracted(k) for k in _kernels(spec)]
    assert len(parts) == 9
    assert all(p and p < everything for p in parts)
    assert sum(len(p) for p in parts) == len(everything)
    # L = 0 has one target: it cannot split, and the plan says so.
    assert not _admitted(0, 3)


# ---------------------------------------------------------------------------
# 3. An untouched p is skipped
# ---------------------------------------------------------------------------


def _untouched_cases() -> list[tuple[Spec, int]]:
    out = []
    for L, n in GRID:
        spec = pq_spec(L, n_batches=n)
        for b, dag in enumerate(plan(spec).dags):
            if len(batch_contraction(spec, dag)) < len(bra_functions(spec)):
                out.append((spec, b))
    return out


def test_an_untouched_p_gets_no_slot_no_reduce_and_no_atomic() -> None:
    cases = _untouched_cases()
    assert cases, "no (class, n_batches) leaves a bra function untouched"
    for spec, b in cases:
        dag, k = plan(spec).dags[b], _kernels(spec)[b]
        touched = [i for i, _, _ in batch_contraction(spec, dag)]
        assert len(touched) < len(bra_functions(spec))
        stmts = list(walk(k.body))
        (jsh,) = [s for s in stmts if isinstance(s, Decl) and s.name == JSH]
        assert jsh.shape[0] == Lit(len(touched))
        reduces = [s for s in stmts if isinstance(s, WarpReduce)]
        assert [r.dst.name for r in reduces] == [f"Jp{i}" for i in touched]
        atomics = [s for s in stmts if isinstance(s, AtomicAdd)]
        assert [a.dst.at[1] for a in atomics] == [Lit(i) for i in touched]


def test_an_untouched_p_is_left_exactly_alone() -> None:
    """Interpreted: one batch kernel onto a zeroed J writes only its p."""
    spec, b = _untouched_cases()[0]
    dag, k = plan(spec).dags[b], _kernels(spec)[b]
    touched = {i for i, _, _ in batch_contraction(spec, dag)}
    args = _inputs(spec, n_bra=2, n_ket=3, seed=11)
    grid, block = launch_shape(2, 2)
    J = Interpreter(k, launch=Launch(grid, block, 2), boys=boys).run(**args)["J"]
    for i in range(len(bra_functions(spec))):
        if i not in touched:
            assert np.all(J[:, i] == 0.0)
    assert np.any(J[:, sorted(touched)] != 0.0)


def test_an_untouched_p_and_an_empty_batch_are_different_things() -> None:
    """Every batch the plan admits touches at least one p; a batch with no
    target is refused by the plan (#64), not built as an empty kernel."""
    for L, n in GRID:
        spec = pq_spec(L, n_batches=n)
        assert all(batch_contraction(spec, d) for d in plan(spec).dags)


# ---------------------------------------------------------------------------
# 4. The screen still breaks, per batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lanes", [1, 4])
def test_the_screen_is_a_break_in_every_batch(lanes: int) -> None:
    spec = pq_spec(4, n_batches=3)
    args = _inputs(spec, n_bra=3, n_ket=24, seed=25)
    bounds = [args["Ip"][b] * args["Iq"] * args["Dmax"] for b in range(3)]
    args["tau"] = float(np.median(bounds[1]))
    above = [int((bb >= args["tau"]).sum()) for bb in bounds]
    n = args["n_ket"]
    assert all(lanes < a < n - lanes for a in above)
    got, its = _run(spec, args, lanes)
    _assert_matches(got, *_reference(spec, args))
    want = sum(len(range(y, a, lanes)) + any(j >= a for j in range(y, n, lanes))
               for a in above for y in range(lanes))
    assert len(its) == 3
    for it in its:
        assert it.iterations[KET] == want < args["n_bra"] * n


def test_every_batch_carries_the_sort_precondition() -> None:
    for k in _kernels(pq_spec(9, n_batches=9)):
        assert Note(SORTED_KETS) in list(walk(k.body))


# ---------------------------------------------------------------------------
# 5. [ab|cd] is never formed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("L", "n_batches"), [(4, 3), (9, 6), (16, 9)])
def test_the_shared_accumulator_is_the_only_array(L: int, n_batches: int) -> None:
    spec = pq_spec(L, n_batches=n_batches)
    for dag, k in zip(plan(spec).dags, _kernels(spec), strict=True):
        arrays = [s for s in walk(k.body) if isinstance(s, Decl) and s.shape]
        assert [d.name for d in arrays] == [JSH]
        assert arrays[0].space is Space.SHARED
        assert arrays[0].shape[0] == Lit(len(batch_contraction(spec, dag)))
        (loop,) = [s for s in walk(k.body) if isinstance(s, For)]
        assert all(not d.shape for d in walk(loop.body) if isinstance(d, Decl))


# ---------------------------------------------------------------------------
# 6. Atomic throughout when split; plain when not
# ---------------------------------------------------------------------------


def _writes(k: Kernel, kind: type) -> list:
    return [s for s in walk(k.body) if isinstance(s, kind)
            and isinstance(s.dst, Index) and s.dst.array == "J"]


@pytest.mark.parametrize(("L", "n_batches"), GRID)
def test_a_batch_kernel_atomics_into_j_and_never_assigns(L: int, n_batches: int):
    spec = pq_spec(L, n_batches=n_batches)
    for dag, k in zip(plan(spec).dags, _kernels(spec), strict=True):
        assert not any(isinstance(s, Raw) for s in walk(k.body))
        if n_batches == 1:
            assert k == build_pq_kernel(spec)
            assert not _writes(k, AtomicAdd)
            assert len(_writes(k, Assign)) == len(bra_functions(spec))
            continue
        assert not _writes(k, Assign)
        assert len(_writes(k, AtomicAdd)) == len(batch_contraction(spec, dag))
        assert Note(ATOMIC_NOTE) in list(walk(k.body))
        (lane0,) = [s for s in walk(k.body) if isinstance(s, If)
                    and set(s.then) == set(_writes(k, AtomicAdd))]
        assert lane0.then


def test_one_batch_delegates_to_build_pq_kernel() -> None:
    for L in (0, 4, 16):
        spec = pq_spec(L)
        assert build_pq_kernels(spec) == (build_pq_kernel(spec),)
    with pytest.raises(ValueError, match="atomics"):
        build_pq_kernel(pq_spec(7, n_batches=3))
    with pytest.raises(TypeError):
        build_pq_kernels(7)  # type: ignore[arg-type]


def test_kernel_names_follow_the_recurrence_convention() -> None:
    spec = pq_spec(6, n_batches=9)
    names = [k.name for k in _kernels(spec)]
    want = [f"pq_L6_p3_q3_{d}{a}_{b}" for d, a, b in
            (dag.batch for dag in plan(spec).dags)]
    assert names == want
    assert names[:3] == ["pq_L6_p3_q3_x0_2", "pq_L6_p3_q3_x3_3",
                         "pq_L6_p3_q3_x4_6"]
    assert len(set(names)) == len(names)
    assert build_pq_kernels(spec) == _kernels(spec)  # deterministic
