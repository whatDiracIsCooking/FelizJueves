"""`kernel_sx`: the cross-basis overlap kernel, `Target.SX`, interpreted.

THE WITNESS is `ao.overlap.cross_overlap_matrix`: `overlap_block` of every
ordered shell pair `(i in A, j in B)`, placed by `ao.one_electron.
rectangular`. The kernel's records are `launch.cross_overlap_launch`'s, in
its parameter order (`XClass.arrays`), and the per-record statements are
`kernel_s.record_values`', so the only new IR is the scatter.

EXACT, NOT A TOLERANCE. Interpreted with the records in order, each shell
pair's records add in `primitive_pairs` order from the caller's `0.0`,
which is `overlap_block`'s own accumulation, and with no mirror and no
triangle read the whole rectangular matrix is `np.array_equal` to the host
-- a stronger statement than `S`'s, whose diagonal shell pairs are `==` only
to a rearranged block (test_kernel_s.py).

THE CASES. Per class, every ordered class (ss) through (ff) on HF/cc-pVTZ
against ITSELF as two bases -- it is a legal pair of bases, the one tracked
basis reaching f, and it puts same-centre shell pairs (`off_a == off_b`
whenever `i == j`) through a kernel with no diagonal rule, so a scatter that
special-cased equal offsets would fail here; and the whole matrix of the
three tracked libcint cross cases, both orientations. libcint itself is the
host's witness (test_ao_cross_overlap.py), not this file's. Fast tier:
nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.overlap import cross_overlap_matrix
from felizjueves.ao.pairs import overlap_bound
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import AtomicAdd, If, Intent, walk
from felizjueves.kernel_s import build_s_kernel, record_values
from felizjueves.kernel_sx import (
    ATOMIC_REASON,
    ZEROED_S,
    build_sx_kernel,
    launch_shape,
    sx_spec,
)
from felizjueves.launch import XClass, XLaunch, cross_overlap_launch
from felizjueves.spec import Operator, Spec, Target

LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]


@cache
def _basis(name: str):
    return ob.from_fixture(manifest.load(manifest.FIXTURES / f"{name}.json"))


@cache
def _cross(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    rows, _ = manifest.cross_overlap(fx)
    return ob.from_fixture(rows), ob.from_fixture(fx)


def _run(cls: XClass, n_a: int, n_b: int, S: np.ndarray,
         threads: int = 5) -> np.ndarray:
    arr = {k: np.array(v) for k, v in cls.arrays().items()}
    for k in ("off_a", "off_b"):
        arr[k] = arr[k].astype(np.int32)
    n_rec = len(cls.recs)
    grid, block = launch_shape(n_rec, threads)
    return Interpreter(build_sx_kernel(cls.spec()),
                       launch=Launch(grid, block, 1)).run(
        n_rec=n_rec, n_a=n_a, n_b=n_b, S=S, **arr)["S"]


def _whole(plan: XLaunch) -> np.ndarray:
    S = np.zeros((plan.n_a, plan.n_b))
    for c in plan:
        S = _run(c, plan.n_a, plan.n_b, S)
    return S


@cache
def _hf_plan() -> XLaunch:
    b = _basis("hf_ccpvtz")
    return cross_overlap_launch(b, b, tau=0.0, bound=overlap_bound)


# ---------------------------------------------------------------------------
# Per class, on HF/cc-pVTZ against itself
# ---------------------------------------------------------------------------


def test_hf_against_itself_reaches_every_ordered_class_and_equal_offsets():
    plan = _hf_plan()
    assert [(c.la, c.lb) for c in plan] == CLASSES
    assert any(r.off_a == r.off_b for c in plan for r in c.recs)


@pytest.mark.parametrize("cls", CLASSES, ids=[f"{a}{b}" for a, b in CLASSES])
def test_one_launch_over_the_class_equals_the_host(cls):
    plan = _hf_plan()
    b = _basis("hf_ccpvtz")
    (c,) = [c for c in plan if (c.la, c.lb) == cls]
    pairs = [(i, j) for i, j, _, _ in c.blocks]
    got = _run(c, b.n_ao, b.n_ao, np.zeros((b.n_ao, b.n_ao)))
    want = np.array(cross_overlap_matrix(b, b, pairs=pairs))
    assert np.array_equal(got, want)  # owned elements, and 0.0 elsewhere


# ---------------------------------------------------------------------------
# The whole matrix, on the tracked cross cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("swapped", [False, True], ids=["AB", "BA"])
@pytest.mark.parametrize("name", sorted(p.stem
                                        for p in manifest.cross_fixtures()))
def test_the_whole_rectangular_matrix(name, swapped):
    A, B = _cross(name)
    if swapped:
        A, B = B, A
    plan = cross_overlap_launch(A, B, tau=0.0, bound=overlap_bound)
    S = _whole(plan)
    assert S.shape == (A.n_ao, B.n_ao)
    assert np.array_equal(S, np.array(cross_overlap_matrix(A, B)))


def test_a_screened_launch_leaves_the_dropped_blocks_exactly_zero():
    A, B = _cross("h2o_sto3g_x_631g_moved")
    bounds = sorted(overlap_bound(a, b) for a in A.shells for b in B.shells)
    plan = cross_overlap_launch(A, B, tau=bounds[len(bounds) // 3],
                                bound=overlap_bound)
    assert plan.dropped
    S = _whole(plan)
    assert np.array_equal(S, np.array(cross_overlap_matrix(A, B,
                                                           pairs=plan.kept)))


# ---------------------------------------------------------------------------
# The contract, structurally
# ---------------------------------------------------------------------------


def test_the_key_is_an_ordered_overlap_pair():
    assert sx_spec(2, 1) == Spec(Operator.OVERLAP, 0, (2, 1), Target.SX)
    assert build_sx_kernel((2, 1)).name == "sx_21"
    assert build_sx_kernel((1, 2)).name == "sx_12"
    with pytest.raises(ValueError, match="Target.SX"):
        build_sx_kernel(Spec(Operator.OVERLAP, 0, (1, 1), Target.S))
    with pytest.raises(ValueError):
        build_sx_kernel(Spec(Operator.OVERLAP, 0, (1, 1), Target.SX,
                             ("recycle",)))


def test_the_parameter_block():
    k = build_sx_kernel((1, 2))
    assert [p.name for p in k.params] == [
        "n_rec", "n_a", "n_b", "zeta_ab", "XPA", "XPB", "s", "off_a",
        "off_b", "S"]
    (out,) = [p for p in k.params if p.intent is Intent.OUT]
    assert out.name == "S"
    assert [d.name for d in out.shape] == ["n_a", "n_b"]  # rectangular
    notes = {getattr(st, "prov", None) for st in walk(k.body)}
    assert {ZEROED_S, ATOMIC_REASON} <= notes


@pytest.mark.parametrize("cls", CLASSES, ids=[f"{a}{b}" for a, b in CLASSES])
def test_one_add_per_element_no_mirror_no_diagonal_branch(cls):
    la, lb = cls
    k = build_sx_kernel(cls)
    adds = [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    na, nb = (la + 1) * (la + 2) // 2, (lb + 1) * (lb + 2) // 2
    dsts = sorted((a.dst.at[0].args[1].value, a.dst.at[1].args[1].value)
                  for a in adds)
    assert dsts == [(i, j) for i in range(na) for j in range(nb)]
    assert all(a.dst.at[0].args[0].name == "oa"
               and a.dst.at[1].args[0].name == "ob" for a in adds)
    # one guard (rec < n_rec), no offset comparison
    assert len([s for s in walk(k.body) if isinstance(s, If)]) == 1


@pytest.mark.parametrize("cls", CLASSES, ids=[f"{a}{b}" for a, b in CLASSES])
def test_the_record_statements_are_the_s_kernels(cls):
    head, values = record_values(*cls)
    sx = build_sx_kernel(cls).body
    s = build_s_kernel(cls).body
    (gx,) = [st for st in sx if isinstance(st, If)]
    (gs,) = [st for st in s if isinstance(st, If)]
    n = len(head) + len(values)
    assert tuple(gx.then[:n]) == tuple(gs.then[:n]) == (*head, *values)


@pytest.mark.parametrize("cls", [(0, 0), (2, 1), (1, 2), (3, 3)],
                         ids=["00", "21", "12", "33"])
def test_both_backends_emit_it(cls):
    k = build_sx_kernel(cls)
    for text in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)):
        assert f"sx_{cls[0]}{cls[1]}" in text
        assert "atomicAdd" in text
