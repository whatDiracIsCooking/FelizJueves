"""`e_fragment`: the `E` coefficients as device code, through the interpreter.

`E` has one independent check, the pointwise Hermite identity
(test/test_ao_hermite.py), and `ao.hermite` is what passes it. So the
fragments are checked against `ao.hermite.e_axis` / `e_block`, and the axis
fragment against the identity directly as well:

- the AXIS fragment, interpreted alone, equals `e_axis` over randomised
  `(zeta, X_PA, X_PB)` for every `i, j <= 3` on every axis;
- the BLOCK fragment equals `e_block` for every pair class (ss) through (ff),
  all 16 ordered `(la, lb)`, with the entries `t_d > a_d + b_d` ABSENT (left
  as the consumer's NaN) rather than written as zeros;
- dressed by `ao.pairs.dress` -- the one module that dresses -- the block is
  `ao.pairs.PrimitivePair.block`, row for row, so the layout is the pair
  record's and no dressing is inside the fragment.

EXACT EQUALITY, NOT A TOLERANCE. The fragments perform the reference's
operations in the reference's order (i raised first through `X_PA`, then j
through `X_PB`; terms `1/2zeta`, `X`, `(t+1)`; `exp((zeta X_PA) X_PB)`;
products `(E_x E_y) E_z`), and the interpreter is fp64 Python, so fp64
round-off between them is zero. A mismatch is a defect, not noise. The
identity check carries `test_ao_hermite`'s stated `K * eps * scale` bound.

The reach is f by the issue's budget; every class runs in the fast tier
(the (ff) block interprets in well under a second).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
from test_ao_hermite import EPS, K, _derived, _lambda, _pair

from felizjueves import e_fragment as ef
from felizjueves import ir
from felizjueves.ao.hermite import e_axis, e_block, hermite_order
from felizjueves.ao.pairs import dress, primitive_pairs
from felizjueves.ao.shell import Shell, cartesians
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.interp import interpret, interpret_fragment
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Factor,
    Fn,
    Inline,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Raw,
    Scalar,
    Var,
    Weighted,
    walk,
)

LMAX = 3  # (ss) through (ff)
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
NEW_FACTORS = {Factor.HALF_INV_ZETA, Factor.PA_X, Factor.PA_Y, Factor.PA_Z,
               Factor.PB_X, Factor.PB_Y, Factor.PB_Z}


def _geometry(rng: random.Random):
    """A random primitive pair: `zeta`, `P`, `A`, `B` in 3D."""
    alpha, beta = rng.uniform(0.1, 5.0), rng.uniform(0.1, 5.0)
    A = tuple(rng.uniform(-1.5, 1.5) for _ in range(3))
    B = tuple(rng.uniform(-1.5, 1.5) for _ in range(3))
    zeta = alpha + beta
    P = tuple((alpha * A[d] + beta * B[d]) / zeta for d in range(3))
    return alpha, beta, zeta, P, A, B


def _scalars(la: int, lb: int, zeta: float, P, A, B, axes=(0, 1, 2)) -> dict:
    out = {"zeta": zeta}
    if la + lb:
        out["half_inv_zeta"] = 0.5 / zeta
    for d in axes:
        out[Factor.pa(d).name_in_kernel] = P[d] - A[d]
        out[Factor.pb(d).name_in_kernel] = P[d] - B[d]
    return out


def _absent(la: int, lb: int) -> np.ndarray:
    """True where `t_d > a_d + b_d` on some axis: the structural zeros."""
    rows, cols = ef.block_shape(la, lb)
    mask = np.zeros((rows, cols), bool)
    for r, (a, b) in enumerate(ef.block_rows(la, lb)):
        for c, t in enumerate(hermite_order(la + lb)):
            mask[r, c] = any(t[d] > a[d] + b[d] for d in range(3))
    return mask


def _run_block(la: int, lb: int, zeta, P, A, B, fill=np.nan) -> np.ndarray:
    frag = ef.block_fragment(la, lb)
    E = np.full(ef.block_shape(la, lb), fill)
    return interpret_fragment(frag, E=E, **_scalars(la, lb, zeta, P, A, B))["E"]


# ---------------------------------------------------------------------------
# The axis fragment, alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [0, 1, 2])
def test_the_axis_fragment_is_e_axis(d: int) -> None:
    frag = ef.axis_fragment(LMAX, LMAX, d)
    rng = random.Random(152 + d)
    for _ in range(20):
        _, _, zeta, P, A, B = _geometry(rng)
        out = interpret_fragment(frag, **_scalars(LMAX, LMAX, zeta, P, A, B, (d,)))
        pa, pb = P[d] - A[d], P[d] - B[d]
        for i in range(LMAX + 1):
            for j in range(LMAX + 1):
                want = e_axis(i, j, zeta, pa, pb)
                got = tuple(out[ef.e_name(d, i, j, t)] for t in range(i + j + 1))
                assert got == want, (d, i, j, zeta, pa, pb)  # bitwise


def test_the_axis_fragment_satisfies_the_hermite_identity() -> None:
    # The independent check, applied to the fragment's own output.
    frag = ef.axis_fragment(LMAX, LMAX, 0)
    rng = random.Random(1520)
    for _ in range(10):
        alpha, beta, A, B = _pair(rng)
        zeta, P, _ = _derived(alpha, beta, A, B)
        out = interpret_fragment(frag, zeta=zeta, half_inv_zeta=0.5 / zeta,
                                 PA_x=P - A, PB_x=P - B)
        x = P + np.array([rng.uniform(-2.5, 2.5) for _ in range(9)]) / math.sqrt(zeta)
        for i in range(LMAX + 1):
            for j in range(LMAX + 1):
                lhs = (x - A) ** i * (x - B) ** j * np.exp(-alpha * (x - A) ** 2) \
                    * np.exp(-beta * (x - B) ** 2)
                terms = np.array([out[ef.e_name(0, i, j, t)] * _lambda(t, zeta, P, x)
                                  for t in range(i + j + 1)])
                scale = np.abs(terms).sum(axis=0) + np.abs(lhs)
                assert np.all(np.abs(lhs - terms.sum(axis=0)) <= K * EPS * scale)


@pytest.mark.parametrize("la,lb", CLASSES)
def test_entries_above_i_plus_j_are_absent_not_zeroed(la: int, lb: int) -> None:
    frag = ef.axis_fragment(la, lb, 1)
    names = {s.name for s in frag.body if isinstance(s, Decl)}
    assert names == {ef.e_name(1, i, j, t) for i in range(la + 1)
                     for j in range(lb + 1) for t in range(i + j + 1)}
    assert not any(isinstance(e, Lit) for s in frag.body if isinstance(s, Decl)
                   for e in ir._subexprs(s.init))


# ---------------------------------------------------------------------------
# The block fragment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("la,lb", CLASSES)
def test_the_block_fragment_is_e_block(la: int, lb: int) -> None:
    rng = random.Random(1000 * la + lb)
    absent = _absent(la, lb)
    rows = ef.block_rows(la, lb)
    for _ in range(3):
        _, _, zeta, P, A, B = _geometry(rng)
        got = _run_block(la, lb, zeta, P, A, B)
        assert np.all(np.isnan(got[absent]))  # never written
        assert not np.any(np.isnan(got[~absent]))  # every other entry is
        want = np.array([e_block(a, b, zeta, P, A, B) for a, b in rows])
        assert np.all(want[absent] == 0.0)  # what the absence stands for
        assert np.array_equal(got[~absent], want[~absent])  # bitwise


def test_the_block_layout_is_the_pair_records() -> None:
    assert all(ef.cartesians(l) == cartesians(l) for l in range(6))  # noqa: E741
    for la, lb in CLASSES:
        assert ef.block_rows(la, lb) == tuple(
            (a, b) for a in cartesians(la) for b in cartesians(lb))
        assert ef.block_shape(la, lb)[1] == len(hermite_order(la + lb))


@pytest.mark.parametrize("la,lb", [(0, 0), (1, 0), (2, 1), (1, 3)])
def test_dressed_by_ao_pairs_the_block_is_the_pair_record(la: int, lb: int) -> None:
    # The dressing lives in ao.pairs.dress, applied here OUTSIDE the
    # fragment: with it, the fragment's block is PrimitivePair.block. No
    # fudge factor -- the fragment carries no (pi/zeta)^{3/2}, no c_a c_b.
    sa = Shell(la, (0.1, -0.4, 0.7), (3.1, 0.6), (0.4, 0.7))
    sb = Shell(lb, (-0.5, 0.2, 0.3), (1.7,), (1.0,))
    for pp in primitive_pairs(sa, sb):
        ca, cb = sa.coefficients[pp.i], sb.coefficients[pp.j]
        got = _run_block(la, lb, pp.zeta, pp.P, sa.centre, sb.centre, fill=0.0)
        dressed = tuple(dress(tuple(row), ca=ca, cb=cb, zeta=pp.zeta)
                        for row in got)
        assert dressed == pp.block


# ---------------------------------------------------------------------------
# Structure: the recurrence is Weighted, and the result is recorded
# ---------------------------------------------------------------------------


def _seed_ok(e: ir.Expr) -> bool:
    return isinstance(e, Call) and e.fn is Fn.EXP


@pytest.mark.parametrize("la,lb", CLASSES)
def test_every_recurrence_step_is_weighted(la: int, lb: int) -> None:
    # The structural result: no step needed a Call tree or a Raw. Seeds are
    # the one EXP each axis; products are the block's MUL pairs; every other
    # value is one Weighted.
    frag = ef.block_fragment(la, lb)
    decls = [s for s in frag.body if isinstance(s, Decl)]
    seeds = [s for s in decls if _seed_ok(s.init)]
    assert [s.name for s in seeds] == [ef.e_name(d, 0, 0, 0) for d in range(3)]
    assert all(isinstance(s.init, Weighted) for s in decls if s not in seeds)
    assigns = [s for s in frag.body if isinstance(s, Assign)]
    assert assigns and all(
        isinstance(s.value, Call) and s.value.fn is Fn.MUL for s in assigns)
    assert not any(isinstance(s, Raw) for s in walk(frag.body))
    assert len(assigns) == int((~_absent(la, lb)).sum())


def test_the_provenance_is_cited_once_per_fragment() -> None:
    for frag in (ef.axis_fragment(2, 2, 0), ef.block_fragment(2, 2)):
        notes = [s for s in walk(frag.body) if isinstance(s, Note)]
        assert len(notes) == 1 and "eqs. 2.20-2.21" in notes[0].prov


def test_every_new_factor_is_used() -> None:
    used = {t.g for s in ef.block_fragment(1, 1).body if isinstance(s, Decl)
            for e in ir._subexprs(s.init) if isinstance(e, Weighted)
            for t in e.terms}
    assert NEW_FACTORS <= used
    assert NEW_FACTORS == set(Factor) - {Factor.ONE, Factor.PQ_X, Factor.PQ_Y,
                                         Factor.PQ_Z}


def test_the_binds_are_the_contract() -> None:
    assert ef.block_fragment(1, 2).binds == (
        "zeta", "half_inv_zeta", "PA_x", "PB_x", "PA_y", "PB_y", "PA_z",
        "PB_z", "E")
    assert ef.block_fragment(1, 2).array_binds == {"E"}
    # (ss) raises nothing, so 1/2zeta would be a stale bind: it is absent.
    assert "half_inv_zeta" not in ef.block_fragment(0, 0).binds
    assert ef.axis_fragment(0, 2, 2).binds == (
        "zeta", "half_inv_zeta", "PA_z", "PB_z")


def test_the_axis_body_is_the_one_source_of_the_block() -> None:
    # No nested Inline (#149): the block is composed at build time from the
    # same axis builder, so its axis statements ARE the axis fragments'.
    block = ef.block_fragment(2, 3).body
    assert not any(isinstance(s, Inline) for s in walk(block))
    for d in range(3):
        axis = ef.axis_fragment(2, 3, d).body[1:]  # minus its Note
        start = block.index(axis[0])
        assert block[start:start + len(axis)] == axis


def test_bad_input_refused() -> None:
    with pytest.raises(ValueError):
        ef.axis_fragment(-1, 0, 0)
    with pytest.raises(ValueError):
        ef.axis_fragment(0, 0, 3)
    with pytest.raises(ValueError):
        ef.block_fragment(1, 1.0)  # type: ignore[arg-type]


def test_the_interpreter_does_not_know_what_e_is() -> None:
    src = (Path(__file__).resolve().parents[1] / "src" / "felizjueves"
           / "interp.py").read_text().lower()
    assert "hermite" not in src and "e_fragment" not in src


# ---------------------------------------------------------------------------
# A kernel that splices the block; Fn.EXP spelled by both backends
# ---------------------------------------------------------------------------

PROBE_CLASS = (1, 1)


def e_block_probe(la: int = PROBE_CLASS[0], lb: int = PROBE_CLASS[1]) -> Kernel:
    """One thread computes one primitive pair's undressed block into `E`.

    Every scalar bind is bound to the host's own name (no `const` lines);
    `half_inv_zeta` is the host's `0.5 / zeta`, formed once.
    """
    frag = ef.block_fragment(la, lb)
    rows, cols = ef.block_shape(la, lb)
    scalars = [b for b in frag.binds if b not in ("half_inv_zeta", "E")]
    return Kernel(f"e_block_probe_{la}{lb}", (
        *(Param(n, Scalar.F64) for n in scalars),
        Param("E", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(rows), Lit(cols))),
    ), (
        Decl("half_inv_zeta", Scalar.F64,
             init=Call(Fn.DIV, (Lit(0.5), Var("zeta")))),
        Inline(frag, {b: Var(b) for b in frag.binds}),
    ))


GOLDEN = Path(__file__).resolve().parent / "golden"
CUDA_GOLDEN = GOLDEN / "cuda" / "e_block_probe_11.cu"
HIP_GOLDEN = GOLDEN / "hip" / "e_block_probe_11.hip"


def test_the_probe_kernel_interprets_to_e_block() -> None:
    la, lb = PROBE_CLASS
    rng = random.Random(11)
    _, _, zeta, P, A, B = _geometry(rng)
    k = e_block_probe()
    s = _scalars(la, lb, zeta, P, A, B)
    del s["half_inv_zeta"]
    got = interpret(k, E=np.full(ef.block_shape(la, lb), np.nan), **s)["E"]
    absent = _absent(la, lb)
    want = np.array([e_block(a, b, zeta, P, A, B)
                     for a, b in ef.block_rows(la, lb)])
    assert np.all(np.isnan(got[absent]))
    assert np.array_equal(got[~absent], want[~absent])


def test_the_cuda_golden_is_unchanged() -> None:
    # Regenerate (never hand-edit):
    #   PYTHONPATH=src:test uv run python -c "from test_e_fragment import \
    #   e_block_probe as k; from felizjueves.device import SM_86; \
    #   from felizjueves.emit.cuda import emit; print(emit(k(), SM_86), \
    #   end='')" > test/golden/cuda/e_block_probe_11.cu
    assert cuda.emit(e_block_probe(), SM_86) == CUDA_GOLDEN.read_text()


def test_the_hip_golden_is_unchanged() -> None:
    #   ... the same with GFX1200 and felizjueves.emit.hip, into
    #   test/golden/hip/e_block_probe_11.hip
    assert hip.emit(e_block_probe(), GFX1200) == HIP_GOLDEN.read_text()


def test_exp_is_spelled_by_both_backends() -> None:
    k = e_block_probe()
    for text in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)):
        assert text.count(" = exp(((zeta * ") == 3  # one seed per axis
        assert "// fragment e_block_11" in text
    c = cuda.emit(k, SM_86).splitlines()
    h = hip.emit(k, GFX1200).splitlines()
    assert h[2] == "#include <hip/hip_runtime.h>" and c[2:] == h[3:]


@pytest.mark.cuda
def test_the_cuda_probe_compiles(nvcc, compile_device_source) -> None:
    # Compile only: a device compiler needs no card.
    compile_device_source(nvcc, cuda.emit(e_block_probe(), SM_86), link=False)


@pytest.mark.hip
def test_the_hip_probe_compiles(hipcc, compile_device_source) -> None:
    compile_device_source(hipcc, hip.emit(e_block_probe(), GFX1200), link=False)
