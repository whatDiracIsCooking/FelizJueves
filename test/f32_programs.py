"""Two F32 IR programs (#338): the parity witnesses for the fp32 semantics.

Both are one thread per element `i`, launched as exactly `n` threads in
blocks of `BLOCK` (so every warp is full and no `If` guards a warp
reduction). Inputs: `x`, `y` float32 in [0.5, 2), `d` float64 in [0.5, 2),
all positive, so no sum in either program cancels except where a column is
built to.

- `EXACT` (`f32_exact`) uses no multiply whose result feeds an add or a
  subtract, so no FMA can form on either compiler: the card must equal the
  interpreter BITWISE. It covers every F32 spelling but `expf` and a
  multi-term `Weighted` with a non-unit factor: + - * /, `sqrtf`, both
  casts, an I32 converting into F32, the `f` literal, a loop-carried
  accumulator, `WarpReduce`, `AtomicAdd`, `LT`, and a unit-factor
  `Weighted` (`-1.0f * a`, exact, so a contraction cannot change it).
- `CONTRACTING` (`f32_contracting`) adds what a card may round differently:
  a ten-step recurrence of `Weighted` terms `PQ_x * r + (k-1) * r'` (FMA
  candidates on both compilers), `expf`, and an F32 SHARED array. It is held
  to the declared fp32 tier (test/oracle/tolerance.py), per column.

Each program has columns that DISCRIMINATE: `(a + 2^24) - 2^24` and
`((a + b) - a) - b` are zero or a rounding error in float32 and exactly `a`
or zero without it. An interpreter that does not round every operation
(`interp._round32` mutated to the identity) misses the card there by the
whole column's scale -- which is what test_f32_device.py asserts, so the
rounding is witnessed by a card, not by the interpreter's own idea of fp32.
"""

from __future__ import annotations

import numpy as np

from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Factor,
    Fn,
    For,
    Index,
    Intent,
    Kernel,
    Lit,
    Param,
    Scalar,
    Space,
    Term,
    Var,
    WarpReduce,
    Weighted,
)
from felizjueves.ir.interp import Launch, interpret

BLOCK = 64
N = 128  # elements: two blocks, four 32-lane warps
LOOP = 100  # the accumulator's trip count
BIG = 16777216.0  # 2^24: float32's integer ulp is 2 above it
EXACT_COLS, EXACT_COLS64 = 12, 2
CONTRACTING_COLS = 14


def _f32(v: float) -> Call:
    return Call(Fn.TO_F32, (Lit(v),))


def _c(fn: Fn, *args) -> Call:
    return Call(fn, tuple(args))


V = Var


def _params(cols: int, cols64: int) -> tuple[Param, ...]:
    out = (Param("n", Scalar.I32),
           Param("x", Scalar.F32, array=True),
           Param("y", Scalar.F32, array=True),
           Param("d", Scalar.F64, array=True),
           Param("out", Scalar.F32, array=True, intent=Intent.OUT,
                 shape=(V("n"), Lit(cols))))
    if cols64:
        out += (Param("out64", Scalar.F64, array=True, intent=Intent.OUT,
                      shape=(V("n"), Lit(cols64))),)
    return out


def _head() -> tuple:
    i = V("i")
    return (
        Decl("i", Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X),
                                      _c(Fn.MUL, Call(Fn.BLOCK_X),
                                         Call(Fn.BLOCK_DIM_X)))),
        Decl("a", Scalar.F32, init=Index("x", (i,))),
        Decl("b", Scalar.F32, init=Index("y", (i,))),
        Decl("s", Scalar.F32, init=_c(Fn.ADD, V("a"), V("b"))),
    )


def _out(col: int, arr: str = "out") -> Index:
    return Index(arr, (V("i"), Lit(col)))


def exact_kernel() -> Kernel:
    a, b, s = V("a"), V("b"), V("s")
    big = _f32(BIG)
    body = (
        *_head(),
        Assign(_out(0), _c(Fn.SUB, _c(Fn.SUB, s, a), b)),
        Assign(_out(1), _c(Fn.DIV, a, b)),
        Assign(_out(2), _c(Fn.SQRT, s)),
        Assign(_out(3), _c(Fn.MUL, a, b)),
        Assign(_out(4), _c(Fn.SUB, _c(Fn.TO_F32, Index("d", (V("i"),))), a)),
        Assign(_out(5), _c(Fn.SUB, _c(Fn.ADD, a, Lit(3)), a)),
        Assign(_out(6), _c(Fn.SUB, _c(Fn.ADD, a, big), big)),
        Decl("acc", Scalar.F32, init=_f32(0.0)),
        For("k", Lit(0), Lit(LOOP), Lit(1), (
            Assign(V("acc"), _c(Fn.ADD, V("acc"), _c(Fn.DIV, a, Lit(7)))),)),
        Assign(_out(7), V("acc")),
        Decl("red", Scalar.F32),
        WarpReduce(V("red"), _c(Fn.DIV, a, b)),
        Assign(_out(8), V("red")),
        Assign(_out(9), b),
        AtomicAdd(_out(9), a),
        Assign(_out(10), _c(Fn.LT, s, _f32(2.5))),
        Assign(_out(11), Weighted((Term(1, Factor.ONE, s),
                                   Term(-1, Factor.ONE, a),
                                   Term(-1, Factor.ONE, b)))),
        Assign(_out(0, "out64"), _c(Fn.SUB, _c(Fn.TO_F64, s),
                                    Index("d", (V("i"),)))),
        Assign(_out(1, "out64"), _c(Fn.TO_F64, _c(Fn.DIV, a, b))),
    )
    return Kernel("f32_exact", _params(EXACT_COLS, EXACT_COLS64), body)


def contracting_kernel() -> Kernel:
    a, b, s = V("a"), V("b"), V("s")
    big = _f32(BIG)
    pq = Factor.PQ_X
    body: list = [
        *_head(),
        Decl("PQ_x", Scalar.F32, init=_c(Fn.DIV, b, Lit(2))),
        Decl("r0", Scalar.F32, init=a),
        Decl("r1", Scalar.F32, init=Weighted((Term(1, pq, V("r0")),))),
    ]
    for k in range(2, 11):
        body.append(Decl(f"r{k}", Scalar.F32, init=Weighted((
            Term(1, pq, V(f"r{k - 1}")), Term(k - 1, Factor.ONE, V(f"r{k - 2}"))))))
    body += [Assign(_out(k - 1), V(f"r{k}")) for k in range(1, 11)]
    body += [
        Assign(_out(10), _c(Fn.EXP, _c(Fn.SUB, _f32(0.0), a))),
        Decl("sh", Scalar.F32, shape=(Lit(BLOCK),), space=Space.SHARED),
        Assign(Index("sh", (Call(Fn.THREAD_X),)), _c(Fn.EXP, b)),
        Assign(_out(11), _c(Fn.MUL, Index("sh", (Call(Fn.THREAD_X),)),
                            _c(Fn.TO_F32, Index("d", (V("i"),))))),
        Assign(_out(12), _c(Fn.SUB, _c(Fn.ADD, a, big), big)),
        Assign(_out(13), _c(Fn.SUB, _c(Fn.SUB, s, a), b)),
    ]
    return Kernel("f32_contracting", _params(CONTRACTING_COLS, 0), tuple(body))


# The columns built to discriminate rounding from its absence.
DISCRIMINATING = {"f32_exact": (0, 6), "f32_contracting": (12, 13)}


def inputs(seed: int = 338, n: int = N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "x": rng.uniform(0.5, 2.0, n).astype(np.float32),
        "y": rng.uniform(0.5, 2.0, n).astype(np.float32),
        "d": rng.uniform(0.5, 2.0, n),
    }


def launch(lanes: int = 32, n: int = N) -> Launch:
    assert n % BLOCK == 0
    return Launch(grid=n // BLOCK, block=(BLOCK, 1), lanes=lanes)


def run(kernel: Kernel, data: dict[str, np.ndarray], lanes: int = 32
        ) -> dict[str, np.ndarray]:
    """The kernel interpreted on `data`: its OUT arrays, shaped (n, cols)."""
    n = data["x"].size
    params = {p.name: p for p in kernel.params}
    outs = {name: np.zeros(n * params[name].shape[1].value, dtype=dt)
            for name, dt in (("out", np.float32), ("out64", np.float64))
            if name in params}
    got = interpret(kernel, launch=launch(lanes, n), n=n, **data, **outs)
    return {k: v.reshape(n, -1) for k, v in got.items()}


def numpy_exact(data: dict[str, np.ndarray], lanes: int = 32) -> dict:
    """`EXACT` in numpy float32, op by op: an evaluation of the same program
    that shares no code with the interpreter."""
    f = np.float32
    a, b, d = data["x"], data["y"], data["d"]
    n = a.size
    s = a + b
    out = np.zeros((n, EXACT_COLS), dtype=f)
    out[:, 0] = (s - a) - b
    out[:, 1] = a / b
    out[:, 2] = np.sqrt(s)
    out[:, 3] = a * b
    out[:, 4] = d.astype(f) - a
    out[:, 5] = (a + f(3)) - a
    out[:, 6] = (a + f(BIG)) - f(BIG)
    acc = np.zeros(n, dtype=f)
    for _ in range(LOOP):
        acc = acc + a / f(7)
    out[:, 7] = acc
    q = (a / b).reshape(-1, lanes)
    o = lanes // 2
    while o:
        q = q + q[:, np.arange(lanes) ^ o]
        o //= 2
    out[:, 8] = q.reshape(-1)
    out[:, 9] = b + a
    out[:, 10] = (s < f(2.5)).astype(f)
    out[:, 11] = (s + f(-1) * a) + f(-1) * b
    out64 = np.stack([s.astype(np.float64) - d,
                      (a / b).astype(np.float64)], axis=1)
    return {"out": out, "out64": out64}
