"""The one-electron hessian kernels: `Target.S`, `T` and `V` at
`derivative_order = 2`, built as `Kernel`s.

    build_h1e_kernel(spec | (target, la, lb), lowstack=None) -> Kernel
    h1e_lowstack(spec | (target, la, lb))     -> bool     the d2v form's rule
    h1e_name(spec | (target, la, lb))         -> str      d2s_ / d2t_ / d2v_{la}{lb}
    h1e_spec(target, la, lb)                  -> Spec     the class's d2 key
    launch_shape(n_rec, threads)              -> (grid, block)

Per primitive-pair record of one pair class `(la, lb)` (exponents `alpha`
on `A`, `beta` on `B`, the host's one scalar `s = c_a c_b (pi/zeta)^{3/2}`),
the contribution of the pair's block to

    H[A, k, B, l] += SUM_mn M_mn d2X_mn / dR_{A,k} dR_{B,l}

(`felizjueves.hessian`'s package docstring; `hessian.host_1e` is the
reference, by another route: it differentiates each 3-D function as a
list of moved Cartesians, the kernel each axis's factor). `kernel_g1e`'s
record head, weights `w_r` and launch are reused unchanged, the tables two
indices above the class:

    E = e_fragment.axis_tables_fragment(la + 2, lb + 2)    S, V
        e_fragment.axis_tables_fragment(la + 2, lb + 4)    T

AN OPERATOR PER AXIS. A component `d/dX_k d/dY_l` (`X, Y` in `A, B`)
puts `X` on axis `k` and `Y` on axis `l` -- both on one axis when
`k == l` -- and every other axis keeps its plain factor. On one axis
factor `f_ij` (`E^{ij}_t`, `S_ij = E^{ij}_0`, or the kinetic `K_ij`),

    dA f_ij = 2 alpha f_{i+1,j} - i f_{i-1,j}
    dB f_ij = 2 beta  f_{i,j+1} - j f_{i,j-1}

applied once or twice (`dA dA`, `dA dB`, `dB dB`), each operated factor a
pooled local `d{op}{tag}{x}_{i}_{j}_{t}`, the `- n f` term ABSENT where
`n = 0` or its source has no entry `t`. Then per record

- `d2s`: `hXY_kl = SUM_r w_r PROD_d (op_d E)^{a_d b_d}_0`.
- `d2t`: the kinetic bracket `SUM_m PROD_d (op_d F_m,d)`, `F_m,d = K` on
  `d == m`, `S` elsewhere (`kernel_g1e`'s `gt`, both factors operated).
- `d2v`: the Hermite densities, ONCE, outside the nucleus loop,

      H^{ops}_t = SUM_r w_r PROD_d (op_d E)^{a_d b_d}_{t_d}

  for every operator triple a component, a first derivative or nothing
  needs (`|t| <= L + n_ops`); then per nucleus `C` (serially, as
  `kernel_v`): `PQ = P - C`, the plan's `[t]^(0)` for every
  `|t| <= L + 2` (`kernel_v.v_plan((la, lb + 2))`: the plan layer at
  Hermite order `L + 2`, `recurrence_body` unchanged), and

      hXY_kl -= Z_C SUM_t H^{XY,kl}_t [t]^(0)                   centres
      H[X, k, C, l] += s Z_C SUM_t H^{X_k}_t [t + 1_l]^(0)     and mirrored
      H[C, k, C, l] -= s Z_C SUM_t H^{0}_t [t + 1_k + 1_l]^(0) nucleus

Then `H[A, k, A, l] += s hAA_kl` (`k <= l` formed, mirrored), `BB`
likewise, `H[A, k, B, l] += s hAB_kl` and `H[B, l, A, k]` the same value:
36 `AtomicAdd`s per record (and 45 per nucleus for `V`).

THE PARAMETER BLOCK is `kernel_g1e`'s, with the output `H[n_atom, 3,
n_atom, 3]` in place of `G[n_atom, 3]`. `M` is read, never written, and
need not be symmetric. `H` is ZEROED BY THE CALLER before a build's first
hessian launch; every kernel only adds.

ONE THREAD PER RECORD, `rec = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by
`rec < n_rec`, as at d1.

ROUND-OFF, NOT BITWISE, against the host: the association differs (per
axis here, per 3-D function there), and `V` regroups by Hermite index and
runs the plan's MILD recurrence. The interpreted test states its bound
(test/test_kernel_h1e.py).

THE LOW-STACK `d2v`, the default from `la + lb = 4`. The densities above
are straight-line text: every operator triple times every Hermite index
times every row, unrolled, so `d2v_22` is ~1.08 MB of source and `d2v_33`
~4.9 MB, over the repo's 512 KiB file cap, with every density live across
the nucleus loop. `build_h1e_kernel(cls)` therefore emits, whenever
`h1e_lowstack(cls)` (`V` with `la + lb + 2 >= kernel_v.LOWSTACK_MIN_LSUM`,
the Hermite order at which `V`'s own low-stack form takes over: (pf), (dd),
(df), (ff) and their mirrors through f), a kernel with the SAME name,
parameters, launch and `H` contract and no densities:

    tables (la + 2, lb + 2), the record head      unchanged
    W[n_rows]                                     the weights, as loops
    per nucleus C:
        PQ, [t]^(0) for |t| <= L + 2             the same prelude and plan
        R[i] = [t_i]^(0)                          hermite_targets order
        for each block row (a, b)                 e_fragment.cartesian_loops
            F{op}[d, t] = (op E)^{a_d b_d}_t      per axis, op in "" A B AA
                                                  AB BB, as guarded loops
            for each operator triple              rolled over t per axis
                q += W[row] SUM_t PROD_d F{o_d}[d, t_d] R[t + up]
        hXY_kl -= Z_C qXY_kl; the 45 nucleus atomics of s Z_C q

Each `F` entry is `operated`'s rule term for term (the `- n f` term behind
an `If` on `0 < n` and its source's reach), so it is the unrolled pooled
local's value bitwise; the sums are regrouped by row instead of by Hermite
index, so the two forms agree to rounding, and both are held to the host
to the same bound (test/test_kernel_h1e.py). The frame is the tables, `W`,
`R`, the six `F` vectors and 45 accumulators; the text is the recurrence
plus a fixed set of loops. Below the rule the default is the unrolled
kernel above, its text unchanged byte for byte (`d2v_22`'s unrolled text
pinned by digest in test/golden/hessian_digests.json); `lowstack=True` /
`False` forces either form of `V` at any class. `S` and `T` have one form
(`d2t_33`, their largest, is ~456 KB) and refuse `lowstack=True`.

A builder: vendor-neutral IR. It imports neither `ao/` nor `hessian.host_1e`
(its witness), nor the interpreter, nor `reference/`.
"""

from __future__ import annotations

from collections.abc import Callable

from felizjueves.gradient.kernel_g1e import (
    NUC,
    REC,
    Axis,
    Locals,
    axis_reader,
    launch_shape,
    record_head,
    weights,
)
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Expr,
    Fn,
    For,
    If,
    Index,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
)
from felizjueves.one_electron.kernel_v import LOWSTACK_MIN_LSUM, v_plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import OPERATOR_OF, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "COMPONENTS",
    "HESSIAN_1E",
    "ZEROED_H",
    "axis_ops",
    "build_h1e_kernel",
    "h1e_lowstack",
    "h1e_name",
    "h1e_spec",
    "launch_shape",
]

HESSIAN_1E = (Target.S, Target.T, Target.V)
_PREFIX = {Target.S: "d2s", Target.T: "d2t", Target.V: "d2v"}

ATOMIC_REASON = (
    "AtomicAdd: every record of every pair on an atom adds into that atom's "
    "blocks of H")

ZEROED_H = (
    "PRECONDITION: H is zeroed by the caller before the first hessian "
    "launch of a build; this kernel only adds")

# The record's centre-centre components `(X, k, Y, l)`: `AA` and `BB` with
# `k <= l` (each symmetric, mirrored on the scatter), `AB` all nine.
COMPONENTS: tuple[tuple[str, int, str, int], ...] = (
    *(("A", k, "A", l) for k in range(3) for l in range(k, 3)),  # noqa: E741
    *(("B", k, "B", l) for k in range(3) for l in range(k, 3)),  # noqa: E741
    *(("A", k, "B", l) for k in range(3) for l in range(3)),  # noqa: E741
)

Ops = tuple[str, str, str]
Op = Callable[[str, int, int, int, int], Expr]  # (op, d, i, j, t) -> expr


def h1e_spec(target: Target, la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of `target` at derivative order 2."""
    return Spec(OPERATOR_OF[target], 2, (la, lb), target)


def _class(cls: Spec | tuple[Target, int, int]) -> tuple[Target, int, int]:
    spec = cls if isinstance(cls, Spec) else h1e_spec(*cls)
    if spec.target not in HESSIAN_1E or spec.derivative_order != 2:
        raise ValueError(f"{spec.key()} is not a d2 S, T or V key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the hessian kernels take no levers")
    la, lb = spec.am
    return spec.target, la, lb


def h1e_name(cls: Spec | tuple[Target, int, int]) -> str:
    """The name `build_h1e_kernel(cls)` gives its kernel, unbuilt."""
    target, la, lb = _class(cls)
    return f"{_PREFIX[target]}_{la}{lb}"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _fold(terms: list[Expr]) -> Expr:
    out = terms[0]
    for t in terms[1:]:
        out = _c(Fn.ADD, out, t)
    return out


def _prod(f: list[Expr]) -> Expr:
    return _mul(_mul(f[0], f[1]), f[2])


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def axis_ops(x: str, k: int, y: str, l: int) -> Ops:  # noqa: E741
    """The operator on each axis of the component `d/dX_k d/dY_l`, each
    spelled sorted (`"AB"`, never `"BA"`)."""
    ops = ["", "", ""]
    ops[k] += x
    ops[l] += y
    return tuple("".join(sorted(o)) for o in ops)  # type: ignore[return-value]


def operated(base: Axis, pool: Locals, tag: str) -> Op:
    """`op` applied to `base`'s factor `f^{ij}_t` on axis `d` (module
    docstring): the last side of `op` is applied last, each step a pooled
    local. `base` must hold every entry `t <= i + j`, so an operated
    factor holds `t <= i + j + len(op)`."""

    def get(op: str, d: int, i: int, j: int, t: int) -> Expr:
        if not op:
            return base(d, i, j, t)
        rest, side = op[:-1], op[-1]

        def make() -> Expr:
            up = (i + 1, j) if side == "A" else (i, j + 1)
            two = Var("ea2" if side == "A" else "eb2")
            out: Expr = _mul(two, get(rest, d, *up, t))
            n = i if side == "A" else j
            lo = (i - 1, j) if side == "A" else (i, j - 1)
            if n and t <= lo[0] + lo[1] + len(rest):
                out = _c(Fn.SUB, out, _mul(Lit(float(n)),
                                           get(rest, d, *lo, t)))
            return out
        return pool.get(f"d{op}{tag}{'xyz'[d]}_{i}_{j}_{t}", make)
    return get


def _h(x: str, k: int, y: str, l: int) -> str:  # noqa: E741
    return f"h{x}{y}_{k}{l}"


def _s_values(la: int, lb: int, F: Op) -> list[Stmt]:
    rows = ef.block_rows(la, lb)
    out: list[Stmt] = []
    for comp in COMPONENTS:
        ops = axis_ops(*comp)
        terms = [_mul(Var(f"w{r}"), _prod([F(ops[d], d, a[d], b[d], 0)
                                           for d in range(3)]))
                 for r, (a, b) in enumerate(rows)]
        out.append(Decl(_h(*comp), Scalar.F64, init=_fold(terms)))
    return out


def _t_values(la: int, lb: int, E: Axis, pool: Locals) -> list[Stmt]:
    """`d2t`'s components (module docstring)."""
    eb = Var("ebk")

    def S(d: int, i: int, j: int, t: int) -> Expr:
        return E(d, i, j, 0)

    def K(d: int, i: int, j: int, t: int) -> Var:
        def make() -> Expr:  # e_fragment._shift's expression, restated
            k = _c(Fn.ADD, _mul(_mul(_mul(Lit(-2.0), eb), eb), S(d, i, j + 2, 0)),
                   _mul(_mul(eb, Lit(float(2 * j + 1))), S(d, i, j, 0)))
            if j >= 2:
                k = _c(Fn.SUB, k, _mul(Lit(0.5 * j * (j - 1)),
                                       S(d, i, j - 2, 0)))
            return k
        return pool.get(f"K{'xyz'[d]}_{i}_{j}", make)

    oS, oK = operated(S, pool, "S"), operated(K, pool, "K")
    rows = ef.block_rows(la, lb)
    out: list[Stmt] = []
    for comp in COMPONENTS:
        ops = axis_ops(*comp)
        terms = []
        for r, (a, b) in enumerate(rows):
            parts = [_prod([(oK if d == m else oS)(ops[d], d, a[d], b[d], 0)
                            for d in range(3)]) for m in range(3)]
            bracket = _c(Fn.ADD, _c(Fn.ADD, parts[0], parts[1]), parts[2])
            terms.append(_mul(Var(f"w{r}"), bracket))
        out.append(Decl(_h(*comp), Scalar.F64, init=_fold(terms)))
    return out


def _density_name(ops: Ops, n: int) -> str:
    return "H_" + "_".join(o or "n" for o in ops) + f"_{n}"


def _v_values(la: int, lb: int, F: Op) -> list[Stmt]:
    """`d2v`'s Hermite densities, nucleus loop and nucleus atomics (module
    docstring)."""
    L = la + lb
    rows = ef.block_rows(la, lb)
    first = {(x, k): axis_ops(x, k, "", k) for x in "AB" for k in range(3)}
    triples: list[Ops] = [("", "", ""), *first.values(),
                          *(axis_ops(*c) for c in COMPONENTS)]
    dens: list[Stmt] = []
    names: dict[tuple[Ops, tuple[int, int, int]], str] = {}
    for ops in dict.fromkeys(triples):
        for n, t in enumerate(hermite_targets(L + sum(map(len, ops)))):
            terms = [_mul(Var(f"w{r}"), _prod([F(ops[d], d, a[d], b[d], t[d])
                                               for d in range(3)]))
                     for r, (a, b) in enumerate(rows)
                     if all(t[d] <= a[d] + b[d] + len(ops[d])
                            for d in range(3))]
            if terms:
                names[ops, t] = _density_name(ops, n)
                dens.append(Decl(names[ops, t], Scalar.F64,
                                 init=_fold(terms)))

    def contract(ops: Ops, up: tuple[int, int, int]) -> Expr:
        """`SUM_t H^{ops}_t [t + up]^(0)`."""
        return _fold([_mul(Var(nm), Var(node_name((
            tuple(t[d] + up[d] for d in range(3)), 0))))
            for (o, t), nm in names.items() if o == ops])

    def unit(*axes: int) -> tuple[int, int, int]:
        return tuple(sum(a == d for a in axes) for d in range(3))  # type: ignore[return-value]

    dag, order = v_plan((la, lb + 2))
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    nuc = Var(NUC)

    def at(a: Expr, k: int, b: Expr, l: int) -> Index:  # noqa: E741
        return Index("H", (a, Lit(k), b, Lit(l)))

    per_nuc: list[Stmt] = [
        Decl("Zc", Scalar.F64, init=Index("Z", (nuc,))),
        *(Decl(nm, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                       Index("Cnuc", (nuc, Lit(d)))))
          for d, nm in enumerate(PQ_PARAMS)),
        *prelude,
        *recurrence_body(dag, order, fundamental),
        Note("hXY_kl -= Z_C SUM_t H^{XY,kl}_t [t]^(0): the centres; "
             "H[X, k, C, l] += s Z_C SUM_t H^{X_k}_t [t + 1_l]^(0), "
             "mirrored; H[C, k, C, l] -= s Z_C SUM_t H^0_t "
             "[t + 1_k + 1_l]^(0): the nucleus"),
        Decl("skz", Scalar.F64, init=_mul(Var("sk"), Var("Zc"))),
        Decl("mskz", Scalar.F64, init=_mul(Lit(-1.0), Var("skz"))),
    ]
    for comp in COMPONENTS:
        h = Var(_h(*comp))
        per_nuc.append(Assign(h, _c(Fn.SUB, h, _mul(
            Var("Zc"), contract(axis_ops(*comp), (0, 0, 0))))))
    per_nuc.append(Note(ATOMIC_REASON))
    for x, atom in (("A", "aa"), ("B", "ab")):
        for k in range(3):
            for l in range(3):  # noqa: E741
                nm = f"x{x}_{k}{l}"
                per_nuc += [
                    Decl(nm, Scalar.F64, init=_mul(Var("skz"), contract(
                        first[x, k], unit(l)))),
                    AtomicAdd(at(Var(atom), k, nuc, l), Var(nm)),
                    AtomicAdd(at(nuc, l, Var(atom), k), Var(nm)),
                ]
    for k in range(3):
        for l in range(k, 3):  # noqa: E741
            nm = f"xC_{k}{l}"
            per_nuc.append(Decl(nm, Scalar.F64, init=_mul(
                Var("mskz"), contract(("", "", ""), unit(k, l)))))
            per_nuc.append(AtomicAdd(at(nuc, k, nuc, l), Var(nm)))
            if k != l:
                per_nuc.append(AtomicAdd(at(nuc, l, nuc, k), Var(nm)))
    return [
        *dens,
        *(Decl(f"P{d}", Scalar.F64, init=Index(f"p_{a}", (Var(REC),)))
          for d, a in enumerate("xyz")),
        Note("theta = sqrt(zeta): a unit point-charge ket (kernel_v)"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, Var(ef.ZETA))),
        *(Decl(_h(*c), Scalar.F64, init=Lit(0.0)) for c in COMPONENTS),
        Note("the nuclei, serially inside the record (kernel_v's mapping)"),
        For(NUC, Lit(0), Var("n_atom"), Lit(1), tuple(per_nuc)),
    ]


# ---------------------------------------------------------------------------
# The low-stack d2v (module docstring, THE LOW-STACK `d2v`)
# ---------------------------------------------------------------------------

R_ARRAY = "R"  # [t]^(0) of one nucleus, hermite_targets(L + 2) order
W_ARRAY = "W"  # the record's weights, block_rows order
_OPS = ("", "A", "B", "AA", "AB", "BB")  # every operator axis_ops spells


def _f_array(op: str) -> str:
    """The row's operated factors `(op E)^{a_d b_d}_t`, `[3, L + 3]`."""
    return f"F{op or 'n'}"


def _iadd(x: Expr, y: Expr) -> Call:
    return _c(Fn.ADD, x, y)


def _cart_index(tag: str) -> Expr:
    """The position in `cartesians(l)` of the `cartesian_loops` nest
    tagged `tag`: `u (u+1)/2 + w`."""
    u, w = Var(f"{tag}_u"), Var(f"{tag}_w")
    return _iadd(_c(Fn.DIV, _mul(u, _iadd(u, Lit(1))), Lit(2)), w)


def _operated_value(op: str, d: int, i: Expr, j: Expr, t: Expr, dst: str,
                    lb_t: int, L_t: int) -> list[Stmt]:  # noqa: N803
    """Declare `dst = (op E)^{ij}_t` on axis `d`, `i`, `j`, `t` I32
    expressions: `operated`'s rule term for term, its `- n f` term behind
    an `If` on `0 < n` and `t <= ` its source's reach, so the value is the
    unrolled pooled local's bitwise. The base is the record head's tables
    at `(la_t, lb_t)`, `L_t = la_t + lb_t`."""
    if not op:
        return [Decl(dst, Scalar.F64, init=Index(ef.AXIS_ARRAYS[d], (
            ef.axis_table_index(i, j, t, lb_t, L_t),)))]
    rest, side = op[:-1], op[-1]
    one = Lit(1)
    if side == "A":
        up, lo, n, two = ((_iadd(i, one), j), (_c(Fn.SUB, i, one), j), i,
                          Var("ea2"))
    else:
        up, lo, n, two = ((i, _iadd(j, one)), (i, _c(Fn.SUB, j, one)), j,
                          Var("eb2"))
    out = _operated_value(rest, d, *up, t, dst + "u", lb_t, L_t)
    out.append(Decl(dst, Scalar.F64, init=_mul(two, Var(dst + "u"))))
    reach = _iadd(_iadd(lo[0], lo[1]), Lit(len(rest) + 1))
    out.append(If(_mul(_c(Fn.LT, Lit(0), n), _c(Fn.LT, t, reach)), (
        *_operated_value(rest, d, *lo, t, dst + "l", lb_t, L_t),
        Assign(Var(dst), _c(Fn.SUB, Var(dst), _mul(n, Var(dst + "l")))),
    )))
    return out


def _unit(*axes: int) -> tuple[int, int, int]:
    return tuple(sum(a == d for a in axes) for d in range(3))  # type: ignore[return-value]


def _contractions(la: int, lb: int
                  ) -> list[tuple[Ops, list[tuple[tuple[int, int, int], str]]]]:
    """Every operator triple the record needs, each with the shifts `up`
    of `R[t + up]` it is contracted against and their accumulators: the 21
    components at `up = 0`, the six first derivatives at `1_l`, and the
    plain densities at `1_k + 1_l` (module docstring)."""
    out: list[tuple[Ops, list[tuple[tuple[int, int, int], str]]]] = [
        (axis_ops(*c), [((0, 0, 0), "q" + _h(*c)[1:])]) for c in COMPONENTS]
    out += [(axis_ops(x, k, "", k),
             [(_unit(l), f"q{x}_{k}{l}") for l in range(3)])  # noqa: E741
            for x in "AB" for k in range(3)]
    out.append((("", "", ""), [(_unit(k, l), f"qC_{k}{l}") for k in range(3)
                               for l in range(k, 3)]))  # noqa: E741
    return out


def _rolled(ops: Ops, ups: list[tuple[tuple[int, int, int], str]],
            a: tuple[Expr, Expr, Expr], b: tuple[Expr, Expr, Expr],
            width: int) -> For:
    """`q += W[row] SUM_t PROD_d F{o_d}[d, t_d] R[t + up]` for every
    `(up, q)` of `ups` at once: three loops over the row's reachable `t`
    (`t_d <= a_d + b_d + len(o_d)`), the product formed per axis."""
    t = (Var("q_tx"), Var("q_ty"), Var("q_tz"))

    def f(d: int) -> Index:
        return Index(_f_array(ops[d]), (_iadd(Lit(d * width), t[d]),))

    def top(d: int) -> Expr:
        return _iadd(_iadd(a[d], b[d]), Lit(len(ops[d]) + 1))

    def at(up: tuple[int, int, int]) -> Index:
        return Index(R_ARRAY, (ef.hermite_index(tuple(  # type: ignore[arg-type]
            _iadd(t[d], Lit(up[d])) if up[d] else t[d] for d in range(3))),))

    z = For(t[2].name, Lit(0), top(2), Lit(1), (
        Decl("q_e", Scalar.F64, init=_mul(Var("q_exy"), f(2))),
        *(Assign(Var(q), _iadd(Var(q), _mul(Var("q_e"), at(up))))
          for up, q in ups),
    ))
    y = For(t[1].name, Lit(0), top(1), Lit(1), (
        Decl("q_exy", Scalar.F64, init=_mul(Var("q_ex"), f(1))), z))
    return For(t[0].name, Lit(0), top(0), Lit(1), (
        Decl("q_ex", Scalar.F64, init=_mul(Var("wr"), f(0))), y))


def _v_lowstack_values(la: int, lb: int) -> list[Stmt]:
    """The low-stack `d2v`'s weights, nucleus loop and nucleus atomics
    (module docstring, THE LOW-STACK `d2v`): the same `hXY_kl` and the
    same 45 nucleus atomics per nucleus as `_v_values`, regrouped by
    row."""
    L = la + lb
    lb_t, L_t = lb + 2, L + 4  # the record head's tables
    na, nb = len(ef.cartesians(la)), len(ef.cartesians(lb))
    n_rows = na * nb
    ts = hermite_targets(L + 2)
    width = L + 3  # one axis's operated factor, t <= a_d + b_d + 2
    nuc = Var(NUC)

    # --- the weights, as kernel_g1e.weights' rule over loops ----------------
    i, j = Var("i"), Var("j")

    def m(row: Expr, col: Expr) -> Index:
        return Index("M", (row, col))

    def fill(value: Expr) -> For:
        return For("i", Lit(0), Lit(na), Lit(1), (
            For("j", Lit(0), Lit(nb), Lit(1), (
                Assign(Index(W_ARRAY, (_iadd(_mul(i, Lit(nb)), j),)),
                       value),)),))

    off = fill(_iadd(m(_iadd(Var("oa"), i), _iadd(Var("ob"), j)),
                     m(_iadd(Var("ob"), j), _iadd(Var("oa"), i))))
    w_fill: list[Stmt] = [Decl(W_ARRAY, Scalar.F64, (Lit(n_rows),))]
    if la != lb:  # a diagonal shell pair has la == lb
        w_fill.append(off)
    else:
        differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                    _c(Fn.LT, Var("ob"), Var("oa")))
        w_fill.append(If(differ, (off,), (fill(m(_iadd(Var("oa"), i),
                                                 _iadd(Var("oa"), j))),)))

    # --- per row: the operated factors, then every contraction -------------
    contractions = _contractions(la, lb)
    def cell(a: tuple[Expr, Expr, Expr], b: tuple[Expr, Expr, Expr]
             ) -> tuple[Stmt, ...]:
        out: list[Stmt] = [
            Decl("row", Scalar.I32, init=_iadd(
                _mul(_cart_index("a"), Lit(nb)), _cart_index("b"))),
            Decl("wr", Scalar.F64, init=Index(W_ARRAY, (Var("row"),))),
            Note("F{op}[d, t] = (op E)^{a_d b_d}_t on each axis, operated's "
                 "rule with the row's indices at run time"),
        ]
        for op in _OPS:
            for d in range(3):
                top = _iadd(_iadd(a[d], b[d]), Lit(len(op) + 1))
                out.append(For("q_t", Lit(0), top, Lit(1), (
                    *_operated_value(op, d, a[d], b[d], Var("q_t"), "o",
                                     lb_t, L_t),
                    Assign(Index(_f_array(op), (_iadd(Lit(d * width),
                                                      Var("q_t")),)),
                           Var("o")),
                )))
        out.append(Note("q += W[row] SUM_t PROD_d F{o_d}[d, t_d] R[t + up], "
                        "rolled over t per axis"))
        out += [_rolled(ops, ups, a, b, width) for ops, ups in contractions]
        return tuple(out)

    rows = ef.cartesian_loops(la, ("ax", "ay", "az"), lambda a: (
        ef.cartesian_loops(lb, ("bx", "by", "bz"),
                           lambda b: cell(a, b), tag="b"),), tag="a")

    # --- per nucleus ------------------------------------------------------
    dag, order = v_plan((la, lb + 2))
    held = {r for (r, mm) in order if mm == 0}
    assert set(ts) <= held, "the recurrence never forms every [t]^(0)"
    prelude, fundamental = fundamental_prelude(
        [mm for (r, mm) in order if not any(r)])
    accs = [q for _, ups in contractions for _, q in ups]

    def at_h(x: Expr, k: int, y: Expr, l: int) -> Index:  # noqa: E741
        return Index("H", (x, Lit(k), y, Lit(l)))

    per_nuc: list[Stmt] = [
        Decl("Zc", Scalar.F64, init=Index("Z", (nuc,))),
        *(Decl(nm, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                       Index("Cnuc", (nuc, Lit(d)))))
          for d, nm in enumerate(PQ_PARAMS)),
        *prelude,
        *recurrence_body(dag, order, fundamental),
        Note("[t]^(0) into the local R, hermite_targets order, so the "
             "rolled contractions can index it"),
        *(Assign(Index(R_ARRAY, (Lit(n),)), Var(node_name((tt, 0))))
          for n, tt in enumerate(ts)),
        *(Decl(q, Scalar.F64, init=Lit(0.0)) for q in accs),
        Note("per block row (e_fragment.cartesian_loops)"),
        rows,
        Note("hXY_kl -= Z_C qXY_kl: the centres; H[X, k, C, l] += s Z_C "
             "qX_kl, mirrored; H[C, k, C, l] -= s Z_C qC_kl: the nucleus"),
        Decl("skz", Scalar.F64, init=_mul(Var("sk"), Var("Zc"))),
        Decl("mskz", Scalar.F64, init=_mul(Lit(-1.0), Var("skz"))),
    ]
    for comp in COMPONENTS:
        h = Var(_h(*comp))
        per_nuc.append(Assign(h, _c(Fn.SUB, h, _mul(
            Var("Zc"), Var("q" + _h(*comp)[1:])))))
    per_nuc.append(Note(ATOMIC_REASON))
    for x, atom in (("A", "aa"), ("B", "ab")):
        for k in range(3):
            for l in range(3):  # noqa: E741
                nm = f"x{x}_{k}{l}"
                per_nuc += [
                    Decl(nm, Scalar.F64, init=_mul(Var("skz"),
                                                   Var(f"q{x}_{k}{l}"))),
                    AtomicAdd(at_h(Var(atom), k, nuc, l), Var(nm)),
                    AtomicAdd(at_h(nuc, l, Var(atom), k), Var(nm)),
                ]
    for k in range(3):
        for l in range(k, 3):  # noqa: E741
            nm = f"xC_{k}{l}"
            per_nuc.append(Decl(nm, Scalar.F64, init=_mul(
                Var("mskz"), Var(f"qC_{k}{l}"))))
            per_nuc.append(AtomicAdd(at_h(nuc, k, nuc, l), Var(nm)))
            if k != l:
                per_nuc.append(AtomicAdd(at_h(nuc, l, nuc, k), Var(nm)))
    return [
        Note(LOWSTACK_NOTE),
        *w_fill,
        *(Decl(f"P{d}", Scalar.F64, init=Index(f"p_{a}", (Var(REC),)))
          for d, a in enumerate("xyz")),
        Note("theta = sqrt(zeta): a unit point-charge ket (kernel_v)"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, Var(ef.ZETA))),
        *(Decl(_h(*c), Scalar.F64, init=Lit(0.0)) for c in COMPONENTS),
        Decl(R_ARRAY, Scalar.F64, (Lit(len(ts)),)),
        *(Decl(_f_array(op), Scalar.F64, (Lit(3 * width),)) for op in _OPS),
        Note("the nuclei, serially inside the record (kernel_v's mapping)"),
        For(NUC, Lit(0), Var("n_atom"), Lit(1), tuple(per_nuc)),
    ]


LOWSTACK_NOTE = (
    "LOW-STACK: no Hermite densities; per nucleus [t]^(0) into a local R, "
    "and per block row the operated per-axis factors F{op}[d, t] and every "
    "contraction as loops")


def _scatter() -> list[Stmt]:
    """The centre-centre blocks (module docstring): 36 atomics."""
    out: list[Stmt] = [Note(ATOMIC_REASON)]
    atom = {"A": Var("aa"), "B": Var("ab")}
    for x, k, y, l in COMPONENTS:  # noqa: E741
        v = _mul(Var("sk"), Var(_h(x, k, y, l)))
        out.append(AtomicAdd(Index("H", (atom[x], Lit(k), atom[y], Lit(l))),
                             v))
        if x != y or k != l:  # the mirror: (B, l, A, k), or (X, l, X, k)
            out.append(AtomicAdd(Index("H", (atom[y], Lit(l), atom[x],
                                             Lit(k))), v))
    return out


def h1e_lowstack(cls: Spec | tuple[Target, int, int]) -> bool:
    """THE SELECTION RULE (module docstring, THE LOW-STACK `d2v`): True for
    `V` when its Hermite order `la + lb + 2 >= LOWSTACK_MIN_LSUM`, where
    `build_h1e_kernel` emits the looped form by default; False below, and
    for `S` and `T`, which have one form."""
    target, la, lb = _class(cls)
    return target is Target.V and la + lb + 2 >= LOWSTACK_MIN_LSUM


def build_h1e_kernel(cls: Spec | tuple[Target, int, int], *,
                     name: str | None = None,
                     lowstack: bool | None = None) -> Kernel:
    """The hessian kernel of one one-electron pair class at d2.

    `cls` is a d2 `Target.S` / `T` / `V` `Spec` or `(target, la, lb)`.
    See the module docstring. `lowstack` picks `V`'s form: `None` (the
    default) is `h1e_lowstack(cls)`, `True` / `False` force either; `S`
    and `T` refuse `True`.
    """
    target, la, lb = _class(cls)
    if lowstack is None:
        lowstack = h1e_lowstack(cls)
    if lowstack and target is not Target.V:
        raise ValueError(f"{h1e_name(cls)}: only d2v has a low-stack form")
    lb_t = lb + 4 if target is Target.T else lb + 2
    head = record_head(la + 2, lb_t)
    E = axis_reader(la + 2, lb_t)
    pool = Locals()
    if target is Target.T:
        head.append(Decl("ebk", Scalar.F64, init=Index("eb_ket", (Var(REC),))))
        values = _t_values(la, lb, E, pool)
    elif target is Target.S:
        values = _s_values(la, lb, operated(E, pool, ""))
    elif lowstack:
        values = _v_lowstack_values(la, lb)
    else:
        values = _v_values(la, lb, operated(E, pool, ""))
    what = {Target.S: "overlap", Target.T: "kinetic energy",
            Target.V: "nuclear attraction (the operator's centre-nucleus "
                      "and nucleus-nucleus terms included)"}[target]
    per_rec = (*head, *values, *_scatter()) if lowstack else (
        *head, *weights(la, lb),
        Note("the operated factors, each declared where first read"),
        *pool.decls, *values, *_scatter())
    form = ", LOW-STACK" if lowstack else ""
    body: tuple[Stmt, ...] = (
        Note(f"hessian of the {what}, class (la, lb) = ({la}, {lb}){form}: "
             "H[A, k, B, l] += SUM_mn M_mn d2X_mn/dR_{A,k}dR_{B,l} "
             "(felizjueves.hessian)"),
        Note(ZEROED_H),
        Note("launch: one thread per primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, Var(REC), Var("n_rec")), per_rec),
    )
    nr, nao, nat = Var("n_rec"), Var("n_ao"), Var("n_atom")
    params = [
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_atom", Scalar.I32),
        _f64("zeta_ab", (nr,)),
        *(_f64(f"pa_{a}", (nr,)) for a in "xyz"),
        *(_f64(f"pb_{a}", (nr,)) for a in "xyz"),
        *((_f64(f"p_{a}", (nr,)) for a in "xyz") if target is Target.V
          else ()),
        _f64("s_ab", (nr,)),
        _f64("ea_bra", (nr,)),
        _f64("eb_ket", (nr,)),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        _i32("atom_a", (nr,)),
        _i32("atom_b", (nr,)),
        _f64("M", (nao, nao)),
    ]
    if target is Target.V:
        params += [_f64("Z", (nat,)), _f64("Cnuc", (nat, Lit(3)))]
    params.append(Param("H", Scalar.F64, array=True, intent=Intent.OUT,
                        shape=(nat, Lit(3), nat, Lit(3))))
    return Kernel(name or h1e_name((target, la, lb)), tuple(params), body)
