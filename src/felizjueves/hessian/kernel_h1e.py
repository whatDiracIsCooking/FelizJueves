"""The one-electron hessian kernels: `Target.S`, `T` and `V` at
`derivative_order = 2`, built as `Kernel`s.

    build_h1e_kernel(spec | (target, la, lb)) -> Kernel   one per class
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
from felizjueves.one_electron.kernel_v import v_plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import OPERATOR_OF, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "COMPONENTS",
    "HESSIAN_1E",
    "ZEROED_H",
    "axis_ops",
    "build_h1e_kernel",
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


def build_h1e_kernel(cls: Spec | tuple[Target, int, int], *,
                     name: str | None = None) -> Kernel:
    """The hessian kernel of one one-electron pair class at d2.

    `cls` is a d2 `Target.S` / `T` / `V` `Spec` or `(target, la, lb)`.
    See the module docstring.
    """
    target, la, lb = _class(cls)
    lb_t = lb + 4 if target is Target.T else lb + 2
    head = record_head(la + 2, lb_t)
    E = axis_reader(la + 2, lb_t)
    pool = Locals()
    if target is Target.T:
        head.append(Decl("ebk", Scalar.F64, init=Index("eb_ket", (Var(REC),))))
        values = _t_values(la, lb, E, pool)
    elif target is Target.S:
        values = _s_values(la, lb, operated(E, pool, ""))
    else:
        values = _v_values(la, lb, operated(E, pool, ""))
    what = {Target.S: "overlap", Target.T: "kinetic energy",
            Target.V: "nuclear attraction (the operator's centre-nucleus "
                      "and nucleus-nucleus terms included)"}[target]
    per_rec = (*head, *weights(la, lb),
               Note("the operated factors, each declared where first read"),
               *pool.decls, *values, *_scatter())
    body: tuple[Stmt, ...] = (
        Note(f"hessian of the {what}, class (la, lb) = ({la}, {lb}): "
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
