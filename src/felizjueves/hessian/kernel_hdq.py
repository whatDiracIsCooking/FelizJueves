"""The ket derivative densities: `Target.DQ` at `derivative_order = 1`, the
J chain's pre-contraction differentiated on each ket centre.

    build_hdq_kernel(spec | (lc, ld)) -> Kernel   one per ket pair class
    hdq_name(spec | (lc, ld))         -> str      hdq_{lc}{ld}
    hdq_spec(lc, ld)                  -> Spec     the class's d1 key
    n_q1(lc, ld)                      -> int      D1's last extent
    launch_shape(n_rec, threads)      -> (grid, block)

What the Coulomb Hessian's bra-ket term contracts the `[p|q]` kernel with
(`felizjueves.hessian`'s package docstring). Per ket primitive-pair record
(one thread; exponents `gamma` on the first shell `c`, `delta` on the
second `d`, the record scalar `s`), with the weights of
`gradient.kernel_g1e.weights` at `M = D` -- `D_cd + D_dc` off the diagonal,
so the factor 2 of an off-diagonal ket is in them -- and the per-axis
tables at `(lc + 1, ld + 1)`:

    D1[l,     rec, n] = s SUM_r w_r (dC_l E^{cd})^r_{t_n}
    D1[3 + l, rec, n] = s SUM_r w_r (dD_l E^{cd})^r_{t_n}

over `t_n` in `hermite_targets(lc + ld + 1)`; an entry no row reaches is
stored `0.0`, so every element of the record's three-dimensional slab is
written and nothing need be zeroed. These are `kernel_g1e`'s Hermite
densities `HA`, `HB` (`hermite_densities`, shared), scaled by `s`.

WHAT THE CALLER DOES WITH IT. Slab `l` is the density of a ket of atom
`atom_c`, axis `l`; slab `3 + l` of a ket of `atom_d`. The `[p|q]` kernel
of the class with the bra raised by one and the ket raised by one
(`kernel_hjab.hjab_pq1_spec`) runs once per atom `C` and axis `l` over the
kets touching `C`, each carrying its slab of `C`: that gather, like the
J chain's sort, is the host's (`hessian.host.coulomb_dkets` is its
reference). The kernel never reads an atom.

THE PARAMETER BLOCK, `kernel_gjab`'s record spelling for the ket:

    params  n_rec n_ao (I32)
            zeta_ab[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s  ea_bra  eb_ket
            off_a  off_b  [n_rec]  (I32)
            M[n_ao, n_ao]          the density D
            D1[6, n_rec, n_q1]     (OUT, every element stored)

(`zeta_ab` is the ket's `eta`, `XPA` / `XPB` its `Q - C` / `Q - D`,
`ea_bra` / `eb_ket` its `gamma` / `delta`: the names are the record
head's, shared with the bra kernels.)

BIT FOR BIT, interpreted, against `hessian.host.ket_derivative_records`
(the same sums in the same order; test/test_kernel_hessian.py).

THE ROLLED FORM (#690), the default from `lc + ld = LOWSTACK_MIN_LSUM`.
The densities above are straight-line text, every Hermite index times
every row: `hdq_33` is ~860 KB of source, ~49 KB of `sm_86` stack and
~114 s of nvcc (data/measured/hjab.json). From the floor the kernel has
the SAME name, parameters, launch and `D1` contract and no named density:

    a local HD[6 n_q1], zeroed;
    for each block row (a, b) (`kernel_g1e.lowstack_rows`), weight `wt`
    from M (`lowstack_weight`):
        F{op}[d, t] = (op E)^{a_d b_d}_t, op in "" A B
                                         (`kernel_h1e.operated_value`)
        HD[3 side + k, t] += wt ((F[0, t_x] F[1, t_y]) F[2, t_z])
                                         rolled over the reachable t
    D1[i, rec, n] = s HD[i, n]           every element stored

Each `F` entry is the pooled local's value bitwise, the rows are visited
in `block_rows` order and the product associates as the unrolled one, so
the two forms are the SAME sums in the same order: bit for bit, and both
bitwise the host. The rolled pieces are shared with `kernel_hjab`'s
rolled form (`fill_factors`, `rolled_terms`, `first_densities`).

A builder: vendor-neutral IR. It imports neither `ao/` nor
`hessian.host` nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from collections.abc import Callable

from felizjueves.gradient.kernel_g1e import (
    REC,
    Locals,
    axis_reader,
    derivative_reader,
    hermite_densities,
    launch_shape,
    lowstack_differ,
    lowstack_rows,
    lowstack_weight,
    record_head,
    weights,
)
from felizjueves.hessian.kernel_h1e import f_array, operated_value
from felizjueves.ir import (
    Assign,
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
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "HD_ARRAY",
    "LOWSTACK_MIN_LSUM",
    "build_hdq_kernel",
    "fill_factors",
    "first_densities",
    "hdq_lowstack",
    "hdq_name",
    "hdq_spec",
    "launch_shape",
    "n_q1",
    "rolled_terms",
    "zeroed",
]

# The lc + ld floor of the rolled default (#690): the smallest lc + ld with
# an inadmissible unrolled class in data/measured/hjab.json
# (`devtools/measure.sh hjab-compile`, the rule declared before any row).
LOWSTACK_MIN_LSUM: int | None = 5

HD_ARRAY = "HD"  # the six first-derivative densities, [6, n_q1]

LOWSTACK_NOTE = (
    "ROLLED (#690): no named density; per block row, its weight read from "
    "M, the operated per-axis factors F{op}[d, t] and the six first-"
    "derivative densities accumulated into a local HD as rolled loops")

Cart = tuple[Expr, Expr, Expr]

# record_head's atom scalars: this kernel never reads an atom.
_UNREAD = frozenset({"aa", "ab"})


def hdq_spec(lc: int, ld: int) -> Spec:
    """The ket pair class `(lc, ld)` at derivative order 1, `Target.DQ`."""
    return Spec(Operator.COULOMB, 1, (lc, ld), Target.DQ)


def n_q1(lc: int, ld: int) -> int:
    """`D1`'s last extent: the Hermite functions of order `lc + ld + 1`."""
    return len(hermite_targets(lc + ld + 1))


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else hdq_spec(*cls)
    if spec.target is not Target.DQ or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 Target.DQ key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the Hessian kernels take no levers")
    lc, ld = spec.am
    return lc, ld


def hdq_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_hdq_kernel(cls)` gives its kernel, unbuilt."""
    lc, ld = _class(cls)
    return f"hdq_{lc}{ld}"


def hdq_lowstack(cls: Spec | tuple[int, int]) -> bool:
    """THE SELECTION RULE (#690): True when `LOWSTACK_MIN_LSUM` is set and
    `lc + ld >= LOWSTACK_MIN_LSUM`; there `build_hdq_kernel` emits the
    rolled form by default."""
    lc, ld = _class(cls)
    return LOWSTACK_MIN_LSUM is not None and lc + ld >= LOWSTACK_MIN_LSUM


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _iadd(x: Expr, y: Expr) -> Call:
    return _c(Fn.ADD, x, y)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


# --- the rolled form, shared with kernel_hjab --------------------------------


def fill_factors(ops: tuple[str, ...], a: Cart, b: Cart, width: int,
                 lb_t: int, L_t: int) -> list[Stmt]:  # noqa: N803
    """`F{op}[d * width + t] = (op E)^{a_d b_d}_t` for every `op` of `ops`
    and every reachable `t <= a_d + b_d + len(op)`, from the record head's
    tables at `(la_t, lb_t)`, `L_t = la_t + lb_t`
    (`kernel_h1e.operated_value`, bitwise the unrolled pooled local)."""
    out: list[Stmt] = []
    for op in ops:
        for d in range(3):
            top = _iadd(_iadd(a[d], b[d]), Lit(len(op) + 1))
            out.append(For("q_t", Lit(0), top, Lit(1), (
                *operated_value(op, d, a[d], b[d], Var("q_t"), "o", lb_t,
                                L_t),
                Assign(Index(f_array(op), (_iadd(Lit(d * width),
                                                  Var("q_t")),)), Var("o")),
            )))
    return out


def rolled_terms(ops: tuple[str, str, str], a: Cart, b: Cart, width: int,
                 body: Callable[[Cart, Expr], tuple[Stmt, ...]]) -> For:
    """Three loops over the row's reachable `t` (`t_d <= a_d + b_d +
    len(ops[d])`), the term `q_e = wt ((F0[t_x] F1[t_y]) F2[t_z])` -- the
    unrolled densities' association -- handed to `body(t, q_e)`."""
    t = (Var("q_tx"), Var("q_ty"), Var("q_tz"))

    def f(d: int) -> Index:
        return Index(f_array(ops[d]), (_iadd(Lit(d * width), t[d]),))

    def top(d: int) -> Expr:
        return _iadd(_iadd(a[d], b[d]), Lit(len(ops[d]) + 1))

    z = For(t[2].name, Lit(0), top(2), Lit(1), (
        Decl("q_e", Scalar.F64, init=_mul(Var("wt"), _mul(
            _mul(f(0), f(1)), f(2)))),
        *body(t, Var("q_e"))))
    y = For(t[1].name, Lit(0), top(1), Lit(1), (z,))
    return For(t[0].name, Lit(0), top(0), Lit(1), (y,))


def first_densities(a: Cart, b: Cart, width: int, n_q: int) -> list[Stmt]:
    """`HD[(3 side + k) n_q + idx(t)] += wt PROD_d (op_d E)_{t_d}`, `op_k =
    side`, for the six first derivatives: the unrolled `HA{k}_n`, `HB{k}_n`
    row by row."""
    out: list[Stmt] = []
    for s, side in enumerate("AB"):
        for k in range(3):
            ops = tuple(side if d == k else "" for d in range(3))
            base = Lit((3 * s + k) * n_q)

            def acc(t: Cart, e: Expr, base=base) -> tuple[Stmt, ...]:
                at = Index(HD_ARRAY, (_iadd(base, ef.hermite_index(t)),))
                return (Assign(at, _c(Fn.ADD, at, e)),)
            out.append(rolled_terms(ops, a, b, width, acc))  # type: ignore[arg-type]
    return out


def zeroed(name: str, n: int) -> list[Stmt]:
    """A local F64 `name[n]`, every element set to 0.0."""
    return [Decl(name, Scalar.F64, (Lit(n),)),
            For("z_i", Lit(0), Lit(n), Lit(1),
                (Assign(Index(name, (Var("z_i"),)), Lit(0.0)),))]


def _rolled_values(lc: int, ld: int) -> list[Stmt]:
    """THE ROLLED FORM (module docstring): the same sums as the unrolled
    densities and stores, row by row."""
    L = lc + ld
    width = L + 2  # one axis's operated factor, t <= a_d + b_d + 1
    n = n_q1(lc, ld)
    ops = ("", "A", "B")

    def elem(a: Cart, b: Cart) -> tuple[Stmt, ...]:
        return (*lowstack_weight(lc, ld),
                *fill_factors(ops, a, b, width, ld + 1, L + 2),
                *first_densities(a, b, width, n))

    rec = Var(REC)
    return [
        Note(LOWSTACK_NOTE),
        *lowstack_differ(lc, ld),
        *(Decl(f_array(op), Scalar.F64, (Lit(3 * width),)) for op in ops),
        *zeroed(HD_ARRAY, 6 * n),
        Note("per block row (a, b), weight wt from M: HD[3 side + k, t] += "
             "wt PROD_d (op_d E)^{a_d b_d}_{t_d}, rolled over t"),
        lowstack_rows(lc, ld, elem),
        Note("D1[i, rec, n] = s HD[i, n]: every element stored"),
        For("i", Lit(0), Lit(6), Lit(1), (
            For("n", Lit(0), Lit(n), Lit(1), (
                Assign(Index("D1", (Var("i"), rec, Var("n"))), _mul(
                    Var("sk"), Index(HD_ARRAY, (_iadd(
                        _mul(Var("i"), Lit(n)), Var("n")),)))),)),)),
    ]


def build_hdq_kernel(cls: Spec | tuple[int, int], *,
                     name: str | None = None,
                     lowstack: bool | None = None) -> Kernel:
    """The ket derivative-density kernel of one ket pair class.

    `cls` is a d1 `Target.DQ` `Spec` or `(lc, ld)`. See the module
    docstring. `lowstack` picks the form: `None` (the default) is
    `hdq_lowstack(cls)`, `True` / `False` force either at any class. Both
    forms have the same name, parameters, launch and `D1` contract.
    """
    lc, ld = _class(cls)
    if lowstack is None:
        lowstack = hdq_lowstack(cls)
    head = [s for s in record_head(lc + 1, ld + 1, interleaved=True)
            if not (isinstance(s, Decl) and s.name in _UNREAD)]
    if lowstack:
        return _kernel(lc, ld, (*head, *_rolled_values(lc, ld)), name,
                       ", ROLLED")
    E = axis_reader(lc + 1, ld + 1)
    pool = Locals()
    dA = derivative_reader(E, pool, "A")
    dB = derivative_reader(E, pool, "B")
    dens, names = hermite_densities(lc, ld, E, dA, dB, with_h0=False)
    stores: list[Stmt] = [
        Note("D1[3 side + l, rec, n] = s SUM_r w_r (d_side,l E)^r_n: every "
             "element stored, 0.0 where no row reaches t_n"),
    ]
    rec = Var(REC)
    for side, letter in enumerate("AB"):
        for k in range(3):
            for n in range(n_q1(lc, ld)):
                nm = names.get((f"{letter}{k}", n))
                value: Expr = (Lit(0.0) if nm is None
                               else _c(Fn.MUL, Var("sk"), Var(nm)))
                stores.append(Assign(
                    Index("D1", (Lit(3 * side + k), rec, Lit(n))), value))
    per_rec = (*head, *weights(lc, ld), *pool.decls, *dens, *stores)
    return _kernel(lc, ld, per_rec, name, "")


def _kernel(lc: int, ld: int, per_rec: tuple[Stmt, ...], name: str | None,
            form: str) -> Kernel:
    rec = Var(REC)
    body: tuple[Stmt, ...] = (
        Note(f"the ket derivative densities, ket class (lc, ld) = ({lc}, "
             f"{ld}){form}: the J chain's D_q differentiated on each ket "
             "centre (felizjueves.hessian)"),
        Note("launch: one thread per ket primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, rec, Var("n_rec")), per_rec),
    )
    nr, nao = Var("n_rec"), Var("n_ao")

    def f64(nm: str, shape: tuple[Expr, ...]) -> Param:
        return Param(nm, Scalar.F64, array=True, shape=shape)

    def i32(nm: str) -> Param:
        return Param(nm, Scalar.I32, array=True, shape=(nr,))

    params = (
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        f64("zeta_ab", (nr,)),
        f64("XPA", (nr, Lit(3))),
        f64("XPB", (nr, Lit(3))),
        f64("s", (nr,)),
        f64("ea_bra", (nr,)),
        f64("eb_ket", (nr,)),
        i32("off_a"),
        i32("off_b"),
        f64("M", (nao, nao)),
        Param("D1", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(6), nr, Lit(n_q1(lc, ld)))),
    )
    return Kernel(name or hdq_name((lc, ld)), params, body)
