"""The one-electron derivative MATRICES (#687): `S`, `T` and `V` at
`derivative_order = 1` in a matrix form (`spec.FORM_TARGETS`), built as
`Kernel`s.

    build_ip1e_kernel(spec | (target, la, lb[, form])) -> Kernel  one per class
    ip1e_name(spec | (target, la, lb[, form]))         -> str
                                     ips_ / ipt_ / ipv_ / iprv_{la}{lb}
    ip1e_spec(target, la, lb, form="ip1")              -> Spec   the d1 key
    launch_shape(n_rec, threads)                       -> (grid, block)

WHY A MATRIX. An analytic RHF Hessian's CPHF right-hand side, per
perturbation `x = (atom X, axis k)`, needs the derivative Fock matrix `F^x =
h^x + J^x[D] - 1/2 K^x[D]` and `S^x` explicitly, occ x virt. A d1 kernel of
`gradient.kernel_g1e` writes a CONTRACTION, `G[atom, k] += SUM_mn M_mn
dX_mn/dR`, which cannot give them. These kernels write the one-electron
half, `S^x` and `h^x = T^x + V^x`, as `[3, n_ao, n_ao]` Cartesian matrices,
row-major (spherical is the caller's, `U X U^T`).

THE ip1 FORM (`ips_`, `ipt_`, `ipv_`; key ``overlap/d1.ip1/(la,lb)/b1/s/
literal``): the derivative on the centre of the matrix's FIRST index,

    X1[k, m, n] += d X_mn / dA_k        A the centre of m

libcint's `int1e_ipovlp`, `int1e_ipkin`, `int1e_ipnuc`, NEGATED (libcint
differentiates in the electron's coordinate). All atoms in one launch per
pair class, the `gs` / `gt` / `gv` launch: one thread per primitive-pair
record of shell pair `i <= j` on centres `A`, `B`, writing its `(m, n)`
block with the `dA` rule and, off the diagonal, its `(n, m)` block with
the `dB` rule (`gradient.kernel_g1e`'s rules, record head and per-axis
tables `axis_tables_fragment(la + 1, lb + 1)`, `lb + 3` for `T`, reused
unchanged). A DIAGONAL shell pair (`off_a == off_b`, possible only when
`la == lb`) writes the `(m, n)` block alone: its records are every ORDERED
primitive pair, so the `dA` sum over them is already the block's whole
first-index derivative. Atom `X`'s matrix is the caller's: the rows of `m`
on `X` of `X1`, plus that slab's transpose.

- `ips`: per row and axis, the `t = 0` factors with axis `k`
  differentiated -- `gs`'s expression with no weight.
- `ipt`: `gt`'s bracket `K_x S_y S_z + S_x K_y S_z + S_x S_y K_z`, the
  factor on axis `k` replaced by its derivative, per row.
- `ipv`: the basis function's derivative with every nucleus's operator
  held fixed. Per record the nuclei's potential in Hermite space, once,

      U_t = -SUM_C Z_C [t]^(0)(P - C)       |t| <= L + 1

  (per nucleus, serially, `[t]^(0)` from `kernel_v.v_plan((la, lb + 1))`
  into a local `R` as it forms, `recurrence.store_targets`, then added
  into `U`), then per block row `(a, b)` and axis `k`, `2 alpha SUM_t
  E^{a+1_k,b}_t U_t - a_k SUM_t E^{a-1_k,b}_t U_t` (and `B`'s), each SUM
  rolled over the per-axis tables (`kernel_g1e.lowstack_derivative`, the
  low-stack `gv`'s). There is no unrolled `ipv`: `U` is the only Hermite
  quantity, and the frame is the tables, `R` and `U` at every class.

THE rinv FORM (`iprv_`; ``nuclear/d1.rinv/(la,lb)/b1/v/literal``): `V`'s
OPERATOR term, which no first-index derivative holds -- `d/dC_k` of the
operator at nucleus `C` moved with its nucleus,

    XC[k, m, n] += d V^C_mn / dC_k = s Z_C SUM_t E^{ab}_t [t + 1_k]^(0)(P - C)

(`d/dC_k [t]^(0)(P - C) = -[t + 1_k]^(0)`, times `V^C`'s `-Z_C`), which is
`-Z_C (iprinv + iprinv^T)` with libcint's `rinv` origin at `C`. ONE LAUNCH
PER NUCLEUS, its index the kernel argument `nuc` -- the form the consumer
expects, as PySCF's `with_rinv_at_nucleus` -- writing `[3, n_ao, n_ao]`
for that nucleus alone; `kernel_g1e`'s `gv` has the same term reduced
against `M` (its Hellmann-Feynman row). The matrix is symmetric, so a row
writes one value to `(m, n)` and, off the diagonal, to `(n, m)`; the
tables are the class's own, `axis_tables_fragment(la, lb)`, and the sum
`ef.rolled_e_sum(..., up=k)`.

With both, for `V`: `dV/dR_X = (X1 rows on X) + (that)^T + XC(X)`, and
translational invariance is `X1 + X1^T + SUM_C XC(C) = 0`
(test/test_kernel_ip1e.py holds it). `S` and `T` have no operator term.

THE PARAMETER BLOCK, `kernel_g1e`'s per-record spelling, less what a
matrix does not read (`M`, `atom_a`, `atom_b`) and plus its own out:

    params  n_rec n_ao [n_atom: ipv, iprv] [nuc: iprv] (I32)
            zeta_ab  pa_x pa_y pa_z  pb_x pb_y pb_z  [p_x p_y p_z: ipv, iprv]
                s_ab  [ea_bra eb_ket: ip1]                  [n_rec] per record
            off_a  off_b  [n_rec]  (I32)
            [Z[n_atom]  Cnuc[n_atom, 3]: ipv, iprv]          the nuclei
            X1[3, n_ao, n_ao] (ip1) / XC[3, n_ao, n_ao] (iprv)
                                               (OUT, read and AtomicAdd-ed)

The out is ZEROED BY THE CALLER before a build's first launch into it;
every kernel only adds. `n_cart(la) n_cart(lb)` rows times three axes times
(one, or two off the diagonal) `AtomicAdd`s per record: a shell's rows are
written by every record of every pair it is in.

ROUND-OFF, NOT BITWISE, against the host (`gradient.host.overlap_ip1`,
`kinetic_ip1`, `potential_ip1`, `potential_rinv`): the host sums each row
with its own association, `ipv` folds the nuclei into `U` first, and the
plan's MILD recurrence forms `[t]^(0)`. test/test_kernel_ip1e.py states
its bound.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
(its witness), nor the interpreter, nor `reference/`.
"""

from __future__ import annotations

from collections.abc import Callable

from felizjueves.gradient.kernel_g1e import (
    REC,
    Locals,
    axis_reader,
    derivative_reader,
    launch_shape,
    lowstack_derivative,
    lowstack_rows,
)
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Expr,
    Factor,
    Fn,
    For,
    If,
    Index,
    Inline,
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
    recurrence_body,
    store_targets,
)
from felizjueves.one_electron.kernel_v import v_plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import FORM_TARGETS, OPERATOR_OF, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "IP1E",
    "ZEROED_X",
    "build_ip1e_kernel",
    "ip1e_name",
    "ip1e_spec",
    "launch_shape",
    "out_array",
]

NUC = "c"  # the nucleus loop index, serial inside the record (ipv)
R_ARRAY = "R"  # [t]^(0) of one nucleus, hermite_targets(L + 1) order
U_ARRAY = "U"  # -SUM_C Z_C [t]^(0)(P - C), the same order (ipv)

# Every (target, form) built, in ladder order.
IP1E = ((Target.S, "ip1"), (Target.T, "ip1"), (Target.V, "ip1"),
        (Target.V, "rinv"))
_PREFIX = {(Target.S, "ip1"): "ips", (Target.T, "ip1"): "ipt",
           (Target.V, "ip1"): "ipv", (Target.V, "rinv"): "iprv"}

ATOMIC_REASON = (
    "AtomicAdd: every record of every shell pair a shell is in adds into "
    "that shell's rows and columns of the matrix")

ZEROED_X = (
    "PRECONDITION: the out matrix is zeroed by the caller before the first "
    "launch of a build into it; this kernel only adds")

Cls = Spec | tuple[Target, int, int] | tuple[Target, int, int, str]


def ip1e_spec(target: Target, la: int, lb: int, form: str = "ip1") -> Spec:
    """The pair class `(la, lb)` of `target` at d1 in matrix form `form`."""
    return Spec(OPERATOR_OF[target], 1, (la, lb), target, form=form)


def _class(cls: Cls) -> tuple[Target, str, int, int]:
    spec = cls if isinstance(cls, Spec) else ip1e_spec(*cls)
    if spec.derivative_order != 1 or spec.form not in FORM_TARGETS:
        raise ValueError(f"{spec.key()} is not a d1 matrix-form key")
    la, lb = spec.am
    return spec.target, spec.form, la, lb


def ip1e_name(cls: Cls) -> str:
    """The name `build_ip1e_kernel(cls)` gives its kernel, unbuilt."""
    target, form, la, lb = _class(cls)
    return f"{_PREFIX[target, form]}_{la}{lb}"


def out_array(cls: Cls) -> str:
    """The kernel's one OUT: `X1` for ip1, `XC` for rinv."""
    return "XC" if _class(cls)[1] == "rinv" else "X1"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _add(x: Expr, y: Expr) -> Call:
    return _c(Fn.ADD, x, y)


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def _head(la_t: int, lb_t: int, *, rule: bool) -> list[Stmt]:
    """Record `rec`'s scalars and the axis tables at `(la_t, lb_t)`, inlined:
    `kernel_g1e.record_head`'s, less the atoms (a matrix is indexed by
    AO) and, without the derivative `rule`, the exponents."""
    rec = Var(REC)
    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA),
                              ef.HALF_INV_ZETA: Var(ef.HALF_INV_ZETA)}
    for d, a in enumerate("xyz"):
        binds[Factor.pa(d).name_in_kernel] = Index(f"pa_{a}", (rec,))
        binds[Factor.pb(d).name_in_kernel] = Index(f"pb_{a}", (rec,))
    for arr in ef.AXIS_ARRAYS:
        binds[arr] = Var(arr)
    exps: list[Stmt] = [
        Decl("ea2", Scalar.F64, init=_mul(Lit(2.0), Index("ea_bra", (rec,)))),
        Decl("eb2", Scalar.F64, init=_mul(Lit(2.0), Index("eb_ket", (rec,)))),
    ] if rule else []
    n = ef.axis_table_extent(la_t, lb_t)
    up = ("one index above the class on each side, for the derivative rule"
          if rule else "the class's own")
    frag = ef.axis_tables_fragment(la_t, lb_t)
    binds = {k: v for k, v in binds.items() if k in frag.binds}
    half = [Decl(ef.HALF_INV_ZETA, Scalar.F64,
                 init=_c(Fn.DIV, Lit(0.5), Var(ef.ZETA)))
            ] if ef.HALF_INV_ZETA in binds else []  # (ss) tables read none
    return [
        Decl(ef.ZETA, Scalar.F64, init=Index("zeta_ab", (rec,))),
        *half,
        Decl("sk", Scalar.F64, init=Index("s_ab", (rec,))),
        *exps,
        Decl("oa", Scalar.I32, init=Index("off_a", (rec,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (rec,))),
        Note(f"the undressed per-axis E^{{ij}}_t, i <= {la_t}, j <= {lb_t}: "
             f"{up}"),
        *(Decl(arr, Scalar.F64, (Lit(n),)) for arr in ef.AXIS_ARRAYS),
        Inline(frag, binds),
    ]


def _differ(la: int, lb: int) -> list[Stmt]:
    """For `la == lb`, the I32 `differ`: 1 off the diagonal (`oa != ob`),
    where a record also writes the transposed block."""
    if la != lb:
        return []
    return [Decl("differ", Scalar.I32, init=_add(
        _c(Fn.LT, Var("oa"), Var("ob")), _c(Fn.LT, Var("ob"), Var("oa"))))]


def _write(la: int, lb: int, out: str, k: int, i: Expr, j: Expr,
           va: Expr, vb: Expr) -> list[Stmt]:
    """`out[k, oa + i, ob + j] += va` and, off the diagonal, `out[k, ob +
    j, oa + i] += vb`."""
    m, n = _add(Var("oa"), i), _add(Var("ob"), j)
    mine = AtomicAdd(Index(out, (Lit(k), m, n)), va)
    other = AtomicAdd(Index(out, (Lit(k), n, m)), vb)
    if la != lb:
        return [mine, other]
    return [mine, If(Var("differ"), (other,))]


def _s_values(la: int, lb: int, dA, dB, E) -> list[Stmt]:
    """`ips`: per row `r = (a, b)` and axis `k`, `PROD_d f_d` with `f_k`
    the `dA` (`dB`) rule at `t = 0`."""
    out: list[Stmt] = []
    nb = len(ef.cartesians(lb))
    for r, (a, b) in enumerate(ef.block_rows(la, lb)):
        for k in range(3):
            v = []
            for D in (dA, dB):
                f = [D(d, a[d], b[d], 0) if d == k else E(d, a[d], b[d], 0)
                     for d in range(3)]
                v.append(_mul(Var("sk"), _mul(_mul(f[0], f[1]), f[2])))
            out += _write(la, lb, "X1", k, Lit(r // nb), Lit(r % nb), *v)
    return out


def _t_values(la: int, lb: int, E, pool: Locals) -> list[Stmt]:
    """`ipt`: `gt`'s bracket per row, the factor on axis `k` differentiated
    (`kernel_g1e._t_values`' expressions, with no weight)."""
    eb = Var("ebk")

    def S(d: int, i: int, j: int) -> Expr:
        return E(d, i, j, 0)

    def K(d: int, i: int, j: int) -> Var:
        def make() -> Expr:  # e_fragment._shift's expression, restated
            k = _add(_mul(_mul(_mul(Lit(-2.0), eb), eb), S(d, i, j + 2)),
                     _mul(_mul(eb, Lit(float(2 * j + 1))), S(d, i, j)))
            if j >= 2:
                k = _c(Fn.SUB, k, _mul(Lit(0.5 * j * (j - 1)), S(d, i, j - 2)))
            return k
        return pool.get(f"K{'xyz'[d]}_{i}_{j}", make)

    def deriv(f: Callable[[int, int, int], Expr], side: str, d: int, i: int,
              j: int, tag: str) -> Var:
        def make() -> Expr:
            if side == "A":
                out: Expr = _mul(Var("ea2"), f(d, i + 1, j))
                if i:
                    out = _c(Fn.SUB, out, _mul(Lit(float(i)), f(d, i - 1, j)))
            else:
                out = _mul(Var("eb2"), f(d, i, j + 1))
                if j:
                    out = _c(Fn.SUB, out, _mul(Lit(float(j)), f(d, i, j - 1)))
            return out
        return pool.get(f"d{side}{tag}{'xyz'[d]}_{i}_{j}", make)

    out: list[Stmt] = []
    nb = len(ef.cartesians(lb))
    for r, (a, b) in enumerate(ef.block_rows(la, lb)):
        for k in range(3):
            v = []
            for side in "AB":
                parts = []
                for m in range(3):  # the bracket's term with K on axis m
                    f: list[Expr] = []
                    for d in range(3):
                        if d == k:
                            f.append(deriv(K if d == m else S, side, d, a[d],
                                           b[d], "K" if d == m else "S"))
                        else:
                            f.append(K(d, a[d], b[d]) if d == m
                                     else S(d, a[d], b[d]))
                    parts.append(_mul(_mul(f[0], f[1]), f[2]))
                v.append(_mul(Var("sk"), _add(_add(parts[0], parts[1]),
                                              parts[2])))
            out += _write(la, lb, "X1", k, Lit(r // nb), Lit(r % nb), *v)
    return out


def _nucleus(la: int, lb: int, nuc: Expr) -> tuple[list[Stmt], int]:
    """Nucleus `nuc`'s `Zc`, `P - C` and `[t]^(0)`, `|t| <= L + 1`, each
    stored into `R` as it forms; and the number of targets."""
    L = la + lb
    dag, order = v_plan((la, lb + 1))
    t1 = hermite_targets(L + 1)
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    return [
        Decl("Zc", Scalar.F64, init=Index("Z", (nuc,))),
        *(Decl(nm, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                       Index("Cnuc", (nuc, Lit(d)))))
          for d, nm in enumerate(PQ_PARAMS)),
        *prelude,
        Note(f"[t]^(0) for every |t| <= L + 1 = {L + 1}, each stored into "
             "R as it forms (recurrence.store_targets)"),
        *store_targets(recurrence_body(dag, order, fundamental),
                       frozenset(t1), L + 1, R_ARRAY),
    ], len(t1)


def _point(la: int, lb: int) -> list[Stmt]:
    """`P` and `theta = sqrt(zeta)`, and the local `R`."""
    n = len(hermite_targets(la + lb + 1))
    return [
        *(Decl(f"P{d}", Scalar.F64, init=Index(f"p_{a}", (Var(REC),)))
          for d, a in enumerate("xyz")),
        Note("theta = sqrt(zeta): a unit point-charge ket (kernel_v)"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, Var(ef.ZETA))),
        Decl(R_ARRAY, Scalar.F64, (Lit(n),)),
    ]


def _rows(la: int, lb: int, elem) -> For:
    """`kernel_g1e.lowstack_rows`' nest, each element given `rb` first."""
    def body(a, b) -> tuple[Stmt, ...]:
        return (Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),
                *elem(a, b))
    return lowstack_rows(la, lb, body)


def _v_values(la: int, lb: int) -> list[Stmt]:
    """`ipv`'s body after the head (module docstring)."""
    nuc, h = Var(NUC), Var("h")
    per_nuc, n_t = _nucleus(la, lb, nuc)
    less = _c(Fn.SUB, Index(U_ARRAY, (h,)),
              _mul(Var("Zc"), Index(R_ARRAY, (h,))))
    per_nuc += [
        Note("U_t -= Z_C [t]^(0): the nuclei's potential in Hermite space"),
        For("h", Lit(0), Lit(n_t), Lit(1),
            (Assign(Index(U_ARRAY, (h,)), less),)),
    ]

    def u_read(t: Expr) -> Expr:
        return Index(U_ARRAY, (t,))

    def elem(a, b) -> tuple[Stmt, ...]:
        out: list[Stmt] = [
            *(Decl(f"g{s}{k}", Scalar.F64, init=Lit(0.0))
              for s in "AB" for k in range(3)),
            *lowstack_derivative(la, lb, a, b, u_read, "sk", Fn.ADD),
        ]
        for k in range(3):
            out += _write(la, lb, "X1", k, Var("ra"), Var("rb"),
                          Var(f"gA{k}"), Var(f"gB{k}"))
        return tuple(out)

    return [
        *_differ(la, lb),
        *_point(la, lb),
        Decl(U_ARRAY, Scalar.F64, (Lit(n_t),)),
        For("h", Lit(0), Lit(n_t), Lit(1),
            (Assign(Index(U_ARRAY, (h,)), Lit(0.0)),)),
        Note("the nuclei, serially inside the record (kernel_v's mapping)"),
        For(NUC, Lit(0), Var("n_atom"), Lit(1), tuple(per_nuc)),
        Note("per block row (a, b) and axis k: sk (2 alpha SUM_t E^{a+1_k,b}_t "
             "U_t - a_k SUM_t E^{a-1_k,b}_t U_t) into (m, n), and B's rule "
             "into (n, m), each SUM rolled over the per-axis tables "
             "(kernel_g1e.lowstack_derivative)"),
        _rows(la, lb, elem),
    ]


def _rinv_values(la: int, lb: int) -> list[Stmt]:
    """`iprv`'s body after the head (module docstring, THE rinv FORM)."""
    per_nuc, _ = _nucleus(la, lb, Var("nuc"))

    def r_read(t: Expr) -> Expr:
        return Index(R_ARRAY, (t,))

    def elem(a, b) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for k in range(3):
            v = _mul(Var("skz"), Var(f"hC{k}"))
            out += [*ef.rolled_e_sum(la, lb, a, b, r_read, f"hC{k}",
                                     tag=f"pC{k}", up=k),
                    *_write(la, lb, "XC", k, Var("ra"), Var("rb"), v, v)]
        return tuple(out)

    return [
        *_differ(la, lb),
        *_point(la, lb),
        Note("nucleus nuc alone: its operator term, one launch per nucleus"),
        *per_nuc,
        Decl("skz", Scalar.F64, init=_mul(Var("sk"), Var("Zc"))),
        Note("per block row (a, b) and axis k: s Z_C SUM_t E^{ab}_t "
             "R[t + 1_k] into (m, n) and (n, m), the sum rolled over the "
             "per-axis tables (e_fragment.rolled_e_sum)"),
        _rows(la, lb, elem),
    ]


def build_ip1e_kernel(cls: Cls, *, name: str | None = None) -> Kernel:
    """The derivative-matrix kernel of one one-electron pair class.

    `cls` is a d1 matrix-form `Spec` (`spec.FORM_TARGETS`) or `(target, la,
    lb[, form])`, `form` "ip1" by default. See the module docstring."""
    target, form, la, lb = _class(cls)
    out = "XC" if form == "rinv" else "X1"
    pool = Locals()
    if form == "rinv":
        head = _head(la, lb, rule=False)
        values = _rinv_values(la, lb)
    else:
        lb_t = lb + 3 if target is Target.T else lb + 1
        head = _head(la + 1, lb_t, rule=True)
        E = axis_reader(la + 1, lb_t)
        if target is Target.S:
            values = _s_values(la, lb, derivative_reader(E, pool, "A"),
                               derivative_reader(E, pool, "B"), E)
        elif target is Target.T:
            head.append(Decl("ebk", Scalar.F64,
                             init=Index("eb_ket", (Var(REC),))))
            values = _t_values(la, lb, E, pool)
        else:
            values = _v_values(la, lb)
    if target is not Target.V:
        values = [*_differ(la, lb),
                  Note("the derivative factors, each declared where first "
                       "read"),
                  *pool.decls, Note(ATOMIC_REASON), *values]
    else:
        values = [Note(ATOMIC_REASON), *values]
    what = {
        (Target.S, "ip1"): "X1[k, m, n] += dS_mn/dA_k, A the centre of m",
        (Target.T, "ip1"): "X1[k, m, n] += dT_mn/dA_k, A the centre of m",
        (Target.V, "ip1"): ("X1[k, m, n] += dV_mn/dA_k, A the centre of m, "
                            "every nucleus's operator held fixed"),
        (Target.V, "rinv"): ("XC[k, m, n] += dV^C_mn/dC_k, nucleus C = nuc's "
                             "operator term"),
    }[target, form]
    body: tuple[Stmt, ...] = (
        Note(f"derivative matrix, class (la, lb) = ({la}, {lb}): {what} "
             "(hessian.kernel_ip1e)"),
        Note(ZEROED_X),
        Note("launch: one thread per primitive-pair record"),
        Decl(REC, Scalar.I32, init=_add(Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, Var(REC), Var("n_rec")), (*head, *values)),
    )
    nr, nao, nat = Var("n_rec"), Var("n_ao"), Var("n_atom")
    nuclear = target is Target.V
    params = [
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        *((Param("n_atom", Scalar.I32),) if nuclear else ()),
        *((Param("nuc", Scalar.I32),) if form == "rinv" else ()),
        _f64("zeta_ab", (nr,)),
        *(_f64(f"pa_{a}", (nr,)) for a in "xyz"),
        *(_f64(f"pb_{a}", (nr,)) for a in "xyz"),
        *((_f64(f"p_{a}", (nr,)) for a in "xyz") if nuclear else ()),
        _f64("s_ab", (nr,)),
        *((_f64("ea_bra", (nr,)), _f64("eb_ket", (nr,)))
          if form == "ip1" else ()),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        *((_f64("Z", (nat,)), _f64("Cnuc", (nat, Lit(3)))) if nuclear
          else ()),
        Param(out, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(3), nao, nao)),
    ]
    return Kernel(name or ip1e_name(cls), tuple(params), body)
