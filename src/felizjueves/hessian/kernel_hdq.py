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

ROUND-OFF, NOT BITWISE, against `hessian.host.ket_derivative_records`
(the same sums, the host's grouping of `s`); the bound is in
test/test_kernel_hessian.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor
`hessian.host` nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.gradient.kernel_g1e import (
    REC,
    Locals,
    axis_reader,
    derivative_reader,
    hermite_densities,
    launch_shape,
    record_head,
    weights,
)
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Fn,
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
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "build_hdq_kernel",
    "hdq_name",
    "hdq_spec",
    "launch_shape",
    "n_q1",
]

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


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def build_hdq_kernel(cls: Spec | tuple[int, int], *,
                     name: str | None = None) -> Kernel:
    """The ket derivative-density kernel of one ket pair class.

    `cls` is a d1 `Target.DQ` `Spec` or `(lc, ld)`. See the module
    docstring.
    """
    lc, ld = _class(cls)
    head = [s for s in record_head(lc + 1, ld + 1, interleaved=True)
            if not (isinstance(s, Decl) and s.name in _UNREAD)]
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
    body: tuple[Stmt, ...] = (
        Note(f"the ket derivative densities, ket class (lc, ld) = ({lc}, "
             f"{ld}): the J chain's D_q differentiated on each ket centre "
             "(felizjueves.hessian)"),
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
