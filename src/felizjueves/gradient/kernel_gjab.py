"""The Coulomb-energy gradient kernel: `Target.JAB` at `derivative_order =
1`, the J chain's bra back-transform differentiated.

    build_gjab_kernel(spec | (la, lb)) -> Kernel   one per bra pair class
    gjab_name(spec | (la, lb))         -> str      gjab_{la}{lb}
    gjab_spec(la, lb)                  -> Spec     the class's d1 key
    gjab_pq_spec(la, lb, lc, ld)       -> Spec     the [p|q] class it reads
    launch_shape(n_rec, threads)       -> (grid, block)

THE CHAIN. `E_J = 1/2 SUM D_ab D_cd (ab|cd)`, and its ket derivative equals
its bra derivative (the package docstring), so

    dE_J / dR_{A,k} = SUM_ab D_ab SUM_t (dA_k E^{ab})_t J_t

with `J_t = SUM_q [t|q] D_q` the J chain's `J_p` of the bra record -- one
Hermite order HIGHER than `J`'s own, `|t| <= la + lb + 1`, because
`dA E^{ab}` reaches `E^{a+1,b}`. That `J_p` is the ordinary d0 `[p|q]`
kernel of the quartet class with the bra raised by one,
`gjab_pq_spec(la, lb, lc, ld) = Spec(COULOMB, 0, (la + 1, lb, lc, ld),
Target.J)` (so `split` reads `(la + lb + 1, lc + ld)`), over the same bra
records and the J chain's kets (`DQ`, unchanged). Nothing in the `[p|q]`
kernel or its plan is new; a gradient build launches it one class up.

Per bra record (one thread), with the weights of `kernel_g1e.weights`
at `M = D` and the per-axis tables at `(la + 1, lb + 1)`:

    HA_k,t = SUM_r w_r (dA_k E)^{ab}_t,  HB_k,t likewise   |t| <= L + 1
    gA_k   = SUM_t HA_k,t Jp[row, t]                        gB_k likewise
    G[atom_a, k] += s gA_k,   G[atom_b, k] += s gB_k

(`kernel_g1e.hermite_densities`, shared; `s` the host's record scalar).
The result is the whole `dE_J` -- no half, no factor 2: the half of
`E_J` and the bra-plus-ket two cancel.

THE PARAMETER BLOCK, `kernel_jab`'s (the J chain's interleaved spelling)
plus the derivative's and the gradient's:

    params  n_rec n_ao n_atom n_rows (I32)
            zeta_ab[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s  ea_bra  eb_ket
            off_a  off_b  row  atom_a  atom_b  [n_rec]  (I32)
            Jp[n_rows, n_p1]      J_p at order L + 1 (gjab_pq_spec's buffer)
            M[n_ao, n_ao]         the density D
            G[n_atom, 3]          (OUT, read and AtomicAdd-ed)

with `n_p1 = len(hermite_targets(la + lb + 1))`. `G` is ZEROED BY THE
CALLER; every record only adds.

ROUND-OFF, NOT BITWISE, against `gradient.host.coulomb_gradient`, which
sums row by row; the stated bound is in test/test_kernel_gjab.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.gradient.kernel_g1e import (
    REC,
    ZEROED_G,
    Locals,
    axis_reader,
    derivative_reader,
    hermite_densities,
    launch_shape,
    record_head,
    scatter_gradient,
    weights,
)
from felizjueves.ir import (
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
    "build_gjab_kernel",
    "gjab_name",
    "gjab_pq_spec",
    "gjab_spec",
    "launch_shape",
    "n_p1",
]


def gjab_spec(la: int, lb: int) -> Spec:
    """The bra pair class `(la, lb)` at derivative order 1, `Target.JAB`."""
    return Spec(Operator.COULOMB, 1, (la, lb), Target.JAB)


def gjab_pq_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The d0 `[p|q]` class whose `J_p` the gradient of bra class
    `(la, lb)` against ket class `(lc, ld)` reads: the bra raised by one."""
    return Spec(Operator.COULOMB, 0, (la + 1, lb, lc, ld), Target.J)


def n_p1(la: int, lb: int) -> int:
    """The second extent of `Jp`: the Hermite functions of order `L + 1`."""
    return len(hermite_targets(la + lb + 1))


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else gjab_spec(*cls)
    if spec.target is not Target.JAB or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 Target.JAB key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    la, lb = spec.am
    return la, lb


def gjab_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_gjab_kernel(cls)` gives its kernel, unbuilt."""
    la, lb = _class(cls)
    return f"gjab_{la}{lb}"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def build_gjab_kernel(cls: Spec | tuple[int, int], *,
                      name: str | None = None) -> Kernel:
    """The Coulomb-energy gradient kernel of one bra pair class.

    `cls` is a d1 `Target.JAB` `Spec` or `(la, lb)`. See the module
    docstring.
    """
    la, lb = _class(cls)
    head = record_head(la + 1, lb + 1, interleaved=True)
    head.append(Decl("jr", Scalar.I32, init=Index("row", (Var(REC),))))
    E = axis_reader(la + 1, lb + 1)
    pool = Locals()
    dA = derivative_reader(E, pool, "A")
    dB = derivative_reader(E, pool, "B")
    dens, names = hermite_densities(la, lb, E, dA, dB, with_h0=False)
    values: list[Stmt] = [
        Note("gA_k = SUM_t HA_k,t J_t (and B), J_t = Jp[row, t] at Hermite "
             f"order {la + lb + 1}: the bra derivative of E_J"),
    ]
    for side in "AB":
        for k in range(3):
            terms = [_c(Fn.MUL, Var(names[f"{side}{k}", n]),
                        Index("Jp", (Var("jr"), Lit(n))))
                     for n in range(n_p1(la, lb))
                     if (f"{side}{k}", n) in names]
            v: Expr = terms[0]
            for t in terms[1:]:
                v = _c(Fn.ADD, v, t)
            values.append(Decl(f"g{side}{k}", Scalar.F64, init=v))
    per_rec = (*head, *weights(la, lb), *pool.decls, *dens, *values,
               *scatter_gradient())
    body: tuple[Stmt, ...] = (
        Note(f"gradient of E_J, bra class (la, lb) = ({la}, {lb}): "
             "G[A, k] += SUM_ab D_ab SUM_t (dA_k E^ab)_t J_t "
             "(felizjueves.gradient)"),
        Note(ZEROED_G),
        Note("launch: one thread per bra primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, Var(REC), Var("n_rec")), per_rec),
    )
    nr, nao, nat, nrows = (Var("n_rec"), Var("n_ao"), Var("n_atom"),
                           Var("n_rows"))

    def f64(nm: str, shape: tuple[Expr, ...]) -> Param:
        return Param(nm, Scalar.F64, array=True, shape=shape)

    def i32(nm: str) -> Param:
        return Param(nm, Scalar.I32, array=True, shape=(nr,))

    params = (
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_atom", Scalar.I32),
        Param("n_rows", Scalar.I32),
        f64("zeta_ab", (nr,)),
        f64("XPA", (nr, Lit(3))),
        f64("XPB", (nr, Lit(3))),
        f64("s", (nr,)),
        f64("ea_bra", (nr,)),
        f64("eb_ket", (nr,)),
        i32("off_a"),
        i32("off_b"),
        i32("row"),
        i32("atom_a"),
        i32("atom_b"),
        f64("Jp", (nrows, Lit(n_p1(la, lb)))),
        f64("M", (nao, nao)),
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3))),
    )
    return Kernel(name or gjab_name((la, lb)), params, body)
