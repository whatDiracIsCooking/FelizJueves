"""The ip1 bra half of the Coulomb derivative matrix: `Target.JX` at
`derivative_order = 1`, the J chain's bra back-transform differentiated and
scattered into `(m, n)` (#688).

    build_jx_kernel(spec | (la, lb)) -> Kernel   one per bra class
    jx_name(spec | (la, lb))         -> str      jx_{la}{lb}
    jx_spec(la, lb)                  -> Spec     the class's d1 key
    jx_pq_spec(la, lb, lc, ld)       -> Spec     the [p|q] class it reads
    n_p1(la, lb)                     -> int      Jp's second extent
    launch_shape(n_rec, threads)     -> (grid, block)

WHAT IT IS FOR. The two-electron Coulomb part of the derivative Fock
matrix `F^x` (the CPHF right-hand side of the RHF Hessian): for atom `X`
and axis `k`,

    J^X_mn = SUM_ls D_ls d(mn|ls) / dR_{X,k}

as an AO matrix. It splits into a bra half and a ket half
(`hessian.host_jx`, docs/implementation/hessian.md, "The Coulomb
derivative matrix"). The bra half is the ip1 form,

    J1[k, m, n] = SUM_ls (dA_k m n|ls) D_ls,   A the centre of m,

which PySCF's `grad.rhf.get_jk` returns as `vj` (`-int2e_ip1` contracted
`ijkl,lk->ij`: libcint's `nabla` is `-d/dA`), and the caller takes atom
`X`'s rows plus their transpose:

    J^X_bra[k] = P_X J1[k] + (P_X J1[k])^T,  P_X the rows of AOs on X.

THIS KERNEL WRITES J1, all atoms in one launch per bra class. The chain is
`coulomb_gradient`'s (`gradient.kernel_gjab`) up to its last step: the
ket phase (`DQ`) unchanged, the raised `[p|q]` at `jx_pq_spec` =
`gjab_pq_spec` unchanged, `J_t` over `|t| <= la + lb + 1`; then, per bra
record (one thread) and Cartesian pair `(a, b)` of the class,

    gA_k = s SUM_t (dA_k E^{ab})_t Jp[row, t]
    gB_k = s SUM_t (dB_k E^{ab})_t Jp[row, t]
    J1[k, oa + i, ob + j] += gA_k        the derivative on m = oa + i
    J1[k, ob + j, oa + i] += gB_k        on n = ob + j, the first index

with `dA E^{ij}_t = 2 alpha E^{i+1,j}_t - i E^{i-1,j}_t` (and `dB`) on
axis `k` -- `gjab`'s rule, with no weight `M`: where `gjab` contracts the
rows with `D` and adds into `G[atom]`, this kernel scatters each row as
`jab` does. J1 is NOT symmetric, so there is no mirror: the `B` term is the
transposed element.

A DIAGONAL SHELL PAIR (`off_a == off_b`, so `la == lb`) writes the `A`
term alone, every `(i, j)`: its records are every ORDERED primitive pair
(`ao.pairs.primitive_pairs`), so the `B` term of `(i, j)` at primitives
`(p, q)` is the `A` term of `(j, i)` at `(q, p)`, already written. The
offset test is `jab`'s, `(off_a < off_b) + (off_b < off_a)`; for `la !=
lb` no pair is diagonal and no branch is built.

ONE FORM, ROLLED (`gjab`'s low-stack shape): per record the head and the
per-axis tables one index up (`kernel_g1e.record_head`, without the
atoms), then per block row `(a, b)` the six sums as rolled loops over the
tables (`kernel_g1e.lowstack_derivative`, its weight the record scalar
`s`) reading the record's `Jp` row where it lies, and three or six
`AtomicAdd`s. Every row is its own output, so the unrolled form's text
grows as rows x Hermite targets (no density folds the rows first, as in
`gjab`); whether an unrolled form pays below some `L` is a measurement
nobody has made, and is not built.

THE PARAMETER BLOCK, `kernel_jab`'s (the J chain's interleaved spelling)
plus the derivative's exponents:

    params  n_rec n_ao n_rows (I32)
            zeta_ab[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s  ea_bra  eb_ket
            off_a  off_b  row  [n_rec]  (I32)
            Jp[n_rows, n_p1]      J_p at order L + 1 (jx_pq_spec's buffer)
            J1[3, n_ao, n_ao]     (OUT, read and AtomicAdd-ed)

with `n_p1 = len(hermite_targets(la + lb + 1))`. `J1` is ZEROED BY THE
CALLER; every record only adds. No `M`, no atoms: the kernel never sees an
atom, the caller's gather does (above).

THE KET HALF NEEDS NO KERNEL. `SUM_ls (mn|d_X(ls)) D_ls` is the ordinary J
build over the derivative kets of atom `X`: `hdq`'s `D1` slabs gathered per
`(X, axis)` as `hjab`'s bra-ket term gathers them, through the d0 `[p|q]`
of `(la, lb | lc + 1, ld)` and the d0 `jab` (`hessian.host_jx` is its
reference; test/test_kernel_jx.py runs that chain interpreted).

ROUND-OFF, NOT BITWISE, against `hessian.host_jx.coulomb_ip1`, which forms
each derivative before the sum; the bound is in test/test_kernel_jx.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor the host
references nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.gradient.kernel_g1e import (
    REC,
    launch_shape,
    lowstack_derivative,
    lowstack_rows,
    record_head,
)
from felizjueves.gradient.kernel_gjab import gjab_pq_spec, n_p1
from felizjueves.ir import (
    AtomicAdd,
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
from felizjueves.mmd import e_fragment as ef
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "ZEROED_J1",
    "build_jx_kernel",
    "jx_name",
    "jx_pq_spec",
    "jx_spec",
    "launch_shape",
    "n_p1",
]

ATOMIC_REASON = (
    "AtomicAdd: several primitive records of one bra shell pair add into the "
    "same J1 elements (kernel_jab's reason, without the mirror)")

ZEROED_J1 = (
    "PRECONDITION: J1 is zeroed by the caller before the first JX launch of "
    "a build; this kernel only adds")

_DIAGONAL = (
    "diagonal bra shell pair (off_a == off_b): the A term alone, every "
    "(i, j) -- the B term of (i, j) at primitives (p, q) is the A term of "
    "(j, i) at (q, p), a record of its own")


def jx_spec(la: int, lb: int) -> Spec:
    """The bra pair class `(la, lb)` at derivative order 1, `Target.JX`."""
    return Spec(Operator.COULOMB, 1, (la, lb), Target.JX)


def jx_pq_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The d0 `[p|q]` class whose `J_p` bra class `(la, lb)` reads against
    ket class `(lc, ld)`: the bra raised by one, `gjab`'s."""
    return gjab_pq_spec(la, lb, lc, ld)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else jx_spec(*cls)
    if spec.target is not Target.JX:
        raise ValueError(f"{spec.key()} is not a Target.JX key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the JX kernel takes no levers")
    la, lb = spec.am
    return la, lb


def jx_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_jx_kernel(cls)` gives its kernel, unbuilt."""
    la, lb = _class(cls)
    return f"jx_{la}{lb}"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _rows(la: int, lb: int) -> list[Stmt]:
    """Per block row `(a, b)`: the six sums and the scatter (module
    docstring)."""
    def jp_read(h: Expr) -> Expr:
        return Index("Jp", (Var("jr"), h))

    def at(k: int, row: Expr, col: Expr) -> Index:
        return Index("J1", (Lit(k), row, col))

    m = _c(Fn.ADD, Var("oa"), Var("ra"))
    n = _c(Fn.ADD, Var("ob"), Var("rb"))

    def elem(a: tuple[Expr, Expr, Expr], b: tuple[Expr, Expr, Expr]
             ) -> tuple[Stmt, ...]:
        a_adds = tuple(AtomicAdd(at(k, m, n), Var(f"gA{k}")) for k in range(3))
        b_adds = tuple(AtomicAdd(at(k, n, m), Var(f"gB{k}")) for k in range(3))
        return (
            Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),
            *(Decl(f"g{s}{k}", Scalar.F64, init=Lit(0.0))
              for s in "AB" for k in range(3)),
            *lowstack_derivative(la, lb, a, b, jp_read, "sk", Fn.ADD),
            Note(ATOMIC_REASON),
            *a_adds,
            *((If(Var("offd"), b_adds),) if la == lb else b_adds),
        )

    out: list[Stmt] = []
    if la == lb:
        out += [
            Note(_DIAGONAL),
            Decl("offd", Scalar.I32, init=_c(
                Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))),
        ]
    out += [
        Note("per block row (a, b): gA_k = s SUM_t (dA_k E^{ab})_t Jp[row, "
             "t] (the Cartesian rule, 2 alpha E^{a+1_k,b} - a_k "
             "E^{a-1_k,b}), gB likewise, each rolled over the per-axis "
             "tables (e_fragment.rolled_e_sum); Hermite order "
             f"{la + lb + 1}; J1[k, m, n] += gA_k, J1[k, n, m] += gB_k"),
        lowstack_rows(la, lb, elem),
    ]
    return out


def build_jx_kernel(cls: Spec | tuple[int, int], *,
                    name: str | None = None) -> Kernel:
    """The ip1 bra half of the Coulomb derivative matrix, one bra pair
    class.

    `cls` is a `Target.JX` `Spec` (d1, the only order it has) or `(la,
    lb)`. See the module docstring.
    """
    la, lb = _class(cls)
    head = record_head(la + 1, lb + 1, interleaved=True, atoms=False)
    head.append(Decl("jr", Scalar.I32, init=Index("row", (Var(REC),))))
    per_rec = (*head, *_rows(la, lb))
    body: tuple[Stmt, ...] = (
        Note(f"ip1 bra half of the Coulomb derivative matrix, bra class "
             f"(la, lb) = ({la}, {lb}): J1[k, m, n] += SUM_t (dA_k E^mn)_t "
             "J_t, J1[k, n, m] += SUM_t (dB_k E^mn)_t J_t "
             "(felizjueves.hessian.kernel_jx)"),
        Note(ZEROED_J1),
        Note("launch: one thread per bra primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, Var(REC), Var("n_rec")), per_rec),
    )
    nr, nao, nrows = Var("n_rec"), Var("n_ao"), Var("n_rows")

    def f64(nm: str, shape: tuple[Expr, ...]) -> Param:
        return Param(nm, Scalar.F64, array=True, shape=shape)

    def i32(nm: str) -> Param:
        return Param(nm, Scalar.I32, array=True, shape=(nr,))

    params = (
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
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
        f64("Jp", (nrows, Lit(n_p1(la, lb)))),
        Param("J1", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(3), nao, nao)),
    )
    return Kernel(name or jx_name((la, lb)), params, body)
