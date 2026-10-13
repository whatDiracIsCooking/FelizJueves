"""The kinetic-energy kernel, `Target.T`, built as a `Kernel` (#238).

    build_t_kernel(spec | (la, lb)) -> Kernel     one per pair class
    t_spec(la, lb)                  -> Spec       the class's key
    launch_shape(n_rec, threads)    -> (grid, block)

`T_ab = <a| -1/2 grad^2 |b>` on the same axis bodies as `S`, read two
angular momenta higher (`ao/kinetic.py`'s module docstring). Per
primitive-pair record `k` of one pair class `(la, lb)` and per Cartesian
pair `r = (a, b)`,

    v = s[k] * KE[r],   KE[r] = K_x S_y S_z + S_x K_y S_z + S_x S_y K_z
    K_ij = -2 eb^2 E^{i,j+2}_0 + eb (2j+1) E^{ij}_0 - 1/2 j(j-1) E^{i,j-2}_0

with `KE` written by `e_fragment.kinetic_fragment(la, lb)` from the axis
tables at `lb + 2`, and `v` added into the AO matrix by `kernel_jab`'s
scatter rule, UNCHANGED (as in `kernel_s.py`):

    off-diagonal (off_a != off_b):  T[off_a+i, off_b+j] += v;  T[off_b+j, off_a+i] += v
    diagonal     (off_a == off_b):  only i >= j;  T[off_a+i, off_a+j] += v,
                                    and for i != j also T[off_a+j, off_a+i] += v

THE PARAMETER BLOCK: `kernel_s`'s, plus the ket exponent per record.

    params  n_rec n_ao (I32)
            zeta_ab[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s[n_rec]
            eb_ket[n_rec]                                        per record
            off_a[n_rec]  off_b[n_rec]  (I32)
            T[n_ao, n_ao]  (OUT, read and AtomicAdd-ed)

- `eb_ket` is the KET primitive's exponent `beta`. The shift is the one
  place in this tree a single primitive's exponent enters a fragment, so it
  is a parameter of its own, read into the fragment's scalar bind `eb` --
  never reconstructed from `zeta` (which would need `alpha` too, and
  round). `eb_ket`, not `eb`, for the hygiene reason `zeta_ab` is not
  `zeta`: `ir` refuses a host array named like a fragment's scalar bind.
- `T` is `Intent.OUT`, READ by every `AtomicAdd`: ZEROED BY THE CALLER
  before the first `T` launch of a build, carried as a `Note`.

NO PREFACTOR IS FORMED HERE, and no `K_AB` is reinstated: `s` is the host's
one scalar per record, `c_a c_b (pi/zeta)^{3/2}` (`ao.pairs.dress`), and the
fragment's per-axis seed `exp(zeta X_PA X_PB)` already carries `K_AB` in
every term of the bracket. One `(pi/zeta)^{3/2}` is right for all three
terms, since each carries exactly one factor per axis (`ao/kinetic.py`).

ONE THREAD PER RECORD, `rec = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by
`rec < n_rec`. Interpreted (threads in order, blocks in order) it runs the
records IN ORDER.

WHY `==` IN THE INTERPRETER. `ao.kinetic.kinetic_block` forms, per record,
`S_ij = e_axis(i, j)[0]` for `j <= lb + 2`, `K_ij` by `kinetic_axis`
(`((-2.0 beta) beta) S_{i,j+2} + (beta (2j+1)) S_ij`, then minus
`(1/2 j(j-1)) S_{i,j-2}` only for `j >= 2`), the bracket
`((K_x S_y) S_z + (S_x K_y) S_z) + (S_x S_y) K_z`, dresses it as
`scale * bracket`, and adds it into an accumulator from `0.0`, records in
`primitive_pairs` order. The fragment performs exactly those operations in
that association (its axis entries are `ao.hermite.e_axis`'s, raised in
its order -- test/test_e_fragment.py), `v = sk * KE[r]` is `scale * x`, and
the kernel's `+=` chain starts from the caller's zeroed `T` -- the same
`0.0 + x` as the host's. So the reference is `kinetic_block` placed by
`kernel_jab`'s rule, held bitwise (test/test_kernel_t.py); against
`kinetic_matrix`, which reads a diagonal pair's UPPER triangle, a stated
bound (`kernel_s.py` says why).

On a card the atomics' order differs run to run; that tolerance is a device
test's, not this module's.

A builder: vendor-neutral IR. It imports neither `ao/` (the reference it is
checked against) nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves import e_fragment as ef
from felizjueves.ir import (
    AtomicAdd,
    Call,
    Decl,
    Expr,
    Factor,
    Fn,
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
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "REC",
    "ZEROED_T",
    "build_t_kernel",
    "launch_shape",
    "t_spec",
]

REC = "rec"  # the record index: one thread per primitive-pair record

ATOMIC_REASON = (
    "AtomicAdd: several primitive records of one shell pair, and both mirror "
    "positions, add into the same T elements (kernel_jab's scatter rule)")

ZEROED_T = (
    "PRECONDITION: T is zeroed by the caller before the first T launch of a "
    "build; this kernel only adds")

_DIAGONAL = (
    "diagonal shell pair (off_a == off_b): the lower triangle i >= j only, "
    "each mirrored once -- kernel_jab's diagonal rule, so the diagonal is not "
    "double-counted")


def t_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` as a generation key, `Target.T`."""
    return Spec(Operator.KINETIC, 0, (la, lb), Target.T)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else t_spec(*cls)
    if spec.target is not Target.T:
        raise ValueError(f"{spec.key()} is not a Target.T key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the T kernel takes no levers")
    la, lb = spec.am
    return la, lb


def launch_shape(n_rec: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per record, `ceil(n_rec/threads)`
    blocks. No `WarpReduce`, so a partial last block is legal."""
    if type(n_rec) is not int or n_rec < 1:
        raise ValueError(f"n_rec = {n_rec!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_rec // threads), (threads, 1)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def build_t_kernel(cls: Spec | tuple[int, int], *,
                   name: str | None = None) -> Kernel:
    """The kinetic-energy kernel of one pair class.

    `cls` is a `Target.T` `Spec` or `(la, lb)`. See the module docstring.
    """
    la, lb = _class(cls)
    frag = ef.kinetic_fragment(la, lb)
    n_rows_cart = len(ef.block_rows(la, lb))
    nb = len(ef.cartesians(lb))
    rec = Var(REC)

    # --- the record's inputs, and the bracket KE from the fragment ----------
    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA), ef.EB: Var(ef.EB),
                              ef.KE_ARRAY: Var(ef.KE_ARRAY)}
    head: list[Stmt] = [
        Decl(ef.ZETA, Scalar.F64, init=Index("zeta_ab", (rec,))),
    ]
    if ef.HALF_INV_ZETA in frag.binds:
        head.append(Decl(ef.HALF_INV_ZETA, Scalar.F64,
                         init=_c(Fn.DIV, Lit(0.5), Var(ef.ZETA))))
        binds[ef.HALF_INV_ZETA] = Var(ef.HALF_INV_ZETA)
    for d in range(3):
        for f, arr in ((Factor.pa(d), "XPA"), (Factor.pb(d), "XPB")):
            if f.name_in_kernel in frag.binds:
                binds[f.name_in_kernel] = Index(arr, (rec, Lit(d)))
    head += [
        Decl(ef.EB, Scalar.F64, init=Index("eb_ket", (rec,))),
        Decl("sk", Scalar.F64, init=Index("s", (rec,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (rec,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (rec,))),
        Note("the undressed kinetic bracket K_x S_y S_z + S_x K_y S_z + "
             "S_x S_y K_z of this record, from the fragment (axis tables at "
             "lb + 2): rows cartesians(la) x cartesians(lb)"),
        Decl(ef.KE_ARRAY, Scalar.F64, (Lit(n_rows_cart),)),
        Inline(frag, binds),
    ]

    # --- v per Cartesian pair: the host's one scalar times the bracket -------
    values: list[Stmt] = [
        Note("T_ab = s KE_ab: s = c_a c_b (pi/zeta)^{3/2} is the host's "
             "(ao.pairs.dress); no prefactor and no K_AB is formed here"),
    ]
    for r in range(n_rows_cart):
        values.append(Decl(f"v{r}", Scalar.F64, init=_c(
            Fn.MUL, Var("sk"), Index(ef.KE_ARRAY, (Lit(r),)))))

    # --- the scatter: kernel_jab's rule ------------------------------------
    def at(row: Expr, col: Expr) -> Index:
        return Index("T", (row, col))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    off_diag: list[Stmt] = [Note(ATOMIC_REASON)]
    for i in range(len(ef.cartesians(la))):
        for j in range(nb):
            vr = Var(f"v{i * nb + j}")
            off_diag += [AtomicAdd(at(plus("oa", i), plus("ob", j)), vr),
                         AtomicAdd(at(plus("ob", j), plus("oa", i)), vr)]
    scatter: Stmt
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    if la == lb:
        diag: list[Stmt] = [Note(_DIAGONAL), Note(ATOMIC_REASON)]
        for i in range(nb):
            for j in range(i + 1):
                vr = Var(f"v{i * nb + j}")
                diag.append(AtomicAdd(at(plus("oa", i), plus("oa", j)), vr))
                if i != j:
                    diag.append(AtomicAdd(at(plus("oa", j), plus("oa", i)), vr))
        scatter = If(differ, tuple(off_diag), tuple(diag))
    else:
        scatter = If(differ, tuple(off_diag))

    per_rec = (*head, *values,
               Note("scatter into T_mn and its mirror T_nm"), scatter)
    body: tuple[Stmt, ...] = (
        Note(f"kinetic energy, class (la, lb) = ({la}, {lb}): T_ab by the "
             "fixed ket shift on the E axis tables read at lb + 2 "
             "(ao/kinetic.py)"),
        Note(ZEROED_T),
        Note("launch: one thread per primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, rec, Var("n_rec")), per_rec),
    )
    nr, nao = Var("n_rec"), Var("n_ao")
    params = (
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        _f64("zeta_ab", (nr,)),
        _f64("XPA", (nr, Lit(3))),
        _f64("XPB", (nr, Lit(3))),
        _f64("s", (nr,)),
        _f64("eb_ket", (nr,)),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        Param("T", Scalar.F64, array=True, intent=Intent.OUT, shape=(nao, nao)),
    )
    return Kernel(name or f"t_{la}{lb}", params, body)
