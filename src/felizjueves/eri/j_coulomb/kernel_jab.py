"""The bra-transform-and-scatter kernel, `Target.JAB`, built as a `Kernel` (#154).

    build_jab_kernel(spec | (la, lb)) -> Kernel     one per bra pair class
    jab_name(spec | (la, lb))         -> str        its name, unbuilt
    jab_spec(la, lb)                  -> Spec       the class's key
    jab_launch_shape(n_rec, threads)  -> (grid, block)

Step 4 of the J chain (docs/implementation/j_engine.md, "The J chain as
device code", "Step 4"): per bra primitive-pair record `k` of one bra pair
class `(la, lb)` and per Cartesian pair `r = (i, j)`,

    v = SUM_{p in hermite_order(la+lb)} (s[k] E[r, p]) Jp[row[k], p]

added into the AO matrix by the scatter rule:

    off-diagonal (off_a != off_b):  J[off_a+i, off_b+j] += v;  J[off_b+j, off_a+i] += v
    diagonal     (off_a == off_b):  only i >= j;  J[off_a+i, off_a+j] += v,
                                    and for i != j also J[off_a+j, off_a+i] += v

THE PARAMETER BLOCK is the contract's, with two spellings forced by the IR:

    params  n_rec n_ao n_rows (I32)
            zeta_ab[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s[n_rec]   per record
            off_a[n_rec]  off_b[n_rec]  row[n_rec]  (I32)
            Jp[n_rows, n_p]                     the bra class's J_p buffer
            J[n_ao, n_ao]  (OUT, read and AtomicAdd-ed)

- `zeta_ab`, not the contract's `zeta`: the `E` block fragment binds a
  SCALAR named `zeta`, and `ir`'s hygiene (#149) refuses a host array of the
  same name -- a fragment is not a macro, nothing is renamed. The kernel
  reads `zeta_ab[rec]` into a local `zeta` and binds that.
- `n_rows` is the first extent of `Jp`: a `Param.shape` extent must be a
  scalar `Param`, and the contract writes `Jp[n_rows, n_p]` without saying
  where `n_rows` comes from. It is the number of rows of the bra class's
  buffer, which a whole-class launch shares across shell pairs.
- `J` is `Intent.OUT`: the IR's only writable global. It is READ by every
  `AtomicAdd` (`dst += value`), so its contents on entry matter -- ZEROED BY
  THE CALLER before a build's first `JAB` launch (`build_pq_kernels`'
  convention), carried here as a `Note`.

ONE THREAD PER RECORD, `rec = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by
`rec < n_rec` (`jab_launch_shape`). The mapping is not part of the numerical
contract; it is the simplest one, and in the interpreter (threads in order,
blocks in order) it runs the records IN ORDER, which is what makes the
interpreted kernel `==` to `ao.j.jab_from_host` (below).

`E` FROM THE FRAGMENT, INLINED. `e_fragment.block_fragment(la, lb)` writes the
undressed block into a LOCAL array `E[n_rows_cart, n_p]`, from `zeta`,
`1/2zeta` (a local `0.5 / zeta`, formed once, as `ao.hermite.e_axis` forms
`h`), and the host-supplied `X_PA = P - A`, `X_PB = P - B` per axis. Only the
entries the fragment writes are read: an entry with `t_d > a_d + b_d` on some
axis is exactly zero, and its term is OMITTED from `v`, not multiplied in --
the interpreter refuses a read of an unwritten local, so a stray read fails.
The dressing is the host's one scalar per record, `s` (`ao.density.EPair`); no
`(pi/zeta)^{3/2}` is formed here.

THE DIAGONAL. On a diagonal bra shell pair the host's assign-then-mirror
leaves the lower triangle mirrored (j_engine.md, "The diagonal rule"); the
kernel writes exactly that, so every element of `J` is added to once per
record. A diagonal pair exists only when `la == lb`; for `la != lb` the
diagonal branch is not built. "Off-diagonal" is tested as `off_a != off_b`,
spelled `(off_a < off_b) + (off_b < off_a)`, so it assumes no order between
the two offsets.

WHY ATOMIC, and why `==` in the interpreter. Several records of one shell
pair and both mirror positions add into the same elements, so every write is
an `AtomicAdd` with that reason as its `Note`. `v` is a left fold over `p`
in Hermite order (`((t_0 + t_1) + t_2) ...`, the host's `acc += ...` with
its `+0.0` start and its zero-`E` terms dropped -- both change at most the
sign of a zero), and the `+=` chain from a zeroed `J` with records in order
is the host's `jab[r] += acc` from `0.0` followed by its single add into the
zeroed `J`. So interpreted, the kernel is `==` to `ao.j.jab_from_host`. On a
card the atomics' order differs run to run; that tolerance is the device
test's, not this module's.

A builder: vendor-neutral IR. It imports neither `ao/` (the reference it is
checked against) nor the interpreter nor `reference/`.
"""

from __future__ import annotations

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
from felizjueves.mmd import e_fragment as ef
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "REC",
    "ZEROED_J",
    "build_jab_kernel",
    "jab_name",
    "jab_launch_shape",
    "jab_spec",
]

REC = "rec"  # the record index: one thread per bra primitive-pair record

ATOMIC_REASON = (
    "AtomicAdd: several primitive records of one bra shell pair, and both "
    "mirror positions, add into the same J elements (j_engine.md, 'Step 4', "
    "'Why atomic')")

ZEROED_J = (
    "PRECONDITION: J is zeroed by the caller before the first JAB launch of "
    "a build -- build_pq_kernels' launch convention; this kernel only adds")

_DIAGONAL = (
    "diagonal bra shell pair (off_a == off_b): the lower triangle i >= j "
    "only, each mirrored once -- what the host's assign-then-mirror leaves, "
    "so the diagonal is not double-counted (j_engine.md, 'The diagonal rule')")


def jab_spec(la: int, lb: int) -> Spec:
    """The bra pair class `(la, lb)` as a generation key, `Target.JAB`."""
    return Spec(Operator.COULOMB, 0, (la, lb), Target.JAB)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else jab_spec(*cls)
    if spec.target is not Target.JAB:
        raise ValueError(f"{spec.key()} is not a Target.JAB key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the JAB kernel takes no levers")
    la, lb = spec.am
    return la, lb


def jab_launch_shape(n_rec: int, threads: int = 32
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


def jab_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_jab_kernel(cls)` gives its kernel, without building
    it (#419)."""
    la, lb = _class(cls)
    return f"jab_{la}{lb}"


def build_jab_kernel(cls: Spec | tuple[int, int], *,
                     name: str | None = None) -> Kernel:
    """The bra-transform-and-scatter kernel of one bra pair class.

    `cls` is a `Target.JAB` `Spec` or `(la, lb)`. See the module docstring.
    """
    la, lb = _class(cls)
    frag = ef.block_fragment(la, lb)
    rows = ef.block_rows(la, lb)
    ts = hermite_targets(la + lb)
    n_rows_cart, n_p = len(rows), len(ts)
    nb = len(ef.cartesians(lb))
    rec = Var(REC)

    # --- the record's inputs, and E from the fragment ----------------------
    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA), ef.E_ARRAY: Var(ef.E_ARRAY)}
    head: list[Stmt] = [
        Decl(ef.ZETA, Scalar.F64, init=Index("zeta_ab", (rec,))),
    ]
    if ef.HALF_INV_ZETA in frag.binds:
        head.append(Decl(ef.HALF_INV_ZETA, Scalar.F64,
                         init=_c(Fn.DIV, Lit(0.5), Var(ef.ZETA))))
        binds[ef.HALF_INV_ZETA] = Var(ef.HALF_INV_ZETA)
    for d in range(3):
        binds[Factor.pa(d).name_in_kernel] = Index("XPA", (rec, Lit(d)))
        binds[Factor.pb(d).name_in_kernel] = Index("XPB", (rec, Lit(d)))
    head += [
        Decl("sk", Scalar.F64, init=Index("s", (rec,))),
        Decl("jr", Scalar.I32, init=Index("row", (rec,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (rec,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (rec,))),
        Note("the undressed E block of this record, from the fragment: "
             "rows cartesians(la) x cartesians(lb), columns hermite_order"),
        Decl(ef.E_ARRAY, Scalar.F64, (Lit(n_rows_cart), Lit(n_p))),
        Inline(frag, binds),
    ]

    # --- v per Cartesian pair: the bra transform ---------------------------
    transform: list[Stmt] = [
        Note("J_ab = SUM_p (s E^ab_p) J_p, [JEngine, v2, eqs. 20-22]: the bra "
             "back-transform; a structurally zero E term is omitted"),
    ]
    for r, (a, b) in enumerate(rows):
        terms = [
            _c(Fn.MUL, _c(Fn.MUL, Var("sk"), Index(ef.E_ARRAY, (Lit(r), Lit(c)))),
               Index("Jp", (Var("jr"), Lit(c))))
            for c, t in enumerate(ts)
            if not any(t[d] > a[d] + b[d] for d in range(3))
        ]
        v: Expr = terms[0]
        for term in terms[1:]:
            v = _c(Fn.ADD, v, term)
        transform.append(Decl(f"v{r}", Scalar.F64, init=v))

    # --- the scatter --------------------------------------------------------
    def at(row: Expr, col: Expr) -> Index:
        return Index("J", (row, col))

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

    per_rec = (*head, *transform,
               Note("scatter into J_mn and its mirror J_nm"), scatter)
    body: tuple[Stmt, ...] = (
        Note(f"bra transform and scatter, class (la, lb) = ({la}, {lb}): "
             "j_engine.md, 'The J chain as device code', step 4"),
        Note(ZEROED_J),
        Note("launch: one thread per bra primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, rec, Var("n_rec")), per_rec),
    )
    nr, nao = Var("n_rec"), Var("n_ao")
    params = (
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_rows", Scalar.I32),
        _f64("zeta_ab", (nr,)),
        _f64("XPA", (nr, Lit(3))),
        _f64("XPB", (nr, Lit(3))),
        _f64("s", (nr,)),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        _i32("row", (nr,)),
        _f64("Jp", (Var("n_rows"), Lit(n_p))),
        Param("J", Scalar.F64, array=True, intent=Intent.OUT, shape=(nao, nao)),
    )
    return Kernel(name or jab_name((la, lb)), params, body)
