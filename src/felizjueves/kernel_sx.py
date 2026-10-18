"""The cross-basis overlap kernel, `Target.SX`, built as a `Kernel`.

    build_sx_kernel(spec | (la, lb)) -> Kernel     one per ORDERED pair class
    sx_spec(la, lb)                  -> Spec       the class's key
    launch_shape(n_rec, threads)     -> (grid, block)

`S_{B1,B2}[m, n] = <m|n>`, `m` an AO of the ROW basis `B1` and `n` of the
COLUMN basis `B2`: the rectangular `n_a x n_b` overlap between two basis
sets (one molecule in two bases, or two molecules). The integral is
`Target.S`'s exactly -- the `t = 0` column of the dressed block, per
primitive-pair record `k` of a pair class `(la, lb)` and per Cartesian pair
`r = (i, j)`,

    v = s[k] * ((Ex_{a_x b_x 0} * Ey_{a_y b_y 0}) * Ez_{a_z b_z 0})

and the statements that form it are `kernel_s.record_values`', UNCHANGED.
What differs is the MATRIX, so only the scatter is this module's:

    S[off_a + i, off_b + j] += v        for every (i, j); nothing else

with `off_a` the shell's AO offset in `B1` and `off_b` in `B2`. There is no
mirror (`S[n, m]` is not an element of an `n_a x n_b` matrix), and no
diagonal rule (`off_a == off_b` means nothing across two bases: those are
offsets into different index sets). Every element is owned by exactly one
ordered shell pair `(i in B1, j in B2)`.

THE CLASS IS ORDERED. `(la, lb)` and `(lb, la)` are different kernels and
both are reached by any two bases with both shell types: `la` is always the
row basis's. `Target.S` gets its ordered classes because a basis keys a
pair by shell index; here there is no `i <= j` to begin with.

THE PARAMETER BLOCK:

    params  n_rec n_a n_b (I32)
            zeta_ab[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s[n_rec]   per record
            off_a[n_rec]  off_b[n_rec]  (I32)
            S[n_a, n_b]  (OUT, read and AtomicAdd-ed)

`kernel_s`'s, with `n_ao` split into the row and column counts. `S` is
ZEROED BY THE CALLER before the first launch of a build (a `Note`).
`AtomicAdd` because a shell pair's records are separate threads and all add
into its block.

WHY `==` IN THE INTERPRETER. Interpreted (threads in order, blocks in
order), the records of a shell pair add in `primitive_pairs` order from the
caller's `0.0`, which is `ao.overlap.overlap_block(sa, sb)`'s accumulation
exactly (`kernel_s`'s docstring has the two admitted departures; neither
changes a bit). With no triangle read, the whole matrix is `==` to
`ao.overlap.cross_overlap_matrix` -- not merely to a rearranged block, as
`S` is. On a card the atomics' order differs run to run.

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
from felizjueves.kernel_s import REC, launch_shape, record_values
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "ZEROED_S",
    "build_sx_kernel",
    "launch_shape",
    "sx_spec",
]

ATOMIC_REASON = (
    "AtomicAdd: the primitive records of one ordered shell pair (i in B1, "
    "j in B2) are separate threads and add into the same S elements; no "
    "mirror, no diagonal rule")

ZEROED_S = (
    "PRECONDITION: S is zeroed by the caller before the first SX launch of "
    "a build; this kernel only adds")


def sx_spec(la: int, lb: int) -> Spec:
    """The ordered pair class `(la in B1, lb in B2)` as a key, `Target.SX`."""
    return Spec(Operator.OVERLAP, 0, (la, lb), Target.SX)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else sx_spec(*cls)
    if spec.target is not Target.SX:
        raise ValueError(f"{spec.key()} is not a Target.SX key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the SX kernel takes no levers")
    la, lb = spec.am
    return la, lb


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def build_sx_kernel(cls: Spec | tuple[int, int], *,
                    name: str | None = None) -> Kernel:
    """The cross-basis overlap kernel of one ordered pair class.

    `cls` is a `Target.SX` `Spec` or `(la, lb)`. See the module docstring.
    """
    la, lb = _class(cls)
    nb = len(ef.cartesians(lb))
    rec = Var(REC)
    head, values = record_values(la, lb)

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for i in range(len(ef.cartesians(la))):
        for j in range(nb):
            scatter.append(AtomicAdd(Index("S", (plus("oa", i), plus("ob", j))),
                                     Var(f"v{i * nb + j}")))

    per_rec = (*head, *values,
               Note("scatter into S[m in B1, n in B2]: one add per element"),
               *scatter)
    body: tuple[Stmt, ...] = (
        Note(f"cross-basis overlap, ordered class (la, lb) = ({la}, {lb}): "
             "S_{B1,B2} as the t = 0 column of the dressed E block, la the "
             "row basis's shell"),
        Note(ZEROED_S),
        Note("launch: one thread per primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, rec, Var("n_rec")), per_rec),
    )
    nr, na, nbv = Var("n_rec"), Var("n_a"), Var("n_b")
    params = (
        Param("n_rec", Scalar.I32),
        Param("n_a", Scalar.I32),
        Param("n_b", Scalar.I32),
        _f64("zeta_ab", (nr,)),
        _f64("XPA", (nr, Lit(3))),
        _f64("XPB", (nr, Lit(3))),
        _f64("s", (nr,)),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        Param("S", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(na, nbv)),
    )
    return Kernel(name or f"sx_{la}{lb}", params, body)
