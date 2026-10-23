"""The overlap kernel, `Target.S`, built as a `Kernel` (#237).

    build_s_kernel(spec | (la, lb)) -> Kernel     one per pair class
    s_name(spec | (la, lb))         -> str        its name, unbuilt
    s_spec(la, lb)                  -> Spec       the class's key
    launch_shape(n_rec, threads)    -> (grid, block)

The first one-electron kernel, and the cheapest consumer of the `E`
fragment. `S_ab = <a|b>` is the `t = 0` column of the dressed block
(`ao/overlap.py`'s module docstring): per primitive-pair record `k` of one
pair class `(la, lb)` and per Cartesian pair `r = (i, j)`,

    v = s[k] * ((Ex_{a_x b_x 0} * Ey_{a_y b_y 0}) * Ez_{a_z b_z 0})

added into the AO matrix by `kernel_jab`'s scatter rule, UNCHANGED:

    off-diagonal (off_a != off_b):  S[off_a+i, off_b+j] += v;  S[off_b+j, off_a+i] += v
    diagonal     (off_a == off_b):  only i >= j;  S[off_a+i, off_a+j] += v,
                                    and for i != j also S[off_a+j, off_a+i] += v

`S` is symmetric and that rule is already proved (test/test_kernel_jab.py);
reusing it is what lets the interpreted kernel be bitwise `==` to the host.

THE PARAMETER BLOCK:

    params  n_rec n_ao (I32)
            zeta_ab  pa_x pa_y pa_z  pb_x pb_y pb_z  s  [n_rec]   per record
            off_a  off_b  [n_rec]  (I32)
            S[n_ao, n_ao]  (OUT, read and AtomicAdd-ed)

- Every per-record array is flat, one pointer per scalar: `P - A` is the
  three arrays `pa_x pa_y pa_z`, not one `XPA[n_rec, 3]`, so with one
  thread per record a warp's lanes read consecutive doubles on every load.
  They cross the call as plain positional parameters (`pa_name`,
  `pb_name`): `kernel_sx` imports the spelling, `kernel_t` and `kernel_v`
  restate it.

- `zeta_ab`, not `zeta`, for the hygiene reason `kernel_jab` documents: the
  fragment binds a SCALAR named `zeta`, and `ir` refuses a host array of the
  same name. The kernel reads `zeta_ab[rec]` into a local `zeta`.
- `S` is `Intent.OUT`, READ by every `AtomicAdd`: ZEROED BY THE CALLER
  before the first `S` launch of a build, carried as a `Note`.

NO PREFACTOR IS FORMED HERE. `s` is the host's one scalar per record,
`c_a c_b (pi/zeta)^{3/2}` as `ao.pairs.dress` forms it (`ao.density.epair`'s
`s`); the kernel neither raises to a power nor names `pi`. And no `K_AB`
is reinstated: the fragment's seed `exp(zeta X_PA X_PB)` per axis already
carries it (milestone 11's design pin). `1/2zeta` is a local `0.5 / zeta`,
formed once, as `ao.hermite.e_axis` forms it.

`E` FROM THE FRAGMENT, INLINED. `e_fragment.overlap_fragment(la, lb)` writes
the undressed `t = 0` column into a LOCAL array `E0[n_rows_cart]` -- not
the whole block: the other columns would be dead code here.

ONE THREAD PER RECORD, `rec = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by
`rec < n_rec` (`launch_shape`). Interpreted (threads in order, blocks in
order) it runs the records IN ORDER.

WHY `==` IN THE INTERPRETER. The host (`ao.overlap.overlap_block`) forms
`E^{ab}_0 = ((1.0 E_x) E_y) E_z` (`ao.hermite.e_block`), multiplies by
`scale` (`ao.pairs.dress`, the same scalar the host passes as `s`), and adds
it into an accumulator that starts at `0.0`, record by record in
`primitive_pairs` order. The kernel performs the same multiplications in the
same association, and its `+=` chain starts from the caller's zeroed `S`.
Two departures, neither of which changes a bit:

  (1) a structurally zero `E` entry is ABSENT, not computed: the host's
      `e_block` writes `0.0` where `t_d > a_d + b_d`. At `t = 0` there is no
      such entry (`0 <= a_d + b_d` always), so no product `S` reads is
      affected; the axis tables declare exactly the entries the host's
      `e_axis` returns (`t <= i + j`).
  (2) the host's `0.0` accumulator start, and its leading `1.0 *`: `0.0 + x`
      is `x` for every `x` except `-0.0` (which becomes `+0.0`), and the
      kernel's first add into a zeroed `S` is the same `0.0 + x`; `1.0 * x`
      is `x` exactly. So both change at most the sign of a zero, and here
      not even that, since the operations are the same.

The diagonal rule keeps the LOWER triangle of a diagonal shell pair's block
(`kernel_jab`'s, `ao.j.jab_from_host`'s); `ao.one_electron.matrix` reads the
UPPER. For a symmetric operator the two agree to rounding, not necessarily
bitwise -- `E^{ij}` and `E^{ji}` are raised along different paths -- so the
bitwise witness is `overlap_block` with `kernel_jab`'s rule applied, and
`overlap_matrix` is held to a stated tolerance (test/test_kernel_s.py).

On a card the atomics' order differs run to run; that tolerance is a device
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
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "AXES",
    "REC",
    "ZEROED_S",
    "build_s_kernel",
    "s_name",
    "launch_shape",
    "pa_name",
    "pb_name",
    "record_values",
    "s_spec",
]

REC = "rec"  # the record index: one thread per primitive-pair record

AXES = "xyz"  # the per-axis suffix of every record vector's arrays


def pa_name(d: int) -> str:
    """The parameter holding axis `d` of `P - A`, one double per record."""
    return f"pa_{AXES[d]}"


def pb_name(d: int) -> str:
    """The parameter holding axis `d` of `P - B`, one double per record."""
    return f"pb_{AXES[d]}"


ATOMIC_REASON = (
    "AtomicAdd: several primitive records of one shell pair, and both mirror "
    "positions, add into the same S elements (kernel_jab's scatter rule)")

ZEROED_S = (
    "PRECONDITION: S is zeroed by the caller before the first S launch of a "
    "build; this kernel only adds")

_DIAGONAL = (
    "diagonal shell pair (off_a == off_b): the lower triangle i >= j only, "
    "each mirrored once -- kernel_jab's diagonal rule, so the diagonal is not "
    "double-counted")


def s_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` as a generation key, `Target.S`."""
    return Spec(Operator.OVERLAP, 0, (la, lb), Target.S)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else s_spec(*cls)
    if spec.target is not Target.S:
        raise ValueError(f"{spec.key()} is not a Target.S key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the S kernel takes no levers")
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


def record_values(la: int, lb: int) -> tuple[list[Stmt], list[Stmt]]:
    """The per-record statements of class `(la, lb)` up to the scatter:
    `head` reads record `rec`'s inputs into locals (`zeta`, `sk`, `oa`,
    `ob`) and inlines `E^{ab}_0`; `values` forms `v{r} = sk * E0[r]` for
    every Cartesian pair `r` of `block_rows(la, lb)`. Shared, unchanged, by
    `kernel_sx` (the cross-basis overlap), whose scatter alone differs."""
    frag = ef.overlap_fragment(la, lb)
    n_rows_cart = len(ef.block_rows(la, lb))
    rec = Var(REC)

    # --- the record's inputs, and E^{ab}_0 from the fragment --------------
    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA),
                              ef.E0_ARRAY: Var(ef.E0_ARRAY)}
    head: list[Stmt] = [
        Decl(ef.ZETA, Scalar.F64, init=Index("zeta_ab", (rec,))),
    ]
    if ef.HALF_INV_ZETA in frag.binds:
        head.append(Decl(ef.HALF_INV_ZETA, Scalar.F64,
                         init=_c(Fn.DIV, Lit(0.5), Var(ef.ZETA))))
        binds[ef.HALF_INV_ZETA] = Var(ef.HALF_INV_ZETA)
    for d in range(3):
        binds[Factor.pa(d).name_in_kernel] = Index(pa_name(d), (rec,))
        binds[Factor.pb(d).name_in_kernel] = Index(pb_name(d), (rec,))
    head += [
        Decl("sk", Scalar.F64, init=Index("s_ab", (rec,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (rec,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (rec,))),
        Note("the undressed t = 0 column E^{ab}_0 of this record, from the "
             "fragment: rows cartesians(la) x cartesians(lb)"),
        Decl(ef.E0_ARRAY, Scalar.F64, (Lit(n_rows_cart),)),
        Inline(frag, binds),
    ]

    # --- v per Cartesian pair: the host's one scalar times E^{ab}_0 ---------
    values: list[Stmt] = [
        Note("S_ab = s E^{ab}_0: s = c_a c_b (pi/zeta)^{3/2} is the host's "
             "(ao.pairs.dress); no prefactor and no K_AB is formed here"),
    ]
    for r in range(n_rows_cart):
        values.append(Decl(f"v{r}", Scalar.F64, init=_c(
            Fn.MUL, Var("sk"), Index(ef.E0_ARRAY, (Lit(r),)))))
    return head, values


def s_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_s_kernel(cls)` gives its kernel, without building
    it (#419)."""
    la, lb = _class(cls)
    return f"s_{la}{lb}"


def build_s_kernel(cls: Spec | tuple[int, int], *,
                   name: str | None = None) -> Kernel:
    """The overlap kernel of one pair class.

    `cls` is a `Target.S` `Spec` or `(la, lb)`. See the module docstring.
    """
    la, lb = _class(cls)
    nb = len(ef.cartesians(lb))
    rec = Var(REC)
    head, values = record_values(la, lb)

    # --- the scatter: kernel_jab's rule ------------------------------------
    def at(row: Expr, col: Expr) -> Index:
        return Index("S", (row, col))

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
               Note("scatter into S_mn and its mirror S_nm"), scatter)
    body: tuple[Stmt, ...] = (
        Note(f"overlap, class (la, lb) = ({la}, {lb}): S_ab as the t = 0 "
             "column of the dressed E block (ao/overlap.py)"),
        Note(ZEROED_S),
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
        *(_f64(pa_name(d), (nr,)) for d in range(3)),
        *(_f64(pb_name(d), (nr,)) for d in range(3)),
        _f64("s_ab", (nr,)),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        Param("S", Scalar.F64, array=True, intent=Intent.OUT, shape=(nao, nao)),
    )
    return Kernel(name or s_name((la, lb)), params, body)
