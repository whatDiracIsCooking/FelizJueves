"""The bra-transform-and-scatter kernel, `Target.JAB`, built as a `Kernel` (#154).

    build_jab_kernel(spec | (la, lb), lowstack=None) -> Kernel  one per class
    jab_lowstack(spec | (la, lb))     -> bool       the form's selection rule
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
interpreted kernel `==` to `eri.j_coulomb.j.jab_from_host` (below).

`E` FROM THE FRAGMENT, INLINED. `e_fragment.block_fragment(la, lb)` writes the
undressed block into a LOCAL array `E[n_rows_cart, n_p]`, from `zeta`,
`1/2zeta` (a local `0.5 / zeta`, formed once, as `mmd.hermite.e_axis` forms
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
zeroed `J`. So interpreted, the kernel is `==` to `eri.j_coulomb.j.jab_from_host`. On a
card the atomics' order differs run to run; that tolerance is the device
test's, not this module's.

THE LOW-STACK FORM (#586, stage 2), the default where it is needed: `l_1 +
l_2 >= LOWSTACK_MIN_PAIR_L = 6` -- (ff), and past f (gd), (gf), ... The
block above is `E[100][84]` per thread at (ff), 67 KB: on sm_86 a 71 KB
stack frame (`jab_33`), which CUDA reserves for every resident thread (~7 GB
of a 10 GB RTX 3080). `build_jab_kernel(cls)` therefore emits, whenever
`jab_lowstack(cls)` (`kernel_dq`'s rule on the bra; the two constants are
pinned equal by a test), a kernel with the SAME name, parameters, launch and
output contract and no `E` block:

    tables = e_fragment.axis_tables_fragment(la, lb)  Ex Ey Ez, local,
                                      3 x (la+1)(lb+1)(la+lb+1) doubles
    for each Cartesian pair (a, b)    e_fragment.cartesian_loops, as loops;
                                      on a diagonal pair only i >= j
        v = 0
        for p in hermite_order, over  |t| ascending, t_x then t_y
            the row's reachable t     descending, each loop clipped to
                                      t_d <= a_d + b_d
            v += (s E^ab_t) Jp[row, hermite_index(t)]
                                      E = (Ex Ey) Ez, formed here
        scatter v: the rule above, its two AtomicAdds, in place

THE SAME BITS. `E` is `block_fragment`'s entry, `(E_x E_y) E_z` from the
same recurrence, every term is the unrolled form's `(s E) Jp` and `v` is the
same left fold over the same reachable `p` in `hermite_order` (from `+0.0`,
which changes only the sign of a zero), and the `AtomicAdd`s come in the
same order (`(i, j)` row-major, the mirror after its element). So
interpreted, the low-stack form is `==` to the unrolled one and to the host.
`e_fragment.rolled_e_sum` is NOT used: it sums `t` in another order with a
factored product, round-off, where this kernel's contract is bitwise. Below
L = 6 -- through d, and the mixed f classes (sf), (pf), (df) -- the default
is the unrolled kernel, its text unchanged byte for byte (those units
already fit: 0 to 936 B on sm_86, no spill, and the low-stack form is slower
there, up to 3.6x for JAB at (pf)); `lowstack=True` / `False` forces either
form at any class. At (ff) the frame is the three tables, 336 doubles.

A builder: vendor-neutral IR. It imports neither `ao/` (the reference it is
checked against) nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from collections.abc import Callable

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
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "LOWSTACK_MIN_PAIR_L",
    "REC",
    "ZEROED_J",
    "build_jab_kernel",
    "jab_lowstack",
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


LOWSTACK_MIN_PAIR_L = 6  # the low-stack form at l1 + l2 >= 6 (#586), = kernel_dq's


def jab_lowstack(cls: Spec | tuple[int, int]) -> bool:
    """THE SELECTION RULE (#586): True when the class's pair angular
    momentum `l_1 + l_2 >= LOWSTACK_MIN_PAIR_L` -- (ff), and past f (gd),
    (gf), ... -- where the unrolled form's per-thread `E` block,
    `n_rows x n_t` doubles, outgrows the stack: on sm_86 the unrolled
    (df) frame (L = 5, 60 x 56) is 728 B for DQ and 936 B for JAB, no
    spill, and (ff)'s (L = 6, 100 x 84) is about 72 KB. A class's block
    size is a function of L more than of its split (gd's 90 x 84 sits
    beside (ff), gp's 45 x 56 below (df)), so the cut is on L; there
    `build_jab_kernel` emits the low-stack form by default. Below it the
    unrolled form stays the default and its text is unchanged: it fits,
    and it is faster (#586's timing)."""
    return sum(_class(cls)) >= LOWSTACK_MIN_PAIR_L


def build_jab_kernel(cls: Spec | tuple[int, int], *,
                     name: str | None = None,
                     lowstack: bool | None = None) -> Kernel:
    """The bra-transform-and-scatter kernel of one bra pair class.

    `cls` is a `Target.JAB` `Spec` or `(la, lb)`. See the module docstring.
    `lowstack` picks the form: `None` (the default) is `jab_lowstack(cls)`,
    so below L = 6 the unrolled kernel and from L = 6 the low-stack one (THE
    LOW-STACK FORM); `True` / `False` force either, at any class. Both
    have the same name, parameters, launch and output contract.
    """
    la, lb = _class(cls)
    if lowstack is None:
        lowstack = jab_lowstack((la, lb))
    if lowstack:
        return _build_lowstack(la, lb, name or jab_name((la, lb)))
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
    return Kernel(name or jab_name((la, lb)), _params(n_p), body)


def _params(n_p: int) -> tuple[Param, ...]:
    """The parameter block (module docstring), one for both forms."""
    nr, nao = Var("n_rec"), Var("n_ao")
    return (
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


# ---------------------------------------------------------------------------
# The low-stack form (#586, stage 2)
# ---------------------------------------------------------------------------


LOWSTACK_NOTE = (
    "LOW-STACK (#586): per-axis E tables (3 x (la+1)(lb+1)(la+lb+1) "
    "doubles) in place of the E^{ab}_t block; the rows and the Hermite "
    "index as loops, each E entry formed (E_x E_y) E_z where it is used, "
    "the unrolled form's product, so the arithmetic is the same")


def _position(tag: str) -> Expr:
    """The position in `cartesians(l)` of the Cartesian the
    `e_fragment.cartesian_loops` nest tagged `tag` has open:
    `u (u+1) / 2 + w`, an I32 expression."""
    u, w = Var(f"{tag}_u"), Var(f"{tag}_w")
    return _c(Fn.ADD, _c(Fn.DIV, _c(Fn.MUL, u, _c(Fn.ADD, u, Lit(1))),
                         Lit(2)), w)


def _clip_low(name: str, value: Expr) -> tuple[Stmt, ...]:
    """`name = max(value, 0)` as an I32 local: `Fn` has no `max`."""
    return (Decl(name, Scalar.I32, init=value),
            If(_c(Fn.LT, Var(name), Lit(0)), (Assign(Var(name), Lit(0)),)))


def _clip_high(name: str, value: Expr, cap: Expr) -> tuple[Stmt, ...]:
    """`name = min(value, cap)` as an I32 local."""
    return (Decl(name, Scalar.I32, init=value),
            If(_c(Fn.LT, cap, Var(name)), (Assign(Var(name), cap),)))


def hermite_order_sum(a: tuple[Expr, Expr, Expr], b: tuple[Expr, Expr, Expr],
                      term: Callable[[tuple[Expr, Expr, Expr]], Expr],
                      dst: str) -> tuple[Stmt, ...]:
    """`dst = SUM_t term(t)` over the reachable `t` of row `(a, b)`
    (`t_d <= a_d + b_d`), as a left fold from `+0.0` IN `hermite_order`:
    `|t| = n` ascending, then `t_x` descending (`u = n - t_x` ascending),
    then `t_y` descending (`w = t_z` ascending), each loop clipped to the
    reachable range so no iteration is skipped by a branch. `term` gets
    `t` as three I32 expressions. Declares the F64 `dst` and I32 locals
    `n`, `u`, `w`, `s{x,y,z}`, `u0`, `w0`, `w1`, `t{x,y,z}`."""
    sx, sy, sz = (Var(f"s{c}") for c in "xyz")
    n, u, w = Var("n"), Var("u"), Var("w")
    tx, ty, tz = Var("tx"), Var("ty"), Var("tz")
    w_loop = For("w", Var("w0"), _c(Fn.ADD, Var("w1"), Lit(1)), Lit(1), (
        Decl("ty", Scalar.I32, init=_c(Fn.SUB, u, w)),
        Decl("tz", Scalar.I32, init=w),
        Assign(Var(dst), _c(Fn.ADD, Var(dst), term((tx, ty, tz)))),
    ))
    u_loop = For("u", Var("u0"), _c(Fn.ADD, n, Lit(1)), Lit(1), (
        Decl("tx", Scalar.I32, init=_c(Fn.SUB, n, u)),
        *_clip_low("w0", _c(Fn.SUB, u, sy)),
        *_clip_high("w1", u, sz),
        w_loop,
    ))
    n_loop = For("n", Lit(0), _c(Fn.ADD, _c(Fn.ADD, _c(Fn.ADD, sx, sy), sz),
                                 Lit(1)), Lit(1), (
        *_clip_low("u0", _c(Fn.SUB, n, sx)),
        u_loop,
    ))
    return (
        *(Decl(f"s{c}", Scalar.I32, init=_c(Fn.ADD, a[d], b[d]))
          for d, c in enumerate("xyz")),
        Decl(dst, Scalar.F64, init=Lit(0.0)),
        n_loop,
    )


def _build_lowstack(la: int, lb: int, name: str) -> Kernel:
    """`build_jab_kernel`'s kernel with no `E` block (module docstring, THE
    LOW-STACK FORM): the same parameters, launch and scatter, the same bits."""
    L = la + lb
    n_p = len(hermite_targets(L))
    frag = ef.axis_tables_fragment(la, lb)
    rec = Var(REC)

    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA),
                              **{n: Var(n) for n in ef.AXIS_ARRAYS}}
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
    extent = Lit(ef.axis_table_extent(la, lb))
    head += [
        Decl("sk", Scalar.F64, init=Index("s", (rec,))),
        Decl("jr", Scalar.I32, init=Index("row", (rec,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (rec,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (rec,))),
        Note(LOWSTACK_NOTE),
        Note("the record's per-axis E^{ij}_t, one local table per axis "
             "(e_fragment.axis_tables_fragment), undressed; no E^{ab}_t "
             "block is formed"),
        *(Decl(n, Scalar.F64, (extent,)) for n in ef.AXIS_ARRAYS),
        Inline(frag, binds),
        Note("off-diagonal (off_a != off_b), as an I32: the scatter rule's "
             "test, (off_a < off_b) + (off_b < off_a)"),
        Decl("offd", Scalar.I32, init=_c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                                         _c(Fn.LT, Var("ob"), Var("oa")))),
    ]

    def at(row: Expr, col: Expr) -> Index:
        return Index("J", (row, col))

    i, j = Var("i"), Var("j")

    def row(a: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def cell(b: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            base = [f"e{c}0" for c in "xyz"]

            def term(t: tuple[Expr, Expr, Expr]) -> Expr:
                def e(d: int) -> Index:
                    return Index(ef.AXIS_ARRAYS[d],
                                 (_c(Fn.ADD, Var(base[d]), t[d]),))
                ee = _c(Fn.MUL, _c(Fn.MUL, e(0), e(1)), e(2))
                return _c(Fn.MUL, _c(Fn.MUL, Var("sk"), ee),
                          Index("Jp", (Var("jr"), ef.hermite_index(t))))

            first = AtomicAdd(at(_c(Fn.ADD, Var("oa"), i),
                                 _c(Fn.ADD, Var("ob"), j)), Var("v"))
            mirror = AtomicAdd(at(_c(Fn.ADD, Var("ob"), j),
                                  _c(Fn.ADD, Var("oa"), i)), Var("v"))
            work: tuple[Stmt, ...] = (
                *(Decl(base[d], Scalar.I32,
                       init=ef.axis_table_index(a[d], b[d], 0, lb, L))
                  for d in range(3)),
                *hermite_order_sum(a, b, term, "v"),
                Note(ATOMIC_REASON),
            )
            pos = (Decl("i", Scalar.I32, init=_position("a")),
                   Decl("j", Scalar.I32, init=_position("b")))
            if la != lb:  # no diagonal pair: every element and its mirror
                return (*pos, *work, first, mirror)
            return (*pos, If(
                _c(Fn.ADD, Var("offd"), _c(Fn.LT, j, _c(Fn.ADD, i, Lit(1)))),
                (*work, first,
                 If(_c(Fn.ADD, Var("offd"), _c(Fn.LT, j, i)), (mirror,)))),)
        return (ef.cartesian_loops(lb, ("bx", "by", "bz"), cell, tag="b"),)

    per_rec = (
        *head,
        Note("J_ab = SUM_p (s E^ab_p) J_p, [JEngine, v2, eqs. 20-22], per "
             "Cartesian pair, rolled: p over the row's reachable t in "
             "hermite_order, a left fold, then scattered into J_mn and its "
             "mirror J_nm in place"),
        *((Note(_DIAGONAL + "; off-diagonal, every (i, j) and its mirror"),)
          if la == lb else ()),
    )
    rows = ef.cartesian_loops(la, ("ax", "ay", "az"), row, tag="a")
    # la != lb: the unrolled form's If(differ, off_diag) with no else, kept
    # (a record with off_a == off_b writes nothing there, so here too)
    per_rec = (*per_rec, rows if la == lb else If(Var("offd"), (rows,)))
    body: tuple[Stmt, ...] = (
        Note(f"bra transform and scatter, class (la, lb) = ({la}, {lb}), "
             "LOW-STACK (#586): j_engine.md, 'The J chain as device code', "
             "step 4"),
        Note(ZEROED_J),
        Note("launch: one thread per bra primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, rec, Var("n_rec")), per_rec),
    )
    return Kernel(name, _params(n_p), body)
