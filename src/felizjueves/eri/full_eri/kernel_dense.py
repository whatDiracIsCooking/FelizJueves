"""The dense-ERI kernel, `Target.ERI_DENSE`: a row block of `M_J` or `M_K`.

    build_dense_kernel(spec | (la, lb, lc, ld), layout, lowstack=None)
                                                        -> Kernel  per class
    dense_spec(la, lb, lc, ld)                          -> Spec    the key
    launch_shape(n_task, threads, rows=1)               -> (grid, block)

THE MATRICES (`full_eri.dense`): with AO pairs unpacked, `(m, n) -> m n_ao +
n`, the J layout `M_J[(m,n), (l,s)] = (mn|ls)` and the K layout
`M_K[(m,n), (l,s)] = (ml|ns)`, each dense `n_ao^2 x n_ao^2`, so `J = M_J
vec(D)` and `K = M_K vec(D)` for ANY `D` -- no pair packing, no `Dp`, no
symmetric-`D` assumption.

ONE KERNEL FAMILY, A LAYOUT ARGUMENT. Both layouts are the same block
`(ab|cd)` of one ORDERED shell quartet `(A B|C D)`, placed two ways:

    J   M_J[(a, b), (c, d)] = (ab|cd)      (m = a, n = b, l = c, s = d)
    K   M_K[(a, c), (b, d)] = (ab|cd)      (m = a, l = b, n = c, s = d)

so `build_dense_kernel(cls, "j")` (`eri_dj_{la}{lb}{lc}{ld}`) and
`build_dense_kernel(cls, "k")` (`eri_dk_...`) share everything up to the
scatter -- `kernel_eri.contraction`, unchanged: one thread per task, the
`Target.J` quartet's MMD plan for `am`, the bra and ket `E` blocks inlined,
the record loops serial, one accumulator per element in registers. The key
is the class's, not the layout's: the layout changes where a value goes,
never the value.

NO SYMMETRY. The class is the AO-order quartet of EVERY ordering (`la > lb`,
the ket's pair class before the bra's), and a task is one ordered quartet of
shells; nothing is mirrored and no element is skipped. Every element of
either matrix belongs to exactly one ordered quartet, so each is one plain
store by one thread -- no atomics, and the output is bit-reproducible run to
run. Against `kernel_eri`'s unique-quartet build each unique integral is
computed by every ordering of its quartet: up to 8x (`full_eri.dense`).

THE ROW RANGE. In both layouts the row's `m` is in the bra's first shell `A`,
so a launch over the tasks whose `A` is in a shell range `[sh0, sh1)` (the
staging's, `dense.dense_classes(basis, (sh0, sh1))`) writes exactly the
contiguous rows `[m0 n_ao, (m0 + n_m) n_ao)`, every column, and launches over
disjoint shell ranges write disjoint row blocks whose union is the matrix.
`eri` is the LOCAL block, `n_row = n_m n_ao` rows: element `(row, col)` is
`eri[row - m0 n_ao, col]`, flat `(row - m0 n_ao) n_col + col`, so the int32
bound is `n_row n_col < 2^31` per launch (`dense.row_range` checks it), not
`n_ao^4`. The kernel does not guard: a task whose `A` is outside the range
writes outside the block, so only the staging's tasks may be launched.

    params  n_task n_brec n_krec n_bpair n_kpair n_ao m0 n_row n_col (I32)
            zeta_ab[n_brec] P[n_brec, 3] XPA XPB [n_brec, 3] s_ab[n_brec]
            zeta_cd[n_krec] Q[n_krec, 3] XQC XQD [n_krec, 3] s_cd[n_krec]
            brec0 brec1 off_a off_b [n_bpair]  krec0 krec1 off_c off_d
            [n_kpair]  task_bra task_ket [n_task]  (I32)
            eri[n_row, n_col]  (OUT, each element written once)

`m0` is the range's first AO `m` (`off[sh0]`), `n_col = n_ao^2`. Elements of
classes not launched are untouched: the caller zeroes first.

THE LOW-STACK FORM, every class with a d or f shell (`kernel_eri.eri_lowstack`,
the same rule; `lowstack=True` forces it at any class): `kernel_eri`'s
module docstring has it -- `kernel_eri.lowstack_body` unchanged, one thread
per (task, bra Cartesian `a`) -- its `n_b` rows serially, where the packed
kernel takes one row a thread -- so `task_threads(cls) = n_a` threads per
task (`launch_shape(n_task, threads, rows=n_a)`), each element
accumulated in place in `eri` (the first record pair stores). Here every
element of a thread's rows is its own, so nothing is skipped and nothing
mirrored; the layout places each one as above. Same name, parameters and
output as the one-thread-per-task form.

A builder: vendor-neutral IR. It imports neither `ao/` nor the interpreter
nor `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.full_eri.kernel_eri import (
    contraction,
    eri_lowstack,
    input_params,
    launch,
    lowstack_body,
    lowstack_launch,
)
from felizjueves.eri.full_eri.kernel_eri import (
    launch_shape as _eri_launch_shape,
)
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Fn,
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
    "LAYOUTS",
    "OUT",
    "SCALARS",
    "build_dense_kernel",
    "dense_spec",
    "launch_shape",
    "task_threads",
]

OUT = "eri"
LAYOUTS = ("j", "k")
SCALARS = ("n_task", "n_brec", "n_krec", "n_bpair", "n_kpair", "n_ao", "m0",
           "n_row", "n_col")


def dense_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The ordered quartet class `(la lb|lc ld)`, `Target.ERI_DENSE`."""
    return Spec(Operator.COULOMB, 0, (la, lb, lc, ld), Target.ERI_DENSE)


def _spec(cls: Spec | tuple[int, int, int, int]) -> Spec:
    spec = cls if isinstance(cls, Spec) else dense_spec(*cls)
    if spec.target is not Target.ERI_DENSE:
        raise ValueError(f"{spec.key()} is not a Target.ERI_DENSE key")
    return spec


def task_threads(cls: Spec | tuple[int, int, int, int],
                 lowstack: bool | None = None) -> int:
    """Threads per task of the dense kernel: 1 in the one-thread-per-task
    form, `n_a` (one bra Cartesian `a`, its `n_b` rows, each) in the
    low-stack form (`lowstack=None` is `kernel_eri.eri_lowstack(cls)`)."""
    am = cls.am if isinstance(cls, Spec) else tuple(cls)
    if lowstack is None:
        lowstack = eri_lowstack(am)
    return len(ef.cartesians(am[0])) if lowstack else 1


def launch_shape(n_task: int, threads: int = 32, *, rows: int = 1
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: `rows` threads per task, as `kernel_eri`'s
    (`task_threads`)."""
    return _eri_launch_shape(n_task, threads, rows=rows)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def build_dense_kernel(cls: Spec | tuple[int, int, int, int], layout: str,
                       *, name: str | None = None,
                       lowstack: bool | None = None) -> Kernel:
    """The dense-ERI kernel of one ordered class in `layout` (`"j"` or
    `"k"`; module docstring). `lowstack` as `kernel_eri.build_eri_kernel`'s:
    `None` is `kernel_eri.eri_lowstack(cls)`."""
    spec = _spec(cls)
    if spec.levers:
        raise ValueError(f"{spec.key()}: the dense ERI kernel takes no "
                         "levers")
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r} is not one of {LAYOUTS}")
    if lowstack is None:
        lowstack = eri_lowstack(spec)
    if lowstack:
        return _build_lowstack(spec, layout, name)
    la, lb, lc, ld = spec.am
    na, nb = len(ef.cartesians(la)), len(ef.cartesians(lb))
    nc, nd = len(ef.cartesians(lc)), len(ef.cartesians(ld))

    def add(*xs: Expr) -> Expr:
        out = xs[0]
        for x in xs[1:]:
            out = _c(Fn.ADD, out, x)
        return out

    def times_nao(x: Expr) -> Expr:
        return _c(Fn.MUL, x, Var("n_ao"))

    # The local row is (m - m0) n_ao + (n), the column (l) n_ao + s, with
    # (n, l) = (b, c) in J and (c, b) in K.
    second, third = (("ob", nb, "b"), ("oc", nc, "c")) if layout == "j" \
        else (("oc", nc, "c"), ("ob", nb, "b"))
    row_idx = [Decl(f"pr{a}_{x}", Scalar.I32, init=add(
        times_nao(_c(Fn.SUB, add(Var("oa"), Lit(a)), Var("m0"))),
        Var(second[0]), Lit(x)))
        for a in range(na) for x in range(second[1])]
    col_idx = [Decl(f"pc{y}_{d}", Scalar.I32, init=add(
        times_nao(add(Var(third[0]), Lit(y))), Var("od"), Lit(d)))
        for y in range(third[1]) for d in range(nd)]
    stores: list[Stmt] = []
    for a in range(na):
        for b in range(nb):
            for c in range(nc):
                for d in range(nd):
                    r, k = a * nb + b, c * nd + d
                    row, col = (f"pr{a}_{b}", f"pc{c}_{d}") if layout == "j" \
                        else (f"pr{a}_{c}", f"pc{b}_{d}")
                    stores.append(Assign(Index(OUT, (Var(row), Var(col))),
                                         Var(f"acc{r}_{k}")))
    where = ("M_J[(a,b), (c,d)]" if layout == "j" else "M_K[(a,c), (b,d)]")
    per_task: tuple[Stmt, ...] = (
        *contraction(spec.am),
        Note("the local row and the column of every element: row (m - m0) "
             "n_ao + n, column l n_ao + s"),
        *row_idx,
        *col_idx,
        Note(f"scatter: (ab|cd) to {where}, one plain store per element, "
             "no mirror, no atomics -- each element belongs to one task"),
        *stores,
    )
    what = ("M_J[(m,n), (l,s)] = (mn|ls)" if layout == "j"
            else "M_K[(m,n), (l,s)] = (ml|ns)")
    body = launch((Note(f"dense ERI, {layout.upper()} layout, class ({la},"
                        f"{lb}|{lc},{ld}) at L = {la + lb + lc + ld}: {what}, "
                        "rows [m0 n_ao, m0 n_ao + n_row) "
                        "(eri/full_eri/dense.py, eri/eri.py)"),),
                  per_task)
    params = (
        *input_params(SCALARS),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Var("n_row"), Var("n_col"))),
    )
    return Kernel(name or f"eri_d{layout}_{la}{lb}{lc}{ld}", params, body)


def _params() -> tuple[Param, ...]:
    return (
        *input_params(SCALARS),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Var("n_row"), Var("n_col"))),
    )


def _build_lowstack(spec: Spec, layout: str, name: str | None) -> Kernel:
    """`build_dense_kernel`'s low-stack form (module docstring, THE
    LOW-STACK FORM): `kernel_eri.lowstack_body`, each element of the
    thread's rows placed by the layout's rule."""
    la, lb, lc, ld = spec.am

    def add(*xs: Expr) -> Expr:
        out = xs[0]
        for x in xs[1:]:
            out = _c(Fn.ADD, out, x)
        return out

    m_row = _c(Fn.MUL, _c(Fn.SUB, add(Var("oa"), Var("ra")), Var("m0")),
               Var("n_ao"))

    def element(ci: Expr, di: Expr) -> tuple[Index, None]:
        if layout == "j":  # M_J[(a, b), (c, d)]
            row = add(m_row, Var("ob"), Var("rb"))
            col = add(_c(Fn.MUL, add(Var("oc"), ci), Var("n_ao")), Var("od"),
                      di)
        else:  # M_K[(a, c), (b, d)]
            row = add(m_row, Var("oc"), ci)
            col = add(_c(Fn.MUL, add(Var("ob"), Var("rb")), Var("n_ao")),
                      Var("od"), di)
        return Index(OUT, (row, col)), None

    what = ("M_J[(m,n), (l,s)] = (mn|ls)" if layout == "j"
            else "M_K[(m,n), (l,s)] = (ml|ns)")
    rows = len(ef.cartesians(lb))
    body = lowstack_launch(
        (Note(f"dense ERI, {layout.upper()} layout, class ({la},{lb}|{lc},"
              f"{ld}) at L = {la + lb + lc + ld}, LOW-STACK: {what}, rows "
              "[m0 n_ao, m0 n_ao + n_row) (eri/full_eri/dense.py, "
              "eri/eri.py)"),),
        spec.am, rows, lowstack_body(spec.am, rows, element))
    return Kernel(name or f"eri_d{layout}_{la}{lb}{lc}{ld}", _params(), body)
