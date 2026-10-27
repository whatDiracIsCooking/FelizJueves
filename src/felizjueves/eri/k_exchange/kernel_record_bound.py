"""K's dressed record bound kernel, `Target.RECORD_BOUND`:
`eri.k_exchange.k.record_bounds`.

    build_record_bound_kernel(spec | (la, lb), lowstack=None) -> Kernel
                                               one per ordered class
    record_bound_lowstack(spec | (la, lb))     -> bool     the form's rule
    record_bound_spec(la, lb)                  -> Spec     the class's key
    record_bound_name(spec | (la, lb))         -> str      its name, unbuilt
    launch_shape(n_rec, threads)               -> (grid, block)

K's screen (`eri.k_exchange.k`, "THE SCREEN", #200) skips a (bra, ket) record pair on
`(I_ac * I_bd) * Dm < tau`, and each `I` is the record's DRESSED Schwarz
scalar, `eri.k_exchange.k.record_bounds`:

    I = max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p| I_p(zeta)

for a record of the ORDERED shell pair `(a, b)`: `s = c_a c_b (pi/zeta)^{3/2}`
(`ao.pairs.dress`), `E` the undressed Hermite block (which carries the pair
factor `exp(-mu |AB|^2)` in its seed), and `I_p` the pair-factor-free bound
of `kernel_pair_bound` taken one `p` at a time. `eri.k_exchange.k`'s docstring proves it
bounds; `Target.PAIR_BOUND`'s `SUM_p I_p` is J's `Ip` / `Iq`, a different
number with neither the coefficients nor the pair factor in it, and it does
not decay with `|AB|`, so a density-decay factor (#484) read against it never
fires.

PER RECORD, one thread: the undressed block `E[r, j]` of
`e_fragment.block_fragment(la, lb)`, inlined, rows `block_rows(la, lb)`
(`a` outer) and columns `hermite_targets(la + lb)`; then, with `x =
sqrt(zeta)`, each `I_p = C_p x^{|p|} sqrt(x)`
(`kernel_pair_bound.hermite_bound_coefficients`, the same closed form at
`PQ = 0`, one `p` at a time); then per first-shell Cartesian `i`

    acc_i = SUM_{k in cartesians(lb)} SUM_{j} |s * E[i n_b + k, j]| * I_j

`k` outer, `j` inner, the host's term order, each term `|s * e| * I`
exactly as the host forms it, and `bound = max_i acc_i`. An `E` entry that is
structurally zero (`t_d > a_d + b_d`) is ABSENT, never read: the host adds
`|s * 0| * I = +0.0` there, and `+0.0` added to a sum of non-negative terms
that started at `+0.0` is the identity. `|.|` and `max` are spelled with `LT`
and `SUB`, as in `kernel_dq`.

ROUND-OFF, NOT BITWISE, against `eri.k_exchange.k.record_bounds` through `reference.pq`:
the host forms each `I_p` from the `[p|q]` recurrence at `PQ = 0` and sums
with `fsum`; the kernel uses the closed form and a plain left-to-right sum.
Every term is non-negative, so the sum's relative error is a few ulps per
record (test/test_kernel_record_bound.py states the bar). The screen compares
a product of these to a `tau` many orders above that.

THE CLASS IS AN ORDERED PAIR. The key is `(la, lb)` with `la` the FIRST
shell's momentum -- the record's output index (K's `a` on a bra, `b` on a
ket) -- and the max runs over its Cartesians. So `(la, lb)` and `(lb, la)`
are different kernels with different values (a `(p, s)` record maxes over
three rows of one term each, an `(s, p)` record sums them in one row), and a
consumer launches one per ordered class over records stored in that
orientation, `XPA = P - A` with `A` the first shell's centre. Never batched:
there are no Hermite targets to partition (`spec._PAIR_BATCH_REFUSAL`).

THE DENSITY IS NOT AN INPUT. The bound is a property of the record alone, so
a consumer computes it once per basis, not per SCF iteration.

    params  n_rec (I32)
            zeta_ab[n_rec]  XPA[n_rec, 3]  XPB[n_rec, 3]  s_ab[n_rec]
            bound[n_rec] (OUT)   I = eri.k_exchange.k.record_bounds, written whole

THE LOW-STACK FORM (#586, stage 3), the default from f up. The block
above is `E[n_ab][n_p]` per thread, `E[100][84]` at (ff), 67 KB: on sm_86 a
76,608 B stack frame with 12 KB of spill, reserved by CUDA for every
resident thread (~8 GB of a 10 GB RTX 3080), and ~32 KB at (df) / (fd).
`build_record_bound_kernel(cls)` therefore emits, whenever
`record_bound_lowstack(cls)` (a shell with `l >= LOWSTACK_MIN_L = 3`), a
kernel with the SAME name, parameters, launch and `bound` contract and no
`E^{ab}` block:

    tables = e_fragment.axis_tables_fragment(la, lb)   Ex Ey Ez, local
    |E{x,y,z}^{ij}_t| in place, over the reachable t <= i + j only
    I[j] = C_j x^{|t_j|} sqrt(x)                       a local, n_p doubles
    for each first-shell Cartesian a          e_fragment.cartesian_loops
        acc = 0
        for each second-shell Cartesian b
            acc += SUM_t |E_x||E_y||E_z| I[t]   e_fragment.rolled_e_sum
        acc = |s| acc;  bnd = max(bnd, acc)

That is the same number: `|s E^{ab}_t| I_t = |s| |E_x| |E_y| |E_z| I_t`
term by term (`E^{ab}_t = E_x E_y E_z`, `I_t > 0`), over the same reachable
terms, and the same max over the first shell. The frame is the three tables
and `I` (420 doubles at (ff)). The products and sums are associated
differently from the unrolled form's, so the two agree to rounding, at the
same `16 eps` against the host (test/test_kernel_record_bound.py). Through
d the unrolled text is unchanged byte for byte (pinned by sha256);
`lowstack=True` / `False` forces either form at any class.

MEASURED (#586; `nvcc -O3 -arch=sm_86 -rdc=true -DFJ_WAVEFRONT=32 -Xptxas
-v`, CUDA 13.0, and `hipcc --offload-arch=gfx1200 -O3
-Rpass-analysis=kernel-resource-usage`, ROCm 7.2, compile only; both on
`emit.gpu`'s text, source in bytes), unrolled (534a8e7) / low-stack:

    unit  source            sm_86 stack     spill st     regs      nvcc
    03    15,838 / 9,208    0 / 256         0 / 0        70 / 64   0.7 / 0.7 s
    30    16,621 / 9,184    0 / 256         0 / 0        70 / 64   0.7 / 0.7 s
    13    60,805 / 14,235   0 / 1,248       0 / 0       254 / 96   0.9 / 0.8 s
    31    61,420 / 14,235   16 / 1,248      16 / 0      255 / 106  0.9 / 1.0 s
    23   169,304 / 21,071   32,000 / 2,176  7,024 / 0   255 / 96   2.8 / 0.9 s
    32   169,672 / 21,071   31,744 / 2,176  5,744 / 0   255 / 118  2.7 / 0.9 s
    33   386,304 / 29,835   76,608 / 3,360  12,056 / 0  255 / 162  29.0 / 0.9 s

    unit  gfx1200 scratch/lane   VGPRs
    03    0 / 0                  70 / 96
    30    0 / 0                  121 / 94
    13    1,212 / 1,264          192 / 146
    31    1,300 / 1,264          192 / 152
    23    5,068 / 2,192          192 / 102
    32    4,492 / 2,192          192 / 102
    33    10,260 / 3,376         192 / 98

At (ff) the sm_86 frame is exactly the four local arrays (3 x 112 + 84
doubles), no spill: 3,360 B x 104,448 resident threads is ~350 MB of the
RTX 3080, against ~8 GB. `03` / `30` grow from no frame to 256 B (part
of the tables, indexed by loop variables, stays in memory), far inside the
8 KB budget.

Nothing here is CUDA- or HIP-specific, and nothing here imports `ao`,
`reference/`, the interpreter, a backend or a card.
"""

from __future__ import annotations

from felizjueves.eri.kernel_pair_bound import hermite_bound_coefficients
from felizjueves.ir import (
    Assign,
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
from felizjueves.mmd.e_fragment import (
    AXIS_ARRAYS,
    E_ARRAY,
    HALF_INV_ZETA,
    ZETA,
    axis_table_extent,
    axis_table_index,
    axis_tables_fragment,
    block_fragment,
    block_rows,
    block_shape,
    cartesian_loops,
    rolled_e_sum,
)
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "LOWSTACK_MIN_L",
    "PARAMS",
    "build_record_bound_kernel",
    "launch_shape",
    "record_bound_lowstack",
    "record_bound_name",
    "record_bound_spec",
]

# The parameter block, in order.
PARAMS = ("n_rec", "zeta_ab", "XPA", "XPB", "s_ab", "bound")

REC = "k"  # the record index: one thread per record

BOUND_IS_RECORD_BOUNDS = (
    "bound[k] = ao.k.record_bounds = max_{i in a} SUM_{k in b} SUM_p "
    "|s E^{a_i b_k}_p| I_p(zeta): rows over the FIRST shell, the dressed E, "
    "I_p = C_p x^|p| sqrt(x) with x = sqrt(zeta) (kernel_record_bound.py, "
    "module docstring)"
)


def record_bound_spec(la: int, lb: int) -> Spec:
    """The ordered pair class `(la, lb)` as a generation key,
    `Target.RECORD_BOUND`."""
    return Spec(Operator.COULOMB, 0, (la, lb), Target.RECORD_BOUND)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 2:
        spec = record_bound_spec(*cls)
    else:
        raise TypeError(f"build_record_bound_kernel takes (la, lb) or a Spec, "
                        f"not {cls!r}")
    if spec.target is not Target.RECORD_BOUND:
        raise ValueError(f"{spec.key()} is not a Target.RECORD_BOUND key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the record bound takes no levers")
    la, lb = spec.am
    return la, lb


def launch_shape(n_rec: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`, `grid = ceil(n_rec / threads)`: one thread per
    record. The mapping is not part of the numerical contract."""
    if type(n_rec) is not int or n_rec < 1:
        raise ValueError(f"n_rec = {n_rec!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_rec // threads), (threads, 1)


def record_bound_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_record_bound_kernel(cls)` gives its kernel, without
    building it."""
    la, lb = _class(cls)
    return f"record_bound_{la}{lb}"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


LOWSTACK_MIN_L = 3  # the low-stack form from f up (#586)


def record_bound_lowstack(cls: Spec | tuple[int, int]) -> bool:
    """THE SELECTION RULE (#586): True when the class has a shell of
    `l >= LOWSTACK_MIN_L` (f or above), where `build_record_bound_kernel`
    emits the low-stack form by default; False through d, where the unrolled
    form stays the default and its text is unchanged."""
    return max(_class(cls)) >= LOWSTACK_MIN_L


def build_record_bound_kernel(cls: Spec | tuple[int, int], *,
                              name: str | None = None,
                              lowstack: bool | None = None) -> Kernel:
    """The dressed record bound kernel of one ordered pair class (module
    docstring).

    `cls` is a `Target.RECORD_BOUND` `Spec` or `(la, lb)`, `la` the first
    shell's momentum. `lowstack` picks the form: `None` (the default) is
    `record_bound_lowstack(cls)`, so through d the unrolled kernel and at f
    the low-stack one (THE LOW-STACK FORM); `True` / `False` force either,
    at any class. Both have the same name, parameters, launch and `bound`
    contract.
    """
    la, lb = _class(cls)
    if lowstack is None:
        lowstack = record_bound_lowstack((la, lb))
    if lowstack:
        return _build_lowstack(la, lb, name)
    L = la + lb
    frag = block_fragment(la, lb)
    rows = block_rows(la, lb)
    n_ab, n_p = block_shape(la, lb)
    n_b = (lb + 1) * (lb + 2) // 2
    ts = hermite_targets(L)
    coeffs = hermite_bound_coefficients(L)
    assert len(coeffs) == n_p and n_ab % n_b == 0
    k = Var(REC)

    def rec(arr: str, *more: Expr) -> Index:
        return Index(arr, (k, *more))

    bindings: dict[str, Expr] = {ZETA: Var(ZETA), E_ARRAY: Var(E_ARRAY)}
    head: list[Stmt] = [Decl(ZETA, Scalar.F64, init=rec("zeta_ab"))]
    if HALF_INV_ZETA in frag.binds:
        head.append(Decl(HALF_INV_ZETA, Scalar.F64,
                         init=_c(Fn.DIV, Lit(0.5), Var(ZETA))))
        bindings[HALF_INV_ZETA] = Var(HALF_INV_ZETA)
    for d in range(3):
        bindings[Factor.pa(d).name_in_kernel] = rec("XPA", Lit(d))
        bindings[Factor.pb(d).name_in_kernel] = rec("XPB", Lit(d))

    def xp(n: int) -> Var:
        return Var(f"xp{n}")

    def ip(j: int) -> Var:
        return Var(f"ip{j}")

    per_rec: list[Stmt] = [
        Note(BOUND_IS_RECORD_BOUNDS),
        *head,
        Decl("sk", Scalar.F64, init=rec("s_ab")),
        Note("I_p = C_p x^|p| sqrt(x), x = sqrt(zeta): kernel_pair_bound's "
             "closed form at PQ = 0, one p at a time"),
        Decl("x", Scalar.F64, init=_c(Fn.SQRT, Var(ZETA))),
        Decl(xp(0).name, Scalar.F64, init=_c(Fn.SQRT, Var("x"))),
        *(Decl(xp(n).name, Scalar.F64, init=_c(Fn.MUL, xp(n - 1), Var("x")))
          for n in range(1, L + 1)),
        *(Decl(ip(j).name, Scalar.F64,
               init=_c(Fn.MUL, Lit(coeffs[j]), xp(sum(t))))
          for j, t in enumerate(ts)),
        Note("the undressed E block of this record: e_fragment.block_fragment, "
             "inlined; the dressing s arrives as a host scalar"),
        Decl(E_ARRAY, Scalar.F64, (Lit(n_ab), Lit(n_p))),
        Inline(frag, bindings),
        Note("acc_i = SUM_{k in b} SUM_p |s * E[i n_b + k, p]| * I_p, k outer, "
             "p inner, the host's term order; an absent E entry (t_d > a_d + "
             "b_d) is a structural zero and is skipped. bound = max_i acc_i"),
        Decl("ae", Scalar.F64, init=Lit(0.0)),
        Decl("bnd", Scalar.F64, init=Lit(0.0)),
    ]
    ae = Var("ae")
    for i in range(n_ab // n_b):
        acc = Var(f"acc{i}")
        per_rec.append(Decl(acc.name, Scalar.F64, init=Lit(0.0)))
        for r in range(i * n_b, (i + 1) * n_b):
            a, b = rows[r]
            for j, t in enumerate(ts):
                if any(t[d] > a[d] + b[d] for d in range(3)):
                    continue
                per_rec += [
                    Assign(ae, _c(Fn.MUL, Var("sk"),
                                  Index(E_ARRAY, (Lit(r), Lit(j))))),
                    If(_c(Fn.LT, ae, Lit(0.0)),
                       (Assign(ae, _c(Fn.SUB, Lit(0.0), ae)),)),
                    Assign(acc, _c(Fn.ADD, acc, _c(Fn.MUL, ae, ip(j)))),
                ]
        per_rec.append(If(_c(Fn.LT, Var("bnd"), acc),
                          (Assign(Var("bnd"), acc),)))
    per_rec.append(Assign(rec("bound"), Var("bnd")))

    return Kernel(name or record_bound_name((la, lb)), _params(),
                  _launch(la, lb, "", per_rec))


def _launch(la: int, lb: int, form: str, per_rec: list[Stmt]
            ) -> tuple[Stmt, ...]:
    """The kernel body around one record's statements: one thread per
    record, guarded by `n_rec`. `form` tags the head note."""
    k = Var(REC)
    return (
        Note(f"K's dressed record bound I = ao.k.record_bounds of an ordered "
             f"pair class ({la},{lb}){form}, rows over the first shell: kb's "
             f"Ip or Iq"),
        Note("launch: one thread per record, k = threadIdx.x + blockIdx.x * "
             "blockDim.x"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, k, Var("n_rec")), tuple(per_rec)),
    )


def _params() -> tuple[Param, ...]:
    """The parameter block (module docstring), one for both forms."""
    nr = Var("n_rec")

    def f64(n: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(n, Scalar.F64, array=True, intent=intent, shape=shape)

    params = (
        Param("n_rec", Scalar.I32),
        f64("zeta_ab", nr), f64("XPA", nr, Lit(3)), f64("XPB", nr, Lit(3)),
        f64("s_ab", nr),
        f64("bound", nr, intent=Intent.OUT),
    )
    assert tuple(p.name for p in params) == PARAMS
    return params


# ---------------------------------------------------------------------------
# The low-stack form (#586, stage 3)
# ---------------------------------------------------------------------------

I_ARRAY = "I"  # I_p of the record, hermite_targets(la + lb) order

LOWSTACK_NOTE = (
    "LOW-STACK (#586): per-axis E tables (3 x (la+1)(lb+1)(la+lb+1) "
    "doubles) in place of the E^{ab}_t block, their entries taken |.| in "
    "place, I_p in a local I, the row sums as loops: |s E_t| I_t = |s| "
    "|E_x| |E_y| |E_z| I_t term by term")


def _abs_in_place(at: Index | Var) -> If:
    return If(_c(Fn.LT, at, Lit(0.0)), (Assign(at, _c(Fn.SUB, Lit(0.0), at)),))


def _build_lowstack(la: int, lb: int, name: str | None) -> Kernel:
    """`build_record_bound_kernel`'s kernel with no `E^{ab}` block (module
    docstring, THE LOW-STACK FORM): the same parameters, launch and
    `bound`."""
    L = la + lb
    frag = axis_tables_fragment(la, lb)
    ts = hermite_targets(L)
    coeffs = hermite_bound_coefficients(L)
    assert len(coeffs) == len(ts)
    k = Var(REC)

    def rec(arr: str, *more: Expr) -> Index:
        return Index(arr, (k, *more))

    bindings: dict[str, Expr] = {ZETA: Var(ZETA),
                                 **{n: Var(n) for n in AXIS_ARRAYS}}
    head: list[Stmt] = [Decl(ZETA, Scalar.F64, init=rec("zeta_ab"))]
    if HALF_INV_ZETA in frag.binds:
        head.append(Decl(HALF_INV_ZETA, Scalar.F64,
                         init=_c(Fn.DIV, Lit(0.5), Var(ZETA))))
        bindings[HALF_INV_ZETA] = Var(HALF_INV_ZETA)
    for d in range(3):
        bindings[Factor.pa(d).name_in_kernel] = rec("XPA", Lit(d))
        bindings[Factor.pb(d).name_in_kernel] = rec("XPB", Lit(d))

    def xp(n: int) -> Var:
        return Var(f"xp{n}")

    # |E^{ij}_t| of every reachable entry, in place; t > i + j is never
    # written by the fragment and never read here or by rolled_e_sum.
    ti, tj, tt = Var("ti"), Var("tj"), Var("tt")
    at = axis_table_index(ti, tj, tt, lb, L)
    absolute = For(ti.name, Lit(0), Lit(la + 1), Lit(1), (
        For(tj.name, Lit(0), Lit(lb + 1), Lit(1), (
            For(tt.name, Lit(0), _c(Fn.ADD, _c(Fn.ADD, ti, tj), Lit(1)),
                Lit(1), (
                    Decl("ix", Scalar.I32, init=at),
                    *(_abs_in_place(Index(n, (Var("ix"),)))
                      for n in AXIS_ARRAYS),
                )),)),))

    acc, bnd = Var("acc"), Var("bnd")

    def row(a: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def cell(b: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            return (
                *rolled_e_sum(la, lb, a, b,
                              lambda h: Index(I_ARRAY, (h,)), "u"),
                Assign(acc, _c(Fn.ADD, acc, Var("u"))),
            )
        return (
            Decl(acc.name, Scalar.F64, init=Lit(0.0)),
            cartesian_loops(lb, ("bx", "by", "bz"), cell, tag="b"),
            Assign(acc, _c(Fn.MUL, Var("sk"), acc)),
            If(_c(Fn.LT, bnd, acc), (Assign(bnd, acc),)),
        )

    extent = Lit(axis_table_extent(la, lb))
    per_rec: list[Stmt] = [
        Note(BOUND_IS_RECORD_BOUNDS),
        Note(LOWSTACK_NOTE),
        *head,
        Note("|s|: the dressing's sign is the coefficients', and the bound "
             "takes |s E| term by term"),
        Decl("sk", Scalar.F64, init=rec("s_ab")),
        _abs_in_place(Var("sk")),
        Note("I_p = C_p x^|p| sqrt(x), x = sqrt(zeta): kernel_pair_bound's "
             "closed form at PQ = 0, one p at a time, into the local I"),
        Decl("x", Scalar.F64, init=_c(Fn.SQRT, Var(ZETA))),
        Decl(xp(0).name, Scalar.F64, init=_c(Fn.SQRT, Var("x"))),
        *(Decl(xp(n).name, Scalar.F64, init=_c(Fn.MUL, xp(n - 1), Var("x")))
          for n in range(1, L + 1)),
        Decl(I_ARRAY, Scalar.F64, (Lit(len(ts)),)),
        *(Assign(Index(I_ARRAY, (Lit(j),)),
                 _c(Fn.MUL, Lit(coeffs[j]), xp(sum(t))))
          for j, t in enumerate(ts)),
        Note("the record's per-axis E^{ij}_t, one local table per axis "
             "(e_fragment.axis_tables_fragment); no E^{ab}_t block is formed"),
        *(Decl(n, Scalar.F64, (extent,)) for n in AXIS_ARRAYS),
        Inline(frag, bindings),
        Note("|E| per axis, over the reachable entries only"),
        absolute,
        Note("acc_a = |s| SUM_{b} SUM_t |E_x||E_y||E_z| I_t over the reachable "
             "t (e_fragment.rolled_e_sum), b in cartesians order; bound = "
             "max_a acc_a"),
        Decl(bnd.name, Scalar.F64, init=Lit(0.0)),
        cartesian_loops(la, ("ax", "ay", "az"), row, tag="a"),
        Assign(rec("bound"), bnd),
    ]
    return Kernel(name or record_bound_name((la, lb)), _params(),
                  _launch(la, lb, ", LOW-STACK (#586)", per_rec))
