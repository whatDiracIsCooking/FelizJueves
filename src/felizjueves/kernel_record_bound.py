"""K's dressed record bound kernel, `Target.RECORD_BOUND`: `ao.k.record_bounds`.

    build_record_bound_kernel(spec | (la, lb)) -> Kernel   one per ordered class
    record_bound_spec(la, lb)                  -> Spec     the class's key
    record_bound_name(spec | (la, lb))         -> str      its name, unbuilt
    launch_shape(n_rec, threads)               -> (grid, block)

K's screen (`ao.k`, "THE SCREEN", #200) skips a (bra, ket) record pair on
`(I_ac * I_bd) * Dm < tau`, and each `I` is the record's DRESSED Schwarz
scalar, `ao.k.record_bounds`:

    I = max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p| I_p(zeta)

for a record of the ORDERED shell pair `(a, b)`: `s = c_a c_b (pi/zeta)^{3/2}`
(`ao.pairs.dress`), `E` the undressed Hermite block (which carries the pair
factor `exp(-mu |AB|^2)` in its seed), and `I_p` the pair-factor-free bound
of `kernel_pair_bound` taken one `p` at a time. `ao.k`'s docstring proves it
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

ROUND-OFF, NOT BITWISE, against `ao.k.record_bounds` through `reference.pq`:
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
            bound[n_rec] (OUT)   I = ao.k.record_bounds, written whole

Nothing here is CUDA- or HIP-specific, and nothing here imports `ao`,
`reference/`, the interpreter, a backend or a card.
"""

from __future__ import annotations

from felizjueves.e_fragment import (
    E_ARRAY,
    HALF_INV_ZETA,
    ZETA,
    block_fragment,
    block_rows,
    block_shape,
)
from felizjueves.ir import (
    Assign,
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
from felizjueves.kernel_pair_bound import hermite_bound_coefficients
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "PARAMS",
    "build_record_bound_kernel",
    "launch_shape",
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


def build_record_bound_kernel(cls: Spec | tuple[int, int], *,
                              name: str | None = None) -> Kernel:
    """The dressed record bound kernel of one ordered pair class (module
    docstring).

    `cls` is a `Target.RECORD_BOUND` `Spec` or `(la, lb)`, `la` the first
    shell's momentum.
    """
    la, lb = _class(cls)
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

    body: tuple[Stmt, ...] = (
        Note(f"K's dressed record bound I = ao.k.record_bounds of an ordered "
             f"pair class ({la},{lb}), rows over the first shell: kb's Ip or "
             f"Iq"),
        Note("launch: one thread per record, k = threadIdx.x + blockIdx.x * "
             "blockDim.x"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, k, Var("n_rec")), tuple(per_rec)),
    )
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
    return Kernel(name or record_bound_name((la, lb)), params, body)
