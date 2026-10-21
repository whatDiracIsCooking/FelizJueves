"""The record bound kernel, `Target.PAIR_BOUND`: each record's `Ip` / `Iq`.

    build_pair_bound_kernel(spec | (la, lb)) -> Kernel   one per pair class
    pair_bound_spec(la, lb)                  -> Spec     the class's key
    pair_bound_name(spec | (la, lb))         -> str      its name, unbuilt
    pair_bound_coefficients(L)               -> (D_0, ..., D_L)
    hermite_bound_coefficients(L)            -> (C_p for p in Hermite order)
    launch_shape(n_rec, threads)             -> (grid, block)

The `[p|q]` kernel screens a (bra, ket) record pair on `Dmax * Ip * Iq < tau`
(`kernel.py`), and each `I` is the record's pair-factor-free Schwarz bound
`ao.pairs.pair_bound(zeta, L)`: the sum over `p` in `hermite_order(L)` of

    I_p = sqrt( (-1)^{|p|} [2p]^{(0)}(PQ = 0, theta = sqrt(zeta/2)) )

(ao/pairs.py, "THE BOUND, AND WHY IT IS PAIR-FACTOR-FREE", which proves it
bounds). Until this target the bound was host Python only. A record's bound
depends on its exponent `zeta` and its class's Hermite order `L = la + lb`,
nothing else, so this is one thread per record and no `E`, no recurrence and
no Boys: the self-integral has a closed form.

THE CLOSED FORM. At `PQ = 0` the Hermite recurrence `R_{t+1}^{(m)} = t
R_{t-1}^{(m+1)} + X R_t^{(m+1)}` loses its `X` term, so per axis `R_{2a}^{(m)}
= (2a - 1)!! R_0^{(m+a)}`, and the fundamental is the kernel's own
`[0]^{(m)} = (2/sqrt(pi)) theta (-2 theta^2)^m F_m(0)` with `F_m(0) =
1/(2m + 1)` (reference/pq.py's `W0` convention, which `kernel.py` keeps).
With `2 theta^2 = zeta` the sign `(-1)^{|p|}` cancels and

    (-1)^{|p|} [2p]^{(0)} = (2/sqrt(pi)) sqrt(zeta/2) zeta^{|p|}
                            PROD_d (2 p_d - 1)!! / (2|p| + 1)

So with `x = sqrt(zeta)`, `I = sqrt(x) SUM_{n=0..L} D_n x^n`, where

    D_n = (2/sqrt(pi))^{1/2} 2^{-1/4}
          SUM_{|p| = n} sqrt(PROD_d (2 p_d - 1)!! / (2n + 1))

are constants of `L` alone, folded at build time (`pair_bound_coefficients`,
exact rationals under one `sqrt` each, summed with `fsum`). The kernel is two
`SQRT`s and a Horner fold, highest power first.

ROUND-OFF, NOT BITWISE, against `ao.pairs.pair_bound` through
`reference.pq.one_centre`: the host runs the recurrence and sums the `I_p`
with `fsum`, the kernel evaluates the closed form. The two derivations share
no code, which is what makes the host a witness of the closed form; they
agree to a few ulps (test/test_kernel_pair_bound.py states the bound). The
screen compares the product to a `tau` many orders above that, so a bound
low by an ulp moves no decision a test can see; it is still a bound to
rounding, not a rigorous one, exactly as the host's is.

THE CLASS IS A PAIR, LIKE `DQ`'S. The key is `(la, lb)` so a consumer that
keys every per-pair family by its pair class indexes this one the same way;
the text depends on `la + lb` only, so `(0, 2)` and `(1, 1)` emit the same
body under two names. Never batched: there are no Hermite targets to
partition, only a closed form per record (`spec._PAIR_BATCH_REFUSAL`).

    params  n_rec (I32)
            zeta[n_rec]       the record's exponent, zeta for a bra, eta for a ket
            bound[n_rec] (OUT)  I = pair_bound(zeta, la + lb), written whole

Nothing here is CUDA- or HIP-specific, and nothing here imports `ao`,
`reference/`, the interpreter, a backend or a card.
"""

from __future__ import annotations

import math
from fractions import Fraction

from felizjueves.ir import (
    Assign,
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
    "build_pair_bound_kernel",
    "hermite_bound_coefficients",
    "launch_shape",
    "pair_bound_coefficients",
    "pair_bound_name",
    "pair_bound_spec",
]

REC = "k"  # the record index: one thread per record

BOUND_IS_PAIR_BOUND = (
    "bound[k] = ao.pairs.pair_bound(zeta[k], L) = SUM_p sqrt((-1)^|p| "
    "[2p]^(0)(PQ = 0, theta = sqrt(zeta/2))), in its closed form at PQ = 0: "
    "sqrt(x) SUM_n D_n x^n with x = sqrt(zeta) (kernel_pair_bound.py, module "
    "docstring)"
)


def pair_bound_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` as a generation key, `Target.PAIR_BOUND`."""
    return Spec(Operator.COULOMB, 0, (la, lb), Target.PAIR_BOUND)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 2:
        spec = pair_bound_spec(*cls)
    else:
        raise TypeError(f"build_pair_bound_kernel takes (la, lb) or a Spec, "
                        f"not {cls!r}")
    if spec.target is not Target.PAIR_BOUND:
        raise ValueError(f"{spec.key()} is not a Target.PAIR_BOUND key")
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


def pair_bound_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_pair_bound_kernel(cls)` gives its kernel, without
    building it."""
    la, lb = _class(cls)
    return f"pair_bound_{la}{lb}"


def _double_factorial(k: int) -> int:
    """`k!!`, with `(-1)!! = 1`."""
    return math.prod(range(k, 0, -2)) if k > 0 else 1


def pair_bound_coefficients(L: int) -> tuple[float, ...]:
    """`(D_0, ..., D_L)`: the Horner coefficients of the closed form in
    `x = sqrt(zeta)` (module docstring), the constant `(2/sqrt(pi))^{1/2}
    2^{-1/4}` folded in. Each `sqrt` is of an exact rational."""
    if type(L) is not int or L < 0:
        raise ValueError(f"L = {L!r} is not a non-negative int")
    by_order: list[list[float]] = [[] for _ in range(L + 1)]
    for p in hermite_targets(L):
        n = sum(p)
        c = Fraction(math.prod(_double_factorial(2 * x - 1) for x in p),
                     2 * n + 1)
        by_order[n].append(math.sqrt(c))
    k = math.sqrt(2.0 / math.sqrt(math.pi)) * 2.0 ** -0.25
    return tuple(k * math.fsum(terms) for terms in by_order)


def hermite_bound_coefficients(L: int) -> tuple[float, ...]:
    """`C_p` for each `p` in `hermite_targets(L)` order: the ONE-`p` terms of
    the closed form, `I_p = C_p x^{|p|} sqrt(x)` with `x = sqrt(zeta)` and
    `C_p = (2/sqrt(pi))^{1/2} 2^{-1/4} sqrt(PROD_d (2 p_d - 1)!! / (2|p| +
    1))` (module docstring). `pair_bound_coefficients` sums these per order;
    `kernel_record_bound` weights each `I_p` by a dressed `|E_p|`, so it
    needs them apart."""
    if type(L) is not int or L < 0:
        raise ValueError(f"L = {L!r} is not a non-negative int")
    k = math.sqrt(2.0 / math.sqrt(math.pi)) * 2.0 ** -0.25
    return tuple(
        k * math.sqrt(Fraction(math.prod(_double_factorial(2 * x - 1)
                                         for x in p), 2 * sum(p) + 1))
        for p in hermite_targets(L))


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def build_pair_bound_kernel(cls: Spec | tuple[int, int], *,
                            name: str | None = None) -> Kernel:
    """The record bound kernel of one pair class (module docstring).

    `cls` is a `Target.PAIR_BOUND` `Spec` or `(la, lb)`.
    """
    la, lb = _class(cls)
    L = la + lb
    coeffs = pair_bound_coefficients(L)
    k = Var(REC)

    # Horner, highest power first: acc = D_L; acc = acc * x + D_n.
    horner: list[Stmt] = [Decl("acc", Scalar.F64, init=Lit(coeffs[L]))]
    for n in range(L - 1, -1, -1):
        horner.append(Assign(Var("acc"), _c(
            Fn.ADD, _c(Fn.MUL, Var("acc"), Var("x")), Lit(coeffs[n]))))

    per_rec: tuple[Stmt, ...] = (
        Note(BOUND_IS_PAIR_BOUND),
        Decl("x", Scalar.F64, init=_c(Fn.SQRT, Index("zeta", (k,)))),
        *horner,
        Assign(Index("bound", (k,)),
               _c(Fn.MUL, Var("acc"), _c(Fn.SQRT, Var("x")))),
    )
    body: tuple[Stmt, ...] = (
        Note(f"the record bound I = pair_bound(zeta, L) of a pair class with "
             f"la + lb = L = {L}: the [p|q] kernel's Ip or Iq"),
        Note("launch: one thread per record, k = threadIdx.x + blockIdx.x * "
             "blockDim.x"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, k, Var("n_rec")), per_rec),
    )
    nr = Var("n_rec")
    params = (
        Param("n_rec", Scalar.I32),
        Param("zeta", Scalar.F64, array=True, shape=(nr,)),
        Param("bound", Scalar.F64, array=True, intent=Intent.OUT, shape=(nr,)),
    )
    return Kernel(name or pair_bound_name((la, lb)), params, body)
