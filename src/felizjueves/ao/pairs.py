"""Primitive-pair data for the `[p|q]` kernel, and a Schwarz bound that bounds.

    primitive_pairs(sa, sb) -> tuple[PrimitivePair, ...]   one per (i, j)
    dress(e, ...)           -> the ONE place the three dressings multiply in
    hermite_bound(zeta, p, *, one_centre) -> I_p >= 0
    pair_bound(zeta, L, *, one_centre)    -> I_pair = sum_p I_p
    dmax(dq)                              -> max_q |D_q|, of the DRESSED D_q

A *primitive pair* of an ordered shell pair `(sa, sb)` is one exponent from
each, `alpha = sa.exponents[i]`, `beta = sb.exponents[j]`; it becomes one bra
(or ket) index record of the kernel's parameter block (`kernel.py`): a centre
`P = (alpha A + beta B)/zeta`, an exponent `zeta = alpha + beta`, and the
DRESSED Hermite coefficient block -- one row per Cartesian component pair
`(a, b)` in `cartesians(la) x cartesians(lb)` order, one column per Hermite
index in `hermite_order(la + lb)` order (= `plan.targets.hermite_targets`).

WHERE THE THREE DRESSINGS GO, decided once (milestone 4, "The physics this
milestone owns"; docs/implementation/j_engine.md, "The `E` transform").
`ao/hermite.py` produces McMurchie & Davidson's PURE `E` (entry 46 of
docs/references.md, eqs. 2.20-2.22). `dress` multiplies in, and nothing else
in this package does:

    the contraction coefficients   c_a c_b             (ao.shell.Shell)
    the primitive normalisation    N(la, alpha) N(lb, beta)   (ao.shell.primitive_norm)
    the pair factor                (pi / zeta)^{3/2}

The kernel keeps `(2/sqrt(pi)) theta` inside `[0]^{(m)}` (reference/pq.py,
`fundamental`), the host applies `(pi/zeta)^{3/2}` on the bra and
`(pi/eta)^{3/2}` on the ket, and

    (2/sqrt(pi)) theta (pi/zeta)^{3/2} (pi/eta)^{3/2}
        = 2 pi^{5/2} / (zeta eta sqrt(zeta + eta))

is McMurchie & Davidson's `lambda` -- the statement that the placement is
complete, asserted by test/test_ao_pairs.py.

THE BOUND, AND WHY IT IS PAIR-FACTOR-FREE (milestone 4; [JEngine, v2, eqs.
7-13] for `[p|q]`, McMurchie & Davidson for the Hermite Gaussians `Lambda_p`).
`G(p, q) = <Lambda_p, Lambda_q>_Coulomb` is the Coulomb inner product of two
Hermite Gaussians -- a positive-definite form -- so Cauchy-Schwarz holds
exactly:

    |G(p, q)| <= sqrt(G(p, p)) sqrt(G(q, q))

`G(p, q) = (pi/zeta)^{3/2} (pi/eta)^{3/2} [p|q]`, and the positive constants
cancel on both sides (each side carries one of each), leaving the kernel's own
undressed quantity bounded by

    I_p = sqrt( (-1)^{|p|} [2p]^{(0)}(PQ = 0, theta = sqrt(zeta/2)) )  >= 0

-- the self-integral of the pair with itself: `PQ = 0`, and `theta` is the
`zeta = eta` case of `theta^2 = zeta eta / (zeta + eta)`, i.e. `zeta/2`
(NOT `zeta`: that gives a bound that holds on compact pairs and fails on
diffuse ones). Then, for the kets of one primitive pair,

    |sum_q [p|q] D_q| <= sum_q I_p I_q |D_q| <= I_p (sum_q I_q) max_q |D_q|

which is exactly the product `Ip * (Iq * Dmax)` the emitted kernel screens on:
`Iq = pair_bound(eta, lq)`, `Dmax = dmax(D_q)` over the dressed ket vector. So
`I` carries no pair factor and `D` carries both of the ket's. The bra record's
scalar is `pair_bound(zeta, lp)` too: `sum_p I_p >= I_p` for every `p`, so it
bounds every `J_p` the record accumulates. test/test_ao_pairs.py proves the
inequality over randomised geometry rather than taking it on authority.

`(-1)^{|p|} [2p]^{(0)}` is a diagonal of a positive-definite form, so it is
non-negative. At `PQ = 0` the recurrence leaves only the `(r_d - 1)` terms,
whose signs are those of `[0]^{(|p|)} ~ (-2 theta^2)^{|p|}` -- so the product
is non-negative EXACTLY, not merely to rounding. A negative value therefore
means the evaluator's sign convention is wrong, and `hermite_bound` raises on
it; it never takes `abs` to hide one.

`Dmax` is over the DRESSED `D_q = sum_{cd} D_cd E^{cd}_q (dressed)` -- not the
AO density's largest element, and not computable from one.

THE EVALUATOR IS INJECTED. `one_centre(r, m, pq, theta) -> [r]^{(m)}` arrives
as an argument (tests pass `reference.pq.one_centre`); importing `reference`
here would make the bound share ancestry with the integrals it screens
(test/test_layering.py, rule "ao").

Plain tuples and floats, like `ao/shell.py`: `src/felizjueves` has no runtime
dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from felizjueves.ao.hermite import e_block, hermite_order
from felizjueves.ao.shell import Shell, cartesians, primitive_norm

Vec = tuple[float, float, float]
Ang = tuple[int, int, int]
OneCentre = Callable[[Ang, int, Vec, float], float]


@dataclass(frozen=True)
class PrimitivePair:
    """One bra (or ket) index record: `P`, `zeta`, and the dressed `E` block.

    `block[k]` is the dressed `E^{ab}_t` over `hermite_order(L)` for the k-th
    Cartesian pair `(a, b)` of `cartesians(la) x cartesians(lb)`, a outer.
    `i` and `j` are the primitives' indices in their shells.
    """

    i: int
    j: int
    P: Vec
    zeta: float
    L: int
    block: tuple[tuple[float, ...], ...]


def dress(e: Sequence[float], *, ca: float, cb: float, na: float, nb: float,
          zeta: float) -> tuple[float, ...]:
    """`c_a c_b N_a N_b (pi/zeta)^{3/2} E` -- the ONLY place any of the three
    dressings (module docstring) multiplies into a Hermite coefficient."""
    if not zeta > 0.0:
        raise ValueError(f"zeta = {zeta!r} is not > 0")
    scale = ca * cb * na * nb * (math.pi / zeta) ** 1.5
    return tuple(scale * x for x in e)


def primitive_pairs(sa: Shell, sb: Shell) -> tuple[PrimitivePair, ...]:
    """Every primitive pair of the ORDERED shell pair `(sa, sb)`, i outer."""
    A, B = sa.centre, sb.centre
    L = sa.l + sb.l
    carts = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
    out = []
    for i, (alpha, ca) in enumerate(zip(sa.exponents, sa.coefficients,
                                        strict=True)):
        na = primitive_norm(sa.l, alpha)
        for j, (beta, cb) in enumerate(zip(sb.exponents, sb.coefficients,
                                           strict=True)):
            nb = primitive_norm(sb.l, beta)
            zeta = alpha + beta
            P = tuple((alpha * A[d] + beta * B[d]) / zeta for d in range(3))
            block = tuple(
                dress(e_block(a, b, zeta, P, A, B),
                      ca=ca, cb=cb, na=na, nb=nb, zeta=zeta)
                for a, b in carts
            )
            out.append(PrimitivePair(i, j, P, zeta, L, block))
    return tuple(out)


def hermite_bound(zeta: float, p: Ang, *, one_centre: OneCentre) -> float:
    """`I_p = sqrt((-1)^{|p|} [2p]^{(0)}(PQ = 0, theta = sqrt(zeta/2)))`.

    Raises if the radicand is negative: that is a sign-convention error in
    the evaluator, not rounding (module docstring), and is never `abs`ed.
    """
    if not zeta > 0.0:
        raise ValueError(f"zeta = {zeta!r} is not > 0")
    if len(p) != 3 or any(type(c) is not int or c < 0 for c in p):
        raise ValueError(f"p = {p!r} is not a Hermite index of three ints >= 0")
    r = (2 * p[0], 2 * p[1], 2 * p[2])
    v = one_centre(r, 0, (0.0, 0.0, 0.0), math.sqrt(zeta / 2.0))
    s = -v if sum(p) % 2 else v
    if s < 0.0:
        raise ValueError(
            f"(-1)^|p| [2p]^(0) = {s!r} < 0 at p = {p}, zeta = {zeta!r}: a "
            "diagonal of a positive-definite form cannot be negative, so the "
            "evaluator's sign convention is wrong"
        )
    return math.sqrt(s)


def hermite_bounds(zeta: float, L: int, *,
                   one_centre: OneCentre) -> tuple[float, ...]:
    """`I_p` for every `p` in `hermite_order(L)`."""
    return tuple(hermite_bound(zeta, p, one_centre=one_centre)
                 for p in hermite_order(L))


def pair_bound(zeta: float, L: int, *, one_centre: OneCentre) -> float:
    """`I_pair = sum_p I_p` over `hermite_order(L)` -- the record's `Ip`/`Iq`."""
    return math.fsum(hermite_bounds(zeta, L, one_centre=one_centre))


def dmax(dq: Sequence[float]) -> float:
    """`max_q |D_q|` over the DRESSED ket vector -- the record's `Dmax`."""
    if not len(dq):
        raise ValueError("an empty ket vector has no Dmax")
    return max(abs(float(x)) for x in dq)
