"""Primitive-pair data for the `[p|q]` kernel, and a Schwarz bound that bounds.

    primitive_pairs(sa, sb) -> tuple[PrimitivePair, ...]   one per (i, j)
    dress(e, ...)           -> the ONE place the two dressings multiply in
    hermite_bound(zeta, p, *, one_centre) -> I_p >= 0
    pair_bound(zeta, L, *, one_centre)    -> I_pair = sum_p I_p
    dmax(dq)                              -> max_q |D_q|, of the DRESSED D_q
    one_electron_bound(sa, sb, *, T, weight) -> >= max |S or V block|  (#236)
    overlap_bound / kinetic_bound / potential_bound -> the three, per pair
    point_charge_bounds(zeta, L, *, one_centre) -> B_t >= |[t]^(0)|, any C
    pair_list(basis, *, tau, bound)       -> PairList, the screened shell pairs
    primitive_schwarz_bound(pp, *, one_centre)  -> G_[ab] of one primitive pair
    primitive_schwarz_bounds(sa, sb, *, one_centre) -> G_[ab] per primitive pair
    prune(bounds, *, eps)                 -> the kept prefix, bound-descending
    narrowed(values)                      -> an operand's fp32 packing (#347)

A *primitive pair* of an ordered shell pair `(sa, sb)` is one exponent from
each, `alpha = sa.exponents[i]`, `beta = sb.exponents[j]`; it becomes one bra
(or ket) index record of the kernel's parameter block (`kernel.py`): a centre
`P = (alpha A + beta B)/zeta`, an exponent `zeta = alpha + beta`, and the
DRESSED Hermite coefficient block -- one row per Cartesian component pair
`(a, b)` in `cartesians(la) x cartesians(lb)` order, one column per Hermite
index in `hermite_order(la + lb)` order (= `plan.targets.hermite_targets`).

WHERE THE DRESSINGS GO, decided once (milestone 4, "The physics this
milestone owns"; docs/implementation/j_engine.md, "The `E` transform").
`ao/hermite.py` produces McMurchie & Davidson's PURE `E` (entry 46 of
docs/references.md, eqs. 2.20-2.22). `dress` multiplies in, and nothing else
in this package does:

    the contraction coefficients   c_a c_b             (ao.shell.Shell)
    the pair factor                (pi / zeta)^{3/2}

THE PRIMITIVE NORMALISATION IS NOT A DRESSING (#85, correcting #83). A
`Shell`'s coefficients multiply UNNORMALISED primitives (ao/shell.py's
module docstring, the one stated convention), so any normalisation is
already IN them -- put there by whoever built the shell from basis-file
coefficients, via `ao.shell.primitive_norm`, or by `test/oracle/basis.py`
from libcint's. Multiplying `N(la, alpha) N(lb, beta)` in here as well
normalised twice: a clean per-class constant against libcint's `int2e`,
which test/test_ao_eri.py now pins.

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

THE ONE-ELECTRON PAIR BOUND, PROVED (milestone 11, #236). Beside the Schwarz
bound `J` uses, a one-electron block has a bound of its own, read off the
SAME dressed Hermite coefficients the block is built from -- so no safety
multiplier is guessed and none can drift:

    one_electron_bound(sa, sb, *, T, weight) >= max_{a,b} |block[a][b]|

For `S` (`overlap_bound`, `T = 0`) the block IS `SUM_pp block_pp[ab][0]`
(ao/overlap.py), and the triangle inequality gives

    |S_ab| <= SUM_pp max_{ab} |c_a c_b (pi/zeta)^{3/2} E^{ab}_0[pp]|

-- equality for an `s`-`s` pair whose terms share a sign (one row, no
cancellation), which test/test_ao_pairs.py asserts. For `V`
(`potential_bound`, `T = la + lb`) the block is
`-SUM_C Z_C SUM_pp SUM_t block_pp[ab][t] [t]^(0)(P - C, sqrt(zeta))`
(ao/potential.py), and every `[t]^(0)` is bounded uniformly in `C` by
`point_charge_bounds` (below), so

    |V_ab| <= (SUM_C |Z_C|) SUM_pp max_{ab} SUM_t |block_pp[ab][t]| B_t(zeta)

For `T` (`kinetic_bound`) the block is the fixed `K_ij` combination of 1D
overlaps read at `lb + 2` (ao/kinetic.py); its bound is that combination
with every coefficient and every overlap replaced by its absolute value,

    |K_ij| <= 2 beta^2 |S_{i,j+2}| + beta (2j+1) |S_ij| + 1/2 j(j-1) |S_{i,j-2}|
    |T_ab| <= |K_x| |S_y| |S_z| + |S_x| |K_y| |S_z| + |S_x| |S_y| |K_z|

dressed by `dress` with `|c_a| |c_b|`, maximised over `ab`, summed over pp.

IN FLOATING POINT, TOO, FOR `S` AND `T`. `overlap_bound` and `kinetic_bound`
perform the block's own operations, in the block's own order, on the
absolute values of its operands. Rounding to nearest is monotone,
`|fl(xy)| = fl(|x||y|)` and `|fl(x + y)| <= fl(|x| + |y|)`, so by induction
the computed bound is >= the computed block element -- not merely the exact
one. `V`'s `B_t` is an analytic bound on `[t]^(0)`, so for `V` the statement
is the exact-arithmetic one; it is loose everywhere except at `t = 0,
C = P`, and the tests check it on the computed blocks.

THE POINT-CHARGE BOUND `B_t`. `[t]^(0)(x, theta)` is `d^t/dx^t` of
`(2/sqrt(pi)) theta F_0(theta^2 |x|^2)`, and `F_0(T) = INT_0^1 exp(-T u^2)
du`, so `[t]^(0) = (2/sqrt(pi)) theta INT_0^1 PROD_d d^{t_d} exp(-theta^2
u^2 x_d^2) du`. Write each 1D Gaussian as its Fourier integral, `exp(-a x^2)
= E[exp(i K x)]` with `K ~ N(0, 2a)`: then `|d^n exp(-a x^2)| <= E|K|^n =
(2a)^{n/2} m_n` for every `x`, `m_n = E|Z|^n` of a standard normal --
`(n-1)!!` for even `n` (attained at `x = 0`), and `<= sqrt((n-2)!! n!!)` for
odd `n` by Cauchy-Schwarz on the two neighbouring even moments. With
`a = theta^2 u^2` and `INT_0^1 u^{|t|} du = 1/(|t| + 1)`:

    |[t]^(0)(x, theta)| <= [0]^(0)(0, theta) (2 theta^2)^{|t|/2}
                           PROD_d m_{t_d} / (|t| + 1)        for every x

at `theta = sqrt(zeta)` (ao/potential.py). `[0]^(0)(0, theta)` is read from
the INJECTED evaluator, so this module forms no constant of the operator and
`pi` still appears in `dress` only. Every factor is proved; none is fitted.

THE PAIR LIST. `pair_list(basis, *, tau, bound)` takes the bound as a
CALLABLE `bound(sa, sb) -> float` -- the way `launch.py` takes `bound=` --
so one list serves all three matrices, and the question of whose bound it is
never arises. A shell pair `i <= j` is DROPPED when `bound < tau`; then every
element of its block is `< tau`. `ao.one_electron.matrix(..., pairs=)`
builds from the kept pairs only. The decay is `exp(-mu |A - B|^2)` through
`E^{00}_0`, a genuine distance screen rather than a Schwarz product, which is
why a one-electron build is `O(N)` work on an extended system -- and why a
compact molecule screens almost nothing (test/test_ao_pairs.py records it).

THE CONTRACTED SCHWARZ BOUND OF [Fock] (#285; [Fock, eq. 22, §3.2 decision
8]). The fused Fock kernel screens a CONTRACTED quartet `(ab|cd)` on
`G_ab G_cd`, and deletes a primitive pair of `|ab)` on its own `G_[ab]`. Both
are read off the bound above -- no second bound is introduced. For one
primitive pair `pp` of `(sa, sb)` (its dressed block, `primitive_pairs`):

    G_[ab](pp) = max_k SUM_{p in hermite_order(la + lb)} |block_pp[k][p]| I_p(zeta_pp)

with `I_p = hermite_bound(zeta_pp, p)`, the max over the Cartesian pairs `k`.
THE PROOF that it bounds: an element of the contracted block is (ao/eri.py)

    (ab|cd)[k][l] = SUM_{pp, qq} SUM_{p, q} E_pp[k][p] [p|q] E_qq[l][q]

and `|[p|q]| <= I_p(zeta) I_q(eta)` (THE BOUND above), so

    |(ab|cd)[k][l]| <= SUM_{pp, qq} (SUM_p |E_pp[k][p]| I_p) (SUM_q |E_qq[l][q]| I_q)
                    <= SUM_{pp, qq} G_[ab](pp) G_[cd](qq)
                     = (SUM_pp G_[ab](pp)) (SUM_qq G_[cd](qq)).

The last line holds for ANY subsets of the primitive pairs summed on each
side: the same bound, restricted to the kept primitives, bounds the block
computed from the kept primitives only. That is why pruning (below) keeps
the proof -- the bound still bounds after pruning -- and the deleted
primitive pairs move an element by at most `(SUM_dropped G_[ab]) G_cd +
G_ab' (SUM_dropped G_[cd])`, `G_ab'` the kept sum. The statement is the
exact-arithmetic one, like `V`'s; the tests check it on computed blocks.
For an `s`-`s` primitive pair `G_[ab]` is `|block[0][0]| I_0`, the existing
`hermite_bound` times the dressed coefficient (a test pins it). `[Fock]`'s
eq. 22 defines `G_ab` as the exact `sqrt((ab|ab))`-type factor; this one is
an upper bound on it that needs no ERI, and is the bound `ao.k` and the
J engine already rest on.

`prune(bounds, *, eps)` is [Fock]'s primitive-level screen: the primitive
pairs ordered DESCENDING by `G_[ab]` (stably, by index on ties) and the
LONGEST TAIL whose `math.fsum` is `< eps` deleted -- so the kept set is a
PREFIX of the bound order, and the deleted sum is `< eps`: `fsum` is
correctly rounded and `eps` is a float, so a rounded sum below `eps` is an
exact one below it. Which `eps` makes a deletion insignificant is the
caller's (`launch_f.py`); `eps = 0` deletes nothing.

Plain tuples and floats, like `ao/shell.py`: `src/felizjueves` has no runtime
dependencies.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.mmd.hermite import e_axis, e_block, hermite_order

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


def dress(e: Sequence[float], *, ca: float, cb: float,
          zeta: float) -> tuple[float, ...]:
    """`c_a c_b (pi/zeta)^{3/2} E` -- the ONLY place either dressing (module
    docstring) multiplies into a Hermite coefficient. No normalisation: that
    is already in the coefficients."""
    if not zeta > 0.0:
        raise ValueError(f"zeta = {zeta!r} is not > 0")
    scale = ca * cb * (math.pi / zeta) ** 1.5
    return tuple(scale * x for x in e)


def primitive_pairs(sa: Shell, sb: Shell) -> tuple[PrimitivePair, ...]:
    """Every primitive pair of the ORDERED shell pair `(sa, sb)`, i outer."""
    A, B = sa.centre, sb.centre
    L = sa.l + sb.l
    carts = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
    out = []
    for i, (alpha, ca) in enumerate(zip(sa.exponents, sa.coefficients,
                                        strict=True)):
        for j, (beta, cb) in enumerate(zip(sb.exponents, sb.coefficients,
                                           strict=True)):
            zeta = alpha + beta
            P = tuple((alpha * A[d] + beta * B[d]) / zeta for d in range(3))
            block = tuple(
                dress(e_block(a, b, zeta, P, A, B),
                      ca=ca, cb=cb, zeta=zeta)
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


# ---------------------------------------------------------------------------
# The one-electron pair bound (#236; module docstring, "THE ONE-ELECTRON PAIR
# BOUND, PROVED")
# ---------------------------------------------------------------------------

Weight = Callable[[float], Sequence[float]]
Bound = Callable[[Shell, Shell], float]


def one_electron_bound(sa: Shell, sb: Shell, *, T: int = 0,
                       weight: Weight | None = None) -> float:
    """`SUM_pp max_{ab} SUM_{t in hermite_order(T)} |block_pp[ab][t]| w_t`.

    `block_pp` is `primitive_pairs(sa, sb)`'s dressed block -- the
    coefficients the integral is built from -- and `w_t = weight(zeta)[t]`
    (all 1 when `weight` is None). `T = 0` with no weight bounds `S`;
    `T = la + lb` with `point_charge_bounds` bounds `V` per unit charge
    (module docstring).
    """
    if type(T) is not int or not 0 <= T <= sa.l + sb.l:
        raise ValueError(f"T = {T!r} is not an int in [0, la + lb = "
                         f"{sa.l + sb.l}]")
    n = len(hermite_order(T))
    total = 0.0
    for pp in primitive_pairs(sa, sb):
        if weight is None:
            w: tuple[float, ...] = (1.0,) * n
        else:
            w = tuple(weight(pp.zeta))[:n]
            if len(w) != n or any(not x >= 0.0 for x in w):
                raise ValueError(
                    f"weight({pp.zeta!r}) does not give {n} weights >= 0")
        worst = 0.0
        for row in pp.block:
            v = 0.0
            for e, x in zip(row[:n], w, strict=True):
                v += abs(e) * x
            worst = max(worst, v)
        total += worst
    return total


def overlap_bound(sa: Shell, sb: Shell) -> float:
    """`>= max |overlap_block(sa, sb)|`, in floating point too (`T = 0`)."""
    return one_electron_bound(sa, sb, T=0)


def _double_factorial(n: int) -> int:
    """`n!!` for `n >= -1` (`(-1)!! = 0!! = 1`)."""
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


def abs_normal_moment_bound(n: int) -> float:
    """`>= E|Z|^n`, `Z` a standard normal: `(n-1)!!` (exact) for even `n`,
    `sqrt((n-2)!! n!!)` (Cauchy-Schwarz between `E Z^{n-1}` and `E Z^{n+1}`)
    for odd `n`."""
    if type(n) is not int or n < 0:
        raise ValueError(f"n = {n!r} is not an int >= 0")
    if n % 2 == 0:
        return float(_double_factorial(n - 1))
    return math.sqrt(_double_factorial(n - 2) * _double_factorial(n))


def point_charge_bounds(zeta: float, L: int, *,
                        one_centre: OneCentre) -> tuple[float, ...]:
    """`B_t >= |[t]^(0)(x, sqrt(zeta))|` for EVERY `x`, over `hermite_order(L)`.

    `B_t = [0]^(0)(0, theta) (2 theta^2)^{|t|/2} PROD_d m_{t_d} / (|t| + 1)`
    with `theta = sqrt(zeta)` and `[0]^(0)(0, theta)` from the injected
    evaluator (module docstring, "THE POINT-CHARGE BOUND").
    """
    if not zeta > 0.0:
        raise ValueError(f"zeta = {zeta!r} is not > 0")
    theta = math.sqrt(zeta)
    r0 = one_centre((0, 0, 0), 0, (0.0, 0.0, 0.0), theta)
    if not r0 > 0.0:
        raise ValueError(f"[0]^(0)(0, theta) = {r0!r} is not > 0: the "
                         "evaluator's convention is wrong")
    out = []
    for t in hermite_order(L):
        n = sum(t)
        m = math.prod(abs_normal_moment_bound(c) for c in t)
        out.append(r0 * (2.0 * zeta) ** (0.5 * n) * m / (n + 1))
    return tuple(out)


def potential_bound(sa: Shell, sb: Shell,
                    nuclei: Sequence[tuple[float, Vec]], *,
                    one_centre: OneCentre) -> float:
    """`>= max |potential_block(sa, sb, nuclei)|`: `SUM_C |Z_C|` times the
    `T = la + lb` bound weighted by `point_charge_bounds` (exact arithmetic;
    module docstring). Uniform in the nuclei's positions: `V` is long-ranged
    in `C`, and only the pair's own `exp(-mu |A - B|^2)` decays."""
    z = math.fsum(abs(float(n[0])) for n in nuclei)
    L = sa.l + sb.l
    per_charge = one_electron_bound(
        sa, sb, T=L,
        weight=lambda zeta: point_charge_bounds(zeta, L,
                                                one_centre=one_centre))
    return z * per_charge


def kinetic_bound(sa: Shell, sb: Shell) -> float:
    """`>= max |kinetic_block(sa, sb)|`, in floating point too: the block's
    own operations, in its order, on absolute values (module docstring)."""
    A, B = sa.centre, sb.centre
    la, lb = sa.l, sb.l
    carts = [(a, b) for a in cartesians(la) for b in cartesians(lb)]
    total = 0.0
    for alpha, ca in zip(sa.exponents, sa.coefficients, strict=True):
        for beta, cb in zip(sb.exponents, sb.coefficients, strict=True):
            zeta = alpha + beta
            P = tuple((alpha * A[d] + beta * B[d]) / zeta for d in range(3))
            S = [tuple(tuple(abs(e_axis(i, j, zeta, P[d] - A[d],
                                        P[d] - B[d])[0])
                             for j in range(lb + 3))
                       for i in range(la + 1))
                 for d in range(3)]
            bracket = []
            for a, b in carts:
                s = [S[d][a[d]][b[d]] for d in range(3)]
                k = []
                for d in range(3):
                    row, j = S[d][a[d]], b[d]
                    kd = (2.0 * beta * beta * row[j + 2]
                          + beta * (2 * j + 1) * row[j])
                    if j >= 2:
                        kd += 0.5 * j * (j - 1) * row[j - 2]
                    k.append(kd)
                bracket.append(k[0] * s[1] * s[2] + s[0] * k[1] * s[2]
                               + s[0] * s[1] * k[2])
            total += max(dress(bracket, ca=abs(ca), cb=abs(cb), zeta=zeta))
    return total


@dataclass(frozen=True)
class PairList:
    """The unordered shell pairs `(i, j)`, `i <= j`, of a basis, split by a
    bound at `tau`: `kept` (bound >= tau) and `dropped` (bound < tau), each
    in `ao.one_electron.matrix`'s row-major order (`i` outer)."""

    tau: float
    kept: tuple[tuple[int, int], ...]
    dropped: tuple[tuple[int, int], ...]

    @property
    def n_pairs(self) -> int:
        return len(self.kept) + len(self.dropped)

    @property
    def skip_fraction(self) -> float:
        return len(self.dropped) / self.n_pairs if self.n_pairs else 0.0


def _finite_non_negative(x: object) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and 0.0 <= x < math.inf)


def pair_list(basis: Basis, *, tau: float, bound: Bound) -> PairList:
    """Screen every shell pair `i <= j` of `basis`: dropped when
    `bound(sa, sb) < tau`, so every element of a dropped block is `< tau`.

    `bound` is injected (`overlap_bound`, `kinetic_bound`, or a closure over
    `potential_bound`'s nuclei and evaluator); a value that is not a finite
    number `>= 0` is refused, never clipped. `tau = 0` drops nothing.
    """
    if not isinstance(basis, Basis):
        raise ValueError(f"{basis!r} is not a Basis")
    if not _finite_non_negative(tau):
        raise ValueError(f"tau = {tau!r} is not a finite number >= 0")
    kept, dropped = [], []
    shells = basis.shells
    for i, sa in enumerate(shells):
        for j in range(i, len(shells)):
            b = bound(sa, shells[j])
            if not _finite_non_negative(b):
                raise ValueError(
                    f"bound(shell {i}, shell {j}) = {b!r} is not a finite "
                    "number >= 0")
            (kept if b >= tau else dropped).append((i, j))
    return PairList(float(tau), tuple(kept), tuple(dropped))


def cross_pair_list(basis_a: Basis, basis_b: Basis, *, tau: float,
                    bound: Bound) -> PairList:
    """Screen every ORDERED shell pair `(i in basis_a, j in basis_b)` --
    the cross-basis overlap's (`ao.one_electron.rectangular`) -- as
    `pair_list` screens `i <= j` of one basis: dropped when `bound(sa, sb)
    < tau`, in row-major order (`i` outer). `PairList`'s `i <= j` reading
    does not apply here; its two lists index the two bases."""
    for name, b in (("basis_a", basis_a), ("basis_b", basis_b)):
        if not isinstance(b, Basis):
            raise ValueError(f"{name} {b!r} is not a Basis")
    if not _finite_non_negative(tau):
        raise ValueError(f"tau = {tau!r} is not a finite number >= 0")
    kept, dropped = [], []
    for i, sa in enumerate(basis_a.shells):
        for j, sb in enumerate(basis_b.shells):
            b = bound(sa, sb)
            if not _finite_non_negative(b):
                raise ValueError(
                    f"bound(shell {i} of A, shell {j} of B) = {b!r} is not a "
                    "finite number >= 0")
            (kept if b >= tau else dropped).append((i, j))
    return PairList(float(tau), tuple(kept), tuple(dropped))


# ---------------------------------------------------------------------------
# The contracted Schwarz bound of [Fock] and primitive pruning (#285; module
# docstring, "THE CONTRACTED SCHWARZ BOUND OF [Fock]")
# ---------------------------------------------------------------------------


def primitive_schwarz_bound(pp: PrimitivePair, *,
                            one_centre: OneCentre) -> float:
    """`G_[ab](pp) = max_k SUM_p |block[k][p]| I_p(zeta)`: the per-primitive
    factor of the contracted Schwarz bound, from `hermite_bound`."""
    ips = hermite_bounds(pp.zeta, pp.L, one_centre=one_centre)
    return max(math.fsum(abs(e) * i for e, i in zip(row, ips, strict=True))
               for row in pp.block)


def primitive_schwarz_bounds(sa: Shell, sb: Shell, *,
                             one_centre: OneCentre) -> tuple[float, ...]:
    """`G_[ab]` for every primitive pair of `(sa, sb)`, in `primitive_pairs`
    order (i outer); their sum is the contracted `G_ab`."""
    return tuple(primitive_schwarz_bound(pp, one_centre=one_centre)
                 for pp in primitive_pairs(sa, sb))


def prune(bounds: Sequence[float], *, eps: float) -> tuple[int, ...]:
    """The indices KEPT, in descending-bound order (stable): every index
    except the longest tail of that order whose `math.fsum` is `< eps`.

    Refuses a bound or an `eps` that is not a finite number `>= 0`.
    """
    if not _finite_non_negative(eps):
        raise ValueError(f"eps = {eps!r} is not a finite number >= 0")
    for k, g in enumerate(bounds):
        if not _finite_non_negative(g):
            raise ValueError(f"bounds[{k}] = {g!r} is not a finite number >= 0")
    order = sorted(range(len(bounds)), key=lambda k: -bounds[k])
    cut = len(order)
    while cut and math.fsum(bounds[k] for k in order[cut - 1:]) < eps:
        cut -= 1
    return tuple(order[:cut])


def narrowed(values: Iterable[float]) -> tuple[float, ...]:
    """The host's FP32 packing of an operand array (#347's `kmixed-de`): each
    value rounded ONCE to the nearest float32 (ties to even, as C's
    `(float)x`), returned as the Python floats that are exactly those
    float32s -- a narrowed COPY, the fp64 values untouched. Flat, in the
    given order; a caller lays it out as the fp64 array it narrows. Refuses
    a value outside float32's finite range (it would overflow, not round)."""
    out = []
    for k, x in enumerate(values):
        x = float(x)
        try:
            f = struct.unpack("f", struct.pack("f", x))[0]
        except OverflowError:
            f = math.inf
        if math.isfinite(x) and not math.isfinite(f):
            raise ValueError(f"values[{k}] = {x!r} overflows float32")
        out.append(f)
    return tuple(out)
