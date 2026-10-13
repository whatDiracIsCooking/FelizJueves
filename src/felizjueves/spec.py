"""The one key every later subsystem joins on.

A `Spec` names one thing the generator could be asked to produce: which
operator, at which derivative order, for which angular-momentum class, building
which target, with which levers pulled away from the paper. Plan, cost model,
emitter and scoreboards will each key their records on it, so a record made by
one is findable by the others.

SEVERAL FIELDS HAD EXACTLY ONE LEGAL VALUE, AND THAT WAS THE POINT. The
operator was the Coulomb operator and the derivative order is 0. They are
fields anyway because adding a field later means retrofitting every consumer
that was keyed without it; widening an enum does not.

THE OPERATOR IS THE FIRST SUCH FIELD TO BE WIDENED (#232), and the sentence
above is what that tests. Milestone 11 adds the one-electron operators:
`OVERLAP` (S_ab = <a|b>), `KINETIC` (T_ab = <a|-1/2 nabla^2|b>) and `NUCLEAR`
(V_ab = -SUM_C Z_C <a|1/r_C|b>), each with its own target -- `Target.S`, `T`
and `V`, each a PAIR class (la, lb) like `DQ` and `JAB`. A target has
exactly one operator (`OPERATOR_OF`) and the validator refuses any other
pairing, so `Spec(COULOMB, 0, (1, 1), Target.S)` is never keyed. No consumer
outside this module changed: a one-electron key reads
``overlap/d0/(2,1)/b1/s/literal``, the shape every key already had, and every
pre-existing key is byte-identical. `DERIVATIVE_ORDERS` stays (0,):
gradients and Hessians are out of milestone 11's scope.

THE TARGET DECIDES THE CLASS'S SHAPE (#150). `Target.J` is the [p|q] kernel
of a QUARTET class (l1, l2 | l3, l4). `Target.DQ` (the pre-contracted ket
vector) and `Target.JAB` (the bra back-transform and the scatter) act on one
shell PAIR, so their `am` is a PAIR class (la, lb). That is a widening of
`am` and of the enum, not a new field: `AM_LENGTH` says which target takes
which length, and the validator refuses a mismatch either way.

A PAIR TARGET IS UNBATCHED, FOR A REASON STATED PER TARGET (#232). `DQ` and
`JAB` (and `S` and `T`, which run no recurrence tower at all) are never
batched for a structural reason: batching partitions the Hermite targets of
a quartet's [p|q] tower, and those kernels have none to partition. That
reason is FALSE for `V`, whose Hermite targets are every t with
|t| <= la + lb (the J engine with the ket a unit point charge), so `V` is
unbatched by milestone 11's DECISION, not by that rule; a batched `V` is a
follow-up. Either way `n_batches = 1` is a pair target's only legal value
today, and the refusal names which reason applies (`_PAIR_BATCH_REFUSAL`).

A K CLASS IS NAMED IN HERMITE ORDER, NOT AO ORDER (#196). `Target.K` is the
exchange matrix, K_ab = SUM_cd D_cd (ac|bd), on the same [p|q] recurrence:
its bra Hermite pair is (a, c) and its ket pair (b, d). So a K class's `am`
is `(la, lc, lb, ld)` -- the two Hermite pair orders, BRA FIRST -- and never
the AO quartet order (la, lb, lc, ld). The sum L is the same either way, so
the same Tables 1-3 row is indexed; the split (lp, lq) = (la + lc, lb + ld)
is not, and the plan depends on the split. A `Spec` built in AO order plans
the wrong kernel and still returns a plausible number. So named, a K class
maps onto the identical DAG, MILD tree, sieve, batching, schedule and cost as
the J class with the same `am`; only the digestion differs. It is a
widening, not a new field: `AM_LENGTH` takes 4, the key spells the quartet
`(la,lc|lb,ld)` with target slot `k`, and K admits batching exactly as
`Target.J` does.

HOW A SPEC MEETS THE [JEngine] TABLES, stated once so the next change honours
it rather than choosing its own: for this engine the Hermite class is the
total angular momentum, L = l1 + l2 + l3 + l4 (the [p|q] class has
lp = l1 + l2 and lq = l3 + l4, and L = lp + lq). A Spec's four shells index a
table row through `total_am`, their sum, and through nothing else.

THE BATCH COUNT IS PART OF THE KEY. A kernel is emitted per class AND per
batch, and [JEngine] Tables 1-3 are indexed by (L, batches), so `n_batches`
changes the artifact; it is a field, not a loose call argument. It is 1, or a
positive multiple of 3 -- finer splits break the three-way symmetry
([JEngine, v2, §5]). `is_batch_count` is that rule, stated once; for a
pair target the count is 1 and only 1 (above).

Values only, plus two derived reads: `total_am` above, and `key`, a string
form of the key for file names and JSON, where a dataclass cannot go.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Operator(StrEnum):
    COULOMB = "coulomb"  # 1/r12
    # The one-electron operators (#232, milestone 11).
    OVERLAP = "overlap"  # 1: S_ab = <a|b>
    KINETIC = "kinetic"  # -1/2 nabla^2: T_ab
    NUCLEAR = "nuclear"  # -SUM_C Z_C / r_C: V_ab


class Target(StrEnum):
    """The MATRIX being built -- not the hardware.

    The card a kernel is compiled for is a `felizjueves.device.Arch`, which is
    deliberately not part of a `Spec`; `device.py` says why and what keys a
    measurement instead.
    """

    J = "j"  # the Coulomb matrix, [JEngine] Algorithm 1
    DQ = "dq"  # the pre-contracted ket vector D_q of one shell pair
    JAB = "jab"  # the bra back-transform of J_p and its scatter into J_ab
    # The exchange matrix K_ab = SUM_cd D_cd (ac|bd); its `am` is
    # (la, lc, lb, ld), the Hermite pairing, bra first (#196, module docstring).
    K = "k"
    # The one-electron matrices (#232), each a PAIR class (la, lb).
    S = "s"  # overlap
    T = "t"  # kinetic energy
    V = "v"  # nuclear attraction


# How many shells a target's class names: a quartet (l1, l2 | l3, l4) for the
# [p|q] kernel, a pair (la, lb) for the two per-pair families (#150), and a
# quartet (la, lc | lb, ld) in Hermite order for the exchange matrix (#196),
# and a pair (la, lb) for each one-electron matrix (#232).
AM_LENGTH: dict[Target, int] = {
    Target.J: 4, Target.DQ: 2, Target.JAB: 2, Target.K: 4,
    Target.S: 2, Target.T: 2, Target.V: 2,
}


# The one operator each target is built from (#232); any other pairing is
# refused. The four two-electron targets are Coulomb's.
OPERATOR_OF: dict[Target, Operator] = {
    Target.J: Operator.COULOMB, Target.DQ: Operator.COULOMB,
    Target.JAB: Operator.COULOMB, Target.K: Operator.COULOMB,
    Target.S: Operator.OVERLAP, Target.T: Operator.KINETIC,
    Target.V: Operator.NUCLEAR,
}


# Why a pair target refuses `n_batches > 1`, per target (#232, module
# docstring): a structural reason for the targets with no Hermite targets to
# partition, and a decision for `V`, of which the structural reason is false.
_STRUCTURAL = (
    "never batched -- batching partitions the Hermite targets of a quartet, "
    "and this pair kernel has none to partition"
)
_PAIR_BATCH_REFUSAL: dict[Target, str] = {
    Target.DQ: _STRUCTURAL,
    Target.JAB: _STRUCTURAL,
    Target.S: _STRUCTURAL,
    Target.T: _STRUCTURAL,
    Target.V: (
        "unbatched by milestone 11's decision -- V does have Hermite targets "
        "(every t with |t| <= la + lb), so the structural reason does not "
        "apply; a batched V kernel is a follow-up"
    ),
}


# Geometric derivatives are a later operator family; order 0 only for now.
DERIVATIVE_ORDERS = (0,)


def is_batch_count(n: object) -> bool:
    """1, or a positive multiple of 3 ([JEngine, v2, §5])."""
    return type(n) is int and (n == 1 or (n > 0 and n % 3 == 0))


@dataclass(frozen=True)
class Spec:
    operator: Operator
    derivative_order: int
    # (l1, l2 | l3, l4) for `Target.J`: the angular momentum of each of the
    # four shells; (la, lc | lb, ld) for `Target.K` -- the Hermite pairing,
    # bra first, NOT the AO order (la, lb, lc, ld); (la, lb) for a pair
    # target (`AM_LENGTH`).
    am: tuple[int, int, int, int] | tuple[int, int]
    target: Target
    # Named departures from the paper, sorted and unique. () is the literal
    # [JEngine] reconstruction.
    levers: tuple[str, ...] = ()
    # 1, or a multiple of 3: finer splits break the three-way symmetry
    # ([JEngine, v2, §5]).
    n_batches: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.operator, Operator):
            raise ValueError(f"operator {self.operator!r} is not an Operator")
        if not isinstance(self.target, Target):
            raise ValueError(f"target {self.target!r} is not a Target")
        if self.operator is not OPERATOR_OF[self.target]:
            raise ValueError(
                f"target {self.target} is built from operator "
                f"{OPERATOR_OF[self.target]}, not {self.operator}"
            )
        if (
            type(self.derivative_order) is not int
            or self.derivative_order not in DERIVATIVE_ORDERS
        ):
            raise ValueError(
                f"derivative order {self.derivative_order!r} not in "
                f"{DERIVATIVE_ORDERS}"
            )
        if len(self.am) not in (2, 4) or not all(
            type(x) is int and x >= 0 for x in self.am
        ):
            raise ValueError(f"am {self.am!r} is not two or four non-negative ints")
        want = AM_LENGTH[self.target]
        if len(self.am) != want:
            shape = "pair" if want == 2 else "quartet"
            hermite = (
                " in Hermite order (la, lc | lb, ld)"
                if self.target is Target.K else ""
            )
            raise ValueError(
                f"target {self.target} takes a {shape} class ({want} shells)"
                f"{hermite}, "
                f"not am {self.am!r}"
            )
        if not all(isinstance(lever, str) and lever for lever in self.levers):
            raise ValueError(f"levers {self.levers!r} must be non-empty strings")
        if tuple(sorted(set(self.levers))) != self.levers:
            raise ValueError(f"levers {self.levers!r} must be sorted and unique")
        if not is_batch_count(self.n_batches):
            raise ValueError(
                f"n_batches {self.n_batches!r} is neither 1 nor a positive "
                "multiple of 3"
            )
        if want == 2 and self.n_batches != 1:
            raise ValueError(
                f"n_batches {self.n_batches!r} for pair target {self.target}: "
                f"{_PAIR_BATCH_REFUSAL[self.target]}"
            )

    @property
    def total_am(self) -> int:
        """The sum of `am`, a quartet's or a pair's alike.

        For a quartet (`Target.J`) this is L, the row the Spec indexes in
        [JEngine] Tables 1-3. A PAIR class (`Target.DQ`, `Target.JAB`,
        and the one-electron `Target.S`, `T`, `V`) indexes
        NO published table row: its sum is la + lb, one pair's Hermite
        order, and must never be scored against a [JEngine] table.

        A K quartet (`Target.K`, `am = (la, lc, lb, ld)`) indexes the SAME
        Tables 1-3 row as the J quartet with the same `am`: L is the sum
        either way. The pairing is what differs -- K pairs (a, c) and
        (b, d), so the AO order (la, lb, lc, ld) has the same L but a
        different (lp, lq) split whenever lb != lc, and the sum alone cannot
        catch that mistake.
        """
        return sum(self.am)

    def key(self) -> str:
        """E.g. ``coulomb/d0/(4,4|4,4)/b1/j/literal``.

        A pair class spells ``coulomb/d0/(2,1)/b1/dq/literal`` -- no ``|``,
        so it cannot collide with a quartet's; a one-electron class spells
        ``overlap/d0/(2,1)/b1/s/literal``. A quartet key is
        byte-identical to the one before #150: it is the join key of every
        measured row.
        """
        if len(self.am) == 4:
            l1, l2, l3, l4 = self.am
            cls = f"({l1},{l2}|{l3},{l4})"
        else:
            la, lb = self.am
            cls = f"({la},{lb})"
        levers = "+".join(self.levers) or "literal"
        return (
            f"{self.operator}/d{self.derivative_order}/{cls}/"
            f"b{self.n_batches}/{self.target}/{levers}"
        )
