"""The one key every later subsystem joins on.

A `Spec` names one thing the generator could be asked to produce: which
operator, at which derivative order, for which angular-momentum class, building
which target, with which levers pulled away from the paper. Plan, cost model,
emitter and scoreboards will each key their records on it, so a record made by
one is findable by the others.

SEVERAL FIELDS HAVE EXACTLY ONE LEGAL VALUE TODAY, AND THAT IS THE POINT. The
operator is the Coulomb operator, the derivative order is 0 and the target is
the J matrix. They are fields anyway because adding a field later means
retrofitting every consumer that was keyed without it; widening an enum does
not.

HOW A SPEC MEETS THE [JEngine] TABLES, stated once so the next change honours
it rather than choosing its own: for this engine the Hermite class is the
total angular momentum, L = l1 + l2 + l3 + l4 (the [p|q] class has
lp = l1 + l2 and lq = l3 + l4, and L = lp + lq). A Spec's four shells index a
table row through `total_am`, their sum, and through nothing else.

THE BATCH COUNT IS PART OF THE KEY. A kernel is emitted per class AND per
batch, and [JEngine] Tables 1-3 are indexed by (L, batches), so `n_batches`
changes the artifact; it is a field, not a loose call argument. It is 1, or a
positive multiple of 3 -- finer splits break the three-way symmetry
([JEngine, v2, §5]). `is_batch_count` is that rule, stated once.

Values only, plus two derived reads: `total_am` above, and `key`, a string
form of the key for file names and JSON, where a dataclass cannot go.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Operator(StrEnum):
    COULOMB = "coulomb"  # 1/r12


class Target(StrEnum):
    J = "j"  # the Coulomb matrix, [JEngine] Algorithm 1


# Geometric derivatives are a later operator family; order 0 only for now.
DERIVATIVE_ORDERS = (0,)


def is_batch_count(n: object) -> bool:
    """1, or a positive multiple of 3 ([JEngine, v2, §5])."""
    return type(n) is int and (n == 1 or (n > 0 and n % 3 == 0))


@dataclass(frozen=True)
class Spec:
    operator: Operator
    derivative_order: int
    # (l1, l2 | l3, l4): the angular momentum of each of the four shells.
    am: tuple[int, int, int, int]
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
        if (
            type(self.derivative_order) is not int
            or self.derivative_order not in DERIVATIVE_ORDERS
        ):
            raise ValueError(
                f"derivative order {self.derivative_order!r} not in "
                f"{DERIVATIVE_ORDERS}"
            )
        if len(self.am) != 4 or not all(
            type(x) is int and x >= 0 for x in self.am
        ):
            raise ValueError(f"am {self.am!r} is not four non-negative ints")
        if not all(isinstance(lever, str) and lever for lever in self.levers):
            raise ValueError(f"levers {self.levers!r} must be non-empty strings")
        if tuple(sorted(set(self.levers))) != self.levers:
            raise ValueError(f"levers {self.levers!r} must be sorted and unique")
        if not is_batch_count(self.n_batches):
            raise ValueError(
                f"n_batches {self.n_batches!r} is neither 1 nor a positive "
                "multiple of 3"
            )

    @property
    def total_am(self) -> int:
        """L, the row a Spec indexes in [JEngine] Tables 1-3."""
        return sum(self.am)

    def key(self) -> str:
        """E.g. ``coulomb/d0/(4,4|4,4)/b1/j/literal``."""
        l1, l2, l3, l4 = self.am
        levers = "+".join(self.levers) or "literal"
        return (
            f"{self.operator}/d{self.derivative_order}/({l1},{l2}|{l3},{l4})/"
            f"b{self.n_batches}/{self.target}/{levers}"
        )
