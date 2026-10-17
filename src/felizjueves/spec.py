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

AN F CLASS IS NAMED IN AO ORDER, NOT HERMITE ORDER (#272). `Target.F` is the
FUSED Fock build of [Fock] §3 on the Head-Gordon-Pople recurrence: one kernel
digests each contracted quartet (ab|cd) into both J and K. The HGP VRR pairs
the bra (a, b) and the ket (c, d) AS WRITTEN, so an F class's `am` is the AO
quartet order `(la, lb, lc, ld)` and the key spells `(la,lb|lc,ld)` with
target slot `f`. That is the OPPOSITE convention to `Target.K` above: the tree
now holds two quartet targets with two orderings, and the mistake is silent in
both directions -- a K class spelled in AO order, or an F class spelled in
Hermite order, is a legal `Spec` naming a different class. The same four
shells therefore give different specs and different keys under `F` and `K`.
`Target.F` is a widening of the enum, not a `scheme` field (milestone 12, pin
4): every pre-existing key is byte-identical. The trigger that would force a
`scheme` field is named: the first fused MMD `F`, or the first HGP `J`/`K`.

A SCHWARZ CLASS IS A PAIR THAT RUNS A QUARTET. `Target.SCHWARZ` is the
matrix `sqrt((mn|mn))` over every AO pair `(m, n)`, the Schwarz factor of
`|(mn|rs)| <= sqrt((mn|mn)) sqrt((rs|rs))`. Its class is the shell pair
`(la, lb)` -- one kernel per pair class, `am` a pair, unbatched by decision
(`_PAIR_BATCH_REFUSAL`) -- but the recurrence it runs is the quartet
`(la lb|la lb)`'s, at L = 2 (la + lb). So `total_am` is la + lb and indexes no
table row, and the plan is asked for the `Target.J` quartet, never for this
key (`kernel_schwarz.schwarz_plan`).

A CROSS-OVERLAP CLASS IS AN ORDERED PAIR ACROSS TWO BASES. `Target.SX` is
the rectangular overlap `S_{B1,B2}[m, n] = <m|n>` with `m` an AO of one
basis and `n` of another (the same molecule in two basis sets, or two
molecules). Its class `(la, lb)` is ORDERED -- `la` the row basis's shell,
`lb` the column basis's -- because no symmetry relates `(la, lb)` to
`(lb, la)` across two bases; its operator is `OVERLAP`, as `Target.S`'s,
and it is a separate target because the MATRIX differs (rectangular, no
mirror, no diagonal rule), not the integral (kernel_sx.py). Unbatched for
`S`'s structural reason (`_PAIR_BATCH_REFUSAL`).

A METRIC CLASS IS A PAIR OF SINGLE-CENTER SHELLS (#410). `Target.METRIC` is
the two-center Coulomb metric `(P|Q) = INT INT P(r1) Q(r2) / r12` of an
auxiliary (density-fitting) basis, Cartesian. Its class is the pair of aux
shells `(lP, lQ)` -- `am` a pair, operator `COULOMB`, key
``coulomb/d0/(lP,lQ)/b1/metric/literal`` -- and the recurrence it runs is the
`Target.J` quartet `(lP 0|lQ 0)`'s at L = lP + lQ: an aux primitive is a
Hermite "pair" with `zeta = alpha`, `P = A` and `X_PA = 0`, whose `E` is
`e_fragment.center_fragment`. So `total_am` is lP + lQ, the L of that
quartet, but the key indexes no table row; the plan is asked for the quartet,
never for this key. Unbatched by DECISION, like `V` and `SCHWARZ`
(`_PAIR_BATCH_REFUSAL`): it has Hermite targets that could be partitioned.

AN F CLASS IS UNBATCHED, FOR ITS OWN REASON (#272). It is not the pair rule:
a pair target is unbatched because a pair has no Hermite targets to
partition, whereas an HGP class has no Hermite targets AT ALL, batched or not
-- its recurrence is the VRR/HRR, not the [p|q] tower, so [JEngine]'s
partition does not apply to it. `n_batches = 1` is its only legal value
today. The field is a widening path, not a dead end: its future meaning for
`F` is VRR target grouping (`_F_BATCH_REFUSAL`). For the same reason the plan
layer, which is MMD machinery keyed on `total_am`, refuses an F spec
(`plan.dag.refuse_hgp`).

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

A KERNEL-BODY LEVER IS LEGAL PER TARGET (#337). `levers` holds names from
several layers: the plan's (`no-sieve`, `tuned`), lowering's (`recycle`)
and, since milestone 13, the kernel builders' own (`levers.VOCABULARY`). The
last kind is checked HERE, against each lever's legality table, and an
illegal `(target, lever)` is refused with the table's stated reason; a name
outside the vocabulary passes through to its own layer, as before. The key
spells every lever alike, so every pre-existing key is byte-identical.

Values only, plus two derived reads: `total_am` above, and `key`, a string
form of the key for file names and JSON, where a dataclass cannot go.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from felizjueves import levers as _levers


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
    # The fused Fock build of [Fock] §3 on the HGP recurrence; its `am` is
    # (la, lb, lc, ld), the AO order -- NOT K's Hermite pairing (#272,
    # module docstring).
    F = "f"
    # The Schwarz factor sqrt((mn|mn)) of every AO pair, a PAIR class
    # (la, lb) whose kernel runs the quartet (la lb|la lb)'s recurrence
    # (kernel_schwarz.py, module docstring "A SCHWARZ CLASS").
    SCHWARZ = "schwarz"
    # The rectangular overlap S_{B1,B2} between two bases, an ORDERED pair
    # class (la in B1, lb in B2) (kernel_sx.py; module docstring "A
    # CROSS-OVERLAP CLASS").
    SX = "sx"
    # The two-center Coulomb metric (P|Q) of an auxiliary basis, a PAIR class
    # (lP, lQ) of single-center shells whose recurrence is the quartet
    # (lP 0|lQ 0)'s (#410; module docstring "A METRIC CLASS").
    METRIC = "metric"


# How many shells a target's class names: a quartet (l1, l2 | l3, l4) for the
# [p|q] kernel, a pair (la, lb) for the two per-pair families (#150), and a
# quartet (la, lc | lb, ld) in Hermite order for the exchange matrix (#196),
# and a pair (la, lb) for each one-electron matrix (#232), and a quartet
# (la, lb | lc, ld) in AO order for the fused Fock build (#272), and a pair
# (lP, lQ) of auxiliary shells for the Coulomb metric (#410).
AM_LENGTH: dict[Target, int] = {
    Target.J: 4, Target.DQ: 2, Target.JAB: 2, Target.K: 4, Target.F: 4,
    Target.S: 2, Target.T: 2, Target.V: 2, Target.SCHWARZ: 2,
    Target.SX: 2, Target.METRIC: 2,
}


# The one operator each target is built from (#232); any other pairing is
# refused. The two-electron targets are Coulomb's, the metric among them.
OPERATOR_OF: dict[Target, Operator] = {
    Target.J: Operator.COULOMB, Target.DQ: Operator.COULOMB,
    Target.JAB: Operator.COULOMB, Target.K: Operator.COULOMB,
    Target.F: Operator.COULOMB, Target.SCHWARZ: Operator.COULOMB,
    Target.S: Operator.OVERLAP, Target.T: Operator.KINETIC,
    Target.V: Operator.NUCLEAR, Target.SX: Operator.OVERLAP,
    Target.METRIC: Operator.COULOMB,
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
    Target.SX: _STRUCTURAL,
    Target.T: _STRUCTURAL,
    Target.V: (
        "unbatched by milestone 11's decision -- V does have Hermite targets "
        "(every t with |t| <= la + lb), so the structural reason does not "
        "apply; a batched V kernel is a follow-up"
    ),
    Target.SCHWARZ: (
        "unbatched by decision -- the class runs the quartet (la lb|la lb)'s "
        "recurrence, whose Hermite targets could be partitioned, but the "
        "kernel is one thread per shell pair and a batched Schwarz kernel "
        "is a follow-up nobody has declared"
    ),
    Target.METRIC: (
        "unbatched by decision -- the metric (P|Q) runs the quartet "
        "(lP 0|lQ 0)'s recurrence, whose Hermite targets (every t with "
        "|t| <= lP + lQ) could be partitioned, so the structural reason does "
        "not apply; a batched metric kernel is a follow-up nobody has "
        "declared"
    ),
}


# Why `Target.F` refuses `n_batches > 1` (#272, module docstring): its own
# reason, not the pair one.
_F_BATCH_REFUSAL = (
    "an HGP class is never batched -- it has no Hermite targets at all, "
    "batched or not, so [JEngine]'s partition does not apply; the field's "
    "future meaning for F is VRR target grouping, not yet built"
)


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
    # bra first, NOT the AO order (la, lb, lc, ld); (la, lb | lc, ld) for
    # `Target.F` -- the AO order, NOT K's pairing; (la, lb) for a pair
    # target (`AM_LENGTH`).
    am: tuple[int, int, int, int] | tuple[int, int]
    target: Target
    # Named departures from the paper, sorted and unique. () is the literal
    # [JEngine] reconstruction. A kernel-body lever (`levers.VOCABULARY`)
    # must be legal on `target` (#337).
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
                if self.target is Target.K
                else " in AO order (la, lb | lc, ld)"
                if self.target is Target.F else ""
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
        for lever in self.levers:
            why = _levers.refusal(self.target, lever)
            if why is not None:
                raise ValueError(
                    f"lever {lever!r} on target {self.target}: {why}")
        if not is_batch_count(self.n_batches):
            raise ValueError(
                f"n_batches {self.n_batches!r} is neither 1 nor a positive "
                "multiple of 3"
            )
        if self.target is Target.F and self.n_batches != 1:
            raise ValueError(
                f"n_batches {self.n_batches!r} for target {self.target}: "
                f"{_F_BATCH_REFUSAL}"
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

        An F quartet (`Target.F`, `am = (la, lb, lc, ld)`, the AO order)
        indexes NO [JEngine] table row: an HGP class has no Hermite targets,
        so its sum names no Tables 1-3 row, and it must never be scored
        against one. The plan layer, which keys on this sum, refuses an F
        spec (`plan.dag.refuse_hgp`).

        A METRIC pair (`Target.METRIC`, `am = (lP, lQ)`) sums to lP + lQ,
        the L of the `Target.J` quartet `(lP 0|lQ 0)` whose recurrence it
        runs; the pair key itself still indexes NO table row.
        """
        return sum(self.am)

    def key(self) -> str:
        """E.g. ``coulomb/d0/(4,4|4,4)/b1/j/literal``.

        A pair class spells ``coulomb/d0/(2,1)/b1/dq/literal`` -- no ``|``,
        so it cannot collide with a quartet's; a one-electron class spells
        ``overlap/d0/(2,1)/b1/s/literal``. A quartet key is
        byte-identical to the one before #150: it is the join key of every
        measured row. An F class spells ``coulomb/d0/(2,1|1,0)/b1/f/literal``
        -- the AO order, target slot ``f`` (#272).
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

    def kernel_body_levers(self) -> tuple[str, ...]:
        """The levers a kernel BUILDER reads (`levers.VOCABULARY`, #337), in
        key order; the rest belong to the plan or to lowering."""
        return _levers.split(self.levers)[0]

    def without_kernel_body_levers(self) -> Spec:
        """This spec as the plan layer must see it: its kernel-body levers
        removed (the plan refuses any lever it does not own), every other
        field kept. `self` itself when it carries none (#337)."""
        body, rest = _levers.split(self.levers)
        if not body:
            return self
        return Spec(self.operator, self.derivative_order, self.am,
                    self.target, rest, self.n_batches)


def lever_provenance(name: str) -> str:
    """The provenance line a builder records for kernel-body lever `name`
    (`levers.provenance`, re-exported so a builder reads the vocabulary
    through the key it already depends on)."""
    return _levers.provenance(name)
