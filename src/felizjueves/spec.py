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
pre-existing key is byte-identical. `DERIVATIVE_ORDERS` stayed (0,):
gradients were out of milestone 11's scope (since widened, below).

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
4): every pre-existing key is byte-identical. Pin 4 named the trigger that
would force a second axis: the first fused MMD `F`, or the first HGP
`J`/`K`. It fired with milestone 14's HGP `J`/`K` and is RESOLVED, by an
`engine` field rather than a `scheme` field, because `Target` already
names the scheme (below). The first fused MMD `F` (#326) needs no further
field: it is `Spec(Target.F, ..., engine=Engine.MMD)`, refused until it is
built.

THE RECURRENCE IS A FIELD, `engine` (#401). That trigger has come: milestone
14 builds the first HGP `J`/`K`. [Fock] confounds the recurrence with the
digestion (§1, §4.3); `Target` already names the digestion (`J`/`K` split,
`F` fused), so the axis a `Spec` could not spell was the RECURRENCE --
`Engine.MMD` (McMurchie-Davidson, [JEngine]) or `Engine.HGP`
(Head-Gordon-Pople, [Fock] §3.1). Every target has a NATIVE engine
(`NATIVE_ENGINE`: `F` -> HGP, every other target -> MMD), and
`Spec(..., engine=None)`, the default, resolves to it, so every existing
constructor builds the spec it always did. The key spells the engine ONLY
when it is not the native one, as a suffix of the target slot
(`Spec.target_slot`): ``coulomb/d0/(1,1|1,0)/b1/j.hgp/literal``,
``.../k.hgp/...``, and (#326, not yet legal) ``.../f.mmd/...``. Still the
same six segments; `.` and not `@`, which separates the arch in
`MeasuredKey`. So every pre-existing key is byte-identical and no stored key
carries a suffix. A non-native engine is legal only where it is being built:
HGP on `J` and `K`, on the one-electron `S`, `T`, `V` and `DIPOLE`
(`one_electron/kernel_hgp.py`), and on the two DF targets `METRIC` and `ERI3`
(`eri/density_fit/kernel_hgp.py`, s and p shells tested). Every other
non-native pairing is refused with its reason (`_ENGINE_REFUSAL`), MMD `F`
among them until #326.

`am` IS THE PAIRING THE RECURRENCE USES -- one rule, under either engine. A
J class is `(la, lb | lc, ld)`: its ERI is (ab|cd), bra (a, b), under MMD
and HGP alike. A K class is `(la, lc | lb, ld)` under BOTH engines: its ERI
is (ac|bd), so the HGP bra pair is (a, c) exactly as MMD's Hermite bra pair
is -- an HGP `K` is NOT spelled in `F`'s AO order. An F class is
`(la, lb | lc, ld)`: the fused build digests (ab|cd) itself.

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

AN ERI3 CLASS IS A TRIPLE THAT RUNS A QUARTET (#441). `Target.ERI3` is the
three-center Coulomb integral `(mn|P) = INT INT m(r1) n(r1) P(r2) / r12` of
density fitting: an AO shell pair `(m, n)` on the bra and ONE auxiliary
(single-center) shell `P` on the ket. Its class is the TRIPLE
`(la, lb, lP)` -- the first target whose class names three shells, `am` of
length 3, operator `COULOMB`, key ``coulomb/d0/(la,lb|lP)/b1/eri3/literal``
-- and the recurrence it runs is the `Target.J` quartet `(la lb|lP 0)`'s at
L = la + lb + lP: the bra is an ordinary Hermite pair, the ket an aux
primitive as in `Target.METRIC` (`zeta = alpha`, `P = A`, `X_PA = 0`). The
triple is NOT an AO quartet with a shell dropped: its third shell is an aux
shell of another basis, not an AO, and no fourth shell was ever there to
drop. So `total_am` is la + lb + lP, the L of that quartet, but the key
indexes no [JEngine] table row; the plan is asked for the quartet, never for
this key. Unbatched by DECISION, like `METRIC` (`_PAIR_BATCH_REFUSAL`): its
Hermite targets (every `t` with `|t| <= la + lb + lP`) could be partitioned,
and a batched three-center kernel is a follow-up nobody has filed.

A DIPOLE CLASS IS A ONE-ELECTRON PAIR; ITS ORDER IS NOT IN THE KEY (#518).
`Target.DIPOLE` is the matrix of Cartesian dipole moment integrals
`mu^d_ab = <a| q (r_d - C_d) |b>`, d in {x, y, z}, of operator
`Operator.DIPOLE` -- the multipole moments of order `e = 1` about a point
`C`, scaled by a charge `q` (both runtime inputs, not key fields). Its class
is the shell pair `(la, lb)`, `am` a pair, key
``dipole/d0/(la,lb)/b1/dipole/literal``, and one kernel writes all three
components. The integrals are McMurchie-Davidson's `E` (`dress`'s
`sqrt(pi/zeta)` aside) contracted with the Hermite moments
`M^e_t = INT x_C^e Lambda_t dx`, which obey
`M^{e+1}_t = t M^e_{t-1} + X_PC M^e_t + M^e_{t+1} / (2 zeta)`
(`e_fragment.multipole_fragment`, #519): Helgaker, Jorgensen & Olsen,
*Molecular Electronic-Structure Theory*, §9.5 (docs/references.md entry
48). The equation number inside §9.5 was not checked against the text, and
entries 39 / 46 (McMurchie & Davidson) were not found to state the moments.
The ORDER `e` IS A BUILDER ARGUMENT, NOT A KEY FIELD: a quadrupole would be
its own `Target.QUADRUPOLE` sharing the builder, so `Spec` gains no field
and no stored key moves. Unbatched for `S`'s structural reason
(`_PAIR_BATCH_REFUSAL`): one thread per record over an E fragment, no [p|q]
tower, so no Hermite targets of a quartet to partition. HGP is legal, as
for `S` (below).

AN HGP ONE-ELECTRON CLASS. `Target.S`, `T`, `V` and `DIPOLE` are legal on
`Engine.HGP` too: `one_electron.kernel_hgp` builds each on Head-Gordon &
Pople's split of Obara & Saika's one-electron recurrences (a VRR on the bra
centre per primitive pair -- `hgp.vrr`'s eq 25 at `f = 0`, its point-charge
limit for `V` -- the contraction, then `hgp.hrr`'s eq 28 on contracted
values). The key spells the engine as a slot suffix,
``overlap/d0/(1,0)/b1/s.hgp/literal``, so no MMD key moves; `am` is the same
pair `(la, lb)`, unbatched for the same reason as the MMD class's (an HGP
class has no Hermite targets at all). The builder takes s and p shells only
(`kernel_hgp.L_MAX`); `SX` stays refused (`_NO_HGP_ONE_E`).

AN F CLASS IS UNBATCHED, FOR ITS OWN REASON (#272). It is not the pair rule:
a pair target is unbatched because a pair has no Hermite targets to
partition, whereas an HGP class has no Hermite targets AT ALL, batched or not
-- its recurrence is the VRR/HRR, not the [p|q] tower, so [JEngine]'s
partition does not apply to it. `n_batches = 1` is its only legal value
today. The field is a widening path, not a dead end: its future meaning for
`F` is VRR target grouping (`_F_BATCH_REFUSAL`). For the same reason the plan
layer, which is MMD machinery keyed on `total_am`, refuses an F spec
(`plan.dag.refuse_hgp`).

A GRADIENT CLASS IS ITS INTEGRAL'S CLASS AT `derivative_order = 1`. The
field reserved above is the second to be widened: `DERIVATIVE_ORDERS` is
`(0, 1)`, and order 1 is legal on `GRADIENT_TARGETS` alone -- `S`, `T`,
`V`, `JAB` and `K`, each on MMD -- every other target refusing it
(`_GRADIENT_REFUSAL`). A d1 kernel does not write the derivative integrals
(a `[n_atom, 3, n_ao, n_ao]` tensor); it writes their CONTRACTION with a
matrix `M`, `G[atom, k] += SUM_mn M_mn dX_mn/dR_{atom,k}`, which is what an
energy gradient needs (`felizjueves.gradient`'s package docstring). `am` is
the integral's pair `(la, lb)`, unchanged, so
``overlap/d1/(2,1)/b1/s/literal`` and ``overlap/d0/(2,1)/b1/s/literal``
differ in one segment and every d0 key is byte-identical. `JAB` at d1 is
the Coulomb energy's bra derivative: it reads `J_p` of a `Target.J` class
whose bra order is one higher (`gradient.kernel_gjab.gjab_pq_spec`), a d0
key. `K` at d1 is the exchange energy's bra derivative, its `am` K's
Hermite pairing `(la, lc, lb, ld)`; it runs the plan of the d0 K class with
the bra raised by one (`gradient.kernel_gk.gk_pq_spec`), and its builder
refuses `n_batches != 1`. Unbatched, for each target's d0 reason.

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

A KERNEL-BODY LEVER IS LEGAL PER TARGET SLOT (#337, #401). `levers` holds
names from several layers: the plan's (`no-sieve`, `tuned`), lowering's
(`recycle`) and, since milestone 13, the kernel builders' own
(`levers.VOCABULARY`). The last kind is checked HERE, against each lever's
legality table, and an illegal `(slot, lever)` is refused with the table's
stated reason; a name outside the vocabulary passes through to its own
layer, as before. The table is keyed by `target_slot`, not by the target
alone, because `j` and `k` now exist under both engines and a lever written
for one engine's kernel has nothing to act on in the other's: `ordered` is
legal on `j.hgp` / `k.hgp` only. The key spells every lever alike, so every
pre-existing key is byte-identical.

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
    # The dipole moment operator (#518, milestone 19).
    DIPOLE = "dipole"  # q (r - C): mu_ab = <a|q (r - C)|b>, e = 1


class Target(StrEnum):
    """The MATRIX being built -- not the hardware.

    The card a kernel is compiled for is a
    `felizjueves.performance.device.Arch`, which is deliberately not part of
    a `Spec`; `device.py` says why and what keys a measurement instead.
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
    # The three-center DF integral (mn|P), a TRIPLE class (la, lb, lP): an AO
    # pair on the bra, one aux shell on the ket, whose recurrence is the
    # quartet (la lb|lP 0)'s (#441; module docstring "AN ERI3 CLASS").
    ERI3 = "eri3"
    # The record bound I = ao.pairs.pair_bound(zeta, la + lb) of each
    # primitive-pair record, the [p|q] kernel's Ip / Iq, a PAIR class (la, lb)
    # (kernel_pair_bound.py, module docstring "THE CLASS IS A PAIR").
    PAIR_BOUND = "pair_bound"
    # K's DRESSED record bound, eri.k_exchange.k.record_bounds: max over the first
    # shell's Cartesians of SUM_{k, p} |s E^{a_i b_k}_p| I_p(zeta), an
    # ORDERED pair class (la, lb) (kernel_record_bound.py, module docstring
    # "THE CLASS IS AN ORDERED PAIR").
    RECORD_BOUND = "record_bound"
    # The Cartesian dipole moment integrals <a| q (r - C) |b>, all three
    # components, a PAIR class (la, lb); the multipole order e = 1 is the
    # builder's argument, not a key field (#518; module docstring "A
    # DIPOLE CLASS").
    DIPOLE = "dipole"


class Engine(StrEnum):
    """The RECURRENCE a kernel runs (#401, module docstring)."""

    MMD = "mmd"  # McMurchie-Davidson: the [p|q] tower of [JEngine]
    HGP = "hgp"  # Head-Gordon-Pople: the VRR/HRR of [Fock] §3.1


# How many shells a target's class names: a quartet (l1, l2 | l3, l4) for the
# [p|q] kernel, a pair (la, lb) for the two per-pair families (#150), and a
# quartet (la, lc | lb, ld) in Hermite order for the exchange matrix (#196),
# and a pair (la, lb) for each one-electron matrix (#232), and a quartet
# (la, lb | lc, ld) in AO order for the fused Fock build (#272), and a pair
# (lP, lQ) of auxiliary shells for the Coulomb metric (#410), and a triple
# (la, lb | lP) -- an AO pair and one aux shell -- for the three-center
# integral (#441), and a pair (la, lb) for the dipole matrix (#518).
AM_LENGTH: dict[Target, int] = {
    Target.J: 4, Target.DQ: 2, Target.JAB: 2, Target.K: 4, Target.F: 4,
    Target.S: 2, Target.T: 2, Target.V: 2, Target.SCHWARZ: 2,
    Target.SX: 2, Target.METRIC: 2, Target.ERI3: 3, Target.PAIR_BOUND: 2,
    Target.RECORD_BOUND: 2, Target.DIPOLE: 2,
}


# The one operator each target is built from (#232); any other pairing is
# refused. The two-electron targets are Coulomb's, the metric and the
# three-center integral among them; the dipole is its own operator (#518).
OPERATOR_OF: dict[Target, Operator] = {
    Target.J: Operator.COULOMB, Target.DQ: Operator.COULOMB,
    Target.JAB: Operator.COULOMB, Target.K: Operator.COULOMB,
    Target.F: Operator.COULOMB, Target.SCHWARZ: Operator.COULOMB,
    Target.S: Operator.OVERLAP, Target.T: Operator.KINETIC,
    Target.V: Operator.NUCLEAR, Target.SX: Operator.OVERLAP,
    Target.METRIC: Operator.COULOMB, Target.ERI3: Operator.COULOMB,
    Target.PAIR_BOUND: Operator.COULOMB, Target.RECORD_BOUND: Operator.COULOMB,
    Target.DIPOLE: Operator.DIPOLE,
}


# Each target's NATIVE engine (#401): the one its key does not spell. `F` is
# the fused build on HGP (#272); every other target is MMD's.
NATIVE_ENGINE: dict[Target, Engine] = {
    t: Engine.HGP if t is Target.F else Engine.MMD for t in Target
}


# Why a (target, engine) other than the native one is refused (#401). Legal
# off-native: HGP `J` and HGP `K` (milestone 14 builds them), HGP `S`, `T`,
# `V` and `DIPOLE` (`one_electron.kernel_hgp`), and HGP `METRIC` and `ERI3`
# (`eri/density_fit/kernel_hgp.py`).
_NO_HGP_PAIR = (
    "the HGP engine is a quartet ERI recurrence (VRR/HRR, [Fock] §3.1); "
    "this pair kernel runs no ERI recurrence at all, so there is nothing "
    "for HGP to replace"
)
_NO_HGP_ONE_E = (
    "the HGP engine is the two-electron VRR/HRR of [Fock] §3.1, and its "
    "one-electron instance (one_electron.kernel_hgp) covers S, T, V and the "
    "dipole only; this one-electron kernel is McMurchie-Davidson's E "
    "fragment, and an HGP instance of it is a follow-up nobody has declared"
)


def _mmd_only(what: str) -> str:
    return (f"{what} runs the Target.J quartet's MMD plan by construction "
            "(its Hermite targets, the parity-dead drop, the E fragment "
            "inlined); an HGP instance is a follow-up nobody has declared")


_ENGINE_REFUSAL: dict[tuple[Target, Engine], str] = {
    (Target.DQ, Engine.HGP): _NO_HGP_PAIR,
    (Target.JAB, Engine.HGP): _NO_HGP_PAIR,
    (Target.PAIR_BOUND, Engine.HGP): (
        "the record bound is a closed form per primitive-pair record, with "
        "no ERI recurrence for HGP to replace"
    ),
    (Target.RECORD_BOUND, Engine.HGP): (
        "the dressed record bound is the E fragment and a closed form per "
        "primitive-pair record, with no ERI recurrence for HGP to replace"
    ),
    # S, T, V and DIPOLE are legal on HGP (`one_electron.kernel_hgp`, s and p
    # shells): the module docstring's "AN HGP ONE-ELECTRON CLASS".
    (Target.SX, Engine.HGP): _NO_HGP_ONE_E,
    (Target.SCHWARZ, Engine.HGP): _mmd_only(
        "the Schwarz kernel (la lb|la lb)"),
    (Target.F, Engine.MMD): (
        "the fused F is built on HGP only ([Fock] §3.2); a fused MMD F is "
        "#326, not yet built"
    ),
}


# Why a pair (or, since #441, a triple) target refuses `n_batches > 1`, per
# target (#232, module docstring): a structural reason for the targets with no
# Hermite targets to partition, and a decision for `V`, `SCHWARZ`, `METRIC`
# and `ERI3`, of which the structural reason is false.
_STRUCTURAL = (
    "never batched -- batching partitions the Hermite targets of a quartet, "
    "and this pair kernel has none to partition"
)
_PAIR_BATCH_REFUSAL: dict[Target, str] = {
    Target.DQ: _STRUCTURAL,
    Target.JAB: _STRUCTURAL,
    Target.PAIR_BOUND: _STRUCTURAL,
    Target.RECORD_BOUND: _STRUCTURAL,
    Target.S: _STRUCTURAL,
    Target.SX: _STRUCTURAL,
    Target.DIPOLE: _STRUCTURAL,
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
    Target.ERI3: (
        "unbatched by decision -- the three-center (mn|P) runs the quartet "
        "(la lb|lP 0)'s recurrence, whose Hermite targets (every t with "
        "|t| <= la + lb + lP) could be partitioned, so the structural reason "
        "does not apply; a batched three-center kernel is a follow-up nobody "
        "has filed"
    ),
}


# The word a validator message uses for a class of `n` shells.
_SHAPE: dict[int, str] = {2: "pair", 3: "triple", 4: "quartet"}


# Why `Target.F` refuses `n_batches > 1` (#272, module docstring): its own
# reason, not the pair one.
_F_BATCH_REFUSAL = (
    "an HGP class is never batched -- it has no Hermite targets at all, "
    "batched or not, so [JEngine]'s partition does not apply; the field's "
    "future meaning for F is VRR target grouping, not yet built"
)


# The geometric derivative order (module docstring, "A GRADIENT CLASS"):
# 0, the integral itself, everywhere; 1, the nuclear gradient contracted
# with a matrix, on the targets in `GRADIENT_TARGETS` only.
DERIVATIVE_ORDERS = (0, 1)

# The targets whose first derivative is built (`felizjueves.gradient`):
# `S`, `T`, `V` (its Hellmann-Feynman term included), `JAB`, the J
# chain's bra back-transform read as the Coulomb energy's bra derivative,
# and `K`, the exchange energy's bra derivative (`gradient.kernel_gk`).
# Each on its native engine, MMD, only.
GRADIENT_TARGETS = frozenset((Target.S, Target.T, Target.V, Target.JAB,
                              Target.K))

_GRADIENT_REFUSAL = (
    "a first derivative is built for S, T, V, JAB and K on the MMD engine only "
    "(felizjueves.gradient); this target's gradient is a follow-up"
)


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
    # target; (la, lb, lP) for `Target.ERI3`, an AO pair and one aux shell
    # (`AM_LENGTH`).
    am: tuple[int, int, int, int] | tuple[int, int, int] | tuple[int, int]
    target: Target
    # Named departures from the paper, sorted and unique. () is the literal
    # [JEngine] reconstruction. A kernel-body lever (`levers.VOCABULARY`)
    # must be legal on `target` (#337).
    levers: tuple[str, ...] = ()
    # 1, or a multiple of 3: finer splits break the three-way symmetry
    # ([JEngine, v2, §5]).
    n_batches: int = 1
    # The recurrence (#401). None, the default, resolves to the target's
    # native engine (`NATIVE_ENGINE`), so after construction it is always an
    # `Engine`; the key spells it only when it is not the native one.
    engine: Engine | None = None

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
        if self.engine is None:
            object.__setattr__(self, "engine", NATIVE_ENGINE[self.target])
        if not isinstance(self.engine, Engine):
            raise ValueError(f"engine {self.engine!r} is not an Engine")
        why = _ENGINE_REFUSAL.get((self.target, self.engine))
        if why is not None:
            raise ValueError(
                f"engine {self.engine} on target {self.target}: {why}")
        if (
            type(self.derivative_order) is not int
            or self.derivative_order not in DERIVATIVE_ORDERS
        ):
            raise ValueError(
                f"derivative order {self.derivative_order!r} not in "
                f"{DERIVATIVE_ORDERS}"
            )
        if self.derivative_order and (
            self.target not in GRADIENT_TARGETS
            or self.engine is not NATIVE_ENGINE[self.target]
        ):
            raise ValueError(
                f"derivative order {self.derivative_order} on target "
                f"{self.target_slot}: {_GRADIENT_REFUSAL}")
        if len(self.am) not in (2, 3, 4) or not all(
            type(x) is int and x >= 0 for x in self.am
        ):
            raise ValueError(
                f"am {self.am!r} is not two, three or four non-negative ints")
        want = AM_LENGTH[self.target]
        if len(self.am) != want:
            shape = _SHAPE[want]
            hermite = (
                " in Hermite order (la, lc | lb, ld)"
                if self.target is Target.K
                else " in AO order (la, lb | lc, ld)"
                if self.target is Target.F
                else " (la, lb | lP), an AO pair and one aux shell"
                if self.target is Target.ERI3 else ""
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
            why = _levers.refusal(self.target_slot, lever)
            if why is not None:
                raise ValueError(
                    f"lever {lever!r} on target {self.target_slot}: {why}")
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
        if want in (2, 3) and self.n_batches != 1:
            raise ValueError(
                f"n_batches {self.n_batches!r} for {_SHAPE[want]} target "
                f"{self.target}: "
                f"{_PAIR_BATCH_REFUSAL[self.target]}"
            )

    @property
    def target_slot(self) -> str:
        """The key's target segment (#401): the target's slot, plus
        ``.<engine>`` when the engine is not the target's native one --
        ``j``, ``f``, ``j.hgp``, ``k.hgp``. Kernel-body lever legality is
        per slot (`levers.SLOTS`)."""
        if self.engine is NATIVE_ENGINE[self.target]:
            return self.target.value
        return f"{self.target.value}.{self.engine.value}"

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

        An ERI3 triple (`Target.ERI3`, `am = (la, lb, lP)`) sums to
        la + lb + lP, the L of the `Target.J` quartet `(la lb|lP 0)` whose
        recurrence it runs; the triple key itself indexes NO table row.
        """
        return sum(self.am)

    def key(self) -> str:
        """E.g. ``coulomb/d0/(4,4|4,4)/b1/j/literal``.

        A pair class spells ``coulomb/d0/(2,1)/b1/dq/literal`` -- no ``|``,
        so it cannot collide with a quartet's; a one-electron class spells
        ``overlap/d0/(2,1)/b1/s/literal``. A quartet key is
        byte-identical to the one before #150: it is the join key of every
        measured row. An F class spells ``coulomb/d0/(2,1|1,0)/b1/f/literal``
        -- the AO order, target slot ``f`` (#272). An ERI3 triple spells
        ``coulomb/d0/(2,1|3)/b1/eri3/literal`` -- the bra pair, ``|``, the
        one aux shell (#441). A non-native engine suffixes the target slot:
        ``coulomb/d0/(1,1|1,0)/b1/j.hgp/literal`` (#401, `target_slot`).
        """
        if len(self.am) == 4:
            l1, l2, l3, l4 = self.am
            cls = f"({l1},{l2}|{l3},{l4})"
        elif len(self.am) == 3:
            la, lb, lp = self.am
            cls = f"({la},{lb}|{lp})"
        else:
            la, lb = self.am
            cls = f"({la},{lb})"
        levers = "+".join(self.levers) or "literal"
        return (
            f"{self.operator}/d{self.derivative_order}/{cls}/"
            f"b{self.n_batches}/{self.target_slot}/{levers}"
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
                    self.target, rest, self.n_batches, self.engine)


def lever_provenance(name: str) -> str:
    """The provenance line a builder records for kernel-body lever `name`
    (`levers.provenance`, re-exported so a builder reads the vocabulary
    through the key it already depends on)."""
    return _levers.provenance(name)
