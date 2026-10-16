"""`Spec`, the join key: frozen, canonical, and narrow on purpose."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools

import pytest

from felizjueves.spec import AM_LENGTH, OPERATOR_OF, Operator, Spec, Target


def _spec(**kw) -> Spec:
    base = dict(
        operator=Operator.COULOMB, derivative_order=0, am=(4, 4, 4, 4), target=Target.J
    )
    return Spec(**{**base, **kw})


def test_key_names_every_field() -> None:
    assert _spec().key() == "coulomb/d0/(4,4|4,4)/b1/j/literal"
    assert _spec(am=(10, 2, 3, 1), levers=("a", "b"), n_batches=9).key() == (
        "coulomb/d0/(10,2|3,1)/b9/j/a+b"
    )


def test_the_two_arms_of_a_lever_ab_are_two_keys() -> None:
    # #55: the recycle arm is told apart by its key, not by a note; levers
    # from two layers (the plan's no-sieve, the builder's recycle) compose.
    literal, lever = _spec(n_batches=3), _spec(levers=("recycle",), n_batches=3)
    assert literal.key() == "coulomb/d0/(4,4|4,4)/b3/j/literal"
    assert lever.key() == "coulomb/d0/(4,4|4,4)/b3/j/recycle"
    assert _spec(levers=("no-sieve", "recycle")).key().endswith("/no-sieve+recycle")
    with pytest.raises(ValueError):
        _spec(levers=("recycle", "no-sieve"))  # sorted, or refused


def test_the_tuned_arm_is_its_own_key() -> None:
    # #161: the searched board's plan is keyed apart from literal and from
    # no-sieve, so the store can tell the arms apart; sorted like any lever.
    assert _spec(levers=("tuned",), n_batches=3).key() == (
        "coulomb/d0/(4,4|4,4)/b3/j/tuned")
    keys = {_spec(levers=lv).key() for lv in ((), ("tuned",), ("no-sieve",))}
    assert len(keys) == 3
    assert _spec(levers=("recycle", "tuned")).key().endswith("/recycle+tuned")
    with pytest.raises(ValueError):
        _spec(levers=("tuned", "recycle"))  # sorted, or refused
    with pytest.raises(ValueError):
        _spec(levers=("tuned", "tuned"))  # unique, or refused


def test_the_batch_count_is_part_of_the_key() -> None:
    # A kernel is emitted per class AND per batch, so two batch counts are two
    # artifacts and must be two keys.
    assert _spec().n_batches == 1
    assert _spec(n_batches=3) != _spec(n_batches=6)
    assert len({_spec(n_batches=n).key() for n in (1, 3, 6, 9)}) == 4


def test_total_am_is_the_table_row() -> None:
    # The convention the next PR must honour: a Spec indexes a [JEngine] table
    # row by L = l1 + l2 + l3 + l4, and by nothing else.
    assert _spec(am=(4, 4, 4, 4)).total_am == 16
    assert _spec(am=(3, 0, 2, 2)).total_am == 7
    assert _spec(am=(4, 3, 0, 0)).total_am == _spec(am=(0, 0, 3, 4)).total_am


def test_equal_values_are_one_key() -> None:
    assert _spec() == _spec() and hash(_spec()) == hash(_spec())
    assert len({_spec(), _spec(), _spec(levers=("x",))}) == 2


def test_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _spec().derivative_order = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        {"operator": "coulomb"},
        {"target": "k"},
        {"derivative_order": 1},
        {"derivative_order": False},
        {"am": (4, 4, 4)},
        {"am": (4, 4, 4, -1)},
        {"levers": ("b", "a")},
        {"levers": ("a", "a")},
        {"levers": ("",)},
        {"n_batches": 0},
        {"n_batches": 2},
        {"n_batches": 4},
        {"n_batches": -3},
        {"n_batches": 3.0},
        {"n_batches": True},
        {"am": (4, 4)},  # J takes a quartet, not a pair
        {"am": (4,)},
        {"am": ()},
    ],
)
def test_illegal_values_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        _spec(**bad)


# #150: the pair targets. A pair class is (la, lb); a quartet (l1, l2 | l3, l4).

PAIR_TARGETS = (Target.DQ, Target.JAB)


def test_the_target_enum_is_widened_not_renamed() -> None:
    # #196 appends `k`; the earlier members keep their place and value.
    # #232 appends `s`, `t` and `v` the same way, and #272 appends `f`.
    # The Schwarz matrix appends `schwarz`, a pair class, and the
    # cross-basis overlap `sx`, an ordered pair class.
    assert [t.value for t in Target] == [
        "j", "dq", "jab", "k", "s", "t", "v", "f", "schwarz", "sx"]
    assert AM_LENGTH == {
        Target.J: 4, Target.DQ: 2, Target.JAB: 2, Target.K: 4,
        Target.S: 2, Target.T: 2, Target.V: 2, Target.F: 4,
        Target.SCHWARZ: 2, Target.SX: 2,
    }


@pytest.mark.parametrize("target", PAIR_TARGETS)
def test_a_pair_target_takes_two_shells_and_only_two(target: Target) -> None:
    assert _spec(target=target, am=(2, 1)).am == (2, 1)
    for am in ((4, 4, 4, 4), (2, 1, 0), (2,), (2, -1), (2, 1.0)):
        with pytest.raises(ValueError):
            _spec(target=target, am=am)


def test_j_takes_four_shells_and_only_four() -> None:
    assert _spec(am=(1, 0, 1, 0)).am == (1, 0, 1, 0)
    for am in ((1, 0), (1, 0, 1), (1, 0, 1, 0, 0)):
        with pytest.raises(ValueError):
            _spec(am=am)


@pytest.mark.parametrize("target", PAIR_TARGETS)
def test_total_am_is_the_sum_for_a_pair_too(target: Target) -> None:
    assert _spec(target=target, am=(3, 2)).total_am == 5
    assert _spec(target=target, am=(0, 0)).total_am == 0
    assert _spec(am=(3, 2, 0, 0)).total_am == 5  # same sum, different classes
    assert "NO published table row" in Spec.total_am.__doc__


@pytest.mark.parametrize("target", PAIR_TARGETS)
@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_a_pair_target_is_never_batched(target: Target, n: int) -> None:
    # A legal quartet batch count is still refused for a pair: batching
    # partitions a quartet's Hermite targets, and a pair has none.
    assert _spec(n_batches=n).n_batches == n
    with pytest.raises(ValueError, match="never batched"):
        _spec(target=target, am=(2, 1), n_batches=n)
    assert _spec(target=target, am=(2, 1)).n_batches == 1


def test_a_pair_key_is_spelled_without_a_bar() -> None:
    assert _spec(target=Target.DQ, am=(2, 1)).key() == "coulomb/d0/(2,1)/b1/dq/literal"
    assert _spec(target=Target.JAB, am=(0, 3), levers=("x",)).key() == (
        "coulomb/d0/(0,3)/b1/jab/x"
    )


def test_a_pair_key_never_collides_with_a_quartet_key() -> None:
    ls = range(4)
    quartets = {
        _spec(am=(a, b, c, d), levers=lv).key()
        for a in ls for b in ls for c in ls for d in ls for lv in ((), ("x",))
    }
    pairs = {
        _spec(target=t, am=(a, b), levers=lv).key()
        for t in PAIR_TARGETS for a in ls for b in ls for lv in ((), ("x",))
    }
    assert len(quartets) == 4**4 * 2 and len(pairs) == 2 * 4**2 * 2
    assert quartets.isdisjoint(pairs)
    # Not just a different target slot: the class field alone tells them apart.
    q_classes = {k.split("/")[2] for k in quartets}
    p_classes = {k.split("/")[2] for k in pairs}
    assert q_classes.isdisjoint(p_classes)
    assert all("|" in c for c in q_classes) and not any("|" in c for c in p_classes)


def test_a_quartet_key_is_byte_identical_to_before_the_widening() -> None:
    # The join key of 122 measured rows: these strings are pre-#150 literals.
    assert _spec(am=(0, 0, 0, 0)).key() == "coulomb/d0/(0,0|0,0)/b1/j/literal"
    assert _spec(am=(8, 0, 8, 0), n_batches=12, levers=("recycle",)).key() == (
        "coulomb/d0/(8,0|8,0)/b12/j/recycle"
    )


# #232: the one-electron operators and their three pair targets.

ONE_ELECTRON = {
    Target.S: Operator.OVERLAP,
    Target.T: Operator.KINETIC,
    Target.V: Operator.NUCLEAR,
}


def _one(target: Target, **kw) -> Spec:
    return _spec(operator=ONE_ELECTRON[target], target=target, am=(2, 1), **kw)


def test_the_operator_enum_is_widened_not_renamed() -> None:
    # The first widening of a field that had one legal value since the key
    # was designed: COULOMB keeps its place and its value.
    assert [o.value for o in Operator] == ["coulomb", "overlap", "kinetic", "nuclear"]
    assert (Operator.OVERLAP, Operator.KINETIC, Operator.NUCLEAR) == (
        "overlap", "kinetic", "nuclear")


def test_operator_of_pins_each_target_to_one_operator() -> None:
    assert OPERATOR_OF == {
        Target.J: Operator.COULOMB, Target.DQ: Operator.COULOMB,
        Target.JAB: Operator.COULOMB, Target.K: Operator.COULOMB,
        Target.S: Operator.OVERLAP, Target.T: Operator.KINETIC,
        Target.V: Operator.NUCLEAR, Target.F: Operator.COULOMB,
        Target.SCHWARZ: Operator.COULOMB, Target.SX: Operator.OVERLAP,
    }
    assert set(OPERATOR_OF) == set(Target) == set(AM_LENGTH)


@pytest.mark.parametrize("target", list(Target))
@pytest.mark.parametrize("operator", list(Operator))
def test_every_target_operator_pairing_is_admitted_or_refused(
    target: Target, operator: Operator
) -> None:
    # Both directions: the pinned pairing is keyed, every other is refused --
    # Spec(COULOMB, 0, (1, 1), S) as much as Spec(OVERLAP, 0, (1, 1), DQ).
    am = (1,) * AM_LENGTH[target]
    if OPERATOR_OF[target] is operator:
        assert Spec(operator, 0, am, target).operator is operator
    else:
        want = OPERATOR_OF[target]
        with pytest.raises(ValueError, match=f"built from operator {want}"):
            Spec(operator, 0, am, target)


def test_coulomb_overlap_is_refused_not_keyed() -> None:
    with pytest.raises(ValueError, match="target s is built from operator overlap"):
        Spec(Operator.COULOMB, 0, (1, 1), Target.S)


@pytest.mark.parametrize("target", list(ONE_ELECTRON))
def test_a_one_electron_target_takes_a_pair(target: Target) -> None:
    assert AM_LENGTH[target] == 2
    assert _one(target).am == (2, 1)
    assert _one(target).total_am == 3
    op = ONE_ELECTRON[target]
    for am in ((2, 1, 0, 0), (2, 1, 0), (2,), (2, -1)):
        with pytest.raises(ValueError):
            _spec(operator=op, target=target, am=am)


@pytest.mark.parametrize("target", list(ONE_ELECTRON))
def test_a_one_electron_target_is_derivative_order_zero(target: Target) -> None:
    with pytest.raises(ValueError, match="derivative order"):
        _spec(operator=ONE_ELECTRON[target], target=target, am=(2, 1),
              derivative_order=1)


def test_a_one_electron_key_has_the_shape_every_key_has() -> None:
    assert _one(Target.S).key() == "overlap/d0/(2,1)/b1/s/literal"
    assert _one(Target.T).key() == "kinetic/d0/(2,1)/b1/t/literal"
    assert _one(Target.V).key() == "nuclear/d0/(2,1)/b1/v/literal"
    assert _one(Target.V, levers=("x",)).key() == "nuclear/d0/(2,1)/b1/v/x"


def test_one_electron_keys_never_collide_with_each_other_or_coulomb() -> None:
    ls = range(4)
    one = {Spec(ONE_ELECTRON[t], 0, (a, b), t).key()
           for t in ONE_ELECTRON for a in ls for b in ls}
    coulomb_pairs = {_spec(target=t, am=(a, b)).key()
                     for t in PAIR_TARGETS for a in ls for b in ls}
    assert len(one) == 3 * 4**2
    assert one.isdisjoint(coulomb_pairs)


# The batching refusal, per target: which reason applies is in the message.

@pytest.mark.parametrize("n", [3, 6, 9, 12])
@pytest.mark.parametrize("target", [Target.DQ, Target.JAB, Target.S, Target.T])
def test_a_pair_without_hermite_targets_is_refused_structurally(
    target: Target, n: int
) -> None:
    with pytest.raises(ValueError) as info:
        _spec(operator=OPERATOR_OF[target], target=target, am=(2, 1), n_batches=n)
    msg = str(info.value)
    assert "never batched" in msg and "partitions the Hermite targets" in msg
    assert "decision" not in msg


@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_v_is_unbatched_by_decision_not_by_the_structural_rule(n: int) -> None:
    with pytest.raises(ValueError) as info:
        _one(Target.V, n_batches=n)
    msg = str(info.value)
    assert "unbatched by milestone 11's decision" in msg
    assert "|t| <= la + lb" in msg
    # The false reason is not given: V does have Hermite targets.
    assert "never batched" not in msg and "none to partition" not in msg
    assert _one(Target.V).n_batches == 1


# Every pre-existing key, byte for byte: a FIXED list of literals written
# before #232 (J, K, DQ, JAB; every lever spelling the store uses). Nothing
# in milestone 11 may move one -- they are the join key of every measured row.
PRE_232_KEYS = [
    ((Operator.COULOMB, 0, (0, 0, 0, 0), Target.J), {},
     "coulomb/d0/(0,0|0,0)/b1/j/literal"),
    ((Operator.COULOMB, 0, (4, 4, 4, 4), Target.J), {},
     "coulomb/d0/(4,4|4,4)/b1/j/literal"),
    ((Operator.COULOMB, 0, (4, 0, 3, 0), Target.J), {"n_batches": 3},
     "coulomb/d0/(4,0|3,0)/b3/j/literal"),
    ((Operator.COULOMB, 0, (8, 0, 8, 0), Target.J),
     {"n_batches": 12, "levers": ("recycle",)},
     "coulomb/d0/(8,0|8,0)/b12/j/recycle"),
    ((Operator.COULOMB, 0, (7, 0, 6, 0), Target.J),
     {"n_batches": 6, "levers": ("no-sieve",)},
     "coulomb/d0/(7,0|6,0)/b6/j/no-sieve"),
    ((Operator.COULOMB, 0, (5, 0, 5, 0), Target.J),
     {"n_batches": 9, "levers": ("tuned",)},
     "coulomb/d0/(5,0|5,0)/b9/j/tuned"),
    ((Operator.COULOMB, 0, (2, 1, 2, 1), Target.K), {},
     "coulomb/d0/(2,1|2,1)/b1/k/literal"),
    ((Operator.COULOMB, 0, (2, 1), Target.DQ), {},
     "coulomb/d0/(2,1)/b1/dq/literal"),
    ((Operator.COULOMB, 0, (0, 3), Target.JAB), {"levers": ("x",)},
     "coulomb/d0/(0,3)/b1/jab/x"),
    ((Operator.COULOMB, 0, (3, 3), Target.JAB), {},
     "coulomb/d0/(3,3)/b1/jab/literal"),
]


@pytest.mark.parametrize(("args", "kw", "key"), PRE_232_KEYS)
def test_every_pre_existing_key_is_byte_identical(args, kw, key) -> None:
    assert Spec(*args, **kw).key() == key


# #272 widens the enum with `Target.F`; every key above, and milestone 11's
# one-electron keys below (literals written before #272), must not move.
PRE_272_ONE_ELECTRON_KEYS = [
    ((Operator.OVERLAP, 0, (2, 1), Target.S), "overlap/d0/(2,1)/b1/s/literal"),
    ((Operator.KINETIC, 0, (0, 3), Target.T), "kinetic/d0/(0,3)/b1/t/literal"),
    ((Operator.NUCLEAR, 0, (3, 3), Target.V), "nuclear/d0/(3,3)/b1/v/literal"),
]


@pytest.mark.parametrize(("args", "key"), PRE_272_ONE_ELECTRON_KEYS)
def test_every_one_electron_key_is_byte_identical_after_f(args, key) -> None:
    assert Spec(*args).key() == key


# #337 puts a lever-legality check into `Spec.__post_init__`. It is a
# refusal, never a respelling: every key of this enumeration -- every target,
# every class up to l = 2, the plan and lowering levers, an unknown name and
# every legal batch count -- is digested here, the digest computed on
# origin/main before #337.
KEYS_BEFORE_337 = (
    3672, "e744293fb16ed978c9b6574c0e020451ae9300d5e5e35ae9aad2ef25e2244bca")
# The targets the digest was taken over: a later widening (`Target.SCHWARZ`)
# adds keys of its own and must move none of these.
PRE_337_TARGETS = (Target.J, Target.DQ, Target.JAB, Target.K, Target.S,
                   Target.T, Target.V, Target.F)


def test_every_key_is_byte_identical_after_the_lever_vocabulary() -> None:
    keys = []
    for t in PRE_337_TARGETS:
        n = AM_LENGTH[t]
        for am in itertools.product(range(3), repeat=n):
            for lv in ((), ("no-sieve",), ("recycle",), ("tuned",),
                       ("no-sieve", "recycle"), ("x",)):
                for b in ((1, 3, 6) if n == 4 and t is not Target.F else (1,)):
                    keys.append(Spec(OPERATOR_OF[t], 0, am, t, lv, b).key())
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    assert (len(keys), digest) == KEYS_BEFORE_337


# The cross-basis overlap S_{B1,B2}: an ORDERED pair class of the overlap
# operator, never batched for S's structural reason, its key a pair's.


def test_the_cross_overlap_key_is_an_ordered_overlap_pair() -> None:
    a = Spec(Operator.OVERLAP, 0, (2, 1), Target.SX)
    b = Spec(Operator.OVERLAP, 0, (1, 2), Target.SX)
    assert a.key() == "overlap/d0/(2,1)/b1/sx/literal"
    assert a.key() != b.key()
    assert a.key() != Spec(Operator.OVERLAP, 0, (2, 1), Target.S).key()
    with pytest.raises(ValueError, match="never batched"):
        Spec(Operator.OVERLAP, 0, (1, 1), Target.SX, n_batches=3)
    with pytest.raises(ValueError, match="built from operator"):
        Spec(Operator.COULOMB, 0, (1, 1), Target.SX)
    with pytest.raises(ValueError, match="pair"):
        Spec(Operator.OVERLAP, 0, (1, 1, 1, 1), Target.SX)
