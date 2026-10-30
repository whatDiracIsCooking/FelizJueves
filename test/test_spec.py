"""`Spec`, the join key: frozen, canonical, and narrow on purpose."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools

import pytest

from felizjueves.performance.measured_store import target_of
from felizjueves.spec import (
    _PAIR_BATCH_REFUSAL,
    AM_LENGTH,
    GRADIENT_OFF_NATIVE,
    GRADIENT_TARGETS,
    HESSIAN_TARGETS,
    OPERATOR_OF,
    Engine,
    Operator,
    Spec,
    Target,
)


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
        {"derivative_order": 1},  # legal on S, T, V and JAB only, not J
        {"derivative_order": 2},
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
    # cross-basis overlap `sx`, an ordered pair class, and #410 the
    # Coulomb metric `metric`, a pair of aux shells, and #441 the
    # three-center integral `eri3`, a triple, and the record bound
    # `pair_bound`, a pair class, and #499 K's dressed `record_bound`, an
    # ordered pair class, and #518 the dipole matrix `dipole`, a pair class,
    # and the quadrupole matrix `quadrupole`, a pair class.
    assert [t.value for t in Target] == [
        "j", "dq", "jab", "k", "s", "t", "v", "f", "schwarz", "sx", "metric",
        "eri3", "pair_bound", "record_bound", "dipole", "quadrupole"]
    assert AM_LENGTH == {
        Target.J: 4, Target.DQ: 2, Target.JAB: 2, Target.K: 4,
        Target.S: 2, Target.T: 2, Target.V: 2, Target.F: 4,
        Target.SCHWARZ: 2, Target.SX: 2, Target.METRIC: 2, Target.ERI3: 3,
        Target.PAIR_BOUND: 2, Target.RECORD_BOUND: 2, Target.DIPOLE: 2,
        Target.QUADRUPOLE: 2,
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
    # was designed: COULOMB keeps its place and its value. #518 appends
    # `dipole` the same way, and the quadrupole appends `quadrupole`.
    assert [o.value for o in Operator] == [
        "coulomb", "overlap", "kinetic", "nuclear", "dipole", "quadrupole"]
    assert (Operator.OVERLAP, Operator.KINETIC, Operator.NUCLEAR,
            Operator.DIPOLE) == ("overlap", "kinetic", "nuclear", "dipole")


def test_operator_of_pins_each_target_to_one_operator() -> None:
    assert OPERATOR_OF == {
        Target.J: Operator.COULOMB, Target.DQ: Operator.COULOMB,
        Target.JAB: Operator.COULOMB, Target.K: Operator.COULOMB,
        Target.S: Operator.OVERLAP, Target.T: Operator.KINETIC,
        Target.V: Operator.NUCLEAR, Target.F: Operator.COULOMB,
        Target.SCHWARZ: Operator.COULOMB, Target.SX: Operator.OVERLAP,
        Target.METRIC: Operator.COULOMB, Target.ERI3: Operator.COULOMB,
        Target.PAIR_BOUND: Operator.COULOMB,
        Target.RECORD_BOUND: Operator.COULOMB,
        Target.DIPOLE: Operator.DIPOLE,
        Target.QUADRUPOLE: Operator.QUADRUPOLE,
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
def test_a_one_electron_target_takes_orders_one_and_two_and_no_higher(
        target: Target) -> None:
    # The gradient (felizjueves.gradient) and the hessian
    # (felizjueves.hessian): orders 1 and 2 are legal, each key differs
    # from the integral's in the `d` segment alone, and order 3 is refused.
    d0 = _spec(operator=ONE_ELECTRON[target], target=target, am=(2, 1))
    for order in (1, 2):
        dn = _spec(operator=ONE_ELECTRON[target], target=target, am=(2, 1),
                   derivative_order=order)
        assert dn.key() == d0.key().replace("/d0/", f"/d{order}/")
    with pytest.raises(ValueError, match="derivative order"):
        _spec(operator=ONE_ELECTRON[target], target=target, am=(2, 1),
              derivative_order=3)


def test_order_two_is_refused_on_a_non_native_engine() -> None:
    with pytest.raises(ValueError, match="derivative order 2 on target v.hgp"):
        Spec(Operator.NUCLEAR, 2, (1, 0), Target.V, engine=Engine.HGP)


@pytest.mark.parametrize("target", [t for t in Target
                                    if t not in GRADIENT_TARGETS])
def test_order_one_is_refused_outside_the_gradient_targets(
        target: Target) -> None:
    am = (1,) * AM_LENGTH[target]
    with pytest.raises(ValueError, match="derivative order 1 on target"):
        Spec(OPERATOR_OF[target], 1, am, target)


def test_k_takes_order_one_in_its_hermite_pairing() -> None:
    # The exchange gradient (gradient.kernel_gk): K's am, bra first, and the
    # key differs from the integral's in the `d` segment alone.
    d1 = Spec(Operator.COULOMB, 1, (2, 1, 1, 0), Target.K)
    d0 = Spec(Operator.COULOMB, 0, (2, 1, 1, 0), Target.K)
    assert Target.K in GRADIENT_TARGETS
    assert d1.key() == d0.key().replace("/d0/", "/d1/")
    # ... and on HGP too, an off-native d1 key beside J on HGP (gradient.kernel_ghk)
    hgp = Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.K, engine=Engine.HGP)
    assert hgp.key() == "coulomb/d1/(1,0|0,0)/b1/k.hgp/literal"
    assert GRADIENT_OFF_NATIVE == {(Target.J, Engine.HGP),
                                   (Target.K, Engine.HGP), (Target.T, Engine.HGP),
                                   (Target.S, Engine.HGP), (Target.V, Engine.HGP),
                                   (Target.ERI3, Engine.HGP),
                                   (Target.METRIC, Engine.HGP)}
    with pytest.raises(ValueError, match="derivative order 1 on target k.rys"):
        Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.K, engine=Engine.RYS)


def test_order_one_is_refused_on_a_non_native_engine() -> None:
    # S and V on HGP are off-native gradients (gradient.kernel_hgs,
    # gradient.kernel_ghv), each keyed as its HGP integral's class in the `d`
    # segment alone; on Rys, and the dipole on HGP, they are not
    assert Spec(Operator.OVERLAP, 1, (1, 0), Target.S,
                engine=Engine.HGP).key() == "overlap/d1/(1,0)/b1/s.hgp/literal"
    d0 = Spec(Operator.NUCLEAR, 0, (1, 0), Target.V, engine=Engine.HGP)
    d1 = Spec(Operator.NUCLEAR, 1, (1, 0), Target.V, engine=Engine.HGP)
    assert d1.key() == d0.key().replace("/d0/", "/d1/")
    assert d1.key() == "nuclear/d1/(1,0)/b1/v.hgp/literal"
    with pytest.raises(ValueError, match="derivative order 1 on target s.rys"):
        Spec(Operator.OVERLAP, 1, (1, 0), Target.S, engine=Engine.RYS)
    with pytest.raises(ValueError, match="derivative order 1 on target v.rys"):
        Spec(Operator.NUCLEAR, 1, (1, 0), Target.V, engine=Engine.RYS)
    with pytest.raises(ValueError, match="derivative order 1 on target "
                                         "dipole.hgp"):
        Spec(Operator.DIPOLE, 1, (1, 0), Target.DIPOLE, engine=Engine.HGP)
    assert Spec(Operator.COULOMB, 1, (1, 0), Target.JAB).key() == \
        "coulomb/d1/(1,0)/b1/jab/literal"


@pytest.mark.parametrize("target", [t for t in Target
                                    if t not in HESSIAN_TARGETS])
def test_order_two_is_refused_outside_the_hessian_targets(
        target: Target) -> None:
    am = (1,) * AM_LENGTH[target]
    with pytest.raises(ValueError, match="derivative order 2 on target"):
        Spec(OPERATOR_OF[target], 2, am, target)


def test_jab_takes_order_two_and_dq_order_one_for_the_hessian() -> None:
    # The Coulomb Hessian (felizjueves.hessian): JAB at d2 and DQ at d1, each
    # key differing from the integral's in the `d` segment alone; order 3 is
    # refused everywhere.
    # (S, T and V at d2 are the one-electron Hessians, hessian.kernel_h1e;
    # K at d2 the exchange energy's, hessian.kernel_hkx; METRIC at d2 the DF
    # metric's, hessian.kernel_d2metric; ERI3 at d2 the three-center DF
    # integral's, hessian.kernel_d2eri3.)
    assert HESSIAN_TARGETS == frozenset({Target.JAB, Target.S, Target.T,
                                         Target.V, Target.K, Target.METRIC,
                                         Target.ERI3})
    assert Target.DQ in GRADIENT_TARGETS
    d2 = Spec(Operator.COULOMB, 2, (2, 1), Target.JAB)
    assert d2.key() == "coulomb/d2/(2,1)/b1/jab/literal"
    dq = Spec(Operator.COULOMB, 1, (2, 1), Target.DQ)
    assert dq.key() == "coulomb/d1/(2,1)/b1/dq/literal"
    with pytest.raises(ValueError, match="derivative order"):
        Spec(Operator.COULOMB, 3, (2, 1), Target.JAB)
    with pytest.raises(ValueError, match="derivative order 2 on target dq"):
        Spec(Operator.COULOMB, 2, (2, 1), Target.DQ)


def test_k_takes_order_two_in_its_hermite_pairing() -> None:
    # The exchange Hessian (hessian.kernel_hkx): K's am, bra first, the key
    # differing from the integral's in the `d` segment alone; MMD only.
    d2 = Spec(Operator.COULOMB, 2, (2, 1, 1, 0), Target.K)
    d0 = Spec(Operator.COULOMB, 0, (2, 1, 1, 0), Target.K)
    assert d2.key() == d0.key().replace("/d0/", "/d2/")
    with pytest.raises(ValueError, match="derivative order 2 on target k.hgp"):
        Spec(Operator.COULOMB, 2, (1, 0, 0, 0), Target.K, engine=Engine.HGP)


def test_the_metric_takes_order_two_on_mmd_only() -> None:
    # hessian.kernel_d2metric: the integral's pair at d2, the key differing
    # in the `d` segment alone; the HGP metric (metric.hgp) refuses.
    d0 = Spec(Operator.COULOMB, 0, (1, 2), Target.METRIC)
    d2 = Spec(Operator.COULOMB, 2, (1, 2), Target.METRIC)
    assert d2.key() == d0.key().replace("/d0/", "/d2/")
    assert d2.key() == "coulomb/d2/(1,2)/b1/metric/literal"
    with pytest.raises(ValueError, match="derivative order 2 on target "
                                         "metric.hgp"):
        Spec(Operator.COULOMB, 2, (1, 2), Target.METRIC, engine=Engine.HGP)


def test_the_df_targets_take_order_one_and_both_on_hgp_too() -> None:
    # felizjueves.gradient's kernel_gmetric / kernel_geri3: the integral's
    # class at d1, the key differing in the `d` segment alone; both on HGP
    # too (kernel_hgmetric, kernel_hgeri3: GRADIENT_OFF_NATIVE).
    for target, am in ((Target.METRIC, (1, 2)), (Target.ERI3, (0, 1, 2))):
        d0 = Spec(Operator.COULOMB, 0, am, target)
        d1 = Spec(Operator.COULOMB, 1, am, target)
        assert d1.key() == d0.key().replace("/d0/", "/d1/")
    h0 = Spec(Operator.COULOMB, 0, (0, 1, 2), Target.ERI3, engine=Engine.HGP)
    h1 = Spec(Operator.COULOMB, 1, (0, 1, 2), Target.ERI3, engine=Engine.HGP)
    assert h1.key() == h0.key().replace("/d0/", "/d1/") == \
        "coulomb/d1/(0,1|2)/b1/eri3.hgp/literal"
    assert (Target.ERI3, Engine.HGP) in GRADIENT_OFF_NATIVE
    m0 = Spec(Operator.COULOMB, 0, (1, 2), Target.METRIC, engine=Engine.HGP)
    m1 = Spec(Operator.COULOMB, 1, (1, 2), Target.METRIC, engine=Engine.HGP)
    assert m1.key() == m0.key().replace("/d0/", "/d1/") == \
        "coulomb/d1/(1,2)/b1/metric.hgp/literal"
    assert (Target.METRIC, Engine.HGP) in GRADIENT_OFF_NATIVE


def test_eri3_takes_order_two_on_mmd_only() -> None:
    # hessian.kernel_d2eri3: the triple at d2, the key differing from the
    # integral's in the `d` segment alone; the HGP eri3 (eri3.hgp) refuses.
    d0 = Spec(Operator.COULOMB, 0, (0, 1, 2), Target.ERI3)
    d2 = Spec(Operator.COULOMB, 2, (0, 1, 2), Target.ERI3)
    assert d2.key() == d0.key().replace("/d0/", "/d2/")
    with pytest.raises(ValueError, match="derivative order 2 on target "
                                         "eri3.hgp"):
        Spec(Operator.COULOMB, 2, (0, 1, 2), Target.ERI3, engine=Engine.HGP)


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


# #410: the two-center Coulomb metric (P|Q), a pair (lP, lQ) of auxiliary
# shells on the Coulomb operator, unbatched by decision.


def _metric(am=(2, 1), **kw) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.METRIC, **kw)


def test_the_metric_key_is_a_coulomb_pair_and_round_trips() -> None:
    a = _metric()
    assert a.key() == "coulomb/d0/(2,1)/b1/metric/literal"
    assert a.total_am == 3  # the L of the quartet (2 0|1 0) it runs
    assert _metric((1, 2)).key() == "coulomb/d0/(1,2)/b1/metric/literal"
    # Read back by a parser that does not call Spec.key(): the class and
    # the target slot recover the Spec exactly.
    for lp in range(5):
        for lq in range(5):
            s = _metric((lp, lq))
            op, d, cls, b, t, lv = s.key().split("/")
            assert cls == f"({lp},{lq})" and t == "metric" and lv == "literal"
            back = Spec(Operator(op), int(d[1:]),
                        tuple(int(x) for x in cls[1:-1].split(",")),
                        Target(t), n_batches=int(b[1:]))
            assert back == s and back.key() == s.key()
            assert target_of(f"{s.key()}@sm_86/nvcc 13.0") is Target.METRIC
    assert "Target.METRIC" in Spec.total_am.__doc__


def test_a_metric_key_never_collides_with_any_other_target_key() -> None:
    ls = range(5)
    metric = {_metric((a, b)).key() for a in ls for b in ls}
    assert len(metric) == 25
    others = set()
    for t in Target:
        if t is Target.METRIC:
            continue
        shapes = itertools.product(ls, repeat=AM_LENGTH[t])
        others |= {Spec(OPERATOR_OF[t], 0, am, t).key() for am in shapes}
    assert metric.isdisjoint(others)
    # The quartet it runs, (lP 0|lQ 0), is a different key, and the class
    # field alone tells them apart (no `|` in a pair).
    assert _metric((2, 1)).key() != _spec(am=(2, 0, 1, 0)).key()
    assert all("|" not in k.split("/")[2] for k in metric)


@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_a_batched_metric_spec_is_refused_by_decision(n: int) -> None:
    with pytest.raises(ValueError, match="unbatched by decision") as err:
        _metric(n_batches=n)
    assert "pair target metric" in str(err.value)
    assert "Hermite targets" in _PAIR_BATCH_REFUSAL[Target.METRIC]
    assert _metric().n_batches == 1


def test_the_metric_takes_a_pair_on_coulomb_only() -> None:
    for op in (Operator.OVERLAP, Operator.KINETIC, Operator.NUCLEAR):
        with pytest.raises(ValueError, match="built from operator coulomb"):
            Spec(op, 0, (1, 1), Target.METRIC)
    for am in ((1, 0, 1, 0), (1, 0, 1), (1,), (1, -1), (1, 1.0)):
        with pytest.raises(ValueError):
            _metric(am)


# #441: the three-center integral (mn|P), a TRIPLE (la, lb, lP) -- an AO pair
# on the bra, one aux shell on the ket -- running the quartet (la lb|lP 0).


def _eri3(am=(2, 1, 3), **kw) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.ERI3, **kw)


def test_the_eri3_key_is_a_coulomb_triple_and_round_trips() -> None:
    a = _eri3()
    assert a.key() == "coulomb/d0/(2,1|3)/b1/eri3/literal"
    assert Target.ERI3 == "eri3" and AM_LENGTH[Target.ERI3] == 3
    assert OPERATOR_OF[Target.ERI3] is Operator.COULOMB
    for am in itertools.product(range(4), repeat=3):
        s = _eri3(am)
        op, d, cls, b, t, lv = s.key().split("/")
        la, lb, lp = am
        assert cls == f"({la},{lb}|{lp})" and t == "eri3" and lv == "literal"
        bra, ket = cls[1:-1].split("|")
        back = Spec(Operator(op), int(d[1:]),
                    (*(int(x) for x in bra.split(",")), int(ket)),
                    Target(t), n_batches=int(b[1:]))
        assert back == s and back.key() == s.key()
        assert target_of(f"{s.key()}@sm_86/nvcc 13.0") is Target.ERI3


def test_eri3_total_am_is_the_quartet_it_runs_and_no_table_row() -> None:
    # L of the J quartet (la lb|lP 0), exactly as SCHWARZ and METRIC.
    assert _eri3((2, 1, 3)).total_am == 6
    assert _eri3((2, 1, 3)).total_am == _spec(am=(2, 1, 3, 0)).total_am
    assert _eri3((0, 0, 0)).total_am == 0
    doc = Spec.total_am.__doc__
    assert "Target.ERI3" in doc and "indexes NO table row" in doc


def test_eri3_is_built_from_coulomb_only() -> None:
    for op in (Operator.OVERLAP, Operator.KINETIC, Operator.NUCLEAR):
        with pytest.raises(ValueError, match="built from operator coulomb"):
            Spec(op, 0, (1, 1, 1), Target.ERI3)


@pytest.mark.parametrize("am", [(2, 1), (2, 1, 3, 0), (1, 0, 1, 0)])
def test_eri3_refuses_a_pair_or_a_quartet_naming_triple(am) -> None:
    with pytest.raises(ValueError, match=r"takes a triple class \(3 shells\)"):
        _eri3(am)


@pytest.mark.parametrize("am", [(2,), (2, 1, -1), (2, 1, 1.0), ()])
def test_eri3_refuses_a_malformed_am(am) -> None:
    with pytest.raises(ValueError):
        _eri3(am)


def test_a_triple_is_refused_by_every_other_target_naming_its_shape() -> None:
    # Both directions: a three-shell am on a pair or quartet target names
    # that target's shape, never "triple".
    for t in Target:
        if t is Target.ERI3:
            continue
        shape = {2: "pair", 4: "quartet"}[AM_LENGTH[t]]
        with pytest.raises(ValueError, match=f"takes a {shape} class") as err:
            Spec(OPERATOR_OF[t], 0, (1, 1, 1), t)
        assert "triple" not in str(err.value)


@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_a_batched_eri3_spec_is_refused_by_decision(n: int) -> None:
    with pytest.raises(ValueError, match="unbatched by decision") as err:
        _eri3(n_batches=n)
    assert "triple target eri3" in str(err.value)
    why = _PAIR_BATCH_REFUSAL[Target.ERI3]
    assert "Hermite targets" in why and "structural reason does not" in why
    assert "never batched" not in str(err.value)  # not the structural one
    assert _eri3().n_batches == 1


def test_an_eri3_key_never_collides_with_any_other_target_key() -> None:
    ls = range(4)
    eri3 = {_eri3(am).key() for am in itertools.product(ls, repeat=3)}
    assert len(eri3) == 64
    others = set()
    for t in Target:
        if t is Target.ERI3:
            continue
        shapes = itertools.product(ls, repeat=AM_LENGTH[t])
        others |= {Spec(OPERATOR_OF[t], 0, am, t).key() for am in shapes}
    assert eri3.isdisjoint(others)
    # The quartet it runs, (la lb|lP 0), is a different key: a triple's ket
    # holds one shell, a quartet's two.
    assert _eri3((2, 1, 3)).key() != _spec(am=(2, 1, 3, 0)).key()
    assert all(k.split("/")[2].count(",") == 1 for k in eri3)


def test_the_docstring_says_the_triple_is_not_a_quartet_minus_a_shell() -> None:
    import felizjueves.spec as spec_module

    doc = spec_module.__doc__
    assert "AN ERI3 CLASS IS A TRIPLE THAT RUNS A QUARTET" in doc
    assert "NOT an AO quartet with a shell dropped" in doc


# #518: the dipole matrix <a| q (r - C) |b>, a pair (la, lb) on its own
# operator, never batched for S's structural reason. The multipole order
# e = 1 is the builder's argument: no field, so no key spells it.


def _dipole(am=(2, 1), **kw) -> Spec:
    return Spec(Operator.DIPOLE, 0, am, Target.DIPOLE, **kw)


def test_the_dipole_key_is_a_dipole_pair_and_round_trips() -> None:
    a = _dipole()
    assert a.key() == "dipole/d0/(2,1)/b1/dipole/literal"
    assert a.key() != _dipole((1, 2)).key()
    assert a.total_am == 3
    assert [f.name for f in dataclasses.fields(Spec)] == [
        "operator", "derivative_order", "am", "target", "levers",
        "n_batches", "engine"]  # no multipole-order field
    for la in range(5):
        for lb in range(5):
            s = _dipole((la, lb))
            op, d, cls, b, t, lv = s.key().split("/")
            assert (op, cls, t, lv) == (
                "dipole", f"({la},{lb})", "dipole", "literal")
            back = Spec(Operator(op), int(d[1:]),
                        tuple(int(x) for x in cls[1:-1].split(",")),
                        Target(t), n_batches=int(b[1:]))
            assert back == s and back.key() == s.key()
            assert target_of(f"{s.key()}@sm_86/nvcc 13.0") is Target.DIPOLE


def test_the_dipole_takes_its_own_operator_and_a_pair() -> None:
    for op in Operator:
        if op is Operator.DIPOLE:
            continue
        with pytest.raises(ValueError, match="built from operator dipole"):
            Spec(op, 0, (1, 1), Target.DIPOLE)
    for am in ((2, 1, 0, 0), (2, 1, 0), (2,), (2, -1)):
        with pytest.raises(ValueError):
            _dipole(am)
    with pytest.raises(ValueError, match="derivative order"):
        Spec(Operator.DIPOLE, 1, (2, 1), Target.DIPOLE)


@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_a_batched_dipole_is_refused_structurally(n: int) -> None:
    with pytest.raises(ValueError) as info:
        _dipole(n_batches=n)
    msg = str(info.value)
    assert "pair target dipole" in msg
    assert "never batched" in msg and "partitions the Hermite targets" in msg
    assert "decision" not in msg
    assert _PAIR_BATCH_REFUSAL[Target.DIPOLE] == _PAIR_BATCH_REFUSAL[Target.S]
    assert _dipole().n_batches == 1


def test_the_dipole_admits_hgp_as_s_does() -> None:
    # one_electron.kernel_hgp: S, T, V and the dipole are legal on HGP, keyed
    # by the engine slot; the native engine stays MMD.
    hgp = _dipole(engine=Engine.HGP)
    assert hgp.engine is Engine.HGP and hgp.target_slot == "dipole.hgp"
    assert _dipole().engine is Engine.MMD
    assert hgp.key() == _dipole().key().replace("/dipole/literal",
                                                "/dipole.hgp/literal")


def test_a_dipole_key_never_collides_with_any_other_target_key() -> None:
    ls = range(4)
    dipole = {_dipole((a, b)).key() for a in ls for b in ls}
    assert len(dipole) == 16
    others = set()
    for t in Target:
        if t is Target.DIPOLE:
            continue
        shapes = itertools.product(ls, repeat=AM_LENGTH[t])
        others |= {Spec(OPERATOR_OF[t], 0, am, t).key() for am in shapes}
    assert dipole.isdisjoint(others)
    # Not S's key either, though both are one-electron pairs of one shape.
    assert _dipole().key() != Spec(Operator.OVERLAP, 0, (2, 1), Target.S).key()


def test_the_docstring_says_the_order_is_not_in_the_key() -> None:
    import felizjueves.spec as spec_module

    doc = spec_module.__doc__
    assert "A DIPOLE CLASS IS A ONE-ELECTRON PAIR" in doc
    assert "IS A BUILDER ARGUMENT, NOT A KEY FIELD" in doc
    assert "§9.5 (docs/references.md entry" in doc
