"""#272: `Target.F`, the fused Fock class, named in AO order.

An F class's `am` is `(la, lb, lc, ld)` -- the HGP VRR pairs the bra (a, b)
and the ket (c, d) as written -- the OPPOSITE of `Target.K`'s Hermite pairing
`(la, lc, lb, ld)`. These tests pin the widening (no new field, every earlier
key unmoved), the shape refusal, the F-specific batch refusal (not the pair
one), the `total_am` docstring's F case, and that the same four shells give
different specs and keys under F and K.
Fast: constructs keys only; builds, interprets and compiles nothing.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from felizjueves.spec import (
    _PAIR_BATCH_REFUSAL,
    AM_LENGTH,
    D1_ONLY_TARGETS,
    OPERATOR_OF,
    Operator,
    Spec,
    Target,
    is_batch_count,
)


def _base_order(target: Target) -> int:
    """The lowest derivative order `target` is keyed at: 1 for a d1-only
    target (#689, `spec.D1_ONLY_TARGETS`), 0 for every other."""
    return 1 if target in D1_ONLY_TARGETS else 0


def _f(am: tuple[int, ...], **kw) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.F, **kw)


def _k(am: tuple[int, ...], **kw) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, **kw)


def test_f_is_a_widening_not_a_new_field() -> None:
    assert Target.F == "f" and AM_LENGTH[Target.F] == 4
    assert OPERATOR_OF[Target.F] is Operator.COULOMB
    assert [f.name for f in dataclasses.fields(Spec)] == [
        "operator", "derivative_order", "am", "target", "levers", "n_batches",
        "engine",  # #401's field, the recurrence; F added none
    ]


@pytest.mark.parametrize("am", [(1, 0), (0, 0), (2, 1)])
def test_an_f_spec_with_a_pair_am_is_refused_naming_the_shape(am) -> None:
    with pytest.raises(
        ValueError, match=r"quartet class \(4 shells\) in AO order \(la, lb \| lc, ld\)"
    ):
        _f(am)


@pytest.mark.parametrize("am", [(1, 1, 0), (1, 0, 0, 0, 0), ()])
def test_an_f_spec_with_any_other_length_is_refused(am) -> None:
    with pytest.raises(ValueError):
        _f(am)


@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_f_refuses_batching_with_its_own_reason_not_the_pair_one(n: int) -> None:
    assert is_batch_count(n)  # legal for J and K; refused for F alone
    with pytest.raises(ValueError) as exc:
        _f((1, 0, 1, 0), n_batches=n)
    msg = str(exc.value)
    assert "for target f" in msg and "HGP class" in msg
    assert "no Hermite targets at all" in msg
    assert "VRR target grouping" in msg  # the field's named future meaning
    for pair_reason in set(_PAIR_BATCH_REFUSAL.values()):
        assert pair_reason not in msg
    assert "pair" not in msg


@pytest.mark.parametrize("n", [0, 2, 4, -3])
def test_f_refuses_the_batch_counts_every_target_refuses(n: int) -> None:
    with pytest.raises(ValueError, match="neither 1 nor a positive multiple of 3"):
        _f((1, 0, 1, 0), n_batches=n)


def test_an_f_key_spells_the_ao_quartet_and_its_target() -> None:
    assert _f((2, 1, 1, 0)).key() == "coulomb/d0/(2,1|1,0)/b1/f/literal"
    assert _f((0, 0, 0, 0)).n_batches == 1


@pytest.mark.parametrize("shells", list(itertools.product(range(3), repeat=4)))
def test_the_same_four_shells_are_different_specs_under_f_and_k(
    shells: tuple[int, int, int, int],
) -> None:
    # The four shells a, b, c, d of one quartet: F spells them (a, b | c, d)
    # as written, K spells them (a, c | b, d). Different spec, different key,
    # whatever the shells -- the target slot alone separates them.
    la, lb, lc, ld = shells
    f = _f((la, lb, lc, ld))
    k = _k((la, lc, lb, ld))
    assert f != k
    assert f.key() != k.key()
    assert f.key().split("/")[4] == "f" and k.key().split("/")[4] == "k"
    # Where lb != lc the classes themselves differ too, not just the slot.
    if lb != lc:
        assert f.am != k.am


def test_an_f_key_never_collides_with_any_other_target_key() -> None:
    ls = range(3)
    quartets = list(itertools.product(ls, repeat=4))
    f = {_f(am).key() for am in quartets}
    assert len(f) == len(quartets)
    others = set()
    for t in Target:
        if t is Target.F:
            continue
        shapes = itertools.product(ls, repeat=AM_LENGTH[t])
        others |= {Spec(OPERATOR_OF[t], _base_order(t), am, t).key() for am in shapes}
    assert f.isdisjoint(others)


def test_total_am_says_f_indexes_no_jengine_row() -> None:
    doc = Spec.total_am.__doc__
    assert "Target.F" in doc and "NO [JEngine] table row" in doc
    assert "never be scored" in doc


def test_the_spec_docstring_states_both_quartet_orderings() -> None:
    import felizjueves.spec as spec

    doc = spec.__doc__
    assert "A K CLASS IS NAMED IN HERMITE ORDER" in doc
    assert "AN F CLASS IS NAMED IN AO ORDER" in doc
    assert "silent in" in doc and "both directions" in doc
