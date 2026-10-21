"""#196: `Target.K`, a quartet class named in the Hermite pairing.

A K class's `am` is `(la, lc, lb, ld)` -- bra pair (a, c), ket pair (b, d) --
not the AO quartet order. These tests pin the widening (no new field, the
J/DQ/JAB keys unmoved), the shape refusal, and the one mistake the sum cannot
catch: the AO order names a different (lp, lq) split whenever lb != lc.
Fast: constructs keys only; builds, interprets and compiles nothing.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from felizjueves.kernel import k_spec, pq_spec, split
from felizjueves.spec import AM_LENGTH, Operator, Spec, Target, is_batch_count


def _k(am: tuple[int, ...], **kw) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, **kw)


def test_k_is_a_widening_not_a_new_field() -> None:
    assert Target.K == "k" and AM_LENGTH[Target.K] == 4
    assert [f.name for f in dataclasses.fields(Spec)] == [
        "operator", "derivative_order", "am", "target", "levers", "n_batches",
        "engine",  # #401's field, the recurrence; K added none
    ]


@pytest.mark.parametrize("am", [(1, 0), (0, 0), (2, 1)])
def test_a_k_spec_with_a_pair_am_is_refused_naming_the_shape(am) -> None:
    # As a pair target with a quartet is refused today (#150).
    with pytest.raises(ValueError, match=r"quartet class.*\(la, lc \| lb, ld\)"):
        _k(am)


@pytest.mark.parametrize("am", [(1, 1, 0), (1, 0, 0, 0, 0), ()])
def test_a_k_spec_with_any_other_length_is_refused(am) -> None:
    with pytest.raises(ValueError):
        _k(am)


@pytest.mark.parametrize("n", [1, 3, 6, 9, 12])
def test_k_admits_batching_exactly_as_j_does(n: int) -> None:
    assert _k((1, 0, 1, 0), n_batches=n).n_batches == n
    assert pq_spec(2, n_batches=n).n_batches == n


@pytest.mark.parametrize("n", [0, 2, 4, 5, -3])
def test_k_refuses_the_batch_counts_j_refuses(n: int) -> None:
    assert not is_batch_count(n)
    with pytest.raises(ValueError):
        _k((1, 0, 1, 0), n_batches=n)


def test_a_k_key_spells_the_hermite_quartet_and_its_target() -> None:
    assert _k((1, 0, 0, 1)).key() == "coulomb/d0/(1,0|0,1)/b1/k/literal"
    assert _k((1, 1, 0, 0), n_batches=3, levers=("recycle",)).key() == (
        "coulomb/d0/(1,1|0,0)/b3/k/recycle"
    )


def test_a_k_key_never_collides_with_a_j_dq_or_jab_key() -> None:
    ls = range(3)
    quartets = list(itertools.product(ls, repeat=4))
    k = {_k(am).key() for am in quartets}
    j = {Spec(Operator.COULOMB, 0, am, Target.J).key() for am in quartets}
    pairs = {
        Spec(Operator.COULOMB, 0, (a, b), t).key()
        for t in (Target.DQ, Target.JAB) for a in ls for b in ls
    }
    assert len(k) == len(quartets)
    assert k.isdisjoint(j) and k.isdisjoint(pairs)


@pytest.mark.parametrize("L", range(17))
def test_k_spec_splits_l_as_pq_spec_does(L: int) -> None:
    # Pure arithmetic on the key -- no kernel is built at any L.
    s = k_spec(L)
    assert s.target is Target.K and s.n_batches == 1 and s.levers == ()
    assert s.total_am == L == pq_spec(L).total_am
    assert split(s) == split(pq_spec(L)) == ((L + 1) // 2, L // 2)
    la, lc, lb, ld = s.am
    # Each pair's order distributed over its two shells, larger first.
    assert (la, lc) == ((la + lc + 1) // 2, (la + lc) // 2)
    assert (lb, ld) == ((lb + ld + 1) // 2, (lb + ld) // 2)


def test_k_spec_small_cells_are_spelled_out() -> None:
    assert k_spec(0).am == (0, 0, 0, 0)
    assert k_spec(1).am == (1, 0, 0, 0)
    assert k_spec(2).am == (1, 0, 1, 0)
    assert k_spec(2, n_batches=3).key() == "coulomb/d0/(1,0|1,0)/b3/k/literal"


def test_k_spec_refuses_a_bad_l() -> None:
    for bad in (-1, 1.0, "2", True):
        with pytest.raises(ValueError):
            k_spec(bad)
    with pytest.raises(ValueError):
        k_spec(2, n_batches=2)


@pytest.mark.parametrize(
    "shells", list(itertools.product(range(2), repeat=4)) + [(0, 2, 0, 0)]
)
def test_the_ao_order_names_a_different_split_whenever_lb_differs_from_lc(
    shells: tuple[int, int, int, int],
) -> None:
    # The same four shells a, b, c, d of K_ab = SUM_cd D_cd (ac|bd): the
    # right spelling pairs (a, c) | (b, d); the AO-order mistake pairs
    # (a, b) | (c, d). Same L, so the same table row -- different plan.
    la, lb, lc, ld = shells
    hermite = _k((la, lc, lb, ld))
    ao_mistake = _k((la, lb, lc, ld))
    assert hermite.total_am == ao_mistake.total_am
    assert split(hermite) == (la + lc, lb + ld)
    if lb != lc:
        assert split(hermite) != split(ao_mistake)
    else:
        assert split(hermite) == split(ao_mistake)


def test_total_am_says_k_indexes_the_j_row_and_the_pairing_differs() -> None:
    doc = Spec.total_am.__doc__
    assert "SAME" in doc and "Tables 1-3 row" in doc
    assert "pairing is what differs" in doc
    assert _k((1, 0, 1, 0)).total_am == Spec(
        Operator.COULOMB, 0, (1, 0, 1, 0), Target.J).total_am
