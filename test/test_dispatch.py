"""`dispatch`: the batch count of a class, from measured wall-clock (#145).

Fast tier, no device markers: no compiler, no card, and no tracked row --
every case builds its own `Measured` rows. The tracked cases (the resolver
recomputing data/measured/batch_recommendation.json from kernels.json, the
default store being the tracked one, the tracked answer's input dependence,
and the tracked cards being characterised) are checks of measurement
artefacts: measure/measure_dispatch.py (`devtools/measure.sh check`).

DEFERRED to #144: running one on-device cell at the dispatched count; that
issue writes the on-device whole-molecule test, and `dispatch.choose(lp, lq,
arch, input)` is the entry point it calls.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.pairs import pair_bound
from felizjueves.launch import class_spec, launches
from felizjueves.mmd.kernel import build_pq_kernels, pq_spec
from felizjueves.performance import batchreport, dispatch
from felizjueves.performance import measured_store as measured
from felizjueves.performance.device import (
    SM_86,
    Arch,
    MeasuredKey,
    Uncharacterised,
    Vendor,
)
from felizjueves.reference.pq import one_centre

BATCHES = batchreport.BATCHES


def _split(L: int) -> tuple[int, int]:
    return ((L + 1) // 2, L // 2)


def _store(wide: dict[int, tuple[int, int]],
           near: dict[int, tuple[int, int]] | None = None, *, L: int = 13,
           arch: Arch = SM_86) -> dict[str, measured.Measured]:
    near = wide if near is None else near
    out = {}
    for b in BATCHES:
        (wm, wi), (nm, ni) = wide[b], near[b]
        out[MeasuredKey(pq_spec(L, n_batches=b), arch, "tc").key()] = (
            measured.Measured(
                runtime_ns=wm, repeats=31, spread_ns=wi, spill_store=0,
                spill_load=0, registers=255, toolchain="tc", arch=arch.name,
                host="h", gpu="g", date="2026-10-12",
                raw={"wallclock:near:median_ns": nm,
                     "wallclock:near:q1_ns": 0, "wallclock:near:q3_ns": ni}))
    return out


# --- the rule, on synthetic rows ----------------------------------------------


def test_the_choice_is_the_best_median_and_carries_its_spread() -> None:
    wide = {1: (1000, 10), 3: (990, 20), 6: (900, 5), 9: (905, 30),
            12: (2000, 5)}
    c = dispatch.choose(*_split(13), SM_86, "wide", store=_store(wide))
    assert c.measured and c.n_batches == 6
    assert (c.median_ns, c.iqr_ns, c.repeats) == (900, 5, 31)
    assert (c.runner_up, c.margin_ns) == (9, 5)
    assert not c.resolved  # 5 <= max(5, 30)
    assert c.within_spread == (6, 9)
    assert "best median" in c.reason


def test_a_tie_goes_to_the_fewer_batches() -> None:
    wide = {1: (1000, 1), 3: (500, 1), 6: (500, 1), 9: (500, 1), 12: (600, 1)}
    c = dispatch.choose(*_split(13), SM_86, "wide", store=_store(wide))
    assert c.n_batches == 3 and c.runner_up == 6 and c.margin_ns == 0


def test_the_input_axis_is_explicit_and_never_averaged() -> None:
    wide = {b: (1000 + b, 1) for b in BATCHES}       # 1 wins
    near = {b: (5000 - b, 1) for b in BATCHES}       # 12 wins
    store = _store(wide, near)
    lp, lq = _split(13)
    assert dispatch.choose(lp, lq, SM_86, "wide", store=store).n_batches == 1
    assert dispatch.choose(lp, lq, SM_86, "near", store=store).n_batches == 12
    for bad in (None, "both", "mean", ""):
        with pytest.raises(ValueError, match="never averaged"):
            dispatch.choose(lp, lq, SM_86, bad, store=store)
        with pytest.raises(ValueError):
            dispatch.Resolver(SM_86, bad, store=store)


def test_wall_clock_only_spills_and_registers_move_nothing() -> None:
    wide = {b: (1000 - b, 1) for b in BATCHES}
    store = _store(wide)
    moved = {k: replace(r, spill_store=10**6, spill_load=10**6, registers=1)
             for k, r in store.items()}
    lp, lq = _split(13)
    assert (dispatch.choose(lp, lq, SM_86, "wide", store=store)
            == dispatch.choose(lp, lq, SM_86, "wide", store=moved))


def test_a_class_with_no_row_falls_back_to_one_and_says_so() -> None:
    store = _store({b: (1000 - b, 1) for b in BATCHES})  # L = 13 only: 12 wins
    # not a measured split at all
    c = dispatch.choose(2, 2, SM_86, "wide", store=store)
    assert (c.n_batches, c.measured) == (dispatch.FALLBACK, False) == (1, False)
    assert "not a measured split" in c.reason and c.median_ns is None
    # the OTHER split of a measured L is not read off that row
    lp, lq = _split(13)
    assert dispatch.choose(lp, lq, SM_86, "wide", store=store).n_batches == 12
    c = dispatch.choose(lq, lp, SM_86, "wide", store=store)
    assert (c.n_batches, c.measured) == (1, False)
    # a measured split with no row on this card
    c = dispatch.choose(*_split(16), SM_86, "wide", store=store)
    assert (c.n_batches, c.measured) == (1, False)
    assert "no row on sm_86" in c.reason


def test_a_partly_timed_class_is_refused_not_chosen_over_the_rest() -> None:
    store = _store({b: (1000 - b, 1) for b in BATCHES})
    key = MeasuredKey(pq_spec(13, n_batches=12), SM_86, "tc").key()
    gone = {k: v for k, v in store.items() if k != key}
    with pytest.raises(KeyError):
        dispatch.choose(*_split(13), SM_86, "wide", store=gone)


@pytest.mark.parametrize("which", dispatch.INPUTS)
def test_an_uncharacterised_card_gets_no_choice(which) -> None:
    bare = Arch(Vendor.NVIDIA, "sm_86", wavefront=32)  # 0.0: not characterised
    assert not bare.characterised
    rows = _store({b: (1000 + b, 1) for b in BATCHES}, arch=bare)
    for lp, lq in [*dispatch.measured_classes(), (0, 0)]:
        with pytest.raises(Uncharacterised):
            dispatch.choose(lp, lq, bare, which, store=rows)
    with pytest.raises(Uncharacterised):
        dispatch.Resolver(bare, which, store=rows)
    with pytest.raises(Uncharacterised):
        dispatch.table(bare, which, store=rows)


# --- the tracked file, recomputed from the store ------------------------------


def test_the_measured_classes_are_the_subgrid_splits() -> None:
    assert dispatch.measured_classes() == {
        _split(L): L for L in measured.SUBGRID_LS}


# --- injected into the launch plan --------------------------------------------

BOUND = partial(pair_bound, one_centre=one_centre)


@cache
def _h2o():
    fx = manifest.load(manifest.FIXTURES / "h2o_sto3g_j.json")
    return ob.from_fixture(fx), np.array(fx.density)


def _plan(**kw):
    basis, D = _h2o()
    return launches(basis, D, **{"bound": BOUND, "n_batches": 1, "tau": 0.0,
                                 **kw})


def test_launches_builds_each_class_at_the_injected_count() -> None:
    asked = []

    def resolve(lp: int, lq: int) -> int:
        asked.append((lp, lq))
        return 3 if (lp, lq) == (2, 2) else 1

    P = _plan(resolve=resolve)
    assert sorted(asked) == sorted((c.lp, c.lq) for c in P)
    for c in P:
        n = 3 if (c.lp, c.lq) == (2, 2) else 1
        assert c.n_batches == n
        assert c.kernels == build_pq_kernels(class_spec(c.lp, c.lq, n))
    assert any((c.lp, c.lq) == (2, 2) for c in P)


def test_without_a_resolver_the_plan_is_unchanged() -> None:
    assert _plan() == _plan(resolve=lambda lp, lq: 1)


def test_an_injected_count_the_plan_refuses_is_refused_as_a_fixed_one() -> None:
    with pytest.raises(ValueError, match="the plan refuses"):
        _plan(resolve=lambda lp, lq: 7)  # not a batch count the plan admits
    P = _plan(resolve=lambda lp, lq: 7, strict=False)
    assert all(c.n_batches == 1 for c in P)
    assert {(lp, lq) for lp, lq, _ in P.refused} == {(c.lp, c.lq) for c in P}


def test_the_dispatch_resolver_on_h2o_falls_back_everywhere() -> None:
    # STO-3G reaches no measured split (L <= 4): every class is built at 1,
    # and the resolver says why for each.
    res = dispatch.Resolver(SM_86, "near", store=_store(
        {b: (1000 + b, 1) for b in BATCHES}))  # a measured L = 13 class
    P = _plan(resolve=res)
    assert P == _plan()
    assert all(not res.choice(c.lp, c.lq).measured for c in P)
