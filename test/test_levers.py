"""#337: the kernel-body lever vocabulary, its legality table, and the seam.

`levers.VOCABULARY` ships EMPTY -- no lever's behaviour lands with #337 -- so
the seam is exercised here with a PROBE lever declared by the test itself
(`monkeypatch.setitem`), the way each later lever issue declares its entry.
The probe changes nothing but the provenance `Note` every builder records.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from felizjueves import levers
from felizjueves.eri.k_exchange.kernel_k import build_k_kernel, build_k_kernels
from felizjueves.ir import Note
from felizjueves.ir.lower import LEVERS as LOWERING
from felizjueves.mmd.kernel import (
    build_pq_kernel,
    build_pq_kernels,
    builder_plan,
    k_spec,
    lever_notes,
    pq_spec,
)
from felizjueves.plan.pipeline import TUNED, plan
from felizjueves.plan.sieve import NO_SIEVE
from felizjueves.spec import (
    AM_LENGTH,
    NATIVE_ENGINE,
    OPERATOR_OF,
    Engine,
    Spec,
    Target,
)

PROBE = "probe"
K_REASON = "the probe's own reason for K"


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> levers.Lever:
    lever = levers.Lever.of(PROBE, 337, "the seam's witness; no statement "
                            "changes", {"j", "k"})
    monkeypatch.setitem(levers.VOCABULARY, PROBE, lever)
    return lever


@pytest.fixture
def probe_j_only(monkeypatch: pytest.MonkeyPatch) -> levers.Lever:
    lever = levers.Lever.of(PROBE, 337, "J only", {"j"}, k=K_REASON)
    monkeypatch.setitem(levers.VOCABULARY, PROBE, lever)
    return lever


def _spec(target: Target, *lv: str, n_batches: int = 1,
          engine: Engine | None = None) -> Spec:
    am = {2: (2, 1), 3: (2, 1, 1), 4: (1, 0, 1, 0)}[AM_LENGTH[target]]
    return Spec(OPERATOR_OF[target], 0, am, target, tuple(lv), n_batches,
                engine)


# ---------------------------------------------------------------------------
# The vocabulary, as declared
# ---------------------------------------------------------------------------


def test_the_target_slots_are_spec_s_targets() -> None:
    assert levers.TARGET_SLOTS == tuple(t.value for t in Target)
    assert levers.SLOTS == levers.TARGET_SLOTS + levers.ENGINE_SLOTS
    assert set(levers.DEFAULT_REFUSALS) == set(levers.SLOTS)
    assert all(levers.DEFAULT_REFUSALS.values())


def test_the_engine_slots_are_spec_s_legal_off_native_pairings() -> None:
    # #401: every (target, engine) Spec admits off its native engine is an
    # engine slot, spelled as the key spells it, and nothing else is.
    off_native = set()
    for target in Target:
        for engine in Engine:
            if engine is NATIVE_ENGINE[target]:
                continue
            try:
                spec = _spec(target, engine=engine)
            except ValueError:
                continue
            off_native.add(spec.target_slot)
    assert off_native == set(levers.ENGINE_SLOTS) == {
        "j.hgp", "k.hgp", "s.hgp", "t.hgp", "v.hgp", "dipole.hgp",
        "metric.hgp", "eri3.hgp"}


def test_the_other_layers_names_are_their_owners() -> None:
    assert set(levers.OTHER_LAYERS) == {NO_SIEVE, TUNED} | set(LOWERING)


def test_every_declared_lever_is_well_formed() -> None:
    # Empty at #337; every later entry is held to this.
    for name, lever in levers.VOCABULARY.items():
        assert lever.name == name
        assert name not in levers.OTHER_LAYERS
        assert lever.legal | set(lever.refusals) == set(levers.SLOTS)


def test_no_lever_behaviour_lands_with_the_seam() -> None:
    # Empty at #337; every entry since is a later lever issue's own.
    assert all(lv.issue != 337 for lv in levers.VOCABULARY.values())


# ---------------------------------------------------------------------------
# A declaration is refused unless its table is total and its name free
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("args", "match"), [
    (("", 1, "x", {"j"}), "non-empty"),
    (("a+b", 1, "x", {"j"}), "without '\\+'"),
    ((NO_SIEVE, 1, "x", {"j"}), "owned by the plan layer"),
    (("tuned", 1, "x", {"j"}), "owned by the plan layer"),
    (("p", 0, "x", {"j"}), "issue"),
    (("p", 1, "", {"j"}), "must say what it changes"),
    (("p", 1, "x", set()), "legal on no target"),
])
def test_a_malformed_declaration_is_refused(args, match) -> None:
    with pytest.raises(ValueError, match=match):
        levers.Lever.of(*args)


def test_an_incomplete_or_overlapping_table_is_refused() -> None:
    with pytest.raises(ValueError, match=r"missing \['f'\]"):
        levers.Lever("p", 1, "x", frozenset({"j"}),
                     {s: "no" for s in levers.SLOTS
                      if s not in ("j", "f")})
    with pytest.raises(ValueError, match=r"missing \['k.hgp'\]"):
        levers.Lever("p", 1, "x", frozenset({"j"}),
                     {s: "no" for s in levers.SLOTS
                      if s not in ("j", "k.hgp")})
    with pytest.raises(ValueError, match="unknown \\['z'\\]"):
        levers.Lever("p", 1, "x", frozenset({"j"}),
                     {**{s: "no" for s in levers.SLOTS if s != "j"},
                      "z": "no"})
    with pytest.raises(ValueError, match="both legal and refused"):
        levers.Lever("p", 1, "x", frozenset({"j"}),
                     {s: "no" for s in levers.SLOTS})
    with pytest.raises(ValueError, match="states no reason"):
        levers.Lever("p", 1, "x", frozenset({"j"}),
                     {s: "" for s in levers.SLOTS if s != "j"})
    with pytest.raises(ValueError, match="legal or no target"):
        levers.Lever.of("p", 1, "x", {"j"}, j="but it is legal")


# ---------------------------------------------------------------------------
# Spec: each illegal (target, lever) refused, with its reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [t for t in Target
                                    if t not in (Target.J, Target.K)])
def test_each_illegal_target_refuses_with_its_stated_reason(
        probe, target) -> None:
    why = levers.DEFAULT_REFUSALS[target.value]
    with pytest.raises(ValueError) as err:
        _spec(target, PROBE)
    assert str(err.value) == f"lever {PROBE!r} on target {target}: {why}"


def test_a_lever_s_own_reason_beats_the_default(probe_j_only) -> None:
    with pytest.raises(ValueError) as err:
        _spec(Target.K, PROBE)
    assert str(err.value) == f"lever {PROBE!r} on target k: {K_REASON}"
    assert _spec(Target.J, PROBE).levers == (PROBE,)


def test_a_name_outside_the_vocabulary_passes_to_its_own_layer() -> None:
    # The plan, lowering and unknown names are judged where they always were.
    for target in Target:
        for lv in (NO_SIEVE, TUNED, *LOWERING, "anything"):
            assert levers.refusal(target, lv) is None
            assert _spec(target, lv).levers == (lv,)


def test_a_legal_lever_is_spelled_in_the_key(probe) -> None:
    spec = pq_spec(4, (PROBE,), n_batches=3)
    assert spec.key() == "coulomb/d0/(2,0|2,0)/b3/j/probe"
    assert pq_spec(4, (NO_SIEVE, PROBE)).key().endswith("/no-sieve+probe")


# ---------------------------------------------------------------------------
# The builders: the lever reaches the Kernel; the plan never sees it
# ---------------------------------------------------------------------------


def _without_lever_notes(kernel, spec):
    notes = set(lever_notes(spec))
    return tuple(s for s in kernel.body if s not in notes)


def test_the_plan_layer_never_sees_a_kernel_lever(probe) -> None:
    spec = pq_spec(4, (PROBE,))
    with pytest.raises(ValueError, match="no lever"):
        plan(spec)  # the plan layer owns no kernel lever, and says so
    assert _dags(builder_plan(spec)) == _dags(plan(pq_spec(4)))


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_a_plan_lever_beside_a_kernel_lever_still_reaches_the_plan(
        probe) -> None:
    # at L = 7 the sieve bites
    both = pq_spec(7, (NO_SIEVE, PROBE))
    assert _dags(builder_plan(both)) == _dags(plan(pq_spec(7, (NO_SIEVE,))))
    assert _dags(plan(pq_spec(7, (NO_SIEVE,)))) != _dags(plan(pq_spec(7)))


def _dags(p) -> tuple:
    return tuple((d.batch, d.targets, dict(d.parents)) for d in p.dags)


def test_no_lever_no_note() -> None:
    for spec in (pq_spec(4), pq_spec(4, (NO_SIEVE,))):
        assert lever_notes(spec) == ()
        assert spec.kernel_body_levers() == ()
        assert spec.without_kernel_body_levers() is spec


def test_the_spec_splits_its_own_levers(probe) -> None:
    spec = pq_spec(7, (NO_SIEVE, PROBE), n_batches=3)
    assert spec.kernel_body_levers() == (PROBE,)
    assert spec.without_kernel_body_levers() == pq_spec(7, (NO_SIEVE,), 3)


@pytest.mark.parametrize("n_batches", [1, 3])
def test_a_lever_reaches_the_pq_kernel_as_provenance(probe, n_batches) -> None:
    spec = pq_spec(4, (PROBE,), n_batches=n_batches)
    note = Note(levers.provenance(PROBE))
    assert note.prov == ("lever probe (#337): the seam's witness; no "
                         "statement changes")
    assert lever_notes(spec) == (note,)
    literal = build_pq_kernels(pq_spec(4, n_batches=n_batches))
    built = build_pq_kernels(spec)
    assert len(built) == len(literal) == n_batches
    for got, want in zip(built, literal, strict=True):
        assert got.body[1] == note  # right under the kernel's head line
        assert got.name == want.name and got.params == want.params
        assert _without_lever_notes(got, spec) == want.body
    if n_batches == 1:
        assert build_pq_kernel(spec) == built[0]


@pytest.mark.parametrize("n_batches", [1, 3])
def test_a_lever_reaches_the_k_kernel_as_provenance(probe, n_batches) -> None:
    literal_spec = k_spec(4, n_batches=n_batches)
    spec = replace(literal_spec, levers=(PROBE,))
    literal = build_k_kernels(literal_spec)
    built = build_k_kernels(spec)
    assert len(built) == len(literal) == n_batches
    for got, want in zip(built, literal, strict=True):
        assert got.body[1] == Note(levers.provenance(PROBE))
        assert got.name == want.name and got.params == want.params
        assert _without_lever_notes(got, spec) == want.body
    if n_batches == 1:
        assert build_k_kernel(spec) == built[0]


# ---------------------------------------------------------------------------
# #401: legality sees the engine; `ordered` is legal exactly on HGP J / K
# ---------------------------------------------------------------------------

ORDERED = "ordered"


def _every_spec_slot() -> list[tuple[Target, Engine]]:
    """Every (target, engine) a lever-free Spec admits."""
    out = []
    for target in Target:
        for engine in Engine:
            try:
                _spec(target, engine=engine)
            except ValueError:
                continue
            out.append((target, engine))
    return out


def test_ordered_is_declared_on_the_engine_slots_only() -> None:
    lever = levers.VOCABULARY[ORDERED]
    assert (lever.name, lever.issue) == ("ordered", 401)
    assert lever.legal == {"j.hgp", "k.hgp"}
    for slot in ("j", "k", "f"):
        assert lever.refusals[slot] != levers.DEFAULT_REFUSALS[slot]
    for slot in set(levers.TARGET_SLOTS) - {"j", "k", "f"}:
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]


def test_ordered_is_legal_exactly_on_hgp_j_and_k() -> None:
    legal = set()
    for target, engine in _every_spec_slot():
        try:
            spec = _spec(target, ORDERED, engine=engine)
        except ValueError as err:
            slot = _spec(target, engine=engine).target_slot
            assert str(err) == (f"lever {ORDERED!r} on target {slot}: "
                                f"{levers.VOCABULARY[ORDERED].refusals[slot]}")
            continue
        legal.add((target, engine))
        assert spec.key().endswith(f"/{spec.target_slot}/ordered")
    assert legal == {(Target.J, Engine.HGP), (Target.K, Engine.HGP)}


@pytest.mark.parametrize(("target", "match"), [
    (Target.J, "Hermite split"),
    (Target.K, "Hermite pairs"),
    (Target.F, "8-fold orbit"),
])
def test_ordered_is_refused_on_mmd_j_k_and_on_f_with_its_reason(
        target, match) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(target, ORDERED)


@pytest.mark.parametrize("name", sorted(set(levers.VOCABULARY) - {ORDERED}))
@pytest.mark.parametrize("target", [Target.J, Target.K])
def test_no_other_kernel_lever_is_legal_on_an_hgp_split_slot(
        name, target) -> None:
    # Each earlier lever is MMD's (J, K) or the fused F's: an HGP J / K
    # kernel has neither surface, so its slot's default refusal stands.
    slot = f"{target.value}.hgp"
    assert levers.refusal(slot, name) == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match=f"on target {slot}: "):
        _spec(target, name, engine=Engine.HGP)


def test_legality_is_per_slot_not_per_target(probe) -> None:
    # The probe is legal on MMD j / k (its `legal`), and refused on the same
    # targets under HGP: the slot, not the target, is what the table keys.
    assert _spec(Target.J, PROBE).levers == (PROBE,)
    for target in (Target.J, Target.K):
        with pytest.raises(ValueError, match=f"on target {target.value}.hgp"):
            _spec(target, PROBE, engine=Engine.HGP)
