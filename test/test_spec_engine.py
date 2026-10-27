"""#401: `Spec.engine`, the recurrence as a key field.

`Target` names the digestion (`J`/`K` split, `F` fused); `Spec.engine` names
the recurrence (`Engine.MMD`, `Engine.HGP`). Every target has a native engine
(`NATIVE_ENGINE`: F -> HGP, every other target -> MMD), `engine=None`
resolves to it, and the key spells the engine only off-native, as a suffix of
the target slot (`.../j.hgp/...`). HGP is legal on J, K and F only; MMD F is
refused until #326. Everything here is L <= 4.
"""

from __future__ import annotations

import itertools

import pytest

from felizjueves.mmd.kernel import k_spec, pq_spec
from felizjueves.performance.device import MeasuredKey, by_name
from felizjueves.performance.measured_store import engine_of, split_key, target_of
from felizjueves.spec import (
    AM_LENGTH,
    NATIVE_ENGINE,
    OPERATOR_OF,
    Engine,
    Spec,
    Target,
)

_AM = {2: (2, 1), 3: (2, 1, 1), 4: (1, 1, 1, 0)}
_CARDS = (("sm_86", "nvcc 13.0"), ("gfx1200", "hipcc / ROCm 7.2.4"))

# The off-native pairings milestone 14 builds, the HGP DF kernels
# (`eri/density_fit/kernel_hgp.py`), and nothing else.
LEGAL_OFF_NATIVE = {(Target.J, Engine.HGP), (Target.K, Engine.HGP),
                    (Target.METRIC, Engine.HGP), (Target.ERI3, Engine.HGP)}


def _spec(target: Target, engine: Engine | None = None, *,
          am: tuple[int, ...] | None = None, n_batches: int = 1) -> Spec:
    return Spec(OPERATOR_OF[target], 0, am or _AM[AM_LENGTH[target]],
                target, (), n_batches, engine)


# ---------------------------------------------------------------------------
# Native defaults
# ---------------------------------------------------------------------------


def test_the_native_engine_is_hgp_for_f_and_mmd_for_every_other_target(
        ) -> None:
    assert set(NATIVE_ENGINE) == set(Target)
    assert NATIVE_ENGINE[Target.F] is Engine.HGP
    assert all(e is Engine.MMD for t, e in NATIVE_ENGINE.items()
               if t is not Target.F)
    assert [e.value for e in Engine] == ["mmd", "hgp"]


@pytest.mark.parametrize("target", list(Target))
def test_engine_none_resolves_to_the_native_one(target) -> None:
    spec = _spec(target)
    assert spec.engine is NATIVE_ENGINE[target]
    explicit = _spec(target, NATIVE_ENGINE[target])
    assert explicit == spec and hash(explicit) == hash(spec)
    assert explicit.key() == spec.key()
    assert spec.target_slot == target.value


def test_every_existing_constructor_is_unchanged() -> None:
    # The helpers every consumer builds through: no engine argument, the
    # native engine, the key with no suffix.
    for s in (pq_spec(4), pq_spec(4, n_batches=3), k_spec(3),
              Spec(OPERATOR_OF[Target.F], 0, (1, 1, 1, 0), Target.F)):
        assert s.engine is NATIVE_ENGINE[s.target]
        assert "." not in s.key().split("/")[4]


def test_an_engine_that_is_not_an_engine_is_refused() -> None:
    with pytest.raises(ValueError, match="is not an Engine"):
        _spec(Target.J, "hgp")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The key spelling
# ---------------------------------------------------------------------------


def test_the_key_spells_an_off_native_engine_as_a_slot_suffix() -> None:
    j = _spec(Target.J, Engine.HGP, am=(1, 1, 1, 0))
    assert j.key() == "coulomb/d0/(1,1|1,0)/b1/j.hgp/literal"
    assert j.target_slot == "j.hgp"
    k = _spec(Target.K, Engine.HGP, am=(1, 1, 1, 0), n_batches=3)
    assert k.key() == "coulomb/d0/(1,1|1,0)/b3/k.hgp/literal"
    # Still the same six segments, and no '@' (the arch separator).
    for s in (j, k):
        assert len(s.key().split("/")) == len(_spec(Target.J).key().split("/"))
        assert "@" not in s.key()


def test_hgp_k_keeps_k_s_pairing_not_f_s_ao_order() -> None:
    # `am` is the pairing the recurrence uses: (ac|bd) for K under either
    # engine. The same am names the same Hermite class, MMD or HGP.
    mmd, hgp = _spec(Target.K, am=(2, 0, 1, 1)), _spec(
        Target.K, Engine.HGP, am=(2, 0, 1, 1))
    assert mmd.am == hgp.am
    assert mmd.key().replace("/k/", "/k.hgp/") == hgp.key()


def test_an_hgp_j_key_never_collides_with_an_mmd_j_key() -> None:
    mmd, hgp = set(), set()
    for am in itertools.product(range(2), repeat=4):  # L <= 4
        for n in (1, 3):
            for target in (Target.J, Target.K):
                mmd.add(_spec(target, am=am, n_batches=n).key())
                hgp.add(_spec(target, Engine.HGP, am=am, n_batches=n).key())
    assert len(mmd) == len(hgp) == 2 * 2 * 16
    assert not mmd & hgp
    for a, tc in _CARDS:
        m = {MeasuredKey(_spec(Target.J, am=(1, 1, 1, 0)), by_name(a),
                         tc).key()}
        h = {MeasuredKey(_spec(Target.J, Engine.HGP, am=(1, 1, 1, 0)),
                         by_name(a), tc).key()}
        assert not m & h


def test_specs_differing_only_in_engine_are_different_specs() -> None:
    assert _spec(Target.J) != _spec(Target.J, Engine.HGP)
    assert len({_spec(Target.J), _spec(Target.J, Engine.HGP)}) == 2


def test_the_engine_survives_without_kernel_body_levers() -> None:
    s = _spec(Target.J, Engine.HGP)
    assert s.without_kernel_body_levers() is s
    lv = Spec(s.operator, 0, s.am, s.target, ("ordered",), 1, Engine.HGP)
    assert lv.without_kernel_body_levers() == s


# ---------------------------------------------------------------------------
# Every refusal, with its reason
# ---------------------------------------------------------------------------


_PAIRINGS = [(t, e) for t in Target for e in Engine
             if e is not NATIVE_ENGINE[t]]


@pytest.mark.parametrize(("target", "engine"), _PAIRINGS,
                         ids=[f"{t.value}-{e.value}" for t, e in _PAIRINGS])
def test_every_off_native_pairing_is_legal_or_refused_with_its_reason(
        target, engine) -> None:
    if (target, engine) in LEGAL_OFF_NATIVE:
        assert _spec(target, engine).engine is engine
        return
    with pytest.raises(ValueError) as err:
        _spec(target, engine)
    msg = str(err.value)
    head = f"engine {engine.value} on target {target.value}: "
    assert msg.startswith(head) and len(msg) > len(head) + 20


def test_mmd_f_is_refused_until_326() -> None:
    with pytest.raises(ValueError, match="#326"):
        _spec(Target.F, Engine.MMD, am=(1, 1, 1, 0))


@pytest.mark.parametrize("target", [Target.DQ, Target.JAB, Target.S,
                                    Target.T, Target.V, Target.SX,
                                    Target.SCHWARZ, Target.PAIR_BOUND,
                                    Target.RECORD_BOUND])
def test_hgp_is_refused_on_every_target_but_j_k_f_metric_eri3(target) -> None:
    with pytest.raises(ValueError, match=f"engine hgp on target {target}"):
        _spec(target, Engine.HGP)


def test_the_other_refusals_still_apply_under_hgp() -> None:
    with pytest.raises(ValueError, match="quartet"):
        _spec(Target.J, Engine.HGP, am=(1, 1))
    with pytest.raises(ValueError, match="multiple of 3"):
        _spec(Target.K, Engine.HGP, n_batches=2)


# ---------------------------------------------------------------------------
# target_of / engine_of round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", list(Target))
@pytest.mark.parametrize("engine", list(Engine))
def test_target_of_and_engine_of_round_trip(target, engine) -> None:
    try:
        spec = _spec(target, engine)
    except ValueError:
        return  # refused above, never keyed
    for a, tc in _CARDS:
        key = MeasuredKey(spec, by_name(a), tc).key()
        assert split_key(key) == (spec.key(), a, tc)
        assert target_of(key) is target
        assert engine_of(key) is engine


@pytest.mark.parametrize("bad", [
    "coulomb/d0/(1,1|1,0)/b1/j.mmd/literal@sm_86/nvcc 13.0",  # native spelt
    "coulomb/d0/(1,1|1,0)/b1/f.hgp/literal@sm_86/nvcc 13.0",  # native spelt
    "coulomb/d0/(1,1|1,0)/b1/j.xyz/literal@sm_86/nvcc 13.0",
    "coulomb/d0/(1,1|1,0)/b1/q.hgp/literal@sm_86/nvcc 13.0",
])
def test_a_slot_no_spec_spells_is_refused(bad) -> None:
    for read in (target_of, engine_of):
        with pytest.raises(ValueError):
            read(bad)
