"""#340: `Target.F` on the lever axis -- F's legality and `kernel_f`'s plumbing.

`levers.VOCABULARY` is still EMPTY (no lever's behaviour lands with #340), so
the plumbing is exercised with PROBE levers the test declares itself
(`monkeypatch.setitem`), the way #350-#361 will each declare their entry:
one legal on F alone, and one legal on J and K but not F.

What is held here:
  - F's default refusal is the HGP reason, not a placeholder, and a lever
    whose entry does not name F is refused by `Spec` with it;
  - an F spec carrying a legal lever is keyed, still refuses a batch count,
    and builds: the kernel is the literal one plus one provenance `Note`
    under its head line, for every class with L <= 4, with and without the
    ERI tap, and emits as the literal text plus that one comment;
  - a lever outside the vocabulary (the plan's, lowering's, or any other
    name) is refused by `build_f_kernel`, because no layer around the fused
    kernel would read it.

The literal output's byte-identity is test_emit_fock.py's goldens and
`fock_digests.json`, unchanged by #340. Fast tier: no compiler, no card.
"""

from __future__ import annotations

import pytest

from felizjueves import levers
from felizjueves.emit import cuda, hip
from felizjueves.eri.fock.kernel_f import (
    F_NO_OUTER_LAYER,
    build_f_kernel,
    counted,
    f_class_spec,
    kernel_counts,
)
from felizjueves.hgp import vrr
from felizjueves.ir import Note
from felizjueves.ir.lower import LEVERS as LOWERING
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.plan.pipeline import TUNED
from felizjueves.plan.sieve import NO_SIEVE
from felizjueves.spec import _F_BATCH_REFUSAL, Operator, Spec, Target

F_PROBE = "f-probe"
MMD_PROBE = "mmd-probe"
# The user's cap for milestone 13: every class here has L <= 4.
CLASSES = [c for c in vrr.CLASSES if sum(c) <= 4]


@pytest.fixture
def f_probe(monkeypatch: pytest.MonkeyPatch) -> levers.Lever:
    lever = levers.Lever.of(F_PROBE, 340, "the F seam's witness; no "
                            "statement changes", {"f"})
    monkeypatch.setitem(levers.VOCABULARY, F_PROBE, lever)
    return lever


@pytest.fixture
def mmd_probe(monkeypatch: pytest.MonkeyPatch) -> levers.Lever:
    lever = levers.Lever.of(MMD_PROBE, 340, "a J/K lever", {"j", "k"})
    monkeypatch.setitem(levers.VOCABULARY, MMD_PROBE, lever)
    return lever


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


def test_f_s_default_refusal_is_the_hgp_reason() -> None:
    why = levers.DEFAULT_REFUSALS["f"]
    assert "#340" not in why  # the placeholder #337 left is gone
    assert "HGP" in why and "[p|q]" in why and "Hermite" in why
    assert "n_batches stays 1" in why
    # the batch field's own refusal is unchanged and still F's
    assert "never batched" in _F_BATCH_REFUSAL


def test_a_lever_not_naming_f_is_refused_on_f_with_its_reason(
        mmd_probe) -> None:
    with pytest.raises(ValueError) as err:
        f_class_spec(1, 0, 1, 0, (MMD_PROBE,))
    assert str(err.value) == (f"lever {MMD_PROBE!r} on target f: "
                              f"{levers.DEFAULT_REFUSALS['f']}")


def test_every_declared_lever_not_naming_f_is_refused_on_f() -> None:
    # The real vocabulary (#344's `fused` among it), not a probe.
    for name, lever in levers.VOCABULARY.items():
        if "f" not in lever.legal:
            with pytest.raises(ValueError, match="lever .* on target f: "):
                f_class_spec(1, 0, 1, 0, (name,))


def test_a_lever_naming_f_alone_is_refused_on_every_other_target(
        f_probe) -> None:
    for target in Target:
        if target is not Target.F:
            assert levers.refusal(target, F_PROBE) == \
                levers.DEFAULT_REFUSALS[target.value]
    assert levers.refusal(Target.F, F_PROBE) is None


def test_a_legal_f_lever_is_spelled_in_the_key_and_batches_stay_one(
        f_probe) -> None:
    spec = f_class_spec(2, 1, 1, 0, (F_PROBE,))
    assert spec.key() == "coulomb/d0/(2,1|1,0)/b1/f/f-probe"
    assert spec.kernel_body_levers() == (F_PROBE,)
    assert spec.without_kernel_body_levers() == f_class_spec(2, 1, 1, 0)
    with pytest.raises(ValueError, match="never batched"):
        Spec(Operator.COULOMB, 0, (2, 1, 1, 0), Target.F, (F_PROBE,), 3)


# ---------------------------------------------------------------------------
# The builder: the lever reaches the Kernel as provenance, and nothing else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eris", [False, True])
@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: "".join(map(str, c)))
def test_a_legal_lever_reaches_the_f_kernel_as_provenance(
        f_probe, cls, eris) -> None:
    note = Note(levers.provenance(F_PROBE))
    assert note.prov == ("lever f-probe (#340): the F seam's witness; no "
                         "statement changes")
    literal = build_f_kernel(cls, eris=eris)
    got = build_f_kernel(f_class_spec(*cls, (F_PROBE,)), eris=eris)
    assert got.body[1] == note  # right under the kernel's head line
    assert got.name == literal.name and got.params == literal.params
    assert got.body[:1] + got.body[2:] == literal.body
    assert counted(got) == counted(literal) == kernel_counts(cls)


def test_no_lever_no_note() -> None:
    for cls in CLASSES:
        spec = f_class_spec(*cls)
        assert spec.kernel_body_levers() == ()
        assert build_f_kernel(spec) == build_f_kernel(cls)
        assert not any(isinstance(s, Note) and s.prov.startswith("lever ")
                       for s in build_f_kernel(spec).body)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_the_lever_build_emits_as_the_literal_text_plus_its_note(
        f_probe, backend) -> None:
    emit, arch = {"cuda": (cuda.emit, SM_86), "hip": (hip.emit, GFX1200)}[
        backend]
    cls = (1, 0, 1, 0)
    literal = emit(build_f_kernel(cls), arch).splitlines()
    got = emit(build_f_kernel(f_class_spec(*cls, (F_PROBE,))),
               arch).splitlines()
    extra = [ln for ln in got if "f-probe" in ln]
    assert len(extra) == 1 and len(got) == len(literal) + 1
    i = got.index(extra[0])
    assert got[:i] + got[i + 1:] == literal


@pytest.mark.parametrize("lever", [NO_SIEVE, TUNED, *sorted(LOWERING),
                                   "anything"])
def test_a_lever_no_layer_around_f_reads_is_refused(lever) -> None:
    spec = f_class_spec(1, 0, 1, 0, (lever,))  # Spec passes it on, as ever
    with pytest.raises(ValueError) as err:
        build_f_kernel(spec)
    assert str(err.value) == (f"{spec.key()}: lever {lever!r} "
                              f"{F_NO_OUTER_LAYER}")


def test_beside_a_legal_lever_an_outer_one_is_still_refused(f_probe) -> None:
    spec = f_class_spec(1, 0, 1, 0, tuple(sorted((F_PROBE, NO_SIEVE))))
    with pytest.raises(ValueError, match=f"lever {NO_SIEVE!r} is not"):
        build_f_kernel(spec)
