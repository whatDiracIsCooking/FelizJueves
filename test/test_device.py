"""`Arch`, the card as data, and `MeasuredKey`, what keys a measurement."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from felizjueves.device import (
    ARCHS,
    GFX1200,
    SM_86,
    Arch,
    MeasuredKey,
    Vendor,
    by_name,
    from_env,
)
from felizjueves.spec import Operator, Spec, Target

EMIT_DIR = Path(__file__).resolve().parent.parent / "src" / "felizjueves" / "emit"


def _spec(**kw) -> Spec:
    base = dict(
        operator=Operator.COULOMB, derivative_order=0, am=(4, 4, 4, 4), target=Target.J
    )
    return Spec(**{**base, **kw})


def test_the_declared_archs_are_exactly_this_box() -> None:
    # RTX 3080 (sm_86) and RX 9060 XT (gfx1200, RDNA4): one per vendor.
    assert ARCHS == (SM_86, GFX1200)
    assert {a.name for a in ARCHS} == {"sm_86", "gfx1200"}
    assert SM_86.vendor is Vendor.NVIDIA and GFX1200.vendor is Vendor.AMD
    assert SM_86.wavefront == 32 and GFX1200.wavefront == 32
    assert str(SM_86) == "sm_86"


def test_an_arch_is_a_frozen_value() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SM_86.wavefront = 64  # type: ignore[misc]
    assert Arch(Vendor.NVIDIA, "sm_86", 32) == SM_86
    with pytest.raises(ValueError):
        Arch("nvidia", "sm_86", 32)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Arch(Vendor.AMD, "gfx1200", 0)


def test_arch_and_target_are_different_types() -> None:
    # "target" is the matrix (spec.Target); the card is an Arch.
    assert not isinstance(SM_86, Target)
    assert "arch" not in {f.name for f in dataclasses.fields(Spec)}


def test_from_env_follows_config_sh_defaults_and_pins() -> None:
    assert from_env(Vendor.NVIDIA, {}) is SM_86
    assert from_env(Vendor.AMD, {}) is GFX1200
    assert from_env(Vendor.NVIDIA, {"CUDA_ARCH": "86"}) is SM_86
    assert from_env(Vendor.AMD, {"GPU_TARGETS": "gfx1200"}) is GFX1200
    assert from_env(Vendor.AMD, {"HIP_ARCH": "gfx1200", "GPU_TARGETS": "x"}) is GFX1200
    # A pin to a card nobody declared fails loudly instead of drifting.
    with pytest.raises(ValueError, match="not declared"):
        from_env(Vendor.NVIDIA, {"CUDA_ARCH": "90"})
    with pytest.raises(ValueError, match="not declared"):
        by_name("gfx942")


def test_config_sh_spells_the_same_defaults() -> None:
    # The shell's copy cannot import Python; assert the two agree instead.
    config = (Path(__file__).resolve().parent.parent / "devtools" / "config.sh")
    text = config.read_text()
    assert f"CUDA_ARCH=${{CUDA_ARCH:-{SM_86.name.removeprefix('sm_')}}}" in text
    assert f"HIP_ARCH=${{HIP_ARCH:-{GFX1200.name}}}" in text


def test_conftest_reads_the_arch_from_this_module() -> None:
    import conftest

    assert f"sm_{conftest.CUDA_ARCH}" == from_env(Vendor.NVIDIA).name
    assert conftest.HIP_ARCH == from_env(Vendor.AMD).name


def test_two_rows_differing_only_in_card_do_not_collide() -> None:
    # The #28 hazard, asserted before any data file exists: Spec alone cannot
    # tell an sm_86 row from a gfx1200 one for the same class and batch count.
    spec = _spec(am=(3, 3, 3, 3), n_batches=3)
    rows = {
        MeasuredKey(spec, SM_86, "nvcc 13.0"): "row A",
        MeasuredKey(spec, GFX1200, "nvcc 13.0"): "row B",
        MeasuredKey(spec, GFX1200, "hipcc / ROCm 7.2.4"): "row C",
        MeasuredKey(spec, SM_86, "nvcc 13.1"): "row D",
    }
    assert len(rows) == 4
    assert len({k.key() for k in rows}) == 4
    a, b = MeasuredKey(spec, SM_86, "t"), MeasuredKey(spec, GFX1200, "t")
    assert a.spec == b.spec and a != b and a.key() != b.key()
    assert MeasuredKey(spec, SM_86, "nvcc 13.0").key() == (
        "coulomb/d0/(3,3|3,3)/b3/j/literal@sm_86/nvcc 13.0"
    )
    with pytest.raises(ValueError):
        MeasuredKey(spec, "sm_86", "nvcc 13.0")  # type: ignore[arg-type]


def _emit_sources() -> list[Path]:
    return sorted(EMIT_DIR.rglob("*.py")) if EMIT_DIR.is_dir() else []


def test_no_arch_string_or_wavefront_literal_in_emit() -> None:
    # Vacuous until emit/ lands, and bites the moment it does: a backend reads
    # the arch name and the wavefront width off its Arch argument, it never
    # spells them. (Both cards are wave32, so a literal 32 is the easy bake-in.)
    names = {a.name for a in ARCHS}
    widths = {a.wavefront for a in ARCHS}
    found: list[str] = []
    for path in _emit_sources():
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.Constant):
                continue
            v = node.value
            if isinstance(v, str) and any(n in v for n in names):
                found.append(f"{path.name}:{node.lineno}: arch string {v!r}")
            elif type(v) is int and v in widths:
                found.append(f"{path.name}:{node.lineno}: wavefront literal {v}")
    assert not found, "read these off the Arch instead:\n" + "\n".join(found)
