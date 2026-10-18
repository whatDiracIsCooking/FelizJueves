"""`Arch`, the card as data, and `MeasuredKey`, what keys a measurement."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from felizjueves.device import (
    ARCHS,
    COMPUTE_BOUND,
    GFX1200,
    MEMORY_BOUND,
    QUANTITIES,
    SM_86,
    Arch,
    MeasuredKey,
    Uncharacterised,
    Vendor,
    by_name,
    from_env,
    recorded_ceilings,
    require_characterised,
    verdict,
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


def _kfd_tree(root: Path, *gfx_target_versions: int) -> Path:
    """An amdkfd topology with one node per argument, in kfd's own layout."""
    nodes = root / "nodes"
    for i, gfxver in enumerate(gfx_target_versions):
        node = nodes / str(i)
        node.mkdir(parents=True)
        # The real file carries ~60 fields; the two that matter here, in kfd's
        # `name value` spelling, plus one whose prefix could be mis-matched.
        (node / "properties").write_text(
            f"cpu_cores_count 24\nsimd_count 0\n"
            f"gfx_target_version_prefix_trap 999\ngfx_target_version {gfxver}\n"
        )
    return nodes


def test_an_amd_card_is_an_enumerated_kfd_node_not_just_dev_kfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#431: `/dev/kfd` exists whenever the module is loaded, bound card or not.

    An amdgpu that does not know the card's PCI id -- kernel 6.8's in-tree one
    against gfx1200's 1002:7590 -- loads, claims nothing, and leaves kfd with
    only the CPU node. Keying presence on the device node therefore reported a
    card on a box that had none: `hip_device` did not skip, so every `gpu`
    test FAILED instead of skipping, and in a container that read as broken
    passthrough. Presence is an enumerated GPU node, and nothing else.
    """
    import conftest

    cpu_only = _kfd_tree(tmp_path / "cpu-only", 0)
    monkeypatch.setattr(conftest, "KFD_NODES", cpu_only)
    assert conftest._has_amd_device() is False

    # The CPU node is still node 0 when a GPU is there; gfx1200 -> 120000.
    with_gpu = _kfd_tree(tmp_path / "with-gpu", 0, 120000)
    monkeypatch.setattr(conftest, "KFD_NODES", with_gpu)
    assert conftest._has_amd_device() is True


def test_no_kfd_topology_at_all_is_simply_no_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import conftest

    monkeypatch.setattr(conftest, "KFD_NODES", tmp_path / "absent")
    assert conftest._has_amd_device() is False


def test_the_two_reasons_for_no_amd_card_read_differently(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix differs, so the skip reason has to: a missing `/dev/kfd` is a
    passthrough question, an empty topology is a HOST driver question (#431)."""
    import conftest

    seen: list[str] = []
    monkeypatch.setattr(
        conftest.Path, "exists", lambda self: (seen.append(str(self)), False)[1]
    )
    absent = conftest._no_amd_reason()
    assert "/dev/kfd" in seen and "--device=/dev/kfd" in absent

    monkeypatch.setattr(conftest.Path, "exists", lambda self: True)
    unbound = conftest._no_amd_reason()
    assert "no GPU node" in unbound and "HOST" in unbound
    assert "passthrough" in unbound and absent != unbound


def test_the_shell_probe_asks_the_same_question_as_conftest() -> None:
    """The tier's reading and the test suite's must not drift (#431).

    lib.sh cannot import Python, so assert the shared rule textually, the way
    test_config_sh_spells_the_same_defaults does for the arch defaults.
    """
    devtools = Path(__file__).resolve().parent.parent / "devtools"
    lib = (devtools / "lib.sh").read_text()
    tier = (devtools / "kernel-tier.sh").read_text()

    # One rule, in one place: the same sysfs path and the same field.
    assert "amd_gpu_present()" in lib
    assert "/sys/class/kfd/kfd/topology/nodes" in lib
    assert "gfx_target_version" in lib

    # And the tier asks for it, rather than for the device node that is there
    # whether or not a card is bound.
    assert "hip)  amd_gpu_present && device_present=1" in tier
    assert "hip)  [ -e /dev/kfd ] && device_present=1" not in tier


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


# -- the ceilings (#109) -------------------------------------------------------

UNCHARACTERISED = Arch(Vendor.AMD, "gfx9999", wavefront=64)  # deliberately


def test_an_uncharacterised_arch_gets_no_ridge_and_no_verdict() -> None:
    # #109's acceptance: a third card, declared with no measured ceilings,
    # is refused -- no default ridge, no verdict -- rather than scored.
    a = UNCHARACTERISED
    assert a.fp64_flops_per_s == 0.0 and a.mem_bandwidth_bytes_per_s == 0.0
    assert not a.characterised
    with pytest.raises(Uncharacterised, match="not characterised"):
        a.ridge  # noqa: B018
    for intensity in (0.0, 1.0, 1e6):
        with pytest.raises(Uncharacterised):
            verdict(a, intensity)
    # Half a characterisation is still none.
    half = Arch(Vendor.AMD, "gfx9999", 64, fp64_flops_per_s=1e12)
    with pytest.raises(Uncharacterised, match="mem_bandwidth_bytes_per_s"):
        half.ridge  # noqa: B018
    with pytest.raises(Uncharacterised):
        require_characterised(half)


def test_a_characterised_arch_has_a_ridge_and_verdicts() -> None:
    a = Arch(Vendor.NVIDIA, "sm_test", 32, fp64_flops_per_s=400e9,
             mem_bandwidth_bytes_per_s=100e9)
    assert a.characterised and a.ridge == pytest.approx(4.0)
    assert verdict(a, 3.9) == MEMORY_BOUND
    assert verdict(a, 4.0) == COMPUTE_BOUND
    with pytest.raises(ValueError):
        verdict(a, -1.0)
    for bad in (-1.0, float("nan"), True, "1e12"):
        with pytest.raises(ValueError):
            Arch(Vendor.NVIDIA, "sm_test", 32, fp64_flops_per_s=bad)  # type: ignore[arg-type]


def test_ceilings_are_observations_not_identity() -> None:
    # Equality and hashing are the compiler target's; a (re-)probe must not
    # split one card into two keys.
    bare = Arch(Vendor.NVIDIA, "sm_86", 32)
    assert bare == SM_86 and hash(bare) == hash(SM_86)
    assert MeasuredKey(_spec(), bare, "t").key() == MeasuredKey(_spec(), SM_86,
                                                                "t").key()


def test_the_declared_archs_read_their_ceilings_from_the_record() -> None:
    for arch in ARCHS:
        assert {q: getattr(arch, q) for q in QUANTITIES} == recorded_ceilings(
            arch.name)


def test_a_missing_record_is_uncharacterised(tmp_path: Path) -> None:
    assert recorded_ceilings("sm_86", tmp_path / "absent.json") == {
        q: 0.0 for q in QUANTITIES}
    part = tmp_path / "part.json"
    part.write_text('{"ceilings": {"sm_86": {"fp64_flops_per_s": '
                    '{"value": 4e11}}}}')
    got = recorded_ceilings("sm_86", part)
    assert got == {"fp64_flops_per_s": 4e11, "mem_bandwidth_bytes_per_s": 0.0}
    assert not Arch(Vendor.NVIDIA, "sm_86", 32, **got).characterised


def test_device_spells_no_datasheet_peak() -> None:
    # #109's hazard: the ridge is measured on this box. No float literal of a
    # plausible peak's size (>= 1e9) may appear in device.py.
    src = Path(__file__).resolve().parent.parent / "src" / "felizjueves" / "device.py"
    big = [n.value for n in ast.walk(ast.parse(src.read_text()))
           if isinstance(n, ast.Constant) and type(n.value) in (int, float)
           and abs(n.value) >= 1e9]
    assert not big, f"a datasheet-sized constant in device.py: {big}"


CEILING_NAMES = {"fp64_flops_per_s", "mem_bandwidth_bytes_per_s", "ridge",
                 "verdict", "require_characterised", "characterised",
                 "recorded_ceilings", "ceilings"}


def test_nothing_in_the_literature_layer_reads_the_ceilings() -> None:
    # #109's hazard: a scoreboard reading a field a re-probe rewrites destroys
    # a frozen observation condition. Names, not just imports (test_layering
    # holds the imports): an attribute access is invisible to an import walk.
    root = Path(__file__).resolve().parent.parent / "src" / "felizjueves"
    found = []
    for path in sorted((root / "literature").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            name = (node.attr if isinstance(node, ast.Attribute) else
                    node.id if isinstance(node, ast.Name) else None)
            if name in CEILING_NAMES:
                found.append(f"{path.name}:{node.lineno}: {name}")
    assert not found, "\n".join(found)


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
