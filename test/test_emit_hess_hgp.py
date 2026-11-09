"""The HGP two-electron Hessian kernels emitted for both vendors: golden
texts for `(ss|ss)`, digests of every other ladder unit through `(pp|pp)`,
the `--hess-hgp-l` ladder, and the compile.

THE LADDER. `python -m felizjueves.emit --hess-hgp-l [LMAX]` writes
`d2f_`, `d2hj_` and `d2hk_{la}{lb}{lc}{ld}` (`hessian.kernel_hgp2`) for
every F class of milestone 12's ladder (`emit.__main__.fock_classes`: AO
order, canonical) with each `l <= LMAX` (1 when the flag is bare: 6
classes, 18 units). It accepts 0..2 (validated through `(dd|dd)`,
interpreted, test_kernel_hgp2.py), exits 2 above that, is NEVER in the
bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/d2{f,hj,hk}_0000.{cu,hip}`;
every other unit through `(pp|pp)` is pinned by size and sha256 in
`test/golden/hgp2_digests.json` (`slow` above `L + 2 = 4`, `tiers.by_l`:
the kernel builds the class two orders up). A unit is large (`(pp|pp)`'s
is 3.7 MB of CUDA; a d class 0.5 to 66 MB), so the d classes are neither
digested nor compiled. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hess_hgp.py --write

THE COMPILE is test_emit_grad.py's: each unit through `(pp|pp)` alone
(`-c`), the kernel tier's flags, FAILING on a warning located in the
generated unit (#92). Needs a compiler, never a card. The ids are
`<slot>_<class>` so the compile sample (`tiers.sample_compiles`) keeps one
unit of each of the three targets.
"""

from __future__ import annotations

import hashlib
import json
import sys

import pytest
from test_emit_grad import BACKEND, GOLDEN, _run, compile_kernel
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.hessian.kernel_hgp2 import (
    HGP2_TARGETS,
    build_d2_kernel,
    d2_name,
    d2_spec,
)

CLASSES = emit_cli.hess_hgp_classes(1)
UNITS = [d2_spec(t, c) for c in CLASSES for t in HGP2_TARGETS]
GOLDEN_UNITS = [d2_spec(t, (0, 0, 0, 0)) for t in HGP2_TARGETS]
DIGESTS = GOLDEN / "hgp2_digests.json"
DIGESTED = [u for u in UNITS if u not in GOLDEN_UNITS]


def _name(unit) -> str:
    return d2_name(unit)


def _id(unit) -> str:
    return f"{unit.target_slot}_{''.join(map(str, unit.am))}"


def _text(unit, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_d2_kernel(unit), arch)


def _digest(text: str) -> dict:
    b = text.encode()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def write_goldens() -> None:
    """Rewrite the goldens and digests (module docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        for u in GOLDEN_UNITS:
            (GOLDEN / backend / f"{_name(u)}.{ext}").write_text(
                _text(u, backend))
        digests[backend] = {_name(u): _digest(_text(u, backend))
                            for u in DIGESTED}
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_the_fock_ladder_through_lmax() -> None:
    assert CLASSES == emit_cli.fock_classes(1) and len(UNITS) == 18
    assert emit_cli.DEFAULT_HESS_HGP_L == 1 and emit_cli.HESS_HGP_L_MAX == 2
    assert emit_cli.hess_hgp_classes(0) == [(0, 0, 0, 0)]
    assert emit_cli.hess_hgp_classes(2) == emit_cli.fock_classes(2)


def test_the_ladder_refuses_above_dd() -> None:
    for bad in (-1, 3, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.2"):
            emit_cli.hess_hgp_classes(bad)


# --- the goldens and the digests ------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", GOLDEN_UNITS, ids=_id)
def test_the_golden_is_unchanged(unit, backend: str) -> None:
    path = GOLDEN / backend / f"{_name(unit)}.{BACKEND[backend][2]}"
    assert _text(unit, backend) == path.read_text()


def test_the_golden_set_is_exactly_ss_ss() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for pre in ("d2f_", "d2hj_", "d2hk_")
                 for p in (GOLDEN / backend).glob(f"{pre}*.{ext}")}
        assert files == {_name(u) for u in GOLDEN_UNITS}


def test_the_digests_are_every_other_ladder_unit() -> None:
    d = json.loads(DIGESTS.read_text())
    for backend in BACKEND:
        assert set(d[backend]) == {_name(u) for u in DIGESTED}


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", [by_l(u, u.total_am + 2, id=_id(u))
                                  for u in DIGESTED])
def test_the_digest_is_unchanged(unit, backend: str) -> None:
    want = json.loads(DIGESTS.read_text())[backend][_name(unit)]
    assert _digest(_text(unit, backend)) == want


def test_the_vendors_differ_only_in_the_prologue() -> None:
    for u in GOLDEN_UNITS:
        c, h = _text(u, "cuda"), _text(u, "hip")
        assert c.split("\n")[2:] == h.split("\n")[3:]
        assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_write_is_atomic_into_h_with_one_shared_record() -> None:
    for backend in BACKEND:
        for u in GOLDEN_UNITS:
            text = _text(u, backend)
            assert text.count("atomicAdd(&H[") == 144
            assert text.count("__shared__") == 1


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_hess_hgp_l_zero_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hess-hgp-l", "0") == 0
    names = {_name(u) for u in GOLDEN_UNITS}
    assert {p.name for p in tmp_path.iterdir()} == {
        f"dq_0_0.{ext}", f"jab_00.{ext}", *(f"{n}.{ext}" for n in names)}
    for n in names:
        assert (tmp_path / f"{n}.{ext}").read_text() == (
            GOLDEN / backend / f"{n}.{ext}").read_text()


def test_the_bare_default_writes_no_hgp_hessian_unit(tmp_path) -> None:
    assert _run(tmp_path, "cuda") == 0
    assert not any(p.name.startswith(("d2f_", "d2hj_", "d2hk_"))
                   for p in tmp_path.iterdir())


def test_kernel_gen_cmd_asks_for_no_hgp_hessian_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--hess-hgp-l" not in line


@pytest.mark.parametrize("argv", [
    ["--hess-hgp-l", "3"],
    ["--hess-hgp-l", "-1"],
    ["--hess-hgp-l", "x"],
    ["--hess-hgp-l", "1", "--class", "0", "0", "0", "0"],
    ["--hess-hgp-l", "1", "--shell-ls", "0"],
    ["--hess-hgp-l", "1", "--fock-l", "0"],
    ["--hess-hgp-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_hess_hgp_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_hess_hgp_l_above_dd_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hess-hgp-l", "3") == 2
    err = capsys.readouterr().err
    assert "--hess-hgp-l 3 is outside 0..2" in err
    assert "validated through (dd|dd) only" in err


# --- the compile (a compiler, no card) ----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_nvcc_compiles_the_hgp_hessian_unit(unit, nvcc, tmp_path) -> None:
    compile_kernel(nvcc, "cuda", build_d2_kernel(unit), tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_hipcc_compiles_the_hgp_hessian_unit(unit, hipcc, tmp_path) -> None:
    compile_kernel(hipcc, "hip", build_d2_kernel(unit), tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hess_hgp.py "
                 "--write")
    write_goldens()
