"""The HGP DF Hessian kernels emitted for both vendors: golden texts (or a
digest) for the three metric and six three-center s/p classes, the
`--hgp-hess-metric-l` and `--hgp-hess-eri3-l` ladders, and a compile of each
through nvcc (sm_86) and hipcc (gfx1200).

THE LADDERS. `python -m felizjueves.emit --hgp-hess-metric-l [LMAX]` writes
`hgd2metric_{lP}{lQ}` (`hessian.kernel_hgd2metric`) for every `lP <= lQ <=
LMAX`, and `--hgp-hess-eri3-l [LMAX]` writes `hgd2eri3_{la}{lb}{lP}`
(`hessian.kernel_hgd2eri3`) for every `la <= lb <= LMAX`, `lP <= LMAX`:
`--hgp-metric-l`'s and `--hgp-eri3-l`'s classes and bound (0..
`HGP_DF_L_MAX` = 1, bare 1), never in the bare default nor in
KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/hgd2{metric,eri3}_*.{cu,hip}`, every
unit but those over the pre-commit hook's 512 KiB cap (`hgd2eri3_111`, 539
KB), which are pinned instead by byte count and sha256 in
`test/golden/hgd2_digests.json`. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hgp_hess_df.py --write

THE COMPILE is test_emit_grad.py's: each unit alone (`-c`), the kernel
tier's flags, FAILING on any warning located in the generated unit (#92).
Needs a compiler, never a card: the compile cases are `cuda` / `hip`.
"""

from __future__ import annotations

import hashlib
import json
import sys

import pytest
import test_emit_hgp_df as thd
from test_emit_grad import BACKEND, GOLDEN, _names, _run, compile_kernel

from felizjueves.emit import __main__ as emit_cli
from felizjueves.hessian.kernel_hgd2eri3 import build_hgd2eri3_kernel
from felizjueves.hessian.kernel_hgd2metric import build_hgd2metric_kernel
from felizjueves.ir import Raw, walk

UNITS = [*(("hgd2metric", c) for c in thd.METRIC),
         *(("hgd2eri3", c) for c in thd.ERI3)]
BUILD = {"hgd2metric": build_hgd2metric_kernel,
         "hgd2eri3": build_hgd2eri3_kernel}
DIGESTS = GOLDEN / "hgd2_digests.json"
CAP = 512 * 1024  # bytes: .pre-commit-config.yaml's --maxkb=512


def _name(unit) -> str:
    return f"{unit[0]}_{''.join(map(str, unit[1]))}"


def _text(unit, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(BUILD[unit[0]](unit[1]), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_goldens() -> None:
    """Rewrite every HGP DF Hessian golden and the digest file (module
    docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        digests[backend] = {}
        for unit in UNITS:
            text = _text(unit, backend)
            path = GOLDEN / backend / f"{_name(unit)}.{ext}"
            if len(text.encode()) <= CAP:
                path.write_text(text)
            else:
                digests[backend][_name(unit)] = _digest(text)
                path.unlink(missing_ok=True)
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


def _pinned(unit, backend: str, text: str) -> None:
    path = GOLDEN / backend / f"{_name(unit)}.{BACKEND[backend][2]}"
    pinned = json.loads(DIGESTS.read_text())[backend]
    if _name(unit) in pinned:
        assert _digest(text) == pinned[_name(unit)]
        assert not path.exists()
    else:
        assert text == path.read_text()


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


# --- the ladders ---------------------------------------------------------------


def test_the_ladders_through_p_are_the_s_and_p_classes() -> None:
    assert emit_cli.hgp_hess_metric_classes(1) == thd.METRIC
    assert emit_cli.hgp_hess_eri3_classes(1) == thd.ERI3
    assert emit_cli.hgp_hess_metric_classes(0) == [(0, 0)]
    assert emit_cli.hgp_hess_eri3_classes(0) == [(0, 0, 0)]
    for f in (emit_cli.hgp_hess_metric_classes,
              emit_cli.hgp_hess_eri3_classes):
        for bad in (-1, 2, 1.0, True):
            with pytest.raises(ValueError, match="s and p shells only"):
                f(bad)


def test_no_raw_escape_hatch() -> None:
    for unit in UNITS:
        k = BUILD[unit[0]](unit[1])
        assert k.name == _name(unit)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens ---------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", UNITS, ids=_name)
def test_the_hgp_hess_df_golden_is_unchanged(unit, backend: str) -> None:
    _pinned(unit, backend, _text(unit, backend))


def test_the_golden_set_is_exactly_the_nine_units() -> None:
    pinned = json.loads(DIGESTS.read_text())
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"hgd2*_*.{ext}")}
        assert files | set(pinned[backend]) == {_name(u) for u in UNITS}
        assert not files & set(pinned[backend])
        assert set(pinned[backend]) == {"hgd2eri3_111"}


def test_the_vendors_differ_only_in_the_prologue() -> None:
    for unit in (("hgd2metric", (1, 1)), ("hgd2eri3", (0, 1, 1))):
        c, h = _text(unit, "cuda"), _text(unit, "hip")
        assert c.split("\n")[2:] == h.split("\n")[3:]
        assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_writes_are_atomic_adds_into_h_alone() -> None:
    for backend in BACKEND:
        text = _text(("hgd2metric", (1, 1)), backend)
        assert text.count("atomicAdd(&H[") == 36 and "__shared__" not in text
        text = _text(("hgd2eri3", (0, 1, 1)), backend)
        assert text.count("atomicAdd(&H[") == 81 and "__shared__" not in text


# --- the CLI path --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_flags_bare_write_the_nine_pinned_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--L", "0", "--hgp-hess-metric-l",
                "--hgp-hess-eri3-l") == 0
    assert _names(tmp_path) == {f"pq_L0_p0_q0.{ext}"} | {
        f"{_name(u)}.{ext}" for u in UNITS}
    for u in UNITS:
        _pinned(u, backend, (tmp_path / f"{_name(u)}.{ext}").read_text())


def test_they_add_their_units_beside_the_d0_and_d1_hgp_ladders(
        tmp_path) -> None:
    assert _run(tmp_path, "hip", "--L", "0", "--hgp-eri3-l", "0",
                "--hgp-grad-metric-l", "0", "--hgp-hess-eri3-l", "0",
                "--hgp-hess-metric-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.hip", "heri3_000.hip",
                                "hgmetric_00.hip", "hgd2eri3_000.hip",
                                "hgd2metric_00.hip"}


def test_the_bare_default_and_kernel_gen_cmd_write_none(tmp_path) -> None:
    from felizjueves.performance import measured
    assert _run(tmp_path, "cuda") == 0
    assert not any(n.startswith("hgd2") for n in _names(tmp_path))
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "--hgp-hess" not in line


@pytest.mark.parametrize("argv", [
    ["--hgp-hess-metric-l", "2"],
    ["--hgp-hess-eri3-l", "2"],
    ["--hgp-hess-eri3-l", "-1"],
    ["--hgp-hess-metric-l", "x"],
    ["--hgp-hess-eri3-l", "1", "--class", "0", "0", "0", "0"],
    ["--hgp-hess-metric-l", "1", "--fock-l", "0"],
    ["--hgp-hess-eri3-l", "1", "--hgp-j-l", "0"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


# --- the compile (a compiler, no card) -----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("unit", UNITS, ids=_name)
def test_nvcc_compiles_the_hgp_hessian_unit(unit, nvcc, tmp_path) -> None:
    compile_kernel(nvcc, "cuda", BUILD[unit[0]](unit[1]), tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", UNITS, ids=_name)
def test_hipcc_compiles_the_hgp_hessian_unit(unit, hipcc, tmp_path) -> None:
    compile_kernel(hipcc, "hip", BUILD[unit[0]](unit[1]), tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hgp_hess_df.py "
                 "--write")
    write_goldens()
