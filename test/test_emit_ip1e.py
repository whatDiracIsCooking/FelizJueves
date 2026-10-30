"""The one-electron derivative-matrix kernels (#687) emitted for both
vendors: golden texts for the pair classes (ss) through (dd), the `--ip-one-e-l`
ladder, and a compile of each through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --ip-one-e-l [LMAX]` writes, for every
`la <= lb <= LMAX` (2 when the flag is bare), `ips_`, `ipt_`, `ipv_` and
`iprv_{la}{lb}` (`hessian.kernel_ip1e`). It accepts 0..3 (the kernels are
validated through (ff), interpreted), exits 2 above that, is NEVER in the
bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/{ips,ipt,ipv,iprv}_{la}{lb}.{cu,hip}`
for the six classes (ss) (sp) (sd) (pp) (pd) (dd). Every unit through (ff),
both orders, is under the repo's 512 KiB file cap (asserted; the largest,
`ipt_33`, is ~210 KB). Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_ip1e.py --write

THE COMPILE. Each golden is compiled alone (`-c`) with the kernel tier's
own flags and FAILS on any warning located in the generated unit (#92's
filter; `test_emit_grad.compile_kernel`). Needs a compiler, never a card.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_emit_grad import compile_kernel
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.hessian.kernel_ip1e import IP1E, build_ip1e_kernel, ip1e_name
from felizjueves.ir import Raw, walk
from felizjueves.performance.device import GFX1200, SM_86

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
CAP = 512 * 1024  # the check-added-large-files hook's --maxkb
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
CLASSES = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
FAMILY = dict(zip(("ips", "ipt", "ipv", "iprv"), IP1E, strict=True))
PREFIXES = ("ips_", "ipt_", "ipv_", "iprv_")
UNITS = [(f, c) for c in CLASSES for f in FAMILY]
ORDERED = [(la, lb) for la in range(4) for lb in range(4)]


def _kernel(family: str, cls):
    target, form = FAMILY[family]
    return build_ip1e_kernel((target, *cls, form))


def _name(family: str, cls) -> str:
    target, form = FAMILY[family]
    return ip1e_name((target, *cls, form))


def _id(u) -> str:
    return _name(*u)


def _text(family: str, cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(_kernel(family, cls), arch)


def write_goldens() -> None:
    """Rewrite every derivative-matrix golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for f, c in UNITS:
            text = _text(f, c, backend)
            if len(text.encode()) > CAP:
                raise SystemExit(f"{_name(f, c)} is over the {CAP} B cap")
            (GOLDEN / backend / f"{_name(f, c)}.{ext}").write_text(text)


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_pair_class_through_lmax() -> None:
    assert emit_cli.ip_one_e_classes(2) == CLASSES
    assert emit_cli.DEFAULT_IP_ONE_E_L == 2 and emit_cli.IP_ONE_E_L_MAX == 3
    assert emit_cli.ip_one_e_classes(0) == [(0, 0)]
    assert len(emit_cli.ip_one_e_classes(3)) == 10


def test_the_ladder_refuses_above_ff() -> None:
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            emit_cli.ip_one_e_classes(bad)


def test_the_names_are_the_builders() -> None:
    for f, c in UNITS:
        assert _kernel(f, c).name == _name(f, c) == f + "_" + "".join(
            map(str, c))


def test_no_raw_escape_hatch() -> None:
    for f, c in UNITS:
        k = _kernel(f, c)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_the_ip1_golden_is_unchanged(unit, backend: str) -> None:
    text = _text(*unit, backend)
    path = GOLDEN / backend / f"{_name(*unit)}.{BACKEND[backend][2]}"
    assert len(text.encode()) <= CAP
    assert text == path.read_text()


def test_the_golden_set_is_exactly_the_six_classes() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for f in FAMILY
                 for p in (GOLDEN / backend).glob(f"{f}_[0-9]*.{ext}")}
        assert files == {_name(*u) for u in UNITS}


@pytest.mark.parametrize("cls", [by_l(c, 1 + sum(c), id=f"{c[0]}{c[1]}")
                                 for c in ORDERED])
def test_every_unit_through_ff_is_under_the_cap(cls) -> None:
    for backend in BACKEND:
        for f in FAMILY:
            size = len(_text(f, cls, backend).encode())
            assert size <= CAP, f"{_name(f, cls)}.{backend}: {size} B"


@pytest.mark.parametrize("unit", [("ips", (1, 2)), ("iprv", (1, 1))], ids=_id)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(*unit, "cuda"), _text(*unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_every_write_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        for f in FAMILY:
            text = _text(f, (1, 1), backend)
            out = "XC" if f == "iprv" else "X1"
            assert f"atomicAdd(&{out}[" in text
            assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_ip_one_e_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--ip-one-e-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(*u)}.{ext}" for u in UNITS}
    for u in UNITS:
        name = _name(*u)
        text = (tmp_path / f"{name}.{ext}").read_text()
        assert text == (GOLDEN / backend / f"{name}.{ext}").read_text()


def test_ip_one_e_l_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--ip-one-e-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "ips_00.cu", "ipt_00.cu",
                                "ipv_00.cu", "iprv_00.cu"}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_ip1_unit(backend, tmp_path) -> None:
    assert _run(tmp_path, backend) == 0
    assert not any(n.startswith(PREFIXES) for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_ip1_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--ip-one-e-l" not in line


def test_the_ip1_names_cannot_collide_with_any_other_ladder(tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--one-e-l", "3", "--mu-l",
                "--schwarz-l", "--sx-l", "--metric-l", "--eri3-l",
                "--grad-l", "3", "--hgp-one-e-l", "--hess-l", "2",
                "--hess-one-e-l", "3", "--grad-df-l") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(PREFIXES) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--ip-one-e-l", "1") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 12 and not mine & others


@pytest.mark.parametrize("argv", [
    ["--ip-one-e-l", "4"],
    ["--ip-one-e-l", "-1"],
    ["--ip-one-e-l", "x"],
    ["--ip-one-e-l", "1", "--class", "0", "0", "0", "0"],
    ["--ip-one-e-l", "1", "--shell-ls", "0"],
    ["--ip-one-e-l", "1", "--fock-l", "0"],
    ["--ip-one-e-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--ip-one-e-l", "1", "--n-batches", "3"],
    ["--ip-one-e-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_an_ip_one_e_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_ip_one_e_l_above_ff_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--ip-one-e-l", "4") == 2
    err = capsys.readouterr().err
    assert "--ip-one-e-l 4 is outside 0..3" in err
    assert "validated through (ff) only" in err


# --- the compile (a compiler, no card) ----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_nvcc_compiles_the_ip1_unit(unit, nvcc, tmp_path) -> None:
    compile_kernel(nvcc, "cuda", _kernel(*unit), tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_hipcc_compiles_the_ip1_unit(unit, hipcc, tmp_path) -> None:
    compile_kernel(hipcc, "hip", _kernel(*unit), tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_ip1e.py --write")
    write_goldens()
