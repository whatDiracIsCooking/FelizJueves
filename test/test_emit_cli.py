"""`python -m felizjueves.emit` as KERNEL_GEN_CMD's contract (#156).

ONE LADDER PER KERNEL FAMILY. The no-argument default writes the `[p|q]`
ladder (L = 0..16, unchanged: the measured store pins each unit's
`source_sha256`) PLUS the pair ladder (`DQ` and `JAB` for every pair class
`(la, lb)`, `la <= lb <= 3`: 10 classes, 20 units). `--L` alone writes only
`[p|q]`, `--pair-l` alone only the pair ladder. `--one-e-l LMAX` (#242)
ADDS the one-electron ladder (`s_`, `t_`, `v_` per pair class `la <= lb <=
LMAX`) beside whichever J ladders those select; it is never in the bare
default, which stays 37 units with every pinned `source_sha256`. The fused
Fock kernel (`--fock-class` / `--fock-l`, #287) is a separate path, tested
in test/test_emit_fock.py; here only its absence from the default.

THE LEVER LADDERS (#339) are flags read from a registry
(`lever_ladder.flags()`, over `levers.VOCABULARY`): `--klever-<name>-pq` and
`--klever-<name>-k` add lever units beside the J ladders, and
`--klever-<name>-fock-l` sits on the F path. Beside the shipped `fused`
lever (#344, its units checked against its goldens), the tests below declare
a probe lever; no lever unit is in the bare default.

THE SELECTOR writes, per `(class, n_batches)` cell, the chain one class
needs -- its ket's `DQ`, its `[p|q]` batch kernels, its bra's `JAB` -- with
an `order` file naming them in that launch order.

Exit status: 0 written, 2 could not run -- no third meaning.

Fast tier: nothing here compiles anything. The bare default builds the
`[p|q]` ladder through L = 16, so every test that runs it is `slow`, and the
rest are split at `tiers.FAST_L_MAX` by their classes' L (`la + lb` for a
pair unit, `2 (la + lb)` for a Schwarz one). Names are the exception: they
come from each builder's name function, unbuilt, so the collision checks
cover every L in the fast tier (#419).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_ao_j_interpreted import _case, _classes, _shell_ams
from tiers import by_l, is_fast, split_by_l

from felizjueves import levers
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.ir import lever_ladder
from felizjueves.kernel_dq import build_dq_kernel, dq_name
from felizjueves.kernel_eri3 import eri3_name
from felizjueves.kernel_jab import build_jab_kernel, jab_name
from felizjueves.kernel_k import build_k_kernel, build_k_kernels
from felizjueves.mmd.kernel import (
    build_pq_kernel,
    build_pq_kernels,
    k_spec,
    pq_name,
    pq_spec,
)
from felizjueves.performance.device import GFX1200, SM_86

ROOT = Path(__file__).resolve().parent.parent
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}

PAIR_CLASSES_F = [(la, lb) for la in range(4) for lb in range(la, 4)]


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(tmp_path: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(tmp_path), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


def _pq_names(ext: str) -> set[str]:
    return {f"{pq_name(L)}.{ext}" for L in range(17)}


def _pair_names(ext: str, classes=PAIR_CLASSES_F) -> set[str]:
    return ({f"{dq_name(c)}.{ext}" for c in classes}
            | {f"{jab_name(c)}.{ext}" for c in classes})


# --- the names, without building (#419) ---------------------------------------
#
# Each builder names its kernel through a name function beside it (`pq_name`,
# `dq_name`, `jab_name`, `emit.ONE_E_NAMES`), so the names of every rung, L
# up to 16 and pair classes through `PAIR_L_MAX`, are checked here without
# building one IR. That the builder uses its name function is checked by
# building, so it is split at the cut.


def _assert_the_names_are_the_builders(keep) -> None:
    for L in range(17):
        if keep(L):
            assert pq_name(L) == build_pq_kernel(L).name
    for c in emit_cli.pair_classes(emit_cli.PAIR_L_MAX):
        if not keep(sum(c)):
            continue
        assert dq_name(c) == build_dq_kernel(c).name
        assert jab_name(c) == build_jab_kernel(c).name
        for name, build in zip(emit_cli.ONE_E_NAMES, emit_cli.ONE_E_BUILDERS,
                               strict=True):
            assert name(c) == build(c).name


def test_each_name_function_is_its_builders_name() -> None:
    _assert_the_names_are_the_builders(is_fast)


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_each_name_function_is_its_builders_name_above_l4() -> None:
    _assert_the_names_are_the_builders(lambda L: not is_fast(L))


# --- the ladders ------------------------------------------------------------


def test_the_pair_ladder_is_ss_through_ff() -> None:
    assert emit_cli.DEFAULT_PAIR_L == 3
    assert emit_cli.pair_classes(3) == PAIR_CLASSES_F
    assert len(PAIR_CLASSES_F) == 10
    assert all(la <= lb for la, lb in emit_cli.pair_classes(emit_cli.PAIR_L_MAX))


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_default_writes_both_ladders_and_nothing_else(backend, tmp_path):
    emit, arch, ext = BACKEND[backend]
    assert _run(tmp_path, backend) == 0
    pq, pair = _pq_names(ext), _pair_names(ext)
    assert len(pq) == 17 and len(pair) == 20 and not pq & pair
    assert _names(tmp_path) == pq | pair
    for c in PAIR_CLASSES_F:
        for k in (build_dq_kernel(c), build_jab_kernel(c)):
            assert (tmp_path / f"{k.name}.{ext}").read_text() == emit(k, arch)




def test_the_pair_names_cannot_collide_with_the_pq_naming() -> None:
    for c in emit_cli.pair_classes(emit_cli.PAIR_L_MAX):
        for name in (dq_name(c), jab_name(c)):
            assert not name.startswith("pq_")


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_l_alone_writes_only_the_pq_ladder(backend, tmp_path):
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--L", "1") == 0
    assert _names(tmp_path) == {f"pq_L1_p1_q0.{ext}"}


@pytest.mark.parametrize("lmax", split_by_l(
    [1, 3], lambda lmax: 2 * lmax, ids=lambda lmax: f"pair-l{lmax}"))
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_pair_l_alone_writes_only_the_pair_ladder(backend, lmax, tmp_path):
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", str(lmax)) == 0
    want = {1: [(0, 0), (0, 1), (1, 1)], 3: PAIR_CLASSES_F}[lmax]
    assert _names(tmp_path) == _pair_names(ext, want)


def test_l_and_pair_l_together_write_both(tmp_path):
    assert _run(tmp_path, "cuda", "--L", "0", "--pair-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "dq_0_0.cu", "jab_00.cu"}


# --- the one-electron ladder (#242) ------------------------------------------


def _one_e_names(ext: str, lmax: int) -> set[str]:
    return {f"{name(c)}.{ext}" for c in emit_cli.one_e_classes(lmax)
            for name in emit_cli.ONE_E_NAMES}


def test_the_one_e_ladder_is_s_t_v_per_pair_class() -> None:
    assert emit_cli.DEFAULT_ONE_E_L == 3
    assert emit_cli.one_e_classes(3) == PAIR_CLASSES_F
    names = [k.name for k in emit_cli.one_e_kernels(2)]
    assert names == [f"{f}_{la}{lb}" for la, lb in emit_cli.pair_classes(2)
                     for f in "stv"]
    assert len(names) == 18


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_is_still_37_units_and_no_one_e_unit(backend, tmp_path):
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend) == 0
    got = _names(tmp_path)
    assert len(got) == 37
    assert not got & _one_e_names(ext, emit_cli.ONE_E_L_MAX)


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_default_ladder_holds_no_fock_unit(backend, tmp_path):
    """#287: the fused Fock kernel is behind `--fock-class` / `--fock-l`
    only. Neither the bare default nor KERNEL_GEN_CMD's `--one-e-l 2` writes
    an `f_` unit, and the default's 37 names are exactly the J chain's
    (their bytes: `test_the_pq_ladder_still_matches_the_measured_pins`)."""
    _, _, ext = BACKEND[backend]
    for argv in ([], ["--one-e-l", "2"]):
        d = tmp_path / ("bare" if not argv else "one-e")
        assert _run(d, backend, *argv) == 0
        assert not any(n.startswith("f_") for n in _names(d))
    assert _names(tmp_path / "bare") == _pq_names(ext) | _pair_names(ext)


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_one_e_l_adds_the_ladder_beside_the_j_chain(backend, tmp_path):
    """`--one-e-l 2` (KERNEL_GEN_CMD's) writes the 37 J units, byte for byte
    the bare command's, plus the 18 one-electron units in `emit`'s text."""
    emit, arch, ext = BACKEND[backend]
    bare, flagged = tmp_path / "bare", tmp_path / "flagged"
    assert _run(bare, backend) == 0
    assert _run(flagged, backend, "--one-e-l", "2") == 0
    one_e = _one_e_names(ext, 2)
    assert len(one_e) == 18
    assert _names(flagged) == _names(bare) | one_e
    for name in _names(bare):
        assert (flagged / name).read_bytes() == (bare / name).read_bytes()
    for k in emit_cli.one_e_kernels(2):
        assert (flagged / f"{k.name}.{ext}").read_text() == emit(k, arch)


@pytest.mark.slow  # bare `--one-e-l` is through (ff), L = 6
def test_one_e_l_bare_is_through_ff(tmp_path):
    assert _run(tmp_path, "cuda", "--L", "0", "--one-e-l") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu"} | _one_e_names("cu", 3)


def test_one_e_l_adds_to_the_ladder_flags(tmp_path):
    assert _run(tmp_path, "cuda", "--pair-l", "0", "--one-e-l", "0") == 0
    assert _names(tmp_path) == {"dq_0_0.cu", "jab_00.cu", "s_00.cu",
                                "t_00.cu", "v_00.cu"}


def test_the_one_e_names_cannot_collide_with_the_j_chain() -> None:
    assert emit_cli.ONE_E_L_MAX == emit_cli.PAIR_L_MAX  # one class list
    pairs = emit_cli.pair_classes(emit_cli.PAIR_L_MAX)
    j = ({pq_name(L) for L in range(17)}
         | {name(c) for c in pairs for name in (dq_name, jab_name)})
    one_e = {name(c) for c in pairs for name in emit_cli.ONE_E_NAMES}
    assert not j & one_e
    assert all(n[:2] in ("s_", "t_", "v_") for n in one_e)


# --- the Schwarz ladder -------------------------------------------------------


def test_the_schwarz_ladder_is_one_unit_per_pair_class_through_gg():
    assert emit_cli.schwarz_classes(1) == [(0, 0), (0, 1), (1, 1)]
    assert len(emit_cli.schwarz_classes(emit_cli.SCHWARZ_L_MAX)) == 15
    assert emit_cli.SCHWARZ_L_MAX == 4  # (gg|gg) at L = 16, [p|q]'s top
    with pytest.raises(ValueError):
        emit_cli.schwarz_classes(emit_cli.SCHWARZ_L_MAX + 1)


def test_schwarz_l_adds_its_units_beside_the_ladder_flags(tmp_path):
    from felizjueves.kernel_schwarz import build_schwarz_kernel

    assert _run(tmp_path, "cuda", "--L", "0", "--schwarz-l", "1") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "schwarz_00.cu",
                                "schwarz_01.cu", "schwarz_11.cu"}
    emit, arch, _ = BACKEND["cuda"]
    assert (tmp_path / "schwarz_01.cu").read_text() == emit(
        build_schwarz_kernel((0, 1)), arch)


@pytest.mark.slow  # bare `--schwarz-l` is through (dd), (dd|dd) at L = 8
def test_schwarz_l_bare_is_through_dd(tmp_path):
    assert _run(tmp_path, "hip", "--pair-l", "0", "--schwarz-l") == 0
    assert _names(tmp_path) == {"dq_0_0.hip", "jab_00.hip"} | {
        f"schwarz_{a}{b}.hip" for a, b in emit_cli.schwarz_classes(
            emit_cli.DEFAULT_SCHWARZ_L)}


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_the_bare_default_writes_no_schwarz_unit(backend, tmp_path):
    assert _run(tmp_path, backend) == 0
    assert not any(n.startswith("schwarz_") for n in _names(tmp_path))


@pytest.mark.parametrize("argv", [
    ["--schwarz-l", "-1"],
    ["--schwarz-l", str(emit_cli.SCHWARZ_L_MAX + 1)],
    ["--schwarz-l", "1", "--class", "0", "0", "0", "0"],
    ["--schwarz-l", "1", "--fock-l", "0"],
])
def test_a_schwarz_l_it_cannot_honour_is_a_2(argv, tmp_path):
    assert _run(tmp_path, "cuda", *argv) == 2
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


# --- the cross-basis overlap ladder -------------------------------------------


def test_the_sx_ladder_is_every_ordered_class():
    assert emit_cli.sx_classes(1) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len(emit_cli.sx_classes(emit_cli.DEFAULT_SX_L)) == 9
    with pytest.raises(ValueError):
        emit_cli.sx_classes(emit_cli.ONE_E_L_MAX + 1)


def test_sx_l_adds_its_units_beside_the_ladder_flags(tmp_path):
    from felizjueves.one_electron.kernel_sx import build_sx_kernel

    assert _run(tmp_path / "a", "cuda", "--L", "0", "--sx-l", "1") == 0
    assert _names(tmp_path / "a") == {"pq_L0_p0_q0.cu", "sx_00.cu",
                                      "sx_01.cu", "sx_10.cu", "sx_11.cu"}
    emit, arch, _ = BACKEND["cuda"]
    assert (tmp_path / "a" / "sx_10.cu").read_text() == emit(
        build_sx_kernel((1, 0)), arch)
    assert _run(tmp_path / "b", "hip", "--pair-l", "0", "--sx-l") == 0
    assert _names(tmp_path / "b") == {"dq_0_0.hip", "jab_00.hip"} | {
        f"sx_{a}{b}.hip" for a, b in emit_cli.sx_classes(
            emit_cli.DEFAULT_SX_L)}


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_the_bare_default_writes_no_sx_unit(backend, tmp_path):
    assert _run(tmp_path, backend) == 0
    assert not any(n.startswith("sx_") for n in _names(tmp_path))


@pytest.mark.parametrize("argv", [
    ["--sx-l", "-1"],
    ["--sx-l", str(emit_cli.ONE_E_L_MAX + 1)],
    ["--sx-l", "1", "--class", "0", "0", "0", "0"],
    ["--sx-l", "1", "--fock-l", "0"],
])
def test_an_sx_l_it_cannot_honour_is_a_2(argv, tmp_path):
    assert _run(tmp_path, "cuda", *argv) == 2
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


# --- the DF metric ladder (#415; goldens in test/test_emit_metric.py) ---------


def test_the_metric_ladder_bounds_the_total_l():
    assert emit_cli.metric_classes(2) == [(0, 0), (0, 1), (0, 2), (1, 1)]
    assert len(emit_cli.metric_classes(emit_cli.DEFAULT_METRIC_L)) == 9
    assert emit_cli.metric_classes(40)  # any L: listed, never built here


def test_metric_l_adds_its_units_beside_the_ladder_flags(tmp_path):
    from felizjueves.kernel_metric import build_metric_kernel

    assert _run(tmp_path, "hip", "--pair-l", "0", "--metric-l", "2") == 0
    assert _names(tmp_path) == {"dq_0_0.hip", "jab_00.hip", "metric_00.hip",
                                "metric_01.hip", "metric_02.hip",
                                "metric_11.hip"}
    emit, arch, _ = BACKEND["hip"]
    assert (tmp_path / "metric_02.hip").read_text() == emit(
        build_metric_kernel((0, 2)), arch)


def test_kernel_gen_cmd_asks_for_no_metric_unit():
    cfg = (ROOT / "devtools" / "config.sh").read_text()
    (line,) = [ln for ln in cfg.splitlines()
               if ln.lstrip().startswith("KERNEL_GEN_CMD=")]
    assert "felizjueves.emit" in line and "--metric-l" not in line


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_metric_unit(backend, tmp_path):
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend) == 0
    got = _names(tmp_path)
    assert len(got) == 37 and got == _pq_names(ext) | _pair_names(ext)
    assert not any(n.startswith("metric_") for n in got)


@pytest.mark.parametrize("argv", [
    ["--metric-l", "-1"],
    ["--metric-l", "x"],
    ["--metric-l", "1", "--class", "0", "0", "0", "0"],
    ["--metric-l", "1", "--fock-l", "0"],
])
def test_a_metric_l_it_cannot_honour_is_a_2(argv, tmp_path):
    assert _run(tmp_path, "cuda", *argv) == 2
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


# --- the three-center ladder (#445; goldens in test/test_emit_eri3.py) --------


def test_the_eri3_ladder_bounds_the_total_l():
    assert emit_cli.eri3_classes(1) == [(0, 0, 0), (0, 0, 1), (0, 1, 0)]
    assert len(emit_cli.eri3_classes(emit_cli.DEFAULT_ERI3_L)) == 22
    assert emit_cli.eri3_classes(40)  # any L: listed, never built here


def test_the_eri3_names_cannot_collide_with_any_other_ladder():
    # #419's shape: every name from its name function, no IR built, every
    # rung of every ladder the CLI writes beside it.
    pairs = emit_cli.pair_classes(emit_cli.PAIR_L_MAX)
    others = ({emit_cli.source_name(L, "cuda") for L in range(17)}
              | {f"{name(c)}.cu" for c in pairs
                 for name in (dq_name, jab_name, *emit_cli.ONE_E_NAMES)}
              | {f"schwarz_{a}{b}.cu" for a, b in emit_cli.schwarz_classes(
                  emit_cli.SCHWARZ_L_MAX)}
              | {f"sx_{a}{b}.cu" for a, b in emit_cli.sx_classes(
                  emit_cli.ONE_E_L_MAX)}
              | {"metric_" + "".join(map(str, c)) + ".cu"
                 for c in emit_cli.metric_classes(16)}
              | {f"f_{''.join(map(str, c))}.cu" for c in emit_cli.FOCK_CLASSES})
    classes = emit_cli.eri3_classes(40)
    eri3 = [f"{eri3_name(c)}.cu" for c in classes]
    assert len(set(eri3)) == len(eri3)  # no two classes share a file
    assert not set(eri3) & others
    assert all(n.startswith("eri3_") for n in eri3)
    assert not any(n.startswith("eri3_") for n in others)


def test_the_eri3_name_separates_its_ls_only_above_nine():
    # Every l <= 9: three digits run together, the L <= 4 goldens' spelling.
    assert eri3_name((2, 2, 0)) == "eri3_220"
    assert eri3_name((9, 9, 9)) == "eri3_999"
    # Once any l is two digits the ls are `_`-separated, so the classes
    # that ran together as `eri3_0110` are two names.
    assert eri3_name((0, 11, 0)) == "eri3_0_11_0"
    assert eri3_name((0, 1, 10)) == "eri3_0_1_10"
    assert eri3_name((1, 10, 0)) != eri3_name((11, 0, 0))


def test_eri3_l_11_writes_every_class_to_its_own_file(tmp_path, monkeypatch):
    # The overwrite #456 had: (0,11,0) and (0,1,10) to one file. Names
    # only -- the builder is stubbed, so no IR is built at L = 11.
    import types

    from felizjueves.kernel_eri3 import eri3_spec

    monkeypatch.setattr(emit_cli, "build_eri3_kernel", lambda c: (
        types.SimpleNamespace(name=eri3_name(eri3_spec(*c)))))
    monkeypatch.setattr(emit_cli, "_write", lambda backend, arch, d, k: (
        d / f"{k.name}.cu").open("x").close() or d / f"{k.name}.cu")
    assert _run(tmp_path, "cuda", "--L", "0", "--eri3-l", "11") == 0
    want = {f"{eri3_name(c)}.cu" for c in emit_cli.eri3_classes(11)}
    assert len(want) == len(emit_cli.eri3_classes(11))
    assert _names(tmp_path) == want | {"pq_L0_p0_q0.cu"}
    assert {"eri3_0_11_0.cu", "eri3_0_1_10.cu"} <= want


def test_eri3_l_adds_its_units_beside_the_ladder_flags(tmp_path):
    from felizjueves.kernel_eri3 import build_eri3_kernel

    assert _run(tmp_path, "hip", "--pair-l", "0", "--eri3-l", "1") == 0
    assert _names(tmp_path) == {"dq_0_0.hip", "jab_00.hip", "eri3_000.hip",
                                "eri3_001.hip", "eri3_010.hip"}
    emit, arch, _ = BACKEND["hip"]
    assert (tmp_path / "eri3_010.hip").read_text() == emit(
        build_eri3_kernel((0, 1, 0)), arch)


def test_kernel_gen_cmd_asks_for_no_eri3_unit():
    cfg = (ROOT / "devtools" / "config.sh").read_text()
    (line,) = [ln for ln in cfg.splitlines()
               if ln.lstrip().startswith("KERNEL_GEN_CMD=")]
    assert "felizjueves.emit" in line and "--eri3-l" not in line


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_eri3_unit(backend, tmp_path):
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend) == 0
    got = _names(tmp_path)
    assert len(got) == 37 and got == _pq_names(ext) | _pair_names(ext)
    assert not any(n.startswith("eri3_") for n in got)


@pytest.mark.parametrize("argv", [
    ["--eri3-l", "-1"],
    ["--eri3-l", "x"],
    ["--eri3-l", "1", "--class", "0", "0", "0", "0"],
    ["--eri3-l", "1", "--shell-ls", "0"],
    ["--eri3-l", "1", "--fock-l", "0"],
    ["--eri3-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--eri3-l", "1", "--n-batches", "3"],
])
def test_an_eri3_l_it_cannot_honour_is_a_2(argv, tmp_path):
    assert _run(tmp_path, "cuda", *argv) == 2
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


@pytest.mark.parametrize("argv", [
    ["--one-e-l", "-1"],
    ["--one-e-l", str(emit_cli.ONE_E_L_MAX + 1)],
    ["--one-e-l", "x"],
    ["--one-e-l", "1", "--class", "0", "0", "0", "0"],
    ["--one-e-l", "1", "--shell-ls", "0", "1"],
])
def test_a_one_e_l_it_cannot_honour_is_a_2(argv, tmp_path):
    assert _run(tmp_path, "cuda", *argv) == 2
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


# --- exit status ------------------------------------------------------------


@pytest.mark.parametrize("argv", [
    ["--pair-l", "-1"],
    ["--pair-l", str(emit_cli.PAIR_L_MAX + 1)],
    ["--pair-l", "x"],
    ["--pair-l", "1", "--class", "0", "0", "0", "0"],
    ["--pair-l", "1", "--shell-ls", "0", "1"],
])
def test_a_pair_l_it_cannot_honour_is_a_2(argv, tmp_path):
    assert _run(tmp_path, "cuda", *argv) == 2
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


# --- the selector writes the chain ------------------------------------------


@pytest.mark.parametrize("name", [  # the basis's top class: (pp|pp), (dd|dd)
    by_l("h2o_sto3g_j", 4), by_l("h2o_ccpvdz_j", 8)])
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_selector_writes_every_kernel_the_chain_needs(name, backend, tmp_path):
    """Each cell holds its ket's DQ, its [p|q] batches and its bra's JAB, and
    nothing else; `order` names them DQ -> [p|q] batches -> JAB."""
    emit, arch, ext = BACKEND[backend]
    basis = _case(name)[1]
    ls = [s.l for s in basis.shells]
    assert _run(tmp_path, backend, "--shell-ls", *map(str, ls)) == 0
    classes = _classes(basis)
    assert set(emit_cli.classes_reached(ls)) == classes
    assert _names(tmp_path) == {emit_cli.cell_dir(am, 1) for am in classes}
    dq_seen, jab_seen = set(), set()
    for am in classes:
        cell = tmp_path / emit_cli.cell_dir(am, 1)
        dq = build_dq_kernel(am[2:])
        jab = build_jab_kernel(am[:2])
        pq = build_pq_kernels(emit_cli.class_spec(am, 1))
        want = [dq, *pq, jab]
        order = cell.joinpath("order").read_text().split()
        assert order == [f"{k.name}.{ext}" for k in want]
        assert _names(cell) == {"order", *order}
        for k in want:
            assert (cell / f"{k.name}.{ext}").read_text() == emit(k, arch)
        dq_seen.add(am[2:])
        jab_seen.add(am[:2])
    # Every pair class the sweep reaches has its DQ and its JAB somewhere.
    assert dq_seen == jab_seen == _shell_ams(basis)


def test_the_selector_order_is_dq_batches_jab_for_a_split_cell(tmp_path):
    am = (1, 1, 1, 1)
    assert _run(tmp_path, "cuda", "--class", *map(str, am), "--n-batches", "3") == 0
    order = (tmp_path / emit_cli.cell_dir(am, 3) / "order").read_text().split()
    pq = build_pq_kernels(emit_cli.class_spec(am, 3))
    assert len(pq) == 3
    assert order == ["dq_1_1.cu", *(f"{k.name}.cu" for k in pq), "jab_11.cu"]


# --- the lever ladders (#339) -------------------------------------------------
#
# The flags are DATA: one per (kernel-body lever, family), read from
# `levers.VOCABULARY` through `lever_ladder.flags()` each time the parser is
# built. Beside the shipped `fused`, these tests declare a PROBE lever
# (`monkeypatch.setitem`), the way a lever issue declares its entry -- and the
# CLI is never edited for it.

PROBE = "probe"


def _declare(monkeypatch, legal, name=PROBE):
    monkeypatch.setitem(levers.VOCABULARY, name, levers.Lever.of(
        name, 339, "the ladder's witness; no statement changes", legal))


def _probe_flags() -> list[str]:
    return [lf.flag for lf in lever_ladder.flags() if lf.lever == PROBE]


def test_only_a_declared_lever_has_a_flag(tmp_path):
    # The shipped vocabulary: one flag per (lever, family it is legal on) --
    # `fused` (#344) on J, `split` (#341) on K, and whatever follows.
    suffix = {s: f.suffix for s, f in lever_ladder.FAMILIES.items()}
    assert [lf.flag for lf in lever_ladder.flags()] == [
        f"--klever-{name}-{suffix[s]}" for name in sorted(levers.VOCABULARY)
        for s in lever_ladder.FAMILIES if s in levers.VOCABULARY[name].legal]
    got = {lf.flag for lf in lever_ladder.flags()}
    assert {"--klever-fused-pq", "--klever-split-k"} <= got
    assert _run(tmp_path, "cuda", "--klever-probe-pq", "1") == 2


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_fused_ladder_writes_the_fused_goldens_under_lever_names(
        backend, tmp_path):
    """`--klever-fused-pq 4` writes #344's fused golden for (2,0|2,0) --
    the same text, its kernel renamed `<literal>_lv_fused` -- beside the
    literal unit, which does not move."""
    emit, arch, ext = BACKEND[backend]
    assert _run(tmp_path / "lit", backend, "--L", "4") == 0
    assert _run(tmp_path / "lv", backend, "--L", "4",
                "--klever-fused-pq", "4") == 0
    lv = tmp_path / "lv"
    assert _names(lv) == {f"pq_L4_p2_q2.{ext}", f"pq_L4_p2_q2_lv_fused.{ext}"}
    assert (lv / f"pq_L4_p2_q2.{ext}").read_bytes() == \
        (tmp_path / "lit" / f"pq_L4_p2_q2.{ext}").read_bytes()
    golden = (ROOT / "test" / "golden" / backend / "fused"
              / f"pq_L4_p2_q2.{ext}").read_text()
    got = (lv / f"pq_L4_p2_q2_lv_fused.{ext}").read_text()
    # the kernel's name, in its header comment and its signature, is the
    # only difference
    assert golden.count("pq_L4_p2_q2") == 2
    assert got == golden.replace("pq_L4_p2_q2", "pq_L4_p2_q2_lv_fused")


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_a_declared_lever_moves_no_unflagged_unit(backend, monkeypatch,
                                                  tmp_path):
    """Declaring a lever adds its flags and nothing else: an unflagged
    command writes the same units, byte for byte (L <= 2 here; the bare
    37-unit default is the `slow` twin below)."""
    argv = ("--L", "0", "1", "2", "--pair-l", "1")
    assert _run(tmp_path / "before", backend, *argv) == 0
    _declare(monkeypatch, {"j", "k", "f"})
    assert _probe_flags() == [
        "--klever-probe-pq", "--klever-probe-k", "--klever-probe-fock-l"]
    assert _run(tmp_path / "after", backend, *argv) == 0
    names = _names(tmp_path / "before")
    assert _names(tmp_path / "after") == names and len(names) == 9
    for n in names:
        assert (tmp_path / "after" / n).read_bytes() == \
            (tmp_path / "before" / n).read_bytes()


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_a_declared_lever_gets_its_flags_and_the_default_is_unmoved(
        backend, monkeypatch, tmp_path):
    """Declaring a lever adds its flags and nothing else: the bare command
    still writes the 37 J-chain units, byte for byte (the pins above)."""
    _declare(monkeypatch, {"j", "k", "f"})
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend) == 0
    assert _names(tmp_path) == _pq_names(ext) | _pair_names(ext)
    for L in range(17):
        k = build_pq_kernel(L)
        assert (tmp_path / f"{k.name}.{ext}").read_text() == \
            BACKEND[backend][0](k, BACKEND[backend][1])


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_pq_lever_ladder_adds_its_units_beside_the_literal(
        backend, monkeypatch, tmp_path):
    _declare(monkeypatch, {"j"})
    emit, arch, ext = BACKEND[backend]
    lit = tmp_path / "lit"
    assert _run(lit, backend, "--L", "1", "2") == 0
    assert _run(tmp_path / "lv", backend, "--L", "1", "2",
                "--klever-probe-pq", "0", "2") == 0
    lv = tmp_path / "lv"
    assert _names(lv) == _names(lit) | {f"pq_L0_p0_q0_lv_probe.{ext}",
                                        f"pq_L2_p1_q1_lv_probe.{ext}"}
    for n in _names(lit):  # the literal units did not move
        assert (lv / n).read_bytes() == (lit / n).read_bytes()
    for L in (0, 2):
        k = build_pq_kernel(pq_spec(L, levers=(PROBE,)))
        k = replace(k, name=f"{k.name}_lv_probe")
        assert (lv / f"{k.name}.{ext}").read_text() == emit(k, arch)
        assert "lever probe (#339)" in emit(k, arch)  # the builder's Note


def test_the_k_lever_ladder_adds_beside_the_selected_ladders(monkeypatch,
                                                            tmp_path):
    _declare(monkeypatch, {"j", "k"})
    assert _run(tmp_path, "hip", "--L", "0", "--klever-probe-k", "1", "2") == 0
    ks = [build_k_kernel(replace(k_spec(L), levers=(PROBE,))) for L in (1, 2)]
    assert _names(tmp_path) == {"pq_L0_p0_q0.hip"} | {
        f"{k.name}_lv_probe.hip" for k in ks}
    for k in ks:
        assert "lever probe (#339)" in (
            tmp_path / f"{k.name}_lv_probe.hip").read_text()


@pytest.mark.slow  # the bare default builds L up to 16
def test_a_lever_flag_alone_adds_to_the_default_ladders(monkeypatch, tmp_path):
    _declare(monkeypatch, {"j", "k"})
    assert _run(tmp_path, "cuda", "--klever-probe-pq", "0",
                "--klever-probe-k", "2") == 0
    k2 = build_k_kernel(replace(k_spec(2), levers=(PROBE,)))
    assert _names(tmp_path) == (_pq_names("cu") | _pair_names("cu")
                                | {"pq_L0_p0_q0_lv_probe.cu",
                                   f"{k2.name}_lv_probe.cu"})
    assert k2.name.startswith("k_")


def test_the_f_lever_flag_is_on_the_fock_path(monkeypatch, tmp_path):
    _declare(monkeypatch, {"f"})
    assert _probe_flags() == ["--klever-probe-fock-l"]
    assert _run(tmp_path / "a", "cuda", "--klever-probe-fock-l", "0") == 0
    assert _names(tmp_path / "a") == {"f_0000_lv_probe.cu"}
    assert _run(tmp_path / "b", "hip", "--fock-l", "0",
                "--klever-probe-fock-l", "1") == 0
    assert _names(tmp_path / "b") == {"f_0000.hip"} | {
        f"f_{''.join(map(str, c))}_lv_probe.hip"
        for c in emit_cli.fock_classes(1)}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_split_k_ladder_writes_every_element_kernel(backend, tmp_path):
    """#341's `split` fans a K class out into `_e{i}_{j}` element kernels;
    `--klever-split-k` writes every one, `build_k_kernels`' text renamed
    `<name>_lv_split`."""
    emit, arch, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--L", "0", "--klever-split-k", "1", "2") == 0
    want = [k for L in (1, 2)
            for k in build_k_kernels(replace(k_spec(L), levers=("split",)))]
    assert len(want) > 2 and all("_e" in k.name for k in want)
    assert _names(tmp_path) == {"pq_L0_p0_q0." + ext} | {
        f"{k.name}_lv_split.{ext}" for k in want}
    for k in want:
        lv = replace(k, name=f"{k.name}_lv_split")
        assert (tmp_path / f"{lv.name}.{ext}").read_text() == emit(lv, arch)


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_an_f_lever_unit_differs_from_literal_by_its_note_and_name(
        backend, monkeypatch, tmp_path):
    """#340's CLI half: an F-legal lever's F units through the ladder are
    `kernel_f`'s literal text plus the lever's provenance Note (#340) --
    and, as every lever unit, the `_lv_` name. Nothing else moves."""
    _declare(monkeypatch, {"f"})
    emit, arch, ext = BACKEND[backend]
    assert _run(tmp_path / "lit", backend, "--fock-l", "1") == 0
    assert _run(tmp_path / "lv", backend, "--klever-probe-fock-l", "1") == 0
    note = "lever probe (#339): the ladder's witness; no statement changes"
    for c in emit_cli.fock_classes(1):
        name = "f_" + "".join(map(str, c))
        lit = (tmp_path / "lit" / f"{name}.{ext}").read_text()
        got = (tmp_path / "lv" / f"{name}_lv_probe.{ext}").read_text()
        assert note not in lit and got.count(note) == 1
        renamed = got.replace(f"{name}_lv_probe", name)
        dropped = [ln for ln in renamed.splitlines()
                   if ln not in lit.splitlines()]
        assert dropped == [ln for ln in renamed.splitlines() if note in ln]
        assert [ln for ln in renamed.splitlines() if note not in ln] == \
            lit.splitlines()


@pytest.mark.parametrize("argv", [
    ["--klever-probe-k", "1"],  # illegal on K: no flag at all
    ["--klever-probe-pq", "-1"],
    ["--klever-probe-pq", "x"],
    ["--klever-probe-pq", "1", "--class", "0", "0", "0", "0"],
    ["--klever-probe-pq", "1", "--fock-l", "0"],
    ["--klever-probe-fock-l", str(emit_cli.FOCK_L_MAX + 1)],
    ["--klever-probe-fock-l", "0", "--L", "0"],
    ["--klever-probe-fock-l", "0", "--one-e-l", "0"],
])
def test_a_lever_flag_it_cannot_honour_is_a_2(argv, monkeypatch, tmp_path):
    _declare(monkeypatch, {"j", "f"})
    assert _run(tmp_path, "cuda", *argv) == 2
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


# --- the HGP split J / K path (#407; goldens in test/test_emit_hjk.py) --------


def test_the_bare_default_never_enters_the_hgp_path(monkeypatch, tmp_path):
    # The bare command's 37 units are built elsewhere (slow); here the J
    # ladders are stubbed and the HGP writer made to fail if reached.
    def boom(*a, **k):
        raise AssertionError("the bare default reached the HGP path")

    monkeypatch.setattr(emit_cli, "generate_hgp", boom)
    monkeypatch.setattr(emit_cli, "hgp_selection", boom)
    monkeypatch.setattr(emit_cli, "generate", lambda *a: [])
    monkeypatch.setattr(emit_cli, "generate_pairs", lambda *a: [])
    assert _run(tmp_path, "cuda") == 0
    assert _run(tmp_path, "hip", "--one-e-l", "0") == 0


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_hgp_jk_unit(backend, tmp_path):
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend) == 0
    got = _names(tmp_path)
    assert len(got) == 37 and got == _pq_names(ext) | _pair_names(ext)
    assert not any(n.startswith(("hj_", "hk_")) for n in got)


def test_kernel_gen_cmd_asks_for_no_hgp_jk_unit():
    cfg = (ROOT / "devtools" / "config.sh").read_text()
    (line,) = [ln for ln in cfg.splitlines()
               if ln.lstrip().startswith("KERNEL_GEN_CMD=")]
    assert "felizjueves.emit" in line and "--hgp" not in line


def test_the_hgp_jk_names_cannot_collide_with_any_other_ladder():
    from felizjueves.kernel_hj import hj_class_spec, hj_name
    from felizjueves.kernel_hk import hk_class_spec, hk_name

    # Ordered names only (no DAG built); a canonical name is an ordered
    # class's digits behind `hj_` / `hk_`, the same prefix test.
    classes = emit_cli.hgp_ordered_classes(4)
    names = [n(key(*c, ordered=o)) for c in classes for o in (False, True)
             for n, key in ((hj_name, hj_class_spec), (hk_name, hk_class_spec))
             if o or sum(c) <= 2]
    assert all(n.startswith(("hj_", "hk_")) for n in names)
    pairs = emit_cli.pair_classes(emit_cli.PAIR_L_MAX)
    others = ({emit_cli.source_name(L, "cuda")[:-3] for L in range(17)}
              | {name(c) for c in pairs
                 for name in (dq_name, jab_name, *emit_cli.ONE_E_NAMES)}
              | {eri3_name(c) for c in emit_cli.eri3_classes(8)}
              | {f"f_{''.join(map(str, c))}" for c in emit_cli.FOCK_CLASSES})
    assert not set(names) & others
    assert not any(n.startswith(("hj_", "hk_")) for n in others)
