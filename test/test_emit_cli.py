"""`python -m felizjueves.emit` as KERNEL_GEN_CMD's contract (#156).

ONE LADDER PER KERNEL FAMILY. The no-argument default writes the `[p|q]`
ladder (L = 0..16, unchanged: the measured store pins each unit's
`source_sha256`) PLUS the pair ladder (`DQ` and `JAB` for every pair class
`(la, lb)`, `la <= lb <= 3`: 10 classes, 20 units). `--L` alone writes only
`[p|q]`, `--pair-l` alone only the pair ladder.

THE SELECTOR writes, per `(class, n_batches)` cell, the chain one class
needs -- its ket's `DQ`, its `[p|q]` batch kernels, its bra's `JAB` -- with
an `order` file naming them in that launch order.

Exit status: 0 written, 2 could not run -- no third meaning.

Fast tier: nothing here compiles anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_ao_j_interpreted import _case, _classes, _shell_ams

from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.kernel import build_pq_kernel, build_pq_kernels
from felizjueves.kernel_dq import build_dq_kernel
from felizjueves.kernel_jab import build_jab_kernel

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
    return {f"{build_pq_kernel(L).name}.{ext}" for L in range(17)}


def _pair_names(ext: str, classes=PAIR_CLASSES_F) -> set[str]:
    return ({f"{build_dq_kernel(c).name}.{ext}" for c in classes}
            | {f"{build_jab_kernel(c).name}.{ext}" for c in classes})


# --- the ladders ------------------------------------------------------------


def test_the_pair_ladder_is_ss_through_ff() -> None:
    assert emit_cli.DEFAULT_PAIR_L == 3
    assert emit_cli.pair_classes(3) == PAIR_CLASSES_F
    assert len(PAIR_CLASSES_F) == 10
    assert all(la <= lb for la, lb in emit_cli.pair_classes(emit_cli.PAIR_L_MAX))


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
        for k in (build_dq_kernel(c), build_jab_kernel(c)):
            assert not k.name.startswith("pq_")


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_l_alone_writes_only_the_pq_ladder(backend, tmp_path):
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--L", "1") == 0
    assert _names(tmp_path) == {f"pq_L1_p1_q0.{ext}"}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_pair_l_alone_writes_only_the_pair_ladder(backend, tmp_path):
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "1") == 0
    assert _names(tmp_path) == _pair_names(ext, [(0, 0), (0, 1), (1, 1)])
    other = tmp_path / "f"
    assert _run(other, backend, "--pair-l", "3") == 0
    assert _names(other) == _pair_names(ext)


def test_l_and_pair_l_together_write_both(tmp_path):
    assert _run(tmp_path, "cuda", "--L", "0", "--pair-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "dq_0_0.cu", "jab_00.cu"}


def test_the_pq_ladder_still_matches_the_measured_pins(tmp_path):
    """Every literal one-batch row's `source_sha256` is the text the default
    writes: the `[p|q]` ladder's bytes did not move."""
    rows = json.loads(
        (ROOT / "data" / "measured" / "kernels.json").read_text())["rows"]
    for backend in BACKEND:
        assert _run(tmp_path / backend, backend) == 0
    checked = 0
    for key, row in rows.items():
        if "/b1/j/literal@" not in key:
            continue  # a split cell, or a lever's arm: not the ladder's unit
        backend = {"sm_86": "cuda", "gfx1200": "hip"}[row["arch"]]
        path = tmp_path / backend / f"{row['kernel']}.{BACKEND[backend][2]}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == \
            row["source_sha256"], key
        checked += 1
    assert checked == 34  # L = 0..16 on both arches


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


@pytest.mark.parametrize("name", ["h2o_sto3g_j", "h2o_ccpvdz_j"])
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
