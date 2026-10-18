"""The cross-tree ptxas baseline on sm_86 (#289), held against its file.

data/measured/fock_baseline.json puts the generated F kernel's ptxas -v
numbers beside the predecessor's frozen PTXAS_MEASURED rows, both under the
predecessor's flags. It is a QUALITY BASELINE, NEVER A NUMERIC WITNESS (same
ancestry; data/literature/fock/limits.json), and nothing here grades a delta:
there is no threshold and no ratchet. What is tested:

  shape     every row carries both trees' four numbers, and each tree its
            toolchain, arch, flags and date;
  mapping   the predecessor's `a >= b` keys and this repo's `la <= lb` ladder
            (hgp.vrr.CLASSES) are joined by an explicit rule, asserted a
            bijection on the eleven cells before any number is read;
  boundary  the file's structural reading (where the register ceiling falls,
            and which classes crossed) is recomputed from its own rows;
  source    each row's `source_sha256` is the unit this tree emits today, and
            agrees with kernels.json's sm_86 F row where one exists (#288);
  compile   (`cuda`: a compiler, no card) each unit re-compiled under the
            file's flags reproduces its row -- the check #290 re-runs; (dd|dd)
            is also `slow`.

A CHECK OF MEASUREMENT ARTEFACTS, moved from test/ (the mapping's
bijection, which is code, stayed as test/test_fock_baseline.py, and is
imported here): `devtools/measure.sh check` runs the reads, `recheck
--backend cuda` the compile cases at the bottom; the source check of a row
above L = FAST_L_MAX (tiers.py), which builds and emits it, is `slow`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from test_fock_baseline import (
    PREDECESSOR_CELLS,
    canonical,
)
from tiers import by_l

from felizjueves.device import SM_86
from felizjueves.emit import cuda
from felizjueves.hgp import vrr
from felizjueves.kernel_f import build_f_kernel
from felizjueves.measured import parse_ptxas

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data" / "measured" / "fock_baseline.json"
KERNELS = ROOT / "data" / "measured" / "kernels.json"
FLAGS = "-dc -O2 --std=c++17 -arch=sm_86"
FIELDS = ("registers", "spill_store_bytes", "spill_load_bytes",
          "stack_frame_bytes")
SLOW = {(2, 2, 2, 2)}  # ~15 s under the predecessor's flags


def _doc() -> dict:
    return json.loads(BASELINE.read_text())


def _rows() -> list[dict]:
    return _doc()["rows"]


def label(cls) -> str:
    return "{}{}|{}{}".format(*("spdf"[x] for x in cls))


def _fits(nums: dict) -> bool:
    return nums["spill_store_bytes"] == 0 and nums["spill_load_bytes"] == 0


# -- shape ------------------------------------------------------------------------


def test_the_file_says_what_it_is_not() -> None:
    doc = _doc()
    assert doc["schema"] == 1
    assert "NEVER A NUMERIC WITNESS" in doc["about"]
    assert "No threshold, no ratchet" in doc["about"]
    assert doc["flags"] == FLAGS and doc["arch"] == "sm_86"


@pytest.mark.parametrize("tree", ["predecessor", "this_tree"])
def test_each_tree_records_its_toolchain_arch_flags_and_date(tree) -> None:
    side = _doc()[tree]
    assert side["toolchain"].startswith("nvcc 13.0")
    assert side["arch"] == "sm_86"
    assert side["flags"] == FLAGS
    assert len(side["date"]) == 10 and side["date"].startswith("2026-")


def test_the_predecessor_rows_carry_their_provenance() -> None:
    p = _doc()["predecessor"]
    assert p["repo"] == "whatDiracIsCooking/fock-build-generator-private"
    assert p["path"] == "test/hgp/brc/ptxas_data.py"
    assert p["table"] == "PTXAS_MEASURED"
    assert len(p["blob_sha"]) == 40


def test_every_row_has_both_trees_four_numbers() -> None:
    for r in _rows():
        for tree in ("predecessor", "this_tree"):
            nums = r[tree]
            assert all(type(nums[f]) is int and nums[f] >= 0 for f in FIELDS)
            # the ceiling: a spilling class pins at it, in either tree
            if not _fits(nums):
                assert nums["registers"] == _doc()["register_ceiling"] == 255
        assert r["L"] == sum(r["class"]) == sum(r["predecessor_class"])


def test_the_file_joins_by_the_mapping_and_nothing_else() -> None:
    rows = _rows()
    assert [tuple(r["predecessor_class"]) for r in rows] == list(PREDECESSOR_CELLS)
    for r in rows:
        p, o = tuple(r["predecessor_class"]), tuple(r["class"])
        assert o == canonical(p)
        assert r["predecessor_label"] == label(p) and r["label"] == label(o)
        assert r["this_tree"]["kernel"] == "f_" + "".join(map(str, o))


def test_the_boundary_targets_are_recomputed() -> None:
    for r in _rows():
        assert r["boundary_targets"] == {
            "predecessor_orientation": len(vrr.target_set(r["predecessor_class"])),
            "this_orientation": len(vrr.target_set(r["class"]))}


# -- the structural reading -------------------------------------------------------


def _side(tree: str) -> dict:
    rows = _rows()
    fits = [r for r in rows if _fits(r[tree])]
    spills = [r for r in rows if not _fits(r[tree])]
    big = max(fits, key=lambda r: r[tree]["registers"])
    small = min(spills, key=lambda r: r[tree]["spill_store_bytes"])
    return {"fits": [r["predecessor_label"] for r in fits],
            "spills": [r["predecessor_label"] for r in spills],
            "largest_fitting": {"class": big["predecessor_label"],
                                "registers": big[tree]["registers"]},
            "smallest_spilling": {"class": small["predecessor_label"],
                                  "spill_store_bytes":
                                      small[tree]["spill_store_bytes"]}}


def test_the_predecessor_boundary_is_where_its_table_puts_it() -> None:
    # ds|ps fits at 233 registers, pp|ps spills 264 B (the issue's statement)
    got = _side("predecessor")
    assert got["largest_fitting"] == {"class": "ds|ps", "registers": 233}
    assert got["smallest_spilling"] == {"class": "pp|ps",
                                        "spill_store_bytes": 264}


@pytest.mark.parametrize("tree", ["predecessor", "this_tree"])
def test_the_recorded_boundary_is_the_rows(tree) -> None:
    rec = dict(_doc()["boundary"][tree])
    rec.pop("between")
    assert rec == _side(tree)


def test_the_crossed_classes_are_the_rows() -> None:
    b = _doc()["boundary"]
    pred, ours = _side("predecessor"), _side("this_tree")
    crossed = [{"class": c,
                "predecessor": "spills" if c in pred["spills"] else "fits",
                "this_tree": "spills" if c in ours["spills"] else "fits"}
               for c in (r["predecessor_label"] for r in _rows())
               if (c in pred["spills"]) != (c in ours["spills"])]
    assert b["crossed"] == crossed
    assert b["same_place"] is (not crossed)


def test_the_reading_is_written_down() -> None:
    reading = _doc()["reading"]
    assert reading["match"].strip() and reading["mismatch"].strip()
    assert 1 <= len(reading["hypotheses"]) <= 2
    assert all("hypothesis" in h for h in reading["hypotheses"])


def test_the_deferred_tiers_addendum_is_the_kernels_json_rows() -> None:
    # #290: on sm_86 (repo flags) every class of the ladder with at most 40
    # boundary targets fits and every class with 76 or more spills -- the
    # split by target count, not by L, that the addendum states
    rows = json.loads(KERNELS.read_text())["rows"]
    assert "#290" in _doc()["reading"]["deferred_tier"]
    fits, spills = set(), set()
    for cls in vrr.CLASSES:
        key = "coulomb/d0/({},{}|{},{})/b1/f/literal@sm_86/nvcc 13.0".format(
            *cls)
        n = len(vrr.target_set(cls))
        (fits if rows[key]["spill_store"] == 0 else spills).add(n)
    assert max(fits) == 40 and min(spills) == 76
    assert len(vrr.target_set((0, 0, 2, 2))) == 31 in fits  # L = 4, fits


# -- the source each row measured -------------------------------------------------


def _source(cls) -> str:
    return cuda.emit(build_f_kernel(tuple(cls)), SM_86)


@pytest.mark.parametrize("row", [by_l(r, r["L"], id=r["label"])
                                 for r in _rows()])
def test_each_row_measured_the_unit_this_tree_emits(row) -> None:
    sha = hashlib.sha256(_source(row["class"]).encode()).hexdigest()
    assert row["this_tree"]["source_sha256"] == sha


def test_the_source_is_the_one_kernels_json_measured() -> None:
    # #288's sm_86 F rows (repo flags) measured the same text this file
    # compiled under the predecessor's: same unit, two flag sets.
    rows = json.loads(KERNELS.read_text())["rows"]
    f86 = {r["kernel"]: r["source_sha256"] for k, r in rows.items()
           if "/f/" in k and r["arch"] == "sm_86"}
    assert f86
    ours = {r["this_tree"]["kernel"]: r["this_tree"]["source_sha256"]
            for r in _rows()}
    for kernel, sha in f86.items():
        if kernel in ours:
            assert ours[kernel] == sha, kernel


# -- the compile (cuda: a compiler, no card) --------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize(
    "row",
    [pytest.param(r, id=r["label"],
                  marks=[pytest.mark.slow]
                  if tuple(r["predecessor_class"]) in SLOW else [])
     for r in _rows()])
def test_the_row_reproduces_under_the_predecessor_flags(row, nvcc, tmp_path) -> None:
    name = row["this_tree"]["kernel"]
    src = tmp_path / f"{name}.cu"
    src.write_text(_source(row["class"]))
    out = subprocess.run(
        [nvcc.path, *FLAGS.split(), "-Xptxas", "-v", "-c", str(src),
         "-o", str(tmp_path / f"{name}.o")],
        capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr[-4000:]
    got = parse_ptxas(out.stdout + out.stderr, name)
    assert {f: got[f] for f in FIELDS} == {f: row["this_tree"][f] for f in FIELDS}
