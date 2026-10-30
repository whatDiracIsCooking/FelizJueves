"""The DF Hessian kernels' two forms (#694): compiled across the ladder.

    devtools/measure.sh d2-df-lowstack-compile [--backend cuda|hip]

OPT-IN: each test skips unless FJ_D2_DF_LOWSTACK_STORE names the store
(measure.sh sets it to config.sh's MEASURE_D2_DF_LOWSTACK_STORE, the
tracked data/measured/d2_df_lowstack.json).

THE COMPILE (no card: the compiler alone, so a device-less box with nvcc
takes these rows). Every class of the d1 DF ladder's reach, one derivative
order up -- `d2eri3` `la <= lb <= 3`, `lP <= 6` (70 classes) and `d2metric`
`lP <= lQ <= 6` (28) -- in its LOW-STACK form, and the UNROLLED form level by
level in `L` (every class of one `L`, then the next), per target, through
the first level on which some class is inadmissible on this arch
(DECLARED); the unrolled forms above that level are not compiled and are
recorded `not_compiled`. Each unit is compiled ALONE, one at a time, with
#606's harness (`measure_df_lowstack._compile`: the measured store's flags,
`os.wait4` for the compiler's peak RSS, a `COMPILE_BUDGET_S = 300` s kill):
source bytes, compile wall-clock, peak RSS, registers, spill, stack (nvcc's
per-kernel frame before device link; hipcc's ScratchSize). One row per
(arch, toolchain, unit, form). This is the compile time and the frame
#694 asks to record.

THE CLOCK is not measured here (no card took these rows).

THE RULE is the store's `declared`, written before any row (DECLARED): the
d1 sweep's (measure_df_grad_lowstack.py), read over the arches that have
rows, with one change: a low-stack frame is recorded, not held to #606's
14 KB, which the two-up bra tables alone nearly fill at f. The first rows
are `sm_86` only: the box that took them has nvcc and no hipcc, so the
`gfx1200` rows are owed, and the check below reads the rule over `sm_86`
alone until they land.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pytest
from measure_df_lowstack import COMPILE_BUDGET_S, STACK_BUDGET_B, _compile

from felizjueves.hessian import kernel_d2eri3 as ke
from felizjueves.hessian import kernel_d2metric as km
from felizjueves.performance import measured

STORE = "FJ_D2_DF_LOWSTACK_STORE"
SCHEMA = 1

ERI3_LADDER = [(a, b, p) for a in range(4) for b in range(a, 4)
               for p in range(7)]
METRIC_LADDER = [(p, q) for p in range(7) for q in range(p, 7)]

DECLARED = {
    "issue": 694,
    "question": "which DF Hessian classes (d2eri3 through (ff|i), d2metric "
                "through (i|i)) the low-stack gate must cover, and whether "
                "the low-stack form keeps every class within #606's ~14 KB "
                "of stack and the 300 s compile budget",
    "rule": "per class, over every arch with rows: a class's UNROLLED form "
            "is admissible when its stack is at most STACK_BUDGET_B = "
            "14,000 B and its compile finishes within COMPILE_BUDGET_S = "
            "300 s; one not compiled on an arch (above the first L level "
            "with an inadmissible class there) is inadmissible. Per target "
            "the gate is an L floor, LOWSTACK_MIN_L = the smallest L with "
            "an inadmissible class: every class below it must be admissible "
            "(so it stays unrolled), every class at or above it is gated. "
            "Every low-stack unit must compile within COMPILE_BUDGET_S on "
            "every arch with rows; its frame is RECORDED, not gated: the "
            "d2 forms read the bra's per-axis E two indices up, 9.5 KB of "
            "tables alone at (ff|.) before R, U and the operated factors, "
            "so #606's 14 KB is not a bound this form can meet at f, and "
            "an exploratory compile before this rule (d2eri3_336, 24,544 "
            "B on sm_86) said so",
    "expected": "the unrolled d2 forms hold 28 operator triples' densities "
                "(d2eri3) and read the tower two up, so both floors sit at "
                "or below the d1 ones (geri3 5, gmetric 10); the low-stack "
                "d2eri3 frames are R (n_t(L + 2)), U, the bra tables two "
                "indices up and six operated factors per axis, ~20-25 KB "
                "at (ff|.) with spill; the low-stack d2metric ones under "
                "14 KB",
    "arches": "sm_86 first (nvcc, no hipcc on the box); gfx1200 owed",
    "clock": "not measured (module docstring)",
    "declared": "2026-10-30, before any row of this store",
}


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the DF Hessian low-stack records are opt-in: set "
                    f"{STORE} (devtools/measure.sh d2-df-lowstack-compile)")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 694,
            "about": __doc__.split("\n\n")[1], "compile": []}


def _write(path: Path, store: dict) -> None:
    path.write_text(json.dumps(store, indent=1) + "\n")


def _declare(path: Path, store: dict) -> None:
    declared = json.loads(json.dumps(DECLARED))
    if "declared" not in store:
        store["declared"] = declared
        _write(path, store)
    assert store["declared"] == declared, (
        "the store's declared rule is not this file's: its rows would be "
        "read under another rule")


def _save(path: Path, row: dict) -> None:
    key = ("arch", "toolchain", "unit", "form")
    store = _load(path)
    rows = [r for r in store["compile"]
            if tuple(r[k] for k in key) != tuple(row[k] for k in key)]
    rows.append(row)
    rows.sort(key=lambda r: tuple(str(r[k]) for k in key))
    store["compile"] = rows
    _write(path, store)


def _build(target: str, cls, lowstack: bool):
    if target == "d2eri3":
        return ke.build_d2eri3_kernel(cls, lowstack=lowstack)
    return km.build_d2metric_kernel(cls, lowstack=lowstack)


def name(target: str, cls) -> str:
    return (ke.d2eri3_name(cls) if target == "d2eri3"
            else km.d2metric_name(cls))


def ladder(target: str) -> list[tuple[int, ...]]:
    return ERI3_LADDER if target == "d2eri3" else METRIC_LADDER


def gated(target: str, cls) -> bool:
    return (ke.d2eri3_lowstack(cls) if target == "d2eri3"
            else km.d2metric_lowstack(cls))


def floor(target: str) -> int:
    return ke.LOWSTACK_MIN_L if target == "d2eri3" else km.LOWSTACK_MIN_L


def _ok(r: dict) -> bool:
    return not r["over_budget"] and r["stack_bytes"] <= STACK_BUDGET_B


def _compile_rows(backend: str, request, tmp_path: Path) -> None:
    path = _store_path()
    request.getfixturevalue("nvcc" if backend == "cuda" else "hipcc")
    b = measured.BACKENDS[backend]
    _declare(path, _load(path))
    head = {"arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
            "host": measured.host(),
            "date": datetime.date.today().isoformat()}
    only = os.environ.get("FJ_D2_DF_LOWSTACK_ONLY")  # "low" / "unrolled"
    targets = os.environ.get("FJ_D2_DF_LOWSTACK_TARGETS",
                             "d2metric,d2eri3").split(",")

    def one(target: str, cls, low: bool) -> dict:
        kern = _build(target, cls, low)
        r = _compile(backend, kern, tmp_path)
        row = {**head, "target": target, "am": list(cls), "L": sum(cls),
               "form": "lowstack" if low else "unrolled", **r}
        _save(path, row)
        print(f"{backend} {r['unit']} {row['form']}: "
              + ("OVER BUDGET" if r["over_budget"] else
                 f"{r['compile_s']} s, r{r['registers']} spill "
                 f"{r['spill_store']} stack {r['stack_bytes']}"),
              flush=True)
        return row

    for target in targets:
        if only != "unrolled":
            for cls in ladder(target):
                one(target, cls, True)
        if only == "low":
            continue
        stopped = None
        for L in sorted({sum(c) for c in ladder(target)}):
            level = [c for c in ladder(target) if sum(c) == L]
            if stopped is not None:
                for cls in level:
                    _save(path, {**head, "target": target, "am": list(cls),
                                 "L": L, "form": "unrolled",
                                 "unit": name(target, cls),
                                 "not_compiled": True,
                                 "reason": f"above L = {stopped}, the first "
                                           "level with an inadmissible "
                                           "class (DECLARED)"})
                continue
            rows = [one(target, cls, False) for cls in level]
            if not all(_ok(r) for r in rows):
                stopped = L


# --- the rule, read back from the tracked rows (`measure.sh check`) -------------

TRACKED = (Path(__file__).resolve().parent.parent
           / "data/measured/d2_df_lowstack.json")


def arches(store: dict) -> list[str]:
    """The arches with rows (DECLARED's `arches`: gfx1200's are owed)."""
    return sorted({r["arch"] for r in store["compile"]})


def admissible(store: dict, unit: str) -> bool:
    """DECLARED's rule on the unrolled form of `unit`, over every arch with
    rows."""
    rows = {r["arch"]: r for r in store["compile"]
            if r["unit"] == unit and r["form"] == "unrolled"}
    return all(a in rows and not rows[a].get("not_compiled")
               and _ok(rows[a]) for a in arches(store))


def test_the_gate_is_what_the_measured_rows_rule():
    store = json.loads(TRACKED.read_text())
    assert store["declared"] == json.loads(json.dumps(DECLARED))
    assert "sm_86" in arches(store)
    for target in ("d2eri3", "d2metric"):
        bad = [c for c in ladder(target)
               if not admissible(store, name(target, c))]
        assert floor(target) == min(sum(c) for c in bad), target
        for c in ladder(target):
            assert gated(target, c) == (sum(c) >= floor(target)), c
            if not gated(target, c):
                assert c not in bad, (target, c)
    # every low-stack row of the ladder is inside both budgets, per arch
    low = [r for r in store["compile"] if r["form"] == "lowstack"]
    assert len(low) == len(arches(store)) * (len(ERI3_LADDER)
                                             + len(METRIC_LADDER))
    for r in low:  # the frame is recorded, not gated (DECLARED)
        assert not r["over_budget"], r["unit"]
        assert r["compile_s"] <= COMPILE_BUDGET_S, r["unit"]
        assert r["stack_bytes"] is not None, r["unit"]


@pytest.mark.cuda
def test_cuda_d2_df_lowstack_compile(request, tmp_path) -> None:
    _compile_rows("cuda", request, tmp_path)


@pytest.mark.hip
def test_hip_d2_df_lowstack_compile(request, tmp_path) -> None:
    _compile_rows("hip", request, tmp_path)
