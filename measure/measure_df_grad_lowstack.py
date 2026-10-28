"""The DF gradient kernels' two forms (#613): compiled across the ladder.

    devtools/measure.sh df-grad-lowstack-compile [--backend cuda|hip]

OPT-IN: each test skips unless FJ_DF_GRAD_LOWSTACK_STORE names the store
(measure.sh sets it to config.sh's MEASURE_DF_GRAD_LOWSTACK_STORE, the
tracked data/measured/df_grad_lowstack.json).

THE COMPILE (no card). Every class of the ladder FreeBreakfast emits --
`geri3` `la <= lb <= 3`, `lP <= 6` (70 classes) and `gmetric` `lP <= lQ <=
6` (28) -- in its LOW-STACK form, and the UNROLLED form level by level in
`L` (every class of one `L`, then the next), per target, through the first
level on which some class is inadmissible on this arch (DECLARED); the
unrolled forms above that level are not compiled (multi-MB units, 26 MB at
(ff|i)) and are recorded `not_compiled`. Each unit is compiled ALONE, one
at a time, with #606's harness (`measure_df_lowstack._compile`: the
measured store's flags, `os.wait4` for the compiler's peak RSS, a
`COMPILE_BUDGET_S = 300` s kill): source bytes, compile wall-clock, peak
RSS, registers, spill, stack (nvcc's per-kernel frame before device link;
hipcc's ScratchSize). One row per (arch, toolchain, unit, form).

THE CLOCK is not measured here: #613 asks for registers, stack, spill and
compile time; #606 found the d0 low-stack forms 1.46x to 4.58x slower, and
nothing here says the d1 ones are not.

THE RULE is the store's `declared`, written before any row (DECLARED). An
earlier exploratory sweep, ordered by an estimate of the unrolled form's
held doubles and stopped after three consecutive inadmissible classes, was
discarded before this store was written: the estimate did not track the
frame (the unrolled (df|s) at 621 doubles was 22,520 B of sm_86 stack, the
(ss|g) at 915 was 1,448 B), so it stopped short of the L = 5 level.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pytest
from measure_df_lowstack import COMPILE_BUDGET_S, STACK_BUDGET_B, _compile
from test_kernel_device import _toolchain

from felizjueves.gradient import kernel_geri3 as kg
from felizjueves.gradient import kernel_gmetric as km
from felizjueves.performance import measured

STORE = "FJ_DF_GRAD_LOWSTACK_STORE"
SCHEMA = 1

ERI3_LADDER = [(a, b, p) for a in range(4) for b in range(a, 4)
               for p in range(7)]
METRIC_LADDER = [(p, q) for p in range(7) for q in range(p, 7)]

DECLARED = {
    "issue": 613,
    "question": "which DF gradient classes (geri3 through (ff|i), gmetric "
                "through (i|i)) the low-stack gate must cover, and whether "
                "the low-stack form keeps every class within #606's ~14 KB "
                "of stack and the 300 s compile budget",
    "rule": "per class, over both arches: a class's UNROLLED form is "
            "admissible when its stack is at most STACK_BUDGET_B = 14,000 B "
            "on sm_86 and on gfx1200 and its compile finishes within "
            "COMPILE_BUDGET_S = 300 s on both; one not compiled on an arch "
            "(above the first L level with an inadmissible class there) "
            "is inadmissible. Per target the gate is an L floor, "
            "LOWSTACK_MIN_L = the smallest L with an inadmissible class: "
            "every class below it must be admissible (so it stays "
            "unrolled), every class at or above it is gated. Every "
            "low-stack unit must itself be within both budgets on both "
            "arches; where one is not, the form is changed, never the "
            "budget",
    "expected": "the unrolled d1 forms hold the derivative densities per "
                "axis (~6x d0's), so geri3's floor is far below d0's "
                "L = 10; the low-stack frames are R (n_t(L + 1)), U and the "
                "bra tables one index up, about 14 KB at (ff|i)",
    "clock": "not measured (module docstring)",
    "declared": "2026-10-28, before any row of this store",
}


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the DF gradient low-stack records are opt-in: set "
                    f"{STORE} (devtools/measure.sh df-grad-lowstack-compile)")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 613,
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
    if target == "geri3":
        return kg.build_geri3_kernel(cls, lowstack=lowstack)
    return km.build_gmetric_kernel(cls, lowstack=lowstack)


def name(target: str, cls) -> str:
    return kg.geri3_name(cls) if target == "geri3" else km.gmetric_name(cls)


def ladder(target: str) -> list[tuple[int, ...]]:
    return ERI3_LADDER if target == "geri3" else METRIC_LADDER


def gated(target: str, cls) -> bool:
    return (kg.geri3_lowstack(cls) if target == "geri3"
            else km.gmetric_lowstack(cls))


def floor(target: str) -> int:
    return kg.LOWSTACK_MIN_L if target == "geri3" else km.LOWSTACK_MIN_L


def _ok(r: dict) -> bool:
    return not r["over_budget"] and r["stack_bytes"] <= STACK_BUDGET_B


def _compile_rows(backend: str, request, tmp_path: Path) -> None:
    path = _store_path()
    _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    _declare(path, _load(path))
    head = {"arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
            "host": measured.host(),
            "date": datetime.date.today().isoformat()}
    only = os.environ.get("FJ_DF_GRAD_LOWSTACK_ONLY")  # "low" / "unrolled"

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

    for target in ("gmetric", "geri3"):
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
           / "data/measured/df_grad_lowstack.json")
ARCHES = ("sm_86", "gfx1200")


def admissible(store: dict, unit: str) -> bool:
    """DECLARED's rule on the unrolled form of `unit`, over both arches."""
    rows = {r["arch"]: r for r in store["compile"]
            if r["unit"] == unit and r["form"] == "unrolled"}
    return all(a in rows and not rows[a].get("not_compiled")
               and _ok(rows[a]) for a in ARCHES)


def test_the_gate_is_what_the_measured_rows_rule():
    store = json.loads(TRACKED.read_text())
    assert store["declared"] == json.loads(json.dumps(DECLARED))
    for target in ("geri3", "gmetric"):
        bad = [c for c in ladder(target)
               if not admissible(store, name(target, c))]
        assert floor(target) == min(sum(c) for c in bad), target
        for c in ladder(target):
            assert gated(target, c) == (sum(c) >= floor(target)), c
            if not gated(target, c):
                assert c not in bad, (target, c)
    # every low-stack row of the ladder is inside both budgets, on both arches
    low = [r for r in store["compile"] if r["form"] == "lowstack"]
    assert len(low) == 2 * (len(ERI3_LADDER) + len(METRIC_LADDER))
    for r in low:
        assert not r["over_budget"], r["unit"]
        assert r["stack_bytes"] <= STACK_BUDGET_B, (r["unit"], r["arch"])
        assert r["compile_s"] <= COMPILE_BUDGET_S, r["unit"]


@pytest.mark.cuda
def test_cuda_df_grad_lowstack_compile(request, tmp_path) -> None:
    _compile_rows("cuda", request, tmp_path)


@pytest.mark.hip
def test_hip_df_grad_lowstack_compile(request, tmp_path) -> None:
    _compile_rows("hip", request, tmp_path)
