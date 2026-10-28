"""The one-electron gradient kernels `gs_`, `gt_`, `gv_` (#614): compiled
through (ff).

    devtools/measure.sh g1e-compile [--backend cuda|hip]

OPT-IN: each test skips unless FJ_G1E_STORE names the store (measure.sh
sets it to config.sh's MEASURE_G1E_STORE, the tracked
data/measured/g1e.json).

THE COMPILE (no card). Every ordered class `(la, lb)`, `la, lb <= 3`, of
`gs`, `gt` and `gv` in its UNROLLED form (48 units), and every `gv` class
in its LOW-STACK form (16): each unit compiled ALONE, one at a time, with
#606's harness (`measure_df_lowstack._compile`: the measured store's
flags, `os.wait4` for the compiler's peak RSS, a `COMPILE_BUDGET_S = 300`
s kill): source bytes, compile wall-clock, peak RSS, registers, spill,
stack (nvcc's per-kernel frame before device link; hipcc's ScratchSize).
One row per (arch, toolchain, unit, form). A separate store, not
`measured_store` / `kernels.json`, whose keys and rows are `[p|q]` / K /
F-shaped: #606's and #613's choice.

THE CLOCK is not measured: #614 asks for registers, stack, spill and
compile time.

THE RULE is the store's `declared`, written before any row (DECLARED).
Before it, a hand probe of nine `sm_86` and seven `gfx1200` units (the
issue's question: does `gv` at (ff) need #586's path?) read the unrolled
`gv_33` at 50,016 B of `sm_86` stack and `gv_22` at 9,752 B; the budget is
#613's gradient-family 14,000 B (the d1 siblings'), not #586's d0 8 KB,
under which (dd) -- run on both cards by #607 -- would be gated too.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pytest
from measure_df_lowstack import COMPILE_BUDGET_S, STACK_BUDGET_B, _compile
from test_kernel_device import _toolchain

from felizjueves.gradient import kernel_g1e as kg
from felizjueves.performance import measured
from felizjueves.spec import Target

STORE = "FJ_G1E_STORE"
SCHEMA = 1

TARGETS = {"gs": Target.S, "gt": Target.T, "gv": Target.V}
LADDER = [(a, b) for a in range(4) for b in range(4)]

DECLARED = {
    "issue": 614,
    "question": "whether gs, gt or gv needs a low-stack form through (ff), "
                "as d0 V did (#586), and which classes its gate must cover",
    "rule": "per target and class, over both arches: a class's UNROLLED "
            "form is admissible when its stack is at most STACK_BUDGET_B = "
            "14,000 B (#613's gradient-family budget) on sm_86 and on "
            "gfx1200 and its compile finishes within COMPILE_BUDGET_S = "
            "300 s on both. A target with an inadmissible class gets a "
            "low-stack form gated by an la + lb floor, LOWSTACK_MIN_LSUM = "
            "the smallest la + lb with an inadmissible class: every class "
            "below it must be admissible (so stays unrolled, its text "
            "unchanged), every class at or above it is gated. Every "
            "low-stack unit must itself be within both budgets on both "
            "arches; where one is not, the form is changed, never the "
            "budget. A target with no inadmissible class keeps one form",
    "expected": "gs and gt hold no Hermite density and fit everywhere; gv "
                "holds H0 and six derivative densities across its nucleus "
                "loop, so it is over budget from (df) (21 KB on sm_86 by "
                "hand) and at (ff) (50 KB), floor 5; the low-stack gv is "
                "the per-axis tables one index up and R, under 8 KB at (ff)",
    "clock": "not measured (module docstring)",
    "declared": "2026-10-28, before any row of this store",
}


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the g1e compile records are opt-in: set {STORE} "
                    "(devtools/measure.sh g1e-compile)")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 614,
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


def name(target: str, cls) -> str:
    return kg.g1e_name((TARGETS[target], *cls))


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

    def one(target: str, cls, low: bool) -> None:
        kern = kg.build_g1e_kernel((TARGETS[target], *cls), lowstack=low)
        r = _compile(backend, kern, tmp_path)
        row = {**head, "target": target, "am": list(cls), "L": sum(cls),
               "form": "lowstack" if low else "unrolled", **r}
        _save(path, row)
        print(f"{backend} {r['unit']} {row['form']}: "
              + ("OVER BUDGET" if r["over_budget"] else
                 f"{r['compile_s']} s, r{r['registers']} spill "
                 f"{r['spill_store']} stack {r['stack_bytes']}"),
              flush=True)

    for target in TARGETS:
        for cls in LADDER:
            one(target, cls, False)
    for cls in LADDER:
        one("gv", cls, True)


# --- the rule, read back from the tracked rows (`measure.sh check`) -------------

TRACKED = (Path(__file__).resolve().parent.parent / "data/measured/g1e.json")
ARCHES = ("sm_86", "gfx1200")


def admissible(store: dict, unit: str) -> bool:
    """DECLARED's rule on the unrolled form of `unit`, over both arches."""
    rows = {r["arch"]: r for r in store["compile"]
            if r["unit"] == unit and r["form"] == "unrolled"}
    return all(a in rows and _ok(rows[a]) for a in ARCHES)


def test_the_gate_is_what_the_measured_rows_rule():
    store = json.loads(TRACKED.read_text())
    assert store["declared"] == json.loads(json.dumps(DECLARED))
    for target, t in TARGETS.items():
        bad = [c for c in LADDER if not admissible(store, name(target, c))]
        if target != "gv":
            assert not bad, (target, bad)
            assert not any(kg.g1e_lowstack((t, *c)) for c in LADDER)
            continue
        assert kg.LOWSTACK_MIN_LSUM == min(sum(c) for c in bad)
        for c in LADDER:
            assert kg.g1e_lowstack((t, *c)) == (
                sum(c) >= kg.LOWSTACK_MIN_LSUM), c
    low = [r for r in store["compile"] if r["form"] == "lowstack"]
    assert len(low) == 2 * len(LADDER)
    for r in low:
        assert not r["over_budget"], r["unit"]
        assert r["stack_bytes"] <= STACK_BUDGET_B, (r["unit"], r["arch"])
        assert r["compile_s"] <= COMPILE_BUDGET_S, r["unit"]


@pytest.mark.cuda
def test_cuda_g1e_compile(request, tmp_path) -> None:
    _compile_rows("cuda", request, tmp_path)


@pytest.mark.hip
def test_hip_g1e_compile(request, tmp_path) -> None:
    _compile_rows("hip", request, tmp_path)
