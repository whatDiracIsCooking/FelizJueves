"""The exchange-Hessian kernels `hkx_` (#691), compiled.

    devtools/measure.sh hkx-compile [--backend cuda|hip]

OPT-IN: each test skips unless FJ_HKX_STORE names the store (measure.sh
sets it to config.sh's MEASURE_HKX_STORE, the tracked data/measured/hkx.json).

THE COMPILE (no card). Every K class `(la, lc | lb, ld)`, each `l <= 3`
(256), in its DEFAULT form (`hessian.kernel_hkx.hkx_lowstack`: unrolled
below `LOWSTACK_MIN_L`, rolled from it), plus each s, p class in its OTHER
form (the unrolled s, p classes from the floor, the rolled ones below it:
the floor's evidence). Each unit compiled ALONE in its own process with
#606's harness (`measure_df_lowstack._compile`: the measured store's flags,
`os.wait4` for the peak RSS) under a `COMPILE_BUDGET_S` kill (a unit killed
there is recorded `over_budget` with no figures), `FJ_HKX_JOBS` (default 6)
such processes at once on the host: contention only lengthens a wall clock,
never shortens it. Source bytes, compile wall-clock, peak RSS, registers,
spill, stack (nvcc's per-kernel frame; hipcc's ScratchSize). One row per
(arch, toolchain, unit, form). This is the issue's "compile times
recorded" and "frames recorded"; whether each class LAUNCHES on the RTX
3080 is a card's question (test/test_hkx_device.py), not this store's.

THE CLOCK is not measured.

THE RULE is the store's `declared`, written before any row (DECLARED).
Before it, a hand probe read (unrolled) `hkx_11_10` at 16,936 B of `sm_86`
stack and `hkx_11_11` at 33,952 B (70 s); (rolled, the run-time
component loops) `hkx_11_11` 3,728 B, `hkx_21_10` 4,544 B, `hkx_33_33`
19,984 B, each under 4 s of nvcc alone, and `hkx_33_33` 16,864 B of
`gfx1200` scratch (7 s).

A FIRST ROLLED FORM, which unrolled the derivative components in Python
(~160 rolled sums per row, ~420 KB of text a unit), was swept once into
this store's first draft on 2026-10-31 and discarded with it, never
committed: 75-90 s of nvcc a unit alone, 27 units killed at 120 s under
eight concurrent compiles. It is why the components are rolled at run time.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from measure_df_lowstack import STACK_BUDGET_B, _compile

from felizjueves.hessian import kernel_hkx as hk
from felizjueves.performance import measured

STORE = "FJ_HKX_STORE"
SCHEMA = 1
COMPILE_BUDGET_S = 120.0  # per unit: the user's per-.cu budget, declared

LADDER = [(a, c, b, d) for a in range(4) for c in range(4)
          for b in range(4) for d in range(4)]
SP = [am for am in LADDER if max(am) <= hk.UNROLLED_L_MAX]

DECLARED = {
    "issue": 691,
    "question": "does every hkx class through (ff|ff) compile within the "
                "per-.cu budget in its default form, and where does the "
                "unrolled form's frame leave the gradient family's budget",
    "rule": "per unit, over both arches: admissible when its compile "
            "finishes within COMPILE_BUDGET_S = 120 s (the user's per-.cu "
            "budget) on sm_86 and on gfx1200 (a unit killed there is "
            "over_budget); every default unit must be admissible, and where "
            "one is not, the form is changed, never the budget. The stack "
            "is RECORDED, NOT GATED against #613's STACK_BUDGET_B = 14,000 "
            "B (as #694's d2eri3): the bra tables two indices up and R at "
            "L + 2 are ~11 KB alone at (ff|ff), and the frame's gate is the "
            "card's (every default unit launches on the RTX 3080, "
            "test_hkx_device.py). The floor LOWSTACK_MIN_L is the smallest "
            "L with an unrolled s, p class over STACK_BUDGET_B on sm_86",
    "expected": "rolled <= 15 s of nvcc and of hipcc a unit, frames "
                "<= ~21 KB (sm_86) / ~17 KB (gfx1200) at (ff|ff); the "
                "unrolled s, p classes over 14,000 B from L = 3 (sm_86), so "
                "LOWSTACK_MIN_L = 3",
    "clock": "not measured (module docstring)",
    "declared": "2026-10-31, before any row of this store",
}


def units() -> list[tuple[tuple[int, int, int, int], bool]]:
    """`(class, lowstack)` of every unit the sweep compiles: each class's
    default form, then each s, p class's other form."""
    out = [(am, hk.hkx_lowstack(am)) for am in LADDER]
    out += [(am, not hk.hkx_lowstack(am)) for am in SP]
    return out


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the hkx compile records are opt-in: set {STORE} "
                    "(devtools/measure.sh hkx-compile)")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 691,
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


KEY = ("arch", "toolchain", "unit", "form")


def _save(path: Path, new: list[dict]) -> None:
    store = _load(path)
    keys = {tuple(r[k] for k in KEY) for r in new}
    rows = [r for r in store["compile"]
            if tuple(r[k] for k in KEY) not in keys] + new
    rows.sort(key=lambda r: tuple(str(r[k]) for k in KEY))
    store["compile"] = rows
    _write(path, store)


def _compile_rows(backend: str, request) -> None:
    path = _store_path()
    request.getfixturevalue("nvcc" if backend == "cuda" else "hipcc")
    b = measured.BACKENDS[backend]
    _declare(path, _load(path))
    head = {"arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
            "host": measured.host(),
            "date": datetime.date.today().isoformat()}
    only = os.environ.get("FJ_HKX_ONLY")  # e.g. "hkx_33_33,hkx_11_11"
    todo = [(am, low) for am, low in units()
            if not only or hk.hkx_name(am) in only.split(",")]
    jobs = int(os.environ.get("FJ_HKX_JOBS", "6"))

    def one(item):
        am, low = item
        kern = hk.build_hkx_kernel(am, lowstack=low)
        with tempfile.TemporaryDirectory() as d:
            r = _compile(backend, kern, Path(d), budget=COMPILE_BUDGET_S)
        row = {**head, "am": list(am), "L": sum(am),
               "form": "rolled" if low else "unrolled",
               "default": low == hk.hkx_lowstack(am), **r}
        print(f"{backend} {r['unit']} {row['form']}: "
              + ("OVER BUDGET" if r["over_budget"] else
                 f"{r['compile_s']} s, r{r['registers']} spill "
                 f"{r['spill_store']} stack {r['stack_bytes']}"),
              flush=True)
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(one, todo))
    _save(path, rows)


# --- the rule, read back from the tracked rows (`measure.sh check`) -------------

TRACKED = Path(__file__).resolve().parent.parent / "data/measured/hkx.json"
ARCHES = ("sm_86", "gfx1200")


def _rows() -> list[dict]:
    store = json.loads(TRACKED.read_text())
    assert store["declared"] == json.loads(json.dumps(DECLARED))
    return store["compile"]


def test_every_hkx_unit_is_admissible_on_both_arches():
    rows = _rows()
    want = {(a, hk.hkx_name(am), "rolled" if low else "unrolled")
            for a in ARCHES for am, low in units()}
    assert {(r["arch"], r["unit"], r["form"]) for r in rows} == want
    for r in rows:
        if r["default"]:
            assert not r["over_budget"], (r["unit"], r["arch"])
            assert r["compile_s"] <= COMPILE_BUDGET_S, (r["unit"], r["arch"])


def test_the_floor_is_what_the_unrolled_rows_rule():
    # LOWSTACK_MIN_L: the smallest L with an unrolled s, p class over the
    # gradient family's stack budget on sm_86
    over = sorted(r["L"] for r in _rows()
                  if r["arch"] == "sm_86" and r["form"] == "unrolled"
                  and (r["over_budget"] or r["stack_bytes"] > STACK_BUDGET_B))
    assert over and over[0] == hk.LOWSTACK_MIN_L
    assert all(r["stack_bytes"] <= STACK_BUDGET_B for r in _rows()
               if r["arch"] == "sm_86" and r["form"] == "unrolled"
               and r["L"] < hk.LOWSTACK_MIN_L)


@pytest.mark.cuda
def test_cuda_hkx_compile(request) -> None:
    _compile_rows("cuda", request)


@pytest.mark.hip
def test_hip_hkx_compile(request) -> None:
    _compile_rows("hip", request)
