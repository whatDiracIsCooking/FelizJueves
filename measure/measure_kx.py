"""The exchange derivative matrix kernels `kx1_` / `kxc_` (#689), compiled.

    devtools/measure.sh kx-compile [--backend cuda|hip]

OPT-IN: each test skips unless FJ_KX_STORE names the store (measure.sh
sets it to config.sh's MEASURE_KX_STORE, the tracked data/measured/kx.json).

THE COMPILE (no card). Every K class `(la, lc | lb, ld)`, each `l <= 3`
(256), in both forms (`hessian.kernel_kx`: `kx1`, the derivative on the
matrix index, and `kxc`, on the contracted one), each in the ONE form the
builder has (rolled: `kernel_gk`'s #618 shape with a matrix scatter).
Each unit compiled ALONE, one at a time, with #606's harness
(`measure_df_lowstack._compile`: the measured store's flags, `os.wait4`
for the peak RSS) under a `COMPILE_BUDGET_S` kill: a unit killed there is
recorded `over_budget` with no figures. Source bytes, compile wall-clock,
peak RSS, registers, spill, stack (nvcc's per-kernel frame; hipcc's
ScratchSize). One row per (arch, toolchain, unit). This is the issue's
"local-memory frame recorded"; whether each class LAUNCHES on the RTX
3080 is a card's question, not this store's.

THE CLOCK is not measured.

THE RULE is the store's `declared`, written before any row (DECLARED).
Before it, a hand probe on `sm_86` read `kx1_33_33` at 9,824 B of stack
(13.3 s, cold) and `kxc_33_33` at 9,824 B (4.3 s); on `gfx1200` 10,192 B
and 10,176 B of scratch.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pytest
from measure_df_lowstack import STACK_BUDGET_B, _compile

from felizjueves.hessian import kernel_kx as kx
from felizjueves.performance import measured
from felizjueves.spec import Target

STORE = "FJ_KX_STORE"
SCHEMA = 1
COMPILE_BUDGET_S = 30.0  # per unit, declared before any row

LADDER = [(a, c, b, d) for a in range(4) for c in range(4)
          for b in range(4) for d in range(4)]
FORMS = (Target.KX1, Target.KXC)

DECLARED = {
    "issue": 689,
    "question": "does every kx1 / kxc class through (ff|ff) fit the "
                "gradient family's stack budget in its one (rolled) form",
    "rule": "per unit, over both arches: admissible when its stack is at "
            "most STACK_BUDGET_B = 14,000 B (#613's gradient-family "
            "budget) on sm_86 and on gfx1200 and its compile finishes "
            "within COMPILE_BUDGET_S = 30 s on both (a unit killed there is "
            "over_budget). Every unit must be admissible; where one is not, "
            "the form is changed (more loops rolled, tables moved, the "
            "kernel split), never a budget",
    "expected": "gk's rolled form plus a 3 n_a scatter row and one table "
                "side fewer: under 11 KB at (ff|ff) on sm_86 (9,824 B by "
                "hand), each compile under 15 s",
    "clock": "not measured (module docstring)",
    "declared": "2026-10-30, before any row of this store",
}


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the kx compile records are opt-in: set {STORE} "
                    "(devtools/measure.sh kx-compile)")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 689,
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
    key = ("arch", "toolchain", "unit")
    store = _load(path)
    rows = [r for r in store["compile"]
            if tuple(r[k] for k in key) != tuple(row[k] for k in key)]
    rows.append(row)
    rows.sort(key=lambda r: tuple(str(r[k]) for k in key))
    store["compile"] = rows
    _write(path, store)


def _compile_rows(backend: str, request, tmp_path: Path) -> None:
    path = _store_path()
    # a compiler only: a compile needs no card (gk-compile asked for one)
    request.getfixturevalue("nvcc" if backend == "cuda" else "hipcc")
    b = measured.BACKENDS[backend]
    _declare(path, _load(path))
    head = {"arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
            "host": measured.host(),
            "date": datetime.date.today().isoformat()}
    only = os.environ.get("FJ_KX_ONLY")  # e.g. "kx1_33_33,kxc_33_33"
    for cls in LADDER:
        for form in FORMS:
            kern = kx.build_kx_kernel(cls, form=form)
            if only and kern.name not in only.split(","):
                continue
            r = _compile(backend, kern, tmp_path, budget=COMPILE_BUDGET_S)
            _save(path, {**head, "am": list(cls), "L": sum(cls),
                         "form": form.value, **r})
            print(f"{backend} {r['unit']}: "
                  + ("OVER BUDGET" if r["over_budget"] else
                     f"{r['compile_s']} s, r{r['registers']} spill "
                     f"{r['spill_store']} stack {r['stack_bytes']}"),
                  flush=True)


# --- the rule, read back from the tracked rows (`measure.sh check`) -------------

TRACKED = Path(__file__).resolve().parent.parent / "data/measured/kx.json"
ARCHES = ("sm_86", "gfx1200")


def test_every_kx_unit_is_admissible_on_both_arches():
    store = json.loads(TRACKED.read_text())
    assert store["declared"] == json.loads(json.dumps(DECLARED))
    rows = store["compile"]
    want = {(a, kx.kx_name(c, form=f)) for a in ARCHES for c in LADDER
            for f in FORMS}
    assert {(r["arch"], r["unit"]) for r in rows} == want
    for r in rows:
        assert not r["over_budget"], (r["unit"], r["arch"])
        assert r["stack_bytes"] <= STACK_BUDGET_B, (r["unit"], r["arch"])
        assert r["compile_s"] <= COMPILE_BUDGET_S, (r["unit"], r["arch"])


@pytest.mark.cuda
def test_cuda_kx_compile(request, tmp_path) -> None:
    _compile_rows("cuda", request, tmp_path)


@pytest.mark.hip
def test_hip_kx_compile(request, tmp_path) -> None:
    _compile_rows("hip", request, tmp_path)
