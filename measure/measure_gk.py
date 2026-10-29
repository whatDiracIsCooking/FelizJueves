"""The exchange-energy gradient kernel `gk_` (#618): its rolled form and
the unrolled one, compiled.

    devtools/measure.sh gk-compile [--backend cuda|hip]

OPT-IN: each test skips unless FJ_GK_STORE names the store (measure.sh
sets it to config.sh's MEASURE_GK_STORE, the tracked data/measured/gk.json).

THE COMPILE (no card). Every K class `(la, lc | lb, ld)`, each `l <= 3`
(256), in its ROLLED form (`kernel_gk`'s THE ROLLED FORM, forced), and the
UNROLLED form of the classes with every `l <= 2` LEVEL BY LEVEL in `L =
la + lc + lb + ld`, from 0 up to and including the first level that holds
an inadmissible class, and no further (#613's order). The unrolled form of
a class with an f shell is NOT ATTEMPTED (the user's decision: no new
unrolled f text is built, emitted or compiled). Each unit compiled ALONE,
one at a time, with #606's harness (`measure_df_lowstack._compile`: the
measured store's flags, `os.wait4` for the peak RSS) under a
`COMPILE_BUDGET_S = 10` s kill, the user's per-compile budget for #618: a
unit killed there is recorded `over_budget` with no figures. Source bytes,
compile wall-clock, peak RSS, registers, spill, stack (nvcc's per-kernel
frame; hipcc's ScratchSize). One row per (arch, toolchain, unit, form).

THE CLOCK is not measured, and THE BLOCK MAPPING is not built (the
issue's second form, `kernel_kblock`'s one warp per `K_AB` block): by the
user's decision only a rolled / low-stack block form was in scope, and
only if cheap; #571 and #583 already measured the energy K's rolled block
kernels slower than `kb` or neutral, so it is recorded as not built
(docs/implementation/gradient.md).

THE RULE is the store's `declared`, written before any row (DECLARED).
Before it, a hand probe on `sm_86` read the unrolled `gk_22_10` at 14,880
B of stack (7.3 s) and `gk_22_00` at 11,696 B, and the rolled `gk_33_33`
at 11,616 B in 2.8 s.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pytest
from measure_df_lowstack import STACK_BUDGET_B, _compile
from test_kernel_device import _toolchain

from felizjueves.gradient import kernel_gk as kk
from felizjueves.performance import measured

STORE = "FJ_GK_STORE"
SCHEMA = 1
COMPILE_BUDGET_S = 10.0  # the user's per-compile budget for #618

LADDER = [(a, c, b, d) for a in range(4) for c in range(4)
          for b in range(4) for d in range(4)]
SPD = [c for c in LADDER if max(c) <= kk.UNROLLED_L_MAX]

DECLARED = {
    "issue": 618,
    "question": "which gk classes the rolled form's gate must cover, "
                "through (ff|ff)",
    "rule": "per class, over both arches: a class's UNROLLED form is "
            "admissible when its stack is at most STACK_BUDGET_B = 14,000 B "
            "(#613's gradient-family budget) on sm_86 and on gfx1200 and "
            "its compile finishes within COMPILE_BUDGET_S = 10 s on both "
            "(the user's per-compile budget; a unit killed there is "
            "over_budget). A class with an f shell has no admissible "
            "unrolled form: it is not attempted (the user's decision). The "
            "rolled form is gated by LOWSTACK_MIN_L = the smallest L = la + "
            "lc + lb + ld holding an inadmissible class with every l <= 2, "
            "and by any f shell: every class below the floor with every "
            "l <= 2 must be admissible (so stays unrolled, its text "
            "unchanged), every class at or above it, and every f class, is "
            "gated. The unrolled form is compiled level by level in L up to "
            "and including the floor's level, no further. Every rolled unit "
            "must itself be within both budgets on both arches; where one "
            "is not, the form is changed (more loops rolled, tables moved, "
            "the kernel split), never a budget",
    "expected": "the unrolled form spills from (pp|ss) and is over the "
                "stack budget from L = 5 ((dd|ps), 14,880 B on sm_86 by "
                "hand), so the floor is 5; the rolled form is under 12 KB "
                "at (ff|ff) on sm_86 and compiles in a few seconds",
    "clock": "not measured; the block mapping is not built (module "
             "docstring)",
    "declared": "2026-10-29, before any row of this store",
}


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the gk compile records are opt-in: set {STORE} "
                    "(devtools/measure.sh gk-compile)")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 618,
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
    only = os.environ.get("FJ_GK_ONLY")  # e.g. "gk_33_33,gk_22_10"

    def one(cls, low: bool) -> dict | None:
        kern = kk.build_gk_kernel(cls, lowstack=low)
        if only and kern.name not in only.split(","):
            return None
        r = _compile(backend, kern, tmp_path, budget=COMPILE_BUDGET_S)
        row = {**head, "am": list(cls), "L": sum(cls),
               "form": "rolled" if low else "unrolled",
               "default": low == kk.gk_lowstack(cls), **r}
        _save(path, row)
        print(f"{backend} {r['unit']} {row['form']}: "
              + ("OVER BUDGET" if r["over_budget"] else
                 f"{r['compile_s']} s, r{r['registers']} spill "
                 f"{r['spill_store']} stack {r['stack_bytes']}"),
              flush=True)
        return row

    for L in range(max(sum(c) for c in SPD) + 1):  # unrolled, level by level
        for c in SPD:
            if sum(c) == L:
                one(c, False)
        # the floor is over both arches: stop at the first level with an
        # inadmissible row on ANY arch the store holds (this one included)
        if any(r["form"] == "unrolled" and r["L"] == L and not _ok(r)
               for r in _load(path)["compile"]):
            break
    for cls in LADDER:
        one(cls, True)


# --- the rule, read back from the tracked rows (`measure.sh check`) -------------

TRACKED = Path(__file__).resolve().parent.parent / "data/measured/gk.json"
ARCHES = ("sm_86", "gfx1200")


def _rows(store: dict, unit: str, form: str) -> dict:
    return {r["arch"]: r for r in store["compile"]
            if r["unit"] == unit and r["form"] == form}


def admissible(store: dict, cls) -> bool | None:
    """DECLARED's rule on the unrolled form of `cls`, over both arches:
    False when a row on either arch is inadmissible, True when both arches
    have an admissible row, None otherwise (no level-by-level row reached
    it on some arch)."""
    rows = _rows(store, kk.gk_name(cls), "unrolled")
    if any(not _ok(r) for r in rows.values()):
        return False
    return True if all(a in rows for a in ARCHES) else None


def test_the_gate_is_what_the_measured_rows_rule():
    store = json.loads(TRACKED.read_text())
    assert store["declared"] == json.loads(json.dumps(DECLARED))
    seen = {c: admissible(store, c) for c in SPD}
    bad = [c for c, ok in seen.items() if ok is False]
    floor = min(sum(c) for c in bad)
    assert kk.LOWSTACK_MIN_L == floor
    # every level through the floor's was compiled on both arches
    for c, ok in seen.items():
        if sum(c) <= floor:
            assert ok is not None, c
    for c in LADDER:
        want = sum(c) >= floor or max(c) > kk.UNROLLED_L_MAX
        assert kk.gk_lowstack(c) == want, c
        if not want:
            assert seen[c] is True, c
    rolled = [r for r in store["compile"] if r["form"] == "rolled"]
    assert len(rolled) == len(ARCHES) * len(LADDER)
    for r in rolled:
        assert not r["over_budget"], (r["unit"], r["arch"])
        assert r["stack_bytes"] <= STACK_BUDGET_B, (r["unit"], r["arch"])
        assert r["compile_s"] <= COMPILE_BUDGET_S, r["unit"]


@pytest.mark.cuda
def test_cuda_gk_compile(request, tmp_path) -> None:
    _compile_rows("cuda", request, tmp_path)


@pytest.mark.hip
def test_hip_gk_compile(request, tmp_path) -> None:
    _compile_rows("hip", request, tmp_path)
