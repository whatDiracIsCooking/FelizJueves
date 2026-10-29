"""The Coulomb-energy gradient kernel `gjab_` (#619): compiled through (ff).

    devtools/measure.sh gjab-compile [--backend cuda|hip]

OPT-IN: each test skips unless FJ_GJAB_STORE names the store (measure.sh
sets it to config.sh's MEASURE_GJAB_STORE, the tracked
data/measured/gjab.json).

THE COMPILE (no card). Every ordered class `(la, lb)`, `la, lb <= 3`, of
`gjab` in its UNROLLED form (16 units) and in its LOW-STACK form (16): each
unit compiled ALONE, one at a time, with #606's harness
(`measure_df_lowstack._compile`: the measured store's flags, `os.wait4` for
the compiler's peak RSS, a `COMPILE_BUDGET_S = 300` s kill): source bytes,
compile wall-clock, peak RSS, registers, spill, stack (nvcc's per-kernel
frame before device link; hipcc's ScratchSize). One row per (arch,
toolchain, unit, form). A separate store, as #614's `g1e.json`.

THE CLOCK is not measured: #619 asks for registers, spills, stack and
compile time.

THE RULE is the store's `declared`, written before any row (DECLARED),
#614's rule on `gv` restated for `gjab`. The tracked store holds the rule
and NO ROW yet: the session that wrote it had neither compiler. While a
class lacks a row on either arch, `kernel_gjab.LOWSTACK_MIN_LSUM` must be
None (no class defaults to the low-stack form); once every unrolled class
has both arches' rows, the floor must be what the rule reads from them
(`test_the_gate_is_what_the_measured_rows_rule`, `measure.sh check`).
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pytest
from measure_df_lowstack import COMPILE_BUDGET_S, STACK_BUDGET_B, _compile
from test_kernel_device import _toolchain

from felizjueves.gradient import kernel_gjab as kj
from felizjueves.performance import measured

STORE = "FJ_GJAB_STORE"
SCHEMA = 1

LADDER = [(a, b) for a in range(4) for b in range(4)]

DECLARED = {
    "issue": 619,
    "question": "whether gjab needs a low-stack form through (ff), as gv "
                "did (#614), and which classes its gate must cover",
    "rule": "per class, over both arches: a class's UNROLLED form is "
            "admissible when its stack is at most STACK_BUDGET_B = 14,000 B "
            "(#613's gradient-family budget) on sm_86 and on gfx1200 and "
            "its compile finishes within COMPILE_BUDGET_S = 300 s on both. "
            "If any class is inadmissible, the low-stack form is gated by "
            "an la + lb floor, LOWSTACK_MIN_LSUM = the smallest la + lb "
            "with an inadmissible class: every class below it must be "
            "admissible (so stays unrolled, its text unchanged), every "
            "class at or above it is gated. Every low-stack unit must "
            "itself be within both budgets on both arches; where one is "
            "not, the form is changed, never the budget. With no "
            "inadmissible class, gjab keeps one form (LOWSTACK_MIN_LSUM "
            "None)",
    "expected": "the unrolled gjab holds the six derivative densities over "
                "hermite_targets(L + 1) and the pooled derivative factors "
                "as straight-line locals (~210 KB of text at (dd)), like "
                "gv's but without H0 or a nucleus loop, so it is expected "
                "over budget somewhere in (df)-(ff); the low-stack form "
                "holds the per-axis tables one index up and no R (Jp is "
                "read from global memory), expected under gv's low-stack "
                "7,712 B at (ff)",
    "clock": "not measured (module docstring)",
    "declared": "2026-10-29, before any row of this store",
}


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the gjab compile records are opt-in: set {STORE} "
                    "(devtools/measure.sh gjab-compile)")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 619,
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

    def one(cls, low: bool) -> None:
        kern = kj.build_gjab_kernel(cls, lowstack=low)
        r = _compile(backend, kern, tmp_path)
        row = {**head, "target": "gjab", "am": list(cls), "L": sum(cls),
               "form": "lowstack" if low else "unrolled", **r}
        _save(path, row)
        print(f"{backend} {r['unit']} {row['form']}: "
              + ("OVER BUDGET" if r["over_budget"] else
                 f"{r['compile_s']} s, r{r['registers']} spill "
                 f"{r['spill_store']} stack {r['stack_bytes']}"),
              flush=True)

    for low in (False, True):
        for cls in LADDER:
            one(cls, low)


# --- the rule, read back from the tracked rows (`measure.sh check`) -------------

TRACKED = (Path(__file__).resolve().parent.parent / "data/measured/gjab.json")
ARCHES = ("sm_86", "gfx1200")


def _rows(store: dict, unit: str, form: str) -> dict[str, dict]:
    return {r["arch"]: r for r in store["compile"]
            if r["unit"] == unit and r["form"] == form}


def complete(store: dict) -> bool:
    """Every unrolled class has a row on both arches."""
    return all(set(ARCHES) <= set(_rows(store, kj.gjab_name(c), "unrolled"))
               for c in LADDER)


def admissible(store: dict, unit: str) -> bool:
    """DECLARED's rule on the unrolled form of `unit`, over both arches."""
    rows = _rows(store, unit, "unrolled")
    return all(a in rows and _ok(rows[a]) for a in ARCHES)


def floor(store: dict) -> int | None:
    """DECLARED's `LOWSTACK_MIN_LSUM`, read from complete rows."""
    bad = [c for c in LADDER if not admissible(store, kj.gjab_name(c))]
    return min(sum(c) for c in bad) if bad else None


def test_the_gate_is_what_the_measured_rows_rule():
    store = json.loads(TRACKED.read_text())
    assert store["declared"] == json.loads(json.dumps(DECLARED))
    if not complete(store):
        assert kj.LOWSTACK_MIN_LSUM is None, (
            "a gjab floor without both arches' rows for every class")
        return
    assert kj.LOWSTACK_MIN_LSUM == floor(store)
    for c in LADDER:
        gated = kj.gjab_lowstack(c)
        assert gated == (kj.LOWSTACK_MIN_LSUM is not None
                         and sum(c) >= kj.LOWSTACK_MIN_LSUM), c
        if not gated:
            assert admissible(store, kj.gjab_name(c)), c
    if kj.LOWSTACK_MIN_LSUM is None:
        return
    for c in LADDER:
        rows = _rows(store, kj.gjab_name(c), "lowstack")
        for a in ARCHES:
            assert a in rows, (c, a)
            r = rows[a]
            assert not r["over_budget"], r["unit"]
            assert r["stack_bytes"] <= STACK_BUDGET_B, (r["unit"], a)
            assert r["compile_s"] <= COMPILE_BUDGET_S, r["unit"]


@pytest.mark.cuda
def test_cuda_gjab_compile(request, tmp_path) -> None:
    _compile_rows("cuda", request, tmp_path)


@pytest.mark.hip
def test_hip_gjab_compile(request, tmp_path) -> None:
    _compile_rows("hip", request, tmp_path)
