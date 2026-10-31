"""The Coulomb-Hessian kernels `hjab_` / `hdq_` (#690) and the `[p|q]`
classes they read, compiled.

    devtools/measure.sh hjab-compile [--backend cuda|hip]

OPT-IN: each test skips unless FJ_HJAB_STORE names the store (measure.sh
sets it to config.sh's MEASURE_HJAB_STORE, the tracked
data/measured/hjab.json).

THE COMPILE (no card). Every ordered pair class `(la, lb)`, each `l <= 3`
(16), of `hjab` and of `hdq`, in its DEFAULT form (`kernel_hjab.
hjab_lowstack`: unrolled below `la + lb = 3`, rolled from it;
`kernel_hdq.hdq_lowstack`: from `lc + ld = 5`), plus each class in its
OTHER form where that is buildable at all -- every `hdq` class, and the
`hjab` classes through `la + lb = 4` (the unrolled (df) / (ff) are 2.4 /
4.8 MB of straight-line text): the floors' evidence. And every d0 `[p|q]`
class the ladder reads (`hjab_pq2_spec` / `hjab_pq1_spec` over every
`la, lb, lc, ld <= 3`, as Hermite classes `(lp, lq)`, one batch: the
raised towers to `L = 14`), each the J chain's `class_kernels`. Each unit
compiled ALONE in its own process with #606's harness
(`measure_df_lowstack._compile`: the measured store's flags, `os.wait4`
for the peak RSS) under a `COMPILE_BUDGET_S` kill (a unit killed there is
recorded `over_budget` with no figures), `FJ_HJAB_JOBS` (default 6) such
processes at once on the host: contention only lengthens a wall clock,
never shortens it. Source bytes, compile wall-clock, peak RSS, registers,
spill, stack (nvcc's per-kernel frame; hipcc's ScratchSize). One row per
(arch, toolchain, unit, form). This is the issue's "compile times
recorded" and "frames recorded"; whether each class LAUNCHES on the RTX
3080 is a card's question (test/test_hess_j_device.py).

THE CLOCK is not measured.

THE RULE is the store's `declared`, written before any row (DECLARED).
Before it, a hand probe (nvcc 13.0, each alone, 2026-10-31) read the
unrolled `hjab_11` at 8,016 B of `sm_86` stack (2.4 s), `hjab_12` 22,056 B
(13 s), `hjab_22` 50,368 B (186 s); the unrolled `hdq_22` 9,568 B, `hdq_23`
20,096 B (27 s), `hdq_33` 49,040 B (114 s); the rolled `hjab_33` 20,528 B
(8.5 s), `hdq_33` 12,992 B (3 s); and `pq_L14_p8_q6` / `pq_L14_p7_q7` at
26 / 28 s (52 / 47 s of hipcc).
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

from felizjueves.eri.j_coulomb.launch import class_kernels, class_spec
from felizjueves.hessian import kernel_hdq as hd
from felizjueves.hessian import kernel_hjab as hj
from felizjueves.mmd.kernel import split
from felizjueves.performance import measured

STORE = "FJ_HJAB_STORE"
SCHEMA = 1
COMPILE_BUDGET_S = 120.0  # per unit: the user's per-.cu budget, declared

CLASSES = [(a, b) for a in range(4) for b in range(4)]
HJAB_OTHER_L_MAX = 4  # the unrolled hjab above it is megabytes of text

DECLARED = {
    "issue": 690,
    "question": "does every hjab / hdq class through (ff), and every [p|q] "
                "class they read (to L = 14), compile within the per-.cu "
                "budget in its default form, and where does each unrolled "
                "form's frame leave the gradient family's budget",
    "rule": "per unit, over both arches: admissible when its compile "
            "finishes within COMPILE_BUDGET_S = 120 s (the user's per-.cu "
            "budget) on sm_86 and on gfx1200 (a unit killed there is "
            "over_budget); every default unit and every [p|q] unit must be "
            "admissible, and where one is not, the form is changed, never "
            "the budget. The stack is RECORDED, NOT GATED against #613's "
            "STACK_BUDGET_B = 14,000 B (as #691's hkx): the bra tables two "
            "indices up are ~9.5 KB alone at (ff), and the frame's gate is "
            "the card's (every default unit launches on the RTX 3080, "
            "test_hess_j_device.py). Each floor (hjab LOWSTACK_MIN_LSUM, "
            "hdq LOWSTACK_MIN_LSUM) is the smallest la + lb with an "
            "unrolled class over STACK_BUDGET_B (or over budget) on sm_86",
    "expected": "rolled hjab / hdq <= 20 s of nvcc and of hipcc a unit, "
                "frames <= ~21 KB (sm_86) at (ff); the unrolled hjab over "
                "14,000 B from la + lb = 3 and the unrolled hdq from "
                "lc + ld = 5, so the floors 3 and 5; every [p|q] unit "
                "<= ~60 s",
    "clock": "not measured (module docstring)",
    "declared": "2026-10-31, before any row of this store",
}


def pq_classes() -> list[tuple[int, int]]:
    """Every Hermite `[p|q]` class `(lp, lq)` the ladder through (ff)
    reads, sorted."""
    out = set()
    for la, lb in CLASSES:
        for lc, ld in CLASSES:
            out.add(split(hj.hjab_pq2_spec(la, lb, lc, ld)))
            out.add(split(hj.hjab_pq1_spec(la, lb, lc, ld)))
    return sorted(out)


def units() -> list[tuple[str, tuple[int, int], bool | None]]:
    """`(family, class, lowstack)` of every unit the sweep compiles: each
    hjab / hdq class's default form, then its other form (module
    docstring), then every `[p|q]` class (`lowstack` None)."""
    out: list[tuple[str, tuple[int, int], bool | None]] = []
    out += [("hjab", c, hj.hjab_lowstack(c)) for c in CLASSES]
    out += [("hdq", c, hd.hdq_lowstack(c)) for c in CLASSES]
    out += [("hjab", c, not hj.hjab_lowstack(c)) for c in CLASSES
            if sum(c) <= HJAB_OTHER_L_MAX]
    out += [("hdq", c, not hd.hdq_lowstack(c)) for c in CLASSES]
    out += [("pq", c, None) for c in pq_classes()]
    return out


def _default(family: str, cls, low) -> bool:
    if family == "pq":
        return True
    rule = hj.hjab_lowstack if family == "hjab" else hd.hdq_lowstack
    return low == rule(cls)


def _kernel(family: str, cls, low):
    if family == "hjab":
        return hj.build_hjab_kernel(cls, lowstack=low)
    if family == "hdq":
        return hd.build_hdq_kernel(cls, lowstack=low)
    (k,) = class_kernels(class_spec(*cls))
    return k


def _form(family: str, low) -> str:
    return "one-batch" if family == "pq" else (
        "rolled" if low else "unrolled")


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the hjab compile records are opt-in: set {STORE} "
                    "(devtools/measure.sh hjab-compile)")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 690,
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
    only = os.environ.get("FJ_HJAB_ONLY")  # e.g. "hjab_33,pq_L14_p8_q6"
    todo = [u for u in units()
            if not only or _kernel(*u).name in only.split(",")]
    jobs = int(os.environ.get("FJ_HJAB_JOBS", "6"))

    def one(item):
        family, cls, low = item
        kern = _kernel(family, cls, low)
        with tempfile.TemporaryDirectory() as d:
            r = _compile(backend, kern, Path(d), budget=COMPILE_BUDGET_S)
        L = sum(cls) + (2 if family == "hjab" else 1 if family == "hdq"
                        else 0)
        row = {**head, "family": family, "am": list(cls), "L": L,
               "form": _form(family, low),
               "default": _default(family, cls, low), **r}
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

TRACKED = Path(__file__).resolve().parent.parent / "data/measured/hjab.json"
ARCHES = ("sm_86", "gfx1200")


def _rows() -> list[dict]:
    store = json.loads(TRACKED.read_text())
    assert store["declared"] == json.loads(json.dumps(DECLARED))
    return store["compile"]


def test_every_hjab_unit_is_admissible_on_both_arches():
    rows = _rows()
    want = {(a, _kernel(*u).name, _form(u[0], u[2]))
            for a in ARCHES for u in units()}
    assert {(r["arch"], r["unit"], r["form"]) for r in rows} == want
    for r in rows:
        if r["default"]:
            assert not r["over_budget"], (r["unit"], r["arch"])
            assert r["compile_s"] <= COMPILE_BUDGET_S, (r["unit"], r["arch"])


@pytest.mark.parametrize("family, floor", [
    ("hjab", hj.LOWSTACK_MIN_LSUM), ("hdq", hd.LOWSTACK_MIN_LSUM)])
def test_the_floor_is_what_the_unrolled_rows_rule(family, floor):
    # the smallest la + lb with an unrolled class over the gradient
    # family's stack budget (or over the compile budget) on sm_86
    unrolled = [r for r in _rows() if r["arch"] == "sm_86"
                and r["family"] == family and r["form"] == "unrolled"]
    over = sorted(sum(r["am"]) for r in unrolled
                  if r["over_budget"] or r["stack_bytes"] > STACK_BUDGET_B)
    assert over and over[0] == floor
    assert all(r["stack_bytes"] <= STACK_BUDGET_B for r in unrolled
               if sum(r["am"]) < floor)


@pytest.mark.cuda
def test_cuda_hjab_compile(request) -> None:
    _compile_rows("cuda", request)


@pytest.mark.hip
def test_hip_hjab_compile(request) -> None:
    _compile_rows("hip", request)
