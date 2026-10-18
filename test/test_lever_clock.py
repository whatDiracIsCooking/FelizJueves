"""#363's lever-clock verdict rule, on SYNTHETIC rows (no card, no store).

`measured_store.lever_clock_ab` / `lever_clock_verdict` /
`lever_may_become_the_default` are the rule measured_store.py's THE LEVER
CLOCK declared before any timed KERNEL row existed; these cases hold the
code to it on rows built here. The sweep that times the arms, and the check
that the tracked rows read the recorded verdicts, are measurement and live in
measure/measure_lever_clock_device.py (`devtools/measure.sh levers` and
`devtools/measure.sh check`).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from felizjueves import measured_store
from felizjueves.spec import Spec


def literal(spec: Spec) -> Spec:
    """The arm's literal control: the same class with no lever."""
    return replace(spec, levers=())


def _synthetic_store(arm: str, verdicts: list[str]) -> dict:
    """A store of `arm`'s classes on sm_86, each timed so that `lever_ab`
    reads the given per-class verdict (win, loss, neutral, mixed)."""
    from felizjueves.device import SM_86, MeasuredKey
    prov = dict(toolchain="nvcc 13.0", arch="sm_86", host="h", gpu="g",
                date="2026-10-16")
    # (control wide, arm wide, control near, arm near) medians, IQR 10
    shape = {"win": (1000, 500, 1000, 1000), "loss": (500, 1000, 500, 500),
             "neutral": (1000, 1005, 1000, 1000),
             "mixed": (1000, 500, 500, 1000)}
    cells = measured_store.lever_clock_cells(arm)
    assert len(cells) == len(verdicts)
    store: dict = {}

    def st(p, m):
        return {f"{p}median_ns": m, f"{p}q1_ns": m - 5, f"{p}q3_ns": m + 5}

    for spec, v in zip(cells, verdicts, strict=True):
        cw, aw, cn, an = shape[v]
        raw = {**st("wallclock:control:", cw), **st("wallclock:near:", an),
               **st("wallclock:control:near:", cn),
               **dict.fromkeys(measured_store.GPU_UTIL_FIELDS, 1)}
        lit = measured_store.Measured(registers=1, spill_store=0,
                                      spill_load=0, **prov)
        row = measured_store.Measured(registers=1, spill_store=0,
                                      spill_load=0, runtime_ns=aw,
                                      spread_ns=10, repeats=31, raw=raw,
                                      **prov)
        for s, r in ((literal(spec), lit), (spec, row)):
            store = measured_store.merge(
                store, MeasuredKey(s, SM_86, "nvcc 13.0"), r)
    return store


@pytest.mark.parametrize(("verdicts", "want"), [
    (["win", "neutral", "win", "neutral", "neutral"], "faster"),
    (["loss", "neutral", "neutral", "neutral", "loss"], "slower"),
    (["neutral"] * 5, "not shown"),
    (["win", "loss", "neutral", "neutral", "neutral"], "not shown"),
    (["win", "mixed", "win", "win", "win"], "not shown"),
])
def test_the_declared_verdict_rule(verdicts, want) -> None:
    store = _synthetic_store("fused", verdicts)
    got = measured_store.lever_clock_ab(store, "sm_86", "fused")
    assert [ab.verdict for ab in got.values()] == verdicts
    assert measured_store.lever_clock_verdict(store, "sm_86", "fused") == want
    assert not measured_store.lever_may_become_the_default(
        store, "fused", arches=("sm_86",))


def test_an_untimed_cell_is_refused_never_skipped() -> None:
    store = _synthetic_store("fused", ["win"] * 5)
    assert measured_store.lever_may_become_the_default(
        store, "fused", arches=("sm_86",))
    with pytest.raises(KeyError):  # gfx1200: nothing timed there
        measured_store.lever_may_become_the_default(store, "fused")
    store.pop(next(k for k in store if "/fused@" in k))
    with pytest.raises(KeyError):
        measured_store.lever_clock_verdict(store, "sm_86", "fused")
