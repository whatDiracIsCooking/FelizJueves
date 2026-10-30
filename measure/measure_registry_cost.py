"""The case registry's cost estimates (#652): refreshed and re-derived.

    devtools/measure.sh registry-cost      (no compiler, no card)

OPT-IN: the refresh skips unless FJ_REGISTRY_COST_STORE names the store
(measure.sh sets it to config.sh's MEASURE_REGISTRY_COST_STORE, the tracked
data/measured/registry_cost.json).

THE REFRESH collects the whole of test/ as test_registry.py does (every
device case, resolved to its registry keys), times each from the junit
reports and writes a figure for every registry case, tier and backend --
measured where every test covering it was timed, else modelled
(test/registry/cost.py says how). Its inputs:

  FJ_COMPILE_TIMINGS     the compile-only junit report (default
                         test/compile-only-timings.xml, the tracked full
                         set; regenerate it with `pytest -m "(cuda or hip)
                         and not gpu" --junitxml=...` on a box with both
                         compilers)
  FJ_KERNEL_TIER_REPORTS the `gpu` reports: a directory searched for
                         `*/pytest.xml` (kernel-tier.sh's --report layout;
                         default config.sh's KERNEL_TIER_REPORTS) or one
                         junit file. None is tracked: a refresh with none
                         models every `gpu` figure.

It prints the leave-one-out error of each fit. Nothing reads the store but
the registry, and nothing gates on it.

THE CHECK (`devtools/measure.sh check`): the stored modelled figures and
fits are what the stored measured ones fit to, and the tracked compile
report is the one the store was refreshed from.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import registry
from registry import cost

STORE = "FJ_REGISTRY_COST_STORE"
ROOT = cost.ROOT


def _reports(spec: str) -> list[Path]:
    p = ROOT / spec
    if p.is_file():
        return [p]
    return sorted(p.glob("*/pytest.xml")) if p.is_dir() else []


def _source(path: Path) -> dict:
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {"path": str(rel), "sha256": cost.sha256(path)}


def _summary(store: dict) -> str:
    lines = ["held-out error (leave-one-out factor exp|log(pred/meas)|):"]
    for k, e in store["held_out"].items():
        lines.append(f"  {k:12s} n={e['n']:4d}  median {e['median_factor']}"
                     f"  p90 {e['p90_factor']}")
    for f in store["fits"]:
        e = f["held_out"]
        lines.append(
            f"  {f['family']:20s} {f['tier']:7s} {f['backend']:4s} "
            f"a={f['a']} b={f['b']} n={f['n']} L={f['L'][0]}..{f['L'][-1]}"
            f"  loo n={e['n']} median {e['median_factor']}"
            f" p90 {e['p90_factor']}")
    lines.append(f"counts: {store['counts']}")
    return "\n".join(lines)


def test_registry_cost_refresh(tmp_path) -> None:
    out = os.environ.get(STORE)
    if not out:
        pytest.skip(f"the registry cost refresh is opt-in: set {STORE} "
                    "(devtools/measure.sh registry-cost)")
    compile_xml = ROOT / os.environ.get(
        "FJ_COMPILE_TIMINGS", str(cost.COMPILE_TIMINGS.relative_to(ROOT)))
    reports = _reports(os.environ.get("FJ_KERNEL_TIER_REPORTS",
                                      ".kernel-tier-reports"))
    cases = registry.collect(tmp_path / "cases.json")
    measured = cost.measured_figures(
        cases, cost.junit_times([compile_xml]), cost.junit_times(reports))
    store = cost.build(registry.family_of(), measured, {
        "compile": [_source(compile_xml)],
        "gpu": [_source(p) for p in reports]})
    Path(ROOT / out).write_text(json.dumps(store, indent=1) + "\n")
    print(_summary(store))


def test_the_registry_cost_store_recomputes() -> None:
    store = json.loads(cost.STORE.read_text())
    assert store["schema"] == cost.SCHEMA
    again = cost.build(registry.family_of(), cost.stored_measured(store),
                       store["sources"])
    assert again["fits"] == store["fits"]
    assert again["held_out"] == store["held_out"]
    assert again["rows"] == store["rows"]
    tracked = cost.COMPILE_TIMINGS.relative_to(ROOT)
    for s in store["sources"]["compile"]:
        if s["path"] == str(tracked):
            assert s["sha256"] == cost.sha256(cost.COMPILE_TIMINGS), (
                f"{tracked} changed since the store was refreshed: run "
                "devtools/measure.sh registry-cost")
