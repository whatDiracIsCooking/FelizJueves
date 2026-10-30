"""What a registry case costs (#652): seconds per tier, per backend.

    Estimate(seconds, basis, source)   basis "measured" or "modelled"
    Cost(compile, gpu)                 each {backend: Estimate}
    STORE                              -> data/measured/registry_cost.json
    load(path=STORE)                   -> {Key: Cost}
    junit_times(paths)                 -> {(classname, name): [seconds]}
    measured_figures(cases, compile_times, gpu_times)
                                       -> {(Key, tier, backend): (seconds, n)}
    build(families, measured, sources) -> the store, as a dict

WHAT A FIGURE IS. A case's `compile` figure on a backend is the summed
wall-clock of every compile-only test case (`cuda` or `hip`, no `gpu`) of
that backend that covers its key, as a junit report times it (generation
and the device compile together); its `gpu` figure the same over the `gpu`
test cases, which compile and then run. The sum is what drawing the case
costs when every test covering it runs: a case that covers several keys (a
`Span`) is counted in full against each, so the figures are an upper bound
for a sampler, never an apportioned share.

MEASURED. A figure is measured when at least one test case of its tier and
backend covers the key and EVERY such case has a time. The compile times are
test/compile-only-timings.xml (the full compile-only set, cold, no sccache);
the `gpu` times are kernel-tier reports (`<report dir>/pytest.xml`), when
any are given -- none is tracked, so a refresh off this box measures none.

MODELLED. Every other figure, from a fit of `log(seconds) = a + b L` by
least squares over the measured figures of that tier and backend, taken
from the first group that has at least MIN_POINTS of them over at least
two values of L: the case's own ladder (family, engine, quantity, deriv),
then its family, then every family pooled. A `gpu` figure with no `gpu`
fit at all is the case's own `compile` figure on that backend (a `gpu` test
compiles what the compile-only test compiles; the launch on its small
inputs is not modelled, so it is a lower bound). `source` names which, and
says when L is past the fit's highest measured L: an exponential carried
beyond its data, an order of magnitude at best. A case with no tier on a
backend (interpreted only, or never compiled) still gets a modelled
figure: what adding the test would cost.

HELD OUT. Leave-one-out over the measured figures: each predicted by the
chain above with itself removed from every fit, and charged to the group
that predicted it, as the factor `exp |log(predicted / measured)|` -- its
median and 90th percentile per fit and over all.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rules import BACKEND_CARD, Key

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data/measured/registry_cost.json"
COMPILE_TIMINGS = ROOT / "test/compile-only-timings.xml"
SCHEMA = 1
BASES = frozenset({"measured", "modelled"})
COST_TIERS = ("compile", "gpu")
BACKENDS = tuple(BACKEND_CARD)
MIN_POINTS = 3
# A junit time of 0.000 s is below the clock, not free.
FLOOR_S = 1e-3


@dataclass(frozen=True)
class Estimate:
    seconds: float
    basis: str  # "measured" or "modelled"
    source: str  # "<n> cases", "fit:<family>", "fit:pooled" or "compile"

    def __post_init__(self) -> None:
        if self.basis not in BASES:
            raise ValueError(f"basis {self.basis!r} not in {sorted(BASES)}")


@dataclass(frozen=True)
class Cost:
    """A case's figures: `compile` and `gpu`, each by backend."""

    compile: Mapping[str, Estimate]
    gpu: Mapping[str, Estimate]

    def of(self, tier: str, backend: str) -> Estimate:
        return getattr(self, tier)[backend]


def load(path: Path = STORE) -> dict[Key, Cost]:
    if not path.exists():
        return {}
    store = json.loads(path.read_text())
    return {Key(r["engine"], r["quantity"], r["deriv"], r["L"]): Cost(
        *({b: Estimate(r[t][b]["s"], r[t][b]["basis"], r[t][b]["source"])
           for b in BACKENDS} for t in COST_TIERS))
        for r in store["rows"]}


# -- measuring -----------------------------------------------------------------


def junit_key(nodeid: str) -> tuple[str, str]:
    """A pytest nodeid as a junit report names it: (classname, name)."""
    path, *parts = nodeid.split("::")
    module = path.removesuffix(".py").replace("/", ".")
    return ".".join([module, *parts[:-1]]), parts[-1]


def junit_times(paths: Iterable[Path]) -> dict[tuple[str, str], list[float]]:
    """Every passed or failed test case's time in the junit reports; a skip
    timed nothing and is left out."""
    out: dict[tuple[str, str], list[float]] = {}
    for path in paths:
        for tc in ET.parse(path).iter("testcase"):
            if tc.find("skipped") is not None:
                continue
            out.setdefault((tc.get("classname", ""), tc.get("name", "")),
                           []).append(float(tc.get("time", "0")))
    return out


def measured_figures(cases: Iterable[Any],
                     compile_times: Mapping[tuple[str, str], list[float]],
                     gpu_times: Mapping[tuple[str, str], list[float]],
                     ) -> dict[tuple[Key, str, str], tuple[float, int]]:
    """(key, tier, backend) -> (summed seconds, cases) over the collected
    device cases (`registry.CollectedCase`), where every case is timed; a
    case timed in several reports counts at its median."""
    by: dict[tuple[Key, str, str], list[float | None]] = {}
    for c in cases:
        times = (compile_times if c.tier == "compile" else gpu_times).get(
            junit_key(c.nodeid))
        t = statistics.median(times) if times else None
        for key in c.keys:
            by.setdefault((key, c.tier, c.backend), []).append(t)
    return {k: (round(max(sum(ts), FLOOR_S), 3), len(ts)) for k, ts in by.items()
            if all(t is not None for t in ts)}


# -- modelling -----------------------------------------------------------------

Point = tuple[int, float]  # (L, seconds)


@dataclass(frozen=True)
class Fit:
    a: float
    b: float
    n: int

    def __call__(self, L: int) -> float:
        return math.exp(self.a + self.b * L)


def fit(points: list[Point]) -> Fit | None:
    """`log(s) = a + b L` by least squares, or None under MIN_POINTS points
    or a single L."""
    if len(points) < MIN_POINTS or len({L for L, _ in points}) < 2:
        return None
    xs = [L for L, _ in points]
    ys = [math.log(max(s, FLOOR_S)) for _, s in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = (sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
         / sum((x - mx) ** 2 for x in xs))
    return Fit(my - b * mx, b, len(points))


def _factor(predicted: float, measured: float) -> float:
    return math.exp(abs(math.log(max(predicted, FLOOR_S)
                                 / max(measured, FLOOR_S))))


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(q * len(xs)) - 1)]


def _groups(key: Key, families: Mapping[Key, str]) -> tuple[str, ...]:
    """The fits a figure is predicted from, first that can be fitted wins:
    its own ladder, its family, every family."""
    family = families[key]
    return (f"{family}/{key.engine}/{key.quantity}/d{key.deriv}", family,
            "pooled")


def build(families: Mapping[Key, str],
          measured: Mapping[tuple[Key, str, str], tuple[float, int]],
          sources: dict[str, Any]) -> dict[str, Any]:
    """The store: a figure for every key in `families` (key -> its family),
    tier and backend, measured where `measured` has it, else modelled."""
    points: dict[tuple[str, str, str], list[tuple[Key, Point]]] = {}
    for (key, tier, backend), (s, _) in sorted(measured.items()):
        if key not in families:
            continue
        for group in _groups(key, families):
            points.setdefault((group, tier, backend), []).append(
                (key, (key.L, s)))

    fits: dict[tuple[str, str, str], Fit | None] = {
        g: fit([p for _, p in ps]) for g, ps in points.items()}

    top = {g: max(L for _, (L, _) in ps) for g, ps in points.items()}

    def predict(key: Key, tier: str, backend: str,
                fits=fits) -> tuple[float, str, str] | None:
        for group in _groups(key, families):
            f = fits.get((group, tier, backend))
            if f is not None:
                past = top[(group, tier, backend)]
                note = (f", extrapolated past L {past}" if key.L > past
                        else "")
                return f(key.L), f"fit:{group}{note}", group
        return None

    rows = []
    for key in sorted(families):
        row: dict[str, Any] = {**key._asdict(), "family": families[key]}
        for tier in COST_TIERS:
            row[tier] = {}
            for backend in BACKENDS:
                got = measured.get((key, tier, backend))
                if got is not None:
                    est = {"s": got[0], "basis": "measured",
                           "source": f"{got[1]} cases"}
                else:
                    p = predict(key, tier, backend)
                    if p is None and tier == "gpu":
                        c = row["compile"][backend]
                        p = (c["s"], "compile", "")
                    if p is None:
                        raise ValueError(f"{key}: no {tier} figure on "
                                         f"{backend} and nothing to fit")
                    est = {"s": round(max(p[0], FLOOR_S), 3), "basis": "modelled",
                           "source": p[1]}
                row[tier][backend] = est
        rows.append(row)

    # Leave-one-out: each measured figure predicted without itself.
    held: dict[tuple[str, str, str], list[float]] = {}
    for (key, tier, backend), (s, _) in sorted(measured.items()):
        if key not in families:
            continue
        loo = {}
        for group in _groups(key, families):
            g = (group, tier, backend)
            loo[g] = fit([p for k, p in points[g] if k != key])
        p = predict(key, tier, backend, fits=loo)
        if p is None:
            continue
        held.setdefault((p[2], tier, backend), []).append(_factor(p[0], s))
        held.setdefault(("all", tier, backend), []).append(_factor(p[0], s))

    def error(fs: list[float] | None) -> dict[str, Any]:
        if not fs:
            return {"n": 0, "median_factor": None, "p90_factor": None}
        return {"n": len(fs), "median_factor": round(statistics.median(fs), 3),
                "p90_factor": round(_quantile(fs, 0.9), 3)}

    fit_rows = []
    for (group, tier, backend), f in sorted(fits.items()):
        Ls = sorted({p[0] for _, p in points[(group, tier, backend)]})
        fit_rows.append({
            "family": group, "tier": tier, "backend": backend,
            "a": None if f is None else round(f.a, 6),
            "b": None if f is None else round(f.b, 6),
            "n": len(points[(group, tier, backend)]), "L": Ls,
            "held_out": error(held.get((group, tier, backend)))})
    basis_count: dict[str, int] = {}
    for row in rows:
        for tier in COST_TIERS:
            for backend in BACKENDS:
                k = f"{tier}/{backend}/{row[tier][backend]['basis']}"
                basis_count[k] = basis_count.get(k, 0) + 1
    return {
        "schema": SCHEMA, "issue": 652,
        "about": __doc__.split("WHAT A FIGURE IS.")[1].strip(),
        "sources": sources,
        "counts": dict(sorted(basis_count.items())),
        "held_out": {f"{t}/{b}": error(held.get(("all", t, b)))
                     for t in COST_TIERS for b in BACKENDS},
        "fits": fit_rows,
        "rows": rows,
    }


def stored_measured(store: Mapping[str, Any]
                    ) -> dict[tuple[Key, str, str], tuple[float, int]]:
    """The store's measured figures, as `build` takes them."""
    out = {}
    for r in store["rows"]:
        key = Key(r["engine"], r["quantity"], r["deriv"], r["L"])
        for tier in COST_TIERS:
            for backend in BACKENDS:
                e = r[tier][backend]
                if e["basis"] == "measured":
                    out[(key, tier, backend)] = (
                        e["s"], int(e["source"].split()[0]))
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
