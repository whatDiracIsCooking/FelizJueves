"""The registry-sampled selection (#653): which compile and `gpu` cases run.

    Candidate(nodeid, pool, keys, cost_s, families)
    Knobs.load()                    -> the FJ_* knobs, env over config.sh
    seed_of(knobs, root)            -> (seed, merge-base, HEAD)
    changed_paths(base, root)       -> what the diff against `base` touches
    touched_families(paths)         -> the families those paths touch
    draw(candidates, ...)           -> Draw: what is kept, and why
    shard(nodeids, cost, i, n)      -> the i-th of n disjoint shards
    Plugin                          -> the pytest plugin conftest registers

THE BANDS. Every collected device case (`registry.device_cases`: `gpu`, or
`cuda` / `hip` without `gpu`) falls in a POOL, its (tier, backend) --
`gpu/cuda`, `compile/hip`, ... -- and in one band of it, by the classes it
covers (its registry keys):

- `min`: every key at L <= MIN_L_MAX (2). Every such case runs; it is never
  sampled.
- `max`: some key at the top L its ladder -- its `(engine, quantity,
  deriv)` -- reaches among the pool's candidates: the highest class of that
  ladder this run could reach (after `-m`, so `not high_l` and `not slow`
  have already cut it).
- `med`: the rest, 2 < L < that top.
- `unsampled`: a case that covers no key -- an `Allow` (a whole molecule, a
  program, the generator's machinery), or one the registry cannot resolve
  (test_registry.py fails on that, not this). It always runs.

`med` and `max` are each drawn without replacement, per pool, until the
band's budget (`FJ_BUDGET_MED`, `FJ_BUDGET_MAX`, seconds of `cost_s`) is
spent: the case that crosses it is the last drawn, so a budget is overshot
by at most one case. `FJ_K_MED` / `FJ_K_MAX` cap the count as well. A case's
cost (`cost_of`) is its keys' registry figure for its tier and backend
(#652, `cost.py`), shared evenly among the pool's cases that cover each
key, and the largest share over its keys; `FJ_DEFAULT_COST_S` when a key
has no figure. A budget is per pool, so it reads as "seconds per compiler,
or per card".

THE SEED is `FJ_SAMPLE_SEED` when set, else a hash of the merge-base with
`FJ_SAMPLE_BASE` (origin/main) and HEAD: a run on one commit replays its
draw exactly, and a new commit draws afresh. Candidates are ordered by
nodeid before any draw, so collection order (or an xdist worker's) cannot
move it.

CHANGE WEIGHTING. The paths the diff against the merge-base touches
(committed, staged, unstaged and untracked) name families: each family
module's PATHS, plus the files its TESTS name, plus EVERY_FAMILY (the IR,
the emitters, the MMD engine, the planner, the runtime header and `spec`),
which touch them all. A touched family gets one `med` and one `max` case
in every pool that has one, drawn first and whatever the budget ("forced");
the budget's remainder is then drawn with the touched families'
cases `FJ_TOUCH_WEIGHT` times as likely (Efraimidis-Spirakis keys,
`u ** (1 / w)`).

SHARDING. `--fj-shard i/n` (1 <= i <= n) keeps the i-th of n disjoint
shards of what the draw kept, every kept item of the run included, split by
cost (longest first onto the lightest shard, ties by nodeid): the union of
the n shards is the unsharded run.

OPT IN. The plugin acts only under `--fj-sample` (or `FJ_SAMPLE=on`) and
`--fj-shard`; `FJ_SAMPLE=all` turns the draw off -- every case kept, a
shard still split -- whatever the flag says. Nothing passes the flag yet:
#656's gate.sh is what moves the PR gate's selection onto it. It reads only
test/'s items (measure/ re-exports conftest.py, and its sweeps are not
sampled), and leaves every non-device item alone but for the shard.

THE REPORT. The seed, the budgets, the touched families and every pool's
drawn cases go to `FJ_SAMPLE_REPORT` as JSON and to the terminal summary.
Under xdist the first worker writes it and the controller prints it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from . import FAMILIES, REGISTRY, Key, device_cases, family_of

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test"

MIN_L_MAX = 2
BANDS = ("min", "med", "max")

# A change under any of these touches every family: the IR, its lowering and
# the emitters every kernel goes through, the MMD engine most families run on,
# the planner, the header every generated kernel includes, and the key every
# subsystem joins on.
EVERY_FAMILY = (
    "src/felizjueves/ir/",
    "src/felizjueves/emit/",
    "src/felizjueves/mmd/",
    "src/felizjueves/plan/",
    "include/felizjueves/",
    "src/felizjueves/spec.py",
)

# The knobs: environment name -> its config.sh default's meaning.
KNOBS = ("FJ_SAMPLE", "FJ_SAMPLE_SEED", "FJ_SAMPLE_BASE", "FJ_BUDGET_MED",
         "FJ_BUDGET_MAX", "FJ_K_MED", "FJ_K_MAX", "FJ_DEFAULT_COST_S",
         "FJ_TOUCH_WEIGHT", "FJ_SAMPLE_REPORT")


# -- the family maps ----------------------------------------------------------


KEY_FAMILY: dict[Key, str] = family_of()


def family_paths() -> dict[str, tuple[str, ...]]:
    """Family -> the repo-relative paths (a trailing `/` is a directory)
    whose change touches it: its PATHS, and every file its TESTS name."""
    out: dict[str, tuple[str, ...]] = {}
    for name, module in FAMILIES.items():
        files = sorted({f.partition("::")[0] for f in module.TESTS})
        spelled = []
        for f in files:  # a backend spelled `X` is either backend
            spelled += ([f.replace("_X", f"_{b}") for b in ("cuda", "hip")]
                        if "_X" in f else [f])
        out[name] = (*module.PATHS, *spelled)
    return out


def _under(path: str, prefix: str) -> bool:
    return path.startswith(prefix) if prefix.endswith("/") else path == prefix


def touched_families(paths: Iterable[str],
                     table: Mapping[str, Sequence[str]] | None = None
                     ) -> frozenset[str]:
    """The families the changed `paths` touch (module docstring)."""
    table = family_paths() if table is None else table
    out: set[str] = set()
    for path in paths:
        if any(_under(path, p) for p in EVERY_FAMILY):
            return frozenset(table)
        out |= {name for name, prefixes in table.items()
                if any(_under(path, p) for p in prefixes)}
    return frozenset(out)


# -- the knobs, the seed and the diff -----------------------------------------


def _number(value: str, name: str, cast=float) -> Any:
    """`value` as a number; empty is None (no limit)."""
    if value.strip() == "":
        return None
    try:
        return cast(value)
    except ValueError:
        raise pytest.UsageError(f"{name}={value!r} is not a number") from None


@dataclass(frozen=True)
class Knobs:
    mode: str = ""  # "", "on" or "all"
    seed: str = ""
    base: str = "origin/main"
    budget: Mapping[str, float | None] = field(
        default_factory=lambda: {"med": 120.0, "max": 60.0})
    cap: Mapping[str, int | None] = field(
        default_factory=lambda: {"med": None, "max": None})
    default_cost_s: float = 10.0
    touch_weight: float = 4.0
    report: str = ".kernel-tier-reports/sample.json"

    @classmethod
    def from_values(cls, v: Mapping[str, str]) -> Knobs:
        mode = v.get("FJ_SAMPLE", "").strip().lower()
        if mode not in ("", "on", "all"):
            raise pytest.UsageError(
                f"FJ_SAMPLE={mode!r}: expected empty, `on` or `all`")
        d = cls()
        return cls(
            mode=mode, seed=v.get("FJ_SAMPLE_SEED", ""),
            base=v.get("FJ_SAMPLE_BASE", "") or d.base,
            budget={b: _number(v.get(f"FJ_BUDGET_{b.upper()}", ""),
                               f"FJ_BUDGET_{b.upper()}") for b in ("med", "max")},
            cap={b: _number(v.get(f"FJ_K_{b.upper()}", ""), f"FJ_K_{b.upper()}",
                            int) for b in ("med", "max")},
            default_cost_s=_number(v.get("FJ_DEFAULT_COST_S", ""),
                                   "FJ_DEFAULT_COST_S") or d.default_cost_s,
            touch_weight=_number(v.get("FJ_TOUCH_WEIGHT", ""),
                                 "FJ_TOUCH_WEIGHT") or d.touch_weight,
            report=v.get("FJ_SAMPLE_REPORT", "") or d.report)

    @classmethod
    def load(cls) -> Knobs:
        """Every knob as devtools/config.sh spells it -- by sourcing it, so
        the environment still wins over its defaults."""
        script = ". devtools/config.sh && " + " && ".join(
            f'printf "%s\\0" "${{{k}}}"' for k in KNOBS)
        done = subprocess.run(["bash", "-c", script], cwd=ROOT,
                              capture_output=True, text=True, check=True)
        return cls.from_values(dict(zip(KNOBS, done.stdout.split("\0"),
                                        strict=False)))


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def seed_of(knobs: Knobs, root: Path = ROOT) -> tuple[str, str | None, str | None]:
    """(seed, merge-base, HEAD): FJ_SAMPLE_SEED, else a hash of the two."""
    head = _git(root, "rev-parse", "HEAD")
    base = _git(root, "merge-base", "HEAD", knobs.base) if head else None
    if knobs.seed:
        return knobs.seed, base, head
    spelled = f"{base or head or 'no-git'}..{head or 'no-git'}"
    return hashlib.sha256(spelled.encode()).hexdigest()[:16], base, head


def changed_paths(base: str | None, root: Path = ROOT) -> list[str]:
    """Every path the working tree differs in from `base`, untracked files
    included; nothing without a merge-base."""
    if base is None:
        return []
    diff = _git(root, "diff", "--name-only", base) or ""
    new = _git(root, "ls-files", "--others", "--exclude-standard") or ""
    return sorted({p for p in (*diff.splitlines(), *new.splitlines()) if p})


# -- the draw -----------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One collected device case, as the draw sees it."""

    nodeid: str
    pool: str  # "<tier>/<backend>"
    keys: tuple[Key, ...]
    cost_s: float
    families: frozenset[str]


def cost_of(keys: Sequence[Key], tier: str, backend: str,
            covering: Mapping[Key, int], default: float) -> float:
    """One case's seconds: for each key, its registry figure (#652) for this
    tier and backend, shared among the `covering[key]` cases of the pool
    that cover it -- the figure is the sum over every test covering the key
    (cost.py), so an even share is this case's part -- and the largest share
    over its keys, as the keys of one case are one compile and one launch.
    `default` when a key has no figure."""
    shares = []
    for k in keys:
        case = REGISTRY.get(k)
        if case is None or case.cost_s is None:
            return default
        shares.append(case.cost_s.of(tier, backend).seconds
                      / max(covering.get(k, 1), 1))
    return max(shares) if shares else default


def _ladder(key: Key) -> tuple[str, str, int]:
    return key.engine, key.quantity, key.deriv


def bands_of(candidates: Sequence[Candidate]) -> dict[str, str]:
    """nodeid -> its band, `min` / `med` / `max` / `unsampled`, within one
    pool (module docstring)."""
    top: dict[tuple[str, str, int], int] = {}
    for c in candidates:
        for k in c.keys:
            top[_ladder(k)] = max(top.get(_ladder(k), k.L), k.L)
    out = {}
    for c in candidates:
        if not c.keys:
            out[c.nodeid] = "unsampled"
        elif all(k.L <= MIN_L_MAX for k in c.keys):
            out[c.nodeid] = "min"
        elif any(k.L == top[_ladder(k)] for k in c.keys):
            out[c.nodeid] = "max"
        else:
            out[c.nodeid] = "med"
    return out


def _rng(seed: str, *parts: str) -> random.Random:
    spelled = "\0".join((seed, *parts)).encode()
    return random.Random(int.from_bytes(hashlib.sha256(spelled).digest()[:8]))


@dataclass
class BandDraw:
    candidates: int = 0
    forced: list[str] = field(default_factory=list)
    drawn: list[str] = field(default_factory=list)  # forced first
    spent_s: float = 0.0
    budget_s: float | None = None
    cap: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {"candidates": self.candidates, "budget_s": self.budget_s,
                "cap": self.cap, "spent_s": self.spent_s,
                "forced": self.forced, "drawn": self.drawn}


@dataclass
class PoolDraw:
    min: list[str] = field(default_factory=list)
    unsampled: list[str] = field(default_factory=list)
    med: BandDraw = field(default_factory=BandDraw)
    max: BandDraw = field(default_factory=BandDraw)

    def kept(self) -> list[str]:
        return [*self.min, *self.unsampled, *self.med.drawn, *self.max.drawn]

    def to_json(self) -> dict[str, Any]:
        return {"min": self.min, "unsampled": self.unsampled,
                "med": self.med.to_json(), "max": self.max.to_json()}


@dataclass
class Draw:
    seed: str
    touched: frozenset[str]
    pools: dict[str, PoolDraw]
    sampled: bool

    def kept(self) -> set[str]:
        return {n for p in self.pools.values() for n in p.kept()}


def _draw_band(band: list[Candidate], *, seed: str, pool: str, name: str,
               budget: float | None, cap: int | None, touched: frozenset[str],
               touch_weight: float) -> BandDraw:
    out = BandDraw(candidates=len(band), budget_s=budget, cap=cap)
    rng = _rng(seed, pool, name)
    picked: list[Candidate] = []
    for family in sorted(touched):
        members = [c for c in band if family in c.families and c not in picked]
        if members and not any(family in c.families for c in picked):
            picked.append(rng.choice(members))
    out.forced = [c.nodeid for c in picked]
    limit = math.inf if budget is None else budget
    count = math.inf if cap is None else cap
    rest = [c for c in band if c not in picked]
    weights = [touch_weight if c.families & touched else 1.0 for c in rest]
    keyed = sorted(zip((rng.random() ** (1.0 / w) for w in weights), rest,
                       strict=True), key=lambda kc: -kc[0])
    spent = sum(c.cost_s for c in picked)
    for _, c in keyed:
        if spent >= limit or len(picked) >= count:
            break
        picked.append(c)
        spent += c.cost_s
    out.drawn = [c.nodeid for c in picked]
    out.spent_s = spent
    return out


def draw(candidates: Iterable[Candidate], *, seed: str,
         budget: Mapping[str, float | None],
         cap: Mapping[str, int | None] | None = None,
         touched: frozenset[str] = frozenset(), touch_weight: float = 4.0,
         sampled: bool = True) -> Draw:
    """The kept cases of every pool (module docstring). `sampled=False` keeps
    every case, as FJ_SAMPLE=all does."""
    cap = cap or {}
    by_pool: dict[str, list[Candidate]] = defaultdict(list)
    for c in sorted(candidates, key=lambda c: c.nodeid):
        by_pool[c.pool].append(c)
    pools = {}
    for pool, members in sorted(by_pool.items()):
        bands = bands_of(members)
        out = PoolDraw()
        out.min = [c.nodeid for c in members if bands[c.nodeid] == "min"]
        out.unsampled = [c.nodeid for c in members
                         if bands[c.nodeid] == "unsampled"]
        for name in ("med", "max"):
            band = [c for c in members if bands[c.nodeid] == name]
            if sampled:
                got = _draw_band(band, seed=seed, pool=pool, name=name,
                                 budget=budget.get(name), cap=cap.get(name),
                                 touched=touched, touch_weight=touch_weight)
            else:
                got = BandDraw(candidates=len(band),
                               drawn=[c.nodeid for c in band],
                               spent_s=sum(c.cost_s for c in band))
            setattr(out, name, got)
        pools[pool] = out
    return Draw(seed, touched, pools, sampled)


def shard(nodeids: Iterable[str], cost: Mapping[str, float], index: int,
          count: int) -> list[str]:
    """The `index`-th (1-based) of `count` disjoint shards of `nodeids`:
    longest first onto the lightest shard, ties broken by nodeid and shard
    number, so every shard of one set agrees on the split."""
    if not 1 <= index <= count:
        raise ValueError(f"shard {index}/{count}: expected 1 <= i <= n")
    load = [0.0] * count
    mine = []
    for n in sorted(set(nodeids), key=lambda n: (-cost.get(n, 0.0), n)):
        lightest = min(range(count), key=lambda s: (load[s], s))
        load[lightest] += cost.get(n, 0.0)
        if lightest == index - 1:
            mine.append(n)
    return mine


def parse_shard(spelled: str) -> tuple[int, int]:
    i, sep, n = spelled.partition("/")
    try:
        index, count = int(i), int(n)
    except ValueError:
        index = count = 0
    if not sep or not 1 <= index <= count:
        raise pytest.UsageError(
            f"--fj-shard {spelled!r}: expected i/n with 1 <= i <= n")
    return index, count


# -- the pytest plugin --------------------------------------------------------


def addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("fj-sample", "the registry-sampled selection")
    group.addoption("--fj-sample", action="store_true", default=False,
                    help="draw the compile and gpu cases from the registry "
                         "(test/registry/sample.py; FJ_SAMPLE=all overrides)")
    group.addoption("--fj-shard", default=None, metavar="I/N",
                    help="keep the I-th of N disjoint shards of the kept set")


def candidates_of(items: Iterable[pytest.Item],
                  default_cost: float) -> list[Candidate]:
    cases = list(device_cases(items))
    covering: dict[tuple[str, str], dict[Key, int]] = defaultdict(
        lambda: defaultdict(int))
    for c in cases:
        for k in c.keys:
            covering[c.tier, c.backend][k] += 1
    return [Candidate(
        c.nodeid, f"{c.tier}/{c.backend}", c.keys,
        cost_of(c.keys, c.tier, c.backend, covering[c.tier, c.backend],
                default_cost),
        frozenset(KEY_FAMILY[k] for k in c.keys if k in KEY_FAMILY))
        for c in cases]


class Plugin:
    """Registered by conftest.py's pytest_configure, in every process."""

    TOKEN = "fj_sample_token"

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.wanted = bool(config.getoption("fj_sample", False))
        spelled = config.getoption("fj_shard", None)
        self.shard = parse_shard(spelled) if spelled else None
        self.knobs: Knobs | None = None
        self.report: dict[str, Any] | None = None
        worker = getattr(config, "workerinput", None)
        self.worker = worker.get("workerid") if worker else None
        self.token = worker.get(self.TOKEN) if worker else uuid.uuid4().hex

    def _knobs(self) -> Knobs:
        if self.knobs is None:
            self.knobs = Knobs.load()
        return self.knobs

    def active(self) -> bool:
        return bool(self.wanted or self.shard
                    or (os.environ.get("FJ_SAMPLE", "").strip().lower() == "on"))

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node: Any) -> None:  # xdist controller
        node.workerinput[self.TOKEN] = self.token

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config,
                                      items: list[pytest.Item]) -> None:
        """After `-m` and `-k`: deselect what the draw (and the shard) drop."""
        if not self.active():
            return
        knobs = self._knobs()
        sampled = (self.wanted or knobs.mode == "on") and knobs.mode != "all"
        ours = [i for i in items if i.path.is_relative_to(TEST_DIR)]
        cands = candidates_of(ours, knobs.default_cost_s)
        seed, base, head = seed_of(knobs)
        paths = changed_paths(base)
        touched = touched_families(paths)
        got = draw(cands, seed=seed, budget=knobs.budget, cap=knobs.cap,
                   touched=touched, touch_weight=knobs.touch_weight,
                   sampled=sampled)
        device = {c.nodeid for c in cands}
        kept_device = got.kept()
        keep = [i for i in items
                if i.nodeid not in device or i.nodeid in kept_device]
        if self.shard:
            cost = {c.nodeid: c.cost_s for c in cands}
            mine = set(shard((i.nodeid for i in keep),
                             {i.nodeid: cost.get(i.nodeid, knobs.default_cost_s)
                              for i in keep}, *self.shard))
            keep = [i for i in keep if i.nodeid in mine]
        kept = {i.nodeid for i in keep}
        dropped = [i for i in items if i.nodeid not in kept]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
            items[:] = keep
        self.report = {
            "token": self.token, "seed": seed, "merge_base": base,
            "head": head, "base_ref": knobs.base, "sampled": sampled,
            "budget_s": dict(knobs.budget), "cap": dict(knobs.cap),
            "default_cost_s": knobs.default_cost_s,
            "touch_weight": knobs.touch_weight,
            "changed_paths": len(paths), "touched": sorted(touched),
            "shard": "/".join(map(str, self.shard)) if self.shard else None,
            "pools": {p: d.to_json() for p, d in got.pools.items()},
            "deselected": len(dropped), "kept": sorted(kept),
            "report": knobs.report,
        }
        if self.worker in (None, "gw0"):
            path = ROOT / knobs.report
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.report, indent=1))
            tmp.replace(path)

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if self.worker is not None or not self.active():
            return
        report = self.report
        if report is None:  # the controller: the first worker wrote it
            try:
                report = json.loads((ROOT / self._knobs().report).read_text())
            except (OSError, ValueError):
                return
            if report.get("token") != self.token:
                return
        for line in summary_lines(report):
            terminalreporter.write_line(line)


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    tw = ["", "fj sample: " + ("drawn" if report["sampled"] else
                               "every case (FJ_SAMPLE=all)")
          + f", seed {report['seed']} (merge-base {report['merge_base']}, "
          f"HEAD {report['head']})",
          f"  budgets med {report['budget_s']['med']}s max "
          f"{report['budget_s']['max']}s, caps med {report['cap']['med']} "
          f"max {report['cap']['max']}, default cost "
          f"{report['default_cost_s']}s, touched: "
          + (", ".join(report["touched"]) or "none")
          + (f", shard {report['shard']}" if report["shard"] else "")]
    for pool, d in report["pools"].items():
        tw.append(f"  {pool}: min {len(d['min'])} (all), unsampled "
                  f"{len(d['unsampled'])}, med {len(d['med']['drawn'])}/"
                  f"{d['med']['candidates']} ({d['med']['spent_s']:g}s), max "
                  f"{len(d['max']['drawn'])}/{d['max']['candidates']} "
                  f"({d['max']['spent_s']:g}s)")
        if report["sampled"]:
            for band in ("med", "max"):
                forced = set(d[band]["forced"])
                for n in d[band]["drawn"]:
                    tw.append(f"    {band}{' (forced)' if n in forced else ''}"
                              f": {n}")
    tw.append(f"  {report['deselected']} deselected; report: "
              f"{report['report']}")
    return tw


__all__ = [
    "BANDS", "EVERY_FAMILY", "KEY_FAMILY", "MIN_L_MAX", "Candidate",
    "Draw", "Knobs", "Plugin", "addoption", "bands_of", "candidates_of",
    "changed_paths", "cost_of", "draw", "family_paths", "parse_shard",
    "seed_of", "shard", "summary_lines", "touched_families",
]
