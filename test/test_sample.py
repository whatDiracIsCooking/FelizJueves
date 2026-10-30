"""The registry sampler (#653, test/registry/sample.py): the draw replays, the
`min` band is whole, a budget is overshot by at most one case, a touched
family is always drawn, and the shards partition what was kept.

The draws run over candidates built from the registry itself -- one per
compile and `gpu` case per backend -- so the bands and families are the
real ones; one child pytest checks the plugin end to end on one file."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
from registry import FAMILIES, REGISTRY, sample
from registry.sample import Candidate, Knobs

ROOT = Path(__file__).resolve().parents[1]
BUDGET = {"med": 60.0, "max": 30.0}
SEEDS = [f"seed-{i}" for i in range(12)]


def _candidates(cost=lambda key: 10.0) -> list[Candidate]:
    out = []
    for key, case in REGISTRY.items():
        for tier in sorted(case.tiers & {"compile", "gpu"}):
            for backend in ("cuda", "hip"):
                out.append(Candidate(
                    f"test/t.py::{tier}[{'-'.join(map(str, key))}-{backend}]",
                    f"{tier}/{backend}", (key,), cost(key),
                    frozenset({sample.KEY_FAMILY[key]})))
    # a case that covers no class: an Allow
    out.append(Candidate("test/t.py::molecule[cuda]", "gpu/cuda", (), 10.0,
                         frozenset()))
    return out


CANDS = _candidates()
# A spread of costs, so the budget is crossed at an uneven point.
UNEVEN = _candidates(lambda key: 1.0 + zlib.crc32(repr(key).encode()) % 17)


def _kept(d: sample.Draw) -> dict[str, list[str]]:
    return {p: v.kept() for p, v in d.pools.items()}


def test_the_same_seed_draws_the_same_cases() -> None:
    for seed in SEEDS[:4]:
        one = sample.draw(CANDS, seed=seed, budget=BUDGET)
        shuffled = CANDS[:]
        random.Random(seed).shuffle(shuffled)
        two = sample.draw(shuffled, seed=seed, budget=BUDGET)
        assert _kept(one) == _kept(two)
    draws = {frozenset(sample.draw(CANDS, seed=s, budget=BUDGET).kept())
             for s in SEEDS}
    assert len(draws) > 1, "every seed drew the same cases"


def test_min_and_the_unsampled_are_always_complete() -> None:
    for seed in SEEDS:
        d = sample.draw(CANDS, seed=seed, budget={"med": 0.0, "max": 0.0})
        for pool, p in d.pools.items():
            members = [c for c in CANDS if c.pool == pool]
            bands = sample.bands_of(members)
            assert sorted(p.min) == sorted(
                n for n, b in bands.items() if b == "min")
            assert sorted(p.unsampled) == sorted(
                n for n, b in bands.items() if b == "unsampled")
            assert p.med.drawn == p.max.drawn == []
    assert "test/t.py::molecule[cuda]" in d.kept()


def test_the_bands_follow_each_ladders_top() -> None:
    pool = [c for c in CANDS if c.pool == "gpu/cuda"]
    bands = sample.bands_of(pool)
    top = {}
    for c in pool:
        for k in c.keys:
            top[k[:3]] = max(top.get(k[:3], 0), k.L)
    for c in pool:
        if not c.keys:
            assert bands[c.nodeid] == "unsampled"
            continue
        (k,) = c.keys
        want = ("min" if k.L <= sample.MIN_L_MAX else
                "max" if k.L == top[k[:3]] else "med")
        assert bands[c.nodeid] == want, c


def test_a_budget_is_never_exceeded_by_more_than_one_case() -> None:
    cost = {c.nodeid: c.cost_s for c in UNEVEN}
    for seed in SEEDS:
        for budget in ({"med": 25.0, "max": 7.0}, {"med": 0.5, "max": 100.0}):
            d = sample.draw(UNEVEN, seed=seed, budget=budget)
            for p in d.pools.values():
                for name in ("med", "max"):
                    b = getattr(p, name)
                    spent = sum(cost[n] for n in b.drawn)
                    assert spent == pytest.approx(b.spent_s)
                    if b.drawn:
                        assert spent - cost[b.drawn[-1]] < budget[name]
                    if len(b.drawn) < b.candidates:
                        assert spent >= budget[name]  # it stopped at the budget


def test_a_count_cap_holds() -> None:
    d = sample.draw(CANDS, seed="cap", budget={"med": None, "max": None},
                    cap={"med": 2, "max": 1})
    for p in d.pools.values():
        assert len(p.med.drawn) == min(2, p.med.candidates)
        assert len(p.max.drawn) == min(1, p.max.candidates)


def test_a_touched_family_is_always_drawn() -> None:
    zero = {"med": 0.0, "max": 0.0}
    for family in FAMILIES:
        for seed in SEEDS[:3]:
            d = sample.draw(CANDS, seed=seed, budget=zero,
                            touched=frozenset({family}))
            for pool, p in d.pools.items():
                for name in ("med", "max"):
                    band = getattr(p, name)
                    has = [c for c in CANDS if c.pool == pool
                           and family in c.families
                           and sample.bands_of(
                               [x for x in CANDS if x.pool == pool]
                           )[c.nodeid] == name]
                    drawn = {c.nodeid for c in has} & set(band.drawn)
                    assert bool(drawn) == bool(has), (family, pool, name)
                    assert band.forced == band.drawn  # the budget was zero


def test_a_touched_family_is_likelier_in_the_rest_of_the_draw() -> None:
    family = "gradients"
    pool = "gpu/cuda"

    def share(touched: frozenset[str]) -> float:
        hits = total = 0
        for seed in SEEDS:
            d = sample.draw(CANDS, seed=seed, budget=BUDGET, touched=touched,
                            touch_weight=8.0)
            band = d.pools[pool].med
            rest = [n for n in band.drawn if n not in band.forced]
            fam = {c.nodeid for c in CANDS if family in c.families}
            hits += sum(n in fam for n in rest)
            total += len(rest)
        return hits / total

    assert share(frozenset({family})) > share(frozenset())


def test_every_case_is_kept_when_sampling_is_off() -> None:
    d = sample.draw(CANDS, seed="x", budget={"med": 0.0, "max": 0.0},
                    sampled=False)
    assert d.kept() == {c.nodeid for c in CANDS}


def test_the_shards_partition_the_kept_set() -> None:
    kept = sorted(sample.draw(UNEVEN, seed="s", budget=BUDGET).kept())
    cost = {c.nodeid: c.cost_s for c in UNEVEN}
    for n in (1, 2, 3, 7):
        shards = [sample.shard(kept, cost, i, n) for i in range(1, n + 1)]
        union = [x for s in shards for x in s]
        assert sorted(union) == kept
        assert len(union) == len(set(union))
        assert sample.shard(list(reversed(kept)), cost, 1, n) == shards[0]
        loads = [sum(cost[x] for x in s) for s in shards]
        assert max(loads) - min(loads) <= max(cost.values())
    with pytest.raises(ValueError):
        sample.shard(kept, cost, 0, 2)


@pytest.mark.parametrize("spelled", ["0/2", "3/2", "1", "a/b", "1/0"])
def test_a_bad_shard_is_a_usage_error(spelled: str) -> None:
    with pytest.raises(pytest.UsageError):
        sample.parse_shard(spelled)


def test_the_changed_paths_name_their_families() -> None:
    t = sample.touched_families
    assert t([]) == frozenset()
    assert t(["README.md", "docs/references.md"]) == frozenset()
    assert t(["src/felizjueves/gradient/kernel_gk.py"]) == {"gradients"}
    assert t(["src/felizjueves/hessian/host.py",
              "src/felizjueves/eri/k_exchange/k.py"]) == {"hessians",
                                                           "k-exchange"}
    assert t(["src/felizjueves/hgp/vrr.py"]) >= {"fock-hgp",
                                                  "one-electron-extra"}
    # a family's test files, by either backend's spelling
    tests = [p for p in sample.family_paths()["k-exchange"]
             if p.startswith("test/test_")]
    assert tests and all("k-exchange" in t([p]) for p in tests)
    for every in sample.EVERY_FAMILY:
        assert t([every + "x.py" if every.endswith("/") else every]) == set(
            FAMILIES)


def test_every_family_path_exists() -> None:
    for family, paths in sample.family_paths().items():
        assert paths, family
        missing = [p for p in paths if not (ROOT / p).exists()]
        assert not missing, (family, missing)
    for p in sample.EVERY_FAMILY:
        assert (ROOT / p).exists(), p


def test_the_knobs_parse() -> None:
    k = Knobs.from_values({"FJ_SAMPLE": "ALL", "FJ_BUDGET_MED": "",
                           "FJ_BUDGET_MAX": "7.5", "FJ_K_MED": "3"})
    assert k.mode == "all"
    assert k.budget == {"med": None, "max": 7.5}
    assert k.cap == {"med": 3, "max": None}
    with pytest.raises(pytest.UsageError):
        Knobs.from_values({"FJ_SAMPLE": "maybe"})
    with pytest.raises(pytest.UsageError):
        Knobs.from_values({"FJ_K_MAX": "lots"})


def test_config_sh_spells_every_knob() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("FJ_")}
    done = subprocess.run(
        ["bash", "-c", ". devtools/config.sh && "
         + " && ".join(f'test "${{{n}+set}}" = set' for n in sample.KNOBS)],
        cwd=ROOT, env=env)
    assert done.returncode == 0, "a knob is not set by devtools/config.sh"


def test_the_seed_hashes_the_merge_base_and_head() -> None:
    a = sample.seed_of(Knobs(seed=""))
    assert a == sample.seed_of(Knobs(seed=""))
    assert sample.seed_of(Knobs(seed="fixed"))[0] == "fixed"


def _collect(tmp_path: Path, *args: str, **env: str) -> tuple[list[str], dict]:
    report = tmp_path / f"r{len(list(tmp_path.iterdir()))}.json"
    child = {k: v for k, v in os.environ.items()
             if not k.startswith(("PYTEST_", "FJ_"))}
    child.update(FJ_SAMPLE_SEED="fixed", FJ_SAMPLE_REPORT=str(report),
                 FJ_BUDGET_MED="0.001", FJ_BUDGET_MAX="0.001",
                 **env)
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-p", "no:xdist", "-o", "addopts=",
         "-m", "gpu and cuda", "test/test_one_electron_device.py", *args],
        cwd=ROOT, env=child, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    ids = [line for line in done.stdout.splitlines()
           if line.startswith("test/")]
    return ids, (json.loads(report.read_text()) if report.exists() else {})


def test_the_plugin_deselects_what_it_did_not_draw(tmp_path: Path) -> None:
    every, none = _collect(tmp_path)
    assert not none, "the plugin acted without --fj-sample"
    drawn, report = _collect(tmp_path, "--fj-sample")
    assert set(drawn) < set(every)
    assert sorted(drawn) == report["kept"]
    assert report["seed"] == "fixed" and report["sampled"]
    again, _ = _collect(tmp_path, "--fj-sample")
    assert again == drawn
    off, report = _collect(tmp_path, "--fj-sample", FJ_SAMPLE="all")
    assert off == every and not report["sampled"]
    halves = [_collect(tmp_path, "--fj-sample", f"--fj-shard={i}/2")[0]
              for i in (1, 2)]
    assert not set(halves[0]) & set(halves[1])
    assert sorted(halves[0] + halves[1]) == sorted(drawn)


def test_a_cost_is_the_registry_figure_shared_among_its_coverers() -> None:
    key = next(k for k, c in REGISTRY.items()
               if c.cost_s is not None and "gpu" in c.tiers)
    whole = REGISTRY[key].cost_s.of("gpu", "cuda").seconds
    assert sample.cost_of((key,), "gpu", "cuda", {key: 1}, 99.0) == whole
    assert sample.cost_of((key,), "gpu", "cuda", {key: 4}, 99.0) == (
        pytest.approx(whole / 4))
    gone = key._replace(L=999)
    assert sample.cost_of((key, gone), "gpu", "cuda", {}, 99.0) == 99.0
