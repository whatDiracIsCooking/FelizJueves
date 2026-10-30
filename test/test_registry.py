"""The case registry (#648) against the device cases pytest actually collects,
in both directions.

One collection of the whole of test/ -- every marker, no `-m`, so a narrowed
run cannot hide a case -- in a child pytest with `FJ_REGISTRY_DUMP` set:
conftest.py's collection hook resolves every `gpu` and compile-only case
(`registry.device_cases`) and writes them out. Then:

- every parametrized device case resolves: a rule names its function, and
  the rule reads a class from it (or says, as an `Allow`, why it covers none);
- every key a case covers is a registry case, in the tier the case ran in;
- every registry case's `compile` and `gpu` tiers are collected by some case
  of that tier, and every card it records by a `gpu` case on that card's
  backend;
- no rule, and no `ById` id, names a function or a case that is gone;
- every registry case has a cost (#652): a figure for its compile and its
  `gpu` tests on each backend, each marked measured or modelled, and the
  cost store names no case the registry does not.

The allowlist is the `Allow` rules, each with its reason; it is printed here
so a run shows what covers no case.
"""

from __future__ import annotations

import collections

import pytest
import registry
from registry import BACKEND_CARD, REGISTRY, RULES, Allow, ById, CollectedCase, cost

from felizjueves.spec import Target

# Every target's matrix, and the one uncontracted form a `Spec` admits
# (#692: ERI3 at d1, `uncontracted=True`, spelled apart since it is another
# quantity, the derivative integrals rather than their contraction).
QUANTITIES = {t.value for t in Target} | {"eri3.ip1"}


@pytest.fixture(scope="module")
def collected(tmp_path_factory) -> list[CollectedCase]:
    out = tmp_path_factory.mktemp("registry") / "cases.json"
    try:
        rows = registry.collect(out)
    except RuntimeError as exc:
        pytest.fail(f"collection failed:\n{exc}")
    assert rows, "the collection hook wrote no device case"
    return rows


def _parametrized(collected: list[CollectedCase]) -> list[CollectedCase]:
    return [c for c in collected if c.parametrized]


def test_the_registry_is_well_formed() -> None:
    for key, case in REGISTRY.items():
        assert case.key == key
        assert case.quantity in QUANTITIES, case
        assert case.deriv in (0, 1, 2), case
        assert case.L >= 0, case
        assert case.tiers, f"{key}: no tier"


def test_every_registry_case_has_a_cost() -> None:
    missing = [key for key, case in sorted(REGISTRY.items())
               if case.cost_s is None]
    assert not missing, (
        f"{len(missing)} registry cases have no cost -- run "
        "`devtools/measure.sh registry-cost` and commit "
        f"{cost.STORE.relative_to(cost.ROOT)}:\n  "
        + "\n  ".join(map(str, missing[:20])))
    for key, case in REGISTRY.items():
        for tier in cost.COST_TIERS:
            for backend in BACKEND_CARD:
                e = case.cost_s.of(tier, backend)
                assert e.basis in cost.BASES, (key, tier, backend, e)
                assert e.seconds > 0 and e.source, (key, tier, backend, e)
    stale = sorted(set(cost.load()) - set(REGISTRY))
    assert not stale, f"the cost store names cases the registry does not: {stale}"


def test_every_parametrized_device_case_resolves(collected) -> None:
    unruled = sorted({c.function for c in _parametrized(collected)
                      if not c.ruled})
    assert not unruled, (
        "no registry rule for these device test functions -- add one to "
        "the family's TESTS in test/registry/:\n  " + "\n  ".join(unruled))
    errors = [f"{c.nodeid}: {c.error}" for c in collected if c.error]
    assert not errors, "\n".join(errors)
    empty = [c.nodeid for c in _parametrized(collected)
             if not c.keys and c.allow is None]
    assert not empty, empty


def test_every_collected_key_is_a_registry_case_in_its_tier(collected) -> None:
    bad = []
    for c in collected:
        for key in c.keys:
            case = REGISTRY.get(key)
            if case is None:
                bad.append(f"{c.nodeid}: {key} is not in the registry")
            elif c.tier not in case.tiers:
                bad.append(f"{c.nodeid}: {key} has no {c.tier!r} tier "
                           f"({sorted(case.tiers)})")
    assert not bad, "\n".join(bad[:50]) + f"\n({len(bad)} in all)"


def test_every_registry_case_is_collected_in_each_tier_and_card(collected) -> None:
    seen: dict[tuple, set[str]] = collections.defaultdict(set)
    for c in collected:
        for key in c.keys:
            seen[key].add(c.tier)
            if c.tier == "gpu":
                seen[key].add(BACKEND_CARD[c.backend])
    bad = []
    for key, case in sorted(REGISTRY.items()):
        for tier in sorted(case.tiers & {"compile", "gpu"}):
            if tier not in seen[key]:
                bad.append(f"{key}: no collected {tier!r} case")
        for card in sorted(case.cards):
            if card not in seen[key]:
                bad.append(f"{key}: ran on {card}, but no gpu case of it is "
                           "collected on that card's backend")
    assert not bad, "\n".join(bad[:50]) + f"\n({len(bad)} in all)"


def test_no_rule_names_a_function_or_case_that_is_gone(collected) -> None:
    functions = {c.function for c in collected}
    stale = sorted(set(RULES) - functions)
    assert not stale, "rules for functions no longer collected:\n  " + (
        "\n  ".join(stale))
    ids = collections.defaultdict(set)
    for c in collected:
        ids[c.function].add(c.nodeid.partition("[")[2].removesuffix("]"))
    gone = [f"{function}[{case_id}]"
            for function, rule in RULES.items() if isinstance(rule, ById)
            for case_id in rule.by_id if case_id not in ids[function]]
    assert not gone, "ById ids no longer collected:\n  " + "\n  ".join(gone)


def test_the_allowlist_says_why(collected) -> None:
    allowed = collections.Counter(
        (c.function, c.allow) for c in collected if c.allow is not None)
    for (function, reason), n in sorted(allowed.items()):
        assert reason.strip(), function
        print(f"allowlisted: {function} ({n} cases): {reason}")
    rules = [r for r in RULES.values() if isinstance(r, Allow)]
    assert all(r.reason.strip() for r in rules)
