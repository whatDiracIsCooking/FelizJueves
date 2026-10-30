"""The case registry (#648): every kernel family's cases that actually exist.

    Case(engine, quantity, deriv, L, cards, tiers, cost_s)
    REGISTRY                        -> every family's cases, by key
    RULES                           -> device test function -> its rule
    FAMILIES                        -> family name -> its module
    resolve(item)                   -> Resolution: the keys a collected case
                                       covers, or the allowlist reason it
                                       covers none
    device_cases(items)             -> [CollectedCase] for every `gpu` and
                                       compile-only item, resolved

WHAT A CASE IS. One `(engine, quantity, deriv, L)` -- the recurrence
(`spec.Engine`), the matrix (`spec.Target`'s value), the derivative order,
and the class's total angular momentum: `la + lb + lc + ld` for a quartet,
`la + lb` for a pair, the triple's sum for `eri3`, and `2 (la + lb)` for
`schwarz`, as `tiers.py` defines it. A derivative's L is its class's, not
raised by the order. A lever (`kmixed`, `sliced-hrr`, ...), a low-stack form
or a batch count is not part of the key: they are variants of one case, and
every test of a variant covers it.

`tiers` says which tiers apply: `interpreted` (the IR run on the host, as
the tree-state file records it -- declared, not checked here), `compile` (a
compile-only test under nvcc / hipcc exists) and `gpu` (a test launches it
on a card). `cards` is the tree-state record of which cards a `gpu` test of
it has RUN on -- `rtx3080` (CUDA) and `rx9060xt` (HIP) -- not which could:
the HGP split `J` / `K` has `gpu` tests on both backends and ran on the RTX
3080 only. `cost_s` is a `cost.Cost`: seconds for its `compile` and its
`gpu` tests on each backend, each measured or modelled, read from
data/measured/registry_cost.json (#652; `cost.py` says how, and
`devtools/measure.sh registry-cost` refreshes it).

WHERE IT COMES FROM. Each family module (one per
`.claude/skills/tree-state/references/*.md` kernel file) declares its CASES
by hand from that file, and its TESTS: for every device test function, how a
collected case of it resolves. It is sparse by design -- no Rys Hessians,
the HGP one-electron and split HGP J / K on the RTX 3080 only, and so on --
because it lists what exists, never the cross product.

HOW A COLLECTED CASE RESOLVES. A test function's nodeid, with its backend
spelled `X` (`test_cuda_k_cell...` and `test_hip_k_cell...` are one rule;
`test_emit_cuda.py` and `test_emit_hip.py` too), names one rule:

- `Classes(engine, quantity, deriv)`: one class per case, read from the
  parameters, then the id -- a `Spec` (all four fields), a class tuple,
  `('class', am)`, `(Target, la, lb)`, a unit name such as `hj_0010` or
  `gs_00` (its prefix names engine, quantity and deriv, `UNITS`), or an
  integer `L`. The rule's own fields are the default when they do not name
  them. `otherwise` is the allowlist entry for a function that mixes class
  cells with molecule cells.
- `Span(keys)` (`span(engine, quantity, deriv, Ls)`): every case covers
  these classes -- "every class through (dd)", a program over a synthetic
  basis, a molecule whose basis reaches a known top shell.
- `ById({id: rule})`: a rule per case id, for a function whose cases differ
  in kind; an id it does not name is an error.
- `Allow(reason)`: covers no case, and says why -- a whole molecule's build
  that no family relies on, a generator-machinery check that is no family's
  class, the device Boys' template order. These ARE the allowlist the
  bijection check reads.

test/test_registry.py holds the registry to the collected device cases in
both directions; conftest.py's collection hook writes them out for it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from felizjueves.spec import Spec, Target

from . import (
    cost,
    fock_hgp,
    gradients,
    hessians,
    j_chain,
    k_exchange,
    one_electron_extra,
    rys_and_boys,
    schwarz_overlap_df,
)
from .rules import (
    BACKEND_CARD,
    CARDS,
    ENGINES,
    TIERS,
    Allow,
    ById,
    Case,
    Classes,
    Key,
    Rule,
    Span,
)

__all__ = [
    "BACKEND_CARD", "CARDS", "ENGINES", "FAMILIES", "REGISTRY", "RULES",
    "TIERS", "Allow", "ById", "Case", "Classes", "CollectedCase", "Key",
    "Resolution", "Span", "collect", "cost", "device_cases", "dump_if_asked",
    "family_of", "resolve",
]

FAMILIES = {
    "j-chain": j_chain,
    "k-exchange": k_exchange,
    "fock-hgp": fock_hgp,
    "schwarz-overlap-df": schwarz_overlap_df,
    "one-electron-extra": one_electron_extra,
    "gradients": gradients,
    "hessians": hessians,
    "rys-and-boys": rys_and_boys,
}


def _registry() -> dict[Key, Case]:
    out: dict[Key, Case] = {}
    costs = cost.load()
    for name, module in FAMILIES.items():
        for case in module.CASES:
            if case.key in out:
                raise ValueError(f"{name}: {case.key} declared twice")
            out[case.key] = dataclasses.replace(case,
                                                cost_s=costs.get(case.key))
    return out


def family_of() -> dict[Key, str]:
    """Every registry case's key -> the family that declares it."""
    return {case.key: name for name, module in FAMILIES.items()
            for case in module.CASES}


def _rules() -> dict[str, Rule]:
    out: dict[str, Rule] = {}
    for name, module in FAMILIES.items():
        for function, rule in module.TESTS.items():
            if function in out:
                raise ValueError(f"{name}: a rule for {function} twice")
            out[function] = rule
    return out


REGISTRY: dict[Key, Case] = _registry()
RULES: dict[str, Rule] = _rules()

# A unit name's prefix -> (engine, quantity, deriv): the generator's own
# spelling of a class (`hj_0010`, `gs_00`, `d2eri3_012`, `gk_00_01`).
UNITS: dict[str, tuple[str, str, int]] = {
    "s": ("mmd", "s", 0), "t": ("mmd", "t", 0), "v": ("mmd", "v", 0),
    "dq": ("mmd", "dq", 0), "jab": ("mmd", "jab", 0),
    "mu": ("mmd", "dipole", 0), "quad": ("mmd", "quadrupole", 0),
    "metric": ("mmd", "metric", 0), "eri3": ("mmd", "eri3", 0),
    "gs": ("mmd", "s", 1), "gt": ("mmd", "t", 1), "gv": ("mmd", "v", 1),
    # the d1 matrix forms (#687): a variant of the d1 case, as a lever is
    "ips": ("mmd", "s", 1), "ipt": ("mmd", "t", 1), "ipv": ("mmd", "v", 1),
    "iprv": ("mmd", "v", 1),
    "gjab": ("mmd", "jab", 1), "gk": ("mmd", "k", 1),
    "gmetric": ("mmd", "metric", 1), "geri3": ("mmd", "eri3", 1),
    "ipmetric": ("mmd", "metric", 1), "xeri3": ("mmd", "eri3", 1),
    "hdq": ("mmd", "dq", 2), "hjab": ("mmd", "jab", 2),
    "d2s": ("mmd", "s", 2), "d2t": ("mmd", "t", 2), "d2v": ("mmd", "v", 2),
    "hkx": ("mmd", "k", 2),
    "kx1": ("mmd", "kx1", 1), "kxc": ("mmd", "kxc", 1),
    "d2metric": ("mmd", "metric", 2), "d2eri3": ("mmd", "eri3", 2),
    "f": ("hgp", "f", 0), "hj": ("hgp", "j", 0), "hk": ("hgp", "k", 0),
    "gf": ("hgp", "f", 1),
    "ghk": ("hgp", "k", 1),
    "hs": ("hgp", "s", 0), "ht": ("hgp", "t", 0), "hv": ("hgp", "v", 0),
    "hmu": ("hgp", "dipole", 0),
    "hgs": ("hgp", "s", 1), "ghv": ("hgp", "v", 1),
    "hmetric": ("hgp", "metric", 0), "heri3": ("hgp", "eri3", 0),
    "hgeri3": ("hgp", "eri3", 1),
    "hgmetric": ("hgp", "metric", 1),
    "fr": ("rys", "f", 0), "rj": ("rys", "j", 0), "rk": ("rys", "k", 0),
    "rs": ("rys", "s", 0), "rt": ("rys", "t", 0), "rv": ("rys", "v", 0),
    "rmetric": ("rys", "metric", 0),
    "reri3": ("rys", "eri3", 0),
}
_UNIT = re.compile(r"^([a-z][a-z0-9]*?)_(?:o_)?(\d[\d_]*)$")

_BACKEND = re.compile(r"(?<=_)(cuda|hip|nvcc|hipcc)(?=[_.]|$)")


def function_of(nodeid: str) -> str:
    """`nodeid` without its parameters, its backend spelled `X`."""
    return _BACKEND.sub("X", nodeid.partition("[")[0])


@dataclass(frozen=True)
class Resolution:
    """What one collected case covers: `keys`, or `allow`'s reason for none.
    `rule` is None when no rule names the function at all."""

    keys: tuple[Key, ...] = ()
    allow: str | None = None
    rule: Rule | None = None


def _unit(name: str) -> tuple[tuple[str, str, int], int] | None:
    m = _UNIT.match(name)
    if m is None or m.group(1) not in UNITS:
        return None
    return UNITS[m.group(1)], sum(int(c) for c in m.group(2) if c.isdigit())


Found = tuple[tuple[str, str, int] | str | None, int]


def _from_value(value: Any) -> Found | None:
    """One parameter's L, and what it names of (engine, quantity, deriv): all
    three, the quantity alone (a `str`), or nothing (None)."""
    if isinstance(value, Spec):
        return ((value.engine.value, value.target.value,
                 value.derivative_order), value.total_am)
    if isinstance(getattr(value, "name", None), str) and hasattr(
            value, "params"):  # an ir.Kernel
        return _unit(value.name)
    if isinstance(value, str):
        return _unit(value)
    if not isinstance(value, tuple) or not value:
        L = getattr(value, "L", None)  # performance.asym.AsymCell
        return (None, L) if type(L) is int else None
    if all(type(v) is int for v in value):  # a class tuple
        return None, sum(value)
    head = value[0]
    if head == "class" and len(value) == 2:  # ('class', am)
        return _from_value(value[1])
    if isinstance(head, Target) and all(type(v) is int for v in value[1:]):
        return head.value, sum(value[1:])  # (Target, la, lb)
    if isinstance(head, str) and len(value) == 2 and isinstance(
            value[1], tuple):  # ('gs', (la, lb))
        unit = _unit(f"{head}_0")
        if unit is not None:
            return unit[0], sum(value[1])
    for v in reversed(value):  # ('class', 'j', 'canonical', 'hj_0000')
        found = _unit(v) if isinstance(v, str) else None
        if found is not None:
            return found
    return None


_L_NAMES = ("L", "L_", "l")


def classes_of(params: dict[str, Any], case_id: str,
               rule: Classes) -> Key | None:
    """The one class a `Classes` case names -- from its parameters, then its
    id's tokens -- or None when it names none."""
    what: tuple[str, str, int] | None = None
    quantity: str | None = None
    L: int | None = None
    found_all = [_from_value(v) for k, v in params.items()
                 if k not in _L_NAMES]
    found_all += [_unit(t) for t in case_id.split("-")]
    for found in found_all:
        if found is None:
            continue
        if isinstance(found[0], tuple) and what is None:
            what = found[0]
        elif isinstance(found[0], str) and quantity is None:
            quantity = found[0]
        if L is None:
            L = found[1]
    for k in _L_NAMES:
        if type(params.get(k)) is int:
            L = params[k]
    if L is None:
        return None
    engine, q, deriv = what or (rule.engine, quantity or rule.quantity,
                                rule.deriv)
    if q == "schwarz":
        L *= 2  # tiers.py: (la lb|la lb) runs at 2 (la + lb)
    return Key(engine, q, deriv, L)


def resolve(item: Any) -> Resolution:
    """The keys a collected device case covers (module docstring)."""
    rule = RULES.get(function_of(item.nodeid))
    callspec = getattr(item, "callspec", None)
    case_id = callspec.id if callspec is not None else ""
    if isinstance(rule, ById):
        if case_id not in rule.by_id:
            raise LookupError(f"case id {case_id!r} is not in its ById rule")
        rule = rule.by_id[case_id]
    if rule is None:
        return Resolution()
    if isinstance(rule, Allow):
        return Resolution(allow=rule.reason, rule=rule)
    if isinstance(rule, Span):
        return Resolution(keys=rule.keys, rule=rule)
    key = classes_of(callspec.params if callspec else {}, case_id, rule)
    if key is None:
        if rule.otherwise is None:
            raise LookupError(f"no class in the parameters of {case_id!r}")
        return Resolution(allow=rule.otherwise.reason, rule=rule)
    also = tuple(key._replace(quantity=q) for q in rule.also)
    return Resolution(keys=(key, *also), rule=rule)


# -- the collected device cases -----------------------------------------------


@dataclass(frozen=True)
class CollectedCase:
    """One collected `gpu` or compile-only case, resolved."""

    nodeid: str
    function: str
    tier: str  # "gpu" or "compile"
    backend: str  # "cuda" or "hip"
    parametrized: bool
    keys: tuple[Key, ...] = ()
    allow: str | None = None
    ruled: bool = False
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"nodeid": self.nodeid, "function": self.function,
                "tier": self.tier, "backend": self.backend,
                "parametrized": self.parametrized,
                "keys": [list(k) for k in self.keys], "allow": self.allow,
                "ruled": self.ruled, "error": self.error}

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> CollectedCase:
        return cls(row["nodeid"], row["function"], row["tier"],
                   row["backend"], row["parametrized"],
                   tuple(Key(*k) for k in row["keys"]), row["allow"],
                   row["ruled"], row["error"])


def device_cases(items: Iterable[Any]) -> Iterator[CollectedCase]:
    """Every `gpu` and compile-only (`cuda` or `hip`, no `gpu`) item."""
    for item in items:
        backend = ("cuda" if item.get_closest_marker("cuda") else
                   "hip" if item.get_closest_marker("hip") else None)
        if backend is None:
            continue
        tier = "gpu" if item.get_closest_marker("gpu") else "compile"
        try:
            got = resolve(item)
            error = None
        except LookupError as exc:
            got, error = Resolution(rule=RULES.get(function_of(item.nodeid))), str(exc)
        yield CollectedCase(
            item.nodeid, function_of(item.nodeid), tier, backend,
            getattr(item, "callspec", None) is not None, got.keys, got.allow,
            got.rule is not None, error)


DUMP_ENV = "FJ_REGISTRY_DUMP"
ROOT = Path(__file__).resolve().parents[2]


def dump_if_asked(items: Iterable[Any]) -> None:
    """With `FJ_REGISTRY_DUMP=<path>` set, write every collected device case,
    resolved, to `<path>` as JSON (test/test_registry.py's input)."""
    path = os.environ.get(DUMP_ENV)
    if not path:
        return
    rows = [c.to_json() for c in device_cases(items)]
    Path(path).write_text(json.dumps(rows, indent=0))


def collect(out: Path) -> list[CollectedCase]:
    """Collect the whole of test/ -- every marker, no `-m` -- in a child
    pytest with `FJ_REGISTRY_DUMP=<out>`, and read back its device cases."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env[DUMP_ENV] = str(out)
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-o", "addopts=", "test"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stdout[-4000:] + done.stderr[-4000:])
    return [CollectedCase.from_json(r) for r in json.loads(out.read_text())]
