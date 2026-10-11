"""The import graph of src/felizjueves, as an exact, asserted allow-list.

Three of the kernel milestone's load-bearing rules are import-direction facts:
nothing that emits text may import `reference/` (#20), the interpreter must not
know what McMurchie-Davidson is (#23), and nothing in a built `Kernel` may be
CUDA- or HIP-specific (#25). This file is where they stop being prose.

TWO LAYERS, BOTH LITERAL.

- `RULE_OF` gives every module a rule, and `RULES` says, per rule, what it may
  import and what it must NOT import, each prohibition with its reason. A rule
  whose reason is not recorded gets relaxed by the next person who hits it.
- `EDGES` is the exact set of internal imports each module makes today. It is
  checked against the tree in BOTH directions: an import not listed fails, and
  a listed edge that no longer occurs fails too (a stale entry is drift with a
  green test over it). Every listed edge must also satisfy its module's rule.

So a new module FAILS until it is given a rule and its edges -- that per-PR
edit is the mechanism, not a cost. Do not make the rules coarse to avoid churn:
"anything may import `ir`" passes forever and detects nothing.

WHAT THIS CANNOT SEE. It reads the source with `ast`, so it sees every
`import` statement wherever it is written -- module level, inside a function,
under `if TYPE_CHECKING:` -- but not a RUNTIME reach: `importlib`, an
`__import__` call, or (by design) a callable injected by the caller. An
engine or an oracle passed in as an argument is invisible here, and that
inversion of control is the shape to prefer: `interp.py` takes its Boys
evaluator as `boys=` and the pins pass `reference.boys.boys`, so the
interpreter itself never reaches `reference/`. Nor is this a design review:
it can tell you the layering is what was declared, never that it is right.

Names are relative to `felizjueves`; `__init__` is the root package.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "src" / "felizjueves"
PKG = "felizjueves"


@dataclass(frozen=True)
class Rule:
    why: str
    may: frozenset[str]  # prefixes: "plan" admits "plan" and "plan.*"
    must_not: dict[str, str] = field(default_factory=dict)  # prefix -> reason


def _rule(why: str, may: tuple[str, ...], **must_not: str) -> Rule:
    return Rule(why, frozenset(may), dict(must_not))


_REF = "reference"
_GROUND_TRUTH = ("an import from reference/ makes it share ancestry with the "
                 "ground truth it is checked against, and the pin compares a "
                 "thing to itself (#20)")

RULES: dict[str, Rule] = {
    "root": _rule("the package marker; nothing to compose", ()),
    "spec": _rule("the frozen key every subsystem joins on; it depends on "
                  "nothing so that everything may depend on it", ()),
    "literature": _rule(
        "published tables and the rules that score against them; self-contained "
        "so that a score cannot be bent by the code it scores",
        ("literature",),
        plan="the plan is what literature/ scores; scoring must not import the "
             "candidate"),
    "plan": _rule(
        "host-side planning, [JEngine, v2, sec. 6] steps 1-3; it reads the key "
        "and the published tables, and knows nothing of IR or devices",
        ("spec", "plan", "literature"),
        ir="the plan counts; it emits no instructions",
        reference=_GROUND_TRUTH),
    "reference": _rule(
        "numeric ground truth (#20) with independent ancestry: it may use only "
        "its own primitives (pq -> boys)",
        ("reference",),
        spec="ground truth takes plain numbers, not the generation key",
        plan="the reference must not share the MILD/DAG/sieve path it checks",
        ir="the reference is not built from the IR it checks",
        recurrence="the reference is the recurrence run directly, not the "
                   "builder's version of it"),
    "device": _rule(
        "the hardware vocabulary, `Arch` and `MeasuredKey` (#37): data, keyed "
        "on `Spec`",
        ("spec",),
        ir="an arch is data about a card, not about a kernel",
        emit="backends read `Arch`; `Arch` does not know the backends",
        reference=_GROUND_TRUTH),
    "ir": _rule(
        "the instruction set: types only (#22). May reach `spec` and the device "
        "vocabulary; today it needs neither",
        ("spec", "device"),
        plan="types only -- the IR does not know how a plan is made (#22)",
        reference=_GROUND_TRUTH),
    "interp": _rule(
        "an instruction-set machine for ir.py and nothing more (#23); Boys is "
        "injected (`boys=`), not imported",
        ("ir",),
        reference="same ancestry as its own pin: the interpreter would be "
                  "checked against the module it was built on (#23)",
        plan="it executes instructions; it does not know the recurrence (#23)",
        recurrence="recurrence.py is McMurchie-Davidson; interp.py must not "
                   "know what that is (#23)",
        kernel="the builder is what the interpreter checks",
        emit="interpreting is not emitting; neither may lean on the other"),
    "lower": _rule(
        "IR -> IR rewrites: lower.py (register recycling, #24) and passes.py "
        "(the levers, first `atomize`, #48)",
        ("ir",),
        reference=_GROUND_TRUTH,
        emit="lowering is backend-neutral; a backend consumes its output",
        plan="lowering rewrites instructions, not plans"),
    "builder": _rule(
        "plan -> IR: recurrence.py (the McMurchie-Davidson recurrence as IR, "
        "#23) and kernel.py (the [p|q] contract, Alg. 1, #25). They read the "
        "plan and the key and produce a vendor-neutral `Kernel`; a split "
        "class's kernels are built then `passes.atomize`d (#49), so a "
        "builder may apply a pass",
        ("spec", "plan", "ir", "recurrence", "passes"),
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific "
             "(#25)",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
        interp="the builder is checked by the interpreter, never built on it",
        reference=_GROUND_TRUTH),
    "emit": _rule(
        "IR -> text, one backend per vendor sharing emit/_c.py (#26/#27). Reads "
        "a finished `Kernel` and an `Arch`",
        ("ir", "device", "emit"),
        reference="#20's hazard: a reference an emitter can reach is a "
                  "reference an emitter can be derived from",
        interp="emitting is not interpreting; the golden files and the kernel "
               "tier check the text",
        plan="a backend spells a `Kernel`; it does not decide one",
        kernel="a backend takes any `Kernel`; composing builder + backend is "
               "a driver's job, not a backend's",
        recurrence="as `kernel`: the backend does not know which kernel it is "
                   "spelling"),
    "driver": _rule(
        "CLIs that COMPOSE builder + backend and write or measure the result: "
        "`python -m felizjueves.emit` (KERNEL_GEN_CMD, #28) and "
        "`python -m felizjueves.measured` (#28). Their own rule, so that emit/*'s "
        "stays narrow",
        ("spec", "device", "ir", "kernel", "emit"),
        reference="a driver emits text: #20's hazard applies to it as much as "
                  "to a backend",
        interp="a driver ships kernels; it does not check them -- the tests do",
        plan="a driver goes through the builder, so the kernel it writes is "
             "the kernel the tests built"),
    "analysis": _rule(
        "model against measurement: kernelrank.py ranks lower.KernelCost "
        "against the Measured rows (#54). It reads both sides and feeds "
        "neither",
        ("spec", "device", "kernel", "lower", "measured"),
        plan="a model-vs-measurement result must never reach the "
             "calibrated instrument the literature floors are measured with",
        literature="nothing in KernelCost may feed a literature floor "
                   "(lower.py); the limits it records are read by tests",
        reference=_GROUND_TRUTH),
}

RULE_OF: dict[str, str] = {
    "__init__": "root",
    "spec": "spec",
    "literature": "literature",
    "literature.claims": "literature",
    "literature.limits": "literature",
    "literature.rules": "literature",
    "literature.scoreboard": "literature",
    "literature.tables": "literature",
    "plan": "plan",
    "plan.__main__": "plan",
    "plan.batching": "plan",
    "plan.cost": "plan",
    "plan.dag": "plan",
    "plan.mild": "plan",
    "plan.pipeline": "plan",
    "plan.recovery": "plan",
    "plan.schedule": "plan",
    "plan.search": "plan",
    "plan.sieve": "plan",
    "plan.targets": "plan",
    "reference": "reference",
    "reference.boys": "reference",
    "reference.pq": "reference",
    "device": "device",
    "ir": "ir",
    "interp": "interp",
    "lower": "lower",
    "passes": "lower",
    "recurrence": "builder",
    "kernel": "builder",
    "emit": "emit",
    "emit._c": "emit",
    "emit.cuda": "emit",
    "emit.hip": "emit",
    "emit.__main__": "driver",
    "measured": "driver",
    "kernelrank": "analysis",
}

EDGES: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in {
    "__init__": (),
    "spec": (),
    "literature": (),
    "literature.claims": (),
    "literature.limits": (),
    "literature.rules": ("literature.tables",),
    "literature.scoreboard": ("literature.rules", "literature.tables"),
    "literature.tables": (),
    "plan": (),
    "plan.__main__": ("literature.rules", "literature.scoreboard",
                      "literature.tables", "plan.cost", "plan.pipeline",
                      "plan.recovery", "plan.search", "plan.sieve", "spec"),
    "plan.batching": ("plan.targets", "spec"),
    "plan.cost": ("plan.dag",),
    "plan.dag": ("plan.batching", "plan.mild", "plan.targets", "spec"),
    "plan.mild": ("plan.batching", "plan.targets"),
    "plan.pipeline": ("plan.cost", "plan.dag", "plan.sieve", "spec"),
    "plan.recovery": ("literature.rules", "literature.tables", "plan.batching",
                      "plan.cost", "plan.dag", "plan.sieve", "plan.targets",
                      "spec"),
    "plan.schedule": ("plan.dag",),
    "plan.search": ("plan.batching", "plan.cost", "plan.dag", "plan.recovery",
                    "plan.sieve"),
    "plan.sieve": ("plan.batching", "plan.dag", "plan.targets", "spec"),
    "plan.targets": (),
    "reference": (),
    "reference.boys": (),
    "reference.pq": ("reference.boys",),
    "device": ("spec",),
    "ir": (),
    "interp": ("ir",),
    "lower": ("ir",),
    "passes": ("ir",),
    "recurrence": ("ir", "plan.dag", "plan.pipeline", "plan.schedule",
                   "plan.targets", "spec"),
    "kernel": ("ir", "passes", "plan.dag", "plan.pipeline", "plan.schedule",
               "plan.targets", "recurrence", "spec"),
    "emit": (),
    "emit._c": ("device", "ir"),
    "emit.cuda": ("device", "emit._c", "ir"),
    "emit.hip": ("device", "emit._c", "ir"),
    "emit.__main__": ("device", "emit.cuda", "emit.hip", "kernel"),
    "measured": ("device", "emit.cuda", "emit.hip", "ir", "kernel", "spec"),
    "kernelrank": ("device", "kernel", "lower", "measured", "spec"),
}.items()}

# The rules under which a module writes kernel text, for #20's hazard.
EMITS_TEXT = ("emit", "driver")


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def _name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts == ("__init__",):
        return "__init__"
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _modules() -> dict[str, Path]:
    return {_name(p): p for p in sorted(ROOT.rglob("*.py"))}


def imports_of(source: str, name: str, is_package: bool,
               known: set[str]) -> set[str]:
    """Every felizjueves module `source` imports, at ANY depth of the AST.

    `from felizjueves.emit import cuda` resolves to `emit.cuda` when that is a
    module; otherwise a `from X import name` is an edge to X.
    """
    here = [] if name == "__init__" else name.split(".")
    pkg = here if is_package else here[:-1]
    found: set[str] = set()

    def internal(dotted: str) -> str | None:
        if dotted == PKG:
            return "__init__"
        if dotted.startswith(PKG + "."):
            return dotted[len(PKG) + 1:]
        return None

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (m := internal(alias.name)) is not None:
                    found.add(m)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - (node.level - 1)]
                tail = [node.module] if node.module else []
                m = ".".join(base + tail) or "__init__"
            else:
                m = internal(node.module or "")
            if m is None:
                continue
            for alias in node.names:
                sub = alias.name if m == "__init__" else f"{m}.{alias.name}"
                found.add(sub if sub in known else m)
    found.discard(name)
    return found


def _graph() -> dict[str, set[str]]:
    mods = _modules()
    known = set(mods)
    return {n: imports_of(p.read_text(), n, p.name == "__init__.py", known)
            for n, p in mods.items()}


def _under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


def test_the_walk_sees_function_level_imports() -> None:
    """A late import inside a function is still an import; it must be seen."""
    src = ("def f():\n"
           "    if True:\n"
           "        from felizjueves.reference.boys import boys\n"
           "    import felizjueves.plan.dag\n"
           "    from felizjueves.emit import cuda\n"
           "    from . import ir\n")
    known = {"reference.boys", "plan.dag", "emit", "emit.cuda", "ir"}
    assert imports_of(src, "interp", False, known) == {
        "reference.boys", "plan.dag", "emit.cuda", "ir"}


def test_every_module_has_a_rule() -> None:
    # A new package FAILS until it is given one. This is the whole mechanism.
    mods = set(_modules())
    unruled = sorted(mods - set(RULE_OF))
    assert not unruled, f"modules with no rule in RULE_OF: {unruled}"
    no_edges = sorted(mods - set(EDGES))
    assert not no_edges, f"modules with no EDGES entry: {no_edges}"
    gone = sorted((set(RULE_OF) | set(EDGES)) - mods)
    assert not gone, f"rules for modules that no longer exist: {gone}"
    assert set(RULE_OF.values()) == set(RULES), "a rule nothing uses, or a typo"
    for rule in RULES.values():
        assert rule.why and all(rule.must_not.values()), "every rule needs a reason"


def test_no_import_outside_its_rule() -> None:
    graph = _graph()
    unlisted = {m: sorted(got - EDGES.get(m, frozenset()))
                for m, got in graph.items() if got - EDGES.get(m, frozenset())}
    assert not unlisted, f"imports not on the allow-list: {unlisted}"
    breaches = []
    for m, edges in EDGES.items():
        rule = RULES[RULE_OF[m]]
        for e in sorted(edges):
            banned = [r for p, r in rule.must_not.items() if _under(e, p)]
            if banned:
                breaches.append(f"{m} -> {e}: MUST NOT ({banned[0]})")
            elif not any(_under(e, p) for p in rule.may):
                breaches.append(f"{m} -> {e}: outside rule "
                                f"{RULE_OF[m]!r} (may: {sorted(rule.may)})")
    assert not breaches, "\n".join(breaches)


def test_no_rule_is_stale() -> None:
    # An allow-list entry nothing exercises is removed, not kept "just in case".
    graph = _graph()
    stale = {m: sorted(edges - graph.get(m, set()))
             for m, edges in EDGES.items() if edges - graph.get(m, set())}
    assert not stale, f"allow-listed imports that no longer occur: {stale}"


def test_nothing_that_emits_text_imports_the_reference() -> None:
    # #20's hazard, as a test rather than a sentence. Named after the rule so
    # that a failure reads as the finding it is. TRANSITIVE: an emitter that
    # reaches reference/ through any chain of imports can be derived from it.
    graph = _graph()

    def reach(m: str) -> set[str]:
        seen: set[str] = set()
        todo = [m]
        while todo:
            for e in graph.get(todo.pop(), set()):
                if e not in seen:
                    seen.add(e)
                    todo.append(e)
        return seen

    emitters = sorted(m for m in graph if RULE_OF.get(m) in EMITS_TEXT)
    assert emitters, "no emitters found: the rule names have drifted"
    reached = {m: sorted(e for e in reach(m) if _under(e, _REF))
               for m in emitters}
    reached = {m: r for m, r in reached.items() if r}
    assert not reached, f"text emitters that reach reference/: {reached}"
