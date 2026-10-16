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
                  "nothing but the lever vocabulary (a leaf, #337), so that "
                  "everything may depend on it", ("levers",)),
    "levers": _rule(
        "levers.py (#337): the kernel-body lever vocabulary and its "
        "per-target legality table, as data. A leaf, so that `spec` can "
        "refuse an illegal (target, lever) at construction; targets are "
        "named by their key slot, never by importing `spec`",
        (),
        spec="`spec` imports this module to validate; the reverse is a cycle",
        plan="the plan layer's levers are its own; this vocabulary is the "
             "builders'",
        lower="lowering's levers are its own (`lower.LEVERS`)"),
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
        dispatch="nothing measured may reach the plan's cost model: a "
                 "dispatch choice is admitted by the plan, never imported "
                 "by it (#145)",
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
    "gate": _rule(
        "the card gate (#109): wait for a quiet card before a measurement, "
        "and refuse the phase when it is not. Stdlib only, so that a driver, "
        "the probes and the tests can all reach it",
        (),
        literature="a gate decides whether a measurement is taken; it must "
                   "not know what the measurement will be scored against"),
    "probe": _rule(
        "ceilings.py (#109): the roofline ceiling probes and their record, "
        "behind the gate. It writes what `device` reads",
        ("device", "gate"),
        literature="the published tables must never see a ceiling a re-probe "
                   "rewrites (#109's hazard)",
        plan="the plan counts; a card's measured ceiling must not reach it",
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
        "builder may apply a pass. It reads the kernel-body levers through "
        "the key (`Spec.kernel_body_levers`, #337), never `levers` directly",
        ("spec", "plan", "ir", "recurrence", "passes"),
        levers="the builder reads the lever vocabulary through `spec`, its "
               "one key (#337); the vendor-neutral import set is pinned",
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific "
             "(#25)",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
        interp="the builder is checked by the interpreter, never built on it",
        reference=_GROUND_TRUTH),
    "fragment": _rule(
        "e_fragment.py (#152): device-code `Fragment`s for the AO side's "
        "quantities, first the McMurchie-Davidson `E`. It builds IR (`ir`) "
        "and reads the Hermite order from the plan's targets, which "
        "`ao.hermite` restates and a test pins equal",
        ("ir", "plan.targets"),
        ao="`ao.hermite` is the reference the fragment is interpreted "
           "against, and `ao.shell` the layout it is pinned to: importing "
           "either shares ancestry with its witness (#20); the Cartesian "
           "order is restated and a test asserts the two equal",
        interp="the fragment is checked by the interpreter, never built on "
               "it; and the interpreter must not know what `E` is (#152)",
        emit="a fragment is vendor-neutral IR; the arch enters at emission",
        device="the IR is arch-neutral",
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
        "stays narrow. A driver also LOWERS what it built when the Spec names "
        "a lowering lever (`measured.cell_kernels`, #55): build -> lower -> emit. "
        "`measured` writes its rows into `measured_store` (#94), and compiles "
        "#113's asymmetric study cells through `asym`, and #205's K cells "
        "through `kernel_k`",
        ("spec", "device", "ir", "kernel", "lower", "emit", "measured_store",
         "asym", "kernel_dq", "kernel_jab", "kernel_k"),
        reference="a driver emits text: #20's hazard applies to it as much as "
                  "to a backend",
        interp="a driver ships kernels; it does not check them -- the tests do",
        plan="a driver goes through the builder, so the kernel it writes is "
             "the kernel the tests built"),
    "analysis": _rule(
        "model against measurement: kernelrank.py ranks lower.KernelCost "
        "against the Measured rows (#54), batchreport.py derives a "
        "recommended n_batches from their wall-clock alone (#56; its EDGES "
        "hold it off `lower`), and ridge.py scores #114's declared "
        "ridge-crossing rule off the counter rows (its EDGES hold it to the "
        "store and the card), and crossread.py reads those three against "
        "each other (#115; the one analysis that may import its siblings, "
        "and only the three it cross-reads). It reads both sides and feeds "
        "neither",
        ("spec", "device", "kernel", "lower", "measured_store",
         "kernelrank", "batchreport", "ridge"),
        measured="an analysis reads the STORE (`measured_store`), never the "
                 "driver that compiles into it: a ranking must not need a "
                 "toolchain to be recomputed (#94)",
        plan="a model-vs-measurement result must never reach the "
             "calibrated instrument the literature floors are measured with",
        literature="nothing in KernelCost may feed a literature floor "
                   "(lower.py); the limits it records are read by tests",
        reference=_GROUND_TRUTH),
    "plan_analysis": _rule(
        "planrank.py (#165): the plan layer's counts against the measured "
        "rows -- PlanCost of the literal, no-sieve and tuned plans of a cell "
        "against its spills and its same-session wall-clock. Not `analysis`: "
        "it must read `plan` (the counts are the plan's), which `analysis` "
        "may not. It reads both sides and feeds neither: nothing may import "
        "it (`plan` and `literature` admit no such edge), and a plan chosen "
        "on its reading is #166's hand-recorded decision, not a loop the "
        "code closes. `gate` only for GPU_BUSY_MAX, the clean-phase bound",
        ("spec", "device", "gate", "plan", "measured_store", "kernelrank"),
        literature="the reading may not feed a literature floor: a floor "
                   "moved by what a card did with a kernel is no longer the "
                   "paper's number (#165's hazard)",
        measured="an analysis reads the STORE, never the driver that "
                 "compiles into it: the reading must be recomputable "
                 "without a toolchain (#94)",
        dispatch="the reading is not a choice; dispatch chooses from the "
                 "clock alone, and nothing here may become its input",
        launch="the reading is about one class's kernel, not a molecule's "
               "launch plan, and must not reach what the plan launches",
        reference=_GROUND_TRUTH),
    "dispatch": _rule(
        "dispatch.py (#145): the batch count of a class, chosen from the "
        "measured wall-clock rows by batchreport's rule (best median) and "
        "INJECTED into `launch.launches` as `resolve=`. Not `analysis`: it "
        "feeds the launch plan, by injection. It reads the store and the "
        "card, nothing else",
        ("device", "measured_store", "batchreport"),
        lower="measured rows only, never a model: `lower.KernelCost` is the "
              "model #54's limits record mis-ranking the hardware (#145)",
        kernelrank="a model-vs-measurement ranking is not a choice: the "
                   "clock arbitrates, never a model or byte proxy (#145)",
        ridge="a counter/roofline verdict is a byte proxy, not the clock",
        crossread="the cross-read reports disagreements; it does not choose",
        plan="nothing measured may reach the plan's cost model; the plan "
             "admits the chosen count in `launch`, not here",
        launch="the resolver is injected into the launch plan; it does not "
               "need the plan, and importing it would invite the reverse",
        measured="the resolver reads the STORE, never the driver: a choice "
                 "must not need a toolchain (#94)",
        literature="a dispatch choice is not scored against the published "
                   "tables, and nothing measured may feed a floor",
        reference=_GROUND_TRUTH),
    "study": _rule(
        "asym.py (#113): the asymmetric split, a STUDY that builds a batch "
        "list the generator refuses (plan + the builder's `_pq_kernel` + "
        "`passes.atomize`) and judges its rows against the literal ones read "
        "from the store. Not `builder`: it reads the store; not `analysis`: "
        "it plans",
        ("spec", "device", "ir", "plan", "kernel", "passes", "measured_store"),
        literature="the study's verdict is recorded by hand in claims.json / "
                   "limits.json and read by tests; nothing measured may feed "
                   "a literature floor",
        measured="the study reads the STORE; the driver compiles its cells, "
                 "never the reverse (#94)",
        emit="the study builds kernels; spelling them is the driver's job",
        interp="the kernel tier checks the study's kernels, not the "
               "interpreter",
        reference=_GROUND_TRUTH),
    "store": _rule(
        "measured_store.py (#94): the `Measured` record, its file and read "
        "path, the declared SUBGRID and the verdicts read off the rows. It "
        "names cells (`kernel.pq_spec`) and lowering levers (`lower.LEVERS`) "
        "but knows no toolchain, so the fast tier and the analyses read it "
        "without one (`TOOLCHAIN_STDLIB` below holds the runtime half)",
        ("spec", "device", "kernel", "lower"),
        measured="the driver writes into the store, never the reverse: a store "
                 "that imports its driver drags a toolchain into every reader "
                 "(#94)",
        emit="the store records what a compiler said about text; spelling the "
             "text is the driver's job (#94)",
        ir="a row is keyed on a Spec, not on instructions",
        interp="a measurement is not an interpretation",
        dispatch="the resolver reads the store; a store that imports it "
                 "is a cycle, and a choice would shape its own evidence "
                 "(#145)",
        plan="a stored measurement must never reach the instrument the "
             "literature floors are measured with",
        literature="nothing measured may feed a literature floor",
        reference=_GROUND_TRUTH),
    "ao": _rule(
        "the AO layer (#80, #81, #83, #85, #86): shells, the basis and the "
        "one normalisation convention, the pure Hermite coefficients `E`, the "
        "dressed pair data with its Schwarz bound (whose `[r]^(0)` evaluator "
        "is injected as `one_centre=`, never imported), the ERI block and the "
        "Coulomb matrix (whose `[p|q]` evaluator is injected as `pq=`) -- "
        "input data a "
        "kernel is eventually run on, in plain tuples and floats",
        ("ao",),
        spec="a shell is input data, not a generation key: no `Spec` field "
             "and no `key()` (#80)",
        plan="the AO side restates the Cartesian order rather than importing "
             "it; a test asserts the two agree, so neither drifts silently",
        ir="the AO layer describes basis functions, not instructions",
        emit="the definition of the input must not be shaped by how a kernel "
             "is spelled",
        reference="the reference is checked against what the AO layer feeds "
                  "it; sharing ancestry would compare a thing to itself "
                  "(#20)"),
    "adapter": _rule(
        "ao/drive.py (#87): runs the GENERATED kernel in the interpreter as "
        "ao.j's `jp=` seam -- builder + interp + ao, producing numbers. Not "
        "a `driver`: that rule ships kernel text and must not interpret or "
        "plan, and this module does both (it asks the plan which n_batches "
        "a class admits, #64). Not `ao`: the AO layer may not know IR. "
        "Also ao/staged.py (#143): the same, over a LAUNCH PLAN -- it reads "
        "`launch`'s `ClassLaunch`es, and `passes.atomize`s a one-batch "
        "kernel because one `J_p` per bra class has a writer per ket class. "
        "And ao/drive.py's chain (#155): `Target.DQ` -> `[p|q]` -> "
        "`Target.JAB` composed over that staging, so the adapter reaches "
        "`ao.staged` and builds its own `launch` plan from the DQ kernel's "
        "read-back `Dmax`",
        ("spec", "kernel", "ir", "interp", "plan", "ao", "launch", "passes",
         "kernel_dq", "kernel_jab"),
        reference="the kernel's numbers are compared against reference/; "
                  "Boys arrives by injection (`boys=`), so the adapter "
                  "shares no ancestry with its witness (#20)",
        emit="it produces numbers, not text: a backend's spelling is the "
             "kernel tier's to check, not this adapter's"),
    "launch": _rule(
        "launch.py (#138): the whole-molecule launch plan as a VALUE -- which "
        "class kernels run over which bra and ket index records, in which "
        "order. It reads the key, asks the plan which batch counts a class "
        "admits, builds the class's kernels, and takes its records from the "
        "AO layer (milestone 6 (a)); the bound arrives injected (`bound=`). "
        "`ir` for the `Kernel` type it carries",
        ("spec", "plan", "kernel", "ir", "ao"),
        reference=_GROUND_TRUTH + "; the bound is injected (`bound=`), as "
                  "ao/pairs.py takes its `one_centre=`",
        interp="the plan is data: nothing is launched or interpreted here",
        **{"ao.drive": "the adapter runs kernels; the staged driver beside "
                       "it imports this plan (#143), so the plan importing "
                       "the adapter is a cycle and drags the interpreter in "
                       "-- `class_spec` is restated, and a test pins the "
                       "two equal",
           "ao.staged": "the staged driver (#143) runs this plan; the plan "
                        "importing it is the same cycle as `ao.drive`"},
        emit="the plan is data: no text is emitted (#138's acceptance)",
        measured="a batch count is the plan's admission or an injected "
                 "resolver's (#145), never a toolchain's",
        dispatch="the resolver is injected (`resolve=`, #145), never "
                 "imported: the plan stays data and the policy "
                 "replaceable",
        measured_store="nothing measured may reach the plan by import: a "
                       "dispatch policy arrives injected (#145)",
        literature="a launch plan is not scored against the published "
                   "tables; nothing here may feed a floor"),
    "k_launch": _rule(
        "launch_k.py (#204): the K launch plan as a VALUE, BESIDE `launch` "
        "-- which K class kernels run over which ordered bra and ket index "
        "records, in which order -- and its walk in Python. It imports "
        "from `launch` the part the two plans share (`PairRec`, the stable "
        "sort, the bound check, the batch admission) and from `ao` its "
        "records and the host's staged contraction; the bound, the ket "
        "sort key and the K kernel builder arrive injected (`bound=`, "
        "`key=`, `build=`)",
        ("spec", "ir", "ao", "launch"),
        reference=_GROUND_TRUTH + "; the bound and `pq` are injected",
        interp="the plan is data and the walk is ao.k's arithmetic; "
               "nothing is interpreted here",
        kernel="J's builder is `launch`'s business, and the K builder "
               "(`kernel_k`, #198) arrives injected as `build=`: this plan "
               "names a class's `Spec`, it does not build it",
        plan="the batch admission is `launch._batches_for`'s, shared, not "
             "restated: a second path to the plan could disagree with J's",
        emit="the plan is data: no text is emitted",
        **{"ao.drive": "the adapter runs kernels; a plan importing it drags "
                       "the interpreter in",
           "ao.staged": "the staged J driver reads `launch`'s plans; this "
                        "plan importing it is the same cycle as `launch`'s"},
        measured="no toolchain reaches a plan",
        dispatch="no K dispatch resolver exists (milestone 10's scope); "
                 "when one does it is injected, never imported (#145)",
        measured_store="nothing measured may reach the plan by import",
        literature="a launch plan is not scored against the published "
                   "tables; nothing here may feed a floor"),
    "jab_builder": _rule(
        "kernel_jab.py (#154): the bra-transform-and-scatter kernel, "
        "`Target.JAB` -- key -> IR, with the `E` block fragment inlined "
        "(`e_fragment`) and the Hermite order from the plan's targets",
        ("spec", "ir", "e_fragment", "plan.targets"),
        ao="`ao.j.jab_from_host` is the witness the kernel is interpreted "
           "against; importing the AO layer shares ancestry with it (#20)",
        interp="the builder is checked by the interpreter, never built on it",
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
        reference=_GROUND_TRUTH),
    "dq_builder": _rule(
        "kernel_dq.py (#153): the ket pre-contraction kernel, `Target.DQ` -- "
        "the key, the `E` block fragment (inlined), the Hermite order from "
        "the plan's targets, and IR. ao/drive.py's `KernelDQ` runs it in "
        "the interpreter as `ao.j`'s `dq=` seam",
        ("spec", "ir", "e_fragment", "plan.targets"),
        ao="`ao.j.dq_from_host` is the reference the kernel is interpreted "
           "against, and `ao.pairs.dress` the dressing it must not form: "
           "importing either shares ancestry with its witness (#20); the "
           "weight and the dressing arrive as host scalars",
        interp="the builder is checked by the interpreter, never built on it",
        emit="a builder produces vendor-neutral IR; the arch enters at "
             "emission (#156)",
        device="the `Kernel` is arch-neutral",
        reference=_GROUND_TRUTH),
    "count": _rule(
        "kdup.py (#202): the duplicated-work count, J against K, per class "
        "and batch count -- counted from `plan`'s batch partition, its "
        "`PlanCost` and the `E` nonzero pattern, with no card, no compiler "
        "and no built kernel. Not `analysis`: it must read `plan`, which "
        "`analysis` may not; not `plan_analysis`: it reads no measured row. "
        "It reads the plan and feeds nothing: no module imports it "
        "(test/test_k_duplication.py)",
        ("spec", "plan"),
        kernel="the count is the arithmetic half of a claim settled BEFORE "
               "any K kernel is built (#202): `k_spec` is restated and a "
               "test pins the two equal",
        ir="the count reads no instructions: a builder's output would make "
           "it a measurement of one kernel, not of the plan",
        e_fragment="the `E` nonzero pattern is restated from the plan's "
                   "targets; the fragment is a builder",
        interp="nothing here is run",
        emit="nothing here is spelled",
        measured="no toolchain: the count is recomputed in the fast tier",
        measured_store="no measured row may shape the arithmetic half; "
                       "`SUBGRID_LS` is restated and a test pins it",
        literature="the count scores nothing: the claim is E3's (#207), and "
                   "nothing here may move a floor",
        ao="the AO layer is the K witness's home, not the plan's",
        reference=_GROUND_TRUTH),
    "k_builder": _rule(
        "kernel_k.py (#198): the exchange kernel, `Target.K` -- the staged "
        "contraction as IR: J's plan, schedule and recurrence region "
        "(`plan`, `recurrence`), J's key split and launch mapping "
        "(`kernel`), and two `E` block fragments inlined (`e_fragment`); "
        "a split class's kernels are built then `passes.atomize`d (#203), "
        "as the `builder` rule lets `kernel.py` do",
        ("spec", "plan", "ir", "recurrence", "e_fragment", "kernel",
         "passes", "k_levers"),
        ao="`ao.k.k_quartet` is the witness the kernel is interpreted "
           "against, and `ao.pairs.dress` the dressing it must not form: "
           "importing either shares ancestry with its witness (#20); both "
           "dressings arrive as host scalars",
        interp="the builder is checked by the interpreter, never built on it",
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
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
    "plan.optimum": "plan",
    "plan.order_search": "plan",
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
    "gate": "gate",
    "ceilings": "probe",
    "ir": "ir",
    "interp": "interp",
    "lower": "lower",
    "passes": "lower",
    "recurrence": "builder",
    "kernel": "builder",
    "e_fragment": "fragment",
    "emit": "emit",
    "emit._c": "emit",
    "emit.cuda": "emit",
    "emit.hip": "emit",
    "emit.__main__": "driver",
    "measured": "driver",
    "measured_store": "store",
    "asym": "study",
    "kernelrank": "analysis",
    "batchreport": "analysis",
    "ridge": "analysis",
    "dispatch": "dispatch",
    "crossread": "analysis",
    "planrank": "plan_analysis",
    "ao": "ao",
    "ao.shell": "ao",
    "ao.hermite": "ao",
    "ao.pairs": "ao",
    "ao.eri": "ao",
    "ao.j": "ao",
    "ao.k": "ao",
    "ao.drive": "adapter",
    "ao.staged": "adapter",
    "launch": "launch",
    "launch_k": "k_launch",
    "kernel_jab": "jab_builder",
    "kernel_dq": "dq_builder",
    "kdup": "count",
    "kernel_k": "k_builder",
    "levers": "levers",
}

EDGES: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in {
    "__init__": (),
    "spec": ("levers",),
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
    "plan.optimum": ("plan.cost", "plan.targets"),
    "plan.order_search": ("plan.cost",),
    # #161: the `tuned` lever builds the searched board's recorded plan, so
    # pipeline reads searched_cuts.json through plan.search's loader and
    # replays it with its searched_plan. The builder (kernel) reaches it
    # only through pipeline.plan: no new edge in the `builder` rule.
    "plan.pipeline": ("plan.cost", "plan.dag", "plan.search", "plan.sieve",
                      "spec"),
    "plan.recovery": ("literature.rules", "literature.tables", "plan.batching",
                      "plan.cost", "plan.dag", "plan.sieve", "plan.targets",
                      "spec"),
    "plan.schedule": ("plan.dag",),
    # #172: plan.order_search is the climb and the frontier the searched
    # board's build step runs; plan.search is its one importer (pinned in
    # test_plan_order_search.py), and plan.targets names the orders' type.
    "plan.search": ("plan.batching", "plan.cost", "plan.dag",
                    "plan.order_search", "plan.recovery", "plan.sieve",
                    "plan.targets"),
    "plan.sieve": ("plan.batching", "plan.dag", "plan.targets", "spec"),
    "plan.targets": (),
    "reference": (),
    "reference.boys": (),
    "reference.pq": ("reference.boys",),
    "device": ("spec",),
    "gate": (),
    "ceilings": ("device", "gate"),
    "ir": (),
    "interp": ("ir",),
    "lower": ("ir",),
    "passes": ("ir",),
    "recurrence": ("ir", "plan.dag", "plan.pipeline", "plan.schedule",
                   "plan.targets", "spec"),
    "kernel": ("ir", "passes", "plan.dag", "plan.pipeline", "plan.schedule",
               "plan.targets", "recurrence", "spec"),
    "e_fragment": ("ir", "plan.targets"),
    "emit": (),
    "emit._c": ("device", "ir"),
    "emit.cuda": ("device", "emit._c", "ir"),
    "emit.hip": ("device", "emit._c", "ir"),
    "emit.__main__": ("device", "emit.cuda", "emit.hip", "kernel", "kernel_dq",
                      "kernel_jab", "spec"),
    "measured": ("asym", "device", "emit.cuda", "emit.hip", "ir", "kernel",
                 "kernel_k", "lower", "measured_store", "spec"),
    "measured_store": ("device", "kernel", "lower", "spec"),
    "asym": ("device", "ir", "kernel", "measured_store", "passes",
             "plan.batching", "plan.dag", "plan.sieve", "plan.targets", "spec"),
    "kernelrank": ("device", "kernel", "lower", "measured_store", "spec"),
    "batchreport": ("device", "measured_store"),
    "ridge": ("device", "measured_store"),
    "dispatch": ("batchreport", "device", "measured_store"),
    "crossread": ("batchreport", "device", "kernelrank", "measured_store",
                  "ridge", "spec"),
    # #165: PlanCost (plan.pipeline, plan.cost) against the store's rows;
    # kernelrank for its rank statistic, gate for GPU_BUSY_MAX.
    "planrank": ("device", "gate", "kernelrank", "measured_store", "plan.cost",
                 "plan.pipeline", "spec"),
    "ao": (),
    "ao.shell": (),
    "ao.hermite": (),
    "ao.pairs": ("ao.hermite", "ao.shell"),
    "ao.eri": ("ao.hermite", "ao.pairs", "ao.shell"),
    "ao.j": ("ao.hermite", "ao.pairs", "ao.shell"),
    # #197: K reuses ao.j's density contract and EPair, and ao.eri for its
    # third index path (`k_from_eris`).
    "ao.k": ("ao.eri", "ao.hermite", "ao.j", "ao.shell"),
    "ao.drive": ("ao.j", "ao.shell", "ao.staged", "interp", "ir", "kernel",
                 "kernel_dq", "kernel_jab", "launch", "plan.pipeline", "spec"),
    "ao.staged": ("ao.j", "ao.pairs", "ao.shell", "interp", "ir", "kernel",
                  "launch", "passes"),
    "launch": ("ao.j", "ao.pairs", "ao.shell", "ir", "kernel", "plan.pipeline",
               "spec"),
    # #204: the K plan beside J's -- the shared part from `launch`, the
    # records and the host's staged contraction from `ao`.
    "launch_k": ("ao.j", "ao.k", "ao.shell", "ir", "launch", "spec"),
    "kernel_jab": ("e_fragment", "ir", "plan.targets", "spec"),
    "kernel_dq": ("e_fragment", "ir", "plan.targets", "spec"),
    # #202: the partition, the plan's cost and the targets; nothing built.
    "kdup": ("plan.batching", "plan.pipeline", "plan.targets", "spec"),
    # #337: the plan reached through `kernel.builder_plan`, which strips the
    # kernel-body levers, so `plan.pipeline` is no longer imported here.
    # #367: the lever seams (`k_levers.seams`, the frozen defaults moved
    # there) and the registry that composes a spec's levers (`k_levers`).
    "kernel_k": ("ir", "k_levers", "k_levers.seams", "kernel",
                 "recurrence", "spec"),
    "levers": (),
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


def test_the_launch_plan_reaches_neither_reference_nor_a_launcher() -> None:
    # #138's acceptance, TRANSITIVELY: the plan's bound is injected, and the
    # plan is data -- no chain of imports from it reaches reference/, the
    # interpreter, a backend, or the adapter that runs kernels.
    # #204: the K plan beside it, the same; and it reaches no K builder --
    # `kernel_k` (#198) arrives injected as `build=`.
    graph = _graph()
    for root, extra in (("launch", ()), ("launch_k", ("kernel_k",))):
        seen: set[str] = set()
        todo = [root]
        while todo:
            for e in graph.get(todo.pop(), set()):
                if e not in seen:
                    seen.add(e)
                    todo.append(e)
        for banned in (_REF, "interp", "emit", "ao.drive", "ao.staged",
                       "dispatch", "measured", "measured_store", *extra):
            hit = sorted(e for e in seen if _under(e, banned))
            assert not hit, f"{root} reaches {hit}"


def test_nothing_measured_reaches_back_through_the_resolver() -> None:
    # #145's hazard, TRANSITIVELY: the resolver reads the store, so neither
    # the store nor the plan (its cost model) nor the launch plan may reach it
    # by any chain of imports -- the policy arrives injected (`resolve=`).
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

    for m in sorted(graph):
        if (_under(m, "plan") or m in ("measured_store", "launch")) and (
                "dispatch" in reach(m)):
            raise AssertionError(f"{m} reaches dispatch")


# #94: the store half knows no toolchain. The internal edges above keep it off
# the driver and `emit/`; these are the standard-library modules through which
# a module runs a compiler, finds one, or asks the box who it is -- the driver
# imports every one of them, and the store none.
TOOLCHAIN_STDLIB = frozenset({"subprocess", "shutil", "socket", "tempfile",
                              "concurrent", "os", "time"})


def _stdlib_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_the_measured_store_imports_no_toolchain() -> None:
    store = _stdlib_imports(ROOT / "measured_store.py")
    assert not store & TOOLCHAIN_STDLIB, (
        f"measured_store imports {sorted(store & TOOLCHAIN_STDLIB)}: running a "
        "compiler is the driver's (felizjueves.measured), not the store's (#94)")
    # ...and the list names what the driver really uses, so it is not vacuous
    assert TOOLCHAIN_STDLIB <= _stdlib_imports(ROOT / "measured.py")


def test_the_dq_builder_reaches_neither_its_witness_nor_a_machine() -> None:
    # #153, TRANSITIVELY: the pre-contraction kernel is built from IR, the
    # `E` fragment and the key alone. No chain of imports reaches the AO
    # layer it is checked against (`ao.j.dq_from_host`), the interpreter,
    # reference/, a backend or a card.
    graph = _graph()
    seen: set[str] = set()
    todo = ["kernel_dq"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    for banned in (_REF, "ao", "interp", "emit", "device"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"kernel_dq reaches {hit}"
    assert "kernel_dq" in graph["ao.drive"]  # the adapter is its one runner


# ---------------------------------------------------------------------------
# #233 (milestone 11): the one-electron assembly and the overlap matrix, both
# under the `ao` rule. `ao.one_electron` is the symmetric loop `S`, `T` and
# `V` share, from `ao.shell` alone; `ao.overlap` is `S` as the `t = 0` column
# of the block `ao.pairs` already dresses.
# ---------------------------------------------------------------------------

RULE_OF |= {
    "ao.one_electron": "ao",
    "ao.overlap": "ao",
}
EDGES |= {k: frozenset(v) for k, v in {
    "ao.one_electron": ("ao.shell",),
    "ao.overlap": ("ao.one_electron", "ao.pairs", "ao.shell"),
}.items()}


# ---------------------------------------------------------------------------
# #235 (milestone 11): the nuclear attraction `V`, under the `ao` rule. It is
# the point-charge limit of the J engine's `[p|q]` over the dressed block of
# `ao.pairs`; its `[r]^(0)` evaluator arrives injected (`ao.pairs.OneCentre`),
# so it reaches no `reference/`.
# ---------------------------------------------------------------------------

RULE_OF |= {
    "ao.potential": "ao",
}
EDGES |= {k: frozenset(v) for k, v in {
    "ao.potential": ("ao.hermite", "ao.one_electron", "ao.pairs", "ao.shell"),
}.items()}


# ---------------------------------------------------------------------------
# #234 (milestone 11): the kinetic matrix, under the `ao` rule. `ao.kinetic`
# is `T` by the fixed +/-2 ket shift on `ao.hermite`'s axis tables, dressed
# by `ao.pairs.dress` and assembled by `ao.one_electron`.
# ---------------------------------------------------------------------------

RULE_OF |= {
    "ao.kinetic": "ao",
}
EDGES |= {k: frozenset(v) for k, v in {
    "ao.kinetic": ("ao.hermite", "ao.one_electron", "ao.pairs", "ao.shell"),
}.items()}


# ---------------------------------------------------------------------------
# #237 (milestone 11): the overlap kernel, `Target.S`. A builder like
# `kernel_jab`: the key, IR and the `E` fragment's `t = 0` column, inlined.
# It is checked against `ao.overlap.overlap_block` in the interpreter, so
# neither may be its ancestor, nor `reference/`.
# ---------------------------------------------------------------------------

RULES |= {
    "s_builder": _rule(
        "kernel_s.py (#237): the overlap kernel, `Target.S` -- key -> IR, "
        "with the `E` fragment's `t = 0` column inlined (`e_fragment`) and "
        "`kernel_jab`'s scatter rule restated, not imported",
        ("spec", "ir", "e_fragment"),
        ao="`ao.overlap.overlap_block` is the witness the kernel is "
           "interpreted against, and `ao.pairs.dress` the prefactor it must "
           "not form: importing either shares ancestry with it (#20)",
        interp="the builder is checked by the interpreter, never built on it",
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
        reference=_GROUND_TRUTH),
}
RULE_OF |= {
    "kernel_s": "s_builder",
}
EDGES |= {k: frozenset(v) for k, v in {
    "kernel_s": ("e_fragment", "ir", "spec"),
}.items()}


def test_the_s_builder_reaches_neither_its_witness_nor_a_machine() -> None:
    # #237, TRANSITIVELY: no chain of imports from `kernel_s` reaches the AO
    # layer it is checked against, the interpreter, reference/, a backend or
    # a card.
    graph = _graph()
    seen: set[str] = set()
    todo = ["kernel_s"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert "e_fragment" in seen
    for banned in (_REF, "ao", "interp", "emit", "device"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"kernel_s reaches {hit}"


# ---------------------------------------------------------------------------
# #238 (milestone 11): the kinetic-energy kernel, `Target.T`. `kernel_s`'s
# shape: the key, IR and the `E` fragment's kinetic bracket (the axis tables
# at `lb + 2` and the fixed ket shift), inlined. It is checked against
# `ao.kinetic.kinetic_block` in the interpreter, so neither may be its
# ancestor, nor `reference/`.
# ---------------------------------------------------------------------------

RULES |= {
    "t_builder": _rule(
        "kernel_t.py (#238): the kinetic-energy kernel, `Target.T` -- key -> "
        "IR, with the `E` fragment's kinetic bracket inlined (`e_fragment`) "
        "and `kernel_jab`'s scatter rule restated, not imported",
        ("spec", "ir", "e_fragment"),
        ao="`ao.kinetic.kinetic_block` is the witness the kernel is "
           "interpreted against, and `ao.pairs.dress` the prefactor it must "
           "not form: importing either shares ancestry with it (#20)",
        interp="the builder is checked by the interpreter, never built on it",
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
        reference=_GROUND_TRUTH),
}
RULE_OF |= {
    "kernel_t": "t_builder",
}
EDGES |= {k: frozenset(v) for k, v in {
    "kernel_t": ("e_fragment", "ir", "spec"),
}.items()}


def test_the_t_builder_reaches_neither_its_witness_nor_a_machine() -> None:
    # #238, TRANSITIVELY: no chain of imports from `kernel_t` reaches the AO
    # layer it is checked against, the interpreter, reference/, a backend or
    # a card.
    graph = _graph()
    seen: set[str] = set()
    todo = ["kernel_t"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert "e_fragment" in seen
    for banned in (_REF, "ao", "interp", "emit", "device"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"kernel_t reaches {hit}"


# ---------------------------------------------------------------------------
# #239 (milestone 11): the nuclear-attraction kernel, `Target.V` -- the plan
# layer's first non-J consumer. A builder like `kernel_s`, plus the plan
# layer and `recurrence` (the J engine's `[t]^(0)` statements, imported, not
# rewritten). Checked against `ao.potential.potential_block` in the
# interpreter, so neither may be its ancestor, nor `reference/`.
# ---------------------------------------------------------------------------

RULES |= {
    "v_builder": _rule(
        "kernel_v.py (#239): the nuclear-attraction kernel, `Target.V` -- "
        "key -> IR, with the plan layer's DAG and schedule (`plan`), the J "
        "engine's prelude and recurrence statements (`recurrence`), the `E` "
        "block fragment inlined (`e_fragment`) and `kernel_jab`'s scatter "
        "rule restated, not imported",
        ("spec", "ir", "e_fragment", "plan", "recurrence"),
        ao="`ao.potential.potential_block` is the witness the kernel is "
           "interpreted against, and `ao.pairs.dress` the prefactor it must "
           "not form: importing either shares ancestry with it (#20)",
        interp="the builder is checked by the interpreter, never built on it",
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
        reference=_GROUND_TRUTH),
}
RULE_OF |= {
    "kernel_v": "v_builder",
}
EDGES |= {k: frozenset(v) for k, v in {
    "kernel_v": ("e_fragment", "ir", "plan.dag", "plan.pipeline",
                 "plan.schedule", "plan.targets", "recurrence", "spec"),
}.items()}


def test_the_v_builder_reaches_neither_its_witness_nor_a_machine() -> None:
    # #239, TRANSITIVELY: no chain of imports from `kernel_v` reaches the AO
    # layer it is checked against, the interpreter, reference/, a backend or
    # a card -- and it does reach the plan layer, through `recurrence`.
    graph = _graph()
    seen: set[str] = set()
    todo = ["kernel_v"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert {"e_fragment", "recurrence", "plan.pipeline"} <= seen
    for banned in (_REF, "ao", "interp", "emit", "device"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"kernel_v reaches {hit}"


# ---------------------------------------------------------------------------
# #241 (milestone 11): the one-electron launch, BESIDE `ClassLaunch` in
# `launch.py`. Its records come from the AO layer (`ao.pairs.pair_list` and
# `primitive_pairs`, `ao.j.epair`), and its walker reproduces the host's
# `S`, `T` and `V` from the plan with the host's own axis pieces
# (`ao.hermite.e_axis`, `ao.kinetic.kinetic_axis`), the host's placement
# (`ao.one_electron.matrix`) and nucleus check (`ao.potential`). Still
# under rule "launch": `ao` was already allowed, `one_centre` is injected,
# and no builder of S, T or V is imported -- the plan restates their keys
# (test/test_launch.py pins them equal), it does not build kernels.
# ---------------------------------------------------------------------------

EDGES["launch"] = EDGES["launch"] | frozenset(
    ("ao.hermite", "ao.kinetic", "ao.one_electron", "ao.potential"))


def test_the_one_electron_launch_builds_no_one_electron_kernel() -> None:
    # #241's acceptance ("no kernel is launched or interpreted in this
    # module"), TRANSITIVELY: nothing `launch` reaches is an S, T or V
    # builder, the interpreter, or reference/.
    graph = _graph()
    seen: set[str] = set()
    todo = ["launch"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert {"ao.one_electron", "ao.potential"} <= seen
    for banned in (_REF, "interp", "kernel_s", "kernel_t", "kernel_v"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"launch reaches {hit}"


# ---------------------------------------------------------------------------
# #242 (milestone 11): the generator CLI writes the one-electron ladder
# (`--one-e-l`), so the driver composes the three new builders with the
# backends, as it does `kernel_dq` and `kernel_jab`. Still rule "driver":
# it builds and emits; the builders' own rules keep them off `ao`.
# ---------------------------------------------------------------------------

_ONE_E_BUILDERS = ("kernel_s", "kernel_t", "kernel_v")
RULES["driver"] = Rule(RULES["driver"].why + ", and #242's one-electron "
                       "ladder through `kernel_s`, `kernel_t`, `kernel_v`",
                       RULES["driver"].may | frozenset(_ONE_E_BUILDERS),
                       RULES["driver"].must_not)
EDGES["emit.__main__"] = EDGES["emit.__main__"] | frozenset(_ONE_E_BUILDERS)


# ---------------------------------------------------------------------------
# #273 (milestone 12): engine #2, the HGP recurrence of `[Fock]` §3, as the
# package `hgp/`. The rule is PACKAGE-WIDE: every module under `hgp/` gets
# rule "hgp" here, so a new `hgp.*` module is held to it the moment it exists
# (it still needs its own EDGES entry, the per-PR edit), and the transitive
# test below walks every `hgp.*` module.
# ---------------------------------------------------------------------------

RULES |= {
    "hgp": _rule(
        "hgp/ (#273): engine #2, the Head-Gordon-Pople recurrence of "
        "[Fock] §3 -- the VRR as an (e, f, m) DAG (`hgp.vrr`), and its "
        "siblings. Kept off the MMD engine so that neither is bent into "
        "the other's shape (milestone 12, pin 11)",
        ("hgp", "spec"),
        plan="the plan layer keys on `total_am`, which indexes a [JEngine] "
             "row an HGP class does not have, and its (r, m) DAG is graded "
             "by m -- every edge raises m by one, which its order, sieve and "
             "recycling rest on -- while the HGP VRR has same-m edges (#273): "
             "importing it would bend one engine into the other's shape "
             "(milestone 12, pin 11)",
        reference=_GROUND_TRUTH),
}
RULE_OF |= {m: "hgp" for m in _modules() if _under(m, "hgp")}
EDGES |= {k: frozenset(v) for k, v in {
    "hgp": (),
    "hgp.vrr": (),
    "hgp.hrr": ("hgp.vrr",),  # #274: the class check, Cartesians, targets
    "hgp.cost": ("hgp.hrr", "hgp.vrr"),  # #276: FLOPs off both DAGs
}.items()}

# #279: `hgp.recurrence`, the bridge from `hgp/` to `ir.py` -- the HGP twin
# of the MMD `recurrence` (rule "builder"). It is the one `hgp.*` module that
# may reach the IR, so it gets its own rule; the package-wide prohibitions
# (plan, reference) are restated, and the transitive test below holds it too.
RULES |= {
    "hgp_builder": _rule(
        "hgp/recurrence.py (#279): the HGP VRR (eqs 25-26) and HRR (eq 28) "
        "as IR -- the statements off `hgp.vrr` / `hgp.hrr`'s DAGs, and "
        "standalone recurrence-only `Kernel`s keyed on a `Target.F` `Spec`",
        ("ir", "spec", "hgp"),
        plan=RULES["hgp"].must_not["plan"],
        reference=_GROUND_TRUTH + "; its pins compare against "
                  "reference/pq.py's fundamental and ao/eri.py",
        emit="the builder is vendor-neutral; a backend spells its output",
        interp="the interpreter is what checks the builder; it must not "
               "know HGP, and the builder must not lean on it (#23)"),
}
RULE_OF["hgp.recurrence"] = "hgp_builder"
EDGES["hgp.recurrence"] = frozenset(("hgp.hrr", "hgp.vrr", "ir", "spec"))


# #281: `ao.hgp_drive`, the HGP engine's ERI block INTERPRETED -- the
# `ao.drive` precedent (builder + interp + ao, producing numbers) for engine
# #2. It cannot live under `hgp/` (rule `hgp` admits only `hgp` and `spec`,
# and `hgp_builder` bans `interp`), so it is an adapter of its own: the
# MMD adapter's rule does not admit `hgp`, and widening it would let the
# MMD driver reach the engine it is meant to be independent of.
RULES |= {
    "hgp_adapter": _rule(
        "ao/hgp_drive.py (#281): `hgp.recurrence`'s VRR and HRR kernels run "
        "in the interpreter on contracted shell quartets -- the HGP twin of "
        "`ao.eri.eri_block`, same signature, for the cross-engine pin and "
        "for `ao.fock`'s injected `eri=` slot (#286)",
        ("spec", "ir", "interp", "hgp", "ao"),
        reference=_GROUND_TRUTH + "; Boys arrives by injection (`boys=`), "
                  "and the pin compares against ao/eri.py and libcint",
        plan=RULES["hgp"].must_not["plan"],
        emit="it produces numbers, not text: a backend's spelling is the "
             "kernel tier's to check",
        kernel="the MMD builder is the other engine: reaching it would make "
               "the cross-engine pin share ancestry with its witness",
        **{"ao.eri": "ao/eri.py is the witness this evaluator is pinned "
                     "against; importing it would compare an engine to "
                     "itself",
           "ao.drive": "the MMD adapter runs the other engine"}),
}
RULE_OF["ao.hgp_drive"] = "hgp_adapter"
EDGES["ao.hgp_drive"] = frozenset(("ao.shell", "hgp.hrr", "hgp.recurrence",
                                   "interp", "ir", "spec"))


def test_no_hgp_module_reaches_the_plan_or_the_reference() -> None:
    # Milestone 12, pin 11, TRANSITIVELY and package-wide: no chain of
    # imports from any module under hgp/ reaches plan/ or reference/.
    graph = _graph()
    roots = [m for m in graph if _under(m, "hgp")]
    assert "hgp.vrr" in roots
    for root in roots:
        assert RULE_OF[root] in ("hgp", "hgp_builder"), root
        seen: set[str] = set()
        todo = [root]
        while todo:
            for e in graph.get(todo.pop(), set()):
                if e not in seen:
                    seen.add(e)
                    todo.append(e)
        for banned in ("plan", _REF):
            hit = sorted(e for e in seen if _under(e, banned))
            assert not hit, f"{root} reaches {hit}"


# ---------------------------------------------------------------------------
# #275 (milestone 12): the two-electron Fock witness, under the `ao` rule.
# `ao.fock` sweeps every ordered quartet with the ERI block function
# INJECTED (`eri=`), so it imports no engine: only `ao.j`'s density contract
# and `ao.shell`'s basis.
# ---------------------------------------------------------------------------

RULE_OF |= {
    "ao.fock": "ao",
}
EDGES |= {k: frozenset(v) for k, v in {
    "ao.fock": ("ao.j", "ao.shell"),
}.items()}


# ---------------------------------------------------------------------------
# #285 (milestone 12): the fused Fock build's pair stream, `launch_f`, as a
# VALUE beside `launch` and `launch_k`: the contracted pairs, their pruned
# primitive pairs, the degree-uniform batches, eq. 29's magnitudes and the
# pre-launch screen. Its bound is injected (`bound=`), so it reaches no
# reference/; it builds no kernel and evaluates no ERI.
# ---------------------------------------------------------------------------

RULES |= {
    "f_launch": _rule(
        "launch_f.py (#285): the fused Fock build's pair stream as a VALUE "
        "-- contracted pairs, pruned primitive pairs, batches keyed by "
        "class, contraction degree and eq. 29 magnitude, and the batch "
        "pairs launched. Its records come from the AO layer, the shared "
        "bound check and stable sort from `launch`, the `Spec` it names "
        "from `spec`; the bound arrives injected (`bound=`)",
        ("spec", "ao", "launch"),
        reference=_GROUND_TRUTH + "; the bound is injected (`bound=`), as "
                  "ao/pairs.py takes its `one_centre=`",
        plan="the plan layer keys on `total_am`, which indexes a [JEngine] "
             "row an F class does not have, and refuses an F spec "
             "(milestone 12, pin 11)",
        emit="the plan is data: no text is emitted",
        dispatch="nothing measured may reach a launch plan by import (#145)",
        measured="a launch plan is data, never a toolchain's reading",
        measured_store="nothing measured may reach a launch plan by import "
                       "(#145)"),
}
RULE_OF |= {
    "launch_f": "f_launch",
}
EDGES |= {k: frozenset(v) for k, v in {
    "launch_f": ("ao.j", "ao.pairs", "ao.shell", "launch", "spec"),
}.items()}


def test_the_f_pair_stream_reaches_neither_reference_nor_a_machine() -> None:
    # #285, TRANSITIVELY: no chain of imports from `launch_f` reaches
    # reference/ (its bound is injected), a backend, the measured layer, the
    # dispatch policy, or the adapters that run kernels.
    graph = _graph()
    seen: set[str] = set()
    todo = ["launch_f"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert "ao.pairs" in seen
    for banned in (_REF, "interp", "emit", "ao.drive", "ao.staged",
                   "dispatch", "measured", "measured_store"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"launch_f reaches {hit}"


# ---------------------------------------------------------------------------
# #282 (milestone 12): the fused Fock kernel, `Target.F`, up to the ERIs --
# a builder beside `kernel_k`: the key, IR, and the HGP statements of
# `hgp.recurrence` over `hgp.vrr` / `hgp.hrr`'s DAGs. It is interpreted
# against a host path built from #279's kernels and against `ao/eri.py`, so
# neither the AO layer nor reference/ may be its ancestor, and it keeps off
# the plan layer as hgp/ does (milestone 12, pin 11).
# ---------------------------------------------------------------------------

RULES |= {
    "f_builder": _rule(
        "kernel_f.py (#282): the fused Fock kernel of [Fock] §3.2 up to the "
        "ERIs -- the K_CD x K_AB nest, the contracted boundary accumulators "
        "and the HRR after the loop, its statements from `hgp.recurrence` "
        "over `hgp.vrr` / `hgp.hrr`'s DAGs, keyed on a `Target.F` `Spec`",
        ("spec", "ir", "hgp"),
        plan=RULES["hgp"].must_not["plan"],
        ao="`ao/eri.py` is the integrals' witness and `launch_f`'s records "
           "come from the AO layer: the kernel takes packed host scalars "
           "(`kernel_f.pack`), and importing either shares ancestry (#20)",
        interp="the builder is checked by the interpreter, never built on it",
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
        reference=_GROUND_TRUTH),
}
RULE_OF["kernel_f"] = "f_builder"
EDGES["kernel_f"] = frozenset(
    ("hgp.hrr", "hgp.recurrence", "hgp.vrr", "ir", "spec"))


def test_the_f_builder_reaches_neither_its_witness_nor_a_machine() -> None:
    # #282, TRANSITIVELY: no chain of imports from `kernel_f` reaches the
    # plan layer, the AO layer, reference/, the interpreter, a backend or a
    # card -- nor `launch_f`, whose records it consumes only as packed data.
    graph = _graph()
    seen: set[str] = set()
    todo = ["kernel_f"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert "hgp.recurrence" in seen
    for banned in ("plan", _REF, "ao", "interp", "emit", "device",
                   "launch_f"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"kernel_f reaches {hit}"


# ---------------------------------------------------------------------------
# #286 (milestone 12): `walk_f`, the fused Fock build's pair stream WALKED --
# milestone 6's launch oracle for the fused kernel. It reads `launch_f`'s
# plan by duck type and must not import it (the walker independent of the
# plan builder, m6's discipline); it rebuilds the records from the AO
# layer's shells, builds and packs through `kernel_f`, and runs each launch
# through an INJECTED executor (`run=`), so it reaches no interpreter and
# no reference/. Nor may it reach the witnesses it is checked against.
# ---------------------------------------------------------------------------

RULES |= {
    "f_walk": _rule(
        "walk_f.py (#286): `launch_f`'s plan walked with `kernel_f`'s "
        "kernel -- the plan checked against the basis, the records rebuilt "
        "from the shells, every launch run through an injected executor "
        "(`run=`), `F` closed, and every unique quartet's fate recorded",
        ("spec", "ir", "ao", "kernel_f"),
        launch_f="the walker is independent of the plan builder (m6's "
                 "discipline, #286): it reads the plan by duck type, never "
                 "a second call into, or a type from, the module that made it",
        reference=_GROUND_TRUTH + "; the executor (and its Boys) arrives "
                  "injected (`run=`)",
        interp="the kernel runs through the injected `run=`; the walker "
               "evaluates nothing itself",
        plan=RULES["hgp"].must_not["plan"],
        emit="the walk interprets; no text is emitted",
        device="the walk runs no card",
        dispatch="nothing measured may reach a launch plan by import (#145)",
        measured="a walk is not a toolchain's reading",
        measured_store="nothing measured may reach a walk by import"),
}
RULE_OF["walk_f"] = "f_walk"
EDGES["walk_f"] = frozenset(("ao.j", "ao.shell", "ir", "kernel_f", "spec"))


def test_the_f_walker_reaches_neither_its_plan_builder_nor_its_witness() -> None:
    # #286, TRANSITIVELY: no chain of imports from `walk_f` reaches the plan
    # builder, reference/, the interpreter, a backend, the dispatch policy,
    # or the witnesses the walk is held to (`ao.fock`'s sweep, `ao.eri`,
    # `ao.hgp_drive`).
    graph = _graph()
    seen: set[str] = set()
    todo = ["walk_f"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert "kernel_f" in seen
    for banned in ("launch_f", "launch", _REF, "interp", "emit", "device",
                   "dispatch", "measured", "ao.fock", "ao.eri",
                   "ao.hgp_drive", "engine"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"walk_f reaches {hit}"


# ---------------------------------------------------------------------------
# #292 (milestone 12, G1): `engine`, the protocol both ERI engines satisfy,
# extracted last, and the rule that makes it stick: NOTHING ABOVE THE
# ENGINES MAY IMPORT AN ENGINE BY NAME -- this file's preference for injected
# callables (module docstring: `interp`'s `boys=`), made exact for the new
# boundary.
#
# WHAT "AN ENGINE" IS, by module: the entry points `engine` names and the
# recurrence machinery behind them (`ENGINES`). MMD is `plan/`, `recurrence`,
# `kernel` and `ao.eri`; HGP is `hgp/`, `kernel_f` and `ao.hgp_drive`. The
# modules that are one engine's private consumers (`launch`, `kernel_k`,
# `ao.drive`, `launch_f`, ...) are neither an engine nor above one: their
# own rules above hold them.
#
# WHAT "ABOVE" IS, precisely: (a) every module that imports `engine` -- a
# consumer of the protocol, computed from the tree, so a new one is held the
# moment it appears; and (b) the modules DECLARED engine-neutral here
# (`DECLARED_ABOVE`): today `ao.fock`, the Fock witness whose ERI arrives as
# `eri=` (#275). Such a module may not reach an engine module through any
# chain of imports except through `engine` itself. "By name" therefore
# means: by any import at all, direct or transitive, other than the protocol.
#
# Also tree-wide: no module but `engine` (and the engines' own modules)
# names BOTH engines -- one that does is above them by definition and must
# go through the protocol. Tests are outside `src/` and are witnesses: they
# name both engines on purpose, to compare them.
# ---------------------------------------------------------------------------

ENGINES: dict[str, tuple[str, ...]] = {
    "mmd": ("plan", "recurrence", "kernel", "ao.eri"),
    "hgp": ("hgp", "kernel_f", "ao.hgp_drive"),
}
DECLARED_ABOVE = ("ao.fock",)
_ENGINE_WHY = ("the protocol names the engines' ENTRY POINTS (`ao.eri`, "
               "`ao.hgp_drive`, `kernel`, `kernel_f`) and nothing behind "
               "them: reaching {} would make an engine's private machinery "
               "part of the shared surface (#292's push-back-down list)")

RULES |= {
    "engine": _rule(
        "engine.py (#292): the `Engine` protocol -- name, target, eri, "
        "eri_and_scale, kernel -- and its two implementations, `Mmd` over "
        "`ao.eri` + `kernel` and `Hgp` over `ao.hgp_drive` + `kernel_f`. "
        "The one module in src/ that names both engines; each engine's "
        "special function is injected at construction",
        ("spec", "ir", "ao.shell", "ao.eri", "ao.hgp_drive", "kernel",
         "kernel_f"),
        reference=_GROUND_TRUTH + "; `pq=` and `boys=` arrive injected, as "
                  "`interp` takes `boys=`",
        plan=_ENGINE_WHY.format("plan/ (MILD, the sieve, the partition, "
                                "the total_am row)"),
        hgp=_ENGINE_WHY.format("hgp/ (the VRR/HRR DAGs)"),
        recurrence=_ENGINE_WHY.format("the MMD recurrence"),
        interp="the protocol evaluates nothing itself; HGP's adapter runs "
               "the interpreter, and the protocol must not lean on it",
        emit="a `Kernel` is vendor-neutral; spelling it is a driver's job",
        device="the `Kernel` is arch-neutral; the arch enters at emission"),
}
RULE_OF["engine"] = "engine"
EDGES["engine"] = frozenset(("ao.eri", "ao.hgp_drive", "ao.shell", "ir",
                             "kernel", "kernel_f", "spec"))


def _engine_of(module: str) -> str | None:
    for name, prefixes in ENGINES.items():
        if any(_under(module, p) for p in prefixes):
            return name
    return None


def _above(graph: dict[str, set[str]]) -> set[str]:
    return ({m for m, es in graph.items() if "engine" in es}
            | set(DECLARED_ABOVE))


def engines_named_from_above(graph: dict[str, set[str]]) -> dict[str, list]:
    """For each module above the engines, the engine modules it reaches by
    any chain of imports that does not pass through `engine`."""
    out = {}
    for root in sorted(_above(graph)):
        seen: set[str] = set()
        todo = [root]
        while todo:
            for e in graph.get(todo.pop(), set()):
                if e not in seen and e != "engine":
                    seen.add(e)
                    todo.append(e)
        hit = sorted(e for e in seen if _engine_of(e))
        if hit:
            out[root] = hit
    return out


def test_nothing_above_the_engines_names_an_engine() -> None:
    graph = _graph()
    assert set(DECLARED_ABOVE) <= set(graph), "a declared module is gone"
    assert not any(_engine_of(m) for m in _above(graph)), (
        "a module is declared above the engines and is one of them")
    assert not (hit := engines_named_from_above(graph)), (
        f"modules above the engines that name one (#292): {hit}")


def test_only_the_protocol_names_both_engines() -> None:
    graph = _graph()
    both = sorted(m for m, es in graph.items()
                  if not _engine_of(m) and m != "engine"
                  and m not in GENERATORS
                  and len({_engine_of(e) for e in es} - {None}) == 2)
    assert not both, f"modules naming both engines outside `engine`: {both}"
    named = {_engine_of(e) for e in graph["engine"]} - {None}
    assert named == set(ENGINES), named  # the registry really holds both


def test_the_engine_rule_can_fail() -> None:
    # A module above (it imports `engine`) that also imports `hgp.vrr`, and
    # the declared `ao.fock` reaching `ao.eri` through two hops: both are
    # caught; the route through `engine` itself is not a breach.
    graph = {"engine": {"ao.eri", "kernel_f"}, "ao.eri": set(),
             "kernel_f": {"hgp.vrr"}, "hgp.vrr": set(),
             "consumer": {"engine", "hgp.vrr"}, "via": {"mid"},
             "mid": {"ao.eri"}, "ao.fock": {"via"}, "clean": {"engine"}}
    assert engines_named_from_above(graph) == {
        "ao.fock": ["ao.eri"], "consumer": ["hgp.vrr"]}


# ---------------------------------------------------------------------------
# #287 (milestone 12): the generator CLI writes the fused Fock kernel behind
# `--fock-class` / `--fock-l`, so `emit.__main__` composes `kernel_f` with the
# backends as it composes `kernel` -- and so names BOTH engines. It cannot go
# through the protocol instead: it writes MMD's split batch kernels
# (`kernel.build_pq_kernels`, the selector), which #292 pushed back DOWN as
# engine-private, so it would be a module above the engines that names one.
# The exemption is NARROW and stated as a test: a generator names each engine
# by its ENTRY POINT only (`kernel`, `kernel_f`), never the machinery behind
# it (`plan/`, `recurrence`, `hgp/`, the AO layer).
# ---------------------------------------------------------------------------

GENERATORS = {"emit.__main__": frozenset(("kernel", "kernel_f"))}
RULES["driver"] = Rule(RULES["driver"].why + ", and #287's fused Fock "
                       "kernel through `kernel_f`, by its entry point only",
                       RULES["driver"].may | frozenset(("kernel_f",)),
                       RULES["driver"].must_not)
EDGES["emit.__main__"] = EDGES["emit.__main__"] | frozenset(("kernel_f",))


def test_a_generator_names_the_engines_by_their_entry_points_only() -> None:
    graph = _graph()
    for m, entry in GENERATORS.items():
        named = {e for e in graph[m] if _engine_of(e)}
        assert named <= entry, (
            f"{m} names engine machinery behind an entry point: "
            f"{sorted(named - entry)}")
        assert {_engine_of(e) for e in entry} == set(ENGINES), entry


# ---------------------------------------------------------------------------
# #288 (milestone 12): the `--remeasure` driver compiles the fused Fock
# kernel's F cells (`--target f`), so `measured` composes `kernel_f` with the
# backends as it composes `kernel` -- the second generator naming both
# engines, under #287's same narrow exemption: by ENTRY POINT only
# (`kernel`, `kernel_f`), never `hgp/` or the plan behind them.
# ---------------------------------------------------------------------------

GENERATORS["measured"] = frozenset(("kernel", "kernel_f"))
EDGES["measured"] = EDGES["measured"] | frozenset(("kernel_f",))


# ---------------------------------------------------------------------------
# #367 (milestone 13, SK): the K kernel's lever seams. `k_levers.seams` holds
# `KHooks`, `KContext` and the frozen defaults `kernel_k`'s skeleton calls;
# `k_levers` is the registry a lever module plugs into. Both sit BELOW
# `kernel_k` (it imports them), so neither may reach it back, and they keep
# the `k_builder` rule's prohibitions: what they build is part of the K
# kernel. A K lever module (`k_levers/<name>.py`) is ruled `k_lever` too.
# ---------------------------------------------------------------------------

RULES |= {
    "k_lever": _rule(
        "k_levers/ (#367): the K kernel's lever seams (`seams`: the hooks, "
        "their context and the frozen defaults, statements built from the "
        "plan's targets and schedule, the recurrence's node names and the "
        "`E` fragments) and the registry composing a spec's K levers, one "
        "module per lever; since #345 also the batch kernel's atomic "
        "write (`seams.atomize`, the `passes.atomize` default)",
        ("spec", "plan", "ir", "recurrence", "e_fragment", "k_levers",
         "passes"),
        kernel_k="the skeleton imports the seams: a seam or a lever reaching "
                 "back into `kernel_k` is a cycle",
        ao=RULES["k_builder"].must_not["ao"],
        interp=RULES["k_builder"].must_not["interp"],
        emit=RULES["k_builder"].must_not["emit"],
        device=RULES["k_builder"].must_not["device"],
        reference=_GROUND_TRUTH),
}
RULE_OF |= {m: "k_lever" for m in _modules() if _under(m, "k_levers")}
EDGES |= {k: frozenset(v) for k, v in {
    "k_levers": ("k_levers.seams", "spec"),
    "k_levers.seams": ("e_fragment", "ir", "passes", "plan.schedule",
                       "plan.targets", "recurrence", "spec"),
}.items()}


def test_the_k_seams_reach_neither_the_skeleton_nor_a_machine() -> None:
    # #367, TRANSITIVELY: from the seams and the registry no chain of
    # imports reaches `kernel_k` (the cycle), the K witness in `ao/`, the
    # interpreter, reference/, a backend or a card.
    graph = _graph()
    for root in ("k_levers", "k_levers.seams"):
        seen: set[str] = set()
        todo = [root]
        while todo:
            for e in graph.get(todo.pop(), set()):
                if e not in seen:
                    seen.add(e)
                    todo.append(e)
        for banned in ("kernel_k", _REF, "ao", "interp", "emit", "device"):
            hit = sorted(e for e in seen if _under(e, banned))
            assert not hit, f"{root} reaches {hit}"


# ---------------------------------------------------------------------------
# #368 (milestone 13, SF): the fused F kernel's lever seams, K's (#367) shape.
# `f_levers.seams` holds `FHooks`, `FContext` and the frozen defaults
# `kernel_f`'s skeleton calls; `f_levers` is the registry an F lever module
# plugs into. Both sit BELOW `kernel_f` (it imports them), so neither may
# reach it back, and they keep the `f_builder` rule's prohibitions (no plan
# layer, pin 11): what they build is part of the F kernel. An F lever module
# (`f_levers/<name>.py`) is ruled `f_lever` too.
# ---------------------------------------------------------------------------

RULES |= {
    "f_lever": _rule(
        "f_levers/ (#368): the fused F kernel's lever seams (`seams`: the "
        "hooks, their context and the frozen defaults, statements built from "
        "`hgp.recurrence` over `hgp.vrr` / `hgp.hrr`'s DAGs) and the registry "
        "composing a spec's F levers, one module per lever",
        ("spec", "ir", "hgp", "f_levers"),
        kernel_f="the skeleton imports the seams: a seam or a lever reaching "
                 "back into `kernel_f` is a cycle",
        plan=RULES["f_builder"].must_not["plan"],
        ao=RULES["f_builder"].must_not["ao"],
        interp=RULES["f_builder"].must_not["interp"],
        emit=RULES["f_builder"].must_not["emit"],
        device=RULES["f_builder"].must_not["device"],
        reference=_GROUND_TRUTH),
}
RULE_OF |= {m: "f_lever" for m in _modules() if _under(m, "f_levers")}
EDGES |= {k: frozenset(v) for k, v in {
    "f_levers": ("f_levers.seams", "spec"),
    "f_levers.seams": ("hgp.hrr", "hgp.recurrence", "hgp.vrr", "ir", "spec"),
}.items()}
RULES["f_builder"] = Rule(RULES["f_builder"].why + "; its lever seams live "
                          "in `f_levers/` (#368)",
                          RULES["f_builder"].may | frozenset(("f_levers",)),
                          RULES["f_builder"].must_not)
EDGES["kernel_f"] = EDGES["kernel_f"] | frozenset(
    ("f_levers", "f_levers.seams"))
# #352: the monotone ket key's early break, an F lever module on the seams.
# Its key lives in `launch_f` and its dominating factor in `walk_f`, which
# restate its two names rather than import it (test_f_lever_ket_break.py).
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.ket_break",))
EDGES["f_levers.ket_break"] = frozenset(("f_levers.seams", "ir", "spec"))


def test_the_f_seams_reach_neither_the_skeleton_nor_a_machine() -> None:
    # #368, TRANSITIVELY: from the seams and the registry no chain of
    # imports reaches `kernel_f` (the cycle), the plan layer, the AO layer,
    # reference/, the interpreter, a backend, a card or `launch_f`.
    graph = _graph()
    for root in ("f_levers", "f_levers.seams"):
        seen: set[str] = set()
        todo = [root]
        while todo:
            for e in graph.get(todo.pop(), set()):
                if e not in seen:
                    seen.add(e)
                    todo.append(e)
        for banned in ("kernel_f", "plan", _REF, "ao", "interp", "emit",
                       "device", "launch_f"):
            hit = sorted(e for e in seen if _under(e, banned))
            assert not hit, f"{root} reaches {hit}"


# ---------------------------------------------------------------------------
# #344 (milestone 13): `j_levers/`, the J kernel-body levers' behaviour, one
# module per lever (`fused` first; #346's mixed precision follows). The
# vocabulary entry is `levers`'; the IR the lever builds is here, reached
# from `kernel._pq_kernel` by one hook. Builder-side, so the `builder` rule's
# bans hold: no reference/, no interpreter, no backend, no card.
# ---------------------------------------------------------------------------

RULES |= {
    "j_lever": _rule(
        "j_levers/ (#344): the [p|q] kernel's kernel-body levers, IR in and "
        "IR out, called by `kernel.py`'s hook when `Spec.kernel_body_levers` "
        "names one. It reads the recurrence's node names and the targets' "
        "type; the plan, the batch and the contraction arrive as arguments",
        ("ir", "recurrence", "plan.targets"),
        kernel="`kernel.py` calls the lever; the reverse is a cycle",
        levers="the builder reads the vocabulary through `spec`; a lever's "
               "behaviour does not re-read it",
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
        interp="the lever is checked by the interpreter, never built on it",
        reference=_GROUND_TRUTH),
}
RULE_OF |= {"j_levers": "j_lever", "j_levers.fused": "j_lever",
            "j_levers.mixed": "j_lever"}
EDGES |= {k: frozenset(v) for k, v in {
    "j_levers": (),
    "j_levers.fused": ("ir", "plan.targets", "recurrence"),
    "j_levers.mixed": ("ir", "recurrence"),
}.items()}
RULES["builder"] = Rule(RULES["builder"].why + "; its J kernel-body levers' "
                        "behaviour lives in `j_levers/` (#344)",
                        RULES["builder"].may | frozenset(("j_levers",)),
                        RULES["builder"].must_not)
EDGES["kernel"] = EDGES["kernel"] | frozenset(("j_levers.fused",
                                                "j_levers.mixed"))


# ---------------------------------------------------------------------------
# #341 (milestone 13, M1): the per-element split, the first K lever module.
# `k_levers` imports it to register it; it moves hooks only, so it reaches
# the seams, the IR (its `Note`) and `Spec`, and nothing below `k_lever`'s
# prohibitions.
# ---------------------------------------------------------------------------

EDGES["k_levers"] = EDGES["k_levers"] | frozenset(("k_levers.split",))
EDGES["k_levers.split"] = frozenset(("ir", "k_levers.seams", "spec"))

# #345 (milestone 13, M6): the coalesced K write-back, a K lever module that
# moves `write_back` only; the same reach as #341's.
EDGES["k_levers"] = EDGES["k_levers"] | frozenset(("k_levers.coalesced",))
EDGES["k_levers.coalesced"] = frozenset(("ir", "k_levers.seams", "spec"))

# #349 (milestone 13, M7): the register write-back for small K blocks. It
# moves `write_back` and `atomize` and falls back to (and reuses the atomize
# of) #345's `coalesced`, so it reaches that module too.
EDGES["k_levers"] = EDGES["k_levers"] | frozenset(("k_levers.shuffle",))
EDGES["k_levers.shuffle"] = frozenset(("ir", "k_levers.coalesced",
                                       "k_levers.seams", "spec"))


# #342 (milestone 13, M2): `E^ac` staged in shared memory, a K lever module.
# It moves `bind_e_ac` / `e_ac` only, inlining the `E` fragment it reads off
# the context (and checking it is write-once), so it reaches the seams, the
# IR, the `E` fragment and `Spec`.
EDGES["k_levers"] = EDGES["k_levers"] | frozenset(("k_levers.stage_eac",))
EDGES["k_levers.stage_eac"] = frozenset(("e_fragment", "ir", "k_levers.seams",
                                         "spec"))


# #350 (milestone 13, H1): the batched VRR, the first F lever module. It
# moves four F hooks, groups the boundary with `hgp.vrr`'s new (additive)
# functions and reads `hgp.hrr`'s DAG for its group-count rule's floor, so
# it reaches the seams, `hgp`, the IR and `Spec`, nothing below `f_lever`'s
# prohibitions.
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.batched_vrr",))
EDGES["f_levers.batched_vrr"] = frozenset(("f_levers.seams", "hgp.hrr",
                                           "hgp.recurrence", "hgp.vrr", "ir",
                                           "spec"))

# #355 (milestone 13, H12): mixed-precision F. It moves the `real` / `cast`
# precision hooks and wraps `quartet_scalars`, `vrr`, `hrr` and `eri`; it
# reaches the seams, the IR and `Spec` only (its F32 literal rounding is its
# own, not `j_levers.mixed`'s: `f_lever` does not admit `j_levers`).
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.mixed",))
EDGES["f_levers.mixed"] = frozenset(("f_levers.seams", "ir", "spec"))


# #356 (milestone 13, H2): the sliced HRR, an F lever module composed on
# #350's. It slices with `hgp.hrr`'s new (additive) functions and re-cuts
# `batched_vrr`'s groups at its floor, so it reaches that sibling lever, the
# seams, `hgp`, the IR and `Spec`, nothing below `f_lever`'s prohibitions.
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.sliced_hrr",))
EDGES["f_levers.sliced_hrr"] = frozenset(("f_levers.batched_vrr",
                                          "f_levers.seams", "hgp.hrr",
                                          "hgp.vrr", "ir", "spec"))

# #359 (milestone 13, H3): the interleaved digest per slice, an F lever
# module composed on #356's. It supplies #356's slice body and re-installs
# #350's nest at its floor, naming the HRR nodes it digests after, so it
# reaches both sibling levers, the seams, `hgp`, the IR and `Spec`.
EDGES["f_levers"] = EDGES["f_levers"] | frozenset((
    "f_levers.sliced_site_digest",))
EDGES["f_levers.sliced_site_digest"] = frozenset((
    "f_levers.batched_vrr", "f_levers.seams", "f_levers.sliced_hrr",
    "hgp.hrr", "hgp.recurrence", "hgp.vrr", "ir", "spec"))


# #357 (milestone 13, H6): the dedicated sub-DAG VRR schedule, an F lever
# module that wraps the `vrr` hook only. It composes on #350 (requires it,
# reads its name and group count), schedules with `hgp.vrr`'s new (additive)
# functions, and reaches the seams and `Spec`, nothing below `f_lever`'s
# prohibitions.
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.subdag_vrr",))
EDGES["f_levers.subdag_vrr"] = frozenset(("f_levers.batched_vrr",
                                          "f_levers.seams", "hgp.vrr",
                                          "spec"))


# #360 (milestone 13, H4): the per-bra-element Fock-buffer split, an F lever
# module composed on #356's slices (it wraps `sliced_hrr.contracted_with`),
# so it reaches that sibling lever, the seams, `hgp.vrr` (the Cartesian
# counts of its declared buffer sizes), the IR and `Spec`.
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.slim_fock",))
EDGES["f_levers.slim_fock"] = frozenset(("f_levers.seams",
                                          "f_levers.sliced_hrr", "hgp.vrr",
                                          "ir", "spec"))


# #361 (milestone 13, H5): slice-interior sub-slicing, an F lever module
# composed on #360's reduced buffers (it wraps `slim_fock.contracted_over`
# with a body of #356's `slice_body`s), cutting with `hgp.hrr`'s new
# (additive) `sub_slices`; it reaches those two sibling levers, the seams,
# `hgp`, the IR and `Spec`.
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.sub_slice",))
EDGES["f_levers.sub_slice"] = frozenset(("f_levers.seams",
                                          "f_levers.sliced_hrr",
                                          "f_levers.slim_fock", "hgp.hrr",
                                          "hgp.vrr", "ir", "spec"))


# ---------------------------------------------------------------------------
# #347 (milestone 13, M9): mixed-precision K. `k_levers.mixed` reuses
# `j_levers.mixed`'s F32 prelude and node narrowing, so the two engines'
# fp32 recurrence is one spelling: the `k_lever` rule admits `j_levers` (a
# sibling lever package, builder-side, under the same bans).
# ---------------------------------------------------------------------------

RULES["k_lever"] = Rule(RULES["k_lever"].why + "; a K lever may reuse a J "
                        "lever's IR (`j_levers`, #347's fp32 recurrence)",
                        RULES["k_lever"].may | frozenset(("j_levers",)),
                        RULES["k_lever"].must_not)
EDGES["k_levers"] = EDGES["k_levers"] | frozenset(("k_levers.mixed",))
EDGES["k_levers.mixed"] = frozenset(("e_fragment", "ir", "j_levers.mixed",
                                     "k_levers.seams", "recurrence", "spec"))


# #353 (milestone 13, H10): struct-typed pair-record reads, an F lever
# module. It moves the `field` hook only (its records are `ir.Record`s, its
# reads `ir.Member`s), so it reaches the seams, the IR and `Spec`, nothing
# below `f_lever`'s prohibitions; the spelling of the view is `emit/_c.py`'s.
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.struct_reads",))
EDGES["f_levers.struct_reads"] = frozenset(("f_levers.seams", "ir", "spec"))


# ---------------------------------------------------------------------------
# #339 (milestone 13): the lever ladder as a registry. `lever_ladder` is a
# leaf beside `levers` (data only: the families and the unit name; no
# builder), read by both drivers; the generator CLI now also composes
# `kernel_k` (the K lever ladder) and names `ir.Kernel`; and the store reads
# the kernel-body vocabulary to know a KERNEL arm (`measured_store.kind_of`).
# ---------------------------------------------------------------------------

RULES |= {
    "lever_ladder": _rule(
        "lever_ladder.py (#339): which flags a kernel-body lever gets on the "
        "generator CLI -- the families (J, K, F), their flags and the lever "
        "unit's name -- as data read off `levers.VOCABULARY`. A leaf beside "
        "`levers`, so both drivers read one table",
        ("levers",),
        kernel="the registry names no builder: the CLI maps each family to "
               "its builder, so the table stays data",
        emit="a registry spells no text",
        measured_store="the store reads the vocabulary itself; the ladder "
                       "is the CLI's"),
}
RULE_OF["lever_ladder"] = "lever_ladder"
EDGES["lever_ladder"] = frozenset(("levers",))
RULES["driver"] = Rule(RULES["driver"].why + ", and #339's lever ladders "
                       "through `lever_ladder`",
                       RULES["driver"].may | frozenset(("lever_ladder",)),
                       RULES["driver"].must_not)
EDGES["emit.__main__"] = EDGES["emit.__main__"] | frozenset(
    ("ir", "kernel_k", "lever_ladder"))
EDGES["measured"] = EDGES["measured"] | frozenset(("lever_ladder",))
RULES["store"] = Rule(RULES["store"].why + "; and the kernel-body lever "
                      "vocabulary (`levers`, a leaf), whose entries are its "
                      "KERNEL arms (#339)",
                      RULES["store"].may | frozenset(("levers",)),
                      RULES["store"].must_not)
EDGES["measured_store"] = EDGES["measured_store"] | frozenset(("levers",))


# ---------------------------------------------------------------------------
# #343 (milestone 13, M3): interleaved I_p, the warp arm. It replaces the
# contraction region, so beside the seams and the IR it reads the
# recurrence's node names (`recurrence.node_name`), as the seams do.
# ---------------------------------------------------------------------------

EDGES["k_levers"] = EDGES["k_levers"] | frozenset(("k_levers.interleave",))
EDGES["k_levers.interleave"] = frozenset(("ir", "k_levers.seams",
                                          "recurrence", "spec"))

# #348 (milestone 13, M4): the split arm is the composition interleave+split,
# not a lever of its own. `interleave` reads `split.NAME` to name the arm in
# the element kernels' region (`interleave.arm`); `split` imports nothing of
# `interleave`, so the edge is one way.
EDGES["k_levers.interleave"] = EDGES["k_levers.interleave"] | frozenset(
    ("k_levers.split",))


# ---------------------------------------------------------------------------
# The Schwarz matrix sqrt((mn|mn)): the host reference and staging
# (`ao.schwarz`, under the `ao` rule, `pq` injected) and the kernel builder
# (`kernel_schwarz`, `kernel_v`'s rule restated: the plan layer's DAG for the
# J quartet (la lb|la lb), the recurrence statements, the `E` fragment). The
# generator CLI writes its ladder (`--schwarz-l`), so the driver composes it.
# ---------------------------------------------------------------------------

RULES |= {
    "schwarz_builder": _rule(
        "kernel_schwarz.py: the Schwarz kernel, `Target.SCHWARZ` -- key -> "
        "IR, with the plan layer's DAG and schedule for the J quartet "
        "(la lb|la lb) (`plan`), the J engine's prelude and recurrence "
        "statements (`recurrence`), the `E` block fragment inlined twice "
        "(`e_fragment`) and `kernel_jab`'s scatter rule restated",
        ("spec", "ir", "e_fragment", "plan", "recurrence"),
        ao="`ao.schwarz` (through `ao.eri`) is the witness the kernel is "
           "interpreted against: importing it shares ancestry with it (#20)",
        interp="the builder is checked by the interpreter, never built on it",
        emit="nothing in the produced `Kernel` may be CUDA- or HIP-specific",
        device="the `Kernel` is arch-neutral; the arch enters at emission",
        reference=_GROUND_TRUTH),
}
RULE_OF |= {
    "ao.schwarz": "ao",
    "kernel_schwarz": "schwarz_builder",
}
EDGES |= {k: frozenset(v) for k, v in {
    "ao.schwarz": ("ao.eri", "ao.j", "ao.shell"),
    "kernel_schwarz": ("e_fragment", "ir", "plan.dag", "plan.pipeline",
                       "plan.schedule", "plan.targets", "recurrence", "spec"),
}.items()}
RULES["driver"] = Rule(RULES["driver"].why + ", and the Schwarz ladder "
                       "through `kernel_schwarz`",
                       RULES["driver"].may | frozenset(("kernel_schwarz",)),
                       RULES["driver"].must_not)
EDGES["emit.__main__"] = EDGES["emit.__main__"] | frozenset(
    ("kernel_schwarz",))


def test_the_schwarz_builder_reaches_neither_its_witness_nor_a_machine(
        ) -> None:
    graph = _graph()
    seen: set[str] = set()
    todo = ["kernel_schwarz"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert {"e_fragment", "recurrence", "plan.pipeline"} <= seen
    for banned in (_REF, "ao", "interp", "emit", "device"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"kernel_schwarz reaches {hit}"


# ---------------------------------------------------------------------------
# #351 (milestone 13, H7): the block-level f_ab reduction, an F lever
# module. `f_levers` imports it to register it; it moves `shared_decl` and
# `flush_bra` only, so it reaches the seams, the IR (`Barrier`,
# `WarpReduce`), `hgp.vrr` (the bra block's Cartesian counts, for the
# declared footprint) and `Spec`.
# ---------------------------------------------------------------------------

EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.block_fab",))
EDGES["f_levers.block_fab"] = frozenset(("f_levers.seams", "hgp.vrr", "ir",
                                         "spec"))


# ---------------------------------------------------------------------------
# #358 (milestone 13, H8): the atomic-free write-back. The lever module
# (`f_levers.scatter`) composes block-fab's reduction and moves the two
# flushes and the params; the scatter kernel and its host map
# (`kernel_f_scatter`) are a second builder beside `kernel_f`, under the
# same bans (no plan layer, no AO layer, no reference/, no interpreter, no
# backend, no card), reading the lever's slot layout. `walk_f` launches the
# scatter after a `scatter` launch, so it reaches the scatter builder; the
# plan builder restates the lever's name and declares the footprint itself.
# ---------------------------------------------------------------------------

RULES |= {
    "f_scatter": _rule(
        "kernel_f_scatter.py (#358): the atomic-free F write-back's scatter "
        "kernel (class-agnostic: one thread per element of F, summing its "
        "slots of the launch's buffer W in a fixed order) and its host map, "
        "the slot layout read off `f_levers.scatter`",
        ("spec", "ir", "hgp", "f_levers"),
        plan=RULES["f_builder"].must_not["plan"],
        ao=RULES["f_builder"].must_not["ao"],
        interp=RULES["f_builder"].must_not["interp"],
        emit=RULES["f_builder"].must_not["emit"],
        device=RULES["f_builder"].must_not["device"],
        launch_f="the map is built from packed AO offsets, as `kernel_f.pack` "
                 "takes records; the plan declares its footprint itself",
        reference=_GROUND_TRUTH),
}
RULE_OF["kernel_f_scatter"] = "f_scatter"
EDGES["kernel_f_scatter"] = frozenset(("f_levers.scatter", "f_levers.seams",
                                       "hgp.vrr", "ir"))
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.scatter",))
EDGES["f_levers.scatter"] = frozenset(("f_levers.block_fab", "f_levers.seams",
                                       "hgp.vrr", "ir", "spec"))
RULES["f_walk"] = Rule(RULES["f_walk"].why + "; a `scatter` launch's "
                       "scatter kernel and map through `kernel_f_scatter` "
                       "(#358)",
                       RULES["f_walk"].may | frozenset(("kernel_f_scatter",)),
                       RULES["f_walk"].must_not)
EDGES["walk_f"] = EDGES["walk_f"] | frozenset(("kernel_f_scatter",))


def test_the_scatter_builder_reaches_neither_its_witness_nor_a_machine(
        ) -> None:
    # #358, TRANSITIVELY: from the scatter builder no chain of imports
    # reaches `kernel_f`, the plan layer, the AO layer, reference/, the
    # interpreter, a backend, a card or `launch_f`.
    graph = _graph()
    seen: set[str] = set()
    todo = ["kernel_f_scatter"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert "f_levers.scatter" in seen
    for banned in ("kernel_f", "plan", _REF, "ao", "interp", "emit",
                   "device", "launch_f"):
        hit = sorted(e for e in seen if _under(e, banned))
        assert not hit, f"kernel_f_scatter reaches {hit}"


# #354 (milestone 13, H11): the ETR as a second VRR. `hgp.etr` is a sibling
# DAG under rule "hgp" (package-wide; it reads `hgp.vrr` only), and
# `hgp.cost` counts it with new functions (`etr_cost`), so it now reaches
# `hgp.etr` too. `f_levers.etr` is the F lever: it moves `vrr`, `ket_prim`
# and `quartet_scalars`, arbitrating per class on `hgp.cost` and the live
# model in `hgp.vrr`, so it reaches the seams, `hgp`, the IR and `Spec`,
# nothing below `f_lever`'s prohibitions.
# ---------------------------------------------------------------------------

EDGES["hgp.etr"] = frozenset(("hgp.vrr",))
EDGES["hgp.cost"] = EDGES["hgp.cost"] | frozenset(("hgp.etr",))
EDGES["f_levers"] = EDGES["f_levers"] | frozenset(("f_levers.etr",))
EDGES["f_levers.etr"] = frozenset(("f_levers.seams", "hgp.cost", "hgp.etr",
                                   "hgp.recurrence", "hgp.vrr", "ir", "spec"))
