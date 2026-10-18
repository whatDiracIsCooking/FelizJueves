"""Lowering: liveness over a scheduled `Kernel`, opt-in slot recycling, and
`KernelCost`.

A function of IR, never of text. Three things live here:

- `liveness(kernel)`: for every VALUE -- a per-thread scalar `Decl`
  (`Space.LOCAL`, `shape == ()`) -- the closed interval of statement positions
  over which it is live, and how often it is read.
- `recycle(kernel)`: reuse dead slots, so named scalars become elements of one
  indexed array. **OPT-IN, and it stays opt-in.** `lower()` leaves a kernel
  alone unless asked.
- `KernelCost`, via `kernel_cost(kernel, registers=...)`: an engineering model
  of the IR, with a register budget that is an argument and has no default.

POSITIONS are indices into `ir.walk(kernel.body)`: every statement, nested
ones included, in source order. A value's interval opens at its `Decl` and
closes at its last reference (read or write). A reference inside a `For` the
value was declared OUTSIDE of keeps it live to the end of that loop -- the
outermost such loop -- since the next iteration may read it again; a loop's
bounds count as references inside the loop, because `stop` and `step` are
re-evaluated every iteration. Liveness is keyed by name, so it requires every
value's `Decl` name to be unique within the kernel and raises otherwise.

THE LOOP MODEL (#284), stated because the IR carries no trip count and this
module infers none:

- LIVENESS ACROSS A LOOP BOUNDARY is the rule above, unchanged: a value
  declared before a `For` and referenced inside it is live from its `Decl`
  through the loop's last statement. A value WRITTEN inside such a loop is
  a LOOP-CARRIED accumulator (`loop_carried`): the next iteration reads what
  this one wrote. All of one loop's carried values are live together at
  that loop's last position (each interval opens before the loop and closes
  at or after its end), so `peak_live >= KernelCost.carried`, always, by
  construction rather than by measurement. The fused HGP kernel (#282)
  declares its `|boundary|` contracted accumulators before the `K_CD`
  primitive loop and updates every one inside it, so its `peak_live` is at
  least `|boundary|` -- 961 at `(dd|dd)` -- whatever the VRR does.
- ITS LIMITS. A loop of trip count 0 or 1 is costed exactly as one of a
  thousand; a loop-INVARIANT value (read inside, never written) is held live
  across the loop just as a carried one is, but is not counted in
  `carried`; liveness is not path-sensitive, so both arms of an `If` count;
  and a value declared INSIDE a loop is per iteration (its interval never
  wraps, even where the backend would keep it in a register across trips).
- TIERED COUNTS. `KernelCost.nest` attributes every computing statement to
  the enclosing `For` loops by variable name, outermost first; the counts
  sum to `n_instructions`, which stays static. Weighting a nest by its trip
  counts is the caller's, because only the caller knows them: `tiered` maps
  the nest onto `K^4 / K^2 / K^0` given which loops are the per-quartet
  loop and which are the primitive-pair loops -- the tiers of `hgp.cost`
  (#276), so the two are commensurable per class without either reading the
  other.

INTERVALS ARE CLOSED, ON PURPOSE. A value read for the last time at position p
is still live at p, so the statement at p never writes into the slot of one of
its own inputs. `dst = PQ_x * s[3] + 2 * s[5]` with `dst` allocated to `s[3]`
is correct C (the right-hand side is evaluated first), but it is the one way a
slot allocator can be wrong and still look right -- and the moment a backend
splits a statement, or a pass turns an `Assign` into an accumulate, it is
wrong. Closed intervals rule it out by construction. Values are an interval
graph, so the greedy allocator below uses exactly `peak_live` slots.

WHY RECYCLING IS NOT THE DEFAULT. A named `const double` per intermediate is
what nvcc and hipcc want to see for a kernel of this size: each does its own
register allocation, and an indexed local array can defeat it (an array
indexed dynamically lives in local memory; one indexed statically is only
promoted if the compiler chooses to). The unrolled high-L recurrence kernels
this engine emits are the case where recycling is most likely to *hurt*. It is
a lever, to be switched on when a measurement says so.

THE LEVER'S NAME is `RECYCLE` (`"recycle"`), carried in `Spec.levers` so the
two arms of an A/B are two keys -- two goldens, two `Measured` rows -- rather
than one key and a note. Lowering is the stage AFTER the builder: the driver
(`measured.cell_kernels`) builds the kernels of the Spec without its `LEVERS`
-- the plan layer and the builder never see them -- and then calls
`lower(..., recycle_slots=True)` on each finished kernel, atomized or not.
#55 measured it on both cards: `measured_store.lever_ab` is the verdict's
definition (the store's docstring, A LEVER A/B; since #94 the store and the
`--remeasure` driver are two modules), and test/test_measured.py pins what it
said about the rows.
WHAT #55 FOUND: neither hurt nor help. nvcc 13.0 and hipcc / ROCm 7.2.4 emit
identical machine code for the two arms in every batch kernel of the
subgrid -- both promote the statically indexed slot array back to scalars --
so the lever is a no-op on this box's toolchains and stays opt-in.

A LOWERING LEVER IS NOT A KERNEL-BODY LEVER (#337). The builders' own levers
live in `levers.VOCABULARY`, which reserves `recycle` as lowering's
(`levers.OTHER_LAYERS`, pinned to `LEVERS` by test/test_lower.py), so a
builder never mistakes this one for its own and still hands it to the plan,
which refuses it, exactly as before.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Expr,
    Factor,
    For,
    If,
    Index,
    Kernel,
    Lit,
    Scalar,
    Space,
    Stmt,
    Term,
    Var,
    WarpReduce,
    Weighted,
)

__all__ = [
    "LEVERS",
    "RECYCLE",
    "KernelCost",
    "Liveness",
    "TIERS",
    "kernel_cost",
    "liveness",
    "loop_carried",
    "lower",
    "recycle",
    "spilled",
    "tiered",
]

RECYCLE = "recycle"
"""The `Spec.levers` name of slot recycling (#55): `lower(recycle_slots=True)`."""

LEVERS = frozenset({RECYCLE})
"""Every `Spec.levers` name this module applies: what a driver strips before
building and hands to `lower` after."""


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


def _is_value(s: Stmt) -> bool:
    return isinstance(s, Decl) and s.space is Space.LOCAL and not s.shape


def _reads(e: Expr | None) -> Iterator[str]:
    """Every name an expression reads: `Var`s, `Index` subscripts, `Term`
    sources, and the scalar a non-unit `Factor` binds."""
    if e is None:
        return
    if isinstance(e, Var):
        yield e.name
    elif isinstance(e, Index):
        for i in e.at:
            yield from _reads(i)
    elif isinstance(e, Call):
        for a in e.args:
            yield from _reads(a)
    elif isinstance(e, Weighted):
        for t in e.terms:
            if t.g.name_in_kernel is not None:
                yield t.g.name_in_kernel
            yield from _reads(t.src)


def _lvalue_reads(dst: Var | Index) -> Iterator[str]:
    """Names an lvalue READS (an `Index`'s subscripts; a `Var` reads none)."""
    if isinstance(dst, Index):
        for i in dst.at:
            yield from _reads(i)


def _refs(s: Stmt) -> tuple[list[str], list[str]]:
    """(reads, writes) of one statement, nested bodies excluded."""
    if isinstance(s, Decl):
        return [n for d in s.shape for n in _reads(d)] + list(_reads(s.init)), []
    if isinstance(s, (Assign, WarpReduce, AtomicAdd)):
        reads = list(_lvalue_reads(s.dst)) + list(_reads(s.value))
        writes = [s.dst.name] if isinstance(s.dst, Var) else []
        return reads, writes
    if isinstance(s, For):
        return [n for e in (s.start, s.stop, s.step) for n in _reads(e)], []
    if isinstance(s, If):
        return list(_reads(s.cond)), []
    return [], []


def _positions(
    body: tuple[Stmt, ...],
) -> tuple[list[Stmt], list[tuple[int, ...]], dict[int, int]]:
    """`walk` order, each statement's enclosing loops (by the For's position,
    outermost first; a For's own header counts as inside it), and each loop's
    last position."""
    flat: list[Stmt] = []
    loops: list[tuple[int, ...]] = []
    end: dict[int, int] = {}

    def visit(stmts: tuple[Stmt, ...], enclosing: tuple[int, ...]) -> None:
        for s in stmts:
            pos = len(flat)
            flat.append(s)
            if isinstance(s, For):
                loops.append((*enclosing, pos))
                visit(s.body, (*enclosing, pos))
                end[pos] = len(flat) - 1
            else:
                loops.append(enclosing)
                if isinstance(s, If):
                    visit(s.then, enclosing)
                    visit(s.orelse, enclosing)

    visit(body, ())
    return flat, loops, end


@dataclass(frozen=True)
class Liveness:
    """Every value's live interval `[start, end]` (closed, positions into
    `stmts`), and how many times it is read."""

    stmts: tuple[Stmt, ...]  # ir.walk order
    start: Mapping[str, int]  # the Decl
    end: Mapping[str, int]  # the last reference, extended across loops
    reads: Mapping[str, int]

    def live_at(self, pos: int) -> frozenset[str]:
        return frozenset(v for v, s in self.start.items() if s <= pos <= self.end[v])

    @property
    def peak(self) -> int:
        """The most values live at once."""
        if not self.start:
            return 0
        delta = [0] * (len(self.stmts) + 1)
        for v, s in self.start.items():
            delta[s] += 1
            delta[self.end[v] + 1] -= 1
        best = run = 0
        for d in delta:
            run += d
            best = max(best, run)
        return best


def liveness(kernel: Kernel) -> Liveness:
    """Last use, and so live interval, per value of `kernel`."""
    flat, loops, loop_end = _positions(kernel.body)
    start: dict[str, int] = {}
    for pos, s in enumerate(flat):
        if _is_value(s):
            if s.name in start:
                raise ValueError(
                    f"liveness needs unique value names; {s.name!r} is declared twice"
                )
            start[s.name] = pos
    end = dict(start)
    reads = dict.fromkeys(start, 0)
    for pos, s in enumerate(flat):
        r, w = _refs(s)
        for name, is_read in [(n, True) for n in r] + [(n, False) for n in w]:
            if name not in start:
                continue  # a Param, a loop variable, or an array
            d = start[name]
            if pos < d or (pos == d and is_read):
                raise ValueError(f"{name!r} is referenced before it is declared")
            reads[name] += is_read
            last = pos
            outer = next((lp for lp in loops[pos] if lp > d), None)
            if outer is not None:
                last = loop_end[outer]
            end[name] = max(end[name], last)
    return Liveness(
        tuple(flat),
        MappingProxyType(start),
        MappingProxyType(end),
        MappingProxyType(reads),
    )


def loop_carried(kernel: Kernel) -> Mapping[int, frozenset[str]]:
    """Per `For` (by its position in `ir.walk` order), the values declared
    BEFORE it and written inside it, nested loops included: its loop-carried
    accumulators (module docstring, THE LOOP MODEL). A loop that carries
    nothing is absent."""
    flat, loops, _ = _positions(kernel.body)
    start = {s.name: pos for pos, s in enumerate(flat) if _is_value(s)}
    out: dict[int, set[str]] = {}
    for pos, s in enumerate(flat):
        for name in _refs(s)[1]:
            d = start.get(name)
            if d is None:
                continue
            for lp in loops[pos]:
                if lp > d and lp != pos:
                    out.setdefault(lp, set()).add(name)
    return MappingProxyType({lp: frozenset(v) for lp, v in sorted(out.items())})


# ---------------------------------------------------------------------------
# Recycling (opt-in)
# ---------------------------------------------------------------------------


def _allocate(live: Liveness, names: list[str]) -> dict[str, int]:
    """Greedy interval colouring by start: a slot is free again only AFTER its
    value's last position, so a writer never lands on one of its inputs."""
    slot: dict[str, int] = {}
    active: list[str] = []
    free: list[int] = []
    n_slots = 0
    for v in sorted(names, key=lambda v: (live.start[v], v)):
        s = live.start[v]
        for a in [a for a in active if live.end[a] < s]:
            active.remove(a)
            free.append(slot[a])
        if free:
            free.sort()
            slot[v] = free.pop(0)
        else:
            slot[v] = n_slots
            n_slots += 1
        active.append(v)
    return slot


def recycle(kernel: Kernel, *, array: str = "slot") -> Kernel:
    """Reuse dead slots: named scalars become an indexed array.

    OPT-IN; see the module docstring for why this is not the default. Every
    per-thread fp64 scalar `Decl` becomes an element of one LOCAL fp64 array
    named `array` (prefixed with underscores until it collides with nothing),
    declared first; two values share an element only when their closed live
    intervals are disjoint. A `Decl` with an `init` becomes an `Assign` to its
    element, one without is dropped, and every reference is rewritten.
    Excluded, and left named: I32 values, and a scalar a `Factor` binds by
    name (`PQ_x`...), which must stay a `Param` or `Decl` of that name.

    Cost a kernel BEFORE recycling it: the slot array is not a value, so
    `liveness` of the result no longer sees what it holds.
    """
    live = liveness(kernel)
    factor_names = {f.name_in_kernel for f in Factor} - {None}
    names = [
        s.name
        for s in live.stmts
        if _is_value(s) and s.type is Scalar.F64 and s.name not in factor_names
    ]
    if not names:
        return kernel
    taken = {p.name for p in kernel.params} | {
        s.name for s in live.stmts if isinstance(s, Decl)
    } | {s.var for s in live.stmts if isinstance(s, For)}
    while array in taken:
        array = "_" + array
    slot = _allocate(live, names)

    def ref(name: str) -> Index:
        return Index(array, (Lit(slot[name]),))

    def lv(dst: Var | Index) -> Var | Index:
        if isinstance(dst, Var):
            return ref(dst.name) if dst.name in slot else dst
        return Index(dst.array, tuple(ex(i) for i in dst.at))

    def ex(e: Expr) -> Expr:
        if isinstance(e, Var):
            return ref(e.name) if e.name in slot else e
        if isinstance(e, Index):
            return Index(e.array, tuple(ex(i) for i in e.at))
        if isinstance(e, Call):
            return Call(e.fn, tuple(ex(a) for a in e.args))
        if isinstance(e, Weighted):
            return Weighted(tuple(Term(t.c, t.g, lv(t.src)) for t in e.terms))
        return e

    def st(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for s in stmts:
            if isinstance(s, Decl) and s.name in slot and _is_value(s):
                if s.init is not None:
                    out.append(Assign(ref(s.name), ex(s.init)))
            elif isinstance(s, Decl):
                init = None if s.init is None else ex(s.init)
                out.append(
                    Decl(s.name, s.type, tuple(ex(d) for d in s.shape), s.space, init)
                )
            elif isinstance(s, Assign):
                out.append(Assign(lv(s.dst), ex(s.value)))
            elif isinstance(s, WarpReduce):
                out.append(WarpReduce(lv(s.dst), ex(s.value)))
            elif isinstance(s, AtomicAdd):
                out.append(AtomicAdd(lv(s.dst), ex(s.value)))
            elif isinstance(s, For):
                out.append(
                    For(s.var, ex(s.start), ex(s.stop), ex(s.step), st(s.body))
                )
            elif isinstance(s, If):
                out.append(If(ex(s.cond), st(s.then), st(s.orelse)))
            else:
                out.append(s)  # Break, Note, Raw: nothing to rewrite
        return tuple(out)

    n_slots = max(slot.values()) + 1
    decl = Decl(array, Scalar.F64, (Lit(n_slots),), Space.LOCAL)
    return Kernel(kernel.name, kernel.params, (decl, *st(kernel.body)))


def lower(kernel: Kernel, *, recycle_slots: bool = False) -> Kernel:
    """The lowering entry point. Recycling is OFF unless asked for."""
    return recycle(kernel) if recycle_slots else kernel


# ---------------------------------------------------------------------------
# KernelCost
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelCost:
    """An engineering model of a kernel, read from its IR.

    **This is not `PlanCost`, and the two may never merge.** `PlanCost`
    (plan/cost.py) reads the DAG and is a calibrated instrument: it is what
    the literature floors of `[JEngine]` Tables 1-3 are measured with.
    `KernelCost` reads the IR and is a model, free to change whenever a better
    one turns up. **Nothing in `KernelCost` may feed a literature floor** --
    otherwise an improvement to a register model silently moves a
    published-table ratchet. test/test_lower.py checks that nothing under
    plan/ or literature/ imports this module.

    - `peak_live`: the most values live at once, from `liveness`.
    - `traffic`: spill traffic, in fp64 words, under the register budget
      `registers` it was computed with (see `spilled`): each spilled value
      costs one store plus one load per read.
    - `n_instructions`: static count of computing statements -- `Decl` with an
      `init`, `Assign`, `WarpReduce`, `AtomicAdd` -- not dynamic, so a loop
      body counts once.
    - `nest`: the same statements, attributed to their enclosing `For` loops
      (`((loop var, ...outermost first), count)`, in first-occurrence order;
      `()` is outside every loop). The counts sum to `n_instructions`; the
      trip counts that would weight them are not in the IR (`tiered`).
    - `carried`: the most loop-carried values of any one `For`
      (`loop_carried`); `peak_live >= carried` by construction (module
      docstring, THE LOOP MODEL).

    `nest` and `carried` (#284) are added beside the first three, which they
    leave byte-identical: liveness, spilling and the flat count are the same
    functions of the IR they were.
    """

    peak_live: int
    traffic: int
    n_instructions: int
    registers: int  # the budget `traffic` assumed; part of the number
    nest: tuple[tuple[tuple[str, ...], int], ...]
    carried: int

    def by_nest(self) -> Mapping[tuple[str, ...], int]:
        return MappingProxyType(dict(self.nest))


TIERS = ("K4", "K2", "K0")


def tiered(
    cost: KernelCost, *, per: str, contraction: Sequence[str]
) -> Mapping[str, int]:
    """`cost.nest` as `hgp.cost`'s tiers, given the loop structure the IR
    does not name: `per` is the loop over contracted quartets (one iteration
    each), and each loop in `contraction` runs over one side's primitive
    PAIRS, a degree `K^2` for shells of degree `K`. A statement inside `per`
    and `n` contraction loops is `K^(2n)` (`n` <= 2: `K4`, `K2`, `K0`);
    one outside `per` is per block or per thread, not per quartet, and is
    `"outside"`. The four sum to `n_instructions`. Refuses a nest deeper
    than two contraction loops, a contraction loop outside `per`, and loop
    names the kernel does not have."""
    contraction = tuple(contraction)
    seen = {v for key, _ in cost.nest for v in key}
    missing = [v for v in (per, *contraction) if v not in seen]
    if missing:
        raise ValueError(f"no computing statement sits in loop(s) {missing}")
    out = dict.fromkeys((*TIERS, "outside"), 0)
    for key, n in cost.nest:
        depth = sum(v in contraction for v in key)
        if per not in key:
            if depth:
                raise ValueError(f"contraction loop outside {per!r}: {key}")
            out["outside"] += n
        elif depth > 2:
            raise ValueError(f"{key}: more than two contraction loops")
        else:
            out[f"K{2 * depth}"] += n
    return MappingProxyType(out)


def spilled(live: Liveness, registers: int) -> frozenset[str]:
    """The values a budget of `registers` spills, under one declared model:
    sweep the intervals by start and, whenever more than `registers` are
    live, evict the one whose interval ends last (which minimises the number
    of values spilled, for unit-cost intervals)."""
    if type(registers) is not int or registers < 0:
        raise ValueError(f"registers = {registers!r} is not a non-negative int")
    out: set[str] = set()
    active: list[str] = []
    for v in sorted(live.start, key=lambda v: (live.start[v], v)):
        s = live.start[v]
        active = [a for a in active if live.end[a] >= s]
        active.append(v)
        if len(active) > registers:
            victim = max(active, key=lambda a: (live.end[a], a))
            active.remove(victim)
            out.add(victim)
    return frozenset(out)


def kernel_cost(kernel: Kernel, *, registers: int) -> KernelCost:
    """`KernelCost` of `kernel` under a budget of `registers` values.

    `registers` has no default on purpose: an unstated budget turns a model
    into a number nobody can reproduce.
    """
    live = liveness(kernel)
    out = spilled(live, registers)
    computing = (Assign, WarpReduce, AtomicAdd)
    flat, loops, _ = _positions(kernel.body)
    nest: dict[tuple[str, ...], int] = {}
    for s, enclosing in zip(flat, loops, strict=True):
        if isinstance(s, computing) or (isinstance(s, Decl) and s.init is not None):
            key = tuple(flat[lp].var for lp in enclosing)  # type: ignore[attr-defined]
            nest[key] = nest.get(key, 0) + 1
    carried = loop_carried(kernel)
    return KernelCost(
        peak_live=live.peak,
        traffic=sum(1 + live.reads[v] for v in out),
        n_instructions=sum(nest.values()),
        registers=registers,
        nest=tuple(nest.items()),
        carried=max((len(v) for v in carried.values()), default=0),
    )
