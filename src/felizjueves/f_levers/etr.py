"""The ETR as a second VRR, arbitrated per class on the cost model (#354, lane H11).

The fused F kernel forms its boundary `[e0|f0]^(0)` by `hgp.vrr`'s KET
CASCADE: eq 25 up the bra to `|e| = la + lb`, then eq 26 up the ket at every
Boys order. `hgp.etr` builds the same boundary the other way Head-Gordon and
Pople offer: eq 25 up the bra to `|e| = L`, then the ELECTRON-TRANSFER
RELATION at `m = 0`,

    [e|f+1_d] = etr_c_d [e|f] + e_d inv2q [e-1_d|f] + f_d inv2q [e|f-1_d]
              - etr_pq [e+1_d|f],
    etr_pq = p / q,   etr_c_d = QC_d + etr_pq PA_d          (per quartet)

(`hgp/etr.py` derives it). This lever swaps the VRR body for the ETR's on
the classes where the model prefers it and the accuracy gate admits it, and
leaves the literal kernel alone on the others.

THE ARBITRATION, DECLARED (`RULE`, `choice`). Per class, the two VRR DAGs
are scored on two numbers of this tree's own models:

    peak   hgp.vrr.vrr_peak(dag): the statement-order live model #350's
           group rule reads (values live at once executing the emitted
           order, the boundary live out) -- the register-pressure opinion
    x      hgp.cost's K^4 FLOPs per primitive quartet (`cost` for the
           cascade, `etr_cost` for the ETR): the work opinion

and the model prefers the ETR (`Score.rule`) when `(peak, x)` is
LEXICOGRAPHICALLY LOWER for it; a tie on both keeps the cascade (the
literal kernel). The HRR, the boundary and its accumulation are identical
under both, so only the VRR side is compared. The model's preference: the
ETR on every class with a non-`s` bra shell, the cascade on the six
`(ss|..)` classes -- the pure-ket classes, where the ETR raises `|e|` to
`lc + ld` only to transfer it all back.

THE ACCURACY GATE, DECLARED (`ACCURACY`, `ACCURACY_TOL`, `gate`). The
transfer's `- (p/q) [e+1_d|f]` term cancels against the others when the
bra's exponent sum is large beside the ket's (a tight bra, a diffuse ket),
and every transfer step multiplies what was lost by `p/q`; no rewrite of
the coefficients removes that (the cancellation is between the terms, not
inside one). On H2O/cc-pVDZ's contracted shells -- the per-class device
cells' inputs, which the synthetic shells and the s/p molecules of the
first interpreted tests did not reach -- the ETR forced misses the F
tolerance on `(ss|dd)` (5.99e-13 of scale) and `(sp|pd)` (6.92e-13), the
one a cascade class and the other a class the model ships with the ETR;
#354's first device run failed on exactly those two, on both cards. So the
ETR SHIPS ONLY WHERE THE MODEL PREFERS IT AND `gate` ADMITS IT: a class
whose interpreted ETR F is recorded within `ACCURACY_TOL` (the fused F's
`TOL`) on those inputs. A refused class keeps the cascade; so does a class
with no record (`UNVERIFIED`: every class above L = 4, milestone 13's cap
-- unchecked, so not shipped). `OVERRIDES` (empty) wins over both. The
result on the 21 classes (`data/hgp/etr_arbitration.json`, held to
`render()` by test/test_f_lever_etr.py, which names the command that
rewrites it, and the record recomputed there): the ETR on the six classes
with L <= 4 the model prefers and the gate admits, `(sp|sp)`, `(sp|sd)`,
`(sp|pp)`, `(sd|sd)`, `(sd|pp)` and `(pp|pp)`; the cascade on the rest.

THE COST HALF IS A MODEL'S OPINION, NOT A MEASUREMENT. On most classes the two
opinions DISAGREE: the ETR lowers the peak but costs more FLOPs (it carries
`|e|` to `L`, and each transfer step reads four values) until `(pd|dd)`
and `(dd|dd)`, where it wins both. Neither number is a register count; the
compiler's (ptxas / the AMDGPU backend) is, and test/test_f_lever_etr_device.py
prints both recurrences' figures on every class with L <= 4 so the
disagreement is measured rather than assumed. A measured selector (#362)
may override any entry (but not ship the ETR past the accuracy gate's
refusal); the rule here is what ships until one does.

THE KERNEL, when the class arbitrates to the ETR. The `vrr` hook emits, in
the ETR DAG's order: the per-quartet `etr_pq` and the `etr_c_d` of every
axis a transfer reduces along; each leaf `S F_m(T)` (the literal
fundamental); each bra node by `hgp.recurrence.vrr_reduction` (eq 25,
verbatim); each transfer node as a sum of products in the term order above
(an integer multiplier `k > 1` one extra multiply by `Lit(k)`; the
transfer's sign a subtract) -- not a `Weighted`, because `etr_c_d` and
`etr_pq` are not in `ir.Factor`'s closed vocabulary and this lever does not
widen it. The V names are the cascade's (`vrr_name`), so `accumulate` and
everything after it are the literal hooks. Because a transfer reads `QC_d`,
`PA_d` and `inv2q` through `Call`s, which `FContext.vneed` does not see,
`ket_prim` and `quartet_scalars` are wrapped to declare exactly those of
them the default did not (the literal spellings: the ket record's `X_d`,
the staged bra record's `X_d`, `0.5 / q`).

ON A CLASS THAT ARBITRATES TO THE CASCADE `apply` returns the hooks
unchanged: the kernel is the literal one plus the provenance Note.

NOT BITWISE: the ETR is the same integrals by a different recurrence, so
the boundary differs from the literal one in rounding. It is held, with the
existing tolerances (never loosened), to `ao/eri.py`'s MMD integrals and to
libcint (through the tracked fixtures), and its `F` to `ao.fock`'s sweep,
on every class with L <= 4 (test/test_f_lever_etr.py) -- and on H2O/cc-pVDZ,
where it does not hold on two classes, which is THE ACCURACY GATE above.

COMPOSITION.
  - `batched-vrr` (#350): composes. #350's nest hands the `vrr` hook a
    context whose `vdag` is a cascade group's closure; the hook takes the
    ETR's closure of THAT GROUP'S TARGETS (`hgp.vrr.group_dag` over the ETR
    DAG), so each group computes its boundary values by the ETR, each node
    the same expression as in the ungrouped ETR kernel. `F` and the tap are
    the `etr` kernel's bit for bit (asserted). The group count stays #350's
    rule, read on the cascade's peaks: a model mismatch recorded here, not
    corrected.
  - `subdag-vrr` (#357): REFUSED (`SUBDAG_REFUSAL`). It re-schedules the
    cascade group's closure and hands the reordered DAG to the `vrr` hook;
    the ETR replaces that DAG, so the schedule would be silently discarded.
  - Levers that move other hooks (`block-fab`, `ket-break`,
    `struct-reads`) are untouched by this one; the record reads it adds go
    through `ctx.read`, so `struct-reads`' `field` hook sees them.

Like the seams, nothing here imports `plan/`, `ao/`, the interpreter,
`reference/`, a backend, a device or `kernel_f`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from functools import cache
from types import MappingProxyType

from felizjueves.f_levers.seams import AB_SH, KAB, KCD, REC, FContext, FHooks
from felizjueves.hgp import cost as _cost
from felizjueves.hgp import etr as _etr
from felizjueves.hgp import vrr as _vrr
from felizjueves.hgp.recurrence import (
    OUT,
    VRR_SCALARS,
    boundary_order,
    fundamental_prelude,
    hgp_class,
    vrr_name,
    vrr_reduction,
)
from felizjueves.hgp.vrr import AmClass, Node
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Factor,
    Fn,
    Index,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.spec import Spec

__all__ = [
    "ACCURACY",
    "ACCURACY_INPUT",
    "ACCURACY_TOL",
    "ADMITTED",
    "CASCADE",
    "ETR",
    "NAME",
    "OVERRIDES",
    "PQ",
    "REFUSED",
    "RULE",
    "SUBDAG_REFUSAL",
    "Score",
    "UNVERIFIED",
    "apply",
    "build_etr_vrr_kernel",
    "choice",
    "closure",
    "etr_body",
    "forced",
    "gate",
    "needs",
    "render",
    "report",
    "score",
]

NAME = "etr"
ETR, CASCADE = "etr", "cascade"
PQ = "etr_pq"  # p / q, the transfer coefficient
_C = ("etr_c_x", "etr_c_y", "etr_c_z")  # QC_d + (p/q) PA_d
_X = REC.index("X_x")  # the record's PA / QC offset

RULE = (
    "per class, score the ket cascade (hgp.vrr.build) and the ETR "
    "(hgp.etr.build) on (peak, x): peak = hgp.vrr.vrr_peak, the "
    "statement-order live model (#350's), x = hgp.cost's K^4 FLOPs per "
    "primitive quartet (cost / etr_cost); the ETR when its (peak, x) is "
    "lexicographically lower, else the cascade (a tie on both keeps the "
    "literal kernel); then THE ACCURACY GATE: the ETR ships only on a "
    "class whose interpreted ETR F is recorded within the F tolerance "
    "(ACCURACY, ACCURACY_TOL), else the cascade; OVERRIDES wins. The cost "
    "half is a model's opinion, not a measurement: a measured selector "
    "(#362) may override it")

SUBDAG_REFUSAL = (
    "etr + subdag-vrr: #357 re-schedules the ket cascade's group closure and "
    "hands it to the vrr hook; the ETR replaces that DAG with its own, so the "
    "dedicated schedule would be silently discarded. A per-group schedule of "
    "the ETR's closure is a lever nobody has declared")

# Per-class overrides of the rule (class -> ETR or CASCADE). Empty: every
# class ships the rule's choice. An entry is a decision, stated beside it.
OVERRIDES: Mapping[AmClass, str] = MappingProxyType({})

# THE ACCURACY GATE (module docstring). The F tolerance it is held to: the
# fused F's `TOL` of `scale_G` (test/test_fock_build.py; equality asserted by
# test/test_f_lever_etr.py), never loosened.
ACCURACY_TOL = 1e-13
ACCURACY_INPUT = (
    "H2O/cc-pVDZ, launch_f's plan at tau = 0, n = 1 (test_fock_ondevice.py's "
    "PART1): every launch of the class, the kernel the ETR forced "
    "(`forced(ETR)`), interpreted, against ao.fock's sweep restricted to the "
    "quartets those launches meet, worst |F - G| / scale_G (scale_G: the "
    "sweep on |D| with ao.eri's absolute-value twin)")
# The gate's record: class -> that worst error, recomputed by
# test/test_f_lever_etr.py (to 1%; the verdict exactly). Only the classes
# with L <= 4 (milestone 13's cap); a class absent is UNVERIFIED and never
# ships the ETR. Interpreted fp64, no FMA contraction: the card's figure
# differs in rounding (#354's device run, ETR forced: (ss|dd) 3.4e-13 and
# (sp|pd) 7.8e-13 of scale on both cards, the other eleven at most 5.9e-14).
ACCURACY: Mapping[AmClass, float] = MappingProxyType({
    (0, 0, 0, 0): 8.54e-16,
    (0, 0, 0, 1): 4.69e-15,
    (0, 0, 0, 2): 1.89e-14,
    (0, 0, 1, 1): 2.90e-15,
    (0, 0, 1, 2): 3.19e-14,
    (0, 0, 2, 2): 5.99e-13,  # REFUSED
    (0, 1, 0, 1): 3.33e-15,
    (0, 1, 0, 2): 1.66e-14,
    (0, 1, 1, 1): 1.11e-14,
    (0, 1, 1, 2): 6.92e-13,  # REFUSED
    (0, 2, 0, 2): 2.98e-14,
    (0, 2, 1, 1): 1.47e-14,
    (1, 1, 1, 1): 3.90e-16,
})
ADMITTED, REFUSED, UNVERIFIED = "admitted", "refused", "unverified"


def gate(cls: AmClass) -> str:
    """The accuracy gate's verdict on the ETR for `cls`: `ADMITTED` when
    its recorded error is within `ACCURACY_TOL`, `REFUSED` when not,
    `UNVERIFIED` when there is no record (L > 4)."""
    cls = tuple(cls)
    if cls not in ACCURACY:
        return UNVERIFIED
    return ADMITTED if ACCURACY[cls] <= ACCURACY_TOL else REFUSED


# ---------------------------------------------------------------------------
# The arbitration (module docstring, THE ARBITRATION)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Score:
    """One class's two VRR sides on the model, and the rule's choice."""

    cls: AmClass
    cascade_peak: int
    etr_peak: int
    cascade_x: int  # K^4 FLOPs per primitive quartet
    etr_x: int
    cascade_total: int  # x + y + z: an uncontracted quartet
    etr_total: int
    cascade_nodes: int
    etr_nodes: int

    @property
    def rule(self) -> str:
        return (ETR if (self.etr_peak, self.etr_x)
                < (self.cascade_peak, self.cascade_x) else CASCADE)

    @property
    def peak_margin(self) -> int:
        """Live values the ETR saves (negative: it costs them)."""
        return self.cascade_peak - self.etr_peak

    @property
    def x_margin(self) -> int:
        """K^4 FLOPs the ETR saves (negative: it costs them)."""
        return self.cascade_x - self.etr_x

    @property
    def opinions_agree(self) -> bool:
        """Do the two numbers point the same way (or does one tie)?"""
        return self.peak_margin * self.x_margin >= 0


@cache
def score(cls: AmClass) -> Score:
    cls = tuple(cls)
    c, e = _vrr.build(cls), _etr.build(cls)
    cc, ec = _cost.cost(cls), _cost.etr_cost(cls)
    return Score(cls, _vrr.vrr_peak(c), _vrr.vrr_peak(e), cc.x, ec.x,
                 cc.total, ec.total, c.n_nodes, e.n_nodes)


def choice(cls: AmClass) -> str:
    """The shipped recurrence of `cls`: `OVERRIDES`, else the model's rule
    AND the accuracy gate -- the ETR only where the model prefers it and
    `gate` admits it."""
    cls = tuple(cls)
    if cls in OVERRIDES:
        return OVERRIDES[cls]
    return (ETR if score(cls).rule == ETR and gate(cls) == ADMITTED
            else CASCADE)


ABOUT = (
    "Per-class arbitration of the F kernel's VRR side between the ket "
    "cascade (hgp.vrr, [Fock] eqs 25-26) and the electron-transfer "
    "relation (hgp.etr, #354), from src/felizjueves/f_levers/etr.py. Rule: "
    + RULE + ". peak_margin and x_margin are cascade minus ETR (positive: "
    "the ETR saves); opinions_agree is whether the two point the same way; "
    "rule is the model's choice. accuracy is the ETR forced, worst error / "
    "scale on " + ACCURACY_INPUT + "; gate is admitted when it is within "
    f"{ACCURACY_TOL:g}, refused when not, unverified with no record (L > "
    "4, milestone 13's cap); choice ships the ETR only where rule says etr "
    "and gate is admitted. "
    "Model numbers only, calibrated against nothing: the compiler's "
    "registers and spills for both recurrences are printed by "
    "test/test_f_lever_etr_device.py, not stored here.")


def report() -> dict:
    rows = []
    for cls in _vrr.CLASSES:
        s = score(cls)
        rows.append({
            "class": list(cls),
            "cascade": {"peak": s.cascade_peak, "x": s.cascade_x,
                        "total": s.cascade_total, "nodes": s.cascade_nodes},
            "etr": {"peak": s.etr_peak, "x": s.etr_x, "total": s.etr_total,
                    "nodes": s.etr_nodes},
            "peak_margin": s.peak_margin,
            "x_margin": s.x_margin,
            "opinions_agree": s.opinions_agree,
            "rule": s.rule,
            "accuracy": ACCURACY.get(cls),
            "gate": gate(cls),
            "choice": choice(cls),
        })
    return {"about": ABOUT, "classes": rows}


def render() -> str:
    return json.dumps(report(), indent=1) + "\n"


# ---------------------------------------------------------------------------
# The ETR body as IR
# ---------------------------------------------------------------------------


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


@cache
def _etr_dag(cls: AmClass) -> _vrr.VrrDag:
    return _etr.build(tuple(cls))


@cache
def _closure(cls: AmClass, targets: tuple[Node, ...]) -> _vrr.VrrDag:
    dag = _etr_dag(cls)
    if targets == dag.targets:
        return dag
    return _vrr.group_dag(dag, targets)


def closure(ctx: FContext) -> _vrr.VrrDag:
    """The ETR DAG's closure of `ctx.vdag`'s targets (every boundary value,
    or one #350 group's)."""
    return _closure(tuple(ctx.cls), tuple(ctx.vdag.targets))


def _axis(dag: _vrr.VrrDag, node: Node) -> int:
    return next(i for i, c in enumerate(node[1]) if c)


def needs(dag: _vrr.VrrDag) -> frozenset[str]:
    """The scalars the transfer statements read outside a `Weighted`:
    `QC_d` and `PA_d` of every transfer axis, and `inv2q` when a lowering
    term exists."""
    out: set[str] = set()
    for n in dag.order:
        if _etr.is_transfer(n):
            d = _axis(dag, n)
            out |= {Factor.qc(d).value, Factor.pa(d).value}
            if any(t.coef == "inv2q" for t in dag.terms[n]):
                out.add(Factor.INV2Q.value)
    return frozenset(out)


def _transfer(dag: _vrr.VrrDag, node: Node, absolute: bool = False
              ) -> Expr:
    """`node`'s ETR as a sum of products, in `hgp.etr.reduce`'s term order;
    each child checked to be the shift its coefficient says. `absolute`:
    every term added (the |.| twin, `build_etr_vrr_kernel`)."""
    e, f, m = node
    d = _axis(dag, node)
    ts = dag.terms[node]
    g = _vrr._minus(f, d)
    want = {"etr_c": [(e, g, 0)], "p_over_q": [(_etr._plus(e, d), g, 0)],
            "inv2q": [x for x in ((_vrr._minus(e, d), g, 0) if e[d] else None,
                                  (e, _vrr._minus(g, d), 0) if g[d] else None)
                      if x is not None]}
    coef = {"etr_c": Var(_C[d]), "inv2q": Var(Factor.INV2Q.value),
            "p_over_q": Var(PQ)}
    out: Expr | None = None
    for t in ts:
        if m != 0 or t.coef not in want or t.child not in want[t.coef]:
            raise ValueError(f"{node}: term {t} is not the ETR on axis {d}")
        prod: Expr = _c(Fn.MUL, coef[t.coef], Var(vrr_name(t.child)))
        if abs(t.mult) != 1:
            prod = _c(Fn.MUL, Lit(float(abs(t.mult))), prod)
        if out is None:
            if t.mult < 0:
                raise ValueError(f"{node}: the first term is negative")
            out = prod
        else:
            out = _c(Fn.SUB if t.mult < 0 and not absolute else Fn.ADD,
                     out, prod)
    assert out is not None
    return out


def etr_body(dag: _vrr.VrrDag, fundamental, pq: Expr, *,
             absolute: bool = False) -> tuple[Stmt, ...]:
    """One `Decl` per node of the ETR `dag`, in its order, after the
    per-quartet `etr_pq = pq` and `etr_c_d` of every transfer axis. FP64,
    as `hgp.recurrence.vrr_body`'s: a precision lever retypes by wrapping
    the `vrr` hook. `absolute` adds every transfer term (the |.| twin's
    half that a `Weighted`'s `|c|` cannot reach)."""
    real = Scalar.F64
    axes = sorted({_axis(dag, n) for n in dag.order if _etr.is_transfer(n)})
    body: list[Stmt] = [Note(
        "#354: the boundary by eq 25 to |e| = L, then the electron-transfer "
        "relation at m = 0 ([e|f+1_d] = etr_c_d [e|f] + e_d inv2q [e-1_d|f] "
        "+ f_d inv2q [e|f-1_d] - etr_pq [e+1_d|f]; hgp/etr.py)")]
    if axes:
        body.append(Decl(PQ, real, init=pq))
        for d in axes:
            body.append(Decl(_C[d], real, init=_c(
                Fn.ADD, Var(Factor.qc(d).value),
                _c(Fn.MUL, Var(PQ), Var(Factor.pa(d).value)))))
    for node in dag.order:
        if _vrr.is_leaf(node):
            value = fundamental(node[2])
        elif _etr.is_transfer(node):
            value = _transfer(dag, node, absolute)
        else:
            value = vrr_reduction(dag, node)
        body.append(Decl(vrr_name(node), real, init=value))
    return tuple(body)


# ---------------------------------------------------------------------------
# The hooks
# ---------------------------------------------------------------------------


def _vrr_hook(ctx: FContext) -> tuple[Stmt, ...]:
    return etr_body(closure(ctx), ctx.fundamental,
                    _c(Fn.DIV, Var("p"), Var("q")))


def _missing(ctx: FContext) -> frozenset[str]:
    return needs(closure(ctx)) - ctx.vneed


def _wrap_ket_prim(prev):
    def ket_prim(ctx: FContext) -> tuple[Stmt, ...]:
        miss = _missing(ctx)
        extra = tuple(
            ctx.scalar(Factor.qc(d).value, ctx.read(
                "ket_prim", ctx.k, Var(KCD), Lit(_X + d)), "vrr")
            for d in range(3) if Factor.qc(d).value in miss)
        return (*prev(ctx), *extra)
    return ket_prim


def _wrap_quartet_scalars(prev):
    def quartet_scalars(ctx: FContext) -> tuple[Stmt, ...]:
        miss = _missing(ctx)
        extra = [ctx.scalar(Factor.pa(d).value, ctx.read(
            AB_SH, Var(KAB), Lit(_X + d)), "vrr")
            for d in range(3) if Factor.pa(d).value in miss]
        if Factor.INV2Q.value in miss:
            extra.append(ctx.scalar(Factor.INV2Q.value, _c(
                Fn.DIV, Lit(0.5), Var("q")), "vrr"))
        return (*prev(ctx), *extra)
    return quartet_scalars


def forced(recurrence: str | None = None):
    """The lever's `apply` with the recurrence fixed (`ETR` or `CASCADE`;
    `None`: the class's `choice`). `apply` is `forced()`; `forced(ETR)` is
    an experiment's arm (both recurrences on every class), never shipped
    under the name."""
    if recurrence not in (None, ETR, CASCADE):
        raise ValueError(f"recurrence {recurrence!r} is not {ETR!r} or "
                         f"{CASCADE!r}")

    def apply_r(hooks: FHooks, spec: Spec) -> FHooks:
        if "subdag-vrr" in spec.levers:
            raise ValueError(SUBDAG_REFUSAL)
        which = choice(tuple(spec.am)) if recurrence is None else recurrence
        if which == CASCADE:
            return hooks
        return dataclasses.replace(
            hooks, vrr=_vrr_hook,
            ket_prim=_wrap_ket_prim(hooks.ket_prim),
            quartet_scalars=_wrap_quartet_scalars(hooks.quartet_scalars))

    return apply_r


apply = forced()


# ---------------------------------------------------------------------------
# A standalone ETR kernel: hgp.recurrence.build_vrr_kernel's twin
# ---------------------------------------------------------------------------


def build_etr_vrr_kernel(spec: Spec, *, name: str | None = None,
                         absolute: bool = False) -> Kernel:
    """`hgp.recurrence.build_vrr_kernel` with the ETR body: the same
    parameters (`VRR_SCALARS`, then `out`) and the same `out` layout
    (`boundary_order`), so an interpreted driver of the one runs the other.
    `etr_pq` is `inv2q / inv2p` (= p / q) from the bound scalars.

    `absolute=True` is half of this engine's |.| twin (`ao.hgp_drive`'s
    "THE SCALE"): every transfer term ADDED; `ao.hgp_drive.absolute_kernel`
    then flips the `Weighted` coefficients, and the kernel run on |scalars|
    gives `SUM |term|` through the ETR's own sums -- the yardstick for ITS
    rounding, whose `- p/q` transfer cancels where the cascade's does not."""
    cls = hgp_class(spec)
    dag = _etr_dag(cls)
    prelude, fundamental = fundamental_prelude()
    pq = _c(Fn.DIV, Var(Factor.INV2Q.value), Var(Factor.INV2P.value))
    body: list[Stmt] = [*prelude, *etr_body(dag, fundamental, pq,
                                            absolute=absolute)]
    for i, (e, f) in enumerate(boundary_order(cls)):
        body.append(Assign(Index(OUT, (Lit(i),)), Var(vrr_name((e, f, 0)))))
    params = (*(Param(n, Scalar.F64) for n in VRR_SCALARS),
              Param(OUT, Scalar.F64, array=True, intent=Intent.OUT))
    return Kernel(name or f"hgp_etr_{''.join(map(str, cls))}", params,
                  tuple(body))
