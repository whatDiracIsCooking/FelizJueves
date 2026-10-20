"""The HGP recurrences as IR: the VRR (eqs 25-26) and HRR (eq 28) of `[Fock]` §3.1.

This is the bridge from `hgp/` to `ir.py`, the second engine's twin of the MMD
`recurrence.py`, and built on the same two-layer shape so the two are
recognisably the same kind of object:

- `vrr_body(dag, fundamental)` / `hrr_body(dag, boundary)` -- the statements
  alone, one `Decl` per DAG node in the DAG's build order, each node a LOCAL
  F64 scalar (`vrr_name`, `hrr_name`). The caller decides what a fundamental
  is (an `Expr` per Boys order `m`) and what a boundary value is (an `Expr`
  per `(e, f)`), and where the outputs go.
- `build_vrr_kernel(spec)` / `build_hrr_kernel(spec)` -- standalone
  recurrence-only `Kernel`s with scalar params, so each half is
  interpretable (and golden-able) before the fused kernel (#282) exists.

THIS MODULE KNOWS HGP; `interp.py` MUST NOT. The interpreter executes what is
produced here; test/test_hgp_recurrence.py pins it against a direct recursion
of eqs 25, 26 and 28 and against `ao/eri.py` (by way of `reference/`), neither
of which builds a DAG. THE DIRECTION IS READ OFF THE DAG, never re-derived:
`hgp.vrr` and `hgp.hrr` chose the axis and the side, and `vrr_reduction` /
`hrr_reduction` only spell, and check, what the DAG says.

THE VRR STATEMENT, one node `(e, f, m)` = `[e,0|f,0]^(m)` (`hgp.vrr`):

    Decl V{e}__{f}__{m} = Weighted( SUM_k mult_k * g_k(d) * child_k )

with `g_k` the `Factor` of the DAG term's coefficient name on the reduced
axis `d` (`VRR_FACTOR`): `PA WP inv2p inv2p*a_over_p` (eq 25) and `QC WQ
inv2q inv2q*alpha_over_q inv2pq` (eq 26), the signed `-e_d` riding in `c`
(`ir.Factor`, "THE SIGN LIVES IN `c`"). A leaf `[0,0|0,0]^(m)` is
`fundamental(m)`.

THE HRR STATEMENT, one node `(a, b, c, d)` = `(ab|cd)` (`hgp.hrr`):

    Decl H{a}__{b}__{c}__{d} = Weighted( 1 * 1 * transferred + 1 * AB_k|CD_k * lowered )

and a boundary node `(e,0|f,0)` is `boundary(e, f)` -- one read, as
`hgp.hrr.HrrDag.n_statements` counts it.

WHAT THE SCALARS ARE (the values a caller binds; the IR carries only names).
`p = alpha + beta`, `q = gamma + delta`, `P`, `Q` the pair centres, `PQ = P -
Q`, and `W = (p P + q Q)/(p + q)`:

    PA = P - A    QC = Q - C    inv2p = 1/(2p)    inv2q = 1/(2q)
    inv2pq = 1/(2(p+q))
    WP = W - P = -q/(p+q) PQ          WQ = W - Q = +p/(p+q) PQ
    inv2p_alpha_over_p = (1/2p)(rho/p)    inv2q_alpha_over_q = (1/2q)(rho/q)

where `rho = p q/(p+q)` is the REDUCED exponent -- `ir.Factor`'s "alpha" --
and is exactly FJ's `theta^2` (`reference/pq.py`). These are Obara-Saika's
coefficients for a seed whose `m` dependence is `F_m(T)` alone.

THE SIGN OF `WQ` IS DERIVED, NOT TRANSCRIBED. `[Fock]` eq 26 prints the
`(m+1)` displacement of the ket VRR as `-PQ p/(p+q)`; under the convention
eqs 25 and 28 fix (`PA = P - A`, `CD = C - D`, so `PQ = P - Q`) the ket
displacement is `W - Q = +p/(p+q) PQ`, and only that sign reproduces the
integrals (test/test_hgp_recurrence.py evaluates both against `ao/eri.py` on
a quartet with `P != Q`; the claim is `fock/eq26-ket-vrr-sign`,
asserted by #291). The IR names `W - Q` itself, so the sign is a property of
the bound VALUE, never of the instruction stream.

THE FUNDAMENTAL: ONE, AND IT IS FJ'S. `reference/pq.py` pins

    [0]^(m) = (2/sqrt(pi)) theta (-2 theta^2)^m F_m(T),
    theta^2 = p q/(p+q),  T = theta^2 |PQ|^2,

without pair factors (they ride on the dressed `E` in the MMD engine). The
HGP VRR needs the primitive `[00|00]^(m)` WITH the pair factors and WITHOUT
the `(-2 theta^2)^m`, because Obara-Saika's coefficients above already carry
the derivative factors MMD's PQ-only recurrence puts in the seed:

    [00|00]^(m) = U_P U_Q (-2 theta^2)^(-m) [0]^(m)
                = U_P U_Q (2/sqrt(pi)) theta F_m(T),
    U_P = (pi/p)^(3/2) K_ab,  K_ab = exp(-alpha beta/p |AB|^2),
    U_Q = (pi/q)^(3/2) K_cd,  K_cd = exp(-gamma delta/q |CD|^2).

`[Fock]` §3.1's `R0_m = pref F_m(T)` with `pref = sqrt(4 pi^5) K_ab K_cd /
sqrt(p^2 q^2 (p+q))` is the SAME quantity: `U_P U_Q (2/sqrt(pi)) theta =
pi^3/(pq)^(3/2) * 2/sqrt(pi) * sqrt(pq/(p+q)) K_ab K_cd = 2 pi^(5/2) K_ab
K_cd / (p q sqrt(p+q))`, and `2 pi^(5/2) = sqrt(4 pi^5)`. The factor
placement: `[Fock]` puts all of it in `pref`; FJ's `[0]^(m)` keeps
`(2/sqrt(pi)) theta` and `(-2 theta^2)^m` in the fundamental and the pair
factors on `E`; this engine binds `U_P U_Q` as ONE scalar, `pair`, and
spells the seed as `S F_m(T)` with `S = pair * ((2/sqrt(pi)) * theta)`, the
`m`-independent half of FJ's weight `W_m`. test/test_hgp_recurrence.py
asserts both identities pointwise against `reference.pq.fundamental`.

`[Fock]` eq 14 (§2.1) prints the seed as `U_P U_Q theta^m F_m(T)`; against
this derivation it lacks `(2/sqrt(pi)) theta` at every `m` and carries a
`theta^m` the coefficients of eqs 25-26 do not expect (claim
`fock/eq14-seed-inconsistent`, asserted by #291). The seed here is derived, not
transcribed from eq 14; the test shows eq 14's fails on `(ss|ss)` and, with
its normalisation repaired, still fails one VRR step up.

fp64 throughout (ir.Scalar). No import from `plan/`, `reference/` or `emit/`
(test/test_layering.py, rule "hgp.recurrence").
"""

from __future__ import annotations

import math
from collections.abc import Callable

from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.vrr import AmClass, Cart
from felizjueves.ir import (
    Assign,
    Boys,
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
    Term,
    Var,
    Weighted,
)
from felizjueves.spec import Spec, Target

OUT = "out"
BOUNDARY = "boundary"
PQ_PARAMS = ("PQ_x", "PQ_y", "PQ_z")
THETA = "theta"
PAIR = "pair"  # U_P U_Q, the two pair factors' product (module docstring)
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)

Fundamental = Callable[[int], Expr]
Boundary = Callable[[Cart, Cart], Expr]

# `hgp.vrr`'s coefficient names -> the `Factor` on reduced axis `d`.
VRR_FACTOR: dict[str, Callable[[int], Factor]] = {
    "PA": Factor.pa,
    "WP": Factor.wp,
    "inv2p": lambda d: Factor.INV2P,
    "inv2p*a_over_p": lambda d: Factor.INV2P_ALPHA_OVER_P,
    "QC": Factor.qc,
    "WQ": Factor.wq,
    "inv2q": lambda d: Factor.INV2Q,
    "inv2q*alpha_over_q": lambda d: Factor.INV2Q_ALPHA_OVER_Q,
    "inv2pq": lambda d: Factor.INV2PQ,
}
# `hgp.hrr`'s coefficient names -> their `Factor`.
HRR_FACTOR: dict[str, Factor] = {
    "1": Factor.ONE,
    **{f"AB_{a}": Factor.ab(k) for k, a in enumerate(hrr.AXES)},
    **{f"CD_{a}": Factor.cd(k) for k, a in enumerate(hrr.AXES)},
}

# The scalars each standalone kernel binds, in parameter order.
VRR_SCALARS: tuple[str, ...] = (
    *(Factor.pa(d).value for d in range(3)),
    *(Factor.wp(d).value for d in range(3)),
    *(Factor.qc(d).value for d in range(3)),
    *(Factor.wq(d).value for d in range(3)),
    Factor.INV2P.value, Factor.INV2Q.value, Factor.INV2PQ.value,
    Factor.INV2P_ALPHA_OVER_P.value, Factor.INV2Q_ALPHA_OVER_Q.value,
    *PQ_PARAMS, THETA, PAIR,
)
HRR_SCALARS: tuple[str, ...] = (
    *(Factor.ab(d).value for d in range(3)),
    *(Factor.cd(d).value for d in range(3)),
)


def _digits(v: Cart) -> str:
    return "_".join(str(c) for c in v)


def vrr_name(node: vrr.Node) -> str:
    """The LOCAL scalar holding `[e,0|f,0]^(m)`: `V{ex}_{ey}_{ez}__{f..}__{m}`."""
    e, f, m = node
    return f"V{_digits(e)}__{_digits(f)}__{m}"


def hrr_name(node: hrr.Node) -> str:
    """The LOCAL scalar holding `(ab|cd)`: `H{a}__{b}__{c}__{d}`."""
    return "H" + "__".join(_digits(v) for v in node)


def _differs_on(child: Cart, parent: Cart, by: int) -> int | None:
    """The one axis on which `parent - child == by`, others equal; else None."""
    diff = [p - c for p, c in zip(parent, child, strict=True)]
    axes = [k for k, x in enumerate(diff) if x]
    if len(axes) != 1 or diff[axes[0]] != by:
        return None
    return axes[0]


def vrr_reduction(dag: vrr.VrrDag, node: vrr.Node) -> Weighted:
    """The `Weighted` that forms `node` from its DAG terms.

    The axis is read off the first term (`PA` or `QC`, which lowers `e` or
    `f` by one at the same `m`); every term's child is then checked to be the
    shift its coefficient name says on that axis, so a malformed DAG raises
    here instead of building a kernel that computes something else."""
    e, f, m = node
    ts = dag.terms[node]
    if not ts:
        raise ValueError(f"{node} is a leaf; it has no reduction")
    ket = ts[0].coef == "QC"
    if ts[0].coef not in ("PA", "QC"):
        raise ValueError(f"{node}: first term {ts[0].coef!r} is not PA or QC")
    side, other = (f, e) if ket else (e, f)
    d = _differs_on(ts[0].child[1 if ket else 0], side, 1)
    if d is None:
        raise ValueError(f"{node}: first term {ts[0]} lowers no single axis")
    terms = []
    for t in ts:
        ce, cf, cm = t.child
        cside, cother = (cf, ce) if ket else (ce, cf)
        lower = t.coef in (("inv2q", "inv2q*alpha_over_q") if ket
                           else ("inv2p", "inv2p*a_over_p"))
        cross = t.coef == "inv2pq"
        if t.coef not in (vrr.KET if ket else vrr.BRA):
            raise ValueError(f"{node}: {t.coef!r} mixes eq 25 and eq 26")
        if cross:  # [e-1_d | f-1_d]^(m+1): one off each side, on d
            ok = (_differs_on(ce, e, 1) == d and _differs_on(cf, f, 1) == d
                  and cm == m + 1)
        else:
            ok = (_differs_on(cside, side, 2 if lower else 1) == d
                  and cother == other
                  and cm == m + (t.coef in ("WP", "WQ", "inv2p*a_over_p",
                                            "inv2q*alpha_over_q")))
        if not ok:
            raise ValueError(f"{node}: term {t} is not eq "
                             f"{26 if ket else 25} on axis {d}")
        terms.append(Term(t.mult, VRR_FACTOR[t.coef](d), Var(vrr_name(t.child))))
    return Weighted(tuple(terms))


def vrr_body(dag: vrr.VrrDag, fundamental: Fundamental) -> tuple[Stmt, ...]:
    """One `Decl` per node of `dag`, in `dag.order`: a leaf is
    `fundamental(m)`, every other node its `vrr_reduction`."""
    body: list[Stmt] = [Note("[Fock] eqs 25-26, the HGP VRR (reduction form; "
                             "W - Q derived, not eq 26's printed sign)")]
    for node in dag.order:
        value = (fundamental(node[2]) if vrr.is_leaf(node)
                 else vrr_reduction(dag, node))
        body.append(Decl(vrr_name(node), Scalar.F64, init=value))
    return tuple(body)


def hrr_reduction(dag: hrr.HrrDag, node: hrr.Node) -> Weighted:
    """`(ab|cd) = transferred + AB_k|CD_k * lowered`, read off the DAG and
    checked: the transfer moves one unit on axis `k` from `b` to `a` (or `d`
    to `c`), the lowered term drops it from `b` (`d`)."""
    ts = dag.terms[node]
    if len(ts) != 2 or ts[0].coef != "1":
        raise ValueError(f"{node}: terms {ts} are not eq 28's two")
    coef = ts[1].coef
    if coef not in HRR_FACTOR or coef == "1":
        raise ValueError(f"{node}: {coef!r} is not AB_k or CD_k")
    ket = coef.startswith("CD")
    k = hrr.AXES.index(coef[-1])
    lo, hi = (2, 3) if ket else (0, 1)  # (c, d) or (a, b)
    moved, lowered = ts[0].child, ts[1].child
    for i, (n, mv, lw) in enumerate(zip(node, moved, lowered, strict=True)):
        if i == lo:
            ok = _differs_on(n, mv, 1) == k and lw == n
        elif i == hi:
            ok = _differs_on(mv, n, 1) == k and lw == mv
        else:
            ok = mv == n and lw == n
        if not ok:
            raise ValueError(f"{node}: terms {ts} are not eq 28 on axis {k}")
    return Weighted((Term(1, Factor.ONE, Var(hrr_name(moved))),
                     Term(1, HRR_FACTOR[coef], Var(hrr_name(lowered)))))


def hrr_body(dag: hrr.HrrDag, boundary: Boundary) -> tuple[Stmt, ...]:
    """One `Decl` per node of `dag`, in `dag.order`: a boundary node
    `(e,0|f,0)` is `boundary(e, f)`, every other node its `hrr_reduction`."""
    body: list[Stmt] = [Note("[Fock] eq 28, the HGP HRR on contracted values")]
    for node in dag.order:
        value = (boundary(node[0], node[2]) if hrr.is_boundary(node)
                 else hrr_reduction(dag, node))
        body.append(Decl(hrr_name(node), Scalar.F64, init=value))
    return tuple(body)


def _mul(a: Expr, b: Expr) -> Call:
    return Call(Fn.MUL, (a, b))


def _add(a: Expr, b: Expr) -> Call:
    return Call(Fn.ADD, (a, b))


# The local array the prelude's `Boys` fills: `boys[m] = F_m(T)` (#301).
# `recurrence.BOYS_ARRAY`'s name, spelled here because `hgp/` stays off the
# MMD recurrence (test_layering.py).
BOYS_ARRAY = "boys"


def fundamental_prelude(top: int) -> tuple[tuple[Stmt, ...], Fundamental]:
    """Statements forming `T`, the seed scale `S` and every Boys order to
    `top`, and the `Fundamental`.

        T = (theta theta) |PQ|^2,   S = pair ((2/sqrt(pi)) theta),
        boys[m] = F_m(T), m = 0..top,   [00|00]^(m) = S boys[m]

    (module docstring, "THE FUNDAMENTAL"). `top` is the class's L, the
    highest leaf order its VRR reaches; a leaf above it is refused."""
    pqx, pqy, pqz = (Var(n) for n in PQ_PARAMS)
    theta = Var(THETA)
    stmts: tuple[Stmt, ...] = (
        Note("[00|00]^(m) = U_P U_Q (2/sqrt(pi)) theta F_m(T) = U_P U_Q "
             "(-2 theta^2)^(-m) [0]^(m), reference/pq.py's fundamental"),
        Decl("T", Scalar.F64, init=_mul(_mul(theta, theta), _add(
            _add(_mul(pqx, pqx), _mul(pqy, pqy)), _mul(pqz, pqz)))),
        Decl("S", Scalar.F64, init=_mul(Var(PAIR), _mul(
            Lit(_TWO_OVER_SQRT_PI), theta))),
        Decl(BOYS_ARRAY, Scalar.F64, shape=(Lit(top + 1),)),
        Boys(BOYS_ARRAY, top, Var("T")),
    )

    def fundamental(m: int) -> Expr:
        if not 0 <= m <= top:
            raise ValueError(f"Boys order {m} above the prelude's top {top}")
        return _mul(Var("S"), Index(BOYS_ARRAY, (Lit(m),)))

    return stmts, fundamental


def hgp_class(spec: Spec) -> AmClass:
    """The `(la, lb, lc, ld)` of a `Target.F` spec; anything else refused."""
    if not isinstance(spec, Spec) or spec.target is not Target.F:
        raise ValueError(f"the HGP recurrences build Target.F classes, not "
                         f"{getattr(spec, 'target', spec)!r}")
    return tuple(spec.am)  # type: ignore[return-value]


def boundary_order(cls: AmClass) -> tuple[tuple[Cart, Cart], ...]:
    """The boundary `(e, f)` in `hgp.vrr.targets` order: the VRR kernel's
    `out` layout and the HRR kernel's `boundary` layout, the same list."""
    return tuple((e, f) for e, f, _ in vrr.targets(cls))


def _name(prefix: str, cls: AmClass) -> str:
    return f"{prefix}_{''.join(str(x) for x in cls)}"


def build_vrr_kernel(spec: Spec, *, name: str | None = None) -> Kernel:
    """A VRR-only Kernel: `out[i] = [e_i,0|f_i,0]^(0)`, `boundary_order`.

    Scalar params `VRR_SCALARS` (the eq 25-26 coefficients, then `PQ_x PQ_y
    PQ_z theta pair` for the fundamental), and `out` (F64, OUT)."""
    cls = hgp_class(spec)
    dag = vrr.build(cls)
    prelude, fundamental = fundamental_prelude(sum(cls))
    body: list[Stmt] = [*prelude, *vrr_body(dag, fundamental)]
    for i, (e, f) in enumerate(boundary_order(cls)):
        body.append(Assign(Index(OUT, (Lit(i),)), Var(vrr_name((e, f, 0)))))
    params = (*(Param(n, Scalar.F64) for n in VRR_SCALARS),
              Param(OUT, Scalar.F64, array=True, intent=Intent.OUT))
    return Kernel(name or _name("hgp_vrr", cls), params, tuple(body))


def build_hrr_kernel(spec: Spec, *, first: str = "bra",
                     name: str | None = None) -> Kernel:
    """An HRR-only Kernel: `out[i] = (a b | c d)_i` in `hgp.hrr.targets`
    order, from `boundary[j]` = `(e_j 0 | f_j 0)` in `boundary_order`.

    Scalar params `AB_x AB_y AB_z CD_x CD_y CD_z`; `boundary` (F64, IN) and
    `out` (F64, OUT). `first` is `hgp.hrr.build`'s side order."""
    cls = hgp_class(spec)
    dag = hrr.build(cls, first)
    slot = {ef: i for i, ef in enumerate(boundary_order(cls))}
    if set(slot) != dag.boundary:  # #274 holds this; refuse rather than misread
        raise ValueError(f"{cls}: HRR boundary is not the VRR target set")

    def boundary(e: Cart, f: Cart) -> Expr:
        return Index(BOUNDARY, (Lit(slot[(e, f)]),))

    body: list[Stmt] = list(hrr_body(dag, boundary))
    for i, node in enumerate(dag.targets):
        body.append(Assign(Index(OUT, (Lit(i),)), Var(hrr_name(node))))
    params = (*(Param(n, Scalar.F64) for n in HRR_SCALARS),
              Param(BOUNDARY, Scalar.F64, array=True),
              Param(OUT, Scalar.F64, array=True, intent=Intent.OUT))
    return Kernel(name or _name(f"hgp_hrr_{first}", cls), params, tuple(body))
