"""A FLOP model for an HGP class, read off its DAGs and tiered `K^4 / K^2 / K^0`.

An ENGINEERING INSTRUMENT, CALIBRATED AGAINST NOTHING. `[Fock]` publishes no
FLOP table, and the 2020 letter's three cells are out of scope (milestone 12,
pin 2). So nothing here has an `exact >= floor` ratchet or a published
comparison: it compares schedules of this repo's own DAGs, and that is all.
It is `plan/cost.py`'s (`PlanCost`'s) SIBLING, not `lower.py`'s
(`KernelCost`'s): it reads the VRR DAG (`hgp.vrr`, #273) and the HRR DAG
(`hgp.hrr`, #274) -- never the lowered IR -- and feeds no floor. It imports
neither `lower` nor `plan/` (test/test_layering.py, rule "hgp";
test/test_lower.py holds the `lower` half package-wide).

THE TIERS are where the work sits in the primitive loop nest of the fused
kernel -- `[Fock]`'s own uncontracted / half-contracted / fully contracted
taxonomy. For a class whose shells all have contraction degree `K`, the
FLOPs per contracted quartet are `x K^4 + y K^2 + z`:

- `x`, `K^4` (uncontracted, once per primitive quartet): the quartet geometry
  (`p+q`, its inverse, `PQ`, `WP`, `WQ`, the composite coefficients, the Boys
  argument and the seed prefactor), one multiply per Boys leaf
  `[0,0|0,0]^(m)` the DAG holds (the seed), every VRR step, and one add per
  boundary value accumulated into its contracted register;
- `y`, `K^2` (half-contracted, once per primitive PAIR, bra and ket twin):
  `p`, `1/p`, `P`, `PA`, `1/(2p)`, `ab/p` and `Kab`, and the ket's `q`, `Q`,
  `QC`, `1/(2q)`, `cd/q`, `Kcd`. `y` is the bra pair's FLOPs plus the ket
  pair's; with unequal degrees they weigh `K_a K_b` and `K_c K_d`. `1/(2p)`
  is not in #276's list (`p, P, PA, Kab`); it is here because eq 25 reads it
  and it is a pair quantity, so it hoists;
- `z`, `K^0` (fully contracted, once per contracted quartet): every HRR step
  (eq 28, `x = y + AB_k z`: one multiply and one add), and the shell-pair
  separations `AB`, `|AB|^2`, `CD`, `|CD|^2` -- the same for every primitive
  of the pair, so computed once per quartet (`|AB|^2` feeds `Kab`).

THE COUNTING CONVENTION. One `+`, `-`, `*` or `/` is one FLOP; no fused
multiply-add. An integer coefficient of +-1 costs nothing, and a sign costs
nothing (it turns an add into a subtract, or is folded into the constant of a
multiply). A VRR node `sum_i mult_i coef_i child_i` costs, per term, one
multiply by `coef_i` plus one more when `|mult_i| > 1`, and `n - 1` adds to
combine its `n` terms: the term structure of eqs 25-26 exactly as
`hgp.vrr.reduce` lists it, each lower term absent when its index is zero (no
factoring of `e_d/(2p) (x - a/p y)` into one multiply). Composite
coefficients (`inv2p*a_over_p`, `inv2q*alpha_over_q`) are formed once per
quartet, one multiply each. Constants (`2 pi^(5/2)`, `1/2`) are compile-time.
TRANSCENDENTALS ARE NAMED, NOT HIDDEN: each quartet calls `boys` once (the
orders `0..m_max` of the leaves the DAG holds) and one `rsqrt(p+q)`; each
primitive pair calls one `exp`. None of them is counted as FLOPs; they are
reported beside the triple (`Cost.transcendentals`), per tier.

EQ 26's SIGN. #271's claim `fock/eq26-ket-vrr-sign` (asserted by #291) says eq 26's
`(m+1)` displacement is misprinted (`WQ = +zeta/(zeta+eta) PQ`, not `-`). A
sign does not change a FLOP count, so the claim cannot move any number here;
`WQ` is counted as the three multiplies either sign costs.

GEOMETRY IS READ OFF THE DAG TOO. A geometry step is counted only if a
coefficient the DAG reads (or the seed, which every class has) needs it:
`(ss|ss)` forms no `PA`, `WP` or `1/(2p)`. `GEOMETRY` is the table, each step
with its tier, FLOPs, named transcendental and the steps it reads; `needed`
is the closure.

PRUNING IS ON BY DEFAULT, and that is a decision: `cost(cls)` counts
`hgp.vrr.build(cls)`'s pruned DAG. The predecessor shipped the same saving
opt-in only because its FLOP ratchet sat downstream of the lowering; nothing
here is ratcheted, so there is no reason to count dead nodes.
`cost(cls, prune=False)` counts the forward build, and `report()` carries
both for every class -- the saving made visible, not asserted as a win.

THE HRR BUILD ORDER is `hgp.hrr.build`'s default, bra HRR first
(`first="bra"`); #274 pins both orders' statement counts, and this model
counts the default only.

`python -m felizjueves.hgp.cost` prints `report()` as the JSON tracked at
`data/hgp/cost.json` (test/test_hgp_cost.py holds the two equal).

Pure Python: no card, no compiler, no import from `plan/`, `reference/`,
`lower` or `ir`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from felizjueves.hgp import etr, hrr, vrr
from felizjueves.hgp.vrr import CLASSES, AmClass

TIERS = ("K4", "K2", "K0")


@dataclass(frozen=True)
class Step:
    """One geometry step: `flops` FLOPs at `tier`, reading `needs`."""

    tier: str
    flops: int
    needs: tuple[str, ...] = ()
    transcendental: str | None = None
    side: str = ""  # "bra" / "ket" for a K2 pair step, else ""
    formula: str = ""


def _pair(side: str) -> dict[str, Step]:
    """The primitive-pair steps of one side; the ket twin renames them."""
    a, b, p, P, A, Kab, mu, AB = (
        ("a", "b", "p", "P", "A", "Kab", "mu", "AB") if side == "bra"
        else ("c", "d", "q", "Q", "C", "Kcd", "nu", "CD"))
    inv, inv2 = f"inv_{p}", f"inv2{p}"
    pa = f"{P}{A}"
    return {
        p: Step("K2", 1, (), None, side, f"{p} = {a} + {b}"),
        inv: Step("K2", 1, (p,), None, side, f"{inv} = 1/{p}"),
        P: Step("K2", 12, (inv,), None, side,
                f"{P}_k = ({a} {A}_k + {b} {'B' if side == 'bra' else 'D'}_k)"
                f" {inv}, 3 axes x (2 mul + 1 add + 1 mul)"),
        pa: Step("K2", 3, (P,), None, side, f"{pa}_k = {P}_k - {A}_k"),
        inv2: Step("K2", 1, (inv,), None, side, f"{inv2} = (1/2) {inv}"),
        mu: Step("K2", 2, (inv,), None, side, f"{mu} = {a} {b} {inv}"),
        Kab: Step("K2", 4, (mu, f"{AB}2"), "exp", side,
                  f"{Kab} = c_{a} c_{b} {inv} exp(-{mu} |{AB}|^2): "
                  f"{mu}*|{AB}|^2, c_{a}*c_{b}, *{inv}, *exp"),
    }


def _separation(AB: str) -> dict[str, Step]:
    return {
        AB: Step("K0", 3, (), None, "", f"{AB}_k = {AB[0]}_k - {AB[1]}_k"),
        f"{AB}2": Step("K0", 5, (AB,), None, "",
                       f"|{AB}|^2 = sum_k {AB}_k^2: 3 mul + 2 add"),
    }


GEOMETRY: Mapping[str, Step] = MappingProxyType({
    **_pair("bra"), **_pair("ket"), **_separation("AB"), **_separation("CD"),
    "zeta_eta": Step("K4", 1, ("p", "q"), formula="p + q"),
    "inv_zeta_eta": Step("K4", 1, ("zeta_eta",), formula="1/(p + q)"),
    "eta_frac": Step("K4", 1, ("inv_zeta_eta", "q"), formula="q/(p + q)"),
    "zeta_frac": Step("K4", 1, ("inv_zeta_eta", "p"), formula="p/(p + q)"),
    "PQ": Step("K4", 3, ("P", "Q"), formula="PQ_k = P_k - Q_k"),
    "WP": Step("K4", 3, ("eta_frac", "PQ"),
               formula="WP_k = -q/(p+q) PQ_k (eq 25)"),
    "WQ": Step("K4", 3, ("zeta_frac", "PQ"),
               formula="WQ_k = +p/(p+q) PQ_k (eq 26 as derived; the printed "
                       "sign costs the same)"),
    "inv2pq": Step("K4", 1, ("inv_zeta_eta",), formula="(1/2) inv_zeta_eta"),
    "inv2p*a_over_p": Step("K4", 1, ("inv2p", "eta_frac"),
                           formula="inv2p * q/(p+q)"),
    "inv2q*alpha_over_q": Step("K4", 1, ("inv2q", "zeta_frac"),
                               formula="inv2q * p/(p+q)"),
    "rho": Step("K4", 1, ("p", "eta_frac"), formula="rho = p q/(p+q)"),
    "PQ2": Step("K4", 5, ("PQ",), formula="|PQ|^2: 3 mul + 2 add"),
    "T": Step("K4", 1, ("rho", "PQ2"), formula="T = rho |PQ|^2"),
    "boys": Step("K4", 0, ("T",), "boys", formula="F_0..F_mmax(T)"),
    "rsqrt": Step("K4", 0, ("zeta_eta",), "rsqrt",
                  formula="1/sqrt(p + q)"),
    "pref": Step("K4", 3, ("Kab", "Kcd", "rsqrt"),
                 formula="2 pi^(5/2) Kab Kcd rsqrt: 3 mul"),
})

# Every class forms the seed `pref F_m(T)`, whatever its DAG reads.
ALWAYS = ("pref", "boys")
# A DAG coefficient name -> the geometry step that forms it ("1" is free).
_COEF_STEP = {**{c: c for c in vrr.BRA + vrr.KET},
              **{f"AB_{k}": "AB" for k in hrr.AXES},
              **{f"CD_{k}": "CD" for k in hrr.AXES}}


def needed(roots: Iterable[str],
           table: Mapping[str, Step] = GEOMETRY) -> frozenset[str]:
    """The geometry steps `roots` read, transitively, roots included
    (`table` is `GEOMETRY` unless a second recurrence adds steps, #354)."""
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        s = todo.pop()
        if s not in seen:
            seen.add(s)
            todo.extend(table[s].needs)
    return frozenset(seen)


def vrr_node_flops(terms: tuple[vrr.Term, ...]) -> int:
    """One VRR step: a multiply per term (two when `|mult| > 1`) and `n - 1`
    adds. A leaf (no terms) costs 0 here; its seed multiply is counted apart."""
    if not terms:
        return 0
    return sum(1 + (abs(t.mult) != 1) for t in terms) + len(terms) - 1


def hrr_node_flops(terms: tuple[hrr.Term, ...]) -> int:
    """One HRR step: `coef` "1" free, each other coefficient a multiply, and
    `n - 1` adds. A boundary node (no terms) is a read: 0."""
    if not terms:
        return 0
    return sum(t.coef != "1" for t in terms) + len(terms) - 1


@dataclass(frozen=True)
class Cost:
    """The FLOPs of one class, by tier, each tier split into its sources."""

    cls: AmClass
    pruned: bool
    k4: Mapping[str, int]  # geometry, seed, vrr, accumulate
    k2: Mapping[str, int]  # bra, ket
    k0: Mapping[str, int]  # geometry, hrr
    transcendentals: Mapping[str, Mapping[str, int]]  # tier -> name -> calls
    boys_max_order: int
    steps: frozenset[str]  # the geometry steps counted

    @property
    def x(self) -> int:
        return sum(self.k4.values())

    @property
    def y(self) -> int:
        return sum(self.k2.values())

    @property
    def z(self) -> int:
        return sum(self.k0.values())

    @property
    def triple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)

    @property
    def total(self) -> int:
        """`x + y + z`: the FLOPs of an uncontracted (`K = 1`) quartet."""
        return self.x + self.y + self.z

    def at(self, k: int) -> int:
        """`x K^4 + y K^2 + z`: every shell of contraction degree `k`."""
        if type(k) is not int or k < 1:
            raise ValueError(f"contraction degree {k!r} is not an int >= 1")
        return self.x * k**4 + self.y * k**2 + self.z


def _geometry(steps: frozenset[str], tier: str, side: str = "",
              table: Mapping[str, Step] = GEOMETRY) -> int:
    return sum(table[s].flops for s in steps
               if table[s].tier == tier and table[s].side == side)


def cost(cls: AmClass, prune: bool = True) -> Cost:
    """The tiered FLOPs of `(la lb | lc ld)`: the VRR DAG (pruned unless
    `prune=False`) and the default (bra-first) HRR DAG."""
    return _cost(cls, vrr.build(cls, prune=prune), prune, GEOMETRY,
                 _COEF_STEP)


def _cost(cls: AmClass, v: vrr.VrrDag, prune: bool,
          table: Mapping[str, Step], coef_step: Mapping[str, str]) -> Cost:
    """`cost`'s count over the VRR-side DAG `v` (the ket cascade's, or the
    ETR's, #354), with its geometry `table` and coefficient map."""
    h = hrr.build(cls)
    coefs = {t.coef for ts in v.terms.values() for t in ts}
    coefs |= {t.coef for ts in h.terms.values() for t in ts} - {"1"}
    steps = needed([*ALWAYS, *(coef_step[c] for c in coefs)], table)
    leaves = [n for n in v.terms if vrr.is_leaf(n)]
    k4 = {
        "geometry": _geometry(steps, "K4", table=table),
        "seed": len(leaves),
        "vrr": sum(vrr_node_flops(t) for t in v.terms.values()),
        "accumulate": len(v.targets),
    }
    k2 = {"bra": _geometry(steps, "K2", "bra", table),
          "ket": _geometry(steps, "K2", "ket", table)}
    k0 = {"geometry": _geometry(steps, "K0", table=table),
          "hrr": sum(hrr_node_flops(t) for t in h.terms.values())}
    trans: dict[str, dict[str, int]] = {t: {} for t in TIERS}
    for s in sorted(steps):
        st = table[s]
        if st.transcendental:
            d = trans[st.tier]
            d[st.transcendental] = d.get(st.transcendental, 0) + 1
    return Cost(tuple(cls), prune, MappingProxyType(k4), MappingProxyType(k2),
                MappingProxyType(k0),
                MappingProxyType({t: MappingProxyType(d)
                                  for t, d in trans.items()}),
                max(n[2] for n in leaves), steps)


# ---------------------------------------------------------------------------
# The ETR as a second VRR (#354): its FLOPs, additive
# ---------------------------------------------------------------------------
#
# `hgp.etr.build(cls)` reaches the same boundary as `hgp.vrr.build(cls)` by
# eq 25 to |e| = L and the electron-transfer relation at m = 0. Its count is
# `cost`'s convention, unchanged: a node is one multiply per term (two when
# |mult| > 1) plus n - 1 adds, the transfer's -p/q a sign (free). Two
# quartet steps are new and live only in `ETR_GEOMETRY` (so `GEOMETRY`,
# `cost`, `report` and data/hgp/cost.json are untouched): `p/q` (one
# multiply by the ket pair's `1/q`; the kernel's one divide) and `etr_c_k =
# QC_k + (p/q) PA_k` (a multiply and an add per axis), both K^4. The ket's
# `1/(2q)` is `GEOMETRY`'s K^2 ket step, read by both lowering terms.

ETR_GEOMETRY: Mapping[str, Step] = MappingProxyType({
    **GEOMETRY,
    "p_over_q": Step("K4", 1, ("p", "inv_q"),
                     formula="p/q: the ETR's transfer coefficient"),
    "etr_c": Step("K4", 6, ("p_over_q", "PA", "QC"),
                  formula="etr_c_k = QC_k + (p/q) PA_k, 3 axes x (1 mul + "
                          "1 add)"),
})
_ETR_COEF_STEP = {**_COEF_STEP, **{c: c for c in etr.ETR}}


def etr_cost(cls: AmClass, prune: bool = True) -> Cost:
    """`cost` with the VRR side replaced by the ETR DAG (`hgp.etr.build`):
    the same HRR, the same boundary, the same convention (above)."""
    return _cost(cls, etr.build(cls, prune=prune), prune, ETR_GEOMETRY,
                 _ETR_COEF_STEP)


ABOUT = (
    "FLOPs per contracted quartet of each of milestone 12's 21 HGP classes, "
    "x K^4 + y K^2 + z, from src/felizjueves/hgp/cost.py (#276), which reads "
    "the VRR DAG (hgp.vrr, #273) and the bra-first HRR DAG (hgp.hrr, #274), "
    "never the lowered IR. AN ENGINEERING INSTRUMENT CALIBRATED AGAINST "
    "NOTHING: [Fock] publishes no FLOP table, so there is no floor, no "
    "ratchet and no published comparison. Convention: one +, -, *, / is one "
    "FLOP (no FMA); an integer coefficient of +-1 and a sign cost nothing; a "
    "VRR node costs one multiply per term (two when |mult| > 1) plus n - 1 "
    "adds, the terms of eqs 25-26 as hgp.vrr.reduce lists them; an HRR node "
    "one multiply and one add; transcendentals (boys, rsqrt per quartet; exp "
    "per primitive pair) are named under 'transcendentals', not counted. "
    "x (K^4): quartet geometry, one seed multiply per Boys leaf, every VRR "
    "step, one add per boundary value accumulated; y (K^2): the bra pair's "
    "and the ket pair's geometry; z (K^0): every HRR step and the shell-pair "
    "separations. 'pruned' (the default, by decision: nothing downstream is "
    "ratcheted) counts the pruned VRR DAG; 'unpruned' the forward build; "
    "'saving' is unpruned total - pruned total, reported, not asserted as a "
    "win. Eq 26's misprinted sign (fock/eq26-ket-vrr-sign, asserted) moves no "
    "count: a sign costs nothing.")


def _record(c: Cost) -> dict:
    return {"x": c.x, "y": c.y, "z": c.z, "total": c.total,
            "K4": dict(c.k4), "K2": dict(c.k2), "K0": dict(c.k0)}


def report() -> dict:
    """Every class's pruned and unpruned triple, as tracked in
    `data/hgp/cost.json`."""
    rows = []
    for cls in CLASSES:
        p, u = cost(cls), cost(cls, prune=False)
        rows.append({
            "class": list(cls),
            "pruned": _record(p),
            "unpruned": _record(u),
            "saving": u.total - p.total,
            "boys_max_order": p.boys_max_order,
            "transcendentals": {t: dict(d) for t, d in
                                p.transcendentals.items()},
        })
    return {"about": ABOUT, "hrr_first": "bra", "classes": rows}


def render() -> str:
    return json.dumps(report(), indent=1) + "\n"


if __name__ == "__main__":
    print(render(), end="")
