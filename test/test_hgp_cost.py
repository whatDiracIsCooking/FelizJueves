"""`hgp/cost.py` (#276): FLOPs read off the HGP DAGs, tiered `K^4 / K^2 / K^0`.

Fast tier: pure Python, no toolchain, no card. The model is calibrated
against nothing (`[Fock]` publishes no FLOP table), so the triples are pinned
from the code in `data/hgp/cost.json`; what is checked here is that the file
is what the code computes, that the tiers sum and are attributable, that the
default counts the pruned DAG, and that the module stays off the IR side.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from felizjueves.hgp import cost, hrr, vrr
from felizjueves.hgp.cost import GEOMETRY, Cost
from felizjueves.hgp.vrr import CLASSES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hgp" / "cost.json"
SRC = ROOT / "src" / "felizjueves" / "hgp" / "cost.py"


def test_the_tracked_file_is_what_the_code_computes() -> None:
    # Regenerate with `PYTHONPATH=src python -m felizjueves.hgp.cost`.
    assert DATA.read_text() == cost.render()


def test_the_file_pins_all_21_classes_with_the_convention_stated() -> None:
    doc = json.loads(DATA.read_text())
    assert [tuple(r["class"]) for r in doc["classes"]] == list(CLASSES)
    about = doc["about"]
    for phrase in ("one +, -, *, / is one FLOP",
                   "integer coefficient of +-1",
                   "transcendentals", "CALIBRATED AGAINST NOTHING",
                   "pruned", "unpruned"):
        assert phrase in about, phrase
    assert doc["hrr_first"] == "bra"


@pytest.mark.parametrize("cls", CLASSES)
def test_the_tiers_sum_to_the_total(cls: tuple[int, int, int, int]) -> None:
    for prune in (True, False):
        c = cost.cost(cls, prune=prune)
        assert c.x == sum(c.k4.values())
        assert c.y == c.k2["bra"] + c.k2["ket"]
        assert c.z == sum(c.k0.values())
        assert c.total == c.x + c.y + c.z == c.at(1)
        assert c.at(3) == 81 * c.x + 9 * c.y + c.z
        # Every counted geometry step lands in exactly one tier bucket.
        geo = sum(GEOMETRY[s].flops for s in c.steps)
        assert geo == c.k4["geometry"] + c.y + c.k0["geometry"]


def test_y_is_reconstructed_from_the_pair_geometry_steps_alone() -> None:
    # (pp|pp), spelled by hand from the formulas, without the GEOMETRY table:
    # p = a + b (1), 1/p (1), P_k = (a A_k + b B_k)/p (3 x 4 = 12),
    # PA_k = P_k - A_k (3), 1/(2p) (1), mu = a b / p (2),
    # Kab = c_a c_b /p exp(-mu |AB|^2) (mu*AB2, c_a*c_b, */p, *exp: 4).
    bra = 1 + 1 + 12 + 3 + 1 + 2 + 4
    c = cost.cost((1, 1, 1, 1))
    assert c.k2 == {"bra": bra, "ket": bra}
    assert c.y == 2 * bra == 48
    # ... and it is exactly the K2 steps the class reads.
    assert c.y == sum(GEOMETRY[s].flops for s in c.steps
                      if GEOMETRY[s].tier == "K2")
    # (ss|ss) reads no PA and no 1/(2p): 24 - 3 - 1 per side.
    assert cost.cost((0, 0, 0, 0)).k2 == {"bra": 20, "ket": 20}
    # (ss|sp): only the ket reads QC, and nothing reads 1/(2q).
    assert cost.cost((0, 0, 0, 1)).k2 == {"bra": 20, "ket": 23}


def test_ss_ss_by_hand() -> None:
    # K4: p+q, 1/(p+q), q/(p+q), PQ (3), rho, |PQ|^2 (5), T, pref (3) = 16;
    # one seed multiply, no VRR step, one accumulate.
    # K0: AB (3) + |AB|^2 (5), CD likewise; no HRR step.
    c = cost.cost((0, 0, 0, 0))
    assert dict(c.k4) == {"geometry": 16, "seed": 1, "vrr": 0, "accumulate": 1}
    assert dict(c.k0) == {"geometry": 16, "hrr": 0}
    assert c.triple == (18, 40, 16)
    assert {t: dict(d) for t, d in c.transcendentals.items()} == {
        "K4": {"boys": 1, "rsqrt": 1}, "K2": {"exp": 2}, "K0": {}}
    assert c.boys_max_order == 0


def test_a_vrr_step_costs_its_terms() -> None:
    # [0,0|2_x,0]^(0) from QC, WQ, inv2q (mult 1), inv2q*alpha_over_q (-1):
    # four multiplies, three adds.
    node = ((0, 0, 0), (2, 0, 0), 0)
    terms = vrr.reduce(node)
    assert [abs(t.mult) for t in terms] == [1, 1, 1, 1]
    assert cost.vrr_node_flops(terms) == 7
    # [0,0|3_x,0]: the lower term's mult is 2, one more multiply each.
    terms = vrr.reduce(((0, 0, 0), (3, 0, 0), 0))
    assert [abs(t.mult) for t in terms] == [1, 1, 2, 2]
    assert cost.vrr_node_flops(terms) == 9
    # A ket step with the e_d/(2(p+q)) term too: five terms.
    terms = vrr.reduce(((1, 0, 0), (1, 0, 0), 0))
    assert [t.coef for t in terms] == ["QC", "WQ", "inv2pq"]
    assert cost.vrr_node_flops(terms) == 5
    assert cost.vrr_node_flops(()) == 0


@pytest.mark.parametrize("cls", CLASSES)
def test_each_tier_is_attributable_to_its_dag(cls) -> None:
    c = cost.cost(cls)
    v, h = vrr.build(cls), hrr.build(cls)
    assert c.k4["vrr"] == sum(cost.vrr_node_flops(t) for t in v.terms.values())
    assert c.k4["seed"] == sum(map(vrr.is_leaf, v.terms))
    assert c.k4["accumulate"] == len(vrr.target_set(cls)) == h.n_boundary
    # An HRR step is one multiply and one add; the default order is bra-first.
    assert c.k0["hrr"] == 2 * h.n_interior
    assert c.k0["geometry"] == 16


def test_the_dd_dd_hrr_tier_is_274s_statement_count() -> None:
    # #274: 5,293 two-term updates at (dd|dd) under the bra-first order.
    assert cost.cost((2, 2, 2, 2)).k0["hrr"] == 2 * 5293


@pytest.mark.parametrize("cls", CLASSES)
def test_pruning_is_on_by_default_and_only_the_k4_tier_moves(cls) -> None:
    p, u = cost.cost(cls), cost.cost(cls, prune=False)
    assert p.pruned and not u.pruned
    assert cost.cost(cls).triple == p.triple
    assert p.x <= u.x and (p.y, p.z) == (u.y, u.z)
    doc = {tuple(r["class"]): r for r in json.loads(DATA.read_text())["classes"]}
    row = doc[cls]
    assert (row["pruned"]["x"], row["unpruned"]["x"]) == (p.x, u.x)
    assert row["saving"] == u.total - p.total


def test_the_saving_is_reported_not_asserted() -> None:
    # Classes with no bra VRR have nothing to prune; every other has some.
    saving = {cls: cost.cost(cls, prune=False).total - cost.cost(cls).total
              for cls in CLASSES}
    assert {c for c, s in saving.items() if s == 0} == {
        c for c in CLASSES if c[0] + c[1] == 0}
    assert saving[(2, 2, 2, 2)] == 9001


def test_a_contraction_degree_is_checked() -> None:
    c = cost.cost((0, 0, 0, 0))
    for bad in (0, -1, 1.5, None):
        with pytest.raises(ValueError):
            c.at(bad)  # type: ignore[arg-type]
    assert isinstance(c, Cost)


def test_every_dag_coefficient_has_a_geometry_step() -> None:
    names = set(vrr.BRA + vrr.KET) | (set(hrr.COEFS) - {"1"})
    for n in names:
        assert cost._COEF_STEP[n] in GEOMETRY, n
    for s in GEOMETRY.values():
        assert s.tier in cost.TIERS
        assert set(s.needs) <= set(GEOMETRY)
        assert (s.side != "") == (s.tier == "K2")


def test_cost_reads_the_dag_not_the_ir() -> None:
    # #276: PlanCost's sibling, not KernelCost's. It imports neither the
    # lowering nor the plan layer, and not the IR either: a DAG is its input.
    names: set[str] = set()
    for node in ast.walk(ast.parse(SRC.read_text())):
        if isinstance(node, ast.ImportFrom):
            names |= {node.module or ""} | {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
    parts = {p for n in names for p in n.split(".")}
    assert not parts & {"lower", "plan", "ir", "interp", "reference",
                        "KernelCost", "PlanCost"}, names
