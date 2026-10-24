"""The duplicated-work count, J against K (#202): every number recomputed.

`src/felizjueves/eri/k_exchange/kdup.py` counts, per K class and batch count, the work a
split duplicates for J and for K at the same `Spec.am`, from the plan's
partition and the `E` nonzero pattern alone; this file holds the tracked
record, data/literature/jengine/k_duplication.json, to that count.

THE TIERS. Fast tier, no marker, no toolchain, no card, no PySCF. Under the
milestone's rule that no test builds, interprets, compiles or runs a class
with L > 2, the rows with L <= 2 are recomputed here in the fast tier (the
plan admits 1 and 3 batches there, nothing finer) and the SUBGRID_LS rows
(L = 7, 10, 13, 16) by `test_subgrid_rows_recomputed`, marked `slow` -- plan
arithmetic only, but kept off the gate by that rule. The summary, the cell
set and the structural invariants are read off every stored row in the
fast tier.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from felizjueves.eri.k_exchange import kdup
from felizjueves.literature.claims import load_claims
from felizjueves.mmd.hermite import e_block, hermite_order
from felizjueves.mmd.kernel import k_spec, pq_spec, split
from felizjueves.performance.measured_store import SUBGRID, SUBGRID_LS
from felizjueves.plan.batching import in_batch, partition
from felizjueves.plan.pipeline import plan
from felizjueves.spec import Target

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "felizjueves"
CLAIMS = REPO / "data" / "literature" / "jengine" / "claims.json"
SMALL = [(L, n) for L, n in kdup.cells() if L <= 2 and kdup.admitted(L, n) is None]


@pytest.fixture(scope="module")
def stored() -> dict:
    return json.loads(kdup.DATA.read_text())


def _row(stored: dict, L: int, n: int) -> dict:
    hits = [r for r in stored["rows"] if (r["L"], r["n_batches"]) == (L, n)]
    assert len(hits) == 1, f"{len(hits)} rows for L = {L}, n_batches = {n}"
    return hits[0]


# --- the restated names ------------------------------------------------------


@pytest.mark.parametrize("L", range(17))
def test_the_class_is_kernel_k_spec(L) -> None:
    # A Spec built, nothing planned or built: k_spec only names the class.
    j, k = kdup.specs(L, 1)
    assert k == k_spec(L)
    assert j == replace(k, target=Target.J)
    assert split(j) == split(pq_spec(L))  # J's counts are pq_spec's too


def test_the_subgrid_is_the_stores() -> None:
    assert kdup.SUBGRID_LS == SUBGRID_LS
    assert set(kdup.BATCHES) == {s.n_batches for s in SUBGRID}


@pytest.mark.parametrize(("la", "lb"), [(a, b) for a in range(3) for b in range(3)])
def test_the_nonzero_pattern_is_the_hosts(la, lb) -> None:
    # `E^{ab}_t` at a generic geometry: exactly the entries kdup calls
    # structurally nonzero are nonzero, at every pair up to (dd).
    A, B, zeta = (0.1, -0.3, 0.7), (0.9, 0.4, -0.2), 1.3
    P = tuple((0.6 * A[d] + 0.7 * B[d]) / zeta for d in range(3))
    for a in kdup.cartesians(la):
        for b in kdup.cartesians(lb):
            vals = e_block(a, b, zeta, P, A, B)
            s = tuple(a[d] + b[d] for d in range(3))
            want = set(kdup.nonzero(s, la + lb))
            got = {t for t, v in zip(hermite_order(la + lb), vals, strict=True)
                   if v != 0.0}
            assert got == want, (a, b)


def test_a_dd_pair_writes_336_entries_of_1260() -> None:
    # The milestone description's figure, from the count's pattern: of the
    # 36 x 35 = 1260 slots of a (dd) block, 336 are structurally nonzero
    # (written) and 924 are absent -- the 336 is the PRESENT count, as
    # test_the_nonzero_pattern_is_the_hosts confirms against mmd.hermite.
    sums = kdup.row_sums(2, 2)
    cols = len(kdup.nonzero((4, 4, 4), 4))
    present = sum(m * len(kdup.nonzero(s, 4)) for s, m in sums.items())
    assert (sum(sums.values()) * cols, present) == (1260, 336)


# --- the record ---------------------------------------------------------------


def test_the_record_states_its_rule_and_cells(stored) -> None:
    assert stored["schema"] == 1
    assert stored["rule"] == kdup.RULE
    assert kdup.CLAIM in stored["about"]
    seen = [(r["L"], r["n_batches"]) for r in stored["rows"]]
    refused = [(r["L"], r["n_batches"]) for r in stored["refused"]]
    assert sorted(seen + refused) == sorted(kdup.cells())
    assert len(set(seen)) == len(seen)
    for r in stored["refused"]:
        assert r["reason"] == kdup.admitted(r["L"], r["n_batches"])
    for L, n in seen:
        assert kdup.admitted(L, n) is None


def test_the_plan_admits_only_one_and_three_batches_at_small_L() -> None:
    assert SMALL == [(0, 1), (1, 1), (1, 3), (2, 1), (2, 3)]


@pytest.mark.parametrize(("L", "n"), SMALL)
def test_small_rows_recomputed(stored, L, n) -> None:
    assert _row(stored, L, n) == json.loads(json.dumps(kdup.count(L, n)))


@pytest.mark.slow
@pytest.mark.parametrize("L", SUBGRID_LS)
def test_subgrid_rows_recomputed(stored, L) -> None:
    # Plan arithmetic only, but L > 2: run by #231, not the push gate.
    for n in kdup.BATCHES:
        if kdup.admitted(L, n) is None:
            assert _row(stored, L, n) == json.loads(json.dumps(kdup.count(L, n)))


def test_the_summary_is_read_off_the_rows(stored) -> None:
    assert stored["summary"] == kdup.summary(stored["rows"])
    assert kdup.dumps(stored) == kdup.DATA.read_text()


def test_the_partitioned_terms_are_never_duplicated(stored) -> None:
    # Each (p, q) has one r = p + q, hence one batch: J's products and K's
    # stage 1 are partitioned, and a one-batch row duplicates nothing.
    for r in stored["rows"]:
        assert r["j"]["dup_contraction_fma"] == 0
        assert r["k"]["dup_stage1_fma"] == 0
        assert r["j"]["contraction_fma"] == r["rows"]["n_p"] * r["rows"]["n_q"]
        if r["n_batches"] == 1:
            assert r["direction"] is None
            for side in ("j", "k", "recurrence"):
                assert all(v == 0 for key, v in r[side].items()
                           if key.startswith("dup_")), (r["L"], side)


def test_the_verdict_rule() -> None:
    assert kdup.verdict(["K>J", "K>J"]) == kdup.SUPPORTS
    assert kdup.verdict(["K=J", "K<J"]) == kdup.REFUTES
    assert kdup.verdict(["K>J", "K=J"]) == kdup.CANNOT_SEPARATE
    assert kdup.verdict([]) == kdup.CANNOT_SEPARATE
    assert kdup.direction(3, 3) == "K=J"


# --- an independent count at small L ------------------------------------------


def _brute(L: int, n: int) -> dict[str, int]:
    """Stage 2 and 3 and J's write-backs, row by row, no bitmask, no sums."""
    la, lc, lb, ld = kdup.k_class(L)
    lp, lq = la + lc, lb + ld
    batches = partition(L, n)
    P, Q = hermite_order(lp), hermite_order(lq)

    def nz(t, a, c):
        return all(t[d] <= a[d] + c[d] for d in range(3))

    def add(p, q):
        return tuple(p[d] + q[d] for d in range(3))

    s2 = s3 = jw = 0
    for B in batches:
        jw += sum(any(in_batch(add(p, q), *B) for q in Q) for p in P)
        for b in kdup.cartesians(lb):
            for d in kdup.cartesians(ld):
                held = [p for p in P if any(
                    nz(q, b, d) and in_batch(add(p, q), *B) for q in Q)]
                for a in kdup.cartesians(la):
                    for c in kdup.cartesians(lc):
                        k = sum(nz(p, a, c) for p in held)
                        s2 += k
                        s3 += k > 0
    return {"stage2_fma": s2, "stage3_fma": s3, "reduction_writes": jw}


@pytest.mark.parametrize(("L", "n"), SMALL)
def test_an_unoptimised_count_agrees(L, n) -> None:
    row = kdup.count(L, n)
    got = _brute(L, n)
    assert got == {"stage2_fma": row["k"]["stage2_fma"],
                   "stage3_fma": row["k"]["stage3_fma"],
                   "reduction_writes": row["j"]["reduction_writes"]}


@pytest.mark.parametrize(("L", "n"), SMALL)
def test_k_and_j_share_one_plan(L, n) -> None:
    j, k = kdup.specs(L, n)
    assert plan(k).cost == plan(j).cost


# --- the layering: reads the plan, feeds nothing -------------------------------


def test_the_count_imports_no_builder_and_no_toolchain() -> None:
    probe = ("import sys, felizjueves.eri.k_exchange.kdup; "
             "print(sorted(m for m in sys.modules if m.startswith('felizjueves')"
             " or m == 'numpy'))")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    out = subprocess.run([sys.executable, "-c", probe], env=env, check=True,
                         capture_output=True, text=True).stdout
    loaded = set(ast.literal_eval(out))
    banned = {"felizjueves." + m for m in (
        "kernel", "ir", "interp", "recurrence", "passes", "lower", "emit",
        "e_fragment", "kernel_dq", "kernel_jab", "kernel_k", "measured",
        "measured_store", "device", "ao", "reference", "launch", "dispatch")}
    assert not {m for m in loaded if m in banned
                or any(m.startswith(b + ".") for b in banned)}, loaded
    assert "numpy" not in loaded


def test_nothing_imports_the_count() -> None:
    for path in SRC.rglob("*.py"):
        if path.name == "kdup.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            assert not any(n.split(".")[-1] == "kdup" for n in names), path


def test_the_claim_it_counts_for_is_left_pending() -> None:
    # Scoring is E3's (#207): this issue moves no record.
    rec = {c.id: c for c in load_claims(CLAIMS)}[kdup.CLAIM]
    assert rec.status == "pending"
    assert "duplicated contraction work must be strictly larger" in rec.reason
