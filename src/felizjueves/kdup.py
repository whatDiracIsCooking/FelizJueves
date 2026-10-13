"""The duplicated-work count: J against K, per class and batch count (#202).

    python -m felizjueves.kdup              # the record, as JSON
    python -m felizjueves.kdup --write      # rewrite DATA
    python -m felizjueves.kdup --markdown   # the per-cell table

(with `PYTHONPATH=src uv run` in front, as for every entry point here.)

THE QUESTION. `[JEngine, v2, §6]` says recomputing the fundamentals `[r]^(0)`
in several kernels "has more merit when applied to the calculation of the
exchange matrix K due to the additional contractions that must be completed
at the shell quartet level". Splitting a class into `n_batches` kernels makes
every kernel that writes the same output element redo some work. This module
counts that duplicated work for `Target.J` and for `Target.K` at the same
`Spec.am` and `n_batches`, with NO card, NO compiler and NO built kernel:
from `plan.batching`'s partition of the Hermite targets, the plan's own cost
(`plan.pipeline`, read and unchanged), and the structurally nonzero pattern
of the McMurchie-Davidson `E` -- `E^{ac}_p != 0` exactly when
`p_d <= a_d + c_d` on every axis `d`. It is the ARITHMETIC half of
`jengine/k-recompute-has-more-merit` (claims.json, whose `reason` fixes the
rule: "K's duplicated contraction work must be strictly larger than J's");
the measured half is E2's, and scoring the record is E3's (#207), not this
module's.

THE CLASSES. One K class per L, `kernel.k_spec`'s rule restated (this module
must not import the builder; test/test_k_duplication.py pins the two equal):
`(lp, lq) = (ceil(L/2), floor(L/2))`, each pair's order split over its two
shells larger first, `am = (la, lc, lb, ld)`, the Hermite pairing, bra first.
The J row of a cell uses the SAME `am` with `Target.J`: the plan depends on
`L` and `n_batches` only, and J's contraction on `(lp, lq)` only, so the J
counts are those of `pq_spec(L)` as well. The cells are L = 0, 1, 2 and
`SUBGRID_LS` (`measured_store.SUBGRID_LS`, restated and pinned likewise),
each at every `n_batches` in `BATCHES` the plan admits; a refused split
(`batching.partition`'s empty-batch refusal, #64) is listed with its reason,
never counted. At L <= 2 the plan admits 1 and 3 batches only (L = 0: 1).

THE COUNTING RULE, per primitive quartet (one bra pair record `(a,c)`, one
ket pair record `(b,d)`), stated before any number. A batch `B` holds the
Hermite targets `r = p + q` the partition gives it; it touches the bra
functions `P_B = {p : p + q in B for some q}`.

  shared (J and K alike: one plan, discipline 1 of the milestone)
    recurrence   `PlanCost.flops` at `n_batches` minus at one batch (the
                 sieved literal plan, plan/cost.py's FLOP convention); the
                 summed per-batch intermediates minus the one-batch count;
                 and `n_batches - 1` extra Boys evaluations (each batch
                 kernel evaluates the Boys function for itself,
                 `jengine/batch-overlap-is-fundamentals-and-axials`).
  J            `J_p += sum_q [p+q] D_q` (`D_q` pre-contracted per ket
               record by `DQ`, the back-transform `JAB` once per bra class:
               neither is split).
    contraction  the `(p, q)` products with `p + q in B`, summed over
                 batches. Each product has exactly one `r`, so the batches
                 PARTITION them: never duplicated (asserted, not assumed).
    reduction    `sum_B |P_B| - n_p`: the extra partial `J_p` write-backs a
                 split forces. Counted HIGH: as if paid per quartet (one ket
                 record per bra record), though the kernel accumulates `J_p`
                 over its ket loop and writes once per bra record.
  K            the staged contraction (milestone description), per batch:
    stage 1      `I[p] = sum_q [p|q] E^{bd}_q` per `(b,d)` row, only
                 `q <= b + d` (structurally nonzero `E^{bd}_q`) and
                 `p + q in B`. A partition again: never duplicated.
    stage 2      `V(a,c) = sum_p E^{ac}_p I[p]` per `(a,c)` row and `(b,d)`
                 row, only for `p` with `I[p]` structurally nonzero IN THIS
                 BATCH for that `(b,d)` row and `E^{ac}_p` structurally
                 nonzero. A `p` touched by several batches is contracted by
                 each: this is the duplication.
    stage 3      `K[a,b] += D[c,d] V(a,c)`, once per `(a,c)`, `(b,d)` row
                 and batch in which that `V` is structurally nonzero.
    Counted LOW: only structurally nonzero terms; no K write-back; no `E`
    re-evaluation (each batch kernel would recompute its `E` blocks, which
    the split K kernel of B3 decides, not this count).

  FLOPs: a multiply-add is 2 (plan/cost.py's convention 1), a partial
  write-back 1 (one add). DUPLICATED CONTRACTION FLOPs:
      J   1 * reduction_dup + 2 * contraction_dup
      K   2 * (stage1_dup + stage2_dup + stage3_dup)
  `ratio_contraction` is K's over J's; `ratio_total` adds the shared
  recurrence FLOPs to both. Both are null where the denominator is 0.

THE DIRECTION, per split cell: `K>J` if K's duplicated contraction FLOPs are
strictly larger, `K=J`, `K<J`. Per class, over its split cells: SUPPORTS if
`K>J` at every one, REFUTES if at none, CANNOT SEPARATE otherwise (or with no
split cell). Because J is counted high and K low, a `K>J` is robust to both
simplifications; a `K<=J` would not be, and is reported as found.

THIS FEEDS NOTHING. It reads the plan and builds no kernel; no module imports
it (test/test_k_duplication.py), so no count here reaches `plan`'s cost model,
the measured store or a literature floor; its rule (`count` in
test/test_layering.py) forbids those imports.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

from felizjueves.plan.batching import in_batch, partition
from felizjueves.plan.pipeline import plan
from felizjueves.plan.targets import R, hermite_targets
from felizjueves.spec import Operator, Spec, Target

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "literature" / "jengine" / "k_duplication.json"

SUBGRID_LS = (7, 10, 13, 16)  # measured_store.SUBGRID_LS, restated and pinned
SMALL_LS = (0, 1, 2)  # the fast tier's cells (the milestone's L <= 2 rule)
LS = SMALL_LS + SUBGRID_LS
BATCHES = (1, 3, 6, 9, 12)  # measured_store.SUBGRID's batch counts

SUPPORTS, REFUTES, CANNOT_SEPARATE = "supports", "refutes", "cannot separate"
CLAIM = "jengine/k-recompute-has-more-merit"


def k_class(L: int) -> tuple[int, int, int, int]:
    """`kernel.k_spec(L).am`, restated: `(la, lc, lb, ld)`, bra first."""
    if type(L) is not int or L < 0:
        raise ValueError(f"L = {L!r} is not an int >= 0")
    lp, lq = (L + 1) // 2, L // 2
    return ((lp + 1) // 2, lp // 2, (lq + 1) // 2, lq // 2)


def specs(L: int, n_batches: int) -> tuple[Spec, Spec]:
    """The J and the K `Spec` of one cell: the same `am`, the same count."""
    am = k_class(L)
    return (Spec(Operator.COULOMB, 0, am, Target.J, (), n_batches),
            Spec(Operator.COULOMB, 0, am, Target.K, (), n_batches))


def cartesians(l: int) -> tuple[R, ...]:  # noqa: E741
    """Every `(i, j, k)` with `i + j + k = l`, read off the plan's targets."""
    return tuple(t for t in hermite_targets(l) if sum(t) == l)


def nonzero(s: R, order: int) -> tuple[R, ...]:
    """The Hermite functions `t`, `|t| <= order`, with `E_t` structurally
    nonzero for a pair row whose per-axis sum is `s` (`t_d <= s_d` on every
    axis)."""
    return tuple(t for t in hermite_targets(order)
                 if all(t[d] <= s[d] for d in range(3)))


def row_sums(l1: int, l2: int) -> Counter[R]:
    """The per-axis sums `a + c` of a pair's rows, with their multiplicity."""
    return Counter(tuple(a[d] + c[d] for d in range(3))
                   for a in cartesians(l1) for c in cartesians(l2))


def admitted(L: int, n_batches: int) -> str | None:
    """None if the plan admits the split, else its refusal, verbatim."""
    try:
        partition(L, n_batches)
    except ValueError as e:
        return str(e)
    return None


def _batch_of(L: int, n_batches: int) -> dict[R, int]:
    batches = partition(L, n_batches)
    out: dict[R, int] = {}
    for r in hermite_targets(L):
        hits = [i for i, b in enumerate(batches) if in_batch(r, *b)]
        if len(hits) != 1:
            raise AssertionError(f"target {r} in {len(hits)} batches, not 1")
        out[r] = hits[0]
    return out


def _add(p: R, q: R) -> R:
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


@cache
def _recurrence(L: int, n_batches: int) -> dict[str, int]:
    j = specs(L, n_batches)[0]
    one = plan(specs(L, 1)[0])
    split = plan(j)
    inter = sum(b.intermediates for b in split.batches)
    inter_one = sum(b.intermediates for b in one.batches)
    return {
        "flops": split.cost.flops,
        "flops_one_batch": one.cost.flops,
        "dup_flops": split.cost.flops - one.cost.flops,
        "intermediates": inter,
        "intermediates_one_batch": inter_one,
        "dup_intermediates": inter - inter_one,
        "boys_evaluations": n_batches,
        "dup_boys_evaluations": n_batches - 1,
    }


def count(L: int, n_batches: int) -> dict[str, Any]:
    """One row: the cell's counts for both targets, per the module's rule."""
    reason = admitted(L, n_batches)
    if reason is not None:
        raise ValueError(f"L = {L}, n_batches = {n_batches} is refused: {reason}")
    la, lc, lb, ld = k_class(L)
    lp, lq = la + lc, lb + ld
    P, Q = hermite_targets(lp), hermite_targets(lq)
    pbit = {p: 1 << i for i, p in enumerate(P)}
    batch_of = _batch_of(L, n_batches)
    nb = n_batches

    # J: products (p, q) per batch, and the bra functions each batch touches.
    j_fma = [0] * nb
    j_touch = [0] * nb
    for p in P:
        for q in Q:
            b = batch_of[_add(p, q)]
            j_fma[b] += 1
            j_touch[b] |= pbit[p]
    j_out = [m.bit_count() for m in j_touch]

    # K: per distinct ket row sum s_bd, the I[p] each batch holds; per bra
    # row sum s_ac, the p with E^{ac}_p structurally nonzero.
    bra = row_sums(la, lc)
    ket = row_sums(lb, ld)
    e_ac = {s: sum(pbit[p] for p in nonzero(s, lp)) for s in bra}
    s1 = [0] * nb
    s2 = [0] * nb
    s3 = [0] * nb
    s2_one = s3_one = 0
    for s_bd, m_bd in ket.items():
        held = [0] * nb
        for q in nonzero(s_bd, lq):
            for p in P:
                b = batch_of[_add(p, q)]
                s1[b] += m_bd
                held[b] |= pbit[p]
        union = 0
        for b in range(nb):
            union |= held[b]
            for s_ac, m_ac in bra.items():
                hit = held[b] & e_ac[s_ac]
                s2[b] += m_bd * m_ac * hit.bit_count()
                s3[b] += m_bd * m_ac * (hit != 0)
        for s_ac, m_ac in bra.items():
            hit = union & e_ac[s_ac]
            s2_one += m_bd * m_ac * hit.bit_count()
            s3_one += m_bd * m_ac * (hit != 0)
    s1_one = sum(m * len(nonzero(s, lq)) * len(P) for s, m in ket.items())

    j = {
        "contraction_fma": sum(j_fma),
        "dup_contraction_fma": sum(j_fma) - len(P) * len(Q),
        "reduction_writes": sum(j_out),
        "dup_reduction_writes": sum(j_out) - len(P),
    }
    j["dup_contraction_flops"] = (j["dup_reduction_writes"]
                                  + 2 * j["dup_contraction_fma"])
    k = {
        "stage1_fma": sum(s1), "dup_stage1_fma": sum(s1) - s1_one,
        "stage2_fma": sum(s2), "dup_stage2_fma": sum(s2) - s2_one,
        "stage3_fma": sum(s3), "dup_stage3_fma": sum(s3) - s3_one,
    }
    k["dup_contraction_flops"] = 2 * (k["dup_stage1_fma"] + k["dup_stage2_fma"]
                                      + k["dup_stage3_fma"])
    rec = _recurrence(L, n_batches)
    jd, kd, rd = j["dup_contraction_flops"], k["dup_contraction_flops"], \
        rec["dup_flops"]
    js, ks = specs(L, n_batches)
    return {
        "L": L,
        "n_batches": n_batches,
        "am": list(k_class(L)),
        "lp": lp,
        "lq": lq,
        "keys": {"j": js.key(), "k": ks.key()},
        "rows": {"bra": sum(bra.values()), "ket": sum(ket.values()),
                 "n_p": len(P), "n_q": len(Q)},
        "recurrence": rec,
        "j": j,
        "k": k,
        "ratio_contraction": kd / jd if jd else None,
        "ratio_total": (kd + rd) / (jd + rd) if jd + rd else None,
        "direction": direction(jd, kd) if n_batches > 1 else None,
    }


def direction(j_dup: int, k_dup: int) -> str:
    return "K>J" if k_dup > j_dup else "K=J" if k_dup == j_dup else "K<J"


def cells() -> tuple[tuple[int, int], ...]:
    return tuple((L, n) for L in LS for n in BATCHES)


def verdict(directions: Iterable[str]) -> str:
    """SUPPORTS if `K>J` at every split cell, REFUTES at none, else neither."""
    d = list(directions)
    if d and all(x == "K>J" for x in d):
        return SUPPORTS
    if d and not any(x == "K>J" for x in d):
        return REFUTES
    return CANNOT_SEPARATE


def trend(ratios: list[float]) -> str:
    """How `ratio_contraction` moves with `n_batches` over a class's splits."""
    if len(ratios) < 2:
        return "one split"
    steps = [b - a for a, b in zip(ratios, ratios[1:], strict=False)]
    if all(s > 0 for s in steps):
        return "rising"
    if all(s < 0 for s in steps):
        return "falling"
    if all(s == 0 for s in steps):
        return "flat"
    return "mixed"


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per class and overall: the direction and the verdict, off the rows."""
    per: dict[str, Any] = {}
    for L in LS:
        split = [r for r in rows if r["L"] == L and r["n_batches"] > 1]
        dirs = [r["direction"] for r in split]
        ratios = [r["ratio_contraction"] for r in split
                  if r["ratio_contraction"] is not None]
        per[str(L)] = {
            "split_cells": [r["n_batches"] for r in split],
            "directions": dirs,
            "ratio_trend": trend(ratios),
            "verdict": verdict(dirs),
        }
    every = [r["direction"] for r in rows if r["n_batches"] > 1]
    sub = [r["direction"] for r in rows
           if r["n_batches"] > 1 and r["L"] in SUBGRID_LS]
    return {"per_class": per, "subgrid": verdict(sub), "all": verdict(every)}


RULE = (
    "Per primitive quartet, from plan.batching's partition and the E nonzero "
    "pattern (E^{ac}_p != 0 iff p_d <= a_d + c_d) only. Shared by J and K: "
    "the recurrence's extra FLOPs and intermediates (the sieved literal plan, "
    "plan/cost.py, at n_batches minus at one batch) and n_batches - 1 extra "
    "Boys evaluations. J: the J_p += [p+q] D_q products (partitioned by r, "
    "never duplicated) and sum_B |P_B| - n_p extra partial J_p write-backs, "
    "counted HIGH (as if per quartet). K: stage 1 (I[p] = sum_q [p|q] E^bd_q, "
    "partitioned), stage 2 (V(a,c) = sum_p E^ac_p I[p] over the p each batch "
    "holds for that (b,d) row) and stage 3 (K[a,b] += D[c,d] V(a,c) per "
    "batch with V structurally nonzero), counted LOW (nonzero terms only; no "
    "K write-back; no E re-evaluation). FLOPs: multiply-add 2, write-back 1. "
    "Duplicated = summed over batches minus the one-batch count. Direction "
    "per split cell: K>J iff K's duplicated contraction FLOPs exceed J's. "
    "Per class: supports if K>J at every admitted split, refutes if at none, "
    "cannot separate otherwise.")


def record() -> dict[str, Any]:
    rows, refused = [], []
    for L, n in cells():
        reason = admitted(L, n)
        if reason is None:
            rows.append(count(L, n))
        else:
            refused.append({"L": L, "n_batches": n, "reason": reason})
    return {
        "schema": 1,
        "about": (
            "The duplicated-work count of #202: for each K class and batch "
            "count, the work a split duplicates across the kernels that "
            "write the same output element, for J and for K at the same "
            "Spec.am, and their ratio -- the arithmetic half of "
            f"{CLAIM} (claims.json), counted with no card, no compiler and "
            "no built kernel. Written by `PYTHONPATH=src uv run python -m "
            "felizjueves.kdup --write`; recomputed by "
            "test/test_k_duplication.py (L <= 2 in the fast tier, L > 2 "
            "`slow`). It scores nothing: the claim is E3's (#207)."),
        "rule": RULE,
        "classes": ("kernel.k_spec(L): (lp, lq) = (ceil(L/2), floor(L/2)), "
                    "am = (la, lc, lb, ld) bra first; the J row at the same "
                    "am"),
        "rows": rows,
        "refused": refused,
        "summary": summary(rows),
    }


def dumps(rec: dict[str, Any]) -> str:
    return json.dumps(rec, indent=1, ensure_ascii=False) + "\n"


def markdown(rec: dict[str, Any]) -> str:
    out = ["| L | am | n | recurrence dup FLOPs | J dup FLOPs | K dup FLOPs "
           "| K/J contraction | K/J total | direction |",
           "|---|---|---|---|---|---|---|---|---|"]

    def fmt(x: float | None) -> str:
        return "--" if x is None else f"{x:.4g}"

    for r in rec["rows"]:
        if r["n_batches"] == 1:
            continue
        am = "({},{}\\|{},{})".format(*r["am"])  # escaped for the table
        out.append(
            f"| {r['L']} | {am} | {r['n_batches']} "
            f"| {r['recurrence']['dup_flops']} "
            f"| {r['j']['dup_contraction_flops']} "
            f"| {r['k']['dup_contraction_flops']} "
            f"| {fmt(r['ratio_contraction'])} | {fmt(r['ratio_total'])} "
            f"| {r['direction']} |")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    rec = record()
    if argv == ["--write"]:
        DATA.write_text(dumps(rec))
    elif argv == ["--markdown"]:
        sys.stdout.write(markdown(rec))
    elif not argv:
        sys.stdout.write(dumps(rec))
    else:
        sys.stderr.write("usage: python -m felizjueves.kdup "
                         "[--write | --markdown]\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
