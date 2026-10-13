# [JEngine] tables

`[JEngine]` is entry 35 of `docs/references.md`. Every table here was
transcribed from the ChemRxiv v2 preprint (doi:10.26434/chemrxiv-2025-2lntb-v2),
PDF page 12 / printed page 11. Each JSON file records its own provenance. The
PDF is not tracked.

| File | Table | Quantity |
|---|---|---|
| `table1.json` | 1 | total FLOPs to evaluate every `[r](0)` |
| `table2.json` | 2 | maximum intermediates `[r](m)`, m > 0, per batch |
| `table3.json` | 3 | maximum targets `[r](0)` per batch |
| `limits.json` | — | what a model will structurally not predict, what cannot be reproduced here, or what was declined, with its mechanism: seventeen `StructuralLimit`s, namely the two refused `exact_given_inputs` (#108), the searched board's record (no residual since #172, the earlier spaces' residual kept as `previous_residual`), §4.1's comparison (#111), what the roofline answer cannot see (#114), the sort-key verdict (#147), the absent `Launch` IR and `LaunchCost`, the record-versus-shell-pair screen granularity, and what the on-card launch-plan build does not reach (#148); and milestone 7's three (#158): the pre- and post-contraction kernels as this repo's own design, because the paper defers them to `[Fock]`, the instruction form's reach (`E` in `Weighted`, and the four work-arounds), and what the generated chain on the cards does not reach; and #160's three: the seed order's spread over the 58 rules, and Alg. 3's re-expansion direction and deletion reach as reconstruction choices; and #164's: the sieve's spill pattern on our cards, which fails the published one on both; and #166's: the refusal to ship the tuned plan, on measured wall-clock |
| `claims.json` | — | the non-numeric claims, one record each, with status |
| `coverage.json` | — | the reading `claims.json` is the inventory of, unit by unit |
| `mild_vs_optimum.json` | — | MILD's gap against this tree's computed optimum, replacing §4.1's comparison (#111) |
| `seed_order_spread.json` | — | per cell, the spread of FLOPs and intermediates over the 58 seed-order rules, and Alg. 3's three open choices with the tests that pin them (#160) |

## The claims inventory, and how it stays closed

`claims.json` holds what `[JEngine, v2]` asserts **without** a number: a
tie-break, a worked example, a split's shape, a direction of change, a defect
in the text. Each record is `asserted` by a named `claim`-marked test, or
`pending` / `unverifiable` with its reason (`felizjueves.literature.claims`).
The counts by status live in the file and nowhere else; recompute them rather
than quoting them.

`coverage.json` records the reading (#120): every unit of the paper — the
abstract, each numbered section and subsection, each algorithm and each figure
caption — with exactly the records whose `source` cites it, and a note saying
what else the unit asserts and why that is not a record (content implemented
and tested elsewhere, a claim about a cited work, a number, the experiment's
conditions). An `eq. N` in a `source` resolves to the section holding it.
`test/test_claims.py` holds the two files against each other in both
directions and pins the unit list to the paper's structure, so:

- **a new record** must cite a unit by its `source`, and that unit's `claims`
  must gain its id, in the same change;
- **a new reading** of a unit that finds a claim must add the record, or say
  in the unit's note why it is not one.

Tables are not units, and a record may not cite one: Tables 1–3 are the
scoreboards, and a cell is never duplicated as a claim.

## Table 4 is deliberately absent

Table 4 is on the same page: the average spills (bytes of loads and stores per
kernel), or register utilisation where a kernel does not spill, for L = 7..16
and 1, 3, 6 and 9 batches. That is another 40 cells. It is **not transcribed,
and that is not an oversight.**

A FLOP or intermediate count can be computed from the recurrence alone, but a
spill count cannot. It is the output of an emitted kernel, run through a
particular compiler and version, for a particular register file. The paper's
numbers are from its own kernels on an A100, sm_80. Those observation
conditions are frozen into the table and cannot be recovered from it, so no
model in this repo can score against it. This project's own kernels, compilers
and cards (sm_86, gfx1200) are a different experiment, and their numbers must
never be compared against Table 4's or read as reproducing it.

This is also the one item on which the repo differs from its predecessor
`jgen`, which scores a register-pressure proxy against Table 4
(`docs/implementation/j_engine.md`, "Parity with the predecessor"). Which
plan ships is not a second difference: it was decided on measured grounds
(#166).

If Table 4 is ever wanted, for example to show that spills fall as the batch
count rises, record it as a `StructuralLimit` in `limits.json`, with the
observation conditions as its mechanism. Do not add it as a scored table.

The same applies to the bold cells in Tables 1–3, which mark the fastest
measured batch count at each L. They are timings on that A100, so they are not
recorded as values.
