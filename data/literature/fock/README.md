# [Fock] §3, read and declared

`[Fock]` is entry 23 of `docs/references.md` (Palethorpe, Stocks and Barca,
JCTC **20**, 10424–10442, 2024; doi:10.1021/acs.jctc.4c00994). It was already a
reference here — `eri/k_exchange/k.py` and `kernel_k.py` cite its §4.3 as prior art for
`K` — and this directory, added by #271 for milestone 12, is the first time a
section of it is **reconstructed** rather than cited: §3, the HGP recurrence
(§3.1, eqs. 25–28) and the fused Fock kernel built on it (§3.2, eq. 29, Fig. 1).

The text read is the arXiv preprint, arXiv:2407.21445v1, whose section,
equation and figure numbers are the ones milestone 12 cites; `coverage.json`
records the provenance. The journal version was not available to the reading.

| File | What it holds |
|---|---|
| `coverage.json` | the reading, unit by unit: every numbered section, and inside §3 each of eqs. 25–29, each design decision of §3.2 (numbered by the reading, in text order, since the paper does not number them) and each of Fig. 1's four components; each unit `read` or `not-read`, with the records that cite it and a note in this repo's vocabulary |
| `claims.json` | the scorable non-numeric claims, one record each (`felizjueves.literature.claims`); declared `pending` by #271 before anything was built, each `reason` naming the rule that would score it, and scored by #291 against the built tree: an `asserted` record names the `claim`-marked test that holds it, on evidence no weaker than its declared rule; an `unverifiable` one says what would settle it and on what hardware |
| `limits.json` | what cannot be scored here and why (`felizjueves.literature.limits`): the tables are A100 / V100 timings, §6 is whole-program wall-clock, there is no published register table for the §3.2 kernel, no `[JEngine]` cell or floor moves, the screens' efficacy is unmeasured on compact fixtures (and what the pair stream's screens remove there is counted, #285, not compared), the cross-tree ptxas baseline is never a numeric witness (#289), the symmetry-versus-screening trade is not decided on our cards, and what the fused `F` does not reach: `f` and above, residency across SCF iterations, gradients, levers, batching, cooperative staging, and the 2x2's two off-diagonal cells (#293) |

There is **no `tables.json`**, by decision (milestone 12, pin 2): `[Fock]`'s
tables are timings on hardware this box does not have, so milestone 12 scores
no published cell.

## How the inventory stays closed

The same rules as `../jengine/`, checked per paper. `test/test_claims.py`
discovers every `<paper>/claims.json` under `data/literature/` and holds each
paper's records against the `claim`-marked tests whose ids carry that paper's
prefix, both ways; `test/test_fock_coverage.py` holds `coverage.json` against
`claims.json` both ways and pins the unit list. So:

- **a new record** must name its units in `source` (each token a unit name,
  or `eqs. N-M`), and each of those units' `claims` must gain its id in the
  same change;
- **a record moved to `asserted`** must name the `claim`-marked test that holds
  it, and that test must exist;
- **a new reading** of a `not-read` unit flips its status, and adds a record
  for any claim it finds or says in the note why it is not one.

The counts by status live in `claims.json` and nowhere else; recompute them
rather than quoting them. A doc that does quote them spells them with the key,
"N [Fock] records: A asserted, U unverifiable, P pending", and
`test/test_claims.py` holds every such quote to the file.

No record rests on `data/measured/fock_baseline.json` (#289): it is a quality
baseline of common ancestry, never a numeric witness, and
`test/test_fock_coverage.py` refuses a record that cites it.
