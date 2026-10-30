# `test/golden/numeric/` — the numerical golden store (#655)

libcint/PySCF arrays that tests compare against, one plain float64 `.npy`
per array, keyed by registry case (`test/registry/`, #648), with
`manifest.json` recording how each was produced. The schema, the recipe
(`ENTRIES`), the loader and the one comparison helper are
`test/oracle/numeric.py`; its docstring is the long form of this file.

This is **not** the text goldens next door (`../cuda/`, `../hip/`,
`../gpu/`, `../*_digests.json`). Those hold what the generator emits and
are checked with no compiler. Nothing here is kernel text.

## What is here

| Entry | Kind | Registry case | What |
|---|---|---|---|
| `j_h2o_sto3g_bse` | npy | `(mmd, j, 0, 4)` | J, H2O, `data/basis/sto-3g.json`, `h2o_sto3g_j`'s density |
| `j_h2o_631g_bse` | npy | `(mmd, j, 0, 4)` | J, H2O, `data/basis/6-31g.json`, `h2o_631g_j`'s density |
| `j_h2o_ccpvdz_bse` | npy | `(mmd, j, 0, 8)` | J, H2O, `data/basis/cc-pVDZ.json` unreduced, `h2o_ccpvdz_j`'s density |
| `metric_h2o_ccpvdz_jkfit_bse` | npy | `(mmd, metric, 0, 6)` | `(P\|Q)`, `data/aux_basis/cc-pVDZ-JKFIT.json` on H2O |
| `metric_h2o_def2_universal_jkfit_bse` | npy | `(mmd, metric, 0, 8)` | `(P\|Q)`, `data/aux_basis/def2-universal-JKFIT.json` on H2O |
| `j_h2_sto3g`, `j_h2o_sto3g`, `j_h2o_631g`, `j_h2o_ccpvdz`, `j_h10_chain_ccpvdz`, `j_co_ccpvtz` | fixture | `(mmd, j, 0, 0…12)` | each `_j` fixture's `vj` |
| `metric_h2o_ccpvtz_jkfit`, `metric_prim_spd_f_auxi` | fixture | `(mmd, metric, 0, 8 / 12)` | each metric set's `int2c2e_cart` |

An **npy** entry is computed over a basis file from `data/` (#650), read
exactly as the file states it. That means libcint's shell order and **no**
free-primitive reduction, which is how FreeBreakfast reads it. The molecule
and the density are **referenced** from a tracked `test/oracle/fixtures/`
case (the manifest names it and records the density's sha256), never
copied. The manifest also records the basis file's sha256 and FreeBreakfast
commit as its `provenance.json` gives them, plus PySCF, libcint and numpy
versions, date, host and command.

A **fixture** entry indexes an array that `test/oracle/fixtures/` already
holds, so the store has one key per golden without a second copy. Its
regeneration is still `oracle.generate`'s, with its own `oracle` test.

None of the npy arrays is a fixture's under another name. STO-3G and 6-31G
are BSE's rounding of those sets, not PySCF's library's, and differ from
the fixtures' `vj` by ~5e-8 relative. The cc-pVDZ general contraction left
unreduced gives different AOs (16% apart). `test_golden_numeric.py` checks
that no npy entry equals a fixture array.

## Using it

```python
from oracle import numeric as golden

golden.compare("j_h2o_ccpvdz_bse", J)            # fp64 tier, scale max|want|
golden.compare(name, M_sub, index=(rows, rows))  # a block of the golden
golden.basis(name), golden.atoms(name), golden.density(name)  # inputs, no PySCF
```

`compare` is `oracle.tolerance.assert_within` against the golden. A float32
result is judged at the fp32 tier and never at the fp64 one.

## Regenerating

```bash
uv sync --group oracle
```

```bash
PYTHONPATH=test uv run --group oracle python -m oracle.generate --numeric --regenerate
```

`--check` in place of `--regenerate`, or `pytest -m oracle
test/test_golden_numeric.py`, rebuilds each npy array in memory and requires
it **bit for bit** equal to the tracked file. A drift is news about PySCF
until proven otherwise. Regenerate only after deciding which side is right,
and say so in the PR. To add an entry, add it to `ENTRIES` in
`test/oracle/numeric.py`, then run `--regenerate <name>`.

## Budget

Plain `.npy`, uncompressed, on purpose. Git already deflates every blob, so
an `.npz` would save nothing in the repository (this seed is 296 KB on disk
and 135 KB deflated). The C++ harness (`test/harness/npy.h`, #649) reads
plain `.npy`.

- **One array: at most 256 KiB**, half the repo's 512 KiB
  `check-added-large-files` cap. That is about a 180 × 180 float64 matrix.
  Anything bigger is a sign the case is too big for a golden. Use a smaller
  molecule, or a row block as the metric fixtures do.
- **The whole store: at most 1 MiB** on disk, arrays plus manifest.
- **Index rather than copy.** If `test/oracle/fixtures/` has the array, add
  a fixture entry, not an npy one.

`test_golden_numeric.py` enforces both size limits.
