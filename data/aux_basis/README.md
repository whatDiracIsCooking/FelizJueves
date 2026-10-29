# `data/aux_basis/`

The density-fitting (auxiliary) basis sets of
[FreeBreakfast](https://github.com/whatDiracIsCooking/FreeBreakfast), copied
**verbatim** (#650), beside the orbital sets in `../basis/` (whose README has
the reader, the Cartesian caveat and the comparison with the fixtures).

| | |
|---|---|
| copied from | FreeBreakfast `data/aux_basis/` at commit [`67a5bb4`](https://github.com/whatDiracIsCooking/FreeBreakfast/tree/67a5bb47cd9268a6d2c15978a6b3230e8b8b3c81/data/aux_basis) (its `main`, 2026-10-29) |
| format | BSE JSON, the same as `../basis/`, read by `felizjueves.ao.bse` |
| checked by | `test/test_ao_bse.py` (fast tier) |
| record | `provenance.json`: the commit, and per file its `name`, size, sha256 and original source |
| size | 15,234,880 bytes in 11 files; the acCD sets are 1.2–2.8 MB each |

## What is here, and where each came from

| File | `name` | Original source |
|---|---|---|
| `cc-pVDZ-JKFIT.json` | `cc-pVDZ-JKFIT` | PySCF 2.14.0's `cc-pvdz-jkfit.dat` (Molpro's `weigend_jkfit.libmol`), converted |
| `cc-pVTZ-JKFIT.json` | `cc-pVTZ-JKFIT` | PySCF 2.14.0's `cc-pvtz-jkfit.dat`, converted |
| `def2-universal-JKFIT.json` | `def2-universal-JKFIT` | PySCF 2.14.0's `def2-universal-jkfit.dat` (BSE's copy of Turbomole 7.3's), converted |
| `cc-pVDZ-acCD.json` | `cc-pVDZ-acCD` | **generated** by FreeBreakfast's `tools/atomic_cd` (Lehtola's acCD, 1e-7 / 1e-5, `--max-l 4`) over `../basis/cc-pVDZ.json` |
| `<orbital>-acCD-6-6.json` (7) | `<orbital>-acCD-6-6` | **generated** the same way at 1e-6 / 1e-6, `--max-l 6`, over cc-pVDZ, cc-pVTZ, aug-cc-pVDZ, aug-cc-pVTZ, def2-SVP, def2-TZVP and 6-31G** |

**The JKFIT sets are PySCF's tables, not BSE's**, so a PySCF reference fits
over the same numbers: `cc-pVTZ-JKFIT.json` reproduces the converted aux basis
of the DF fixtures (`h2o_ccpvtz_jkfit_metric`, `h2o_sto3g_jkfit_3c`) to
1e-14, with no reduction, which `test_ao_bse.py` checks. The acCD sets are
published nowhere; FreeBreakfast's `data/aux_basis/README.md`, at the commit
above, has their method, settings and measured DF errors.

No set here passes i (l = 6), the DF kernels' aux ceiling: `cc-pVDZ-JKFIT`
stops at f, `cc-pVTZ-JKFIT` and `cc-pVDZ-acCD` at g, and
`def2-universal-JKFIT` and the acCD-6-6 sets reach i.

## Refreshing

As `../basis/`: copy FreeBreakfast's files, rewrite `provenance.json`, and
`test_ao_bse.py` holds the two together.
