# `data/basis/`

The orbital basis sets of [FreeBreakfast](https://github.com/whatDiracIsCooking/FreeBreakfast),
this generator's main consumer, copied **verbatim** (#650) so that a test here
can run on the same sets FreeBreakfast does. `aux_basis/` beside it holds the
fitting sets, copied the same way.

| | |
|---|---|
| copied from | FreeBreakfast `data/basis/` at commit [`67a5bb4`](https://github.com/whatDiracIsCooking/FreeBreakfast/tree/67a5bb47cd9268a6d2c15978a6b3230e8b8b3c81/data/basis) (its `main`, 2026-10-29) |
| format | [Basis Set Exchange](https://www.basissetexchange.org/) (BSE) JSON, `molssi_bse_schema` 0.1 |
| reader | `felizjueves.ao.bse` (no normalisation); `test/oracle/basis.py`'s `from_bse` maps a set onto our convention |
| checked by | `test/test_ao_bse.py` (fast tier) |
| record | `provenance.json`: the commit, and per file its `name`, size, sha256 and original source |
| size | 1,913,439 bytes in 11 files |

## What is here, and where each came from

| File | `name` | Original source |
|---|---|---|
| `sto-3g.json` | `STO-3G` | BSE v1, revised 2018-06-19, as served |
| `6-31g.json` | `6-31G` | BSE v1, 2018-06-19; d shells from K relabelled `gto_spherical` |
| `6-31gstar.json` | `6-31G*` | BSE v1, 2018-06-19; d shells from Li relabelled `gto_spherical` |
| `6-31gstarstar.json` | `6-31G**` | BSE v1, 2018-06-19; d shells from Li relabelled `gto_spherical` |
| `def2-SVP.json`, `def2-TZVP.json` | `def2-SVP`, `def2-TZVP` | BSE v1, 2018-09-07, as served |
| `cc-pVDZ.json`, `cc-pVTZ.json` | `cc-pVDZ`, `cc-pVTZ` | BSE v1, 2018-10-03, as served (no K) |
| `aug-cc-pVDZ.json`, `aug-cc-pVTZ.json` | `aug-cc-pVDZ`, `aug-cc-pVTZ` | BSE v1, 2018-10-03, as served (no K, Ca) |
| `minao.json` | `MINAO` | **not BSE**: PySCF 2.14.0's `pyscf/gto/basis/minao.py`, converted (no K) |

Every set covers Z 1–36 but where the table says otherwise. Each document
keeps its BSE header (`description`, `references`, `revision_date`,
`revision_description`), so the provenance also travels inside the file.
FreeBreakfast's own `data/basis/README.md`, at the commit above, has the
download and conversion commands and the reasoning for each choice; it is not
copied, because its build instructions are FreeBreakfast's.

## What a Cartesian reader should know

**This tree is Cartesian throughout; FreeBreakfast is spherical-only.**
`ao.bse` records `function_type` and does not act on it, so a `gto_spherical`
d read here is six Cartesian functions, not five — exactly what the libcint
fixtures' `cart=True` does with the same tag. So FreeBreakfast's relabelling
of the Pople sets' d shells (Cartesian in BSE, spherical there) changes
nothing here, and an AO count here is not FreeBreakfast's from d up: H2O in
cc-pVDZ is 25 Cartesian AOs here and 24 spherical ones there.

**These are not, function for function, the sets the libcint fixtures were
written over.** `test/oracle/generate.py` used PySCF's own library, and:

- STO-3G and 6-31G agree shell for shell, but only to PySCF's rounding (at
  worst 9.3e-7 relative on an exponent, 5.2e-7 absolute on a coefficient).
- cc-pVDZ and cc-pVTZ agree bit for bit only after a reduction: BSE writes
  each general contraction over every primitive, including the ones that
  also stand alone as an uncontracted function, and PySCF's tables drop those
  components (cc-pVDZ's O s: BSE's three functions over nine primitives,
  PySCF's two over eight plus the 0.3023 alone). The span is the same and so
  is an SCF energy; the AOs, and so a J or K matrix, are not.
  `oracle.basis.free_primitive_reduced` is the reduction, and
  `test_ao_bse.py` pins both that it closes the gap and that the gap is real.

A fixture regenerated over these files, rather than PySCF's library, would
make the comparison exact without the reduction. None is yet.

## Refreshing

Copy FreeBreakfast's files over these, then rewrite `provenance.json` (the
commit, and each file's size and sha256); `test_ao_bse.py` fails until the
record matches the bytes. A refresh that changes a set's shells may also move
the comparison against the fixtures, which is the point of having it.
