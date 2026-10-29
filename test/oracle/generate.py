"""Write the tracked oracle fixtures from PySCF -- the only module that imports it.

    PYTHONPATH=test uv run --group oracle \
        python -m oracle.generate --regenerate [NAME ...]    # or --check

`--regenerate` rewrites `fixtures/<name>.json` (every case, or the named ones);
`--check` rebuilds in memory and compares against what is tracked, the same
comparison the `oracle`-marked test makes. Exit 0 agree / written, 1 drift,
2 could not run (PySCF absent -- `uv sync --group oracle`). The shape is
`felizjueves.performance.measured --remeasure`'s: one command rewrites the data
file, and the file carries who wrote it.

WHAT A FIXTURE IS NOT: the truth. It records what libcint computed, with the
versions, host and date in its provenance block. A drift is news about PySCF
until proven otherwise; regenerate only after deciding which side is right,
and say so in the PR that changes the file.

Adding a case is an entry in `CASES` plus `--regenerate <name>`. A case's
inputs (`mol`, `basis`, `cart`) are part of what `--check` compares; the
volatile provenance (versions, date, host) is not.

An "rhf" density (#486, `Case.density`) is the one input not recomputed on
`--check`: it is converged when its owner is regenerated, recorded, and read
back from the tracked file thereafter (`case_density`), so every array stays
bitwise reproducible; `--check` holds a fresh SCF to `RHF_DENSITY_TOL` of it.
Regenerate the owner (`h10_chain_ccpvdz_j`) and its `_k` together, in that
order: the `_k` reads the `_j`'s recorded density.

`--df-gradient` (with `--regenerate` or `--check`, no names) is the DF
gradient witnesses instead, both in `df_gradient/`: #613's
(`df_gradient_build`), libcint's derivative integrals on the aux-i sets'
inputs contracted per class, loaded by `manifest.df_gradient`; and #620's
(`df_rhf_build`), PySCF's DF-RHF two-electron gradient of H2O/STO-3G
against JKFIT's s and p shells, loaded by `manifest.df_rhf`.

`--k-gradient` (the same way) is the exchange-gradient witness instead
(#618, `k_gradient_build`): libcint's `int2e_ip1_cart` on `co_ccpvtz_j`'s
inputs contracted with its recorded density, per K class and whole,
written to `k_gradient/` and loaded by `manifest.k_gradient`.

A GRADIENT case (#550, `Case.record_gradient`, `<m>_grad`) records its
converged SCF (`mo_energy`, `mo_occ`, `mo_coeff`) the way an "rhf" density
is recorded, and computes everything else from it: the density, `W`, the
derivative integrals, `vj1` / `vk1`, each per-atom term and PySCF's total
(`gradient_arrays`). THE SIGN CONVENTIONS are `manifest`'s docstring's:
every libcint `ip` integral is `<nabla m| O |n>`, `nabla` the electron
coordinate's derivative on the FIRST function, which is `-d/dA` for `m` on
atom A -- so `grad.rhf.get_hcore` and `get_ovlp` are the integrals NEGATED,
`grad.rhf.get_jk` returns `-vj, -vk`, and each recorded term is PySCF's
`grad_elec` split by term. `manifest.check_gradient` recomputes every term
from the stored arrays and runs on every written file.

`--quadrupole` (the same way) is the quadrupole witness instead
(`quadrupole_build`): libcint's `int1e_rr_cart` on `hf_ccpvtz_spd_dipole`'s
inputs at its origin and at 0, written to `quadrupole/` and loaded by
`manifest.quadrupole_integrals`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import socket
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import (
    AO_SCHEMA,
    AUX_OVLP_ARRAY,
    COLS_SEP,
    CROSS_PREFIX,
    DIP0_ARRAY,
    DIP_ARRAY,
    DIP_CHARGES_ARRAY,
    DIP_COORDS_ARRAY,
    DIPOLE_ORIGIN_ARRAY,
    ERI_PREFIX,
    FIXTURES,
    GRAD_TOTAL,
    GRADIENT_SCHEMA,
    GRADIENT_TERMS,
    HCORE_ARRAY,
    J_ARRAY,
    K_ARRAY,
    KIN_ARRAY,
    METRIC_ARRAY,
    METRIC_SCHEMA,
    NO_DENSITY_RECIPE,
    NUC_ARRAY,
    ONE_E_HALVES,
    ONE_E_PIECE_SUFFIXES,
    OVLP_ARRAY,
    R0_ARRAY,
    R_ARRAY,
    ROWS_SEP,
    SCHWARZ_ARRAY,
    THREE_CENTER_PREFIX,
    THREE_CENTER_SCHEMA,
    dumps,
    encode_array,
    gradient_fixture,
    load,
)

COMMAND = ("PYTHONPATH=test uv run --group oracle python -m oracle.generate "
           "--regenerate")

# Provenance fields that legitimately differ between two honest runs.
VOLATILE = ("pyscf", "libcint", "numpy", "date", "host")


@dataclass(frozen=True)
class Case:
    description: str
    atom: str
    unit: str
    basis: str
    cart: bool
    charge: int = 0
    spin: int = 0
    # Write the `converted` slot: the basis in our convention (oracle.basis).
    record_basis: bool = False
    # Write `int2e:i,j,k,l` arrays: libcint's Cartesian ERI block for one
    # representative shell quartet per canonical angular-momentum class
    # (`eri_quartets`, #85). Needs `record_basis`: i, j, k, l index its shells.
    record_eri: bool = False
    # Only classes with min_eri_L <= la + lb + lc + ld <= max_eri_L. A
    # class's block size is fixed by its l's (no smaller representative
    # exists), and the repo's 512 KB check-added-large-files hook caps a
    # fixture: every class of an s-to-f basis is ~1.2 MB, L <= 8 is ~330 KB.
    # So one basis's classes are split across several cases by L (#121),
    # each a file under the cap, together covering every class once.
    max_eri_L: int | None = None
    min_eri_L: int | None = None
    # Write the `vj` array (#86): PySCF's `get_jk` Coulomb matrix for the
    # recorded `density`, so both sides of the J comparison see identical
    # input. Needs `record_basis` (our side builds its shells from it).
    record_j: bool = False
    # Write the `vk` array (#199): PySCF's `get_jk` exchange matrix for the
    # recorded `density`. Needs `record_basis`, as `record_j` does.
    record_k: bool = False
    # Write the one-electron arrays (#240): libcint's `int1e_ovlp`,
    # `int1e_kin`, `int1e_nuc` and PySCF's `hcore`. Only where `record_basis`
    # holds (our side builds its shells from it). Off for the five
    # HF/cc-pVTZ ERI files alone: they are `hf_ccpvtz`'s molecule and basis
    # exactly, which carries the four, and four 50 x 50 arrays would put
    # four of the five over the 512 KB hook (test_ao_eri.py pins it).
    record_one_e: bool = True
    # #570: write only this half of the four (`manifest.ONE_E_HALVES`, by
    # index): a ONE-ELECTRON PIECE, for a molecule whose four nao x nao
    # arrays do not fit one file under the 512 KB hook. Needs
    # `record_one_e`; the case is density-free.
    one_e_half: int | None = None
    # Write the `int2e_diag` array: libcint's `(mn|mn)` for every AO pair,
    # the square of the Schwarz factor (`eri/schwarz.py`). Needs
    # `record_basis`, as `record_j` does.
    record_schwarz: bool = False
    # #517: write the dipole arrays (manifest: THE DIPOLE FIXTURES) --
    # libcint's `int1e_r_cart` at `DIPOLE_ORIGIN` and at 0, and PySCF's RHF
    # `dip_moment` for the recorded density at both. Needs `record_basis`.
    record_dipole: bool = False
    # #520: write the dipole INTEGRALS only (`DIPOLE_INT_ARRAYS`): libcint's
    # `int1e_r_cart` at `DIPOLE_ORIGIN` and at 0, cross-checked against the
    # origin shift. For a density-free case (`density="none"`), which has no
    # `dip_moment` to record. Needs `record_basis` and `cart`.
    record_dipole_integrals: bool = False
    # #520: drop every library shell above this l (None keeps them all), so
    # a fixture holds nothing above it. The basis's name in `provenance`
    # then says so (`_basis_name`).
    max_l: int | None = None
    # Write `int1e_ovlp_cross:<rows>`: libcint's rectangular overlap
    # `<m|n>` with `m` an AO of case `rows` (another case, whose molecule
    # and basis are the ROWS) and `n` of this case (the COLUMNS) --
    # `gto.intor_cross`, the oracle of `S_{B1,B2}` (`Target.SX`). Needs
    # `record_basis` here and on `rows`.
    cross_rows: str | None = None
    # #411, a metric piece (manifest: THE METRIC PIECES): this case's
    # molecule and basis are the PARENT's (`parent`, a tracked case), and the
    # fixture is `df.make_auxmol(parent, aux_basis)`'s -- its shells, AOs and
    # rows [r0, r1) of libcint's `int2c2e_cart` and `int1e_ovlp`, full width.
    # No density and no AO arrays.
    aux_basis: str | None = None
    aux_rows: tuple[int, int] | None = None
    metric_set: str | None = None
    parent: str | None = None
    # #442, a three-center piece (manifest: THE THREE-CENTER PIECES): as a
    # metric piece, the fixture is the aux basis's, on `parent`'s molecule;
    # it records aux columns [c0, c1) of libcint's `int3c2e_cart` (`(mn|P)`,
    # m, n AOs of `parent`), full AO width. `aux_rows` stays None.
    three_center_set: str | None = None
    aux_cols: tuple[int, int] | None = None
    # #486: the recorded density's recipe -- "hilbert" (every case before
    # #486: `hilbert_density`, a function of nao alone) or "rhf" (PySCF's
    # converged RHF density, which decays with distance). An "rhf" density
    # is RECORDED and is the input from then on (`case_density`). "none"
    # (#520): the fixture records no density (manifest: THE DENSITY-FREE
    # DIPOLE FIXTURE), so nothing computed from one may be recorded either.
    density: str = "hilbert"
    # "rhf" only: the case whose tracked fixture owns the density. A `_k`
    # names its `_j`, so the pair's densities are one matrix, bit for bit
    # (`manifest.check_pair`); None means this case converges its own.
    density_from: str | None = None
    # #550: a GRADIENT fixture (schema 5, manifest: THE GRADIENT FIXTURES):
    # the converged SCF recorded, and the RHF nuclear gradient's pieces
    # (`gradient_arrays`). Needs `record_basis`, `cart` and density "rhf";
    # `gradient_sibling` is the `_j` case whose inputs it shares, or None.
    record_gradient: bool = False
    gradient_sibling: str | None = None


# A basis given inline rather than by library name: NWChem text, one
# "#BASIS SET" block per element (what `gto.basis.parse(text, symb)` splits on).
# One primitive per l = 0..3 on each element, with different exponents, so the
# single-primitive overlap is not one exponent pair repeated.
PRIM_SPDF = """\
#BASIS SET: one primitive per l = 0..3
He S
  1.3 1.0
He P
  0.9 1.0
He D
  0.7 1.0
He F
  0.55 1.0
#BASIS SET: one primitive per l = 0..3
Ne S
  2.1 1.0
Ne P
  1.4 1.0
Ne D
  1.1 1.0
Ne F
  0.8 1.0
"""


def hilbert_density(nao: int) -> np.ndarray:
    """D[i][j] = 1/(i+j+1): symmetric, positive definite, and no RNG.

    Each entry is one correctly rounded IEEE division, so the bytes are the
    same on any numpy and any host -- the density is an INPUT the later `J`
    comparison feeds both sides, and must not itself be a source of drift.
    Its entries are distinct off the symmetry, so a transposed or permuted
    index shows up in `J`.
    """
    i = np.arange(nao, dtype=np.float64)
    return 1.0 / (i[:, None] + i[None, :] + 1.0)


DENSITY_RECIPE = ("hilbert: D[i][j] = 1/(i+j+1) -- symmetric positive "
                  "definite, no RNG, reproducible bit-for-bit on any host")

# #486: the converged RHF density. Converging an SCF is not bit-reproducible
# (a thread count or a BLAS reorders the sums), so the density is RECORDED
# when its owner case is regenerated and is an input from then on: every
# array is computed from the recorded matrix, exactly as from a hilbert one.
# `--check` re-converges and holds the fresh density to RHF_DENSITY_TOL of
# the recorded one, a tolerance and never bitwise.
RHF_CONV_TOL = 1e-12  # energy, Eh
RHF_CONV_TOL_GRAD = 1e-10  # orbital gradient
RHF_DENSITY_TOL = 1e-8  # max |D_fresh - D_recorded|, --check only
RHF_MAX_CYCLE = 200  # #550's gradient cases (`rhf_mos`) only
RHF_DENSITY_RECIPE = (
    f"rhf: PySCF's converged closed-shell RHF density (total, 2 C_occ "
    f"C_occ^T, symmetrised as (D + D^T) / 2; conv_tol {RHF_CONV_TOL:g}, "
    f"conv_tol_grad {RHF_CONV_TOL_GRAD:g}), recorded; every array is "
    "computed from the recorded matrix, which --check holds to "
    f"{RHF_DENSITY_TOL:g} of a fresh SCF, not bitwise")
NO_DENSITY_RECIPE_TEXT = (
    f"{NO_DENSITY_RECIPE} a density-free fixture records no density and "
    "nothing computed from one (#520)")
DENSITY_RECIPES = {"hilbert": DENSITY_RECIPE, "rhf": RHF_DENSITY_RECIPE,
                   "none": NO_DENSITY_RECIPE_TEXT}

_H2O = "O 0 0 0; H 1.43 0.21 -1.07; H -1.38 0.35 -1.12"


def _hf_eri_by_L(L: int, classes: str) -> Case:
    """#121: HF/cc-pVTZ's classes with la + lb + lc + ld == L, one file per
    L = 9..12 -- the f-heavy blocks hf_ccpvtz_eri's 512 KB cap leaves out.
    Same molecule, basis and representative rule as hf_ccpvtz_eri, so the
    five files together hold all 55 canonical classes, each exactly once."""
    return Case(
        description=(
            f"#121's split of #85's s-to-f case: hf_ccpvtz_eri's molecule and "
            f"basis, with libcint's int2e for one representative shell "
            f"quartet per angular-momentum class with la + lb + lc + ld = "
            f"{L}: {classes}. With hf_ccpvtz_eri (L <= 8) and its L = 9..12 "
            f"siblings, all 55 classes of HF/cc-pVTZ, each file under the "
            f"512 KB hook."
        ),
        atom="F 0 0 0; H 0.4 -0.7 1.6",
        unit="Bohr",
        basis="cc-pvtz",
        cart=True,
        record_basis=True,
        record_eri=True,
        min_eri_L=L,
        max_eri_L=L,
        record_one_e=False,  # hf_ccpvtz carries them (Case.record_one_e)
    )

CASES: dict[str, Case] = {
    "h2_sto3g": Case(
        description=(
            "Seed fixture: H2 at R = 1.4 bohr on z, STO-3G, Cartesian. AO "
            "metadata, the libcint overlap, and a recorded symmetric density "
            "-- the schema's first user."
        ),
        atom="H 0 0 0; H 0 0 1.4",
        unit="Bohr",
        basis="sto-3g",
        cart=True,
    ),
    "prim_spdf": Case(
        description=(
            "#84's single-primitive case: He and Ne, one uncontracted "
            "primitive per l = 0..3 on each, the second atom off every axis "
            "so all Cartesian components couple. Records the converted basis "
            "and libcint's overlap."
        ),
        atom="He 0 0 0; Ne 0.3 -0.5 1.2",
        unit="Bohr",
        basis=PRIM_SPDF,
        cart=True,
        record_basis=True,
    ),
    "hf_ccpvtz": Case(
        description=(
            "#84's contracted case: HF in cc-pVTZ, Cartesian -- s to f on F, "
            "s to d on H, a general contraction (one F s shell carries two "
            "contractions) and H off every axis. Records the converted basis "
            "and libcint's overlap."
        ),
        atom="F 0 0 0; H 0.4 -0.7 1.6",
        unit="Bohr",
        basis="cc-pvtz",
        cart=True,
        record_basis=True,
    ),
    "h2o_631g_eri": Case(
        description=(
            "#85's contracted s/p case: H2O in 6-31G, Cartesian, no atom on "
            "an axis -- contraction depth (O 1s is six primitives) at low l, "
            "which is what exercises the dressing. Records the converted "
            "basis and libcint's int2e for one representative shell quartet "
            "per angular-momentum class."
        ),
        atom="O 0 0 0; H 1.43 0.21 -1.07; H -1.38 0.35 -1.12",
        unit="Bohr",
        basis="6-31g",
        cart=True,
        record_basis=True,
        record_eri=True,
    ),
    "hf_ccpvtz_eri": Case(
        description=(
            "#85's s-to-f case: hf_ccpvtz's molecule and basis (a general "
            "contraction, H off every axis), with libcint's int2e for one "
            "representative shell quartet per angular-momentum class with "
            "la + lb + lc + ld <= 8: 46 of the 55, f included, (dd|dd) and "
            "(ff|pp) the largest."
        ),
        atom="F 0 0 0; H 0.4 -0.7 1.6",
        unit="Bohr",
        basis="cc-pvtz",
        cart=True,
        record_basis=True,
        record_eri=True,
        max_eri_L=8,
        record_one_e=False,  # hf_ccpvtz carries them (Case.record_one_e)
    ),
    # #121: the nine classes over hf_ccpvtz_eri's cap, one file per L.
    "hf_ccpvtz_eri_l9": _hf_eri_by_L(9, "(ff|fs) (ff|dp) (fd|fp) (fd|dd)"),
    "hf_ccpvtz_eri_l10": _hf_eri_by_L(10, "(ff|fp) (ff|dd) (fd|fd)"),
    "hf_ccpvtz_eri_l11": _hf_eri_by_L(11, "(ff|fd)"),
    "hf_ccpvtz_eri_l12": _hf_eri_by_L(12, "(ff|ff)"),
    # #86's J cases: the recorded hilbert density, contracted by PySCF's
    # `get_jk`. One geometry for every H2O case (h2o_631g_eri's, no atom on
    # an axis), so the three bases differ in nothing else.
    "h2_sto3g_j": Case(
        description=(
            "#86's smallest J case: H2 in STO-3G, Cartesian, the bond off "
            "every axis. Records the converted basis and PySCF's get_jk J "
            "for the recorded density."
        ),
        atom="H 0 0 0; H 0.3 -0.4 1.3",
        unit="Bohr",
        basis="sto-3g",
        cart=True,
        record_basis=True,
        record_j=True,
    ),
    "h2o_sto3g_j": Case(
        description=(
            "#86: H2O in STO-3G, Cartesian, no atom on an axis -- s and p, "
            "three-primitive contractions. Records the converted basis and "
            "PySCF's get_jk J for the recorded density."
        ),
        atom=_H2O,
        unit="Bohr",
        basis="sto-3g",
        cart=True,
        record_basis=True,
        record_j=True,
    ),
    "h2o_631g_j": Case(
        description=(
            "#86's 6-31G case: H2O, Cartesian, no atom on an axis -- the O 1s "
            "six-primitive contraction and split valence. Records the "
            "converted basis and PySCF's get_jk J for the recorded density."
        ),
        atom=_H2O,
        unit="Bohr",
        basis="6-31g",
        cart=True,
        record_basis=True,
        record_j=True,
    ),
    "h2o_ccpvdz_j": Case(
        description=(
            "#86's d case (slow): H2O in cc-pVDZ, Cartesian (six d "
            "components), no atom on an axis, general contractions on O. "
            "Records the converted basis and PySCF's get_jk J for the "
            "recorded density."
        ),
        atom=_H2O,
        unit="Bohr",
        basis="cc-pvdz",
        cart=True,
        record_basis=True,
        record_j=True,
    ),
}


def _k_case(j_name: str) -> Case:
    """#199: the `_j` case's molecule, basis and `cart` exactly -- so the
    recorded hilbert density (a function of `nao` alone) and the converted
    basis are the `_j` fixture's, bit for bit -- with PySCF's `get_jk` K in
    place of J. A K disagreement then localises to the contraction, not to
    the input; `test_ao_k.py` checks the two files' inputs are identical."""
    j = CASES[j_name]
    return replace(
        j,
        description=(
            f"#199: {j_name}'s molecule, basis and recorded density, with "
            "PySCF's get_jk exchange matrix K (vk_il = sum_jk (ij|kl) D_jk) "
            "in place of J."
        ),
        record_j=False,
        record_k=True,
    )


# #199's K cases: one beside every `_j` case.
CASES.update({
    j[:-1] + "k": _k_case(j)
    for j in ("h2_sto3g_j", "h2o_sto3g_j", "h2o_631g_j", "h2o_ccpvdz_j")
})

# #486: five H2 on z (bond 1.4 bohr, 4.0 bohr between molecules), the
# atoms at z = 5.4 k + {0, 1.4}. Ten electrons, 50 Cartesian AOs; H20 would
# be 100 AOs, whose `_k` alone (density, ovlp, vk) is ~660 KB, over the
# 512 KB hook.
_H10_CHAIN = "; ".join(f"H 0 0 {round(5.4 * k + d, 1)}"
                       for k in range(5) for d in (0.0, 1.4))

CASES["h10_chain_ccpvdz_j"] = Case(
    description=(
        "#486: an H10 chain -- five H2 on z, bond 1.4 bohr, 4.0 bohr "
        "between molecules -- in cc-pVDZ, Cartesian, with PySCF's converged "
        "RHF density recorded: the first fixture whose density decays with "
        "distance (#484's per-block density factor can fire on it). Records "
        "the converted basis and PySCF's get_jk J for the recorded density."
    ),
    atom=_H10_CHAIN,
    unit="Bohr",
    basis="cc-pvdz",
    cart=True,
    record_basis=True,
    record_j=True,
    density="rhf",
)
CASES["h10_chain_ccpvdz_k"] = replace(
    _k_case("h10_chain_ccpvdz_j"), density_from="h10_chain_ccpvdz_j")


def _schwarz_case(j_name: str) -> Case:
    """The `_j` case's molecule, basis and `cart` exactly (so the converted
    basis is the `_j` fixture's, bit for bit), with libcint's `(mn|mn)` for
    every AO pair in place of J and no one-electron arrays -- the oracle of
    the Schwarz matrix, kept small."""
    j = CASES[j_name]
    return replace(
        j,
        description=(
            f"{j_name}'s molecule and basis, with libcint's (mn|mn) for every "
            "Cartesian AO pair (int2e's diagonal, int2e_diag): the square of "
            "the Schwarz factor."
        ),
        record_j=False,
        record_one_e=False,
        record_schwarz=True,
    )


# The Schwarz cases: s and p only, every class at L = 2 (la + lb) <= 4.
CASES.update({
    j[:-1] + "schwarz": _schwarz_case(j)
    for j in ("h2_sto3g_j", "h2o_sto3g_j", "h2o_631g_j")
})


# #517: the dipole fixtures' non-zero origin C, Bohr: every component
# non-zero, their magnitudes pairwise distinct and no atom on it, so `X_PC`
# is exercised and a swapped component order cannot pass.
DIPOLE_ORIGIN = (0.37, -0.81, 1.23)


def _dipole_case(j_name: str) -> Case:
    """#517, the `_schwarz` pattern: the `_j` case's molecule, basis, `cart`
    and density recipe exactly (so the converted basis and the recorded
    density are the `_j` fixture's, bit for bit; `manifest.check_dipole`
    asserts it), with the dipole arrays in place of J and no one-electron
    arrays."""
    j = CASES[j_name]
    return replace(
        j,
        description=(
            f"#517: {j_name}'s molecule, basis and recorded density, with "
            "libcint's int1e_r_cart (<m|r - C|n>, components x, y, z) under "
            f"mol.with_common_orig(C) at C = {DIPOLE_ORIGIN} bohr and at "
            "C = 0, and PySCF's RHF dip_moment (a.u., nuclear minus "
            "electronic) for the recorded density at both, with the nuclear "
            "charges and coordinates it used."
        ),
        record_j=False,
        record_one_e=False,
        record_dipole=True,
    )


# The dipole cases: one beside each H2O `_j` case, s/p to d.
CASES.update({
    j[:-1] + "dipole": _dipole_case(j)
    for j in ("h2o_sto3g_j", "h2o_631g_j", "h2o_ccpvdz_j")
})


# #520: the off-centre (dd) witness. #517's H2O fixtures have one d shell,
# on O, so every (dd) block libcint gives them is one-centre. HF/cc-pVTZ with
# its f shells removed (F 4s3p2d, H 3s2p1d; 40 Cartesian AOs) has d on both
# atoms, so every class (ss) through (dd) occurs between the two centres.
# No density, so no dip_moment: the integrals, at both origins, and the four
# one-electron arrays (`int1e_ovlp` for the origin-shift identity).
CASES["hf_ccpvtz_spd_dipole"] = replace(
    CASES["hf_ccpvtz"],
    description=(
        "#520: hf_ccpvtz's molecule in Cartesian cc-pVTZ with every f shell "
        "removed (F 4s3p2d, H 3s2p1d; nothing above d), so every class (ss) "
        "through (dd) occurs between the two centres. libcint's int1e_r_cart "
        "(<m|r - C|n>, components x, y, z) under mol.with_common_orig(C) at "
        f"C = {DIPOLE_ORIGIN} bohr and at C = 0, and the one-electron "
        "arrays. No density and no dip_moment."
    ),
    max_l=2,
    density="none",
    record_dipole_integrals=True,
)


def _cross_case(rows: str, cols: str, description: str, *,
                atom: str | None = None) -> Case:
    """`cols`' molecule, basis and `cart` (or the same basis at `atom`),
    with libcint's overlap against `rows`' AOs. Its own one-electron arrays
    are left to its `_j` twin when the inputs are the twin's exactly."""
    c = CASES[cols]
    return replace(
        c,
        description=description,
        atom=atom or c.atom,
        record_j=False,
        record_one_e=atom is not None,
        cross_rows=rows,
    )


# The cross-basis overlap cases (`Target.SX`): rows H2O/STO-3G (s, p), columns
# a larger basis -- the minimal-to-larger projection's overlap. Both
# orientations of every class are reached by the test reading the transpose.
CASES.update({
    "h2o_sto3g_x_631g": _cross_case(
        "h2o_sto3g_j", "h2o_631g_j",
        "S_{B1,B2}: H2O, rows STO-3G (h2o_sto3g_j's AOs), columns 6-31G "
        "(h2o_631g_j's inputs exactly), Cartesian: libcint's "
        "gto.intor_cross('int1e_ovlp'), one molecule in two bases."),
    "h2o_sto3g_x_ccpvdz": _cross_case(
        "h2o_sto3g_j", "h2o_ccpvdz_j",
        "S_{B1,B2}: H2O, rows STO-3G (h2o_sto3g_j's AOs), columns cc-pVDZ "
        "(h2o_ccpvdz_j's inputs exactly, d included), Cartesian: libcint's "
        "gto.intor_cross('int1e_ovlp')."),
    "h2o_sto3g_x_631g_moved": _cross_case(
        "h2o_sto3g_j", "h2o_631g_j",
        "S_{B1,B2} across two geometries: rows H2O/STO-3G (h2o_sto3g_j's "
        "AOs), columns H2O/6-31G displaced and distorted, so no column "
        "shell sits on a row shell's centre: libcint's "
        "gto.intor_cross('int1e_ovlp'). Carries its own one-electron arrays "
        "(no twin has its inputs).",
        atom="O 0.31 -0.22 0.47; H 1.62 0.55 -0.71; H -1.05 0.18 -0.93"),
})


def _metric_piece(parent: str, aux_basis: str, set_name: str,
                  rows: tuple[int, int]) -> Case:
    """Rows [r0, r1) of `aux_basis`'s metric on `parent`'s molecule."""
    c = CASES[parent]
    r0, r1 = rows
    return replace(
        c,
        description=(
            f"#411: the auxiliary basis {aux_basis} on {parent}'s molecule "
            f"(df.make_auxmol, Cartesian): rows [{r0}, {r1}) of libcint's "
            "two-centre Coulomb metric int2c2e_cart and of its overlap "
            f"int1e_ovlp, full width. With the other {set_name} pieces, the "
            "whole matrices, each file under the 512 KB hook "
            "(manifest.metric_set reassembles them)."
        ),
        record_basis=False, record_eri=False, record_j=False,
        record_k=False, record_one_e=False, record_schwarz=False,
        cross_rows=None, aux_basis=aux_basis, aux_rows=rows,
        metric_set=set_name, parent=parent,
    )


# #411's metric set: cc-pVTZ-jkfit (O to g, H to f; 45 shells, 166 AOs) on
# H2O/cc-pVDZ, split by row block at shell boundaries (ao_loc 43, 81, 121)
# into four files, each well under the 512 KB cap the whole matrix is over.
JKFIT_SET = "h2o_ccpvtz_jkfit_metric"
JKFIT_ROWS = ((0, 43), (43, 81), (81, 121), (121, 166))
CASES.update({
    f"{JKFIT_SET}{ROWS_SEP}{r0}_{r1}": _metric_piece(
        "h2o_ccpvdz_j", "cc-pvtz-jkfit", JKFIT_SET, (r0, r1))
    for r0, r1 in JKFIT_ROWS
})


def _three_center_piece(parent: str, aux_basis: str, set_name: str,
                        cols: tuple[int, int]) -> Case:
    """Aux columns [c0, c1) of `(mn|P)`, m, n of `parent`, P of
    `aux_basis` on `parent`'s molecule."""
    c = CASES[parent]
    c0, c1 = cols
    return replace(
        c,
        description=(
            f"#442: the auxiliary basis {aux_basis} on {parent}'s molecule "
            f"(df.make_auxmol, Cartesian): aux columns [{c0}, {c1}) of "
            f"libcint's three-center Coulomb integral int3c2e_cart (mn|P), "
            f"m and n AOs of {parent}, full AO width "
            "(manifest.three_center_set reassembles the tensor)."
        ),
        record_basis=False, record_eri=False, record_j=False,
        record_k=False, record_one_e=False, record_schwarz=False,
        cross_rows=None, aux_basis=aux_basis, aux_cols=cols,
        three_center_set=set_name, parent=parent,
    )


# #442's three-center set: cc-pVTZ-jkfit (166 aux AOs) against
# H2O/STO-3G's 7 AOs. 7 x 7 x 166 values are ~200 KB, under the 512 KB cap,
# so the set is one piece.
THREE_CENTER_SET = "h2o_sto3g_jkfit_3c"
THREE_CENTER_COLS = ((0, 166),)
CASES.update({
    f"{THREE_CENTER_SET}{COLS_SEP}{c0}_{c1}": _three_center_piece(
        "h2o_sto3g_j", "cc-pvtz-jkfit", THREE_CENTER_SET, (c0, c1))
    for c0, c1 in THREE_CENTER_COLS
})


# #606: the aux ladder through i. A small AO parent, `prim_spd_f` -- s, p
# and d on He, one contracted f on Ne (20 AOs), every pair class through
# (ff) -- and an inline aux basis with every l = 0..6 on BOTH atoms (168
# Cartesian aux AOs), so every eri3 class `la <= lb <= 3`, `lP <= 6` and
# every metric class through (i|i) occurs, each also with the aux shell on
# another centre than the bra pair's (the (ff) pair is one-centre on Ne; an
# i aux shell is also on He). The 168-column tensor of 20 x 20 AOs does not
# fit one file under the 512 KB hook, and an i shell alone (28 columns) does,
# so the tensor is split at aux shell boundaries into five pieces and the
# metric (two 168-wide arrays) by rows into four.
PRIM_SPD_F = """\
#BASIS SET: #606's parent: s, p, d on He, one contracted f on Ne
He S
  1.3 1.0
He P
  0.9 1.0
He D
  0.7 1.0
#BASIS SET: #606's parent: s, p, d on He, one contracted f on Ne
Ne F
  1.4 0.6
  0.55 0.5
"""
AUX_I = """\
#BASIS SET: #606's aux ladder, l = 0..6 on each atom
He S
  1.9 1.0
He P
  1.5 1.0
He D
  1.2 1.0
He F
  1.0 1.0
He G
  0.85 1.0
He H
  0.75 1.0
He I
  0.65 1.0
#BASIS SET: #606's aux ladder, l = 0..6 on each atom
Ne S
  2.4 1.0
Ne P
  1.9 1.0
Ne D
  1.5 1.0
Ne F
  1.2 1.0
Ne G
  1.0 1.0
Ne H
  0.85 1.0
Ne I
  1.6 0.6
  0.7 0.5
"""
CASES["prim_spd_f"] = Case(
    description=(
        "#606's AO parent for the aux ladder through i: He with one "
        "primitive s, p and d, Ne with one f contracted over two primitives, "
        "Ne off every axis. Every pair class (la <= lb <= 3) occurs. Records "
        "the converted basis and libcint's one-electron arrays."
    ),
    atom="He 0 0 0; Ne 0.3 -0.5 1.2",
    unit="Bohr",
    basis=PRIM_SPD_F,
    cart=True,
    record_basis=True,
)
AUX_I_METRIC_SET = "prim_spd_f_auxi_metric"
AUX_I_METRIC_ROWS = ((0, 56), (56, 104), (104, 140), (140, 168))
AUX_I_3C_SET = "prim_spd_f_auxi_3c"
AUX_I_3C_COLS = ((0, 35), (35, 84), (84, 119), (119, 140), (140, 168))
CASES.update({
    f"{AUX_I_METRIC_SET}{ROWS_SEP}{r0}_{r1}": _metric_piece(
        "prim_spd_f", AUX_I, AUX_I_METRIC_SET, (r0, r1))
    for r0, r1 in AUX_I_METRIC_ROWS
})
CASES.update({
    f"{AUX_I_3C_SET}{COLS_SEP}{c0}_{c1}": _three_center_piece(
        "prim_spd_f", AUX_I, AUX_I_3C_SET, (c0, c1))
    for c0, c1 in AUX_I_3C_COLS
})


# #570: the f witness of a whole-molecule build. CO in Cartesian cc-pVTZ
# (C and O each 4s3p2d1f; 18 shells, 70 AOs): f on BOTH centres and
# heteronuclear, so every class through (ff|ff) occurs between two
# different atoms and no A <-> B swap hides behind a symmetry. H2O or HF in
# cc-pVTZ carry f on one atom only, so their (ff|ff) is one-centre. The
# bond is ~2.10 bohr, off every axis. A 70 x 70 array is ~127 KB here, so
# a file holds three under the 512 KB hook: `_j` and `_k` the density, the
# legacy `ovlp` and vj / vk; the four one-electron arrays go in two
# density-free ONE-ELECTRON PIECES beside the `_j` (`manifest.one_e_fixture`
# reassembles them, refusing pieces whose inputs differ from the `_j`'s).
_CO = "C 0.13 -0.21 0.17; O 0.77 0.86 1.86"

CASES["co_ccpvtz_j"] = Case(
    description=(
        "#570's f case: CO in cc-pVTZ, Cartesian (ten f components), off "
        "every axis, f on both atoms. Records the converted basis and "
        "PySCF's get_jk J for the recorded density; the one-electron arrays "
        "are its two pieces', co_ccpvtz_j_st and co_ccpvtz_j_vh."
    ),
    atom=_CO,
    unit="Bohr",
    basis="cc-pvtz",
    cart=True,
    record_basis=True,
    record_j=True,
    record_one_e=False,  # in the pieces: four 70 x 70 arrays do not fit
)
CASES["co_ccpvtz_k"] = _k_case("co_ccpvtz_j")
CASES.update({
    f"co_ccpvtz_j{suffix}": replace(
        CASES["co_ccpvtz_j"],
        description=(
            f"#570: co_ccpvtz_j's molecule and basis, with half of the "
            f"one-electron arrays ({', '.join(half)}): a density-free "
            "one-electron piece, reassembled with its sibling by "
            "manifest.one_e_fixture('co_ccpvtz_j')."
        ),
        record_j=False,
        record_one_e=True,
        one_e_half=h,
        density="none",
    )
    for h, (half, suffix) in enumerate(zip(ONE_E_HALVES,
                                           ONE_E_PIECE_SUFFIXES,
                                           strict=True))
})


def _grad_case(j_name: str) -> Case:
    """#550: the `_j` case's molecule, basis and `cart` exactly (so the
    converted basis is the `_j` fixture's, bit for bit;
    `manifest.check_gradient` asserts it), with its own converged RHF and
    the gradient arrays in place of J and the one-electron arrays."""
    j = CASES[j_name]
    return replace(
        j,
        description=(
            f"#550: {j_name}'s molecule and basis with PySCF's converged RHF "
            "(mo_energy, mo_occ, mo_coeff recorded; density and W = "
            "make_rdm1e from them), libcint's int1e_ipovlp / ipkin / ipnuc "
            "and int1e_iprinv at each nucleus, grad.rhf.get_jk's vj1 / vk1, "
            "each per-atom term of the RHF nuclear gradient (h, S/Pulay, J, "
            "K, V_nn) and nuc_grad_method().kernel()'s total."
        ),
        record_j=False,
        record_one_e=False,
        density="rhf",
        record_gradient=True,
        gradient_sibling=j_name,
    )


# #550's gradient cases: one beside each H2O `_j` case (cc-pVDZ: d shells,
# so J/K classes to L = 8, read by `slow` tests only), and one low-symmetry
# s/p molecule with no `_j` sibling -- NH2F, C1 (no plane: the two N-H
# bonds differ), so no Cartesian component of any atom's gradient is zero
# by symmetry and a sign error has nowhere to hide.
CASES.update({
    j[:-1] + "grad": _grad_case(j)
    for j in ("h2o_sto3g_j", "h2o_631g_j", "h2o_ccpvdz_j")
})
CASES["nh2f_631g_grad"] = Case(
    description=(
        "#550's low-symmetry case: NH2F in 6-31G, Cartesian, C1 (the two "
        "N-H bonds differ, no atom on an axis, no plane of symmetry). "
        "PySCF's converged RHF, libcint's derivative integrals, vj1 / vk1, "
        "each per-atom term of the RHF nuclear gradient and its total."
    ),
    atom=("N 0 0 0; F 1.31 0.42 -2.18; H -1.02 1.57 0.33; "
          "H 0.48 -0.96 1.71"),
    unit="Bohr",
    basis="6-31g",
    cart=True,
    record_basis=True,
    record_one_e=False,
    density="rhf",
    record_gradient=True,
)


def _host() -> str:
    # The same marking felizjueves.performance.measured uses: a container's hostname is
    # its id, not the box's.
    name = socket.gethostname()
    return f"{name} (container)" if Path("/.dockerenv").exists() else name


def _libcint_version() -> str:
    import pyscf

    header = Path(pyscf.__file__).parent / "lib" / "deps" / "include" / "cint.h"
    try:
        m = re.search(r'#define\s+CINT_VERSION\s+"([^"]+)"', header.read_text())
    except OSError:
        m = None
    return m.group(1) if m else "unknown (no cint.h shipped with this pyscf)"


def canonical_classes(ls: Sequence[int]) -> list[tuple[int, int, int, int]]:
    """Every `(la lb|lc ld)` with la >= lb, lc >= ld, (la, lb) >= (lc, ld)
    over the angular momenta `ls` -- one per 8-fold-symmetry orbit."""
    lv = sorted(set(ls), reverse=True)
    pairs = [(a, b) for a in lv for b in lv if a >= b]
    return [(*bra, *ket) for bra in pairs for ket in pairs if bra >= ket]


def eri_quartets(shells: Sequence[Any], max_L: int | None = None,
                 min_L: int | None = None,
                 ) -> list[tuple[int, int, int, int]]:
    """One representative shell quartet per canonical class (with total
    angular momentum <= `max_L` and >= `min_L`, where given),
    deterministically.

    `shells` are `converted` records ({atom, l, exponents, ...}) or anything
    with those fields as keys. Among a class's quartets, prefer the most
    distinct atoms (geometry), then the most distinct shells (no accidental
    symmetry), then the most primitives (contraction depth), then the lowest
    indices.
    """
    by_l: dict[int, list[int]] = {}
    for i, s in enumerate(shells):
        by_l.setdefault(s["l"], []).append(i)

    def key(q: tuple[int, int, int, int]) -> tuple:
        atoms = len({shells[i]["atom"] for i in q})
        prims = sum(len(shells[i]["exponents"]) for i in q)
        return (-atoms, -len(set(q)), -prims, q)

    out = []
    for cls in canonical_classes(list(by_l)):
        if max_L is not None and sum(cls) > max_L:
            continue
        if min_L is not None and sum(cls) < min_L:
            continue
        cands = [(i, j, k, m) for i in by_l[cls[0]] for j in by_l[cls[1]]
                 for k in by_l[cls[2]] for m in by_l[cls[3]]]
        out.append(min(cands, key=key))
    return out


def eri_arrays(mol: Any, converted: dict[str, Any],
               max_L: int | None = None,
               min_L: int | None = None) -> dict[str, Any]:
    """libcint's Cartesian `int2e` for each of `eri_quartets`, cut to our
    shells' AO ranges -- one block per class, never the whole tensor."""
    shells = converted["shells"]
    # Our shell s is contraction `ctr[s]` of libcint shell `lib[s]`; within a
    # libcint shell the AOs are contraction-major (oracle.basis.from_mole).
    lib, ctr = [], []
    for i in range(mol.nbas):
        for c in range(int(mol.bas_nctr(i))):
            lib.append(i)
            ctr.append(c)
    out = {}
    for q in eri_quartets(shells, max_L, min_L):
        sl: list[int] = []
        for s in q:
            sl += [lib[s], lib[s] + 1]
        full = mol.intor("int2e", shls_slice=tuple(sl))  # cart=True: int2e_cart
        cut = []
        for s in q:
            n = (shells[s]["l"] + 1) * (shells[s]["l"] + 2) // 2
            cut.append(slice(ctr[s] * n, (ctr[s] + 1) * n))
        out[ERI_PREFIX + ",".join(str(s) for s in q)] = encode_array(
            np.ascontiguousarray(full[tuple(cut)]))
    return out


def _mole(case: Case) -> Any:
    """The PySCF `Mole` of `case`."""
    from pyscf import gto

    basis: Any = case.basis
    symbols = sorted({s.split()[0] for s in case.atom.split(";") if s.strip()})
    if "\n" in case.basis:  # inline NWChem text, not a library name
        basis = {s: gto.basis.parse(case.basis, symb=s) for s in symbols}
    if case.max_l is not None:  # #520: the library basis, high l dropped
        if isinstance(basis, str):
            basis = {s: gto.basis.load(basis, s) for s in symbols}
        basis = {s: [sh for sh in b if sh[0] <= case.max_l]
                 for s, b in basis.items()}
    return gto.M(atom=case.atom, unit=case.unit, basis=basis, cart=case.cart,
                 charge=case.charge, spin=case.spin, verbose=0)


def _basis_name(case: Case) -> str:
    """The basis as `provenance.basis` records it: the name, and with
    `max_l` (#520) the cut, so a truncated basis never reads as the
    library's."""
    if case.max_l is None:
        return case.basis
    return f"{case.basis}, shells with l <= {case.max_l} only"


def rhf_density(mol: Any) -> np.ndarray:
    """#486: PySCF's converged closed-shell RHF density (the total density,
    `make_rdm1`), symmetrised so it is exactly symmetric -- `(D + D^T) / 2`
    is, because floating-point addition commutes. Refuses an SCF that did
    not converge."""
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = RHF_CONV_TOL
    mf.conv_tol_grad = RHF_CONV_TOL_GRAD
    mf.kernel()
    if not mf.converged:
        raise ValueError("RHF did not converge")
    dm = np.asarray(mf.make_rdm1(), dtype=np.float64)
    return 0.5 * (dm + dm.T)


def rhf_mos(mol: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """#550: PySCF's converged closed-shell RHF `(mo_energy, mo_coeff,
    mo_occ)`, as `rhf_density` converges it. Refuses an SCF that did not
    converge. Up to `RHF_MAX_CYCLE` iterations: NH2F/6-31G, far from its
    equilibrium on purpose, takes 56 to these tolerances, over PySCF's 50."""
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = RHF_CONV_TOL
    mf.conv_tol_grad = RHF_CONV_TOL_GRAD
    mf.max_cycle = RHF_MAX_CYCLE
    mf.kernel()
    if not mf.converged:
        raise ValueError("RHF did not converge")
    return (np.asarray(mf.mo_energy, dtype=np.float64),
            np.asarray(mf.mo_coeff, dtype=np.float64),
            np.asarray(mf.mo_occ, dtype=np.float64))


def case_mos(name: str, mol: Any, *, fresh: bool = False,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """#550: a gradient case's `(mo_energy, mo_coeff, mo_occ)`: the RECORDED
    ones from its tracked fixture, or a fresh `rhf_mos` when regenerated
    (`fresh`) or not yet tracked -- an "rhf" density's rule."""
    path = FIXTURES / f"{name}.json"
    if fresh or not path.exists():
        return rhf_mos(mol)
    a = load(path).arrays
    return (np.array(a["mo_energy"]), np.array(a["mo_coeff"]),
            np.array(a["mo_occ"]))


def mos_density(mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
    """PySCF's RHF `make_rdm1` of recorded orbitals, symmetrised so it is
    exactly symmetric (`rhf_density`'s rule)."""
    from pyscf import scf

    dm = np.asarray(scf.hf.make_rdm1(mo_coeff, mo_occ), dtype=np.float64)
    return 0.5 * (dm + dm.T)


def _tracked_density(name: str) -> np.ndarray | None:
    path = FIXTURES / f"{name}.json"
    return np.array(load(path).density) if path.exists() else None


def case_density(name: str, mol: Any, *,
                 fresh: bool = False) -> np.ndarray | None:
    """The density `name`'s arrays are computed from.

    hilbert: `hilbert_density(nao)`. none (#520): None, no density.
    rhf: the RECORDED matrix -- the tracked
    fixture of `density_from` (a `_k`'s `_j`, which must be tracked first),
    or of `name` itself; a fresh `rhf_density` only for an owner regenerated
    (`fresh`) or not yet tracked.
    """
    case = CASES[name]
    nao = int(mol.nao_nr())
    if case.density == "hilbert":
        return hilbert_density(nao)
    if case.density == "none":
        return None
    if case.density != "rhf":
        raise ValueError(f"{name}: unknown density recipe {case.density!r}")
    if case.record_gradient:  # #550: the recorded orbitals' density
        _, c, occ = case_mos(name, mol, fresh=fresh)
        return mos_density(c, occ)
    if case.density_from is not None:
        dm = _tracked_density(case.density_from)
        if dm is None:
            raise ValueError(f"{name}: its density is {case.density_from}'s "
                             f"recorded one; regenerate {case.density_from} "
                             "first")
    else:
        dm = None if fresh else _tracked_density(name)
        if dm is None:
            dm = rhf_density(mol)
    if dm.shape != (nao, nao):
        raise ValueError(f"{name}: density has shape {dm.shape}, nao {nao}")
    return dm


def build(name: str, *, fresh_density: bool = False) -> dict[str, Any]:
    """The fixture object for `name`, computed now by PySCF/libcint, from
    `case_density(name, mol, fresh=fresh_density)`."""
    import pyscf

    case = CASES[name]
    mol = parent_mol = _mole(case)
    if case.aux_basis is not None:
        from pyscf import df

        aux_basis: Any = case.aux_basis
        if "\n" in aux_basis:  # #606: inline NWChem text, per element
            from pyscf import gto

            aux_basis = {s: gto.basis.parse(aux_basis, symb=s)
                         for s in sorted({mol.atom_pure_symbol(i)
                                          for i in range(mol.natm)})}
        mol = df.make_auxmol(mol, aux_basis)
        if not mol.cart:
            raise ValueError(f"{name}: make_auxmol did not inherit cart")

    coords = mol.atom_coords(unit="Bohr")
    atoms = [{"symbol": mol.atom_pure_symbol(i),
              "charge": int(mol.atom_charge(i)),
              "coord": [float(x) for x in coords[i]]}
             for i in range(mol.natm)]

    # Mole._basis is the basis as parsed from the library, keyed by atom
    # symbol and NOT normalised; libcint's shells are atom-major in this order.
    shells = []
    for ia in range(mol.natm):
        for entry in mol._basis[mol.atom_symbol(ia)]:
            ang, rows = entry[0], entry[1:]
            if not isinstance(rows[0], list | tuple):
                raise ValueError(f"{name}: kappa-carrying shell not supported")
            shells.append({
                "atom": ia,
                "l": int(ang),
                "exponents": [float(r[0]) for r in rows],
                "coefficients": [[float(c) for c in r[1:]] for r in rows],
            })
    if len(shells) != mol.nbas:
        raise ValueError(f"{name}: {len(shells)} shells parsed, mol has {mol.nbas}")

    nao = int(mol.nao_nr())
    ao_loc = mol.ao_loc_nr(cart=case.cart)
    mos = None
    if case.record_gradient:  # #550: one SCF, the density derived from it
        mos = case_mos(name, mol, fresh=fresh_density)
        dm = mos_density(mos[1], mos[2])
    else:
        dm = case_density(name, mol, fresh=fresh_density)
    out = {
        "schema": AO_SCHEMA,
        "name": name,
        "description": case.description,
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": _libcint_version(),
            "numpy": np.__version__,
            "date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
            "host": _host(),
            "command": f"{COMMAND} {name}",
            "mol": {"atom": case.atom, "unit": case.unit,
                    "charge": case.charge, "spin": case.spin},
            "basis": case.aux_basis or _basis_name(case),
            "cart": case.cart,
        },
        "atoms": atoms,
        "shells": shells,
        "nao": nao,
        "ao_labels": list(mol.ao_labels()),
        "ao_loc": [int(x) for x in ao_loc],
        "arrays": {
            "density": encode_array(dm),
            "ovlp": encode_array(mol.intor("int1e_ovlp")),
        } if dm is not None else {
            "ovlp": encode_array(mol.intor("int1e_ovlp")),
        },
        "density_recipe": DENSITY_RECIPES[case.density],
    }
    if dm is None and (case.record_j or case.record_k or case.record_dipole):
        raise ValueError(f"{name}: no density, so no vj, vk or dip_moment")
    if case.aux_basis is not None and case.three_center_set is not None:
        return _three_center_fixture(name, case, parent_mol, mol, out)
    if case.aux_basis is not None:
        return _metric_fixture(name, case, mol, out)
    if case.record_basis:
        from .basis import converted_record

        out["converted"] = converted_record(mol)
    if case.record_eri:
        if not (case.record_basis and case.cart):
            raise ValueError(f"{name}: record_eri needs record_basis and cart")
        out["arrays"].update(eri_arrays(mol, out["converted"],
                                        case.max_eri_L, case.min_eri_L))
    if case.record_j:
        if not case.record_basis:
            raise ValueError(f"{name}: record_j needs record_basis")
        out["arrays"][J_ARRAY] = encode_array(j_array(mol, dm))
    if case.record_k:
        if not case.record_basis:
            raise ValueError(f"{name}: record_k needs record_basis")
        out["arrays"][K_ARRAY] = encode_array(k_array(mol, dm))
    if case.record_schwarz:
        if not case.record_basis:
            raise ValueError(f"{name}: record_schwarz needs record_basis")
        out["arrays"][SCHWARZ_ARRAY] = encode_array(schwarz_array(mol))
    if case.record_dipole:
        if not (case.record_basis and case.cart):
            raise ValueError(f"{name}: record_dipole needs record_basis and "
                             "cart")
        out["arrays"].update(dipole_arrays(mol, dm, DIPOLE_ORIGIN))
    if case.record_dipole_integrals:
        if not (case.record_basis and case.cart) or case.record_dipole:
            raise ValueError(f"{name}: record_dipole_integrals needs "
                             "record_basis and cart, and not record_dipole")
        out["arrays"].update(dipole_integral_arrays(mol, DIPOLE_ORIGIN))
    if case.record_gradient:
        if not (case.record_basis and case.cart and case.density == "rhf"
                and not case.record_one_e):
            raise ValueError(f"{name}: record_gradient needs record_basis, "
                             "cart, density rhf and no one-electron arrays")
        out["schema"] = GRADIENT_SCHEMA
        out["gradient"] = {"sibling": case.gradient_sibling}
        del out["arrays"]["ovlp"]  # the sibling's; not a gradient array
        if case.gradient_sibling is not None:
            out["converted"] = sibling_converted(case.gradient_sibling,
                                                 out["converted"])
        assert mos is not None and dm is not None
        out["arrays"].update(gradient_arrays(mol, *mos, dm))
    if case.cross_rows is not None:
        rows = CASES[case.cross_rows]
        if not (case.record_basis and rows.record_basis):
            raise ValueError(f"{name}: cross_rows needs record_basis on both")
        if rows.cart != case.cart:
            raise ValueError(f"{name}: rows and columns differ in cart")
        out["arrays"][CROSS_PREFIX + case.cross_rows] = encode_array(
            cross_overlap_array(_mole(rows), mol))
    if case.record_basis and case.record_one_e:
        one_e = one_e_arrays(mol)
        if case.one_e_half is not None:
            if dm is not None:
                raise ValueError(f"{name}: a one-electron piece is "
                                 "density-free")
            one_e = {k: one_e[k] for k in ONE_E_HALVES[case.one_e_half]}
        out["arrays"].update(one_e)
    elif case.one_e_half is not None:
        raise ValueError(f"{name}: one_e_half needs record_basis and "
                         "record_one_e")
    return out


def _metric_fixture(name: str, case: Case, mol: Any,
                    out: dict[str, Any]) -> dict[str, Any]:
    """`out` (the aux basis's common fixture fields) as a schema-3 metric
    piece: no density, the `metric` record, and rows [r0, r1) of libcint's
    `int2c2e_cart` and `int1e_ovlp`. `int2c2e_cart` is checked bitwise
    against `int2c2e` (cart=True, so they must be the same call) before it
    is written; what is RECORDED is `int2c2e_cart`'s."""
    from .basis import converted_record

    if case.aux_rows is None or case.metric_set is None or case.parent is None:
        raise ValueError(f"{name}: a metric piece needs rows, set and parent")
    r0, r1 = case.aux_rows
    m = np.ascontiguousarray(mol.intor("int2c2e_cart"))
    if not np.array_equal(m, mol.intor("int2c2e")):
        raise ValueError(f"{name}: int2c2e_cart and int2c2e differ")
    s = np.ascontiguousarray(mol.intor("int1e_ovlp"))
    out["schema"] = METRIC_SCHEMA
    out["density_recipe"] = ("none: a metric piece is not an AO fixture and "
                             "records no density (#411)")
    out["metric"] = {"set": case.metric_set, "rows": [r0, r1],
                     "parent": case.parent,
                     "converted": converted_record(mol)}
    out["arrays"] = {METRIC_ARRAY: encode_array(m[r0:r1]),
                     AUX_OVLP_ARRAY: encode_array(s[r0:r1])}
    return out


def _three_center_fixture(name: str, case: Case, mol: Any, auxmol: Any,
                          out: dict[str, Any]) -> dict[str, Any]:
    """`out` (the aux basis's common fixture fields) as a schema-4
    three-center piece: no density, the `three_center` record, and aux
    columns [c0, c1) of libcint's `int3c2e_cart` through
    `df.incore.aux_e2(mol, auxmol, aosym="s1")`, shape (nao, nao, c1 - c0).
    Checked bitwise against `int3c2e` (cart=True, so the same call) before
    it is written; what is RECORDED is `int3c2e_cart`'s."""
    from pyscf import df

    from .basis import converted_record

    if case.aux_cols is None or case.parent is None:
        raise ValueError(f"{name}: a three-center piece needs cols and parent")
    c0, c1 = case.aux_cols
    t = np.ascontiguousarray(df.incore.aux_e2(mol, auxmol,
                                              intor="int3c2e_cart",
                                              aosym="s1"))
    nao = int(mol.nao_nr())
    if t.shape != (nao, nao, int(auxmol.nao_nr())):
        raise ValueError(f"{name}: int3c2e_cart has shape {t.shape}")
    if not np.array_equal(t, df.incore.aux_e2(mol, auxmol, intor="int3c2e",
                                              aosym="s1")):
        raise ValueError(f"{name}: int3c2e_cart and int3c2e differ")
    out["schema"] = THREE_CENTER_SCHEMA
    out["density_recipe"] = ("none: a three-center piece is not an AO "
                             "fixture and records no density (#442)")
    out["three_center"] = {"set": case.three_center_set, "cols": [c0, c1],
                           "parent": case.parent,
                           "converted": converted_record(auxmol)}
    out["arrays"] = {THREE_CENTER_PREFIX + case.parent:
                     encode_array(t[:, :, c0:c1])}
    return out


def cross_overlap_array(rows: Any, cols: Any) -> np.ndarray:
    """libcint's `<m|n>`, `m` an AO of `rows` and `n` of `cols`:
    `gto.intor_cross('int1e_ovlp', rows, cols)`. Cross-checked, before it is
    written, against a second route -- the two molecules concatenated
    (`gto.conc_mol`) and the off-diagonal block of its `int1e_ovlp` -- to
    1e-14 of the largest element; what is RECORDED is `intor_cross`'s."""
    from pyscf import gto

    s = np.ascontiguousarray(gto.intor_cross("int1e_ovlp", rows, cols))
    na = int(rows.nao_nr())
    other = gto.conc_mol(rows, cols).intor("int1e_ovlp")[:na, na:]
    err = float(np.max(np.abs(s - other)))
    if err > 1e-14 * float(np.max(np.abs(s))):
        raise ValueError(f"intor_cross and conc_mol's block differ by "
                         f"{err:.3e}")
    return s


def one_e_arrays(mol: Any) -> dict[str, Any]:
    """#240: libcint's `S`, `T`, `V` (`int1e_ovlp`, `int1e_kin`,
    `int1e_nuc`: point nuclei, `-Z_C` already in) and PySCF's
    `scf.hf.get_hcore`. `get_hcore` is cross-checked against `T + V` to 1e-14
    of the largest element before it is written (no ECP, no pseudo, no
    finite nuclei here, so it must be that sum); what is RECORDED is
    `get_hcore`'s, the matrix a later `Tr(D h)` is read against."""
    from pyscf import scf

    s = mol.intor("int1e_ovlp")
    t = mol.intor("int1e_kin")
    v = mol.intor("int1e_nuc")
    h = np.asarray(scf.hf.get_hcore(mol), dtype=np.float64)
    err = float(np.max(np.abs(h - (t + v))))
    if err > 1e-14 * float(np.max(np.abs(h))):
        raise ValueError(f"get_hcore and int1e_kin + int1e_nuc differ by "
                         f"{err:.3e}")
    return {OVLP_ARRAY: encode_array(s), KIN_ARRAY: encode_array(t),
            NUC_ARRAY: encode_array(v), HCORE_ARRAY: encode_array(h)}


def r_array(mol: Any, origin: Sequence[float]) -> np.ndarray:
    """libcint's `<m|r - C|n>` at common origin `C`, (3, nao, nao), x, y, z:
    `int1e_r_cart` under `mol.with_common_orig(C)`, checked bitwise against
    `int1e_r` (cart=True, so the same call) before it is returned."""
    with mol.with_common_orig(origin):
        r = np.ascontiguousarray(mol.intor("int1e_r_cart", comp=3))
        if not np.array_equal(r, mol.intor("int1e_r", comp=3)):
            raise ValueError("int1e_r_cart and int1e_r differ")
    nao = int(mol.nao_nr())
    if r.shape != (3, nao, nao):
        raise ValueError(f"int1e_r_cart has shape {r.shape}")
    return r


def _shift_checked_r(mol: Any, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(r_C, r_0)`, `int1e_r_cart` at `c` and at 0, refused unless
    `r_C = r_0 - C S` per component to 1e-14 of the largest `|r_0|`."""
    r, r0 = r_array(mol, c), r_array(mol, np.zeros(3))
    s = mol.intor("int1e_ovlp")
    shifted = r0 - c[:, None, None] * s[None]
    err = float(np.max(np.abs(r - shifted)))
    if err > 1e-14 * float(np.max(np.abs(r0))):
        raise ValueError(f"int1e_r_cart at C and at 0 - C S differ by "
                         f"{err:.3e}")
    return r, r0


def dipole_integral_arrays(mol: Any,
                           origin: Sequence[float]) -> dict[str, Any]:
    """#520: the integral half of `dipole_arrays` alone (no density, no
    `dip_moment`): `C`, and `int1e_r_cart` at `C` and at 0, with the same
    origin-shift cross-check before they are written."""
    c = np.asarray(origin, dtype=np.float64)
    r, r0 = _shift_checked_r(mol, c)
    return {DIPOLE_ORIGIN_ARRAY: encode_array(c), R_ARRAY: encode_array(r),
            R0_ARRAY: encode_array(r0)}


def dipole_arrays(mol: Any, dm: np.ndarray,
                  origin: Sequence[float]) -> dict[str, Any]:
    """#517: `int1e_r_cart` at `origin` and at 0, and PySCF's RHF
    `scf.hf.dip_moment(mol, dm, unit="AU", origin=...)` at both, with the
    nuclear charges and coordinates it reads (`atom_charges`,
    `atom_coords`). Cross-checked before they are written: the origin shift
    `r_C = r_0 - C S` per component to 1e-14 of the largest `|r_0|` (which
    pins the component order against the origin), and each `dip_moment`
    against `sum_A Z_A (R_A - C) - einsum("xij,ji->x", r_C, dm)` from the
    RECORDED `r_C` to 1e-13 of its scale; what is RECORDED is libcint's and
    `dip_moment`'s own."""
    from pyscf import scf

    c = np.asarray(origin, dtype=np.float64)
    zero = np.zeros(3)
    r, r0 = _shift_checked_r(mol, c)
    charges = np.asarray(mol.atom_charges(), dtype=np.float64)
    coords = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    out = {DIPOLE_ORIGIN_ARRAY: encode_array(c), R_ARRAY: encode_array(r),
           R0_ARRAY: encode_array(r0),
           DIP_CHARGES_ARRAY: encode_array(charges),
           DIP_COORDS_ARRAY: encode_array(coords)}
    for name, at, ints in ((DIP_ARRAY, c, r), (DIP0_ARRAY, zero, r0)):
        dip = np.asarray(scf.hf.dip_moment(mol, dm, unit="AU", origin=at,
                                           verbose=0), dtype=np.float64)
        nuc = np.einsum("i,ix->x", charges, coords - at[None, :])
        el = np.einsum("xij,ji->x", ints, dm)
        scale = float(np.max(np.abs(nuc)) + np.max(np.abs(el)))
        err = float(np.max(np.abs(dip - (nuc - el))))
        if err > 1e-13 * scale:
            raise ValueError(f"dip_moment and nuc - Tr(r D) differ by "
                             f"{err:.3e}")
        out[name] = encode_array(dip)
    return out


def j_array(mol: Any, dm: np.ndarray) -> np.ndarray:
    """PySCF's Coulomb matrix for the recorded density `dm`: `scf.hf.get_jk`
    with
    `hermi=1` (the density is exactly symmetric) and no `vhfopt`, so no
    integral screening on PySCF's side either. Cross-checked against the full
    `int2e` tensor contracted by `einsum`, to 1e-13 of the largest element,
    so a fixture is never written from a `get_jk` that disagrees with
    libcint's own integrals; what is RECORDED is `get_jk`'s."""
    from pyscf import scf

    vj, _ = scf.hf.get_jk(mol, dm, hermi=1, with_k=False)
    vj = np.asarray(vj, dtype=np.float64)
    full = np.einsum("ijkl,kl->ij", mol.intor("int2e"), dm)
    err = float(np.max(np.abs(vj - full)))
    if err > 1e-13 * float(np.max(np.abs(full))):
        raise ValueError(f"get_jk and einsum(int2e, dm) differ by {err:.3e}")
    return vj


def k_array(mol: Any, dm: np.ndarray) -> np.ndarray:
    """PySCF's exchange matrix for the recorded density `dm`: `scf.hf.get_jk`
    with
    `hermi=1` and no `vhfopt`, as `j_array`, keeping the K half. PySCF's
    convention is `vk_il = sum_jk (ij|kl) D_jk`, which by the ERI's
    symmetry is `eri.k_exchange.k`'s `K_ab = sum_cd D_cd (ac|bd)`. Cross-checked against
    the full `int2e` tensor contracted by `einsum` with exactly that index
    pattern, to 1e-13 of the largest element, before it is written; what is
    RECORDED is `get_jk`'s."""
    from pyscf import scf

    _, vk = scf.hf.get_jk(mol, dm, hermi=1, with_j=False)
    vk = np.asarray(vk, dtype=np.float64)
    full = np.einsum("acbd,cd->ab", mol.intor("int2e"), dm)
    err = float(np.max(np.abs(vk - full)))
    if err > 1e-13 * float(np.max(np.abs(full))):
        raise ValueError(f"get_jk and einsum(int2e, dm) differ by {err:.3e}")
    return vk


def schwarz_array(mol: Any) -> np.ndarray:
    """libcint's `(mn|mn)` for every AO pair: the full `int2e` tensor's
    diagonal `einsum("ijij->ij")`. Cross-checked, before it is written,
    against a second route through libcint -- one `shls_slice` call per shell
    pair `(i, j)` against itself, its component diagonal -- to 1e-14 of the
    largest element; what is RECORDED is the full tensor's."""
    full = np.ascontiguousarray(np.einsum("ijij->ij", mol.intor("int2e")))
    loc = mol.ao_loc_nr()
    other = np.zeros_like(full)
    for i in range(mol.nbas):
        for j in range(mol.nbas):
            blk = mol.intor("int2e", shls_slice=(i, i + 1, j, j + 1,
                                                 i, i + 1, j, j + 1))
            ni, nj = loc[i + 1] - loc[i], loc[j + 1] - loc[j]
            other[loc[i]:loc[i + 1], loc[j]:loc[j + 1]] = np.einsum(
                "ijij->ij", blk.reshape(ni, nj, ni, nj))
    err = float(np.max(np.abs(full - other)))
    if err > 1e-14 * float(np.max(np.abs(full))):
        raise ValueError(f"int2e's diagonal and the per-pair slices differ "
                         f"by {err:.3e}")
    return full


def sibling_converted(sibling: str, fresh: dict[str, Any]) -> dict[str, Any]:
    """#550: the tracked `sibling`'s recorded `converted` record -- the `_j`
    fixture's input, which a gradient fixture shares exactly -- refused
    unless `fresh` (this host's `converted_record`) is it in every field but
    the normalised coefficients, and those within 1e-15 of each value. The
    normalisation's last bit is host-dependent (h2o_631g_j's own coefficients
    differ by one ulp between the host that wrote it and others, on one
    PySCF and libcint), so the recorded input is adopted, as an "rhf"
    density is."""
    import json

    rec = json.loads((FIXTURES / f"{sibling}.json").read_text())["converted"]
    same = (rec["common_factor"] == fresh["common_factor"]
            and len(rec["shells"]) == len(fresh["shells"]))
    for a, b in zip(rec["shells"], fresh["shells"], strict=False):
        same = same and all(a[k] == b[k] for k in a if k != "coefficients")
        ca, cb = np.array(a["coefficients"]), np.array(b["coefficients"])
        same = same and ca.shape == cb.shape and bool(
            np.all(np.abs(ca - cb) <= 1e-15 * np.abs(ca)))
    if not same:
        raise ValueError(f"{sibling}'s recorded converted basis is not this "
                         "case's to 1e-15")
    return rec


def gradient_arrays(mol: Any, mo_energy: np.ndarray, mo_coeff: np.ndarray,
                    mo_occ: np.ndarray, dm: np.ndarray) -> dict[str, Any]:
    """#550: the gradient fixture's arrays (manifest: THE GRADIENT
    FIXTURES) for the recorded orbitals and their density `dm`.

    The terms are PySCF's `grad.rhf.grad_elec` split by term, from `dm`
    and `W = make_rdm1e(mo_energy, mo_coeff, mo_occ)`: `grad:h` the
    `hcore_generator` derivative contracted with `dm`; `grad:s` `-2 sum
    s1[:, s(A)] W[s(A)]` with `s1 = get_ovlp = -int1e_ipovlp`; `grad:j` /
    `grad:k` `2 sum vj1[:, s(A)] D[s(A)]` and `-1/2` of that with `vk1`,
    `get_jk`'s (already negated); `grad:nuc` `grad_nuc`. `grad:total` is
    `nuc_grad_method().kernel()` on the recorded orbitals. Cross-checked
    before they are written: `vj1` / `vk1` against `int2e_ip1` contracted
    by `einsum` (`-(nabla m n|k l) D_lk` and `-(nabla m n|k l) D_nk`) to
    1e-13 of the largest element, and the derivative integrals against
    `get_hcore` / `get_ovlp` exactly; what is RECORDED is PySCF's."""
    from pyscf import scf
    from pyscf.grad import rhf as rhf_grad

    mf = scf.RHF(mol)
    mf.mo_energy, mf.mo_coeff, mf.mo_occ = mo_energy, mo_coeff, mo_occ
    mf.converged = True
    g = mf.nuc_grad_method()
    g.verbose = 0
    total = np.asarray(g.kernel(mo_energy=mo_energy, mo_coeff=mo_coeff,
                                mo_occ=mo_occ), dtype=np.float64)
    dme = np.asarray(g.make_rdm1e(mo_energy, mo_coeff, mo_occ),
                     dtype=np.float64)
    ipovlp = mol.intor("int1e_ipovlp", comp=3)
    ipkin = mol.intor("int1e_ipkin", comp=3)
    ipnuc = mol.intor("int1e_ipnuc", comp=3)
    if not (np.array_equal(g.get_ovlp(mol), -ipovlp)
            and np.array_equal(g.get_hcore(mol), -(ipkin + ipnuc))):
        raise ValueError("get_ovlp / get_hcore are not the negated ip "
                         "integrals")
    iprinv = []
    for ia in range(mol.natm):
        with mol.with_rinv_at_nucleus(ia):
            iprinv.append(mol.intor("int1e_iprinv", comp=3))
    vj1, vk1 = (np.asarray(v, dtype=np.float64)
                for v in rhf_grad.get_jk(mol, dm))
    eri1 = mol.intor("int2e_ip1", comp=3)
    for got, want, what in (
            (vj1, -np.einsum("xmnkl,lk->xmn", eri1, dm), "vj1"),
            (vk1, -np.einsum("xmnkl,nk->xml", eri1, dm), "vk1")):
        err = float(np.max(np.abs(got - want)))
        if err > 1e-13 * float(np.max(np.abs(want))):
            raise ValueError(f"get_jk's {what} and einsum(int2e_ip1, dm) "
                             f"differ by {err:.3e}")
    s1 = g.get_ovlp(mol)
    hcore_deriv = g.hcore_generator(mol)
    aoslices = mol.aoslice_by_atom()
    terms = {n: np.zeros((mol.natm, 3)) for n in GRADIENT_TERMS}
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2:]
        terms["grad:h"][ia] = np.einsum("xij,ij->x", hcore_deriv(ia), dm)
        terms["grad:s"][ia] = -2 * np.einsum("xij,ij->x", s1[:, p0:p1],
                                             dme[p0:p1])
        terms["grad:j"][ia] = 2 * np.einsum("xij,ij->x", vj1[:, p0:p1],
                                            dm[p0:p1])
        terms["grad:k"][ia] = -np.einsum("xij,ij->x", vk1[:, p0:p1],
                                         dm[p0:p1])
    terms["grad:nuc"] = np.asarray(g.grad_nuc(mol), dtype=np.float64)
    return {
        "mo_energy": encode_array(mo_energy),
        "mo_occ": encode_array(mo_occ),
        "mo_coeff": encode_array(mo_coeff),
        "dme": encode_array(dme),
        "int1e_ipovlp": encode_array(ipovlp),
        "int1e_ipkin": encode_array(ipkin),
        "int1e_ipnuc": encode_array(ipnuc),
        "int1e_iprinv": encode_array(np.stack(iprinv)),
        "vj1": encode_array(vj1),
        "vk1": encode_array(vk1),
        **{n: encode_array(terms[n]) for n in GRADIENT_TERMS},
        GRAD_TOTAL: encode_array(total),
    }


def comparable(obj: dict[str, Any]) -> str:
    """The fixture's bytes with the volatile provenance blanked: what must match."""
    obj = dict(obj)
    obj["provenance"] = {k: v for k, v in obj["provenance"].items()
                         if k not in VOLATILE}
    return dumps(obj)


def drift(name: str) -> str | None:
    """None if a fresh build matches the tracked file; else what differs."""
    import json

    path = FIXTURES / f"{name}.json"
    tracked = json.loads(path.read_text())
    load(path)  # the tracked file must itself be valid
    fresh = build(name)
    tp, fp = tracked["provenance"], fresh["provenance"]
    if comparable(fresh) == comparable(tracked):
        case = CASES[name]
        if case.density != "rhf" or case.density_from is not None:
            return None
        # #486: the arrays are the recorded density's exactly; the density
        # itself is held to a fresh SCF within a tolerance, never bitwise.
        mol = _mole(case)
        dm = (mos_density(*rhf_mos(mol)[1:]) if case.record_gradient
              else rhf_density(mol))
        err = float(np.max(np.abs(dm - np.array(load(path).density))))
        if err <= RHF_DENSITY_TOL:
            return None
        return (f"{path}: PySCF {fp['pyscf']} now converges an RHF density "
                f"{err:.3e} (max abs) from the one recorded by pyscf "
                f"{tp['pyscf']} on {tp['date']} ({tp['host']}), over "
                f"RHF_DENSITY_TOL = {RHF_DENSITY_TOL:g}. Decide which is "
                f"right before running `{COMMAND} {name}`.")
    diffs = [k for k in fresh if k != "provenance" and fresh[k] != tracked[k]]
    diffs += [f"provenance.{k}" for k in fp
              if k not in VOLATILE and fp[k] != tp[k]]
    return (f"{path}: PySCF {fp['pyscf']} / libcint {fp['libcint']} now "
            f"compute {diffs} differently from what pyscf {tp['pyscf']} / "
            f"libcint {tp['libcint']} computed on {tp['date']} ({tp['host']}). "
            "A fixture records what libcint computed on a date, not the truth: "
            "this is news about PySCF until proven otherwise. Decide which is "
            f"right before running `{COMMAND} {name}`.")


# ---------------------------------------------------------------------------
# #613: the DF gradient witness (manifest.df_gradient)
# ---------------------------------------------------------------------------

DF_GRADIENT_COMMAND = ("PYTHONPATH=test uv run --group oracle python -m "
                       "oracle.generate --df-gradient")


def _class_grads(keys, atoms, contrib) -> list[dict[str, Any]]:
    """Per class, the sum of `contrib(block)` over the blocks keyed so."""
    out: dict[tuple[int, ...], np.ndarray] = {}
    for key, atom, g in contrib:
        out.setdefault(key, np.zeros((atoms, 3)))[atom] += g
    return [{"am": list(k), "G": [float(x) for x in out[k].ravel()]}
            for k in sorted(out)]


def df_gradient_build() -> dict[str, Any]:
    """The DF gradient witness (manifest.df_gradient), computed now: per
    class of the aux-i sets' molecule, libcint's `int3c2e_ip1_cart` /
    `int3c2e_ip2_cart` contracted with `df_gradient_gam` and
    `int2c2e_ip1_cart` with `df_gradient_w`. An electron-coordinate
    derivative on a function is minus the derivative on its centre, so
    `d(mn|P)/dR_{atom(m)} = -ip1[k, m, n, P]`, the `n` side `-ip1[k, n, m,
    P]` (`(m nabla n|P) = (nabla n m|P)`), the aux side `-ip2[k, m, n, P]`,
    and `d(P|Q)/dR_{atom(P)} = -ip1[k, P, Q]`, the `Q` side `-ip1[k, Q,
    P]`."""
    import pyscf
    from pyscf import df, gto

    from .manifest import (
        DF_GRADIENT,
        DF_GRADIENT_SCHEMA,
        df_gradient_gam,
        df_gradient_w,
    )

    case = CASES["prim_spd_f"]
    mol = _mole(case)
    aux = {s: gto.basis.parse(AUX_I, symb=s)
           for s in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})}
    auxmol = df.make_auxmol(mol, aux)
    nao, naux, nat = int(mol.nao_nr()), int(auxmol.nao_nr()), mol.natm
    ip1 = df.incore.aux_e2(mol, auxmol, intor="int3c2e_ip1_cart",
                           aosym="s1", comp=3)
    ip2 = df.incore.aux_e2(mol, auxmol, intor="int3c2e_ip2_cart",
                           aosym="s1", comp=3)
    m1 = auxmol.intor("int2c2e_ip1_cart", comp=3)
    if ip1.shape != (3, nao, nao, naux) or m1.shape != (3, naux, naux):
        raise ValueError(f"unexpected shapes {ip1.shape} {m1.shape}")
    gam, w = df_gradient_gam(nao, naux), df_gradient_w(naux)
    loc, aloc = mol.ao_loc_nr(cart=True), auxmol.ao_loc_nr(cart=True)

    def sl(lc, i):
        return slice(int(lc[i]), int(lc[i + 1]))

    def eri3():
        for i in range(mol.nbas):
            for j in range(mol.nbas):
                li, lj = mol.bas_angular(i), mol.bas_angular(j)
                for p in range(auxmol.nbas):
                    key = (min(li, lj), max(li, lj), auxmol.bas_angular(p))
                    si, sj, sp = sl(loc, i), sl(loc, j), sl(aloc, p)
                    g = gam[si, sj, sp]
                    yield key, mol.bas_atom(i), -np.einsum(
                        "kmnp,mnp->k", ip1[:, si, sj, sp], g)
                    yield key, mol.bas_atom(j), -np.einsum(
                        "knmp,mnp->k", ip1[:, sj, si, sp], g)
                    yield key, auxmol.bas_atom(p), -np.einsum(
                        "kmnp,mnp->k", ip2[:, si, sj, sp], g)

    def metric():
        for p in range(auxmol.nbas):
            for q in range(auxmol.nbas):
                lp, lq = auxmol.bas_angular(p), auxmol.bas_angular(q)
                key = (min(lp, lq), max(lp, lq))
                sp, sq = sl(aloc, p), sl(aloc, q)
                yield key, auxmol.bas_atom(p), -np.einsum(
                    "kpq,pq->k", m1[:, sp, sq], w[sp, sq])
                yield key, auxmol.bas_atom(q), -np.einsum(
                    "kqp,pq->k", m1[:, sq, sp], w[sp, sq])

    return {
        "schema": DF_GRADIENT_SCHEMA,
        "name": DF_GRADIENT,
        "description": (
            "#613: libcint's int3c2e_ip1_cart / int3c2e_ip2_cart and "
            "int2c2e_ip1_cart on prim_spd_f's molecule against the aux-i "
            "basis (prim_spd_f_auxi_3c / _metric's), contracted per class "
            "into G[atom, k] with manifest.df_gradient_gam / df_gradient_w "
            "(oracle.generate.df_gradient_build has the signs)."),
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": _libcint_version(),
            "numpy": np.__version__,
            "date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
            "host": _host(),
            "command": DF_GRADIENT_COMMAND,
            "mol": {"atom": case.atom, "unit": case.unit,
                    "charge": case.charge, "spin": case.spin},
            "basis": _basis_name(case),
            "cart": case.cart,
            "aux_basis": AUX_I,
            "parent": "prim_spd_f",
            "nao": nao,
            "naux": naux,
        },
        "three_center_set": AUX_I_3C_SET,
        "metric_set": AUX_I_METRIC_SET,
        "eri3": _class_grads(None, nat, eri3()),
        "gmetric": _class_grads(None, nat, metric()),
    }


def df_gradient_main(regenerate: bool) -> int:
    """`--df-gradient`: rewrite both witnesses in `df_gradient/` (#613's
    per-class contractions, #620's DF-RHF gradient), or (`--check`)
    compare."""
    from .manifest import DF_GRADIENT, DF_RHF, df_gradient, df_rhf

    return (_witness_main(regenerate, DF_GRADIENT, df_gradient_build,
                         df_gradient)
            | _witness_main(regenerate, DF_RHF, df_rhf_build, df_rhf))


# #620: the DF-RHF two-electron gradient (manifest.df_rhf)

def df_rhf_build() -> dict[str, Any]:
    """The DF-RHF two-electron gradient (#620, manifest.df_rhf), computed
    now: PySCF's `scf.RHF(mol).density_fit(aux).nuc_grad_method()` at the
    closed-form `D = occ C C^T` (`df_rhf_c`, `mo_coeff` / `mo_occ` tagged so
    K is built from `C`). `get_jk`'s `vj` / `vk` are electron-coordinate
    derivatives on the bra AO (`-d/dR` of the function, so `+d/dR` of the
    centre with PySCF's sign), each AO row counted twice by `(mn|ls) =
    (nm|ls)` symmetry, plus its `.aux` (the aux-centre and metric terms):
    `G_X[A] = 2 SUM_{m on A} vX[:, m, :] . D[m, :] + vX.aux[A]`. `G` is
    `get_veff`'s the same way, `E_J - 1/2 E_K`. ONE OpenMP thread: PySCF's
    threaded DF-K reduces in a run-dependent order (~1e-14 apart run to
    run), and `--check` compares bytes."""
    import pyscf
    from pyscf import gto, lib, scf

    from .manifest import (
        DF_RHF,
        DF_RHF_AUX_MAX_L,
        DF_RHF_OCC,
        DF_RHF_SCHEMA,
        DF_RHF_SET,
        df_rhf_c,
    )

    piece = CASES[f"{DF_RHF_SET}{COLS_SEP}{THREE_CENTER_COLS[0][0]}_"
                  f"{THREE_CENTER_COLS[0][1]}"]
    case = CASES[piece.parent]
    mol = _mole(case)
    symbols = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    aux = {s: [sh for sh in gto.basis.load(piece.aux_basis, s)
               if sh[0] <= DF_RHF_AUX_MAX_L] for s in symbols}
    mf = scf.RHF(mol).density_fit(auxbasis=aux)
    mf.with_df.build()
    auxmol = mf.with_df.auxmol
    if not auxmol.cart:
        raise ValueError("the aux molecule is not Cartesian")
    nao, naux, nat = int(mol.nao_nr()), int(auxmol.nao_nr()), mol.natm
    nocc = mol.nelectron // 2
    c = df_rhf_c(nao, nocc)
    mo = np.zeros((nao, nao))
    mo[:, :nocc] = c
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = DF_RHF_OCC
    d = DF_RHF_OCC * c @ c.T
    dm = lib.tag_array(d, mo_coeff=mo, mo_occ=mo_occ)
    with lib.with_omp_threads(1):
        vj, vk = mf.get_jk(mol, dm)
        g = mf.nuc_grad_method()
        gj, gk = g.get_jk(mol, dm)
        veff = g.get_veff(mol, dm)
    sl = mol.aoslice_by_atom()

    def total(v) -> list[float]:
        out = np.asarray(v.aux, dtype=float).reshape(-1, nat, 3)[0].copy()
        for a in range(nat):
            p0, p1 = sl[a, 2:]
            out[a] += 2 * np.einsum("xij,ij->x", v[:, p0:p1], d[p0:p1])
        return [float(x) for x in out.ravel()]

    return {
        "schema": DF_RHF_SCHEMA,
        "name": DF_RHF,
        "description": (
            "#620: PySCF's DF-RHF two-electron nuclear gradient (aux "
            "response on) of H2O/STO-3G against cc-pVTZ-JKFIT's s and p "
            "shells (h2o_sto3g_jkfit_3c's inputs, aux l <= 1), at D = occ "
            "C C^T of manifest.df_rhf_c: g_j of E_J, g_k of E_K = 1/2 SUM "
            "D D (ml|ns), g of E_J - 1/2 E_K (get_veff), each [atom, k] "
            "(oracle.generate.df_rhf_build has the signs)."),
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": _libcint_version(),
            "numpy": np.__version__,
            "date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
            "host": _host(),
            "command": DF_GRADIENT_COMMAND,
            "mol": {"atom": case.atom, "unit": case.unit,
                    "charge": case.charge, "spin": case.spin},
            "basis": _basis_name(case),
            "cart": case.cart,
            "aux_basis": piece.aux_basis,
            "aux_max_l": DF_RHF_AUX_MAX_L,
            "parent": piece.parent,
            "nao": nao,
            "naux": naux,
            "occ": DF_RHF_OCC,
        },
        "three_center_set": DF_RHF_SET,
        "nocc": nocc,
        "e_j": float(0.5 * np.einsum("ij,ji->", d, vj)),
        "e_k": float(0.5 * np.einsum("ij,ji->", d, vk)),
        "g_j": total(gj),
        "g_k": total(gk),
        "g": total(veff),
    }


def _witness_main(regenerate: bool, name: str, build, load) -> int:
    import json

    from .manifest import DF_GRADIENT_DIR

    path = DF_GRADIENT_DIR / f"{name}.json"
    fresh = build()
    if regenerate:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(fresh))
        load()
        print(f"wrote {path}")
        return 0
    tracked = json.loads(path.read_text())
    if comparable(fresh) == comparable(tracked):
        print(f"{name}: matches the tracked witness")
        return 0
    print(f"{path}: PySCF now computes it differently from the tracked "
          "witness; news about PySCF until proven otherwise")
    return 1


# ---------------------------------------------------------------------------
# #618: the exchange-gradient witness (manifest.k_gradient)
# ---------------------------------------------------------------------------

K_GRADIENT_COMMAND = ("PYTHONPATH=test uv run --group oracle python -m "
                      "oracle.generate --k-gradient")
K_GRADIENT_PARENT = "co_ccpvtz_j"


def k_gradient_build() -> dict[str, Any]:
    """The exchange-gradient witness (manifest.k_gradient), computed now:
    libcint's `int2e_ip1_cart` on `K_GRADIENT_PARENT`'s molecule and basis,
    contracted with that fixture's recorded density `D`. An
    electron-coordinate derivative on a function is minus the derivative on
    its centre, so with `ip1[k, a, c, b, d] = (nabla_k a c|b d)` the bra
    derivative of the quartet `(ac|bd)` (`kernel_gk`'s pairing) is
    `-SUM D_ab D_cd ip1[k, a, c, b, d]` on `atom(a)` and `-SUM D_ab D_cd
    ip1[k, c, a, b, d]` on `atom(c)` (`(ac|bd) = (ca|bd)`), each into the K
    class `(l_a, l_c | l_b, l_d)`. One `shls_slice` per first shell; `D` is
    symmetric (asserted)."""
    import pyscf

    from .manifest import (
        K_GRADIENT,
        K_GRADIENT_SCHEMA,
        density_sha256,
    )

    case = CASES[K_GRADIENT_PARENT]
    mol = _mole(case)
    D = _tracked_density(K_GRADIENT_PARENT)
    if D is None:
        raise ValueError(f"{K_GRADIENT_PARENT} is not tracked")
    if not np.array_equal(D, D.T):
        raise ValueError(f"{K_GRADIENT_PARENT}'s density is not symmetric")
    nao, nat, n = int(mol.nao_nr()), mol.natm, mol.nbas
    if D.shape != (nao, nao):
        raise ValueError(f"density {D.shape}, nao {nao}")
    loc = mol.ao_loc_nr(cart=True)
    ls = [int(mol.bas_angular(i)) for i in range(n)]
    by_l = {lv: np.concatenate([np.arange(loc[i], loc[i + 1])
                                for i in range(n) if ls[i] == lv])
            for lv in sorted(set(ls))}
    out: dict[tuple[int, int, int, int], np.ndarray] = {}

    def add(key, atom, g) -> None:
        out.setdefault(key, np.zeros((nat, 3)))[atom] += g

    for s in range(n):
        ip = mol.intor("int2e_ip1_cart", comp=3,
                       shls_slice=(s, s + 1, 0, n, 0, n, 0, n))
        ss = slice(int(loc[s]), int(loc[s + 1]))
        if ip.shape != (3, ss.stop - ss.start, nao, nao, nao):
            raise ValueError(f"unexpected shape {ip.shape}")
        atom = int(mol.bas_atom(s))
        for lt, T in by_l.items():
            for lu, U in by_l.items():
                for lv, V in by_l.items():
                    blk = ip[:, :, T][:, :, :, U][:, :, :, :, V]
                    add((ls[s], lt, lu, lv), atom, -np.einsum(
                        "xstuv,su,tv->x", blk, D[ss][:, U], D[np.ix_(T, V)]))
                    add((lt, ls[s], lu, lv), atom, -np.einsum(
                        "xstuv,tu,sv->x", blk, D[np.ix_(T, U)], D[ss][:, V]))
    whole = sum(out.values())
    return {
        "schema": K_GRADIENT_SCHEMA,
        "name": K_GRADIENT,
        "description": (
            f"#618: libcint's int2e_ip1_cart on {K_GRADIENT_PARENT}'s "
            "molecule and basis, contracted with its recorded density into "
            "dE_K/dR (E_K = 1/2 SUM D_ab D_cd (ac|bd), the bra derivative "
            "of every ordered quartet, both bra centres), per K class "
            "(la, lc | lb, ld) and whole (oracle.generate.k_gradient_build "
            "has the signs)."),
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": _libcint_version(),
            "numpy": np.__version__,
            "date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
            "host": _host(),
            "command": K_GRADIENT_COMMAND,
            "mol": {"atom": case.atom, "unit": case.unit,
                    "charge": case.charge, "spin": case.spin},
            "basis": _basis_name(case),
            "cart": case.cart,
            "nao": nao,
            "density_sha256": density_sha256(D),
        },
        "parent": K_GRADIENT_PARENT,
        "whole": [float(x) for x in whole.ravel()],
        "classes": [{"am": list(k), "G": [float(x) for x in out[k].ravel()]}
                    for k in sorted(out)],
    }


def k_gradient_main(regenerate: bool) -> int:
    """`--k-gradient`: rewrite the witness, or (`--check`) compare."""
    import json

    from .manifest import K_GRADIENT, K_GRADIENT_DIR, k_gradient

    path = K_GRADIENT_DIR / f"{K_GRADIENT}.json"
    fresh = k_gradient_build()
    if regenerate:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(fresh))
        k_gradient()
        print(f"wrote {path}")
        return 0
    tracked = json.loads(path.read_text())
    if comparable(fresh) == comparable(tracked):
        print(f"{K_GRADIENT}: matches the tracked witness")
        return 0
    print(f"{path}: PySCF now computes it differently from the tracked "
          "witness; news about PySCF until proven otherwise")
    return 1


# ---------------------------------------------------------------------------
# The quadrupole witness (manifest.quadrupole_integrals)
# ---------------------------------------------------------------------------

QUADRUPOLE_COMMAND = ("PYTHONPATH=test uv run --group oracle python -m "
                      "oracle.generate --quadrupole")


def rr_array(mol: Any, origin: Sequence[float]) -> np.ndarray:
    """libcint's `<m|(r - C)_d (r - C)_e|n>` at common origin `C`, the six
    components `manifest.QUADRUPOLE_COMPONENTS`, (6, nao, nao): the nine of
    `int1e_rr_cart` under `mol.with_common_orig(C)`, refused unless each
    mirrored component (yx, zx, zy) equals its twin bitwise."""
    from .manifest import RR_KEPT

    nao = int(mol.nao_nr())
    with mol.with_common_orig(origin):
        rr = np.ascontiguousarray(mol.intor("int1e_rr_cart", comp=9))
    if rr.shape != (9, nao, nao):
        raise ValueError(f"int1e_rr_cart has shape {rr.shape}")
    for d in range(3):
        for e in range(d):
            if not np.array_equal(rr[3 * d + e], rr[3 * e + d]):
                raise ValueError(f"int1e_rr_cart component {3 * d + e} is "
                                 f"not its mirror {3 * e + d}")
    return np.ascontiguousarray(rr[list(RR_KEPT)])


def quadrupole_build() -> dict[str, Any]:
    """The quadrupole witness (manifest.quadrupole_integrals), computed now:
    `rr_array` at the parent's origin and at 0, cross-checked before it is
    returned by the origin shift `Q_de(C) = Q_de(0) - C_d r_e(0) -
    C_e r_d(0) + C_d C_e S` to 1e-14 of the largest `|Q(0)|`."""
    import pyscf

    from .manifest import (
        QUADRUPOLE,
        QUADRUPOLE_COMPONENTS,
        QUADRUPOLE_SCHEMA,
        SPD_DIPOLE,
    )

    case = CASES[SPD_DIPOLE]
    mol = _mole(case)
    c = np.asarray(DIPOLE_ORIGIN, dtype=np.float64)
    rr, rr0 = rr_array(mol, c), rr_array(mol, np.zeros(3))
    r0 = r_array(mol, np.zeros(3))
    s = mol.intor("int1e_ovlp")
    pairs = [(d, e) for d in range(3) for e in range(d, 3)]
    for k, (d, e) in enumerate(pairs):
        want = rr0[k] - c[d] * r0[e] - c[e] * r0[d] + c[d] * c[e] * s
        err = float(np.max(np.abs(rr[k] - want)))
        if err > 1e-14 * float(np.max(np.abs(rr0))):
            raise ValueError(f"int1e_rr_cart's {QUADRUPOLE_COMPONENTS[k]} at "
                             f"C and its shift from 0 differ by {err:.3e}")
    return {
        "schema": QUADRUPOLE_SCHEMA,
        "name": QUADRUPOLE,
        "description": (
            f"libcint's int1e_rr_cart (<m|(r - C)_d (r - C)_e|n>) on "
            f"{SPD_DIPOLE}'s molecule and basis under mol.with_common_orig(C) "
            f"at C = {DIPOLE_ORIGIN} bohr (the parent's dipole_origin) and at "
            "C = 0; the six components xx, xy, xz, yy, yz, zz of libcint's "
            "nine, the mirrored three checked equal first."),
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": _libcint_version(),
            "numpy": np.__version__,
            "date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
            "host": _host(),
            "command": QUADRUPOLE_COMMAND,
            "mol": {"atom": case.atom, "unit": case.unit,
                    "charge": case.charge, "spin": case.spin},
            "basis": _basis_name(case),
            "cart": case.cart,
            "nao": int(mol.nao_nr()),
        },
        "parent": SPD_DIPOLE,
        "components": list(QUADRUPOLE_COMPONENTS),
        "arrays": {
            "origin": encode_array(c),
            "int1e_rr_cart:origin": encode_array(rr),
            "int1e_rr_cart:zero": encode_array(rr0),
        },
    }


def quadrupole_main(regenerate: bool) -> int:
    """`--quadrupole`: rewrite the witness, or (`--check`) compare."""
    import json

    from .manifest import QUADRUPOLE, QUADRUPOLE_DIR, quadrupole_integrals

    path = QUADRUPOLE_DIR / f"{QUADRUPOLE}.json"
    fresh = quadrupole_build()
    if regenerate:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(fresh))
        quadrupole_integrals()
        print(f"wrote {path}")
        return 0
    tracked = json.loads(path.read_text())
    if comparable(fresh) == comparable(tracked):
        print(f"{QUADRUPOLE}: matches the tracked witness")
        return 0
    print(f"{path}: PySCF now computes it differently from the tracked "
          "witness; news about PySCF until proven otherwise")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m oracle.generate",
                                 description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--regenerate", action="store_true",
                      help="rewrite the tracked fixture(s)")
    mode.add_argument("--check", action="store_true",
                      help="compare a fresh build against the tracked fixture(s)")
    ap.add_argument("names", nargs="*", help=f"cases (default all: {list(CASES)})")
    ap.add_argument("--df-gradient", action="store_true",
                    help="the DF gradient witness (#613, "
                         "manifest.df_gradient) instead of the fixtures")
    ap.add_argument("--quadrupole", action="store_true",
                    help="the quadrupole witness "
                         "(manifest.quadrupole_integrals) instead of the "
                         "fixtures")
    ap.add_argument("--k-gradient", action="store_true",
                    help="the exchange-gradient witness (#618, "
                         "manifest.k_gradient) instead of the fixtures")
    args = ap.parse_args(argv)
    if args.df_gradient and args.names:
        ap.error("--df-gradient takes no case names")
    if args.k_gradient and (args.names or args.df_gradient
                            or args.quadrupole):
        ap.error("--k-gradient takes no case names and no other witness")
    if args.quadrupole and (args.names or args.df_gradient):
        ap.error("--quadrupole takes no case names and no --df-gradient")

    unknown = [n for n in args.names if n not in CASES]
    if unknown:
        ap.error(f"unknown case(s) {unknown}; known: {list(CASES)}")
    try:
        import pyscf  # noqa: F401
    except ImportError:
        print("could not run: PySCF is not installed -- uv sync --group oracle",
              file=sys.stderr)
        return 2
    # `oracle.basis` (the converted slot) imports felizjueves.ao.shell. pytest
    # gets `src` from `pythonpath`; the project is virtual (nothing installed),
    # so run as `python -m oracle.generate` it is put on the path here.
    src = str(Path(__file__).resolve().parents[2] / "src")
    if src not in sys.path:
        sys.path.append(src)
    if args.df_gradient:
        return df_gradient_main(args.regenerate)
    if args.quadrupole:
        return quadrupole_main(args.regenerate)
    if args.k_gradient:
        return k_gradient_main(args.regenerate)

    status = 0
    for name in args.names or list(CASES):
        if args.regenerate:
            path = FIXTURES / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dumps(build(name, fresh_density=True)))
            if CASES[name].record_gradient:
                gradient_fixture(name)  # #550: every term recomputed
            else:
                load(path)
            print(f"wrote {path}")
        else:
            msg = drift(name)
            print(msg or f"{name}: matches the tracked fixture")
            status |= msg is not None
    return status


if __name__ == "__main__":
    sys.exit(main())
