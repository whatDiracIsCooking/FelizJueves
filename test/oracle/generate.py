"""Write the tracked oracle fixtures from PySCF -- the only module that imports it.

    PYTHONPATH=test uv run --group oracle \
        python -m oracle.generate --regenerate [NAME ...]    # or --check

`--regenerate` rewrites `fixtures/<name>.json` (every case, or the named ones);
`--check` rebuilds in memory and compares against what is tracked, the same
comparison the `oracle`-marked test makes. Exit 0 agree / written, 1 drift,
2 could not run (PySCF absent -- `uv sync --group oracle`). The shape is
`felizjueves.measured --remeasure`'s: one command rewrites the data file, and
the file carries who wrote it.

WHAT A FIXTURE IS NOT: the truth. It records what libcint computed, with the
versions, host and date in its provenance block. A drift is news about PySCF
until proven otherwise; regenerate only after deciding which side is right,
and say so in the PR that changes the file.

Adding a case is an entry in `CASES` plus `--regenerate <name>`. A case's
inputs (`mol`, `basis`, `cart`) are part of what `--check` compares; the
volatile provenance (versions, date, host) is not.
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
    ERI_PREFIX,
    FIXTURES,
    HCORE_ARRAY,
    J_ARRAY,
    K_ARRAY,
    KIN_ARRAY,
    NUC_ARRAY,
    OVLP_ARRAY,
    SCHEMA,
    SCHWARZ_ARRAY,
    dumps,
    encode_array,
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
    # Write the `int2e_diag` array: libcint's `(mn|mn)` for every AO pair,
    # the square of the Schwarz factor (`ao/schwarz.py`). Needs
    # `record_basis`, as `record_j` does.
    record_schwarz: bool = False


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


def _host() -> str:
    # The same marking felizjueves.measured uses: a container's hostname is
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


def build(name: str) -> dict[str, Any]:
    """The fixture object for `name`, computed now by PySCF/libcint."""
    import pyscf
    from pyscf import gto

    case = CASES[name]
    basis: Any = case.basis
    if "\n" in case.basis:  # inline NWChem text, not a library name
        symbols = {s.split()[0] for s in case.atom.split(";") if s.strip()}
        basis = {s: gto.basis.parse(case.basis, symb=s) for s in sorted(symbols)}
    mol = gto.M(atom=case.atom, unit=case.unit, basis=basis, cart=case.cart,
                charge=case.charge, spin=case.spin, verbose=0)

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
    out = {
        "schema": SCHEMA,
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
            "basis": case.basis,
            "cart": case.cart,
        },
        "atoms": atoms,
        "shells": shells,
        "nao": nao,
        "ao_labels": list(mol.ao_labels()),
        "ao_loc": [int(x) for x in ao_loc],
        "arrays": {
            "density": encode_array(hilbert_density(nao)),
            "ovlp": encode_array(mol.intor("int1e_ovlp")),
        },
        "density_recipe": DENSITY_RECIPE,
    }
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
        out["arrays"][J_ARRAY] = encode_array(j_array(mol))
    if case.record_k:
        if not case.record_basis:
            raise ValueError(f"{name}: record_k needs record_basis")
        out["arrays"][K_ARRAY] = encode_array(k_array(mol))
    if case.record_schwarz:
        if not case.record_basis:
            raise ValueError(f"{name}: record_schwarz needs record_basis")
        out["arrays"][SCHWARZ_ARRAY] = encode_array(schwarz_array(mol))
    if case.record_basis and case.record_one_e:
        out["arrays"].update(one_e_arrays(mol))
    return out


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


def j_array(mol: Any) -> np.ndarray:
    """PySCF's Coulomb matrix for the recorded density: `scf.hf.get_jk` with
    `hermi=1` (the density is exactly symmetric) and no `vhfopt`, so no
    integral screening on PySCF's side either. Cross-checked against the full
    `int2e` tensor contracted by `einsum`, to 1e-13 of the largest element,
    so a fixture is never written from a `get_jk` that disagrees with
    libcint's own integrals; what is RECORDED is `get_jk`'s."""
    from pyscf import scf

    dm = hilbert_density(int(mol.nao_nr()))
    vj, _ = scf.hf.get_jk(mol, dm, hermi=1, with_k=False)
    vj = np.asarray(vj, dtype=np.float64)
    full = np.einsum("ijkl,kl->ij", mol.intor("int2e"), dm)
    err = float(np.max(np.abs(vj - full)))
    if err > 1e-13 * float(np.max(np.abs(full))):
        raise ValueError(f"get_jk and einsum(int2e, dm) differ by {err:.3e}")
    return vj


def k_array(mol: Any) -> np.ndarray:
    """PySCF's exchange matrix for the recorded density: `scf.hf.get_jk` with
    `hermi=1` and no `vhfopt`, as `j_array`, keeping the K half. PySCF's
    convention is `vk_il = sum_jk (ij|kl) D_jk`, which by the ERI's
    symmetry is `ao.k`'s `K_ab = sum_cd D_cd (ac|bd)`. Cross-checked against
    the full `int2e` tensor contracted by `einsum` with exactly that index
    pattern, to 1e-13 of the largest element, before it is written; what is
    RECORDED is `get_jk`'s."""
    from pyscf import scf

    dm = hilbert_density(int(mol.nao_nr()))
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
    if comparable(fresh) == comparable(tracked):
        return None
    tp, fp = tracked["provenance"], fresh["provenance"]
    diffs = [k for k in fresh if k != "provenance" and fresh[k] != tracked[k]]
    diffs += [f"provenance.{k}" for k in fp
              if k not in VOLATILE and fp[k] != tp[k]]
    return (f"{path}: PySCF {fp['pyscf']} / libcint {fp['libcint']} now "
            f"compute {diffs} differently from what pyscf {tp['pyscf']} / "
            f"libcint {tp['libcint']} computed on {tp['date']} ({tp['host']}). "
            "A fixture records what libcint computed on a date, not the truth: "
            "this is news about PySCF until proven otherwise. Decide which is "
            f"right before running `{COMMAND} {name}`.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m oracle.generate",
                                 description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--regenerate", action="store_true",
                      help="rewrite the tracked fixture(s)")
    mode.add_argument("--check", action="store_true",
                      help="compare a fresh build against the tracked fixture(s)")
    ap.add_argument("names", nargs="*", help=f"cases (default all: {list(CASES)})")
    args = ap.parse_args(argv)

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

    status = 0
    for name in args.names or list(CASES):
        if args.regenerate:
            path = FIXTURES / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dumps(build(name)))
            load(path)
            print(f"wrote {path}")
        else:
            msg = drift(name)
            print(msg or f"{name}: matches the tracked fixture")
            status |= msg is not None
    return status


if __name__ == "__main__":
    sys.exit(main())
