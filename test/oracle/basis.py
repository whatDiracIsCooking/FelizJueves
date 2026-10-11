"""A PySCF `Mole` as our shells -- and libcint's three conventions, pinned.

`from_mole(mol)` turns a PySCF molecule into a `felizjueves.ao.shell.Basis` in
OUR convention (coefficients multiplying UNNORMALISED Cartesian primitives
x^i y^j z^k exp(-alpha r^2); `ao/shell.py`'s module docstring), so that a
contracted function of ours and libcint's AO of the same index are the same
function, with the same AO index. `from_fixture(fx)` does the same from a
tracked fixture's `converted` record, with no PySCF. `converted_record(mol)` is
what `generate.py` writes into that record.

THE THREE CONVENTIONS -- each was wrong once in the predecessor project, and
each shows up as a clean per-angular-momentum constant. `test_oracle_basis.py`
pins each with a test, not a comment, because a PySCF version can move them:

1. `cart=True`. PySCF defaults to solid harmonics from `d` up; this tree is
   Cartesian throughout. `from_mole` refuses a spherical `Mole` outright, and
   the AO-count assertion is the guard (a `d` shell is 5 AOs, not 6, without).
2. `mol._libcint_ctr_coeff(i)`, NOT `mol.bas_ctr_coeff(i)`. The former is what
   libcint integrates: the file's coefficient times `gto_norm(l, alpha)` (the
   RADIAL normaliser, 1/sqrt(int r^(2l+2) e^(-2 alpha r^2) dr)) times a
   contraction normaliser. The latter divides `gto_norm` back out.
3. `common_factor(l)` -- libcint's `CINTcommon_fac_sp`, applied inside every
   integral routine and therefore invisible in `_env`: 1/sqrt(4 pi) for s,
   sqrt(3/(4 pi)) for p, 1 from d up.

So OUR coefficient for primitive p of contraction c is

    _libcint_ctr_coeff(i)[p, c] * common_factor(l)

and NOTHING else: applying `ao.shell.primitive_norm` on top normalises twice
(the hazard the issue names; only the overlap sees it).

HOW THE TWO NORMALISATIONS LINE UP. Our `primitive_norm(l, a)` = N makes the
AXIS component x^l unit; libcint's `gto_norm` normalises the radial part, and
since the angular average of (x/r)^(2l) is 1/(2l+1),

    gto_norm(l, a) = N(l, a) * sqrt(4 pi / (2l + 1)).

Times `common_factor`: for l = 0, 1, sqrt(4 pi/(2l+1)) * common_factor(l) = 1
exactly, so a libcint s or p primitive IS ours times N. From d up the common
factor is 1, so a libcint Cartesian primitive is N * sqrt(4 pi/(2l+1)) -- its
axis component has self-overlap 4 pi/(2l+1) (4 pi/5 for xx, 4 pi/7 for xxx),
not 1. The RATIOS between components are the same in both conventions,
(2i-1)!!(2j-1)!!(2k-1)!!/(2l-1)!!, because both are one factor per l. None of
this is a correction we apply: our coefficients absorb libcint's whole
normalisation, so the two sides are the same functions.

This module lives in `test/`, deliberately: `src/felizjueves` has zero runtime
dependencies. It never imports PySCF at module scope -- `common_factor` and
`from_fixture` are fast-tier -- and the functions taking a `mol` only call its
methods; `measured_common_factor` imports `pyscf.gto` when called.
"""

from __future__ import annotations

import math
from typing import Any

from felizjueves.ao.shell import Basis, Shell, primitive_norm

from .manifest import Fixture

# libcint's CINTcommon_fac_sp: l = 0 and 1 carry the real-solid-harmonic
# prefactor, l >= 2 nothing. Pinned against PySCF's data, per l, by
# test_oracle_basis.py -- here it is only written down.
_COMMON_FACTOR = {0: 1.0 / math.sqrt(4.0 * math.pi),
                  1: math.sqrt(3.0 / (4.0 * math.pi))}

# The exponent of the one-primitive probe `measured_common_factor` builds.
PROBE_EXPONENT = 1.0


def common_factor(l: int) -> float:  # noqa: E741
    """libcint's `common_fac`: 1/sqrt(4pi) (s), sqrt(3/4pi) (p), 1 (d and up)."""
    if type(l) is not int or l < 0:
        raise ValueError(f"angular momentum {l!r} is not an int >= 0")
    return _COMMON_FACTOR.get(l, 1.0)


def _require_cart(mol: Any) -> None:
    if not mol.cart:
        raise ValueError(
            "a spherical Mole (cart=False): this tree is Cartesian throughout, "
            "and from d up PySCF's AOs would be solid harmonics -- build the "
            "Mole with cart=True")


def _shells(mol: Any, coeff: Any) -> Basis:
    shells = []
    for i in range(mol.nbas):
        l = int(mol.bas_angular(i))  # noqa: E741
        centre = tuple(float(x) for x in mol.bas_coord(i))
        exps = tuple(float(a) for a in mol.bas_exp(i))
        cs = coeff(i)  # [primitive, contraction]
        for c in range(cs.shape[1]):
            shells.append(Shell(l, centre, exps,
                                tuple(float(x) * common_factor(l)
                                      for x in cs[:, c])))
    return Basis(tuple(shells))


def from_mole(mol: Any) -> Basis:
    """Our shells for `mol`, one per contraction, in libcint's AO order.

    A libcint shell with n contractions becomes n consecutive `Shell`s: its
    AOs are contraction-major, then Cartesian component (`ao_labels` shows
    1s, 2s, ... then 2px, 2py, ...), which is exactly n `Shell`s in a row.
    `Basis.offsets` is the AO offset map; `mol.ao_loc_nr()[i]` is the offset
    of the first `Shell` libcint shell i became.
    """
    _require_cart(mol)
    return _shells(mol, mol._libcint_ctr_coeff)


def from_mole_file_coefficients(mol: Any) -> Basis:
    """The WRONG conversion, kept so a test can show it fails: the same, but
    with `bas_ctr_coeff` (gto_norm divided back out)."""
    _require_cart(mol)
    return _shells(mol, mol.bas_ctr_coeff)


def from_fixture(fx: Fixture) -> Basis:
    """Our shells from a fixture's `converted` record. Numpy only, no PySCF."""
    if fx.converted is None:
        raise ValueError(f"{fx.path}: records no converted basis")
    return Basis(tuple(Shell(s.l, s.centre, s.exponents, s.coefficients)
                       for s in fx.converted.shells))


def measured_common_factor(l: int) -> float:  # noqa: E741
    """libcint's common factor for `l`, as PySCF's data yields it.

    One atom, one primitive x^l e^(-a r^2), a = PROBE_EXPONENT. libcint's axis
    AO is cf * c * x^l e^(-a r^2), with c = `_libcint_ctr_coeff`, so its
    self-overlap S (from `int1e_ovlp`) is cf^2 c^2 / N^2, N our
    `primitive_norm`. Hence cf = sqrt(S) N / c. Nothing here assumes the
    value the constant table holds.
    """
    from pyscf import gto

    mol = gto.M(atom="He 0 0 0", unit="Bohr", cart=True, verbose=0,
                basis={"He": [[l, [PROBE_EXPONENT, 1.0]]]})
    s = mol.intor("int1e_ovlp")
    axis = 0  # cartesians(l)[0] is (l, 0, 0)
    c = float(mol._libcint_ctr_coeff(0)[0, 0])
    return math.sqrt(float(s[axis, axis])) * primitive_norm(l, PROBE_EXPONENT) / c


def converted_record(mol: Any) -> dict[str, Any]:
    """The fixture's `converted` slot for `mol` (see `manifest`'s schema)."""
    basis = from_mole(mol)
    atom_of = [int(mol.bas_atom(i)) for i in range(mol.nbas)
               for _ in range(int(mol.bas_nctr(i)))]
    lmax = max(s.l for s in basis.shells)
    return {
        "common_factor": [measured_common_factor(lq) for lq in range(lmax + 1)],
        "shells": [{"atom": a, "l": s.l, "centre": list(s.centre),
                    "exponents": list(s.exponents),
                    "coefficients": list(s.coefficients)}
                   for a, s in zip(atom_of, basis.shells, strict=True)],
    }


def ao_offsets(basis: Basis) -> tuple[int, ...]:
    """The AO offset map, with the total appended: libcint's `ao_loc` shape."""
    return (*basis.offsets, basis.n_ao)


__all__ = ["PROBE_EXPONENT", "ao_offsets", "common_factor", "converted_record",
           "from_fixture", "from_mole", "from_mole_file_coefficients",
           "measured_common_factor"]
