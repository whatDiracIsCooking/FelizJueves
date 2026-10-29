"""A PySCF `Mole` as our shells -- and libcint's three conventions, pinned.

`from_mole(mol)` turns a PySCF molecule into a `felizjueves.ao.shell.Basis` in
OUR convention (coefficients multiplying UNNORMALISED Cartesian primitives
x^i y^j z^k exp(-alpha r^2); `ao/shell.py`'s module docstring), so that a
contracted function of ours and libcint's AO of the same index are the same
function, with the same AO index. `from_fixture(fx)` does the same from a
tracked fixture's `converted` record, with no PySCF. `converted_record(mol)` is
what `generate.py` writes into that record. `from_bse(basis_set, atoms)` does
it from a `data/basis/` file read by `felizjueves.ao.bse` (#650), with
PySCF's normalisation computed here, and `free_primitive_reduced` is the one
difference between those files and PySCF's own library.

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
from collections.abc import Sequence
from typing import Any

from felizjueves.ao.bse import BasisSet, Contraction
from felizjueves.ao.shell import Basis, Shell, primitive_norm

from .manifest import Fixture, MetricSet

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


def from_fixture(fx: Fixture | MetricSet) -> Basis:
    """Our shells from a fixture's `converted` record -- or, for a metric set
    (#411, `manifest.metric_set`), the aux basis its pieces record. Numpy
    only, no PySCF."""
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


def _gaussian_int(n: int, a: float) -> float:
    """int_0^inf r^n e^(-a r^2) dr -- PySCF's `gto.gaussian_int`."""
    return math.gamma((n + 1) / 2.0) / (2.0 * a ** ((n + 1) / 2.0))


def gto_norm(l: int, a: float) -> float:  # noqa: E741
    """libcint's radial normaliser, 1/sqrt(int r^(2l+2) e^(-2 a r^2) dr)."""
    return 1.0 / math.sqrt(_gaussian_int(2 * l + 2, 2.0 * a))


def libcint_order(contractions: Sequence[Contraction]) -> tuple[Contraction, ...]:
    """An element's contractions as libcint lays them out: by angular
    momentum, file order within one (PySCF's `make_bas_env`)."""
    return tuple(sorted(contractions, key=lambda c: c.l))


def free_exponents(contractions: Sequence[Contraction], l: int) -> frozenset[float]:  # noqa: E741
    """The exponents that stand alone at `l`: a contracted function that is
    one primitive with coefficient 1, whether its own shell or a unit column
    of a general contraction."""
    out = set()
    for c in contractions:
        if c.l != l:
            continue
        for row in c.coefficients:
            nonzero = [(a, x) for a, x in zip(c.exponents, row, strict=True) if x]
            if len(nonzero) == 1 and nonzero[0][1] == 1.0:
                out.add(nonzero[0][0])
    return frozenset(out)


def free_primitive_reduced(
        contractions: Sequence[Contraction]) -> tuple[Contraction, ...]:
    """`contractions` with every contracted function's components on a free
    exponent (`free_exponents`) dropped, one `Contraction` per function:
    every contracted one first, in file order, then every free primitive.

    This is the form PySCF's own library carries the correlation-consistent
    sets in, and BSE's are not (#650): BSE's cc-pVDZ O s has three columns
    over nine primitives, the third the free 0.3023, and the first two carry
    a 0.3023 component; PySCF's has the two over eight plus the 0.3023
    alone. A multiple of a free primitive added to a contraction leaves the
    SPAN unchanged, so the two give one SCF energy -- but different AOs, so
    a J matrix in one is not the J matrix in the other. Zero entries are
    dropped from every function; on a segmented set (STO-3G, 6-31G) that is
    the only change, and it changes nothing a `Shell` sees.
    """
    out = []
    for l in sorted({c.l for c in contractions}):  # noqa: E741
        free = free_exponents(contractions, l)
        contracted, alone = [], []
        for c in contractions:
            if c.l != l:
                continue
            for row in c.coefficients:
                keep = [(a, x) for a, x in zip(c.exponents, row, strict=True) if x]
                if len(keep) == 1 and keep[0][1] == 1.0:
                    alone.append(keep[0][0])
                    continue
                keep = [(a, x) for a, x in keep if a not in free]
                contracted.append(Contraction(
                    l, tuple(a for a, _ in keep), (tuple(x for _, x in keep),),
                    c.function_type))
        out.extend(contracted)
        out.extend(Contraction(l, (a,), ((1.0,),), "gto") for a in alone)
    return tuple(out)


def bse_coefficients(c: Contraction) -> tuple[tuple[float, ...], ...]:
    """OUR coefficients for each of `c`'s contracted functions: what
    `from_mole` gives for the same file read by PySCF, computed here.

    PySCF's `_nomalize_contracted_ao` multiplies each file coefficient by
    `gto_norm(l, alpha)` and then scales the contraction to unit RADIAL norm;
    `common_factor(l)` is then applied as `from_mole` applies it. A zero
    coefficient stays a zero, so a general contraction keeps every primitive
    in every column, as libcint does.
    """
    l = c.l  # noqa: E741
    g = [gto_norm(l, a) for a in c.exponents]
    out = []
    for row in c.coefficients:
        w = [x * gp for x, gp in zip(row, g, strict=True)]
        s = sum(wp * wq * _gaussian_int(2 * l + 2, ap + aq)
                for wp, ap in zip(w, c.exponents, strict=True)
                for wq, aq in zip(w, c.exponents, strict=True))
        scale = common_factor(l) / math.sqrt(s)
        out.append(tuple(wp * scale for wp in w))
    return tuple(out)


def from_bse(basis_set: BasisSet,
             atoms: Sequence[tuple[int, tuple[float, float, float]]],
             *, reduce: bool = False) -> Basis:
    """Our shells for `atoms` ((Z, centre in Bohr) each) in `basis_set`, in
    libcint's AO order -- `from_mole` for a Mole built over the same file
    with `cart=True`, with no PySCF. One `Shell` per contracted function.
    `reduce=True` takes each element's `free_primitive_reduced` form."""
    shells = []
    for z, centre in atoms:
        cs = basis_set.contractions(z)
        for c in (free_primitive_reduced(cs) if reduce else libcint_order(cs)):
            shells.extend(Shell(c.l, centre, c.exponents, coeffs)
                          for coeffs in bse_coefficients(c))
    return Basis(tuple(shells))


def ao_offsets(basis: Basis) -> tuple[int, ...]:
    """The AO offset map, with the total appended: libcint's `ao_loc` shape."""
    return (*basis.offsets, basis.n_ao)


__all__ = ["PROBE_EXPONENT", "ao_offsets", "bse_coefficients", "common_factor",
           "converted_record", "free_exponents", "free_primitive_reduced",
           "from_bse", "from_fixture", "from_mole",
           "from_mole_file_coefficients", "gto_norm", "libcint_order",
           "measured_common_factor"]
