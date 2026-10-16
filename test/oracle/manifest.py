"""The fixture schema, its loader, and the provenance record -- NUMPY ONLY.

Nothing here imports PySCF, so a fast-tier test can load and validate every
tracked fixture on a box where the `oracle` group was never installed. The
writer is `generate.py`; this module only defines what a fixture IS and refuses
anything that is not one, with a message naming the file.

THE SHAPE (schema 2), one JSON object per file under `fixtures/`:

    schema       2 (#240: schema 1 had no one-electron arrays; a schema-1
                 file is refused by name, not read as one with them absent)
    name         the file's stem
    description  free text: what the case is for
    provenance   {pyscf, libcint, numpy, date, host, command, mol, basis, cart}
                 -- exactly these keys. Versions, an ISO date, the host (marked
                 "(container)" inside one, as `measured` marks it), the command
                 that wrote the file, the molecule as PySCF was given it
                 ({atom, unit, charge, spin}), the basis name, and whether the
                 AOs are Cartesian.
    atoms        [{symbol, charge, coord}]          coord in Bohr
    shells       [{atom, l, exponents, coefficients}]
                 the basis as PySCF's library parsed it (`Mole._basis`), in
                 libcint's shell order; coefficients[k][c] is primitive k of
                 contraction c, NOT normalised -- normalisation conventions are
                 #84's to pin, so the raw numbers are what is recorded.
    nao          number of AOs
    ao_labels    PySCF's `ao_labels()`, one per AO
    ao_loc       shell offsets into the AO list, len(shells) + 1 entries
    arrays       {name: {shape, values}} -- values flattened row-major
                 `density` (nao x nao, exactly symmetric) is required.
                 `int2e:i,j,k,l` (OPTIONAL, #85; needs `converted`) is
                 libcint's Cartesian `(ij|kl)` block for converted shells
                 i, j, k, l, shape (n_i, n_j, n_k, n_l), AOs in `cartesians`
                 order -- one shell quartet, never the whole tensor
                 `vj` (OPTIONAL, #86; needs `converted`) is PySCF's
                 `get_jk` Coulomb matrix (nao x nao) for `density`, AO order
                 `vk` (OPTIONAL, #199; needs `converted`) is PySCF's
                 `get_jk` exchange matrix (nao x nao) for `density`, AO
                 order: `vk_il = sum_jk (ij|kl) D_jk`, which is our
                 `K_ab = sum_cd D_cd (ac|bd)`
                 `int1e_ovlp`, `int1e_kin`, `int1e_nuc`, `hcore` (OPTIONAL,
                 #240; each needs `converted`, all four or none) are
                 libcint's `S`, `T`, `V` (point nuclei, `-Z_C` folded in)
                 and PySCF's `get_hcore` (`T + V`), nao x nao, AO order.
                 `int1e_ovlp` is the same `intor` call as `ovlp` and must
                 equal it exactly; the legacy `ovlp` stays in every file.
                 `int1e_ovlp_cross:<rows>` (OPTIONAL; needs `converted`) is
                 libcint's rectangular overlap `<m|n>`, shape
                 (rows' nao, nao): `m` an AO of the tracked fixture `<rows>`
                 (which must carry `converted` and the same `cart`), `n` of
                 this one -- the oracle of `S_{B1,B2}` (`Target.SX`);
                 `cross_overlap(fx)` loads the partner and checks it.
    density_recipe  how the recorded density was made
    converted    OPTIONAL. {common_factor, shells}: the basis in OUR convention
                 (`ao.shell.Shell`, one entry per contraction, in AO order) as
                 `oracle.basis` converts it -- coefficients are libcint's
                 normalised `_libcint_ctr_coeff` times `common_factor(l)`, to
                 multiply unnormalised Cartesian primitives -- and
                 `common_factor`, libcint's `common_fac` per l = 0.. as PySCF's
                 data yields it. Present only where a case records it; it lets
                 a later test build our shells with no PySCF. Cartesian only.

THE FOCK PAIR (#280, milestone 12 pin 10). No fixture records the
two-electron Fock matrix; it is COMBINED, never stored, from a `<m>_j` /
`<m>_k` pair, which `fock_pair(m)` loads and `check_pair` asserts share
their inputs before anything is combined: the raw `shells`, the `density`
(exactly), `ao_loc`, `converted.common_factor` and `provenance.mol` (the
issue's five), and also `provenance.basis` / `cart`, `atoms`, `nao`,
`ao_labels`, the converted shells, the density recipe, and every array the
two files both carry (`ovlp`, the one-electron matrices) -- each exactly.
A pair regenerated at different times from different molecules would pass
numerically on a small case and be meaningless, so a mismatch is a
`FixtureError` naming both files and the field, never a tolerance. The
combined quantity is `FockPair.two_electron() = vj - vk / 2`: PySCF's RHF
`vj - 0.5 vk` for the total closed-shell density, `FOCK_EXCHANGE_FRACTION`
= 1/2 (derived in `felizjueves.ao.fock`, which this module must not import;
test/test_oracle_fock.py asserts the two constants agree). `fock_pairs()`
names every tracked molecule that has both halves.

FULL PRECISION. Floats are written by `json`, i.e. by `repr`, which is the
shortest string that round-trips the double exactly, so "the numbers match"
can mean `==` and not a tolerance. `dumps` is the one serialiser, and a
tracked fixture is byte-for-byte `dumps(load(...))`.

`python -m oracle.manifest` (PYTHONPATH=test) loads every tracked fixture and
prints one line each; devtools/doctor.sh's oracle row runs exactly that.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 2
FIXTURES = Path(__file__).resolve().parent / "fixtures"

PROVENANCE_KEYS = (
    "pyscf", "libcint", "numpy", "date", "host", "command", "mol", "basis", "cart",
)
MOL_KEYS = ("atom", "unit", "charge", "spin")
TOP_KEYS = (
    "schema", "name", "description", "provenance", "atoms", "shells", "nao",
    "ao_labels", "ao_loc", "arrays", "density_recipe",
)
OPTIONAL_KEYS = ("converted",)
REQUIRED_ARRAYS = ("density",)
ERI_PREFIX = "int2e:"  # an ERI block's array name: int2e:i,j,k,l (#85)
J_ARRAY = "vj"  # PySCF get_jk's Coulomb matrix for `density` (#86)
K_ARRAY = "vk"  # PySCF get_jk's exchange matrix for `density` (#199)
# #240: libcint's one-electron matrices and PySCF's core Hamiltonian.
OVLP_ARRAY = "int1e_ovlp"
KIN_ARRAY = "int1e_kin"
NUC_ARRAY = "int1e_nuc"
HCORE_ARRAY = "hcore"
SCHWARZ_ARRAY = "int2e_diag"  # libcint's (mn|mn) for every AO pair
# libcint's <m|n>, m an AO of the named fixture (rows), n of this one
CROSS_PREFIX = "int1e_ovlp_cross:"
ONE_E_ARRAYS = (OVLP_ARRAY, KIN_ARRAY, NUC_ARRAY, HCORE_ARRAY)


class FixtureError(ValueError):
    """A fixture file that is not a valid fixture. The message names the file."""


@dataclass(frozen=True)
class Provenance:
    """Who computed a fixture, with what, and from which inputs."""

    pyscf: str
    libcint: str
    numpy: str
    date: str
    host: str
    command: str
    mol: dict[str, Any]
    basis: str
    cart: bool


@dataclass(frozen=True)
class Atom:
    symbol: str
    charge: int
    coord: tuple[float, float, float]  # Bohr


@dataclass(frozen=True)
class Shell:
    """One contracted shell, as PySCF's basis library gives it (unnormalised)."""

    atom: int
    l: int  # noqa: E741 -- the angular momentum's own name
    exponents: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]  # [primitive][contraction]


@dataclass(frozen=True)
class ConvertedShell:
    """One contraction in our convention (`felizjueves.ao.shell.Shell`'s fields
    plus the atom it sits on). Plain data: this module must not import `src`."""

    atom: int
    l: int  # noqa: E741
    centre: tuple[float, float, float]  # Bohr
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class Converted:
    common_factor: tuple[float, ...]  # indexed by l
    shells: tuple[ConvertedShell, ...]


@dataclass(frozen=True)
class Fixture:
    path: Path
    name: str
    description: str
    provenance: Provenance
    atoms: tuple[Atom, ...]
    shells: tuple[Shell, ...]
    nao: int
    ao_labels: tuple[str, ...]
    ao_loc: tuple[int, ...]
    arrays: dict[str, np.ndarray]  # read-only
    density_recipe: str
    converted: Converted | None = None

    @property
    def density(self) -> np.ndarray:
        return self.arrays["density"]


# --- serialisation ---------------------------------------------------------


def encode_array(a: np.ndarray) -> dict[str, Any]:
    """`{shape, values}`, row-major, every value a Python float (full repr)."""
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "values": [float(x) for x in a.ravel()]}


def dumps(obj: dict[str, Any]) -> str:
    """The one serialiser: what a tracked fixture's bytes are."""
    return json.dumps(obj, indent=1, allow_nan=False) + "\n"


# --- loading and validation ------------------------------------------------


def _fail(path: Path, msg: str) -> FixtureError:
    return FixtureError(f"{path}: {msg}")


def _is_num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _floats(path: Path, where: str, xs: Any, n: int | None = None) -> tuple:
    if not isinstance(xs, list) or not all(_is_num(x) for x in xs):
        raise _fail(path, f"{where} must be a list of numbers")
    if n is not None and len(xs) != n:
        raise _fail(path, f"{where} has {len(xs)} entries, expected {n}")
    if not all(math.isfinite(x) for x in xs):
        raise _fail(path, f"{where} has a non-finite value")
    return tuple(float(x) for x in xs)


def _provenance(path: Path, p: Any) -> Provenance:
    if not isinstance(p, dict):
        raise _fail(path, "provenance must be an object")
    missing = [k for k in PROVENANCE_KEYS if k not in p]
    if missing:
        raise _fail(path, f"provenance is missing key(s) {missing}")
    extra = sorted(set(p) - set(PROVENANCE_KEYS))
    if extra:
        raise _fail(path, f"provenance has unknown key(s) {extra}")
    for k in ("pyscf", "libcint", "numpy", "date", "host", "command", "basis"):
        if not isinstance(p[k], str) or not p[k]:
            raise _fail(path, f"provenance.{k} must be a non-empty string")
    if not isinstance(p["cart"], bool):
        raise _fail(path, "provenance.cart must be true or false")
    mol = p["mol"]
    if not isinstance(mol, dict) or sorted(mol) != sorted(MOL_KEYS):
        raise _fail(path, f"provenance.mol must have exactly the keys {MOL_KEYS}")
    return Provenance(**{k: p[k] for k in PROVENANCE_KEYS})


def _arrays(path: Path, raw: Any) -> dict[str, np.ndarray]:
    if not isinstance(raw, dict):
        raise _fail(path, "arrays must be an object")
    missing = [k for k in REQUIRED_ARRAYS if k not in raw]
    if missing:
        raise _fail(path, f"arrays is missing {missing}")
    out = {}
    for name, a in raw.items():
        if not isinstance(a, dict) or sorted(a) != ["shape", "values"]:
            raise _fail(path, f"arrays.{name} must be {{shape, values}}")
        shape = a["shape"]
        if not (isinstance(shape, list) and shape
                and all(isinstance(d, int) and d > 0 for d in shape)):
            raise _fail(path, f"arrays.{name}.shape must be positive integers")
        need = math.prod(shape)
        values = a["values"]
        if isinstance(values, list) and len(values) != need:
            raise _fail(path, f"arrays.{name} is truncated or padded: "
                              f"{len(values)} values for shape {shape} "
                              f"(needs {need})")
        vals = _floats(path, f"arrays.{name}.values", values, need)
        arr = np.array(vals, dtype=np.float64).reshape(shape)
        arr.setflags(write=False)
        out[name] = arr
    return out


def validate(path: Path, raw: Any) -> Fixture:
    """Turn a parsed JSON object into a `Fixture`, or raise naming `path`."""
    path = Path(path)
    if not isinstance(raw, dict):
        raise _fail(path, "a fixture must be a JSON object")
    missing = [k for k in TOP_KEYS if k not in raw]
    if missing:
        raise _fail(path, f"missing key(s) {missing}")
    extra = sorted(set(raw) - set(TOP_KEYS) - set(OPTIONAL_KEYS))
    if extra:
        raise _fail(path, f"unknown key(s) {extra}")
    if raw["schema"] == 1:
        raise _fail(path, f"schema 1, this loader reads {SCHEMA}: the bump to "
                          "2 (#240) added the one-electron arrays -- "
                          "regenerate it (python -m oracle.generate "
                          "--regenerate)")
    if raw["schema"] != SCHEMA:
        raise _fail(path, f"schema {raw['schema']!r}, this loader reads {SCHEMA}")
    if raw["name"] != path.stem:
        raise _fail(path, f"name {raw['name']!r} is not the file's stem")

    prov = _provenance(path, raw["provenance"])

    atoms = []
    for i, a in enumerate(raw["atoms"] if isinstance(raw["atoms"], list) else []):
        if not isinstance(a, dict) or sorted(a) != ["charge", "coord", "symbol"]:
            raise _fail(path, f"atoms[{i}] must be {{symbol, charge, coord}}")
        atoms.append(Atom(a["symbol"], int(a["charge"]),
                          _floats(path, f"atoms[{i}].coord", a["coord"], 3)))
    if not atoms:
        raise _fail(path, "atoms must be a non-empty list")

    shells = []
    for i, s in enumerate(raw["shells"] if isinstance(raw["shells"], list) else []):
        keys = ["atom", "coefficients", "exponents", "l"]
        if not isinstance(s, dict) or sorted(s) != keys:
            raise _fail(path, f"shells[{i}] must be {{atom, l, exponents, "
                              "coefficients}")
        if not (isinstance(s["atom"], int) and 0 <= s["atom"] < len(atoms)):
            raise _fail(path, f"shells[{i}].atom is not an atom index")
        if not (isinstance(s["l"], int) and s["l"] >= 0):
            raise _fail(path, f"shells[{i}].l must be a non-negative integer")
        exps = _floats(path, f"shells[{i}].exponents", s["exponents"])
        if not exps or min(exps) <= 0:
            raise _fail(path, f"shells[{i}].exponents must be positive")
        coefs = s["coefficients"]
        if not isinstance(coefs, list) or len(coefs) != len(exps):
            raise _fail(path, f"shells[{i}].coefficients needs one row per "
                              f"primitive ({len(exps)})")
        rows = [_floats(path, f"shells[{i}].coefficients[{k}]", r)
                for k, r in enumerate(coefs)]
        if len({len(r) for r in rows}) != 1 or not rows[0]:
            raise _fail(path, f"shells[{i}].coefficients is ragged")
        shells.append(Shell(s["atom"], s["l"], exps, tuple(rows)))
    if not shells:
        raise _fail(path, "shells must be a non-empty list")

    nao = raw["nao"]
    if not (isinstance(nao, int) and nao > 0):
        raise _fail(path, "nao must be a positive integer")
    ao_loc = raw["ao_loc"]
    if not (isinstance(ao_loc, list) and all(isinstance(x, int) for x in ao_loc)
            and len(ao_loc) == len(shells) + 1 and ao_loc[0] == 0
            and ao_loc[-1] == nao):
        raise _fail(path, f"ao_loc must be {len(shells) + 1} integer offsets "
                          f"from 0 to nao={nao}")
    for i, sh in enumerate(shells):
        ncomp = (sh.l + 1) * (sh.l + 2) // 2 if prov.cart else 2 * sh.l + 1
        width = ao_loc[i + 1] - ao_loc[i]
        if width != ncomp * len(sh.coefficients[0]):
            raise _fail(path, f"ao_loc gives shell {i} {width} AOs, its l and "
                              "contractions need "
                              f"{ncomp * len(sh.coefficients[0])}")
    labels = raw["ao_labels"]
    if not (isinstance(labels, list) and len(labels) == nao
            and all(isinstance(x, str) for x in labels)):
        raise _fail(path, f"ao_labels must be {nao} strings")

    arrays = _arrays(path, raw["arrays"])
    d = arrays["density"]
    if d.shape != (nao, nao):
        raise _fail(path, f"density has shape {d.shape}, expected ({nao}, {nao})")
    if not np.array_equal(d, d.T):
        raise _fail(path, "density is not exactly symmetric")
    for k in ("description", "density_recipe"):
        if not isinstance(raw[k], str) or not raw[k]:
            raise _fail(path, f"{k} must be a non-empty string")

    converted = (_converted(path, raw["converted"], prov, len(atoms), shells,
                            nao)
                 if "converted" in raw else None)
    _check_eri(path, arrays, converted)
    one_e = [n for n in ONE_E_ARRAYS if n in arrays]
    if one_e and len(one_e) != len(ONE_E_ARRAYS):
        raise _fail(path, f"arrays has {one_e} of the one-electron arrays "
                          f"{list(ONE_E_ARRAYS)}: all four or none")
    for name in (J_ARRAY, K_ARRAY, SCHWARZ_ARRAY, *ONE_E_ARRAYS):
        if name not in arrays:
            continue
        if converted is None:
            raise _fail(path, f"arrays.{name} needs the converted basis "
                              "its AOs are in")
        if arrays[name].shape != (nao, nao):
            raise _fail(path, f"arrays.{name} has shape "
                              f"{arrays[name].shape}, expected "
                              f"({nao}, {nao})")
    if SCHWARZ_ARRAY in arrays:
        q = arrays[SCHWARZ_ARRAY]
        # (mn|mn) and (nm|nm) are equal exactly but computed apart, so the
        # record is symmetric to rounding, not bitwise
        if not np.all(np.abs(q - q.T) <= 1e-14 * np.max(np.abs(q))):
            raise _fail(path, f"arrays.{SCHWARZ_ARRAY} is not symmetric to "
                              "1e-14 of its largest element")
        if np.any(q < 0.0):
            raise _fail(path, f"arrays.{SCHWARZ_ARRAY} has a negative "
                              "(mn|mn)")
    for name, a in arrays.items():
        if not name.startswith(CROSS_PREFIX):
            continue
        rows = name[len(CROSS_PREFIX):]
        if not rows or rows != Path(rows).name or rows == raw["name"]:
            raise _fail(path, f"arrays.{name} must name another fixture "
                              "(its stem) as its rows")
        if converted is None:
            raise _fail(path, f"arrays.{name} needs the converted basis "
                              "its columns are in")
        if a.ndim != 2 or a.shape[1] != nao:
            raise _fail(path, f"arrays.{name} has shape {a.shape}, expected "
                              f"(rows' nao, {nao})")
    if one_e and not np.array_equal(arrays[OVLP_ARRAY], arrays["ovlp"]):
        raise _fail(path, f"arrays.{OVLP_ARRAY} is not exactly arrays.ovlp")

    return Fixture(path, raw["name"], raw["description"], prov, tuple(atoms),
                   tuple(shells), nao, tuple(labels), tuple(ao_loc), arrays,
                   raw["density_recipe"], converted)


def _converted(path: Path, c: Any, prov: Provenance, n_atoms: int,
               shells: list[Shell], nao: int) -> Converted:
    if not isinstance(c, dict) or sorted(c) != ["common_factor", "shells"]:
        raise _fail(path, "converted must be {common_factor, shells}")
    if not prov.cart:
        raise _fail(path, "converted is Cartesian only, and provenance.cart "
                          "is false")
    common = _floats(path, "converted.common_factor", c["common_factor"])
    out = []
    raw = c["shells"] if isinstance(c["shells"], list) else []
    for i, s in enumerate(raw):
        keys = ["atom", "centre", "coefficients", "exponents", "l"]
        if not isinstance(s, dict) or sorted(s) != keys:
            raise _fail(path, f"converted.shells[{i}] must be {{atom, l, "
                              "centre, exponents, coefficients}")
        if not (isinstance(s["atom"], int) and 0 <= s["atom"] < n_atoms):
            raise _fail(path, f"converted.shells[{i}].atom is not an atom index")
        if not (isinstance(s["l"], int) and 0 <= s["l"] < len(common)):
            raise _fail(path, f"converted.shells[{i}].l has no common_factor")
        centre = _floats(path, f"converted.shells[{i}].centre", s["centre"], 3)
        exps = _floats(path, f"converted.shells[{i}].exponents", s["exponents"])
        if not exps or min(exps) <= 0:
            raise _fail(path, f"converted.shells[{i}].exponents must be positive")
        coefs = _floats(path, f"converted.shells[{i}].coefficients",
                        s["coefficients"], len(exps))
        out.append(ConvertedShell(s["atom"], s["l"], centre, exps, coefs))
    want = sum(len(sh.coefficients[0]) for sh in shells)
    if len(out) != want:
        raise _fail(path, f"converted has {len(out)} shells, the basis has "
                          f"{want} contractions")
    n = sum((s.l + 1) * (s.l + 2) // 2 for s in out)
    if n != nao:
        raise _fail(path, f"converted shells give {n} AOs, nao is {nao}")
    return Converted(common, tuple(out))


def _eri_quartet(name: str) -> tuple[int, ...] | None:
    """`int2e:i,j,k,l` -> (i, j, k, l); None for any other array name."""
    if not name.startswith(ERI_PREFIX):
        return None
    try:
        q = tuple(int(x) for x in name[len(ERI_PREFIX):].split(","))
    except ValueError:
        return ()
    return q if len(q) == 4 and min(q) >= 0 else ()


def _check_eri(path: Path, arrays: dict[str, np.ndarray],
               converted: Converted | None) -> None:
    for name, a in arrays.items():
        q = _eri_quartet(name)
        if q is None:
            continue
        if not q:
            raise _fail(path, f"arrays.{name} is not int2e:i,j,k,l")
        if converted is None:
            raise _fail(path, f"arrays.{name} needs the converted basis its "
                              "indices name")
        if max(q) >= len(converted.shells):
            raise _fail(path, f"arrays.{name} names a shell the converted "
                              "basis does not have")
        want = tuple((converted.shells[s].l + 1) * (converted.shells[s].l + 2)
                     // 2 for s in q)
        if a.shape != want:
            raise _fail(path, f"arrays.{name} has shape {a.shape}, its shells "
                              f"need {want}")


def eri_blocks(fx: Fixture) -> dict[tuple[int, int, int, int], np.ndarray]:
    """Every `int2e:i,j,k,l` block of `fx`, keyed by its converted-shell
    quartet (validated on load)."""
    return {q: a for name, a in fx.arrays.items()
            if (q := _eri_quartet(name))}


def load(path: Path | str) -> Fixture:
    """Read and validate one fixture file. Every failure names the file."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except OSError as e:
        raise _fail(path, f"cannot read: {e}") from e
    except json.JSONDecodeError as e:
        raise _fail(path, f"not valid JSON (truncated?): {e}") from e
    return validate(path, raw)


def tracked() -> list[Path]:
    """Every tracked fixture file, sorted."""
    return sorted(FIXTURES.glob("*.json"))


# --- the cross-basis overlap ------------------------------------------------


def cross_overlap(fx: Fixture) -> tuple[Fixture, np.ndarray]:
    """`(rows, S)` of a fixture carrying `int1e_ovlp_cross:<rows>`: the row
    fixture, loaded from beside `fx`, and the `rows.nao x fx.nao` array.
    Refuses a row fixture that is absent, has no converted basis, differs in
    `cart`, or whose `nao` is not the array's row count."""
    names = [n for n in fx.arrays if n.startswith(CROSS_PREFIX)]
    if len(names) != 1:
        raise _fail(fx.path, f"carries {len(names)} {CROSS_PREFIX}* arrays, "
                             "not one")
    (name,) = names
    rpath = fx.path.parent / f"{name[len(CROSS_PREFIX):]}.json"
    if not rpath.exists():
        raise _fail(fx.path, f"arrays.{name}: no fixture {rpath.name}")
    rows = load(rpath)
    S = fx.arrays[name]
    if rows.converted is None:
        raise _fail(fx.path, f"arrays.{name}: {rpath.name} has no converted "
                             "basis")
    if rows.provenance.cart != fx.provenance.cart:
        raise _fail(fx.path, f"arrays.{name}: {rpath.name} differs in cart")
    if S.shape != (rows.nao, fx.nao):
        raise _fail(fx.path, f"arrays.{name} has shape {S.shape}, expected "
                             f"({rows.nao}, {fx.nao})")
    return rows, S


def cross_fixtures() -> list[Path]:
    """Every tracked fixture carrying a cross-basis overlap, sorted."""
    return [p for p in tracked()
            if any(n.startswith(CROSS_PREFIX) for n in load(p).arrays)]


# --- the Fock pair (#280) ----------------------------------------------------

J_SUFFIX, K_SUFFIX = "_j", "_k"
# G = J - K / N_SPIN for the total closed-shell density (felizjueves.ao.fock's
# derivation; PySCF's RHF vj - 0.5 vk). Not imported: this module stays numpy.
FOCK_EXCHANGE_FRACTION = 0.5


@dataclass(frozen=True)
class FockPair:
    """A `<name>_j` / `<name>_k` fixture pair that `check_pair` accepted."""

    name: str
    j: Fixture
    k: Fixture

    @property
    def density(self) -> np.ndarray:
        return self.j.density

    @property
    def vj(self) -> np.ndarray:
        return self.j.arrays[J_ARRAY]

    @property
    def vk(self) -> np.ndarray:
        return self.k.arrays[K_ARRAY]

    def two_electron(self) -> np.ndarray:
        """`G = vj - vk / 2`, PySCF's two-electron Fock matrix for the pair's
        (total, closed-shell) density. `F = hcore + G`."""
        g = self.vj - FOCK_EXCHANGE_FRACTION * self.vk
        g.setflags(write=False)
        return g


def _pair_fail(fj: Fixture, fk: Fixture, msg: str) -> FixtureError:
    return FixtureError(f"{fj.path} / {fk.path}: not a Fock pair: {msg}")


def check_pair(fj: Fixture, fk: Fixture) -> None:
    """Refuse, naming both files and the field, unless `fj` (carrying `vj`)
    and `fk` (carrying `vk`) were computed from the same inputs. Every
    comparison is exact (module docstring: THE FOCK PAIR)."""
    if J_ARRAY not in fj.arrays:
        raise _pair_fail(fj, fk, f"the first carries no {J_ARRAY}")
    if K_ARRAY not in fk.arrays:
        raise _pair_fail(fj, fk, f"the second carries no {K_ARRAY}")
    if fj.converted is None or fk.converted is None:
        raise _pair_fail(fj, fk, "both must record the converted basis")
    pj, pk = fj.provenance, fk.provenance
    fields = (  # the issue's five first, so the message names the first rot
        ("shells", fj.shells, fk.shells),
        ("ao_loc", fj.ao_loc, fk.ao_loc),
        ("converted.common_factor", fj.converted.common_factor,
         fk.converted.common_factor),
        ("provenance.mol", pj.mol, pk.mol),
        ("provenance.basis", pj.basis, pk.basis),
        ("provenance.cart", pj.cart, pk.cart),
        ("atoms", fj.atoms, fk.atoms),
        ("nao", fj.nao, fk.nao),
        ("ao_labels", fj.ao_labels, fk.ao_labels),
        ("converted.shells", fj.converted.shells, fk.converted.shells),
        ("density_recipe", fj.density_recipe, fk.density_recipe),
    )
    for name, a, b in fields[:3]:
        if a != b:
            raise _pair_fail(fj, fk, f"{name} differ")
    if not np.array_equal(fj.density, fk.density):
        raise _pair_fail(fj, fk, "density differ")
    for name, a, b in fields[3:]:
        if a != b:
            raise _pair_fail(fj, fk, f"{name} differ")
    shared = sorted(set(fj.arrays) & set(fk.arrays) - {"density"})
    for name in shared:
        if not np.array_equal(fj.arrays[name], fk.arrays[name]):
            raise _pair_fail(fj, fk, f"arrays.{name} differ")


def fock_pair(name: str, fixtures: Path = FIXTURES) -> FockPair:
    """Load `<name>_j` and `<name>_k` and check them as a pair."""
    fj = load(fixtures / f"{name}{J_SUFFIX}.json")
    fk = load(fixtures / f"{name}{K_SUFFIX}.json")
    check_pair(fj, fk)
    return FockPair(name, fj, fk)


def fock_pairs() -> list[str]:
    """Every tracked molecule with both a `_j` and a `_k` fixture, sorted.
    By file name only; `fock_pair` is what checks one."""
    stems = {p.stem for p in tracked()}
    return sorted(s[:-len(J_SUFFIX)] for s in stems
                  if s.endswith(J_SUFFIX)
                  and s[:-len(J_SUFFIX)] + K_SUFFIX in stems)


def main(argv: list[str] | None = None) -> int:
    """Load every tracked fixture; one line each. Exit 1 on the first bad one."""
    paths = [Path(p) for p in (argv or [])] or tracked()
    if not paths:
        print(f"no fixtures under {FIXTURES}")
        return 1
    for p in paths:
        try:
            f = load(p)
        except FixtureError as e:
            print(f"BAD {e}")
            return 1
        pv = f.provenance
        print(f"{f.name}: nao={f.nao}, generated {pv.date} on {pv.host} "
              f"(pyscf {pv.pyscf}, libcint {pv.libcint})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
