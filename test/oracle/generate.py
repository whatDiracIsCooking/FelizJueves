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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import FIXTURES, SCHEMA, dumps, encode_array, load

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
}


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
    return out


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
