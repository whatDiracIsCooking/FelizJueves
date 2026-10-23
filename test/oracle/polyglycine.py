"""Polyglycine/6-31G inputs for measure/ -- shells and a converged RHF density.

    PYTHONPATH=test:src uv run --group oracle python -m oracle.polyglycine [N ...]

writes `.measure-cache/polyglycine/gly{N}_631g.npz` (default N = 5 10 15)
from the tracked geometry `data/molecules/gly{N}.xyz` ([Fock, §6]'s
gly_n = H(C2H3NO)_n OH, neutral, closed shell). Cartesian 6-31G, so every
shell is s or p and every class is at most (pp|pp).

WHY A CACHE AND NOT A FIXTURE. gly15 has 643 AOs; its density is 413,449
doubles, ~9 MB as a fixture's JSON, far over the 512 KiB hook. Nothing here is
an oracle for a correctness test -- the inputs only set what a timed build
screens on -- so the file is untracked, and every measure row that reads one
records its `sha256` and `e_rhf`, which is what makes two runs comparable.
The density is PySCF's converged RHF one, symmetrised as
`generate.rhf_density` does, but under looser tolerances (`CONV_TOL`,
`CONV_TOL_GRAD`): gly5 does not reach `generate`'s 1e-12 / 1e-10, and a
density that only decides what a screen skips needs no more. The shells are
`basis.converted_record`'s, read back by `load` with no PySCF.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GEOMETRIES = REPO / "data" / "molecules"
CACHE = REPO / ".measure-cache" / "polyglycine"
BASIS = "6-31g"
DEFAULT_N = (5, 10, 15)
CONV_TOL = 1e-9  # energy, Eh
CONV_TOL_GRAD = 1e-6  # orbital gradient
MAX_CYCLE = 200
COMMAND = ("PYTHONPATH=test:src uv run --group oracle "
           "python -m oracle.polyglycine")


def xyz_path(n: int) -> Path:
    return GEOMETRIES / f"gly{n}.xyz"


def cache_path(n: int) -> Path:
    return CACHE / f"gly{n}_631g.npz"


def _atoms(path: Path) -> str:
    lines = path.read_text().splitlines()
    count = int(lines[0].split()[0])
    body = [ln.split() for ln in lines[2:2 + count]]
    if len(body) != count or any(len(b) != 4 for b in body):
        raise ValueError(f"{path}: not an xyz file of {count} atoms")
    return "; ".join(" ".join(b) for b in body)


def write(n: int) -> Path:
    """Run the SCF for gly{n} and write its cache file."""
    import pyscf
    from pyscf import gto, scf

    from .basis import converted_record

    geometry = xyz_path(n)
    mol = gto.M(atom=_atoms(geometry), unit="Angstrom", basis=BASIS,
                cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = CONV_TOL, CONV_TOL_GRAD
    mf.max_cycle = MAX_CYCLE
    mf.kernel()
    if not mf.converged:
        raise ValueError(f"gly{n}: RHF did not converge")
    dm = np.asarray(mf.make_rdm1(), dtype=np.float64)
    dm = 0.5 * (dm + dm.T)
    # The energy of the recorded density, not the SCF's last iterate's.
    e_rhf = float(mf.energy_tot(dm))
    meta = {
        "molecule": f"gly{n}",
        "basis": BASIS,
        "cart": True,
        "nao": int(mol.nao_nr()),
        "n_electrons": int(mol.nelectron),
        "e_rhf": e_rhf,
        "conv_tol": CONV_TOL,
        "conv_tol_grad": CONV_TOL_GRAD,
        "geometry": str(geometry.relative_to(REPO)),
        "geometry_sha256": hashlib.sha256(geometry.read_bytes()).hexdigest(),
        "pyscf": pyscf.__version__,
        "date": _dt.date.today().isoformat(),
        "command": COMMAND,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(n)
    np.savez(out, density=dm,
             converted=np.array(json.dumps(converted_record(mol))),
             meta=np.array(json.dumps(meta)))
    return out


class Inputs:
    """One cache file read back: `basis` (ours), `density`, `meta`, `sha256`."""

    def __init__(self, path: Path) -> None:
        from felizjueves.ao.shell import Basis, Shell

        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        with np.load(path) as z:
            self.density = np.array(z["density"])
            converted = json.loads(str(z["converted"]))
            self.meta: dict[str, Any] = json.loads(str(z["meta"]))
        self.basis = Basis(tuple(
            Shell(l=s["l"], centre=tuple(s["centre"]),
                  exponents=tuple(s["exponents"]),
                  coefficients=tuple(s["coefficients"]))
            for s in converted["shells"]))
        if self.density.shape != (self.basis.n_ao,) * 2:
            raise ValueError(f"{path}: density {self.density.shape} against "
                             f"{self.basis.n_ao} AOs")


def load(n: int) -> Inputs:
    """gly{n}'s cache, or FileNotFoundError naming the command that writes it."""
    path = cache_path(n)
    if not path.exists():
        raise FileNotFoundError(f"{path} is absent -- write it with "
                                f"`{COMMAND} {n}`")
    inputs = Inputs(path)
    geometry = xyz_path(n)
    if (hashlib.sha256(geometry.read_bytes()).hexdigest()
            != inputs.meta["geometry_sha256"]):
        raise ValueError(f"{path} was written from another {geometry}; "
                         f"rewrite it with `{COMMAND} {n}`")
    return inputs


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("n", nargs="*", type=int, default=list(DEFAULT_N))
    args = p.parse_args(argv)
    try:
        import pyscf  # noqa: F401
    except ImportError:
        print("polyglycine: PySCF absent -- `uv sync --group oracle`",
              file=sys.stderr)
        return 2
    for n in args.n:
        out = write(n)
        meta = load(n).meta
        print(f"{out.relative_to(REPO)}: nao {meta['nao']}, "
              f"E_RHF {meta['e_rhf']:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
