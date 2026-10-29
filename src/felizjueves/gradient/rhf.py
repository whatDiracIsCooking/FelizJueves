"""The closed-shell RHF nuclear gradient, assembled from its terms (#565).

    Runners(overlap, kinetic, potential, coulomb, exchange)
    host_runners(one_centre=)                       -> Runners, `gradient.host`
    rhf_gradient(basis, nuclei, D, W, runners)      -> RhfGradient
    TERMS, SIGN                                     the terms, and each factor

THE ENERGY. With `D` the TOTAL density (`2 C_occ C_occ^T`) and `W` the
total energy-weighted one (`2 C_occ diag(eps_occ) C_occ^T`, PySCF's
`make_rdm1e`),

    E = SUM D h + E_J - 1/2 E_K + V_nn,
    E_J = 1/2 SUM D_ab D_cd (ab|cd),  E_K = 1/2 SUM D_ab D_cd (ac|bd)

and, the orbitals being stationary and orthonormal through `S`,

    dE/dR = - SUM W dS/dR + SUM D dT/dR + SUM D dV/dR
            + dE_J/dR - 1/2 dE_K/dR + dV_nn/dR

`V`'s term carries its Hellmann-Feynman part (`gv_` computes it per
nucleus; #557 superseded). Each runner returns `G[A][k] = SUM M dX/dR_{A,k}`
(the package docstring's contract: `M = W` for `S`, `D` for `T` and `V`;
`dE_J` and `dE_K` themselves for J and K), so the sign and the half are
applied HERE, once (`SIGN`): a term of `RhfGradient` is its signed
contribution and the six add to `total`. `V_nn^x` is
`gradient.host.nuclear_repulsion_gradient` (#552 superseded), on the host
whichever runners ran.

THE RUNNERS ARE INJECTED, `(basis, nuclei, M) -> n_atom x 3`:
`host_runners` here (the references), `gradient.interpreted.
interpreted_runners` (every term from the generated kernels, run in
`ir.interp`), a device harness's on a card. Nothing here knows which ran.

THE SUM. `total[A][k]` adds the six signed terms in `TERMS` order, left to
right, so it is a fixed function of the terms; each term's own summation
order is its runner's (the device kernels write `G` by `atomicAdd`, #552
superseded, so a term from a card is not bit-reproducible).

Pure Python floats: every runner's `G` is read through `float()`, and the
result is tuples.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from felizjueves.ao.pairs import OneCentre
from felizjueves.ao.shell import Basis
from felizjueves.gradient import host
from felizjueves.one_electron.potential import Nucleus

Matrix = Sequence[Sequence[float]]
Grad = tuple[tuple[float, float, float], ...]
Runner = Callable[[Basis, Sequence[Nucleus], Matrix], Sequence[Sequence[float]]]

TERMS = ("overlap", "kinetic", "potential", "coulomb", "exchange", "nuclear")
# The factor each runner's G enters dE/dR with (module docstring).
SIGN = {"overlap": -1.0, "kinetic": 1.0, "potential": 1.0, "coulomb": 1.0,
        "exchange": -0.5, "nuclear": 1.0}


@dataclass(frozen=True)
class Runners:
    """One `G = SUM M dX/dR` per electronic term (module docstring)."""

    overlap: Runner  # M = W
    kinetic: Runner  # M = D
    potential: Runner  # M = D, Hellmann-Feynman included
    coulomb: Runner  # dE_J/dR
    exchange: Runner  # dE_K/dR, E_K = 1/2 SUM D_ab D_cd (ac|bd)


@dataclass(frozen=True)
class RhfGradient:
    """Every term's signed contribution to `dE/dR` and their sum, each
    `n_atom x 3`, xyz fastest."""

    overlap: Grad
    kinetic: Grad
    potential: Grad
    coulomb: Grad
    exchange: Grad
    nuclear: Grad
    total: Grad

    def term(self, name: str) -> Grad:
        if name not in (*TERMS, "total"):
            raise ValueError(f"no term {name!r}; the terms are {TERMS}")
        return getattr(self, name)


def host_runners(*, one_centre: OneCentre) -> Runners:
    """`gradient.host`'s references as runners, `[t]^(0)` injected."""
    return Runners(
        overlap=lambda b, n, M: host.overlap_gradient(b, M, n),
        kinetic=lambda b, n, M: host.kinetic_gradient(b, M, n),
        potential=lambda b, n, M: host.potential_gradient(
            b, M, n, one_centre=one_centre),
        coulomb=lambda b, n, M: host.coulomb_gradient(
            b, M, n, one_centre=one_centre),
        exchange=lambda b, n, M: host.exchange_gradient(
            b, M, n, one_centre=one_centre))


def _grad(G: Sequence[Sequence[float]], n_atom: int, name: str) -> Grad:
    rows = [list(r) for r in G]
    if len(rows) != n_atom or any(len(r) != 3 for r in rows):
        raise ValueError(f"{name}: G is not {n_atom} x 3")
    f = SIGN[name]
    return tuple((f * float(r[0]), f * float(r[1]), f * float(r[2]))
                 for r in rows)


def rhf_gradient(basis: Basis, nuclei: Sequence[Nucleus], D: Matrix,
                 W: Matrix, runners: Runners) -> RhfGradient:
    """`dE_RHF/dR` and its terms (module docstring), `D` and `W` total."""
    nuc = tuple(nuclei)
    n = len(nuc)
    host.shell_atoms(basis, nuc)  # every shell on a nucleus, or refuse here
    terms: dict[str, Grad] = {}
    for name in TERMS[:-1]:
        M = W if name == "overlap" else D
        terms[name] = _grad(getattr(runners, name)(basis, nuc, M), n, name)
    terms["nuclear"] = _grad(host.nuclear_repulsion_gradient(nuc), n,
                             "nuclear")
    total = []
    for a in range(n):
        row = [0.0, 0.0, 0.0]
        for k in range(3):
            for name in TERMS:
                row[k] += terms[name][a][k]
        total.append((row[0], row[1], row[2]))
    return RhfGradient(**terms, total=tuple(total))


__all__ = ["SIGN", "TERMS", "RhfGradient", "Runners", "host_runners",
           "rhf_gradient"]
