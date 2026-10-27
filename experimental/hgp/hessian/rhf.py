"""The closed-shell RHF nuclear hessian with every two-electron piece by HGP.

    rhf_hessian(shells, atoms, natm, C, e, nocc, one) -> H, parts

The expression is ``experimental/mmd/hessian``'s (its ``hessian.py``
docstring derives it):

    H_xy = [E_nuc^xy + SUM D h^xy - SUM W S^xy]          (one, the caller's)
         + 1/2 SUM D D (mn|ls)^xy - 1/4 SUM D D (ml|sn)^xy  (J, K: grad2, fused)
         + SUM D^y F^x - SUM W^y S^x                     (response)

with ``F^x = h^x + G^x[D]`` the skeleton Fock derivative. Every
two-electron quantity is this repo's HGP engine: ``(ab|cd)`` and
``d(ab|cd)/dR`` from ``experimental/hgp/gradient``'s ``two_electron`` (for
``F``, ``F^x`` and the CPHF's orbital hessian), the skeleton ``J`` and ``K``
from ``grad2.two_electron_hessian``. The CPHF solve is
``experimental/mmd/hessian``'s ``cphf``, imported unchanged: it is
engine-free numpy on the tensors it is handed.

The one-electron and nuclear terms come from the caller (``one``), because
``experimental/hgp/one_electron`` has no derivatives yet; ``check.py`` takes
them from libcint, as the gradient spike's table 7 does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gradient"),
                str(HERE.parents[1] / "mmd" / "hessian")]

import grad as G  # noqa: E402
import grad2 as G2  # noqa: E402
from hessian import cphf  # noqa: E402  (experimental/mmd/hessian)

__all__ = ["rhf_hessian"]


def rhf_hessian(shells, atoms, natm, C, e, nocc, one):
    """The RHF hessian ``[natm, 3, natm, 3]`` and its parts ``one``, ``J``,
    ``K``, ``response``.

    ``C``, ``e``: converged MO coefficients and energies, occupied first;
    ``one``: ``{"h": [nao, nao], "dh": [natm, 3, nao, nao], "dS": [natm, 3,
    nao, nao], "H": [natm, 3, natm, 3]}``, the core hamiltonian, its and the
    overlap's first derivatives, and the one-electron and nuclear skeleton
    ``E_nuc^xy + SUM D h^xy - SUM W S^xy``."""
    Co = C[:, :nocc]
    D = 2.0 * Co @ Co.T
    eri, deri = G.two_electron(shells, atoms, natm)
    F0 = one["h"] + G.fock(eri, D)
    F1 = one["dh"] + G.fock_deriv(deri, D)
    D1, W1, _ = cphf(C, e, nocc, eri, F0, F1, one["dS"])
    two = G2.two_electron_hessian(shells, atoms, natm, D)
    parts = {
        "one": one["H"],
        "J": two["J"],
        "K": two["K"],
        "response": np.einsum("akmn,blmn->akbl", F1, D1)
        - np.einsum("akmn,blmn->akbl", one["dS"], W1),
    }
    return sum(parts.values()), parts
