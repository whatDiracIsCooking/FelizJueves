"""The theta convention, held where theta is USED -- the in-repo anchor.

[JEngine, v2, eq. 13] writes the fundamental with a `theta` it never defines.
docs/implementation/j_engine.md ("Two defects in the source", #8) reads it as
the conventional McMurchie-Davidson reduced exponent

    theta^2 = zeta eta / (zeta + eta)

THE RECORD'S WITNESS IS ELSEWHERE. Since #90, `jengine/theta-undefined` is
discharged from outside the repo -- `J_mn` from the interpreted generated
kernel against PySCF/libcint (test/test_ao_j_interpreted.py,
`test_interpreted_j_matches_reference_and_libcint`, which carries the
`claim` mark). This file carries no mark and stays for two things that
witness cannot give: a closed-form check with no fixture at all, and the
ABSOLUTE normalisation of the fundamental. libcint sees only the product
`(2/sqrt(pi)) theta (pi/zeta)^{3/2} (pi/eta)^{3/2}`, so a constant moved
between the fundamental and `ao.pairs.dress` passes it; it fails here
(`jengine/prefactor-factorisation`'s record says so).

A doc pins a reading; only the code that forms the number can get it wrong.
The kernel forms theta in its per-pair prelude (`kernel.build_pq_kernel`,
spelled by `emit/`); the host forms the same expression in
`reference.pq.theta_of` (the oracle), `ao/eri.py` and `ao/j.py`. This file
holds it at the oracle and the kernel, against an anchor that shares ancestry
with neither:

    [0]^(0) = (2/sqrt(pi)) theta F_0(theta^2 R^2) = erf(theta R) / R

the Coulomb interaction of two Gaussian charges with the (pi/zeta)^(3/2) and
(pi/eta)^(3/2) pair factors removed, in closed form via `math.erf` -- no Boys
quadrature, no recurrence, no IR. And it
shows the test can fail: the readings a slip would produce (theta taken as the
reduced exponent itself, unrooted; sqrt(zeta + eta); sqrt(zeta eta)) each move
the interpreted `J_p` by far more than the tolerance.

Interpreting and emitting only: nothing compiles and nothing launches.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.kernel import (
    bra_functions,
    build_pq_kernel,
    ket_functions,
    launch_shape,
    pq_spec,
)
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import fundamental, pq, theta_of

# zeta eta / (zeta + eta) = 0.5367 -- far from 1, so theta, theta^2 and the
# other readings all differ by O(1), not by rounding.
ZETA, ETA = 0.7, 2.3
P, Q = np.array([0.31, -0.12, 0.45]), np.array([-0.20, 0.27, 0.05])

# What a wrong reading of eq. 13 would compute in place of theta.
WRONG = {
    "theta-as-reduced-exponent": lambda z, e: z * e / (z + e),
    "sqrt-of-sum": lambda z, e: math.sqrt(z + e),
    "sqrt-of-product": lambda z, e: math.sqrt(z * e),
}


def _interpret(L: int, zeta: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """J_p of the generated [p|q] kernel, one bra, one ket; and the D_q used."""
    spec = pq_spec(L)
    n_p, n_q = len(bra_functions(spec)), len(ket_functions(spec))
    D = np.random.default_rng(L).uniform(-1.0, 1.0, (1, n_q))
    lanes = 4
    grid, block = launch_shape(1, lanes)
    it = Interpreter(build_pq_kernel(spec), launch=Launch(grid, block, lanes),
                     boys=boys)
    J = it.run(
        n_bra=1, n_ket=1, tau=0.0,
        P=P[None, :].copy(), zeta=np.array([zeta]), Ip=np.array([1.0]),
        Q=Q[None, :].copy(), eta=np.array([eta]), Iq=np.array([1.0]),
        Dmax=np.array([1.0]), D=D, J=np.zeros((1, n_p)),
    )["J"]
    return J[0], D[0]


def _reference_J(L: int, D: np.ndarray, theta: float) -> np.ndarray:
    spec = pq_spec(L)
    PQ = tuple(float(c) for c in P - Q)
    return np.array([
        sum(pq(p, q, PQ, theta) * D[j] for j, q in enumerate(ket_functions(spec)))
        for p in bra_functions(spec)
    ])


def test_theta_is_the_reduced_exponent_where_it_is_used() -> None:
    R = float(np.linalg.norm(P - Q))
    want_theta = math.sqrt(ZETA * ETA / (ZETA + ETA))

    # 1. The oracle: theta_of forms that theta, and the fundamental it feeds is
    #    the closed-form Coulomb erf(theta R) / R -- an anchor with no Boys.
    assert theta_of(ZETA, ETA) == pytest.approx(want_theta, rel=1e-15)
    coulomb = math.erf(want_theta * R) / R
    assert fundamental(0, tuple(P - Q), theta_of(ZETA, ETA)) == pytest.approx(
        coulomb, rel=1e-12)

    # 2. The kernel, interpreted: at L = 0, J = [0]^(0) D, the same closed form;
    #    at L = 2 (lp = lq = 1, m up to 2) J_p goes through t2 = theta^2 in both
    #    T and the (-2 theta^2)^m weights, and must match the oracle's [p|q].
    J0, D0 = _interpret(0, ZETA, ETA)
    assert J0[0] == pytest.approx(coulomb * D0[0], rel=1e-12)
    J2, D2 = _interpret(2, ZETA, ETA)
    want2 = _reference_J(2, D2, theta_of(ZETA, ETA))
    scale = float(np.abs(want2).max())
    np.testing.assert_allclose(J2, want2, rtol=0, atol=1e-12 * scale)

    # 3. And the check discriminates: every slip-reading moves J by O(1).
    for name, wrong in WRONG.items():
        alt = _reference_J(2, D2, wrong(ZETA, ETA))
        assert np.abs(J2 - alt).max() > 1e-3 * scale, name
        assert abs(J0[0] - math.erf(wrong(ZETA, ETA) * R) / R * D0[0]) > 1e-3, name


@pytest.mark.parametrize(
    ("emit", "arch"), [(cuda.emit, SM_86), (hip.emit, GFX1200)], ids=["cuda", "hip"])
def test_the_emitted_prelude_forms_theta_that_way_and_only_once(emit, arch) -> None:
    # The interpreted IR is what the backends spell; this is the spelling, so a
    # backend that re-formed theta on its own would show here.
    text = emit(build_pq_kernel(2), arch)
    assert re.findall(r"double theta = (.*);", text) == [
        "sqrt(((ze * et) / (ze + et)))"]
    assert re.findall(r"double ze = (.*);", text) == ["zeta[bra]"]
    assert re.findall(r"double et = (.*);", text) == ["eta[k]"]
    assert re.findall(r"double t2 = (.*);", text) == ["(theta * theta)"]
