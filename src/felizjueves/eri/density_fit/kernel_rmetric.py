"""The DF Coulomb metric `(P|Q)` on Rys quadrature: `Target.METRIC` on
`Engine.RYS`.

    build_rmetric_kernel(spec | (lP, lQ))  -> Kernel   `rmetric_{lP}{lQ}`
    rmetric_spec(lP, lQ)                   -> Spec     `.../metric.rys/...`
    rmetric_name(spec | (lP, lQ))          -> str
    RYS_METRIC_L_MAX                       -> 9: `N = L // 2 + 1 <= 5`

The metric of milestone 15 (`kernel_metric`, MMD) and its HGP twin
(`kernel_hgp.build_hmetric_kernel`) with the vertical recurrence replaced by
the `N = (lP + lQ) // 2 + 1` point Rys rule (`ir.Rys`, the device
`fj_rys<N>`, `felizjueves.rys.roots`) and, per node, the per-axis 2D
recurrence (`felizjueves.rys.recurrence.root_loop`). It fills
`kernel_hgp.metric_kernel`'s skeleton, so an `rmetric_` unit is a DROP-IN
for the `metric_` / `hmetric_` unit of the same class: the same parameters
in the same order, the same staging (`MetricClass.arrays()`, `s` passed as
`s_aux`), one thread per staged aux shell pair, its two record ranges
walked serially (bra outer), and the plain mirrored store (`kernel_jab`'s
diagonal rule), no atomics, so runs are bit-reproducible.

A METRIC CLASS IS `(lP 0 | lQ 0)` WITH EXPONENT-0 PARTNERS (`kernel_hgp`'s
module docstring): an aux Gaussian is the pair `(P, 1)` whose second member
is an s function of exponent 0 on the same centre, so `p = alpha_P`, `P =
A_P`, `U_P = s_aux` (`c (pi/alpha)^{3/2}`, `K = 1`), and likewise on the
ket. Both `P - A` and `Q - C` are zero, so the 2D recurrence's seeds are
`C00 = x (W - P)` and `D00 = x (W - Q)` (`root_loop(..., single_centre=
True)`): the zero terms are not spelled. There is no HRR (`b = d = 0`): the
boundary `[e0|f0]^(0)` summed over the nodes and the record pairs is the
contracted `(P|Q)` itself, so

    (P|Q) += S SUM_n w_n I_x(e_x, f_x) I_y(e_y, f_y) I_z(e_z, f_z)

per record pair, `S = U_P U_Q (2/sqrt(pi)) theta` and `(x_n, w_n)` the
rule at `T = rho |A_P - A_Q|^2` (`rys.recurrence.quartet_rule`, shared with
`kernel_frys`, `kernel_rj` and `kernel_rk`, its text unchanged).

ROUND-OFF, NOT BITWISE, against the host (`eri.density_fit.metric`, MMD
on `reference.pq`), the MMD kernel and the HGP twin: another sum.

THE REACH: `N <= rys.roots.N_MAX = 5`, so `lP + lQ <= 9` ((g|h); the
`cc-pvtz-jkfit` set's whole reach is (g|g)); a class past it is refused. Unbatched
(the metric's decision) and no lever.

A builder: vendor-neutral IR. It imports neither `ao/` (its witnesses), nor
the interpreter, nor `reference/`, nor `plan/`, nor a backend
(test/test_layering.py, rule `rys_metric_builder`).
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_hgp import (
    BRA,
    KET,
    S_AUX,
    metric_acc,
    metric_kernel,
)
from felizjueves.hgp.vrr import cartesians
from felizjueves.ir import Decl, Expr, Index, Kernel, Lit, Note, Scalar, Stmt, Var
from felizjueves.rys import recurrence as rr
from felizjueves.rys.roots import N_MAX, n_roots
from felizjueves.spec import Engine, Operator, Spec, Target

__all__ = ["RYS_METRIC_L_MAX", "build_rmetric_kernel", "rmetric_name",
           "rmetric_spec"]

RYS_METRIC_L_MAX = 2 * N_MAX - 1  # lP + lQ: the rule's N = L // 2 + 1 <= 5


def rmetric_spec(lP: int, lQ: int) -> Spec:
    """The aux class `(lP, lQ)` on the Rys engine: `.../metric.rys/...`."""
    return Spec(Operator.COULOMB, 0, (lP, lQ), Target.METRIC,
                engine=Engine.RYS)


def _spec(cls: Spec | tuple[int, int]) -> Spec:
    spec = cls if isinstance(cls, Spec) else rmetric_spec(*cls)
    if spec.target is not Target.METRIC or spec.engine is not Engine.RYS:
        raise ValueError(f"{spec.key()} is not a Rys Target.METRIC key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the Rys metric kernel takes no "
                         "levers")
    L = sum(spec.am)
    if n_roots(L) > N_MAX:
        raise ValueError(f"{spec.key()}: L = {L} needs {n_roots(L)} Rys "
                         f"nodes; rys.roots stops at N = {N_MAX} ((g|h))")
    return spec


def rmetric_name(cls: Spec | tuple[int, int]) -> str:
    """`rmetric_` and the two `l`s, `_`-separated once any is two digits."""
    am = cls.am if isinstance(cls, Spec) else tuple(cls)
    return "rmetric_" + ("_" if max(am) > 9 else "").join(map(str, am))


def build_rmetric_kernel(cls: Spec | tuple[int, int], *,
                         name: str | None = None) -> Kernel:
    """The Rys metric kernel of one aux class (module docstring); its
    parameters, launch and output are `kernel_metric.build_metric_kernel`'s.
    """
    spec = _spec(cls)
    lP, lQ = spec.am
    am = (lP, 0, lQ, 0)
    n = n_roots(lP + lQ)
    bnd = [(e, f) for e in cartesians(lP) for f in cartesians(lQ)]
    ip, iq = Var(BRA), Var(KET)

    def scalar(n: str, e: Expr) -> Decl:
        return Decl(n, Scalar.F64, init=e)

    bra: tuple[Stmt, ...] = (
        Decl("p", Scalar.F64, init=Index("alpha", (ip,))),
        *(Decl(f"Pb{d}", Scalar.F64, init=Index("A", (ip, Lit(d))))
          for d in range(3)),
        Note("U_P = s_aux of the bra record: c (pi/alpha)^(3/2), K = 1"),
        Decl("Ub", Scalar.F64, init=Index(S_AUX, (ip,))),
    )
    ket: tuple[Stmt, ...] = (
        Decl("q", Scalar.F64, init=Index("alpha", (iq,))),
        *(Decl(f"Qk{d}", Scalar.F64, init=Index("A", (iq, Lit(d))))
          for d in range(3)),
        Decl("Uq", Scalar.F64, init=Index(S_AUX, (iq,))),
        *rr.quartet_rule(am, scalar),
        Note("single-centre sides: P - A = Q - C = 0, so C00 = x (W - P) "
             "and D00 = x (W - Q)"),
        *rr.root_loop(am, bnd, metric_acc, n_roots=n, single_centre=True),
    )
    return metric_kernel(
        lP, lQ, bra, ket, name=name or rmetric_name(spec),
        title=(f"DF Coulomb metric on Rys quadrature, class (lP, lQ) = "
               f"({lP}, {lQ}): (P|Q) = ({lP}0|{lQ}0) with exponent-0 "
               f"partners, N = {n} nodes per record pair and the 2D VRR per "
               "node (eri/density_fit/kernel_rmetric.py)"))
