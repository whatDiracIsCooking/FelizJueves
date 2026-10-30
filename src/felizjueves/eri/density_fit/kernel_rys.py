"""The three-center DF integral on Rys quadrature: `Target.ERI3` on `Engine.RYS`.

    build_reri3_kernel(spec | (la, lb, lP))  -> Kernel   `reri3_{la}{lb}{lP}`
    reri3_spec(la, lb, lP)                   -> Spec     `.../eri3.rys/...`
    reri3_name(spec | (la, lb, lP))          -> str      its name, unbuilt
    RYS_ERI3_L_MAX                           -> 9: `la + lb + lP` the rule reaches

`(mn|P)` of milestone 16 (`kernel_eri3`, McMurchie & Davidson) and of the
HGP twin (`kernel_hgp.build_heri3_kernel`) on the third engine. It is the
HGP kernel with ONE thing replaced, through `kernel_hgp.three_center_kernel`:
the per-aux-record vertical recurrence. Where `heri3_` runs `[Fock]` eqs
25-26 off one Boys evaluation per (bra record, aux record), `reri3_` runs
the `N = L // 2 + 1` point Rys rule (`ir.Rys`, the device `fj_rys<N>`,
`felizjueves.rys.roots`) and, per node, the per-axis 2D recurrence
(`felizjueves.rys.recurrence`), whose products are the same contracted
boundary `[e0|f0]^(0)` the bra HRR (eq 28) reads. So the key's built class
(`(max, min | lP 0)`, `kernel_hgp`'s "THE BRA ORIENTATION"), the bra
record's scalars (`U_P = s_ab K_AB`), the thread mapping (one thread per
cell, an AO shell pair crossed with an aux shell), the HRR, the plain
mirrored scatter and the parameters are `kernel_hgp`'s own. It is to
`kernel_hgp` what `eri/hjk/kernel_rj.py` is to `kernel_hj`.

A DROP-IN. A `reri3_` unit has the signature, launch and output of the
`heri3_` and `eri3_` units of the class: `eri.density_fit.eri3`'s staging,
`eri3_drive` and test/eri3_driver.py serve all three (`engine=`).

THE AUX SIDE IS SINGLE-CENTRE. An aux Gaussian is the pair `(P, 1)` with an
exponent-0 s partner on its own centre (`kernel_hgp`'s module docstring), so
`q = alpha`, `Q = C = A_k`, `U_Q = s_aux`, and the ket's `QC = Q - C = 0`:
the 2D VRR's `D00 = x (W - Q)` (`rys.recurrence`, "A SINGLE-CENTRE KET"),
its zero term not spelled. The ket span is `lP`, the bra's `la + lb`.

PER (BRA RECORD, AUX RECORD), `rys.recurrence.quartet_rule`'s scalars with
the bra record's `Pb*`, `p`, `Ub` and the aux record's `Qk*`, `q`, `Uq`:
`PQ`, `p + q`, `rho`, `theta`, `W - P`, `W - Q`, the `1/(2x)` the class
reads, `T = rho |PQ|^2`, `S = U_P U_Q (2/sqrt(pi)) theta`, the rule into
`rys_x[N]` / `rys_w[N]`, then the root loop, which adds every boundary
value straight into the contracted accumulators (`kernel_hgp.acc_name`):
the root sum and the record sums are one sum.

THE REACH: `N <= rys.roots.N_MAX = 5`, so `la + lb + lP <= 9`; a class
past it is refused. No batching, no lever (`eri3.rys` admits none).

ROUND-OFF, NOT BITWISE, against the host (`eri.density_fit.eri3`, MMD on
`reference.pq`), the MMD kernel and the HGP one: another recurrence and a
quadrature for the Boys function (`rys.roots`: within 3.4e-15 of a 40-digit
route, `N <= 5`, `T` in `[0, 1e6]`). test/test_kernel_rys_df.py states the
bounds.

A builder: vendor-neutral IR. It imports neither `ao/` nor the interpreter
nor `reference/` nor `plan/` nor `mmd/`; of `eri/density_fit/` only
`kernel_hgp`, whose skeleton it fills (test/test_layering.py, rule
`rys_df_builder`).
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_hgp import (
    S_AUX,
    acc_name,
    three_center_kernel,
)
from felizjueves.hgp import vrr
from felizjueves.hgp.vrr import AmClass, Cart
from felizjueves.ir import Decl, Expr, Index, Kernel, Lit, Note, Scalar, Stmt, Var
from felizjueves.rys import recurrence as rr
from felizjueves.rys.roots import N_MAX, n_roots
from felizjueves.spec import Engine, Operator, Spec, Target

__all__ = ["RYS_ERI3_L_MAX", "build_reri3_kernel", "reri3_name",
           "reri3_spec"]

RYS_ERI3_L_MAX = 2 * N_MAX - 1  # 9: the highest L with n_roots(L) <= N_MAX


def reri3_spec(la: int, lb: int, lP: int) -> Spec:
    """The class `(la, lb, lP)` on the Rys engine: `.../eri3.rys/...`."""
    return Spec(Operator.COULOMB, 0, (la, lb, lP), Target.ERI3,
                engine=Engine.RYS)


def _spec(cls: Spec | tuple[int, int, int]) -> Spec:
    spec = cls if isinstance(cls, Spec) else reri3_spec(*cls)
    if spec.target is not Target.ERI3 or spec.engine is not Engine.RYS:
        raise ValueError(f"{spec.key()} is not a Rys Target.ERI3 key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the Rys (mn|P) kernel takes no "
                         "levers")
    L = sum(spec.am)
    if n_roots(L) > N_MAX:
        raise ValueError(f"{spec.key()}: L = {L} needs {n_roots(L)} Rys "
                         f"nodes; rys.roots stops at N = {N_MAX} "
                         f"(L <= {RYS_ERI3_L_MAX})")
    return spec


def reri3_name(cls: Spec | tuple[int, int, int]) -> str:
    """`reri3_` and the three `l`s, `_`-separated once any is two digits:
    `r` before `kernel_eri3.eri3_name`'s spelling (`reri3_011`)."""
    am = cls.am if isinstance(cls, Spec) else tuple(cls)
    return "reri3_" + ("_" if max(am) > 9 else "").join(map(str, am))


def _ket_body(cls: AmClass, ket: Var
              ) -> tuple[tuple[Stmt, ...], list[tuple[Cart, Cart]]]:
    """One aux record's statements inside the bra loop (module docstring,
    "PER (BRA RECORD, AUX RECORD)"), and the boundary in `hgp.vrr.targets`
    order."""
    def scalar(name: str, e: Expr) -> Decl:
        return Decl(name, Scalar.F64, init=e)

    bnd = [(e, f) for e, f, _ in vrr.targets(cls)]
    body: tuple[Stmt, ...] = (
        Note("the aux record: q = alpha, Q = C = A_k, U_Q = s_aux (an "
             "exponent-0 partner: QC = 0)"),
        Decl("q", Scalar.F64, init=Index("alpha", (ket,))),
        *(Decl(f"Qk{d}", Scalar.F64, init=Index("A", (ket, Lit(d))))
          for d in range(3)),
        Decl("Uq", Scalar.F64, init=Index(S_AUX, (ket,))),
        *rr.quartet_rule(cls, scalar),
        *rr.root_loop(cls, bnd, acc_name, n_roots=n_roots(sum(cls)),
                      ket_centred=True),
    )
    return body, bnd


def build_reri3_kernel(cls: Spec | tuple[int, int, int], *,
                       name: str | None = None) -> Kernel:
    """The Rys three-center kernel of one class (module docstring); its
    parameters, launch and output are `kernel_eri3.build_eri3_kernel`'s."""
    spec = _spec(cls)
    return three_center_kernel(
        spec, _ket_body, name=name or reri3_name(spec),
        title="three-center DF integral by Rys quadrature "
              f"(N = {n_roots(sum(spec.am))} nodes)",
        origin="eri/density_fit/kernel_rys.py", bra=("p", "Pb", "Ub"))
