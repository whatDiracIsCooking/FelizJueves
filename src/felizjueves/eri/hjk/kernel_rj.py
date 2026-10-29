"""The split Coulomb kernel on Rys quadrature: `Target.J` on `Engine.RYS`.

    build_rj_kernel(spec | (la, lb, lc, ld), *, eris=, density=) -> Kernel
    rj_class_spec(la, lb, lc, ld, *, ordered=False) -> Spec  .../j.rys/...
    rj_name(spec)               -> "rj_..." (canonical) / "rj_o_..." (ordered)
    hgp_twin(spec)              -> the same key on Engine.HGP
    rys_nest(am, boundary)      -> kernel_hj.Nest, the Rys engine's part

`kernel_hj`'s split `J` (`J[a_i][b_j] += D[c_k][d_l] (a_i b_j | c_k d_l)`
over every ordered shell quartet, by atomic add; `[Fock]` §3.2's mapping,
one block per bra shell pair, one thread per contracted quartet, the
`K_AB x K_CD` contraction inside, a one-block screen that skips) with ONE
thing replaced: the vertical recurrence. Where `hj_` runs `[Fock]` eqs
25-26 off one Boys evaluation per primitive quartet, `rj_` runs the
`N = L // 2 + 1` point Rys rule (`ir.Rys`, the device `fj_rys<N>`,
`felizjueves.rys.roots`) and, per node, the per-axis 2D recurrence
(`felizjueves.rys.recurrence`), whose products are the same contracted
boundary `[e0|f0]^(0)` the HGP HRR (eq 28) reads. So the staging, the
screen, the HRR, the tap, the deposit, both class modes and the parameters
are `kernel_hj`'s own, through its seam (`kernel_hj.Nest`,
`split_j_kernel`): this module is a `Nest` and a key. It is to `kernel_hj`
what `eri/fock/kernel_frys.py` is to `kernel_f`, and the two Rys builders
share their quartet scalars and rule (`rys.recurrence.quartet_rule`) and
root loop (`rys.recurrence.root_loop`).

A DROP-IN. An `rj_` unit has the signature, grid and output of the `hj_`
unit of the same class and mode: `launch_hjk`'s plan, `kernel_f.pack`ed
records, `walk_hjk` (its `build=` hook) and test/hjk_driver.py's harness
serve both unchanged. test/test_kernel_rj.py holds the two kernels' IR
equal outside the `K_CD` loop.

THE KEY. `Spec(..., Target.J, engine=Engine.RYS)`, keyed `.../j.rys/...`,
`am` the AO-order quartet `(la, lb | lc, ld)` as HGP and MMD `J`; the class
mode is HGP J's -- canonical (the literal: one kernel per orbit of the
8-fold symmetry, built on its representative, `rj_{rep}`, a runtime
`perm`) or `ordered` (the lever, legal on `j.rys` as on `j.hgp`: one kernel
per ordered class, `rj_o_{class}`). The structure (the built class, the
orientations, the parameters) is read off the HGP twin (`hgp_twin`), the
same key with the engine swapped, so it cannot drift from `kernel_hj`'s.

THE NEST. Bra outer, ket inner, as `kernel_hj`: per bra primitive pair the
record's `P`, `P - A` (when the bra steps), `p`, `U`; per ket primitive
pair its `Q`, `Q - C` (when the ket steps), `q`, `U`, then the quartet
scalars, `T = rho |PQ|^2`, `S = U_P U_Q (2/sqrt(pi)) theta`, the rule into
`rys_x[N]` / `rys_w[N]`, and the root loop, which adds every boundary value
straight into the accumulators (the root sum and the contraction are one
sum, as eq 27 is for HGP). Not bitwise against `eri/hgp_drive.py` (a
different sum); held to it and to MMD by test/test_kernel_rj.py's bounds.

THE REACH: `N <= rys.roots.N_MAX = 5`, so `L = la + lb + lc + ld <= 8`
(`(dd|dd)`); a class past it is refused. No batching (one batch, as
`j.hgp`), no other lever.

Nothing here imports `plan/`, `ao/`, `reference/`, the interpreter, a
backend or a device (test/test_layering.py, rule `rj_builder`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from felizjueves.eri.hjk import kernel_hj as HJ
from felizjueves.hgp.vrr import AmClass, Cart
from felizjueves.ir import Decl, Expr, Kernel, Scalar
from felizjueves.rys import recurrence as rr
from felizjueves.rys.roots import N_MAX, n_roots
from felizjueves.spec import Engine, Operator, Spec, Target

__all__ = ["build_rj_kernel", "hgp_twin", "rj_class_spec", "rj_name",
           "rys_nest"]


def rj_class_spec(la: int, lb: int, lc: int, ld: int, *,
                  ordered: bool = False) -> Spec:
    """The Rys `J` class `(la lb | lc ld)`, one batch, carrying the
    `ordered` lever when asked."""
    return Spec(Operator.COULOMB, 0, (la, lb, lc, ld), Target.J,
                (HJ.ORDERED,) if ordered else (), engine=Engine.RYS)


def _spec_of(cls: AmClass | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = rj_class_spec(*cls)
    else:
        raise TypeError("build_rj_kernel takes (la, lb, lc, ld) or a Spec, "
                        f"not {cls!r}")
    if spec.target is not Target.J or spec.engine is not Engine.RYS:
        raise ValueError(f"{spec.key()}: the Rys J kernel builds Target.J on "
                         "Engine.RYS only")
    L = sum(spec.am)
    if n_roots(L) > N_MAX:
        raise ValueError(f"{spec.key()}: L = {L} needs {n_roots(L)} Rys "
                         f"nodes; rys.roots stops at N = {N_MAX} ((dd|dd))")
    return spec


def hgp_twin(cls: AmClass | Spec) -> Spec:
    """The same key on `Engine.HGP`: `kernel_hj` validates it (the batch
    count, the levers) and reads the structure off it."""
    twin = replace(_spec_of(cls), engine=Engine.HGP)
    HJ.built_class(twin)  # kernel_hj's own refusals
    return twin


def rj_name(cls: AmClass | Spec) -> str:
    """`rj_{built class}` canonical, `rj_o_{class}` ordered: `kernel_hj`'s
    names with the engine's letter, `_`-separated once any `l` is above 9."""
    return "r" + HJ.hj_name(hgp_twin(cls))[1:]


def rys_nest(am: AmClass, bnd: Sequence[tuple[Cart, Cart]]) -> HJ.Nest:
    """The Rys engine's nest (module docstring, "THE NEST")."""
    def scalar(name: str, e: Expr) -> Decl:
        return Decl(name, Scalar.F64, init=e)

    n = n_roots(sum(am))
    need = rr.scalar_needs(am) & ({f"PA_{a}" for a in "xyz"}
                                  | {f"QC_{a}" for a in "xyz"})
    body = (
        *rr.quartet_rule(am, scalar),
        *rr.root_loop(am, bnd, HJ.acc_name, n_roots=n),
    )
    return HJ.Nest(frozenset(need), body,
                   "(ab|cd) by Rys quadrature, J class ({cls}) [{mode}]: "
                   f"N = {n} nodes per primitive quartet and the 2D VRR per "
                   "node into the HGP boundary inside the K_AB x K_CD nest, "
                   "HRR eq 28 after it, then J_ab += D_cd (ab|cd), split "
                   "(eri/hjk/kernel_rj.py)")


def build_rj_kernel(cls: AmClass | Spec, *, name: str | None = None,
                    eris: bool = False,
                    density: tuple[int, int] = HJ.COULOMB_DENSITY) -> Kernel:
    """The split `J` kernel of one class on Rys quadrature (module
    docstring). `cls` is a `Target.J`, `Engine.RYS` `Spec` (or `(la, lb,
    lc, ld)`, canonical); `eris=` and `density=` are
    `kernel_hj.build_hj_kernel`'s."""
    return HJ.split_j_kernel(hgp_twin(cls), rys_nest,
                             name=name or rj_name(cls), eris=eris,
                             density=density)
