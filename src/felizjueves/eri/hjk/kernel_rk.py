"""The split exchange kernel on Rys quadrature: `Target.K` on `Engine.RYS`.

    build_rk_kernel(spec | (la, lc, lb, ld), *, eris=, density=) -> Kernel
    rk_class_spec(la, lc, lb, ld, *, ordered=False) -> Spec  .../k.rys/...
    rk_name(spec)               -> "rk_..." (canonical) / "rk_o_..." (ordered)
    hgp_twin(spec)              -> the same key on Engine.HGP
    rys_nest(am, boundary)      -> kernel_hj.Nest, the Rys engine's part

`kernel_hk`'s split `K` (`K[a_i][b_j] += D[c_k][d_l] (a_i c_k | b_j d_l)`
over every ordered shell quartet, by atomic add; `[Fock]` §3.2's mapping,
one block per bra shell pair, one thread per contracted quartet, the
`K_AB x K_CD` contraction inside, a one-block screen that skips) with ONE
thing replaced: the vertical recurrence. Where `hk_` runs `[Fock]` eqs
25-26 off one Boys evaluation per primitive quartet, `rk_` runs the
`N = L // 2 + 1` point Rys rule (`ir.Rys`, the device `fj_rys<N>`,
`felizjueves.rys.roots`) and, per node, the per-axis 2D recurrence
(`felizjueves.rys.recurrence`) into the same contracted boundary
`[e0|f0]^(0)` the HGP HRR (eq 28) reads. It is to `kernel_hk` exactly what
`eri/hjk/kernel_rj.py` is to `kernel_hj`, through `kernel_hk`'s seam
(`split_k_kernel`, taking a `kernel_hj.Nest`), and its nest IS
`kernel_rj.rys_nest` with K's opening note: the recurrence of a built class
`(x0 x1 | x2 x3)` does not know whether the deposit is `J`'s or `K`'s.

A DROP-IN. An `rk_` unit has the signature, grid and output of the `hk_`
unit of the same class and mode: `launch_hjk`'s plan (screen `"k"`),
`kernel_f.pack`ed records, `walk_hjk` (its `build=` hook) and
test/hjk_driver.py's harness serve both unchanged. test/test_kernel_rk.py
holds the two kernels' IR equal outside the `K_CD` loop.

THE KEY. `Spec(..., Target.K, engine=Engine.RYS)`, keyed `.../k.rys/...`,
`am` K's Hermite pairing `(la, lc | lb, ld)` as HGP and MMD `K` (#196,
#401); the class mode is HGP K's -- canonical (`rk_{rep}`, a runtime
`perm`) or `ordered` (`rk_o_{class}`). The structure (the built class, the
orientations, the parameters) is read off the HGP twin (`hgp_twin`), so it
cannot drift from `kernel_hk`'s.

THE REACH: `N <= rys.roots.N_MAX = 5`, so `L = la + lc + lb + ld <= 8`
(`(dd|dd)`); a class past it is refused. No batching, no other lever.

Nothing here imports `plan/`, `ao/`, `reference/`, the interpreter, a
backend or a device (test/test_layering.py, rule `rk_builder`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from felizjueves.eri.hjk import kernel_hj as HJ
from felizjueves.eri.hjk import kernel_hk as HK
from felizjueves.eri.hjk import kernel_rj as RJ
from felizjueves.hgp.vrr import AmClass, Cart
from felizjueves.ir import Kernel
from felizjueves.rys.roots import N_MAX, n_roots
from felizjueves.spec import Engine, Operator, Spec, Target

__all__ = ["build_rk_kernel", "hgp_twin", "rk_class_spec", "rk_name",
           "rys_nest"]


def rk_class_spec(la: int, lc: int, lb: int, ld: int, *,
                  ordered: bool = False) -> Spec:
    """The Rys `K` class `(la, lc | lb, ld)` -- K's Hermite pairing -- one
    batch, carrying the `ordered` lever when asked."""
    return Spec(Operator.COULOMB, 0, (la, lc, lb, ld), Target.K,
                (HK.ORDERED,) if ordered else (), engine=Engine.RYS)


def _spec_of(cls: AmClass | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = rk_class_spec(*cls)
    else:
        raise TypeError("build_rk_kernel takes (la, lc, lb, ld) or a Spec, "
                        f"not {cls!r}")
    if spec.target is not Target.K or spec.engine is not Engine.RYS:
        raise ValueError(f"{spec.key()}: the Rys K kernel builds Target.K on "
                         "Engine.RYS only")
    L = sum(spec.am)
    if n_roots(L) > N_MAX:
        raise ValueError(f"{spec.key()}: L = {L} needs {n_roots(L)} Rys "
                         f"nodes; rys.roots stops at N = {N_MAX} ((dd|dd))")
    return spec


def hgp_twin(cls: AmClass | Spec) -> Spec:
    """The same key on `Engine.HGP`: `kernel_hk` validates it (the batch
    count, the levers) and reads the structure off it."""
    twin = replace(_spec_of(cls), engine=Engine.HGP)
    HK.built_class(twin)  # kernel_hk's own refusals
    return twin


def rk_name(cls: AmClass | Spec) -> str:
    """`rk_{built class}` canonical, `rk_o_{class}` ordered: `kernel_hk`'s
    names with the engine's letter, `_`-separated once any `l` is above 9."""
    return "r" + HK.hk_name(hgp_twin(cls))[1:]


def rys_nest(am: AmClass, bnd: Sequence[tuple[Cart, Cart]]) -> HJ.Nest:
    """The Rys engine's nest (`kernel_rj.rys_nest`, statement for
    statement) under K's opening note."""
    return replace(RJ.rys_nest(am, bnd), note=(
        "(x0 x1|x2 x3) by Rys quadrature, K class ({cls}) [{mode}]: "
        f"N = {n_roots(sum(am))} nodes per primitive quartet and the 2D VRR "
        "per node into the HGP boundary inside the K_AB x K_CD nest, HRR eq "
        "28 after it, then K_ab += D_cd (ac|bd), split "
        "(eri/hjk/kernel_rk.py)"))


def build_rk_kernel(cls: AmClass | Spec, *, name: str | None = None,
                    eris: bool = False,
                    density: tuple[int, int] = HK.K_DENSITY) -> Kernel:
    """The split `K` kernel of one class on Rys quadrature (module
    docstring). `cls` is a `Target.K`, `Engine.RYS` `Spec` (or `(la, lc,
    lb, ld)`, canonical); `eris=` and `density=` are
    `kernel_hk.build_hk_kernel`'s."""
    return HK.split_k_kernel(hgp_twin(cls), rys_nest,
                             name=name or rk_name(cls), eris=eris,
                             density=density)
