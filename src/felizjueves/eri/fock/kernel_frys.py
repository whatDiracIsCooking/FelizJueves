"""The fused Fock kernel on Rys quadrature: `Target.F` on `Engine.RYS`.

    build_frys_kernel((la, lb, lc, ld) | spec, *, eris=, weights=) -> Kernel
    frys_class_spec(la, lb, lc, ld)        -> Spec   coulomb/d0/(..)/b1/f.rys/literal
    frys_name(cls)                         -> "fr_{la}{lb}{lc}{ld}"
    RYS_HOOKS                              the seams it moves

`kernel_f`'s fused scheme (`[Fock]` §3.2: one block per bra shell pair, one
thread per contracted quartet, the `K_CD x K_AB` contraction inside, the
six-block screen that skips, the derived 8-fold weights and the six
thread-private buffers, `F <- F + F^T` on the host) with ONE thing replaced:
the vertical recurrence. Where `kernel_f` runs `[Fock]` eqs 25-26 on one
Boys evaluation per primitive quartet, this kernel runs the `N = L // 2 + 1`
point Rys rule (`ir.Rys`, the device `fj_rys<N>`, `felizjueves.rys.roots`)
and, per node, the per-axis 2D recurrence (`felizjueves.rys.recurrence`),
whose products are the same contracted boundary `[e0|f0]^(0)` the HGP HRR
reads. So the accumulators, the HRR (eq 28, on contracted values), the tap,
the digestion, the staging, the screen, the parameters and the launch are
`kernel_f`'s own -- not re-derived: this module is a set of `FHooks` on its
seams (`eri/fock/levers/seams.py`, #368) and a head note. A launch plan,
`pack`ed records, `density_blocks` and `close` serve both kernels unchanged,
and an `fr_` unit is a drop-in for the `f_` unit of the same class (same
signature, same grid, same output).

WHY THE HRR STAYS HGP'S. #468's spike carries the transfers in 2D, per node
and per primitive quartet (`I(i, j+1, k, l) = I(i+1, j, k, l) + AB I(i, j,
k, l)`); here they run once per CONTRACTED quartet as eq 28, which is the
cheaper place and the code `kernel_f` already pins (`hgp.hrr`). The boundary
`(e0|f0)` with `la <= |e| <= la + lb`, `lc <= |f| <= lc + ld` is exactly the
2D VRR's `I(e_d, f_d)` product, so the seam is the boundary and nothing
moves across it.

THE SEAMS MOVED: `ket_prim` (the ket record's reads, `Q - C` only when the
ket steps), `quartet_scalars` (the bra record, the quartet geometry and the
2D coefficients' scalars, `T`, `S`, then the rule into `rys_x[N]`,
`rys_w[N]`), `vrr` (the root loop, which adds into the accumulators itself)
and `accumulate` (nothing: the root loop is eq 27). Every other hook is
`kernel_f`'s default, so the two kernels differ in the primitive nest's body
and nowhere else (test/test_kernel_frys.py diffs them).

NO LEVER. The slot `f.rys` admits no kernel-body lever (`levers`), and an
outer-layer one is refused here, as `kernel_f` refuses it.

ACCURACY. The rule is within 7e-15 relative of a 40-digit route for every
`N <= 5`, `T` in `[0, 1e6]` (`rys.roots`); the ERIs and `F` are held to the
HGP kernel's and to libcint's, interpreted, to the tolerances
test/test_kernel_frys.py states. The device `fj_rys` is the host rule's
text: run as host C++ it is bitwise the host's on that grid
(test/test_rys_roots.py). NOT RUN ON A CARD in the session that built it
(no card, no nvcc / hipcc): the units passed clang's CUDA front end only.

Nothing here imports `plan/`, `ao/`, `reference/`, the interpreter, a
backend or a device.
"""

from __future__ import annotations

from dataclasses import replace

from felizjueves.eri.fock.kernel_f import WEIGHTS, Weights
from felizjueves.eri.fock.levers.seams import (
    AB_SH,
    DEFAULT_HOOKS,
    KAB,
    KCD,
    FContext,
    FHooks,
    acc_name,
)
from felizjueves.hgp.recurrence import (
    _TWO_OVER_SQRT_PI,
    PAIR,
    PQ_PARAMS,
    THETA,
    hgp_class,
)
from felizjueves.hgp.vrr import AmClass
from felizjueves.ir import (
    Call,
    Decl,
    Expr,
    Fn,
    If,
    Kernel,
    Lit,
    Note,
    Rys,
    Scalar,
    Space,
    Stmt,
    Var,
)
from felizjueves.rys import recurrence as rr
from felizjueves.rys.roots import N_MAX, n_roots
from felizjueves.spec import Engine, Operator, Spec, Target

__all__ = ["RYS_HOOKS", "build_frys_kernel", "frys_class_spec", "frys_name"]

_P, _X, _ZETA, _U = 0, 3, 6, 7  # field offsets in seams.REC
_AXES = "xyz"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def frys_class_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The F class `(la lb | lc ld)` on the Rys engine: AO order, one batch."""
    return Spec(Operator.COULOMB, 0, (la, lb, lc, ld), Target.F,
                engine=Engine.RYS)


def frys_name(cls: AmClass) -> str:
    return "fr_" + "".join(map(str, cls))


def _spec_of(cls: AmClass | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = frys_class_spec(*cls)
    else:
        raise TypeError("build_frys_kernel takes (la, lb, lc, ld) or a Spec, "
                        f"not {cls!r}")
    hgp_class(spec)  # refuses anything but Target.F
    if spec.engine is not Engine.RYS:
        raise ValueError(f"{spec.key()}: build_frys_kernel builds Target.F on "
                         f"Engine.RYS, not {spec.engine}")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the fused F on Rys takes no lever")
    if n_roots(sum(spec.am)) > N_MAX:
        raise ValueError(f"{spec.key()}: L = {sum(spec.am)} needs "
                         f"{n_roots(sum(spec.am))} Rys nodes; rys.roots "
                         f"stops at N = {N_MAX} ((dd|dd))")
    return spec


# --- the hooks ----------------------------------------------------------------


def _need(ctx: FContext) -> frozenset[str]:
    return rr.scalar_needs(ctx.cls)


def ket_prim(ctx: FContext) -> tuple[Stmt, ...]:
    """The ket primitive pair: `Q`, `Q - C` when the ket steps, `q`, `U`."""
    def ket(f: int) -> Expr:
        return ctx.read("ket_prim", ctx.k, Var(KCD), Lit(f))

    need = _need(ctx)
    return (
        Note("the ket primitive pair, from global memory (this thread's |cd))"),
        *(ctx.scalar(f"Qk{d}", ket(_P + d), "vrr") for d in range(3)),
        *(ctx.scalar(f"QC_{a}", ket(_X + d), "vrr")
          for d, a in enumerate(_AXES) if f"QC_{a}" in need),
        ctx.scalar("q", ket(_ZETA), "vrr"),
        ctx.scalar("Uq", ket(_U), "vrr"),
    )


def quartet_scalars(ctx: FContext) -> tuple[Stmt, ...]:
    """The bra primitive pair, the quartet geometry, `T`, `S` and the Rys
    rule (module docstring)."""
    need = _need(ctx)
    n = n_roots(sum(ctx.cls))

    def sh(f: int) -> Expr:
        return ctx.read(AB_SH, Var(KAB), Lit(f))

    def scalar(name: str, e: Expr) -> Decl:
        return ctx.scalar(name, e, "vrr")

    pq = [Var(v) for v in PQ_PARAMS]
    theta = Var(THETA)
    out: list[Stmt] = [
        Note("the bra primitive pair, from shared memory"),
        *(scalar(f"Pb{d}", sh(_P + d)) for d in range(3)),
        *(scalar(f"PA_{a}", sh(_X + d)) for d, a in enumerate(_AXES)
          if f"PA_{a}" in need),
        scalar("p", sh(_ZETA)),
        scalar("Ub", sh(_U)),
        Note("quartet geometry: W - P and W - Q with kernel_f's derived "
             "signs, and the 2D recurrence's scalars"),
        *(scalar(name, _c(Fn.SUB, Var(f"Pb{d}"), Var(f"Qk{d}")))
          for d, name in enumerate(PQ_PARAMS)),
        scalar("ppq", _c(Fn.ADD, Var("p"), Var("q"))),
        scalar("rho", _c(Fn.DIV, _c(Fn.MUL, Var("p"), Var("q")), Var("ppq"))),
        scalar(THETA, _c(Fn.SQRT, Var("rho"))),
    ]
    wp = any(f"WP_{a}" in need for a in _AXES)
    wq = any(f"WQ_{a}" in need for a in _AXES)
    if wp or "q_ppq" in need:
        out.append(scalar("q_ppq", _c(Fn.DIV, Var("q"), Var("ppq"))))
    if wq or "p_ppq" in need:
        out.append(scalar("p_ppq", _c(Fn.DIV, Var("p"), Var("ppq"))))
    for d, a in enumerate(_AXES):
        if f"WP_{a}" in need:
            out.append(scalar(f"WP_{a}", _c(Fn.SUB, Lit(0.0), _c(
                Fn.MUL, Var("q_ppq"), pq[d]))))
        if f"WQ_{a}" in need:
            out.append(scalar(f"WQ_{a}", _c(Fn.MUL, Var("p_ppq"), pq[d])))
    for name, x in (("inv2p", "p"), ("inv2q", "q"), ("inv2pq", "ppq")):
        if name in need:
            out.append(scalar(name, _c(Fn.DIV, Lit(0.5), Var(x))))
    out += [
        scalar(PAIR, _c(Fn.MUL, Var("Ub"), Var("Uq"))),
        Note(f"the Rys rule: N = {n} nodes and weights of (1/2) x^(-1/2) "
             "exp(-T x) dx on [0, 1] at T = rho |PQ|^2 (rys/roots.py); "
             "S = U_P U_Q (2/sqrt(pi)) theta, the HGP fundamental's scale"),
        Decl("T", Scalar.F64, init=_c(Fn.MUL, _c(Fn.MUL, theta, theta), _c(
            Fn.ADD, _c(Fn.ADD, _c(Fn.MUL, pq[0], pq[0]),
                       _c(Fn.MUL, pq[1], pq[1])), _c(Fn.MUL, pq[2], pq[2])))),
        Decl(rr.WEIGHT_SCALE, Scalar.F64, init=_c(Fn.MUL, Var(PAIR), _c(
            Fn.MUL, Lit(_TWO_OVER_SQRT_PI), theta))),
        Decl(rr.NODES, Scalar.F64, (Lit(n),), Space.LOCAL),
        Decl(rr.WEIGHTS, Scalar.F64, (Lit(n),), Space.LOCAL),
        Rys(rr.NODES, rr.WEIGHTS, n, Var("T")),
    ]
    return tuple(out)


def vrr(ctx: FContext) -> tuple[Stmt, ...]:
    """The root loop: per node the 2D tables and every boundary value,
    added into `kernel_f`'s accumulators."""
    return rr.root_loop(ctx.cls, ctx.bnd, acc_name,
                        n_roots=n_roots(sum(ctx.cls)))


def accumulate(ctx: FContext) -> tuple[Stmt, ...]:
    return ()  # the root loop adds into the accumulators (eq 27's place)


RYS_HOOKS = replace(DEFAULT_HOOKS, ket_prim=ket_prim,
                    quartet_scalars=quartet_scalars, vrr=vrr,
                    accumulate=accumulate)


def build_frys_kernel(cls: AmClass | Spec, *, name: str | None = None,
                      weights: Weights = WEIGHTS, eris: bool = False
                      ) -> Kernel:
    """The fused Fock kernel of one F class on Rys quadrature (module
    docstring): `kernel_f`'s skeleton and seams, the Rys hooks. `cls` is
    `(la, lb, lc, ld)` in AO order or a `Target.F` / `Engine.RYS` `Spec`;
    `weights` and `eris=` are `kernel_f.build_f_kernel`'s."""
    spec = _spec_of(cls)
    cls = hgp_class(spec)
    if not isinstance(weights, Weights):
        raise TypeError(f"weights {weights!r} is not a Weights")
    la, lb, lc, ld = cls
    if name is None:
        name = frys_name(cls)
    hooks: FHooks = RYS_HOOKS
    ctx = FContext.of(spec, cls, name, weights, eris, hooks)
    body: tuple = (
        Note(f"(ab|cd) by Rys quadrature, F class ({la}{lb}|{lc}{ld}) in AO "
             f"order: N = {n_roots(sum(cls))} nodes per primitive quartet, "
             "the 2D VRR per node into the HGP boundary, HRR eq 28 after the "
             "K_CD x K_AB nest, then [Fock] §3.2's digestion into F "
             "(eri/fock/kernel_frys.py)"),
        Note("launch: grid = n_bra, blockDim = (threads, 1); one block per "
             "bra shell pair; the host closes F <- F + F^T after every launch"),
        Decl("bra", Scalar.I32, init=Call(Fn.BLOCK_X, ())),
        *hooks.shared_decl(ctx),
        If(_c(Fn.LT, ctx.bra, Var("n_bra")), tuple(hooks.per_bra(ctx))),
    )
    return hooks.finish(ctx, Kernel(name, hooks.params(ctx), body))
