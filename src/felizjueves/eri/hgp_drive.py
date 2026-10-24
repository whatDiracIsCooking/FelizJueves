"""`(ab|cd)` of a CONTRACTED shell quartet by the HGP engine, interpreted (#281).

    eri_block(bra_pair, ket_pair, *, boys, first="bra")  -> (ab|cd)
    eri_block_and_scale(bra_pair, ket_pair, *, boys, first="bra")
                                                         -> ((ab|cd), scale)
    primitive_scalars(sa, sb, sc, sd, i, j, k, l)        -> the VRR's binding

The thin evaluator behind the cross-engine pin: `hgp.recurrence`'s two
standalone kernels (#279) run in `interp.Interpreter`, composed exactly as
`[Fock]` §3.1 composes them --

    boundary(e, f) = SUM_{i j k l} [e,0|f,0]^(0)_{ijkl}     VRR, per primitive
    (ab|cd)        = HRR(boundary; AB, CD)                  once, contracted

-- so the `K^4` contraction sum finishes BEFORE the HRR reads it (milestone
12, pin 5: eq 28 holds only on contracted values). The fused kernel (#282)
does the same sum in registers inside one launch; here it is one VRR launch
per primitive quartet and a host-side `+=` in primitive order, then one HRR
launch. Nothing about the recurrence is restated: every integral comes out
of the IR the builder made.

THE SIGNATURE IS `eri.eri.eri_block`'s, deliberately: two ordered shell pairs
in, the block out as a tuple of rows (`ab` over `cartesians(la) x
cartesians(lb)`, a outer; columns `cd`, c outer), the evaluator's own
special function as a keyword. So `functools.partial(eri_block, boys=boys)`
is an `eri(bra, ket)` for `eri.fock.fock`'s injected sweep, the same slot
`functools.partial(eri.eri.eri_block, pq=pq)` fills today -- the swap #286
makes, with no edit to `eri.fock.fock`. Any ordered quartet is accepted (the HGP
DAGs are built for any `(la, lb, lc, ld)`; the 21-class canonical ladder is
what the kernels are named and measured on, not a restriction here).

THE BINDING (`primitive_scalars`) is `hgp.recurrence`'s module docstring,
"WHAT THE SCALARS ARE", spelled for one primitive quartet `(i, j | k, l)`,
with ONE addition: the four contraction coefficients ride on `pair`,

    pair = (c_i c_j (pi/p)^(3/2) K_ab) (c_k c_l (pi/q)^(3/2) K_cd),

because every VRR value is linear in the seed scale `S = pair (2/sqrt(pi))
theta` (each statement is a weighted sum of earlier ones, and the leaves are
`S F_m(T)`). `Shell.coefficients` already carry any normalisation
(`ao/shell.py`), so this is `ao.pairs.dress`'s `c_a c_b (pi/zeta)^{3/2}`
placed on the seed instead of on `E`. `WQ` is the DERIVED `W - Q = +p/(p+q)
PQ`, not eq 26's printed sign (`fock/eq26-ket-vrr-sign`, asserted by #291).

THE SCALE, `eri.eri.eri_block_and_scale`'s twin for THIS engine: the same two
kernels with every `Weighted` coefficient `c` replaced by `|c|`
(`absolute_kernel`), run on `|scalar|` for every bound scalar, so each
statement adds the absolute values of what the real one adds and the result
is `SUM |term|` through the whole VRR, the contraction sum and the HRR. It
is the yardstick for THIS engine's rounding: the HRR's `AB`/`CD` transfers
cancel, which the MMD scale does not see, so a tolerance stated against the
MMD scale alone would be a statement about the wrong program's sums.

WHY AN ADAPTER UNDER `ao/` AND NOT `hgp/eri.py`. Producing a number from the
IR needs the interpreter, and `hgp/` may not import it (rule `hgp_builder`:
the interpreter checks the builder and must not be leaned on by it) nor
anything outside `hgp`/`spec`/`ir` (rule `hgp`). `eri/j_coulomb/drive.py` is the MMD
precedent for exactly this -- builder + `interp` + `ao`, producing numbers --
so this module follows it, under its own layering rule
(test/test_layering.py, `hgp_adapter`): it may reach `hgp`, `ir`, `interp`,
`spec` and `ao`, and not `reference/` (Boys arrives as `boys=`, as
`interp` requires), `plan/` (pin 11) or a backend.

Kernels are memoised per `(class, first)` (building is pure); results are not.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from functools import cache

import numpy as np

from felizjueves.ao.shell import Shell
from felizjueves.hgp import hrr
from felizjueves.hgp import recurrence as R
from felizjueves.ir import Assign, Decl, Kernel, Note, Term, Weighted
from felizjueves.ir import Boys as BoysFill
from felizjueves.ir.interp import Interpreter
from felizjueves.spec import Operator, Spec, Target

__all__ = ["absolute_kernel", "eri_block", "eri_block_and_scale", "f_spec",
           "primitive_scalars"]

Block = tuple[tuple[float, ...], ...]
Boys = Callable[[int, float], float]
AmClass = tuple[int, int, int, int]
_AXES = "xyz"


def f_spec(cls: AmClass) -> Spec:
    """The `Target.F` key of the AO-order class `(la, lb, lc, ld)`."""
    return Spec(Operator.COULOMB, 0, tuple(cls), Target.F)


def absolute_kernel(kernel: Kernel) -> Kernel:
    """`kernel` with every `Weighted` term's `c` replaced by `|c|`.

    Only the statement kinds `hgp.recurrence`'s standalone kernels hold are
    accepted (`Note`, `Decl`, `Assign`, and the prelude's `Boys` fill);
    anything else is refused, so a kernel this transform does not understand
    cannot yield a scale that is silently not one. The non-`Weighted` inits
    (`T`, `S`, the seed `S boys[m]`) are products of the bound scalars and
    Boys values, already nonnegative when those are, and `F_m(T) > 0`."""
    body = []
    for st in kernel.body:
        if not isinstance(st, (Note, Decl, Assign, BoysFill)):
            raise ValueError(f"absolute_kernel: {type(st).__name__} in "
                             f"{kernel.name} is not a recurrence statement")
        if isinstance(st, Decl) and isinstance(st.init, Weighted):
            st = replace(st, init=Weighted(tuple(
                Term(abs(t.c), t.g, t.src) for t in st.init.terms)))
        body.append(st)
    return replace(kernel, name=f"{kernel.name}_abs", body=tuple(body))


@cache
def _kernels(cls: AmClass, first: str, absolute: bool
             ) -> tuple[Kernel, Kernel]:
    spec = f_spec(cls)
    vk, hk = R.build_vrr_kernel(spec), R.build_hrr_kernel(spec, first=first)
    if absolute:
        return absolute_kernel(vk), absolute_kernel(hk)
    return vk, hk


def _sub(u, v) -> tuple[float, float, float]:
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def primitive_scalars(sa: Shell, sb: Shell, sc: Shell, sd: Shell,
                      i: int, j: int, k: int, l: int  # noqa: E741
                      ) -> dict[str, float]:
    """Every scalar `R.VRR_SCALARS` names, for the primitive quartet
    `(sa_i sb_j | sc_k sd_l)` (module docstring, "THE BINDING")."""
    a, b = sa.exponents[i], sb.exponents[j]
    c, d = sc.exponents[k], sd.exponents[l]
    A, B, C, D = sa.centre, sb.centre, sc.centre, sd.centre
    p, q = a + b, c + d
    P = tuple((a * A[x] + b * B[x]) / p for x in range(3))
    Q = tuple((c * C[x] + d * D[x]) / q for x in range(3))
    PQ = _sub(P, Q)
    W = tuple((p * P[x] + q * Q[x]) / (p + q) for x in range(3))
    rho = p * q / (p + q)
    AB, CD = _sub(A, B), _sub(C, D)
    kab = math.exp(-a * b / p * (AB[0] ** 2 + AB[1] ** 2 + AB[2] ** 2))
    kcd = math.exp(-c * d / q * (CD[0] ** 2 + CD[1] ** 2 + CD[2] ** 2))
    s: dict[str, float] = {}
    for x, ax in enumerate(_AXES):
        s[f"PA_{ax}"] = P[x] - A[x]
        s[f"WP_{ax}"] = W[x] - P[x]
        s[f"QC_{ax}"] = Q[x] - C[x]
        s[f"WQ_{ax}"] = p / (p + q) * PQ[x]  # W - Q, derived (docstring)
        s[f"PQ_{ax}"] = PQ[x]
    s["inv2p"] = 1.0 / (2.0 * p)
    s["inv2q"] = 1.0 / (2.0 * q)
    s["inv2pq"] = 1.0 / (2.0 * (p + q))
    s["inv2p_alpha_over_p"] = 1.0 / (2.0 * p) * (rho / p)
    s["inv2q_alpha_over_q"] = 1.0 / (2.0 * q) * (rho / q)
    s[R.THETA] = math.sqrt(rho)
    s[R.PAIR] = ((sa.coefficients[i] * sb.coefficients[j]
                  * (math.pi / p) ** 1.5 * kab)
                 * (sc.coefficients[k] * sd.coefficients[l]
                    * (math.pi / q) ** 1.5 * kcd))
    return s


def _run(bra_pair, ket_pair, boys: Boys, first: str, absolute: bool
         ) -> np.ndarray:
    """VRR per primitive quartet, summed in primitive order, then one HRR;
    with `absolute`, the |.|-twin kernels on |scalars| (module docstring)."""
    sa, sb = bra_pair
    sc, sd = ket_pair
    cls: AmClass = (sa.l, sb.l, sc.l, sd.l)
    vk, hk = _kernels(cls, first, absolute)
    fix = abs if absolute else (lambda x: x)
    vrr = Interpreter(vk, boys=boys)
    n_bnd = len(R.boundary_order(cls))
    boundary = np.zeros(n_bnd)
    for i in range(len(sa.exponents)):
        for j in range(len(sb.exponents)):
            for k in range(len(sc.exponents)):
                for l in range(len(sd.exponents)):  # noqa: E741
                    s = primitive_scalars(sa, sb, sc, sd, i, j, k, l)
                    ins = {n: fix(s[n]) for n in R.VRR_SCALARS}
                    boundary += vrr.run(**ins, out=np.zeros(n_bnd))[R.OUT]
    AB, CD = _sub(sa.centre, sb.centre), _sub(sc.centre, sd.centre)
    geo = {**{f"AB_{ax}": AB[x] for x, ax in enumerate(_AXES)},
           **{f"CD_{ax}": CD[x] for x, ax in enumerate(_AXES)}}
    return Interpreter(hk).run(**{n: fix(geo[n]) for n in R.HRR_SCALARS},
                               boundary=boundary,
                               out=np.zeros(len(hrr.targets(cls))))[R.OUT]


def _rows(flat: np.ndarray, sa: Shell, sb: Shell, sc: Shell,
          sd: Shell) -> Block:
    """`hgp.hrr.targets` order (a, b, c, d nested, `cartesians` each) as
    rows `ab`, columns `cd`."""
    cols = sc.n_cart * sd.n_cart
    return tuple(tuple(float(v) for v in flat[r * cols:(r + 1) * cols])
                 for r in range(sa.n_cart * sb.n_cart))


def eri_block(bra_pair: tuple[Shell, Shell], ket_pair: tuple[Shell, Shell],
              *, boys: Boys, first: str = "bra") -> Block:
    """`(ab|cd)` of the ordered quartet `(sa sb|sc sd)` by VRR (per primitive
    quartet, summed) then HRR (once): rows `ab`, columns `cd`, each in
    `cartesians` order with the first index outer -- `eri.eri.eri_block`'s
    layout and signature, with `boys=` in place of `pq=`."""
    flat = _run(bra_pair, ket_pair, boys, first, False)
    return _rows(flat, *bra_pair, *ket_pair)


def eri_block_and_scale(bra_pair: tuple[Shell, Shell],
                        ket_pair: tuple[Shell, Shell], *, boys: Boys,
                        first: str = "bra") -> tuple[Block, Block]:
    """`(ab|cd)` and this engine's absolute-value twin (module docstring,
    "THE SCALE"), both in `eri_block`'s layout."""
    blk = _run(bra_pair, ket_pair, boys, first, False)
    scale = _run(bra_pair, ket_pair, boys, first, True)
    return (_rows(blk, *bra_pair, *ket_pair),
            _rows(scale, *bra_pair, *ket_pair))
